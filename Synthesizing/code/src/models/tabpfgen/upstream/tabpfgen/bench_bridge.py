"""
Benchmark bridge (runs inside the TabPFGen container).

staged train.csv + JSON config (written by the host adapter) -> synthetic CSV.

- categorical / ordinal / binary / boolean columns are label-encoded over their
  sorted observed domain and decoded back to domain values (never float codes);
- integer columns are rounded; numeric columns optionally clipped to the train range;
- missing feature values are imputed for TabPFGen and re-injected at the empirical
  per-column rate (optional);
- the TabPFN fit set is capped at the model's pretraining sample limit using a
  stratified subsample (explicit + logged);
- generation runs in chunks, the TabPFN label model is fitted once and reused,
  rows are appended to a partial CSV, and a calibrated ETA is logged.
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

CATEGORICAL_KINDS = {"categorical", "binary", "ordinal", "boolean", "bool"}


def _log(msg: str) -> None:
    print(f"[TabPFGen] {msg}", flush=True)


def _sort_domain(values):
    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        return sorted(values, key=str)


def _is_categorical(series: pd.Series, kind: str) -> bool:
    if kind in CATEGORICAL_KINDS:
        return True
    if kind in {"continuous", "integer"}:
        return False
    return (
        series.dtype == object
        or pd.api.types.is_bool_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def encode_column(series: pd.Series, kind: str, name: str):
    miss = series.isna().to_numpy()
    rate = float(miss.mean()) if len(miss) else 0.0
    if miss.all():
        raise ValueError(f"column {name!r} is entirely missing; cannot model it")
    if _is_categorical(series, kind):
        domain = _sort_domain(pd.unique(series.dropna()).tolist())
        mapping = {v: i for i, v in enumerate(domain)}
        codes = series.map(mapping).to_numpy(dtype=float)
        unmapped = ~miss & np.isnan(codes)
        if unmapped.any():
            raise ValueError(f"column {name!r}: failed to encode {int(unmapped.sum())} values")
        fill = int(np.bincount(codes[~miss].astype(int)).argmax())
        codes[miss] = fill
        return codes, {"kind": "categorical", "domain": domain, "missing_rate": rate}
    numeric = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & numeric.isna()
    if bad.any():
        examples = series[bad].astype(str).head(5).tolist()
        raise ValueError(f"numeric column {name!r} has non-numeric values: {examples}")
    vals = numeric.to_numpy(dtype=float)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    vals = np.where(np.isnan(vals), float(np.nanmean(vals)), vals)
    return vals, {
        "kind": "integer" if kind == "integer" else "continuous",
        "min": lo,
        "max": hi,
        "missing_rate": rate,
    }


def decode_column(values: np.ndarray, enc: dict, name: str, clip: bool):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"column {name!r}: generator produced {int((~np.isfinite(values)).sum())} non-finite values"
        )
    if enc["kind"] == "categorical":
        domain = enc["domain"]
        codes = np.clip(np.rint(values), 0, len(domain) - 1).astype(int)
        out = np.empty(len(codes), dtype=object)
        dom = np.empty(len(domain), dtype=object)
        dom[:] = domain
        out[:] = dom[codes]
        return pd.Series(out, dtype=object)
    if clip:
        values = np.clip(values, enc["min"], enc["max"])
    if enc["kind"] == "integer":
        return pd.Series(np.rint(values).astype(np.int64), dtype="Int64")
    return pd.Series(values, dtype=float)


def stratified_subsample(y: np.ndarray, cap: int, is_clf: bool, rng) -> np.ndarray:
    n = len(y)
    if n <= cap:
        return np.arange(n)
    if is_clf:
        groups = np.asarray(y)
    else:
        edges = np.unique(np.quantile(y, np.linspace(0, 1, 21)[1:-1]))
        groups = np.searchsorted(edges, y, side="right")
    uniq, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(uniq) > cap:
        raise ValueError(f"cannot keep every stratum: {len(uniq)} strata > fit cap {cap}")
    exact = counts * cap / n
    alloc = np.minimum(np.maximum(np.floor(exact).astype(int), 1), counts)
    diff = cap - int(alloc.sum())
    order = np.argsort(-(exact - alloc))
    while diff != 0:
        progressed = False
        for g in order if diff > 0 else np.argsort(-alloc):
            if diff > 0 and alloc[g] < counts[g]:
                alloc[g] += 1
                diff -= 1
                progressed = True
            elif diff < 0 and alloc[g] > 1:
                alloc[g] -= 1
                diff += 1
                progressed = True
            if diff == 0:
                break
        if not progressed:
            break
    idx = [
        rng.choice(np.where(inverse == g)[0], size=int(k), replace=False)
        for g, k in enumerate(alloc)
        if k > 0
    ]
    return np.sort(np.concatenate(idx))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = cfg.get("seed")
    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)

    # Checkpoints must be present unless downloads were explicitly allowed.
    if cfg.get("backend", "local") != "client" and not cfg.get("allow_download"):
        for kind, path in (cfg.get("checkpoints") or {}).items():
            if path and not os.path.isfile(path):
                raise FileNotFoundError(
                    f"TabPFN {kind} checkpoint not found in container: {path}. "
                    "Run the adapter preflight (`python -m models.tabpfgen.adapter.tabpfgen_adapter check`) "
                    "or set TABPFGEN_ALLOW_DOWNLOAD=1."
                )

    from tabpfgen import TabPFGen

    df = pd.read_csv(cfg["train_csv"], encoding="utf-8-sig", low_memory=False)
    columns = list(df.columns)
    target_col = cfg["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"target column {target_col!r} not in CSV")
    kinds = {k: str(v).lower() for k, v in (cfg.get("column_kinds") or {}).items()}
    is_clf = bool(cfg["is_classification"])
    num_rows = int(cfg["num_rows"])
    if num_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {num_rows}")

    target_missing = df[target_col].isna()
    if target_missing.any():
        _log(f"dropped {int(target_missing.sum())} rows with missing target {target_col!r}")
        df = df.loc[~target_missing].reset_index(drop=True)
    if df.empty:
        raise ValueError("no rows left after dropping missing targets")

    feature_cols = [c for c in columns if c != target_col]
    if not feature_cols:
        raise ValueError("TabPFGen needs at least one feature column besides the target")

    X = np.empty((len(df), len(feature_cols)), dtype=np.float32)
    enc = {}
    for j, col in enumerate(feature_cols):
        X[:, j], enc[col] = encode_column(df[col], kinds.get(col, ""), col)
    n_cat = sum(e["kind"] == "categorical" for e in enc.values())
    _log(f"features: {len(feature_cols)} ({n_cat} categorical/ordinal/boolean label-encoded)")

    if is_clf:
        target_domain = _sort_domain(pd.unique(df[target_col]).tolist())
        if len(target_domain) < 2:
            raise ValueError(
                f"classification target {target_col!r} has {len(target_domain)} class; need >= 2"
            )
        t_map = {v: i for i, v in enumerate(target_domain)}
        y = df[target_col].map(t_map).to_numpy(dtype=np.int64)
        target_enc = {"kind": "categorical", "domain": target_domain, "missing_rate": 0.0}
        _log(f"classification target {target_col!r}: {len(target_domain)} classes")
    else:
        y_vals, target_enc = encode_column(df[target_col], kinds.get(target_col, ""), target_col)
        if target_enc["kind"] == "categorical":
            raise ValueError(f"regression target {target_col!r} is not numeric")
        y = y_vals.astype(np.float64)

    n_all = len(X)
    cap = int(cfg["fit_max_rows"])
    idx = stratified_subsample(y, cap, is_clf, rng)
    Xf, yf = X[idx], y[idx]
    if len(idx) < n_all:
        _log(
            f"fit rows {n_all} -> {len(idx)} (cap={cap}, TabPFN {cfg['model_version']} "
            f"pretraining limit={cfg['model_sample_limit']}; stratified by "
            f"{'class' if is_clf else 'target quantile'})"
        )
    else:
        _log(f"fit rows {n_all} (cap={cap}, no subsampling)")

    gen = TabPFGen(
        n_sgld_steps=int(cfg["n_sgld_steps"]),
        sgld_step_size=float(cfg["sgld_step_size"]),
        sgld_noise_scale=float(cfg["sgld_noise_scale"]),
        device=cfg.get("device") or "auto",
        model_version=cfg.get("model_version"),
        classifier_model_path=(cfg.get("checkpoints") or {}).get("classifier"),
        regressor_model_path=(cfg.get("checkpoints") or {}).get("regressor"),
        n_estimators=cfg.get("n_estimators"),
        fit_mode=cfg.get("fit_mode"),
        ignore_pretraining_limits=bool(cfg.get("ignore_pretraining_limits")),
        many_class=cfg.get("many_class", "auto"),
        label_sampling=cfg.get("label_sampling", "sample"),
        regression_sampling=cfg.get("regression_sampling", "predictive"),
        progress_every=int(cfg.get("progress_every", 0)),
        seed=seed,
    )
    device = gen.device

    chunk_rows = cfg.get("chunk_rows")
    if not chunk_rows:
        budget = int(cfg.get("pairwise_budget", 64_000_000))
        chunk_rows = int(min(16384, max(256, budget // max(1, len(Xf)))))
    chunk_rows = max(1, min(int(chunk_rows), num_rows))
    n_chunks = math.ceil(num_rows / chunk_rows)
    _log(
        f"plan: rows={num_rows} chunk_rows={chunk_rows} chunks={n_chunks} "
        f"sgld_steps={gen.n_sgld_steps} step_size={gen.sgld_step_size} "
        f"noise={gen.sgld_noise_scale} device={device} model={cfg.get('model_version')} "
        f"n_estimators={cfg.get('n_estimators')} fit_mode={cfg.get('fit_mode')} "
        f"balance_classes={cfg.get('balance_classes')} label_sampling={gen.label_sampling} "
        f"regression_sampling={gen.regression_sampling}"
    )

    # Fail fast: load weights + fit label model before any SGLD work.
    t0 = time.perf_counter()
    gen.prepare(Xf, yf, "classification" if is_clf else "regression")
    _log(f"preflight: TabPFN weights loaded and fitted in {time.perf_counter() - t0:.1f}s")

    # Calibrate SGLD step cost on a chunk-sized batch.
    calib_steps = 5
    xs = torch.randn(chunk_rows, Xf.shape[1], device=device)
    xt = torch.randn(len(Xf), Xf.shape[1], device=device)
    ys = torch.zeros(chunk_rows, device=device)
    yt = torch.zeros(len(Xf), device=device)
    gen._sgld_step(xs, ys, xt, yt)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(calib_steps):
        xs = gen._sgld_step(xs, ys, xt, yt)
    _sync(device)
    per_step = (time.perf_counter() - t0) / calib_steps
    del xs, xt, ys, yt
    est = per_step * gen.n_sgld_steps * n_chunks
    _log(
        f"estimate: sgld {per_step * 1000:.1f} ms/step x {gen.n_sgld_steps} steps x "
        f"{n_chunks} chunks ~= {est / 60:.1f} min (+ TabPFN label prediction per chunk)"
    )

    out_csv = cfg["output_csv"]
    partial = out_csv + ".partial.csv"
    if os.path.exists(partial):
        os.remove(partial)
    clip = bool(cfg.get("clip_to_train_range", True))
    reinject = bool(cfg.get("reinject_missing", True))

    written = 0
    t_start = time.perf_counter()
    for chunk_i in range(n_chunks):
        take = min(chunk_rows, num_rows - written)
        t_chunk = time.perf_counter()
        if is_clf:
            Xp, yp = gen.generate_classification(
                Xf, yf, n_samples=take, balance_classes=bool(cfg.get("balance_classes"))
            )
        else:
            Xp, yp = gen.generate_regression(Xf, yf, n_samples=take)
        Xp, yp = np.asarray(Xp), np.asarray(yp)
        if len(Xp) != take or len(yp) != take:
            raise RuntimeError(
                f"chunk {chunk_i}: requested {take} rows, got X={len(Xp)} y={len(yp)}"
            )
        frame = {}
        for j, col in enumerate(feature_cols):
            s = decode_column(Xp[:, j], enc[col], col, clip)
            rate = enc[col]["missing_rate"]
            if reinject and rate > 0:
                s[rng.random(take) < rate] = None if s.dtype == object else np.nan
            frame[col] = s
        frame[target_col] = decode_column(yp, target_enc, target_col, clip)
        pd.DataFrame(frame)[columns].to_csv(
            partial, mode="a", header=(written == 0), index=False
        )
        written += take
        dt = time.perf_counter() - t_chunk
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (chunk_i + 1) * (n_chunks - chunk_i - 1)
        _log(
            f"chunk {chunk_i + 1}/{n_chunks}: +{take} rows in {dt:.1f}s "
            f"(total={written}/{num_rows}, elapsed={elapsed / 60:.1f} min, ETA={eta / 60:.1f} min)"
        )

    os.replace(partial, out_csv)
    _log(f"saved {written} rows -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1])
