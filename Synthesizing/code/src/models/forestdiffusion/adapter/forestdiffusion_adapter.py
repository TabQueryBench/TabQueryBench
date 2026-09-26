"""ForestDiffusion（CPU）适配器：数值列 + 类别列整数编码后拟合树扩散（flow matching），joblib 保存模型。

Encoding (see `_encode_frame`):
- continuous/integer -> numeric (float64 keeps epoch-like integers exact); integer
  columns are passed as `int_indexes` (rounded by the model).
- everything else (categorical/binary/ordinal/boolean/text/...) -> integer codes
  passed as `cat_indexes` (one-hot inside ForestDiffusion). NaN is an explicit
  "__nan__" level decoded back to NaN.
- numeric NaN -> hot-deck imputed + binary missing-indicator column (bin index);
  generated rows with indicator 1 get NaN re-injected. Keeps X NaN-free so
  ForestDiffusion can train one multi-output XGBoost per noise level (p_in_one)
  instead of one per column.
- target: regression (registry task_type or numeric data_type) -> numeric column;
  classification -> categorical column in cat_indexes.

Training rows: no cap by default. FORESTDIFFUSION_MAX_TRAIN_ROWS=<n> subsamples
explicitly (logged).

Hyper-parameters (`_resolve_hparams`, logged + written to forestdiffusion_hparams.json):
  FORESTDIFFUSION_PROFILE = auto (default) | upstream
    upstream: n_t=50, duplicate_K=100, n_estimators=100, max_depth=7 (paper defaults)
    auto    : n_t=50, n_estimators=100, max_depth=7,
              duplicate_K = clamp(round(FORESTDIFFUSION_TARGET_EXPANDED_ROWS / rows), 1, 100)
              with TARGET_EXPANDED_ROWS default 2_000_000 (K=100 for <=20k rows)
  Explicit overrides: FORESTDIFFUSION_N_T / _DUPLICATE_K / _N_ESTIMATORS / _MAX_DEPTH /
  _ETA / _N_BATCH / _N_JOBS / _XGB_NTHREAD (0 = all cores) / _GEN_BATCH_SIZE.
  FORESTDIFFUSION_MAX_CAT_LEVELS (default 100, 0 = unlimited): non-target categoricals with more
  levels keep the top (cap-1) levels; the rest are bucketed into "__other__" and resampled from
  their empirical distribution at decode (each one-hot level is an extra regression output).
  FORESTDIFFUSION_MULTI_STRATEGY=multi_output_tree (opt-in) for vector-leaf XGBoost trees.
  `epochs` (runner --epochs) maps to n_estimators when FORESTDIFFUSION_N_ESTIMATORS is unset.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import load_features_json
from core.runner.config import MODEL_DOCKER_MAP, get_synthetic_benchmark_root

_NAN_SENTINEL = "__nan__"
_OTHER_TOKEN = "__other__"
_ISNA_PREFIX = "__fd_isna__"
_NUMERIC_DTYPES = {"continuous", "integer", "numeric", "numerical", "float", "int"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else float(default)


def _resolve_task_type(
    manifest_path: Optional[str], target_col: str, target_dtype: str
) -> Tuple[str, str]:
    """Return ("classification"|"regression", source)."""
    if manifest_path:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            reg_path = manifest.get("field_registry_json")
            if reg_path and Path(reg_path).is_file():
                with open(reg_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                tt = str(registry.get("task_type") or "").strip().lower()
                if "regress" in tt:
                    return "regression", "field_registry.task_type"
                if any(k in tt for k in ("class", "binary", "multiclass")):
                    return "classification", "field_registry.task_type"
                for field in registry.get("fields") or []:
                    if field.get("name") == target_col:
                        sem = str(field.get("semantic_type") or "").lower()
                        if sem in ("continuous", "integer"):
                            return "regression", "field_registry.target.semantic_type"
                        if sem:
                            return "classification", "field_registry.target.semantic_type"
        except Exception as exc:
            print(f"[ForestDiffusion] WARNING: could not read task_type from registry: {exc}")
    if target_dtype in ("continuous", "integer"):
        return "regression", "features.data_type"
    return "classification", "features.data_type"


class ForestDiffusionAdapter(BaseModelAdapter):
    @property
    def model_name(self) -> str:
        return "forestdiffusion"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["forestdiffusion"]

    @property
    def _needs_gpu(self) -> bool:
        return False

    def _extra_volumes(self):
        r = get_synthetic_benchmark_root()
        return [
            (
                r / "third_party" / "ForestDiffusion" / "Python-Package" / "base-ForestDiffusion",
                "/workspace/base-ForestDiffusion",
            )
        ]

    @staticmethod
    def _encode_frame(
        df: pd.DataFrame,
        features: List[dict],
        task_type: str,
        seed: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """返回 X(float64) 与 meta（列顺序、cat/int/bin 下标、类别 level、缺失指示列）。"""
        rng = np.random.default_rng(seed)
        target = next(f["feature_name"] for f in features if f.get("is_target"))
        ordered = [f for f in features if f.get("feature_name") in df.columns and not f.get("is_target")]
        ordered += [f for f in features if f.get("feature_name") == target]

        cols_order: List[str] = []
        parts: List[np.ndarray] = []
        cat_idx: List[int] = []
        int_idx: List[int] = []
        bin_idx: List[int] = []
        categorical_levels: Dict[str, List[str]] = {}
        missing_indicators: Dict[str, str] = {}
        rare_levels: Dict[str, Dict[str, List[Any]]] = {}
        max_cat_levels = _env_int("FORESTDIFFUSION_MAX_CAT_LEVELS", 100)
        all_nan_cols: List[str] = []
        indicators: List[Tuple[str, np.ndarray]] = []

        for feat in ordered:
            name = feat["feature_name"]
            dtype = str(feat.get("data_type", "") or "").strip().lower()
            is_target = name == target
            if is_target:
                as_numeric = task_type == "regression"
            elif dtype in _NUMERIC_DTYPES:
                as_numeric = True
            elif not dtype:
                as_numeric = pd.api.types.is_numeric_dtype(df[name].dtype)
            else:
                as_numeric = False
            if as_numeric:
                casted = pd.to_numeric(df[name], errors="coerce")
                bad = df[name].notna() & casted.isna()
                if bad.any():
                    if is_target:
                        raise ValueError(
                            f"ForestDiffusion: regression target '{name}' has non-numeric values: "
                            f"{df.loc[bad, name].astype(str).head(5).tolist()}"
                        )
                    print(
                        f"[ForestDiffusion] WARNING: column '{name}' declared {dtype!r} but has "
                        f"non-numeric values; encoding it as categorical"
                    )
                    as_numeric = False
            pos = len(cols_order)
            if as_numeric:
                v = casted.to_numpy(dtype=np.float64)
                mask = ~np.isfinite(v)
                if mask.all():
                    if is_target:
                        raise ValueError(f"ForestDiffusion: target '{name}' is entirely NaN")
                    all_nan_cols.append(name)
                    print(f"[ForestDiffusion] column '{name}' is entirely NaN; excluded and emitted as NaN")
                    continue
                if mask.any():
                    v = v.copy()
                    v[mask] = rng.choice(v[~mask], size=int(mask.sum()), replace=True)
                    ind = f"{_ISNA_PREFIX}{name}"
                    missing_indicators[name] = ind
                    indicators.append((ind, mask.astype(np.float64)))
                parts.append(v.reshape(-1, 1))
                if dtype == "integer" and np.allclose(v, np.round(v)):
                    int_idx.append(pos)
            else:
                s = df[name].astype(object).where(df[name].notna(), _NAN_SENTINEL).astype(str)
                n_levels = int(s.nunique(dropna=False))
                if not is_target and max_cat_levels > 1 and n_levels > max_cat_levels:
                    counts = s.value_counts(dropna=False)
                    keep = set(counts.index[: max_cat_levels - 1])
                    rare = counts.iloc[max_cat_levels - 1:]
                    rare_levels[name] = {
                        "values": [str(x) for x in rare.index.tolist()],
                        "probs": [float(x) / float(rare.sum()) for x in rare.tolist()],
                    }
                    s = s.where(s.isin(keep), _OTHER_TOKEN)
                    print(
                        f"[ForestDiffusion] high-cardinality column '{name}': {n_levels} levels > "
                        f"FORESTDIFFUSION_MAX_CAT_LEVELS={max_cat_levels}; kept top {len(keep)} "
                        f"(covering {float(counts.iloc[: max_cat_levels - 1].sum()) / len(s):.2%} of rows), "
                        f"{len(rare)} rare levels bucketed into '{_OTHER_TOKEN}' and resampled at decode"
                    )
                codes, levels = pd.factorize(s, sort=True)
                categorical_levels[name] = [str(x) for x in levels.tolist()]
                parts.append(codes.astype(np.float64).reshape(-1, 1))
                if len(levels) > 1:
                    cat_idx.append(pos)
            cols_order.append(name)

        for ind_name, ind in indicators:
            pos = len(cols_order)
            parts.append(ind.reshape(-1, 1))
            cols_order.append(ind_name)
            bin_idx.append(pos)

        X = np.hstack(parts).astype(np.float64)
        meta = {
            "column_names": cols_order,
            "output_columns": list(df.columns),
            "target_col": target,
            "task_type": task_type,
            "cat_indexes": cat_idx,
            "int_indexes": int_idx,
            "bin_indexes": bin_idx,
            "categorical_levels": categorical_levels,
            "missing_indicators": missing_indicators,
            "rare_levels": rare_levels,
            "max_cat_levels": max_cat_levels,
            "all_nan_cols": all_nan_cols,
        }
        return X, meta

    @staticmethod
    def _resolve_hparams(n_rows: int, n_cols: int, epochs: Optional[int]) -> Dict[str, Any]:
        profile = os.environ.get("FORESTDIFFUSION_PROFILE", "auto").strip().lower() or "auto"
        if profile not in ("auto", "upstream"):
            raise ValueError("FORESTDIFFUSION_PROFILE must be 'auto' or 'upstream'")
        reasons: Dict[str, str] = {}
        n = max(1, int(n_rows))
        if os.environ.get("FORESTDIFFUSION_DUPLICATE_K", "").strip():
            dup_k = _env_int("FORESTDIFFUSION_DUPLICATE_K", 100)
            reasons["duplicate_K"] = "env"
        elif profile == "upstream":
            dup_k = 100
            reasons["duplicate_K"] = "upstream default"
        else:
            target_rows = _env_int("FORESTDIFFUSION_TARGET_EXPANDED_ROWS", 2_000_000)
            dup_k = int(min(100, max(1, round(target_rows / n))))
            reasons["duplicate_K"] = f"auto: clamp(round({target_rows}/rows), 1, 100)"
        if os.environ.get("FORESTDIFFUSION_N_ESTIMATORS", "").strip():
            n_est = _env_int("FORESTDIFFUSION_N_ESTIMATORS", 100)
            reasons["n_estimators"] = "env"
        elif epochs:
            n_est = int(epochs)
            reasons["n_estimators"] = "runner epochs"
        else:
            n_est = 100
            reasons["n_estimators"] = "upstream default"
        n_jobs = _env_int("FORESTDIFFUSION_N_JOBS", 1)
        xgb_nthread = _env_int("FORESTDIFFUSION_XGB_NTHREAD", 0)
        hp = {
            "profile": profile,
            "train_rows": n,
            "train_cols": int(n_cols),
            "n_t": max(2, _env_int("FORESTDIFFUSION_N_T", 50)),
            "duplicate_K": max(1, dup_k),
            "n_estimators": max(1, n_est),
            "max_depth": max(1, _env_int("FORESTDIFFUSION_MAX_DEPTH", 7)),
            "eta": _env_float("FORESTDIFFUSION_ETA", 0.3),
            "n_batch": max(0, _env_int("FORESTDIFFUSION_N_BATCH", 1)),
            "n_jobs": n_jobs if n_jobs != 0 else 1,
            "xgb_nthread": xgb_nthread,  # 0 -> all cores in container
            "xgb_verbosity": _env_int("FORESTDIFFUSION_XGB_VERBOSITY", 1),
            "gen_batch_size": max(1, _env_int("FORESTDIFFUSION_GEN_BATCH_SIZE", 50_000)),
            "seed": _env_int("FORESTDIFFUSION_SEED", 666),
            # opt-in: "multi_output_tree" (one vector-leaf tree per round instead of one tree per
            # output); much cheaper with many one-hot outputs, deviates from the paper setup.
            "multi_strategy": os.environ.get("FORESTDIFFUSION_MULTI_STRATEGY", "").strip(),
            "reasons": reasons,
        }
        if not 0 <= hp["xgb_verbosity"] <= 3:
            raise ValueError("FORESTDIFFUSION_XGB_VERBOSITY must be one of 0, 1, 2, 3")
        hp["expanded_rows_per_fit"] = n * hp["duplicate_K"]
        hp["xgb_fits"] = hp["n_t"]  # p_in_one: one multi-output booster per noise level (per class if label_y)
        return hp

    def train(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        epochs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        csv_path, json_path = self._resolve_model_inputs(csv_path, json_path, kwargs)
        # Hard guard: training must only consume staged train split.
        csv_name = Path(csv_path).name.lower()
        if csv_name in {"val.csv", "test.csv"}:
            raise ValueError(
                f"{self.model_name}: training input must be train split, got '{csv_name}'"
            )
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._validate_model_inputs(csv_path, json_path, require_target=True, strict_numeric_cast=False)
        df = self.read_staged_csv(csv_path)
        features = load_features_json(json_path)

        max_train_rows = _env_int("FORESTDIFFUSION_MAX_TRAIN_ROWS", 0)
        if max_train_rows > 0 and len(df) > max_train_rows:
            print(
                f"[ForestDiffusion] FORESTDIFFUSION_MAX_TRAIN_ROWS={max_train_rows}: "
                f"subsampling {len(df)} -> {max_train_rows} training rows (seed 42)"
            )
            df = df.sample(n=max_train_rows, random_state=42).reset_index(drop=True)

        target = next(f["feature_name"] for f in features if f.get("is_target"))
        target_dtype = str(next(f for f in features if f.get("is_target")).get("data_type", "")).lower()
        task_type, task_source = _resolve_task_type(
            kwargs.get("model_input_manifest"), target, target_dtype
        )
        print(f"[ForestDiffusion] target={target} task_type={task_type} (source={task_source})")
        X, meta = self._encode_frame(df, features, task_type)
        hp = self._resolve_hparams(X.shape[0], X.shape[1], epochs)
        # Workload estimate: ForestDiffusion one-hot encodes cat_indexes (drop_first) and, with the
        # default multi_strategy, XGBoost grows one tree per output per boosting round per noise level.
        n_outputs = X.shape[1] + sum(
            max(0, len(meta["categorical_levels"][meta["column_names"][i]]) - 2) for i in meta["cat_indexes"]
        )
        trees_per_round = 1 if hp["multi_strategy"] == "multi_output_tree" else n_outputs
        hp["n_outputs_onehot"] = int(n_outputs)
        hp["xgb_tree_builds"] = int(hp["n_t"] * hp["n_estimators"] * trees_per_round)
        hp["xgb_row_tree_units"] = float(hp["xgb_tree_builds"] * hp["expanded_rows_per_fit"])
        warn_units = _env_float("FORESTDIFFUSION_WARN_ROW_TREE_UNITS", 2e10)
        if hp["xgb_row_tree_units"] > warn_units:
            print(
                f"[ForestDiffusion] WARNING: estimated workload {hp['xgb_row_tree_units']:.3g} row x tree builds "
                f"({hp['xgb_tree_builds']} trees on {hp['expanded_rows_per_fit']} expanded rows) exceeds "
                f"{warn_units:.3g}; training may take many hours. Reduce explicitly with FORESTDIFFUSION_N_T / "
                "_DUPLICATE_K / _N_ESTIMATORS / _MAX_CAT_LEVELS, or cap rows with --max-train-rows / "
                "FORESTDIFFUSION_MAX_TRAIN_ROWS."
            )
        meta["hparams"] = hp
        print(f"[ForestDiffusion] resolved hyper-parameters: {json.dumps(hp)}")
        (work_dir / "forestdiffusion_hparams.json").write_text(json.dumps(hp, indent=2), encoding="utf-8")

        script = f"""
import numpy as np, joblib, json, os, threading, time
from ForestDiffusion import ForestDiffusionModel
X = np.load("/tmp/fd_X.npy")
with open("/tmp/fd_meta.json") as f:
    meta = json.load(f)
hp = meta["hparams"]
nthread = hp["xgb_nthread"] if hp["xgb_nthread"] > 0 else max(1, (os.cpu_count() or 1) // max(1, hp["n_jobs"]))
print(
    "[ForestDiffusion] train config: "
    f"rows={{X.shape[0]}} cols={{X.shape[1]}} n_t={{hp['n_t']}} n_estimators={{hp['n_estimators']}} "
    f"duplicate_K={{hp['duplicate_K']}} max_depth={{hp['max_depth']}} n_batch={{hp['n_batch']}} "
    f"n_jobs={{hp['n_jobs']}} xgb_nthread={{nthread}} cat={{meta['cat_indexes']}} int={{meta['int_indexes']}} bin={{meta['bin_indexes']}}",
    flush=True,
)
_stop_heartbeat = False
def _heartbeat():
    started = time.time()
    interval = max(60, int(os.environ.get("FORESTDIFFUSION_HEARTBEAT_SECONDS", "900")))
    while not _stop_heartbeat:
        time.sleep(interval)
        if not _stop_heartbeat:
            print(f"[ForestDiffusion] heartbeat training_seconds={{int(time.time() - started)}}", flush=True)
threading.Thread(target=_heartbeat, daemon=True).start()
t0 = time.time()
try:
    m = ForestDiffusionModel(
        X, n_t=hp["n_t"], n_estimators=hp["n_estimators"], duplicate_K=hp["duplicate_K"],
        n_jobs=hp["n_jobs"], n_batch=hp["n_batch"], model="xgboost", diffusion_type="flow",
        max_depth=hp["max_depth"], eta=hp["eta"], tree_method="hist", seed=hp["seed"],
        cat_indexes=meta["cat_indexes"], int_indexes=meta["int_indexes"], bin_indexes=meta["bin_indexes"],
        verbosity=hp["xgb_verbosity"], nthread=nthread,
        **({{"multi_strategy": hp["multi_strategy"]}} if hp["multi_strategy"] else {{}}),
    )
finally:
    _stop_heartbeat = True
print(f"[ForestDiffusion] fit seconds={{time.time() - t0:.1f}}", flush=True)
joblib.dump((m, meta), "/tmp/fd_model.joblib")
print("ForestDiffusion train OK")
"""
        np.save(work_dir / "_fd_X_host.npy", X)
        (work_dir / "_fd_meta_host.json").write_text(json.dumps(meta), encoding="utf-8")
        c_x = self._to_container_path(work_dir / "_fd_X_host.npy")
        c_meta = self._to_container_path(work_dir / "_fd_meta_host.json")
        c_out = self._to_container_path(work_dir / "forestdiffusion_model.joblib")

        full_script = f"""
import os, shutil, json
with open('/tmp/pgrep', 'w') as _f:
    _f.write("#!/usr/bin/env python3\\n")
    _f.write("import subprocess, sys\\n")
    _f.write("ppid = sys.argv[-1]\\n")
    _f.write("out = subprocess.check_output(['ps', '-o', 'pid=', '--ppid', str(ppid)], text=True)\\n")
    _f.write("print(out, end='')\\n")
os.chmod('/tmp/pgrep', 0o755)
os.environ['PATH'] = '/tmp:' + os.environ.get('PATH', '')
shutil.copy(r'{c_x}', '/tmp/fd_X.npy')
with open(r'{c_meta}') as f:
    open('/tmp/fd_meta.json','w').write(f.read())
{script}
shutil.copy('/tmp/fd_model.joblib', r'{c_out}')
"""
        bridge = self._write_bridge_script(work_dir, "_fd_train.py", full_script)
        log = work_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            r = self._run_docker(
                ["python", self._to_container_path(bridge)],
                extra_env={"PYTHONUNBUFFERED": "1"},
            )
            _write_docker_log(
                log,
                r.stdout or "",
                r.stderr or "",
                getattr(r, "bench_timing", None),
            )
        except Exception as e:
            _write_docker_log(
                log,
                getattr(e, "stdout", "") or "",
                getattr(e, "stderr", "") or "",
                getattr(e, "bench_timing", None),
            )
            raise
        md = work_dir / "models_fd"
        md.mkdir(parents=True, exist_ok=True)
        mp = md / "model.joblib"
        src_model = work_dir / "forestdiffusion_model.joblib"
        # Models reach GBs on large tables (1016 trees on 493k rows ~ 1 GB): hardlink instead of a
        # second full in-memory copy; generate() loads work_dir/forestdiffusion_model.joblib anyway.
        mp.unlink(missing_ok=True)
        try:
            os.link(src_model, mp)
        except OSError:
            import shutil
            shutil.copyfile(src_model, mp)
        return {"model_path": mp, "work_dir": work_dir}

    def generate(
        self,
        model_path: Path,
        output_csv: Path,
        num_rows: int = 1000,
        csv_path: Optional[Path] = None,
        json_path: Optional[Path] = None,
        **kwargs,
    ) -> Path:
        model_path, output_csv = Path(model_path), Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        work_dir = model_path.parent.parent if model_path.parent.name.startswith("models_") else model_path.parent
        c_model = self._to_container_path(work_dir / "forestdiffusion_model.joblib")
        c_out = self._to_container_path(output_csv)
        gen_bs_env = os.environ.get("FORESTDIFFUSION_GEN_BATCH_SIZE", "").strip()
        seed = _env_int("FORESTDIFFUSION_GEN_SEED", 0)
        script = f"""
import joblib, time, numpy as np, pandas as pd
m, meta = joblib.load(r'{c_model}')
n_total = int({int(num_rows)})
bs = int({gen_bs_env or 0}) or int((meta.get("hparams") or {{}}).get("gen_batch_size", 50000))
np.random.seed({seed})
_rng = np.random.default_rng({seed})
chunks, done, t0 = [], 0, time.time()
while done < n_total:
    b = min(bs, n_total - done)
    chunks.append(m.generate(batch_size=b))
    done += b
    print(f"[ForestDiffusion] generated {{done}}/{{n_total}} rows ({{time.time() - t0:.1f}}s)", flush=True)
arr = np.concatenate(chunks, axis=0)
df = pd.DataFrame(arr, columns=meta["column_names"])
for col, levels in (meta.get("categorical_levels") or {{}}).items():
    if col not in df.columns:
        continue
    vals = pd.to_numeric(df[col], errors="coerce").round().fillna(0).astype(int)
    vals = vals.clip(lower=0, upper=max(0, len(levels) - 1))
    s = pd.Series([levels[i] for i in vals], index=df.index, dtype=object)
    rare = (meta.get("rare_levels") or {{}}).get(col)
    if rare:
        mask = (s == "{_OTHER_TOKEN}").to_numpy()
        if mask.any():
            p = np.asarray(rare["probs"], dtype=float)
            s.loc[mask] = _rng.choice(np.asarray(rare["values"], dtype=object), size=int(mask.sum()), p=p / p.sum())
    df[col] = s.where(s != "{_NAN_SENTINEL}", other=np.nan)
for col, ind in (meta.get("missing_indicators") or {{}}).items():
    if col in df.columns and ind in df.columns:
        df.loc[pd.to_numeric(df[ind], errors="coerce").round() >= 1, col] = np.nan
for col in meta.get("all_nan_cols") or []:
    df[col] = np.nan
df = df[meta.get("output_columns") or [c for c in df.columns if not c.startswith("{_ISNA_PREFIX}")]]
df.to_csv(r'{c_out}', index=False)
print("saved", len(df), "null rates", df.isna().mean().round(4).to_dict())
"""
        bridge = self._write_bridge_script(work_dir, "_fd_gen.py", script)
        gen_log = output_csv.parent / f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            r = self._run_docker(
                ["python", self._to_container_path(bridge)],
                extra_env={"PYTHONUNBUFFERED": "1"},
            )
            _write_docker_log(
                gen_log,
                r.stdout or "",
                r.stderr or "",
                getattr(r, "bench_timing", None),
            )
        except Exception as e:
            _write_docker_log(
                gen_log,
                getattr(e, "stdout", "") or "",
                getattr(e, "stderr", "") or "",
                getattr(e, "bench_timing", None),
            )
            raise
        return self._postprocess_generated_csv(
            output_csv, csv_path, json_path, num_rows, **kwargs
        )
