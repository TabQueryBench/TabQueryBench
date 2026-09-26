#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NAN_SENTINEL = "__nan__"


def _load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _normalize_cat_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.where(~s.isna(), NAN_SENTINEL)
    s = s.astype(str)
    return s.replace("nan", NAN_SENTINEL)


def _write_results_checkpoint(results: list[dict[str, Any]], manifest_out: Path) -> None:
    if not results:
        return
    pd.DataFrame(results).to_csv(manifest_out, index=False)


def _clear_dir_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def _update_runtime_result(runtime_path: Path, csv_path: Path) -> None:
    if not runtime_path.exists():
        return
    try:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        return
    artifacts = data.setdefault("artifacts", {})
    artifacts["synthetic_csv"] = str(csv_path)
    runtime_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_overlay_dir(tmp: Path, source: Path, target_name: str) -> None:
    if tmp.exists():
        _clear_dir_tree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name in {target_name, "runtime_result.json"}:
            continue
        (tmp / child.name).symlink_to(child)
    if (source / "runtime_result.json").exists():
        (tmp / "runtime_result.json").write_text(
            (source / "runtime_result.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def _prepare_output_target(run_dir: Path, original_csv: Path) -> tuple[Path, bool]:
    target_name = original_csv.name

    if run_dir.is_symlink():
        source = run_dir.resolve()
        tmp = run_dir.parent / f".{run_dir.name}.__overlay_tmp__"
        _build_overlay_dir(tmp, source, target_name)
        return tmp / target_name, True

    preferred = run_dir / target_name
    if preferred.exists():
        if os.access(preferred, os.W_OK):
            return preferred, False
        if os.access(run_dir.parent, os.W_OK):
            source = run_dir.parent / f".{run_dir.name}.__source__"
            if not source.exists():
                run_dir.rename(source)
            tmp = run_dir.parent / f".{run_dir.name}.__overlay_tmp__"
            _build_overlay_dir(tmp, source, target_name)
            return tmp / target_name, True
        raise PermissionError(f"Target CSV is not writable: {preferred}")

    if os.access(run_dir, os.W_OK):
        return preferred, False
    if os.access(run_dir.parent, os.W_OK):
        source = run_dir.parent / f".{run_dir.name}.__source__"
        if not source.exists():
            run_dir.rename(source)
        tmp = run_dir.parent / f".{run_dir.name}.__overlay_tmp__"
        _build_overlay_dir(tmp, source, target_name)
        return tmp / target_name, True
    raise PermissionError(f"Run dir is not writable: {run_dir}")


def _finalize_overlay(run_dir: Path, overlay_csv: Path) -> Path:
    tmp = overlay_csv.parent
    runtime_path = tmp / "runtime_result.json"
    _update_runtime_result(runtime_path, tmp / overlay_csv.name)
    if run_dir.is_symlink():
        run_dir.unlink()
    elif run_dir.exists():
        _clear_dir_tree(run_dir)
    tmp.rename(run_dir)
    return run_dir / overlay_csv.name


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int_series(arr: Any, index: pd.Index | None = None) -> pd.Series:
    np_arr = np.asarray(arr)
    if np_arr.ndim == 2 and np_arr.shape[1] == 1:
        np_arr = np_arr[:, 0]
    numeric = pd.to_numeric(pd.Series(np_arr, index=index), errors="coerce")
    rounded = np.rint(numeric.to_numpy(dtype="float64", na_value=np.nan))
    vals = [pd.NA if np.isnan(v) else int(v) for v in rounded]
    return pd.Series(vals, index=index, dtype="Int64")


def _mapping_from_raw_and_codes(raw_series: pd.Series, code_series: pd.Series) -> dict[int, str]:
    raw = _normalize_cat_series(raw_series).reset_index(drop=True)
    codes = pd.to_numeric(code_series.reset_index(drop=True), errors="coerce").round().astype("Int64")
    if len(raw) != len(codes):
        raise ValueError(f"raw/code length mismatch: {len(raw)} vs {len(codes)}")
    mapping: dict[int, str] = {}
    for code, value in zip(codes.tolist(), raw.tolist()):
        if code is pd.NA or code is None:
            continue
        code = int(code)
        prev = mapping.get(code)
        if prev is None:
            mapping[code] = value
        elif prev == value:
            continue
        elif prev == NAN_SENTINEL and value != NAN_SENTINEL:
            mapping[code] = value
        elif prev != NAN_SENTINEL and value == NAN_SENTINEL:
            continue
        else:
            raise ValueError(f"inconsistent code mapping for code={code}: {prev!r} vs {value!r}")
    return dict(sorted(mapping.items()))


def _repair_with_mapping(series: pd.Series, mapping: dict[int, str]) -> tuple[pd.Series, dict[str, Any]]:
    if not mapping:
        return series.copy(), {
            "strategy": "empty_mapping_noop",
            "mapping_size": 0,
            "changed_count": 0,
            "remaining_bad_count": None,
        }

    valid_codes = np.asarray(sorted(mapping), dtype=int)
    out_values: list[str] = []
    changed = 0
    support_set = set(mapping.values())
    for value in series.tolist():
        if pd.isna(value):
            sval = NAN_SENTINEL
        else:
            sval = str(value)
            if sval == "nan":
                sval = NAN_SENTINEL

        if sval in support_set:
            out_values.append(sval)
            continue

        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(num):
            code = int(valid_codes[0])
        else:
            code = int(round(float(num)))
            if code < int(valid_codes.min()):
                code = int(valid_codes.min())
            if code > int(valid_codes.max()):
                code = int(valid_codes.max())
            if code not in mapping:
                code = int(valid_codes[np.argmin(np.abs(valid_codes - code))])
        repaired = mapping[code]
        out_values.append(repaired)
        if repaired != sval:
            changed += 1

    out = pd.Series(out_values, index=series.index, name=series.name)
    remaining_bad_count = int((~out.isin(list(support_set))).sum())
    return out, {
        "strategy": "decode_recovery_round_clamp_inverse_map",
        "mapping_size": len(mapping),
        "mapping_sample": json.dumps({str(k): mapping[k] for k in list(mapping)[:10]}, ensure_ascii=False),
        "changed_count": int(changed),
        "remaining_bad_count": remaining_bad_count,
    }


def _support_projection_spec(raw_series: pd.Series) -> dict[str, Any]:
    support = _normalize_cat_series(raw_series)
    pairs: dict[float, str] = {}
    for sval in pd.unique(support):
        num = pd.to_numeric(pd.Series([sval]), errors="coerce").iloc[0]
        if pd.isna(num):
            continue
        key = float(num)
        pairs.setdefault(key, str(sval))
    if not pairs:
        raise ValueError("no numeric support values available for projection")
    return {
        "__spec_type__": "numeric_support_projection",
        "support": sorted((k, v) for k, v in pairs.items()),
    }


def _repair_with_numeric_support(series: pd.Series, spec: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    support_pairs = list(spec["support"])
    support_nums = np.asarray([float(k) for k, _ in support_pairs], dtype=float)
    support_vals = [str(v) for _, v in support_pairs]
    support_set = set(support_vals)
    out_values: list[str] = []
    changed = 0
    for value in series.tolist():
        sval = NAN_SENTINEL if pd.isna(value) else str(value)
        if sval == "nan":
            sval = NAN_SENTINEL
        if sval in support_set:
            out_values.append(sval)
            continue
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(num):
            repaired = support_vals[0]
        else:
            idx = int(np.argmin(np.abs(support_nums - float(num))))
            repaired = support_vals[idx]
        out_values.append(repaired)
        if repaired != sval:
            changed += 1

    out = pd.Series(out_values, index=series.index, name=series.name)
    remaining_bad_count = int((~out.isin(list(support_set))).sum())
    return out, {
        "strategy": "numeric_support_projection",
        "mapping_size": len(support_vals),
        "mapping_sample": json.dumps(
            {str(k): v for k, v in support_pairs[:10]}, ensure_ascii=False
        ),
        "changed_count": int(changed),
        "remaining_bad_count": remaining_bad_count,
    }


def _deterministic_sorted_unique_mapping(raw_series: pd.Series) -> dict[int, str]:
    cats = sorted(pd.unique(_normalize_cat_series(raw_series)))
    return {i: str(v) for i, v in enumerate(cats)}


def _deterministic_factorize_sort_mapping(raw_series: pd.Series) -> dict[int, str]:
    # pd.factorize(sort=True) code order is the sorted unique order.
    cats = sorted(pd.unique(_normalize_cat_series(raw_series)))
    return {i: str(v) for i, v in enumerate(cats)}


def _tabdiff_tabbyflow_mapping_for_column(run_dir: Path, bad_col: str) -> dict[int, str]:
    bundle_info = next((run_dir / "tabular_bundle").glob("*/info.json"))
    info = _load_json(bundle_info)
    bundle_dir = bundle_info.parent
    train_raw_path = bundle_dir / "train.csv"
    if not train_raw_path.exists():
        train_raw_path = run_dir / "staged" / "public" / "train.csv"
    if not train_raw_path.exists():
        raise FileNotFoundError(f"missing raw train csv for {run_dir}")
    raw_train = _load_df(train_raw_path)

    column_names = info["column_names"]
    num_col_idx = [int(x) for x in info.get("num_col_idx", [])]
    cat_col_idx = [int(x) for x in info.get("cat_col_idx", [])]
    num_col_names = [column_names[i] for i in num_col_idx]
    cat_col_names = [column_names[i] for i in cat_col_idx]
    target_idx = int(info["target_col_idx"][0])
    target_name = column_names[target_idx]

    if bad_col == target_name:
        y_train = np.load(bundle_dir / "y_train.npy", allow_pickle=True)
        return _mapping_from_raw_and_codes(raw_train[bad_col], _as_int_series(y_train))

    if bad_col in cat_col_names:
        cat_pos = cat_col_names.index(bad_col)
        x_cat = np.load(bundle_dir / "X_cat_train.npy", allow_pickle=True)
        if x_cat.ndim == 1:
            x_cat = x_cat.reshape(-1, 1)
        return _mapping_from_raw_and_codes(raw_train[bad_col], _as_int_series(x_cat[:, cat_pos]))

    if bad_col in num_col_names:
        return _support_projection_spec(raw_train[bad_col])

    raise KeyError(
        f"{bad_col} not found among tabdiff/tabbyflow feature columns "
        f"(num={num_col_names}, cat={cat_col_names}, target={target_name})"
    )


def _tabsyn_mapping_for_column(run_dir: Path, bad_col: str) -> dict[int, str]:
    data_info = next((run_dir / "data").glob("*/info.json"))
    data_dir = data_info.parent
    info = _load_json(data_info)
    staged_public = run_dir / "staged" / "public"
    raw_train = _load_df(staged_public / "train.csv")
    enc_train = _load_df(data_dir / "train.csv")

    raw_source = raw_train
    if len(raw_source) != len(enc_train):
        extra_frames = [raw_train]
        val_path = staged_public / "val.csv"
        test_path = staged_public / "test.csv"
        if val_path.exists():
            extra_frames.append(_load_df(val_path))
            if sum(len(df) for df in extra_frames) == len(enc_train):
                raw_source = pd.concat(extra_frames, ignore_index=True)
        if len(raw_source) != len(enc_train) and test_path.exists():
            trial_frames = extra_frames + [_load_df(test_path)]
            if sum(len(df) for df in trial_frames) == len(enc_train):
                raw_source = pd.concat(trial_frames, ignore_index=True)
        if len(raw_source) != len(enc_train):
            raise ValueError(
                f"unable to align tabsyn raw/encoded train lengths: "
                f"train={len(raw_train)} encoded={len(enc_train)}"
            )

    column_names = info["column_names"]
    target_idx = int(info["target_col_idx"][0])
    target_name = column_names[target_idx]

    if bad_col == target_name:
        try:
            return _mapping_from_raw_and_codes(raw_source[bad_col], enc_train[bad_col])
        except ValueError:
            return _support_projection_spec(raw_source[bad_col])
    if bad_col not in raw_source.columns or bad_col not in enc_train.columns:
        raise KeyError(f"{bad_col} not found in tabsyn raw/encoded train columns")
    try:
        return _mapping_from_raw_and_codes(raw_source[bad_col], enc_train[bad_col])
    except ValueError:
        return _support_projection_spec(raw_source[bad_col])


def _deterministic_mapping_for_column(run_dir: Path, bad_col: str, mode: str) -> dict[int, str]:
    raw_train = _load_df(run_dir / "staged" / "public" / "train.csv")
    if mode == "sorted_unique":
        return _deterministic_sorted_unique_mapping(raw_train[bad_col])
    if mode == "factorize_sort":
        return _deterministic_factorize_sort_mapping(raw_train[bad_col])
    raise ValueError(mode)


def _effective_recovery_class(row: pd.Series) -> str:
    model = str(row["model"])
    model_defaults = {
        "tabdiff": "recoverable_exact_saved_bundle",
        "tabbyflow": "recoverable_exact_saved_bundle",
        "tabsyn": "recoverable_exact_raw_vs_encoded_train",
        "tabddpm": "recoverable_deterministic_sorted_unique",
        "forestdiffusion": "recoverable_deterministic_factorize_sort",
        "bayesnet": "recoverable_deterministic_sorted_unique",
    }
    default = model_defaults.get(model)
    cls = str(row["recovery_class"])
    if model in {"tabdiff", "tabbyflow"}:
        if cls in {
            "recoverable_exact_saved_bundle",
            "recoverable_exact_staged_train_plus_xcat",
        }:
            return cls
        return default or cls
    if default is not None:
        return default
    return cls


def _mapping_for_column(row: pd.Series, bad_col: str) -> dict[int, str]:
    run_dir = Path(row["run_dir"])
    cls = _effective_recovery_class(row)
    if cls in {"recoverable_exact_saved_bundle", "recoverable_exact_staged_train_plus_xcat"}:
        return _tabdiff_tabbyflow_mapping_for_column(run_dir, bad_col)
    if cls == "recoverable_exact_raw_vs_encoded_train":
        return _tabsyn_mapping_for_column(run_dir, bad_col)
    if cls == "recoverable_deterministic_sorted_unique":
        return _deterministic_mapping_for_column(run_dir, bad_col, "sorted_unique")
    if cls == "recoverable_deterministic_factorize_sort":
        return _deterministic_mapping_for_column(run_dir, bad_col, "factorize_sort")
    raise ValueError(f"unsupported recovery_class: {cls}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feasibility-csv", required=True)
    ap.add_argument("--resolution-csv", required=True)
    ap.add_argument("--manifest-out", required=True)
    args = ap.parse_args()

    targets = pd.read_csv(args.feasibility_csv)
    targets = targets[targets["recoverable"] == "yes"].copy()
    resolution = pd.read_csv(args.resolution_csv)[
        ["bucket", "dataset", "model", "run_id", "run_dir", "synthetic_csv", "bad_discrete_cols"]
    ].drop_duplicates(subset=["bucket", "dataset", "model", "run_id"])
    targets = targets.merge(
        resolution,
        on=["bucket", "dataset", "model", "run_id", "run_dir", "bad_discrete_cols"],
        how="left",
    )
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, str, str]] = set()
    if manifest_out.exists():
        existing = pd.read_csv(manifest_out)
        if not existing.empty:
            results = existing.to_dict("records")
            done_keys = {
                (str(r["bucket"]), str(r["dataset"]), str(r["model"]), str(r["run_id"]))
                for _, r in existing.iterrows()
                if str(r.get("status", "")).startswith("repaired")
            }
            if done_keys:
                targets = targets[
                    ~targets.apply(
                        lambda r: (
                            str(r["bucket"]),
                            str(r["dataset"]),
                            str(r["model"]),
                            str(r["run_id"]),
                        )
                        in done_keys,
                        axis=1,
                    )
                ].reset_index(drop=True)

    total = len(targets)
    for idx, (_, row) in enumerate(targets.iterrows(), start=1):
        dataset = row["dataset"]
        model = row["model"]
        run_id = row["run_id"]
        run_dir = Path(row["run_dir"])
        target_csv = Path(row["synthetic_csv"])
        print(
            f"[fix-decode] {idx}/{total} bucket={row['bucket']} dataset={dataset} model={model} run_id={run_id}",
            flush=True,
        )
        record: dict[str, Any] = {
            "bucket": row["bucket"],
            "dataset": dataset,
            "model": model,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "target_csv": str(target_csv),
            "recovery_class": row["recovery_class"],
            "effective_recovery_class": _effective_recovery_class(row),
            "status": "pending",
        }

        if not run_dir.exists():
            record["status"] = "failed"
            record["error"] = "run_dir_missing"
            results.append(record)
            if idx % 10 == 0 or idx == total:
                _write_results_checkpoint(results, manifest_out)
            continue

        if pd.isna(row.get("synthetic_csv")):
            record["status"] = "failed"
            record["error"] = "synthetic_csv_missing_after_merge"
            results.append(record)
            if idx % 10 == 0 or idx == total:
                _write_results_checkpoint(results, manifest_out)
            continue

        try:
            syn_df = _load_df(target_csv)
            write_target, is_overlay = _prepare_output_target(run_dir, target_csv)

            col_reports: list[dict[str, Any]] = []
            repaired = False
            for col in [c for c in str(row["bad_discrete_cols"]).split("|") if c]:
                if col not in syn_df.columns:
                    col_reports.append({"column": col, "status": "missing_in_synthetic_csv"})
                    continue
                mapping = _mapping_for_column(row, col)
                if isinstance(mapping, dict) and mapping.get("__spec_type__") == "numeric_support_projection":
                    fixed, info = _repair_with_numeric_support(syn_df[col], mapping)
                else:
                    fixed, info = _repair_with_mapping(syn_df[col], mapping)
                syn_df[col] = fixed
                info["column"] = col
                info["status"] = "ok" if info["remaining_bad_count"] == 0 else "remaining_bad_values"
                col_reports.append(info)
                repaired = True

            if repaired:
                write_target.parent.mkdir(parents=True, exist_ok=True)
                syn_df.to_csv(write_target, index=False)
                final_csv = _finalize_overlay(run_dir, write_target) if is_overlay else write_target
            else:
                final_csv = write_target

            remaining = sum(
                int(item.get("remaining_bad_count") or 0)
                for item in col_reports
                if item.get("remaining_bad_count") is not None
            )
            record["status"] = "repaired" if repaired and remaining == 0 else "repaired_with_remaining"
            record["column_reports_json"] = json.dumps(col_reports, ensure_ascii=False)
            record["remaining_bad_total"] = remaining
            record["final_csv_path"] = str(final_csv)
        except Exception as e:
            record["status"] = "failed"
            record["error"] = f"{type(e).__name__}: {e}"
            print(
                f"[fix-decode][error] bucket={row['bucket']} dataset={dataset} model={model} run_id={run_id} error={type(e).__name__}: {e}",
                flush=True,
            )

        results.append(record)
        _write_results_checkpoint(results, manifest_out)
        print(
            f"[fix-decode][done] {idx}/{total} bucket={row['bucket']} dataset={dataset} model={model} run_id={run_id} status={record['status']}",
            flush=True,
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(manifest_out, index=False)
    summary = {
        "target_count": int(len(targets)),
        "repaired_count": int((out_df["status"] == "repaired").sum()),
        "repaired_with_remaining_count": int((out_df["status"] == "repaired_with_remaining").sum()),
        "failed_count": int((out_df["status"] == "failed").sum()),
    }
    manifest_out.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
