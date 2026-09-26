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


REPLACE = "replace_with_clean_alternative"
PROJECTION_TYPES = {
    "round_then_snap_to_train_support",
    "columnwise_threshold_or_nearest_support_projection",
    "round_threshold_and_support_projection",
}


def _load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _is_numeric_like(v: Any) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


def _is_int_like(v: Any) -> bool:
    try:
        fv = float(v)
        return math.isfinite(fv) and abs(fv - round(fv)) < 1e-9
    except Exception:
        return False


def _nearest_support(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    if support.size == 0:
        return values
    diffs = np.abs(values[:, None] - support[None, :])
    idx = np.argmin(diffs, axis=1)
    return support[idx]


def _repair_series(series: pd.Series, train_series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    support_raw = [v for v in pd.unique(train_series.dropna())]
    if not support_raw:
        return series.copy(), {
            "strategy": "empty_support_noop",
            "support_size": 0,
            "changed_count": 0,
            "remaining_bad_count": 0,
        }

    if not all(_is_numeric_like(v) for v in support_raw):
        return series.copy(), {
            "strategy": "nonnumeric_support_unhandled",
            "support_size": len(support_raw),
            "changed_count": 0,
            "remaining_bad_count": None,
        }

    support = np.asarray([float(v) for v in support_raw], dtype=float)
    support = np.unique(support)
    out = pd.to_numeric(series, errors="coerce").astype(float)
    orig = out.copy()

    support_is_int = all(_is_int_like(v) for v in support)
    support_ints = {int(round(v)) for v in support} if support_is_int else set()

    finite_mask = np.isfinite(out.to_numpy())
    arr = out.to_numpy(copy=True)

    strategy = "nearest_value_projection_to_numeric_support"
    if support_is_int and support_ints == {0, 1}:
        strategy = "binary_threshold_to_0_1_then_support_check"
        arr[finite_mask] = (arr[finite_mask] >= 0.5).astype(float)
    elif support_is_int and support_ints == {-1, 1}:
        strategy = "binary_sign_threshold_to_-1_1_then_support_check"
        arr[finite_mask] = np.where(arr[finite_mask] >= 0.0, 1.0, -1.0)
    elif support_is_int:
        strategy = "round_then_snap_to_allowed_integer_support"
        arr[finite_mask] = np.round(arr[finite_mask])
        arr[finite_mask] = _nearest_support(arr[finite_mask], support)
    else:
        arr[finite_mask] = _nearest_support(arr[finite_mask], support)

    out = pd.Series(arr, index=series.index, name=series.name)

    if support_is_int:
        out = out.round().astype("Int64")
        bad_mask = ~out.isna() & ~out.astype(float).isin([float(x) for x in support_ints])
    else:
        bad_mask = ~out.isna() & ~out.astype(float).isin(list(support))

    changed_mask = (orig != pd.to_numeric(out, errors="coerce")).fillna(False)
    changed_count = int(changed_mask.sum())
    remaining_bad_count = int(bad_mask.sum())
    return out, {
        "strategy": strategy,
        "support_size": int(len(support)),
        "support_sample": [str(v) for v in support[:10]],
        "changed_count": changed_count,
        "remaining_bad_count": remaining_bad_count,
    }


def _select_clean_source(
    row: pd.Series,
    val6000: pd.DataFrame,
    best5090: pd.DataFrame,
) -> tuple[str | None, str | None]:
    combo = str(row["combo_matrix_status"])
    dataset = row["dataset"]
    model = row["model"]
    bucket = row["bucket"]

    if "5-1000" in combo:
        cand = best5090[
            (best5090["dataset"] == dataset)
            & (best5090["model"] == model)
            & (best5090["status"] == "ok")
        ]
        if not cand.empty:
            picked = cand.iloc[0]
            return "5090", str(picked["synthetic_csv"])

    if "6-1000" in combo:
        cand = val6000[
            (val6000["dataset"] == dataset)
            & (val6000["model"] == model)
            & (val6000["status"] == "ok")
        ].copy()
        if not cand.empty:
            cand["bucket_rank"] = (cand["bucket"] != bucket).astype(int)
            cand = cand.sort_values(["bucket_rank", "run_id"])
            picked = cand.iloc[0]
            return "6000", str(picked["synthetic_csv"])

    return None, None


def _copy_csv(src: Path, dst: Path) -> None:
    df = _load_df(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


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
        final_csv = tmp / target_name
        return final_csv, True

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution-csv", required=True)
    ap.add_argument("--validation-6000-csv", required=True)
    ap.add_argument("--validation-5090-csv", required=True)
    ap.add_argument("--rowdata-root", required=True)
    ap.add_argument("--manifest-out", required=True)
    args = ap.parse_args()

    resolution = pd.read_csv(args.resolution_csv)
    val6000 = pd.read_csv(args.validation_6000_csv)
    best5090 = pd.read_csv(args.validation_5090_csv)
    rowdata_root = Path(args.rowdata_root)
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    targets = resolution[
        resolution["best_resolution"].isin({REPLACE, *PROJECTION_TYPES})
    ].copy()

    results: list[dict[str, Any]] = []
    train_cache: dict[str, pd.DataFrame] = {}

    total = len(targets)
    for idx, (_, row) in enumerate(targets.iterrows(), start=1):
        dataset = row["dataset"]
        model = row["model"]
        run_id = row["run_id"]
        target_csv = Path(row["synthetic_csv"])
        run_dir = Path(row["run_dir"])
        if idx % 25 == 0 or idx == 1 or idx == total:
            print(
                f"[fix-discrete] {idx}/{total} bucket={row['bucket']} dataset={dataset} model={model} run_id={run_id}",
                flush=True,
            )
        record: dict[str, Any] = {
            "bucket": row["bucket"],
            "dataset": dataset,
            "model": model,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "target_csv": str(target_csv),
            "best_resolution": row["best_resolution"],
            "status": "pending",
        }

        if not run_dir.exists():
            if row["best_resolution"] == REPLACE and run_dir.parent.exists():
                sibling_runs = sorted(
                    p
                    for p in run_dir.parent.iterdir()
                    if p.name != run_dir.name and (p.is_dir() or p.is_symlink())
                )
                if sibling_runs:
                    record["status"] = "already_replaced"
                    record["replacement_source_server"] = "existing_in_syn_data_success"
                    record["final_csv_path"] = str(sibling_runs[0])
                    results.append(record)
                    if idx % 10 == 0 or idx == total:
                        _write_results_checkpoint(results, manifest_out)
                    continue
            record["status"] = "failed"
            record["error"] = "run_dir_missing"
            results.append(record)
            if idx % 10 == 0 or idx == total:
                _write_results_checkpoint(results, manifest_out)
            continue

        if row["best_resolution"] == REPLACE:
            source_server, source_csv = _select_clean_source(row, val6000, best5090)
            if not source_csv:
                record["status"] = "failed"
                record["error"] = "no_clean_source_found"
                results.append(record)
                if idx % 10 == 0 or idx == total:
                    _write_results_checkpoint(results, manifest_out)
                continue
            src = Path(source_csv)
            write_target, is_overlay = _prepare_output_target(run_dir, target_csv)
            _copy_csv(src, write_target)
            final_csv = _finalize_overlay(run_dir, write_target) if is_overlay else write_target
            record["status"] = "replaced"
            record["replacement_source_server"] = source_server
            record["replacement_source_csv"] = str(src)
            record["final_csv_path"] = str(final_csv)
            results.append(record)
            if idx % 10 == 0 or idx == total:
                _write_results_checkpoint(results, manifest_out)
            continue

        if dataset not in train_cache:
            train_csv = rowdata_root / dataset / f"{dataset}-train.csv"
            train_cache[dataset] = _load_df(train_csv)
        train_df = train_cache[dataset]
        syn_df = _load_df(target_csv)
        write_target, is_overlay = _prepare_output_target(run_dir, target_csv)

        col_reports: list[dict[str, Any]] = []
        repaired = False
        for col in [c for c in str(row["bad_discrete_cols"]).split("|") if c]:
            if col not in syn_df.columns or col not in train_df.columns:
                col_reports.append(
                    {"column": col, "status": "missing_in_csv_or_train"}
                )
                continue
            fixed, info = _repair_series(syn_df[col], train_df[col])
            syn_df[col] = fixed
            info["column"] = col
            info["status"] = (
                "ok" if info["remaining_bad_count"] == 0 else "remaining_bad_values"
            )
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
        results.append(record)
        if idx % 10 == 0 or idx == total:
            _write_results_checkpoint(results, manifest_out)

    out_df = pd.DataFrame(results)
    out_df.to_csv(manifest_out, index=False)
    summary = {
        "target_count": int(len(targets)),
        "replaced_count": int((out_df["status"] == "replaced").sum()),
        "repaired_count": int((out_df["status"] == "repaired").sum()),
        "repaired_with_remaining_count": int(
            (out_df["status"] == "repaired_with_remaining").sum()
        ),
        "failed_count": int((out_df["status"] == "failed").sum()),
    }
    manifest_out.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
