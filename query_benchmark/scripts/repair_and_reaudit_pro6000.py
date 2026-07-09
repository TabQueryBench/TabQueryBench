from __future__ import annotations

import csv
import json
import gc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "artifacts" / "full_synthetic_processing_audit_20260502_134237"
DATA_ROOT = REPO_ROOT / "data"
SKIP_DATASETS = {"c21", "n13"}
MISSING_TEXT = {"", "null", "none", "nan", "na", "n/a", "<null>"}
MODEL_DECODE_FIRST = {"tabsyn"}
MODEL_DECODE_SORTED = {"tabddpm", "tabpfgen"}
MODEL_DECODE_SOFT_INTEGER_TOLERANCE = {"forestdiffusion": 0.25}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_TEXT


def _missing_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if pd.api.types.is_numeric_dtype(series):
        return mask
    normalized = series.astype("string").str.strip().str.lower()
    return mask | normalized.isin(MISSING_TEXT)


def _parse_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(str(value).strip())
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _numeric_like_ratio(series: pd.Series) -> float:
    clean = series[~_missing_mask(series)]
    if clean.empty:
        return 0.0
    parsed = pd.to_numeric(clean.astype(str), errors="coerce")
    return float(parsed.notna().mean())


def _integer_like_ratio(series: pd.Series) -> float:
    clean = series[~_missing_mask(series)]
    if clean.empty:
        return 0.0
    parsed = pd.to_numeric(clean.astype(str), errors="coerce").dropna()
    if parsed.empty:
        return 0.0
    return float(((parsed - parsed.round()).abs() < 1e-8).mean())


def _text_overlap_ratio(real_series: pd.Series, syn_series: pd.Series) -> float | None:
    left = {str(v) for v in real_series[~_missing_mask(real_series)].tolist()}
    right = {str(v) for v in syn_series[~_missing_mask(syn_series)].tolist()}
    if not right:
        return None
    return len(left & right) / len(right)


def _load_field_meta(dataset_id: str) -> dict[str, dict[str, Any]]:
    path = DATA_ROOT / dataset_id / "metadata_core" / "field_registry.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for field in payload.get("fields", []):
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        out[name] = {
            "semantic_type": str(field.get("semantic_type") or "").strip().lower(),
            "declared_type": str(field.get("declared_type") or "").strip().lower(),
            "value_order": [str(item) for item in field.get("value_order") or []],
        }
    return out


def _bucket_for_column(meta: dict[str, Any], real_series: pd.Series) -> str:
    semantic = str(meta.get("semantic_type") or "").lower()
    declared = str(meta.get("declared_type") or "").lower()
    raw_numeric_ratio = _numeric_like_ratio(real_series)
    raw_integer_ratio = _integer_like_ratio(real_series)

    textual_markers = [
        "identifier_string",
        "dna_sequence",
        "categorical",
        "boolean",
        "text",
        "free_text",
        "nominal",
        "identifier",
    ]
    if any(marker in semantic for marker in textual_markers):
        if raw_numeric_ratio < 0.95:
            return "textual_categorical"
        if raw_integer_ratio >= 0.95 or meta.get("value_order"):
            return "discrete_numeric"
        return "continuous_numeric"

    if "numeric_discrete" in semantic or "identifier_numeric" in semantic:
        if raw_numeric_ratio < 0.95:
            return "textual_categorical"
        return "discrete_numeric" if raw_integer_ratio >= 0.95 or meta.get("value_order") else "continuous_numeric"

    if "numeric" in semantic or declared in {"numeric", "integer", "float", "double", "decimal"}:
        if raw_numeric_ratio >= 0.95:
            return "discrete_numeric" if raw_integer_ratio >= 0.95 else "continuous_numeric"
        return "textual_categorical"

    if raw_numeric_ratio < 0.95:
        return "textual_categorical"
    if raw_integer_ratio >= 0.95:
        return "discrete_numeric"
    return "continuous_numeric"


def _tabsyn_order(real_series: pd.Series) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for value in real_series.tolist():
        token = "<NA>" if _is_missing(value) else str(value)
        if token not in seen:
            seen.add(token)
            order.append(token)
    return order


def _sorted_order(real_series: pd.Series) -> list[str]:
    values = {str(value) for value in real_series[~_missing_mask(real_series)].tolist()}
    return sorted(values, key=str)


def _load_dataset_context(dataset_id: str) -> dict[str, Any]:
    real_path = DATA_ROOT / dataset_id / f"{dataset_id}-train.csv"
    field_meta = _load_field_meta(dataset_id)
    real_df = pd.read_csv(real_path, low_memory=False)
    columns: dict[str, dict[str, Any]] = {}
    for column in real_df.columns:
        real_series = real_df[column]
        meta = field_meta.get(column, {})
        bucket = _bucket_for_column(meta, real_series)
        info: dict[str, Any] = {
            "meta": meta,
            "bucket": bucket,
            "real_missing_rate": float(_missing_mask(real_series).mean()),
        }
        if bucket == "textual_categorical":
            missing_mask = _missing_mask(real_series)
            info["real_text_set"] = {str(v) for v in real_series[~missing_mask].tolist()}
            info["tabsyn_order"] = _tabsyn_order(real_series)
            info["sorted_order"] = _sorted_order(real_series)
            info["real_numeric_ratio"] = _numeric_like_ratio(real_series)
        elif bucket == "discrete_numeric":
            info["support"] = _numeric_support(meta, real_series)
            info["real_integer_like_ratio"] = _integer_like_ratio(real_series)
        columns[column] = info
    return {"dataset_id": dataset_id, "real_df": real_df, "columns": columns}


def _decode_textual_column(model_id: str, column_info: dict[str, Any], syn_series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    syn_missing_mask = _missing_mask(syn_series)
    syn_non_missing = syn_series[~syn_missing_mask]
    parsed = pd.to_numeric(syn_non_missing.astype(str), errors="coerce")
    integer_tolerance = float(MODEL_DECODE_SOFT_INTEGER_TOLERANCE.get(model_id, 1e-8))
    if (
        syn_non_missing.empty
        or parsed.notna().mean() < 0.95
        or ((parsed.dropna() - parsed.dropna().round()).abs() <= integer_tolerance).mean() < 0.95
    ):
        return syn_series, {"changed_cells": 0, "restored_missing": 0, "decoded_column": False, "strategy": ""}

    if model_id in MODEL_DECODE_FIRST:
        order = column_info["tabsyn_order"]
        strategy = "tabsyn_first_seen"
        allow_out_of_range_missing = False
    elif model_id in MODEL_DECODE_SORTED:
        order = column_info["sorted_order"]
        strategy = "sorted_unique"
        allow_out_of_range_missing = True
    else:
        return syn_series, {"changed_cells": 0, "restored_missing": 0, "decoded_column": False, "strategy": ""}
    if not order:
        return syn_series, {"changed_cells": 0, "restored_missing": 0, "decoded_column": False, "strategy": strategy}

    parsed_all = pd.to_numeric(syn_series, errors="coerce")
    int_mask = parsed_all.notna() & ((parsed_all - parsed_all.round()).abs() <= integer_tolerance)
    indices = parsed_all.round().astype("Int64")
    in_range_mask = int_mask & indices.ge(0) & indices.lt(len(order))
    modify_mask = in_range_mask.copy()
    updated = syn_series.astype("object").copy()

    if in_range_mask.any():
        mapping = {idx: (pd.NA if token == "<NA>" else token) for idx, token in enumerate(order)}
        updated.loc[in_range_mask] = indices.loc[in_range_mask].map(mapping).astype("object")

    restored_missing = int((in_range_mask & updated.isna()).sum())

    if allow_out_of_range_missing:
        out_of_range_mask = int_mask & ~in_range_mask
        if out_of_range_mask.any():
            updated.loc[out_of_range_mask] = pd.NA
            restored_missing += int(out_of_range_mask.sum())
            modify_mask = modify_mask | out_of_range_mask

    changed_cells = int(modify_mask.sum())
    return updated, {
        "changed_cells": changed_cells,
        "restored_missing": restored_missing,
        "decoded_column": changed_cells > 0 or restored_missing > 0,
        "strategy": strategy,
    }


def _numeric_support(meta: dict[str, Any], real_series: pd.Series) -> np.ndarray:
    value_order = meta.get("value_order") or []
    ordered: list[float] = []
    for item in value_order:
        number = _parse_float(item)
        if number is not None:
            ordered.append(number)
    if ordered:
        return np.array(sorted(set(ordered)), dtype=float)
    parsed = pd.to_numeric(real_series[~_missing_mask(real_series)].astype(str), errors="coerce").dropna()
    return np.array(sorted(set(parsed.tolist())), dtype=float)


def _project_to_support(support: np.ndarray, syn_series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    if support.size == 0:
        return syn_series, {"changed_cells": 0, "strategy": ""}

    parsed = pd.to_numeric(syn_series.astype(str), errors="coerce")
    if parsed.notna().mean() < 0.80:
        return syn_series, {"changed_cells": 0, "strategy": ""}

    support.sort()
    values = parsed.to_numpy(dtype=float, na_value=np.nan)
    valid_mask = np.isfinite(values)
    if not valid_mask.any():
        return syn_series, {"changed_cells": 0, "strategy": ""}

    valid_values = values[valid_mask]
    insert = np.searchsorted(support, valid_values, side="left")
    projected = np.empty_like(valid_values)

    left_edge = insert <= 0
    right_edge = insert >= support.size
    middle = ~(left_edge | right_edge)

    projected[left_edge] = support[0]
    projected[right_edge] = support[-1]
    if middle.any():
        left_vals = support[insert[middle] - 1]
        right_vals = support[insert[middle]]
        projected[middle] = np.where(
            np.abs(valid_values[middle] - left_vals) <= np.abs(valid_values[middle] - right_vals),
            left_vals,
            right_vals,
        )

    changed_cells = int((np.abs(valid_values - projected) > 1e-8).sum())
    if changed_cells == 0:
        return syn_series, {"changed_cells": 0, "strategy": ""}

    updated = syn_series.astype("object").copy()
    output_values: list[Any] = []
    for item in projected.tolist():
        if abs(item - round(item)) < 1e-8:
            output_values.append(int(round(item)))
        else:
            output_values.append(float(item))
    updated.loc[valid_mask] = output_values
    return updated, {"changed_cells": changed_cells, "strategy": "nearest_support_projection"}


def _audit_textual(column_info: dict[str, Any], syn_series: pd.Series) -> tuple[str, str]:
    syn_missing_mask = _missing_mask(syn_series)
    syn_non_missing = syn_series[~syn_missing_mask]
    if syn_non_missing.empty:
        return "all_missing", "synthetic_all_missing"
    syn_numeric_ratio = _numeric_like_ratio(syn_series)
    real_numeric_ratio = float(column_info.get("real_numeric_ratio", 0.0))
    right = {str(v) for v in syn_non_missing.tolist()}
    if not right:
        overlap = None
    else:
        overlap = len(column_info.get("real_text_set", set()) & right) / len(right)
    if real_numeric_ratio < 0.95 and syn_numeric_ratio > 0.95 and (overlap or 0.0) == 0.0:
        return "suspected_not_decoded", "real_textual_syn_numeric_no_overlap"
    if overlap is None:
        return "unknown", "no_non_missing_values"
    if overlap >= 0.95:
        return "verified_direct_compare", ""
    if overlap >= 0.50:
        return "likely_ok_direct_compare", f"overlap={overlap:.3f}"
    return "has_overlap_gap", f"overlap={overlap:.3f}"


def _audit_discrete(column_info: dict[str, Any], syn_series: pd.Series) -> tuple[str, str]:
    support = column_info["support"]
    syn_non_missing = syn_series[~_missing_mask(syn_series)]
    if syn_non_missing.empty:
        return "all_missing", "synthetic_all_missing"
    parsed = pd.to_numeric(syn_non_missing.astype(str), errors="coerce")
    if parsed.isna().any():
        return "contains_non_numeric_values", "failed_numeric_parse"
    valid = parsed.to_numpy(dtype=float)
    if support.size and support.size <= 512:
        support_set = set(float(item) for item in support.tolist())
        miss = sum(1 for item in valid.tolist() if float(item) not in support_set)
        if miss:
            return "outside_support", f"outside_support_ratio={miss / len(valid):.3f}"
        return "in_support", ""
    if float(column_info.get("real_integer_like_ratio", 0.0)) >= 0.95:
        frac_ratio = float(((valid - np.round(valid)) > 1e-8).mean())
        if frac_ratio > 0.0:
            return "contains_fractional_values", f"fractional_ratio={frac_ratio:.3f}"
        return "integer_like", ""
    min_support = float(np.nanmin(support)) if support.size else float(np.nanmin(valid))
    max_support = float(np.nanmax(support)) if support.size else float(np.nanmax(valid))
    outside = sum(1 for item in valid.tolist() if item < min_support or item > max_support)
    if outside:
        return "outside_range", f"outside_range_ratio={outside / len(valid):.3f}"
    return "within_range", ""


def _audit_missing(column_info: dict[str, Any], syn_series: pd.Series) -> tuple[str, str]:
    real_missing_rate = float(column_info.get("real_missing_rate", 0.0))
    syn_missing_rate = float(syn_series.isna().mean())
    if real_missing_rate <= 0:
        return "not_applicable", ""
    if syn_missing_rate > 0:
        return "missing_present", f"syn_missing_rate={syn_missing_rate:.4f}"
    return "missing_lost", "no_missing_values_present"


def _repair_asset(row: dict[str, str], dataset_context: dict[str, Any]) -> dict[str, Any]:
    dataset_id = row["dataset_id"]
    model_id = row["model_id"]
    syn_path = REPO_ROOT / row["synthetic_csv_path"].replace("\\", "/")
    real_df = dataset_context["real_df"]
    columns_info = dataset_context["columns"]
    syn_df = pd.read_csv(syn_path, low_memory=False)

    actions: list[str] = []
    needs_textual_audit = row.get("inverse_encoding_status", "") != "repaired_or_verified" or model_id in MODEL_DECODE_FIRST | MODEL_DECODE_SORTED
    needs_discrete_audit = row.get("discrete_numeric_status", "") == "has_discrete_numeric_issues"
    needs_missing_audit = row.get("missing_status", "") == "has_missing_issues"
    decoded_columns = 0
    projected_columns = 0
    missing_columns = 0
    changed_cells = 0
    trimmed_rows = 0

    if len(syn_df) > len(real_df):
        trimmed_rows = len(syn_df) - len(real_df)
        syn_df = syn_df.iloc[: len(real_df)].copy()
        actions.append(f"trimmed_extra_rows={trimmed_rows}")

    for column in real_df.columns:
        if column not in syn_df.columns:
            continue
        column_info = columns_info.get(column, {})
        bucket = column_info.get("bucket", "continuous_numeric")
        if bucket == "textual_categorical":
            updated, info = _decode_textual_column(model_id, column_info, syn_df[column])
            if info["decoded_column"]:
                syn_df[column] = updated
                decoded_columns += 1
                changed_cells += int(info["changed_cells"])
                if int(info["restored_missing"]) > 0:
                    missing_columns += 1
                    needs_missing_audit = True
                actions.append(f"decoded:{column}:{info['strategy']}")
        elif bucket == "discrete_numeric":
            updated, info = _project_to_support(column_info["support"], syn_df[column])
            if int(info["changed_cells"]) > 0:
                syn_df[column] = updated
                projected_columns += 1
                changed_cells += int(info["changed_cells"])
                needs_discrete_audit = True
                actions.append(f"projected:{column}:{info['strategy']}")

    if actions:
        syn_df.to_csv(syn_path, index=False)

    issue_notes: list[str] = []
    inverse_issue_columns = 0
    missing_issue_columns = 0
    discrete_issue_columns = 0
    verified_columns = 0

    for column in real_df.columns:
        if column not in syn_df.columns:
            issue_notes.append(f"missing_column:{column}")
            continue
        column_info = columns_info.get(column, {})
        bucket = column_info.get("bucket", "continuous_numeric")
        if bucket == "textual_categorical":
            if needs_textual_audit:
                inverse_status, inverse_note = _audit_textual(column_info, syn_df[column])
                if inverse_status in {"suspected_not_decoded", "has_overlap_gap", "unknown", "all_missing"}:
                    inverse_issue_columns += 1
                    issue_notes.append(f"{column}:{inverse_note or inverse_status}")
                else:
                    verified_columns += 1
            else:
                verified_columns += 1
        elif bucket == "discrete_numeric":
            if needs_discrete_audit:
                discrete_status, discrete_note = _audit_discrete(column_info, syn_df[column])
                if discrete_status not in {"in_support", "integer_like", "within_range"}:
                    discrete_issue_columns += 1
                    issue_notes.append(f"{column}:{discrete_note or discrete_status}")
                else:
                    verified_columns += 1
            else:
                verified_columns += 1
        else:
            verified_columns += 1

        if needs_missing_audit and float(column_info.get("real_missing_rate", 0.0)) > 0.0:
            missing_status, missing_note = _audit_missing(column_info, syn_df[column])
            if missing_status == "missing_lost":
                missing_issue_columns += 1
                issue_notes.append(f"{column}:{missing_note}")

    row_count_status = "row_match" if len(syn_df) == len(real_df) else "row_mismatch"
    completion_bucket = (
        "completed"
        if row_count_status == "row_match"
        and inverse_issue_columns == 0
        and missing_issue_columns == 0
        and discrete_issue_columns == 0
        else "remaining_problem"
    )

    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "run_id": row["run_id"],
        "synthetic_csv_path": row["synthetic_csv_path"],
        "completion_bucket": completion_bucket,
        "inverse_issue_columns": inverse_issue_columns,
        "missing_issue_columns": missing_issue_columns,
        "discrete_issue_columns": discrete_issue_columns,
        "row_count_status": row_count_status,
        "verified_columns": verified_columns,
        "decoded_columns": decoded_columns,
        "projected_columns": projected_columns,
        "missing_columns_restored": missing_columns,
        "trimmed_rows": trimmed_rows,
        "changed_cells": changed_cells,
        "actions": " | ".join(actions),
        "remaining_issues": " | ".join(issue_notes),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None_\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = [str(row.get(column, "")).replace("\n", "<br>") for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def main() -> None:
    pending_rows = _read_csv_rows(AUDIT_DIR / "pro6000_pending_local_repair.csv")
    working_rows = [row for row in pending_rows if row["dataset_id"] not in SKIP_DATASETS]
    working_rows.sort(key=lambda row: (row["dataset_id"], row["model_id"], row["run_id"]))

    checkpoint_path = AUDIT_DIR / "pro6000_local_repair_results_checkpoint.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    repaired_rows: list[dict[str, Any]] = []
    current_dataset_id = ""
    dataset_context: dict[str, Any] | None = None
    for index, row in enumerate(working_rows, start=1):
        dataset_id = row["dataset_id"]
        if dataset_id != current_dataset_id:
            dataset_context = _load_dataset_context(dataset_id)
            current_dataset_id = dataset_id
            gc.collect()
        repaired_rows.append(_repair_asset(row, dataset_context))
        _write_csv(checkpoint_path, repaired_rows)
        if index % 10 == 0 or index == len(working_rows):
            print(f"processed {index}/{len(working_rows)} assets", flush=True)

    completed_rows = [row for row in repaired_rows if row["completion_bucket"] == "completed"]
    remaining_rows = [row for row in repaired_rows if row["completion_bucket"] == "remaining_problem"]

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in repaired_rows:
        by_model[row["model_id"]].append(row)
        by_dataset[row["dataset_id"]].append(row)
        by_dataset_model[(row["dataset_id"], row["model_id"])].append(row)

    model_rows = []
    for model_id, rows in sorted(by_model.items()):
        model_rows.append(
            {
                "model_id": model_id,
                "asset_count": len(rows),
                "completed_assets": sum(1 for row in rows if row["completion_bucket"] == "completed"),
                "remaining_problem_assets": sum(1 for row in rows if row["completion_bucket"] == "remaining_problem"),
                "assets_with_inverse_issues": sum(1 for row in rows if int(row["inverse_issue_columns"]) > 0),
                "assets_with_missing_issues": sum(1 for row in rows if int(row["missing_issue_columns"]) > 0),
                "assets_with_discrete_issues": sum(1 for row in rows if int(row["discrete_issue_columns"]) > 0),
                "assets_with_row_mismatch": sum(1 for row in rows if row["row_count_status"] == "row_mismatch"),
            }
        )

    dataset_rows = []
    for dataset_id, rows in sorted(by_dataset.items()):
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "asset_count": len(rows),
                "completed_assets": sum(1 for row in rows if row["completion_bucket"] == "completed"),
                "remaining_problem_assets": sum(1 for row in rows if row["completion_bucket"] == "remaining_problem"),
                "assets_with_inverse_issues": sum(1 for row in rows if int(row["inverse_issue_columns"]) > 0),
                "assets_with_missing_issues": sum(1 for row in rows if int(row["missing_issue_columns"]) > 0),
                "assets_with_discrete_issues": sum(1 for row in rows if int(row["discrete_issue_columns"]) > 0),
                "assets_with_row_mismatch": sum(1 for row in rows if row["row_count_status"] == "row_mismatch"),
            }
        )

    dataset_model_rows = []
    for (dataset_id, model_id), rows in sorted(by_dataset_model.items()):
        dataset_model_rows.append(
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "asset_count": len(rows),
                "completed_assets": sum(1 for row in rows if row["completion_bucket"] == "completed"),
                "remaining_problem_assets": sum(1 for row in rows if row["completion_bucket"] == "remaining_problem"),
                "assets_with_inverse_issues": sum(1 for row in rows if int(row["inverse_issue_columns"]) > 0),
                "assets_with_missing_issues": sum(1 for row in rows if int(row["missing_issue_columns"]) > 0),
                "assets_with_discrete_issues": sum(1 for row in rows if int(row["discrete_issue_columns"]) > 0),
                "assets_with_row_mismatch": sum(1 for row in rows if row["row_count_status"] == "row_mismatch"),
            }
        )

    summary_rows = [
        {"metric": "pro6000_assets_considered_after_skipping_c21_n13", "value": len(working_rows)},
        {"metric": "completed_after_local_repair", "value": len(completed_rows)},
        {"metric": "remaining_problem_assets", "value": len(remaining_rows)},
        {"metric": "assets_with_file_modifications", "value": sum(1 for row in repaired_rows if row["actions"])},
        {"metric": "assets_with_decoded_columns", "value": sum(1 for row in repaired_rows if int(row["decoded_columns"]) > 0)},
        {"metric": "assets_with_projected_discrete_columns", "value": sum(1 for row in repaired_rows if int(row["projected_columns"]) > 0)},
        {"metric": "assets_with_trimmed_rows", "value": sum(1 for row in repaired_rows if int(row["trimmed_rows"]) > 0)},
        {"metric": "remaining_assets_with_inverse_issues", "value": sum(1 for row in remaining_rows if int(row["inverse_issue_columns"]) > 0)},
        {"metric": "remaining_assets_with_missing_issues", "value": sum(1 for row in remaining_rows if int(row["missing_issue_columns"]) > 0)},
        {"metric": "remaining_assets_with_discrete_issues", "value": sum(1 for row in remaining_rows if int(row["discrete_issue_columns"]) > 0)},
        {"metric": "remaining_assets_with_row_mismatch", "value": sum(1 for row in remaining_rows if row["row_count_status"] == "row_mismatch")},
    ]

    _write_csv(AUDIT_DIR / "pro6000_local_repair_results.csv", repaired_rows)
    _write_csv(AUDIT_DIR / "pro6000_postrepair_completed.csv", completed_rows)
    _write_csv(AUDIT_DIR / "pro6000_postrepair_remaining.csv", remaining_rows)
    _write_csv(AUDIT_DIR / "pro6000_postrepair_by_model.csv", model_rows)
    _write_csv(AUDIT_DIR / "pro6000_postrepair_by_dataset.csv", dataset_rows)
    _write_csv(AUDIT_DIR / "pro6000_postrepair_by_dataset_model.csv", dataset_model_rows)
    _write_csv(AUDIT_DIR / "pro6000_postrepair_summary_metrics.csv", summary_rows)

    xlsx_path = AUDIT_DIR / "pro6000_postrepair.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(model_rows).to_excel(writer, sheet_name="by_model", index=False)
        pd.DataFrame(dataset_rows).to_excel(writer, sheet_name="by_dataset", index=False)
        pd.DataFrame(dataset_model_rows).to_excel(writer, sheet_name="by_dataset_model", index=False)
        pd.DataFrame(completed_rows).to_excel(writer, sheet_name="completed", index=False)
        pd.DataFrame(remaining_rows).to_excel(writer, sheet_name="remaining", index=False)
        pd.DataFrame(repaired_rows).to_excel(writer, sheet_name="all_assets", index=False)

    report_lines = [
        "# Pro6000 Local Repair And Reaudit",
        "",
        "## Summary",
        "",
        _markdown_table(summary_rows, ["metric", "value"]),
        "",
        "## Remaining By Model",
        "",
        _markdown_table(
            model_rows,
            [
                "model_id",
                "asset_count",
                "completed_assets",
                "remaining_problem_assets",
                "assets_with_inverse_issues",
                "assets_with_missing_issues",
                "assets_with_discrete_issues",
                "assets_with_row_mismatch",
            ],
        ),
        "",
        "## Remaining Roster",
        "",
        _markdown_table(
            remaining_rows,
            [
                "dataset_id",
                "model_id",
                "run_id",
                "inverse_issue_columns",
                "missing_issue_columns",
                "discrete_issue_columns",
                "row_count_status",
                "actions",
                "remaining_issues",
            ],
        ),
    ]
    (AUDIT_DIR / "pro6000_postrepair_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary_json = {
        "considered_assets": len(working_rows),
        "completed_after_local_repair": len(completed_rows),
        "remaining_problem_assets": len(remaining_rows),
        "xlsx_report": str(xlsx_path),
    }
    (AUDIT_DIR / "pro6000_postrepair_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
