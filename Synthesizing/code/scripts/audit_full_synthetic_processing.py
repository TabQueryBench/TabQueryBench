from __future__ import annotations

import ast
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
LATEST_RUN_PATH = REPO_ROOT / "Evaluation" / "distance" / "LATEST_RUN.json"
IGNORE_MODELS = {"goggle", "codi", "cdtd", "ctdt"}
MISSING_TEXT = {"", "null", "none", "nan", "na", "n/a", "<null>"}


@dataclass
class ColumnPlan:
    column: str
    mapping: dict[str, Any]
    needs_decode: bool
    missing_codes: list[str]
    can_repair: bool
    unresolved_reason: str | None


def _load_latest_run_dir() -> Path:
    payload = json.loads(LATEST_RUN_PATH.read_text(encoding="utf-8"))
    return Path(payload["run_dir"])


def _read_summary_rows(run_dir: Path) -> list[dict[str, str]]:
    summary_path = run_dir / "summaries" / "distance_summary__all_datasets.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_list_literal(text: str | None) -> list[str]:
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _safe_relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_TEXT


def _canonical_token(value: Any) -> str:
    if _is_missing(value):
        return "<NA>"
    if isinstance(value, str):
        text = value.strip()
        try:
            num = float(text)
        except ValueError:
            return f"STR::{text}"
        if not math.isfinite(num):
            return f"STR::{text}"
        if abs(num - round(num)) < 1e-9:
            return f"INT::{int(round(num))}"
        return f"FLOAT::{num:.12g}"
    if isinstance(value, bool):
        return f"INT::{int(value)}"
    if isinstance(value, int):
        return f"INT::{value}"
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return f"INT::{int(round(value))}"
        return f"FLOAT::{value:.12g}"
    return f"STR::{str(value).strip()}"


def _values_equivalent(left: Any, right: Any) -> bool:
    return _canonical_token(left) == _canonical_token(right)


def _numeric_like_ratio(series: pd.Series) -> float:
    clean = series[~series.map(_is_missing)]
    if clean.empty:
        return 0.0
    converted = pd.to_numeric(clean.astype(str), errors="coerce")
    return float(converted.notna().mean())


def _integer_like_ratio(series: pd.Series) -> float:
    clean = series[~series.map(_is_missing)]
    if clean.empty:
        return 0.0
    converted = pd.to_numeric(clean.astype(str), errors="coerce").dropna()
    if converted.empty:
        return 0.0
    delta = (converted - converted.round()).abs()
    return float((delta < 1e-6).mean())


def _load_field_hints(dataset_id: str) -> dict[str, str]:
    candidates = [
        DATA_ROOT / dataset_id / "metadata_core" / "field_registry.json",
        DATA_ROOT / dataset_id / "metadata" / "field_registry.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        hints: dict[str, str] = {}
        for field in payload.get("fields", []) if isinstance(payload, dict) else []:
            if not isinstance(field, dict):
                continue
            name = str(field.get("name") or "").strip()
            if not name:
                continue
            semantic = str(field.get("semantic_type") or "").strip().lower()
            declared = str(field.get("declared_type") or "").strip().lower()
            hints[name] = " ".join(part for part in [semantic, declared] if part).strip()
        return hints
    return {}


def _real_train_path(dataset_id: str) -> Path:
    return DATA_ROOT / dataset_id / f"{dataset_id}-train.csv"


def _candidate_companions(parent: Path) -> list[Path]:
    ordered_patterns = [
        "*__real.csv",
        "*__train.csv",
        "*___goggle_train.csv",
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in ordered_patterns:
        for path in sorted(parent.glob(pattern)):
            if path.is_file() and path not in seen:
                found.append(path)
                seen.add(path)
    return found


def _find_companion_csv(row: dict[str, str]) -> Path | None:
    syn_path = Path(row["synthetic_csv_path"])
    for path in _candidate_companions(syn_path.parent):
        if path != syn_path:
            return path
    return None


def _build_column_plan(original: pd.Series, encoded: pd.Series) -> ColumnPlan | None:
    mapping: dict[str, Any] = {}
    consistent = True
    needs_decode = False
    for orig_value, enc_value in zip(original.tolist(), encoded.tolist()):
        enc_key = _canonical_token(enc_value)
        mapped_value = pd.NA if _is_missing(orig_value) else orig_value
        if enc_key in mapping:
            if not _values_equivalent(mapping[enc_key], mapped_value):
                consistent = False
                break
        else:
            mapping[enc_key] = mapped_value
        if not _is_missing(orig_value) and not _is_missing(enc_value):
            if not _values_equivalent(orig_value, enc_value):
                needs_decode = True
    if not consistent:
        return ColumnPlan(
            column=str(original.name),
            mapping={},
            needs_decode=False,
            missing_codes=[],
            can_repair=False,
            unresolved_reason="mapping_conflict",
        )
    missing_codes = [key for key, value in mapping.items() if key != "<NA>" and _is_missing(value)]
    if not needs_decode and not missing_codes:
        return None
    return ColumnPlan(
        column=str(original.name),
        mapping=mapping,
        needs_decode=needs_decode,
        missing_codes=missing_codes,
        can_repair=True,
        unresolved_reason=None,
    )


def _apply_column_plan(series: pd.Series, plan: ColumnPlan) -> tuple[pd.Series, int, int, list[str]]:
    updated: list[Any] = []
    changed_cells = 0
    restored_missing_cells = 0
    unmapped_tokens: Counter[str] = Counter()
    for value in series.tolist():
        token = _canonical_token(value)
        if token in plan.mapping:
            mapped = plan.mapping[token]
            new_value = pd.NA if _is_missing(mapped) else mapped
            if not _values_equivalent(value, new_value):
                changed_cells += 1
                if _is_missing(new_value):
                    restored_missing_cells += 1
            updated.append(new_value)
            continue
        updated.append(value)
        if token != "<NA>":
            unmapped_tokens[str(value)] += 1
    sample_unmapped = [token for token, _count in unmapped_tokens.most_common(5)]
    return pd.Series(updated, index=series.index, name=series.name), changed_cells, restored_missing_cells, sample_unmapped


def _unique_overlap_ratio(real_series: pd.Series, syn_series: pd.Series) -> float | None:
    real_tokens = {str(value) for value in real_series[~real_series.map(_is_missing)].astype(str).tolist()}
    syn_tokens = {str(value) for value in syn_series[~syn_series.map(_is_missing)].astype(str).tolist()}
    if not syn_tokens:
        return None
    return len(real_tokens & syn_tokens) / len(syn_tokens)


def _bucket_for_column(hint: str, real_series: pd.Series) -> str:
    token = (hint or "").lower()
    if any(word in token for word in ["categorical", "string", "text", "boolean", "ordinal", "free_text"]):
        real_numeric_ratio = _numeric_like_ratio(real_series)
        if "boolean" in token and real_numeric_ratio >= 0.95:
            return "discrete_numeric"
        if "categorical_binary" in token and real_numeric_ratio >= 0.95:
            return "discrete_numeric"
        return "textual_categorical"
    if any(word in token for word in ["identifier", " id", "id ", "numeric_discrete", "discrete", "integer"]):
        return "discrete_numeric"
    if any(word in token for word in ["numeric", "float", "double", "decimal", "continuous"]):
        return "continuous_numeric"
    if _numeric_like_ratio(real_series) < 0.5:
        return "textual_categorical"
    if _integer_like_ratio(real_series) >= 0.95:
        return "discrete_numeric"
    return "continuous_numeric"


def _append_remote_strings(value: Any, out: set[str]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _append_remote_strings(child, out)
        return
    if isinstance(value, list):
        for child in value:
            _append_remote_strings(child, out)
        return
    if not isinstance(value, str):
        return
    text = value.strip()
    if text.startswith("/home/") or text.startswith("/data/") or text.startswith("/workspace/"):
        out.add(text)


def _collect_remote_hints(row: dict[str, str]) -> list[str]:
    hints: set[str] = set()
    asset_dir = Path(row["asset_dir"])
    manifest_path = asset_dir / "manifest.json"
    if manifest_path.exists():
        try:
            _append_remote_strings(json.loads(manifest_path.read_text(encoding="utf-8")), hints)
        except Exception:
            pass
    for meta_path in _parse_list_literal(row.get("metadata_paths")):
        path = Path(meta_path)
        if not path.exists():
            continue
        try:
            _append_remote_strings(json.loads(path.read_text(encoding="utf-8")), hints)
        except Exception:
            continue
    return sorted(hints)


def _assess_inverse_without_companion(
    real_series: pd.Series,
    syn_series: pd.Series,
    bucket: str,
) -> tuple[str, str]:
    syn_non_missing = syn_series[~syn_series.map(_is_missing)]
    if syn_non_missing.empty:
        return "all_missing", "synthetic_all_missing"
    if bucket != "textual_categorical":
        return "not_applicable", ""
    real_numeric_ratio = _numeric_like_ratio(real_series)
    syn_numeric_ratio = _numeric_like_ratio(syn_series)
    overlap_ratio = _unique_overlap_ratio(real_series, syn_series)
    if real_numeric_ratio < 0.5 and syn_numeric_ratio > 0.95 and (overlap_ratio or 0.0) == 0.0:
        return "suspected_not_decoded", "real_textual_syn_numeric_no_overlap"
    if overlap_ratio is None:
        return "unknown", "no_non_missing_values"
    if overlap_ratio >= 0.95:
        return "verified_direct_compare", ""
    if real_numeric_ratio < 0.5 and syn_numeric_ratio < 0.5:
        return "likely_ok_direct_compare", f"string_like_overlap={overlap_ratio:.3f}"
    if overlap_ratio >= 0.50:
        return "likely_ok_direct_compare", f"overlap={overlap_ratio:.3f}"
    return "unknown", f"overlap={overlap_ratio:.3f}"


def _assess_discrete_numeric(series_real: pd.Series, series_syn: pd.Series, bucket: str) -> tuple[str, str]:
    if bucket != "discrete_numeric":
        return "not_applicable", ""
    syn_non_missing = series_syn[~series_syn.map(_is_missing)]
    if syn_non_missing.empty:
        return "all_missing", "synthetic_all_missing"
    parsed = pd.to_numeric(syn_non_missing.astype(str), errors="coerce")
    if parsed.isna().all():
        return "contains_non_numeric_values", "failed_numeric_parse"
    valid = parsed.dropna()
    if valid.empty:
        return "contains_non_numeric_values", "failed_numeric_parse"
    frac_ratio = float(((valid - valid.round()).abs() > 1e-6).mean())
    if frac_ratio > 0.05:
        return "contains_fractional_values", f"fractional_ratio={frac_ratio:.3f}"
    return "looks_integer_like", ""


def _assess_missing_status(
    real_series: pd.Series,
    syn_before: pd.Series,
    syn_after: pd.Series,
    plan: ColumnPlan | None,
    changed_missing_cells: int,
) -> tuple[str, str]:
    real_missing_rate = float(real_series.isna().mean())
    before_missing = float(syn_before.isna().mean())
    after_missing = float(syn_after.isna().mean())
    if real_missing_rate <= 0:
        return "not_applicable", ""
    if changed_missing_cells > 0:
        return "restored_to_missing", f"before={before_missing:.4f};after={after_missing:.4f}"
    if after_missing > 0:
        return "missing_present", f"before={before_missing:.4f};after={after_missing:.4f}"
    if plan is not None and plan.missing_codes:
        return "missing_lost", "coded_missing_not_generated"
    return "missing_lost", "nan_style_missing_not_present"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None_\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for row in rows:
        values = [str(row.get(column, "")).replace("\n", "<br>") for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_excel(output_dir: Path, sheets: dict[str, list[dict[str, Any]]]) -> Path | None:
    xlsx_path = output_dir / "synthetic_processing_audit.xlsx"
    try:
        from openpyxl.utils import get_column_letter

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for sheet_name, rows in sheets.items():
                df = pd.DataFrame(rows)
                if df.empty:
                    df = pd.DataFrame([{}])
                excel_sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=excel_sheet_name, index=False)
                worksheet = writer.sheets[excel_sheet_name]
                worksheet.freeze_panes = "A2"
                worksheet.auto_filter.ref = worksheet.dimensions
                for col_idx, column in enumerate(df.columns, start=1):
                    values = [str(column)]
                    values.extend("" if pd.isna(value) else str(value) for value in df[column].head(200).tolist())
                    width = min(max(len(value) for value in values) + 2, 60)
                    worksheet.column_dimensions[get_column_letter(col_idx)].width = width
        return xlsx_path
    except Exception:
        return None


def _build_overview_tables(
    all_asset_rows: list[dict[str, Any]],
    incomplete_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    df_all = pd.DataFrame(all_asset_rows)
    df_incomplete = pd.DataFrame(incomplete_rows)

    overview_metrics = [
        {"metric": "all_non_ignored_assets", "value": len(all_asset_rows)},
        {"metric": "completed_assets", "value": int((df_all["completion_bucket"] == "completed").sum()) if not df_all.empty else 0},
        {"metric": "incomplete_assets", "value": len(incomplete_rows)},
        {"metric": "excluded_assets", "value": len(excluded_rows)},
        {"metric": "assets_modified_locally_in_this_pass", "value": int((df_all["local_action"] == "modified_csv").sum()) if not df_all.empty else 0},
        {"metric": "assets_with_companion_available", "value": int((df_all["companion_available"] == "yes").sum()) if not df_all.empty else 0},
        {"metric": "assets_with_inverse_issues", "value": int((df_all["inverse_encoding_status"] == "has_inverse_issues").sum()) if not df_all.empty else 0},
        {"metric": "assets_with_missing_issues", "value": int((df_all["missing_status"] == "has_missing_issues").sum()) if not df_all.empty else 0},
        {"metric": "assets_with_discrete_numeric_issues", "value": int((df_all["discrete_numeric_status"] == "has_discrete_numeric_issues").sum()) if not df_all.empty else 0},
        {"metric": "assets_with_row_mismatch", "value": int((df_all["row_count_status"] == "row_mismatch").sum()) if not df_all.empty else 0},
        {"metric": "incomplete_assets_needing_server_lookup", "value": int((df_incomplete["server_lookup_recommended"] == "yes").sum()) if not df_incomplete.empty else 0},
    ]

    overview_by_root: list[dict[str, Any]] = []
    if not df_all.empty:
        for root_name, group in df_all.groupby("root_name", sort=True):
            overview_by_root.append(
                {
                    "root_name": root_name,
                    "total_assets": len(group),
                    "completed_assets": int((group["completion_bucket"] == "completed").sum()),
                    "incomplete_assets": int((group["completion_bucket"] == "incomplete").sum()),
                    "modified_locally_in_this_pass": int((group["local_action"] == "modified_csv").sum()),
                    "companion_available_assets": int((group["companion_available"] == "yes").sum()),
                    "inverse_issue_assets": int((group["inverse_encoding_status"] == "has_inverse_issues").sum()),
                    "missing_issue_assets": int((group["missing_status"] == "has_missing_issues").sum()),
                    "discrete_numeric_issue_assets": int((group["discrete_numeric_status"] == "has_discrete_numeric_issues").sum()),
                    "row_mismatch_assets": int((group["row_count_status"] == "row_mismatch").sum()),
                }
            )

    overview_by_model: list[dict[str, Any]] = []
    if not df_all.empty:
        for (root_name, model_id), group in df_all.groupby(["root_name", "model_id"], sort=True):
            overview_by_model.append(
                {
                    "root_name": root_name,
                    "model_id": model_id,
                    "total_assets": len(group),
                    "completed_assets": int((group["completion_bucket"] == "completed").sum()),
                    "incomplete_assets": int((group["completion_bucket"] == "incomplete").sum()),
                    "modified_locally_in_this_pass": int((group["local_action"] == "modified_csv").sum()),
                    "companion_available_assets": int((group["companion_available"] == "yes").sum()),
                    "inverse_issue_assets": int((group["inverse_encoding_status"] == "has_inverse_issues").sum()),
                    "missing_issue_assets": int((group["missing_status"] == "has_missing_issues").sum()),
                    "discrete_numeric_issue_assets": int((group["discrete_numeric_status"] == "has_discrete_numeric_issues").sum()),
                    "row_mismatch_assets": int((group["row_count_status"] == "row_mismatch").sum()),
                }
            )

    issue_breakdown = [
        {"issue_group": "inverse_encoding_issue_assets", "asset_count": int((df_incomplete["inverse_encoding_status"] == "has_inverse_issues").sum()) if not df_incomplete.empty else 0},
        {"issue_group": "missing_issue_assets", "asset_count": int((df_incomplete["missing_status"] == "has_missing_issues").sum()) if not df_incomplete.empty else 0},
        {"issue_group": "discrete_numeric_issue_assets", "asset_count": int((df_incomplete["discrete_numeric_status"] == "has_discrete_numeric_issues").sum()) if not df_incomplete.empty else 0},
        {"issue_group": "row_mismatch_assets", "asset_count": int((df_incomplete["row_count_status"] == "row_mismatch").sum()) if not df_incomplete.empty else 0},
        {"issue_group": "missing_real_train_csv_assets", "asset_count": int(df_incomplete["issues"].fillna("").str.contains("missing_real_train_csv").sum()) if not df_incomplete.empty else 0},
        {"issue_group": "incomplete_with_companion_available", "asset_count": int((df_incomplete["companion_available"] == "yes").sum()) if not df_incomplete.empty else 0},
        {"issue_group": "incomplete_without_companion", "asset_count": int((df_incomplete["companion_available"] != "yes").sum()) if not df_incomplete.empty else 0},
    ]

    return {
        "overview_metrics": overview_metrics,
        "overview_by_root": overview_by_root,
        "overview_by_model": overview_by_model,
        "issue_breakdown": issue_breakdown,
    }


def _status_priority(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("root_name", "")),
        str(row.get("dataset_id", "")),
        str(row.get("model_id", "")),
        str(row.get("run_id", "")),
    )


def main() -> None:
    run_dir = _load_latest_run_dir()
    summary_rows = _read_summary_rows(run_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "artifacts" / f"full_synthetic_processing_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_asset_rows: list[dict[str, Any]] = []
    completed_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []

    for row in summary_rows:
        if row.get("root_name") not in {"SynOutput", "SynOutput-5090"}:
            continue
        model_id = row.get("model_id", "").lower()
        asset_base = {
            "dataset_id": row["dataset_id"],
            "root_name": row["root_name"],
            "model_id": row["model_id"],
            "run_id": row["run_id"],
            "jsd_from_latest_summary": row.get("jensen_shannon_distance", ""),
            "synthetic_csv_path": _safe_relative(Path(row["synthetic_csv_path"])),
            "asset_dir": _safe_relative(Path(row["asset_dir"])),
            "real_row_count": row.get("real_row_count", ""),
            "synthetic_row_count": row.get("synthetic_row_count", ""),
        }

        if model_id in IGNORE_MODELS:
            excluded_rows.append(
                {
                    **asset_base,
                    "status": "excluded_by_user_request",
                }
            )
            continue

        real_path = _real_train_path(row["dataset_id"])
        syn_path = Path(row["synthetic_csv_path"])
        companion_path = _find_companion_csv(row)
        row_mismatch = "unknown"
        try:
            if row.get("real_row_count") and row.get("synthetic_row_count"):
                row_mismatch = "row_match" if int(float(row["real_row_count"])) == int(float(row["synthetic_row_count"])) else "row_mismatch"
        except Exception:
            row_mismatch = "unknown"

        if not real_path.exists():
            asset_row = {
                **asset_base,
                "companion_csv_path": _safe_relative(companion_path),
                "companion_available": "yes" if companion_path else "no",
                "local_action": "none",
                "overall_status": "incomplete",
                "completion_bucket": "incomplete",
                "inverse_encoding_status": "unknown",
                "missing_status": "unknown",
                "discrete_numeric_status": "unknown",
                "row_count_status": row_mismatch,
                "decoded_columns_fixed": 0,
                "missing_columns_fixed": 0,
                "verified_columns": 0,
                "problem_columns": 0,
                "issues": "missing_real_train_csv",
                "server_lookup_recommended": "yes",
                "remote_hints": " | ".join(_collect_remote_hints(row)),
            }
            all_asset_rows.append(asset_row)
            incomplete_rows.append(asset_row)
            continue

        try:
            real_df = pd.read_csv(real_path)
            syn_df_before = pd.read_csv(syn_path)
        except Exception as exc:
            asset_row = {
                **asset_base,
                "companion_csv_path": _safe_relative(companion_path),
                "companion_available": "yes" if companion_path else "no",
                "local_action": "none",
                "overall_status": "incomplete",
                "completion_bucket": "incomplete",
                "inverse_encoding_status": "unknown",
                "missing_status": "unknown",
                "discrete_numeric_status": "unknown",
                "row_count_status": row_mismatch,
                "decoded_columns_fixed": 0,
                "missing_columns_fixed": 0,
                "verified_columns": 0,
                "problem_columns": 0,
                "issues": f"csv_read_failed:{exc.__class__.__name__}",
                "server_lookup_recommended": "yes",
                "remote_hints": " | ".join(_collect_remote_hints(row)),
            }
            all_asset_rows.append(asset_row)
            incomplete_rows.append(asset_row)
            continue

        syn_df_after = syn_df_before.copy()
        enc_df: pd.DataFrame | None = None
        if companion_path and companion_path.exists():
            try:
                enc_df = pd.read_csv(companion_path)
            except Exception:
                enc_df = None

        hints = _load_field_hints(row["dataset_id"])
        common_columns = [column for column in real_df.columns if column in syn_df_after.columns]
        encoded_common = [column for column in common_columns if enc_df is not None and column in enc_df.columns]
        plans: dict[str, ColumnPlan] = {}
        if enc_df is not None:
            for column in encoded_common:
                plan = _build_column_plan(real_df[column], enc_df[column])
                if plan is not None:
                    plans[column] = plan

        local_action = "none"
        decoded_columns_fixed = 0
        missing_columns_fixed = 0
        verified_columns = 0
        problem_columns = 0
        issue_notes: list[str] = []
        asset_inverse_statuses: Counter[str] = Counter()
        asset_missing_statuses: Counter[str] = Counter()
        asset_discrete_statuses: Counter[str] = Counter()

        for column in common_columns:
            hint = hints.get(column, "")
            bucket = _bucket_for_column(hint, real_df[column])
            syn_before_col = syn_df_before[column]
            syn_after_col = syn_df_after[column]
            plan = plans.get(column)
            changed_cells = 0
            restored_missing_cells = 0
            sample_unmapped: list[str] = []

            if plan is not None and plan.can_repair:
                updated_series, changed_cells, restored_missing_cells, sample_unmapped = _apply_column_plan(syn_after_col, plan)
                syn_after_col = updated_series
                syn_df_after[column] = syn_after_col
                if changed_cells > 0:
                    local_action = "modified_csv"
                if plan.needs_decode and changed_cells > 0:
                    decoded_columns_fixed += 1
                if plan.missing_codes and restored_missing_cells > 0:
                    missing_columns_fixed += 1

            inverse_status = "not_applicable"
            inverse_note = ""
            if bucket == "textual_categorical":
                if plan is not None and plan.can_repair:
                    if sample_unmapped:
                        inverse_status = "unmapped_after_repair"
                        inverse_note = ",".join(sample_unmapped)
                    elif plan.needs_decode or plan.missing_codes:
                        inverse_status = "repaired_via_companion" if changed_cells > 0 else "verified_with_companion"
                    else:
                        inverse_status = "verified_with_companion"
                else:
                    inverse_status, inverse_note = _assess_inverse_without_companion(real_df[column], syn_after_col, bucket)

            missing_status, missing_note = _assess_missing_status(real_df[column], syn_before_col, syn_after_col, plan, restored_missing_cells)
            discrete_status, discrete_note = _assess_discrete_numeric(real_df[column], syn_after_col, bucket)

            notes: list[str] = [note for note in [inverse_note, missing_note, discrete_note, plan.unresolved_reason if plan else ""] if note]
            if inverse_status in {"verified_with_companion", "verified_direct_compare", "likely_ok_direct_compare", "not_applicable"} and missing_status in {"not_applicable", "missing_present", "restored_to_missing"} and discrete_status in {"not_applicable", "looks_integer_like"}:
                verified_columns += 1
            else:
                if inverse_status in {"unmapped_after_repair", "suspected_not_decoded", "all_missing", "unknown"}:
                    problem_columns += 1
                elif missing_status == "missing_lost":
                    problem_columns += 1
                elif discrete_status in {"contains_fractional_values", "contains_non_numeric_values", "all_missing"}:
                    problem_columns += 1
                if notes:
                    issue_notes.append(f"{column}:{' / '.join(notes)}")

            asset_inverse_statuses[inverse_status] += 1
            asset_missing_statuses[missing_status] += 1
            asset_discrete_statuses[discrete_status] += 1

            column_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "root_name": row["root_name"],
                    "model_id": row["model_id"],
                    "run_id": row["run_id"],
                    "column": column,
                    "semantic_bucket": bucket,
                    "companion_available": "yes" if enc_df is not None else "no",
                    "inverse_status": inverse_status,
                    "missing_status": missing_status,
                    "discrete_numeric_status": discrete_status,
                    "orig_missing_rate": round(float(real_df[column].isna().mean()), 6),
                    "syn_missing_rate_before": round(float(syn_before_col.isna().mean()), 6),
                    "syn_missing_rate_after": round(float(syn_after_col.isna().mean()), 6),
                    "notes": " | ".join(notes),
                    "synthetic_csv_path": _safe_relative(syn_path),
                }
            )

        if local_action == "modified_csv":
            syn_df_after.to_csv(syn_path, index=False)

        all_missing_asset = syn_df_after.isna().all(axis=None)
        if all_missing_asset:
            issue_notes.append("synthetic_csv_all_missing")

        incomplete_conditions = [
            row_mismatch == "row_mismatch",
            all_missing_asset,
            any(status in {"unmapped_after_repair", "suspected_not_decoded", "all_missing", "unknown"} for status in asset_inverse_statuses),
            any(status == "missing_lost" for status in asset_missing_statuses),
            any(status in {"contains_fractional_values", "contains_non_numeric_values", "all_missing"} for status in asset_discrete_statuses),
        ]
        is_incomplete = any(incomplete_conditions)
        if is_incomplete:
            overall_status = "partially_repaired" if local_action == "modified_csv" else "incomplete"
            completion_bucket = "incomplete"
        else:
            overall_status = "repaired_locally" if local_action == "modified_csv" else "verified_ok_locally"
            completion_bucket = "completed"

        inverse_summary = (
            "repaired_or_verified"
            if not any(status in {"unmapped_after_repair", "suspected_not_decoded", "all_missing", "unknown"} for status in asset_inverse_statuses)
            else "has_inverse_issues"
        )
        missing_summary = (
            "restored_or_preserved"
            if not any(status == "missing_lost" for status in asset_missing_statuses)
            else "has_missing_issues"
        )
        discrete_summary = (
            "valid_or_not_applicable"
            if not any(status in {"contains_fractional_values", "contains_non_numeric_values", "all_missing"} for status in asset_discrete_statuses)
            else "has_discrete_numeric_issues"
        )

        asset_row = {
            **asset_base,
            "companion_csv_path": _safe_relative(companion_path),
            "companion_available": "yes" if enc_df is not None else "no",
            "local_action": local_action,
            "overall_status": overall_status,
            "completion_bucket": completion_bucket,
            "inverse_encoding_status": inverse_summary,
            "missing_status": missing_summary,
            "discrete_numeric_status": discrete_summary,
            "row_count_status": row_mismatch,
            "decoded_columns_fixed": decoded_columns_fixed,
            "missing_columns_fixed": missing_columns_fixed,
            "verified_columns": verified_columns,
            "problem_columns": problem_columns,
            "issues": " | ".join(issue_notes),
            "server_lookup_recommended": "yes" if completion_bucket == "incomplete" else "no",
            "remote_hints": " | ".join(_collect_remote_hints(row)) if completion_bucket == "incomplete" else "",
        }
        all_asset_rows.append(asset_row)
        if completion_bucket == "completed":
            completed_rows.append(asset_row)
        else:
            incomplete_rows.append(asset_row)

    all_asset_rows.sort(key=_status_priority)
    completed_rows.sort(key=_status_priority)
    incomplete_rows.sort(key=_status_priority)
    excluded_rows.sort(key=_status_priority)
    column_rows.sort(key=lambda item: (_status_priority(item), item["column"]))

    overview_tables = _build_overview_tables(all_asset_rows, incomplete_rows, excluded_rows)

    _write_csv(output_dir / "all_assets_audit.csv", all_asset_rows)
    _write_csv(output_dir / "completed_assets.csv", completed_rows)
    _write_csv(output_dir / "incomplete_assets.csv", incomplete_rows)
    _write_csv(output_dir / "excluded_assets.csv", excluded_rows)
    _write_csv(output_dir / "column_level_audit.csv", column_rows)
    _write_csv(output_dir / "overview_metrics.csv", overview_tables["overview_metrics"])
    _write_csv(output_dir / "overview_by_root.csv", overview_tables["overview_by_root"])
    _write_csv(output_dir / "overview_by_model.csv", overview_tables["overview_by_model"])
    _write_csv(output_dir / "issue_breakdown.csv", overview_tables["issue_breakdown"])

    server_5090_rows = [row for row in incomplete_rows if row["root_name"] == "SynOutput-5090"]
    server_pro6000_rows = [row for row in incomplete_rows if row["root_name"] == "SynOutput"]

    server_5090_lines: list[str] = []
    for item in server_5090_rows:
        server_5090_lines.append(
            "\n".join(
                [
                    f"dataset={item['dataset_id']} model={item['model_id']} run_id={item['run_id']}",
                    f"local_asset_dir={item['asset_dir']}",
                    f"local_synthetic_csv={item['synthetic_csv_path']}",
                    f"issue_summary={item['issues']}",
                    f"remote_hints={item['remote_hints']}",
                ]
            )
        )
    server_pro6000_lines: list[str] = []
    for item in server_pro6000_rows:
        server_pro6000_lines.append(
            "\n".join(
                [
                    f"dataset={item['dataset_id']} model={item['model_id']} run_id={item['run_id']}",
                    f"local_asset_dir={item['asset_dir']}",
                    f"local_synthetic_csv={item['synthetic_csv_path']}",
                    f"issue_summary={item['issues']}",
                    f"remote_hints={item['remote_hints']}",
                ]
            )
        )

    (output_dir / "server_lookup_5090.txt").write_text("\n\n".join(server_5090_lines), encoding="utf-8")
    (output_dir / "server_lookup_pro6000.txt").write_text("\n\n".join(server_pro6000_lines), encoding="utf-8")

    prompt_5090 = f"""你将收到两样输入：
1. 5090 服务器根路径：<SERVER_5090_ROOT>
2. 待调查清单路径：<LOCAL_SERVER_LOOKUP_5090_TXT>
   建议填写为：{output_dir / 'server_lookup_5090.txt'}

你的任务是逐条核查清单里的 run，重点确认这些 synthetic data 是否还能在服务器上找到更多中间文件，从而继续修复：

1. 先根据 `run_id`、`dataset`、`model` 和 `remote_hints` 定位服务端原始 run 目录。
2. 优先查找这些文件或同类文件：
   - `staged/public/train.csv`
   - `staged/public/val.csv`
   - `staged/public/test.csv`
   - `*__real.csv`
   - `*__train.csv`
   - `staged_features.json`
   - `model_input_manifest.json`
   - `staged_input_manifest.json`
   - `public_gate_report.json`
   - 任意 encoder / category mapping / inverse transform 相关 json、pkl、yaml、pt、ckpt
3. 如果发现 companion 训练表或映射文件，判断是否足够支持：
   - 反编码 categorical/text/boolean 列
   - 把 coded missing 恢复回真正缺失值
   - 解释行数为什么不匹配
   - 解释离散数值列为什么生成成了连续值
4. 对每个 run 输出结构化结果，至少包含：
   - dataset_id
   - model_id
   - run_id
   - server_run_dir
   - newly_found_files
   - whether_local_repair_is_possible
   - whether_missing_can_be_restored
   - whether_row_mismatch_can_be_explained
   - whether_discrete_numeric_issue_can_be_fixed
   - final_recommendation
5. 如果没有找到更多文件，也要明确写 `no_additional_files_found`。

注意：
- 不要只说“可能有”，要给出具体文件路径。
- 优先判断“是否足以继续本地修复”，其次再判断“是否必须重生成或重跑”。
"""
    prompt_pro6000 = f"""你将收到两样输入：
1. pro6000 服务器根路径：<SERVER_PRO6000_ROOT>
2. 待调查清单路径：<LOCAL_SERVER_LOOKUP_PRO6000_TXT>
   建议填写为：{output_dir / 'server_lookup_pro6000.txt'}

你的任务是逐条核查清单里的 run，重点确认这些 synthetic data 是否还能在服务器上找到更多中间文件，从而继续修复：

1. 先根据 `run_id`、`dataset`、`model` 和 `remote_hints` 定位服务端原始 run 目录。
2. 优先查找这些文件或同类文件：
   - `staged/public/train.csv`
   - `staged/public/val.csv`
   - `staged/public/test.csv`
   - `*__real.csv`
   - `*__train.csv`
   - `staged_features.json`
   - `model_input_manifest.json`
   - `staged_input_manifest.json`
   - `public_gate_report.json`
   - 任意 encoder / category mapping / inverse transform 相关 json、pkl、yaml、pt、ckpt
3. 如果发现 companion 训练表或映射文件，判断是否足够支持：
   - 反编码 categorical/text/boolean 列
   - 把 coded missing 恢复回真正缺失值
   - 解释行数为什么不匹配
   - 解释离散数值列为什么生成成了连续值
4. 对每个 run 输出结构化结果，至少包含：
   - dataset_id
   - model_id
   - run_id
   - server_run_dir
   - newly_found_files
   - whether_local_repair_is_possible
   - whether_missing_can_be_restored
   - whether_row_mismatch_can_be_explained
   - whether_discrete_numeric_issue_can_be_fixed
   - final_recommendation
5. 如果没有找到更多文件，也要明确写 `no_additional_files_found`。

注意：
- 不要只说“可能有”，要给出具体文件路径。
- 这批资产很多本地没有 companion 文件，所以最重要的是找回能做反编码或恢复 missing 的中间产物。
"""
    (output_dir / "prompt_for_5090_investigator.txt").write_text(prompt_5090, encoding="utf-8")
    (output_dir / "prompt_for_pro6000_investigator.txt").write_text(prompt_pro6000, encoding="utf-8")

    sheet_map = {
        "overview_metrics": overview_tables["overview_metrics"],
        "overview_by_root": overview_tables["overview_by_root"],
        "overview_by_model": overview_tables["overview_by_model"],
        "issue_breakdown": overview_tables["issue_breakdown"],
        "all_assets": all_asset_rows,
        "completed": completed_rows,
        "incomplete": incomplete_rows,
        "excluded": excluded_rows,
        "column_audit": column_rows,
    }
    xlsx_path = _write_excel(output_dir, sheet_map)

    summary_payload = {
        "latest_run_dir": str(run_dir),
        "all_non_ignored_assets": len(all_asset_rows),
        "completed_assets": len(completed_rows),
        "incomplete_assets": len(incomplete_rows),
        "excluded_assets": len(excluded_rows),
        "excel_report": str(xlsx_path) if xlsx_path else "",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    completed_columns = [
        "dataset_id",
        "root_name",
        "model_id",
        "run_id",
        "overall_status",
        "inverse_encoding_status",
        "missing_status",
        "discrete_numeric_status",
        "row_count_status",
        "decoded_columns_fixed",
        "missing_columns_fixed",
        "verified_columns",
        "problem_columns",
        "synthetic_csv_path",
    ]
    incomplete_columns = [
        "dataset_id",
        "root_name",
        "model_id",
        "run_id",
        "overall_status",
        "inverse_encoding_status",
        "missing_status",
        "discrete_numeric_status",
        "row_count_status",
        "issues",
        "synthetic_csv_path",
    ]
    report_lines = [
        "# Full Synthetic Processing Audit",
        "",
        f"- Latest distance run: `{run_dir}`",
        f"- Non-ignored assets audited: `{len(all_asset_rows)}`",
        f"- Completed assets: `{len(completed_rows)}`",
        f"- Incomplete assets: `{len(incomplete_rows)}`",
        f"- Excluded by request: `{len(excluded_rows)}`",
        f"- Excel workbook: `{output_dir / 'synthetic_processing_audit.xlsx'}`",
        f"- 5090 investigator prompt: `{output_dir / 'prompt_for_5090_investigator.txt'}`",
        f"- pro6000 investigator prompt: `{output_dir / 'prompt_for_pro6000_investigator.txt'}`",
        "",
        "## Overview Metrics",
        "",
        _markdown_table(overview_tables["overview_metrics"], ["metric", "value"]),
        "",
        "## Overview By Root",
        "",
        _markdown_table(
            overview_tables["overview_by_root"],
            [
                "root_name",
                "total_assets",
                "completed_assets",
                "incomplete_assets",
                "modified_locally_in_this_pass",
                "companion_available_assets",
                "inverse_issue_assets",
                "missing_issue_assets",
                "discrete_numeric_issue_assets",
                "row_mismatch_assets",
            ],
        ),
        "",
        "## Issue Breakdown",
        "",
        _markdown_table(overview_tables["issue_breakdown"], ["issue_group", "asset_count"]),
        "",
        "## Completed",
        "",
        _markdown_table(completed_rows, completed_columns),
        "",
        "## Incomplete",
        "",
        _markdown_table(incomplete_rows, incomplete_columns),
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"report_dir={output_dir}")


if __name__ == "__main__":
    main()
