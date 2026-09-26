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
PROBLEM_JSD_THRESHOLD = 0.3
MISSING_TEXT = {"", "null", "none", "nan", "na", "n/a", "<null>"}


@dataclass
class ColumnRepairPlan:
    column: str
    mapping: dict[str, Any]
    needs_decode: bool
    missing_codes: list[str]
    orig_missing_rate: float
    enc_missing_rate: float
    syn_missing_rate_before: float
    can_repair: bool
    unresolved_reason: str | None


def _load_latest_run_dir() -> Path:
    payload = json.loads(LATEST_RUN_PATH.read_text(encoding="utf-8"))
    return Path(payload["run_dir"])


def _read_summary_rows(run_dir: Path) -> list[dict[str, str]]:
    summary_path = run_dir / "summaries" / "distance_summary__all_datasets.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _problem_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if row.get("root_name") not in {"SynOutput", "SynOutput-5090"}:
            continue
        if row.get("model_id", "").lower() in IGNORE_MODELS:
            continue
        jsd_text = row.get("jensen_shannon_distance") or ""
        if not jsd_text:
            continue
        try:
            jsd = float(jsd_text)
        except ValueError:
            continue
        if jsd > PROBLEM_JSD_THRESHOLD:
            out.append(row)
    return out


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


def _candidate_companions(parent: Path, model_id: str) -> list[Path]:
    preferred: list[Path] = []
    ordered_patterns = [
        "*__real.csv",
        "*__train.csv",
        "*___goggle_train.csv",
    ]
    for pattern in ordered_patterns:
        preferred.extend(sorted(path for path in parent.glob(pattern) if path.is_file()))
    filtered = [path for path in preferred if model_id not in IGNORE_MODELS or "goggle_train" not in path.name]
    dedup: list[Path] = []
    seen: set[Path] = set()
    for path in filtered:
        if path not in seen:
            dedup.append(path)
            seen.add(path)
    return dedup


def _find_companion_csv(row: dict[str, str]) -> Path | None:
    syn_path = Path(row["synthetic_csv_path"])
    for path in _candidate_companions(syn_path.parent, row["model_id"].lower()):
        if path != syn_path:
            return path
    return None


def _real_train_path(dataset_id: str) -> Path:
    return DATA_ROOT / dataset_id / f"{dataset_id}-train.csv"


def _build_column_plan(
    original: pd.Series,
    encoded: pd.Series,
    synthetic: pd.Series,
) -> ColumnRepairPlan | None:
    mapping: dict[str, Any] = {}
    consistent = True
    needs_decode = False
    for orig_value, enc_value in zip(original.tolist(), encoded.tolist()):
        enc_key = _canonical_token(enc_value)
        mapped_value = pd.NA if _is_missing(orig_value) else orig_value
        if enc_key in mapping:
            prior = mapping[enc_key]
            if not _values_equivalent(prior, mapped_value):
                consistent = False
                break
        else:
            mapping[enc_key] = mapped_value
        if not _is_missing(orig_value) and not _is_missing(enc_value):
            if not _values_equivalent(orig_value, enc_value):
                needs_decode = True

    if not consistent:
        return ColumnRepairPlan(
            column=str(original.name),
            mapping={},
            needs_decode=False,
            missing_codes=[],
            orig_missing_rate=float(original.isna().mean()),
            enc_missing_rate=float(encoded.isna().mean()),
            syn_missing_rate_before=float(synthetic.isna().mean()),
            can_repair=False,
            unresolved_reason="mapping_conflict",
        )

    missing_codes = [
        key
        for key, value in mapping.items()
        if key != "<NA>" and _is_missing(value)
    ]

    if not needs_decode and not missing_codes:
        return None

    return ColumnRepairPlan(
        column=str(original.name),
        mapping=mapping,
        needs_decode=needs_decode,
        missing_codes=missing_codes,
        orig_missing_rate=float(original.isna().mean()),
        enc_missing_rate=float(encoded.isna().mean()),
        syn_missing_rate_before=float(synthetic.isna().mean()),
        can_repair=True,
        unresolved_reason=None,
    )


def _apply_column_plan(series: pd.Series, plan: ColumnRepairPlan) -> tuple[pd.Series, int, int, list[str]]:
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


def _safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except Exception:
        return str(path)


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


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None_\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            text = str(value).replace("\n", "<br>")
            values.append(text)
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    run_dir = _load_latest_run_dir()
    summary_rows = _read_summary_rows(run_dir)
    problem_rows = _problem_rows(summary_rows)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "artifacts" / f"distance_repair_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    handled_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for row in summary_rows:
        model_id = row.get("model_id", "").lower()
        jsd_text = row.get("jensen_shannon_distance") or ""
        jsd = float(jsd_text) if jsd_text else None
        if model_id in IGNORE_MODELS and jsd is not None and jsd > PROBLEM_JSD_THRESHOLD:
            excluded_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "root_name": row["root_name"],
                    "model_id": row["model_id"],
                    "run_id": row["run_id"],
                    "synthetic_csv_path": _safe_relative(Path(row["synthetic_csv_path"])),
                    "reason": "excluded_by_user_request",
                }
            )

    for row in problem_rows:
        dataset_id = row["dataset_id"]
        model_id = row["model_id"]
        root_name = row["root_name"]
        syn_path = Path(row["synthetic_csv_path"])
        companion_path = _find_companion_csv(row)
        real_path = _real_train_path(dataset_id)
        row_count_status = "row_match"
        if row.get("synthetic_row_count") and row.get("real_row_count"):
            try:
                if int(float(row["synthetic_row_count"])) != int(float(row["real_row_count"])):
                    row_count_status = "row_mismatch"
            except ValueError:
                row_count_status = "row_unknown"

        base_info = {
            "dataset_id": dataset_id,
            "root_name": root_name,
            "model_id": model_id,
            "run_id": row["run_id"],
            "jsd_before": row["jensen_shannon_distance"],
            "row_count_status": row_count_status,
            "synthetic_csv_path": _safe_relative(syn_path),
            "asset_dir": _safe_relative(Path(row["asset_dir"])),
            "companion_csv_path": _safe_relative(companion_path) if companion_path else "",
        }

        if not real_path.exists():
            unresolved_rows.append(
                {
                    **base_info,
                    "local_status": "unhandled",
                    "reason": "missing_real_train_csv",
                    "decoded_columns_fixed": 0,
                    "missing_code_columns_fixed": 0,
                    "remaining_unresolved_columns": "",
                    "server_lookup_recommended": "yes",
                    "remote_hints": " | ".join(_collect_remote_hints(row)),
                }
            )
            continue

        if companion_path is None or not companion_path.exists():
            unresolved_rows.append(
                {
                    **base_info,
                    "local_status": "unhandled",
                    "reason": "missing_local_companion_csv",
                    "decoded_columns_fixed": 0,
                    "missing_code_columns_fixed": 0,
                    "remaining_unresolved_columns": "",
                    "server_lookup_recommended": "yes",
                    "remote_hints": " | ".join(_collect_remote_hints(row)),
                }
            )
            continue

        try:
            orig_df = pd.read_csv(real_path)
            enc_df = pd.read_csv(companion_path)
            syn_df = pd.read_csv(syn_path)
        except Exception as exc:
            unresolved_rows.append(
                {
                    **base_info,
                    "local_status": "unhandled",
                    "reason": f"csv_read_failed:{exc.__class__.__name__}",
                    "decoded_columns_fixed": 0,
                    "missing_code_columns_fixed": 0,
                    "remaining_unresolved_columns": "",
                    "server_lookup_recommended": "yes",
                    "remote_hints": " | ".join(_collect_remote_hints(row)),
                }
            )
            continue

        if syn_df.isna().all(axis=None):
            unresolved_rows.append(
                {
                    **base_info,
                    "local_status": "unhandled",
                    "reason": "synthetic_csv_all_missing",
                    "decoded_columns_fixed": 0,
                    "missing_code_columns_fixed": 0,
                    "remaining_unresolved_columns": "",
                    "server_lookup_recommended": "yes",
                    "remote_hints": " | ".join(_collect_remote_hints(row)),
                }
            )
            continue

        common_columns = [column for column in orig_df.columns if column in enc_df.columns and column in syn_df.columns]
        plans: list[ColumnRepairPlan] = []
        unresolved_column_notes: list[str] = []
        for column in common_columns:
            plan = _build_column_plan(orig_df[column], enc_df[column], syn_df[column])
            if plan is None:
                if float(orig_df[column].isna().mean()) > 0 and float(enc_df[column].isna().mean()) > 0 and float(syn_df[column].isna().mean()) == 0:
                    unresolved_column_notes.append(f"{column}:nan_style_missing_lost")
                continue
            if plan.unresolved_reason:
                unresolved_column_notes.append(f"{column}:{plan.unresolved_reason}")
                continue
            plans.append(plan)

        decoded_columns_fixed = 0
        missing_code_columns_fixed = 0
        changed_cells_total = 0
        restored_missing_cells_total = 0
        unmapped_after_fix: list[str] = []

        for plan in plans:
            updated_series, changed_cells, restored_missing_cells, sample_unmapped = _apply_column_plan(syn_df[plan.column], plan)
            syn_df[plan.column] = updated_series
            changed_cells_total += changed_cells
            restored_missing_cells_total += restored_missing_cells
            if plan.needs_decode:
                decoded_columns_fixed += 1
            if plan.missing_codes:
                missing_code_columns_fixed += 1
            if sample_unmapped:
                unmapped_after_fix.append(f"{plan.column}:{','.join(sample_unmapped)}")

        wrote_file = False
        if changed_cells_total > 0:
            syn_df.to_csv(syn_path, index=False)
            wrote_file = True

        post_unresolved = list(unresolved_column_notes)
        if unmapped_after_fix:
            post_unresolved.extend(f"unmapped:{item}" for item in unmapped_after_fix)
        if row_count_status != "row_match":
            post_unresolved.append(row_count_status)

        local_status = "repaired" if wrote_file and not post_unresolved else "partially_repaired" if wrote_file else "unhandled"
        handled_rows.append(
            {
                **base_info,
                "local_status": local_status,
                "decoded_columns_fixed": decoded_columns_fixed,
                "missing_code_columns_fixed": missing_code_columns_fixed,
                "changed_cells": changed_cells_total,
                "restored_missing_cells": restored_missing_cells_total,
                "remaining_unresolved_columns": " | ".join(post_unresolved),
                "compare_against_real": "mapping_verified_from_companion_train",
            }
        )

        if local_status != "repaired":
            unresolved_rows.append(
                {
                    **base_info,
                    "local_status": "needs_server_or_manual_followup",
                    "reason": " | ".join(post_unresolved) if post_unresolved else "no_safe_local_change",
                    "decoded_columns_fixed": decoded_columns_fixed,
                    "missing_code_columns_fixed": missing_code_columns_fixed,
                    "remaining_unresolved_columns": " | ".join(post_unresolved),
                    "server_lookup_recommended": "yes",
                    "remote_hints": " | ".join(_collect_remote_hints(row)),
                }
            )

    handled_rows.sort(key=lambda item: (item["root_name"], item["dataset_id"], item["model_id"], item["run_id"]))
    unresolved_rows.sort(key=lambda item: (item["root_name"], item["dataset_id"], item["model_id"], item["run_id"], item["reason"]))
    excluded_rows.sort(key=lambda item: (item["root_name"], item["dataset_id"], item["model_id"], item["run_id"]))

    _write_csv(output_dir / "handled_assets.csv", handled_rows)
    _write_csv(output_dir / "unhandled_assets.csv", unresolved_rows)
    _write_csv(output_dir / "excluded_models.csv", excluded_rows)

    server_5090_rows = [row for row in unresolved_rows if row["root_name"] == "SynOutput-5090"]
    server_6000_rows = [row for row in unresolved_rows if row["root_name"] == "SynOutput"]

    server_5090_lines = []
    for item in server_5090_rows:
        server_5090_lines.append(
            "\n".join(
                [
                    f"dataset={item['dataset_id']} model={item['model_id']} run_id={item['run_id']}",
                    f"local_asset_dir={item['asset_dir']}",
                    f"local_synthetic_csv={item['synthetic_csv_path']}",
                    f"reason={item['reason']}",
                    f"remote_hints={item['remote_hints']}",
                ]
            )
        )
    server_6000_lines = []
    for item in server_6000_rows:
        server_6000_lines.append(
            "\n".join(
                [
                    f"dataset={item['dataset_id']} model={item['model_id']} run_id={item['run_id']}",
                    f"local_asset_dir={item['asset_dir']}",
                    f"local_synthetic_csv={item['synthetic_csv_path']}",
                    f"reason={item['reason']}",
                    f"remote_hints={item['remote_hints']}",
                ]
            )
        )

    (output_dir / "server_lookup_5090.txt").write_text("\n\n".join(server_5090_lines), encoding="utf-8")
    (output_dir / "server_lookup_pro6000.txt").write_text("\n\n".join(server_6000_lines), encoding="utf-8")

    summary_payload = {
        "latest_run_dir": str(run_dir),
        "problem_asset_count": len(problem_rows),
        "handled_asset_count": len(handled_rows),
        "fully_repaired_count": sum(1 for row in handled_rows if row["local_status"] == "repaired"),
        "partially_repaired_count": sum(1 for row in handled_rows if row["local_status"] == "partially_repaired"),
        "unhandled_or_followup_count": len(unresolved_rows),
        "excluded_asset_count": len(excluded_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    handled_columns = [
        "dataset_id",
        "root_name",
        "model_id",
        "run_id",
        "local_status",
        "decoded_columns_fixed",
        "missing_code_columns_fixed",
        "changed_cells",
        "restored_missing_cells",
        "row_count_status",
        "remaining_unresolved_columns",
        "synthetic_csv_path",
    ]
    unresolved_columns = [
        "dataset_id",
        "root_name",
        "model_id",
        "run_id",
        "local_status",
        "reason",
        "decoded_columns_fixed",
        "missing_code_columns_fixed",
        "row_count_status",
        "synthetic_csv_path",
    ]

    report_lines = [
        "# Problematic Synthetic Data Repair Report",
        "",
        f"- Latest distance run: `{run_dir}`",
        f"- Problematic assets in scope (`JSD > {PROBLEM_JSD_THRESHOLD}`, ignored models excluded): `{len(problem_rows)}`",
        f"- Assets touched locally: `{len(handled_rows)}`",
        f"- Fully repaired: `{summary_payload['fully_repaired_count']}`",
        f"- Partially repaired: `{summary_payload['partially_repaired_count']}`",
        f"- Still unresolved / needs server or manual follow-up: `{len(unresolved_rows)}`",
        f"- Excluded by request (`goggle`, `codi`, `cdtd`): `{len(excluded_rows)}`",
        "",
        "## Handled",
        "",
        _markdown_table(handled_rows, handled_columns),
        "",
        "## Unhandled Or Still Needs Server",
        "",
        _markdown_table(unresolved_rows, unresolved_columns),
        "",
        "## Excluded",
        "",
        _markdown_table(excluded_rows, ["dataset_id", "root_name", "model_id", "run_id", "reason", "synthetic_csv_path"]),
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"report_dir={output_dir}")


if __name__ == "__main__":
    main()
