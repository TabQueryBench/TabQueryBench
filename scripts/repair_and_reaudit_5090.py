from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import repair_and_reaudit_pro6000 as base


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "artifacts" / "full_synthetic_processing_audit_20260502_134237"
RECOVERY_JSON = AUDIT_DIR / "server_recovery_5090_results.json"
IGNORE_MODELS = {"goggle", "codi", "cdtd"}

base.MODEL_DECODE_FIRST = {"tabsyn"}
base.MODEL_DECODE_SORTED = {"tabbyflow", "tabdiff", "forestdiffusion"}


def _load_recovery_rows() -> dict[tuple[str, str, str], dict[str, Any]]:
    payload = json.loads(RECOVERY_JSON.read_text(encoding="utf-8"))
    rows = payload.get("results") or []
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["dataset_id"]), str(row["model_id"]), str(row["run_id"]))
        lookup[key] = row
    return lookup


def _working_rows() -> list[dict[str, str]]:
    rows = base._read_csv_rows(AUDIT_DIR / "incomplete_assets.csv")
    filtered = [
        row
        for row in rows
        if row.get("root_name") == "SynOutput-5090" and row.get("model_id") not in IGNORE_MODELS
    ]
    filtered.sort(key=lambda row: (row["dataset_id"], row["model_id"], row["run_id"]))
    return filtered


def _enrich_result(result: dict[str, Any], audit_row: dict[str, str], recovery_row: dict[str, Any] | None) -> dict[str, Any]:
    enriched = dict(result)
    enriched["root_name"] = audit_row.get("root_name", "")
    enriched["jsd_from_latest_summary"] = audit_row.get("jsd_from_latest_summary", "")
    enriched["original_overall_status"] = audit_row.get("overall_status", "")
    enriched["original_completion_bucket"] = audit_row.get("completion_bucket", "")
    enriched["original_inverse_encoding_status"] = audit_row.get("inverse_encoding_status", "")
    enriched["original_missing_status"] = audit_row.get("missing_status", "")
    enriched["original_discrete_numeric_status"] = audit_row.get("discrete_numeric_status", "")
    enriched["original_row_count_status"] = audit_row.get("row_count_status", "")
    enriched["original_issues"] = audit_row.get("issues", "")
    if recovery_row:
        enriched["recovery_newly_found_file_count"] = len(recovery_row.get("newly_found_files") or [])
        enriched["recovery_skipped_existing_file_count"] = len(recovery_row.get("skipped_existing_files") or [])
        enriched["recovery_skipped_large_file_count"] = len(recovery_row.get("skipped_large_files") or [])
        enriched["recovery_copy_error_count"] = len(recovery_row.get("copy_errors") or [])
        enriched["recovery_whether_local_repair_is_possible"] = recovery_row.get("whether_local_repair_is_possible", "")
        enriched["recovery_whether_missing_can_be_restored"] = recovery_row.get("whether_missing_can_be_restored", "")
        enriched["recovery_whether_row_mismatch_can_be_explained"] = recovery_row.get(
            "whether_row_mismatch_can_be_explained",
            "",
        )
        enriched["recovery_whether_discrete_numeric_issue_can_be_fixed"] = recovery_row.get(
            "whether_discrete_numeric_issue_can_be_fixed",
            "",
        )
        enriched["recovery_final_recommendation"] = recovery_row.get("final_recommendation", "")
        enriched["server_run_dir"] = recovery_row.get("server_run_dir", "")
    else:
        enriched["recovery_newly_found_file_count"] = 0
        enriched["recovery_skipped_existing_file_count"] = 0
        enriched["recovery_skipped_large_file_count"] = 0
        enriched["recovery_copy_error_count"] = 0
        enriched["recovery_whether_local_repair_is_possible"] = ""
        enriched["recovery_whether_missing_can_be_restored"] = ""
        enriched["recovery_whether_row_mismatch_can_be_explained"] = ""
        enriched["recovery_whether_discrete_numeric_issue_can_be_fixed"] = ""
        enriched["recovery_final_recommendation"] = ""
        enriched["server_run_dir"] = ""
    return enriched


def _summarize(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)

    out: list[dict[str, Any]] = []
    for group_key, subset in sorted(groups.items()):
        out.append(
            {
                key: group_key,
                "asset_count": len(subset),
                "completed_assets": sum(1 for row in subset if row["completion_bucket"] == "completed"),
                "remaining_problem_assets": sum(
                    1 for row in subset if row["completion_bucket"] == "remaining_problem"
                ),
                "assets_with_inverse_issues": sum(1 for row in subset if int(row["inverse_issue_columns"]) > 0),
                "assets_with_missing_issues": sum(1 for row in subset if int(row["missing_issue_columns"]) > 0),
                "assets_with_discrete_issues": sum(1 for row in subset if int(row["discrete_issue_columns"]) > 0),
                "assets_with_row_mismatch": sum(1 for row in subset if row["row_count_status"] == "row_mismatch"),
                "assets_modified_locally": sum(1 for row in subset if str(row.get("actions") or "").strip()),
            }
        )
    return out


def _summarize_by_dataset_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset_id"]), str(row["model_id"]))].append(row)

    out: list[dict[str, Any]] = []
    for (dataset_id, model_id), subset in sorted(groups.items()):
        out.append(
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "asset_count": len(subset),
                "completed_assets": sum(1 for row in subset if row["completion_bucket"] == "completed"),
                "remaining_problem_assets": sum(
                    1 for row in subset if row["completion_bucket"] == "remaining_problem"
                ),
                "assets_with_inverse_issues": sum(1 for row in subset if int(row["inverse_issue_columns"]) > 0),
                "assets_with_missing_issues": sum(1 for row in subset if int(row["missing_issue_columns"]) > 0),
                "assets_with_discrete_issues": sum(1 for row in subset if int(row["discrete_issue_columns"]) > 0),
                "assets_with_row_mismatch": sum(1 for row in subset if row["row_count_status"] == "row_mismatch"),
                "assets_modified_locally": sum(1 for row in subset if str(row.get("actions") or "").strip()),
            }
        )
    return out


def main() -> None:
    working_rows = _working_rows()
    recovery_lookup = _load_recovery_rows()
    checkpoint_path = AUDIT_DIR / "postrecovery_5090_local_reaudit_results_checkpoint.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    repaired_rows: list[dict[str, Any]] = []
    current_dataset_id = ""
    dataset_context: dict[str, Any] | None = None
    for index, row in enumerate(working_rows, start=1):
        dataset_id = row["dataset_id"]
        if dataset_id != current_dataset_id:
            dataset_context = base._load_dataset_context(dataset_id)
            current_dataset_id = dataset_id
            gc.collect()
        result = base._repair_asset(row, dataset_context)
        recovery_row = recovery_lookup.get((row["dataset_id"], row["model_id"], row["run_id"]))
        repaired_rows.append(_enrich_result(result, row, recovery_row))
        base._write_csv(checkpoint_path, repaired_rows)
        if index % 10 == 0 or index == len(working_rows):
            print(f"processed {index}/{len(working_rows)} assets", flush=True)

    completed_rows = [row for row in repaired_rows if row["completion_bucket"] == "completed"]
    remaining_rows = [row for row in repaired_rows if row["completion_bucket"] == "remaining_problem"]
    model_rows = _summarize(repaired_rows, "model_id")
    dataset_rows = _summarize(repaired_rows, "dataset_id")
    dataset_model_rows = _summarize_by_dataset_model(repaired_rows)

    recovery_payload = json.loads(RECOVERY_JSON.read_text(encoding="utf-8"))
    recovery_rows = recovery_payload.get("results") or []
    summary_rows = [
        {"metric": "recovery_runs_investigated", "value": recovery_payload.get("runs_investigated", len(recovery_rows))},
        {"metric": "recovery_runs_with_additional_files", "value": sum(1 for row in recovery_rows if row.get("newly_found_files"))},
        {"metric": "recovery_total_copied_files", "value": sum(len(row.get("newly_found_files") or []) for row in recovery_rows)},
        {"metric": "recovery_large_binary_files_not_copied", "value": sum(len(row.get("skipped_large_files") or []) for row in recovery_rows)},
        {"metric": "recovery_copy_errors", "value": sum(len(row.get("copy_errors") or []) for row in recovery_rows)},
        {"metric": "assets_considered", "value": len(working_rows)},
        {"metric": "completed_after_local_reaudit", "value": len(completed_rows)},
        {"metric": "remaining_problem_assets", "value": len(remaining_rows)},
        {"metric": "assets_modified_locally", "value": sum(1 for row in repaired_rows if str(row.get("actions") or "").strip())},
        {"metric": "assets_with_decoded_columns", "value": sum(1 for row in repaired_rows if int(row["decoded_columns"]) > 0)},
        {"metric": "assets_with_projected_discrete_columns", "value": sum(1 for row in repaired_rows if int(row["projected_columns"]) > 0)},
        {"metric": "assets_with_trimmed_rows", "value": sum(1 for row in repaired_rows if int(row["trimmed_rows"]) > 0)},
        {"metric": "remaining_assets_with_inverse_issues", "value": sum(1 for row in remaining_rows if int(row["inverse_issue_columns"]) > 0)},
        {"metric": "remaining_assets_with_missing_issues", "value": sum(1 for row in remaining_rows if int(row["missing_issue_columns"]) > 0)},
        {"metric": "remaining_assets_with_discrete_issues", "value": sum(1 for row in remaining_rows if int(row["discrete_issue_columns"]) > 0)},
        {"metric": "remaining_assets_with_row_mismatch", "value": sum(1 for row in remaining_rows if row["row_count_status"] == "row_mismatch")},
    ]

    base._write_csv(AUDIT_DIR / "postrecovery_5090_local_reaudit_results.csv", repaired_rows)
    base._write_csv(AUDIT_DIR / "postrecovery_5090_completed.csv", completed_rows)
    base._write_csv(AUDIT_DIR / "postrecovery_5090_remaining.csv", remaining_rows)
    base._write_csv(AUDIT_DIR / "postrecovery_5090_by_model.csv", model_rows)
    base._write_csv(AUDIT_DIR / "postrecovery_5090_by_dataset.csv", dataset_rows)
    base._write_csv(AUDIT_DIR / "postrecovery_5090_by_dataset_model.csv", dataset_model_rows)
    base._write_csv(AUDIT_DIR / "postrecovery_5090_summary_metrics.csv", summary_rows)

    report_lines = [
        "# 5090 Post-Recovery Reaudit",
        "",
        "## Summary",
        "",
        base._markdown_table(summary_rows, ["metric", "value"]),
        "",
        "## By Model",
        "",
        base._markdown_table(
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
                "assets_modified_locally",
            ],
        ),
        "",
        "## Remaining Assets",
        "",
        base._markdown_table(
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
    (AUDIT_DIR / "postrecovery_5090_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary_json = {
        "considered_assets": len(working_rows),
        "completed_after_local_reaudit": len(completed_rows),
        "remaining_problem_assets": len(remaining_rows),
        "summary_metrics_csv": str(AUDIT_DIR / "postrecovery_5090_summary_metrics.csv"),
    }
    (AUDIT_DIR / "postrecovery_5090_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
