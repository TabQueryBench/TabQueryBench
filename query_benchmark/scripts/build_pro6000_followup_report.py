from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = REPO_ROOT / "artifacts" / "full_synthetic_processing_audit_20260502_134237"


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _status_bucket(result: dict[str, Any]) -> str:
    if result.get("whether_local_repair_is_possible"):
        return "pending_local_repair_after_server_recovery"
    return "rerun_required_server_run_missing"


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
    audit_dir = DEFAULT_AUDIT_DIR
    incomplete_rows = _read_csv(audit_dir / "incomplete_assets.csv")
    pro6000_rows = [row for row in incomplete_rows if row.get("root_name") == "SynOutput"]

    recovery_payload = json.loads((audit_dir / "pro6000_recovery_results.json").read_text(encoding="utf-8"))
    recovery_rows = recovery_payload["results"]
    recovery_map = {
        (row["dataset_id"], row["model_id"], row["run_id"]): row
        for row in recovery_rows
    }

    merged_rows: list[dict[str, Any]] = []
    missing_recovery_rows: list[dict[str, Any]] = []
    for row in pro6000_rows:
        key = (row["dataset_id"], row["model_id"], row["run_id"])
        recovery = recovery_map.get(key)
        if recovery is None:
            missing_recovery_rows.append(row)
            continue
        merged_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "model_id": row["model_id"],
                "run_id": row["run_id"],
                "status_bucket": _status_bucket(recovery),
                "original_overall_status": row["overall_status"],
                "inverse_encoding_status": row["inverse_encoding_status"],
                "missing_status": row["missing_status"],
                "discrete_numeric_status": row["discrete_numeric_status"],
                "row_count_status": row["row_count_status"],
                "original_issues": row["issues"],
                "whether_local_repair_is_possible": str(bool(recovery["whether_local_repair_is_possible"])).lower(),
                "whether_missing_can_be_restored": str(bool(recovery["whether_missing_can_be_restored"])).lower(),
                "whether_row_mismatch_can_be_explained": str(bool(recovery["whether_row_mismatch_can_be_explained"])).lower(),
                "whether_discrete_numeric_issue_can_be_fixed": str(bool(recovery["whether_discrete_numeric_issue_can_be_fixed"])).lower(),
                "newly_found_file_count": len(recovery.get("newly_found_files", [])),
                "written_file_count": len(recovery.get("synoutput_written_files", [])),
                "final_recommendation": recovery.get("final_recommendation", ""),
                "server_run_dir": recovery.get("server_run_dir", ""),
                "synoutput_asset_dir": recovery.get("synoutput_asset_dir", ""),
                "synthetic_csv_path": row["synthetic_csv_path"],
            }
        )

    merged_rows.sort(key=lambda item: (item["dataset_id"], item["model_id"], item["run_id"]))
    pending_rows = [row for row in merged_rows if row["status_bucket"] == "pending_local_repair_after_server_recovery"]
    rerun_rows = [row for row in merged_rows if row["status_bucket"] == "rerun_required_server_run_missing"]

    by_dataset_model_counter: Counter[tuple[str, str, str]] = Counter()
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in merged_rows:
        grouped_rows[(row["dataset_id"], row["model_id"])].append(row)
        by_dataset_model_counter[(row["dataset_id"], row["model_id"], row["status_bucket"])] += 1

    dataset_model_rows: list[dict[str, Any]] = []
    for (dataset_id, model_id), rows in sorted(grouped_rows.items()):
        dataset_model_rows.append(
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "problem_asset_count": len(rows),
                "pending_local_repair_count": sum(1 for row in rows if row["status_bucket"] == "pending_local_repair_after_server_recovery"),
                "rerun_required_count": sum(1 for row in rows if row["status_bucket"] == "rerun_required_server_run_missing"),
                "inverse_issue_assets": sum(1 for row in rows if row["inverse_encoding_status"] == "has_inverse_issues"),
                "missing_issue_assets": sum(1 for row in rows if row["missing_status"] == "has_missing_issues"),
                "discrete_numeric_issue_assets": sum(1 for row in rows if row["discrete_numeric_status"] == "has_discrete_numeric_issues"),
                "row_mismatch_assets": sum(1 for row in rows if row["row_count_status"] == "row_mismatch"),
            }
        )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged_rows:
        by_model[row["model_id"]].append(row)
        by_dataset[row["dataset_id"]].append(row)

    model_rows: list[dict[str, Any]] = []
    for model_id, rows in sorted(by_model.items()):
        model_rows.append(
            {
                "model_id": model_id,
                "problem_asset_count": len(rows),
                "pending_local_repair_count": sum(1 for row in rows if row["status_bucket"] == "pending_local_repair_after_server_recovery"),
                "rerun_required_count": sum(1 for row in rows if row["status_bucket"] == "rerun_required_server_run_missing"),
                "inverse_issue_assets": sum(1 for row in rows if row["inverse_encoding_status"] == "has_inverse_issues"),
                "missing_issue_assets": sum(1 for row in rows if row["missing_status"] == "has_missing_issues"),
                "discrete_numeric_issue_assets": sum(1 for row in rows if row["discrete_numeric_status"] == "has_discrete_numeric_issues"),
                "row_mismatch_assets": sum(1 for row in rows if row["row_count_status"] == "row_mismatch"),
            }
        )

    dataset_rows: list[dict[str, Any]] = []
    for dataset_id, rows in sorted(by_dataset.items()):
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "problem_asset_count": len(rows),
                "pending_local_repair_count": sum(1 for row in rows if row["status_bucket"] == "pending_local_repair_after_server_recovery"),
                "rerun_required_count": sum(1 for row in rows if row["status_bucket"] == "rerun_required_server_run_missing"),
                "inverse_issue_assets": sum(1 for row in rows if row["inverse_encoding_status"] == "has_inverse_issues"),
                "missing_issue_assets": sum(1 for row in rows if row["missing_status"] == "has_missing_issues"),
                "discrete_numeric_issue_assets": sum(1 for row in rows if row["discrete_numeric_status"] == "has_discrete_numeric_issues"),
                "row_mismatch_assets": sum(1 for row in rows if row["row_count_status"] == "row_mismatch"),
            }
        )

    summary_rows = [
        {"metric": "pro6000_problem_assets_before_local_resync", "value": len(merged_rows)},
        {"metric": "pro6000_pending_local_repair_after_server_recovery", "value": len(pending_rows)},
        {"metric": "pro6000_rerun_required_server_run_missing", "value": len(rerun_rows)},
        {"metric": "pro6000_assets_with_inverse_issues", "value": sum(1 for row in merged_rows if row["inverse_encoding_status"] == "has_inverse_issues")},
        {"metric": "pro6000_assets_with_missing_issues", "value": sum(1 for row in merged_rows if row["missing_status"] == "has_missing_issues")},
        {"metric": "pro6000_assets_with_discrete_numeric_issues", "value": sum(1 for row in merged_rows if row["discrete_numeric_status"] == "has_discrete_numeric_issues")},
        {"metric": "pro6000_assets_with_row_mismatch", "value": sum(1 for row in merged_rows if row["row_count_status"] == "row_mismatch")},
        {"metric": "pro6000_recovery_json_entries", "value": len(recovery_rows)},
        {"metric": "pro6000_recovery_entries_marked_local_repair_possible", "value": sum(1 for row in recovery_rows if row["whether_local_repair_is_possible"])},
        {"metric": "pro6000_recovery_entries_run_dir_missing", "value": sum(1 for row in recovery_rows if not row["whether_local_repair_is_possible"])},
        {"metric": "pro6000_audit_rows_missing_recovery_record", "value": len(missing_recovery_rows)},
    ]

    _write_csv(audit_dir / "pro6000_followup_assets.csv", merged_rows)
    _write_csv(audit_dir / "pro6000_pending_local_repair.csv", pending_rows)
    _write_csv(audit_dir / "pro6000_rerun_candidates.csv", rerun_rows)
    _write_csv(audit_dir / "pro6000_problem_counts_by_dataset_model.csv", dataset_model_rows)
    _write_csv(audit_dir / "pro6000_problem_counts_by_model.csv", model_rows)
    _write_csv(audit_dir / "pro6000_problem_counts_by_dataset.csv", dataset_rows)
    _write_csv(audit_dir / "pro6000_followup_summary_metrics.csv", summary_rows)

    xlsx_path = audit_dir / "pro6000_followup.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(dataset_model_rows).to_excel(writer, sheet_name="by_dataset_model", index=False)
        pd.DataFrame(model_rows).to_excel(writer, sheet_name="by_model", index=False)
        pd.DataFrame(dataset_rows).to_excel(writer, sheet_name="by_dataset", index=False)
        pd.DataFrame(pending_rows).to_excel(writer, sheet_name="pending_local_repair", index=False)
        pd.DataFrame(rerun_rows).to_excel(writer, sheet_name="rerun_candidates", index=False)
        pd.DataFrame(merged_rows).to_excel(writer, sheet_name="all_problem_assets", index=False)

    report_lines = [
        "# Pro6000 Follow-up",
        "",
        "- Current Windows workspace does not yet contain the recovered file contents copied back on server.",
        "- SSH to pro6000 from this machine is currently blocked at `banner exchange`, so a second local repair pass could not be executed here.",
        "",
        "## Summary",
        "",
        _markdown_table(summary_rows, ["metric", "value"]),
        "",
        "## Rerun Candidates",
        "",
        _markdown_table(
            rerun_rows,
            [
                "dataset_id",
                "model_id",
                "run_id",
                "status_bucket",
                "final_recommendation",
            ],
        ),
        "",
        "## By Model",
        "",
        _markdown_table(
            model_rows,
            [
                "model_id",
                "problem_asset_count",
                "pending_local_repair_count",
                "rerun_required_count",
                "inverse_issue_assets",
                "missing_issue_assets",
                "discrete_numeric_issue_assets",
                "row_mismatch_assets",
            ],
        ),
        "",
        "## By Dataset-Model",
        "",
        _markdown_table(
            dataset_model_rows,
            [
                "dataset_id",
                "model_id",
                "problem_asset_count",
                "pending_local_repair_count",
                "rerun_required_count",
                "inverse_issue_assets",
                "missing_issue_assets",
                "discrete_numeric_issue_assets",
                "row_mismatch_assets",
            ],
        ),
    ]
    (audit_dir / "pro6000_followup_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary_json = {
        "pro6000_problem_assets_before_local_resync": len(merged_rows),
        "pro6000_pending_local_repair_after_server_recovery": len(pending_rows),
        "pro6000_rerun_required_server_run_missing": len(rerun_rows),
        "output_dir": str(audit_dir),
        "xlsx_report": str(xlsx_path),
    }
    (audit_dir / "pro6000_followup_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
