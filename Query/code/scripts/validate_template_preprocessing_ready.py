#!/usr/bin/env python3
"""Validate template preprocessing readiness without doing final template selection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.config.settings import DATA_DIR
from tqb_query.data.bundle import load_dataset_bundle
from tqb_query.workload_grounding.question_inventory import build_template_candidate_pool

IGNORED_DATASET_DIRS = {"artifacts", "workload_grounding", "SynData"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate template preprocessing across datasets.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_DIR,
        help="Root directory containing datasets.",
    )
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default="",
        help="Optional comma-separated dataset ids. Defaults to all datasets under data root.",
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "agent_candidate_spec_all_core_v1.json",
        help="Candidate spec JSON path.",
    )
    parser.add_argument(
        "--spec-bucket",
        type=str,
        default="all_core",
        help="Spec bucket used for inventory validation.",
    )
    parser.add_argument(
        "--template-library",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "template_library_v1.jsonl",
        help="Template library JSONL path.",
    )
    parser.add_argument(
        "--portability-report",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "template_portability_report_v1.csv",
        help="Portability report CSV path.",
    )
    parser.add_argument(
        "--candidate-pool-dir",
        "--inventory-dir",
        dest="candidate_pool_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "inventories" / "template_candidate_pools_v1",
        help="Directory where per-dataset candidate-pool JSON files are written.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "reports" / "preprocessing_candidate_pool_validation_v1",
        help="Directory where validation summary files are written.",
    )
    parser.add_argument("--min-templates", type=int, default=10)
    parser.add_argument("--target-templates", type=int, default=12, help=argparse.SUPPRESS)
    parser.add_argument("--min-problems-per-template", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--max-problems-per-template", type=int, default=12, help=argparse.SUPPRESS)
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _list_dataset_ids(data_root: Path) -> list[str]:
    dataset_ids: list[str] = []
    for path in sorted(data_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name in IGNORED_DATASET_DIRS:
            continue
        dataset_ids.append(path.name)
    return dataset_ids


def main() -> None:
    args = parse_args()
    dataset_ids = (
        [item.strip() for item in args.dataset_ids.split(",") if item.strip()]
        if args.dataset_ids.strip()
        else _list_dataset_ids(args.data_root)
    )
    args.candidate_pool_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []

    for dataset_id in dataset_ids:
        record: dict[str, object] = {"dataset_id": dataset_id, "validated_at": _now_iso()}
        try:
            bundle = load_dataset_bundle(dataset_id=dataset_id, data_root=args.data_root, strict=True)
            record["bundle_ready"] = True
            payload = build_template_candidate_pool(
                dataset_id=dataset_id,
                spec_path=args.spec_path,
                spec_bucket=args.spec_bucket,
                core_library_path=args.template_library,
                portability_report_path=args.portability_report,
                data_root=args.data_root,
                min_templates=args.min_templates,
            )
            output_path = args.candidate_pool_dir / f"{dataset_id}_candidate_pool.json"
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            record.update(
                {
                    "status": "ok",
                    "main_csv_path": str(bundle.main_csv_path),
                    "eligible_template_count": payload["eligible_template_count"],
                    "screened_template_count": payload["screened_template_count"],
                    "agent_selection_ready": payload["agent_selection_ready"],
                    "agent_selection_gap": payload["agent_selection_gap"],
                    "screening_status_counts": payload["screening_status_counts"],
                    "portability_counts": payload["portability_counts"],
                    "output_path": str(output_path),
                }
            )
            print(
                f"[validate] dataset_id={dataset_id} status=ok eligible={payload['eligible_template_count']} "
                f"screened={payload['screened_template_count']} ready={payload['agent_selection_ready']}"
            )
        except Exception as exc:  # noqa: BLE001 - validation should continue across datasets.
            record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print(f"[validate] dataset_id={dataset_id} status=failed error={type(exc).__name__}: {exc}")
        records.append(record)

    summary = {
        "generated_at": _now_iso(),
        "dataset_count": len(records),
        "success_count": sum(1 for row in records if row.get("status") == "ok"),
        "failure_count": sum(1 for row in records if row.get("status") != "ok"),
        "records": records,
    }
    (args.report_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (args.report_dir / "validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_id",
                "status",
                "eligible_template_count",
                "screened_template_count",
                "agent_selection_ready",
                "agent_selection_gap",
                "error_type",
                "error_message",
                "output_path",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "dataset_id": row.get("dataset_id"),
                    "status": row.get("status"),
                    "eligible_template_count": row.get("eligible_template_count"),
                    "screened_template_count": row.get("screened_template_count"),
                    "agent_selection_ready": row.get("agent_selection_ready"),
                    "agent_selection_gap": row.get("agent_selection_gap"),
                    "error_type": row.get("error_type"),
                    "error_message": row.get("error_message"),
                    "output_path": row.get("output_path"),
                }
            )


if __name__ == "__main__":
    main()
