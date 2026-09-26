#!/usr/bin/env python3
"""Build dataset-level question inventories from the screened all-core template pool."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.config.settings import DATA_DIR, DEFAULT_USAGE_CSV_PATH, MODEL_PRICING_CONFIG_PATH
from tqb_query.usage.logger import UsageCSVLogger
from tqb_query.usage.pricing import load_pricing_config
from tqb_query.workload_grounding.question_inventory import build_full_question_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full grounded question inventories for one or more datasets.")
    parser.add_argument("--dataset-ids", type=str, default="c2,m4,n1", help="Comma-separated dataset ids.")
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "agent_candidate_spec_all_core_v1.json",
        help="Path to candidate spec JSON.",
    )
    parser.add_argument("--spec-bucket", type=str, default="all_core", help="List bucket inside the candidate spec.")
    parser.add_argument(
        "--template-library",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "template_library_v1.jsonl",
        help="Path to core template library JSONL.",
    )
    parser.add_argument(
        "--portability-report",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "template_portability_report_v1.csv",
        help="Path to portability report CSV.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_DIR,
        help="Root directory containing dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "full_question_inventory_v1",
        help="Directory where dataset inventory JSON files will be written.",
    )
    parser.add_argument(
        "--min-templates",
        type=int,
        default=10,
        help="Minimum number of selected template/problem types per dataset.",
    )
    parser.add_argument(
        "--target-templates",
        type=int,
        default=12,
        help="Planner target for selected template/problem types per dataset.",
    )
    parser.add_argument(
        "--min-problems-per-template",
        type=int,
        default=4,
        help="Minimum number of generated problems per selected template.",
    )
    parser.add_argument(
        "--max-problems-per-template",
        type=int,
        default=12,
        help="Maximum number of generated problems per selected template.",
    )
    parser.add_argument(
        "--planner-model",
        type=str,
        default="",
        help="Optional LLM model used for constrained template/problem planning.",
    )
    parser.add_argument(
        "--usage-csv",
        type=Path,
        default=DEFAULT_USAGE_CSV_PATH,
        help="CSV path for usage logs.",
    )
    parser.add_argument(
        "--pricing-config",
        type=Path,
        default=MODEL_PRICING_CONFIG_PATH,
        help="JSON config path for model pricing.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="inventory",
        help="Prefix used to tag planner-side usage records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_ids = [item.strip() for item in args.dataset_ids.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_run_id = f"{args.run_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    usage_logger = UsageCSVLogger(args.usage_csv)
    pricing_config = load_pricing_config(args.pricing_config)

    summary: dict[str, object] = {
        "dataset_ids": dataset_ids,
        "planner_run_id": batch_run_id,
        "spec_path": str(args.spec_path.resolve()),
        "spec_bucket": args.spec_bucket,
        "template_library": str(args.template_library.resolve()),
        "portability_report": str(args.portability_report.resolve()),
        "planner_model": args.planner_model,
        "inventories": {},
    }

    for dataset_id in dataset_ids:
        payload = build_full_question_inventory(
            dataset_id=dataset_id,
            spec_path=args.spec_path,
            spec_bucket=args.spec_bucket,
            core_library_path=args.template_library,
            portability_report_path=args.portability_report,
            data_root=args.data_root,
            min_templates=args.min_templates,
            target_templates=args.target_templates,
            min_problems_per_template=args.min_problems_per_template,
            max_problems_per_template=args.max_problems_per_template,
            planner_model=args.planner_model or None,
            planner_run_id=f"{batch_run_id}_{dataset_id}",
            usage_logger=usage_logger,
            pricing_config=pricing_config,
        )
        output_path = args.output_dir / f"{dataset_id}_questions.json"
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["inventories"][dataset_id] = {
            "path": str(output_path.resolve()),
            "selected_template_count": payload["selected_template_count"],
            "inventory_count": payload["inventory_count"],
            "problem_count": payload["problem_count"],
            "family_counts": payload["family_counts"],
            "skipped_count": len(payload["skipped"]),
        }
        print(
            f"[inventory] dataset_id={dataset_id} templates={payload['selected_template_count']} "
            f"problems={payload['problem_count']} families={payload['family_counts']} output={output_path}"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[inventory] summary={summary_path}")


if __name__ == "__main__":
    main()
