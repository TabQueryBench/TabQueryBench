#!/usr/bin/env python3
"""Run SQL v2, validation, and/or distance evaluation on hyperparameter train-only outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.analysis.runner import run_sql_analysis
from src.eval.distance.runner import run_distance_evaluation
from src.eval.validation.runner import run_validation_evaluation

DEFAULT_DATASETS = ("c2", "m4", "n3")
HYPERPARAMETER_ROOT_NAME = "Hyperparameter-trainonly-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation for hyperparameter synthetic outputs.")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset ids.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["sql", "distance"],
        choices=["sql", "validation", "distance"],
        help="Which evaluation tasks to run.",
    )
    parser.add_argument("--run-tag-prefix", type=str, default="hyperparameter_eval", help="Prefix for generated run tags.")
    parser.add_argument("--latest-only", action="store_true", help="Use only the latest asset per model/dataset.")
    parser.add_argument("--max-workers", type=int, default=1, help="Dataset-level parallelism.")
    parser.add_argument("--query-row-limit", type=int, default=0, help="Optional SQL row limit.")
    parser.add_argument("--max-sql-per-dataset", type=int, default=0, help="Optional cap on SQL statements per dataset.")
    parser.add_argument("--latex-engine", type=str, default=None, help="Optional LaTeX engine for final bundles.")
    parser.add_argument("--families", nargs="*", default=None, help="Optional family ids to limit SQL analysis.")
    parser.add_argument("--skip-final-publish", action="store_true", help="Do not publish SQL final bundle into Evaluation/analysis/final.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_ids = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not dataset_ids:
        dataset_ids = list(DEFAULT_DATASETS)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_tag = f"{args.run_tag_prefix}_{stamp}"
    results: dict[str, object] = {
        "datasets": dataset_ids,
        "tasks": list(args.tasks),
        "root_names": [HYPERPARAMETER_ROOT_NAME],
        "latest_only": bool(args.latest_only),
    }

    if "sql" in args.tasks:
        sql_result = run_sql_analysis(
            run_tag=f"{base_tag}_sql",
            datasets=dataset_ids,
            latest_only=args.latest_only,
            sql_source_version="v2",
            include_all_sql_statements=True,
            max_sql_per_dataset=max(0, int(args.max_sql_per_dataset)),
            query_row_limit=max(0, int(args.query_row_limit)),
            max_workers=max(1, int(args.max_workers)),
            family_filter=args.families,
            latex_engine=args.latex_engine,
            root_names=[HYPERPARAMETER_ROOT_NAME],
            publish_final=not args.skip_final_publish,
        )
        results["sql"] = {
            "run_dir": str(sql_result["run_dir"]),
            "manifest_path": str(Path(sql_result["run_dir"]) / "manifest.json"),
        }

    if "distance" in args.tasks:
        distance_result = run_distance_evaluation(
            run_tag=f"{base_tag}_distance",
            datasets=dataset_ids,
            latest_only=args.latest_only,
            max_workers=max(1, int(args.max_workers)),
            latex_engine=args.latex_engine,
            root_names=[HYPERPARAMETER_ROOT_NAME],
        )
        results["distance"] = {
            "run_dir": str(distance_result["run_dir"]),
            "manifest_path": str(Path(distance_result["run_dir"]) / "manifest.json"),
        }

    if "validation" in args.tasks:
        validation_result = run_validation_evaluation(
            run_tag=f"{base_tag}_validation",
            datasets=dataset_ids,
            latest_only=args.latest_only,
            max_workers=max(1, int(args.max_workers)),
            root_names=[HYPERPARAMETER_ROOT_NAME],
        )
        results["validation"] = {
            "run_dir": str(validation_result["run_dir"]),
            "manifest_path": str(Path(validation_result["run_dir"]) / "manifest.json"),
        }

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
