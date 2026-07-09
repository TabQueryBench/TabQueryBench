#!/usr/bin/env python3
"""Build one template-grounded dataset query set and compare it to a baseline run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.workload_grounding.queryset_builder import build_template_grounded_queryset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a template-grounded dataset queryset sidecar run.")
    parser.add_argument("--dataset-id", type=str, default="m4", help="Dataset ID to materialize.")
    parser.add_argument(
        "--core-library-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "template_library_v1.jsonl",
        help="Path to the grounded core template library.",
    )
    parser.add_argument(
        "--baseline-run-id",
        type=str,
        default="",
        help="Optional no-template baseline run_id. Defaults to latest matching benchmark run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_template_grounded_queryset(
        dataset_id=args.dataset_id,
        core_library_path=args.core_library_path,
        baseline_run_id=args.baseline_run_id or None,
    )
    print(f"[template-grounded-queryset] run_id={result['run_id']}")
    print(f"[template-grounded-queryset] benchmark_package={result['benchmark_package_dir']}")
    if result.get("baseline_run_id"):
        print(f"[template-grounded-queryset] baseline_run_id={result['baseline_run_id']}")
    metrics = result["grounded_metrics"]
    print(
        "[template-grounded-queryset] "
        f"queries={metrics['query_count']} "
        f"bundles={metrics['bundle_count']} "
        f"production_like_query_rate={metrics['production_like_query_rate']}"
    )


if __name__ == "__main__":
    main()
