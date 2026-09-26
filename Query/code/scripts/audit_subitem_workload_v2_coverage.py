#!/usr/bin/env python3
"""Audit v2 coverage from the explicit query registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.subitem_workload_v2.coverage_gate import summarize_coverage
from tqb_query.subitem_workload_v2.paths import dataset_inventory_path, registry_jsonl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v2 registry coverage.")
    parser.add_argument("--dataset-ids", type=str, default="c2", help="Comma-separated dataset ids.")
    parser.add_argument("--run-id", type=str, required=True, help="Existing v2 run id.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_ids = [item.strip() for item in args.dataset_ids.split(",") if item.strip()]
    summary = summarize_coverage(
        inventory_paths=[dataset_inventory_path(dataset_id) for dataset_id in dataset_ids],
        registry_path=registry_jsonl_path(args.run_id),
        output_dir=args.output_dir,
    )
    print(f"[v2-coverage] summary={summary}")


if __name__ == "__main__":
    main()
