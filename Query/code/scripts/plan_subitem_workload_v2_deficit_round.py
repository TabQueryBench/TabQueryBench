#!/usr/bin/env python3
"""Plan the next v2 deficit round without rerunning SQL generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.subitem_workload_v2.deficit_loop import build_deficit_round_plan
from tqb_query.subitem_workload_v2.paths import V2_DATA_ROOT, dataset_inventory_path, registry_jsonl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a deficit-only v2 follow-up round.")
    parser.add_argument("--dataset-ids", type=str, default="c2", help="Comma-separated dataset ids.")
    parser.add_argument("--run-id", type=str, required=True, help="Existing v2 run id to audit.")
    parser.add_argument("--round-id", type=str, default="deficit_round_01", help="Label for the planned next round.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for deficit round output root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_ids = [item.strip() for item in args.dataset_ids.split(",") if item.strip()]
    output_dir = args.output_dir or (V2_DATA_ROOT / "deficit_rounds" / args.round_id)
    summary = build_deficit_round_plan(
        inventory_paths=[dataset_inventory_path(dataset_id) for dataset_id in dataset_ids],
        registry_path=registry_jsonl_path(args.run_id),
        round_id=args.round_id,
        output_dir=output_dir,
    )
    print(f"[v2-deficit-round] summary={summary}")


if __name__ == "__main__":
    main()
