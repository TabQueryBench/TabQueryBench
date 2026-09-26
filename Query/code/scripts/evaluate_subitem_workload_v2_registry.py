#!/usr/bin/env python3
"""Export registry-backed evaluation artifacts for the v2 workload line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.subitem_workload_v2.evaluation import evaluate_registry
from tqb_query.subitem_workload_v2.paths import V2_EVALUATION_FINAL_DIR, registry_jsonl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a v2 query registry.")
    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (V2_EVALUATION_FINAL_DIR / f"evaluation_{args.run_id}_{args.dataset_id}")
    summary = evaluate_registry(
        registry_path=registry_jsonl_path(args.run_id),
        dataset_id=args.dataset_id,
        run_id=args.run_id,
        output_dir=output_dir,
    )
    print(f"[v2-evaluation] summary={summary}")


if __name__ == "__main__":
    main()
