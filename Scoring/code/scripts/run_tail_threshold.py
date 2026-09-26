#!/usr/bin/env python3
"""Run the global tail-threshold sensitivity experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_scoring.eval.tail_threshold.runner import (
    DEFAULT_THRESHOLD_PCTS,
    build_tail_threshold_preview,
    run_tail_threshold_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global tail-threshold sensitivity diagnostics.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional output run tag.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset ids. Empty means all datasets.",
    )
    parser.add_argument(
        "--root-names",
        type=str,
        default="",
        help=(
            "Optional comma-separated synthetic root names. "
            "Example: TabQueryBench-SynDataSuccess-main"
        ),
    )
    parser.add_argument(
        "--threshold-percentages",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_THRESHOLD_PCTS),
        help="Comma-separated tail thresholds in percentage points.",
    )
    parser.add_argument(
        "--all-asset-runs",
        action="store_true",
        help="Disable latest-only filtering within the same model/server.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers across datasets.",
    )
    parser.add_argument(
        "--representatives-per-prefix",
        type=int,
        default=2,
        help="How many representative datasets to keep for each prefix family.",
    )
    parser.add_argument(
        "--plot-only-from-run-dir",
        type=Path,
        default=None,
        help="Instead of recomputing dataset outputs, rebuild summaries/figures from an existing run dir.",
    )
    return parser.parse_args()


def _parse_threshold_percentages(text: str) -> list[float]:
    values: list[float] = []
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()] or None
    root_names = [item.strip() for item in args.root_names.split(",") if item.strip()] or None
    if args.plot_only_from_run_dir is not None:
        manifest = build_tail_threshold_preview(
            source_run_dir=args.plot_only_from_run_dir,
            run_tag=args.run_tag,
            latest_only=not args.all_asset_runs,
            threshold_percentages=_parse_threshold_percentages(args.threshold_percentages),
            representatives_per_prefix=max(1, args.representatives_per_prefix),
        )
    else:
        manifest = run_tail_threshold_experiment(
            run_tag=args.run_tag,
            datasets=datasets,
            latest_only=not args.all_asset_runs,
            root_names=root_names,
            threshold_percentages=_parse_threshold_percentages(args.threshold_percentages),
            max_workers=max(1, args.max_workers),
            representatives_per_prefix=max(1, args.representatives_per_prefix),
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
