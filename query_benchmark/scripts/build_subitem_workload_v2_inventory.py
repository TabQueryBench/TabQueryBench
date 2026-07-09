#!/usr/bin/env python3
"""Build v2 workload inventories with explicit family/subitem metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import DATA_DIR
from src.eval.subitem_workload_v2.inventory import build_inventories_for_datasets
from src.eval.subitem_workload_v2.paths import default_dataset_ids_for_line_version, normalize_line_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v2 workload inventories.")
    parser.add_argument("--line-version", type=str, choices=["v2", "v3", "v4"], default="v2")
    parser.add_argument("--dataset-ids", type=str, default="", help="Comma-separated dataset ids.")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR, help="Dataset root.")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--planner-kind", type=str, choices=["rule", "cli"], default="rule")
    parser.add_argument("--planner-model", type=str, default="gpt-5.4")
    parser.add_argument("--ai-cli-preset", type=str, default="codex")
    parser.add_argument("--ai-cli-command", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    line_version = normalize_line_version(args.line_version)
    dataset_ids = [item.strip() for item in args.dataset_ids.split(",") if item.strip()]
    if not dataset_ids:
        dataset_ids = list(default_dataset_ids_for_line_version(line_version))
    summary = build_inventories_for_datasets(
        dataset_ids,
        line_version=line_version,
        data_root=args.data_root,
        use_cache=args.use_cache,
        planner_kind=args.planner_kind,
        planner_model=args.planner_model,
        ai_cli_preset=args.ai_cli_preset,
        ai_cli_command=args.ai_cli_command,
    )
    print(f"[{line_version}-inventory] datasets={dataset_ids} summary={summary}")


if __name__ == "__main__":
    main()
