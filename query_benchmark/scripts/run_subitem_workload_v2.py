#!/usr/bin/env python3
"""Execute v2 workload inventories and materialize the v2 registry."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.subitem_workload_v2.paths import (
    dataset_inventory_path,
    default_dataset_ids_for_line_version,
    normalize_line_version,
)
from src.eval.subitem_workload_v2.runner import run_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated v2 workload line.")
    parser.add_argument("--line-version", type=str, choices=["v2", "v3", "v4"], default="v2")
    parser.add_argument("--dataset-ids", type=str, default="", help="Comma-separated dataset ids.")
    parser.add_argument("--engine", type=str, choices=["template", "cli"], default="template")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--ai-cli-preset", type=str, default="codex")
    parser.add_argument("--ai-cli-command", type=str, default="")
    parser.add_argument("--ai-cli-timeout-seconds", type=int, default=120)
    parser.add_argument("--ai-cli-retries", type=int, default=1)
    parser.add_argument("--ai-cli-answer-mode", type=str, choices=["local", "ai"], default="local")
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--sql-timeout-ms", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    line_version = normalize_line_version(args.line_version)
    dataset_ids = [item.strip() for item in args.dataset_ids.split(",") if item.strip()]
    if not dataset_ids:
        dataset_ids = list(default_dataset_ids_for_line_version(line_version))
    run_id = args.run_id or f"subitem_workload_{line_version}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    for dataset_id in dataset_ids:
        summary = run_inventory(
            inventory_path=dataset_inventory_path(dataset_id, line_version=line_version),
            run_id=run_id,
            line_version=line_version,
            engine=args.engine,
            model=args.model,
            ai_cli_preset=args.ai_cli_preset,
            ai_cli_command=args.ai_cli_command,
            ai_cli_timeout_seconds=args.ai_cli_timeout_seconds,
            ai_cli_retries=args.ai_cli_retries,
            ai_cli_answer_mode=args.ai_cli_answer_mode,
            row_limit=args.row_limit,
            sql_timeout_ms=args.sql_timeout_ms,
        )
        print(f"[{line_version}-run] dataset={dataset_id} summary={summary}")


if __name__ == "__main__":
    main()
