#!/usr/bin/env python3
"""Rerun failed v2 workload queries only."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.subitem_workload_v2.failure_rerun import write_failed_rerun_inventories
from src.eval.subitem_workload_v2.paths import normalize_line_version
from src.eval.subitem_workload_v2.runner import run_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun failed subitem workload v2 queries only.")
    parser.add_argument("--line-version", type=str, choices=["v2", "v3", "v4"], default="v2")
    parser.add_argument("--source-run-ids", nargs="+", required=True)
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--ai-cli-preset", type=str, default="codex")
    parser.add_argument("--ai-cli-command", type=str, default="")
    parser.add_argument("--ai-cli-timeout-seconds", type=int, default=360)
    parser.add_argument("--ai-cli-retries", type=int, default=4)
    parser.add_argument("--ai-cli-answer-mode", type=str, choices=["local", "ai"], default="local")
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--sql-timeout-ms", type=int, default=15000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    line_version = normalize_line_version(args.line_version)
    run_id = args.run_id or f"subitem_workload_{line_version}_failed_rerun_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_roots = [PROJECT_ROOT / "logs" / f"subitem_workload_{line_version}" / "runs" / run_id_part for run_id_part in args.source_run_ids]
    inventory_paths = write_failed_rerun_inventories(run_roots=run_roots, rerun_tag=run_id, line_version=line_version)
    print(f"[{line_version}-rerun] inventory_count={len(inventory_paths)} run_id={run_id}")
    for inventory_path in inventory_paths:
        summary = run_inventory(
            inventory_path=inventory_path,
            run_id=run_id,
            line_version=line_version,
            engine="cli",
            model=args.model,
            ai_cli_preset=args.ai_cli_preset,
            ai_cli_command=args.ai_cli_command,
            ai_cli_timeout_seconds=args.ai_cli_timeout_seconds,
            ai_cli_retries=args.ai_cli_retries,
            ai_cli_answer_mode=args.ai_cli_answer_mode,
            row_limit=args.row_limit,
            sql_timeout_ms=args.sql_timeout_ms,
        )
        print(f"[{line_version}-rerun] dataset={summary['dataset_id']} summary={summary}")


if __name__ == "__main__":
    main()
