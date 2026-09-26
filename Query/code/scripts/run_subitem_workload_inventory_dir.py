#!/usr/bin/env python3
"""Run a directory of explicit workload inventory json files."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.config.settings import DATA_DIR
from tqb_query.subitem_workload_v2.paths import normalize_line_version
from tqb_query.subitem_workload_v2.runner import run_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explicit workload inventory json files.")
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--dataset-ids", type=str, default="", help="Optional comma-separated dataset filter.")
    parser.add_argument("--line-version", type=str, choices=["v2", "v3", "v4", "v5", "v6", "v7"], default="v2")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--engine", type=str, choices=["template", "cli"], default="cli")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--ai-cli-preset", type=str, default="codex")
    parser.add_argument("--ai-cli-command", type=str, default="")
    parser.add_argument("--ai-cli-timeout-seconds", type=int, default=120)
    parser.add_argument("--ai-cli-retries", type=int, default=1)
    parser.add_argument("--ai-cli-answer-mode", type=str, choices=["local", "ai"], default="local")
    parser.add_argument(
        "--disable-ai-cli-template-fallback",
        action="store_true",
        help="Fail network-blocked AI CLI questions instead of using deterministic template fallback.",
    )
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--sql-timeout-ms", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    line_version = normalize_line_version(args.line_version)
    run_id = args.run_id or f"subitem_workload_{line_version}_inventorydir_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    inventory_paths = sorted(args.inventory_dir.glob(f"*_minimal_rerun_inventory_{line_version}.json"))
    dataset_filter = {item.strip() for item in args.dataset_ids.split(",") if item.strip()}
    if dataset_filter:
        inventory_paths = [path for path in inventory_paths if path.name.split("_minimal_rerun_inventory_", 1)[0] in dataset_filter]
    if not inventory_paths:
        raise SystemExit(f"No rerun inventories found under {args.inventory_dir}")
    summaries = []
    for inventory_path in inventory_paths:
        summary = run_inventory(
            inventory_path=inventory_path,
            run_id=run_id,
            line_version=line_version,
            data_root=args.data_root,
            engine=args.engine,
            model=args.model,
            ai_cli_preset=args.ai_cli_preset,
            ai_cli_command=args.ai_cli_command,
            ai_cli_timeout_seconds=args.ai_cli_timeout_seconds,
            ai_cli_retries=args.ai_cli_retries,
            ai_cli_answer_mode=args.ai_cli_answer_mode,
            allow_ai_cli_template_fallback=not args.disable_ai_cli_template_fallback,
            row_limit=args.row_limit,
            sql_timeout_ms=args.sql_timeout_ms,
        )
        summaries.append(summary)
        print(f"[{line_version}-inventory-dir] dataset={summary['dataset_id']} summary={summary}")
    print(json.dumps({"run_id": run_id, "dataset_count": len(summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
