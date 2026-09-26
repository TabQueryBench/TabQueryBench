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

from tqb_query.config.settings import DATA_DIR
from tqb_query.subitem_workload_v2.paths import (
    dataset_inventory_path,
    default_dataset_ids_for_line_version,
    normalize_line_version,
)
from tqb_query.subitem_workload_v2.runner import run_inventory
from tqb_query.subitem_workload_v2.paths import line_version_family
from tqb_query.workload_grounding.v10_versions import AGENT_LINE_VERSIONS, resolve_model_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated v2 workload line.")
    parser.add_argument("--line-version", type=str, default="v2", help="v2-v11, or a model artifact version such as v10.1.1_claude-opus-5 or v11.2.2-2_glm-5.3")
    parser.add_argument("--dataset-ids", type=str, default="", help="Comma-separated dataset ids.")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR, help="Dataset root.")
    parser.add_argument("--engine", type=str, choices=["template", "cli"], default="template")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--grounding-version", type=str, default="", help="Model-specific version or run, e.g. v11.2.2-2 or v11.2.2-2_glm-5.3; defaults to the first run.")
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
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    line_version = normalize_line_version(args.line_version)
    family = line_version_family(line_version)
    if family in AGENT_LINE_VERSIONS:
        requested = args.grounding_version or (line_version if line_version != family else "")
        line_version = resolve_model_version(args.model, line_family=family, grounding_version=requested).grounding_version
    dataset_ids = [item.strip() for item in args.dataset_ids.split(",") if item.strip()]
    if not dataset_ids:
        dataset_ids = list(default_dataset_ids_for_line_version(line_version))
    run_id = args.run_id or f"subitem_workload_{line_version}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    for dataset_id in dataset_ids:
        summary = run_inventory(
            inventory_path=dataset_inventory_path(dataset_id, line_version=line_version),
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
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        # The runner already logged this dataset's progress and totals; keep the tail short.
        print(
            f"[{line_version}-run] {dataset_id} registry+{summary['row_count']} rows "
            f"accepted={summary['accepted_count']} -> {Path(summary['registry_path']).name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
