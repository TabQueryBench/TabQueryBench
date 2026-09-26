#!/usr/bin/env python3
"""Build v2 workload inventories with explicit family/subitem metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.config.settings import DATA_DIR
from tqb_query.subitem_workload_v2.inventory import build_inventories_for_datasets, rebuild_inventory_summary
from tqb_query.subitem_workload_v2.paths import default_dataset_ids_for_line_version, normalize_line_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v2 workload inventories.")
    parser.add_argument("--line-version", type=str, default="v2", help="v2-v11, or a model artifact version such as v10.1.1_claude-opus-5 or v11.2.2-2_glm-5.3")
    parser.add_argument("--dataset-ids", type=str, default="", help="Comma-separated dataset ids.")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR, help="Dataset root.")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--planner-kind", type=str, choices=["rule", "cli", "agent-bind", "agent-select-bind"], default="rule",
        help="agent-bind (v10): model binds all applicable templates; agent-select-bind (v11): model also selects templates.",
    )
    parser.add_argument("--planner-model", type=str, default="gpt-5.4")
    parser.add_argument(
        "--ai-cli-preset", type=str, choices=["auto", "codex", "claude", "zai", "custom"], default="auto",
        help="agent-bind auto-routes GPT models to Codex, Claude models to Claude CLI, and GLM models to the Z.AI API.",
    )
    parser.add_argument("--ai-cli-command", type=str, default="")
    parser.add_argument("--grounding-version", type=str, default="", help="Model-specific version or run, e.g. v11.2.2-2 or v11.2.2-2_glm-5.3; defaults to the first run.")
    parser.add_argument("--agent-bind-problems-per-template", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--resume", action="store_true", help="Skip model-version inventories already completed on disk.")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    line_version = normalize_line_version(args.line_version)
    dataset_ids = [item.strip() for item in args.dataset_ids.split(",") if item.strip()]
    if not dataset_ids:
        dataset_ids = list(default_dataset_ids_for_line_version(line_version))
    if args.summary_only:
        summary = rebuild_inventory_summary(
            dataset_ids,
            line_version=line_version,
            planner_kind=args.planner_kind,
        )
    else:
        summary = build_inventories_for_datasets(
            dataset_ids,
            line_version=line_version,
            data_root=args.data_root,
            use_cache=args.use_cache,
            planner_kind=args.planner_kind,
            planner_model=args.planner_model,
            ai_cli_preset=args.ai_cli_preset,
            ai_cli_command=args.ai_cli_command,
            grounding_version=args.grounding_version,
            agent_bind_problems_per_template=args.agent_bind_problems_per_template,
            resume=args.resume,
        )
    print(f"[{summary.get('line_version', line_version)}-inventory] datasets={dataset_ids} summary={summary}")


if __name__ == "__main__":
    main()
