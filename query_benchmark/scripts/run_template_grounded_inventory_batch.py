#!/usr/bin/env python3
"""Run template-grounded dataset workloads batch by batch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import RUNS_DIR, ensure_runtime_dirs


def build_run_id(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset SQL workloads in batch.")
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        default=None,
        help="Directory containing <dataset_id>_questions.json files. Required for api/cli/template engines.",
    )
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default="",
        help="Optional comma-separated dataset filter. Required for cli-all when --inventory-dir is omitted.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["api", "cli", "cli-all", "template"],
        default="api",
        help="Execution engine forwarded to run_template_grounded_sql_agent.py.",
    )
    parser.add_argument("--model", type=str, default="", help="Optional model override.")
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="inventory_batch",
        help="Prefix used for the batch summary run id.",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=0,
        help="Optional cap on how many datasets to run.",
    )
    parser.add_argument(
        "--max-questions-per-dataset",
        type=int,
        default=0,
        help="Optional cap forwarded to each dataset run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to keep running after a dataset failure.",
    )
    parser.add_argument(
        "--parallel-datasets",
        type=int,
        default=1,
        help="How many dataset subprocesses to run concurrently. Defaults to 1 for backward compatibility.",
    )
    parser.add_argument("--ai-cli-preset", type=str, default="codex", help="Forwarded when CLI engines are used.")
    parser.add_argument("--ai-cli-command", type=str, default="", help="Optional custom local AI CLI command.")
    parser.add_argument("--ai-cli-timeout-seconds", type=int, default=120, help="Forwarded when CLI engines are used.")
    parser.add_argument("--ai-cli-retries", type=int, default=1, help="Forwarded when CLI engines are used.")
    parser.add_argument("--ai-cli-answer-mode", type=str, default="local", help="Forwarded when CLI engines are used.")
    parser.add_argument("--local-sql-row-limit", type=int, default=50, help="Forwarded to local runners.")
    parser.add_argument("--local-sql-timeout-ms", type=int, default=10000, help="Forwarded to local runners.")
    parser.add_argument("--cli-all-min-templates", type=int, default=10, help="Forwarded when --engine cli-all is used.")
    parser.add_argument("--cli-all-target-templates", type=int, default=12, help="Forwarded when --engine cli-all is used.")
    parser.add_argument(
        "--cli-all-min-problems-per-template",
        type=int,
        default=4,
        help="Forwarded when --engine cli-all is used.",
    )
    parser.add_argument(
        "--cli-all-max-problems-per-template",
        type=int,
        default=12,
        help="Forwarded when --engine cli-all is used.",
    )
    return parser.parse_args()


def _inventory_files(inventory_dir: Path, dataset_ids: set[str]) -> list[Path]:
    files = sorted(inventory_dir.glob("*_questions.json"))
    if dataset_ids:
        files = [path for path in files if path.name.removesuffix("_questions.json") in dataset_ids]
    return files


def _dataset_jobs(args: argparse.Namespace, dataset_filter: set[str]) -> list[dict[str, Any]]:
    if args.engine == "cli-all":
        if args.inventory_dir is not None:
            dataset_ids = [path.name.removesuffix("_questions.json") for path in _inventory_files(args.inventory_dir, dataset_filter)]
        else:
            dataset_ids = sorted(dataset_filter)
        if not dataset_ids:
            raise ValueError("cli-all batch mode requires --dataset-ids or an --inventory-dir that can imply dataset ids")
        if args.max_datasets > 0:
            dataset_ids = dataset_ids[: args.max_datasets]
        return [{"dataset_id": dataset_id, "inventory_path": None} for dataset_id in dataset_ids]

    if args.inventory_dir is None:
        raise ValueError("--inventory-dir is required unless --engine cli-all is used")
    files = _inventory_files(args.inventory_dir, dataset_filter)
    if args.max_datasets > 0:
        files = files[: args.max_datasets]
    return [
        {
            "dataset_id": path.name.removesuffix("_questions.json"),
            "inventory_path": path,
        }
        for path in files
    ]


def _run_dataset_job(job: dict[str, Any], args: argparse.Namespace, batch_dir: Path) -> dict[str, Any]:
    dataset_id = str(job["dataset_id"])
    inventory_path = job.get("inventory_path")
    child_run_prefix = f"{dataset_id}_{args.run_prefix}" if args.run_prefix.strip() else f"{dataset_id}_{args.engine}"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_template_grounded_sql_agent.py"),
        "--dataset-id",
        dataset_id,
        "--engine",
        args.engine,
        "--ai-cli-preset",
        args.ai_cli_preset,
        "--ai-cli-command",
        args.ai_cli_command,
        "--ai-cli-timeout-seconds",
        str(args.ai_cli_timeout_seconds),
        "--ai-cli-retries",
        str(args.ai_cli_retries),
        "--ai-cli-answer-mode",
        args.ai_cli_answer_mode,
        "--local-sql-row-limit",
        str(args.local_sql_row_limit),
        "--local-sql-timeout-ms",
        str(args.local_sql_timeout_ms),
        "--run-prefix",
        child_run_prefix,
    ]
    if inventory_path is not None:
        cmd.extend(["--questions-json", str(Path(inventory_path))])
    if args.model.strip():
        cmd.extend(["--model", args.model.strip()])
    if args.max_questions_per_dataset > 0:
        cmd.extend(["--max-questions", str(args.max_questions_per_dataset)])
    if args.engine == "cli-all":
        cmd.extend(
            [
                "--cli-all-min-templates",
                str(args.cli_all_min_templates),
                "--cli-all-target-templates",
                str(args.cli_all_target_templates),
                "--cli-all-min-problems-per-template",
                str(args.cli_all_min_problems_per_template),
                "--cli-all-max-problems-per-template",
                str(args.cli_all_max_problems_per_template),
            ]
        )

    print(f"[batch] dataset_id={dataset_id} engine={args.engine} inventory={inventory_path}")
    started_at = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    record = {
        "dataset_id": dataset_id,
        "inventory_path": str(Path(inventory_path).resolve()) if inventory_path is not None else None,
        "command": cmd,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode,
        "status": "completed" if completed.returncode == 0 else "failed",
        "stdout_path": str((batch_dir / f"{dataset_id}_stdout.txt").resolve()),
        "stderr_path": str((batch_dir / f"{dataset_id}_stderr.txt").resolve()),
    }
    (batch_dir / f"{dataset_id}_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (batch_dir / f"{dataset_id}_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    return record


def main() -> None:
    ensure_runtime_dirs()
    args = parse_args()
    dataset_filter = {item.strip() for item in args.dataset_ids.split(",") if item.strip()}
    jobs = _dataset_jobs(args, dataset_filter)

    if args.parallel_datasets > 1 and not args.continue_on_error:
        raise ValueError("--continue-on-error=false is only supported with --parallel-datasets 1")

    batch_run_id = build_run_id(args.run_prefix)
    batch_dir = RUNS_DIR / batch_run_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    if args.parallel_datasets <= 1:
        for job in jobs:
            record = _run_dataset_job(job, args, batch_dir)
            records.append(record)
            if record["status"] == "failed" and not args.continue_on_error:
                break
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel_datasets)) as executor:
            future_map = {
                executor.submit(_run_dataset_job, job, args, batch_dir): job["dataset_id"]
                for job in jobs
            }
            for future in as_completed(future_map):
                records.append(future.result())

    records.sort(key=lambda row: str(row["dataset_id"]))
    summary = {
        "batch_run_id": batch_run_id,
        "inventory_dir": str(args.inventory_dir.resolve()) if args.inventory_dir is not None else None,
        "engine": args.engine,
        "model": args.model,
        "dataset_filter": sorted(dataset_filter),
        "parallel_datasets": max(1, args.parallel_datasets),
        "dataset_count": len(records),
        "completed_dataset_count": sum(1 for row in records if row["status"] == "completed"),
        "failed_dataset_count": sum(1 for row in records if row["status"] == "failed"),
        "records": records,
    }
    (batch_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (batch_dir / "batch_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[batch] summary={batch_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
