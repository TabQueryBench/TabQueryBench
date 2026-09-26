#!/usr/bin/env python3
"""Summarize Codex CLI API usage and estimated USD cost by dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PER_MILLION_USD = 5.00
DEFAULT_CACHED_INPUT_PER_MILLION_USD = 0.50
DEFAULT_OUTPUT_PER_MILLION_USD = 30.00
PRICING_SOURCE_URL = "https://developers.openai.com/api/docs/pricing"
PRICING_MODE = "gpt-5.5 standard short-context API pricing"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_from_manifest_path(path: Path) -> str:
    # logs/subitem_workload_v5/runs/<run_id>/<dataset>/artifacts/<query>/run_manifest.json
    return path.parts[-4]


def calculate_cost(
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    input_per_million_usd: float,
    cached_input_per_million_usd: float,
    output_per_million_usd: float,
) -> float:
    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    return (
        uncached_input_tokens / 1_000_000 * input_per_million_usd
        + max(cached_input_tokens, 0) / 1_000_000 * cached_input_per_million_usd
        + output_tokens / 1_000_000 * output_per_million_usd
    )


def build_rows(
    *,
    run_root: Path,
    run_ids: list[str],
    input_per_million_usd: float,
    cached_input_per_million_usd: float,
    output_per_million_usd: float,
    pricing_source_url: str,
    pricing_mode: str,
) -> list[dict[str, Any]]:
    rows_by_dataset: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "dataset_id": "",
            "run_ids": set(),
            "completed_queries": 0,
            "failed_queries": 0,
            "ai_cli_calls": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "usage_source_counts": defaultdict(int),
            "missing_usage_summary_count": 0,
            "missing_raw_response_count": 0,
        }
    )

    for run_id in run_ids:
        for manifest_path in (run_root / run_id).glob("*/artifacts/*/run_manifest.json"):
            dataset_id = dataset_from_manifest_path(manifest_path)
            row = rows_by_dataset[dataset_id]
            row["dataset_id"] = dataset_id
            row["run_ids"].add(run_id)
            manifest = load_json(manifest_path)
            status = str(manifest.get("status") or "")
            if status == "completed":
                row["completed_queries"] += 1
            elif status == "failed":
                row["failed_queries"] += 1
            usage = manifest.get("usage_summary") or {}
            if not usage:
                row["missing_usage_summary_count"] += 1
            row["ai_cli_calls"] += int(usage.get("ai_cli_calls") or 0)
            row["input_tokens"] += int(usage.get("input_tokens") or 0)
            row["cached_input_tokens"] += int(usage.get("cached_input_tokens") or 0)
            row["output_tokens"] += int(usage.get("output_tokens") or 0)
            row["total_tokens"] += int(usage.get("total_tokens") or 0)
            row["usage_source_counts"][str(usage.get("usage_source") or "none")] += 1

            if int(usage.get("ai_cli_calls") or 0) > 0:
                raw_paths = list(manifest_path.parent.glob("cli/*response*.raw.txt"))
                if not raw_paths:
                    row["missing_raw_response_count"] += 1

    rows: list[dict[str, Any]] = []
    for dataset_id, row in rows_by_dataset.items():
        cost = calculate_cost(
            input_tokens=int(row["input_tokens"]),
            cached_input_tokens=int(row["cached_input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            input_per_million_usd=input_per_million_usd,
            cached_input_per_million_usd=cached_input_per_million_usd,
            output_per_million_usd=output_per_million_usd,
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "run_ids": ";".join(sorted(row["run_ids"])),
                "completed_queries": row["completed_queries"],
                "failed_queries": row["failed_queries"],
                "ai_cli_calls": row["ai_cli_calls"],
                "input_tokens": row["input_tokens"],
                "cached_input_tokens": row["cached_input_tokens"],
                "uncached_input_tokens": max(int(row["input_tokens"]) - int(row["cached_input_tokens"]), 0),
                "output_tokens": row["output_tokens"],
                "total_tokens": row["total_tokens"],
                "input_per_million_usd": input_per_million_usd,
                "cached_input_per_million_usd": cached_input_per_million_usd,
                "output_per_million_usd": output_per_million_usd,
                "estimated_cost_usd": round(cost, 6),
                "usage_source_counts": ";".join(
                    f"{key}:{value}" for key, value in sorted(row["usage_source_counts"].items())
                ),
                "missing_usage_summary_count": row["missing_usage_summary_count"],
                "missing_raw_response_count": row["missing_raw_response_count"],
                "pricing_source_url": pricing_source_url,
                "pricing_mode": pricing_mode,
            }
        )
    return sorted(rows, key=lambda item: item["dataset_id"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset_id",
        "run_ids",
        "completed_queries",
        "failed_queries",
        "ai_cli_calls",
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "total_tokens",
        "input_per_million_usd",
        "cached_input_per_million_usd",
        "output_per_million_usd",
        "estimated_cost_usd",
        "usage_source_counts",
        "missing_usage_summary_count",
        "missing_raw_response_count",
        "pricing_source_url",
        "pricing_mode",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    pricing_source_url: str,
    pricing_mode: str,
) -> None:
    total = {
        "dataset_count": len(rows),
        "completed_queries": sum(int(row["completed_queries"]) for row in rows),
        "failed_queries": sum(int(row["failed_queries"]) for row in rows),
        "ai_cli_calls": sum(int(row["ai_cli_calls"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "cached_input_tokens": sum(int(row["cached_input_tokens"]) for row in rows),
        "uncached_input_tokens": sum(int(row["uncached_input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "estimated_cost_usd": round(sum(float(row["estimated_cost_usd"]) for row in rows), 6),
        "pricing_source_url": pricing_source_url,
        "pricing_mode": pricing_mode,
    }
    path.write_text(json.dumps(total, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-version", default="v5")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--input-per-million-usd", type=float, default=DEFAULT_INPUT_PER_MILLION_USD)
    parser.add_argument(
        "--cached-input-per-million-usd",
        type=float,
        default=DEFAULT_CACHED_INPUT_PER_MILLION_USD,
    )
    parser.add_argument("--output-per-million-usd", type=float, default=DEFAULT_OUTPUT_PER_MILLION_USD)
    parser.add_argument("--pricing-source-url", default=PRICING_SOURCE_URL)
    parser.add_argument("--pricing-mode", default=PRICING_MODE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(
        run_root=args.run_root or Path(f"logs/subitem_workload_{args.line_version}/runs"),
        run_ids=args.run_ids,
        input_per_million_usd=args.input_per_million_usd,
        cached_input_per_million_usd=args.cached_input_per_million_usd,
        output_per_million_usd=args.output_per_million_usd,
        pricing_source_url=args.pricing_source_url,
        pricing_mode=args.pricing_mode,
    )
    write_csv(args.output_csv, rows)
    write_summary(
        args.output_summary,
        rows,
        pricing_source_url=args.pricing_source_url,
        pricing_mode=args.pricing_mode,
    )
    print(json.dumps({"rows": len(rows), "output_csv": str(args.output_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
