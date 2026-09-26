from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def natural_dataset_key(dataset_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", dataset_id)
    if not match:
        return (dataset_id.lower(), -1)
    return (match.group(1).lower(), int(match.group(2)))


def format_int(value: int) -> str:
    return f"{value:,}"


def tex_escape(text: str) -> str:
    escaped = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(escaped.get(ch, ch) for ch in str(text))


@dataclass
class DatasetTokenRow:
    dataset_id: str
    planner_calls: int = 0
    planner_input_tokens: int = 0
    planner_cached_input_tokens: int = 0
    planner_output_tokens: int = 0
    planner_total_tokens: int = 0
    generation_completed_queries: int = 0
    generation_failed_queries: int = 0
    generation_ai_cli_calls: int = 0
    generation_input_tokens: int = 0
    generation_cached_input_tokens: int = 0
    generation_output_tokens: int = 0
    generation_total_tokens: int = 0
    combined_total_tokens: int = 0

    def finalize(self) -> None:
        self.combined_total_tokens = (
            self.planner_total_tokens + self.generation_total_tokens
        )


def iter_inventory_files(inventory_dir: Path, line_version: str) -> Iterable[Path]:
    suffix = f"_inventory_{line_version}.json"
    return sorted(
        inventory_dir.glob(f"*{suffix}"),
        key=lambda p: natural_dataset_key(p.stem.replace(suffix.removesuffix(".json"), "")),
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(inventory_dir: Path, run_roots: list[Path], line_version: str = "v2") -> list[DatasetTokenRow]:
    rows_by_dataset: dict[str, DatasetTokenRow] = {}
    suffix = f"_inventory_{line_version}.json"

    for inventory_path in iter_inventory_files(inventory_dir, line_version):
        dataset_id = inventory_path.name.removesuffix(suffix)
        data = load_json(inventory_path)
        usage = data.get("planner_usage_summary", {}) or {}
        row = rows_by_dataset.setdefault(dataset_id, DatasetTokenRow(dataset_id=dataset_id))
        row.planner_calls += int(usage.get("calls", 0) or 0)
        row.planner_input_tokens += int(usage.get("input_tokens", 0) or 0)
        row.planner_cached_input_tokens += int(usage.get("cached_input_tokens", 0) or 0)
        row.planner_output_tokens += int(usage.get("output_tokens", 0) or 0)
        row.planner_total_tokens += int(usage.get("total_tokens", 0) or 0)

    manifest_paths: list[Path] = []
    for run_root in run_roots:
        if run_root.exists():
            manifest_paths.extend(run_root.rglob("run_manifest.json"))

    for manifest_path in manifest_paths:
        data = load_json(manifest_path)
        dataset_id = data.get("dataset_id")
        if not dataset_id:
            continue
        row = rows_by_dataset.setdefault(dataset_id, DatasetTokenRow(dataset_id=dataset_id))
        status = data.get("status")
        if status == "failed":
            row.generation_failed_queries += 1
            continue
        if status != "completed":
            continue
        usage = data.get("usage_summary", {}) or {}
        row.generation_completed_queries += 1
        row.generation_ai_cli_calls += int(usage.get("ai_cli_calls", 0) or 0)
        row.generation_input_tokens += int(usage.get("input_tokens", 0) or 0)
        row.generation_cached_input_tokens += int(usage.get("cached_input_tokens", 0) or 0)
        row.generation_output_tokens += int(usage.get("output_tokens", 0) or 0)
        row.generation_total_tokens += int(usage.get("total_tokens", 0) or 0)

    rows = sorted(rows_by_dataset.values(), key=lambda row: natural_dataset_key(row.dataset_id))
    for row in rows:
        row.finalize()
    return rows


def totals_row(rows: list[DatasetTokenRow]) -> DatasetTokenRow:
    total = DatasetTokenRow(dataset_id="TOTAL")
    for row in rows:
        total.planner_calls += row.planner_calls
        total.planner_input_tokens += row.planner_input_tokens
        total.planner_cached_input_tokens += row.planner_cached_input_tokens
        total.planner_output_tokens += row.planner_output_tokens
        total.planner_total_tokens += row.planner_total_tokens
        total.generation_completed_queries += row.generation_completed_queries
        total.generation_failed_queries += row.generation_failed_queries
        total.generation_ai_cli_calls += row.generation_ai_cli_calls
        total.generation_input_tokens += row.generation_input_tokens
        total.generation_cached_input_tokens += row.generation_cached_input_tokens
        total.generation_output_tokens += row.generation_output_tokens
        total.generation_total_tokens += row.generation_total_tokens
    total.finalize()
    return total


def write_csv(rows: list[DatasetTokenRow], output_path: Path) -> None:
    total = totals_row(rows)
    fieldnames = [
        "dataset_id",
        "planner_calls",
        "planner_input_tokens",
        "planner_cached_input_tokens",
        "planner_output_tokens",
        "planner_total_tokens",
        "generation_completed_queries",
        "generation_failed_queries",
        "generation_ai_cli_calls",
        "generation_input_tokens",
        "generation_cached_input_tokens",
        "generation_output_tokens",
        "generation_total_tokens",
        "combined_total_tokens",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows + [total]:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def write_markdown(rows: list[DatasetTokenRow], output_path: Path, run_ids: list[str]) -> None:
    total = totals_row(rows)
    lines = [
        "# Token Usage Snapshot",
        "",
        f"- Runs covered: `{', '.join(run_ids)}`",
        f"- Datasets covered: `{len(rows)}`",
        f"- Completed generation queries counted: `{format_int(total.generation_completed_queries)}`",
        f"- Failed generation queries counted: `{format_int(total.generation_failed_queries)}`",
        f"- Planner total tokens: `{format_int(total.planner_total_tokens)}`",
        f"- Generation total tokens: `{format_int(total.generation_total_tokens)}`",
        f"- Combined total tokens: `{format_int(total.combined_total_tokens)}`",
        "",
        "| dataset | planner total | generation total | combined total | completed queries | failed queries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.dataset_id} | {format_int(row.planner_total_tokens)} | {format_int(row.generation_total_tokens)} | "
            f"{format_int(row.combined_total_tokens)} | {format_int(row.generation_completed_queries)} | {format_int(row.generation_failed_queries)} |"
        )
    lines.append(
        f"| TOTAL | {format_int(total.planner_total_tokens)} | {format_int(total.generation_total_tokens)} | "
        f"{format_int(total.combined_total_tokens)} | {format_int(total.generation_completed_queries)} | {format_int(total.generation_failed_queries)} |"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(rows: list[DatasetTokenRow], output_path: Path, caption: str, label: str) -> None:
    total = totals_row(rows)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{{tex_escape(label)}}}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Dataset & Planner Total & Generation Total & Combined Total & Completed & Failed \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row.dataset_id)} & {format_int(row.planner_total_tokens)} & "
            f"{format_int(row.generation_total_tokens)} & {format_int(row.combined_total_tokens)} & "
            f"{format_int(row.generation_completed_queries)} & {format_int(row.generation_failed_queries)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            f"TOTAL & {format_int(total.planner_total_tokens)} & {format_int(total.generation_total_tokens)} & "
            f"{format_int(total.combined_total_tokens)} & {format_int(total.generation_completed_queries)} & {format_int(total.generation_failed_queries)} \\\\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate planner + generation token usage for subitem workload runs.")
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--line-version", type=str, default="v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_roots = [args.run_root / run_id for run_id in args.run_ids]
    line_version = args.line_version.strip().lower()
    rows = build_rows(args.inventory_dir, run_roots, line_version=line_version)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "dataset_token_usage_snapshot.csv")
    write_markdown(rows, args.output_dir / "dataset_token_usage_snapshot.md", args.run_ids)
    write_tex(
        rows,
        args.output_dir / "dataset_token_usage_snapshot.tex",
        caption="Planner and generation token usage snapshot for subitem workload v2.",
        label="tab:subitem_workload_v2_token_snapshot",
    )


if __name__ == "__main__":
    main()
