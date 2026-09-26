#!/usr/bin/env python3
"""Build Phase A sql_high dataset scope artifacts from the master status CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("logs/dataset_scan/dataset_master_status_with_scores_20260403.csv")
DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Phase A scope artifacts for datasets whose sql_level is high "
            "(case-insensitive) from the master status CSV."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the master status CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root for the Phase A workspace. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    return parser.parse_args()


def normalize_token(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_master_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise ValueError(f"Input CSV has no header row: {path}")
    return fieldnames, rows


def select_high_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected_rows: list[dict[str, str]] = []
    for row in rows:
        sql_level = (row.get("sql_level") or "").strip()
        if sql_level.casefold() != "high":
            continue

        enriched_row = dict(row)
        enriched_row["class_type_norm"] = normalize_token(row.get("class_type") or "")
        enriched_row["source_type_norm"] = normalize_token(row.get("source_type") or "")
        enriched_row["dataset_folder_name"] = (row.get("own_id") or "").strip()
        selected_rows.append(enriched_row)
    return selected_rows


def ordered_counter(items: list[str]) -> dict[str, int]:
    counter = Counter(items)
    return {key: counter[key] for key in sorted(counter, key=lambda value: (-counter[value], value))}


def build_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_scope_summary(
    *,
    input_path: Path,
    output_root: Path,
    scope_rows: list[dict[str, str]],
    class_type_counts: dict[str, int],
    source_type_counts: dict[str, int],
    class_type_norm_counts: dict[str, int],
    source_type_norm_counts: dict[str, int],
) -> str:
    ordered_dataset_lines = [
        f"{index}. `{row['own_id']}` - {row['dataset_name']}"
        for index, row in enumerate(scope_rows, start=1)
    ]

    class_type_table = build_markdown_table(
        ["class_type", "count"],
        [[key, value] for key, value in class_type_counts.items()],
    )
    source_type_table = build_markdown_table(
        ["source_type", "count"],
        [[key, value] for key, value in source_type_counts.items()],
    )
    normalized_counts_table = build_markdown_table(
        ["normalized_field", "value", "count"],
        [
            ["class_type_norm", key, value]
            for key, value in class_type_norm_counts.items()
        ]
        + [
            ["source_type_norm", key, value]
            for key, value in source_type_norm_counts.items()
        ],
    )

    lines = [
        "# SQL-High Dataset Scope Summary",
        "",
        f"- Input master CSV: `{input_path.resolve()}`",
        f"- Output root: `{output_root.resolve()}`",
        "- Filter rule: keep rows where `sql_level == \"high\"` after trim + case-insensitive comparison.",
        "- Row order: preserved from the master CSV.",
        "- Original master CSV: not modified.",
        "- New web sources: not collected in Phase A.",
        "",
        "## Totals",
        "",
        f"- Total sql_high dataset count: {len(scope_rows)}",
        "",
        "## Counts by class_type",
        "",
        class_type_table,
        "",
        "## Counts by source_type",
        "",
        source_type_table,
        "",
        "## Normalized Count Cross-Check",
        "",
        (
            "The CSV/JSON outputs include `class_type_norm` and `source_type_norm` "
            "to give a case/format-stable view alongside the original source columns."
        ),
        "",
        normalized_counts_table,
        "",
        "## Full Ordered Dataset List",
        "",
        *ordered_dataset_lines,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    output_root = args.output_root.resolve()
    scope_dir = output_root / "scope"
    script_path = Path(__file__).resolve()

    source_fieldnames, source_rows = read_master_csv(input_path)
    scope_rows = select_high_rows(source_rows)

    output_fieldnames = [
        *source_fieldnames,
        "class_type_norm",
        "source_type_norm",
        "dataset_folder_name",
    ]

    class_type_counts = ordered_counter([(row.get("class_type") or "") for row in scope_rows])
    source_type_counts = ordered_counter([(row.get("source_type") or "") for row in scope_rows])
    class_type_norm_counts = ordered_counter([(row.get("class_type_norm") or "") for row in scope_rows])
    source_type_norm_counts = ordered_counter([(row.get("source_type_norm") or "") for row in scope_rows])

    high_datasets_csv_path = scope_dir / "high_datasets.csv"
    high_datasets_json_path = scope_dir / "high_datasets.json"
    scope_summary_path = scope_dir / "scope_summary.md"
    manifest_path = scope_dir / "run_manifest_phase_a.json"

    write_csv(high_datasets_csv_path, output_fieldnames, scope_rows)
    write_json(high_datasets_json_path, scope_rows)

    scope_summary = build_scope_summary(
        input_path=input_path,
        output_root=output_root,
        scope_rows=scope_rows,
        class_type_counts=class_type_counts,
        source_type_counts=source_type_counts,
        class_type_norm_counts=class_type_norm_counts,
        source_type_norm_counts=source_type_norm_counts,
    )
    scope_summary_path.parent.mkdir(parents=True, exist_ok=True)
    scope_summary_path.write_text(scope_summary, encoding="utf-8")

    manifest = {
        "phase": "A",
        "phase_name": "sql_high_dataset_scope_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(script_path),
        "rerun_command": (
            f"python3 {script_path} --input {input_path} --output-root {output_root}"
        ),
        "input": {
            "master_csv_path": str(input_path),
            "master_csv_sha256": sha256_file(input_path),
            "master_csv_size_bytes": input_path.stat().st_size,
            "source_row_count": len(source_rows),
            "source_columns": source_fieldnames,
        },
        "selection": {
            "sql_level_match_value": "high",
            "case_insensitive": True,
            "preserve_source_order": True,
            "matched_row_count": len(scope_rows),
        },
        "computed_columns": {
            "class_type_norm": "lowercase + trim + non-alphanumeric collapsed to underscores from class_type",
            "source_type_norm": "lowercase + trim + non-alphanumeric collapsed to underscores from source_type",
            "dataset_folder_name": "own_id with leading/trailing whitespace removed",
        },
        "counts": {
            "by_class_type": class_type_counts,
            "by_source_type": source_type_counts,
            "by_class_type_norm": class_type_norm_counts,
            "by_source_type_norm": source_type_norm_counts,
        },
        "ordered_datasets": [
            {
                "order_index": index,
                "own_id": row.get("own_id"),
                "dataset_name": row.get("dataset_name"),
                "dataset_folder_name": row.get("dataset_folder_name"),
            }
            for index, row in enumerate(scope_rows, start=1)
        ],
        "outputs": {
            "output_root": str(output_root),
            "scope_dir": str(scope_dir),
            "high_datasets_csv": str(high_datasets_csv_path),
            "high_datasets_json": str(high_datasets_json_path),
            "scope_summary_md": str(scope_summary_path),
            "run_manifest_phase_a_json": str(manifest_path),
        },
        "notes": [
            "No new web sources were collected in Phase A.",
            "The original master CSV was not modified.",
        ],
    }
    write_json(manifest_path, manifest)

    print(str(high_datasets_csv_path))
    print(str(high_datasets_json_path))
    print(str(scope_summary_path))
    print(str(manifest_path))
    print()
    print("PHASE A DONE")
    print(f"sql_high_count: {len(scope_rows)}")
    print(f"class_type_counts: {json.dumps(class_type_counts, ensure_ascii=False)}")
    print(f"source_type_counts: {json.dumps(source_type_counts, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
