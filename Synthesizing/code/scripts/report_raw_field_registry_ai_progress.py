#!/usr/bin/env python3
"""Report progress for per-column raw field-registry AI review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "raw_data" / "tabular_datasets"


def count_columns(dataset_dir: Path) -> int:
    train_path = dataset_dir / f"{dataset_dir.name}-train.csv"
    if not train_path.exists():
        return 0
    with train_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return len(next(csv.reader(handle)))


def review_counts(dataset_dir: Path) -> tuple[int, int, int, str]:
    reviews_path = dataset_dir / "metadata" / "field_ai_reviews.jsonl"
    success = 0
    errors = 0
    cached = 0
    last_column = ""
    if not reviews_path.exists():
        return success, errors, cached, last_column
    with reviews_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("ai_provider") != "claude":
                continue
            last_column = str(record.get("column") or last_column)
            if record.get("ai_cached"):
                cached += 1
            if record.get("ai_error"):
                errors += 1
            elif isinstance(record.get("ai_review"), dict):
                success += 1
    return success, errors, cached, last_column


def main() -> int:
    parser = argparse.ArgumentParser(description="Show Claude per-column preprocessing review progress.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--show-pending", type=int, default=12)
    args = parser.parse_args()

    rows = []
    total_cols = total_success = total_errors = total_cached = 0
    for dataset_dir in sorted(path for path in args.data_root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        columns = count_columns(dataset_dir)
        success, errors, cached, last_column = review_counts(dataset_dir)
        seen = success + errors
        if columns and seen >= columns:
            status = "complete" if errors == 0 else "complete_with_errors"
        elif seen > 0:
            status = "partial"
        else:
            status = "pending"
        rows.append((dataset_dir.name, columns, success, errors, cached, last_column, status))
        total_cols += columns
        total_success += success
        total_errors += errors
        total_cached += cached

    percent = (total_success + total_errors) / total_cols * 100 if total_cols else 0.0
    print(f"total: {total_success + total_errors}/{total_cols} reviewed ({percent:.2f}%), success={total_success}, errors={total_errors}, cached={total_cached}")
    print()
    print("active/partial/failed datasets:")
    for dataset_id, columns, success, errors, cached, last_column, status in rows:
        if status != "pending":
            print(f"{dataset_id:>4}  {status:20} {success + errors:4}/{columns:<4} success={success:<4} errors={errors:<3} cached={cached:<3} last={last_column}")
    print()
    print(f"next pending datasets (first {args.show_pending}):")
    shown = 0
    for dataset_id, columns, success, errors, cached, last_column, status in rows:
        if status == "pending":
            print(f"{dataset_id:>4}  pending              0/{columns:<4}")
            shown += 1
            if shown >= args.show_pending:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
