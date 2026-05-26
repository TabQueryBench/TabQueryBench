"""Reusable CSV logger for API/token usage records."""

from __future__ import annotations

import csv
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

CSV_COLUMNS = [
    "timestamp",
    "run_id",
    "dataset_id",
    "phase",
    "module",
    "question",
    "model",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
]


@dataclass
class UsageLogRecord:
    timestamp: str
    run_id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    question: str = ""
    dataset_id: str = ""
    phase: str = ""
    module: str = ""

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "phase": self.phase,
            "module": self.module,
            "question": self.question,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": f"{self.cost_usd:.10f}",
        }


class UsageCSVLogger:
    """Append usage records to CSV with automatic header creation."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

    def _ensure_compatible_header(self) -> None:
        if not self.csv_path.exists():
            return

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []

        if header == CSV_COLUMNS:
            return

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            dict_reader = csv.DictReader(f)
            rows = list(dict_reader)

        backup_path = self.csv_path.with_suffix(
            f"{self.csv_path.suffix}.bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        shutil.copy2(self.csv_path, backup_path)

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                normalized = {column: row.get(column, "") for column in CSV_COLUMNS}
                writer.writerow(normalized)

    def append(self, record: UsageLogRecord) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_compatible_header()
        file_exists = self.csv_path.exists()

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record.to_row())
