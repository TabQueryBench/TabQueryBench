"""CSV preprocessing and profiling without benchmark dataset-layout assumptions."""

from __future__ import annotations

import csv
import math
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SqliteProfile:
    db_path: Path
    table_name: str
    row_count: int


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    declared_type: str
    semantic_type: str
    field_role: str
    field_tags: tuple[str, ...]
    is_numeric: bool
    is_categorical: bool
    distinct_count: int
    top_values: tuple[tuple[Any, int], ...]
    min_value: float | None
    max_value: float | None
    q33: float | None
    q50: float | None
    q66: float | None
    q75: float | None
    missing_count: int
    sample_values: tuple[Any, ...]


@dataclass(frozen=True)
class ExternalDatasetProfile:
    dataset_id: str
    sqlite_result: SqliteProfile
    field_stats: dict[str, ColumnProfile]
    row_count: int
    target_column: str | None
    groupable_cols: tuple[str, ...]
    numeric_cols: tuple[str, ...]
    low_card_cols: tuple[str, ...]
    high_card_cols: tuple[str, ...]
    continuous_numeric_cols: tuple[str, ...]
    temporal_cols: tuple[str, ...]
    missing_cols: tuple[str, ...]
    filterable_cols: tuple[str, ...]
    condition_cols: tuple[str, ...]
    warnings: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
            "target_column": self.target_column,
            "groupable_cols": list(self.groupable_cols),
            "numeric_cols": list(self.numeric_cols),
            "high_card_cols": list(self.high_card_cols),
            "temporal_cols": list(self.temporal_cols),
            "missing_cols": list(self.missing_cols),
            "filterable_cols": list(self.filterable_cols),
            "condition_cols": list(self.condition_cols),
            "warnings": list(self.warnings),
            "columns": [
                {
                    **asdict(column),
                    "top_values": [list(value) for value in column.top_values],
                    "sample_values": list(column.sample_values),
                }
                for column in self.field_stats.values()
            ],
            "sqlite": {
                "table_name": self.sqlite_result.table_name,
                "row_count": self.sqlite_result.row_count,
            },
        }


def sanitize_identifier(value: str, fallback: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", str(value).strip())
    if result and result[0].isdigit():
        result = "t_" + result
    return result or fallback


def _number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _looks_temporal(name: str, values: list[str]) -> bool:
    if not values:
        return False
    name_hint = any(token in name.lower() for token in ("date", "time", "timestamp", "created", "updated"))
    parsed = 0
    for value in values[:100]:
        candidate = value.strip().replace("Z", "+00:00")
        try:
            datetime.fromisoformat(candidate)
            parsed += 1
        except ValueError:
            continue
    ratio = parsed / min(len(values), 100)
    return ratio >= (0.6 if name_hint else 0.9)


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def preprocess_csv(
    csv_path: Path,
    *,
    output_dir: Path,
    dataset_id: str,
    target_column: str | None = None,
    excluded_columns: tuple[str, ...] = (),
    max_rows: int = 1_000_000,
    max_columns: int = 256,
) -> ExternalDatasetProfile:
    """Validate, profile, and materialize a user CSV in a job-local SQLite DB."""
    values: dict[str, list[str]] = {}
    counts: dict[str, Counter[str]] = {}
    missing: Counter[str] = Counter()
    numeric_seen: Counter[str] = Counter()
    row_count = 0
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header row")
        raw_names = [str(name) for name in reader.fieldnames]
        names = [name.strip() for name in raw_names]
        if raw_names != names:
            raise ValueError("CSV headers must not contain leading or trailing whitespace")
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("CSV headers must be non-empty and unique")
        if len(names) > max_columns:
            raise ValueError(f"CSV has {len(names)} columns; maximum is {max_columns}")
        unknown_excluded = sorted(set(excluded_columns) - set(names))
        if unknown_excluded:
            raise ValueError("excluded columns not found: " + ", ".join(unknown_excluded))
        active_names = [name for name in names if name not in set(excluded_columns)]
        if not active_names:
            raise ValueError("all CSV columns were excluded")
        values = {name: [] for name in active_names}
        counts = {name: Counter() for name in active_names}
        for row_number, row in enumerate(reader, start=1):
            if row_number > max_rows:
                raise ValueError(f"CSV exceeds maximum row count of {max_rows}")
            if None in row or any(row.get(name) is None for name in names):
                raise ValueError(f"CSV row {row_number} has a different field count than the header")
            for name in active_names:
                value = (row.get(name) or "").strip()
                if value == "":
                    missing[name] += 1
                else:
                    if _number(value) is not None:
                        numeric_seen[name] += 1
                    if len(values[name]) < 20_000:
                        values[name].append(value)
                    if value in counts[name] or len(counts[name]) < 10_000:
                        counts[name][value] += 1
            row_count = row_number
    if row_count == 0:
        raise ValueError("CSV must contain at least one data row")
    if target_column and target_column not in values:
        raise ValueError(f"target column not found or excluded: {target_column}")

    profiles: dict[str, ColumnProfile] = {}
    for name in values:
        observed = values[name]
        numeric_values = [number for value in observed if (number := _number(value)) is not None]
        non_missing_count = row_count - missing[name]
        # Use REAL only when every observed value is numeric. A permissive 95%
        # inference makes the later materialization crash on the remaining text.
        numeric = non_missing_count > 0 and numeric_seen[name] == non_missing_count
        temporal = not numeric and _looks_temporal(name, observed)
        distinct_count = len(counts[name])
        distinct_capped = distinct_count >= 10_000
        name_identifies_id = name.lower() in {"id", "uuid", "key"} or name.lower().endswith(("_id", "_uuid", "_key"))
        identifier = name_identifies_id or (
            not numeric and (distinct_capped or (row_count >= 20 and distinct_count / row_count >= 0.98))
        )
        categorical = not temporal and (not numeric or distinct_count <= 32)
        semantic = "identifier" if identifier else "temporal" if temporal else "numeric_continuous" if numeric and distinct_count > 32 else "categorical"
        profiles[name] = ColumnProfile(
            name=name,
            declared_type="REAL" if numeric else "TEXT",
            semantic_type=semantic,
            field_role="target" if name == target_column else "identifier" if identifier else "feature",
            field_tags=("excluded_from_grouping",) if identifier else (),
            is_numeric=numeric,
            is_categorical=categorical,
            distinct_count=distinct_count,
            top_values=tuple(counts[name].most_common(12)),
            min_value=min(numeric_values) if numeric_values else None,
            max_value=max(numeric_values) if numeric_values else None,
            q33=_quantile(numeric_values, 0.33),
            q50=_quantile(numeric_values, 0.50),
            q66=_quantile(numeric_values, 0.66),
            q75=_quantile(numeric_values, 0.75),
            missing_count=missing[name],
            sample_values=tuple(dict.fromkeys(observed[:100]))[:20],
        )

    table_name = sanitize_identifier(dataset_id, "source_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "input.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        definitions = ", ".join(
            f"{_quote_identifier(name)} {'REAL' if profiles[name].is_numeric else 'TEXT'}" for name in profiles
        )
        conn.execute(f"CREATE TABLE {_quote_identifier(table_name)} ({definitions})")
        names = list(profiles)
        placeholders = ", ".join("?" for _ in names)
        insert = f"INSERT INTO {_quote_identifier(table_name)} VALUES ({placeholders})"
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            batch: list[tuple[Any, ...]] = []
            for row in reader:
                if None in row or any(row.get(name) is None for name in names):
                    raise ValueError("CSV contains a row with a different field count than the header")
                batch.append(tuple(
                    None if (row.get(name) or "").strip() == ""
                    else float(row[name]) if profiles[name].is_numeric
                    else row[name]
                    for name in names
                ))
                if len(batch) >= 1_000:
                    conn.executemany(insert, batch)
                    batch = []
            if batch:
                conn.executemany(insert, batch)
        conn.commit()
    finally:
        conn.close()

    groupable = tuple(
        name for name, column in profiles.items()
        if name != target_column and column.semantic_type != "identifier" and 1 < column.distinct_count <= max(64, int(math.sqrt(row_count) * 2))
    )
    numeric_cols = tuple(name for name, column in profiles.items() if column.is_numeric and column.semantic_type != "identifier")
    low_card = tuple(name for name, column in profiles.items() if 1 < column.distinct_count <= 12)
    high_card = tuple(name for name, column in profiles.items() if column.semantic_type == "identifier" or column.distinct_count >= 20)
    temporal = tuple(name for name, column in profiles.items() if column.semantic_type == "temporal")
    filterable = tuple(name for name, column in profiles.items() if column.distinct_count > 1 and (column.is_numeric or column.distinct_count <= 64))
    missing_cols = tuple(name for name, column in profiles.items() if column.missing_count > 0)
    condition = list(low_card)
    if target_column and target_column not in condition:
        condition.insert(0, target_column)
    warnings: list[str] = []
    if target_column is None:
        warnings.append("target_column_not_supplied; target roles use low-cardinality candidates")
    if not temporal:
        warnings.append("no_temporal_column_detected")
    return ExternalDatasetProfile(
        dataset_id=dataset_id,
        sqlite_result=SqliteProfile(db_path=db_path, table_name=table_name, row_count=row_count),
        field_stats=profiles,
        row_count=row_count,
        target_column=target_column,
        groupable_cols=groupable,
        numeric_cols=numeric_cols,
        low_card_cols=low_card,
        high_card_cols=high_card,
        continuous_numeric_cols=tuple(name for name in numeric_cols if profiles[name].distinct_count > 20),
        temporal_cols=temporal,
        missing_cols=missing_cols,
        filterable_cols=filterable,
        condition_cols=tuple(dict.fromkeys(condition + list(groupable))),
        warnings=tuple(warnings),
    )
