"""CSV to SQLite materialization for dataset-mode SQL QA runs."""

from __future__ import annotations

import csv
import fcntl
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tqb_query.data.bundle import DatasetBundle


def _quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _sanitize_table_name(name: str) -> str:
    table = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if table and table[0].isdigit():
        table = f"t_{table}"
    return table or "dataset_table"


def _csv_fingerprint(csv_path: Path) -> dict[str, Any]:
    stat = csv_path.stat()
    return {
        "csv_path": str(csv_path),
        "csv_resolved_path": str(csv_path.resolve()),
        "csv_size": stat.st_size,
        "csv_mtime_ns": stat.st_mtime_ns,
    }


@dataclass
class SqliteMaterializationResult:
    db_path: Path
    table_name: str
    row_count: int
    cache_hit: bool
    manifest_path: Path

    @property
    def sqlite_uri(self) -> str:
        return f"sqlite:///{self.db_path.resolve()}"


def _load_cache_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _write_cache_manifest(manifest_path: Path, data: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@contextmanager
def _materialization_lock(cache_dir: Path, table_name: str) -> Iterator[None]:
    """Serialize materialization of one dataset across processes.

    Without it, parallel runs of the same dataset each DROP and rebuild the same file and
    the losers fail with "database is locked" or read a half-built table.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / f".{table_name}.materialize.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _materialize_csv(csv_path: Path, db_path: Path, table_name: str) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # A generous busy timeout: a reader that started before the lock was taken may still be draining.
    conn = sqlite3.connect(db_path, timeout=120.0)
    try:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {_quote_identifier(table_name)}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            if not headers:
                raise ValueError(f"CSV has empty header row: {csv_path}")

            quoted_columns = ", ".join(f"{_quote_identifier(col)} TEXT" for col in headers)
            cursor.execute(f"CREATE TABLE {_quote_identifier(table_name)} ({quoted_columns})")

            placeholders = ", ".join("?" for _ in headers)
            insert_sql = f"INSERT INTO {_quote_identifier(table_name)} VALUES ({placeholders})"

            batch: list[list[str]] = []
            row_count = 0
            for row in reader:
                if len(row) < len(headers):
                    row = row + [""] * (len(headers) - len(row))
                elif len(row) > len(headers):
                    row = row[: len(headers)]
                batch.append(row)
                row_count += 1

                if len(batch) >= 1000:
                    cursor.executemany(insert_sql, batch)
                    batch = []

            if batch:
                cursor.executemany(insert_sql, batch)

            conn.commit()
            return row_count
    finally:
        conn.close()


def materialize_dataset_to_sqlite(
    bundle: DatasetBundle,
    use_cache: bool = True,
    cache_dir_name: str = "cache",
) -> SqliteMaterializationResult:
    table_name = _sanitize_table_name(bundle.dataset_id)
    cache_dir = bundle.dataset_dir / cache_dir_name
    db_path = cache_dir / f"{table_name}.sqlite"
    manifest_path = cache_dir / "sqlite_cache_manifest.json"

    fingerprint = _csv_fingerprint(bundle.main_csv_path)
    expected_cache_key = {
        "dataset_id": bundle.dataset_id,
        "table_name": table_name,
        **fingerprint,
    }

    def cached() -> SqliteMaterializationResult | None:
        if not (use_cache and db_path.exists()):
            return None
        manifest = _load_cache_manifest(manifest_path)
        if not (manifest and all(manifest.get(k) == v for k, v in expected_cache_key.items())):
            return None
        return SqliteMaterializationResult(
            db_path=db_path,
            table_name=table_name,
            row_count=int(manifest.get("row_count", 0)),
            cache_hit=True,
            manifest_path=manifest_path,
        )

    hit = cached()
    if hit is not None:
        return hit

    with _materialization_lock(cache_dir, table_name):
        # Another process may have rebuilt the cache while this one waited for the lock.
        hit = cached()
        if hit is not None:
            return hit

        row_count = _materialize_csv(bundle.main_csv_path, db_path, table_name)
        manifest = {
            **expected_cache_key,
            "db_path": str(db_path),
            "row_count": row_count,
            "materialized_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_cache_manifest(manifest_path, manifest)

    return SqliteMaterializationResult(
        db_path=db_path,
        table_name=table_name,
        row_count=row_count,
        cache_hit=False,
        manifest_path=manifest_path,
    )
