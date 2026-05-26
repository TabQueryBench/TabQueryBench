"""SQLite execution helpers for benchmark pipeline."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.benchmark.models import ExecutionResult


def execute_sql(db_path: Path, sql: str, row_limit: int | None = 500) -> ExecutionResult:
    sql_text = sql.strip()
    if not sql_text:
        return ExecutionResult(ok=False, sql=sql, columns=[], rows=[], error="Empty SQL")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql_text)
        if row_limit is None or int(row_limit) <= 0:
            rows = cursor.fetchall()
        else:
            rows = cursor.fetchmany(int(row_limit))
        columns = [col[0] for col in (cursor.description or [])]
        normalized_rows = [list(row) for row in rows]
        return ExecutionResult(ok=True, sql=sql_text, columns=columns, rows=normalized_rows, error=None)
    except Exception as exc:  # noqa: BLE001
        return ExecutionResult(ok=False, sql=sql_text, columns=[], rows=[], error=str(exc))
    finally:
        conn.close()


def format_rows_for_prompt(columns: list[str], rows: list[list[Any]], max_rows: int = 15) -> str:
    if not columns:
        return "<no columns>"
    if not rows:
        return "<empty result>"

    display_rows = rows[:max_rows]
    lines = ["\t".join(columns)]
    for row in display_rows:
        lines.append("\t".join(str(cell) for cell in row))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)
