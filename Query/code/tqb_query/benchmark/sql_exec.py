"""SQLite execution helpers for benchmark pipeline."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqb_query.benchmark.models import ExecutionResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execution_metadata(result: ExecutionResult, *, prefix: str = "") -> dict[str, Any]:
    """Return JSON-friendly execution observability fields for an output row."""
    return {
        f"{prefix}exec_ok": result.ok,
        f"{prefix}exec_engine": result.engine,
        f"{prefix}exec_row_count": result.row_count,
        f"{prefix}exec_timed_out": result.timed_out,
        f"{prefix}exec_error": result.error,
        f"{prefix}exec_started_at": result.started_at,
        f"{prefix}exec_ended_at": result.ended_at,
        f"{prefix}exec_elapsed_ms": result.elapsed_ms,
    }


def summarize_execution_results(results: list[ExecutionResult]) -> dict[str, Any]:
    """Summarize execution observability fields for a manifest."""
    elapsed = [result.elapsed_ms for result in results if result.elapsed_ms is not None]
    return {
        "execution_count": len(results),
        "ok_count": sum(1 for result in results if result.ok),
        "error_count": sum(1 for result in results if not result.ok and not result.timed_out),
        "timeout_count": sum(1 for result in results if result.timed_out),
        "row_count_total": sum(result.row_count for result in results),
        "elapsed_ms_total": round(sum(elapsed), 3) if elapsed else 0.0,
        "elapsed_ms_mean": round(sum(elapsed) / len(elapsed), 3) if elapsed else None,
        "engines": sorted({result.engine for result in results}),
    }


def execute_sql(
    db_path: Path,
    sql: str,
    row_limit: int | None = 500,
    timeout_seconds: float | None = None,
) -> ExecutionResult:
    """Execute SQLite SQL and capture timing metadata without affecting scores.

    ``row_count`` is the number of rows fetched (and therefore respects
    ``row_limit``); callers that need the total result cardinality should issue
    an explicit count query.
    """
    started_at = _utc_now()
    started = time.perf_counter()

    def result(*, ok: bool, sql_value: str, columns: list[str], rows: list[list[Any]], error: str | None = None, timed_out: bool = False) -> ExecutionResult:
        return ExecutionResult(
            ok=ok,
            sql=sql_value,
            columns=columns,
            rows=rows,
            error=error,
            engine="sqlite",
            row_count=len(rows),
            timed_out=timed_out,
            started_at=started_at,
            ended_at=_utc_now(),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    sql_text = sql.strip()
    if not sql_text:
        return result(ok=False, sql_value=sql, columns=[], rows=[], error="Empty SQL")

    conn = sqlite3.connect(db_path)
    deadline = (started + float(timeout_seconds)) if timeout_seconds is not None and timeout_seconds > 0 else None
    timed_out = False

    def _enforce_timeout() -> int:
        nonlocal timed_out
        if deadline is not None and time.perf_counter() >= deadline:
            timed_out = True
            return 1
        return 0

    try:
        if deadline is not None:
            conn.set_progress_handler(_enforce_timeout, 1_000)
        cursor = conn.cursor()
        cursor.execute(sql_text)
        if row_limit is None or int(row_limit) <= 0:
            rows = cursor.fetchall()
        else:
            rows = cursor.fetchmany(int(row_limit))
        columns = [col[0] for col in (cursor.description or [])]
        normalized_rows = [list(row) for row in rows]
        return result(ok=True, sql_value=sql_text, columns=columns, rows=normalized_rows)
    except Exception as exc:  # noqa: BLE001
        return result(
            ok=False,
            sql_value=sql_text,
            columns=[],
            rows=[],
            error="SQL execution timed out" if timed_out else str(exc),
            timed_out=timed_out,
        )
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
