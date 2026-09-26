from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tqb_query.benchmark.sql_exec import execute_sql, execution_metadata, summarize_execution_results


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE items (id INTEGER)")
        conn.executemany("INSERT INTO items VALUES (?)", [(1,), (2,)])
        conn.commit()
    finally:
        conn.close()


class ExecuteSqlTimingTest(unittest.TestCase):
    def test_captures_success_timing_and_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "items.sqlite"
            _make_db(db_path)
            result = execute_sql(db_path, "SELECT id FROM items ORDER BY id")

        self.assertTrue(result.ok)
        self.assertEqual(result.engine, "sqlite")
        self.assertEqual(result.row_count, 2)
        self.assertFalse(result.timed_out)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.started_at)
        self.assertIsNotNone(result.ended_at)
        self.assertIsNotNone(result.elapsed_ms)
        self.assertGreaterEqual(result.elapsed_ms or -1, 0)
        self.assertEqual(execution_metadata(result, prefix="real_")["real_exec_row_count"], 2)

    def test_captures_error_and_timeout_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "items.sqlite"
            _make_db(db_path)
            error_result = execute_sql(db_path, "SELECT missing_column FROM items")
            timeout_result = execute_sql(
                db_path,
                "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x + 1 FROM n WHERE x < 100000000) SELECT sum(x) FROM n",
                timeout_seconds=0.000001,
            )

        self.assertFalse(error_result.ok)
        self.assertFalse(error_result.timed_out)
        self.assertTrue(error_result.error)
        self.assertFalse(timeout_result.ok)
        self.assertTrue(timeout_result.timed_out)
        self.assertEqual(timeout_result.error, "SQL execution timed out")
        summary = summarize_execution_results([error_result, timeout_result])
        self.assertEqual(summary["execution_count"], 2)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["timeout_count"], 1)
