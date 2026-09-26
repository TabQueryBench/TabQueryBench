"""Shared real-train SQLite cache: one materialization per dataset across concurrent runs."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqb_scoring.eval.analysis import runner  # noqa: E402


class SharedRealCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.csv = self.root / "c1-train.csv"
        self.csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        self.target = self.root / "shared" / "real_sqlite" / "c1.sqlite"
        self.calls: list[Path] = []
        self.original = runner.materialize_csv_to_sqlite

        def fake(csv_path: Path, sqlite_path: Path, table_name: str) -> None:
            self.calls.append(Path(sqlite_path))
            time.sleep(0.2)
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            Path(sqlite_path).write_text(f"{csv_path.name}:{table_name}", encoding="utf-8")

        runner.materialize_csv_to_sqlite = fake

    def tearDown(self) -> None:
        runner.materialize_csv_to_sqlite = self.original
        self.tmp.cleanup()

    def test_concurrent_runs_materialize_once(self):
        errors: list[BaseException] = []

        def worker():
            try:
                runner._materialize_shared_real_sqlite(self.csv, self.target, "c1")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        self.assertEqual(errors, [])
        self.assertEqual(len(self.calls), 1, "the database must be materialized exactly once")
        self.assertTrue(self.target.exists())
        self.assertTrue(self.target.with_name(self.target.name + ".ready").exists())
        self.assertFalse(self.target.with_name(self.target.name + ".lock").exists())
        self.assertEqual(self.target.read_text(), "c1-train.csv:c1")

    def test_ready_database_is_reused(self):
        runner._materialize_shared_real_sqlite(self.csv, self.target, "c1")
        runner._materialize_shared_real_sqlite(self.csv, self.target, "c1")
        self.assertEqual(len(self.calls), 1)

    def test_stale_lock_is_taken_over(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        lock = self.target.with_name(self.target.name + ".lock")
        lock.write_text("dead pid", encoding="utf-8")
        import os

        stale = time.time() - runner.SHARED_REAL_CACHE_STALE_SECONDS - 60
        os.utime(lock, (stale, stale))
        runner._materialize_shared_real_sqlite(self.csv, self.target, "c1")
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.target.exists())

    def test_root_resolution(self):
        import os

        self.assertIsNone(runner._shared_real_cache_root(None))
        os.environ[runner.SHARED_REAL_CACHE_ENV] = str(self.root)
        try:
            self.assertEqual(runner._shared_real_cache_root(None), self.root.resolve())
            self.assertEqual(runner._shared_real_cache_root(self.root / "x"), (self.root / "x").resolve())
        finally:
            del os.environ[runner.SHARED_REAL_CACHE_ENV]


if __name__ == "__main__":
    unittest.main()
