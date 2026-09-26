from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tqb_query.db.csv_sqlite import materialize_dataset_to_sqlite


def _write_csv(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b"])
        for i in range(rows):
            writer.writerow([i, f"v{i % 7}"])


class CsvSqliteLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self._tmp.name) / "d1"
        self.csv_path = self.dataset_dir / "d1-train.csv"
        _write_csv(self.csv_path, 500)
        self.bundle = SimpleNamespace(
            dataset_id="d1", dataset_dir=self.dataset_dir, main_csv_path=self.csv_path
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_second_call_hits_cache(self) -> None:
        first = materialize_dataset_to_sqlite(bundle=self.bundle, use_cache=True)
        self.assertFalse(first.cache_hit)
        self.assertEqual(first.row_count, 500)
        second = materialize_dataset_to_sqlite(bundle=self.bundle, use_cache=True)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.row_count, 500)

    def test_stale_manifest_triggers_rebuild(self) -> None:
        materialize_dataset_to_sqlite(bundle=self.bundle, use_cache=True)
        manifest = self.dataset_dir / "cache" / "sqlite_cache_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["dataset_id"] = "stale"  # any cache-key mismatch forces a rebuild
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        again = materialize_dataset_to_sqlite(bundle=self.bundle, use_cache=True)
        self.assertFalse(again.cache_hit)
        self.assertEqual(again.row_count, 500)

    def test_parallel_materialization_does_not_race(self) -> None:
        """Eight threads rebuilding one cold cache used to raise 'database is locked'."""
        with cf.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(
                    lambda _: materialize_dataset_to_sqlite(bundle=self.bundle, use_cache=True),
                    range(8),
                )
            )
        self.assertEqual({r.row_count for r in results}, {500})
        self.assertEqual(sum(1 for r in results if not r.cache_hit), 1, "only one call should build")
        db = self.dataset_dir / "cache" / "d1.sqlite"
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            self.assertEqual(con.execute('SELECT COUNT(*) FROM "d1"').fetchone()[0], 500)
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
