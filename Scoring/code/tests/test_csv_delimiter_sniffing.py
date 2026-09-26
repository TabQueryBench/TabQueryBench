"""Delimiter sniffing must survive quoted free-text fields full of other separators."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqb_scoring.eval.common import materialize_csv_to_sqlite, sniff_csv_delimiter  # noqa: E402

# c19 stores YouTube tags as "tag1|tag2|tag3", which makes csv.Sniffer report pipe.
PIPE_HEAVY_COMMA_CSV = (
    'video_id,title,tags,views\n'
    'a1,First video,"news|politics|daily|live|update|world|tv",100\n'
    'a2,Second video,"music|pop|rock|live|band|tour|clip",200\n'
    'a3,Third video,"game|play|stream|live|fun|review|clip",300\n'
)


class SniffCsvDelimiterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_quoted_pipes_do_not_hide_the_comma_delimiter(self):
        path = self.write("c19_like.csv", PIPE_HEAVY_COMMA_CSV)
        self.assertEqual(sniff_csv_delimiter(path), ",")

    def test_real_pipe_file_still_sniffs_as_pipe(self):
        path = self.write("pipe.csv", "a|b|c\n1|2|3\n4|5|6\n")
        self.assertEqual(sniff_csv_delimiter(path), "|")

    def test_semicolon_file_with_commas_inside_fields(self):
        path = self.write("semi.csv", 'x;y\n"a,b,c";2\n"d,e,f";3\n')
        self.assertEqual(sniff_csv_delimiter(path), ";")

    def test_single_column_file_falls_back_to_comma(self):
        path = self.write("single.csv", "only\n1\n2\n")
        self.assertEqual(sniff_csv_delimiter(path), ",")

    def test_materialized_table_keeps_every_column(self):
        path = self.write("c19_like.csv", PIPE_HEAVY_COMMA_CSV)
        sqlite_path = self.root / "c19_like.sqlite"
        materialize_csv_to_sqlite(path, sqlite_path, "c19")
        connection = sqlite3.connect(sqlite_path)
        try:
            columns = [row[1] for row in connection.execute('PRAGMA table_info("c19")')]
            total = connection.execute('SELECT SUM("views") FROM "c19"').fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(columns, ["video_id", "title", "tags", "views"])
        self.assertEqual(total, 600)


if __name__ == "__main__":
    unittest.main()
