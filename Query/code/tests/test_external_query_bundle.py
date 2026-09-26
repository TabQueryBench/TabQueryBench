from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ExternalBundleSmokeTest(unittest.TestCase):
    def test_cli_generates_traceable_executable_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            work = Path(temp_dir)
            csv_path = work / "orders.csv"
            csv_path.write_text(
                "region,channel,customer_id,revenue,returned,coupon_code\n"
                "east,web,c1,10,no,\nwest,store,c2,20,yes,SAVE\n"
                "east,web,c3,15,no,SAVE\nwest,web,c1,12,no,\n"
                "north,store,c4,24,yes,HELLO\nsouth,web,c5,18,no,\n",
                encoding="utf-8",
            )
            metadata_path = work / "metadata.json"
            metadata_path.write_text(json.dumps({"dataset_id": "orders_smoke", "roles": {"group_col": "region", "group_col_2": "channel", "measure_col": "revenue", "entity_col": "customer_id", "condition_col": "returned", "predicate_col": "channel", "missing_col": "coupon_code"}}), encoding="utf-8")
            output_dir = work / "bundle"
            completed = subprocess.run([sys.executable, str(ROOT / "code/scripts/generate_query_bundle.py"), "--csv", str(csv_path), "--metadata", str(metadata_path), "--output-dir", str(output_dir), "--max-queries", "8"], cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("accepted_queries=", completed.stdout)
            bundle = json.loads((output_dir / "query_bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(bundle["schema_version"], "tabquerybench.query_bundle.v1")
            self.assertEqual(bundle["manifest"]["usage"]["llm_calls"], 0)
            self.assertGreater(len(bundle["queries"]), 0)
            self.assertTrue(all(query["provenance"]["template_id"] for query in bundle["queries"]))
            self.assertTrue((output_dir / "manifest.json").is_file())
            self.assertTrue((output_dir / "selected_queries.sql").is_file())


if __name__ == "__main__":
    unittest.main()
