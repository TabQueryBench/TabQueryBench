from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ImportError:  # local core-only environments may omit optional HTTP deps
    TestClient = None  # type: ignore[assignment]

from tqb_query.query_generation.jobs import JobManager
from tqb_query.query_generation.pipeline import QueryGenerationPipeline
from tqb_query.query_generation.providers import HeuristicBindingProvider


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are not installed")
class QueryGenerationApiTest(unittest.TestCase):
    def test_http_job_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            os.environ["QUERY_GENERATION_JOB_DIR"] = str(Path(temp) / "unused-global")
            from tqb_query.query_generation.api import create_app

            manager = JobManager(
                Path(temp) / "jobs",
                pipeline_factory=lambda: QueryGenerationPipeline(binding_provider=HeuristicBindingProvider()),
                max_workers=1,
            )
            try:
                with TestClient(create_app(manager)) as client:  # type: ignore[misc]
                    csv_text = (
                        "group,segment,id,value,target\n"
                        "a,x,i1,1,yes\nb,y,i2,2,no\na,y,i3,3,yes\nb,x,i4,4,no\n"
                    )
                    response = client.post(
                        "/v1/query-generation-jobs",
                        files={"dataset_file": ("sample.csv", csv_text, "text/csv")},
                        data={"dataset_id": "sample", "target_column": "target", "grounding_attempts": "1"},
                    )
                    self.assertEqual(response.status_code, 202, response.text)
                    job_id = response.json()["job_id"]
                    for _ in range(200):
                        status = client.get(f"/v1/query-generation-jobs/{job_id}").json()
                        if status["status"] in {"completed", "failed"}:
                            break
                        time.sleep(0.02)
                    self.assertEqual(status["status"], "completed", status)
                    self.assertTrue(client.get(f"/v1/query-generation-jobs/{job_id}/result").json()["queries"])
                    self.assertEqual(client.get(f"/v1/query-generation-jobs/{job_id}/artifacts").status_code, 200)
                    self.assertEqual(client.get("/v1/query-generation-jobs/not-valid").status_code, 404)
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
