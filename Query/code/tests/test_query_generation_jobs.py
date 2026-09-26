from __future__ import annotations

import io
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from tqb_query.query_generation.jobs import InvalidJobId, JobConflict, JobManager, JobNotFound
from tqb_query.query_generation.pipeline import QueryGenerationOptions, QueryGenerationPipeline
from tqb_query.query_generation.providers import HeuristicBindingProvider


CSV = (
    "region,channel,id,revenue,target,event_date\n"
    "east,web,c1,10,no,2024-01-01\n"
    "west,store,c2,20,yes,2024-02-01\n"
    "east,store,c3,30,no,2024-03-01\n"
    "west,web,c4,40,yes,2024-04-01\n"
)


class JobManagerTest(unittest.TestCase):
    def make_manager(self, root: Path) -> JobManager:
        return JobManager(
            root,
            pipeline_factory=lambda: QueryGenerationPipeline(binding_provider=HeuristicBindingProvider()),
            max_workers=1,
        )

    def wait_terminal(self, manager: JobManager, job_id: str) -> dict:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = manager.get_status(job_id)
            if state["status"] in {"completed", "failed"}:
                return state
            time.sleep(0.02)
        self.fail("job did not finish")

    def test_job_lifecycle_and_artifact_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = self.make_manager(Path(temp))
            try:
                created = manager.create_job(
                    io.BytesIO(CSV.encode()),
                    filename="orders.csv",
                    options=QueryGenerationOptions(dataset_id="orders", target_column="target", grounding_attempts=1),
                )
                job_id = created["job_id"]
                state = self.wait_terminal(manager, job_id)
                self.assertEqual(state["status"], "completed", state)
                self.assertGreater(len(manager.get_result(job_id)["queries"]), 0)
                archive = manager.get_artifact(job_id)
                with zipfile.ZipFile(archive) as handle:
                    self.assertIn("query_bundle.json", handle.namelist())
                manager.delete_job(job_id)
                with self.assertRaises(JobNotFound):
                    manager.get_status(job_id)
            finally:
                manager.shutdown()

    def test_invalid_ids_and_running_result_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = self.make_manager(Path(temp))
            try:
                with self.assertRaises(InvalidJobId):
                    manager.get_status("../../etc")
                created = manager.create_job(io.BytesIO(CSV.encode()), filename="orders.csv")
                try:
                    manager.get_result(created["job_id"])
                except JobConflict:
                    pass
                self.wait_terminal(manager, created["job_id"])
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
