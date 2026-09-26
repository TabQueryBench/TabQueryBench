"""Persistent, thread-backed jobs for the query-generation API."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .pipeline import QueryGenerationOptions, QueryGenerationPipeline
from .providers import provider_from_environment


JOB_ID_RE = re.compile(r"^qgj_[0-9a-f]{32}$")
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


class JobError(RuntimeError):
    """Base class for errors safe to translate at the HTTP boundary."""


class InvalidJobId(JobError):
    pass


class JobNotFound(JobError):
    pass


class JobConflict(JobError):
    pass


class UploadTooLarge(JobError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_pipeline_factory() -> QueryGenerationPipeline:
    # Constructing this in a worker keeps missing/invalid provider configuration
    # isolated to the submitted job.
    return QueryGenerationPipeline(binding_provider=provider_from_environment())


class JobManager:
    """Own job storage and execute pipelines in a bounded worker pool.

    ``pipeline_factory`` is deliberately injectable. Tests and self-hosted users can
    provide a callable returning an object with the same ``run`` interface as
    :class:`QueryGenerationPipeline` without mutating process-global configuration.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        pipeline_factory: Callable[[], QueryGenerationPipeline] | None = None,
        max_workers: int = 2,
        max_upload_bytes: int = 256 * 1024 * 1024,
        max_pending_jobs: int | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be at least 1")
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_factory = pipeline_factory or _default_pipeline_factory
        self.max_upload_bytes = max_upload_bytes
        self.max_pending_jobs = max_pending_jobs or max_workers * 4
        if self.max_pending_jobs < max_workers:
            raise ValueError("max_pending_jobs must be at least max_workers")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="query-generation")
        self._manager_lock = threading.RLock()
        self._job_locks: dict[str, threading.RLock] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._closed = False
        self._mark_interrupted_jobs()

    def _validate_job_id(self, job_id: str) -> str:
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise InvalidJobId("invalid job_id")
        return job_id

    def _job_dir(self, job_id: str, *, must_exist: bool = True) -> Path:
        job_id = self._validate_job_id(job_id)
        path = (self.root_dir / job_id).resolve()
        if path.parent != self.root_dir:
            raise InvalidJobId("invalid job_id")
        if must_exist and (not path.is_dir() or path.is_symlink()):
            raise JobNotFound(f"job not found: {job_id}")
        return path

    def _lock_for(self, job_id: str) -> threading.RLock:
        with self._manager_lock:
            return self._job_locks.setdefault(job_id, threading.RLock())

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JobError(f"job state is unreadable: {path.parent.name}") from exc
        if not isinstance(payload, dict):
            raise JobError(f"job state is invalid: {path.parent.name}")
        return payload

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_status(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock_for(job_id):
            state_path = self._job_dir(job_id) / "status.json"
            state = self._read_json(state_path) if state_path.exists() else {"job_id": job_id}
            state.update(changes)
            state["updated_at"] = _utc_now()
            self._atomic_json(state_path, state)
            return state

    def _mark_interrupted_jobs(self) -> None:
        for candidate in self.root_dir.iterdir():
            if not candidate.is_dir() or candidate.is_symlink() or not JOB_ID_RE.fullmatch(candidate.name):
                continue
            state_path = candidate / "status.json"
            if not state_path.is_file():
                continue
            try:
                state = self._read_json(state_path)
                if state.get("status") in {"queued", "running"}:
                    state.update({
                        "status": "failed",
                        "phase": "failed",
                        "updated_at": _utc_now(),
                        "finished_at": _utc_now(),
                        "error": {
                            "code": "service_restart",
                            "message": "job was interrupted by a service restart",
                        },
                    })
                    self._atomic_json(state_path, state)
            except (JobError, OSError):
                # A corrupt directory must not prevent the service from starting.
                continue

    def create_job(
        self,
        upload: BinaryIO,
        *,
        filename: str,
        options: QueryGenerationOptions | None = None,
    ) -> dict[str, Any]:
        """Persist an upload and schedule it, returning its initial status."""
        options = options or QueryGenerationOptions()
        if not filename or Path(filename).suffix.lower() != ".csv":
            raise ValueError("dataset_file must have a .csv extension")
        with self._manager_lock:
            if self._closed:
                raise JobConflict("job manager is shut down")
            if len(self._futures) >= self.max_pending_jobs:
                raise JobConflict("query-generation job queue is full")
            job_id = f"qgj_{uuid.uuid4().hex}"
            job_dir = self._job_dir(job_id, must_exist=False)
            job_dir.mkdir(mode=0o700)
            (job_dir / "artifacts").mkdir(mode=0o700)
            input_path = job_dir / "dataset.csv"
            try:
                total = 0
                with input_path.open("xb") as destination:
                    while True:
                        chunk = upload.read(1024 * 1024)
                        if not chunk:
                            break
                        if not isinstance(chunk, bytes):
                            raise ValueError("uploaded file must be binary")
                        total += len(chunk)
                        if total > min(self.max_upload_bytes, options.max_file_bytes):
                            raise UploadTooLarge("uploaded CSV exceeds the configured size limit")
                        destination.write(chunk)
                if total == 0:
                    raise ValueError("uploaded CSV is empty")
                now = _utc_now()
                state = {
                    "job_id": job_id,
                    "status": "queued",
                    "phase": "queued",
                    "created_at": now,
                    "updated_at": now,
                    "input": {"filename": Path(filename).name, "size_bytes": total},
                    "progress": {},
                }
                self._atomic_json(job_dir / "status.json", state)
                future = self._executor.submit(self._run_job, job_id, input_path, options)
                self._futures[job_id] = future
                future.add_done_callback(lambda _future, jid=job_id: self._forget_future(jid))
                return state
            except Exception:
                shutil.rmtree(job_dir, ignore_errors=True)
                self._job_locks.pop(job_id, None)
                raise

    # Concise alias for callers that model this operation as submission.
    submit = create_job

    def _forget_future(self, job_id: str) -> None:
        with self._manager_lock:
            self._futures.pop(job_id, None)

    def _run_job(self, job_id: str, csv_path: Path, options: QueryGenerationOptions) -> None:
        try:
            self._write_status(job_id, status="running", phase="starting", started_at=_utc_now(), error=None)

            def progress(event: dict[str, Any]) -> None:
                event = dict(event)
                phase = str(event.pop("phase", "running"))
                self._write_status(job_id, status="running", phase=phase, progress=event)

            pipeline = self.pipeline_factory()
            bundle = pipeline.run(
                csv_path=csv_path,
                output_dir=self._job_dir(job_id) / "artifacts",
                options=options,
                progress=progress,
            )
            manifest = bundle.get("manifest", {}) if isinstance(bundle, dict) else {}
            self._write_status(
                job_id,
                status="completed",
                phase="completed",
                finished_at=_utc_now(),
                progress={
                    "applicable_templates": manifest.get("applicable_template_count"),
                    "accepted_queries": manifest.get("accepted_query_count"),
                    "skipped_templates": manifest.get("skipped_template_count"),
                },
                error=None,
            )
        except Exception as exc:  # one failed model/pipeline must not affect workers
            try:
                self._write_status(
                    job_id,
                    status="failed",
                    phase="failed",
                    finished_at=_utc_now(),
                    error={"code": type(exc).__name__, "message": str(exc)[:1000]},
                )
            except (JobError, OSError):
                pass

    def get_status(self, job_id: str) -> dict[str, Any]:
        with self._lock_for(self._validate_job_id(job_id)):
            state_path = self._job_dir(job_id) / "status.json"
            if not state_path.is_file():
                raise JobNotFound(f"job not found: {job_id}")
            return self._read_json(state_path)

    def get_result(self, job_id: str) -> dict[str, Any]:
        state = self.get_status(job_id)
        if state.get("status") != "completed":
            raise JobConflict(f"job is {state.get('status', 'not completed')}")
        result_path = self._job_dir(job_id) / "artifacts" / "query_bundle.json"
        if not result_path.is_file() or result_path.is_symlink():
            raise JobError("completed job has no query bundle")
        return self._read_json(result_path)

    def get_artifact(self, job_id: str) -> Path:
        state = self.get_status(job_id)
        if state.get("status") != "completed":
            raise JobConflict(f"job is {state.get('status', 'not completed')}")
        job_dir = self._job_dir(job_id)
        artifacts_dir = job_dir / "artifacts"
        archive = job_dir / "query-generation-artifacts.zip"
        temporary = job_dir / f".artifacts.{uuid.uuid4().hex}.zip"
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(artifacts_dir.rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        resolved = path.resolve()
                        if artifacts_dir.resolve() in resolved.parents:
                            bundle.write(resolved, resolved.relative_to(artifacts_dir.resolve()).as_posix())
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
        return archive

    def delete_job(self, job_id: str) -> None:
        state = self.get_status(job_id)
        if state.get("status") not in TERMINAL_STATES:
            raise JobConflict("a running or queued job cannot be deleted")
        with self._manager_lock, self._lock_for(job_id):
            shutil.rmtree(self._job_dir(job_id))
            self._job_locks.pop(job_id, None)

    delete = delete_job

    def shutdown(self, *, wait: bool = True) -> None:
        with self._manager_lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
