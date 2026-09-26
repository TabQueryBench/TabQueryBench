"""FastAPI application for asynchronous query-generation jobs."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from .jobs import InvalidJobId, JobConflict, JobError, JobManager, JobNotFound, UploadTooLarge
from .pipeline import QueryGenerationOptions


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (InvalidJobId, JobNotFound)):
        return HTTPException(status_code=404, detail="job not found")
    if isinstance(exc, JobConflict):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, UploadTooLarge):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="job storage error")


def _excluded_columns(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("excluded_columns must be a JSON string array or comma-separated string")
    return tuple(dict.fromkeys(item.strip() for item in parsed if item.strip()))


def create_app(manager: JobManager | None = None) -> FastAPI:
    """Create an app; inject ``manager`` for tests or alternate deployment."""
    owns_manager = manager is None
    job_manager = manager or JobManager(
        Path(os.getenv("QUERY_GENERATION_JOB_DIR", "var/query-generation-jobs")),
        max_workers=int(os.getenv("QUERY_GENERATION_MAX_WORKERS", "2")),
        max_upload_bytes=int(os.getenv("QUERY_GENERATION_MAX_UPLOAD_BYTES", str(256 * 1024 * 1024))),
        max_pending_jobs=int(os.getenv("QUERY_GENERATION_MAX_PENDING_JOBS", "8")),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if owns_manager:
            job_manager.shutdown(wait=False)

    app = FastAPI(title="TabQueryBench Query Generation API", version="1.0.0", lifespan=lifespan)
    app.state.job_manager = job_manager

    @app.get("/healthz", tags=["service"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/query-generation-jobs", status_code=status.HTTP_202_ACCEPTED, tags=["jobs"])
    def create_job(
        dataset_file: Annotated[UploadFile, File(description="Single-table CSV dataset")],
        dataset_id: Annotated[str, Form()] = "uploaded_dataset",
        target_column: Annotated[str | None, Form()] = None,
        excluded_columns: Annotated[str | None, Form()] = None,
        bindings_per_template: Annotated[int, Form(ge=1, le=1)] = 1,
        grounding_attempts: Annotated[int, Form(ge=1, le=5)] = 2,
        max_rows: Annotated[int, Form(ge=1, le=1_000_000)] = 1_000_000,
        max_columns: Annotated[int, Form(ge=1, le=256)] = 256,
        query_timeout_seconds: Annotated[float, Form(gt=0, le=60)] = 10.0,
        max_grounding_workers: Annotated[int, Form(ge=1, le=8)] = 4,
    ) -> dict:
        try:
            options = QueryGenerationOptions(
                dataset_id=dataset_id,
                target_column=target_column or None,
                excluded_columns=_excluded_columns(excluded_columns),
                max_rows=max_rows,
                max_columns=max_columns,
                max_file_bytes=job_manager.max_upload_bytes,
                bindings_per_template=bindings_per_template,
                grounding_attempts=grounding_attempts,
                query_timeout_seconds=query_timeout_seconds,
                max_grounding_workers=max_grounding_workers,
            )
            return job_manager.create_job(
                dataset_file.file,
                filename=dataset_file.filename or "",
                options=options,
            )
        except (JobError, ValueError) as exc:
            raise _http_error(exc) from exc
        finally:
            dataset_file.file.close()

    @app.get("/v1/query-generation-jobs/{job_id}", tags=["jobs"])
    def get_job(job_id: str) -> dict:
        try:
            return job_manager.get_status(job_id)
        except JobError as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/query-generation-jobs/{job_id}/result", tags=["jobs"])
    def get_result(job_id: str) -> dict:
        try:
            return job_manager.get_result(job_id)
        except JobError as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/query-generation-jobs/{job_id}/artifacts", tags=["jobs"])
    def download_artifacts(job_id: str) -> FileResponse:
        try:
            archive = job_manager.get_artifact(job_id)
        except JobError as exc:
            raise _http_error(exc) from exc
        return FileResponse(archive, media_type="application/zip", filename=f"{job_id}-artifacts.zip")

    @app.delete("/v1/query-generation-jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["jobs"])
    def delete_job(job_id: str) -> None:
        try:
            job_manager.delete_job(job_id)
        except JobError as exc:
            raise _http_error(exc) from exc

    return app


app = create_app()
