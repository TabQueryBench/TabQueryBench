"""Helpers for v2 upstream contracts and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.bundle import DatasetBundle


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_path_hash(path: Path) -> dict[str, Any]:
    if not path or not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def dataset_fingerprint(bundle: DatasetBundle) -> dict[str, Any]:
    """Fingerprint key dataset inputs for cross-run reproducibility."""
    tracked_paths = [
        bundle.main_csv_path,
        bundle.dataset_profile_path,
        bundle.dataset_contract_path,
        bundle.dataset_description_path,
        bundle.dataset_semantics_path,
        bundle.field_registry_path,
        bundle.family_applicability_path,
        bundle.query_policy_path,
        bundle.validation_policy_path,
        bundle.risk_register_path,
        bundle.uncertainty_register_path,
        bundle.source_info_path,
    ]

    files = [_safe_path_hash(path) for path in tracked_paths]
    digest_payload = "|".join(
        f"{item['path']}:{item['sha256'] or 'missing'}:{item['size_bytes'] or 0}" for item in files
    )
    fingerprint = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return {
        "dataset_id": bundle.dataset_id,
        "fingerprint_sha256": fingerprint,
        "files": files,
    }


def git_revision(project_root: Path) -> dict[str, Any]:
    """Best-effort git revision metadata. Explicitly records unavailability."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
        short = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(status.strip())
        return {
            "available": True,
            "commit": commit,
            "short_commit": short,
            "dirty_worktree": dirty,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "commit": None,
            "short_commit": None,
            "dirty_worktree": None,
            "unavailable_reason": str(exc),
        }


def hash_prompt(text: str) -> str:
    normalized = " ".join((text or "").strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def build_manifest_v2(
    *,
    build_id: str,
    run_id: str,
    dataset_id: str,
    dataset_fingerprint_obj: dict[str, Any],
    git_info: dict[str, Any],
    pipeline_version: str,
    llm_config: dict[str, Any],
    generation_config: dict[str, Any],
    curation_config: dict[str, Any],
    prompt_info: dict[str, Any] | None,
    started_at: str | None = None,
) -> dict[str, Any]:
    return {
        "contract_version": "build_manifest_v2",
        "build_id": build_id,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint_obj,
        "code_revision": git_info,
        "pipeline_version": pipeline_version,
        "llm_config": llm_config,
        "generation_config": generation_config,
        "curation_config": curation_config,
        "prompt_info": prompt_info
        or {
            "available": False,
            "reason": "prompt signatures not available at manifest initialization",
        },
        "timestamps": {
            "started_at": started_at or datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
        },
    }


def finalize_build_manifest_v2(
    manifest: dict[str, Any],
    *,
    prompt_info: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(manifest))
    updated.setdefault("timestamps", {})
    updated["timestamps"]["ended_at"] = datetime.now(timezone.utc).isoformat()
    if prompt_info is not None:
        updated["prompt_info"] = prompt_info
    if summary is not None:
        updated["build_summary"] = summary
    return updated
