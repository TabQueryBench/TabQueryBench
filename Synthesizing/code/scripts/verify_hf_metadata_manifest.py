#!/usr/bin/env python3
"""Verify Hugging Face metadata files against a local sha256 manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("files"), list):
        raise ValueError(f"Manifest has no files list: {path}")
    return payload


def remote_url(repo_id: str, revision: str, path: str) -> str:
    quoted_path = urllib.parse.quote(path, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{quoted_revision}/{quoted_path}"


def fetch_remote_file(url: str, token: str | None, retries: int) -> bytes:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - report exact remote failures.
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def verify_one(repo_id: str, revision: str, token: str | None, retries: int, item: dict[str, Any]) -> dict[str, Any]:
    path = str(item["path"])
    url = remote_url(repo_id, revision, path)
    try:
        payload = fetch_remote_file(url, token=token, retries=retries)
    except urllib.error.HTTPError as exc:
        return {"path": path, "status": "missing_or_http_error", "error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "status": "fetch_error", "error": f"{type(exc).__name__}: {exc}"}

    remote_sha = sha256_bytes(payload)
    expected_sha = str(item["sha256"])
    expected_size = int(item["size"])
    if remote_sha != expected_sha:
        return {
            "path": path,
            "status": "sha256_mismatch",
            "expected_sha256": expected_sha,
            "remote_sha256": remote_sha,
            "expected_size": expected_size,
            "remote_size": len(payload),
        }
    if len(payload) != expected_size:
        return {
            "path": path,
            "status": "size_mismatch",
            "expected_size": expected_size,
            "remote_size": len(payload),
        }
    return {"path": path, "status": "ok", "size": len(payload), "sha256": remote_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-id", default=None, help="Override repo_id from manifest.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    repo_id = args.repo_id or str(manifest.get("repo_id") or "TabQueryBench2026/TabQueryBench")
    token = os.environ.get(args.token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    files = list(manifest["files"])

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [
            executor.submit(verify_one, repo_id, args.revision, token, args.retries, item)
            for item in files
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda row: row["path"])
    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {
        "repo_id": repo_id,
        "revision": args.revision,
        "expected_files": len(files),
        "status_counts": counts,
        "all_ok": counts == {"ok": len(files)},
        "bad_sample": [row for row in results if row["status"] != "ok"][:20],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
