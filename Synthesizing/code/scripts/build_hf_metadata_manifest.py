#!/usr/bin/env python3
"""Build a sha256 manifest for AI-reviewed dataset metadata files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PATTERNS = [
    "Synthesizing/raw_data/tabular_datasets/*/metadata/field_ai_reviews.jsonl",
    "Synthesizing/raw_data/tabular_datasets/*/metadata_core/field_registry.json",
    "Synthesizing/raw_data/tabular_datasets/*/metadata_core/dataset_semantics.yaml",
]

GENERATION_CODE_PATTERNS = [
    "Synthesizing/code/.gitignore",
    "Synthesizing/code/README.md",
    "Synthesizing/code/src/**/*.py",
    "Synthesizing/code/src/**/*.json",
    "Synthesizing/code/synthetic_benchmark/**/*.py",
    "Synthesizing/code/synthetic_benchmark/**/*.yaml",
    "Synthesizing/code/synthetic_benchmark/**/*.yml",
    "Synthesizing/code/synthetic_benchmark/**/*.toml",
    "Synthesizing/code/synthetic_benchmark/**/*.md",
]

PREPROCESSING_CODE_PATTERNS = [
    "Synthesizing/code/scripts/build_raw_field_registry_with_ai_review.py",
    "Synthesizing/code/scripts/finalize_field_registry_for_generation.py",
    "Synthesizing/code/scripts/report_raw_field_registry_ai_progress.py",
    "Synthesizing/code/scripts/run_raw_field_registry_ai_shards.py",
]

RELEASE_CODE_PATTERNS = [
    "Synthesizing/code/scripts/build_hf_metadata_manifest.py",
    "Synthesizing/code/scripts/verify_hf_metadata_manifest.py",
]

SKIP_PARTS = {"__pycache__", ".home", ".ipynb_checkpoints"}


def should_include(path: Path) -> bool:
    if path.name.startswith("._"):
        return False
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    lowered = str(path).lower()
    if any(token in lowered for token in ("checkpoint", "synthetic_data", "/output", "/logs/", "/log/")):
        return False
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(repo_root: Path, repo_id: str, patterns: list[str]) -> dict:
    files = []
    for pattern in patterns:
        for path in sorted(repo_root.glob(pattern)):
            if not path.is_file():
                continue
            if not should_include(path):
                continue
            files.append(
                {
                    "path": str(path.relative_to(repo_root)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--repo-id", default="TabQueryBench2026/TabQueryBench")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "tmp" / "hf_metadata_upload_manifest.json",
    )
    parser.add_argument("--pattern", action="append", default=None, help="Override glob pattern. Can be repeated.")
    parser.add_argument("--include-generation-code", action="store_true", help="Also include synthetic_generation source files.")
    parser.add_argument("--include-preprocessing-code", action="store_true", help="Also include preprocessing helper scripts.")
    parser.add_argument("--include-release-code", action="store_true", help="Also include HF release manifest scripts.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    patterns = args.pattern or DEFAULT_PATTERNS
    if args.include_generation_code and args.pattern is None:
        patterns = patterns + GENERATION_CODE_PATTERNS
    if args.include_preprocessing_code and args.pattern is None:
        patterns = patterns + PREPROCESSING_CODE_PATTERNS
    if args.include_release_code and args.pattern is None:
        patterns = patterns + RELEASE_CODE_PATTERNS
    manifest = build_manifest(repo_root=repo_root, repo_id=args.repo_id, patterns=patterns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "repo_id": manifest["repo_id"],
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
