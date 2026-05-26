#!/usr/bin/env python3
"""Create a V2 refinement baseline snapshot without modifying prior outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_ROOT = Path("logs/sql_high_corpus_build_20260404")
DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404/v2_refinement")
DEFAULT_SCOPE_CSV = Path("logs/sql_high_corpus_build_20260404/scope/high_datasets.csv")
DEFAULT_AUDIT_CSV = Path("logs/sql_high_corpus_build_20260404/global/checkpoint_c_sql_audit.csv")
DEFAULT_MASTER_SQL = Path("logs/sql_high_corpus_build_20260404/global/master_sql_inventory_all.csv")
DEFAULT_SOURCE_INVENTORY = Path("logs/sql_high_corpus_build_20260404/global/all_source_inventory.csv")
DEFAULT_FINAL_INDEX = Path("logs/sql_high_corpus_build_20260404/final/final_index.csv")
DEFAULT_CHECKPOINT_C = Path("logs/sql_high_corpus_build_20260404/global/checkpoint_c_status.json")
DEFAULT_CHECKPOINT_D = Path("logs/sql_high_corpus_build_20260404/final/checkpoint_d_status.json")

BASELINE_DATASET_FIELDNAMES = [
    "own_id",
    "dataset_id",
    "dataset_name",
    "class_type",
    "source_type",
    "total_sql_rows",
    "strict_sql_count",
    "weak_sql_count",
    "collision_risk_sql_count",
    "trustworthy_sql_count",
    "top_strict_sql_count",
    "question_seed_count",
    "source_url_count",
    "checkpoint_c_status",
    "question_taxonomy_readiness",
    "top_strict_sql_nonempty",
    "known_risk_tags",
    "official_source_url",
    "best_sql_source_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a V2 refinement Phase 0 baseline snapshot from an existing "
            "sql_high corpus build."
        )
    )
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE_CSV)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--master-sql", type=Path, default=DEFAULT_MASTER_SQL)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--final-index", type=Path, default=DEFAULT_FINAL_INDEX)
    parser.add_argument("--checkpoint-c", type=Path, default=DEFAULT_CHECKPOINT_C)
    parser.add_argument("--checkpoint-d", type=Path, default=DEFAULT_CHECKPOINT_D)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_int(value: str | int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = value.strip()
    if not text:
        return 0
    return int(text)


def build_index(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {(row.get(key) or "").strip(): row for row in rows}


def unique_source_url_counts(source_rows: list[dict[str, str]]) -> dict[str, int]:
    url_sets: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        own_id = (row.get("own_id") or "").strip()
        source_url = (row.get("source_url") or "").strip()
        if own_id and source_url:
            url_sets[own_id].add(source_url)
    return {own_id: len(urls) for own_id, urls in url_sets.items()}


def build_dataset_rows(
    scope_rows: list[dict[str, str]],
    audit_index: dict[str, dict[str, str]],
    final_index: dict[str, dict[str, str]],
    source_url_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope_row in scope_rows:
        own_id = (scope_row.get("own_id") or "").strip()
        audit_row = audit_index.get(own_id, {})
        final_row = final_index.get(own_id, {})
        top_strict_sql_count = to_int(final_row.get("top_strict_sql_count"))
        row = {
            "own_id": own_id,
            "dataset_id": (scope_row.get("dataset_id") or "").strip(),
            "dataset_name": (scope_row.get("dataset_name") or "").strip(),
            "class_type": (scope_row.get("class_type") or "").strip(),
            "source_type": (scope_row.get("source_type") or "").strip(),
            "total_sql_rows": to_int(audit_row.get("total_sql_rows")),
            "strict_sql_count": to_int(audit_row.get("strict_sql_count")),
            "weak_sql_count": to_int(audit_row.get("weak_sql_count")),
            "collision_risk_sql_count": to_int(audit_row.get("collision_risk_sql_count")),
            "trustworthy_sql_count": to_int(audit_row.get("trustworthy_sql_count")),
            "top_strict_sql_count": top_strict_sql_count,
            "question_seed_count": to_int(final_row.get("question_seed_count")),
            "source_url_count": source_url_counts.get(own_id, 0),
            "checkpoint_c_status": (final_row.get("checkpoint_c_status") or "").strip(),
            "question_taxonomy_readiness": (final_row.get("question_taxonomy_readiness") or "").strip(),
            "top_strict_sql_nonempty": "yes" if top_strict_sql_count > 0 else "no",
            "known_risk_tags": (final_row.get("known_risk_tags") or "").strip(),
            "official_source_url": (final_row.get("official_source_url") or "").strip(),
            "best_sql_source_url": (final_row.get("best_sql_source_url") or "").strip(),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -to_int(row["strict_sql_count"]),
            -to_int(row["collision_risk_sql_count"]),
            row["own_id"],
        )
    )
    return rows


def top_dataset_lines(
    rows: list[dict[str, Any]],
    count_key: str,
    top_n: int,
) -> list[str]:
    ranked = [
        row for row in sorted(rows, key=lambda row: (-to_int(row[count_key]), row["own_id"]))
        if to_int(row[count_key]) > 0
    ][:top_n]
    if not ranked:
        return ["- None"]
    return [
        f"- {row['own_id']} - {row['dataset_name']}: {count_key}={row[count_key]}"
        for row in ranked
    ]


def nonempty_top_strict_lines(rows: list[dict[str, Any]]) -> list[str]:
    selected = [
        row for row in sorted(rows, key=lambda row: (-to_int(row["top_strict_sql_count"]), row["own_id"]))
        if to_int(row["top_strict_sql_count"]) > 0
    ]
    if not selected:
        return ["- None"]
    return [
        f"- {row['own_id']} - {row['dataset_name']}: top_strict_sql_count={row['top_strict_sql_count']}, question_seed_count={row['question_seed_count']}"
        for row in selected
    ]


def build_snapshot_markdown(
    baseline_root: Path,
    dataset_rows: list[dict[str, Any]],
    total_sql_rows: int,
    strict_total: int,
    weak_total: int,
    collision_total: int,
    checkpoint_c_payload: dict[str, Any],
    checkpoint_d_payload: dict[str, Any],
    class_counter: Counter[str],
    source_counter: Counter[str],
) -> str:
    checkpoint_c_status = checkpoint_c_payload.get("overall_status", "UNKNOWN")
    checkpoint_d_status = checkpoint_d_payload.get("status", "UNKNOWN")
    checkpoint_d_ready = checkpoint_d_payload.get("ready_for_human_review")
    checkpoint_c_summary = checkpoint_c_payload.get("global_summary", {})
    checkpoint_d_summary = checkpoint_d_payload.get("summary", {})

    lines = [
        "# V2 Baseline Snapshot",
        "",
        f"- Baseline input root: `{baseline_root.resolve()}`",
        f"- Snapshot generated at UTC: `{utc_now_iso()}`",
        "- This is a read-only baseline package for V2 refinement. No prior outputs were cleaned, replaced, or overwritten.",
        "",
        "## Corpus Totals",
        "",
        f"- Total datasets: {len(dataset_rows)}",
        f"- Total SQL rows: {total_sql_rows}",
        f"- Strict SQL rows: {strict_total}",
        f"- Weak SQL rows: {weak_total}",
        f"- Collision-risk SQL rows: {collision_total}",
        "",
        "## Checkpoint Status",
        "",
        f"- Checkpoint C overall status: `{checkpoint_c_status}`",
        f"- Checkpoint C ready dataset count: {checkpoint_c_summary.get('ready_dataset_count', 'unknown')}",
        f"- Checkpoint C fail dataset count: {checkpoint_c_summary.get('fail_dataset_count', 'unknown')}",
        f"- Checkpoint D status: `{checkpoint_d_status}`",
        f"- Checkpoint D ready_for_human_review: `{checkpoint_d_ready}`",
        f"- Checkpoint D total trustworthy strict SQL items: {checkpoint_d_summary.get('total_trustworthy_strict_sql_items', 'unknown')}",
        "",
        "## Dataset Distribution",
        "",
        f"- Class types: {', '.join(f'{key}={class_counter[key]}' for key in sorted(class_counter))}",
        f"- Source types: {', '.join(f'{key}={source_counter[key]}' for key in sorted(source_counter))}",
        "",
        "## Datasets With Non-Empty top_strict_sql.csv",
        "",
        *nonempty_top_strict_lines(dataset_rows),
        "",
        "## Highest Strict Counts",
        "",
        *top_dataset_lines(dataset_rows, "strict_sql_count", top_n=10),
        "",
        "## Highest Collision Counts",
        "",
        *top_dataset_lines(dataset_rows, "collision_risk_sql_count", top_n=10),
        "",
        "## Additional Baseline Notes",
        "",
        f"- Datasets with non-zero trustworthy SQL counts: {sum(1 for row in dataset_rows if to_int(row['trustworthy_sql_count']) > 0)}",
        f"- Datasets with non-empty top_strict_sql.csv: {sum(1 for row in dataset_rows if row['top_strict_sql_nonempty'] == 'yes')}",
        f"- Datasets marked question-taxonomy ready in the final index: {sum(1 for row in dataset_rows if row['question_taxonomy_readiness'] == 'yes')}",
        f"- Datasets marked checkpoint C FAIL in the final index: {sum(1 for row in dataset_rows if row['checkpoint_c_status'] == 'FAIL')}",
    ]
    return "\n".join(lines)


def build_manifest(
    args: argparse.Namespace,
    output_paths: list[Path],
    dataset_rows: list[dict[str, Any]],
    total_sql_rows: int,
    strict_total: int,
    weak_total: int,
    collision_total: int,
    checkpoint_c_payload: dict[str, Any],
    checkpoint_d_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "v2_phase0_baseline_snapshot",
        "generated_at_utc": utc_now_iso(),
        "input": {
            "baseline_root": str(args.baseline_root.resolve()),
            "scope_csv_path": str(args.scope_csv.resolve()),
            "scope_csv_sha256": sha256_file(args.scope_csv),
            "audit_csv_path": str(args.audit_csv.resolve()),
            "audit_csv_sha256": sha256_file(args.audit_csv),
            "master_sql_path": str(args.master_sql.resolve()),
            "master_sql_sha256": sha256_file(args.master_sql),
            "source_inventory_path": str(args.source_inventory.resolve()),
            "source_inventory_sha256": sha256_file(args.source_inventory),
            "final_index_path": str(args.final_index.resolve()),
            "final_index_sha256": sha256_file(args.final_index),
            "checkpoint_c_path": str(args.checkpoint_c.resolve()),
            "checkpoint_c_sha256": sha256_file(args.checkpoint_c),
            "checkpoint_d_path": str(args.checkpoint_d.resolve()),
            "checkpoint_d_sha256": sha256_file(args.checkpoint_d),
        },
        "summary": {
            "total_datasets": len(dataset_rows),
            "total_sql_rows": total_sql_rows,
            "strict_sql_count_total": strict_total,
            "weak_sql_count_total": weak_total,
            "collision_risk_sql_count_total": collision_total,
            "checkpoint_c_overall_status": checkpoint_c_payload.get("overall_status", "UNKNOWN"),
            "checkpoint_d_status": checkpoint_d_payload.get("status", "UNKNOWN"),
            "checkpoint_d_ready_for_human_review": checkpoint_d_payload.get("ready_for_human_review"),
            "nonempty_top_strict_dataset_count": sum(
                1 for row in dataset_rows if row["top_strict_sql_nonempty"] == "yes"
            ),
        },
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
    }


def main() -> int:
    args = parse_args()
    baseline_dir = args.output_root / "baseline"
    snapshot_path = baseline_dir / "baseline_snapshot.md"
    dataset_table_path = baseline_dir / "baseline_dataset_table.csv"
    manifest_path = baseline_dir / "run_manifest_v2_phase0.json"

    scope_rows = read_csv_rows(args.scope_csv)
    audit_rows = read_csv_rows(args.audit_csv)
    final_rows = read_csv_rows(args.final_index)
    source_rows = read_csv_rows(args.source_inventory)
    master_sql_rows = read_csv_rows(args.master_sql)
    checkpoint_c_payload = json.loads(args.checkpoint_c.read_text(encoding="utf-8"))
    checkpoint_d_payload = json.loads(args.checkpoint_d.read_text(encoding="utf-8"))

    audit_index = build_index(audit_rows, "own_id")
    final_index = build_index(final_rows, "own_id")
    source_url_counts = unique_source_url_counts(source_rows)

    dataset_rows = build_dataset_rows(
        scope_rows=scope_rows,
        audit_index=audit_index,
        final_index=final_index,
        source_url_counts=source_url_counts,
    )

    total_sql_rows = len(master_sql_rows)
    strict_total = sum(to_int(row["strict_sql_count"]) for row in dataset_rows)
    weak_total = sum(to_int(row["weak_sql_count"]) for row in dataset_rows)
    collision_total = sum(to_int(row["collision_risk_sql_count"]) for row in dataset_rows)
    class_counter = Counter((row.get("class_type") or "").strip() or "unknown" for row in scope_rows)
    source_counter = Counter((row.get("source_type") or "").strip() or "unknown" for row in scope_rows)

    write_csv(
        dataset_table_path,
        BASELINE_DATASET_FIELDNAMES,
        [{field: row.get(field, "") for field in BASELINE_DATASET_FIELDNAMES} for row in dataset_rows],
    )
    write_text(
        snapshot_path,
        build_snapshot_markdown(
            baseline_root=args.baseline_root,
            dataset_rows=dataset_rows,
            total_sql_rows=total_sql_rows,
            strict_total=strict_total,
            weak_total=weak_total,
            collision_total=collision_total,
            checkpoint_c_payload=checkpoint_c_payload,
            checkpoint_d_payload=checkpoint_d_payload,
            class_counter=class_counter,
            source_counter=source_counter,
        ),
    )
    manifest_payload = build_manifest(
        args=args,
        output_paths=[snapshot_path, dataset_table_path],
        dataset_rows=dataset_rows,
        total_sql_rows=total_sql_rows,
        strict_total=strict_total,
        weak_total=weak_total,
        collision_total=collision_total,
        checkpoint_c_payload=checkpoint_c_payload,
        checkpoint_d_payload=checkpoint_d_payload,
    )
    write_json(manifest_path, manifest_payload)
    manifest_payload["outputs"] = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in [snapshot_path, dataset_table_path, manifest_path]
    ]
    write_json(manifest_path, manifest_payload)

    for path in [snapshot_path, dataset_table_path, manifest_path]:
        print(str(path.resolve()))
    print("V2 PHASE 0 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
