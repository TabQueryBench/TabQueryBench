#!/usr/bin/env python3
"""Build a supervisor-facing morning review bundle for sql_high corpus outputs."""

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


DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404")
DEFAULT_SCOPE_CSV = Path("logs/sql_high_corpus_build_20260404/scope/high_datasets.csv")
DEFAULT_SOURCE_INVENTORY = Path("logs/sql_high_corpus_build_20260404/global/all_source_inventory.csv")
DEFAULT_MASTER_SQL = Path("logs/sql_high_corpus_build_20260404/global/master_sql_inventory_all.csv")
DEFAULT_CHECKPOINT_C_AUDIT = Path("logs/sql_high_corpus_build_20260404/global/checkpoint_c_sql_audit.csv")
DEFAULT_CHECKPOINT_C_STATUS = Path("logs/sql_high_corpus_build_20260404/global/checkpoint_c_status.json")
DEFAULT_FINAL_INDEX = Path("logs/sql_high_corpus_build_20260404/final/final_index.csv")

DATASET_TABLE_FIELDNAMES = [
    "own_id",
    "dataset_name",
    "strict_sql_count",
    "trustworthy_sql_count",
    "weak_sql_count",
    "collision_risk_sql_count",
    "source_url_count",
    "readiness_for_question_taxonomy",
    "recommended_next_action",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single morning review bundle from existing sql_high corpus "
            "Phase A-D artifacts without starting any new phase."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE_CSV)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--master-sql", type=Path, default=DEFAULT_MASTER_SQL)
    parser.add_argument("--checkpoint-c-audit", type=Path, default=DEFAULT_CHECKPOINT_C_AUDIT)
    parser.add_argument("--checkpoint-c-status", type=Path, default=DEFAULT_CHECKPOINT_C_STATUS)
    parser.add_argument("--final-index", type=Path, default=DEFAULT_FINAL_INDEX)
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


def split_tags(tag_text: str) -> list[str]:
    return [tag.strip() for tag in (tag_text or "").split(";") if tag.strip()]


def build_index(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {(row.get(key) or "").strip(): row for row in rows}


def unique_source_url_counts(source_rows: list[dict[str, str]]) -> dict[str, int]:
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        own_id = (row.get("own_id") or "").strip()
        source_url = (row.get("source_url") or "").strip()
        if own_id and source_url:
            by_dataset[own_id].add(source_url)
    return {own_id: len(urls) for own_id, urls in by_dataset.items()}


def source_specificity_counts(source_rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in source_rows:
        own_id = (row.get("own_id") or "").strip()
        label = (row.get("dataset_specificity_hint") or "").strip() or "unknown"
        if own_id:
            counts[own_id][label] += 1
    return counts


def source_status_counts(source_rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in source_rows:
        own_id = (row.get("own_id") or "").strip()
        status = (row.get("http_status") or "").strip() or "unknown"
        if own_id:
            counts[own_id][status] += 1
    return counts


def recommended_next_action(
    row: dict[str, Any],
    risk_tags: list[str],
) -> str:
    checkpoint_status = (row.get("checkpoint_c_status") or "").strip()
    readiness = (row.get("readiness_for_question_taxonomy") or "").strip()
    trustworthy = to_int(row.get("trustworthy_sql_count"))
    strict = to_int(row.get("strict_sql_count"))
    weak = to_int(row.get("weak_sql_count"))
    collision = to_int(row.get("collision_risk_sql_count"))
    top_strict = to_int(row.get("top_strict_sql_count"))
    source_url_count = to_int(row.get("source_url_count"))

    if readiness == "yes" and top_strict > 0:
        return "Human review curated top strict SQL and question seeds"
    if readiness == "yes" and trustworthy > 0 and top_strict == 0:
        return "Curate a small strict SQL shortlist for supervisor review"
    if readiness == "yes" and trustworthy == 0 and weak > 0:
        return "Manual verify weak SQL and promote only exact benchmark matches"
    if checkpoint_status == "FAIL" and trustworthy > 0:
        return "Filter misleading strict SQL, then rerun the SQL audit"
    if "no_sql_inventory_rows" in risk_tags:
        return "Run targeted source search for explicit benchmark SQL"
    if collision > 0 and trustworthy == 0:
        return "Replace collision-risk SQL with benchmark-specific sources"
    if strict == 0 and weak > 0:
        return "Run targeted source search to replace weak SQL with strict evidence"
    if source_url_count <= 6:
        return "Expand source discovery before any question drafting"
    return "Review provenance and recollect benchmark-specific SQL evidence"


def action_requires_source_search(action: str) -> bool:
    text = (action or "").lower()
    search_markers = (
        "source search",
        "source discovery",
        "benchmark-specific sources",
    )
    return any(marker in text for marker in search_markers)


def build_dataset_rows(
    scope_rows: list[dict[str, str]],
    final_index: dict[str, dict[str, str]],
    audit_index: dict[str, dict[str, str]],
    source_url_count_index: dict[str, int],
    source_specificity_index: dict[str, Counter[str]],
    source_status_index: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    dataset_rows: list[dict[str, Any]] = []
    for scope_row in scope_rows:
        own_id = (scope_row.get("own_id") or "").strip()
        final_row = final_index.get(own_id, {})
        audit_row = audit_index.get(own_id, {})
        risk_tags = split_tags(audit_row.get("primary_risk_tags", ""))

        row: dict[str, Any] = {
            "own_id": own_id,
            "dataset_id": (scope_row.get("dataset_id") or "").strip(),
            "dataset_name": (scope_row.get("dataset_name") or "").strip(),
            "strict_sql_count": to_int(audit_row.get("strict_sql_count")),
            "trustworthy_sql_count": to_int(audit_row.get("trustworthy_sql_count")),
            "weak_sql_count": to_int(audit_row.get("weak_sql_count")),
            "collision_risk_sql_count": to_int(audit_row.get("collision_risk_sql_count")),
            "source_url_count": source_url_count_index.get(own_id, 0),
            "readiness_for_question_taxonomy": (
                (audit_row.get("readiness_for_question_taxonomy") or "").strip()
                or (final_row.get("question_taxonomy_readiness") or "").strip()
            ),
            "checkpoint_c_status": (
                (audit_row.get("dataset_status") or "").strip()
                or (final_row.get("checkpoint_c_status") or "").strip()
            ),
            "top_strict_sql_count": to_int(final_row.get("top_strict_sql_count")),
            "question_seed_count": to_int(final_row.get("question_seed_count")),
            "usable_unique_sql_count": to_int(audit_row.get("usable_unique_sql_count")),
            "potentially_misleading_row_count": to_int(audit_row.get("potentially_misleading_row_count")),
            "duplicate_sql_text_clean_count": to_int(audit_row.get("duplicate_sql_text_clean_count")),
            "over_reliance_on_weak_or_collision": (audit_row.get("over_reliance_on_weak_or_collision") or "").strip(),
            "primary_risk_tags": risk_tags,
            "remediation_actions": (audit_row.get("remediation_actions") or "").strip(),
            "official_source_url": (final_row.get("official_source_url") or "").strip(),
            "best_sql_source_url": (final_row.get("best_sql_source_url") or "").strip(),
            "source_specificity_strict_count": source_specificity_index.get(own_id, Counter()).get("strict", 0),
            "source_specificity_weak_count": source_specificity_index.get(own_id, Counter()).get("weak", 0),
            "source_specificity_collision_risk_count": source_specificity_index.get(own_id, Counter()).get("collision_risk", 0),
            "source_specificity_unknown_count": source_specificity_index.get(own_id, Counter()).get("unknown", 0),
            "source_http_404_count": source_status_index.get(own_id, Counter()).get("404", 0),
            "source_http_429_count": source_status_index.get(own_id, Counter()).get("429", 0),
        }
        row["recommended_next_action"] = recommended_next_action(row, risk_tags)
        row["needs_more_source_search"] = "yes" if action_requires_source_search(row["recommended_next_action"]) else "no"
        dataset_rows.append(row)

    dataset_rows.sort(
        key=lambda row: (
            -to_int(row["trustworthy_sql_count"]),
            -to_int(row["strict_sql_count"]),
            row["own_id"],
        )
    )
    return dataset_rows


def ranking_lines(dataset_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, row in enumerate(dataset_rows, start=1):
        lines.append(
            f"| {index} | {row['own_id']} | {row['dataset_name']} | "
            f"{row['trustworthy_sql_count']} | {row['strict_sql_count']} | "
            f"{row['collision_risk_sql_count']} | {row['readiness_for_question_taxonomy']} | "
            f"{row['recommended_next_action']} |"
        )
    return lines


def bullet_dataset_lines(
    dataset_rows: list[dict[str, Any]],
    include_source_counts: bool = False,
) -> list[str]:
    lines: list[str] = []
    for row in dataset_rows:
        detail = (
            f"{row['own_id']} - {row['dataset_name']}: trustworthy={row['trustworthy_sql_count']}, "
            f"strict={row['strict_sql_count']}, weak={row['weak_sql_count']}, "
            f"collision={row['collision_risk_sql_count']}, readiness={row['readiness_for_question_taxonomy']}"
        )
        if include_source_counts:
            detail += f", source_urls={row['source_url_count']}"
        detail += f". Next: {row['recommended_next_action']}."
        lines.append(f"- {detail}")
    return lines


def build_summary_markdown(
    dataset_rows: list[dict[str, Any]],
    total_dataset_count: int,
    total_sql_items: int,
    total_trustworthy_strict_sql_items: int,
    collision_rows: list[dict[str, Any]],
    more_search_rows: list[dict[str, Any]],
) -> str:
    ready_count = sum(1 for row in dataset_rows if row["readiness_for_question_taxonomy"] == "yes")
    not_ready_count = total_dataset_count - ready_count
    top_ready_rows = [row for row in dataset_rows if row["readiness_for_question_taxonomy"] == "yes" and row["trustworthy_sql_count"] > 0]
    summary_lines = [
        "# Morning Review Summary",
        "",
        "## Total sql_high datasets processed",
        "",
        f"- Total datasets: {total_dataset_count}",
        f"- Ready for question taxonomy: {ready_count}",
        f"- Not yet ready for question taxonomy: {not_ready_count}",
        "",
        "## Total SQL items extracted",
        "",
        f"- Total SQL items extracted in Phase C: {total_sql_items}",
        "",
        "## Total trustworthy strict SQL items",
        "",
        f"- Total trustworthy strict SQL items: {total_trustworthy_strict_sql_items}",
        f"- Datasets with non-zero trustworthy strict SQL: {sum(1 for row in dataset_rows if row['trustworthy_sql_count'] > 0)}",
        f"- Datasets with non-empty curated `top_strict_sql.csv`: {sum(1 for row in dataset_rows if row['top_strict_sql_count'] > 0)}",
        "",
        "## Dataset ranking by trustworthy SQL count",
        "",
        "| Rank | own_id | dataset_name | trustworthy_sql_count | strict_sql_count | collision_risk_sql_count | readiness_for_question_taxonomy | recommended_next_action |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        *ranking_lines(dataset_rows),
        "",
        "## Datasets with highest collision risk",
        "",
    ]
    if collision_rows:
        summary_lines.extend(bullet_dataset_lines(collision_rows))
    else:
        summary_lines.append("- No datasets currently have non-zero `collision_risk_sql_count`.")

    summary_lines.extend(
        [
            "",
            "## Datasets needing more source search",
            "",
        ]
    )
    if more_search_rows:
        summary_lines.extend(bullet_dataset_lines(more_search_rows, include_source_counts=True))
    else:
        summary_lines.append("- No datasets are currently flagged as needing additional source search.")

    summary_lines.extend(
        [
            "",
            "## Supervisor Review Pointers",
            "",
            f"- Highest immediately reviewable datasets with curated strict SQL: {', '.join(row['own_id'] for row in top_ready_rows[:5]) or 'none'}.",
            "- Treat `checkpoint_c_status=FAIL` rows as review targets, not ready-to-author question-taxonomy inputs.",
            "- Use the dataset table CSV for filtering and sorting, and the risks memo for where manual cleanup or extra source search should start.",
        ]
    )
    return "\n".join(summary_lines)


def build_risks_markdown(
    dataset_rows: list[dict[str, Any]],
    checkpoint_c_status_payload: dict[str, Any],
) -> str:
    upstream_status = (checkpoint_c_status_payload.get("overall_status") or "").strip() or "UNKNOWN"
    zero_sql_rows = [row for row in dataset_rows if row["trustworthy_sql_count"] == 0 and row["weak_sql_count"] == 0 and row["collision_risk_sql_count"] == 0]
    misleading_heavy_rows = sorted(
        [row for row in dataset_rows if row["potentially_misleading_row_count"] > 0],
        key=lambda row: (-row["potentially_misleading_row_count"], row["own_id"]),
    )[:8]
    source_search_rows = [row for row in dataset_rows if row["needs_more_source_search"] == "yes"]
    collision_rows = [row for row in dataset_rows if row["collision_risk_sql_count"] > 0]
    risk_lines = [
        "# Morning Review Risks",
        "",
        "## Global Posture",
        "",
        f"- Upstream Phase C checkpoint status remains `{upstream_status}`.",
        f"- Ready for question taxonomy: {sum(1 for row in dataset_rows if row['readiness_for_question_taxonomy'] == 'yes')} of {len(dataset_rows)} datasets.",
        f"- Datasets still needing manual cleanup or more source work: {sum(1 for row in dataset_rows if row['checkpoint_c_status'] == 'FAIL')}.",
        "",
        "## Highest Collision-Risk Datasets",
        "",
    ]
    if collision_rows:
        risk_lines.extend(
            [
                f"- {row['own_id']} - {row['dataset_name']}: collision_risk_sql_count={row['collision_risk_sql_count']}, "
                f"weak_sql_count={row['weak_sql_count']}, source_collision_hints={row['source_specificity_collision_risk_count']}, "
                f"recommended_next_action={row['recommended_next_action']}."
                for row in sorted(
                    collision_rows,
                    key=lambda row: (-row["collision_risk_sql_count"], -row["weak_sql_count"], row["own_id"]),
                )
            ]
        )
    else:
        risk_lines.append("- No datasets currently have non-zero collision-risk SQL rows.")

    risk_lines.extend(
        [
            "",
            "## Datasets Needing More Source Search",
            "",
        ]
    )
    if source_search_rows:
        risk_lines.extend(
            [
                f"- {row['own_id']} - {row['dataset_name']}: source_urls={row['source_url_count']}, "
                f"strict={row['strict_sql_count']}, weak={row['weak_sql_count']}, collision={row['collision_risk_sql_count']}, "
                f"remediation={row['recommended_next_action']}."
                for row in sorted(
                    source_search_rows,
                    key=lambda row: (
                        row["readiness_for_question_taxonomy"] == "yes",
                        -row["collision_risk_sql_count"],
                        row["source_url_count"],
                        row["own_id"],
                    ),
                )
            ]
        )
    else:
        risk_lines.append("- No datasets are currently flagged as needing more source search.")

    risk_lines.extend(
        [
            "",
            "## High-Volume But Risky Inventories",
            "",
        ]
    )
    if misleading_heavy_rows:
        risk_lines.extend(
            [
                f"- {row['own_id']} - {row['dataset_name']}: trustworthy_sql_count={row['trustworthy_sql_count']}, "
                f"potentially_misleading_row_count={row['potentially_misleading_row_count']}, "
                f"duplicate_sql_text_clean_count={row['duplicate_sql_text_clean_count']}, "
                f"recommended_next_action={row['recommended_next_action']}."
                for row in misleading_heavy_rows
            ]
        )
    else:
        risk_lines.append("- No datasets currently have a non-zero misleading-row count.")

    risk_lines.extend(
        [
            "",
            "## Zero-SQL Coverage Datasets",
            "",
        ]
    )
    if zero_sql_rows:
        risk_lines.extend(
            [
                f"- {row['own_id']} - {row['dataset_name']}: no strict, weak, or collision-risk SQL survived; source_urls={row['source_url_count']}; next={row['recommended_next_action']}."
                for row in zero_sql_rows
            ]
        )
    else:
        risk_lines.append("- Every dataset has at least some SQL coverage in the current inventory.")

    risk_lines.extend(
        [
            "",
            "## Review Priorities This Morning",
            "",
            "- First inspect the curated-ready subset with non-empty `top_strict_sql.csv`: `c17`, `m4`, `m8`, `m11`, `m3`.",
            "- Then inspect high-count but audit-failing inventories: `c5`, `c7`, and `n16`.",
            "- Finally prioritize fresh source search for the zero-SQL and collision-heavy datasets before any new extraction or taxonomy work.",
        ]
    )
    return "\n".join(risk_lines)


def build_checkpoint_payload(
    args: argparse.Namespace,
    dataset_rows: list[dict[str, Any]],
    total_sql_items: int,
    total_trustworthy_strict_sql_items: int,
    output_paths: list[Path],
    checkpoint_c_status_payload: dict[str, Any],
) -> dict[str, Any]:
    more_search_rows = [row for row in dataset_rows if row["needs_more_source_search"] == "yes"]
    highest_collision_rows = [
        {
            "own_id": row["own_id"],
            "dataset_name": row["dataset_name"],
            "collision_risk_sql_count": row["collision_risk_sql_count"],
            "recommended_next_action": row["recommended_next_action"],
        }
        for row in sorted(
            [row for row in dataset_rows if row["collision_risk_sql_count"] > 0],
            key=lambda row: (-row["collision_risk_sql_count"], -row["weak_sql_count"], row["own_id"]),
        )[:10]
    ]
    return {
        "checkpoint": "D",
        "phase_name": "morning_review_bundle",
        "generated_at_utc": utc_now_iso(),
        "status": "PASS",
        "ready_for_human_review": True,
        "note": (
            "Morning review bundle generated successfully from existing Phase A-D artifacts. "
            "No new collection or extraction phase was started."
        ),
        "upstream_checkpoint_c_overall_status": checkpoint_c_status_payload.get("overall_status", "UNKNOWN"),
        "input": {
            "output_root": str(args.output_root.resolve()),
            "scope_csv_path": str(args.scope_csv.resolve()),
            "scope_csv_sha256": sha256_file(args.scope_csv),
            "scope_dataset_count": len(dataset_rows),
            "source_inventory_path": str(args.source_inventory.resolve()),
            "source_inventory_sha256": sha256_file(args.source_inventory),
            "master_sql_path": str(args.master_sql.resolve()),
            "master_sql_sha256": sha256_file(args.master_sql),
            "checkpoint_c_audit_path": str(args.checkpoint_c_audit.resolve()),
            "checkpoint_c_audit_sha256": sha256_file(args.checkpoint_c_audit),
            "checkpoint_c_status_path": str(args.checkpoint_c_status.resolve()),
            "checkpoint_c_status_sha256": sha256_file(args.checkpoint_c_status),
            "final_index_path": str(args.final_index.resolve()),
            "final_index_sha256": sha256_file(args.final_index),
        },
        "summary": {
            "total_sql_high_datasets_processed": len(dataset_rows),
            "total_sql_items_extracted": total_sql_items,
            "total_trustworthy_strict_sql_items": total_trustworthy_strict_sql_items,
            "datasets_ready_for_question_taxonomy": sum(
                1 for row in dataset_rows if row["readiness_for_question_taxonomy"] == "yes"
            ),
            "datasets_not_ready_for_question_taxonomy": sum(
                1 for row in dataset_rows if row["readiness_for_question_taxonomy"] != "yes"
            ),
            "datasets_needing_more_source_search_count": len(more_search_rows),
            "datasets_with_nonzero_collision_risk_count": sum(
                1 for row in dataset_rows if row["collision_risk_sql_count"] > 0
            ),
        },
        "datasets_with_highest_collision_risk": highest_collision_rows,
        "datasets_needing_more_source_search": [
            {
                "own_id": row["own_id"],
                "dataset_name": row["dataset_name"],
                "source_url_count": row["source_url_count"],
                "strict_sql_count": row["strict_sql_count"],
                "weak_sql_count": row["weak_sql_count"],
                "collision_risk_sql_count": row["collision_risk_sql_count"],
                "recommended_next_action": row["recommended_next_action"],
            }
            for row in more_search_rows
        ],
        "generated_files": [
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
    final_dir = args.output_root / "final"
    summary_path = final_dir / "morning_review_summary.md"
    table_path = final_dir / "morning_review_dataset_table.csv"
    risks_path = final_dir / "morning_review_risks.md"
    checkpoint_path = final_dir / "checkpoint_d_status.json"

    scope_rows = read_csv_rows(args.scope_csv)
    source_rows = read_csv_rows(args.source_inventory)
    sql_rows = read_csv_rows(args.master_sql)
    audit_rows = read_csv_rows(args.checkpoint_c_audit)
    final_rows = read_csv_rows(args.final_index)
    checkpoint_c_status_payload = json.loads(args.checkpoint_c_status.read_text(encoding="utf-8"))

    final_index = build_index(final_rows, "own_id")
    audit_index = build_index(audit_rows, "own_id")
    source_url_count_index = unique_source_url_counts(source_rows)
    source_specificity_index = source_specificity_counts(source_rows)
    source_status_index = source_status_counts(source_rows)

    dataset_rows = build_dataset_rows(
        scope_rows=scope_rows,
        final_index=final_index,
        audit_index=audit_index,
        source_url_count_index=source_url_count_index,
        source_specificity_index=source_specificity_index,
        source_status_index=source_status_index,
    )

    total_dataset_count = len(dataset_rows)
    total_sql_items = len(sql_rows)
    total_trustworthy_strict_sql_items = sum(row["trustworthy_sql_count"] for row in dataset_rows)
    collision_rows = [
        row
        for row in sorted(
            [row for row in dataset_rows if row["collision_risk_sql_count"] > 0],
            key=lambda row: (-row["collision_risk_sql_count"], -row["weak_sql_count"], row["own_id"]),
        )
    ]
    more_search_rows = [
        row
        for row in sorted(
            [row for row in dataset_rows if row["needs_more_source_search"] == "yes"],
            key=lambda row: (
                row["readiness_for_question_taxonomy"] == "yes",
                -row["collision_risk_sql_count"],
                row["source_url_count"],
                row["own_id"],
            ),
        )
    ]

    write_csv(
        table_path,
        DATASET_TABLE_FIELDNAMES,
        [
            {field: row.get(field, "") for field in DATASET_TABLE_FIELDNAMES}
            for row in dataset_rows
        ],
    )
    write_text(
        summary_path,
        build_summary_markdown(
            dataset_rows=dataset_rows,
            total_dataset_count=total_dataset_count,
            total_sql_items=total_sql_items,
            total_trustworthy_strict_sql_items=total_trustworthy_strict_sql_items,
            collision_rows=collision_rows,
            more_search_rows=more_search_rows,
        ),
    )
    write_text(
        risks_path,
        build_risks_markdown(
            dataset_rows=dataset_rows,
            checkpoint_c_status_payload=checkpoint_c_status_payload,
        ),
    )

    output_paths = [summary_path, table_path, risks_path, checkpoint_path]
    checkpoint_payload = build_checkpoint_payload(
        args=args,
        dataset_rows=dataset_rows,
        total_sql_items=total_sql_items,
        total_trustworthy_strict_sql_items=total_trustworthy_strict_sql_items,
        output_paths=output_paths[:-1],
        checkpoint_c_status_payload=checkpoint_c_status_payload,
    )
    write_json(checkpoint_path, checkpoint_payload)
    checkpoint_payload["generated_files"] = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in output_paths
    ]
    write_json(checkpoint_path, checkpoint_payload)

    for path in output_paths:
        print(str(path.resolve()))
    print("")
    print("READY FOR HUMAN REVIEW")
    print(f"datasets={total_dataset_count}")
    print(f"sql_items={total_sql_items}")
    print(f"trustworthy_strict_sql_items={total_trustworthy_strict_sql_items}")
    print(f"needs_more_source_search={len(more_search_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
