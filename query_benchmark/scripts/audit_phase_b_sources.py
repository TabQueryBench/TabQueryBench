#!/usr/bin/env python3
"""Audit Phase B source discovery quality before SQL extraction."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INPUT = Path("logs/sql_high_corpus_build_20260404/global/all_source_inventory.csv")
DEFAULT_OUTPUT_DIR = Path("logs/sql_high_corpus_build_20260404/global")
OK_STATUSES = {"200", "301", "302", "303", "307", "308"}
TEMP_UNREACHABLE_STATUSES = {"429"}
SEARCH_SOURCE_TYPES = {"github_repo_search", "github_code_search", "kaggle_code_search"}
OFFICIAL_SOURCE_TYPES = {
    "official_dataset_page",
    "official_api",
    "openml_api",
    "openml_task_page",
    "openml_task_search",
    "kaggle_overview_page",
    "kaggle_data_page",
    "kaggle_code_page",
    "readme_or_metadata",
    "paper",
}
GENERIC_COLLISION_PATTERNS = (
    "intro to sql",
    "course answer",
    "course answers",
    "course",
    "bootcamp",
    "tutorial",
    "theDataScienceBootcamp".lower(),
)
TOKEN_STOPWORDS = {
    "and",
    "challenge",
    "classification",
    "customer",
    "data",
    "dataset",
    "datasets",
    "default",
    "for",
    "from",
    "prediction",
    "regression",
    "safe",
    "the",
    "with",
}


@dataclass(frozen=True)
class ExpandedRow:
    own_id: str
    dataset_id: str
    dataset_name: str
    source_type: str
    source_url: str
    source_title: str
    retrieval_method: str
    http_status: str
    relevance_label: str
    dataset_specificity_hint: str
    has_sql_text: str
    notes: str
    shared_with_own_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase B source-discovery outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in TOKEN_STOPWORDS
    ]


def dataset_tokens(dataset_name: str, dataset_id: str) -> list[str]:
    tokens: list[str] = []
    for token in tokenize(dataset_name) + tokenize(dataset_id.split("/", 1)[-1]):
        if token not in tokens:
            tokens.append(token)
    return tokens


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def expand_rows(rows: list[dict[str, str]]) -> list[ExpandedRow]:
    expanded: list[ExpandedRow] = []
    for row in rows:
        own_ids = [token.strip() for token in (row.get("own_id") or "").split(";") if token.strip()]
        dataset_ids = [token.strip() for token in (row.get("dataset_id") or "").split(";") if token.strip()]
        dataset_names = [token.strip() for token in (row.get("dataset_name") or "").split(";") if token.strip()]
        if not own_ids:
            continue
        if len(dataset_ids) == 1 and len(own_ids) > 1:
            dataset_ids = dataset_ids * len(own_ids)
        if len(dataset_names) == 1 and len(own_ids) > 1:
            dataset_names = dataset_names * len(own_ids)
        for index, own_id in enumerate(own_ids):
            expanded.append(
                ExpandedRow(
                    own_id=own_id,
                    dataset_id=dataset_ids[index] if index < len(dataset_ids) else dataset_ids[-1],
                    dataset_name=dataset_names[index] if index < len(dataset_names) else dataset_names[-1],
                    source_type=row.get("source_type") or "",
                    source_url=row.get("source_url") or "",
                    source_title=row.get("source_title") or "",
                    retrieval_method=row.get("retrieval_method") or "",
                    http_status=row.get("http_status") or "",
                    relevance_label=row.get("relevance_label") or "",
                    dataset_specificity_hint=row.get("dataset_specificity_hint") or "",
                    has_sql_text=row.get("has_sql_text") or "",
                    notes=row.get("notes") or "",
                    shared_with_own_ids=[token for token in own_ids if token != own_id],
                )
            )
    return expanded


def compute_token_overlap(tokens: list[str], *texts: str) -> int:
    haystack = " ".join(texts).lower()
    return sum(1 for token in tokens if token in haystack)


def likely_name_collision(row: ExpandedRow, token_overlap: int) -> bool:
    lower_title = row.source_title.lower()
    lower_url = row.source_url.lower()
    lower_notes = row.notes.lower()

    if row.dataset_specificity_hint == "collision_risk":
        return True
    if row.shared_with_own_ids:
        return True
    if any(pattern in lower_title or pattern in lower_url or pattern in lower_notes for pattern in GENERIC_COLLISION_PATTERNS):
        return True
    if row.source_type in OFFICIAL_SOURCE_TYPES and row.dataset_specificity_hint == "strict":
        return False
    if row.source_type in {"github_repo", "github_file", "github_release", "gist", "kaggle_code_or_notebook"}:
        return token_overlap <= 1
    return False


def dataset_order(expanded_rows: list[ExpandedRow], output_dir: Path) -> list[str]:
    scope_path = output_dir.parent / "scope" / "high_datasets.csv"
    if scope_path.exists():
        with scope_path.open("r", encoding="utf-8", newline="") as handle:
            return [row["own_id"] for row in csv.DictReader(handle)]
    seen: list[str] = []
    for row in expanded_rows:
        if row.own_id not in seen:
            seen.append(row.own_id)
    return seen


def audit() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_rows(input_path)
    expanded_rows = expand_rows(source_rows)
    order = dataset_order(expanded_rows, output_dir)
    dataset_meta: dict[str, tuple[str, str]] = {}
    rows_by_dataset: dict[str, list[ExpandedRow]] = defaultdict(list)
    for row in expanded_rows:
        dataset_meta.setdefault(row.own_id, (row.dataset_id, row.dataset_name))
        rows_by_dataset[row.own_id].append(row)

    per_dataset_summary: dict[str, dict[str, object]] = {}
    audit_csv_rows: list[dict[str, object]] = []
    broken_rows: list[dict[str, str]] = []
    shared_url_rows: list[dict[str, object]] = []

    for row in source_rows:
        own_ids = [token.strip() for token in (row.get("own_id") or "").split(";") if token.strip()]
        if len(own_ids) > 1:
            shared_url_rows.append(
                {
                    "source_url": row.get("source_url") or "",
                    "own_ids": own_ids,
                    "dataset_names": [token.strip() for token in (row.get("dataset_name") or "").split(";") if token.strip()],
                    "source_type": row.get("source_type") or "",
                    "http_status": row.get("http_status") or "",
                }
            )

    for own_id in order:
        rows = rows_by_dataset.get(own_id, [])
        dataset_id, dataset_name = dataset_meta.get(own_id, ("", own_id))
        tokens = dataset_tokens(dataset_name, dataset_id)
        strict_count = sum(1 for row in rows if row.dataset_specificity_hint == "strict")
        weak_count = sum(1 for row in rows if row.dataset_specificity_hint == "weak")
        collision_count = sum(1 for row in rows if row.dataset_specificity_hint == "collision_risk")
        unknown_count = sum(1 for row in rows if row.dataset_specificity_hint == "unknown")
        reachable_rows = [row for row in rows if row.http_status in OK_STATUSES]
        reachable_strict_count = sum(1 for row in reachable_rows if row.dataset_specificity_hint == "strict")
        reachable_nonsearch_sqlish_count = sum(
            1
            for row in reachable_rows
            if row.has_sql_text in {"yes", "partial"} and row.source_type not in SEARCH_SOURCE_TYPES
        )
        broken_count = sum(
            1 for row in rows if row.http_status and row.http_status not in OK_STATUSES | TEMP_UNREACHABLE_STATUSES
        )
        temporary_unreachable_count = sum(1 for row in rows if row.http_status in TEMP_UNREACHABLE_STATUSES)
        only_collision_or_unknown_reachable = bool(reachable_rows) and all(
            row.dataset_specificity_hint in {"collision_risk", "unknown"} for row in reachable_rows
        )
        likely_collision_count = 0

        for row in rows:
            overlap = compute_token_overlap(tokens, row.source_title, row.source_url)
            is_reachable = row.http_status in OK_STATUSES
            is_temp_unreachable = row.http_status in TEMP_UNREACHABLE_STATUSES
            is_broken = bool(row.http_status) and row.http_status not in OK_STATUSES | TEMP_UNREACHABLE_STATUSES
            is_shared = bool(row.shared_with_own_ids)
            is_likely_collision = likely_name_collision(row, overlap)
            likely_collision_count += int(is_likely_collision)

            issue_tags: list[str] = []
            if is_broken:
                issue_tags.append("broken_or_unreachable")
            elif is_temp_unreachable:
                issue_tags.append("temporary_unreachable")
            if is_shared:
                issue_tags.append("duplicate_url_across_datasets")
            if is_likely_collision:
                issue_tags.append("likely_name_collision")
            if row.dataset_specificity_hint in {"collision_risk", "unknown"}:
                issue_tags.append("low_specificity")

            audit_csv_rows.append(
                {
                    "own_id": own_id,
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "source_type": row.source_type,
                    "source_url": row.source_url,
                    "source_title": row.source_title,
                    "http_status": row.http_status,
                    "is_reachable": "yes" if is_reachable else "no",
                    "is_broken_or_unreachable": "yes" if is_broken else "no",
                    "is_temporary_unreachable": "yes" if is_temp_unreachable else "no",
                    "is_cross_dataset_duplicate": "yes" if is_shared else "no",
                    "shared_with_own_ids": " ; ".join(row.shared_with_own_ids),
                    "token_overlap_count": overlap,
                    "likely_name_collision": "yes" if is_likely_collision else "no",
                    "dataset_specificity_hint": row.dataset_specificity_hint,
                    "relevance_label": row.relevance_label,
                    "has_sql_text": row.has_sql_text,
                    "audit_issue_tags": " ; ".join(issue_tags),
                    "dataset_strict_count": strict_count,
                    "dataset_weak_count": weak_count,
                    "dataset_collision_risk_count": collision_count,
                    "dataset_unknown_count": unknown_count,
                    "dataset_reachable_strict_count": reachable_strict_count,
                    "dataset_reachable_nonsearch_sqlish_count": reachable_nonsearch_sqlish_count,
                    "dataset_broken_count": broken_count,
                    "dataset_temporary_unreachable_count": temporary_unreachable_count,
                    "dataset_likely_name_collision_count": 0,  # populated later
                    "dataset_only_collision_or_unknown": "yes" if only_collision_or_unknown_reachable else "no",
                    "dataset_readiness_tier": "",
                }
            )

            if is_broken or is_temp_unreachable:
                broken_rows.append(
                    {
                        "own_id": own_id,
                        "dataset_name": dataset_name,
                        "source_type": row.source_type,
                        "http_status": row.http_status,
                        "source_url": row.source_url,
                        "source_title": row.source_title,
                    }
                )

        if only_collision_or_unknown_reachable or reachable_strict_count == 0 or reachable_nonsearch_sqlish_count == 0:
            readiness_tier = "NEEDS_MORE_SOURCES"
            readiness_reason = "Missing a usable strict core or a reachable non-search SQL-capable source."
        elif broken_count == 0 and reachable_strict_count >= 3 and reachable_nonsearch_sqlish_count >= 2:
            readiness_tier = "READY_STRONG"
            readiness_reason = "Has a reachable strict source core plus reachable non-search SQL-capable sources."
        else:
            readiness_tier = "READY_WITH_WARNINGS"
            readiness_reason = "Usable for SQL extraction, but some sources are broken, rate-limited, weak, or collision-prone."

        per_dataset_summary[own_id] = {
            "own_id": own_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "total_sources": len(rows),
            "reachable_sources": len(reachable_rows),
            "strict": strict_count,
            "weak": weak_count,
            "collision_risk": collision_count,
            "unknown": unknown_count,
            "reachable_strict": reachable_strict_count,
            "reachable_nonsearch_sqlish": reachable_nonsearch_sqlish_count,
            "broken_count": broken_count,
            "temporary_unreachable_count": temporary_unreachable_count,
            "likely_name_collision_count": likely_collision_count,
            "only_collision_or_unknown": only_collision_or_unknown_reachable,
            "readiness_tier": readiness_tier,
            "readiness_reason": readiness_reason,
        }

    for row in audit_csv_rows:
        summary = per_dataset_summary[row["own_id"]]
        row["dataset_likely_name_collision_count"] = summary["likely_name_collision_count"]
        row["dataset_readiness_tier"] = summary["readiness_tier"]

    readiness_counter = Counter(summary["readiness_tier"] for summary in per_dataset_summary.values())
    overall_status = "PASS" if readiness_counter.get("NEEDS_MORE_SOURCES", 0) == 0 else "FAIL"
    flagged_only_collision_unknown = [
        summary["own_id"]
        for summary in per_dataset_summary.values()
        if summary["only_collision_or_unknown"]
    ]

    md_path = output_dir / "checkpoint_b_source_audit.md"
    csv_path = output_dir / "checkpoint_b_source_audit.csv"
    json_path = output_dir / "checkpoint_b_status.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(audit_csv_rows[0].keys()) if audit_csv_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_csv_rows)

    readiness_lines = [
        "| own_id | dataset_name | readiness_tier | strict | weak | collision_risk | unknown | broken | temp_unreachable | likely_name_collision | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for own_id in order:
        summary = per_dataset_summary[own_id]
        notes: list[str] = []
        if summary["broken_count"]:
            notes.append(f"{summary['broken_count']} broken")
        if summary["temporary_unreachable_count"]:
            notes.append(f"{summary['temporary_unreachable_count']} temp-unreachable")
        if summary["only_collision_or_unknown"]:
            notes.append("only collision/unknown reachable")
        if not notes:
            notes.append("clean enough to proceed")
        readiness_lines.append(
            "| "
            + " | ".join(
                [
                    own_id,
                    str(summary["dataset_name"]),
                    str(summary["readiness_tier"]),
                    str(summary["strict"]),
                    str(summary["weak"]),
                    str(summary["collision_risk"]),
                    str(summary["unknown"]),
                    str(summary["broken_count"]),
                    str(summary["temporary_unreachable_count"]),
                    str(summary["likely_name_collision_count"]),
                    ", ".join(notes),
                ]
            )
            + " |"
        )

    broken_lines = [
        "| own_id | source_type | http_status | source_title | source_url |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in broken_rows:
        broken_lines.append(
            "| "
            + " | ".join(
                [
                    item["own_id"],
                    item["source_type"],
                    item["http_status"],
                    item["source_title"].replace("|", "/"),
                    item["source_url"],
                ]
            )
            + " |"
        )

    shared_lines = [
        "| source_url | source_type | http_status | own_ids | dataset_names |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in shared_url_rows:
        shared_lines.append(
            "| "
            + " | ".join(
                [
                    item["source_url"],
                    item["source_type"],
                    item["http_status"],
                    ", ".join(item["own_ids"]),
                    ", ".join(item["dataset_names"]),
                ]
            )
            + " |"
        )

    collision_rows = [row for row in audit_csv_rows if row["likely_name_collision"] == "yes"]
    collision_lines = [
        "| own_id | source_type | http_status | source_title | source_url |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in collision_rows[:40]:
        collision_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["own_id"]),
                    str(row["source_type"]),
                    str(row["http_status"]),
                    str(row["source_title"]).replace("|", "/"),
                    str(row["source_url"]),
                ]
            )
            + " |"
        )

    md_content = "\n".join(
        [
            "# Checkpoint B Source Audit",
            "",
            f"- Input inventory: `{input_path}`",
            f"- Generated at UTC: `{utc_now_iso()}`",
            f"- Overall checkpoint status: `{overall_status}`",
            f"- Readiness tier counts: `READY_STRONG={readiness_counter.get('READY_STRONG', 0)}`, `READY_WITH_WARNINGS={readiness_counter.get('READY_WITH_WARNINGS', 0)}`, `NEEDS_MORE_SOURCES={readiness_counter.get('NEEDS_MORE_SOURCES', 0)}`",
            f"- Broken URLs (`404`/other non-OK): `{sum(1 for row in broken_rows if row['http_status'] not in TEMP_UNREACHABLE_STATUSES)}`",
            f"- Temporarily unreachable URLs (`429`): `{sum(1 for row in broken_rows if row['http_status'] in TEMP_UNREACHABLE_STATUSES)}`",
            f"- Duplicate URLs across datasets: `{len(shared_url_rows)}`",
            f"- Likely name-collision rows: `{len(collision_rows)}`",
            "",
            "## Readiness Table",
            "",
            *readiness_lines,
            "",
            "## Broken or Unreachable URLs",
            "",
            *broken_lines,
            "",
            "## Duplicate URLs Across Datasets",
            "",
            *shared_lines,
            "",
            "## Likely Name-Collision Sources",
            "",
            *collision_lines,
            "",
            "## Datasets With Only Collision-Risk or Unknown Reachable Sources",
            "",
            (
                "- None."
                if not flagged_only_collision_unknown
                else "- " + ", ".join(flagged_only_collision_unknown)
            ),
            "",
            "## Decision Rule",
            "",
            "- `PASS` means no dataset is in `NEEDS_MORE_SOURCES`, so SQL extraction can proceed for all datasets.",
            "- `READY_WITH_WARNINGS` still allows extraction, but review the flagged sources first if a dataset depends heavily on rate-limited or collision-prone links.",
            "",
        ]
    )
    md_path.write_text(md_content, encoding="utf-8")

    status_payload = {
        "checkpoint": "B",
        "status": overall_status,
        "generated_at_utc": utc_now_iso(),
        "input_inventory_csv": str(input_path),
        "outputs": {
            "checkpoint_b_source_audit_md": str(md_path),
            "checkpoint_b_source_audit_csv": str(csv_path),
            "checkpoint_b_status_json": str(json_path),
        },
        "summary": {
            "source_row_count": len(source_rows),
            "expanded_dataset_source_row_count": len(audit_csv_rows),
            "broken_or_unreachable_count": sum(
                1 for row in audit_csv_rows if row["is_broken_or_unreachable"] == "yes"
            ),
            "temporary_unreachable_count": sum(
                1 for row in audit_csv_rows if row["is_temporary_unreachable"] == "yes"
            ),
            "duplicate_url_across_dataset_count": len(shared_url_rows),
            "likely_name_collision_count": len(collision_rows),
            "readiness_tier_counts": dict(readiness_counter),
            "datasets_only_collision_or_unknown": flagged_only_collision_unknown,
        },
        "per_dataset": {own_id: per_dataset_summary[own_id] for own_id in order},
        "pass_rule": "PASS if and only if no dataset is NEEDS_MORE_SOURCES.",
    }
    json_path.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for own_id in order:
        summary = per_dataset_summary[own_id]
        print(
            f"{own_id}\t{summary['readiness_tier']}\t"
            f"strict={summary['strict']}\tweak={summary['weak']}\t"
            f"collision_risk={summary['collision_risk']}\tunknown={summary['unknown']}\t"
            f"broken={summary['broken_count']}\ttemp_unreachable={summary['temporary_unreachable_count']}"
        )
    print(f"CHECKPOINT B {overall_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(audit())
