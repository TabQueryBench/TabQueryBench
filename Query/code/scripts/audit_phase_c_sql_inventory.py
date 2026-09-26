#!/usr/bin/env python3
"""Audit Phase C SQL inventory trustworthiness and taxonomy readiness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("logs/sql_high_corpus_build_20260404/global/master_sql_inventory_all.csv")
DEFAULT_OUTPUT_DIR = Path("logs/sql_high_corpus_build_20260404/global")
DEFAULT_SCOPE_CSV = Path("logs/sql_high_corpus_build_20260404/scope/high_datasets.csv")
TOKEN_STOPWORDS = {
    "and",
    "challenge",
    "classification",
    "data",
    "dataset",
    "datasets",
    "default",
    "for",
    "from",
    "high",
    "ii",
    "in",
    "kaggle",
    "of",
    "prediction",
    "risk",
    "sql",
    "the",
    "with",
}
SQL_START_PATTERN = re.compile(
    r"^\s*(with|select|insert\s+into|update|delete\s+from|create\s+"
    r"(or\s+replace\s+)?(table|view|database|schema|function|procedure|index|trigger)"
    r"|drop\s+(table|database|view|schema)|alter\s+table|truncate\s+table|use\s+\w+"
    r"|show\s+\w+|describe\s+\w+|explain\s+\w+|merge\s+into|call\s+\w+)",
    re.IGNORECASE | re.DOTALL,
)
GENERIC_SOURCE_PATTERNS = (
    "kaggle_intro_to_sql",
    "kaggle-course-answer",
    "course-answer",
    "course_answer",
    "course answers",
    "intro_to_sql",
    "intro to sql",
    "bootcamp",
    "tutorial",
)
NONSTANDARD_QUERY_PATTERNS = (
    "opql",
    "openpql",
    "parse_query",
    "boardsuitcount",
    "hero=",
    "villain=",
    "board=",
    "game='holdem'",
    'game="holdem"',
    "equity from hero",
)
CODE_FRAGMENT_PATTERNS = (
    "data.tr$",
    "data.table(",
    "selected features %s",
    "executescalar",
    "return a list with the same length as l",
    "train_test_split(",
    "use openpql_pql_parser::parse_query;",
)
APP_REPO_PATTERNS = (
    "-dbms",
    "service",
    "warehouse",
    "plantation",
    "cultivation",
    "trainer",
    "parser",
    "-system",
    "_system",
)
APP_SCHEMA_TOKENS = {
    "account",
    "accounts",
    "basket",
    "card_holder",
    "chapter",
    "credit_card",
    "cust_order",
    "customer",
    "customer_info",
    "employee",
    "experiment",
    "flush",
    "gatherer",
    "guild",
    "images",
    "merchant",
    "merchant_category",
    "olap",
    "oltp",
    "order",
    "order_details",
    "order_item",
    "order_items",
    "orders",
    "payment",
    "pet",
    "plant",
    "plants",
    "player",
    "search",
    "store",
    "transaction",
    "transactions",
    "user",
    "users",
}
FOREIGN_DATASET_PATTERNS = (
    "bigquery-public-data",
    "chicago_taxi_trips",
    "pet_records.pets",
    "stackoverflow.posts_questions",
    "chicago_public_schools",
    "chicago_crime_data",
    "sqlite_master",
)
CSV_FIELDNAMES = [
    "own_id",
    "dataset_id",
    "dataset_name",
    "total_sql_rows",
    "missing_source_url_count",
    "missing_sql_text_raw_count",
    "duplicate_sql_text_clean_count",
    "duplicate_sql_text_clean_groups",
    "strict_sql_count",
    "weak_sql_count",
    "collision_risk_sql_count",
    "trustworthy_sql_count",
    "confidence_high_count",
    "confidence_medium_count",
    "confidence_low_count",
    "potentially_misleading_row_count",
    "potentially_misleading_ratio",
    "over_reliance_on_weak_or_collision",
    "usable_high_medium_nonmisleading_count",
    "usable_unique_sql_count",
    "readiness_for_question_taxonomy",
    "dataset_status",
    "primary_risk_tags",
    "sample_flagged_sql_item_ids",
    "remediation_actions",
]


@dataclass(frozen=True)
class DatasetScopeRow:
    own_id: str
    dataset_id: str
    dataset_name: str


@dataclass(frozen=True)
class SqlRow:
    own_id: str
    dataset_id: str
    dataset_name: str
    sql_item_id: str
    source_url: str
    source_type: str
    source_title: str
    sql_text_raw: str
    sql_text_clean: str
    dataset_specificity_label: str
    evidence_confidence: str
    source_file_path: str
    extraction_method: str
    is_near_duplicate: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase C SQL inventory outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE_CSV)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in TOKEN_STOPWORDS
    ]


def dataset_tokens(dataset_name: str, dataset_id: str) -> list[str]:
    tokens: list[str] = []
    for token in tokenize(dataset_name) + tokenize(dataset_id.split(":", 1)[-1]):
        if token not in tokens:
            tokens.append(token)
    return tokens


def read_scope(scope_csv: Path) -> list[DatasetScopeRow]:
    with scope_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        DatasetScopeRow(
            own_id=(row.get("own_id") or "").strip(),
            dataset_id=(row.get("dataset_id") or "").strip(),
            dataset_name=(row.get("dataset_name") or "").strip(),
        )
        for row in rows
    ]


def read_sql_rows(path: Path) -> list[SqlRow]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        SqlRow(
            own_id=(row.get("own_id") or "").strip(),
            dataset_id=(row.get("dataset_id") or "").strip(),
            dataset_name=(row.get("dataset_name") or "").strip(),
            sql_item_id=(row.get("sql_item_id") or "").strip(),
            source_url=(row.get("source_url") or "").strip(),
            source_type=(row.get("source_type") or "").strip(),
            source_title=(row.get("source_title") or "").strip(),
            sql_text_raw=(row.get("sql_text_raw") or "").strip(),
            sql_text_clean=(row.get("sql_text_clean") or "").strip(),
            dataset_specificity_label=(row.get("dataset_specificity_label") or "").strip(),
            evidence_confidence=(row.get("evidence_confidence") or "").strip(),
            source_file_path=(row.get("source_file_path") or "").strip(),
            extraction_method=(row.get("extraction_method") or "").strip(),
            is_near_duplicate=(row.get("is_near_duplicate") or "").strip(),
        )
        for row in rows
    ]


def normalize_url_root(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if "github.com" not in parsed.netloc:
        return url
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2:
        return f"{parsed.scheme}://{parsed.netloc}/{parts[0]}/{parts[1]}"
    return url


def leading_sql_candidate(text: str) -> str:
    candidate = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    candidate = re.sub(r"^```[^\n]*\n?", "", candidate)
    candidate = re.sub(r"\n?```$", "", candidate)
    candidate = re.sub(r'^["\']{2,}', "", candidate)
    candidate = re.sub(r'^(query|sql)\s*=\s*[frubFRUB]*["\']{1,3}', "", candidate)
    candidate = re.sub(r"^let\s+\w+\s*=\s*[\"']", "", candidate)
    candidate = re.sub(r"^sql[\"',\s:]+", "", candidate)
    candidate = re.sub(r"^//.*?$", "", candidate, flags=re.MULTILINE)
    candidate = re.sub(r"^>.*?$", "", candidate, flags=re.MULTILINE)
    candidate = re.sub(r"(?s)^/\*.*?\*/", "", candidate)

    lines = candidate.splitlines()
    trimmed_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not trimmed_lines and (not stripped or stripped.startswith("#") or stripped.startswith("--")):
            continue
        trimmed_lines.append(line)
    candidate = "\n".join(trimmed_lines).strip()

    keyword_match = re.search(
        r"(?i)\b(with|select|insert\s+into|update|delete\s+from|create\s+"
        r"(or\s+replace\s+)?(table|view|database|schema|function|procedure|index|trigger)"
        r"|drop\s+(table|database|view|schema)|alter\s+table|truncate\s+table|use\s+\w+"
        r"|show\s+\w+|describe\s+\w+|explain\s+\w+|merge\s+into|call\s+\w+)\b",
        candidate,
    )
    if keyword_match and keyword_match.start() <= 120:
        candidate = candidate[keyword_match.start():].strip()
    return candidate


def standalone_sql_like(text: str) -> bool:
    candidate = leading_sql_candidate(text)
    if not candidate:
        return False
    if SQL_START_PATTERN.search(candidate):
        return True
    lowered = candidate.lower()
    keyword_hits = sum(
        1
        for pattern in (
            r"\bselect\b",
            r"\bfrom\b",
            r"\bwhere\b",
            r"\bgroup\s+by\b",
            r"\border\s+by\b",
            r"\bjoin\b",
            r"\bcreate\s+table\b",
            r"\binsert\s+into\b",
            r"\bupdate\b",
            r"\bdelete\s+from\b",
            r"\bwith\b",
        )
        if re.search(pattern, lowered)
    )
    return keyword_hits >= 3 and (";" in candidate or "\n" in candidate)


def extract_table_tokens(text: str) -> list[str]:
    tables = re.findall(
        r'(?i)\b(?:from|join|into|update|table)\s+[`"]?([a-zA-Z_][a-zA-Z0-9_]*)',
        text or "",
    )
    return [token.lower() for token in tables]


def row_issue_tags(row: SqlRow, ds_tokens: list[str]) -> list[str]:
    tags: list[str] = []
    context = " ".join(
        [
            normalize_url_root(row.source_url),
            row.source_title,
            row.source_file_path,
            row.sql_text_raw[:500],
        ]
    ).lower()
    root = normalize_url_root(row.source_url).lower()
    candidate = leading_sql_candidate(row.sql_text_raw)
    table_tokens = extract_table_tokens(row.sql_text_raw)
    context_overlap = sum(1 for token in ds_tokens if token in context)
    app_token_hits = len({token for token in table_tokens if token in APP_SCHEMA_TOKENS})

    if not row.source_url:
        tags.append("missing_source_url")
    if not row.sql_text_raw:
        tags.append("missing_sql_text_raw")
    if not standalone_sql_like(row.sql_text_raw):
        tags.append("non_sql_or_code_fragment")
    if any(pattern in context for pattern in GENERIC_SOURCE_PATTERNS):
        tags.append("generic_course_or_tutorial_source")
    if any(pattern in context for pattern in NONSTANDARD_QUERY_PATTERNS):
        tags.append("nonstandard_query_language")
    if any(pattern in context for pattern in CODE_FRAGMENT_PATTERNS):
        tags.append("code_fragment_context")
    if row.extraction_method in {"source_string_literal", "ipynb_string_literal"} and not standalone_sql_like(row.sql_text_raw):
        tags.append("string_literal_not_standalone_sql")
    if any(pattern in root for pattern in APP_REPO_PATTERNS):
        tags.append("application_repo_context")
    if app_token_hits >= 2:
        tags.append("application_schema_context")
    if any(pattern in context for pattern in FOREIGN_DATASET_PATTERNS) and context_overlap < 2:
        tags.append("foreign_dataset_context")
    if row.dataset_specificity_label == "collision_risk":
        tags.append("collision_risk_specificity")
    elif row.dataset_specificity_label == "weak":
        tags.append("weak_specificity")
    if row.evidence_confidence == "low":
        tags.append("low_confidence")
    if row.is_near_duplicate == "yes":
        tags.append("near_duplicate_flagged")
    if row.sql_text_raw and len(row.sql_text_raw) > 50_000 and row.sql_text_raw.lower().lstrip().startswith("insert into"):
        tags.append("bulk_insert_usability_risk")

    # Schema-mismatch override for broad-domain but wrong-database rows.
    if (
        context_overlap >= 1
        and app_token_hits >= 2
        and row.dataset_specificity_label == "strict"
    ):
        tags.append("strict_label_likely_overstated")

    # Rows that are weak but still clearly tied to the benchmark name should not be treated as misleading.
    if context_overlap >= 2:
        tags = [tag for tag in tags if tag not in {"weak_specificity", "collision_risk_specificity"}]
    return sorted(set(tags))


def row_is_potentially_misleading(issue_tags: list[str]) -> bool:
    major_tags = {
        "missing_source_url",
        "missing_sql_text_raw",
        "non_sql_or_code_fragment",
        "generic_course_or_tutorial_source",
        "nonstandard_query_language",
        "code_fragment_context",
        "string_literal_not_standalone_sql",
        "application_repo_context",
        "application_schema_context",
        "foreign_dataset_context",
        "strict_label_likely_overstated",
    }
    return any(tag in major_tags for tag in issue_tags)


def primary_risks_for_dataset(
    *,
    total_rows: int,
    strict_sql_count: int,
    weak_sql_count: int,
    collision_sql_count: int,
    potentially_misleading_count: int,
    missing_source_url_count: int,
    missing_sql_text_raw_count: int,
    duplicate_clean_count: int,
    usable_unique_sql_count: int,
) -> list[str]:
    risks: list[str] = []
    if total_rows == 0:
        risks.append("no_sql_inventory_rows")
    if missing_source_url_count:
        risks.append("missing_source_url")
    if missing_sql_text_raw_count:
        risks.append("missing_sql_text_raw")
    low_specificity_count = weak_sql_count + collision_sql_count
    if total_rows and low_specificity_count / total_rows > 0.75:
        risks.append("over_reliance_on_weak_or_collision")
    if strict_sql_count == 0 and total_rows > 0:
        risks.append("no_strict_sql")
    if total_rows and potentially_misleading_count / total_rows > 0.25:
        risks.append("high_misleading_share")
    if duplicate_clean_count > 0:
        risks.append("duplicate_sql_text_clean_present")
    if usable_unique_sql_count < 5:
        risks.append("insufficient_usable_sql_variety")
    return risks


def remediation_actions_for_dataset(risk_tags: list[str], example_roots: list[str]) -> str:
    actions: list[str] = []
    if "no_sql_inventory_rows" in risk_tags:
        actions.append(
            "Collect at least one exact dataset-specific GitHub/file/notebook source with explicit SQL, then rerun Phase B and Phase C for this dataset."
        )
    if "missing_source_url" in risk_tags or "missing_sql_text_raw" in risk_tags:
        actions.append(
            "Repair the affected inventory rows so every SQL item retains a source link and non-empty raw SQL text, then rebuild the global master inventory."
        )
    if "over_reliance_on_weak_or_collision" in risk_tags:
        actions.append(
            "Add exact benchmark-matched SQL sources or relabel the current rows after manual verification so the dataset is not dominated by weak/collision evidence."
        )
    if "high_misleading_share" in risk_tags:
        actions.append(
            "Filter out misleading rows from generic course material, wrong-schema application databases, or non-SQL fragments before using this dataset for taxonomy design."
        )
    if "duplicate_sql_text_clean_present" in risk_tags:
        actions.append(
            "Collapse duplicate `sql_text_clean` rows within the dataset or demote them to alternate-source evidence so the inventory reflects distinct SQL items."
        )
    if "insufficient_usable_sql_variety" in risk_tags:
        actions.append(
            "Collect more distinct SQL tasks so the dataset has at least a small set of unique, reusable question-taxonomy candidates."
        )
    if "no_strict_sql" in risk_tags:
        actions.append(
            "Manual-review the weak rows and either promote truly benchmark-specific SQL or replace them with stricter benchmark-aligned sources."
        )
    if not actions:
        actions.append("No blocking remediation required.")
    if example_roots:
        actions.append("Audit the highest-risk source roots first: " + ", ".join(example_roots[:3]) + ".")
    return " ".join(actions)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def audit() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    scope_csv = args.scope_csv.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sql_rows = read_sql_rows(input_path)
    scope_rows = read_scope(scope_csv)
    dataset_order = [row.own_id for row in scope_rows]
    scope_meta = {row.own_id: row for row in scope_rows}
    rows_by_dataset: dict[str, list[SqlRow]] = defaultdict(list)
    for row in sql_rows:
        rows_by_dataset[row.own_id].append(row)

    audit_csv_rows: list[dict[str, Any]] = []
    dataset_status_rows: list[dict[str, Any]] = []
    fail_datasets: list[dict[str, Any]] = []
    ranked_rows: list[dict[str, Any]] = []
    total_missing_source_url_count = 0
    total_missing_sql_text_raw_count = 0
    total_potentially_misleading_row_count = 0
    total_duplicate_clean_count = 0

    for own_id in dataset_order:
        meta = scope_meta[own_id]
        rows = rows_by_dataset.get(own_id, [])
        ds_tokens = dataset_tokens(meta.dataset_name, meta.dataset_id)
        clean_counter = Counter(row.sql_text_clean for row in rows if row.sql_text_clean)
        duplicate_clean_count = sum(count - 1 for count in clean_counter.values() if count > 1)
        duplicate_clean_groups = sum(1 for count in clean_counter.values() if count > 1)

        issue_tags_by_row: dict[str, list[str]] = {}
        potentially_misleading_rows: list[SqlRow] = []
        usable_rows: list[SqlRow] = []
        risk_roots: Counter[str] = Counter()
        confidence_counter = Counter(row.evidence_confidence for row in rows)
        specificity_counter = Counter(row.dataset_specificity_label for row in rows)

        for row in rows:
            tags = row_issue_tags(row, ds_tokens)
            issue_tags_by_row[row.sql_item_id] = tags
            if row_is_potentially_misleading(tags):
                potentially_misleading_rows.append(row)
                risk_roots[normalize_url_root(row.source_url)] += 1
            else:
                if row.evidence_confidence in {"high", "medium"} and row.source_url and row.sql_text_raw:
                    usable_rows.append(row)

        total_rows = len(rows)
        missing_source_url_count = sum(1 for row in rows if not row.source_url)
        missing_sql_text_raw_count = sum(1 for row in rows if not row.sql_text_raw)
        strict_sql_count = specificity_counter.get("strict", 0)
        weak_sql_count = specificity_counter.get("weak", 0)
        collision_sql_count = specificity_counter.get("collision_risk", 0)
        trustworthy_sql_count = sum(
            1
            for row in rows
            if row.dataset_specificity_label == "strict"
            and row.evidence_confidence in {"high", "medium"}
        )
        potentially_misleading_count = len(potentially_misleading_rows)
        usable_unique_sql_count = len({row.sql_text_clean for row in usable_rows if row.sql_text_clean})
        low_specificity_count = weak_sql_count + collision_sql_count
        over_reliance_flag = (
            "yes"
            if total_rows > 0 and (low_specificity_count / total_rows > 0.75 or strict_sql_count == 0)
            else "no"
        )

        risk_tags = primary_risks_for_dataset(
            total_rows=total_rows,
            strict_sql_count=strict_sql_count,
            weak_sql_count=weak_sql_count,
            collision_sql_count=collision_sql_count,
            potentially_misleading_count=potentially_misleading_count,
            missing_source_url_count=missing_source_url_count,
            missing_sql_text_raw_count=missing_sql_text_raw_count,
            duplicate_clean_count=duplicate_clean_count,
            usable_unique_sql_count=usable_unique_sql_count,
        )

        readiness = (
            "yes"
            if (
                total_rows > 0
                and missing_source_url_count == 0
                and missing_sql_text_raw_count == 0
                and usable_unique_sql_count >= 5
                and potentially_misleading_count <= max(2, int(total_rows * 0.25))
            )
            else "no"
        )
        dataset_status = "PASS" if readiness == "yes" else "FAIL"
        sample_flagged_ids = [
            row.sql_item_id
            for row in potentially_misleading_rows[:6]
        ]
        example_roots = [root for root, _count in risk_roots.most_common(5)]
        remediation_actions = remediation_actions_for_dataset(risk_tags, example_roots)

        row_payload = {
            "own_id": own_id,
            "dataset_id": meta.dataset_id,
            "dataset_name": meta.dataset_name,
            "total_sql_rows": total_rows,
            "missing_source_url_count": missing_source_url_count,
            "missing_sql_text_raw_count": missing_sql_text_raw_count,
            "duplicate_sql_text_clean_count": duplicate_clean_count,
            "duplicate_sql_text_clean_groups": duplicate_clean_groups,
            "strict_sql_count": strict_sql_count,
            "weak_sql_count": weak_sql_count,
            "collision_risk_sql_count": collision_sql_count,
            "trustworthy_sql_count": trustworthy_sql_count,
            "confidence_high_count": confidence_counter.get("high", 0),
            "confidence_medium_count": confidence_counter.get("medium", 0),
            "confidence_low_count": confidence_counter.get("low", 0),
            "potentially_misleading_row_count": potentially_misleading_count,
            "potentially_misleading_ratio": f"{(potentially_misleading_count / total_rows):.3f}" if total_rows else "0.000",
            "over_reliance_on_weak_or_collision": over_reliance_flag,
            "usable_high_medium_nonmisleading_count": len(usable_rows),
            "usable_unique_sql_count": usable_unique_sql_count,
            "readiness_for_question_taxonomy": readiness,
            "dataset_status": dataset_status,
            "primary_risk_tags": "; ".join(risk_tags),
            "sample_flagged_sql_item_ids": "; ".join(sample_flagged_ids),
            "remediation_actions": remediation_actions,
        }
        audit_csv_rows.append(row_payload)
        ranked_rows.append(row_payload)
        dataset_status_rows.append(
            {
                "own_id": own_id,
                "dataset_id": meta.dataset_id,
                "dataset_name": meta.dataset_name,
                "status": dataset_status,
                "readiness_for_question_taxonomy": readiness,
                "trustworthy_sql_count": trustworthy_sql_count,
                "usable_unique_sql_count": usable_unique_sql_count,
                "primary_risk_tags": risk_tags,
                "remediation_actions": remediation_actions,
            }
        )
        if dataset_status == "FAIL":
            fail_datasets.append(dataset_status_rows[-1])

        total_missing_source_url_count += missing_source_url_count
        total_missing_sql_text_raw_count += missing_sql_text_raw_count
        total_potentially_misleading_row_count += potentially_misleading_count
        total_duplicate_clean_count += duplicate_clean_count

    ranked_rows.sort(
        key=lambda row: (
            -int(row["trustworthy_sql_count"]),
            -int(row["usable_unique_sql_count"]),
            row["own_id"],
        )
    )

    overall_status = "PASS" if not fail_datasets else "FAIL"

    ranked_table = build_markdown_table(
        [
            "rank",
            "own_id",
            "dataset_name",
            "trustworthy_sql_count",
            "strict",
            "weak",
            "collision_risk",
            "usable_unique",
            "readiness",
        ],
        [
            [
                index,
                row["own_id"],
                row["dataset_name"],
                row["trustworthy_sql_count"],
                row["strict_sql_count"],
                row["weak_sql_count"],
                row["collision_risk_sql_count"],
                row["usable_unique_sql_count"],
                row["readiness_for_question_taxonomy"],
            ]
            for index, row in enumerate(ranked_rows, start=1)
        ],
    )

    fail_lines: list[str] = []
    for entry in fail_datasets:
        fail_lines.append(
            f"- `{entry['own_id']}` - {entry['dataset_name']}: "
            f"{entry['remediation_actions']}"
        )

    md_lines = [
        "# Checkpoint C SQL Audit",
        "",
        f"- Input: `{input_path}`",
        f"- Scope reference: `{scope_csv}`",
        f"- Generated at UTC: `{utc_now_iso()}`",
        f"- Overall status: **{overall_status}**",
        "",
        "## Global Findings",
        "",
        f"- Datasets in scope: {len(scope_rows)}",
        f"- Datasets represented in master SQL inventory: {len(rows_by_dataset)}",
        f"- Missing `source_url` rows: {total_missing_source_url_count}",
        f"- Missing `sql_text_raw` rows: {total_missing_sql_text_raw_count}",
        f"- Duplicate `sql_text_clean` rows within datasets: {total_duplicate_clean_count}",
        f"- Potentially misleading rows: {total_potentially_misleading_row_count}",
        f"- Datasets ready for taxonomy: {len(scope_rows) - len(fail_datasets)}",
        f"- Datasets not ready for taxonomy: {len(fail_datasets)}",
        "",
        "## Audit Rule Summary",
        "",
        "- `trustworthy_sql_count` = rows with `dataset_specificity_label == strict` and `evidence_confidence in {high, medium}`.",
        "- `over_reliance_on_weak_or_collision` = more than 75% of rows are weak/collision, or the dataset has no strict rows.",
        "- `potentially_misleading_row_count` is driven by generic course sources, wrong-schema application databases, non-SQL/code fragments, foreign dataset context, and overstated strict labels.",
        "- `readiness_for_question_taxonomy = yes` only when the dataset has non-empty SQL coverage, no missing critical fields, at least 5 usable non-misleading unique SQL items, and a low misleading-row share.",
        "",
        "## Ranked Table by trustworthy_sql_count",
        "",
        ranked_table,
        "",
        "## Failing Datasets and Remediation",
        "",
    ]
    if fail_lines:
        md_lines.extend(fail_lines)
    else:
        md_lines.append("- No failing datasets.")
    md_lines.append("")

    audit_md_path = output_dir / "checkpoint_c_sql_audit.md"
    audit_csv_path = output_dir / "checkpoint_c_sql_audit.csv"
    status_json_path = output_dir / "checkpoint_c_status.json"

    audit_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    write_csv(audit_csv_path, CSV_FIELDNAMES, audit_csv_rows)

    status_payload = {
        "checkpoint": "C",
        "phase_name": "sql_inventory_trustworthiness_and_usability_audit",
        "generated_at_utc": utc_now_iso(),
        "input": {
            "master_sql_inventory_path": str(input_path),
            "master_sql_inventory_sha256": sha256_file(input_path),
            "master_sql_inventory_row_count": len(sql_rows),
            "scope_csv_path": str(scope_csv),
            "scope_csv_sha256": sha256_file(scope_csv),
            "scope_dataset_count": len(scope_rows),
        },
        "overall_status": overall_status,
        "pass_definition": (
            "PASS requires every scoped dataset to be ready for question taxonomy under "
            "the audit readiness rule."
        ),
        "global_summary": {
            "missing_source_url_count": total_missing_source_url_count,
            "missing_sql_text_raw_count": total_missing_sql_text_raw_count,
            "duplicate_sql_text_clean_count": total_duplicate_clean_count,
            "potentially_misleading_row_count": total_potentially_misleading_row_count,
            "ready_dataset_count": len(scope_rows) - len(fail_datasets),
            "fail_dataset_count": len(fail_datasets),
        },
        "dataset_results": dataset_status_rows,
        "fail_datasets": fail_datasets,
    }
    status_json_path.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(audit())
