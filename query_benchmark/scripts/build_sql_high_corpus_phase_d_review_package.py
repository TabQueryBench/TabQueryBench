#!/usr/bin/env python3
"""Package final review-ready Phase D artifacts for sql_high datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_phase_c_sql_inventory import (  # noqa: E402
    SqlRow as AuditSqlRow,
    dataset_tokens,
    extract_table_tokens,
    leading_sql_candidate,
    normalize_url_root,
    row_is_potentially_misleading,
    row_issue_tags,
    sha256_file,
    utc_now_iso,
)


DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404")
DEFAULT_SCOPE_CSV = Path("logs/sql_high_corpus_build_20260404/scope/high_datasets.csv")
DEFAULT_MASTER_SQL_INVENTORY = Path("logs/sql_high_corpus_build_20260404/global/master_sql_inventory_all.csv")
DEFAULT_CHECKPOINT_C_AUDIT_CSV = Path("logs/sql_high_corpus_build_20260404/global/checkpoint_c_sql_audit.csv")
TOP_STRICT_LIMIT = 20
QUESTION_SEED_LIMIT = 10
FINAL_INDEX_FIELDNAMES = [
    "own_id",
    "dataset_id",
    "dataset_name",
    "class_type",
    "source_type",
    "checkpoint_c_status",
    "question_taxonomy_readiness",
    "strict_sql_count",
    "weak_sql_count",
    "collision_risk_sql_count",
    "trustworthy_sql_count",
    "top_strict_sql_count",
    "question_seed_count",
    "official_source_url",
    "best_sql_source_url",
    "dataset_card_path",
    "sql_inventory_path",
    "top_strict_sql_path",
    "question_seed_candidates_path",
    "provenance_log_path",
    "known_risk_tags",
]
TOP_STRICT_FIELDNAMES = [
    "selection_rank",
    "own_id",
    "dataset_id",
    "dataset_name",
    "source_sql_item_id",
    "source_url",
    "source_root",
    "source_type",
    "source_title",
    "source_file_path",
    "sql_complexity",
    "query_intent_label",
    "family_tag_guess",
    "evidence_confidence",
    "sql_text_raw",
    "sql_text_clean",
    "selection_notes",
]
QUESTION_SEED_FIELDNAMES = [
    "seed_id",
    "seed_question_text",
    "source_sql_item_id",
    "family_tag_guess",
    "confidence",
    "notes",
]
OFFICIAL_SOURCE_TYPE_PRIORITY = [
    "official_dataset_page",
    "official_api",
    "openml_api",
    "openml_task_page",
    "kaggle_overview_page",
    "kaggle_data_page",
    "readme_or_metadata",
    "paper",
]
SQL_SOURCE_TYPE_PRIORITY = [
    "github_file",
    "gist",
    "github_repo",
    "kaggle_code_or_notebook",
    "kaggle_code_page",
]
RISK_LABELS = {
    "no_sql_inventory_rows": "No explicit SQL inventory exists yet.",
    "over_reliance_on_weak_or_collision": "Evidence is dominated by weak or collision-risk SQL.",
    "no_strict_sql": "No strict SQL evidence survived the audit.",
    "high_misleading_share": "Many rows appear generic, wrong-schema, or non-SQL in context.",
    "duplicate_sql_text_clean_present": "The SQL inventory still contains duplicate cleaned SQL rows.",
    "insufficient_usable_sql_variety": "Too few distinct vetted SQL items remain for question design.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package review-ready per-dataset and global Phase D artifacts from the "
            "audited sql_high corpus workspace."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE_CSV)
    parser.add_argument("--master-sql-inventory", type=Path, default=DEFAULT_MASTER_SQL_INVENTORY)
    parser.add_argument("--checkpoint-c-audit-csv", type=Path, default=DEFAULT_CHECKPOINT_C_AUDIT_CSV)
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_scope_index(scope_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {(row.get("own_id") or "").strip(): row for row in scope_rows}


def build_sql_index(sql_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    rows_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sql_rows:
        rows_by_dataset[(row.get("own_id") or "").strip()].append(row)
    return rows_by_dataset


def build_audit_index(audit_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {(row.get("own_id") or "").strip(): row for row in audit_rows}


def audit_sql_row_from_dict(row: dict[str, str]) -> AuditSqlRow:
    return AuditSqlRow(
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


def source_specificity_counts(source_rows: list[dict[str, str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in source_rows:
        counter[(row.get("dataset_specificity_hint") or "").strip() or "unknown"] += 1
    return counter


def official_links(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    priority = {name: index for index, name in enumerate(OFFICIAL_SOURCE_TYPE_PRIORITY)}
    candidates = [
        row
        for row in source_rows
        if (row.get("source_type") or "").strip() in set(OFFICIAL_SOURCE_TYPE_PRIORITY)
    ]
    candidates.sort(
        key=lambda row: (
            priority.get((row.get("source_type") or "").strip(), 999),
            (row.get("source_url") or "").strip(),
        )
    )
    selected: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for row in candidates:
        url = (row.get("source_url") or "").strip()
        if not url or url in seen_urls:
            continue
        selected.append(row)
        seen_urls.add(url)
        if len(selected) >= 3:
            break
    return selected


def best_available_sql_links(
    *,
    sql_rows: list[dict[str, str]],
    top_strict_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if top_strict_rows:
        selected: list[dict[str, str]] = []
        seen_roots: set[str] = set()
        for row in top_strict_rows:
            root = normalize_url_root(row["source_url"])
            if root in seen_roots:
                continue
            selected.append(
                {
                    "label": row["source_sql_item_id"],
                    "url": row["source_url"],
                    "title": row["source_title"],
                    "note": "curated strict SQL source",
                }
            )
            seen_roots.add(root)
            if len(selected) >= 3:
                break
        return selected

    specificity_rank = {"strict": 0, "weak": 1, "collision_risk": 2}
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    priority = {name: index for index, name in enumerate(SQL_SOURCE_TYPE_PRIORITY)}
    candidates = sorted(
        sql_rows,
        key=lambda row: (
            specificity_rank.get((row.get("dataset_specificity_label") or "").strip(), 9),
            confidence_rank.get((row.get("evidence_confidence") or "").strip(), 9),
            priority.get((row.get("source_type") or "").strip(), 9),
            normalize_url_root(row.get("source_url") or ""),
            row.get("sql_item_id") or "",
        ),
    )
    selected: list[dict[str, str]] = []
    seen_roots: set[str] = set()
    for row in candidates:
        url = (row.get("source_url") or "").strip()
        if not url:
            continue
        root = normalize_url_root(url)
        if root in seen_roots:
            continue
        selected.append(
            {
                "label": (row.get("sql_item_id") or "").strip(),
                "url": url,
                "title": (row.get("source_title") or "").strip(),
                "note": "best available SQL source (not curated strict)",
            }
        )
        seen_roots.add(root)
        if len(selected) >= 3:
            break
    return selected


def candidate_score(row: dict[str, str]) -> int:
    confidence_score = {"high": 4, "medium": 2, "low": 0}
    complexity_score = {"moderate": 3, "advanced": 2, "simple": 1}
    family_score = {
        "aggregation": 4,
        "join_analysis": 4,
        "window_analytics": 4,
        "filtering": 3,
        "basic_projection": 2,
        "schema_ddl": 2,
        "misc_sql": 1,
        "etl_load": 0,
    }
    query_score = {
        "dml_select": 4,
        "ddl_create_table": 2,
        "ddl_create_view": 2,
        "dml_update": 2,
        "dml_delete": 1,
        "dml_insert": 0,
        "session_use": -3,
        "other_sql": 0,
    }
    length = len((row.get("sql_text_clean") or "").strip())
    length_bonus = 0
    if 40 <= length <= 2000:
        length_bonus = 2
    elif 2001 <= length <= 8000:
        length_bonus = 1
    elif length > 15000:
        length_bonus = -2
    return (
        confidence_score.get((row.get("evidence_confidence") or "").strip(), 0)
        + complexity_score.get((row.get("sql_complexity") or "").strip(), 0)
        + family_score.get((row.get("family_tag_guess") or "").strip(), 0)
        + query_score.get((row.get("query_intent_label") or "").strip(), 0)
        + length_bonus
    )


def curated_top_strict_rows(
    *,
    dataset_row: dict[str, str],
    audit_row: dict[str, str],
    sql_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if (audit_row.get("dataset_status") or "").strip() != "PASS":
        return []

    ds_tokens = dataset_tokens(
        dataset_row.get("dataset_name") or "",
        dataset_row.get("dataset_id") or "",
    )
    best_by_clean: dict[str, dict[str, Any]] = {}
    for row in sql_rows:
        if (row.get("dataset_specificity_label") or "").strip() != "strict":
            continue
        if (row.get("evidence_confidence") or "").strip() not in {"high", "medium"}:
            continue
        audit_sql_row = audit_sql_row_from_dict(row)
        issue_tags = row_issue_tags(audit_sql_row, ds_tokens)
        if row_is_potentially_misleading(issue_tags):
            continue
        if (row.get("is_near_duplicate") or "").strip() == "yes":
            continue
        clean_sql = (row.get("sql_text_clean") or "").strip()
        if not clean_sql:
            continue
        candidate = {
            **row,
            "_issue_tags": issue_tags,
            "_source_root": normalize_url_root((row.get("source_url") or "").strip()),
            "_candidate_score": candidate_score(row),
        }
        current = best_by_clean.get(clean_sql)
        if current is None or candidate["_candidate_score"] > current["_candidate_score"]:
            best_by_clean[clean_sql] = candidate

    candidates = sorted(
        best_by_clean.values(),
        key=lambda row: (
            -int(row["_candidate_score"]),
            (row.get("query_intent_label") or "").strip(),
            (row.get("family_tag_guess") or "").strip(),
            (row.get("sql_item_id") or "").strip(),
        ),
    )

    selected: list[dict[str, Any]] = []
    source_root_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for row in candidates:
        if len(selected) >= TOP_STRICT_LIMIT:
            break
        root = row["_source_root"]
        family = (row.get("family_tag_guess") or "").strip() or "unknown"
        if source_root_counts[root] >= 5:
            continue
        if family_counts[family] >= 4 and len(candidates) > TOP_STRICT_LIMIT:
            continue
        selected.append(row)
        source_root_counts[root] += 1
        family_counts[family] += 1

    if len(selected) < TOP_STRICT_LIMIT:
        selected_ids = {row.get("sql_item_id") for row in selected}
        for row in candidates:
            if len(selected) >= TOP_STRICT_LIMIT:
                break
            if row.get("sql_item_id") in selected_ids:
                continue
            root = row["_source_root"]
            if source_root_counts[root] >= 7:
                continue
            selected.append(row)
            selected_ids.add(row.get("sql_item_id"))
            source_root_counts[root] += 1

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        output_rows.append(
            {
                "selection_rank": index,
                "own_id": row.get("own_id") or "",
                "dataset_id": row.get("dataset_id") or "",
                "dataset_name": row.get("dataset_name") or "",
                "source_sql_item_id": row.get("sql_item_id") or "",
                "source_url": row.get("source_url") or "",
                "source_root": row["_source_root"],
                "source_type": row.get("source_type") or "",
                "source_title": row.get("source_title") or "",
                "source_file_path": row.get("source_file_path") or "",
                "sql_complexity": row.get("sql_complexity") or "",
                "query_intent_label": row.get("query_intent_label") or "",
                "family_tag_guess": row.get("family_tag_guess") or "",
                "evidence_confidence": row.get("evidence_confidence") or "",
                "sql_text_raw": row.get("sql_text_raw") or "",
                "sql_text_clean": row.get("sql_text_clean") or "",
                "selection_notes": (
                    "Phase D curated strict SQL. "
                    f"checkpoint_c_status={audit_row.get('dataset_status') or ''}; "
                    f"candidate_score={row['_candidate_score']}; "
                    "row_issue_tags=none"
                ),
            }
        )
    return output_rows


def clean_comment_text(prefix: str) -> str:
    text = prefix or ""
    text = re.sub(r"(?s)^/\*+|\*/$", "", text).strip()
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        stripped = re.sub(r"^(--+|#+|\*+)\s*", "", stripped)
        stripped = normalize_whitespace(stripped)
        if not stripped:
            continue
        cleaned_lines.append(stripped)
    comment = normalize_whitespace(" ".join(cleaned_lines))
    comment = re.sub(r"^[0-9]+[.)]\s*", "", comment)
    return comment


def leading_comment_question(raw_sql: str) -> str:
    candidate = raw_sql or ""
    keyword_match = re.search(
        r"(?i)\b(with|select|insert\s+into|update|delete\s+from|create\s+"
        r"(or\s+replace\s+)?(table|view|database|schema|function|procedure|index|trigger)"
        r"|drop\s+(table|database|view|schema)|alter\s+table|truncate\s+table|use\s+\w+)\b",
        candidate,
    )
    prefix = candidate[: keyword_match.start()] if keyword_match else ""
    comment = clean_comment_text(prefix)
    if not comment:
        return ""
    if any(token in comment.lower() for token in ("phpmyadmin sql dump", "step 1:", "connected to a transient")):
        return ""
    if len(comment) < 12:
        return ""
    comment = comment.rstrip(".")
    if "?" not in comment and comment[0].islower():
        comment = comment[0].upper() + comment[1:]
    if "?" not in comment and comment.lower().startswith(("find ", "calculate ", "count ", "check ", "show ", "create ", "insert ", "update ", "load ")):
        comment = comment + "."
    elif "?" not in comment:
        comment = comment + "?"
    return comment


def group_by_columns(sql_text: str) -> str:
    match = re.search(
        r"(?is)\bgroup\s+by\b\s+(.*?)(?:\border\s+by\b|\bhaving\b|\blimit\b|\)\s*(?:select|as|where|from)\b|;|$)",
        sql_text or "",
    )
    if not match:
        return ""
    group_text = normalize_whitespace(match.group(1))
    group_text = group_text.strip(",")
    lower_group_text = group_text.lower()
    for marker in (" ) select", ") select", ") as", ") where", ") from", " select ", " where "):
        marker_index = lower_group_text.find(marker.strip())
        if marker_index != -1:
            group_text = group_text[:marker_index].strip(" ,)")
            lower_group_text = group_text.lower()
    parts = [normalize_whitespace(part) for part in group_text.split(",") if normalize_whitespace(part)]
    if len(parts) > 3:
        parts = parts[:3]
    group_text = ", ".join(parts)
    return group_text[:80]


def aggregate_functions(sql_text: str) -> list[str]:
    functions = []
    for func in ("count", "avg", "sum", "min", "max", "rank", "dense_rank", "row_number"):
        if re.search(rf"(?i)\b{func}\s*\(", sql_text or ""):
            functions.append(func.upper())
    return functions


def heuristic_seed_text(row: dict[str, Any]) -> str:
    query_intent = (row.get("query_intent_label") or "").strip()
    family = (row.get("family_tag_guess") or "").strip()
    sql_text = row.get("sql_text_raw") or ""
    tables = extract_table_tokens(sql_text)
    primary_table = f"`{tables[0]}`" if tables else "the benchmark table"
    table_list = ", ".join(f"`{table}`" for table in tables[:3]) if tables else "the relevant tables"
    aggs = aggregate_functions(sql_text)
    group_cols = group_by_columns(sql_text)
    has_where = bool(re.search(r"(?i)\bwhere\b", sql_text))

    if query_intent == "ddl_create_table" and tables:
        return f"Create the {primary_table} table used by this benchmark SQL."
    if query_intent == "ddl_create_view" and tables:
        return f"Create the view or derived table built from {table_list} in the source SQL."
    if query_intent == "dml_select":
        if family == "aggregation" or aggs:
            if len(aggs) == 1 and aggs[0] == "COUNT" and not group_cols:
                return f"Count rows in {primary_table}."
            if len(aggs) == 2 and set(aggs) == {"MIN", "MAX"} and not group_cols:
                return f"Compute MIN and MAX for {primary_table}."
            agg_clause = ", ".join(aggs) if aggs else "the requested aggregates"
            text = f"On {primary_table}, compute {agg_clause}"
            if group_cols:
                text += f" grouped by `{group_cols}`"
            if has_where:
                text += " with the source filters applied"
            return text + "."
        if family == "join_analysis" and len(tables) >= 2:
            return f"Join {table_list} and reproduce the analytical result expressed in the source SQL."
        if family == "filtering":
            return f"Filter {primary_table} with the source conditions and return the requested columns."
        return f"Query {primary_table} to reproduce the result pattern in the source SQL."
    if query_intent == "dml_update" and tables:
        return f"Update {primary_table} according to the transformation logic in the source SQL."
    if query_intent == "dml_delete" and tables:
        return f"Delete rows from {primary_table} using the source SQL conditions."
    if query_intent == "dml_insert" and tables and len(sql_text) <= 3000:
        return f"Insert the source rows into {primary_table}."
    if query_intent.startswith("ddl_") and tables:
        return f"Recreate the schema object for {primary_table} shown in the source SQL."
    if query_intent == "other_sql" and family == "join_analysis" and len(tables) >= 2:
        return f"Join {table_list} using the logic shown in the source SQL."
    if query_intent == "other_sql" and aggs:
        agg_clause = ", ".join(aggs)
        return f"Compute {agg_clause} over {primary_table} following the source SQL logic."
    if tables:
        return f"Reproduce the SQL logic from `{row.get('source_sql_item_id')}` using {table_list}."
    return f"Reproduce the SQL logic from `{row.get('source_sql_item_id')}`."


def seed_candidate_score(row: dict[str, Any]) -> int:
    query_intent = (row.get("query_intent_label") or "").strip()
    family = (row.get("family_tag_guess") or "").strip()
    score = 0
    if query_intent == "dml_select":
        score += 4
    elif query_intent == "ddl_create_table":
        score += 2
    elif query_intent == "dml_update":
        score += 2
    elif query_intent == "dml_insert":
        score += 1
    elif query_intent == "session_use":
        score -= 5
    if family in {"aggregation", "join_analysis", "window_analytics"}:
        score += 3
    elif family in {"filtering", "basic_projection", "schema_ddl"}:
        score += 2
    if (row.get("evidence_confidence") or "").strip() == "high":
        score += 2
    if len((row.get("sql_text_raw") or "")) > 15000:
        score -= 3
    return score


def question_seeds_from_top_strict(top_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    candidates = sorted(top_rows, key=lambda row: (-seed_candidate_score(row), int(row["selection_rank"])))
    seeds: list[dict[str, str]] = []
    for row in candidates:
        if len(seeds) >= QUESTION_SEED_LIMIT:
            break
        if (row.get("query_intent_label") or "").strip() == "session_use":
            continue
        if (row.get("query_intent_label") or "").strip() == "dml_insert" and len(row.get("sql_text_raw") or "") > 3000:
            continue
        question_text = leading_comment_question(row.get("sql_text_raw") or "")
        note_parts = ["Derived from curated strict SQL."]
        if question_text:
            note_parts.append("Seed text sourced from leading SQL comments.")
        else:
            question_text = heuristic_seed_text(row)
            note_parts.append("Seed text generated from conservative SQL heuristics.")
        question_text = normalize_whitespace(question_text)
        if not question_text:
            continue
        if len(question_text) > 240:
            question_text = question_text[:237].rstrip() + "..."
        seed_index = len(seeds) + 1
        seeds.append(
            {
                "seed_id": f"{row.get('own_id')}_seed_{seed_index:03d}",
                "seed_question_text": question_text,
                "source_sql_item_id": row.get("source_sql_item_id") or "",
                "family_tag_guess": row.get("family_tag_guess") or "",
                "confidence": "high" if (row.get("evidence_confidence") or "").strip() == "high" else "medium",
                "notes": " ".join(note_parts),
            }
        )
    return seeds


def dataset_purpose_text(scope_row: dict[str, str]) -> str:
    class_type = (scope_row.get("class_type") or "").strip().lower()
    source_type = (scope_row.get("source_type") or "").strip()
    if class_type == "categorical":
        task_phrase = "categorical-label tabular analysis and classification-style SQL reasoning"
    elif class_type == "numerical":
        task_phrase = "numeric/regression-oriented tabular analysis and metric computation"
    elif class_type == "mixed":
        task_phrase = "mixed-feature tabular analysis across categorical and numeric fields"
    else:
        task_phrase = "tabular SQL analysis"
    return (
        f"This `sql_high` dataset is packaged as a {task_phrase} benchmark candidate "
        f"from `{source_type}` for corpus review, question-taxonomy design, and downstream SQL evidence assessment."
    )


def downstream_richness_text(
    *,
    dataset_name: str,
    audit_row: dict[str, str],
    top_strict_count: int,
    seed_count: int,
) -> str:
    usable_unique = int(audit_row.get("usable_unique_sql_count") or 0)
    dataset_status = (audit_row.get("dataset_status") or "").strip()
    if top_strict_count >= 10 and seed_count >= 5:
        return (
            f"`{dataset_name}` looks downstream-task-rich in the packaged review set: "
            f"{top_strict_count} curated strict SQL rows survived the audit and yielded {seed_count} preliminary seed questions."
        )
    if top_strict_count > 0:
        return (
            f"`{dataset_name}` has some downstream-task coverage, but it is narrower than ideal: "
            f"only {top_strict_count} curated strict SQL rows survived packaging."
        )
    if dataset_status == "PASS" and usable_unique >= 5:
        return (
            f"`{dataset_name}` shows potential downstream-task richness in weaker or non-promoted SQL, "
            "but no review-ready strict SQL survived, so no strict seed set was packaged."
        )
    return (
        f"`{dataset_name}` is not downstream-task-rich in the current package because no vetted strict SQL subset "
        "was available for review-ready question seeding."
    )


def risk_lines(audit_row: dict[str, str], top_strict_count: int) -> list[str]:
    risks: list[str] = []
    risk_tags = [tag.strip() for tag in (audit_row.get("primary_risk_tags") or "").split(";") if tag.strip()]
    for tag in risk_tags:
        risks.append(RISK_LABELS.get(tag, tag.replace("_", " ")))
    if top_strict_count == 0:
        risks.append("No curated strict SQL was packaged into `top_strict_sql.csv`.")
    if (audit_row.get("dataset_status") or "").strip() == "FAIL":
        risks.append("Checkpoint C marked the dataset as not ready for question taxonomy without remediation.")
    return risks or ["No major review-blocking risk was recorded beyond minor duplicates."]


def build_dataset_card(
    *,
    dataset_dir: Path,
    scope_row: dict[str, str],
    audit_row: dict[str, str],
    source_rows: list[dict[str, str]],
    top_strict_rows: list[dict[str, Any]],
    question_seeds: list[dict[str, str]],
) -> str:
    source_counts = source_specificity_counts(source_rows)
    official = official_links(source_rows)
    best_sql = best_available_sql_links(sql_rows=read_csv_rows(dataset_dir / "sql" / "sql_inventory.csv"), top_strict_rows=top_strict_rows)
    strict_sql_count = int(audit_row.get("strict_sql_count") or 0)
    weak_sql_count = int(audit_row.get("weak_sql_count") or 0)
    collision_sql_count = int(audit_row.get("collision_risk_sql_count") or 0)
    trustworthy_sql_count = int(audit_row.get("trustworthy_sql_count") or 0)
    duplicate_sql_count = int(audit_row.get("duplicate_sql_text_clean_count") or 0)
    misleading_sql_count = int(audit_row.get("potentially_misleading_row_count") or 0)
    top_strict_count = len(top_strict_rows)
    dataset_status = (audit_row.get("dataset_status") or "").strip()
    readiness = (audit_row.get("readiness_for_question_taxonomy") or "").strip()

    lines = [
        f"# Dataset Card: {scope_row.get('dataset_name') or ''} (`{scope_row.get('own_id') or ''}`)",
        "",
        "## What This Dataset Is For",
        "",
        dataset_purpose_text(scope_row),
        "",
        "## Why Downstream-Task-Rich Or Not",
        "",
        downstream_richness_text(
            dataset_name=scope_row.get("dataset_name") or "",
            audit_row=audit_row,
            top_strict_count=top_strict_count,
            seed_count=len(question_seeds),
        ),
        "",
        "## SQL Evidence Quality Summary",
        "",
        f"- Checkpoint C dataset status: `{dataset_status}`",
        f"- Question taxonomy readiness: `{readiness}`",
        f"- Total SQL rows in Phase C inventory: {audit_row.get('total_sql_rows') or '0'}",
        f"- Trustworthy strict SQL rows from audit: {trustworthy_sql_count}",
        f"- Curated top strict SQL rows packaged in Phase D: {top_strict_count}",
        f"- Preliminary question seed count: {len(question_seeds)}",
        f"- Duplicate `sql_text_clean` rows flagged in audit: {duplicate_sql_count}",
        f"- Potentially misleading SQL rows flagged in audit: {misleading_sql_count}",
        "",
        "## Strict Vs Weak Source Breakdown",
        "",
        f"- Phase B source hints: `strict={source_counts.get('strict', 0)}`, `weak={source_counts.get('weak', 0)}`, `collision_risk={source_counts.get('collision_risk', 0)}`, `unknown={source_counts.get('unknown', 0)}`",
        f"- Phase C SQL labels: `strict={strict_sql_count}`, `weak={weak_sql_count}`, `collision_risk={collision_sql_count}`",
        "",
        "## Key Links",
        "",
    ]

    if official:
        lines.append("- Official / context links:")
        for row in official:
            label = normalize_whitespace(row.get("source_type") or "official")
            title = normalize_whitespace(row.get("source_title") or row.get("source_url") or "")
            lines.append(f"  - `{label}`: [{title}]({row.get('source_url') or ''})")
    else:
        lines.append("- Official / context links: none recorded.")

    if best_sql:
        lines.append("- Best SQL links:")
        for row in best_sql:
            title = normalize_whitespace(row.get("title") or row.get("url") or "")
            note = normalize_whitespace(row.get("note") or "")
            lines.append(f"  - [{title}]({row.get('url') or ''}) ({note})")
    else:
        lines.append("- Best SQL links: none packaged.")

    lines.extend(["", "## Known Risks", ""])
    for risk in risk_lines(audit_row, top_strict_count):
        lines.append(f"- {risk}")
    lines.append("")
    return "\n".join(lines)


def build_provenance_log(
    *,
    dataset_dir: Path,
    scope_row: dict[str, str],
    audit_row: dict[str, str],
    source_inventory_path: Path,
    sql_inventory_path: Path,
    top_strict_rows: list[dict[str, Any]],
    question_seeds: list[dict[str, str]],
    generated_paths: list[Path],
) -> list[dict[str, Any]]:
    log_rows: list[dict[str, Any]] = []
    log_rows.append(
        {
            "event": "dataset_context",
            "generated_at_utc": utc_now_iso(),
            "own_id": scope_row.get("own_id") or "",
            "dataset_id": scope_row.get("dataset_id") or "",
            "dataset_name": scope_row.get("dataset_name") or "",
            "class_type": scope_row.get("class_type") or "",
            "source_type": scope_row.get("source_type") or "",
            "phase_d_packaging_basis": "checkpoint_c_audit + existing phase_c_sql_inventory",
        }
    )
    for path in (source_inventory_path, sql_inventory_path):
        row_count = len(read_csv_rows(path))
        log_rows.append(
            {
                "event": "input_inventory",
                "generated_at_utc": utc_now_iso(),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "row_count": row_count,
            }
        )
    log_rows.append(
        {
            "event": "checkpoint_c_summary",
            "generated_at_utc": utc_now_iso(),
            "dataset_status": audit_row.get("dataset_status") or "",
            "question_taxonomy_readiness": audit_row.get("readiness_for_question_taxonomy") or "",
            "strict_sql_count": audit_row.get("strict_sql_count") or "0",
            "trustworthy_sql_count": audit_row.get("trustworthy_sql_count") or "0",
            "usable_unique_sql_count": audit_row.get("usable_unique_sql_count") or "0",
            "primary_risk_tags": audit_row.get("primary_risk_tags") or "",
        }
    )
    for row in top_strict_rows:
        log_rows.append(
            {
                "event": "top_strict_sql_selected",
                "generated_at_utc": utc_now_iso(),
                "selection_rank": row.get("selection_rank") or 0,
                "source_sql_item_id": row.get("source_sql_item_id") or "",
                "source_url": row.get("source_url") or "",
                "source_root": row.get("source_root") or "",
                "family_tag_guess": row.get("family_tag_guess") or "",
                "evidence_confidence": row.get("evidence_confidence") or "",
                "selection_notes": row.get("selection_notes") or "",
            }
        )
    for row in question_seeds:
        log_rows.append(
            {
                "event": "question_seed_generated",
                "generated_at_utc": utc_now_iso(),
                "seed_id": row.get("seed_id") or "",
                "source_sql_item_id": row.get("source_sql_item_id") or "",
                "confidence": row.get("confidence") or "",
                "notes": row.get("notes") or "",
            }
        )
    for path in generated_paths:
        log_rows.append(
            {
                "event": "output_written",
                "generated_at_utc": utc_now_iso(),
                "path": str(path.resolve()),
            }
        )
    return log_rows


def build_final_overview(
    *,
    final_index_rows: list[dict[str, Any]],
    run_manifest_path: Path,
) -> str:
    ready_rows = [row for row in final_index_rows if row["question_taxonomy_readiness"] == "yes"]
    packaged_rows = [row for row in final_index_rows if int(row["top_strict_sql_count"]) > 0]
    seeded_rows = [row for row in final_index_rows if int(row["question_seed_count"]) > 0]
    lines = [
        "# Final Review Package Overview",
        "",
        f"- Generated at UTC: `{utc_now_iso()}`",
        f"- Dataset count: {len(final_index_rows)}",
        f"- Datasets marked taxonomy-ready at Checkpoint C: {len(ready_rows)}",
        f"- Datasets with non-empty `top_strict_sql.csv`: {len(packaged_rows)}",
        f"- Datasets with non-empty `question_seed_candidates.csv`: {len(seeded_rows)}",
        f"- Run manifest: `{run_manifest_path.resolve()}`",
        "",
        "## Per-Dataset Package Summary",
        "",
        "| own_id | dataset_name | checkpoint_c_status | readiness | strict_sql_count | top_strict_sql_count | question_seed_count |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in final_index_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["own_id"],
                    row["dataset_name"],
                    row["checkpoint_c_status"],
                    row["question_taxonomy_readiness"],
                    str(row["strict_sql_count"]),
                    str(row["top_strict_sql_count"]),
                    str(row["question_seed_count"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    scope_csv = args.scope_csv.resolve()
    master_sql_inventory = args.master_sql_inventory.resolve()
    checkpoint_c_audit_csv = args.checkpoint_c_audit_csv.resolve()
    final_dir = output_root / "final"
    script_path = Path(__file__).resolve()

    scope_rows = read_csv_rows(scope_csv)
    scope_index = build_scope_index(scope_rows)
    sql_rows = read_csv_rows(master_sql_inventory)
    sql_by_dataset = build_sql_index(sql_rows)
    audit_rows = read_csv_rows(checkpoint_c_audit_csv)
    audit_index = build_audit_index(audit_rows)

    final_index_rows: list[dict[str, Any]] = []
    per_dataset_manifest: list[dict[str, Any]] = []

    for scope_row in scope_rows:
        own_id = (scope_row.get("own_id") or "").strip()
        dataset_dir = output_root / "datasets" / own_id
        sql_dir = dataset_dir / "sql"
        questions_dir = dataset_dir / "questions"
        audit_dir = dataset_dir / "audit"
        source_inventory_path = dataset_dir / "sources" / "source_inventory.csv"
        sql_inventory_path = sql_dir / "sql_inventory.csv"
        top_strict_path = sql_dir / "top_strict_sql.csv"
        question_seed_path = questions_dir / "question_seed_candidates.csv"
        provenance_path = audit_dir / "provenance_log.jsonl"
        dataset_card_path = dataset_dir / "dataset_card.md"

        source_rows = read_csv_rows(source_inventory_path)
        dataset_sql_rows = sql_by_dataset.get(own_id, [])
        audit_row = audit_index.get(own_id)
        if audit_row is None:
            audit_row = {
                "own_id": own_id,
                "dataset_id": scope_row.get("dataset_id") or "",
                "dataset_name": scope_row.get("dataset_name") or "",
                "total_sql_rows": str(len(dataset_sql_rows)),
                "strict_sql_count": "0",
                "weak_sql_count": "0",
                "collision_risk_sql_count": "0",
                "trustworthy_sql_count": "0",
                "usable_unique_sql_count": "0",
                "readiness_for_question_taxonomy": "no",
                "dataset_status": "FAIL",
                "primary_risk_tags": "no_sql_inventory_rows",
                "remediation_actions": "Checkpoint C audit row missing; regenerate audit artifacts first.",
            }

        # Reuse the existing Phase C SQL inventory unchanged; packaging adds curation files beside it.
        top_strict_rows = curated_top_strict_rows(
            dataset_row=scope_row,
            audit_row=audit_row,
            sql_rows=dataset_sql_rows,
        )
        question_seeds = question_seeds_from_top_strict(top_strict_rows)

        write_csv(top_strict_path, TOP_STRICT_FIELDNAMES, top_strict_rows)
        write_csv(question_seed_path, QUESTION_SEED_FIELDNAMES, question_seeds)

        dataset_card_text = build_dataset_card(
            dataset_dir=dataset_dir,
            scope_row=scope_row,
            audit_row=audit_row,
            source_rows=source_rows,
            top_strict_rows=top_strict_rows,
            question_seeds=question_seeds,
        )
        dataset_card_path.write_text(dataset_card_text, encoding="utf-8")

        provenance_rows = build_provenance_log(
            dataset_dir=dataset_dir,
            scope_row=scope_row,
            audit_row=audit_row,
            source_inventory_path=source_inventory_path,
            sql_inventory_path=sql_inventory_path,
            top_strict_rows=top_strict_rows,
            question_seeds=question_seeds,
            generated_paths=[dataset_card_path, top_strict_path, question_seed_path, provenance_path],
        )
        write_jsonl(provenance_path, provenance_rows)

        official = official_links(source_rows)
        best_sql = best_available_sql_links(sql_rows=dataset_sql_rows, top_strict_rows=top_strict_rows)
        final_index_rows.append(
            {
                "own_id": own_id,
                "dataset_id": scope_row.get("dataset_id") or "",
                "dataset_name": scope_row.get("dataset_name") or "",
                "class_type": scope_row.get("class_type") or "",
                "source_type": scope_row.get("source_type") or "",
                "checkpoint_c_status": audit_row.get("dataset_status") or "",
                "question_taxonomy_readiness": audit_row.get("readiness_for_question_taxonomy") or "",
                "strict_sql_count": audit_row.get("strict_sql_count") or "0",
                "weak_sql_count": audit_row.get("weak_sql_count") or "0",
                "collision_risk_sql_count": audit_row.get("collision_risk_sql_count") or "0",
                "trustworthy_sql_count": audit_row.get("trustworthy_sql_count") or "0",
                "top_strict_sql_count": len(top_strict_rows),
                "question_seed_count": len(question_seeds),
                "official_source_url": official[0]["source_url"] if official else "",
                "best_sql_source_url": best_sql[0]["url"] if best_sql else "",
                "dataset_card_path": str(dataset_card_path.resolve()),
                "sql_inventory_path": str(sql_inventory_path.resolve()),
                "top_strict_sql_path": str(top_strict_path.resolve()),
                "question_seed_candidates_path": str(question_seed_path.resolve()),
                "provenance_log_path": str(provenance_path.resolve()),
                "known_risk_tags": audit_row.get("primary_risk_tags") or "",
            }
        )
        per_dataset_manifest.append(
            {
                "own_id": own_id,
                "dataset_id": scope_row.get("dataset_id") or "",
                "dataset_name": scope_row.get("dataset_name") or "",
                "checkpoint_c_status": audit_row.get("dataset_status") or "",
                "question_taxonomy_readiness": audit_row.get("readiness_for_question_taxonomy") or "",
                "strict_sql_count": int(audit_row.get("strict_sql_count") or 0),
                "top_strict_sql_count": len(top_strict_rows),
                "question_seed_count": len(question_seeds),
                "dataset_card_path": str(dataset_card_path.resolve()),
                "top_strict_sql_path": str(top_strict_path.resolve()),
                "question_seed_candidates_path": str(question_seed_path.resolve()),
                "provenance_log_path": str(provenance_path.resolve()),
            }
        )

    final_index_path = final_dir / "final_index.csv"
    final_overview_path = final_dir / "final_overview.md"
    run_manifest_path = final_dir / "run_manifest_phase_d.json"

    write_csv(final_index_path, FINAL_INDEX_FIELDNAMES, final_index_rows)
    final_overview_path.write_text(
        build_final_overview(final_index_rows=final_index_rows, run_manifest_path=run_manifest_path),
        encoding="utf-8",
    )

    manifest = {
        "phase": "D",
        "phase_name": "review_ready_artifact_packaging",
        "generated_at_utc": utc_now_iso(),
        "script_path": str(script_path),
        "rerun_command": (
            f"python3 {script_path} --output-root {output_root} --scope-csv {scope_csv} "
            f"--master-sql-inventory {master_sql_inventory} --checkpoint-c-audit-csv {checkpoint_c_audit_csv}"
        ),
        "inputs": {
            "scope_csv_path": str(scope_csv),
            "scope_csv_sha256": sha256_file(scope_csv),
            "master_sql_inventory_path": str(master_sql_inventory),
            "master_sql_inventory_sha256": sha256_file(master_sql_inventory),
            "checkpoint_c_audit_csv_path": str(checkpoint_c_audit_csv),
            "checkpoint_c_audit_csv_sha256": sha256_file(checkpoint_c_audit_csv),
        },
        "outputs": {
            "final_index_csv": str(final_index_path.resolve()),
            "final_overview_md": str(final_overview_path.resolve()),
            "run_manifest_phase_d_json": str(run_manifest_path.resolve()),
        },
        "counts": {
            "dataset_count": len(final_index_rows),
            "datasets_with_top_strict_sql": sum(1 for row in final_index_rows if int(row["top_strict_sql_count"]) > 0),
            "datasets_with_question_seeds": sum(1 for row in final_index_rows if int(row["question_seed_count"]) > 0),
            "total_top_strict_sql_rows": sum(int(row["top_strict_sql_count"]) for row in final_index_rows),
            "total_question_seed_rows": sum(int(row["question_seed_count"]) for row in final_index_rows),
        },
        "per_dataset": per_dataset_manifest,
    }
    write_json(run_manifest_path, manifest)


if __name__ == "__main__":
    main()
