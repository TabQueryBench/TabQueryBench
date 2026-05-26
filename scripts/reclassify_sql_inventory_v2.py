#!/usr/bin/env python3
"""Reclassify baseline SQL inventory rows using the V2 cleaning rulebook."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_phase_c_sql_inventory import (  # noqa: E402
    APP_SCHEMA_TOKENS,
    CODE_FRAGMENT_PATTERNS,
    FOREIGN_DATASET_PATTERNS,
    GENERIC_SOURCE_PATTERNS,
    NONSTANDARD_QUERY_PATTERNS,
    dataset_tokens as build_dataset_tokens,
    extract_table_tokens,
    leading_sql_candidate,
    normalize_url_root,
    standalone_sql_like,
)


DEFAULT_MASTER_SQL = Path("logs/sql_high_corpus_build_20260404/global/master_sql_inventory_all.csv")
DEFAULT_RULES_JSON = Path("logs/sql_high_corpus_build_20260404/v2_refinement/rules/v2_cleaning_rules.json")
DEFAULT_SOURCE_INVENTORY = Path("logs/sql_high_corpus_build_20260404/global/all_source_inventory.csv")
DEFAULT_FINAL_INDEX = Path("logs/sql_high_corpus_build_20260404/final/final_index.csv")
DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404/v2_refinement")

NEW_FIELDNAMES = [
    "v2_specificity_label",
    "v2_specificity_reason_code",
    "v2_specificity_reason_text",
    "v2_dataset_match_signals",
    "v2_source_credibility_tier",
    "v2_keep_candidate",
]

SQL_KEYWORDS = {
    "add",
    "all",
    "alter",
    "and",
    "as",
    "asc",
    "between",
    "by",
    "case",
    "cast",
    "check",
    "column",
    "constraint",
    "count",
    "create",
    "cross",
    "cte",
    "current",
    "date",
    "day",
    "database",
    "default",
    "delete",
    "desc",
    "describe",
    "distinct",
    "drop",
    "else",
    "end",
    "except",
    "exists",
    "explain",
    "false",
    "foreign",
    "from",
    "full",
    "function",
    "go",
    "grant",
    "group",
    "having",
    "if",
    "in",
    "index",
    "inner",
    "insert",
    "int",
    "into",
    "is",
    "join",
    "key",
    "left",
    "like",
    "limit",
    "max",
    "merge",
    "min",
    "month",
    "natural",
    "not",
    "null",
    "on",
    "option",
    "or",
    "order",
    "outer",
    "over",
    "partition",
    "primary",
    "procedure",
    "references",
    "replace",
    "restrict",
    "return",
    "right",
    "row",
    "schema",
    "select",
    "set",
    "show",
    "sum",
    "table",
    "then",
    "to",
    "top",
    "trigger",
    "true",
    "truncate",
    "union",
    "unique",
    "unknown",
    "update",
    "use",
    "using",
    "values",
    "view",
    "when",
    "where",
    "with",
    "year",
}
GENERIC_IDENTIFIER_TOKENS = SQL_KEYWORDS | {
    "dbo",
    "etl",
    "fact",
    "id",
    "idx",
    "iso",
    "main",
    "mysql",
    "olap",
    "oltp",
    "pk",
    "public",
    "ref",
    "sql",
    "sqlite",
    "sys",
    "tmp",
    "varchar",
}
TIER_PRIORITY = {
    "tier_1_official": 1,
    "tier_2_primary_code": 2,
    "tier_3_secondary_explanatory": 3,
    "tier_4_low_trust": 4,
}
SPECIFICITY_PRIORITY = {
    "strict": 1,
    "weak": 2,
    "collision_risk": 3,
    "reject_non_sql": 4,
}
CONFIDENCE_PRIORITY = {
    "high": 1,
    "medium": 2,
    "low": 3,
    "": 4,
}
EXECUTABLE_PRIORITY = {
    "pass": 1,
    "unknown": 2,
    "fail": 3,
    "": 4,
}
SOURCE_TYPES_TIER_1 = {
    "official_dataset_page",
    "official_api",
    "openml_api",
    "openml_task_page",
    "kaggle_overview_page",
    "kaggle_data_page",
}
SQL_ERROR_PATTERNS = (
    "syntax error",
    "unknown column",
    "unknown table",
    "relation does not exist",
    "no such table",
    "sqlstate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reclassify the baseline sql_high master inventory using the V2 rulebook "
            "without executing SQL."
        )
    )
    parser.add_argument("--master-sql", type=Path, default=DEFAULT_MASTER_SQL)
    parser.add_argument("--rules-json", type=Path, default=DEFAULT_RULES_JSON)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--final-index", type=Path, default=DEFAULT_FINAL_INDEX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def normalize_sql_for_duplicate(text: str) -> str:
    candidate = normalize_newlines(text or "").lower().strip()
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = re.sub(r";+\s*$", "", candidate)
    candidate = candidate.replace("``", "`").replace('""', '"')
    return candidate


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def csv_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_index(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {(row.get(key) or "").strip(): row for row in rows}


def source_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        own_id = (row.get("own_id") or "").strip()
        url = (row.get("source_url") or "").strip()
        if own_id and url and (own_id, url) not in index:
            index[(own_id, url)] = row
    return index


def parse_domain(url: str) -> str:
    parsed = urllib.parse.urlsplit(url or "")
    return parsed.netloc.lower()


def dataset_id_local_part(dataset_id: str) -> str:
    return (dataset_id or "").split(":", 1)[-1].strip().lower()


def context_text(row: dict[str, str]) -> str:
    parts = [
        normalize_url_root((row.get("source_url") or "").strip()),
        (row.get("source_title") or "").strip(),
        (row.get("source_file_path") or "").strip(),
        (row.get("sql_text_raw") or "")[:500],
        (row.get("extraction_notes") or "").strip(),
        (row.get("source_seed_url") or "").strip(),
        (row.get("source_seed_title") or "").strip(),
    ]
    return " ".join(part for part in parts if part).lower()


def extract_identifier_tokens(sql_text: str) -> list[str]:
    candidate = leading_sql_candidate(sql_text or "")
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", candidate)
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in GENERIC_IDENTIFIER_TOKENS:
            continue
        if lowered.isdigit():
            continue
        if len(lowered) < 2:
            continue
        if lowered not in seen:
            seen.add(lowered)
            normalized.append(lowered)
    return normalized


def extract_create_table_column_tokens(sql_text: str) -> list[str]:
    candidate = leading_sql_candidate(sql_text or "")
    if not re.search(r"(?i)\bcreate\s+table\b", candidate):
        return []
    lines = normalize_newlines(candidate).splitlines()
    tokens: list[str] = []
    seen: set[str] = set()
    inside_columns = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "(" in stripped and re.search(r"(?i)\bcreate\s+table\b", stripped):
            inside_columns = True
            continue
        if inside_columns and stripped.startswith(")"):
            break
        if not inside_columns:
            continue
        token_match = re.match(r'[`"]?([A-Za-z_][A-Za-z0-9_]*)', stripped)
        if not token_match:
            continue
        token = token_match.group(1).lower()
        if token in {
            "constraint",
            "primary",
            "foreign",
            "unique",
            "check",
            "key",
        }:
            continue
        if token in GENERIC_IDENTIFIER_TOKENS:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def evaluate_condition(value_map: dict[str, Any], condition: dict[str, Any]) -> bool:
    field = condition["field"]
    op = condition["op"]
    expected = condition["value"]
    actual = value_map.get(field)
    if op == "eq":
        return actual == expected
    if op == "not_eq":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "contains_any":
        actual_text = str(actual or "").lower()
        return any(str(item).lower() in actual_text for item in expected)
    if op == "gte":
        return float(actual or 0) >= float(expected)
    if op == "gt":
        return float(actual or 0) > float(expected)
    if op == "lt":
        return float(actual or 0) < float(expected)
    if op == "lte":
        return float(actual or 0) <= float(expected)
    raise ValueError(f"Unsupported operation: {op}")


def matches_all(value_map: dict[str, Any], conditions: list[dict[str, Any]]) -> bool:
    return all(evaluate_condition(value_map, condition) for condition in conditions)


def assign_source_tier(value_map: dict[str, Any], rules: dict[str, Any]) -> tuple[str, str]:
    for tier in sorted(rules["source_credibility_tiers"], key=lambda item: item["priority"]):
        for tier_rule in tier["assign_if_any"]:
            if matches_all(value_map, tier_rule["all"]):
                return tier["tier_id"], tier_rule["id"]
    fallback = rules["source_credibility_tiers"][-1]
    return fallback["tier_id"], fallback["default_reason_code"]


def first_matching_rule(
    value_map: dict[str, Any],
    rules: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for rule in sorted(rules, key=lambda item: item["priority"]):
        if matches_all(value_map, rule["all"]):
            return rule
    return None


def benchmark_roots_for_dataset(
    own_id: str,
    final_index_row: dict[str, str],
    source_rows: list[dict[str, str]],
) -> set[str]:
    roots: set[str] = set()
    for field in ("official_source_url", "best_sql_source_url"):
        url = (final_index_row.get(field) or "").strip()
        if url:
            roots.add(normalize_url_root(url))
    for row in source_rows:
        if (row.get("own_id") or "").strip() != own_id:
            continue
        url = (row.get("source_url") or "").strip()
        if not url:
            continue
        if (row.get("http_status") or "").strip() != "200":
            continue
        source_type = (row.get("source_type") or "").strip()
        specificity_hint = (row.get("dataset_specificity_hint") or "").strip()
        if source_type in SOURCE_TYPES_TIER_1 or specificity_hint == "strict":
            roots.add(normalize_url_root(url))
    return roots


def build_source_rows_by_dataset(source_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        own_id = (row.get("own_id") or "").strip()
        if own_id:
            grouped[own_id].append(row)
    return grouped


def detect_dataset_id_exact_match(dataset_id: str, row: dict[str, str], value_map: dict[str, Any]) -> bool:
    dataset_id_full = (dataset_id or "").strip().lower()
    dataset_local = dataset_id_local_part(dataset_id)
    context = value_map["context_text"]
    source_url = (row.get("source_url") or "").strip().lower()
    if dataset_id_full and dataset_id_full in context:
        return True
    if dataset_local and len(dataset_local) >= 5 and dataset_local in context:
        return True
    if dataset_local.startswith("competitions/") and dataset_local in source_url:
        return True
    if dataset_local.startswith("datasets/") and dataset_local in source_url:
        return True
    if dataset_id_full.startswith("uci:"):
        suffix = dataset_id_full.split(":", 1)[1]
        if f"dataset/{suffix}" in source_url or f"id={suffix}" in source_url:
            return True
    if dataset_id_full.startswith("openml:"):
        suffix = dataset_id_full.split(":", 1)[1]
        if f"id={suffix}" in source_url or f"/{suffix}" in source_url:
            return True
    return False


def dataset_match_quality(
    dataset_token_overlap_count: int,
    dataset_id_exact_match: bool,
    ambiguous_single_token_match: bool,
) -> str:
    if dataset_id_exact_match:
        return "exact_id"
    if dataset_token_overlap_count >= 2:
        return "multi_token"
    if ambiguous_single_token_match:
        return "ambiguous_single_token"
    if dataset_token_overlap_count == 1:
        return "single_token"
    return "none"


def benchmark_task_context_match(
    *,
    dataset_id_exact_match: bool,
    source_root_known: bool,
    dataset_token_overlap_count: int,
    source_hint: str,
    project_domain_mismatch: bool,
) -> str:
    if project_domain_mismatch:
        return "none"
    if dataset_id_exact_match:
        return "strong"
    if source_root_known and dataset_token_overlap_count >= 1:
        return "strong"
    if dataset_token_overlap_count >= 2:
        return "moderate"
    if source_hint == "strict" and dataset_token_overlap_count >= 1:
        return "moderate"
    if dataset_token_overlap_count == 1 or source_hint == "weak":
        return "weak"
    return "none"


def candidate_for_attribute_lexicon(value_map: dict[str, Any]) -> bool:
    if not value_map["standalone_sql_like"]:
        return False
    if value_map["flag_nonstandard_query_language"]:
        return False
    if value_map["flag_code_fragment_context"]:
        return False
    if value_map["flag_application_repo_context"]:
        return False
    if value_map["flag_application_schema_context"]:
        return False
    if value_map["flag_foreign_dataset_context"]:
        return False
    if value_map["hard_negative_source_root_match"] or value_map["hard_negative_table_token_match"]:
        return False
    if value_map["dataset_id_exact_match_in_context"]:
        return True
    if value_map["source_root_matches_known_benchmark_source"] and value_map["dataset_token_overlap_count"] >= 1:
        return True
    if value_map["dataset_token_overlap_count"] >= 2:
        return True
    if value_map["source_seed_specificity_hint"] == "strict" and value_map["dataset_token_overlap_count"] >= 1:
        return True
    return False


def build_attribute_lexicons(
    row_states: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    create_tokens: dict[str, set[str]] = defaultdict(set)
    dataset_tokens_index: dict[str, set[str]] = defaultdict(set)
    anchor_row_count: Counter[str] = Counter()

    for state in row_states:
        own_id = state["own_id"]
        dataset_tokens_index[own_id] = set(state["dataset_tokens"])
        if not candidate_for_attribute_lexicon(state):
            continue
        anchor_row_count[own_id] += 1
        for token in state["create_table_column_tokens"]:
            if token not in dataset_tokens_index[own_id]:
                create_tokens[own_id].add(token)
        for token in state["identifier_tokens"]:
            if token in dataset_tokens_index[own_id]:
                continue
            if token in APP_SCHEMA_TOKENS:
                continue
            counters[own_id][token] += 1

    lexicons: dict[str, dict[str, Any]] = {}
    for own_id in dataset_tokens_index:
        frequent_tokens = {
            token
            for token, count in counters[own_id].items()
            if count >= 2
        }
        token_set = set(create_tokens[own_id]) | frequent_tokens
        lexicons[own_id] = {
            "tokens": sorted(token_set),
            "available": len(token_set) >= 3,
            "anchor_row_count": anchor_row_count[own_id],
            "source": "baseline_consensus_sql_identifiers" if len(token_set) >= 3 else "not_available_in_baseline",
        }
    return lexicons


def duplicate_canonical_sort_key(state: dict[str, Any]) -> tuple[Any, ...]:
    return (
        TIER_PRIORITY.get(state["v2_source_credibility_tier"], 99),
        SPECIFICITY_PRIORITY.get(state["v2_specificity_label"], 99),
        CONFIDENCE_PRIORITY.get((state.get("evidence_confidence") or "").strip(), 99),
        EXECUTABLE_PRIORITY.get((state.get("executable_status") or "").strip(), 99),
        len((state.get("source_url") or "").strip()),
        (state.get("sql_item_id") or "").strip(),
    )


def apply_duplicate_policy(row_states: list[dict[str, Any]]) -> None:
    exact_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    normalized_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for state in row_states:
        state["v2_exact_duplicate_group_size"] = 1
        state["v2_normalized_duplicate_group_size"] = 1
        state["v2_duplicate_status"] = "canonical"
        state["v2_duplicate_of_sql_item_id"] = ""
        state["v2_keep_candidate_predup"] = "yes" if state["v2_specificity_label"] in {"strict", "weak"} else "no"
        if state["v2_specificity_label"] == "reject_non_sql":
            state["v2_keep_candidate"] = "no"
            continue
        raw_key = (
            state["own_id"],
            short_hash(normalize_newlines(state.get("sql_text_raw") or "")),
            normalize_url_root(state.get("source_url") or ""),
        )
        norm_key = (
            state["own_id"],
            short_hash(normalize_sql_for_duplicate((state.get("sql_text_clean") or "").strip() or (state.get("sql_text_raw") or ""))),
        )
        exact_groups[raw_key].append(state)
        normalized_groups[norm_key].append(state)

    for group in exact_groups.values():
        if len(group) <= 1:
            continue
        ordered = sorted(group, key=duplicate_canonical_sort_key)
        canonical = ordered[0]
        for state in ordered:
            state["v2_exact_duplicate_group_size"] = len(group)
        for state in ordered[1:]:
            state["v2_duplicate_status"] = "exact_duplicate"
            state["v2_duplicate_of_sql_item_id"] = canonical["sql_item_id"]

    for group in normalized_groups.values():
        if len(group) <= 1:
            continue
        ordered = sorted(group, key=duplicate_canonical_sort_key)
        canonical = ordered[0]
        for state in ordered:
            state["v2_normalized_duplicate_group_size"] = len(group)
        for state in ordered[1:]:
            if state["v2_duplicate_status"] == "canonical":
                state["v2_duplicate_status"] = "normalized_duplicate"
                state["v2_duplicate_of_sql_item_id"] = canonical["sql_item_id"]

    for state in row_states:
        if state["v2_specificity_label"] == "reject_non_sql":
            state["v2_keep_candidate"] = "no"
        elif state["v2_keep_candidate_predup"] == "yes" and state["v2_duplicate_status"] == "canonical":
            state["v2_keep_candidate"] = "yes"
        else:
            state["v2_keep_candidate"] = "no"


def label_reason_text(
    *,
    label: str,
    reason_code: str,
    state: dict[str, Any],
) -> str:
    matched_tokens = ", ".join(state["matched_dataset_tokens"]) or "none"
    attr_tokens = ", ".join(state["attribute_overlap_tokens"]) or "none"
    mismatch_flags = ", ".join(state["project_domain_mismatch_flags"]) or "none"
    return (
        f"Assigned {label} via {reason_code}. "
        f"dataset_name_match_quality={state['dataset_name_match_quality']}; "
        f"matched_dataset_tokens={matched_tokens}; "
        f"attribute_overlap_tokens={attr_tokens}; "
        f"benchmark_task_context_match={state['benchmark_task_context_match']}; "
        f"source_root_matches_known_benchmark_source={state['source_root_matches_known_benchmark_source']}; "
        f"project_domain_mismatch_flags={mismatch_flags}; "
        f"source_credibility_tier={state['v2_source_credibility_tier']}."
    )


def build_reclassify_summary(
    *,
    master_rows: list[dict[str, str]],
    row_states: list[dict[str, Any]],
    output_csv: Path,
    decisions_path: Path,
    rules_path: Path,
) -> str:
    before_counter = Counter((row.get("dataset_specificity_label") or "").strip() or "unknown" for row in master_rows)
    after_counter = Counter(state["v2_specificity_label"] for state in row_states)
    keep_counter = Counter(state["v2_specificity_label"] for state in row_states if state["v2_keep_candidate"] == "yes")
    tier_counter = Counter(state["v2_source_credibility_tier"] for state in row_states)

    per_dataset: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "dataset_name": "",
        "total": 0,
        "before_strict": 0,
        "before_weak": 0,
        "before_collision_risk": 0,
        "after_strict": 0,
        "after_weak": 0,
        "after_collision_risk": 0,
        "after_reject_non_sql": 0,
        "keep_yes": 0,
    })
    for row, state in zip(master_rows, row_states):
        own_id = state["own_id"]
        record = per_dataset[own_id]
        record["dataset_name"] = state["dataset_name"]
        record["total"] += 1
        before_label = (row.get("dataset_specificity_label") or "").strip() or "unknown"
        after_label = state["v2_specificity_label"]
        record[f"before_{before_label}"] = record.get(f"before_{before_label}", 0) + 1
        record[f"after_{after_label}"] = record.get(f"after_{after_label}", 0) + 1
        if state["v2_keep_candidate"] == "yes":
            record["keep_yes"] += 1

    table_lines = [
        "| own_id | dataset_name | total_rows | before_strict | before_weak | before_collision | after_strict | after_weak | after_collision | after_reject | keep_yes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for own_id, record in sorted(per_dataset.items()):
        table_lines.append(
            f"| {own_id} | {record['dataset_name']} | {record['total']} | "
            f"{record['before_strict']} | {record['before_weak']} | {record['before_collision_risk']} | "
            f"{record['after_strict']} | {record['after_weak']} | {record['after_collision_risk']} | "
            f"{record['after_reject_non_sql']} | {record['keep_yes']} |"
        )

    lines = [
        "# V2 Reclassification Summary",
        "",
        f"- Generated at UTC: `{utc_now_iso()}`",
        f"- Rules file: `{rules_path.resolve()}`",
        f"- Reclassified CSV: `{output_csv.resolve()}`",
        f"- Decisions ledger: `{decisions_path.resolve()}`",
        "- SQL execution was not performed in this phase.",
        "",
        "## Global Counts",
        "",
        f"- Total SQL rows reclassified: {len(master_rows)}",
        f"- Before counts: strict={before_counter.get('strict', 0)}, weak={before_counter.get('weak', 0)}, collision_risk={before_counter.get('collision_risk', 0)}",
        f"- After counts: strict={after_counter.get('strict', 0)}, weak={after_counter.get('weak', 0)}, collision_risk={after_counter.get('collision_risk', 0)}, reject_non_sql={after_counter.get('reject_non_sql', 0)}",
        f"- Keep-candidate rows after duplicate policy: strict={keep_counter.get('strict', 0)}, weak={keep_counter.get('weak', 0)}, total={sum(keep_counter.values())}",
        "",
        "## Source Credibility Tiers",
        "",
        f"- tier_1_official={tier_counter.get('tier_1_official', 0)}",
        f"- tier_2_primary_code={tier_counter.get('tier_2_primary_code', 0)}",
        f"- tier_3_secondary_explanatory={tier_counter.get('tier_3_secondary_explanatory', 0)}",
        f"- tier_4_low_trust={tier_counter.get('tier_4_low_trust', 0)}",
        "",
        "## Per-Dataset Before/After Counts",
        "",
        *table_lines,
    ]
    return "\n".join(lines)


def build_manifest(
    *,
    args: argparse.Namespace,
    row_states: list[dict[str, Any]],
    output_paths: list[Path],
) -> dict[str, Any]:
    after_counter = Counter(state["v2_specificity_label"] for state in row_states)
    keep_counter = Counter(state["v2_specificity_label"] for state in row_states if state["v2_keep_candidate"] == "yes")
    return {
        "phase": "v2_phase1_reclassify_baseline_sql_inventory",
        "generated_at_utc": utc_now_iso(),
        "input": {
            "master_sql_path": str(args.master_sql.resolve()),
            "master_sql_sha256": sha256_file(args.master_sql),
            "rules_json_path": str(args.rules_json.resolve()),
            "rules_json_sha256": sha256_file(args.rules_json),
            "source_inventory_path": str(args.source_inventory.resolve()),
            "source_inventory_sha256": sha256_file(args.source_inventory),
            "final_index_path": str(args.final_index.resolve()),
            "final_index_sha256": sha256_file(args.final_index),
        },
        "summary": {
            "total_rows": len(row_states),
            "after_strict_count": after_counter.get("strict", 0),
            "after_weak_count": after_counter.get("weak", 0),
            "after_collision_risk_count": after_counter.get("collision_risk", 0),
            "after_reject_non_sql_count": after_counter.get("reject_non_sql", 0),
            "keep_candidate_yes_count": sum(keep_counter.values()),
            "keep_candidate_strict_count": keep_counter.get("strict", 0),
            "keep_candidate_weak_count": keep_counter.get("weak", 0),
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
    output_dir = args.output_root / "reclassify"
    output_csv = output_dir / "master_sql_inventory_reclassified_v2.csv"
    decisions_path = output_dir / "reclassify_decisions.jsonl"
    summary_path = output_dir / "reclassify_summary.md"
    manifest_path = output_dir / "run_manifest_v2_phase1.json"

    rules = json.loads(args.rules_json.read_text(encoding="utf-8"))
    master_rows = read_csv_rows(args.master_sql)
    source_rows = read_csv_rows(args.source_inventory)
    final_index_rows = read_csv_rows(args.final_index)

    source_lookup = source_index(source_rows)
    source_rows_by_dataset = build_source_rows_by_dataset(source_rows)
    final_index = build_index(final_index_rows, "own_id")
    ambiguous_cases = {
        case["own_id"]: case for case in rules["name_collision_policy"]["cases"]
    }
    dataset_root_index = {
        own_id: benchmark_roots_for_dataset(own_id, final_index.get(own_id, {}), source_rows_by_dataset.get(own_id, []))
        for own_id in {row["own_id"] for row in master_rows}
    }

    row_states: list[dict[str, Any]] = []
    for row in master_rows:
        own_id = (row.get("own_id") or "").strip()
        dataset_id = (row.get("dataset_id") or "").strip()
        dataset_name = (row.get("dataset_name") or "").strip()
        source_meta = source_lookup.get((own_id, (row.get("source_url") or "").strip()), {})
        source_hint = (source_meta.get("dataset_specificity_hint") or row.get("source_seed_specificity_hint") or "").strip()
        source_http_status = (source_meta.get("http_status") or row.get("source_seed_http_status") or "").strip()
        retrieval_method = (source_meta.get("retrieval_method") or "").strip()
        has_sql_text = (source_meta.get("has_sql_text") or "").strip().lower()
        has_explicit_sql_text = has_sql_text in {"yes", "partial"}

        state: dict[str, Any] = dict(row)
        state["source_http_status"] = source_http_status
        state["http_status"] = source_http_status
        state["retrieval_method"] = retrieval_method
        state["dataset_specificity_hint"] = source_hint or "unknown"
        state["has_explicit_sql_text"] = has_explicit_sql_text
        state["source_url_domain"] = parse_domain((row.get("source_url") or "").strip())
        state["source_root"] = normalize_url_root((row.get("source_url") or "").strip())
        state["context_text"] = context_text(row)
        state["dataset_tokens"] = build_dataset_tokens(dataset_name, dataset_id)
        state["matched_dataset_tokens"] = [token for token in state["dataset_tokens"] if token in state["context_text"]]
        state["dataset_token_overlap_count"] = len(state["matched_dataset_tokens"])
        state["table_tokens"] = extract_table_tokens(row.get("sql_text_raw") or "")
        state["identifier_tokens"] = extract_identifier_tokens(row.get("sql_text_raw") or "")
        state["create_table_column_tokens"] = extract_create_table_column_tokens(row.get("sql_text_raw") or "")
        state["table_token_overlap_count"] = len(set(state["table_tokens"]) & set(state["dataset_tokens"]))
        state["app_schema_hit_count"] = len(set(state["table_tokens"]) & set(APP_SCHEMA_TOKENS))
        state["standalone_sql_like"] = standalone_sql_like(row.get("sql_text_raw") or "")
        state["sql_text_raw_stripped"] = (row.get("sql_text_raw") or "").strip()
        state["dataset_id_exact_match_in_context"] = detect_dataset_id_exact_match(dataset_id, row, state)
        state["source_root_matches_known_benchmark_source"] = state["source_root"] in dataset_root_index.get(own_id, set())
        root_and_context = " ".join(
            [
                state["source_root"],
                (row.get("source_title") or "").lower(),
                (row.get("source_file_path") or "").lower(),
                state["context_text"],
            ]
        )
        state["flag_generic_course_or_tutorial_source"] = any(pattern in root_and_context for pattern in GENERIC_SOURCE_PATTERNS)
        state["flag_nonstandard_query_language"] = any(pattern in state["context_text"] for pattern in NONSTANDARD_QUERY_PATTERNS)
        state["flag_code_fragment_context"] = any(pattern in state["context_text"] for pattern in CODE_FRAGMENT_PATTERNS)
        state["flag_application_repo_context"] = any(pattern in root_and_context for pattern in rules["shared_derivation"]["pattern_sets"]["application_repo_patterns"])
        state["flag_application_schema_context"] = state["app_schema_hit_count"] >= 2
        state["flag_foreign_dataset_context"] = any(pattern in state["context_text"] for pattern in FOREIGN_DATASET_PATTERNS) and state["dataset_token_overlap_count"] < 2
        state["flag_string_literal_not_standalone_sql"] = (
            (row.get("extraction_method") or "").strip() in {"source_string_literal", "ipynb_string_literal"}
            and not state["standalone_sql_like"]
        )
        state["source_explicitly_contains_sql_error"] = any(pattern in state["context_text"] for pattern in SQL_ERROR_PATTERNS)
        state["local_sql_parser_failure"] = False
        state["schema_mapping_available"] = False
        state["unresolved_table_or_column_reference"] = False
        state["local_parse_and_execution_success"] = False
        state["source_displays_query_result_table"] = False
        state["query_text_matches_result_block"] = False
        state["notebook_or_repo_documents_successful_execution_for_same_query"] = False
        ambiguous_case = ambiguous_cases.get(own_id)
        hard_negative_source_root_match = False
        hard_negative_table_token_match = False
        ambiguous_single_token_match = False
        if ambiguous_case:
            hard_negative_source_root_match = any(
                fragment in root_and_context
                for fragment in ambiguous_case.get("hard_negative_source_root_fragments", [])
            )
            hard_negative_table_token_match = any(
                token in set(state["table_tokens"]) | set(state["identifier_tokens"])
                for token in ambiguous_case.get("hard_negative_table_tokens", [])
            )
            single_token_never_sufficient = set(ambiguous_case.get("single_token_never_sufficient", []))
            ambiguous_single_token_match = (
                state["dataset_token_overlap_count"] == 1
                and len(state["matched_dataset_tokens"]) == 1
                and state["matched_dataset_tokens"][0] in single_token_never_sufficient
            )
        state["hard_negative_source_root_match"] = hard_negative_source_root_match
        state["hard_negative_table_token_match"] = hard_negative_table_token_match
        state["ambiguous_single_token_match"] = ambiguous_single_token_match
        state["project_domain_mismatch_flags"] = [
            name
            for name, flag in [
                ("application_repo_context", state["flag_application_repo_context"]),
                ("application_schema_context", state["flag_application_schema_context"]),
                ("foreign_dataset_context", state["flag_foreign_dataset_context"]),
                ("hard_negative_source_root_match", state["hard_negative_source_root_match"]),
                ("hard_negative_table_token_match", state["hard_negative_table_token_match"]),
            ]
            if flag
        ]
        state["project_domain_mismatch"] = bool(state["project_domain_mismatch_flags"])
        state["dataset_name_match_quality"] = dataset_match_quality(
            dataset_token_overlap_count=state["dataset_token_overlap_count"],
            dataset_id_exact_match=state["dataset_id_exact_match_in_context"],
            ambiguous_single_token_match=state["ambiguous_single_token_match"],
        )
        state["benchmark_task_context_match"] = benchmark_task_context_match(
            dataset_id_exact_match=state["dataset_id_exact_match_in_context"],
            source_root_known=state["source_root_matches_known_benchmark_source"],
            dataset_token_overlap_count=state["dataset_token_overlap_count"],
            source_hint=state["dataset_specificity_hint"],
            project_domain_mismatch=state["project_domain_mismatch"],
        )
        state["create_table_detected"] = bool(state["create_table_column_tokens"])
        state["source_seed_specificity_hint"] = (row.get("source_seed_specificity_hint") or "").strip()
        row_states.append(state)

    attribute_lexicons = build_attribute_lexicons(row_states)

    for state in row_states:
        lexicon = attribute_lexicons.get(state["own_id"], {"tokens": [], "available": False, "source": "not_available_in_baseline", "anchor_row_count": 0})
        attribute_tokens = set(lexicon["tokens"])
        identifier_set = set(state["identifier_tokens"]) | set(state["create_table_column_tokens"])
        state["known_attribute_tokens"] = lexicon["tokens"]
        state["known_attribute_tokens_available"] = lexicon["available"]
        state["known_attribute_tokens_source"] = lexicon["source"]
        state["known_attribute_anchor_row_count"] = lexicon["anchor_row_count"]
        state["attribute_overlap_tokens"] = sorted(identifier_set & attribute_tokens)
        state["attribute_overlap_count"] = len(state["attribute_overlap_tokens"])

        state["source_credibility_tier_reason_code"] = ""
        state["v2_source_credibility_tier"], state["source_credibility_tier_reason_code"] = assign_source_tier(state, rules)
        state["source_credibility_tier"] = state["v2_source_credibility_tier"]
        reject_rule = first_matching_rule(state, rules["criteria"]["reject_non_sql"]["rules"])
        state["rejected_non_sql"] = reject_rule is not None

        specificity_label = ""
        specificity_reason_code = ""

        if reject_rule is not None:
            specificity_label = "reject_non_sql"
            specificity_reason_code = reject_rule["reason_code"]
        elif state["hard_negative_source_root_match"] or state["hard_negative_table_token_match"]:
            specificity_label = "collision_risk"
            specificity_reason_code = "ambiguous_case_hard_negative_context"
        else:
            strict_block = rules["criteria"]["strict_dataset_specific"]
            weak_block = rules["criteria"]["weak_related"]
            collision_block = rules["criteria"]["collision_risk"]

            strict_rule = None
            if matches_all(state, strict_block["preconditions_all"]) and not any(
                evaluate_condition(state, condition) for condition in strict_block["forbid_if_any"]
            ):
                strict_rule = first_matching_rule(state, strict_block["assign_if_any"])

            strict_gate = (
                strict_rule is not None
                and state["benchmark_task_context_match"] in {"strong", "moderate"}
                and (
                    not state["known_attribute_tokens_available"]
                    or state["attribute_overlap_count"] >= 1
                    or state["table_token_overlap_count"] >= 1
                    or state["dataset_id_exact_match_in_context"]
                    or state["create_table_detected"]
                )
                and not state["project_domain_mismatch"]
            )
            strict_reason_code = strict_rule["id"] if strict_gate and strict_rule is not None else ""
            if (
                not strict_gate
                and not state["project_domain_mismatch"]
                and not state["flag_generic_course_or_tutorial_source"]
                and state["v2_source_credibility_tier"] in {"tier_1_official", "tier_2_primary_code", "tier_3_secondary_explanatory"}
                and state["known_attribute_tokens_available"]
                and state["attribute_overlap_count"] >= 1
                and state["source_root_matches_known_benchmark_source"]
                and state["benchmark_task_context_match"] == "strong"
            ):
                strict_gate = True
                strict_reason_code = "strict_known_source_root_with_attribute_overlap"
            elif (
                not strict_gate
                and not state["project_domain_mismatch"]
                and not state["flag_generic_course_or_tutorial_source"]
                and state["v2_source_credibility_tier"] in {"tier_1_official", "tier_2_primary_code", "tier_3_secondary_explanatory"}
                and state["known_attribute_tokens_available"]
                and state["attribute_overlap_count"] >= 1
                and state["dataset_token_overlap_count"] >= 2
                and state["benchmark_task_context_match"] in {"strong", "moderate"}
            ):
                strict_gate = True
                strict_reason_code = "strict_multi_token_attribute_overlap"
            state["assigned_strict"] = bool(strict_gate)

            weak_rule = None
            if matches_all(state, weak_block["preconditions_all"]) and not any(
                evaluate_condition(state, condition) for condition in weak_block["forbid_if_any"]
            ):
                weak_rule = first_matching_rule(state, weak_block["assign_if_any"])

            weak_gate = (
                weak_rule is not None
                and state["benchmark_task_context_match"] in {"strong", "moderate", "weak"}
                and not state["project_domain_mismatch"]
            )
            state["assigned_weak"] = bool(weak_gate)

            if strict_gate:
                specificity_label = "strict"
                specificity_reason_code = strict_reason_code
            elif weak_gate:
                specificity_label = "weak"
                specificity_reason_code = weak_rule["id"]
            else:
                collision_rule = None
                if matches_all(state, collision_block["preconditions_all"]):
                    collision_rule = first_matching_rule(state, collision_block["assign_if_any"])
                specificity_label = "collision_risk"
                specificity_reason_code = (
                    collision_rule["id"]
                    if collision_rule is not None
                    else "collision_risk_fallback_default"
                )

        state["v2_specificity_label"] = specificity_label
        state["v2_specificity_reason_code"] = specificity_reason_code
        state["v2_specificity_reason_text"] = label_reason_text(
            label=specificity_label,
            reason_code=specificity_reason_code,
            state=state,
        )

    apply_duplicate_policy(row_states)

    output_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for state in row_states:
        signal_payload = {
            "dataset_name_match_quality": state["dataset_name_match_quality"],
            "matched_dataset_tokens": state["matched_dataset_tokens"],
            "dataset_token_overlap_count": state["dataset_token_overlap_count"],
            "dataset_id_exact_match_in_context": state["dataset_id_exact_match_in_context"],
            "source_root_matches_known_benchmark_source": state["source_root_matches_known_benchmark_source"],
            "benchmark_task_context_match": state["benchmark_task_context_match"],
            "table_tokens": state["table_tokens"],
            "table_token_overlap_count": state["table_token_overlap_count"],
            "identifier_tokens": state["identifier_tokens"],
            "known_attribute_tokens_available": state["known_attribute_tokens_available"],
            "known_attribute_tokens_source": state["known_attribute_tokens_source"],
            "known_attribute_anchor_row_count": state["known_attribute_anchor_row_count"],
            "attribute_overlap_count": state["attribute_overlap_count"],
            "attribute_overlap_tokens": state["attribute_overlap_tokens"],
            "project_domain_mismatch": state["project_domain_mismatch"],
            "project_domain_mismatch_flags": state["project_domain_mismatch_flags"],
            "hard_negative_source_root_match": state["hard_negative_source_root_match"],
            "hard_negative_table_token_match": state["hard_negative_table_token_match"],
            "ambiguous_single_token_match": state["ambiguous_single_token_match"],
            "standalone_sql_like": state["standalone_sql_like"],
            "flag_generic_course_or_tutorial_source": state["flag_generic_course_or_tutorial_source"],
            "flag_nonstandard_query_language": state["flag_nonstandard_query_language"],
            "flag_code_fragment_context": state["flag_code_fragment_context"],
            "flag_application_repo_context": state["flag_application_repo_context"],
            "flag_application_schema_context": state["flag_application_schema_context"],
            "flag_foreign_dataset_context": state["flag_foreign_dataset_context"],
            "source_http_status": state["source_http_status"],
            "source_hint": state["dataset_specificity_hint"],
            "retrieval_method": state["retrieval_method"],
            "duplicate_status": state["v2_duplicate_status"],
            "duplicate_of_sql_item_id": state["v2_duplicate_of_sql_item_id"],
            "exact_duplicate_group_size": state["v2_exact_duplicate_group_size"],
            "normalized_duplicate_group_size": state["v2_normalized_duplicate_group_size"],
        }
        output_row = dict(state)
        output_row["v2_specificity_label"] = state["v2_specificity_label"]
        output_row["v2_specificity_reason_code"] = state["v2_specificity_reason_code"]
        output_row["v2_specificity_reason_text"] = state["v2_specificity_reason_text"]
        output_row["v2_dataset_match_signals"] = csv_json_dumps(signal_payload)
        output_row["v2_source_credibility_tier"] = state["v2_source_credibility_tier"]
        output_row["v2_keep_candidate"] = state["v2_keep_candidate"]
        output_rows.append(output_row)

        decision_rows.append(
            {
                "own_id": state["own_id"],
                "dataset_id": state["dataset_id"],
                "dataset_name": state["dataset_name"],
                "sql_item_id": state["sql_item_id"],
                "source_url": state["source_url"],
                "original_dataset_specificity_label": state.get("dataset_specificity_label", ""),
                "v2_specificity_label": state["v2_specificity_label"],
                "v2_specificity_reason_code": state["v2_specificity_reason_code"],
                "v2_specificity_reason_text": state["v2_specificity_reason_text"],
                "v2_source_credibility_tier": state["v2_source_credibility_tier"],
                "v2_keep_candidate": state["v2_keep_candidate"],
                "v2_duplicate_status": state["v2_duplicate_status"],
                "v2_duplicate_of_sql_item_id": state["v2_duplicate_of_sql_item_id"],
                "signals": signal_payload,
            }
        )

    output_fieldnames = list(master_rows[0].keys()) + NEW_FIELDNAMES
    write_csv(output_csv, output_fieldnames, output_rows)
    write_jsonl(decisions_path, decision_rows)
    write_text(
        summary_path,
        build_reclassify_summary(
            master_rows=master_rows,
            row_states=row_states,
            output_csv=output_csv,
            decisions_path=decisions_path,
            rules_path=args.rules_json,
        ),
    )

    manifest_payload = build_manifest(
        args=args,
        row_states=row_states,
        output_paths=[output_csv, decisions_path, summary_path],
    )
    write_json(manifest_path, manifest_payload)
    manifest_payload["outputs"] = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in [output_csv, decisions_path, summary_path, manifest_path]
    ]
    write_json(manifest_path, manifest_payload)

    before_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    after_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_names: dict[str, str] = {}
    for row, state in zip(master_rows, row_states):
        own_id = state["own_id"]
        dataset_names[own_id] = state["dataset_name"]
        before_by_dataset[own_id][(row.get("dataset_specificity_label") or "").strip() or "unknown"] += 1
        after_by_dataset[own_id][state["v2_specificity_label"]] += 1

    print(str(output_csv.resolve()))
    print(str(decisions_path.resolve()))
    print(str(summary_path.resolve()))
    print(str(manifest_path.resolve()))
    print("")
    print("PER-DATASET BEFORE/AFTER COUNTS")
    for own_id in sorted(dataset_names):
        before = before_by_dataset[own_id]
        after = after_by_dataset[own_id]
        print(
            f"{own_id}\t{dataset_names[own_id]}\t"
            f"before(strict={before.get('strict', 0)},weak={before.get('weak', 0)},collision_risk={before.get('collision_risk', 0)})\t"
            f"after(strict={after.get('strict', 0)},weak={after.get('weak', 0)},collision_risk={after.get('collision_risk', 0)},reject_non_sql={after.get('reject_non_sql', 0)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
