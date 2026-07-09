#!/usr/bin/env python3
"""Analyze V2 SQL assets and map representative rows to a new from-scratch taxonomy."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_FINAL_INDEX = Path("logs/sql_high_corpus_build_20260404/v2_refinement/final_v2/final_index_v2.csv")
DEFAULT_EXECUTE = Path("logs/sql_high_corpus_build_20260404/v2_refinement/execute/sql_executability_v2.csv")
DEFAULT_OUTPUT = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/final_v2/taxonomy_mapping_sample.csv"
)

SAMPLE_FIELDNAMES = [
    "inventory_file_path",
    "own_id",
    "dataset_name",
    "dataset_readiness_v2",
    "sql_item_id",
    "source_url",
    "source_title",
    "recommended_taxonomy_category",
    "assignment_confidence",
    "benchmark_use_recommendation",
    "selection_role",
    "structural_signature",
    "observed_features",
    "notes",
    "sql_snippet",
]

CATEGORY_ORDER = [
    "TABLE_SANITY_PROFILE",
    "FILTERED_COHORT_LOOKUP",
    "SEGMENTED_AGGREGATE",
    "INTERACTION_AGGREGATE",
    "RANK_EXTREME_ANALYSIS",
    "DERIVED_RATE_RULE",
    "DATA_PREPARATION",
    "NON_BENCHMARK_NOISE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify SQL rows using a new structural taxonomy and write a representative mapping sample."
    )
    parser.add_argument("--final-index", type=Path, default=DEFAULT_FINAL_INDEX)
    parser.add_argument("--execute-csv", type=Path, default=DEFAULT_EXECUTE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def root_url(url: str) -> str:
    return url.split("/blob/")[0] if "/blob/" in url else url


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def sql_text(row: dict[str, str]) -> str:
    return normalize_ws(
        row.get("sql_canonical_v2")
        or row.get("sql_text_clean")
        or row.get("sql_text_raw")
        or ""
    )


KEYWORDS: list[tuple[str, str]] = [
    ("create_procedure", r"create\s+(or\s+replace\s+)?procedure\b"),
    ("create_function", r"create\s+(or\s+replace\s+)?function\b"),
    ("create_view", r"create\s+(or\s+replace\s+)?view\b"),
    ("create_table", r"create\s+table\b"),
    ("create_database", r"create\s+database\b"),
    ("alter_table", r"alter\s+table\b"),
    ("drop_database", r"drop\s+database\b"),
    ("drop_table", r"drop\s+table\b"),
    ("insert", r"insert\s+into\b|insert\b"),
    ("update", r"update\b"),
    ("delete", r"delete\s+from\b|delete\b"),
    ("copy", r"copy\s*\("),
    ("use", r"use\s+[a-z_`\"\[]"),
    ("with_select", r"with\b"),
    ("select", r"select\b"),
    ("begin", r"begin\b"),
    ("public_class", r"public\s+class\b"),
]


def first_keyword(sql: str) -> str:
    text = sql.lower()
    best_name = "other"
    best_pos = 10**9
    for name, pattern in KEYWORDS:
        match = re.search(pattern, text, re.I | re.S)
        if match and match.start() < best_pos:
            best_name = name
            best_pos = match.start()
    return best_name


def has(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, re.I | re.S))


def group_key_count(sql: str) -> int:
    match = re.search(
        r"\bgroup\s+by\b\s+(.*?)(?:\border\s+by\b|\bhaving\b|\blimit\b|;|$)",
        sql,
        re.I | re.S,
    )
    if not match:
        return 0
    clause = match.group(1)
    parts = [part.strip() for part in clause.split(",") if part.strip()]
    return len(parts)


def table_source_count(sql: str) -> int:
    text = sql.lower()
    join_ct = len(re.findall(r"\bjoin\b", text))
    from_match = re.search(
        r"\bfrom\b\s+(.*?)(?:\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|;|$)",
        text,
        re.I | re.S,
    )
    if from_match:
        fragment = from_match.group(1)
        return 1 + join_ct + fragment.count(",")
    if join_ct:
        return 1 + join_ct
    return 0


def extract_features(row: dict[str, str]) -> dict[str, Any]:
    sql = sql_text(row)
    lowered = sql.lower()
    keyword = first_keyword(sql)
    features = {
        "keyword": keyword,
        "has_group_by": has(r"\bgroup\s+by\b", sql),
        "has_where": has(r"\bwhere\b", sql),
        "has_order_by": has(r"\border\s+by\b", sql),
        "has_limit": has(r"\blimit\b|fetch\s+first\b", sql),
        "has_window": has(r"\bover\s*\(", sql),
        "has_rank": has(r"\b(rank|dense_rank|row_number)\s*\(", sql),
        "has_agg": has(r"\b(count|sum|avg|min|max|percentile_cont|stddev|variance)\s*\(", sql),
        "has_case": has(r"\bcase\b", sql),
        "has_subquery": has(r"\(\s*select\b", sql),
        "has_join": has(r"\bjoin\b", sql),
        "has_distinct": has(r"\bdistinct\b", sql),
        "has_union_like": has(r"\b(union|intersect|except|minus)\b", sql),
        "has_ratio": (
            "/" in sql
            or "100.0" in lowered
            or "100 *" in lowered
            or "* 100" in lowered
            or "percentage" in lowered
            or " rate" in lowered
            or "ratio" in lowered
        ),
        "select_star": has(r"(^|\bselect\b)\s*\*\b", sql),
        "obvious_code": (
            "query = f" in lowered
            or "try (" in lowered
            or "jtextarea" in lowered
            or "train_test_split" in lowered
            or "axis=1" in lowered
            or "inplace=true" in lowered
            or "random_state=" in lowered
            or "preparedstatement" in lowered
            or "conn.preparestatement" in lowered
            or "matcher(" in lowered
            or "doc[" in lowered
            or "resul" in lowered and "createStatement" in lowered
        ),
        "group_key_count": group_key_count(sql),
        "table_source_count": table_source_count(sql),
        "source_title": row.get("source_title") or "",
        "sql": sql,
    }
    return features


def classify_primary_taxonomy(features: dict[str, Any]) -> str:
    keyword = features["keyword"]
    if keyword in {"use", "create_database", "drop_database", "copy", "begin", "public_class"}:
        return "NON_BENCHMARK_NOISE"
    if keyword in {"create_procedure", "create_function"}:
        return "NON_BENCHMARK_NOISE"
    if features["obvious_code"]:
        return "NON_BENCHMARK_NOISE"
    if keyword in {"create_table", "create_view", "alter_table", "insert", "update", "delete", "drop_table"}:
        return "DATA_PREPARATION"
    if keyword in {"select", "with_select", "other"}:
        if features["has_rank"] or features["has_window"] or (
            features["has_order_by"] and features["has_limit"]
        ):
            return "RANK_EXTREME_ANALYSIS"
        if features["has_ratio"] or (features["has_case"] and features["has_agg"]):
            return "DERIVED_RATE_RULE"
        if features["has_group_by"] and (
            features["group_key_count"] >= 2 or features["table_source_count"] >= 2
        ):
            return "INTERACTION_AGGREGATE"
        if features["has_group_by"] and features["group_key_count"] == 1:
            return "SEGMENTED_AGGREGATE"
        if features["has_where"]:
            return "FILTERED_COHORT_LOOKUP"
        return "TABLE_SANITY_PROFILE"
    return "NON_BENCHMARK_NOISE"


def confidence_label(features: dict[str, Any], category: str, readiness: str) -> str:
    if category in {"NON_BENCHMARK_NOISE", "DATA_PREPARATION"}:
        return "high"
    if category == "RANK_EXTREME_ANALYSIS" and (features["has_rank"] or features["has_window"]):
        return "high"
    if category in {"SEGMENTED_AGGREGATE", "INTERACTION_AGGREGATE"} and features["has_group_by"]:
        return "high"
    if category == "DERIVED_RATE_RULE" and (features["has_ratio"] or features["has_case"]):
        return "high"
    if category == "FILTERED_COHORT_LOOKUP" and features["has_where"] and not features["has_group_by"]:
        return "medium"
    if category == "TABLE_SANITY_PROFILE" and readiness == "NOT_READY":
        return "low"
    return "medium"


def benchmark_recommendation(category: str, readiness: str, features: dict[str, Any]) -> str:
    if category in {"NON_BENCHMARK_NOISE", "DATA_PREPARATION"}:
        return "exclude"
    if readiness == "NOT_READY":
        return "review_only"
    if features["table_source_count"] >= 2:
        return "review_only"
    if category == "TABLE_SANITY_PROFILE" and features["select_star"]:
        return "context_only"
    if category == "TABLE_SANITY_PROFILE":
        return "candidate_low_priority"
    return "candidate"


def structural_signature(features: dict[str, Any]) -> str:
    parts = [features["keyword"]]
    for key, label in [
        ("has_group_by", "group"),
        ("has_where", "where"),
        ("has_order_by", "order"),
        ("has_limit", "limit"),
        ("has_window", "window"),
        ("has_agg", "agg"),
        ("has_case", "case"),
        ("has_subquery", "subquery"),
        ("has_join", "join"),
        ("has_union_like", "setop"),
    ]:
        if features[key]:
            parts.append(label)
    return "|".join(parts)


def observed_features_text(features: dict[str, Any]) -> str:
    items = []
    if features["has_group_by"]:
        items.append(f"group_keys={features['group_key_count']}")
    if features["has_where"]:
        items.append("where")
    if features["has_order_by"]:
        items.append("order_by")
    if features["has_limit"]:
        items.append("limit_or_fetch")
    if features["has_window"]:
        items.append("window")
    if features["has_rank"]:
        items.append("rank")
    if features["has_agg"]:
        items.append("aggregate")
    if features["has_case"]:
        items.append("case")
    if features["has_ratio"]:
        items.append("ratio_or_rate")
    if features["has_subquery"]:
        items.append("subquery")
    if features["has_join"]:
        items.append("join")
    if features["has_union_like"]:
        items.append("set_operation")
    if features["select_star"]:
        items.append("select_star")
    if features["table_source_count"]:
        items.append(f"table_sources≈{features['table_source_count']}")
    return ", ".join(items) if items else "plain_select_like"


def note_text(row: dict[str, str], features: dict[str, Any], category: str, readiness: str) -> str:
    own_id = row.get("own_id") or ""
    title = (row.get("source_title") or "").lower()
    if category == "NON_BENCHMARK_NOISE":
        return (
            "Environment/setup/programmatic SQL or embedded code; keep as negative evidence, not as a benchmark target."
        )
    if category == "DATA_PREPARATION":
        return (
            "Useful for schema/context reconstruction, but it measures data-loading or cleanup behavior rather than synthetic-data fidelity."
        )
    if readiness == "NOT_READY":
        if own_id in {"c7", "c5", "c14", "c15", "c18", "m10", "n16", "n4", "c19"}:
            return (
                "Category assignment is structurally clear, but the surrounding dataset/source alignment is risky, so this row should stay review-only."
            )
        return "Structurally usable, but the dataset is not yet in the trusted V2-ready slice."
    if own_id == "c13":
        return (
            "Structurally rich, but many `c13` rows are multi-source census workflows; treat as a boundary case for a single-table benchmark."
        )
    if category == "TABLE_SANITY_PROFILE":
        return "Good for schema grounding and basic realism checks, but usually not sufficient as a high-value benchmark query on its own."
    if category == "FILTERED_COHORT_LOOKUP":
        return "Useful for condition-fidelity checks because it probes whether local slices of the table behave plausibly."
    if category == "SEGMENTED_AGGREGATE":
        return "Useful benchmark candidate because it tests marginal patterns across a single grouping dimension."
    if category == "INTERACTION_AGGREGATE":
        return "Useful when it stays within one analysis table; multi-source rows need extra review under the single-table benchmark objective."
    if category == "RANK_EXTREME_ANALYSIS":
        return "High-value benchmark candidate because it stresses tails, ordering stability, and rare/extreme segments."
    if category == "DERIVED_RATE_RULE":
        return "High-value benchmark candidate because it tests conditional arithmetic, rates, or bucketed business rules beyond raw counts."
    return "Review manually."


def selection_priority(row: dict[str, Any]) -> tuple[Any, ...]:
    readiness_rank = {"READY": 0, "READY_WITH_WARNINGS": 1, "NOT_READY": 2}.get(
        row["dataset_readiness_v2"], 9
    )
    benchmark_rank = {
        "candidate": 0,
        "candidate_low_priority": 1,
        "context_only": 2,
        "review_only": 3,
        "exclude": 4,
    }.get(row["benchmark_use_recommendation"], 9)
    confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(row["assignment_confidence"], 9)
    sql_len = len(row["sql_snippet"])
    return (
        readiness_rank,
        benchmark_rank,
        confidence_rank,
        row["own_id"],
        row["sql_item_id"],
        abs(sql_len - 140),
    )


def select_representative_sample(enriched_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched_rows:
        by_category[row["recommended_taxonomy_category"]].append(row)

    sample_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    category_targets = {
        "TABLE_SANITY_PROFILE": 6,
        "FILTERED_COHORT_LOOKUP": 6,
        "SEGMENTED_AGGREGATE": 6,
        "INTERACTION_AGGREGATE": 6,
        "RANK_EXTREME_ANALYSIS": 6,
        "DERIVED_RATE_RULE": 6,
        "DATA_PREPARATION": 6,
        "NON_BENCHMARK_NOISE": 6,
    }

    dataset_priority_by_category = {
        "TABLE_SANITY_PROFILE": ["m12", "c17", "m4", "m8", "m11", "c13", "c2"],
        "FILTERED_COHORT_LOOKUP": ["m12", "c17", "m4", "m8", "m11", "c13", "c2"],
        "SEGMENTED_AGGREGATE": ["m12", "c17", "m4", "m8", "m11", "c2", "c13"],
        "INTERACTION_AGGREGATE": ["m12", "c17", "m4", "m8", "m11", "c13", "c2"],
        "RANK_EXTREME_ANALYSIS": ["m12", "c17", "m4", "m8", "m11", "c13", "c2"],
        "DERIVED_RATE_RULE": ["m12", "c17", "m4", "m8", "m11", "c13", "c2"],
        "DATA_PREPARATION": ["c5", "c7", "n16", "c19", "c14", "m10", "c13"],
        "NON_BENCHMARK_NOISE": ["c7", "c5", "c19", "n16", "c14", "m10", "c13"],
    }

    for category in CATEGORY_ORDER:
        candidates = sorted(by_category.get(category, []), key=selection_priority)
        per_dataset: Counter[str] = Counter()
        per_root: Counter[str] = Counter()
        selected = 0

        def try_take(row: dict[str, Any]) -> bool:
            nonlocal selected
            sql_item_id = row["sql_item_id"]
            if sql_item_id in seen_ids:
                return False
            if selected >= category_targets[category]:
                return False
            if per_dataset[row["own_id"]] >= 2:
                return False
            if per_root[row["source_root"]] >= 2:
                return False
            sample_rows.append(row)
            seen_ids.add(sql_item_id)
            per_dataset[row["own_id"]] += 1
            per_root[row["source_root"]] += 1
            selected += 1
            return True

        for own_id in dataset_priority_by_category.get(category, []):
            for row in candidates:
                if row["own_id"] == own_id and try_take(row):
                    break
            if selected >= category_targets[category]:
                break

        if selected < category_targets[category]:
            for row in candidates:
                try_take(row)
                if selected >= category_targets[category]:
                    break

    return sample_rows


def main() -> None:
    args = parse_args()
    final_index_path = args.final_index.resolve()
    execute_path = args.execute_csv.resolve()
    output_path = args.output_csv.resolve()

    final_rows = read_csv_rows(final_index_path)
    execute_rows = read_csv_rows(execute_path)
    readiness_by_id = {(row.get("own_id") or "").strip(): row.get("readiness_label_v2") or "" for row in final_rows}

    enriched_rows: list[dict[str, Any]] = []
    category_counts_all = Counter()
    category_counts_ready_primary = Counter()
    ready_warning_ids = {
        own_id for own_id, label in readiness_by_id.items() if label in {"READY", "READY_WITH_WARNINGS"}
    }

    for row in execute_rows:
        own_id = (row.get("own_id") or "").strip()
        inventory_file_path = (
            Path("logs/sql_high_corpus_build_20260404/datasets") / own_id / "v2" / "sql_inventory_v2.csv"
        ).resolve()
        features = extract_features(row)
        category = classify_primary_taxonomy(features)
        readiness = readiness_by_id.get(own_id, "")
        confidence = confidence_label(features, category, readiness)
        benchmark_use = benchmark_recommendation(category, readiness, features)
        snippet = features["sql"][:220] + ("..." if len(features["sql"]) > 220 else "")
        enriched = {
            "inventory_file_path": str(inventory_file_path),
            "own_id": own_id,
            "dataset_name": row.get("dataset_name") or "",
            "dataset_readiness_v2": readiness,
            "sql_item_id": row.get("sql_item_id") or "",
            "source_url": row.get("source_url") or "",
            "source_root": root_url(row.get("source_url") or ""),
            "source_title": row.get("source_title") or "",
            "recommended_taxonomy_category": category,
            "assignment_confidence": confidence,
            "benchmark_use_recommendation": benchmark_use,
            "selection_role": "",
            "structural_signature": structural_signature(features),
            "observed_features": observed_features_text(features),
            "notes": note_text(row, features, category, readiness),
            "sql_snippet": snippet,
            "is_primary_canonical": row.get("is_primary_canonical") or "",
            "v2_keep_candidate": row.get("v2_keep_candidate") or "",
        }
        enriched_rows.append(enriched)
        category_counts_all[category] += 1
        if (
            own_id in ready_warning_ids
            and enriched["is_primary_canonical"] == "yes"
            and enriched["v2_keep_candidate"] == "yes"
        ):
            category_counts_ready_primary[category] += 1

    sample_rows = select_representative_sample(enriched_rows)
    role_by_category = {
        "TABLE_SANITY_PROFILE": "positive_or_context",
        "FILTERED_COHORT_LOOKUP": "positive_candidate",
        "SEGMENTED_AGGREGATE": "positive_candidate",
        "INTERACTION_AGGREGATE": "positive_or_boundary",
        "RANK_EXTREME_ANALYSIS": "positive_candidate",
        "DERIVED_RATE_RULE": "positive_candidate",
        "DATA_PREPARATION": "anti_pattern_or_context",
        "NON_BENCHMARK_NOISE": "anti_pattern",
    }
    for row in sample_rows:
        row["selection_role"] = role_by_category[row["recommended_taxonomy_category"]]

    sample_rows.sort(
        key=lambda row: (
            CATEGORY_ORDER.index(row["recommended_taxonomy_category"]),
            row["dataset_readiness_v2"],
            row["own_id"],
            row["sql_item_id"],
        )
    )
    write_csv(output_path, SAMPLE_FIELDNAMES, sample_rows)

    summary = {
        "all_rows": {
            "row_count": len(execute_rows),
            "category_counts": category_counts_all,
        },
        "ready_warning_primary_kept": {
            "row_count": sum(
                1
                for row in enriched_rows
                if row["dataset_readiness_v2"] in {"READY", "READY_WITH_WARNINGS"}
                and row["is_primary_canonical"] == "yes"
                and row["v2_keep_candidate"] == "yes"
            ),
            "category_counts": category_counts_ready_primary,
        },
        "sample": {
            "row_count": len(sample_rows),
            "category_counts": Counter(row["recommended_taxonomy_category"] for row in sample_rows),
            "output_csv": str(output_path),
        },
    }
    serializable = json.loads(
        json.dumps(
            summary,
            default=lambda value: dict(value) if isinstance(value, Counter) else str(value),
        )
    )
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
