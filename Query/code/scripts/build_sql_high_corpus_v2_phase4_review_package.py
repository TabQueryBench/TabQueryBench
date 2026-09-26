#!/usr/bin/env python3
"""Rebuild per-dataset final assets from the V2-cleaned SQL inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_phase_c_sql_inventory import (  # noqa: E402
    extract_table_tokens,
    normalize_url_root,
)


DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404")
DEFAULT_DATASET_ROOT = Path("logs/sql_high_corpus_build_20260404/datasets")
DEFAULT_GLOBAL_OUTPUT_DIR = Path("logs/sql_high_corpus_build_20260404/final_v2")
DEFAULT_SCOPE_CSV = Path("logs/sql_high_corpus_build_20260404/scope/high_datasets.csv")
DEFAULT_RECLASSIFY_CSV = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/reclassify/master_sql_inventory_reclassified_v2.csv"
)
DEFAULT_DEDUP_CSV = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/dedup/master_sql_inventory_dedup_v2.csv"
)
DEFAULT_EXECUTE_CSV = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/execute/sql_executability_v2.csv"
)
DEFAULT_GATE_CSV = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/gate/dataset_readiness_v2.csv"
)

TOP_STRICT_LIMIT = 20
QUESTION_SEED_LIMIT = 10

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
    "query_intent_label",
    "family_tag_guess",
    "sql_complexity",
    "evidence_confidence",
    "v2_source_credibility_tier",
    "executable_status_v2",
    "canonical_group_id",
    "sql_fingerprint_v2",
    "sql_text_clean",
    "sql_text_raw",
    "selection_notes",
]

REJECTED_FIELDNAMES = [
    "own_id",
    "dataset_id",
    "dataset_name",
    "sql_item_id",
    "source_url",
    "source_type",
    "source_title",
    "v2_specificity_label",
    "v2_keep_candidate",
    "is_primary_canonical",
    "duplicate_type",
    "duplicate_of_sql_item_id",
    "executable_status_v2",
    "rejected_reason_v2",
    "sql_text_clean",
    "sql_text_raw",
]

QUESTION_SEED_FIELDNAMES = [
    "seed_id",
    "seed_question_text",
    "source_sql_item_id",
    "family_tag_guess",
    "confidence",
    "notes",
]

FINAL_INDEX_FIELDNAMES = [
    "own_id",
    "dataset_id",
    "dataset_name",
    "class_type",
    "source_type",
    "readiness_label_v2",
    "strict_keep_count_v2",
    "strict_keep_ratio_v2",
    "collision_ratio_v2",
    "duplicate_burden_v2",
    "executability_pass_ratio_v2",
    "source_credibility_score_v2",
    "sql_inventory_row_count_v2",
    "sql_inventory_primary_count_v2",
    "taxonomy_candidate_count_v2",
    "rejected_sql_count_v2",
    "top_strict_sql_count_v2",
    "question_seed_count_v2",
    "source_url_count_v2",
    "official_dataset_link",
    "best_sql_source_url_v2",
    "gate_reason_codes_v2",
    "recommended_next_action",
    "dataset_card_v2_path",
    "sql_inventory_v2_path",
    "top_strict_sql_v2_path",
    "rejected_sql_v2_path",
    "question_seed_candidates_v2_path",
]

MORNING_REVIEW_FIELDNAMES = [
    "own_id",
    "dataset_name",
    "readiness_label_v2",
    "strict_keep_count_v2",
    "strict_keep_ratio_v2",
    "collision_ratio_v2",
    "duplicate_burden_v2",
    "executability_pass_ratio_v2",
    "source_credibility_score_v2",
    "top_strict_sql_count_v2",
    "question_seed_count_v2",
    "source_url_count_v2",
    "recommended_next_action",
]

READINESS_SORT = {
    "READY": 0,
    "READY_WITH_WARNINGS": 1,
    "NOT_READY": 2,
}
SOURCE_CREDIBILITY_LABELS = {
    "tier_1_official": "official",
    "tier_2_primary_code": "primary_code",
    "tier_3_secondary_explanatory": "secondary_explanatory",
    "tier_4_low_trust": "low_trust",
}
RISK_REASON_TEXT = {
    "ready_thresholds_met": "All strict gate thresholds were met.",
    "warning_thresholds_met": "Only the warning thresholds were met.",
    "no_sql_rows_v2": "No V2 SQL rows exist for this dataset.",
    "no_strict_kept_sql": "No strict kept canonical SQL remains after V2 cleaning.",
    "strict_kept_sql_too_sparse": "The surviving strict SQL core is too small for taxonomy work.",
    "low_strict_keep_purity": "The kept primary inventory is not dominated by strict rows.",
    "high_collision_contamination": "Collision-risk SQL still dominates the primary inventory.",
    "moderate_collision_contamination": "Collision-risk SQL remains material and needs manual review.",
    "high_duplicate_burden": "Too much of the inventory collapses to duplicates.",
    "moderate_duplicate_burden": "Duplicate burden remains elevated.",
    "low_strict_executability_pass_ratio": "Too few strict rows pass the lightweight executability screen.",
    "borderline_strict_executability_pass_ratio": "Executability is usable but still borderline.",
    "low_source_credibility": "The surviving kept SQL comes from weak sources.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild per-dataset V2 final assets and global final_v2 review tables "
            "from the cleaned execute inventory and readiness gate."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--global-output-dir", type=Path, default=None)
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE_CSV)
    parser.add_argument("--reclassify-csv", type=Path, default=DEFAULT_RECLASSIFY_CSV)
    parser.add_argument("--dedup-csv", type=Path, default=DEFAULT_DEDUP_CSV)
    parser.add_argument("--execute-csv", type=Path, default=DEFAULT_EXECUTE_CSV)
    parser.add_argument("--gate-csv", type=Path, default=DEFAULT_GATE_CSV)
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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def dataset_purpose_text(scope_row: dict[str, str]) -> str:
    class_type = (scope_row.get("class_type") or "").strip().lower()
    source_type = (scope_row.get("source_type") or "").strip()
    if class_type == "categorical":
        task_phrase = "categorical-label tabular analysis and classification-style SQL reasoning"
    elif class_type == "numerical":
        task_phrase = "numeric or regression-style metric analysis"
    elif class_type == "mixed":
        task_phrase = "mixed-feature SQL analysis across categorical and numeric fields"
    else:
        task_phrase = "tabular SQL analysis"
    return (
        f"This `sql_high` dataset is packaged as a V2-cleaned {task_phrase} review candidate "
        f"from `{source_type}`. The package keeps explicit source links, V2 specificity labels, "
        "dedup annotations, and lightweight executability estimates for question-taxonomy review."
    )


def resolve_source_url(row: dict[str, str]) -> str:
    return (row.get("source_url") or "").strip() or (row.get("source_seed_url") or "").strip()


def resolved_inventory_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    resolved: list[dict[str, str]] = []
    for row in rows:
        current = dict(row)
        current["source_url"] = resolve_source_url(row)
        if not (current.get("source_title") or "").strip():
            current["source_title"] = (row.get("source_seed_title") or "").strip()
        if not (current.get("source_type") or "").strip():
            current["source_type"] = (row.get("source_seed_type") or "").strip()
        resolved.append(current)
    return resolved


def taxonomy_candidate(row: dict[str, str]) -> bool:
    if (row.get("is_primary_canonical") or "").strip() != "yes":
        return False
    if (row.get("v2_keep_candidate") or "").strip() != "yes":
        return False
    if (row.get("v2_specificity_label") or "").strip() != "strict":
        return False
    if (row.get("executable_status_v2") or "").strip() == "fail":
        return False
    if not resolve_source_url(row):
        return False
    clean_sql = (row.get("sql_text_clean") or "").strip() or (row.get("sql_text_raw") or "").strip()
    if not clean_sql:
        return False
    if (row.get("query_intent_label") or "").strip() == "session_use":
        return False
    return True


def selection_score(row: dict[str, str]) -> int:
    exec_score = {"pass": 6, "unknown": 2, "fail": -6}
    confidence_score = {"high": 4, "medium": 2, "low": 0}
    complexity_score = {"advanced": 3, "moderate": 3, "simple": 1}
    family_score = {
        "aggregation": 4,
        "join_analysis": 4,
        "window_analytics": 4,
        "filtering": 3,
        "basic_projection": 2,
        "schema_ddl": 1,
        "misc_sql": 1,
    }
    tier_score = {
        "tier_1_official": 5,
        "tier_2_primary_code": 4,
        "tier_3_secondary_explanatory": 2,
        "tier_4_low_trust": -2,
    }
    query_score = {
        "dml_select": 4,
        "ddl_create_table": 1,
        "ddl_create_view": 1,
        "dml_update": 1,
        "dml_delete": 0,
        "dml_insert": 0,
        "other_sql": 0,
    }
    text_length = len((row.get("sql_text_clean") or "").strip())
    length_bonus = 0
    if 40 <= text_length <= 2500:
        length_bonus = 2
    elif 2500 < text_length <= 9000:
        length_bonus = 1
    elif text_length > 20000:
        length_bonus = -2
    return (
        exec_score.get((row.get("executable_status_v2") or "").strip(), 0)
        + confidence_score.get((row.get("evidence_confidence") or "").strip(), 0)
        + complexity_score.get((row.get("sql_complexity") or "").strip(), 0)
        + family_score.get((row.get("family_tag_guess") or "").strip(), 0)
        + tier_score.get((row.get("v2_source_credibility_tier") or "").strip(), 0)
        + query_score.get((row.get("query_intent_label") or "").strip(), 0)
        + length_bonus
    )


def best_sql_sources(rows: list[dict[str, str]], limit: int = 3) -> list[dict[str, str]]:
    if not rows:
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            -selection_score(row),
            normalize_url_root(resolve_source_url(row)),
            (row.get("sql_item_id") or "").strip(),
        ),
    )
    selected: list[dict[str, str]] = []
    seen_roots: set[str] = set()
    for row in ranked:
        url = resolve_source_url(row)
        if not url:
            continue
        root = normalize_url_root(url)
        if root in seen_roots:
            continue
        selected.append(
            {
                "url": url,
                "root": root,
                "title": (row.get("source_title") or "").strip() or url,
                "note": (
                    f"sql_item_id={((row.get('sql_item_id') or row.get('source_sql_item_id') or '').strip())}; "
                    f"exec={(row.get('executable_status_v2') or '').strip() or 'unknown'}; "
                    f"cred={(row.get('v2_source_credibility_tier') or '').strip() or 'unknown'}"
                ),
            }
        )
        seen_roots.add(root)
        if len(selected) >= limit:
            break
    return selected


def curated_top_strict_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    candidate_rows = [row for row in rows if taxonomy_candidate(row)]
    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            -selection_score(row),
            normalize_url_root(resolve_source_url(row)),
            (row.get("sql_item_id") or "").strip(),
        ),
    )
    selected: list[dict[str, Any]] = []
    per_root_counter: Counter[str] = Counter()
    per_family_counter: Counter[str] = Counter()
    for row in ranked:
        if len(selected) >= TOP_STRICT_LIMIT:
            break
        root = normalize_url_root(resolve_source_url(row))
        family = (row.get("family_tag_guess") or "").strip() or "unknown"
        if per_root_counter[root] >= 5:
            continue
        if per_family_counter[family] >= 4 and len(ranked) > TOP_STRICT_LIMIT:
            continue
        selected.append(
            {
                "selection_rank": len(selected) + 1,
                "own_id": (row.get("own_id") or "").strip(),
                "dataset_id": (row.get("dataset_id") or "").strip(),
                "dataset_name": (row.get("dataset_name") or "").strip(),
                "source_sql_item_id": (row.get("sql_item_id") or "").strip(),
                "source_url": resolve_source_url(row),
                "source_root": root,
                "source_type": (row.get("source_type") or "").strip(),
                "source_title": (row.get("source_title") or "").strip(),
                "query_intent_label": (row.get("query_intent_label") or "").strip(),
                "family_tag_guess": (row.get("family_tag_guess") or "").strip(),
                "sql_complexity": (row.get("sql_complexity") or "").strip(),
                "evidence_confidence": (row.get("evidence_confidence") or "").strip(),
                "v2_source_credibility_tier": (row.get("v2_source_credibility_tier") or "").strip(),
                "executable_status_v2": (row.get("executable_status_v2") or "").strip(),
                "canonical_group_id": (row.get("canonical_group_id") or "").strip(),
                "sql_fingerprint_v2": (row.get("sql_fingerprint_v2") or "").strip(),
                "sql_text_clean": (row.get("sql_text_clean") or "").strip(),
                "sql_text_raw": (row.get("sql_text_raw") or "").strip(),
                "selection_notes": (
                    "V2 Phase 4 curated strict SQL. "
                    f"selection_score={selection_score(row)}; "
                    f"readiness_candidate=yes; "
                    f"exec={(row.get('executable_status_v2') or '').strip()}; "
                    f"cred={(row.get('v2_source_credibility_tier') or '').strip()}"
                ),
            }
        )
        per_root_counter[root] += 1
        per_family_counter[family] += 1
    return selected


def clean_comment_text(prefix: str) -> str:
    text = prefix or ""
    text = re.sub(r"(?s)^/\*+|\*/$", "", text).strip()
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        stripped = re.sub(r"^(--+|#+|\*+)\s*", "", stripped)
        stripped = normalize_whitespace(stripped)
        if stripped:
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
    if not comment or len(comment) < 12:
        return ""
    if any(token in comment.lower() for token in ("phpmyadmin sql dump", "step 1:", "connected to a transient")):
        return ""
    if "?" not in comment and comment and comment[0].islower():
        comment = comment[0].upper() + comment[1:]
    if "?" not in comment and comment.lower().startswith(("find ", "calculate ", "count ", "check ", "show ", "create ", "insert ", "update ", "load ")):
        return comment.rstrip(".") + "."
    if "?" not in comment:
        return comment.rstrip(".") + "?"
    return comment


def group_by_columns(sql_text: str) -> str:
    match = re.search(
        r"(?is)\bgroup\s+by\b\s+(.*?)(?:\border\s+by\b|\bhaving\b|\blimit\b|;|$)",
        sql_text or "",
    )
    if not match:
        return ""
    parts = [
        normalize_whitespace(part)
        for part in match.group(1).split(",")
        if normalize_whitespace(part)
    ]
    if len(parts) > 3:
        parts = parts[:3]
    return ", ".join(parts)[:80]


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
        return f"Create the {primary_table} table used by the source SQL."
    if query_intent == "ddl_create_view" and tables:
        return f"Create the derived view built from {table_list}."
    if query_intent == "dml_select":
        if family == "aggregation" or aggs:
            if len(aggs) == 1 and aggs[0] == "COUNT" and not group_cols:
                return f"Count rows in {primary_table}."
            agg_clause = ", ".join(aggs) if aggs else "the requested aggregates"
            text = f"On {primary_table}, compute {agg_clause}"
            if group_cols:
                text += f" grouped by `{group_cols}`"
            if has_where:
                text += " with the source filters applied"
            return text + "."
        if family == "join_analysis" and len(tables) >= 2:
            return f"Join {table_list} and reproduce the analytical result in the source SQL."
        if family == "filtering":
            return f"Filter {primary_table} with the source conditions and return the requested result."
        return f"Query {primary_table} to reproduce the result pattern in the source SQL."
    if query_intent == "dml_update" and tables:
        return f"Update {primary_table} according to the transformation logic in the source SQL."
    if query_intent == "dml_delete" and tables:
        return f"Delete rows from {primary_table} using the source SQL conditions."
    if query_intent == "dml_insert" and tables and len(sql_text) <= 3000:
        return f"Insert the source rows into {primary_table}."
    if tables:
        return f"Reproduce the SQL logic from `{row.get('source_sql_item_id')}` using {table_list}."
    return f"Reproduce the SQL logic from `{row.get('source_sql_item_id')}`."


def seed_candidate_score(row: dict[str, Any]) -> int:
    query_intent = (row.get("query_intent_label") or "").strip()
    family = (row.get("family_tag_guess") or "").strip()
    exec_score = {"pass": 4, "unknown": 1, "fail": -5}
    score = exec_score.get((row.get("executable_status_v2") or "").strip(), 0)
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


def question_seeds_from_top_strict(
    *,
    own_id: str,
    readiness_label: str,
    top_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if readiness_label not in {"READY", "READY_WITH_WARNINGS"}:
        return []

    seeds: list[dict[str, str]] = []
    for row in sorted(top_rows, key=lambda item: (-seed_candidate_score(item), int(item["selection_rank"]))):
        if len(seeds) >= QUESTION_SEED_LIMIT:
            break
        if (row.get("query_intent_label") or "").strip() == "session_use":
            continue
        question_text = leading_comment_question(row.get("sql_text_raw") or "")
        note_parts = ["Derived from V2 curated strict SQL."]
        if question_text:
            note_parts.append("Seed text sourced from leading SQL comments.")
        else:
            question_text = heuristic_seed_text(row)
            note_parts.append("Seed text generated from conservative SQL heuristics.")
        if readiness_label == "READY_WITH_WARNINGS":
            note_parts.append("Dataset is only READY_WITH_WARNINGS; manual review required.")
        question_text = normalize_whitespace(question_text)
        if not question_text:
            continue
        if len(question_text) > 240:
            question_text = question_text[:237].rstrip() + "..."
        exec_status = (row.get("executable_status_v2") or "").strip()
        confidence = "high" if exec_status == "pass" and (row.get("evidence_confidence") or "").strip() == "high" else "medium"
        seed_index = len(seeds) + 1
        seeds.append(
            {
                "seed_id": f"{own_id}_v2_seed_{seed_index:03d}",
                "seed_question_text": question_text,
                "source_sql_item_id": row.get("source_sql_item_id") or "",
                "family_tag_guess": row.get("family_tag_guess") or "",
                "confidence": confidence,
                "notes": " ".join(note_parts),
            }
        )
    return seeds


def rejected_reason(row: dict[str, str]) -> str:
    reasons: list[str] = []
    if (row.get("is_primary_canonical") or "").strip() != "yes":
        reasons.append("non_primary_duplicate")
    if (row.get("v2_keep_candidate") or "").strip() != "yes":
        reasons.append("v2_keep_candidate_no")
    label = (row.get("v2_specificity_label") or "").strip()
    if label == "reject_non_sql":
        reasons.append("reject_non_sql")
    elif label != "strict":
        reasons.append("not_strict")
    if (row.get("executable_status_v2") or "").strip() == "fail":
        reasons.append("executability_fail")
    if (row.get("query_intent_label") or "").strip() == "session_use":
        reasons.append("session_use_not_taxonomy_ready")
    if not resolve_source_url(row):
        reasons.append("missing_source_url")
    if not ((row.get("sql_text_clean") or "").strip() or (row.get("sql_text_raw") or "").strip()):
        reasons.append("empty_sql_text")
    if not reasons:
        reasons.append("not_selected_for_top_strict")
    return ",".join(reasons)


def humanize_reasons(codes: str) -> list[str]:
    parts = [part.strip() for part in codes.split(",") if part.strip()]
    return [RISK_REASON_TEXT.get(part, part.replace("_", " ")) for part in parts]


def downstream_richness_text(readiness_label: str, strict_keep_count: int, top_count: int, seed_count: int) -> str:
    if readiness_label == "READY":
        return (
            f"This dataset is downstream-task-rich in V2: {strict_keep_count} strict kept canonical SQL rows survived cleaning, "
            f"{top_count} were promoted into the review package, and {seed_count} seed questions were generated."
        )
    if readiness_label == "READY_WITH_WARNINGS":
        return (
            f"This dataset has a usable downstream-task core, but it still needs human caution: "
            f"{strict_keep_count} strict kept rows survived and {seed_count} seeds were generated, yet warning thresholds remain active."
        )
    if strict_keep_count > 0:
        return (
            f"This dataset has some V2-cleaned strict SQL, but it is not yet downstream-task-ready under the strict gate. "
            f"Only {strict_keep_count} strict kept rows survived, which was not enough for a clean taxonomy package."
        )
    return "This dataset is not downstream-task-rich in V2 because no strict kept SQL core survived cleaning."


def build_dataset_card(
    *,
    scope_row: dict[str, str],
    gate_row: dict[str, str],
    inventory_rows: list[dict[str, str]],
    top_rows: list[dict[str, Any]],
    question_seeds: list[dict[str, str]],
    best_sources: list[dict[str, str]],
) -> str:
    dataset_name = scope_row.get("dataset_name") or ""
    own_id = scope_row.get("own_id") or ""
    readiness_label = gate_row.get("readiness_label_v2") or "NOT_READY"
    strict_keep_count = int(gate_row.get("strict_keep_count_v2") or 0)
    weak_keep_count = int(gate_row.get("weak_keep_count_v2") or 0)
    collision_primary_count = int(gate_row.get("collision_primary_count_v2") or 0)
    primary_rows = int(gate_row.get("primary_sql_rows_v2") or 0)
    duplicate_dropped = int(gate_row.get("duplicate_dropped_count_v2") or 0)
    keep_candidate_primary = int(gate_row.get("keep_candidate_primary_rows_v2") or 0)
    label_counts = Counter((row.get("v2_specificity_label") or "").strip() for row in inventory_rows)
    exec_counts = Counter((row.get("executable_status_v2") or "").strip() for row in inventory_rows)
    reject_count = label_counts.get("reject_non_sql", 0)
    source_url_count = len({resolve_source_url(row) for row in inventory_rows if resolve_source_url(row)})
    reasons = humanize_reasons(gate_row.get("gate_reason_codes_v2") or "")
    official_link = (scope_row.get("dataset_link") or "").strip()
    if readiness_label == "READY":
        go_no_go = "GO"
    elif readiness_label == "READY_WITH_WARNINGS":
        go_no_go = "GO_WITH_WARNINGS"
    else:
        go_no_go = "NO_GO"

    lines = [
        f"# Dataset Card V2: {dataset_name} (`{own_id}`)",
        "",
        "## What This Dataset Is For",
        "",
        dataset_purpose_text(scope_row),
        "",
        "## Why Downstream-Task-Rich Or Not",
        "",
        downstream_richness_text(readiness_label, strict_keep_count, len(top_rows), len(question_seeds)),
        "",
        "## V2 Readiness Snapshot",
        "",
        f"- Readiness label: `{readiness_label}`",
        f"- Strict kept canonical SQL rows: {strict_keep_count}",
        f"- Top strict SQL rows packaged in `top_strict_sql_v2.csv`: {len(top_rows)}",
        f"- Question seed candidates packaged: {len(question_seeds)}",
        f"- Recommended next action: {gate_row.get('recommended_next_action') or ''}",
        "",
        "## V2 SQL Evidence Summary",
        "",
        f"- Total inventory rows in `sql_inventory_v2.csv`: {len(inventory_rows)}",
        f"- Primary canonical rows after dedup: {primary_rows}",
        f"- Duplicate rows removed from the primary view: {duplicate_dropped}",
        f"- Kept primary rows after V2 relabeling: {keep_candidate_primary}",
        f"- V2 specificity counts: `strict={strict_keep_count}`, `weak={weak_keep_count}`, `collision_risk={collision_primary_count}`, `reject_non_sql={reject_count}`",
        f"- Executability counts across all V2 rows: `pass={exec_counts.get('pass', 0)}`, `fail={exec_counts.get('fail', 0)}`, `unknown={exec_counts.get('unknown', 0)}`",
        f"- Strict keep ratio: {gate_row.get('strict_keep_ratio_v2') or '0'}",
        f"- Collision ratio: {gate_row.get('collision_ratio_v2') or '0'}",
        f"- Duplicate burden: {gate_row.get('duplicate_burden_v2') or '0'} ({duplicate_dropped} duplicates removed from {len(inventory_rows)} total rows)",
        f"- Executability pass ratio: {gate_row.get('executability_pass_ratio_v2') or '0'}",
        f"- Source credibility score: {gate_row.get('source_credibility_score_v2') or '0'}",
        f"- Distinct source URL count in V2 inventory: {source_url_count}",
        f"- `sql_inventory_v2.csv` preserves explicit `source_url` values for every row.",
        "",
        "## Go / No-Go Recommendation",
        "",
        f"- Recommendation: `{go_no_go}`",
        f"- Rationale: {gate_row.get('recommended_next_action') or ''}",
        "",
        "## Key Links",
        "",
    ]

    if official_link:
        lines.append(f"- Official dataset link: [{official_link}]({official_link})")
    else:
        lines.append("- Official dataset link: none recorded in scope metadata.")

    if best_sources:
        lines.append("- Trusted V2 SQL sources:")
        for source in best_sources:
            lines.append(f"- [{source['title']}]({source['url']}) ({source['note']})")
    else:
        lines.append("- Trusted V2 SQL sources: none survived the V2 candidate filter.")

    lines.extend(["", "## Known Risks", ""])
    if reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- No active V2 risk codes were recorded.")
    if readiness_label != "READY":
        lines.append("- This dataset should not be treated as fully taxonomy-ready without the noted remediation.")
    lines.append("")
    return "\n".join(lines)


def render_final_overview(
    *,
    now_utc: str,
    final_rows: list[dict[str, str]],
    label_counts: Counter[str],
    total_inventory_rows: int,
    total_top_rows: int,
    total_seed_rows: int,
    total_rejected_rows: int,
) -> str:
    ready_rows = [row for row in final_rows if row["readiness_label_v2"] == "READY"]
    warning_rows = [row for row in final_rows if row["readiness_label_v2"] == "READY_WITH_WARNINGS"]
    not_ready_rows = [row for row in final_rows if row["readiness_label_v2"] == "NOT_READY"]

    lines = [
        "# Final Overview V2",
        "",
        f"- Generated at UTC: `{now_utc}`",
        f"- Total datasets packaged: {len(final_rows)}",
        f"- Total V2 inventory rows packaged: {total_inventory_rows}",
        f"- Total V2 top strict SQL rows packaged: {total_top_rows}",
        f"- Total V2 rejected rows packaged: {total_rejected_rows}",
        f"- Total V2 question seed rows packaged: {total_seed_rows}",
        f"- `READY`: {label_counts['READY']}",
        f"- `READY_WITH_WARNINGS`: {label_counts['READY_WITH_WARNINGS']}",
        f"- `NOT_READY`: {label_counts['NOT_READY']}",
        "",
        "## Ready Datasets",
        "",
    ]
    if ready_rows:
        for row in sorted(ready_rows, key=lambda item: (-int(item["strict_keep_count_v2"]), item["own_id"])):
            lines.append(
                f"- `{row['own_id']}` {row['dataset_name']}: strict_keep={row['strict_keep_count_v2']}, top_strict={row['top_strict_sql_count_v2']}, seeds={row['question_seed_count_v2']}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Ready With Warnings", ""])
    if warning_rows:
        for row in sorted(warning_rows, key=lambda item: (-int(item["strict_keep_count_v2"]), item["own_id"])):
            lines.append(
                f"- `{row['own_id']}` {row['dataset_name']}: reasons=`{row['gate_reason_codes_v2']}`; next=`{row['recommended_next_action']}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Not Ready Highlights", ""])
    for row in sorted(not_ready_rows, key=lambda item: (-int(item["strict_keep_count_v2"]), item["own_id"]))[:12]:
        lines.append(
            f"- `{row['own_id']}` {row['dataset_name']}: strict_keep={row['strict_keep_count_v2']}, label=`{row['readiness_label_v2']}`, next=`{row['recommended_next_action']}`"
        )

    return "\n".join(lines)


def render_morning_risks(rows: list[dict[str, str]]) -> str:
    collision_sorted = sorted(rows, key=lambda row: (-float(row["collision_ratio_v2"]), row["own_id"]))
    duplicate_sorted = sorted(rows, key=lambda row: (-float(row["duplicate_burden_v2"]), row["own_id"]))
    sparse_sorted = sorted(
        [row for row in rows if int(row["strict_keep_count_v2"]) > 0],
        key=lambda row: (int(row["strict_keep_count_v2"]), row["own_id"]),
    )
    no_sql = [row for row in rows if int(row["sql_inventory_row_count_v2"]) == 0]

    lines = [
        "# Morning Review Risks V2",
        "",
        "## Highest Collision Ratios",
        "",
    ]
    for row in collision_sorted[:10]:
        lines.append(
            f"- `{row['own_id']}` {row['dataset_name']}: collision_ratio={row['collision_ratio_v2']}, next=`{row['recommended_next_action']}`"
        )

    lines.extend(["", "## Highest Duplicate Burden", ""])
    for row in duplicate_sorted[:10]:
        lines.append(
            f"- `{row['own_id']}` {row['dataset_name']}: duplicate_burden={row['duplicate_burden_v2']}, top_strict={row['top_strict_sql_count_v2']}"
        )

    lines.extend(["", "## Smallest Surviving Strict Cores", ""])
    if sparse_sorted:
        for row in sparse_sorted[:10]:
            lines.append(
                f"- `{row['own_id']}` {row['dataset_name']}: strict_keep={row['strict_keep_count_v2']}, readiness=`{row['readiness_label_v2']}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Datasets With No V2 SQL Rows", ""])
    if no_sql:
        for row in no_sql:
            lines.append(f"- `{row['own_id']}` {row['dataset_name']}: {row['recommended_next_action']}")
    else:
        lines.append("- None")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    dataset_root = (args.dataset_root.resolve() if args.dataset_root else (output_root / "datasets").resolve())
    global_output_dir = (
        args.global_output_dir.resolve()
        if args.global_output_dir
        else (output_root / "final_v2").resolve()
    )
    scope_csv = args.scope_csv.resolve()
    reclassify_csv = args.reclassify_csv.resolve()
    dedup_csv = args.dedup_csv.resolve()
    execute_csv = args.execute_csv.resolve()
    gate_csv = args.gate_csv.resolve()

    scope_rows = read_csv_rows(scope_csv)
    reclassify_rows = read_csv_rows(reclassify_csv)
    dedup_rows = read_csv_rows(dedup_csv)
    gate_rows = read_csv_rows(gate_csv)
    execute_rows = resolved_inventory_rows(read_csv_rows(execute_csv))

    def inventory_key_set(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
        return {
            (
                (row.get("own_id") or "").strip(),
                (row.get("sql_item_id") or "").strip(),
            )
            for row in rows
        }

    reclassify_keys = inventory_key_set(reclassify_rows)
    dedup_keys = inventory_key_set(dedup_rows)
    execute_keys = inventory_key_set(execute_rows)
    if reclassify_keys != execute_keys:
        raise RuntimeError(
            "Phase 4 expected reclassify and execute inventories to describe the same SQL row keys, "
            f"but found reclassify={len(reclassify_keys)} keys and execute={len(execute_keys)} keys."
        )
    if dedup_keys != execute_keys:
        raise RuntimeError(
            "Phase 4 expected dedup and execute inventories to describe the same SQL row keys, "
            f"but found dedup={len(dedup_keys)} keys and execute={len(execute_keys)} keys."
        )

    scope_by_id = {(row.get("own_id") or "").strip(): row for row in scope_rows}
    gate_by_id = {(row.get("own_id") or "").strip(): row for row in gate_rows}
    execute_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in execute_rows:
        execute_by_id[(row.get("own_id") or "").strip()].append(row)

    execute_fieldnames = list(read_csv_rows(execute_csv)[0].keys()) if execute_rows else []

    final_rows: list[dict[str, str]] = []
    morning_rows: list[dict[str, str]] = []
    label_counts: Counter[str] = Counter()
    total_inventory_rows = 0
    total_top_rows = 0
    total_seed_rows = 0
    total_rejected_rows = 0

    missing_source_rows: list[tuple[str, str]] = []

    for scope_row in scope_rows:
        own_id = (scope_row.get("own_id") or "").strip()
        dataset_id = (scope_row.get("dataset_id") or "").strip()
        dataset_name = (scope_row.get("dataset_name") or "").strip()
        gate_row = gate_by_id.get(own_id, {})
        inventory_rows = execute_by_id.get(own_id, [])

        for row in inventory_rows:
            if not resolve_source_url(row):
                missing_source_rows.append((own_id, (row.get("sql_item_id") or "").strip()))

        dataset_v2_dir = dataset_root / own_id / "v2"
        sql_inventory_v2_path = dataset_v2_dir / "sql_inventory_v2.csv"
        top_strict_v2_path = dataset_v2_dir / "top_strict_sql_v2.csv"
        rejected_v2_path = dataset_v2_dir / "rejected_sql_v2.csv"
        dataset_card_v2_path = dataset_v2_dir / "dataset_card_v2.md"
        question_seed_v2_path = dataset_v2_dir / "question_seed_candidates_v2.csv"

        top_rows = curated_top_strict_rows(inventory_rows)
        taxonomy_candidates = {row["source_sql_item_id"] for row in top_rows}
        rejected_rows = [
            {
                "own_id": (row.get("own_id") or "").strip(),
                "dataset_id": (row.get("dataset_id") or "").strip(),
                "dataset_name": (row.get("dataset_name") or "").strip(),
                "sql_item_id": (row.get("sql_item_id") or "").strip(),
                "source_url": resolve_source_url(row),
                "source_type": (row.get("source_type") or "").strip(),
                "source_title": (row.get("source_title") or "").strip(),
                "v2_specificity_label": (row.get("v2_specificity_label") or "").strip(),
                "v2_keep_candidate": (row.get("v2_keep_candidate") or "").strip(),
                "is_primary_canonical": (row.get("is_primary_canonical") or "").strip(),
                "duplicate_type": (row.get("duplicate_type") or "").strip(),
                "duplicate_of_sql_item_id": (row.get("duplicate_of_sql_item_id") or "").strip(),
                "executable_status_v2": (row.get("executable_status_v2") or "").strip(),
                "rejected_reason_v2": rejected_reason(row),
                "sql_text_clean": (row.get("sql_text_clean") or "").strip(),
                "sql_text_raw": (row.get("sql_text_raw") or "").strip(),
            }
            for row in inventory_rows
            if (row.get("sql_item_id") or "").strip() not in taxonomy_candidates
        ]

        readiness_label = (gate_row.get("readiness_label_v2") or "NOT_READY").strip()
        question_seeds = question_seeds_from_top_strict(
            own_id=own_id,
            readiness_label=readiness_label,
            top_rows=top_rows,
        )
        best_sources = best_sql_sources(top_rows if top_rows else inventory_rows)

        write_csv(sql_inventory_v2_path, execute_fieldnames, inventory_rows)
        write_csv(top_strict_v2_path, TOP_STRICT_FIELDNAMES, top_rows)
        write_csv(rejected_v2_path, REJECTED_FIELDNAMES, rejected_rows)
        write_csv(question_seed_v2_path, QUESTION_SEED_FIELDNAMES, question_seeds)
        write_text(
            dataset_card_v2_path,
            build_dataset_card(
                scope_row=scope_row,
                gate_row=gate_row,
                inventory_rows=inventory_rows,
                top_rows=top_rows,
                question_seeds=question_seeds,
                best_sources=best_sources,
            ),
        )

        source_url_count = len({resolve_source_url(row) for row in inventory_rows if resolve_source_url(row)})
        best_sql_source_url = best_sources[0]["url"] if best_sources else ""
        primary_count = sum(1 for row in inventory_rows if (row.get("is_primary_canonical") or "").strip() == "yes")
        taxonomy_candidate_count = sum(1 for row in inventory_rows if taxonomy_candidate(row))

        final_row = {
            "own_id": own_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "class_type": (scope_row.get("class_type") or "").strip(),
            "source_type": (scope_row.get("source_type") or "").strip(),
            "readiness_label_v2": readiness_label,
            "strict_keep_count_v2": (gate_row.get("strict_keep_count_v2") or "0").strip(),
            "strict_keep_ratio_v2": (gate_row.get("strict_keep_ratio_v2") or "0").strip(),
            "collision_ratio_v2": (gate_row.get("collision_ratio_v2") or "0").strip(),
            "duplicate_burden_v2": (gate_row.get("duplicate_burden_v2") or "0").strip(),
            "executability_pass_ratio_v2": (gate_row.get("executability_pass_ratio_v2") or "0").strip(),
            "source_credibility_score_v2": (gate_row.get("source_credibility_score_v2") or "0").strip(),
            "sql_inventory_row_count_v2": str(len(inventory_rows)),
            "sql_inventory_primary_count_v2": str(primary_count),
            "taxonomy_candidate_count_v2": str(taxonomy_candidate_count),
            "rejected_sql_count_v2": str(len(rejected_rows)),
            "top_strict_sql_count_v2": str(len(top_rows)),
            "question_seed_count_v2": str(len(question_seeds)),
            "source_url_count_v2": str(source_url_count),
            "official_dataset_link": (scope_row.get("dataset_link") or "").strip(),
            "best_sql_source_url_v2": best_sql_source_url,
            "gate_reason_codes_v2": (gate_row.get("gate_reason_codes_v2") or "").strip(),
            "recommended_next_action": (gate_row.get("recommended_next_action") or "").strip(),
            "dataset_card_v2_path": str(dataset_card_v2_path),
            "sql_inventory_v2_path": str(sql_inventory_v2_path),
            "top_strict_sql_v2_path": str(top_strict_v2_path),
            "rejected_sql_v2_path": str(rejected_v2_path),
            "question_seed_candidates_v2_path": str(question_seed_v2_path),
        }
        final_rows.append(final_row)
        morning_rows.append({field: final_row[field] for field in MORNING_REVIEW_FIELDNAMES})
        label_counts[readiness_label] += 1
        total_inventory_rows += len(inventory_rows)
        total_top_rows += len(top_rows)
        total_seed_rows += len(question_seeds)
        total_rejected_rows += len(rejected_rows)

    if missing_source_rows:
        sample = ", ".join(f"{own_id}:{sql_item_id}" for own_id, sql_item_id in missing_source_rows[:10])
        raise RuntimeError(
            "sql_inventory_v2.csv requires explicit source_url for every row, but some rows still have no source_url. "
            f"Examples: {sample}"
        )

    final_rows.sort(key=lambda row: (READINESS_SORT.get(row["readiness_label_v2"], 9), -int(row["strict_keep_count_v2"]), row["own_id"]))
    morning_rows.sort(key=lambda row: (READINESS_SORT.get(row["readiness_label_v2"], 9), -int(row["strict_keep_count_v2"]), row["own_id"]))

    final_index_v2_path = global_output_dir / "final_index_v2.csv"
    final_overview_v2_path = global_output_dir / "final_overview_v2.md"
    morning_table_v2_path = global_output_dir / "morning_review_dataset_table_v2.csv"
    morning_risks_v2_path = global_output_dir / "morning_review_risks_v2.md"
    manifest_v2_path = global_output_dir / "run_manifest_v2_phase4.json"

    write_csv(final_index_v2_path, FINAL_INDEX_FIELDNAMES, final_rows)
    write_csv(morning_table_v2_path, MORNING_REVIEW_FIELDNAMES, morning_rows)
    write_text(
        final_overview_v2_path,
        render_final_overview(
            now_utc=utc_now_iso(),
            final_rows=final_rows,
            label_counts=label_counts,
            total_inventory_rows=total_inventory_rows,
            total_top_rows=total_top_rows,
            total_seed_rows=total_seed_rows,
            total_rejected_rows=total_rejected_rows,
        ),
    )
    write_text(morning_risks_v2_path, render_morning_risks(final_rows))

    outputs = [
        final_index_v2_path,
        final_overview_v2_path,
        morning_table_v2_path,
        morning_risks_v2_path,
        manifest_v2_path,
    ]

    manifest_payload = {
        "phase": "v2_phase4_final_review_package_rebuild",
        "generated_at_utc": utc_now_iso(),
        "input": {
            "scope_csv_path": str(scope_csv),
            "scope_csv_sha256": sha256_file(scope_csv),
            "reclassify_csv_path": str(reclassify_csv),
            "reclassify_csv_sha256": sha256_file(reclassify_csv),
            "dedup_csv_path": str(dedup_csv),
            "dedup_csv_sha256": sha256_file(dedup_csv),
            "execute_csv_path": str(execute_csv),
            "execute_csv_sha256": sha256_file(execute_csv),
            "gate_csv_path": str(gate_csv),
            "gate_csv_sha256": sha256_file(gate_csv),
            "dataset_root": str(dataset_root),
            "global_output_dir": str(global_output_dir),
        },
        "summary": {
            "dataset_count": len(final_rows),
            "total_inventory_rows_v2": total_inventory_rows,
            "total_top_strict_rows_v2": total_top_rows,
            "total_rejected_rows_v2": total_rejected_rows,
            "total_question_seed_rows_v2": total_seed_rows,
            "ready_count": label_counts["READY"],
            "ready_with_warnings_count": label_counts["READY_WITH_WARNINGS"],
            "not_ready_count": label_counts["NOT_READY"],
        },
        "outputs": [],
    }
    write_json(manifest_v2_path, manifest_payload)
    manifest_payload["outputs"] = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in outputs
    ]
    write_json(manifest_v2_path, manifest_payload)

    for row in final_rows:
        print(
            f"{row['own_id']}\t{row['readiness_label_v2']}\tstrict_keep={row['strict_keep_count_v2']}\ttop_strict={row['top_strict_sql_count_v2']}\tseeds={row['question_seed_count_v2']}"
        )


if __name__ == "__main__":
    main()
