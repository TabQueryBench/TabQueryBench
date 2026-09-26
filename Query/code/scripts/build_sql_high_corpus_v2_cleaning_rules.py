#!/usr/bin/env python3
"""Build a deterministic V2 cleaning rulebook for sql_high corpus refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_phase_c_sql_inventory import (  # noqa: E402
    APP_REPO_PATTERNS,
    APP_SCHEMA_TOKENS,
    CODE_FRAGMENT_PATTERNS,
    FOREIGN_DATASET_PATTERNS,
    GENERIC_SOURCE_PATTERNS,
    NONSTANDARD_QUERY_PATTERNS,
    TOKEN_STOPWORDS,
)


DEFAULT_BASELINE_ROOT = Path("logs/sql_high_corpus_build_20260404")
DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404/v2_refinement")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create machine-readable and markdown V2 cleaning rules for the "
            "sql_high corpus refinement workspace."
        )
    )
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_rulebook(baseline_root: Path) -> dict[str, Any]:
    source_credibility_tiers = [
        {
            "tier_id": "tier_1_official",
            "priority": 1,
            "label": "official_or_platform_primary",
            "assign_if_any": [
                {
                    "id": "official_dataset_page",
                    "all": [
                        {"field": "http_status", "op": "eq", "value": "200"},
                        {
                            "field": "source_type",
                            "op": "in",
                            "value": [
                                "official_dataset_page",
                                "official_api",
                                "openml_api",
                                "openml_task_page",
                                "kaggle_overview_page",
                                "kaggle_data_page",
                            ],
                        },
                    ],
                },
                {
                    "id": "official_domain_fallback",
                    "all": [
                        {"field": "http_status", "op": "eq", "value": "200"},
                        {
                            "field": "source_url_domain",
                            "op": "in",
                            "value": [
                                "archive.ics.uci.edu",
                                "www.openml.org",
                                "openml.org",
                                "www.kaggle.com",
                                "kaggle.com",
                                "huggingface.co",
                            ],
                        },
                    ],
                },
            ],
            "default_reason_code": "official_or_platform_primary_source",
        },
        {
            "tier_id": "tier_2_primary_code",
            "priority": 2,
            "label": "primary_code_or_primary_artifact",
            "assign_if_any": [
                {
                    "id": "code_repository_source",
                    "all": [
                        {"field": "http_status", "op": "eq", "value": "200"},
                        {
                            "field": "source_type",
                            "op": "in",
                            "value": [
                                "github_file",
                                "github_repo",
                                "gist",
                                "kaggle_code_or_notebook",
                                "kaggle_code_page",
                                "paper",
                                "supplement",
                            ],
                        },
                        {"field": "flag_generic_course_or_tutorial_source", "op": "eq", "value": False},
                    ],
                }
            ],
            "default_reason_code": "primary_code_or_primary_research_artifact",
        },
        {
            "tier_id": "tier_3_secondary_explanatory",
            "priority": 3,
            "label": "secondary_explanatory_or_derived_but_usable",
            "assign_if_any": [
                {
                    "id": "secondary_sql_source",
                    "all": [
                        {"field": "http_status", "op": "eq", "value": "200"},
                        {
                            "field": "source_type",
                            "op": "in",
                            "value": [
                                "tutorial_blog",
                                "readme_or_metadata",
                                "paper",
                                "supplement",
                                "unknown",
                            ],
                        },
                        {"field": "has_explicit_sql_text", "op": "eq", "value": True},
                    ],
                }
            ],
            "default_reason_code": "secondary_but_explicit_sql_source",
        },
        {
            "tier_id": "tier_4_low_trust",
            "priority": 4,
            "label": "low_trust_unresolved_or_generic",
            "assign_if_any": [
                {"id": "http_non_200", "all": [{"field": "http_status", "op": "not_eq", "value": "200"}]},
                {
                    "id": "generic_course_or_tutorial",
                    "all": [{"field": "flag_generic_course_or_tutorial_source", "op": "eq", "value": True}],
                },
                {
                    "id": "search_derived_only",
                    "all": [{"field": "retrieval_method", "op": "contains_any", "value": ["search", "serp", "query"]}],
                },
                {
                    "id": "unknown_or_collision_source",
                    "all": [
                        {
                            "field": "dataset_specificity_hint",
                            "op": "in",
                            "value": ["unknown", "collision_risk"],
                        }
                    ],
                },
            ],
            "default_reason_code": "low_trust_or_unresolved_source",
        },
    ]

    ambiguous_name_cases = [
        {
            "own_id": "c7",
            "dataset_name": "Nursery",
            "single_token_never_sufficient": ["nursery"],
            "hard_negative_source_root_fragments": ["nursery-dbms", "greenthumb-plantation", "plantation", "dbms"],
            "hard_negative_table_tokens": ["customer", "orders", "order_items", "payment", "plant", "plants", "users"],
            "strict_requires_any": [
                "dataset_id_exact_match_in_context == true",
                "source_root_matches_known_benchmark_source == true",
                "dataset_token_overlap_count >= 2 and table_token_overlap_count >= 1 and app_schema_hit_count == 0",
            ],
            "collision_example": "Nursery-DBMS or plantation/order-management SQL is collision_risk, not strict for OpenML/UCI Nursery.",
        },
        {
            "own_id": "c5",
            "dataset_name": "Mushroom",
            "single_token_never_sufficient": ["mushroom"],
            "hard_negative_source_root_fragments": ["mushroomcultivation", "kinokosql", "basket", "cultivation"],
            "hard_negative_table_tokens": ["basket", "cust_order", "customer", "gatherer", "guild", "order", "store"],
            "strict_requires_any": [
                "dataset_id_exact_match_in_context == true",
                "dataset_token_overlap_count >= 2 and table_token_overlap_count >= 1 and app_schema_hit_count == 0",
            ],
            "collision_example": "Cultivation-store or basket/order SQL is collision_risk even if it contains the token mushroom.",
        },
        {
            "own_id": "c10",
            "dataset_name": "Poker Hand",
            "single_token_never_sufficient": ["poker", "hand"],
            "hard_negative_source_root_fragments": ["poker-query-language", "trainer", "parser"],
            "hard_negative_table_tokens": ["player", "board", "hero", "villain"],
            "strict_requires_any": [
                "dataset_id_exact_match_in_context == true",
                "dataset_token_overlap_count >= 2 and table_token_overlap_count >= 1 and flag_nonstandard_query_language == false",
            ],
            "collision_example": "Poker trainer or OpenPQL text is reject_non_sql or collision_risk, not strict benchmark SQL.",
        },
        {
            "own_id": "c16",
            "dataset_name": "FiveThirtyEight Comic Characters",
            "single_token_never_sufficient": ["comic", "characters"],
            "hard_negative_source_root_fragments": ["service", "warehouse", "comicshop", "superhero"],
            "hard_negative_table_tokens": ["chapter", "customer", "employee", "images", "pet", "search", "store", "user", "users"],
            "strict_requires_any": [
                "dataset_id_exact_match_in_context == true",
                "dataset_token_overlap_count >= 2 and table_token_overlap_count >= 1 and app_schema_hit_count == 0",
            ],
            "collision_example": "Comic-shop or superhero-service databases are collision_risk unless they explicitly anchor to the FiveThirtyEight benchmark schema.",
        },
        {
            "own_id": "n16",
            "dataset_name": "Credit Card Fraud Detection",
            "single_token_never_sufficient": ["credit", "card", "fraud"],
            "hard_negative_source_root_fragments": ["fraud-detection-system", "merchant", "card-holder"],
            "hard_negative_table_tokens": ["card_holder", "credit_card", "merchant", "merchant_category", "transactions"],
            "strict_requires_any": [
                "dataset_id_exact_match_in_context == true",
                "context_contains_any(['v1','v2','v3','amount','class','time']) == true and app_schema_hit_count == 0",
            ],
            "collision_example": "Operational fraud-system schemas with card_holder and merchant tables are collision_risk, not the Kaggle creditcardfraud benchmark.",
        },
        {
            "own_id": "c19",
            "dataset_name": "Trending YouTube Stats",
            "single_token_never_sufficient": ["youtube", "trending"],
            "hard_negative_source_root_fragments": ["kaggle-course-answer", "intro-to-sql"],
            "hard_negative_table_tokens": ["stackoverflow", "posts_questions", "search"],
            "strict_requires_any": [
                "dataset_id_exact_match_in_context == true",
                "dataset_token_overlap_count >= 2 and table_token_overlap_count >= 1 and flag_generic_course_or_tutorial_source == false",
            ],
            "collision_example": "Generic Kaggle course answers mentioning YouTube are collision_risk unless they explicitly tie to the benchmark dataset.",
        },
    ]

    rulebook = {
        "rulebook_name": "sql_high_v2_cleaning_rules",
        "version": "2.0.0",
        "generated_at_utc": utc_now_iso(),
        "baseline_root": str(baseline_root.resolve()),
        "intent": "Deterministic rules for V2 SQL-inventory cleaning without vague judgment calls.",
        "automation_contract": {
            "required_sql_row_fields": [
                "own_id",
                "dataset_id",
                "dataset_name",
                "sql_item_id",
                "source_url",
                "source_type",
                "source_title",
                "sql_text_raw",
                "sql_text_clean",
                "dataset_specificity_label",
                "evidence_confidence",
                "source_file_path",
                "extraction_method",
                "is_near_duplicate",
            ],
            "optional_join_fields_from_source_inventory": [
                "http_status",
                "retrieval_method",
                "dataset_specificity_hint",
                "has_sql_text",
                "notes",
            ],
            "derived_signals_required": [
                "dataset_tokens",
                "context_text",
                "dataset_token_overlap_count",
                "table_tokens",
                "table_token_overlap_count",
                "app_schema_hit_count",
                "dataset_id_exact_match_in_context",
                "source_root_matches_known_benchmark_source",
                "flag_generic_course_or_tutorial_source",
                "flag_nonstandard_query_language",
                "flag_code_fragment_context",
                "flag_application_repo_context",
                "flag_application_schema_context",
                "flag_foreign_dataset_context",
                "flag_string_literal_not_standalone_sql",
                "ambiguous_single_token_match",
                "source_credibility_tier",
            ],
        },
        "decision_order": [
            {
                "stage": "derive_signals",
                "priority": 1,
                "description": "Normalize input text and compute deterministic overlap and pattern flags before any label is assigned.",
            },
            {
                "stage": "reject_non_sql",
                "priority": 2,
                "description": "If any reject_non_sql rule fires, final label becomes reject_non_sql and no specificity label is assigned.",
            },
            {
                "stage": "detect_duplicates",
                "priority": 3,
                "description": "Compute exact and normalized duplicate groups within each own_id after reject_non_sql decisions.",
            },
            {
                "stage": "source_credibility_tiering",
                "priority": 4,
                "description": "Assign exactly one source credibility tier using the first matching tier in priority order.",
            },
            {
                "stage": "dataset_specificity_classification",
                "priority": 5,
                "description": "Assign strict_dataset_specific, weak_related, or collision_risk using first-match precedence strict -> weak -> collision_risk.",
            },
            {
                "stage": "executable_status_assignment",
                "priority": 6,
                "description": "Assign executable_status pass, fail, or unknown with an explicit reason code after reject and specificity decisions.",
            },
            {
                "stage": "cleaning_action",
                "priority": 7,
                "description": "Emit keep_active, keep_provenance_only, or reject row actions deterministically from the prior stages.",
            },
        ],
        "shared_derivation": {
            "dataset_tokenization": {
                "source_fields": ["dataset_name", "dataset_id_local_part"],
                "steps": [
                    "lowercase",
                    "extract tokens with regex [a-z0-9]+",
                    "drop tokens shorter than 3 characters",
                    "drop stopwords",
                    "preserve original order and deduplicate",
                ],
                "stopwords": sorted(TOKEN_STOPWORDS),
            },
            "context_text": {
                "source_fields_in_order": [
                    "normalize_url_root(source_url)",
                    "source_title",
                    "source_file_path",
                    "sql_text_raw[:500]",
                    "extraction_notes",
                ],
                "steps": ["concatenate with single spaces", "lowercase"],
            },
            "sql_candidate_detection": {
                "standalone_sql_start_regex": (
                    r"^\s*(with|select|insert\s+into|update|delete\s+from|create\s+"
                    r"(or\s+replace\s+)?(table|view|database|schema|function|procedure|index|trigger)"
                    r"|drop\s+(table|database|view|schema)|alter\s+table|truncate\s+table|use\s+\w+"
                    r"|show\s+\w+|describe\s+\w+|explain\s+\w+|merge\s+into|call\s+\w+)"
                ),
                "fallback_keyword_threshold": {
                    "required_sql_keyword_hits": 3,
                    "keywords_checked": [
                        "select",
                        "from",
                        "where",
                        "group by",
                        "order by",
                        "join",
                        "create table",
                        "insert into",
                        "update",
                        "delete from",
                        "with",
                    ],
                    "separator_required": ["semicolon", "newline"],
                },
            },
            "table_token_extraction": {
                "regex": r"(?i)\b(?:from|join|into|update|table)\s+[`\"]?([a-zA-Z_][a-zA-Z0-9_]*)",
                "output_transform": ["lowercase"],
            },
            "overlap_signals": {
                "dataset_token_overlap_count": "count of dataset_tokens present in context_text",
                "table_token_overlap_count": "count of extracted table_tokens that also occur in dataset_tokens",
                "app_schema_hit_count": "count of extracted table_tokens present in app_schema_tokens",
            },
            "pattern_sets": {
                "generic_course_patterns": list(GENERIC_SOURCE_PATTERNS),
                "nonstandard_query_patterns": list(NONSTANDARD_QUERY_PATTERNS),
                "code_fragment_patterns": list(CODE_FRAGMENT_PATTERNS),
                "application_repo_patterns": list(APP_REPO_PATTERNS),
                "application_schema_tokens": sorted(APP_SCHEMA_TOKENS),
                "foreign_dataset_patterns": list(FOREIGN_DATASET_PATTERNS),
            },
        },
        "criteria": {
            "reject_non_sql": {
                "final_label": "reject_non_sql",
                "stage_priority": 2,
                "first_match_wins": False,
                "rules": [
                    {
                        "id": "reject_missing_sql_text_raw",
                        "priority": 10,
                        "all": [{"field": "sql_text_raw_stripped", "op": "eq", "value": ""}],
                        "reason_code": "missing_sql_text_raw",
                    },
                    {
                        "id": "reject_not_standalone_sql",
                        "priority": 20,
                        "all": [{"field": "standalone_sql_like", "op": "eq", "value": False}],
                        "reason_code": "not_standalone_sql",
                    },
                    {
                        "id": "reject_nonstandard_query_language",
                        "priority": 30,
                        "all": [{"field": "flag_nonstandard_query_language", "op": "eq", "value": True}],
                        "reason_code": "nonstandard_query_language",
                    },
                    {
                        "id": "reject_code_fragment_context",
                        "priority": 40,
                        "all": [{"field": "flag_code_fragment_context", "op": "eq", "value": True}],
                        "reason_code": "code_fragment_context",
                    },
                    {
                        "id": "reject_string_literal_not_standalone_sql",
                        "priority": 50,
                        "all": [
                            {
                                "field": "extraction_method",
                                "op": "in",
                                "value": ["source_string_literal", "ipynb_string_literal"],
                            },
                            {"field": "flag_string_literal_not_standalone_sql", "op": "eq", "value": True},
                        ],
                        "reason_code": "string_literal_not_standalone_sql",
                    },
                ],
                "cleaning_action": "drop_from_active_sql_inventory_and_write_to_rejected_log",
            },
            "strict_dataset_specific": {
                "final_label": "strict_dataset_specific",
                "output_label_in_inventory": "strict",
                "preconditions_all": [
                    {"field": "rejected_non_sql", "op": "eq", "value": False},
                    {"field": "source_credibility_tier", "op": "in", "value": ["tier_1_official", "tier_2_primary_code", "tier_3_secondary_explanatory"]},
                ],
                "forbid_if_any": [
                    {"field": "flag_application_repo_context", "op": "eq", "value": True},
                    {"field": "flag_application_schema_context", "op": "eq", "value": True},
                    {"field": "flag_foreign_dataset_context", "op": "eq", "value": True},
                    {"field": "flag_nonstandard_query_language", "op": "eq", "value": True},
                    {"field": "ambiguous_single_token_match", "op": "eq", "value": True},
                    {"field": "source_credibility_tier", "op": "eq", "value": "tier_4_low_trust"},
                ],
                "assign_if_any": [
                    {
                        "id": "strict_dataset_id_anchor",
                        "priority": 10,
                        "all": [{"field": "dataset_id_exact_match_in_context", "op": "eq", "value": True}],
                        "reason_code": "dataset_id_exact_match",
                    },
                    {
                        "id": "strict_known_official_root_plus_schema",
                        "priority": 20,
                        "all": [
                            {"field": "source_root_matches_known_benchmark_source", "op": "eq", "value": True},
                            {"field": "dataset_token_overlap_count", "op": "gte", "value": 1},
                            {"field": "table_token_overlap_count", "op": "gte", "value": 1},
                        ],
                        "reason_code": "known_benchmark_source_root_with_schema_overlap",
                    },
                    {
                        "id": "strict_multi_token_schema_match",
                        "priority": 30,
                        "all": [
                            {"field": "dataset_token_overlap_count", "op": "gte", "value": 2},
                            {"field": "table_token_overlap_count", "op": "gte", "value": 1},
                            {"field": "app_schema_hit_count", "op": "eq", "value": 0},
                        ],
                        "reason_code": "multi_token_dataset_and_schema_overlap",
                    },
                ],
                "else_result": "not_strict",
            },
            "weak_related": {
                "final_label": "weak_related",
                "output_label_in_inventory": "weak",
                "preconditions_all": [
                    {"field": "rejected_non_sql", "op": "eq", "value": False},
                    {"field": "assigned_strict", "op": "eq", "value": False},
                ],
                "forbid_if_any": [
                    {"field": "flag_application_repo_context", "op": "eq", "value": True},
                    {"field": "flag_application_schema_context", "op": "eq", "value": True},
                    {"field": "flag_foreign_dataset_context", "op": "eq", "value": True},
                    {"field": "ambiguous_single_token_match", "op": "eq", "value": True},
                ],
                "assign_if_any": [
                    {
                        "id": "weak_single_anchor_on_credible_source",
                        "priority": 10,
                        "all": [
                            {"field": "source_credibility_tier", "op": "in", "value": ["tier_1_official", "tier_2_primary_code", "tier_3_secondary_explanatory"]},
                            {"field": "dataset_token_overlap_count", "op": "gte", "value": 1},
                        ],
                        "reason_code": "single_dataset_anchor_without_schema_alignment",
                    },
                    {
                        "id": "weak_known_source_root",
                        "priority": 20,
                        "all": [
                            {"field": "source_root_matches_known_benchmark_source", "op": "eq", "value": True},
                            {"field": "dataset_token_overlap_count", "op": "gte", "value": 1},
                        ],
                        "reason_code": "known_benchmark_root_but_insufficient_schema_overlap",
                    },
                ],
                "else_result": "not_weak",
            },
            "collision_risk": {
                "final_label": "collision_risk",
                "output_label_in_inventory": "collision_risk",
                "preconditions_all": [
                    {"field": "rejected_non_sql", "op": "eq", "value": False},
                    {"field": "assigned_strict", "op": "eq", "value": False},
                    {"field": "assigned_weak", "op": "eq", "value": False},
                ],
                "assign_if_any": [
                    {
                        "id": "collision_ambiguous_single_token_match",
                        "priority": 10,
                        "all": [{"field": "ambiguous_single_token_match", "op": "eq", "value": True}],
                        "reason_code": "ambiguous_name_only_match",
                    },
                    {
                        "id": "collision_application_repo",
                        "priority": 20,
                        "all": [{"field": "flag_application_repo_context", "op": "eq", "value": True}],
                        "reason_code": "application_repo_context",
                    },
                    {
                        "id": "collision_application_schema",
                        "priority": 30,
                        "all": [{"field": "flag_application_schema_context", "op": "eq", "value": True}],
                        "reason_code": "application_schema_context",
                    },
                    {
                        "id": "collision_foreign_dataset_context",
                        "priority": 40,
                        "all": [{"field": "flag_foreign_dataset_context", "op": "eq", "value": True}],
                        "reason_code": "foreign_dataset_context",
                    },
                    {
                        "id": "collision_generic_course_without_strong_anchor",
                        "priority": 50,
                        "all": [
                            {"field": "flag_generic_course_or_tutorial_source", "op": "eq", "value": True},
                            {"field": "dataset_token_overlap_count", "op": "lt", "value": 2},
                        ],
                        "reason_code": "generic_course_without_strong_dataset_anchor",
                    },
                    {
                        "id": "collision_no_dataset_anchor",
                        "priority": 60,
                        "all": [{"field": "dataset_token_overlap_count", "op": "eq", "value": 0}],
                        "reason_code": "no_dataset_anchor_in_context",
                    },
                    {
                        "id": "collision_low_trust_default",
                        "priority": 70,
                        "all": [{"field": "source_credibility_tier", "op": "eq", "value": "tier_4_low_trust"}],
                        "reason_code": "low_trust_source_default_to_collision_risk",
                    },
                ],
                "fallback_default": "collision_risk",
            },
        },
        "name_collision_policy": {
            "general_rule": (
                "If a dataset name is ambiguous or domain-generic, a single token match never qualifies for strict. "
                "The row must also satisfy an exact dataset-id anchor, a known benchmark source-root anchor, "
                "or a multi-token-plus-schema overlap rule."
            ),
            "ambiguous_single_token_match_definition": (
                "ambiguous_single_token_match = dataset_token_overlap_count == 1 and the only matched token appears "
                "in single_token_never_sufficient for the dataset's ambiguous case."
            ),
            "hard_negative_precedence": (
                "If any hard_negative_source_root_fragments or hard_negative_table_tokens hit for an ambiguous case, "
                "assign collision_risk unless reject_non_sql already fired."
            ),
            "cases": ambiguous_name_cases,
        },
        "duplicate_policy": {
            "scope": "deduplicate only within each own_id",
            "exact_duplicate": {
                "group_key": [
                    "own_id",
                    "sha256(normalize_newlines(sql_text_raw))",
                    "normalize_url_root(source_url)",
                ],
                "canonical_sort_order": [
                    "source_credibility_tier ascending by tier priority",
                    "dataset_specificity strict before weak before collision_risk",
                    "evidence_confidence high before medium before low",
                    "executable_status pass before unknown before fail",
                    "shorter source_url",
                    "sql_item_id lexical ascending",
                ],
                "action_non_canonical": "keep_for_provenance_only",
                "duplicate_status_value": "exact_duplicate",
            },
            "normalized_duplicate": {
                "normalization_steps": [
                    "start from sql_text_clean if non-empty else sql_text_raw",
                    "replace CRLF and CR with LF",
                    "lowercase",
                    "strip leading and trailing whitespace",
                    "collapse consecutive whitespace to a single space",
                    "strip trailing semicolons",
                    "normalize repeated backticks and double quotes to single token separators",
                ],
                "group_key": ["own_id", "sha256(normalized_sql_text)"],
                "canonical_sort_order": [
                    "source_credibility_tier ascending by tier priority",
                    "dataset_specificity strict before weak before collision_risk",
                    "evidence_confidence high before medium before low",
                    "executable_status pass before unknown before fail",
                    "sql_item_id lexical ascending",
                ],
                "action_non_canonical": "keep_for_provenance_only",
                "duplicate_status_value": "normalized_duplicate",
            },
            "inventory_policy": {
                "canonical_row": "eligible_for_active_sql_inventory = true",
                "non_canonical_rows": "eligible_for_active_sql_inventory = false but retained in provenance and duplicate ledgers",
                "question_seed_policy": "only canonical rows may be used for question_seed_candidates.csv",
            },
        },
        "executable_status_policy": {
            "decision_order": [
                {
                    "status": "fail",
                    "priority": 10,
                    "assign_if_any": [
                        "rejected_non_sql == true",
                        "source_explicitly_contains_sql_error == true",
                        "local_sql_parser_failure == true",
                        "schema_mapping_available == true and unresolved_table_or_column_reference == true",
                    ],
                    "reason_codes": [
                        "reject_non_sql",
                        "source_shows_error",
                        "local_parser_failure",
                        "schema_reference_failure",
                    ],
                },
                {
                    "status": "pass",
                    "priority": 20,
                    "assign_if_any": [
                        "local_parse_and_execution_success == true",
                        "source_displays_query_result_table == true and query_text_matches_result_block == true",
                        "notebook_or_repo_documents_successful_execution_for_same_query == true",
                    ],
                    "reason_codes": [
                        "local_execution_success",
                        "source_shows_query_results",
                        "source_documents_successful_execution",
                    ],
                },
                {
                    "status": "unknown",
                    "priority": 30,
                    "assign_if_any": ["otherwise"],
                    "reason_codes": [
                        "no_execution_evidence",
                        "schema_not_mapped_yet",
                        "not_tested_in_v2",
                    ],
                },
            ],
            "default_status_today": "unknown unless explicit pass/fail evidence exists",
        },
        "cleaning_actions": {
            "reject_non_sql": "remove from active SQL inventory; write to rejected_or_non_sql ledger with reject reason code",
            "exact_duplicate": "retain only canonical row in active inventory; preserve all others in duplicate provenance log",
            "normalized_duplicate": "retain only canonical row in active inventory; preserve all others in duplicate provenance log",
            "strict_dataset_specific": "eligible for trusted active inventory and later question-seed generation if not duplicate",
            "weak_related": "keep in active inventory but exclude from question-seed generation until manually promoted",
            "collision_risk": "keep in audit ledger or collision-risk sidecar; exclude from question-seed generation and top_strict exports",
        },
        "automation_readiness_checks": {
            "must_have_top_level_keys": [
                "decision_order",
                "shared_derivation",
                "criteria",
                "duplicate_policy",
                "executable_status_policy",
                "source_credibility_tiers",
                "name_collision_policy",
            ],
            "required_criteria_keys": [
                "strict_dataset_specific",
                "weak_related",
                "collision_risk",
                "reject_non_sql",
            ],
            "pass_condition": (
                "PASS only if all required sections exist, every classification block contains structured conditions, "
                "the decision order is total and ordered, and name-collision handling is explicit for ambiguous datasets."
            ),
        },
        "source_credibility_tiers": source_credibility_tiers,
    }
    return rulebook


def validate_rulebook(rulebook: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    top_level_required = set(rulebook["automation_readiness_checks"]["must_have_top_level_keys"])
    for key in top_level_required:
        if key not in rulebook:
            issues.append(f"missing_top_level_key:{key}")

    decision_order = rulebook.get("decision_order", [])
    if not decision_order or len(decision_order) < 6:
        issues.append("decision_order_too_short")
    else:
        priorities = [stage.get("priority") for stage in decision_order]
        if priorities != sorted(priorities):
            issues.append("decision_order_priorities_not_sorted")

    criteria = rulebook.get("criteria", {})
    for key in rulebook["automation_readiness_checks"]["required_criteria_keys"]:
        block = criteria.get(key)
        if not block:
            issues.append(f"missing_criteria_block:{key}")
            continue
        if key == "reject_non_sql":
            if not block.get("rules"):
                issues.append("reject_non_sql_has_no_rules")
        else:
            if not block.get("assign_if_any"):
                issues.append(f"{key}_has_no_assign_rules")

    tiers = rulebook.get("source_credibility_tiers", [])
    if len(tiers) < 4:
        issues.append("source_credibility_tiers_incomplete")
    else:
        tier_priorities = [tier.get("priority") for tier in tiers]
        if tier_priorities != sorted(tier_priorities):
            issues.append("source_credibility_tier_priorities_not_sorted")

    collision_cases = rulebook.get("name_collision_policy", {}).get("cases", [])
    if not collision_cases:
        issues.append("name_collision_cases_missing")
    else:
        if not any(case.get("own_id") == "c7" for case in collision_cases):
            issues.append("nursery_name_collision_case_missing")
        for case in collision_cases:
            if not case.get("single_token_never_sufficient"):
                issues.append(f"name_collision_case_missing_single_token_rule:{case.get('own_id', 'unknown')}")
            if not case.get("strict_requires_any"):
                issues.append(f"name_collision_case_missing_strict_requirement:{case.get('own_id', 'unknown')}")

    duplicate_policy = rulebook.get("duplicate_policy", {})
    if not duplicate_policy.get("exact_duplicate") or not duplicate_policy.get("normalized_duplicate"):
        issues.append("duplicate_policy_incomplete")

    executable_policy = rulebook.get("executable_status_policy", {})
    if len(executable_policy.get("decision_order", [])) < 3:
        issues.append("executable_status_policy_incomplete")

    return issues


def build_markdown(rulebook: dict[str, Any]) -> str:
    tiers = rulebook["source_credibility_tiers"]
    criteria = rulebook["criteria"]
    collision_cases = rulebook["name_collision_policy"]["cases"]

    lines = [
        "# V2 Cleaning Rules",
        "",
        f"- Version: `{rulebook['version']}`",
        f"- Generated at UTC: `{rulebook['generated_at_utc']}`",
        f"- Baseline root: `{rulebook['baseline_root']}`",
        f"- Intent: {rulebook['intent']}",
        "",
        "## Deterministic Decision Order",
        "",
    ]
    for stage in rulebook["decision_order"]:
        lines.append(f"- Priority {stage['priority']}: `{stage['stage']}` -> {stage['description']}")

    lines.extend(
        [
            "",
            "## Required Inputs",
            "",
            "- Required SQL-row fields: " + ", ".join(rulebook["automation_contract"]["required_sql_row_fields"]),
            "- Optional joined source fields: " + ", ".join(rulebook["automation_contract"]["optional_join_fields_from_source_inventory"]),
            "- Required derived signals: " + ", ".join(rulebook["automation_contract"]["derived_signals_required"]),
            "",
            "## Shared Derivation Rules",
            "",
            "### Dataset Tokens",
            "",
            "- Source fields: " + ", ".join(rulebook["shared_derivation"]["dataset_tokenization"]["source_fields"]),
            "- Steps: " + "; ".join(rulebook["shared_derivation"]["dataset_tokenization"]["steps"]),
            "- Stopwords: " + ", ".join(rulebook["shared_derivation"]["dataset_tokenization"]["stopwords"]),
            "",
            "### Context and SQL Detection",
            "",
            "- Context fields: " + ", ".join(rulebook["shared_derivation"]["context_text"]["source_fields_in_order"]),
            "- SQL start regex: `" + rulebook["shared_derivation"]["sql_candidate_detection"]["standalone_sql_start_regex"] + "`",
            "- Fallback keyword threshold: "
            + str(rulebook["shared_derivation"]["sql_candidate_detection"]["fallback_keyword_threshold"]["required_sql_keyword_hits"])
            + " keyword hits plus semicolon or newline.",
            "- Table-token regex: `" + rulebook["shared_derivation"]["table_token_extraction"]["regex"] + "`",
            "",
            "## Source Credibility Tiers",
            "",
            "| Tier | Priority | Label | Deterministic meaning |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for tier in tiers:
        lines.append(
            f"| {tier['tier_id']} | {tier['priority']} | {tier['label']} | "
            f"Assigned by the first matching rule in its tier block. |"
        )

    lines.extend(
        [
            "",
            "Tier details:",
        ]
    )
    for tier in tiers:
        lines.append(f"- `{tier['tier_id']}`: default_reason_code=`{tier['default_reason_code']}`.")
        for rule in tier["assign_if_any"]:
            conditions = "; ".join(f"{cond['field']} {cond['op']} {cond['value']}" for cond in rule["all"])
            lines.append(f"- `{tier['tier_id']}.{rule['id']}`: {conditions}")

    lines.extend(
        [
            "",
            "## Specificity Criteria",
            "",
            "### strict_dataset_specific",
            "",
            "- Final output label: `strict`",
            "- Preconditions:",
        ]
    )
    for cond in criteria["strict_dataset_specific"]["preconditions_all"]:
        lines.append(f"- `{cond['field']} {cond['op']} {cond['value']}`")
    lines.append("- Forbidden if any:")
    for cond in criteria["strict_dataset_specific"]["forbid_if_any"]:
        lines.append(f"- `{cond['field']} {cond['op']} {cond['value']}`")
    lines.append("- Assign strict if any rule matches:")
    for rule in criteria["strict_dataset_specific"]["assign_if_any"]:
        lines.append(
            "- `"
            + rule["id"]
            + "`: "
            + "; ".join(f"{cond['field']} {cond['op']} {cond['value']}" for cond in rule["all"])
            + f" -> `{rule['reason_code']}`"
        )

    lines.extend(
        [
            "",
            "### weak_related",
            "",
            "- Final output label: `weak`",
            "- Preconditions:",
        ]
    )
    for cond in criteria["weak_related"]["preconditions_all"]:
        lines.append(f"- `{cond['field']} {cond['op']} {cond['value']}`")
    lines.append("- Forbidden if any:")
    for cond in criteria["weak_related"]["forbid_if_any"]:
        lines.append(f"- `{cond['field']} {cond['op']} {cond['value']}`")
    lines.append("- Assign weak if any rule matches:")
    for rule in criteria["weak_related"]["assign_if_any"]:
        lines.append(
            "- `"
            + rule["id"]
            + "`: "
            + "; ".join(f"{cond['field']} {cond['op']} {cond['value']}" for cond in rule["all"])
            + f" -> `{rule['reason_code']}`"
        )

    lines.extend(
        [
            "",
            "### collision_risk",
            "",
            "- Final output label: `collision_risk`",
            "- Preconditions:",
        ]
    )
    for cond in criteria["collision_risk"]["preconditions_all"]:
        lines.append(f"- `{cond['field']} {cond['op']} {cond['value']}`")
    lines.append("- Assign collision_risk if any rule matches:")
    for rule in criteria["collision_risk"]["assign_if_any"]:
        lines.append(
            "- `"
            + rule["id"]
            + "`: "
            + "; ".join(f"{cond['field']} {cond['op']} {cond['value']}" for cond in rule["all"])
            + f" -> `{rule['reason_code']}`"
        )
    lines.append(f"- Fallback default: `{criteria['collision_risk']['fallback_default']}`")

    lines.extend(
        [
            "",
            "### reject_non_sql",
            "",
            "- Final output label: `reject_non_sql`",
            "- If any reject rule fires, the row is dropped from the active SQL inventory and logged to the rejected ledger.",
        ]
    )
    for rule in criteria["reject_non_sql"]["rules"]:
        lines.append(
            "- `"
            + rule["id"]
            + "`: "
            + "; ".join(f"{cond['field']} {cond['op']} {cond['value']}" for cond in rule["all"])
            + f" -> `{rule['reason_code']}`"
        )

    lines.extend(
        [
            "",
            "## Name Collision Handling",
            "",
            f"- General rule: {rulebook['name_collision_policy']['general_rule']}",
            f"- Ambiguous single-token definition: {rulebook['name_collision_policy']['ambiguous_single_token_match_definition']}",
            f"- Hard-negative precedence: {rulebook['name_collision_policy']['hard_negative_precedence']}",
            "",
            "| own_id | dataset_name | Single token never sufficient | Hard negatives | Strict requires any |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in collision_cases:
        lines.append(
            f"| {case['own_id']} | {case['dataset_name']} | "
            f"{', '.join(case['single_token_never_sufficient'])} | "
            f"{', '.join(case['hard_negative_source_root_fragments'] + case['hard_negative_table_tokens'])} | "
            f"{' OR '.join(case['strict_requires_any'])} |"
        )
    lines.extend(
        [
            "",
            "Collision examples:",
        ]
    )
    for case in collision_cases:
        lines.append(f"- `{case['own_id']}`: {case['collision_example']}")

    lines.extend(
        [
            "",
            "## Duplicate Policy",
            "",
            "- Scope: " + rulebook["duplicate_policy"]["scope"],
            "- Exact duplicate key: " + ", ".join(rulebook["duplicate_policy"]["exact_duplicate"]["group_key"]),
            "- Exact duplicate canonical sort: " + "; ".join(rulebook["duplicate_policy"]["exact_duplicate"]["canonical_sort_order"]),
            "- Exact duplicate non-canonical action: `" + rulebook["duplicate_policy"]["exact_duplicate"]["action_non_canonical"] + "`",
            "- Normalized duplicate steps: " + "; ".join(rulebook["duplicate_policy"]["normalized_duplicate"]["normalization_steps"]),
            "- Normalized duplicate key: " + ", ".join(rulebook["duplicate_policy"]["normalized_duplicate"]["group_key"]),
            "- Normalized duplicate canonical sort: " + "; ".join(rulebook["duplicate_policy"]["normalized_duplicate"]["canonical_sort_order"]),
            "- Question-seed policy: " + rulebook["duplicate_policy"]["inventory_policy"]["question_seed_policy"],
            "",
            "## executable_status Policy",
            "",
        ]
    )
    for block in rulebook["executable_status_policy"]["decision_order"]:
        lines.append(f"- `{block['status']}` priority {block['priority']}: " + "; ".join(block["assign_if_any"]))
        lines.append(f"- `{block['status']}` reason codes: " + ", ".join(block["reason_codes"]))
    lines.append("- Default today: " + rulebook["executable_status_policy"]["default_status_today"])

    lines.extend(
        [
            "",
            "## Cleaning Actions",
            "",
        ]
    )
    for key, value in rulebook["cleaning_actions"].items():
        lines.append(f"- `{key}` -> {value}")

    lines.extend(
        [
            "",
            "## Automation Readiness",
            "",
            "- PASS condition: " + rulebook["automation_readiness_checks"]["pass_condition"],
            "- Required top-level keys: " + ", ".join(rulebook["automation_readiness_checks"]["must_have_top_level_keys"]),
            "- Required criteria blocks: " + ", ".join(rulebook["automation_readiness_checks"]["required_criteria_keys"]),
        ]
    )
    return "\n".join(lines)


def build_checkpoint_payload(
    *,
    rulebook_path: Path,
    markdown_path: Path,
    issues: list[str],
) -> dict[str, Any]:
    status = "PASS" if not issues else "FAIL"
    return {
        "checkpoint": "0",
        "phase_name": "v2_cleaning_rulebook",
        "generated_at_utc": utc_now_iso(),
        "status": status,
        "automation_ready": status == "PASS",
        "pass_condition": (
            "PASS only if the rules are explicit enough to be implemented automatically: ordered decision flow, "
            "structured criteria, duplicate policy, executable_status policy, source tiers, and explicit name-collision handling."
        ),
        "issues": issues,
        "outputs": [
            {
                "path": str(rulebook_path.resolve()),
                "sha256": sha256_file(rulebook_path),
                "size_bytes": rulebook_path.stat().st_size,
            },
            {
                "path": str(markdown_path.resolve()),
                "sha256": sha256_file(markdown_path),
                "size_bytes": markdown_path.stat().st_size,
            },
        ],
    }


def main() -> int:
    args = parse_args()
    rules_dir = args.output_root / "rules"
    json_path = rules_dir / "v2_cleaning_rules.json"
    markdown_path = rules_dir / "v2_cleaning_rules.md"
    checkpoint_path = rules_dir / "checkpoint0_status.json"

    rulebook = build_rulebook(args.baseline_root)
    write_json(json_path, rulebook)
    write_text(markdown_path, build_markdown(rulebook))

    issues = validate_rulebook(rulebook)
    checkpoint = build_checkpoint_payload(
        rulebook_path=json_path,
        markdown_path=markdown_path,
        issues=issues,
    )
    write_json(checkpoint_path, checkpoint)

    print(str(json_path.resolve()))
    print(str(markdown_path.resolve()))
    print(str(checkpoint_path.resolve()))
    print(checkpoint["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
