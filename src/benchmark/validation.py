"""Validation layers for benchmark candidates and question bundles."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from src.benchmark.canonical_sql import stable_hash
from src.benchmark.models import (
    CandidateRecord,
    ExecutionResult,
    QuerySpec,
    StaticDatasetUnderstanding,
    ValidationCategoryResult,
    ValidationResult,
)

if TYPE_CHECKING:
    from src.benchmark.llm_runtime import BenchmarkLLMRuntime

SQLITE_INCOMPATIBLE_PATTERNS = [
    r"\bfield\s*\(",
    r"\bilike\b",
    r"\bdate_trunc\s*\(",
    r"\bregexp\b",
]


RATE_LIKE_COLUMN_HINTS = {"rate", "ratio", "proportion", "pct", "percentage", "probability"}
SUPPORT_LIKE_COLUMN_HINTS = {"support", "count", "total", "n", "size", "denominator"}


def _extract_numeric_values(rows: list[list[Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for cell in row:
            try:
                values.append(float(cell))
            except (TypeError, ValueError):
                continue
    return values


def _canonical_sql(sql: str) -> str:
    text = sql.strip().lower().rstrip(";")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_group_by_columns(sql: str) -> list[str]:
    normalized = _canonical_sql(sql)
    match = re.search(r"group by\s+(.+?)(\s+having|\s+order by|\s+limit|$)", normalized)
    if not match:
        return []
    raw = match.group(1)
    parts = [item.strip() for item in raw.split(",")]
    return [part for part in parts if part]


def _extract_partition_by_columns(sql: str) -> list[str]:
    normalized = _canonical_sql(sql)
    results: list[str] = []
    for match in re.finditer(r"over\s*\(\s*partition by\s+(.+?)(\)|\s+order by)", normalized):
        raw = match.group(1)
        parts = [item.strip() for item in raw.split(",")]
        for part in parts:
            if part and part not in results:
                results.append(part)
    return results


def _sql_has_structural_aggregation(sql: str) -> bool:
    lowered = _canonical_sql(sql)
    return any(token in lowered for token in [" group by ", "count(", "sum(", "avg(", " over (", "having "])


def _is_raw_projection(sql: str) -> bool:
    lowered = _canonical_sql(sql)
    if _sql_has_structural_aggregation(lowered):
        return False
    return bool(re.fullmatch(r"select\s+[\w\s,.*]+\s+from\s+\w+(\s+where\s+.+)?", lowered))


def _infer_semantic_role(query_spec: QuerySpec) -> str:
    if query_spec.variant_semantic_role:
        return query_spec.variant_semantic_role
    for code in query_spec.reason_codes:
        if code.startswith("SQL_VARIANT_SEMANTIC_ROLE_"):
            return code.replace("SQL_VARIANT_SEMANTIC_ROLE_", "").lower()
    return "unknown"


def _contains_sqlite_incompatible(sql: str) -> bool:
    normalized = _canonical_sql(sql)
    return any(re.search(pattern, normalized) for pattern in SQLITE_INCOMPATIBLE_PATTERNS)


def _role_contract_ok(role: str, sql: str) -> bool:
    normalized = _canonical_sql(sql)
    if role in {"count_distribution", "rare_extreme_view", "filtered_stable_view"}:
        return " group by " in normalized and "count(" in normalized
    if role == "within_group_proportion":
        return (" group by " in normalized) and ("over (" in normalized or "rate" in normalized or "/" in normalized)
    if role == "collapsed_target_view":
        return "case when" in normalized and "group by" in normalized
    if role == "ranked_signal_view":
        has_signal = "sum(case" in normalized or "avg(case" in normalized
        has_rate = "focus_rate" in normalized or " rate" in normalized or "_rate" in normalized
        has_order = "order by" in normalized
        weak_sort = "order by support" in normalized or "order by count" in normalized
        return has_signal and has_rate and has_order and not weak_sort
    if role == "focused_target_view":
        return "where" in normalized and "group by" in normalized
    if role.startswith("missing"):
        return " is null" in normalized
    return True


def _degenerate_rate_detected(sql: str, target_column: str) -> bool:
    if not target_column:
        return False
    normalized = _canonical_sql(sql)
    group_cols = _extract_group_by_columns(sql)
    partition_cols = _extract_partition_by_columns(sql)
    target = target_column.lower()

    target_in_group = any(target in col for col in group_cols)
    target_in_partition = any(target in col for col in partition_cols)

    case_target_rate = (
        ("sum(case" in normalized or "avg(case" in normalized)
        and target in normalized
        and (" rate" in normalized or "_rate" in normalized or " proportion" in normalized)
    )

    count_ratio = "count(*) * 1.0 / sum(count(*)) over" in normalized

    if target_in_group and case_target_rate:
        return True
    if target_in_group and target_in_partition and count_ratio:
        return True
    return False


def _no_new_information_pattern(sql: str) -> bool:
    normalized = _canonical_sql(sql)
    if "group by" in normalized and "count(*)" in normalized and "having count(*) >" in normalized:
        return True
    if "group by" in normalized and "count(*)" in normalized and "order by" in normalized and "limit" not in normalized:
        signal_terms = ["rate", "proportion", "case when", "sum(", "avg(", "where"]
        if not any(term in normalized for term in signal_terms):
            return True
    return False


def _column_hint_tokens(column_name: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", column_name.lower())
    return {token for token in cleaned.split("_") if token}


def _support_column_indices(columns: list[str]) -> list[int]:
    indices: list[int] = []
    for idx, name in enumerate(columns):
        tokens = _column_hint_tokens(name)
        if not tokens:
            continue
        if tokens & SUPPORT_LIKE_COLUMN_HINTS and not (tokens & RATE_LIKE_COLUMN_HINTS):
            indices.append(idx)
    return indices


def _query_is_rate_like(query_spec: QuerySpec) -> bool:
    role = _infer_semantic_role(query_spec)
    if role in {"within_group_proportion", "collapsed_target_view", "ranked_signal_view"}:
        return True
    sql = _canonical_sql(query_spec.sql)
    return any(token in sql for token in [" rate", "_rate", " proportion", " / ", "over (partition by"])


def _extract_column_numeric_values(rows: list[list[Any]], col_idx: int) -> list[float]:
    values: list[float] = []
    for row in rows:
        if col_idx >= len(row):
            continue
        try:
            values.append(float(row[col_idx]))
        except (TypeError, ValueError):
            continue
    return values


def _all_between_zero_one(values: list[float]) -> bool:
    if not values:
        return False
    return all(0.0 <= value <= 1.0 for value in values)


def _support_observed_summary(query_spec: QuerySpec, execution_result: ExecutionResult) -> dict[str, Any]:
    if not execution_result.ok:
        return {"available": False, "reason": "execution_failed"}

    indices = _support_column_indices(execution_result.columns)
    values: list[float] = []
    for idx in indices:
        values.extend(_extract_column_numeric_values(execution_result.rows, idx))
    if not values:
        return {
            "available": False,
            "reason": "no_explicit_support_columns",
            "is_rate_like_query": _query_is_rate_like(query_spec),
        }
    return {
        "available": True,
        "support_columns": [execution_result.columns[idx] for idx in indices],
        "min_support": min(values),
        "max_support": max(values),
        "mean_support": round(sum(values) / max(1, len(values)), 4),
        "n_points": len(values),
    }


def _result_fingerprint(execution_result: ExecutionResult) -> str:
    if not execution_result.ok:
        return stable_hash(f"error:{execution_result.error or ''}", length=24)
    payload = {
        "columns": execution_result.columns,
        "sample_rows": execution_result.rows[:50],
        "row_count": len(execution_result.rows),
    }
    return stable_hash(str(payload), length=24)


def build_query_execution_summary_v2(
    *,
    query_spec: QuerySpec,
    execution_result: ExecutionResult,
    validation_result: ValidationResult,
) -> dict[str, Any]:
    validation_codes = list(
        dict.fromkeys(
            validation_result.static_validation.reason_codes
            + validation_result.execution_validation.reason_codes
            + validation_result.sanity_validation.reason_codes
        )
    )
    return {
        "contract_version": "query_execution_summary_v2",
        "query_id": query_spec.query_id,
        "stable_query_id": query_spec.stable_query_id,
        "question_id": query_spec.question_id,
        "stable_question_id": query_spec.stable_question_id,
        "family_id": query_spec.family_id or query_spec.family,
        "intended_facet_id": query_spec.intended_facet_id or "unknown",
        "variant_id": query_spec.variant_id or query_spec.query_id,
        "variant_semantic_role": query_spec.variant_semantic_role,
        "execution_ok": execution_result.ok,
        "row_count": len(execution_result.rows),
        "column_names": list(execution_result.columns),
        "support_observed": _support_observed_summary(query_spec, execution_result),
        "result_fingerprint": _result_fingerprint(execution_result),
        "validation_codes": validation_codes,
        "overall_validation_passed": validation_result.overall_passed,
        "alignment_evidence_codes": [],
        "purity_evidence_codes": [],
        "notes": {
            "execution_error": execution_result.error,
            "canonical_sql_hash": query_spec.canonical_sql_hash,
        },
    }


def _variant_has_reason(variant: CandidateRecord, code: str) -> bool:
    return (
        code in variant.validation.static_validation.reason_codes
        or code in variant.validation.execution_validation.reason_codes
        or code in variant.validation.sanity_validation.reason_codes
    )


def run_static_validation(
    *,
    query_spec: QuerySpec,
    table_name: str,
) -> ValidationCategoryResult:
    passed = True
    reason_codes: list[str] = []
    notes: list[str] = []

    sql_text = _canonical_sql(query_spec.sql)
    if not sql_text:
        passed = False
        reason_codes.append("VAL_STATIC_SQL_EMPTY")
    if sql_text and not sql_text.startswith("select"):
        passed = False
        reason_codes.append("VAL_STATIC_SQL_NOT_SELECT")
    if any(keyword in sql_text for keyword in [" insert ", " update ", " delete ", " drop ", " alter "]):
        passed = False
        reason_codes.append("VAL_STATIC_DML_BLOCKED")
    if table_name.lower() not in sql_text:
        reason_codes.append("VAL_STATIC_TABLE_REFERENCE_WEAK")
        notes.append("SQL does not explicitly mention expected table name.")

    if _contains_sqlite_incompatible(sql_text):
        passed = False
        reason_codes.append("VAL_STATIC_SQLITE_INCOMPATIBLE")

    if not query_spec.target_columns:
        passed = False
        reason_codes.append("VAL_STATIC_TARGET_MISSING")

    if _is_raw_projection(query_spec.sql):
        passed = False
        reason_codes.append("VAL_STATIC_RAW_EXTRACTION")
        notes.append("Raw row extraction is not accepted for benchmark query variants.")

    family_requires_agg = query_spec.family != "missingness_structure"
    if family_requires_agg and not _sql_has_structural_aggregation(query_spec.sql):
        passed = False
        reason_codes.append("VAL_STATIC_FAMILY_NEEDS_AGG")

    if query_spec.family == "missingness_structure" and "null" not in sql_text:
        passed = False
        reason_codes.append("VAL_STATIC_MISSINGNESS_SIGNAL_WEAK")

    semantic_role = _infer_semantic_role(query_spec)
    if semantic_role != "unknown" and not _role_contract_ok(semantic_role, query_spec.sql):
        passed = False
        reason_codes.append("VAL_SEMANTIC_ROLE_MISMATCH")

    target_column = query_spec.target_columns[0] if query_spec.target_columns else ""
    if _degenerate_rate_detected(query_spec.sql, target_column):
        passed = False
        reason_codes.append("VAL_DEGENERATE_RATE")

    if _no_new_information_pattern(query_spec.sql):
        reason_codes.append("VAL_NO_NEW_INFORMATION")

    if len(query_spec.subgroup_columns) + len(query_spec.feature_columns) > 4:
        reason_codes.append("VAL_STATIC_HIGH_DIMENSIONALITY_WARNING")
        notes.append("QuerySpec has high condition/group depth for v1 basic validation.")

    return ValidationCategoryResult(passed=passed, reason_codes=reason_codes, notes=notes)


def run_execution_validation(
    *,
    query_spec: QuerySpec,
    execution_result: ExecutionResult,
    support_thresholds: dict[str, Any],
) -> ValidationCategoryResult:
    passed = True
    reason_codes: list[str] = []
    notes: list[str] = []

    absolute_min_rows = int((support_thresholds or {}).get("absolute_min_rows", 20))

    if not execution_result.ok:
        passed = False
        reason_codes.append("VAL_EXEC_SQL_ERROR")
        notes.append(execution_result.error or "Execution failed")
        return ValidationCategoryResult(passed=passed, reason_codes=reason_codes, notes=notes)

    row_count = len(execution_result.rows)
    if row_count == 0:
        passed = False
        reason_codes.append("VAL_EXEC_EMPTY_RESULT")
        notes.append("SQL executed but returned 0 rows.")

    is_rate_like = _query_is_rate_like(query_spec)
    support_indices = _support_column_indices(execution_result.columns)
    support_values: list[float] = []
    for idx in support_indices:
        support_values.extend(_extract_column_numeric_values(execution_result.rows, idx))

    # Support must come from explicit support-like columns (count/total/size/denominator),
    # not from rate/proportion output values.
    if support_values:
        # Guard against mislabeled support columns that are actually normalized rates.
        if is_rate_like and _all_between_zero_one(support_values):
            reason_codes.append("VAL_EXEC_SUPPORT_NOT_OBSERVED_FOR_RATE")
            notes.append(
                "Support-like columns appear normalized (0..1); treated as rates, not support."
            )
        else:
            max_support = max(support_values)
            min_support = min(support_values)
            if max_support < absolute_min_rows:
                reason_codes.append("VAL_EXEC_LOW_SUPPORT")
                notes.append(
                    f"Maximum support ({max_support:.2f}) < threshold ({absolute_min_rows}); min_support={min_support:.2f}."
                )
            else:
                notes.append(
                    f"Support observed from explicit columns: max_support={max_support:.2f}, min_support={min_support:.2f}."
                )
    else:
        if is_rate_like:
            reason_codes.append("VAL_EXEC_SUPPORT_NOT_OBSERVED_FOR_RATE")
            notes.append(
                "Rate/proportion query without explicit support/count columns; no low-support decision from rate magnitude."
            )
        else:
            numeric_values = _extract_numeric_values(execution_result.rows)
            if numeric_values and row_count <= 2 and _all_between_zero_one(numeric_values):
                reason_codes.append("VAL_EXEC_SUPPORT_HEURISTIC_WEAK")
                notes.append(
                    "Only normalized-looking numeric outputs observed and no support column; support signal is weak."
                )
            elif numeric_values and max(numeric_values) < absolute_min_rows and row_count <= 3:
                reason_codes.append("VAL_EXEC_LOW_SUPPORT")
                notes.append(
                    "Low-support heuristic from non-support numeric outputs (weak confidence)."
                )
            else:
                reason_codes.append("VAL_EXEC_SUPPORT_HEURISTIC_WEAK")
                notes.append("No explicit support column observed; support signal is weak.")

    numeric_values = _extract_numeric_values(execution_result.rows)
    if numeric_values and len(set(round(v, 6) for v in numeric_values)) == 1 and row_count <= 3:
        reason_codes.append("VAL_NO_NEW_INFORMATION")

    if row_count == 1:
        reason_codes.append("VAL_EXEC_SINGLE_ROW_WARNING")

    return ValidationCategoryResult(passed=passed, reason_codes=reason_codes, notes=notes)


def run_question_claim_sanity_validation(
    *,
    llm_runtime: "BenchmarkLLMRuntime",
    query_spec: QuerySpec,
    execution_result: ExecutionResult,
) -> ValidationCategoryResult:
    if not execution_result.ok:
        return ValidationCategoryResult(
            passed=False,
            reason_codes=["VAL_SANITY_SKIPPED_DUE_TO_EXEC_ERROR"],
            notes=["Sanity validation skipped because execution failed."],
        )

    _ = llm_runtime  # Reserved for future LLM-assisted sanity checks.
    reason_codes: list[str] = []
    notes: list[str] = []

    columns = [str(col) for col in execution_result.columns]
    rows = execution_result.rows

    is_answering = True
    is_informative = True

    if not columns or not rows:
        is_answering = False
        reason_codes.append("VAL_SANITY_NO_RESULT_CONTENT")
        notes.append("Execution result lacks columns or rows.")

    semantic_role = _infer_semantic_role(query_spec)
    target_overlap = any(target in columns for target in query_spec.target_columns)
    if query_spec.family != "missingness_structure" and not target_overlap:
        if semantic_role == "collapsed_target_view":
            notes.append("Collapsed target view accepted without explicit raw target column.")
        elif semantic_role == "ranked_signal_view" and any(
            hint in _canonical_sql(query_spec.sql) for hint in ["focus_rate", "sum(case", "avg(case"]
        ):
            notes.append("Ranked signal view accepted with derived target signal columns.")
        else:
            reason_codes.append("VAL_SANITY_TARGET_NOT_EXPLICIT")
            notes.append("Target column not explicit in result columns.")

    if len(rows) <= 1:
        is_informative = False
        reason_codes.append("VAL_SANITY_TRIVIAL")
        notes.append("Single-row result is likely too trivial.")

    numeric_values = _extract_numeric_values(rows)
    if numeric_values and len(set(round(v, 6) for v in numeric_values)) == 1 and len(rows) <= 3:
        is_informative = False
        reason_codes.append("VAL_SANITY_LOW_VARIATION")
        notes.append("Very low variation in numeric outputs.")

    sql_text = _canonical_sql(query_spec.sql)
    question_text = query_spec.research_question.lower()
    keyword_trigger = any(key in question_text for key in ["relationship", "influence", "distribution", "across", "associated"])
    if keyword_trigger and not _sql_has_structural_aggregation(sql_text):
        is_answering = False
        reason_codes.append("VAL_SANITY_RQ_SQL_MISMATCH")
        notes.append("Research question implies structural comparison but SQL lacks grouped aggregation.")

    if query_spec.family == "missingness_structure" and "null" not in sql_text:
        is_answering = False
        reason_codes.append("VAL_SANITY_MISSINGNESS_NOT_OPERATIONALIZED")

    if semantic_role != "unknown" and not _role_contract_ok(semantic_role, query_spec.sql):
        is_answering = False
        reason_codes.append("VAL_SANITY_ROLE_CONTRACT_FAIL")

    if _is_raw_projection(query_spec.sql):
        is_answering = False
        reason_codes.append("VAL_SANITY_RAW_EXTRACTION")

    if _degenerate_rate_detected(query_spec.sql, query_spec.target_columns[0] if query_spec.target_columns else ""):
        is_answering = False
        reason_codes.append("VAL_DEGENERATE_RATE")

    if _no_new_information_pattern(query_spec.sql):
        reason_codes.append("VAL_NO_NEW_INFORMATION")

    passed = is_answering and is_informative
    if not is_answering:
        reason_codes.append("VAL_SANITY_QUESTION_MISMATCH")

    return ValidationCategoryResult(
        passed=passed,
        reason_codes=list(dict.fromkeys(reason_codes)),
        notes=notes,
    )


def run_basic_validation(
    *,
    llm_runtime: "BenchmarkLLMRuntime",
    static_understanding: StaticDatasetUnderstanding,
    query_spec: QuerySpec,
    execution_result: ExecutionResult,
    table_name: str,
) -> ValidationResult:
    thresholds = static_understanding.policy_summary.get("minimum_support_thresholds") or {}

    static_result = run_static_validation(query_spec=query_spec, table_name=table_name)
    execution_result_validation = run_execution_validation(
        query_spec=query_spec,
        execution_result=execution_result,
        support_thresholds=thresholds,
    )
    sanity_result = run_question_claim_sanity_validation(
        llm_runtime=llm_runtime,
        query_spec=query_spec,
        execution_result=execution_result,
    )

    overall_passed = static_result.passed and execution_result_validation.passed and sanity_result.passed
    return ValidationResult(
        static_validation=static_result,
        execution_validation=execution_result_validation,
        sanity_validation=sanity_result,
        overall_passed=overall_passed,
    )


def _tokenize_sql(sql: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9_]+", " ", _canonical_sql(sql))
    return {token for token in normalized.split() if token}


def _variant_information_signature(query_spec: QuerySpec) -> tuple[Any, ...]:
    sql = _canonical_sql(query_spec.sql)
    group_cols = tuple(_extract_group_by_columns(sql))
    has_rate = any(token in sql for token in [" rate", "_rate", " proportion", " over ("])
    has_case = "case when" in sql
    has_filter = " where " in sql or " having " in sql
    has_rank = "order by" in sql and (" desc" in sql or " asc" in sql)
    has_limit = " limit " in sql
    role = _infer_semantic_role(query_spec)
    return (role, group_cols, has_rate, has_case, has_filter, has_rank, has_limit)


def run_bundle_similarity_validation(
    *,
    variants: list[CandidateRecord],
    required_min_pass: int,
) -> tuple[ValidationCategoryResult, dict[str, Any]]:
    reason_codes: list[str] = []
    notes: list[str] = []
    quality_notes: list[str] = []
    pseudo_diversity_flags: list[str] = []

    total = len(variants)
    passed_variants = [item for item in variants if item.accepted_local]
    accepted_count = len(passed_variants)

    passed = True
    if accepted_count < required_min_pass:
        passed = False
        reason_codes.append("BUNDLE_PASS_COUNT_INSUFFICIENT")
        notes.append(f"accepted_variants={accepted_count} required_min_pass={required_min_pass}")

    if not passed_variants:
        passed = False
        reason_codes.append("BUNDLE_NO_ACCEPTED_VARIANTS")
        details = {
            "semantic_diversity_score": 0.0,
            "informational_novelty_score": 0.0,
            "pseudo_diversity_flags": ["no_accepted_variants"],
            "bundle_quality_notes": ["No accepted variants."],
            "bundle_reason_codes": reason_codes,
            "role_distribution": {},
            "no_new_information_count": 0,
        }
        return ValidationCategoryResult(passed=False, reason_codes=reason_codes, notes=notes), details

    roles = [_infer_semantic_role(item.query_spec) for item in passed_variants]
    role_counter = Counter(roles)
    unique_roles = len(role_counter)

    signatures = [_variant_information_signature(item.query_spec) for item in passed_variants]
    signature_counter = Counter(signatures)
    unique_signatures = len(signature_counter)

    semantic_diversity_score = unique_roles / max(1, min(accepted_count, 8))
    informational_novelty_score = unique_signatures / max(1, accepted_count)

    no_new_info_count = sum(1 for item in passed_variants if _variant_has_reason(item, "VAL_NO_NEW_INFORMATION"))
    no_new_info_ratio = no_new_info_count / max(1, accepted_count)
    if no_new_info_count:
        novelty_penalty = min(0.75, no_new_info_ratio * 0.75)
        informational_novelty_score = max(0.0, informational_novelty_score - novelty_penalty)
        for item in passed_variants:
            if _variant_has_reason(item, "VAL_NO_NEW_INFORMATION"):
                item.validation.sanity_validation.reason_codes.append("VAL_BUNDLE_INFORMATION_PENALTY")

    if unique_roles <= 3 and accepted_count >= 6:
        pseudo_diversity_flags.append("role_collision")
        reason_codes.append("VAL_BUNDLE_ROLE_COLLISION")
        for item in passed_variants:
            item.validation.sanity_validation.reason_codes.append("VAL_BUNDLE_ROLE_COLLISION")

    if informational_novelty_score < 0.6:
        pseudo_diversity_flags.append("low_informational_novelty")
        reason_codes.append("VAL_PSEUDO_DIVERSITY")

    duplicate_seen: set[tuple[Any, ...]] = set()
    redundant_count = 0
    for item in passed_variants:
        signature = _variant_information_signature(item.query_spec)
        if signature in duplicate_seen:
            redundant_count += 1
            item.validation.sanity_validation.reason_codes.append("VAL_REDUNDANT_WITHIN_BUNDLE")
            item.validation.sanity_validation.reason_codes.append("VAL_PSEUDO_DIVERSITY")
            pseudo_diversity_flags.append("redundant_signature")
        duplicate_seen.add(signature)

    if any(_variant_has_reason(item, "VAL_DEGENERATE_RATE") for item in passed_variants):
        pseudo_diversity_flags.append("degenerate_statistic")

    if no_new_info_ratio >= 0.38:
        reason_codes.append("BUNDLE_INFORMATIONAL_NOVELTY_WEAK")
        pseudo_diversity_flags.append("too_many_no_new_information")
    else:
        reason_codes.append("BUNDLE_INFORMATIONAL_NOVELTY_PASS")

    if no_new_info_ratio >= 0.5 or no_new_info_count >= max(3, required_min_pass):
        passed = False

    coherence_ok = len({item.query_spec.research_question for item in variants}) == 1 and len({item.query_spec.family for item in variants}) == 1

    if coherence_ok:
        reason_codes.append("BUNDLE_COHERENCE_PASS")
    else:
        passed = False
        reason_codes.append("BUNDLE_COHERENCE_WEAK")

    if semantic_diversity_score >= 0.6 and informational_novelty_score >= 0.6 and not pseudo_diversity_flags:
        reason_codes.append("BUNDLE_SEMANTIC_DIVERSITY_PASS")
    else:
        reason_codes.append("BUNDLE_SEMANTIC_DIVERSITY_WEAK")
        if pseudo_diversity_flags:
            reason_codes.append("BUNDLE_PSEUDO_DIVERSITY")
            passed = False

    token_sets = [_tokenize_sql(item.query_spec.sql) for item in passed_variants]
    jaccard_values: list[float] = []
    pairwise_signals: list[dict[str, Any]] = []
    for idx in range(len(token_sets)):
        for jdx in range(idx + 1, len(token_sets)):
            a = token_sets[idx]
            b = token_sets[jdx]
            if not a and not b:
                continue
            inter = len(a & b)
            union = len(a | b)
            if union > 0:
                jaccard = inter / union
                jaccard_values.append(jaccard)
                left_spec = passed_variants[idx].query_spec
                right_spec = passed_variants[jdx].query_spec
                left_sig = signatures[idx]
                right_sig = signatures[jdx]
                same_signature = left_sig == right_sig
                role_match = left_spec.variant_semantic_role == right_spec.variant_semantic_role
                novelty_heuristic = max(0.0, 1.0 - jaccard - (0.2 if same_signature else 0.0))
                pairwise_signals.append(
                    {
                        "left_query_id": left_spec.query_id,
                        "right_query_id": right_spec.query_id,
                        "left_variant_semantic_role": left_spec.variant_semantic_role,
                        "right_variant_semantic_role": right_spec.variant_semantic_role,
                        "jaccard_similarity": round(jaccard, 4),
                        "same_information_signature": same_signature,
                        "role_match": role_match,
                        "novelty_heuristic": round(novelty_heuristic, 4),
                    }
                )

    if jaccard_values:
        avg_similarity = sum(jaccard_values) / len(jaccard_values)
        notes.append(f"bundle_avg_jaccard_similarity={avg_similarity:.3f}")
        if avg_similarity < 0.2:
            passed = False
            reason_codes.append("BUNDLE_VARIANTS_TOO_DIVERSE")

    notes.append(f"bundle_pass_ratio={accepted_count}/{total}")
    notes.append(f"semantic_diversity_score={semantic_diversity_score:.3f}")
    notes.append(f"informational_novelty_score={informational_novelty_score:.3f}")
    notes.append(f"no_new_information_count={no_new_info_count}")
    notes.append(f"no_new_information_ratio={no_new_info_ratio:.3f}")

    quality_notes.append(f"role_distribution={dict(role_counter)}")
    quality_notes.append(f"unique_signatures={unique_signatures}/{accepted_count}")
    quality_notes.append(f"no_new_information_count={no_new_info_count}")
    quality_notes.append(f"no_new_information_ratio={no_new_info_ratio:.3f}")
    quality_notes.append(f"redundant_signature_count={redundant_count}")
    if pseudo_diversity_flags:
        quality_notes.append(f"pseudo_diversity_flags={sorted(set(pseudo_diversity_flags))}")
    if coherence_ok:
        quality_notes.append("bundle_coherence=pass")
    else:
        quality_notes.append("bundle_coherence=weak")

    details = {
        "semantic_diversity_score": round(semantic_diversity_score, 4),
        "informational_novelty_score": round(informational_novelty_score, 4),
        "pseudo_diversity_flags": sorted(set(pseudo_diversity_flags)),
        "bundle_quality_notes": quality_notes,
        "bundle_reason_codes": sorted(set(reason_codes)),
        "role_distribution": dict(role_counter),
        "no_new_information_count": no_new_info_count,
        "no_new_information_ratio": round(no_new_info_ratio, 4),
        "accepted_variant_count": accepted_count,
        "informative_variant_count": accepted_count - no_new_info_count,
        "pairwise_diversity_signals": pairwise_signals,
        "bundle_diversity_score": round(semantic_diversity_score, 4),
        "bundle_novelty_score": round(informational_novelty_score, 4),
    }

    return ValidationCategoryResult(
        passed=passed,
        reason_codes=list(dict.fromkeys(reason_codes)),
        notes=notes,
    ), details
