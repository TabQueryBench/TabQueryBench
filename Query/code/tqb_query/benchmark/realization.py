"""ResearchQuestion -> QuerySpec -> SQL realization and repair."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any
from uuid import uuid4

from tqb_query.benchmark.canonical_sql import (
    canonical_sql_hash,
    canonicalize_sql,
    stable_question_identity,
    stable_query_identity,
)
from tqb_query.benchmark.llm_runtime import BenchmarkLLMRuntime
from tqb_query.benchmark.models import QuerySpec, ResearchQuestion, StaticDatasetUnderstanding
from tqb_query.benchmark.sql_exemplars import SQLExemplarRepository

SQLITE_INCOMPATIBLE_PATTERNS = [
    r"\bfield\s*\(",
    r"\bilike\b",
    r"\bdate_trunc\s*\(",
    r"\bregexp\b",
]


def _normalize_sql(sql: str) -> str:
    text = sql.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\\n", "", text)
        text = re.sub(r"\\n```$", "", text)
    text = text.strip()
    if text.endswith(";"):
        return text
    return text + ";"


def _family_contract(family: str) -> str:
    contracts = {
        "subgroup_structure": "Must include grouped subgroup/target distribution with meaningful comparison structure.",
        "conditional_dependency_structure": "Must include at least two conditioning/group fields plus target summary.",
        "tail_rarity_structure": "Must expose rare/low-support tail behavior via support-aware summaries.",
        "missingness_structure": "Must explicitly reason about NULL/missingness indicator columns.",
        "cardinality_structure": "Must summarize concentration/cardinality-like support patterns.",
    }
    return contracts.get(family, "Must stay aligned with the research question.")


def _canonical_sql(sql: str) -> str:
    text = sql.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(";")


def _dedupe_sqls(sqls: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in sqls:
        canonical = _canonical_sql(item)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        unique.append(_normalize_sql(item))
    return unique


def _is_raw_projection(sql: str) -> bool:
    lowered = _canonical_sql(sql)
    has_agg = any(token in lowered for token in [" group by ", "count(", "sum(", "avg(", " over (", "having "])
    if has_agg:
        return False
    return bool(re.fullmatch(r"select\s+[\w\s,.*]+\s+from\s+\w+(\s+where\s+.+)?", lowered))


def _semantic_role_code(role: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_")
    return f"SQL_VARIANT_SEMANTIC_ROLE_{normalized.upper()}"


def _diversity_intent_tag(role: str) -> str:
    mapping = {
        "count_distribution": "absolute_count_view",
        "within_group_proportion": "within_group_rate_view",
        "collapsed_target_view": "target_collapse_view",
        "ranked_signal_view": "ranked_signal_view",
        "filtered_stable_view": "support_guard_view",
        "rare_extreme_view": "tail_extreme_view",
        "focused_target_view": "target_focus_view",
        "contrastive_conditional_view": "conditional_contrast_view",
        "missing_indicator_view": "missing_indicator_view",
        "missing_target_interaction": "missing_target_interaction_view",
        "missing_rate_by_subgroup": "missing_rate_view",
        "missing_ranked_view": "missing_ranked_view",
    }
    return mapping.get(role, "unknown")


def _output_semantics(role: str) -> str:
    if role in {"count_distribution", "filtered_stable_view", "rare_extreme_view"}:
        return "grouped_support_table"
    if role in {"within_group_proportion", "collapsed_target_view", "ranked_signal_view", "missing_rate_by_subgroup"}:
        return "grouped_rate_table"
    if role in {"focused_target_view", "contrastive_conditional_view"}:
        return "filtered_grouped_support_table"
    if role.startswith("missing"):
        return "missingness_summary_table"
    return "grouped_summary_table"


def _structure_claim(family: str, role: str) -> str:
    return f"{family}:{role}"


def _infer_secondary_family_candidates(family: str, role: str) -> list[str]:
    if family == "subgroup_structure" and role in {"within_group_proportion", "ranked_signal_view"}:
        return ["conditional_dependency_structure"]
    if family == "conditional_dependency_structure" and role in {"rare_extreme_view", "filtered_stable_view"}:
        return ["tail_rarity_structure"]
    if family == "tail_rarity_structure" and role in {"within_group_proportion", "ranked_signal_view"}:
        return ["subgroup_structure", "conditional_dependency_structure"]
    if family == "cardinality_structure" and role in {"ranked_signal_view"}:
        return ["tail_rarity_structure"]
    return []


def _infer_contamination_hints(
    *,
    family: str,
    role: str,
    subgroup_columns: list[str],
    feature_columns: list[str],
) -> list[str]:
    hints: list[str] = []
    if family != "missingness_structure" and any("missing" in col.lower() for col in subgroup_columns + feature_columns):
        hints.append("possible_missingness_leakage")
    if role == "count_distribution" and len(subgroup_columns) + len(feature_columns) <= 1:
        hints.append("weak_structural_specificity")
    if role == "ranked_signal_view" and family not in {"tail_rarity_structure", "conditional_dependency_structure"}:
        hints.append("ranking_may_shift_family_semantics")
    return hints


def _expected_result_schema(expected_output_shape: str, role: str) -> str:
    base = expected_output_shape or "table"
    if base == "table":
        return _output_semantics(role)
    return base


def _contains_sqlite_incompatible(sql: str) -> bool:
    normalized = _canonical_sql(sql)
    return any(re.search(pattern, normalized) for pattern in SQLITE_INCOMPATIBLE_PATTERNS)


def _extract_groupby_columns_sql(sql: str) -> list[str]:
    normalized = _canonical_sql(sql)
    match = re.search(r"group by\s+(.+?)(order by|having|limit|$)", normalized)
    if not match:
        return []
    clause = (match.group(1) or "").strip()
    if not clause:
        return []
    out: list[str] = []
    for part in clause.split(","):
        token = part.strip()
        if token and token not in out:
            out.append(token)
    return out


def _extract_where_filters(sql: str) -> list[str]:
    normalized = _canonical_sql(sql)
    match = re.search(r"where\s+(.+?)(group by|order by|having|limit|$)", normalized)
    if not match:
        return []
    where_clause = (match.group(1) or "").strip()
    if not where_clause:
        return []
    parts = re.split(r"\s+and\s+", where_clause, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def _infer_aggregate_type(sql: str) -> str:
    normalized = _canonical_sql(sql)
    if "count(" in normalized:
        return "count"
    if "avg(" in normalized:
        return "avg"
    if "sum(" in normalized:
        return "sum"
    if "min(" in normalized:
        return "min"
    if "max(" in normalized:
        return "max"
    return "unknown"


def _infer_direction(role: str) -> str:
    if role in {"within_group_proportion", "collapsed_target_view", "ranked_signal_view"}:
        return "higher"
    if role in {"rare_extreme_view", "missing_ranked_view"}:
        return "present"
    if role == "count_distribution":
        return "higher"
    return "unknown"


def _normalize_claim_type_for_protocol(claim_type: str, role: str) -> str:
    raw = (claim_type or "").strip().lower()
    mapping = {
        "higher_lower_comparison": "higher_lower_comparison",
        "higher_lower": "higher_lower_comparison",
        "distribution": "higher_lower_comparison",
        "rate": "higher_lower_comparison",
        "contrast": "higher_lower_comparison",
        "ranking": "higher_lower_comparison",
        "monotonic_trend": "monotonic_trend",
        "trend": "monotonic_trend",
        "association_direction": "association_direction",
        "association": "association_direction",
        "rare_pattern_presence": "rare_pattern_presence",
        "rare": "rare_pattern_presence",
    }
    normalized = mapping.get(raw, "")
    if normalized:
        return normalized
    if role in {"within_group_proportion", "collapsed_target_view"}:
        return "higher_lower_comparison"
    if role in {"ranked_signal_view"}:
        return "higher_lower_comparison"
    if role in {"rare_extreme_view"}:
        return "rare_pattern_presence"
    return "higher_lower_comparison"


def _family_semantic_roles(family: str, num_variants: int) -> list[str]:
    if family == "missingness_structure":
        base = [
            "missing_indicator_view",
            "missing_target_interaction",
            "missing_rate_by_subgroup",
            "missing_ranked_view",
            "filtered_stable_view",
        ]
    elif family == "tail_rarity_structure":
        base = [
            "count_distribution",
            "within_group_proportion",
            "rare_extreme_view",
            "ranked_signal_view",
            "filtered_stable_view",
        ]
    else:
        base = [
            "count_distribution",
            "within_group_proportion",
            "collapsed_target_view",
            "ranked_signal_view",
            "filtered_stable_view",
        ]

    if num_variants <= len(base):
        return base[:num_variants]

    extra_roles = ["focused_target_view", "contrastive_conditional_view", "rare_extreme_view"]
    roles = list(base)
    idx = 0
    while len(roles) < num_variants:
        roles.append(extra_roles[idx % len(extra_roles)])
        idx += 1
    return roles


def _normalize_role(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized


def _select_focus_target_labels(target_labels: list[str]) -> list[str]:
    if not target_labels:
        return []
    positives = [
        label
        for label in target_labels
        if label.lower() in {"acc", "good", "vgood", "positive", "yes", "true", "1"}
    ]
    if positives:
        return positives
    if len(target_labels) == 1:
        return target_labels
    # Heuristic fallback: use the upper half of ordered labels as "more positive" side.
    midpoint = max(1, len(target_labels) // 2)
    return target_labels[midpoint:]


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _build_collapsed_case(target_column: str, target_labels: list[str]) -> tuple[str, str]:
    focus_labels = _select_focus_target_labels(target_labels)
    if not focus_labels:
        return f"CASE WHEN {target_column} IS NOT NULL THEN 'focus' ELSE 'other' END", "focus"
    in_clause = ", ".join(_sql_quote(label) for label in focus_labels)
    return f"CASE WHEN {target_column} IN ({in_clause}) THEN 'focus' ELSE 'other' END", "focus"


def _dedupe_preserve(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        token = str(item).strip()
        if not token or token in out:
            continue
        out.append(token)
    return out


def _sanitize_structural_columns(columns: list[str], target_column: str, *, max_len: int = 4) -> list[str]:
    out: list[str] = []
    for col in _dedupe_preserve(columns):
        if col == target_column:
            continue
        out.append(col)
        if len(out) >= max_len:
            break
    return out


def _desired_combo_depth(
    *,
    family: str,
    question_token: str,
    max_depth: int,
) -> int:
    if max_depth <= 0:
        return 0
    # 1/2/3 are primary, 4 appears occasionally.
    cycle = [1, 2, 3, 2, 1, 3, 4]
    digest = hashlib.sha1(f"{family}|{question_token}".encode("utf-8")).hexdigest()
    depth = cycle[int(digest[:8], 16) % len(cycle)]
    if family == "conditional_dependency_structure":
        depth = min(depth, 3)
    if family == "missingness_structure":
        depth = min(depth, 2)
    return max(1, min(max_depth, depth))


def _expand_columns_to_depth(columns: list[str], pool: list[str], depth: int) -> list[str]:
    out = list(_dedupe_preserve(columns))
    for col in pool:
        if col in out:
            continue
        out.append(col)
        if len(out) >= depth:
            break
    return out


def _core_fields(subgroup_columns: list[str], feature_columns: list[str], target_column: str) -> list[str]:
    fields: list[str] = []
    for field in subgroup_columns + feature_columns:
        if field == target_column:
            continue
        if field not in fields:
            fields.append(field)
        if len(fields) >= 4:
            break
    return fields


def _role_sql_template(
    *,
    role: str,
    family: str,
    table_name: str,
    target_column: str,
    target_labels: list[str],
    subgroup_columns: list[str],
    feature_columns: list[str],
) -> str:
    core = _core_fields(subgroup_columns, feature_columns, target_column)
    group_with_target = core + [target_column]
    group_clause = ", ".join(group_with_target)
    core_clause = ", ".join(core)

    collapsed_case, _focus_bucket = _build_collapsed_case(target_column, target_labels)
    focus_labels = _select_focus_target_labels(target_labels)
    focus_condition = ""
    if focus_labels:
        focus_condition = f"{target_column} IN ({', '.join(_sql_quote(label) for label in focus_labels)})"
    else:
        focus_condition = f"{target_column} IS NOT NULL"

    if role == "count_distribution":
        return (
            f"SELECT {group_clause}, COUNT(*) AS support "
            f"FROM {table_name} "
            f"GROUP BY {group_clause} "
            f"ORDER BY support DESC, {group_clause} LIMIT 200;"
        )

    if role == "within_group_proportion":
        if core:
            partition_clause = core_clause
            return (
                f"SELECT {group_clause}, COUNT(*) AS support, "
                f"ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY {partition_clause}), 4) AS within_group_rate "
                f"FROM {table_name} "
                f"GROUP BY {group_clause} "
                f"ORDER BY {partition_clause}, within_group_rate DESC, support DESC LIMIT 200;"
            )
        return (
            f"SELECT {target_column}, COUNT(*) AS support, "
            f"ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 4) AS global_rate "
            f"FROM {table_name} "
            f"GROUP BY {target_column} "
            f"ORDER BY global_rate DESC, support DESC;"
        )

    if role == "collapsed_target_view":
        bucket_alias = "target_bucket"
        select_core = f"{core_clause}, " if core_clause else ""
        group_bucket = f"{core_clause}, {bucket_alias}" if core_clause else bucket_alias
        if core_clause:
            rate_expr = f"ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY {core_clause}), 4) AS bucket_rate"
        else:
            rate_expr = "ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 4) AS bucket_rate"
        return (
            f"SELECT {select_core}{collapsed_case} AS {bucket_alias}, COUNT(*) AS support, "
            f"{rate_expr} "
            f"FROM {table_name} "
            f"GROUP BY {group_bucket} "
            f"ORDER BY {group_bucket}, support DESC;"
        )

    if role == "ranked_signal_view":
        select_core = core_clause if core_clause else target_column
        group_rank = core_clause if core_clause else target_column
        return (
            f"SELECT {select_core}, "
            f"SUM(CASE WHEN {focus_condition} THEN 1 ELSE 0 END) AS focus_count, "
            f"COUNT(*) AS total_count, "
            f"ROUND(SUM(CASE WHEN {focus_condition} THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS focus_rate "
            f"FROM {table_name} "
            f"GROUP BY {group_rank} "
            f"HAVING COUNT(*) >= 3 "
            f"ORDER BY focus_rate DESC, total_count DESC LIMIT 40;"
        )

    if role == "filtered_stable_view":
        stable_threshold = 3
        return (
            f"SELECT {group_clause}, COUNT(*) AS support "
            f"FROM {table_name} "
            f"GROUP BY {group_clause} "
            f"HAVING COUNT(*) >= {stable_threshold} "
            f"ORDER BY support DESC, {group_clause} LIMIT 200;"
        )

    if role == "rare_extreme_view":
        return (
            f"SELECT {group_clause}, COUNT(*) AS support "
            f"FROM {table_name} "
            f"GROUP BY {group_clause} "
            f"ORDER BY support ASC, {group_clause} LIMIT 80;"
        )

    if role == "focused_target_view":
        return (
            f"SELECT {group_clause}, COUNT(*) AS support "
            f"FROM {table_name} "
            f"WHERE {focus_condition} "
            f"GROUP BY {group_clause} "
            f"ORDER BY support DESC, {group_clause} LIMIT 120;"
        )

    if role == "contrastive_conditional_view":
        if core:
            first_field = core[0]
            return (
                f"SELECT {group_clause}, COUNT(*) AS support "
                f"FROM {table_name} "
                f"WHERE {first_field} IS NOT NULL "
                f"GROUP BY {group_clause} "
                f"ORDER BY {first_field}, support DESC LIMIT 160;"
            )
        return (
            f"SELECT {target_column}, COUNT(*) AS support "
            f"FROM {table_name} GROUP BY {target_column} ORDER BY support DESC;"
        )

    if role == "missing_indicator_view":
        probe_field = core[0] if core else target_column
        return (
            f"SELECT CASE WHEN {probe_field} IS NULL THEN 'missing' ELSE 'not_missing' END AS missing_flag, COUNT(*) AS support "
            f"FROM {table_name} GROUP BY missing_flag ORDER BY missing_flag;"
        )

    if role == "missing_target_interaction":
        probe_field = core[0] if core else target_column
        return (
            f"SELECT {target_column}, CASE WHEN {probe_field} IS NULL THEN 'missing' ELSE 'not_missing' END AS missing_flag, "
            f"COUNT(*) AS support FROM {table_name} "
            f"GROUP BY {target_column}, missing_flag ORDER BY {target_column}, missing_flag;"
        )

    if role == "missing_rate_by_subgroup":
        probe_field = core[0] if core else target_column
        subgroup_for_missing = core[1] if len(core) > 1 else target_column
        return (
            f"SELECT {subgroup_for_missing}, "
            f"SUM(CASE WHEN {probe_field} IS NULL THEN 1 ELSE 0 END) AS missing_count, "
            f"COUNT(*) AS total_count, "
            f"ROUND(SUM(CASE WHEN {probe_field} IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS missing_rate "
            f"FROM {table_name} GROUP BY {subgroup_for_missing} ORDER BY missing_rate DESC, total_count DESC;"
        )

    if role == "missing_ranked_view":
        probe_field = core[0] if core else target_column
        subgroup_for_missing = core[1] if len(core) > 1 else target_column
        return (
            f"SELECT {subgroup_for_missing}, CASE WHEN {probe_field} IS NULL THEN 'missing' ELSE 'not_missing' END AS missing_flag, "
            f"COUNT(*) AS support FROM {table_name} "
            f"GROUP BY {subgroup_for_missing}, missing_flag ORDER BY support DESC, {subgroup_for_missing};"
        )

    # Default fallback.
    return (
        f"SELECT {group_clause}, COUNT(*) AS support "
        f"FROM {table_name} GROUP BY {group_clause} ORDER BY support DESC, {group_clause} LIMIT 200;"
    )


def _role_sql_expectation_ok(role: str, sql: str, target_column: str) -> bool:
    normalized = _canonical_sql(sql)
    if not normalized:
        return False
    if not normalized.startswith("select"):
        return False
    if _contains_sqlite_incompatible(normalized):
        return False
    if _is_raw_projection(sql):
        return False

    if role in {
        "count_distribution",
        "within_group_proportion",
        "collapsed_target_view",
        "ranked_signal_view",
        "filtered_stable_view",
        "rare_extreme_view",
        "focused_target_view",
        "contrastive_conditional_view",
    } and " group by " not in normalized:
        return False

    if role == "within_group_proportion":
        if "over (" not in normalized and "rate" not in normalized and "/" not in normalized:
            return False

    if role == "collapsed_target_view":
        if "case when" not in normalized:
            return False

    if role == "ranked_signal_view":
        if "focus_rate" not in normalized:
            return False
        if "total_count" not in normalized and "count(*) as" not in normalized:
            return False
        if "order by" not in normalized:
            return False
        if "order by support" in normalized or "order by count" in normalized:
            return False

    if role in {"missing_indicator_view", "missing_target_interaction", "missing_rate_by_subgroup", "missing_ranked_view"}:
        if " is null" not in normalized:
            return False

    # Degenerate rate pattern: target grouped while rate condition also directly uses target.
    if "case when" in normalized and target_column.lower() in normalized and "group by" in normalized:
        group_match = re.search(r"group by\s+(.+?)(\s+order by|\s+having|\s+limit|$)", normalized)
        if group_match:
            group_clause = group_match.group(1)
            if target_column.lower() in group_clause and "rate" in normalized:
                return False

    return True


def _fallback_sql_variants(
    *,
    family: str,
    table_name: str,
    target_column: str,
    target_labels: list[str],
    subgroup_columns: list[str],
    feature_columns: list[str],
    num_variants: int,
    roles: list[str],
) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    for role in roles[:num_variants]:
        sql = _role_sql_template(
            role=role,
            family=family,
            table_name=table_name,
            target_column=target_column,
            target_labels=target_labels,
            subgroup_columns=subgroup_columns,
            feature_columns=feature_columns,
        )
        variants.append((role, sql))
    return variants


def _refresh_query_spec_contract(
    *,
    query_spec: QuerySpec,
    static_understanding: StaticDatasetUnderstanding,
) -> None:
    canonical = canonicalize_sql(query_spec.sql)
    query_spec.canonical_sql = canonical
    query_spec.canonical_sql_hash = canonical_sql_hash(query_spec.sql)
    if not query_spec.family_id:
        query_spec.family_id = query_spec.family
    if not query_spec.variant_id:
        query_spec.variant_id = query_spec.query_id
    if not query_spec.stable_question_id:
        query_spec.stable_question_id = stable_question_identity(
            dataset_id=static_understanding.dataset_id,
            family_id=query_spec.family_id,
            intended_facet_id=query_spec.intended_facet_id,
            question_text=query_spec.research_question,
        )
    query_spec.stable_query_id = stable_query_identity(
        dataset_id=static_understanding.dataset_id,
        family_id=query_spec.family_id,
        intended_facet_id=query_spec.intended_facet_id,
        stable_question_id=query_spec.stable_question_id,
        variant_semantic_role=query_spec.variant_semantic_role,
        canonical_sql=canonical,
    )
    if not query_spec.diversity_intent_tag:
        query_spec.diversity_intent_tag = _diversity_intent_tag(query_spec.variant_semantic_role)
    if query_spec.intended_structure_claim in {"", "unknown"}:
        query_spec.intended_structure_claim = _structure_claim(query_spec.family, query_spec.variant_semantic_role)
    if not query_spec.expected_result_schema or query_spec.expected_result_schema == "unknown":
        query_spec.expected_result_schema = _expected_result_schema(
            query_spec.expected_output_shape,
            query_spec.variant_semantic_role,
        )
    if not query_spec.output_semantics or query_spec.output_semantics == "unknown":
        query_spec.output_semantics = _output_semantics(query_spec.variant_semantic_role)
    if not query_spec.secondary_family_candidates:
        query_spec.secondary_family_candidates = _infer_secondary_family_candidates(
            query_spec.family,
            query_spec.variant_semantic_role,
        )
    if not query_spec.contamination_risk_hints:
        query_spec.contamination_risk_hints = _infer_contamination_hints(
            family=query_spec.family,
            role=query_spec.variant_semantic_role,
            subgroup_columns=query_spec.subgroup_columns,
            feature_columns=query_spec.feature_columns,
        )
    if not query_spec.source_columns:
        dedup: list[str] = []
        for column in query_spec.target_columns + query_spec.subgroup_columns + query_spec.feature_columns:
            if column and column not in dedup:
                dedup.append(column)
        query_spec.source_columns = dedup
    if not query_spec.groupby_columns:
        query_spec.groupby_columns = _extract_groupby_columns_sql(query_spec.sql)
    if not query_spec.aggregate_type or query_spec.aggregate_type == "unknown":
        query_spec.aggregate_type = _infer_aggregate_type(query_spec.sql)
    if not query_spec.measure_column or query_spec.measure_column == "unknown":
        query_spec.measure_column = query_spec.target_columns[0] if query_spec.target_columns else static_understanding.target_column
    if not query_spec.base_filters and not query_spec.optional_filters:
        filters = _extract_where_filters(query_spec.sql)
        if len(filters) > 1:
            query_spec.base_filters = [filters[0]]
            query_spec.optional_filters = filters[1:]
        elif len(filters) == 1:
            query_spec.optional_filters = filters
    if not query_spec.allowed_refinement_columns:
        query_spec.allowed_refinement_columns = list(
            dict.fromkeys([col for col in query_spec.source_columns if col and col not in query_spec.target_columns])
        )
    if not query_spec.comparison_target or query_spec.comparison_target == "unknown":
        query_spec.comparison_target = query_spec.subgroup_columns[0] if query_spec.subgroup_columns else "unknown"
    if not query_spec.direction or query_spec.direction == "unknown":
        query_spec.direction = _infer_direction(query_spec.variant_semantic_role)
    if not query_spec.frozen_slots:
        query_spec.frozen_slots = [
            "base_table",
            "join_graph",
            "aggregate_type",
            "measure_column",
            "comparison_entities",
            "direction_semantics",
            "mandatory_filters",
            "family_label",
        ]
    if not query_spec.editable_slots:
        query_spec.editable_slots = ["optional_filter", "threshold_adjacent_bin", "refinement_column", "population_step"]
    query_spec.query_spec_contract_version = "query_spec_acr_v1"


def realize_query_spec_variants(
    *,
    llm_runtime: BenchmarkLLMRuntime,
    static_understanding: StaticDatasetUnderstanding,
    research_question: ResearchQuestion,
    table_name: str,
    num_variants: int,
    sql_exemplar_repo: SQLExemplarRepository | None = None,
    available_columns: list[str] | None = None,
    exemplar_max_candidates_per_role: int = 4,
) -> list[QuerySpec]:
    role_plan = _family_semantic_roles(research_question.family, num_variants)

    system_prompt = """
You convert a tabular benchmark research question into a QuerySpec scaffold with executable SQLite SQL variants.
Rules:
- Single table only.
- SQL must be read-only SELECT.
- Variants must answer the same research question and stay coherent.
- Variants should prioritize semantic diversity (different analysis views), not just syntax rewrites.
- Avoid raw row extraction.
Return strict JSON with fields:
{
  "claim_type": "...",
  "target_columns": ["..."],
  "subgroup_columns": ["1-4 columns..."],
  "feature_columns": ["1-4 columns..."],
  "expected_output_shape": "...",
  "reason_codes": ["..."],
  "semantic_roles": ["..."],
  "variants": [
    {
      "semantic_role": "...",
      "sql": "...",
      "reason_codes": ["..."]
    }
  ]
}
""".strip()

    user_prompt = (
        f"dataset_id={static_understanding.dataset_id}\n"
        f"table_name={table_name}\n"
        f"family={research_question.family}\n"
        f"family_contract={_family_contract(research_question.family)}\n"
        f"research_question={research_question.question}\n"
        f"related_fields={research_question.related_fields}\n"
        f"target_column={static_understanding.target_column}\n"
        f"target_labels={static_understanding.target_labels}\n"
        f"ordered_fields={static_understanding.ordered_fields}\n"
        f"task_type={static_understanding.task_type}\n"
        f"num_variants={num_variants}\n"
        f"required_semantic_roles={role_plan}\n"
    )

    payload = llm_runtime.invoke_json(
        phase="queryspec_generation",
        module="realization.realize_query_spec_variants",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        question_for_usage=research_question.question,
    )

    claim_type = str(payload.get("claim_type") or "distribution")
    target_columns = _dedupe_preserve([str(v) for v in (payload.get("target_columns") or []) if isinstance(v, str)])
    subgroup_columns = _dedupe_preserve([str(v) for v in (payload.get("subgroup_columns") or []) if isinstance(v, str)])
    feature_columns = _dedupe_preserve([str(v) for v in (payload.get("feature_columns") or []) if isinstance(v, str)])
    expected_output_shape = str(payload.get("expected_output_shape") or "table")
    reason_codes = [str(v) for v in (payload.get("reason_codes") or []) if isinstance(v, str)]

    if not target_columns:
        target_columns = [static_understanding.target_column]
    target_column_ref = target_columns[0] if target_columns else static_understanding.target_column
    related_pool = _sanitize_structural_columns(research_question.related_fields, target_column_ref, max_len=4)
    subgroup_columns = _sanitize_structural_columns(subgroup_columns, target_column_ref, max_len=4)
    feature_columns = _sanitize_structural_columns(feature_columns, target_column_ref, max_len=4)

    if not subgroup_columns and related_pool:
        subgroup_columns = [related_pool[0]]
    if not feature_columns and related_pool:
        feature_columns = [field for field in related_pool if field not in target_columns][:4]

    if related_pool and research_question.family in {
        "subgroup_structure",
        "conditional_dependency_structure",
        "tail_rarity_structure",
        "cardinality_structure",
    }:
        question_token = research_question.stable_question_id or research_question.question_id or research_question.question
        target_depth = _desired_combo_depth(
            family=research_question.family,
            question_token=question_token,
            max_depth=min(4, len(related_pool)),
        )
        subgroup_columns = _expand_columns_to_depth(subgroup_columns, related_pool, target_depth)

    subgroup_columns = _sanitize_structural_columns(subgroup_columns, target_column_ref, max_len=4)
    feature_columns = _sanitize_structural_columns(feature_columns, target_column_ref, max_len=4)
    if not feature_columns and subgroup_columns:
        feature_columns = list(subgroup_columns)

    # Normalize optional role suggestions from model, but keep family-constrained role plan as source of truth.
    payload_roles = payload.get("semantic_roles") if isinstance(payload, dict) else None
    if isinstance(payload_roles, list):
        suggested = [_normalize_role(str(item)) for item in payload_roles if isinstance(item, str)]
        if suggested:
            merged: list[str] = []
            for role in role_plan + suggested:
                role_normalized = _normalize_role(role)
                if role_normalized not in merged:
                    merged.append(role_normalized)
            role_plan = merged[:num_variants]

    variants_by_role: dict[str, tuple[str, list[str]]] = {}
    variant_items = payload.get("variants") if isinstance(payload, dict) else None
    if isinstance(variant_items, list):
        for item in variant_items:
            if not isinstance(item, dict):
                continue
            role = _normalize_role(str(item.get("semantic_role") or ""))
            sql = str(item.get("sql") or "").strip()
            codes = [str(v) for v in (item.get("reason_codes") or []) if isinstance(v, str)]
            if not role or not sql:
                continue
            if role not in variants_by_role:
                variants_by_role[role] = (sql, codes)

    prepared_variants: list[tuple[str, str, list[str], dict[str, Any]]] = []
    columns_for_adapt = (
        list(dict.fromkeys([col for col in (available_columns or []) if col]))
        or list(dict.fromkeys(target_columns + subgroup_columns + feature_columns))
    )

    for role in role_plan:
        role_codes: list[str] = []
        origin = {
            "origin_mode": "de_novo",
            "exemplar_sql_item_id": "",
            "exemplar_own_id": "",
            "exemplar_source_url": "",
            "exemplar_match_score": 0.0,
            "exemplar_transform_notes": [],
        }
        llm_variant = variants_by_role.get(role)
        sql = ""
        exemplar_candidates = []
        if sql_exemplar_repo is not None:
            exemplar_candidates = sql_exemplar_repo.get_candidates(
                dataset_id=static_understanding.dataset_id,
                table_name=table_name,
                available_columns=columns_for_adapt,
                family=research_question.family,
                role=role,
                question=research_question.question,
                related_fields=research_question.related_fields,
                target_column=static_understanding.target_column,
                max_candidates=max(1, exemplar_max_candidates_per_role),
            )
        for candidate in exemplar_candidates:
            candidate_sql = _normalize_sql(candidate.sql)
            if not _role_sql_expectation_ok(role, candidate_sql, static_understanding.target_column):
                continue
            sql = candidate_sql
            origin = {
                "origin_mode": candidate.origin_mode,
                "exemplar_sql_item_id": candidate.sql_item_id,
                "exemplar_own_id": candidate.own_id,
                "exemplar_source_url": candidate.source_url,
                "exemplar_match_score": candidate.match_score,
                "exemplar_transform_notes": list(candidate.transform_notes),
            }
            if candidate.origin_mode == "direct_reuse":
                role_codes.append("SQL_VARIANT_SOURCE_DIRECT_REUSE")
            else:
                role_codes.append("SQL_VARIANT_SOURCE_TEMPLATE_ADAPT")
            role_codes.append("SQL_VARIANT_FROM_EXEMPLAR")
            break

        if llm_variant:
            llm_sql = llm_variant[0]
            if (not sql) and _role_sql_expectation_ok(role, llm_sql, static_understanding.target_column):
                sql = llm_sql
                role_codes.extend(llm_variant[1])
                role_codes.append("SQL_VARIANT_FROM_LLM")
                origin["origin_mode"] = "de_novo"

        if not sql or not _role_sql_expectation_ok(role, sql, static_understanding.target_column):
            sql = _role_sql_template(
                role=role,
                family=research_question.family,
                table_name=table_name,
                target_column=static_understanding.target_column,
                target_labels=static_understanding.target_labels,
                subgroup_columns=subgroup_columns,
                feature_columns=feature_columns,
            )
            role_codes.append("SQL_VARIANT_TEMPLATE_BACKFILL")
            origin["origin_mode"] = "de_novo"

        prepared_variants.append((role, _normalize_sql(sql), role_codes, origin))

    if len(prepared_variants) < num_variants:
        fallback = _fallback_sql_variants(
            family=research_question.family,
            table_name=table_name,
            target_column=static_understanding.target_column,
            target_labels=static_understanding.target_labels,
            subgroup_columns=subgroup_columns,
            feature_columns=feature_columns,
            num_variants=num_variants,
            roles=_family_semantic_roles(research_question.family, num_variants),
        )
        for role, sql in fallback:
            if len(prepared_variants) >= num_variants:
                break
            prepared_variants.append(
                (
                    role,
                    _normalize_sql(sql),
                    ["SQL_VARIANT_FALLBACK"],
                    {
                        "origin_mode": "de_novo",
                        "exemplar_sql_item_id": "",
                        "exemplar_own_id": "",
                        "exemplar_source_url": "",
                        "exemplar_match_score": 0.0,
                        "exemplar_transform_notes": [],
                    },
                )
            )
        reason_codes.append("SQL_VARIANT_FALLBACK")

    # Ensure semantic roles and SQLs are not duplicate placeholders.
    used_pairs: set[tuple[str, str]] = set()
    final_variants: list[tuple[str, str, list[str], dict[str, Any]]] = []
    for role, sql, role_codes, origin in prepared_variants:
        key = (role, _canonical_sql(sql))
        if key in used_pairs:
            continue
        used_pairs.add(key)
        final_variants.append((role, sql, role_codes, origin))
        if len(final_variants) >= num_variants:
            break

    if len(final_variants) < num_variants:
        # Final safety net: generate deterministic variants for missing slots.
        fallback_roles = _family_semantic_roles(research_question.family, num_variants)
        for role in fallback_roles:
            if len(final_variants) >= num_variants:
                break
            sql = _normalize_sql(
                _role_sql_template(
                    role=role,
                    family=research_question.family,
                    table_name=table_name,
                    target_column=static_understanding.target_column,
                    target_labels=static_understanding.target_labels,
                    subgroup_columns=subgroup_columns,
                    feature_columns=feature_columns,
                )
            )
            key = (role, _canonical_sql(sql))
            if key in used_pairs:
                continue
            used_pairs.add(key)
            final_variants.append(
                (
                    role,
                    sql,
                    ["SQL_VARIANT_FALLBACK_HARD"],
                    {
                        "origin_mode": "de_novo",
                        "exemplar_sql_item_id": "",
                        "exemplar_own_id": "",
                        "exemplar_source_url": "",
                        "exemplar_match_score": 0.0,
                        "exemplar_transform_notes": [],
                    },
                )
            )
        reason_codes.append("SQL_VARIANT_FALLBACK_HARD")

    query_specs: list[QuerySpec] = []
    base_codes = list(dict.fromkeys(research_question.reason_codes + reason_codes + ["QS_FROM_RESEARCH_QUESTION"]))

    for idx, (role, sql, variant_codes, origin) in enumerate(final_variants[:num_variants]):
        claim_type_norm = _normalize_claim_type_for_protocol(claim_type, role)
        role_code = _semantic_role_code(role)
        codes = list(dict.fromkeys(base_codes + variant_codes + [role_code, f"SQL_VARIANT_{idx + 1}"]))
        source_columns: list[str] = []
        for column in target_columns + subgroup_columns + feature_columns:
            if column and column not in source_columns:
                source_columns.append(column)

        variant_id = f"{research_question.question_id}_v{idx + 1}"
        stable_question_id = research_question.stable_question_id or stable_question_identity(
            dataset_id=static_understanding.dataset_id,
            family_id=research_question.family,
            intended_facet_id=research_question.intended_facet_id or "unknown",
            question_text=research_question.question,
        )

        query_id = f"qs_{research_question.family}_{uuid4().hex[:8]}_v{idx + 1}"

        spec = QuerySpec(
            query_id=query_id,
            family=research_question.family,
            research_question=research_question.question,
            claim_type=claim_type_norm,
            target_columns=target_columns,
            subgroup_columns=subgroup_columns,
            feature_columns=feature_columns,
            expected_output_shape=expected_output_shape,
            sql=sql,
            status="draft",
            reason_codes=codes,
            variant_semantic_role=role,
            repair_history=[],
            question_id=research_question.question_id,
            family_id=research_question.family_id or research_question.family,
            intended_facet_id=research_question.intended_facet_id or "unknown",
            variant_id=variant_id,
            diversity_intent_tag=_diversity_intent_tag(role),
            intended_structure_claim=_structure_claim(research_question.family, role),
            source_columns=source_columns,
            expected_result_schema=_expected_result_schema(expected_output_shape, role),
            stable_question_id=stable_question_id,
            secondary_family_candidates=_infer_secondary_family_candidates(research_question.family, role),
            contamination_risk_hints=_infer_contamination_hints(
                family=research_question.family,
                role=role,
                subgroup_columns=subgroup_columns,
                feature_columns=feature_columns,
            ),
            comparator_type=research_question.comparator_type,
            output_semantics=_output_semantics(role),
            sql_origin_mode=str(origin.get("origin_mode") or "de_novo"),
            exemplar_sql_item_id=str(origin.get("exemplar_sql_item_id") or ""),
            exemplar_own_id=str(origin.get("exemplar_own_id") or ""),
            exemplar_source_url=str(origin.get("exemplar_source_url") or ""),
            exemplar_match_score=float(origin.get("exemplar_match_score") or 0.0),
            exemplar_transform_notes=[str(v) for v in (origin.get("exemplar_transform_notes") or []) if isinstance(v, str)],
        )
        _refresh_query_spec_contract(query_spec=spec, static_understanding=static_understanding)
        spec.reason_codes = list(dict.fromkeys(spec.reason_codes + ["QSV2_CONTRACT_ENRICHED"]))
        if spec.sql_origin_mode == "direct_reuse":
            spec.reason_codes = list(dict.fromkeys(spec.reason_codes + ["SQL_ORIGIN_DIRECT_REUSE"]))
        elif spec.sql_origin_mode == "template_adapt":
            spec.reason_codes = list(dict.fromkeys(spec.reason_codes + ["SQL_ORIGIN_TEMPLATE_ADAPT"]))
        else:
            spec.reason_codes = list(dict.fromkeys(spec.reason_codes + ["SQL_ORIGIN_DE_NOVO"]))
        query_specs.append(spec)

    return query_specs


def realize_query_spec(
    *,
    llm_runtime: BenchmarkLLMRuntime,
    static_understanding: StaticDatasetUnderstanding,
    research_question: ResearchQuestion,
    table_name: str,
) -> QuerySpec:
    variants = realize_query_spec_variants(
        llm_runtime=llm_runtime,
        static_understanding=static_understanding,
        research_question=research_question,
        table_name=table_name,
        num_variants=1,
    )
    return variants[0]


def repair_sql_level(
    *,
    llm_runtime: BenchmarkLLMRuntime,
    static_understanding: StaticDatasetUnderstanding,
    query_spec: QuerySpec,
    table_name: str,
    failure_reason_codes: list[str],
    execution_error: str,
) -> QuerySpec:
    updated = deepcopy(query_spec)

    system_prompt = """
You are repairing a SQL query for benchmark construction.
Return strict JSON with keys: sql, reason_codes.
Rules:
- Keep single-table SELECT only.
- Address the provided failure reason.
- Preserve the research question intent.
""".strip()

    user_prompt = (
        f"table_name={table_name}\n"
        f"research_question={query_spec.research_question}\n"
        f"family={query_spec.family}\n"
        f"family_contract={_family_contract(query_spec.family)}\n"
        f"semantic_role={query_spec.variant_semantic_role}\n"
        f"current_sql={query_spec.sql}\n"
        f"failure_reason_codes={failure_reason_codes}\n"
        f"execution_error={execution_error}\n"
        f"target_column={static_understanding.target_column}\n"
        f"ordered_fields={static_understanding.ordered_fields}\n"
    )

    payload = llm_runtime.invoke_json(
        phase="repair",
        module="realization.repair_sql_level",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        question_for_usage=query_spec.research_question,
    )

    repaired_sql = str(payload.get("sql") or "").strip()
    repair_codes = [str(v) for v in (payload.get("reason_codes") or []) if isinstance(v, str)]
    if repaired_sql:
        normalized = _normalize_sql(repaired_sql)
        if _role_sql_expectation_ok(query_spec.variant_semantic_role or "count_distribution", normalized, static_understanding.target_column):
            updated.sql = normalized
        else:
            repair_codes.append("REPAIR_SQL_ROLE_MISMATCH")
            updated.sql = _normalize_sql(
                _role_sql_template(
                    role=(query_spec.variant_semantic_role or "count_distribution"),
                    family=query_spec.family,
                    table_name=table_name,
                    target_column=static_understanding.target_column,
                    target_labels=static_understanding.target_labels,
                    subgroup_columns=query_spec.subgroup_columns,
                    feature_columns=query_spec.feature_columns,
                )
            )
            repair_codes.append("REPAIR_SQL_TEMPLATE_FALLBACK")
    else:
        updated.sql = _normalize_sql(
            _role_sql_template(
                role=(query_spec.variant_semantic_role or "count_distribution"),
                family=query_spec.family,
                table_name=table_name,
                target_column=static_understanding.target_column,
                target_labels=static_understanding.target_labels,
                subgroup_columns=query_spec.subgroup_columns,
                feature_columns=query_spec.feature_columns,
            )
        )
        repair_codes.append("REPAIR_SQL_TEMPLATE_FALLBACK")

    updated.repair_history.append(
        {
            "level": "sql",
            "failure_reason_codes": failure_reason_codes,
            "execution_error": execution_error,
            "repair_reason_codes": repair_codes,
            "sql_after_repair": updated.sql,
        }
    )
    updated.reason_codes = list(dict.fromkeys(updated.reason_codes + repair_codes + ["REPAIR_SQL_LEVEL"]))
    _refresh_query_spec_contract(query_spec=updated, static_understanding=static_understanding)
    updated.reason_codes = list(dict.fromkeys(updated.reason_codes + ["QSV2_CONTRACT_REFRESHED"]))
    updated.status = "repaired_sql"
    return updated


def repair_queryspec_level(query_spec: QuerySpec, failure_reason_codes: list[str]) -> QuerySpec:
    updated = deepcopy(query_spec)
    if len(updated.subgroup_columns) > 1:
        updated.subgroup_columns = updated.subgroup_columns[: max(1, len(updated.subgroup_columns) - 1)]
        updated.reason_codes.append("REPAIR_QUERYSPEC_REDUCE_SUBGROUP_DEPTH")
    elif len(updated.feature_columns) > 1:
        updated.feature_columns = updated.feature_columns[: max(1, len(updated.feature_columns) - 1)]
        updated.reason_codes.append("REPAIR_QUERYSPEC_REDUCE_FEATURE_DEPTH")
    else:
        updated.reason_codes.append("REPAIR_QUERYSPEC_NO_STRUCTURAL_CHANGE")

    updated.repair_history.append(
        {
            "level": "queryspec",
            "failure_reason_codes": failure_reason_codes,
            "queryspec_after_repair": {
                "subgroup_columns": updated.subgroup_columns,
                "feature_columns": updated.feature_columns,
            },
        }
    )
    updated.source_columns = list(
        dict.fromkeys(updated.target_columns + updated.subgroup_columns + updated.feature_columns)
    )
    updated.contamination_risk_hints = _infer_contamination_hints(
        family=updated.family,
        role=updated.variant_semantic_role or "unknown",
        subgroup_columns=updated.subgroup_columns,
        feature_columns=updated.feature_columns,
    )
    updated.status = "repaired_queryspec"
    updated.reason_codes = list(dict.fromkeys(updated.reason_codes + ["REPAIR_QUERYSPEC_LEVEL"]))
    return updated


def regenerate_research_question_level(
    *,
    llm_runtime: BenchmarkLLMRuntime,
    research_question: ResearchQuestion,
    static_understanding: StaticDatasetUnderstanding,
) -> ResearchQuestion:
    system_prompt = """
Rewrite the research question to keep the same family but improve answerability and non-triviality.
Return strict JSON with keys: question, related_fields, intent, reason_codes.
""".strip()

    user_prompt = (
        f"family={research_question.family}\n"
        f"old_question={research_question.question}\n"
        f"related_fields={research_question.related_fields}\n"
        f"target_column={static_understanding.target_column}\n"
        f"low_support_guard=avoid overly fragmented predicates\n"
    )

    payload = llm_runtime.invoke_json(
        phase="repair",
        module="realization.regenerate_research_question_level",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        question_for_usage=research_question.question,
    )

    question = str(payload.get("question") or research_question.question).strip()
    related_fields = [str(v) for v in (payload.get("related_fields") or research_question.related_fields) if isinstance(v, str)]
    intent = str(payload.get("intent") or research_question.intent)
    reason_codes = [str(v) for v in (payload.get("reason_codes") or []) if isinstance(v, str)]
    stable_id = stable_question_identity(
        dataset_id=static_understanding.dataset_id,
        family_id=research_question.family_id or research_question.family,
        intended_facet_id=research_question.intended_facet_id or "unknown",
        question_text=question,
    )

    return ResearchQuestion(
        question_id=f"rq_{research_question.family}_{uuid4().hex[:8]}",
        family=research_question.family,
        question=question,
        related_fields=related_fields or research_question.related_fields,
        target=research_question.target,
        intent=intent,
        reason_codes=list(dict.fromkeys(research_question.reason_codes + reason_codes + ["REPAIR_RESEARCH_QUESTION_LEVEL"])),
        family_id=research_question.family_id or research_question.family,
        intended_facet_id=research_question.intended_facet_id or "unknown",
        question_text=question,
        target_columns=research_question.target_columns or ([research_question.target] if research_question.target else []),
        related_columns=related_fields or research_question.related_fields,
        rationale=research_question.rationale or "Repaired research question to improve answerability.",
        evidence_expectation=research_question.evidence_expectation or "Grouped support/target summary.",
        comparator_type=research_question.comparator_type,
        risk_tags=list(research_question.risk_tags),
        uncertainty_tags=list(research_question.uncertainty_tags),
        stable_question_id=stable_id,
    )
