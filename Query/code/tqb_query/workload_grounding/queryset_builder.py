"""Build and compare template-grounded dataset query sets."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqb_query.benchmark.canonical_sql import canonical_sql_hash, canonicalize_sql, stable_hash, stable_question_identity, stable_query_identity
from tqb_query.benchmark.models import CandidateRecord, QuerySpec, QuestionBundleRecord, ResearchQuestion, ValidationCategoryResult
from tqb_query.benchmark.sql_exec import execute_sql
from tqb_query.benchmark.understanding import build_static_understanding
from tqb_query.benchmark.validation import build_query_execution_summary_v2, run_basic_validation, run_bundle_similarity_validation
from tqb_query.config.settings import DATA_DIR, RUNS_DIR
from tqb_query.data.bundle import DatasetBundle, load_dataset_bundle
from tqb_query.db.csv_sqlite import materialize_dataset_to_sqlite
from tqb_query.logging.run_artifacts import RunArtifactWriter


HARD_REJECT_CODES = {
    "VAL_STATIC_SQL_EMPTY",
    "VAL_STATIC_DML_BLOCKED",
    "VAL_STATIC_SQLITE_INCOMPATIBLE",
    "VAL_STATIC_TARGET_MISSING",
    "VAL_STATIC_RAW_EXTRACTION",
    "VAL_STATIC_FAMILY_NEEDS_AGG",
    "VAL_EXEC_SQL_ERROR",
    "VAL_EXEC_EMPTY_RESULT",
    "VAL_SANITY_QUESTION_MISMATCH",
    "VAL_SANITY_RAW_EXTRACTION",
    "VAL_DEGENERATE_RATE",
    "VAL_SANITY_MISSINGNESS_NOT_OPERATIONALIZED",
}

POSITIVE_VALUE_HINTS = {"yes", "true", "1", "good", "vgood", "acc", "positive"}
NEGATIVE_VALUE_HINTS = {"no", "false", "0", "unacc", "negative"}

GROUP_COUNT_IDS = {
    "tpl_clickbench_group_count",
    "tpl_clickbench_filtered_topk_group_count",
    "tpl_clickbench_two_dimensional_topk_count",
}
NUMERIC_AGG_IDS = {
    "tpl_h2o_group_sum",
    "tpl_m4_group_avg_numeric",
    "tpl_m4_support_guarded_group_avg",
    "tpl_m4_two_dimensional_group_avg",
    "tpl_m4_binned_numeric_group_avg",
    "tpl_m4_median_filtered_numeric",
}
RATE_IDS = {"tpl_m4_group_condition_rate", "tpl_m4_group_ratio_two_conditions"}
RANK_IDS = {"tpl_h2o_topn_within_group"}


@dataclass
class FieldStats:
    name: str
    declared_type: str
    semantic_type: str
    field_role: str
    field_tags: list[str]
    use_for_groupby: bool
    use_for_predicate: bool
    use_as_target: bool
    distinct_count: int
    top_values: list[tuple[Any, int]]
    is_numeric: bool
    is_categorical: bool
    min_value: float | None = None
    max_value: float | None = None
    q33: float | None = None
    q50: float | None = None
    q66: float | None = None
    q75: float | None = None


@dataclass
class TemplatePlanItem:
    template_id: str
    question: str
    rationale: str
    bindings: dict[str, Any]
    quality_notes: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _is_numeric(field: dict[str, Any]) -> bool:
    if bool(field.get("numeric_like")) or bool(field.get("integer_like")):
        return True
    tokens = (
        f"{field.get('declared_type', '')} {field.get('storage_type', '')} "
        f"{field.get('semantic_type', '')}"
    ).lower()
    return "numeric" in tokens or "integer" in tokens or "float" in tokens


def _is_categorical(field: dict[str, Any]) -> bool:
    tokens = f"{field.get('declared_type', '')} {field.get('semantic_type', '')}".lower()
    return "categorical" in tokens or "boolean" in tokens or "ordinal" in tokens or "nominal" in tokens


def _quantile(values: list[float], frac: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = frac * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _build_field_stats_from_contract(bundle: DatasetBundle) -> dict[str, FieldStats]:
    registry_fields = bundle.field_registry.get("fields") or []
    contract_columns = {
        str(column.get("name") or "").strip(): column
        for column in (bundle.dataset_contract.get("columns") or [])
        if str(column.get("name") or "").strip()
    }
    stats: dict[str, FieldStats] = {}
    for field in registry_fields:
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        contract_column = contract_columns.get(name, {})
        profile_stats = contract_column.get("profile_stats") or {}
        distinct_count = int(profile_stats.get("unique_count") or field.get("unique_count") or 0)
        example_values = [
            value
            for value in (
                profile_stats.get("example_values")
                or field.get("examples")
                or field.get("domain_values")
                or []
            )
            if value is not None
        ]
        numeric_values: list[float] = []
        is_numeric = _is_numeric(field)
        if is_numeric:
            for value in example_values:
                try:
                    numeric_values.append(float(value))
                except (TypeError, ValueError):
                    continue

        numeric_min = profile_stats.get("min")
        numeric_max = profile_stats.get("max")
        if numeric_min is None:
            numeric_min = field.get("numeric_min")
        if numeric_max is None:
            numeric_max = field.get("numeric_max")
        if numeric_values:
            q33 = _quantile(numeric_values, 0.33)
            q50 = _quantile(numeric_values, 0.50)
            q66 = _quantile(numeric_values, 0.66)
            q75 = _quantile(numeric_values, 0.75)
        elif numeric_min is not None and numeric_max is not None:
            low, high = float(numeric_min), float(numeric_max)
            q33 = low + (high - low) * 0.33
            q50 = low + (high - low) * 0.50
            q66 = low + (high - low) * 0.66
            q75 = low + (high - low) * 0.75
        else:
            q33 = q50 = q66 = q75 = None

        stats[name] = FieldStats(
            name=name,
            declared_type=str(field.get("declared_type") or field.get("storage_type") or ""),
            semantic_type=str(field.get("semantic_type") or ""),
            field_role=str(field.get("role") or ""),
            field_tags=list(field.get("field_tags") or []),
            use_for_groupby=bool(field.get("use_for_groupby")),
            use_for_predicate=bool(field.get("use_for_predicate")),
            use_as_target=bool(field.get("use_as_target")),
            distinct_count=distinct_count,
            top_values=[(value, 0) for value in example_values[:8]],
            is_numeric=is_numeric,
            is_categorical=_is_categorical(field),
            min_value=float(numeric_min) if numeric_min is not None else (min(numeric_values) if numeric_values else None),
            max_value=float(numeric_max) if numeric_max is not None else (max(numeric_values) if numeric_values else None),
            q33=q33,
            q50=q50,
            q66=q66,
            q75=q75,
        )
    return stats


def build_field_stats(bundle: DatasetBundle, table_name: str, db_path: Path) -> dict[str, FieldStats]:
    registry_fields = bundle.field_registry.get("fields") or []
    row_count = int(
        (bundle.dataset_contract.get("row_counts") or {}).get("main")
        or (bundle.dataset_profile.get("summary") or {}).get("n_rows")
        or 0
    )
    if row_count <= 0:
        conn = sqlite3.connect(db_path)
        try:
            row_count = int(conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0] or 0)
        finally:
            conn.close()
    if len(registry_fields) > 64 or row_count >= 100_000:
        return _build_field_stats_from_contract(bundle)

    conn = sqlite3.connect(db_path)
    stats: dict[str, FieldStats] = {}
    try:
        for field in registry_fields:
            name = str(field.get("name") or "").strip()
            if not name:
                continue
            distinct_count = int(conn.execute(f'SELECT COUNT(DISTINCT "{name}") FROM "{table_name}"').fetchone()[0] or 0)
            top_values = conn.execute(
                f'SELECT "{name}", COUNT(*) AS c FROM "{table_name}" '
                f'GROUP BY "{name}" ORDER BY c DESC, "{name}" LIMIT 8'
            ).fetchall()
            numeric_values: list[float] = []
            is_numeric = _is_numeric(field)
            if is_numeric:
                raw_values = conn.execute(
                    f'SELECT "{name}" FROM "{table_name}" WHERE "{name}" IS NOT NULL ORDER BY "{name}"'
                ).fetchall()
                numeric_values = []
                for row in raw_values:
                    try:
                        numeric_values.append(float(row[0]))
                    except (TypeError, ValueError):
                        continue

            stats[name] = FieldStats(
                name=name,
                declared_type=str(field.get("declared_type") or field.get("storage_type") or ""),
                semantic_type=str(field.get("semantic_type") or ""),
                field_role=str(field.get("role") or ""),
                field_tags=list(field.get("field_tags") or []),
                use_for_groupby=bool(field.get("use_for_groupby")),
                use_for_predicate=bool(field.get("use_for_predicate")),
                use_as_target=bool(field.get("use_as_target")),
                distinct_count=distinct_count,
                top_values=[(row[0], int(row[1])) for row in top_values],
                is_numeric=is_numeric,
                is_categorical=_is_categorical(field),
                min_value=min(numeric_values) if numeric_values else None,
                max_value=max(numeric_values) if numeric_values else None,
                q33=_quantile(numeric_values, 0.33),
                q50=_quantile(numeric_values, 0.50),
                q66=_quantile(numeric_values, 0.66),
                q75=_quantile(numeric_values, 0.75),
            )
    finally:
        conn.close()
    return stats


def _first_categorical_value(stats: FieldStats, *, positive: bool | None = None) -> Any:
    if not stats.top_values:
        return None
    values = [value for value, _count in stats.top_values if value is not None]
    if not values:
        return None
    lowered = {str(value).lower(): value for value in values}
    if positive is True:
        for key in POSITIVE_VALUE_HINTS:
            if key in lowered:
                return lowered[key]
    if positive is False:
        for key in NEGATIVE_VALUE_HINTS:
            if key in lowered:
                return lowered[key]
    return values[0]


def _binary_pair(stats: FieldStats) -> tuple[Any, Any] | None:
    values = [value for value, _count in stats.top_values if value is not None]
    if len(values) < 2:
        return None
    positive = _first_categorical_value(stats, positive=True)
    negative = _first_categorical_value(stats, positive=False)
    if positive is not None and negative is not None and positive != negative:
        return positive, negative
    return values[0], values[1]


def _latest_baseline_run(dataset_id: str, runs_root: Path) -> str | None:
    candidates: list[str] = []
    for path in sorted(runs_root.iterdir()):
        if not path.is_dir():
            continue
        run_id = path.name
        if not run_id.startswith(f"{dataset_id}_"):
            continue
        if "_tg" in run_id:
            continue
        if not (path / "benchmark_package" / "queryspecs.json").exists():
            continue
        candidates.append(run_id)
    return candidates[-1] if candidates else None


def _load_template_library(core_library_path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(core_library_path)
    return {row["template_id"]: row for row in rows}


def _m4_plan(template_library: dict[str, dict[str, Any]], field_stats: dict[str, FieldStats], min_group_size: int) -> list[TemplatePlanItem]:
    smoker_pair = _binary_pair(field_stats["smoker"])
    if smoker_pair is None:
        raise ValueError("m4 expected binary smoker field for template grounding.")
    smoker_yes, smoker_no = smoker_pair
    age_q33 = int(round(field_stats["age"].q33 or 30))
    age_q66 = int(round(field_stats["age"].q66 or 50))

    plan = [
        TemplatePlanItem(
            template_id="tpl_clickbench_group_count",
            question="How is the insured population distributed across regions?",
            rationale="Baseline subgroup mass by region is one of the most production-like first-pass dashboard queries.",
            bindings={"group_col": "region"},
            quality_notes=["dashboard_like", "low_cardinality_group"],
        ),
        TemplatePlanItem(
            template_id="tpl_clickbench_filtered_topk_group_count",
            question="Among smokers, which regions contribute the most records?",
            rationale="Adds a realistic slice-and-rank pattern common in operational analytics.",
            bindings={
                "group_col": "region",
                "predicate_col": "smoker",
                "predicate_op": "=",
                "predicate_value": smoker_yes,
                "top_k": 4,
            },
            quality_notes=["filtered_slice", "topk"],
        ),
        TemplatePlanItem(
            template_id="tpl_m4_group_condition_rate",
            question="Which child-count groups have the highest smoker rate?",
            rationale="Condition-rate queries are common KPI-style workload units and map naturally to this dataset.",
            bindings={
                "group_col": "children",
                "condition_col": "smoker",
                "condition_value": smoker_yes,
            },
            quality_notes=["kpi_rate", "binary_condition"],
        ),
        TemplatePlanItem(
            template_id="tpl_m4_group_ratio_two_conditions",
            question="How does the smoker-to-non-smoker ratio vary across regions?",
            rationale="A direct ratio query gives a compact workload view that is still easy to interpret.",
            bindings={
                "group_col": "region",
                "condition_col": "smoker",
                "positive_value": smoker_yes,
                "negative_value": smoker_no,
            },
            quality_notes=["ratio_view", "binary_condition"],
        ),
        TemplatePlanItem(
            template_id="tpl_h2o_group_sum",
            question="Which regions contribute the highest total charges?",
            rationale="Grouped sum on the target measure is a universal production workload pattern.",
            bindings={"group_col": "region", "measure_col": "charges"},
            quality_notes=["numeric_aggregate", "business_kpi"],
        ),
        TemplatePlanItem(
            template_id="tpl_m4_support_guarded_group_avg",
            question="How do average charges vary across child-count groups once low-support groups are filtered out?",
            rationale="Support-guarded averages are much closer to production BI than raw low-support comparisons.",
            bindings={"group_col": "children", "measure_col": "charges", "min_group_size": max(min_group_size, 40)},
            quality_notes=["support_guard", "numeric_aggregate"],
        ),
        TemplatePlanItem(
            template_id="tpl_h2o_topn_within_group",
            question="What are the top three charges observed within each region?",
            rationale="Per-group top-N is a realistic drill-down query that complements aggregate dashboards.",
            bindings={"group_col": "region", "measure_col": "charges", "top_n": 3},
            quality_notes=["window_rank", "drilldown"],
        ),
        TemplatePlanItem(
            template_id="tpl_m4_group_avg_numeric",
            question="How do average charges differ between smokers and non-smokers?",
            rationale="This is a compact, highly reusable subgroup mean comparison.",
            bindings={"group_col": "smoker", "measure_col": "charges"},
            quality_notes=["numeric_aggregate", "classic_slice"],
        ),
        TemplatePlanItem(
            template_id="tpl_m4_two_dimensional_group_avg",
            question="How do average charges vary across sex and smoking-status combinations?",
            rationale="Two-dimensional subgroup averages capture the core interaction structure without overfitting the schema.",
            bindings={"group_col": "sex", "group_col_2": "smoker", "measure_col": "charges"},
            quality_notes=["two_axis_slice", "interaction_view"],
        ),
        TemplatePlanItem(
            template_id="tpl_m4_median_filtered_numeric",
            question="What is the median charge among smokers?",
            rationale="A filtered robust-summary query adds tail-aware coverage without becoming exotic.",
            bindings={
                "measure_col": "charges",
                "predicate_col": "smoker",
                "predicate_op": "=",
                "predicate_value": smoker_yes,
            },
            quality_notes=["robust_summary", "filtered_slice"],
        ),
        TemplatePlanItem(
            template_id="tpl_m4_binned_numeric_group_avg",
            question="How do average charges vary across age bands?",
            rationale="Bucketed numeric analysis is a common dashboard shape and avoids grouping on raw high-cardinality values.",
            bindings={
                "band_col": "age",
                "band_cut_1": age_q33,
                "band_cut_2": age_q66,
                "measure_col": "charges",
            },
            quality_notes=["bucketed_numeric", "dashboard_like"],
        ),
        TemplatePlanItem(
            template_id="tpl_clickbench_two_dimensional_topk_count",
            question="Which region-by-smoking-status combinations are most common?",
            rationale="A heavy-hitter joint distribution query is a realistic workload building block.",
            bindings={"group_col": "region", "group_col_2": "smoker", "top_k": 8},
            quality_notes=["two_axis_slice", "heavy_hitter"],
        ),
    ]
    return [item for item in plan if item.template_id in template_library]


def dataset_plan(
    *,
    dataset_id: str,
    template_library: dict[str, dict[str, Any]],
    field_stats: dict[str, FieldStats],
    min_group_size: int,
) -> list[TemplatePlanItem]:
    if dataset_id == "m4":
        return _m4_plan(template_library, field_stats, min_group_size)
    raise ValueError(f"No curated template-grounded queryset plan is defined yet for dataset_id={dataset_id}.")


def _render_sql(sql_skeleton: str, bindings: dict[str, Any], table_name: str) -> str:
    payload = {"table": table_name}
    payload.update(bindings)
    rendered = sql_skeleton
    for key, value in payload.items():
        token = "{" + key + "}"
        replacement = _sql_literal(value) if key.endswith("_value") else str(value)
        rendered = rendered.replace(token, replacement)
    if "{" in rendered or "}" in rendered:
        missing = sorted(set(re.findall(r"\{([^{}]+)\}", rendered)))
        raise ValueError(f"Unresolved template placeholders: {missing}")
    return rendered.strip()


def _query_target_columns(template_id: str, target_column: str, bindings: dict[str, Any]) -> list[str]:
    if "measure_col" in bindings:
        return [str(bindings["measure_col"])]
    if "target_col" in bindings:
        return [str(bindings["target_col"])]
    if template_id in RATE_IDS:
        return [str(bindings.get("condition_col") or target_column)]
    return [target_column] if target_column else []


def _expected_output_shape(template_id: str) -> str:
    if template_id in GROUP_COUNT_IDS:
        return "grouped_support_table"
    if template_id in RATE_IDS:
        return "grouped_rate_table"
    if template_id in RANK_IDS:
        return "ranked_table"
    return "grouped_summary_table"


def _variant_role(template_id: str) -> str:
    mapping = {
        "tpl_clickbench_group_count": "group_count",
        "tpl_clickbench_filtered_topk_group_count": "filtered_group_count_topk",
        "tpl_clickbench_two_dimensional_topk_count": "two_dimensional_count_topk",
        "tpl_m4_group_condition_rate": "group_condition_rate",
        "tpl_m4_group_ratio_two_conditions": "group_condition_ratio",
        "tpl_h2o_group_sum": "group_sum",
        "tpl_m4_support_guarded_group_avg": "support_guarded_group_avg",
        "tpl_h2o_topn_within_group": "topn_within_group",
        "tpl_m4_group_avg_numeric": "group_avg_numeric",
        "tpl_m4_two_dimensional_group_avg": "two_dimensional_group_avg",
        "tpl_m4_median_filtered_numeric": "filtered_median_numeric",
        "tpl_m4_binned_numeric_group_avg": "binned_numeric_group_avg",
    }
    return mapping.get(template_id, template_id.replace("tpl_", ""))


def _claim_type(template_id: str) -> str:
    if template_id in RATE_IDS:
        return "higher_lower_comparison"
    if template_id == "tpl_m4_median_filtered_numeric":
        return "higher_lower_comparison"
    return "higher_lower_comparison"


def _extract_source_columns(bindings: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    for key in ["group_col", "group_col_2", "band_col", "measure_col", "predicate_col", "condition_col", "target_col"]:
        value = bindings.get(key)
        if not value or not isinstance(value, str):
            continue
        if value not in ordered:
            ordered.append(value)
    return ordered


def _feature_columns(bindings: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    for key in ["band_col", "predicate_col", "condition_col"]:
        value = bindings.get(key)
        if not value or not isinstance(value, str):
            continue
        if value not in ordered:
            ordered.append(value)
    return ordered


def _subgroup_columns(bindings: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    for key in ["group_col", "group_col_2"]:
        value = bindings.get(key)
        if not value or not isinstance(value, str):
            continue
        if value not in ordered:
            ordered.append(value)
    return ordered


def _binding_quality_flags(bindings: dict[str, Any], field_stats: dict[str, FieldStats]) -> list[str]:
    flags: list[str] = []
    for key in ["group_col", "group_col_2"]:
        value = bindings.get(key)
        if not isinstance(value, str):
            continue
        stats = field_stats.get(value)
        if not stats:
            continue
        if stats.is_numeric and stats.distinct_count > 20:
            flags.append(f"{key}:raw_high_card_numeric_grouping")
    if "entity_col" in bindings:
        value = bindings["entity_col"]
        stats = field_stats.get(value) if isinstance(value, str) else None
        if stats and stats.is_numeric:
            flags.append("entity_col:numeric_fallback")
    return flags


def _template_source_distribution(selected_rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for row in selected_rows:
        source = row.get("template_provenance", {}).get("source_query_id") or "unknown"
        prefix = str(source).split("_")[0]
        counter[prefix] += 1
    return dict(counter)


def _pattern_flags(sql: str) -> dict[str, bool]:
    normalized = canonicalize_sql(sql)
    return {
        "grouped_count": "group by" in normalized and "count(" in normalized,
        "filtered_slice": " where " in f" {normalized} " or " having " in f" {normalized} ",
        "numeric_agg": "avg(" in normalized or "sum(" in normalized or "median_measure" in normalized,
        "condition_rate_or_ratio": "case when" in normalized or "_rate" in normalized or "_ratio" in normalized,
        "topk_or_rank": " limit " in f" {normalized} " or "row_number()" in normalized,
        "two_dimensional_grouping": len(_extract_groupby_columns_from_sql(sql)) >= 2,
        "bucketed_numeric": "case when" in normalized and "band_bucket" in normalized,
        "support_guard": "having count(*) >" in normalized,
    }


def _extract_groupby_columns_from_sql(sql: str) -> list[str]:
    normalized = canonicalize_sql(sql)
    match = re.search(r"group by\s+(.+?)(having|order by|limit|$)", normalized)
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _query_has_raw_high_card_grouping(sql: str, field_stats: dict[str, FieldStats]) -> bool:
    if "band_bucket" in canonicalize_sql(sql):
        return False
    for token in _extract_groupby_columns_from_sql(sql):
        if token not in field_stats:
            continue
        stats = field_stats[token]
        if stats.is_numeric and stats.distinct_count > 20 and not stats.use_for_groupby:
            return True
    return False


def _production_like_query(query: dict[str, Any], summary_by_id: dict[str, dict[str, Any]], field_stats: dict[str, FieldStats]) -> bool:
    sql = str(query.get("sql") or "")
    query_id = str(query.get("query_id") or "")
    summary = summary_by_id.get(query_id, {})
    codes = set(summary.get("validation_codes") or [])
    if not summary.get("execution_ok", True):
        return False
    if _query_has_raw_high_card_grouping(sql, field_stats):
        return False
    if "VAL_STATIC_HIGH_DIMENSIONALITY_WARNING" in codes:
        return False
    flags = _pattern_flags(sql)
    if flags["condition_rate_or_ratio"] and "VAL_EXEC_SUPPORT_NOT_OBSERVED_FOR_RATE" in codes:
        return False
    return any(
        [
            flags["grouped_count"],
            flags["numeric_agg"],
            flags["condition_rate_or_ratio"],
            flags["topk_or_rank"],
        ]
    )


def analyze_package(
    *,
    dataset_id: str,
    package_dir: Path,
    field_stats: dict[str, FieldStats],
    source_mode: str,
) -> dict[str, Any]:
    bundles = _load_json(package_dir / "question_bundles.json").get("bundles", [])
    queryspecs = _load_json(package_dir / "queryspecs.json").get("queryspecs", [])
    execution_summaries = _load_json(package_dir / "query_execution_summaries_v2.json").get("summaries", [])

    summary_by_id = {str(row.get("query_id")): row for row in execution_summaries}
    family_counter = Counter(str(item.get("family") or "unknown") for item in bundles)
    warning_counter = Counter()
    for row in execution_summaries:
        for code in row.get("validation_codes") or []:
            warning_counter[str(code)] += 1

    accepted_query_ids = {str(spec.get("query_id") or "") for spec in queryspecs}
    traceable_queries = 0
    if source_mode == "template_grounded":
        catalog_path = package_dir.parent / "template_instance_catalog.json"
        if catalog_path.exists():
            catalog = _load_json(catalog_path).get("instances", [])
            traceable_queries = sum(
                1
                for row in catalog
                if row.get("accepted_local") and str(row.get("query_id") or "") in accepted_query_ids
            )
    else:
        for spec in queryspecs:
            if spec.get("sql_origin_mode") != "de_novo" or spec.get("exemplar_source_url"):
                traceable_queries += 1

    pattern_coverage_counter = Counter()
    raw_high_card_grouping_count = 0
    production_like_count = 0
    for spec in queryspecs:
        flags = _pattern_flags(str(spec.get("sql") or ""))
        for name, enabled in flags.items():
            if enabled:
                pattern_coverage_counter[name] += 1
        if _query_has_raw_high_card_grouping(str(spec.get("sql") or ""), field_stats):
            raw_high_card_grouping_count += 1
        if _production_like_query(spec, summary_by_id, field_stats):
            production_like_count += 1

    query_count = len(queryspecs)
    duplicate_ratio = 0.0
    hashes = [str(spec.get("canonical_sql_hash") or "") for spec in queryspecs if spec.get("canonical_sql_hash")]
    if len(hashes) > 1:
        pair_total = len(hashes) * (len(hashes) - 1) // 2
        dup_pairs = sum(count * (count - 1) // 2 for count in Counter(hashes).values() if count > 1)
        duplicate_ratio = dup_pairs / pair_total if pair_total else 0.0

    return {
        "dataset_id": dataset_id,
        "package_dir": str(package_dir),
        "bundle_count": len(bundles),
        "query_count": query_count,
        "family_distribution": dict(family_counter),
        "traceable_query_rate": round(traceable_queries / max(1, query_count), 6),
        "duplicate_ratio": round(duplicate_ratio, 6),
        "raw_high_card_grouping_rate": round(raw_high_card_grouping_count / max(1, query_count), 6),
        "production_like_query_rate": round(production_like_count / max(1, query_count), 6),
        "pattern_coverage": {key: int(value) for key, value in pattern_coverage_counter.items() if value > 0},
        "validation_warning_counter": dict(warning_counter),
        "top_validation_warnings": warning_counter.most_common(10),
    }


def _comparison_judgement(grounded: dict[str, Any], baseline: dict[str, Any]) -> str:
    grounded_prod = float(grounded["production_like_query_rate"])
    baseline_prod = float(baseline["production_like_query_rate"])
    grounded_trace = float(grounded["traceable_query_rate"])
    baseline_trace = float(baseline["traceable_query_rate"])
    grounded_raw = float(grounded["raw_high_card_grouping_rate"])
    baseline_raw = float(baseline["raw_high_card_grouping_rate"])

    if grounded_prod > baseline_prod and grounded_trace >= baseline_trace and grounded_raw <= baseline_raw:
        return "template_grounded_better_for_virtual_production"
    if grounded_prod >= baseline_prod and grounded_trace > baseline_trace:
        return "template_grounded_modestly_better_for_virtual_production"
    return "mixed_tradeoff"


def _collect_validation_codes(candidate: CandidateRecord) -> list[str]:
    return list(
        dict.fromkeys(
            candidate.validation.static_validation.reason_codes
            + candidate.validation.execution_validation.reason_codes
            + candidate.validation.sanity_validation.reason_codes
        )
    )


def _sidecar_accept(template_id: str, sql: str, candidate: CandidateRecord) -> tuple[bool, list[str]]:
    if not candidate.execution.ok:
        return False, ["execution_failed"]

    codes = set(_collect_validation_codes(candidate))
    waived: set[str] = set()
    normalized = canonicalize_sql(sql)

    if normalized.startswith("with "):
        waived.add("VAL_STATIC_SQL_NOT_SELECT")

    if template_id in {"tpl_m4_group_ratio_two_conditions", "tpl_h2o_topn_within_group"}:
        waived.add("VAL_EXEC_SUPPORT_NOT_OBSERVED_FOR_RATE")

    if template_id == "tpl_m4_median_filtered_numeric":
        waived.update(
            {
                "VAL_STATIC_SQL_NOT_SELECT",
                "VAL_EXEC_SUPPORT_NOT_OBSERVED_FOR_RATE",
                "VAL_EXEC_SINGLE_ROW_WARNING",
                "VAL_SANITY_TRIVIAL",
                "VAL_SANITY_LOW_VARIATION",
            }
        )

    blocking = sorted(code for code in codes if code in HARD_REJECT_CODES and code not in waived)
    if blocking:
        return False, [f"blocking_code:{code}" for code in blocking]

    return True, [f"waived_code:{code}" for code in sorted(waived & codes)]


def build_comparison_report(
    *,
    dataset_id: str,
    grounded_run_id: str,
    grounded_metrics: dict[str, Any],
    baseline_run_id: str,
    baseline_metrics: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    grounded_patterns = set(grounded_metrics.get("pattern_coverage", {}).keys())
    baseline_patterns = set(baseline_metrics.get("pattern_coverage", {}).keys())
    shared_patterns = sorted(grounded_patterns & baseline_patterns)
    grounded_only_patterns = sorted(grounded_patterns - baseline_patterns)
    baseline_only_patterns = sorted(baseline_patterns - grounded_patterns)

    comparison = {
        "dataset_id": dataset_id,
        "grounded_run_id": grounded_run_id,
        "baseline_run_id": baseline_run_id,
        "judgement": _comparison_judgement(grounded_metrics, baseline_metrics),
        "grounded_metrics": grounded_metrics,
        "baseline_metrics": baseline_metrics,
        "pattern_overlap": {
            "shared": shared_patterns,
            "grounded_only": grounded_only_patterns,
            "baseline_only": baseline_only_patterns,
        },
        "delta": {
            "bundle_count": grounded_metrics["bundle_count"] - baseline_metrics["bundle_count"],
            "query_count": grounded_metrics["query_count"] - baseline_metrics["query_count"],
            "traceable_query_rate": round(
                float(grounded_metrics["traceable_query_rate"]) - float(baseline_metrics["traceable_query_rate"]), 6
            ),
            "duplicate_ratio": round(
                float(grounded_metrics["duplicate_ratio"]) - float(baseline_metrics["duplicate_ratio"]), 6
            ),
            "raw_high_card_grouping_rate": round(
                float(grounded_metrics["raw_high_card_grouping_rate"]) - float(baseline_metrics["raw_high_card_grouping_rate"]),
                6,
            ),
            "production_like_query_rate": round(
                float(grounded_metrics["production_like_query_rate"]) - float(baseline_metrics["production_like_query_rate"]),
                6,
            ),
        },
    }

    report = f"""# Template-Grounded Queryset Comparison ({dataset_id})

## Compared runs

- Template-grounded run: `{grounded_run_id}`
- Baseline no-template run: `{baseline_run_id}`

## Headline judgement

`{comparison['judgement']}`

## Metrics

| Metric | Template-grounded | Baseline | Delta |
|---|---:|---:|---:|
| bundle_count | {grounded_metrics['bundle_count']} | {baseline_metrics['bundle_count']} | {comparison['delta']['bundle_count']} |
| query_count | {grounded_metrics['query_count']} | {baseline_metrics['query_count']} | {comparison['delta']['query_count']} |
| traceable_query_rate | {grounded_metrics['traceable_query_rate']:.3f} | {baseline_metrics['traceable_query_rate']:.3f} | {comparison['delta']['traceable_query_rate']:+.3f} |
| duplicate_ratio | {grounded_metrics['duplicate_ratio']:.3f} | {baseline_metrics['duplicate_ratio']:.3f} | {comparison['delta']['duplicate_ratio']:+.3f} |
| raw_high_card_grouping_rate | {grounded_metrics['raw_high_card_grouping_rate']:.3f} | {baseline_metrics['raw_high_card_grouping_rate']:.3f} | {comparison['delta']['raw_high_card_grouping_rate']:+.3f} |
| production_like_query_rate | {grounded_metrics['production_like_query_rate']:.3f} | {baseline_metrics['production_like_query_rate']:.3f} | {comparison['delta']['production_like_query_rate']:+.3f} |

## Pattern coverage

- Shared patterns: {', '.join(shared_patterns) if shared_patterns else 'none'}
- Template-grounded only: {', '.join(grounded_only_patterns) if grounded_only_patterns else 'none'}
- Baseline only: {', '.join(baseline_only_patterns) if baseline_only_patterns else 'none'}

## Reading

- Template-grounded query sets are intentionally smaller and denser. A lower `query_count` is not automatically worse if the retained queries are more traceable and structurally natural.
- `traceable_query_rate` measures whether each query can be tied back to an explicit upstream source. Template-grounded should be close to 1.0 by construction.
- `raw_high_card_grouping_rate` flags queries that group directly on high-cardinality numeric fields without binning. Lower is better for production realism.
- `production_like_query_rate` is a conservative static proxy for virtual production realism: queries must execute, avoid raw high-cardinality grouping, avoid rate queries without observable denominator/support, avoid over-complex groupings, and still match at least one common dashboard/BI pattern.

## Top warnings

- Template-grounded: {grounded_metrics.get('top_validation_warnings') or []}
- Baseline: {baseline_metrics.get('top_validation_warnings') or []}
"""
    return comparison, report


def build_template_grounded_queryset(
    *,
    dataset_id: str,
    core_library_path: Path,
    baseline_run_id: str | None = None,
) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_id=dataset_id, data_root=DATA_DIR, strict=True)
    sqlite_result = materialize_dataset_to_sqlite(bundle=bundle, use_cache=True)
    static_understanding = build_static_understanding(bundle)
    min_group_size = int((static_understanding.policy_summary.get("minimum_support_thresholds") or {}).get("absolute_min_rows", 20))
    field_stats = build_field_stats(bundle, sqlite_result.table_name, sqlite_result.db_path)
    template_library = _load_template_library(core_library_path)
    plan = dataset_plan(
        dataset_id=dataset_id,
        template_library=template_library,
        field_stats=field_stats,
        min_group_size=min_group_size,
    )

    run_id = f"{dataset_id}_tgset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact_writer = RunArtifactWriter(RUNS_DIR, run_id)

    manifest = {
        "run_id": run_id,
        "status": "running",
        "mode": "template_grounded_queryset_v1",
        "dataset_id": dataset_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "template_pool_path": str(core_library_path),
        "sqlite": {
            "db_path": str(sqlite_result.db_path),
            "table_name": sqlite_result.table_name,
            "row_count": sqlite_result.row_count,
            "cache_hit": sqlite_result.cache_hit,
        },
    }
    artifact_writer.write_manifest(manifest)

    instances: list[dict[str, Any]] = []
    bundle_records: list[QuestionBundleRecord] = []
    query_specs: list[QuerySpec] = []
    execution_summaries: list[dict[str, Any]] = []
    bundle_diversity_records: list[dict[str, Any]] = []
    skipped_templates: list[dict[str, Any]] = []

    for item in plan:
        template = template_library.get(item.template_id)
        if template is None:
            skipped_templates.append({"template_id": item.template_id, "reason": "template_missing"})
            continue

        try:
            sql = _render_sql(template["sql_skeleton"], item.bindings, sqlite_result.table_name)
        except Exception as exc:  # noqa: BLE001
            skipped_templates.append({"template_id": item.template_id, "reason": f"render_failed:{exc}"})
            continue

        question_id = f"rq_tg_{stable_hash(f'{dataset_id}|{item.template_id}|{item.question}', 10)}"
        stable_question_id = stable_question_identity(
            dataset_id=dataset_id,
            family_id=str(template.get("primary_family") or "unknown"),
            intended_facet_id=item.template_id,
            question_text=item.question,
        )
        target_columns = _query_target_columns(item.template_id, static_understanding.target_column, item.bindings)
        subgroup_columns = _subgroup_columns(item.bindings)
        feature_columns = _feature_columns(item.bindings)
        canonical_sql = canonicalize_sql(sql)
        variant_role = _variant_role(item.template_id)

        rq = ResearchQuestion(
            question_id=question_id,
            family=str(template.get("primary_family") or "unknown"),
            question=item.question,
            related_fields=_extract_source_columns(item.bindings),
            target=target_columns[0] if target_columns else static_understanding.target_column,
            intent=str(template.get("intent") or "template_grounded"),
            reason_codes=[
                "TEMPLATE_GROUNDED_QUERYSET",
                f"TEMPLATE_ID_{item.template_id}",
                f"SOURCE_WORKLOAD_{template.get('source_workload_id', 'unknown')}",
            ],
            family_id=str(template.get("primary_family") or "unknown"),
            intended_facet_id=item.template_id,
            rationale=item.rationale,
            evidence_expectation="Single-table grouped analytical result.",
            comparator_type="workload_grounded_template",
            stable_question_id=stable_question_id,
        )

        spec = QuerySpec(
            query_id=f"qs_tg_{stable_hash(f'{dataset_id}|{item.template_id}|{sql}', 10)}",
            family=str(template.get("primary_family") or "unknown"),
            research_question=item.question,
            claim_type=_claim_type(item.template_id),
            target_columns=target_columns,
            subgroup_columns=subgroup_columns,
            feature_columns=feature_columns,
            expected_output_shape=_expected_output_shape(item.template_id),
            sql=sql,
            status="draft",
            reason_codes=[
                "TEMPLATE_GROUNDED_QUERYSET",
                f"TEMPLATE_ID_{item.template_id}",
                f"TEMPLATE_STATUS_{template.get('status', 'unknown')}",
            ],
            variant_semantic_role=variant_role,
            question_id=question_id,
            family_id=str(template.get("primary_family") or "unknown"),
            intended_facet_id=item.template_id,
            variant_id=f"{item.template_id}_v1",
            diversity_intent_tag="template_grounded_single_variant",
            intended_structure_claim=f"{template.get('primary_family')}:{item.template_id}",
            source_columns=_extract_source_columns(item.bindings),
            expected_result_schema=_expected_output_shape(item.template_id),
            canonical_sql=canonical_sql,
            canonical_sql_hash=canonical_sql_hash(sql),
            stable_query_id=stable_query_identity(
                dataset_id=dataset_id,
                family_id=str(template.get("primary_family") or "unknown"),
                intended_facet_id=item.template_id,
                stable_question_id=stable_question_id,
                variant_semantic_role=variant_role,
                canonical_sql=canonical_sql,
            ),
            stable_question_id=stable_question_id,
            secondary_family_candidates=[template["secondary_family"]] if template.get("secondary_family") else [],
            contamination_risk_hints=_binding_quality_flags(item.bindings, field_stats),
            comparator_type="workload_grounded_template",
            output_semantics=_expected_output_shape(item.template_id),
            aggregate_type="template_grounded",
            measure_column=str(item.bindings.get("measure_col") or (target_columns[0] if target_columns else "unknown")),
            base_filters=[],
            optional_filters=[],
            groupby_columns=subgroup_columns,
            comparison_target=str(item.bindings.get("condition_col") or item.bindings.get("predicate_col") or "unknown"),
            direction="higher",
            sql_origin_mode="template_grounded",
            exemplar_sql_item_id=item.template_id,
            exemplar_own_id=str(template.get("source_workload_id") or "unknown"),
            exemplar_source_url=str((template.get("provenance") or {}).get("url") or ""),
            exemplar_match_score=1.0,
            exemplar_transform_notes=["template_grounded_materialization"],
        )

        execution = execute_sql(sqlite_result.db_path, sql)
        validation = run_basic_validation(
            llm_runtime=None,  # Reserved but currently unused by the sanity validator.
            static_understanding=static_understanding,
            query_spec=spec,
            execution_result=execution,
            table_name=sqlite_result.table_name,
        )
        candidate = CandidateRecord(
            query_spec=spec,
            validation=validation,
            execution=execution,
            accepted_local=False,
            rejected_reason_codes=[],
            provenance={
                "template_id": item.template_id,
                "template_name": template.get("template_name"),
                "template_provenance": template.get("provenance") or {},
                "bindings": item.bindings,
                "quality_notes": item.quality_notes,
                "binding_quality_flags": _binding_quality_flags(item.bindings, field_stats),
            },
        )
        accepted_local, acceptance_notes = _sidecar_accept(item.template_id, sql, candidate)
        candidate.accepted_local = accepted_local
        if accepted_local:
            spec.status = "accepted_local"
            spec.reason_codes.append("ACCEPT_LOCAL_VALIDATION_PASS")
            if acceptance_notes:
                spec.reason_codes.extend(f"SIDECAR_ACCEPT_{note}" for note in acceptance_notes)
        else:
            spec.status = "rejected_local"
            spec.reason_codes.append("REJECT_LOCAL_VALIDATION_FAIL")
            candidate.rejected_reason_codes = _collect_validation_codes(candidate)

        bundle_validation, bundle_quality = run_bundle_similarity_validation(variants=[candidate], required_min_pass=1)
        bundle_quality["template_grounded"] = True
        bundle_quality["template_id"] = item.template_id

        bundle = QuestionBundleRecord(
            bundle_id=f"qb_tg_{stable_hash(f'{dataset_id}|{item.template_id}', 10)}",
            research_question=rq,
            family=str(template.get("primary_family") or "unknown"),
            variants=[candidate],
            bundle_validation=bundle_validation,
            bundle_quality=bundle_quality,
            accepted_local=accepted_local,
            rejected_reason_codes=candidate.rejected_reason_codes,
            provenance={
                "template_id": item.template_id,
                "source_workload_id": template.get("source_workload_id"),
                "template_provenance": template.get("provenance") or {},
            },
        )

        bundle_diversity_records.append(
            {
                "contract_version": "bundle_diversity_matrix_v2",
                "bundle_id": bundle.bundle_id,
                "question_id": rq.question_id,
                "stable_question_id": stable_question_id,
                "family_id": bundle.family,
                "intended_facet_id": item.template_id,
                "variants": [
                    {
                        "query_id": spec.query_id,
                        "stable_query_id": spec.stable_query_id,
                        "variant_id": spec.variant_id,
                        "variant_semantic_role": spec.variant_semantic_role,
                        "accepted_local": candidate.accepted_local,
                        "canonical_sql_hash": spec.canonical_sql_hash,
                    }
                ],
                "pairwise_signals": [],
                "bundle_diversity_score": bundle_quality.get("semantic_diversity_score", 1.0),
                "bundle_novelty_score": bundle_quality.get("informational_novelty_score", 1.0),
                "bundle_reason_codes": bundle_quality.get("bundle_reason_codes") or bundle_validation.reason_codes,
                "pseudo_diversity_flags": bundle_quality.get("pseudo_diversity_flags") or [],
            }
        )

        instances.append(
            {
                "template_id": item.template_id,
                "template_name": template.get("template_name"),
                "query_id": spec.query_id,
                "question_id": question_id,
                "bindings": item.bindings,
                "question": item.question,
                "rationale": item.rationale,
                "quality_notes": item.quality_notes,
                "template_provenance": template.get("provenance") or {},
                "accepted_local": accepted_local,
                "validation_codes": _collect_validation_codes(candidate),
                "acceptance_notes": acceptance_notes,
            }
        )

        bundle_records.append(bundle)
        query_specs.append(spec)
        execution_summaries.append(
            build_query_execution_summary_v2(
                query_spec=spec,
                execution_result=execution,
                validation_result=validation,
            )
        )

    accepted_bundles = [bundle for bundle in bundle_records if bundle.accepted_local]
    accepted_queryspecs = [spec for spec in query_specs if spec.status == "accepted_local"]
    accepted_query_ids = {spec.query_id for spec in accepted_queryspecs}
    accepted_bundle_ids = {bundle.bundle_id for bundle in accepted_bundles}

    artifact_writer.write_json("template_instance_catalog.json", {"instances": instances, "skipped_templates": skipped_templates})
    artifact_writer.write_json("static_understanding.json", static_understanding.to_dict())
    artifact_writer.write_json("queryset_plan.json", {"items": [item.__dict__ for item in plan]})
    artifact_writer.write_jsonl("query_specs.jsonl", [spec.to_dict() for spec in query_specs])
    artifact_writer.write_jsonl("query_execution_summaries_v2.jsonl", execution_summaries)
    artifact_writer.write_jsonl("bundle_diversity_matrix_v2.jsonl", bundle_diversity_records)
    artifact_writer.write_json("question_bundle_pool.json", {"bundles": [bundle.to_dict() for bundle in bundle_records]})

    benchmark_package_dir = artifact_writer.run_dir / "benchmark_package"
    benchmark_package_dir.mkdir(parents=True, exist_ok=True)
    artifact_writer.write_json("benchmark_package/question_bundles.json", {"bundles": [bundle.to_dict() for bundle in accepted_bundles]})
    artifact_writer.write_json("benchmark_package/queryspecs.json", {"queryspecs": [spec.to_dict() for spec in accepted_queryspecs]})
    artifact_writer.write_json(
        "benchmark_package/query_execution_summaries_v2.json",
        {"summaries": [row for row in execution_summaries if row.get("query_id") in accepted_query_ids]},
    )
    artifact_writer.write_json(
        "benchmark_package/bundle_diversity_matrix_v2.json",
        {"bundles": [row for row in bundle_diversity_records if row.get("bundle_id") in accepted_bundle_ids]},
    )
    artifact_writer.write_text(
        "benchmark_package/selected_sql.sql",
        "\n\n".join(
            [
                f"-- {bundle.bundle_id} | {bundle.family} | template_id={bundle.provenance.get('template_id')} | question={bundle.research_question.question}\n"
                + "\n".join(variant.query_spec.sql.rstrip(";") + ";" for variant in bundle.variants if variant.accepted_local)
                for bundle in accepted_bundles
            ]
        ).rstrip()
        + "\n",
    )

    package_summary = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "generation_mode": "template_grounded_queryset_v1",
        "template_pool_size": len(plan),
        "accepted_bundle_count": len(accepted_bundles),
        "accepted_query_count": len(accepted_queryspecs),
        "family_distribution": dict(Counter(bundle.family for bundle in accepted_bundles)),
        "template_source_distribution": _template_source_distribution([row for row in instances if row["accepted_local"]]),
        "skipped_templates": skipped_templates,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact_writer.write_json("benchmark_package/package_summary.json", package_summary)

    grounded_metrics = analyze_package(
        dataset_id=dataset_id,
        package_dir=benchmark_package_dir,
        field_stats=field_stats,
        source_mode="template_grounded",
    )

    selected_baseline_run_id = baseline_run_id or _latest_baseline_run(dataset_id, RUNS_DIR)
    comparison_payload = None
    comparison_report = None
    if selected_baseline_run_id:
        baseline_package_dir = RUNS_DIR / selected_baseline_run_id / "benchmark_package"
        if baseline_package_dir.exists():
            baseline_metrics = analyze_package(
                dataset_id=dataset_id,
                package_dir=baseline_package_dir,
                field_stats=field_stats,
                source_mode="baseline",
            )
            comparison_payload, comparison_report = build_comparison_report(
                dataset_id=dataset_id,
                grounded_run_id=run_id,
                grounded_metrics=grounded_metrics,
                baseline_run_id=selected_baseline_run_id,
                baseline_metrics=baseline_metrics,
            )
            artifact_writer.write_json(
                f"comparison/against_{selected_baseline_run_id}.json",
                comparison_payload,
            )
            artifact_writer.write_text(
                f"comparison/against_{selected_baseline_run_id}.md",
                comparison_report,
            )

    manifest.update(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "accepted_bundle_count": len(accepted_bundles),
                "accepted_query_count": len(accepted_queryspecs),
                "skipped_template_count": len(skipped_templates),
            },
            "comparison_baseline_run_id": selected_baseline_run_id,
        }
    )
    artifact_writer.write_manifest(manifest)

    return {
        "run_id": run_id,
        "run_dir": artifact_writer.run_dir,
        "benchmark_package_dir": benchmark_package_dir,
        "grounded_metrics": grounded_metrics,
        "baseline_run_id": selected_baseline_run_id,
        "comparison_payload": comparison_payload,
        "comparison_report": comparison_report,
    }
