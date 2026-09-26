"""Template catalog helpers for the v2 workload line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqb_query.config.settings import DATA_DIR

from .contract_spec import TEMPLATE_CONTRACTS, default_facet_ids_for_subitem


LEGACY_TEMPLATE_PATHS: tuple[Path, ...] = (
    DATA_DIR / "workload_grounding" / "template_library_v1.jsonl",
    DATA_DIR / "workload_grounding" / "template_library_extensions_v1.jsonl",
)
PUBLIC_TEMPLATE_PATHS: tuple[Path, ...] = (
    DATA_DIR.parents[1] / "Query_Templates" / "templates" / "core" / "subgroup" / "template_library_subgroup_v1.jsonl",
    DATA_DIR.parents[1] / "Query_Templates" / "templates" / "core" / "conditional" / "template_library_conditional_v1.jsonl",
    DATA_DIR.parents[1] / "Query_Templates" / "templates" / "core" / "tail" / "template_library_tail_v1.jsonl",
    DATA_DIR.parents[1] / "Query_Templates" / "templates" / "core" / "cardinality" / "template_library_cardinality_v1.jsonl",
    DATA_DIR.parents[1] / "Query_Templates" / "templates" / "core" / "missing" / "template_library_missing_v1.jsonl",
    DATA_DIR.parents[1] / "Query_Templates" / "templates" / "extensions" / "template_library_extensions_v1.jsonl",
)

COUNT_SUPPORT_TEMPLATES = {
    "tpl_c2_filtered_group_count_2d",
    "tpl_clickbench_filtered_topk_group_count",
    "tpl_clickbench_group_count",
    "tpl_clickbench_two_dimensional_topk_count",
    "tpl_tail_low_support_group_count_v2",
    "tpl_tail_pairwise_sparse_slice_v2",
    "tpl_rtabench_time_bucket_filtered_count",
}

RATE_SHARE_TEMPLATES = {
    "tpl_c2_two_dim_target_rate",
    "tpl_m4_group_condition_rate",
    "tpl_m4_group_ratio_two_conditions",
    "tpl_tpcds_within_group_share",
    "tpl_tail_target_rate_extremes_v2",
}

KEYED_NUMERIC_TEMPLATES = {
    "tpl_cardinality_high_card_response_stability",
    "tpl_conditional_group_quantiles",
    "tpl_m4_binned_numeric_group_avg",
    "tpl_m4_group_dispersion_rank",
    "tpl_m4_window_partition_avg",
    "tpl_h2o_group_sum",
    "tpl_h2o_two_dimensional_group_sum",
    "tpl_h2o_two_dimensional_robust_summary",
    "tpl_m4_group_avg_numeric",
    "tpl_m4_support_guarded_group_avg",
    "tpl_m4_two_dimensional_group_avg",
    "tpl_tpcds_topk_group_sum",
    "tpl_tpch_two_dimensional_summary",
    "tpl_grouped_percentile_point",
    "tpl_tpcds_subgroup_baseline_outlier",
}

TOPK_RANKING_TEMPLATES = {
    "tpl_tpcds_baseline_gated_extreme_ranking",
    "tpl_clickbench_filtered_distinct_topk",
    "tpl_clickbench_group_distinct_topk",
    "tpl_clickbench_group_summary_topk",
    "tpl_tail_weighted_topk_sum",
    "tpl_tpch_max_aggregate_winner",
    "tpl_tpch_relative_total_threshold",
    "tpl_tpch_thresholded_group_ranking",
}

SCALAR_TEMPLATES = {
    "tpl_m4_median_filtered_numeric",
    "tpl_tpch_filtered_sum_band",
    "tpl_missing_marginal_rate_profile",
    "tpl_threshold_rarity_cdf",
    "tpl_cardinality_continuous_range_envelope",
}

TEMPORAL_TEMPLATES = {
    "tpl_rtabench_time_bucket_filtered_count",
    "tpl_rtabench_time_bucket_group_moving_avg",
    "tpl_tail_drift_ratio",
}

RAW_TAIL_TEMPLATES = {
    "tpl_h2o_topn_within_group",
    "tpl_m4_global_zscore_outliers",
    "tpl_m4_quantile_tail_slice",
}

MISSINGNESS_TEMPLATES = {
    "tpl_missing_marginal_rate_profile",
    "tpl_missing_rate_by_subgroup",
    "tpl_missing_target_interaction",
}


def _semantic_contract_for_template(template_contract: Any) -> dict[str, Any]:
    template_id = str(template_contract.template_id)
    binding_roles = list(template_contract.binding_roles)

    if template_contract.semantic_result_contract:
        return dict(template_contract.semantic_result_contract)

    if template_id in MISSINGNESS_TEMPLATES:
        if template_id == "tpl_missing_marginal_rate_profile":
            scorer_type = "missingness_scalar_rate"
            key_roles: list[str] = []
        elif template_id == "tpl_missing_target_interaction":
            scorer_type = "missingness_interaction_rate"
            key_roles = ["target_col"]
        else:
            scorer_type = "missingness_group_rate"
            key_roles = ["group_col"]
        return {
            "scorer_type": scorer_type,
            "result_shape": "scalar" if not key_roles else "keyed_table",
            "key_roles": key_roles,
            "measure_outputs": ["total_rows", "missing_rows", "missing_rate"],
            "primary_measure": "missing_rate",
            "validity_required_outputs": key_roles + ["missing_rate"],
            "score_components": ["missing_rate_similarity", "key_f1", "support_profile_similarity"],
        }

    if template_id == "tpl_h2o_topn_within_group":
        return {
            "scorer_type": "tail_topn_value_curve",
            "result_shape": "ranked_keyed_tail_curve",
            "key_roles": ["group_col"],
            "measure_outputs": ["measure_value", "measure_rank"],
            "primary_measure": "measure_value",
            "rank_by": "measure_value",
            "rank_direction": "desc",
            "selection_semantics": "within_group_topn",
            "validity_required_outputs": ["group_value", "measure_value", "measure_rank"],
            "score_components": ["within_group_rank_similarity", "tail_value_similarity", "group_key_f1"],
        }

    if template_id == "tpl_m4_global_zscore_outliers":
        return {
            "scorer_type": "tail_outlier_summary",
            "result_shape": "scalar_tail_summary",
            "key_roles": [],
            "measure_outputs": [
                "outlier_count",
                "outlier_rate",
                "mean_abs_z",
                "max_abs_z",
                "positive_tail_share",
                "min_outlier_measure",
                "max_outlier_measure",
            ],
            "primary_measure": "outlier_rate",
            "selection_semantics": "global_zscore_outliers",
            "validity_required_outputs": ["outlier_rate"],
            "score_components": ["tail_rate_similarity", "zscore_magnitude_similarity", "tail_boundary_similarity"],
        }

    if template_id == "tpl_m4_quantile_tail_slice":
        return {
            "scorer_type": "tail_distribution_slice",
            "result_shape": "scalar_tail_summary",
            "key_roles": [],
            "measure_outputs": [
                "tail_count",
                "tail_rate",
                "mean_tail_measure",
                "min_tail_measure",
                "max_tail_measure",
            ],
            "primary_measure": "mean_tail_measure",
            "selection_semantics": "top_quantile_slice",
            "validity_required_outputs": ["tail_rate", "mean_tail_measure"],
            "score_components": ["tail_distribution_similarity", "tail_boundary_similarity", "tail_rate_similarity"],
        }

    if template_id in TEMPORAL_TEMPLATES:
        primary = "drift_ratio" if template_id == "tpl_tail_drift_ratio" else "event_count"
        return {
            "scorer_type": "temporal_drift" if template_id == "tpl_tail_drift_ratio" else "temporal_count_curve",
            "result_shape": "temporal_keyed_table",
            "key_roles": [role for role in ("time_col", "group_col") if role in binding_roles],
            "measure_outputs": ["event_count", "moving_avg_count"] if "moving_avg" in template_id else [primary],
            "primary_measure": primary,
            "rank_by": "time_bucket",
            "rank_direction": "asc",
            "selection_semantics": "time_series",
            "validity_required_outputs": ["time_bucket", primary],
            "score_components": ["pointwise_similarity", "temporal_shape_similarity", "total_mass_similarity"],
        }

    if template_id in {"tpl_cardinality_support_rank_profile", "tpl_cardinality_distinct_share_profile"}:
        return {
            "scorer_type": "distribution_cardinality_profile",
            "result_shape": "keyed_distribution",
            "key_roles": ["group_col"],
            "measure_outputs": ["support", "support_share", "support_rank", "cumulative_support"],
            "primary_measure": "support_share",
            "rank_by": "support",
            "rank_direction": "desc",
            "selection_semantics": "full_distribution",
            "validity_required_outputs": ["value_label", "support"],
            "score_components": ["distribution_similarity", "rank_similarity", "cardinality_similarity"],
        }

    if template_id in COUNT_SUPPORT_TEMPLATES:
        support_col = "event_count" if "time_bucket" in template_id else ("row_count" if template_id == "tpl_c2_filtered_group_count_2d" or template_id == "tpl_clickbench_group_count" else "support")
        return {
            "scorer_type": "keyed_count_distribution",
            "result_shape": "keyed_table",
            "key_roles": [role for role in ("group_col", "group_col_2", "time_col") if role in binding_roles],
            "measure_outputs": [support_col],
            "primary_measure": support_col,
            "rank_by": support_col,
            "rank_direction": "asc" if "tail" in template_id or "sparse" in template_id else "desc",
            "selection_semantics": "tailk" if "tail" in template_id or "sparse" in template_id else "count_profile",
            "validity_required_outputs": [support_col],
            "score_components": ["support_distribution_similarity", "total_mass_similarity", "key_f1"],
        }

    if template_id in RATE_SHARE_TEMPLATES:
        primary = "condition_ratio" if "ratio" in template_id else ("share_within_group" if "share" in template_id else ("focus_rate" if "tail_target" in template_id else ("condition_rate" if "condition_rate" in template_id else "target_rate")))
        return {
            "scorer_type": "keyed_ratio_profile" if "ratio" in template_id else "keyed_rate_profile",
            "result_shape": "keyed_table",
            "key_roles": [role for role in ("group_col", "group_col_2", "item_col") if role in binding_roles],
            "measure_outputs": [primary],
            "primary_measure": primary,
            "rank_by": primary,
            "rank_direction": "desc",
            "selection_semantics": "ranked_rate_profile",
            "validity_required_outputs": [primary],
            "score_components": ["rate_similarity", "direction_consistency", "key_f1"],
        }

    if template_id in TOPK_RANKING_TEMPLATES:
        if template_id == "tpl_tpch_max_aggregate_winner":
            primary = "total_measure"
        elif template_id in {"tpl_clickbench_group_distinct_topk", "tpl_clickbench_filtered_distinct_topk"}:
            primary = "distinct_entities"
        elif template_id == "tpl_clickbench_group_summary_topk":
            primary = "support"
        elif template_id == "tpl_tpch_relative_total_threshold":
            primary = "group_value"
        else:
            primary = "total_measure"
        return {
            "scorer_type": "argmax_selection" if template_id == "tpl_tpch_max_aggregate_winner" else "topk_ranked_measure",
            "result_shape": "ranked_keyed_table",
            "key_roles": [role for role in ("group_col", "group_col_2", "item_col", "entity_col") if role in binding_roles],
            "measure_outputs": [primary, "support", "distinct_entities"],
            "primary_measure": primary,
            "rank_by": primary,
            "rank_direction": "desc",
            "selection_semantics": "winner" if template_id == "tpl_tpch_max_aggregate_winner" else "topk",
            "validity_required_outputs": [primary],
            "score_components": ["set_overlap", "rank_similarity", "measure_similarity"],
        }

    if template_id in SCALAR_TEMPLATES:
        primary = {
            "tpl_m4_median_filtered_numeric": "median_measure",
            "tpl_tpch_filtered_sum_band": "total_measure",
            "tpl_missing_marginal_rate_profile": "missing_rate",
            "tpl_threshold_rarity_cdf": "empirical_cdf_at_threshold",
            "tpl_cardinality_continuous_range_envelope": "range_width",
        }.get(template_id, "value")
        return {
            "scorer_type": "scalar_rate" if primary in {"missing_rate", "empirical_cdf_at_threshold"} else "scalar_numeric",
            "result_shape": "scalar",
            "key_roles": [],
            "measure_outputs": [primary],
            "primary_measure": primary,
            "validity_required_outputs": [primary],
            "score_components": ["scalar_similarity"],
        }

    if template_id in KEYED_NUMERIC_TEMPLATES:
        primary = "avg_response" if template_id == "tpl_cardinality_high_card_response_stability" else "avg_measure"
        return {
            "scorer_type": "keyed_numeric_aggregate",
            "result_shape": "keyed_table",
            "key_roles": [role for role in ("group_col", "group_col_2", "key_col", "item_col", "band_col") if role in binding_roles],
            "measure_outputs": ["avg_measure", "sum_measure", "total_measure", "avg_response", "median_measure", "measure_stddev", "percentile_measure"],
            "primary_measure": primary,
            "rank_by": "primary_measure",
            "rank_direction": "desc",
            "selection_semantics": "numeric_profile",
            "validity_required_outputs": [],
            "score_components": ["numeric_measure_similarity", "key_f1", "rank_similarity"],
        }

    return {
        "scorer_type": "generic_keyed_result",
        "result_shape": "keyed_table",
        "key_roles": [role for role in binding_roles if role.endswith("_col") and role != "measure_col"],
        "measure_roles": [role for role in binding_roles if role == "measure_col"],
        "validity_required_outputs": [],
        "score_components": ["strict_set_score", "key_f1", "measure_similarity"],
    }


def _role_constraints_for_template(template_contract: Any) -> dict[str, Any]:
    if template_contract.role_constraints:
        return dict(template_contract.role_constraints)

    template_id = str(template_contract.template_id)
    distinct_roles: list[list[str]] = []
    if "measure_col" in template_contract.binding_roles:
        distinct_roles.extend([
            ["group_col", "measure_col"],
            ["group_col_2", "measure_col"],
            ["key_col", "measure_col"],
            ["band_col", "measure_col"],
        ])
    if "condition_col" in template_contract.binding_roles:
        distinct_roles.extend([
            ["group_col", "condition_col"],
            ["group_col_2", "condition_col"],
        ])
    if "target_col" in template_contract.binding_roles:
        distinct_roles.extend([
            ["group_col", "target_col"],
            ["group_col_2", "target_col"],
            ["key_col", "target_col"],
        ])
    if "entity_col" in template_contract.binding_roles:
        distinct_roles.extend([
            ["group_col", "entity_col"],
            ["group_col_2", "entity_col"],
        ])
    if "predicate_col" in template_contract.binding_roles:
        distinct_roles.extend([
            ["group_col", "predicate_col"],
            ["group_col_2", "predicate_col"],
        ])

    constraints: dict[str, Any] = {"distinct_roles": distinct_roles}
    if template_id in {
        "tpl_clickbench_group_distinct_topk",
        "tpl_clickbench_filtered_distinct_topk",
        "tpl_clickbench_group_summary_topk",
    }:
        constraints["forbid_distinct_self_count"] = True
    if template_id in RAW_TAIL_TEMPLATES:
        constraints["raw_row_output_forbidden"] = True
    return constraints


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalized_sql_skeleton(template_id: str) -> str | None:
    if template_id == "tpl_h2o_topn_within_group":
        return (
            "WITH ranked AS (\n"
            "    SELECT\n"
            "        {group_col} AS group_value,\n"
            "        {measure_col} AS measure_value,\n"
            "        ROW_NUMBER() OVER (PARTITION BY {group_col} ORDER BY {measure_col} DESC) AS measure_rank\n"
            "    FROM {table}\n"
            "    WHERE {measure_col} IS NOT NULL\n"
            ")\n"
            "SELECT group_value, measure_value, measure_rank\n"
            "FROM ranked\n"
            "WHERE measure_rank <= {top_n}\n"
            "ORDER BY group_value, measure_rank;"
        )
    if template_id == "tpl_m4_global_zscore_outliers":
        return (
            "WITH base AS (\n"
            "    SELECT CAST({measure_col} AS FLOAT) AS measure_value\n"
            "    FROM {table}\n"
            "    WHERE {measure_col} IS NOT NULL\n"
            "), stats AS (\n"
            "    SELECT\n"
            "        AVG(measure_value) AS mean_value,\n"
            "        STDDEV(measure_value) AS stddev_value\n"
            "    FROM base\n"
            "), scored AS (\n"
            "    SELECT\n"
            "        measure_value,\n"
            "        (measure_value - mean_value) / NULLIF(stddev_value, 0) AS z_score\n"
            "    FROM base CROSS JOIN stats\n"
            ")\n"
            "SELECT\n"
            "    SUM(CASE WHEN ABS(z_score) > {z_threshold} THEN 1 ELSE 0 END) AS outlier_count,\n"
            "    AVG(CASE WHEN ABS(z_score) > {z_threshold} THEN 1.0 ELSE 0.0 END) AS outlier_rate,\n"
            "    AVG(CASE WHEN ABS(z_score) > {z_threshold} THEN ABS(z_score) ELSE NULL END) AS mean_abs_z,\n"
            "    MAX(CASE WHEN ABS(z_score) > {z_threshold} THEN ABS(z_score) ELSE NULL END) AS max_abs_z,\n"
            "    AVG(CASE WHEN ABS(z_score) > {z_threshold} AND z_score > 0 THEN 1.0 WHEN ABS(z_score) > {z_threshold} THEN 0.0 ELSE NULL END) AS positive_tail_share,\n"
            "    MIN(CASE WHEN ABS(z_score) > {z_threshold} THEN measure_value ELSE NULL END) AS min_outlier_measure,\n"
            "    MAX(CASE WHEN ABS(z_score) > {z_threshold} THEN measure_value ELSE NULL END) AS max_outlier_measure\n"
            "FROM scored;"
        )
    if template_id == "tpl_m4_quantile_tail_slice":
        return (
            "WITH base AS (\n"
            "    SELECT CAST({measure_col} AS FLOAT) AS measure_value\n"
            "    FROM {table}\n"
            "    WHERE {measure_col} IS NOT NULL\n"
            "), ranked AS (\n"
            "    SELECT\n"
            "        measure_value,\n"
            "        ROW_NUMBER() OVER (ORDER BY measure_value DESC) AS rn,\n"
            "        COUNT(*) OVER () AS total_rows\n"
            "    FROM base\n"
            "), tail AS (\n"
            "    SELECT measure_value, total_rows\n"
            "    FROM ranked\n"
            "    WHERE rn <= ((total_rows + {num_tiles} - 1) / {num_tiles})\n"
            ")\n"
            "SELECT\n"
            "    COUNT(*) AS tail_count,\n"
            "    COUNT(*) * 1.0 / NULLIF(MAX(total_rows), 0) AS tail_rate,\n"
            "    AVG(measure_value) AS mean_tail_measure,\n"
            "    MIN(measure_value) AS min_tail_measure,\n"
            "    MAX(measure_value) AS max_tail_measure\n"
            "FROM tail;"
        )
    return None


def _legacy_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in LEGACY_TEMPLATE_PATHS + PUBLIC_TEMPLATE_PATHS:
        if not path.exists():
            continue
        for row in _load_jsonl_rows(path):
            skeleton = _normalized_sql_skeleton(str(row.get("template_id") or ""))
            if skeleton is not None:
                row = dict(row)
                row["sql_skeleton"] = skeleton
            lookup[row["template_id"]] = row
    return lookup


def _local_sql_skeleton(template_id: str) -> str:
    skeletons = {
        "tpl_tail_low_support_group_count_v2": (
            "SELECT\n"
            "    {group_col},\n"
            "    COUNT(*) AS support\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY support ASC, {group_col}\n"
            "LIMIT {top_k};"
        ),
        "tpl_tail_pairwise_sparse_slice_v2": (
            "SELECT\n"
            "    {group_col},\n"
            "    {group_col_2},\n"
            "    COUNT(*) AS support\n"
            "FROM {table}\n"
            "GROUP BY {group_col}, {group_col_2}\n"
            "ORDER BY support ASC, {group_col}, {group_col_2}\n"
            "LIMIT {top_k};"
        ),
        "tpl_tail_target_rate_extremes_v2": (
            "SELECT\n"
            "    {group_col},\n"
            "    COUNT(*) AS support,\n"
            "    AVG(CASE WHEN {target_col} = {target_value} THEN 1 ELSE 0 END) AS focus_rate\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "HAVING COUNT(*) >= {min_support}\n"
            "ORDER BY focus_rate DESC, support ASC\n"
            "LIMIT {top_k};"
        ),
        "tpl_missing_marginal_rate_profile": (
            "SELECT\n"
            "    COUNT(*) AS total_rows,\n"
            "    SUM(CASE WHEN {missing_col} IS NULL THEN 1 ELSE 0 END) AS missing_rows,\n"
            "    AVG(CASE WHEN {missing_col} IS NULL THEN 1.0 ELSE 0.0 END) AS missing_rate\n"
            "FROM {table};"
        ),
        "tpl_missing_rate_by_subgroup": (
            "SELECT\n"
            "    {group_col},\n"
            "    COUNT(*) AS total_rows,\n"
            "    SUM(CASE WHEN {missing_col} IS NULL THEN 1 ELSE 0 END) AS missing_rows,\n"
            "    AVG(CASE WHEN {missing_col} IS NULL THEN 1.0 ELSE 0.0 END) AS missing_rate\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY missing_rate DESC, total_rows DESC;"
        ),
        "tpl_missing_target_interaction": (
            "SELECT\n"
            "    {target_col},\n"
            "    COUNT(*) AS total_rows,\n"
            "    SUM(CASE WHEN {missing_col} IS NULL THEN 1 ELSE 0 END) AS missing_rows,\n"
            "    AVG(CASE WHEN {missing_col} IS NULL THEN 1.0 ELSE 0.0 END) AS missing_rate\n"
            "FROM {table}\n"
            "GROUP BY {target_col}\n"
            "ORDER BY missing_rate DESC, total_rows DESC;"
        ),
        "tpl_cardinality_support_rank_profile": (
            "WITH grouped AS (\n"
            "    SELECT {group_col} AS value_label, COUNT(*) AS support\n"
            "    FROM {table}\n"
            "    GROUP BY {group_col}\n"
            ")\n"
            "SELECT\n"
            "    value_label,\n"
            "    support,\n"
            "    CAST(support AS FLOAT) / NULLIF(SUM(support) OVER (), 0) AS support_share,\n"
            "    ROW_NUMBER() OVER (ORDER BY support DESC, value_label) AS support_rank\n"
            "FROM grouped\n"
            "ORDER BY support DESC, value_label;"
        ),
        "tpl_cardinality_distinct_share_profile": (
            "WITH grouped AS (\n"
            "    SELECT {group_col} AS value_label, COUNT(*) AS support\n"
            "    FROM {table}\n"
            "    GROUP BY {group_col}\n"
            "), ranked AS (\n"
            "    SELECT\n"
            "        value_label,\n"
            "        support,\n"
            "        CAST(support AS FLOAT) / NULLIF(SUM(support) OVER (), 0) AS support_share,\n"
            "        SUM(support) OVER (ORDER BY support DESC, value_label ROWS UNBOUNDED PRECEDING) AS cumulative_support\n"
            "    FROM grouped\n"
            ")\n"
            "SELECT *\n"
            "FROM ranked\n"
            "ORDER BY support DESC, value_label;"
        ),
        "tpl_cardinality_high_card_response_stability": (
            "SELECT\n"
            "    {key_col},\n"
            "    COUNT(*) AS support,\n"
            "    AVG({measure_col}) AS avg_response\n"
            "FROM {table}\n"
            "GROUP BY {key_col}\n"
            "HAVING COUNT(*) >= {min_support}\n"
            "ORDER BY support DESC, avg_response DESC;"
        ),
        "tpl_cardinality_continuous_range_envelope": (
            "SELECT\n"
            "    MIN({measure_col}) AS min_value,\n"
            "    MAX({measure_col}) AS max_value,\n"
            "    MAX({measure_col}) - MIN({measure_col}) AS range_width\n"
            "FROM {table}\n"
            "WHERE {measure_col} IS NOT NULL;"
        ),
    }
    return skeletons[template_id]


def _local_template_row(template_contract: Any) -> dict[str, Any]:
    provenance = {
        "url": "local://subitem_workload_v2",
        "title": "Locally authored v2 template",
        "source_query_id": template_contract.template_id,
    }
    materialization_bucket = "v2_deterministic" if template_contract.realization_mode == "deterministic" else "v2_agent"
    return {
        "template_id": template_contract.template_id,
        "template_name": template_contract.template_name,
        "source_workload_id": "subitem_workload_v2",
        "primary_family": template_contract.family_id,
        "secondary_family": None,
        "intent": template_contract.notes or f"Locally authored v2 template for {template_contract.template_name}.",
        "sql_skeleton": _local_sql_skeleton(template_contract.template_id),
        "required_roles": list(template_contract.binding_roles),
        "optional_roles": [],
        "constraints": ["single_table_only", f"v2_{template_contract.realization_mode}_template"],
        "single_table_portable": "yes",
        "provenance": provenance,
        "provenance_sources": [provenance],
        "status": "ready",
        "notes": template_contract.notes,
        "materialization_bucket": materialization_bucket,
        "activation_tier": "v2",
        "dialect_sensitive": False,
    }


def build_template_library_rows() -> list[dict[str, Any]]:
    legacy_lookup = _legacy_lookup()
    rows: list[dict[str, Any]] = []
    for contract in TEMPLATE_CONTRACTS:
        if contract.source_catalog == "template_library_v2":
            base_row = _local_template_row(contract)
        else:
            base_row = dict(legacy_lookup[contract.template_id])
        row = dict(base_row)
        row["family_id"] = contract.family_id
        row["realization_mode"] = contract.realization_mode
        row["binding_roles"] = list(contract.binding_roles)
        row["supported_canonical_subitem_ids"] = list(contract.supported_canonical_subitem_ids)
        row["allowed_variant_roles"] = list(contract.allowed_variant_roles)
        if contract.template_default_facet_ids:
            row["default_facet_ids"] = list(contract.template_default_facet_ids)
        else:
            row["default_facet_ids"] = list(
                {
                    facet_id
                    for subitem_id in contract.supported_canonical_subitem_ids
                    for facet_id in default_facet_ids_for_subitem(subitem_id)
                }
            )
        row["gate_priority"] = contract.gate_priority
        row["source_catalog"] = contract.source_catalog
        row["extended_family"] = contract.extended_family
        row["semantic_result_contract"] = _semantic_contract_for_template(contract)
        row["role_constraints"] = _role_constraints_for_template(contract)
        rows.append(row)
    rows.sort(key=lambda item: (str(item["family_id"]), str(item["template_id"])))
    return rows


def write_template_library_jsonl(output_path: Path) -> list[dict[str, Any]]:
    rows = build_template_library_rows()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_template_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {row["template_id"]: row for row in _load_jsonl_rows(path)}
