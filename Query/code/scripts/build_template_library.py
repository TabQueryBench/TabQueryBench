#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build workload-grounded SQL template library v1.")
    parser.add_argument(
        "--catalog",
        default="data/workload_grounding/workload_catalog.csv",
        help="Path to workload catalog CSV.",
    )
    parser.add_argument(
        "--mapping",
        default="data/workload_grounding/workload_to_family_mapping_v1.csv",
        help="Path to workload-to-family mapping CSV.",
    )
    parser.add_argument(
        "--source-bank",
        default="data/workload_grounding/source_query_bank_v1.jsonl",
        help="Path to curated source-query bank JSONL.",
    )
    parser.add_argument(
        "--output",
        default="data/workload_grounding/template_library_v1.jsonl",
        help="Output JSONL path for template library.",
    )
    parser.add_argument(
        "--extension-output",
        default="data/workload_grounding/template_library_extensions_v1.jsonl",
        help="Output JSONL path for extension-only templates kept outside the core library.",
    )
    parser.add_argument(
        "--logs-root",
        default="logs/workload_grounding",
        help="Root directory for run manifests.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. Defaults to current timestamp in YYYYMMDD_HHMMSS format.",
    )
    return parser.parse_args()


DIALECT_NOTES = {
    "tpl_h2o_two_dimensional_robust_summary": (
        "Uses ordered-set percentile and standard-deviation aggregates. "
        "Keep it optional unless the downstream SQL engine supports PERCENTILE_CONT/QUANTILE_CONT-style syntax."
    ),
    "tpl_grouped_percentile_point": (
        "Represents a canonical percentile-point family, but concrete SQL differs across engines "
        "(for example PERCENTILE_CONT, APPROX_QUANTILES, approx_percentile, or quantile-style syntax)."
    ),
    "tpl_conditional_group_quantiles": (
        "Uses percentile syntax plus conditional aggregation/filter semantics. "
        "Keep it optional unless the downstream engine supports ordered-set percentiles and FILTER/If-style conditioning."
    ),
}


def parse_flag_set(raw_value: str | None) -> set[str]:
    if not raw_value:
        return set()
    return {item.strip() for item in raw_value.split(";") if item.strip()}


def parse_id_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(";") if item.strip()]


TEMPLATE_REGISTRY: dict[str, dict[str, Any]] = {
    "group_count_by_category": {
        "intent": "Count rows by a single subgroup dimension to capture baseline subgroup mass.",
        "sql_skeleton": (
            "SELECT {group_col}, COUNT(*) AS row_count\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY row_count DESC;"
        ),
        "required_roles": ["group_col"],
        "optional_roles": [],
        "constraints": ["group_col:groupable", "single_table_only"],
        "status": "ready",
    },
    "two_dimensional_condition_rate": {
        "intent": "Measure how a categorical target rate changes across a pair of subgroup axes.",
        "sql_skeleton": (
            "SELECT {group_col}, {group_col_2},\n"
            "       AVG(CASE WHEN {target_col} = {target_value} THEN 1 ELSE 0 END) AS target_rate\n"
            "FROM {table}\n"
            "GROUP BY {group_col}, {group_col_2}\n"
            "ORDER BY target_rate DESC;"
        ),
        "required_roles": ["group_col", "group_col_2", "target_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "group_col_2:groupable_distinct_from_group_col",
            "target_col:categorical_or_binary",
            "single_table_only",
        ],
        "status": "ready",
    },
    "filtered_group_count_2d": {
        "intent": "Count rows for a filtered slice across two subgroup dimensions.",
        "sql_skeleton": (
            "SELECT {group_col}, {group_col_2}, COUNT(*) AS row_count\n"
            "FROM {table}\n"
            "WHERE {predicate_col} {predicate_op} {predicate_value}\n"
            "GROUP BY {group_col}, {group_col_2}\n"
            "ORDER BY row_count DESC;"
        ),
        "required_roles": ["group_col", "group_col_2", "predicate_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "group_col_2:groupable_distinct_from_group_col",
            "predicate_col:filterable",
            "single_table_only",
        ],
        "status": "ready",
    },
    "group_avg_numeric": {
        "intent": "Compare mean numeric outcomes across subgroups.",
        "sql_skeleton": (
            "SELECT {group_col}, AVG({measure_col}) AS avg_measure\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY avg_measure DESC;"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": ["group_col:groupable", "measure_col:numeric", "single_table_only"],
        "status": "ready",
    },
    "group_sum_by_category": {
        "intent": "Compare total numeric mass across subgroups using a simple grouped sum.",
        "sql_skeleton": (
            "SELECT {group_col}, SUM({measure_col}) AS total_measure\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY total_measure DESC;"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": ["group_col:groupable", "measure_col:numeric", "single_table_only"],
        "status": "ready",
    },
    "group_condition_rate": {
        "intent": "Estimate the proportion of rows meeting a low-cardinality condition within each subgroup.",
        "sql_skeleton": (
            "SELECT {group_col},\n"
            "       AVG(CASE WHEN {condition_col} = {condition_value} THEN 1 ELSE 0 END) AS condition_rate\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY condition_rate DESC;"
        ),
        "required_roles": ["group_col", "condition_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "condition_col:binary_or_low_cardinality_preferred",
            "single_table_only",
        ],
        "status": "ready",
    },
    "median_filtered_numeric": {
        "intent": "Compute a median-like robust center for a filtered numeric slice.",
        "sql_skeleton": (
            "WITH ranked AS (\n"
            "    SELECT {measure_col},\n"
            "           ROW_NUMBER() OVER (ORDER BY {measure_col}) AS row_num,\n"
            "           COUNT(*) OVER () AS total_rows\n"
            "    FROM {table}\n"
            "    WHERE {predicate_col} {predicate_op} {predicate_value}\n"
            ")\n"
            "SELECT AVG({measure_col}) AS median_measure\n"
            "FROM ranked\n"
            "WHERE row_num BETWEEN (total_rows + 1) / 2 AND (total_rows + 2) / 2;"
        ),
        "required_roles": ["measure_col", "predicate_col"],
        "optional_roles": [],
        "constraints": ["measure_col:numeric", "predicate_col:filterable", "single_table_only"],
        "status": "ready",
    },
    "support_guarded_group_avg": {
        "intent": "Compute subgroup averages only when support exceeds a configurable minimum.",
        "sql_skeleton": (
            "SELECT {group_col}, AVG({measure_col}) AS avg_measure, COUNT(*) AS support\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "HAVING COUNT(*) > {min_group_size}\n"
            "ORDER BY {group_col};"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "support_guard:minimum_group_size",
            "single_table_only",
        ],
        "status": "ready",
    },
    "group_ratio_two_conditions": {
        "intent": "Contrast two condition counts within each subgroup as a ratio.",
        "sql_skeleton": (
            "WITH grouped AS (\n"
            "    SELECT {group_col},\n"
            "           SUM(CASE WHEN {condition_col} = {positive_value} THEN 1 ELSE 0 END) AS numerator_count,\n"
            "           SUM(CASE WHEN {condition_col} = {negative_value} THEN 1 ELSE 0 END) AS denominator_count\n"
            "    FROM {table}\n"
            "    GROUP BY {group_col}\n"
            ")\n"
            "SELECT {group_col},\n"
            "       CAST(numerator_count AS FLOAT) / NULLIF(denominator_count, 0) AS condition_ratio\n"
            "FROM grouped\n"
            "ORDER BY condition_ratio DESC;"
        ),
        "required_roles": ["group_col", "condition_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "condition_col:binary_or_low_cardinality_preferred",
            "single_table_only",
        ],
        "status": "ready",
    },
    "group_distinct_topk": {
        "intent": "Find the top subgroups by distinct-entity coverage.",
        "sql_skeleton": (
            "SELECT {group_col}, COUNT(DISTINCT {entity_col}) AS distinct_entities\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY distinct_entities DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "entity_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "entity_col:high_cardinality_preferred",
            "single_table_only",
        ],
        "status": "ready",
    },
    "group_summary_topk": {
        "intent": "Rank subgroups by support while also reporting a numeric mean and distinct-entity coverage.",
        "sql_skeleton": (
            "SELECT {group_col},\n"
            "       COUNT(*) AS support,\n"
            "       AVG({measure_col}) AS avg_measure,\n"
            "       COUNT(DISTINCT {entity_col}) AS distinct_entities\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY support DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "measure_col", "entity_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "entity_col:high_cardinality_preferred",
            "single_table_only",
        ],
        "status": "ready",
    },
    "two_dimensional_topk_count": {
        "intent": "Find the heaviest two-dimensional subgroup combinations by row count.",
        "sql_skeleton": (
            "SELECT {group_col}, {group_col_2}, COUNT(*) AS support\n"
            "FROM {table}\n"
            "GROUP BY {group_col}, {group_col_2}\n"
            "ORDER BY support DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "group_col_2"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "group_col_2:groupable_distinct_from_group_col",
            "single_table_only",
        ],
        "status": "ready",
    },
    "two_dimensional_group_sum": {
        "intent": "Compare total numeric mass across a two-way subgroup grid.",
        "sql_skeleton": (
            "SELECT {group_col}, {group_col_2}, SUM({measure_col}) AS total_measure\n"
            "FROM {table}\n"
            "GROUP BY {group_col}, {group_col_2}\n"
            "ORDER BY total_measure DESC;"
        ),
        "required_roles": ["group_col", "group_col_2", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "group_col_2:groupable_distinct_from_group_col",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "filtered_topk_group_count": {
        "intent": "Rank subgroups by support within a filtered slice.",
        "sql_skeleton": (
            "SELECT {group_col}, COUNT(*) AS support\n"
            "FROM {table}\n"
            "WHERE {predicate_col} {predicate_op} {predicate_value}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY support DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "predicate_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "predicate_col:filterable",
            "single_table_only",
        ],
        "status": "ready",
    },
    "filtered_distinct_topk": {
        "intent": "Rank subgroups by distinct-entity coverage within a filtered slice.",
        "sql_skeleton": (
            "SELECT {group_col}, COUNT(DISTINCT {entity_col}) AS distinct_entities\n"
            "FROM {table}\n"
            "WHERE {predicate_col} {predicate_op} {predicate_value}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY distinct_entities DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "entity_col", "predicate_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "entity_col:high_cardinality_preferred",
            "predicate_col:filterable",
            "single_table_only",
        ],
        "status": "ready",
    },
    "value_range_check": {
        "intent": "Check the observed minimum and maximum of an ordered field.",
        "sql_skeleton": (
            "SELECT MIN({measure_col}) AS min_value, MAX({measure_col}) AS max_value\n"
            "FROM {table};"
        ),
        "required_roles": ["measure_col"],
        "optional_roles": [],
        "constraints": ["measure_col:ordered_or_numeric", "single_table_only"],
        "status": "ready",
    },
    "text_length_having_topk": {
        "intent": "Rank large-support groups by average text length to surface long-tail text behavior.",
        "sql_skeleton": (
            "SELECT {group_col}, AVG(LENGTH({text_col})) AS avg_text_len, COUNT(*) AS support\n"
            "FROM {table}\n"
            "WHERE {text_col} <> ''\n"
            "GROUP BY {group_col}\n"
            "HAVING COUNT(*) > {min_group_size}\n"
            "ORDER BY avg_text_len DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "text_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "text_col:text_like",
            "support_guard:minimum_group_size",
            "single_table_only",
        ],
        "status": "needs_review",
    },
    "two_dimensional_summary": {
        "intent": "Summarize a numeric measure across two grouping axes with an additional filter.",
        "sql_skeleton": (
            "SELECT {group_col}, {group_col_2},\n"
            "       SUM({measure_col}) AS sum_measure,\n"
            "       AVG({measure_col}) AS avg_measure,\n"
            "       COUNT(*) AS support\n"
            "FROM {table}\n"
            "WHERE {predicate_col} {predicate_op} {predicate_value}\n"
            "GROUP BY {group_col}, {group_col_2}\n"
            "ORDER BY {group_col}, {group_col_2};"
        ),
        "required_roles": ["group_col", "group_col_2", "measure_col", "predicate_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "group_col_2:groupable_distinct_from_group_col",
            "measure_col:numeric",
            "predicate_col:ordered_or_numeric_preferred",
            "single_table_only",
        ],
        "status": "ready",
    },
    "filtered_sum_band": {
        "intent": "Aggregate a numeric measure within a numeric band filter.",
        "sql_skeleton": (
            "SELECT SUM({measure_col}) AS total_measure\n"
            "FROM {table}\n"
            "WHERE {band_col} BETWEEN {lower_bound} AND {upper_bound};"
        ),
        "required_roles": ["measure_col", "band_col"],
        "optional_roles": [],
        "constraints": [
            "measure_col:numeric",
            "band_col:ordered_or_numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "topk_group_sum": {
        "intent": "Rank subgroups by total numeric measure under a filter.",
        "sql_skeleton": (
            "SELECT {group_col}, SUM({measure_col}) AS total_measure\n"
            "FROM {table}\n"
            "WHERE {predicate_col} {predicate_op} {predicate_value}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY total_measure DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "measure_col", "predicate_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "predicate_col:filterable",
            "single_table_only",
        ],
        "status": "ready",
    },
    "within_group_share": {
        "intent": "Measure each item's contribution within a parent subgroup using a windowed share-of-total.",
        "sql_skeleton": (
            "SELECT {group_col}, {item_col},\n"
            "       SUM({measure_col}) AS total_measure,\n"
            "       SUM({measure_col}) * 100.0 / SUM(SUM({measure_col})) OVER (PARTITION BY {group_col}) AS share_within_group\n"
            "FROM {table}\n"
            "GROUP BY {group_col}, {item_col}\n"
            "ORDER BY share_within_group DESC;"
        ),
        "required_roles": ["group_col", "item_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "item_col:groupable_or_high_cardinality",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "window_partition_avg": {
        "intent": "Use a window function to report per-group averages without collapsing the row-level relation first.",
        "sql_skeleton": (
            "SELECT DISTINCT {group_col},\n"
            "       AVG({measure_col}) OVER (PARTITION BY {group_col}) AS avg_measure\n"
            "FROM {table}\n"
            "ORDER BY avg_measure DESC;"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "time_bucket_filtered_count": {
        "intent": "Count events per time bucket within a filtered slice.",
        "sql_skeleton": (
            "SELECT DATE_TRUNC('{time_grain}', {time_col}) AS time_bucket,\n"
            "       COUNT(*) AS event_count\n"
            "FROM {table}\n"
            "WHERE {predicate_col} {predicate_op} {predicate_value}\n"
            "GROUP BY time_bucket\n"
            "ORDER BY time_bucket;"
        ),
        "required_roles": ["time_col", "predicate_col"],
        "optional_roles": [],
        "constraints": [
            "time_col:temporal",
            "predicate_col:filterable",
            "single_table_only",
        ],
        "status": "ready",
    },
    "time_bucket_group_moving_avg": {
        "intent": "Compute a rolling moving average over time-bucketed subgroup counts.",
        "sql_skeleton": (
            "WITH bucketed AS (\n"
            "    SELECT DATE_TRUNC('{time_grain}', {time_col}) AS time_bucket,\n"
            "           {group_col},\n"
            "           COUNT(*) AS event_count\n"
            "    FROM {table}\n"
            "    WHERE {predicate_col} {predicate_op} {predicate_value}\n"
            "    GROUP BY time_bucket, {group_col}\n"
            ")\n"
            "SELECT time_bucket,\n"
            "       {group_col},\n"
            "       event_count,\n"
            "       AVG(event_count) OVER (\n"
            "           PARTITION BY {group_col}\n"
            "           ORDER BY time_bucket\n"
            "           ROWS BETWEEN {lookback_rows} PRECEDING AND CURRENT ROW\n"
            "       ) AS moving_avg_count\n"
            "FROM bucketed\n"
            "ORDER BY {group_col}, time_bucket;"
        ),
        "required_roles": ["time_col", "group_col", "predicate_col"],
        "optional_roles": [],
        "constraints": [
            "time_col:temporal",
            "group_col:groupable",
            "predicate_col:filterable",
            "single_table_only",
        ],
        "status": "ready",
    },
    "quantile_tail_slice": {
        "intent": "Select the highest quantile bucket of a numeric measure using NTILE-style ranking.",
        "sql_skeleton": (
            "WITH buckets AS (\n"
            "    SELECT {measure_col},\n"
            "           NTILE({num_tiles}) OVER (ORDER BY {measure_col} DESC) AS tail_bucket\n"
            "    FROM {table}\n"
            ")\n"
            "SELECT {measure_col}\n"
            "FROM buckets\n"
            "WHERE tail_bucket = 1\n"
            "ORDER BY {measure_col} DESC;"
        ),
        "required_roles": ["measure_col"],
        "optional_roles": [],
        "constraints": [
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "group_dispersion_rank": {
        "intent": "Rank subgroups by within-group dispersion of a numeric measure.",
        "sql_skeleton": (
            "SELECT {group_col}, STDDEV({measure_col}) AS measure_stddev\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY measure_stddev DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "global_zscore_outliers": {
        "intent": "Score a numeric measure globally and surface high-z-score outliers.",
        "sql_skeleton": (
            "WITH scored AS (\n"
            "    SELECT *,\n"
            "           ({measure_col} - AVG({measure_col}) OVER ())\n"
            "           / NULLIF(STDDEV({measure_col}) OVER (), 0) AS z_score\n"
            "    FROM {table}\n"
            ")\n"
            "SELECT *\n"
            "FROM scored\n"
            "WHERE ABS(z_score) > {z_threshold}\n"
            "ORDER BY {measure_col} DESC;"
        ),
        "required_roles": ["measure_col"],
        "optional_roles": [],
        "constraints": [
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "relative_total_threshold": {
        "intent": "Keep only groups whose aggregate value exceeds a configurable fraction of the grand total.",
        "sql_skeleton": (
            "WITH grouped AS (\n"
            "    SELECT {group_col}, SUM({measure_col}) AS group_value\n"
            "    FROM {table}\n"
            "    GROUP BY {group_col}\n"
            "), total AS (\n"
            "    SELECT SUM(group_value) AS total_value\n"
            "    FROM grouped\n"
            ")\n"
            "SELECT g.{group_col}, g.group_value\n"
            "FROM grouped AS g\n"
            "CROSS JOIN total AS t\n"
            "WHERE g.group_value > t.total_value * {fraction_threshold}\n"
            "ORDER BY g.group_value DESC;"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "max_aggregate_winner": {
        "intent": "Aggregate by group and keep only the winner whose aggregate value is maximal.",
        "sql_skeleton": (
            "WITH grouped AS (\n"
            "    SELECT {group_col}, SUM({measure_col}) AS total_measure\n"
            "    FROM {table}\n"
            "    GROUP BY {group_col}\n"
            ")\n"
            "SELECT {group_col}, total_measure\n"
            "FROM grouped\n"
            "WHERE total_measure = (SELECT MAX(total_measure) FROM grouped)\n"
            "ORDER BY {group_col};"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "thresholded_group_ranking": {
        "intent": "Rank only those groups whose aggregate value exceeds an explicit threshold.",
        "sql_skeleton": (
            "SELECT {group_col}, SUM({measure_col}) AS total_measure\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "HAVING SUM({measure_col}) > {measure_threshold}\n"
            "ORDER BY total_measure DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "subgroup_baseline_outlier": {
        "intent": "Find entity-level aggregates that are extreme relative to their own subgroup baseline.",
        "sql_skeleton": (
            "WITH entity_totals AS (\n"
            "    SELECT {group_col}, {item_col}, SUM({measure_col}) AS entity_measure\n"
            "    FROM {table}\n"
            "    GROUP BY {group_col}, {item_col}\n"
            "), subgroup_baseline AS (\n"
            "    SELECT {group_col}, AVG(entity_measure) AS subgroup_avg\n"
            "    FROM entity_totals\n"
            "    GROUP BY {group_col}\n"
            ")\n"
            "SELECT e.{group_col}, e.{item_col}, e.entity_measure, b.subgroup_avg\n"
            "FROM entity_totals AS e\n"
            "JOIN subgroup_baseline AS b\n"
            "  ON e.{group_col} = b.{group_col}\n"
            "WHERE e.entity_measure > b.subgroup_avg * {baseline_multiplier}\n"
            "ORDER BY e.entity_measure DESC;"
        ),
        "required_roles": ["group_col", "item_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "item_col:groupable_or_high_cardinality",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "baseline_gated_extreme_ranking": {
        "intent": "Apply a subgroup baseline gate before ranking entities by an extreme aggregate outcome.",
        "sql_skeleton": (
            "WITH item_stats AS (\n"
            "    SELECT {group_col}, {item_col}, AVG({measure_col}) AS avg_measure\n"
            "    FROM {table}\n"
            "    GROUP BY {group_col}, {item_col}\n"
            "), group_baseline AS (\n"
            "    SELECT {group_col}, AVG(avg_measure) AS group_avg\n"
            "    FROM item_stats\n"
            "    GROUP BY {group_col}\n"
            "), eligible AS (\n"
            "    SELECT i.{group_col}, i.{item_col}, i.avg_measure\n"
            "    FROM item_stats AS i\n"
            "    JOIN group_baseline AS g\n"
            "      ON i.{group_col} = g.{group_col}\n"
            "    WHERE i.avg_measure > g.group_avg * {baseline_fraction}\n"
            ")\n"
            "SELECT {group_col}, {item_col}, avg_measure,\n"
            "       RANK() OVER (PARTITION BY {group_col} ORDER BY avg_measure DESC) AS within_group_rank\n"
            "FROM eligible\n"
            "ORDER BY avg_measure DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "item_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "item_col:groupable_or_high_cardinality",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "binned_numeric_group_avg": {
        "intent": "Bin a numeric feature into coarse buckets and compare average outcomes across those bins.",
        "sql_skeleton": (
            "SELECT CASE\n"
            "           WHEN {band_col} < {band_cut_1} THEN 'low'\n"
            "           WHEN {band_col} < {band_cut_2} THEN 'mid'\n"
            "           ELSE 'high'\n"
            "       END AS band_bucket,\n"
            "       AVG({measure_col}) AS avg_measure\n"
            "FROM {table}\n"
            "GROUP BY band_bucket\n"
            "ORDER BY avg_measure DESC;"
        ),
        "required_roles": ["band_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "band_col:ordered_or_numeric",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "weighted_topk_sum": {
        "intent": "Rank groups by weighted aggregate mass while preserving both support and weighted total.",
        "sql_skeleton": (
            "SELECT {group_col},\n"
            "       SUM({measure_col}) AS weighted_total,\n"
            "       COUNT(*) AS support\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "HAVING COUNT(*) >= {min_support}\n"
            "ORDER BY weighted_total DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "support_guard:minimum_group_size",
            "single_table_only",
        ],
        "status": "ready",
    },
    "grouped_percentile_point": {
        "intent": "Report a percentile point such as p95 or p99 for each subgroup instead of returning the raw tail rows.",
        "sql_skeleton": (
            "SELECT {group_col},\n"
            "       PERCENTILE_CONT({percentile_value}) WITHIN GROUP (ORDER BY {measure_col}) AS percentile_measure\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY percentile_measure DESC;"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "conditional_group_quantiles": {
        "intent": "Report subgroup percentile points only for rows satisfying a low-cardinality condition.",
        "sql_skeleton": (
            "SELECT {group_col},\n"
            "       PERCENTILE_CONT({percentile_value}) WITHIN GROUP (ORDER BY {measure_col})\n"
            "           FILTER (WHERE {condition_col} = {condition_value}) AS conditional_percentile\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY conditional_percentile DESC;"
        ),
        "required_roles": ["group_col", "measure_col", "condition_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "condition_col:binary_or_low_cardinality_preferred",
            "single_table_only",
        ],
        "status": "ready",
    },
    "threshold_rarity_cdf": {
        "intent": "Estimate how rare a threshold is by reporting the empirical CDF value at that threshold.",
        "sql_skeleton": (
            "SELECT AVG(CASE WHEN {measure_col} <= {measure_threshold} THEN 1 ELSE 0 END) AS empirical_cdf_at_threshold\n"
            "FROM {table};"
        ),
        "required_roles": ["measure_col"],
        "optional_roles": [],
        "constraints": [
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "tail_drift_ratio": {
        "intent": "Compare current-period to prior-period subgroup volume and flag material negative drift.",
        "sql_skeleton": (
            "WITH period_counts AS (\n"
            "    SELECT {group_col},\n"
            "           SUM(CASE WHEN {time_col} >= {current_period_start} AND {time_col} < {current_period_end} THEN 1 ELSE 0 END) AS current_count,\n"
            "           SUM(CASE WHEN {time_col} >= {previous_period_start} AND {time_col} < {previous_period_end} THEN 1 ELSE 0 END) AS previous_count\n"
            "    FROM {table}\n"
            "    GROUP BY {group_col}\n"
            ")\n"
            "SELECT {group_col}, current_count, previous_count,\n"
            "       CAST(current_count AS FLOAT) / NULLIF(previous_count, 0) AS drift_ratio\n"
            "FROM period_counts\n"
            "WHERE CAST(current_count AS FLOAT) / NULLIF(previous_count, 0) < {drift_ratio_threshold}\n"
            "ORDER BY drift_ratio ASC;"
        ),
        "required_roles": ["group_col", "time_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "time_col:temporal",
            "single_table_only",
        ],
        "status": "ready",
    },
    "two_dimensional_group_avg": {
        "intent": "Compare average numeric outcomes across a two-way subgroup grid.",
        "sql_skeleton": (
            "SELECT {group_col}, {group_col_2}, AVG({measure_col}) AS avg_measure\n"
            "FROM {table}\n"
            "GROUP BY {group_col}, {group_col_2}\n"
            "ORDER BY avg_measure DESC;"
        ),
        "required_roles": ["group_col", "group_col_2", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "group_col_2:groupable_distinct_from_group_col",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "two_dimensional_robust_summary": {
        "intent": "Compare robust center and spread of a numeric measure across a two-way subgroup grid.",
        "sql_skeleton": (
            "SELECT {group_col}, {group_col_2},\n"
            "       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {measure_col}) AS median_measure,\n"
            "       STDDEV({measure_col}) AS measure_stddev\n"
            "FROM {table}\n"
            "GROUP BY {group_col}, {group_col_2}\n"
            "ORDER BY median_measure DESC;"
        ),
        "required_roles": ["group_col", "group_col_2", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "group_col_2:groupable_distinct_from_group_col",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "topn_within_group_by_measure": {
        "intent": "Retain the top-N numeric values within each subgroup using window ranking.",
        "sql_skeleton": (
            "WITH ranked AS (\n"
            "    SELECT {group_col}, {measure_col},\n"
            "           ROW_NUMBER() OVER (PARTITION BY {group_col} ORDER BY {measure_col} DESC) AS measure_rank\n"
            "    FROM {table}\n"
            "    WHERE {measure_col} IS NOT NULL\n"
            ")\n"
            "SELECT {group_col}, {measure_col}, measure_rank\n"
            "FROM ranked\n"
            "WHERE measure_rank <= {top_n}\n"
            "ORDER BY {group_col}, measure_rank;"
        ),
        "required_roles": ["group_col", "measure_col"],
        "optional_roles": [],
        "constraints": [
            "group_col:groupable",
            "measure_col:numeric",
            "single_table_only",
        ],
        "status": "ready",
    },
    "count_distinct_check": {
        "intent": "Validate the observed cardinality of a column via COUNT DISTINCT.",
        "sql_skeleton": (
            "SELECT COUNT(DISTINCT {entity_col}) AS distinct_count\n"
            "FROM {table};"
        ),
        "required_roles": ["entity_col"],
        "optional_roles": [],
        "constraints": ["entity_col:any_column", "single_table_only"],
        "status": "ready",
    },
    "duplicate_key_check": {
        "intent": "Find duplicate values for a candidate key column.",
        "sql_skeleton": (
            "SELECT {key_col}, COUNT(*) AS duplicate_count\n"
            "FROM {table}\n"
            "GROUP BY {key_col}\n"
            "HAVING COUNT(*) > 1\n"
            "ORDER BY duplicate_count DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["key_col"],
        "optional_roles": [],
        "constraints": ["key_col:any_column", "single_table_only"],
        "status": "ready",
    },
    "duplicate_composite_key_check": {
        "intent": "Find duplicate row signatures for a candidate composite key.",
        "sql_skeleton": (
            "SELECT {key_col}, {key_col_2}, COUNT(*) AS duplicate_count\n"
            "FROM {table}\n"
            "GROUP BY {key_col}, {key_col_2}\n"
            "HAVING COUNT(*) > 1\n"
            "ORDER BY duplicate_count DESC\n"
            "LIMIT {top_k};"
        ),
        "required_roles": ["key_col", "key_col_2"],
        "optional_roles": [],
        "constraints": [
            "key_col:any_column",
            "key_col_2:distinct_from_key_col",
            "single_table_only",
        ],
        "status": "ready",
    },
    "missing_rate_check": {
        "intent": "Measure the missingness rate for a chosen column, usually after synthetic missing-value injection.",
        "sql_skeleton": (
            "SELECT AVG(CASE WHEN {missing_col} IS NULL THEN 1 ELSE 0 END) AS missing_rate\n"
            "FROM {table};"
        ),
        "required_roles": ["missing_col"],
        "optional_roles": [],
        "constraints": ["missing_col:any_feature_for_injection", "single_table_only"],
        "status": "ready",
    },
    "impossible_combo_check": {
        "intent": "Count rows that satisfy a domain-forbidden combination of attribute values.",
        "sql_skeleton": (
            "SELECT COUNT(*) AS violating_rows\n"
            "FROM {table}\n"
            "WHERE {condition_col} = {condition_value}\n"
            "  AND {condition_col_2} = {condition_value_2};"
        ),
        "required_roles": ["condition_col", "condition_col_2"],
        "optional_roles": [],
        "constraints": [
            "condition_col:filterable",
            "condition_col_2:distinct_from_condition_col",
            "domain_rule_required",
            "single_table_only",
        ],
        "status": "blocked",
    },
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    catalog_path = Path(args.catalog)
    mapping_path = Path(args.mapping)
    source_bank_path = Path(args.source_bank)
    output_path = Path(args.output)
    extension_output_path = Path(args.extension_output)
    logs_root = Path(args.logs_root)
    run_id = args.run_id or default_run_id()
    run_dir = logs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension_output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog_rows = load_csv(catalog_path)
    mapping_rows = load_csv(mapping_path)
    source_rows = load_jsonl(source_bank_path)

    catalog = {row["workload_id"]: row for row in catalog_rows}
    source_bank = {row["source_query_id"]: row for row in source_rows}

    core_templates: list[dict[str, Any]] = []
    extension_templates: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    duplicate_template_ids: list[str] = []
    seen_template_ids: set[str] = set()

    for row in mapping_rows:
        template_id = (row.get("template_id") or "").strip()
        template_kind = (row.get("template_kind") or "").strip()
        source_query_id = (row.get("source_query_id") or "").strip()
        materialization_bucket = (row.get("materialization_bucket") or "").strip() or "core"
        template_flags = parse_flag_set(row.get("template_flags"))
        additional_source_query_ids = parse_id_list(row.get("additional_source_query_ids"))

        if materialization_bucket not in {"core", "extension", "prior_only"}:
            raise ValueError(
                f"Unknown materialization_bucket={materialization_bucket!r} in mapping_id={row.get('mapping_id')}"
            )

        if materialization_bucket == "prior_only":
            skipped_rows.append(
                {
                    "mapping_id": row.get("mapping_id", "unknown"),
                    "reason": "materialization_bucket=prior_only",
                }
            )
            continue

        if not template_id or not template_kind or not source_query_id:
            skipped_rows.append(
                {
                    "mapping_id": row.get("mapping_id", "unknown"),
                    "reason": "no_template_materialized_for_v1",
                }
            )
            continue

        if template_id in seen_template_ids:
            duplicate_template_ids.append(template_id)
            continue
        seen_template_ids.add(template_id)

        if template_kind not in TEMPLATE_REGISTRY:
            raise KeyError(f"Unknown template_kind={template_kind!r} in mapping_id={row.get('mapping_id')}")
        if row["workload_id"] not in catalog:
            raise KeyError(f"Unknown workload_id={row['workload_id']!r} in mapping_id={row.get('mapping_id')}")
        if source_query_id not in source_bank:
            raise KeyError(f"Unknown source_query_id={source_query_id!r} in mapping_id={row.get('mapping_id')}")
        missing_additional = [qid for qid in additional_source_query_ids if qid not in source_bank]
        if missing_additional:
            raise KeyError(
                f"Unknown additional_source_query_ids={missing_additional!r} in mapping_id={row.get('mapping_id')}"
            )

        spec = TEMPLATE_REGISTRY[template_kind]
        source = source_bank[source_query_id]
        provenance_sources = [
            {
                "url": source["source_url"],
                "title": source["source_title"],
                "source_query_id": source["source_query_label"],
            }
        ]
        for extra_source_query_id in additional_source_query_ids:
            extra_source = source_bank[extra_source_query_id]
            provenance_sources.append(
                {
                    "url": extra_source["source_url"],
                    "title": extra_source["source_title"],
                    "source_query_id": extra_source["source_query_label"],
                }
            )
        template = {
            "template_id": template_id,
            "template_name": row["template_name"],
            "source_workload_id": row["workload_id"],
            "primary_family": row["primary_family"],
            "secondary_family": (row.get("secondary_family") or "").strip() or None,
            "intent": row.get("template_intent") or spec["intent"],
            "sql_skeleton": row.get("sql_skeleton_override") or spec["sql_skeleton"],
            "required_roles": spec["required_roles"],
            "optional_roles": spec["optional_roles"],
            "constraints": spec["constraints"],
            "single_table_portable": row["single_table_portable"],
            "provenance": {
                "url": source["source_url"],
                "title": source["source_title"],
                "source_query_id": source["source_query_label"],
            },
            "provenance_sources": provenance_sources,
            "status": row.get("status_hint") or spec["status"],
            "notes": row.get("template_notes") or row.get("pattern_description") or "",
            "materialization_bucket": materialization_bucket,
            "activation_tier": (
                "extension"
                if materialization_bucket == "extension"
                else ("optional" if "optional" in template_flags else "core")
            ),
            "dialect_sensitive": "dialect_sensitive" in template_flags,
        }
        if template["dialect_sensitive"]:
            template["dialect_notes"] = DIALECT_NOTES.get(
                template_id,
                "Requires SQL features that are not guaranteed across every target engine.",
            )
        if materialization_bucket == "extension":
            extension_templates.append(template)
        else:
            core_templates.append(template)

    with output_path.open("w", encoding="utf-8") as handle:
        for template in core_templates:
            handle.write(json.dumps(template, ensure_ascii=False) + "\n")
    with extension_output_path.open("w", encoding="utf-8") as handle:
        for template in extension_templates:
            handle.write(json.dumps(template, ensure_ascii=False) + "\n")

    manifest = {
        "run_id": run_id,
        "generated_at_utc": now_utc_iso(),
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "catalog": {
                "path": str(catalog_path.resolve()),
                "sha256": sha256_file(catalog_path),
                "row_count": len(catalog_rows),
            },
            "mapping": {
                "path": str(mapping_path.resolve()),
                "sha256": sha256_file(mapping_path),
                "row_count": len(mapping_rows),
            },
            "source_bank": {
                "path": str(source_bank_path.resolve()),
                "sha256": sha256_file(source_bank_path),
                "row_count": len(source_rows),
            },
        },
        "outputs": {
            "template_library": {
                "path": str(output_path.resolve()),
                "row_count": len(core_templates),
            },
            "template_library_extensions": {
                "path": str(extension_output_path.resolve()),
                "row_count": len(extension_templates),
            }
        },
        "skipped_mappings": skipped_rows,
        "duplicate_template_ids": duplicate_template_ids,
        "workload_coverage": sorted(
            {template["source_workload_id"] for template in core_templates + extension_templates}
        ),
    }

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": run_id,
        "template_count": len(core_templates),
        "extension_template_count": len(extension_templates),
        "manifest_path": str(manifest_path.resolve()),
        "output_path": str(output_path.resolve()),
        "extension_output_path": str(extension_output_path.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
