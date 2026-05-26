#!/usr/bin/env python3
"""Apply a restrained tail-template expansion to workload-grounding assets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MAPPING_FIELDNAMES = [
    "mapping_id",
    "workload_id",
    "pattern_name",
    "pattern_description",
    "primary_family",
    "secondary_family",
    "evidence_url",
    "evidence_snippet",
    "single_table_portable",
    "portability_notes",
    "confidence",
    "source_query_id",
    "template_id",
    "template_name",
    "template_kind",
    "status_hint",
    "template_notes",
    "materialization_bucket",
    "template_flags",
    "additional_source_query_ids",
]


WORKLOAD_UPSERTS = [
    {
        "workload_id": "tpcds_altinity_queries",
        "workload_name": "TPC-DS Altinity Query Templates",
        "source_type": "repo",
        "source_url": "https://github.com/Altinity/tpc-ds",
        "has_raw_sql": "yes",
        "sql_access_level": "public_repo_query_sql",
        "notes": "Official public TPC-DS query repository from Altinity used here for tail-oriented subgroup-baseline, drift, and concentration patterns.",
        "reliability_level": "high",
    },
    {
        "workload_id": "bigquery_approx_aggregate_docs",
        "workload_name": "BigQuery Approximate Aggregate Documentation Examples",
        "source_type": "doc",
        "source_url": "https://cloud.google.com/bigquery/docs/reference/standard-sql/approximate_aggregate_functions",
        "has_raw_sql": "yes",
        "sql_access_level": "public_doc_sql_examples",
        "notes": "Official Google Cloud examples for approximate quantiles and weighted top-k style aggregates; used as canonical public evidence for reusable tail-query families.",
        "reliability_level": "high",
    },
    {
        "workload_id": "trino_aggregate_docs",
        "workload_name": "Trino Aggregate Function Documentation",
        "source_type": "doc",
        "source_url": "https://trino.io/docs/current/functions/aggregate.html",
        "has_raw_sql": "partial",
        "sql_access_level": "public_doc_function_syntax",
        "notes": "Official Trino aggregate-function docs used as secondary evidence for approximate percentile and weighted percentile template families.",
        "reliability_level": "high",
    },
    {
        "workload_id": "snowflake_sql_docs",
        "workload_name": "Snowflake SQL Function Documentation",
        "source_type": "doc",
        "source_url": "https://docs.snowflake.com/en/sql-reference/functions/percentile_cont",
        "has_raw_sql": "yes",
        "sql_access_level": "public_doc_sql_examples",
        "notes": "Official Snowflake docs provide grouped percentile examples that strengthen the percentile-point tail family.",
        "reliability_level": "high",
    },
    {
        "workload_id": "clickhouse_aggregate_docs",
        "workload_name": "ClickHouse Aggregate Function Documentation",
        "source_type": "doc",
        "source_url": "https://clickhouse.com/docs",
        "has_raw_sql": "yes",
        "sql_access_level": "public_doc_sql_examples",
        "notes": "Official ClickHouse docs and combinator examples provide public SQL for weighted concentration and conditional quantile tail monitoring.",
        "reliability_level": "high",
    },
    {
        "workload_id": "druid_sql_functions",
        "workload_name": "Apache Druid SQL Functions Documentation",
        "source_type": "doc",
        "source_url": "https://druid.apache.org/docs/latest/querying/sql-functions/",
        "has_raw_sql": "yes",
        "sql_access_level": "public_doc_sql_examples",
        "notes": "Official Druid docs provide sketch-based quantile and rarity-rank examples that map cleanly to tail monitoring templates.",
        "reliability_level": "high",
    },
    {
        "workload_id": "pinot_aggregate_docs",
        "workload_name": "Apache Pinot Aggregation Function Documentation",
        "source_type": "doc",
        "source_url": "https://docs.pinot.apache.org/functions/aggregation/percentile",
        "has_raw_sql": "yes",
        "sql_access_level": "public_doc_sql_examples",
        "notes": "Official Pinot percentile docs are used as secondary public evidence for grouped percentile-point templates.",
        "reliability_level": "high",
    },
]


SOURCE_UPSERTS = [
    {
        "source_query_id": "tpch_q11",
        "workload_id": "tpch_qgen",
        "source_url": "https://raw.githubusercontent.com/electrum/tpch-dbgen/master/queries/11.sql",
        "source_title": "queries/11.sql · electrum/tpch-dbgen",
        "source_query_label": "TPC-H Q11",
        "sql_text": "... value > total_value * 0.0001 ...",
        "retrieval_notes": "Core predicate excerpt retained from the public TPC-H Q11 query template; used to derive the relative-to-total threshold family without inventing full denormalized SQL.",
    },
    {
        "source_query_id": "tpch_q15",
        "workload_id": "tpch_qgen",
        "source_url": "https://raw.githubusercontent.com/electrum/tpch-dbgen/master/queries/15.sql",
        "source_title": "queries/15.sql · electrum/tpch-dbgen",
        "source_query_label": "TPC-H Q15",
        "sql_text": "... total_revenue = (SELECT MAX(total_revenue) ...) ...",
        "retrieval_notes": "Core winner-selection excerpt retained from the public TPC-H Q15 template; enough to ground the aggregate-then-pick-max family.",
    },
    {
        "source_query_id": "tpch_q18",
        "workload_id": "tpch_qgen",
        "source_url": "https://raw.githubusercontent.com/electrum/tpch-dbgen/master/queries/18.sql",
        "source_title": "queries/18.sql · electrum/tpch-dbgen",
        "source_query_label": "TPC-H Q18",
        "sql_text": "... GROUP BY ... HAVING SUM(l_quantity) > 300 ORDER BY ... LIMIT 100;",
        "retrieval_notes": "Threshold-plus-ranking excerpt from the public TPC-H Q18 template; used because the core structure survives single-table abstraction.",
    },
    {
        "source_query_id": "tpch_q22",
        "workload_id": "tpch_qgen",
        "source_url": "https://raw.githubusercontent.com/electrum/tpch-dbgen/master/queries/22.sql",
        "source_title": "queries/22.sql · electrum/tpch-dbgen",
        "source_query_label": "TPC-H Q22",
        "sql_text": "... c_acctbal > AVG(c_acctbal) AND NOT EXISTS (SELECT * FROM orders ...) ...",
        "retrieval_notes": "High-balance inactive-segment excerpt from the public TPC-H Q22 template; kept as evidence because the semantic requirements are stronger than the current core quality bar.",
    },
    {
        "source_query_id": "tpcds_alt_q1",
        "workload_id": "tpcds_altinity_queries",
        "source_url": "https://github.com/Altinity/tpc-ds/blob/master/queries/query_1.sql",
        "source_title": "query_1.sql · Altinity/tpc-ds",
        "source_query_label": "TPC-DS Q1",
        "sql_text": "WHERE ctr1.ctr_total_return > (SELECT AVG(ctr_total_return) * 1.2 FROM customer_total_return ctr2 WHERE ctr1.ctr_store_sk = ctr2.ctr_store_sk)",
        "retrieval_notes": "Core subgroup-baseline predicate excerpt retained from the public Altinity TPC-DS query_1.sql page.",
    },
    {
        "source_query_id": "tpcds_alt_q44",
        "workload_id": "tpcds_altinity_queries",
        "source_url": "https://github.com/Altinity/tpc-ds/blob/master/queries/query_44.sql",
        "source_title": "query_44.sql · Altinity/tpc-ds",
        "source_query_label": "TPC-DS Q44",
        "sql_text": "... RANK() OVER (...) ... HAVING AVG(ss_net_profit) > 0.9 * store_avg ...",
        "retrieval_notes": "Core baseline-gated ranking excerpt retained from the public Altinity TPC-DS query_44.sql page.",
    },
    {
        "source_query_id": "tpcds_alt_q75",
        "workload_id": "tpcds_altinity_queries",
        "source_url": "https://github.com/Altinity/tpc-ds/blob/master/queries/query_75.sql",
        "source_title": "query_75.sql · Altinity/tpc-ds",
        "source_query_label": "TPC-DS Q75",
        "sql_text": "... curr_yr.sales_cnt / prev_yr.sales_cnt < 0.9 ...",
        "retrieval_notes": "Year-over-year decline excerpt retained from the public Altinity TPC-DS query_75.sql page.",
    },
    {
        "source_query_id": "tpcds_alt_q78",
        "workload_id": "tpcds_altinity_queries",
        "source_url": "https://github.com/Altinity/tpc-ds/blob/master/queries/query_78.sql",
        "source_title": "query_78.sql · Altinity/tpc-ds",
        "source_query_label": "TPC-DS Q78",
        "sql_text": "... ROUND(ss_qty / (COALESCE(ws_qty, 0) + COALESCE(cs_qty, 0)), 2) AS ratio ...",
        "retrieval_notes": "Channel-vs-rest concentration ratio excerpt retained from the public Altinity TPC-DS query_78.sql page.",
    },
    {
        "source_query_id": "bigquery_approx_quantiles",
        "workload_id": "bigquery_approx_aggregate_docs",
        "source_url": "https://cloud.google.com/bigquery/docs/reference/standard-sql/approximate_aggregate_functions",
        "source_title": "Approximate aggregate functions | BigQuery | Google Cloud Documentation",
        "source_query_label": "BigQuery APPROX_QUANTILES example",
        "sql_text": "SELECT APPROX_QUANTILES(x, 100)[OFFSET(90)] AS approx_p90 FROM UNNEST([...]) AS x;",
        "retrieval_notes": "Official BigQuery docs example retained as the primary public evidence for grouped percentile-point tail templates.",
    },
    {
        "source_query_id": "bigquery_approx_top_sum",
        "workload_id": "bigquery_approx_aggregate_docs",
        "source_url": "https://cloud.google.com/bigquery/docs/reference/standard-sql/approximate_aggregate_functions",
        "source_title": "Approximate aggregate functions | BigQuery | Google Cloud Documentation",
        "source_query_label": "BigQuery APPROX_TOP_SUM example",
        "sql_text": "SELECT APPROX_TOP_SUM(x, weight, 2) AS approx_top_sum FROM UNNEST([...]) AS x;",
        "retrieval_notes": "Official BigQuery docs example retained as the primary public evidence for weighted top-k concentration templates.",
    },
    {
        "source_query_id": "trino_approx_percentile",
        "workload_id": "trino_aggregate_docs",
        "source_url": "https://trino.io/docs/current/functions/aggregate.html",
        "source_title": "Aggregate functions — Trino Documentation",
        "source_query_label": "Trino approx_percentile",
        "sql_text": "approx_percentile(x, percentage)",
        "retrieval_notes": "Official Trino aggregate docs retained as secondary evidence for percentile-point templates.",
    },
    {
        "source_query_id": "trino_weighted_approx_percentile",
        "workload_id": "trino_aggregate_docs",
        "source_url": "https://trino.io/docs/current/functions/aggregate.html",
        "source_title": "Aggregate functions — Trino Documentation",
        "source_query_label": "Trino weighted approx_percentile",
        "sql_text": "approx_percentile(x, w, percentage)",
        "retrieval_notes": "Official Trino docs retained as evidence for weighted percentile families; kept prior-only in this pass.",
    },
    {
        "source_query_id": "snowflake_percentile_cont",
        "workload_id": "snowflake_sql_docs",
        "source_url": "https://docs.snowflake.com/en/sql-reference/functions/percentile_cont",
        "source_title": "PERCENTILE_CONT | Snowflake Documentation",
        "source_query_label": "Snowflake PERCENTILE_CONT grouped example",
        "sql_text": "SELECT k, PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY v) FROM t GROUP BY k;",
        "retrieval_notes": "Official Snowflake grouped percentile example retained as secondary evidence for percentile-point templates.",
    },
    {
        "source_query_id": "clickhouse_quantile",
        "workload_id": "clickhouse_aggregate_docs",
        "source_url": "https://clickhouse.com/docs/sql-reference/aggregate-functions/reference/quantile",
        "source_title": "quantile | ClickHouse Docs",
        "source_query_label": "ClickHouse quantile example",
        "sql_text": "SELECT quantile(0.95)(expr) FROM t;",
        "retrieval_notes": "Official ClickHouse quantile reference retained as secondary evidence for percentile-point templates.",
    },
    {
        "source_query_id": "clickhouse_approx_top_sum",
        "workload_id": "clickhouse_aggregate_docs",
        "source_url": "https://clickhouse.com/docs/sql-reference/aggregate-functions/reference/approxtopsum",
        "source_title": "approx_top_sum | ClickHouse Docs",
        "source_query_label": "ClickHouse approx_top_sum example",
        "sql_text": "SELECT approx_top_sum(2)(k, w) FROM t;",
        "retrieval_notes": "Official ClickHouse docs retained as secondary evidence for weighted top-k concentration templates.",
    },
    {
        "source_query_id": "clickhouse_quantiles_timing_if",
        "workload_id": "clickhouse_aggregate_docs",
        "source_url": "https://clickhouse.com/docs/examples/aggregate-function-combinators/quantilesTimingIf",
        "source_title": "quantilesTimingIf | ClickHouse Docs",
        "source_query_label": "ClickHouse quantilesTimingIf example",
        "sql_text": "SELECT quantilesTimingIf(0.5, 0.95, 0.99)(response_time_ms, is_successful = 1) FROM requests GROUP BY endpoint;",
        "retrieval_notes": "Official ClickHouse example retained as primary evidence for conditional group quantile templates.",
    },
    {
        "source_query_id": "druid_approx_quantile_ds",
        "workload_id": "druid_sql_functions",
        "source_url": "https://druid.apache.org/docs/latest/querying/sql-functions/",
        "source_title": "All Druid SQL functions | Apache Druid",
        "source_query_label": "Druid APPROX_QUANTILE_DS example",
        "sql_text": "SELECT APPROX_QUANTILE_DS(\"Distance\", 0.95, 128) FROM \"flight-carriers\";",
        "retrieval_notes": "Official Druid sketch-based quantile example retained as secondary evidence for percentile-point templates.",
    },
    {
        "source_query_id": "druid_ds_rank",
        "workload_id": "druid_sql_functions",
        "source_url": "https://druid.apache.org/docs/latest/querying/sql-functions/",
        "source_title": "All Druid SQL functions | Apache Druid",
        "source_query_label": "Druid DS_RANK example",
        "sql_text": "SELECT DS_RANK(DS_QUANTILES_SKETCH(\"Distance\"), 500) AS estimate_rank FROM \"flight-carriers\";",
        "retrieval_notes": "Official Druid docs retained as the primary public evidence for threshold-rarity CDF templates.",
    },
    {
        "source_query_id": "pinot_percentile",
        "workload_id": "pinot_aggregate_docs",
        "source_url": "https://docs.pinot.apache.org/functions/aggregation/percentile",
        "source_title": "percentile | Apache Pinot Docs",
        "source_query_label": "Pinot percentile example",
        "sql_text": "SELECT percentile(homeRuns, 99.9) AS value FROM baseballStats;",
        "retrieval_notes": "Official Pinot docs retained as secondary evidence for high-percentile template families.",
    },
]


MAPPING_UPSERTS = [
    {
        "mapping_id": "map_015",
        "workload_id": "tpch_qgen",
        "pattern_name": "filtered_numeric_band_sum",
        "pattern_description": "Filtered aggregate over a numeric band, abstracted from a forecasting-style revenue query.",
        "primary_family": "conditional_dependency_structure",
        "secondary_family": "tail_rarity_structure",
        "evidence_url": "https://raw.githubusercontent.com/electrum/tpch-dbgen/master/queries/6.sql",
        "evidence_snippet": "TPC-H Q6 sums revenue inside discount and quantity bands after a date-range filter.",
        "single_table_portable": "partial",
        "portability_notes": "Portable when the dataset exposes a numeric measure, an ordered/numeric band column, and a filterable predicate column. The temporal semantics remain collapsed away.",
        "confidence": "high",
        "source_query_id": "tpch_q6",
        "template_id": "tpl_tpch_filtered_sum_band",
        "template_name": "Filtered Sum in Numeric Band",
        "template_kind": "filtered_sum_band",
        "status_hint": "ready",
        "template_notes": "Promoted into the materialized core because the tail review showed this narrow-band threshold slice is a canonical low-support but high-impact pattern rather than a benchmark curiosity.",
        "materialization_bucket": "core",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_040",
        "workload_id": "tpch_qgen",
        "pattern_name": "relative_to_total_extreme_threshold",
        "pattern_description": "Keep only groups whose aggregated value exceeds a tiny fraction of the overall total.",
        "primary_family": "tail_rarity_structure",
        "secondary_family": "conditional_dependency_structure",
        "evidence_url": "https://raw.githubusercontent.com/electrum/tpch-dbgen/master/queries/11.sql",
        "evidence_snippet": "TPC-H Q11 keeps only parts whose stock value exceeds a tiny fraction of the total stock value.",
        "single_table_portable": "partial",
        "portability_notes": "Strong single-table abstraction once the grouped value and total baseline are derived from the same fact table.",
        "confidence": "high",
        "source_query_id": "tpch_q11",
        "template_id": "tpl_tpch_relative_total_threshold",
        "template_name": "Relative-to-Total Extreme Threshold",
        "template_kind": "relative_total_threshold",
        "status_hint": "ready",
        "template_notes": "Canonical low-support but high-impact segment template: entity value above a tiny fraction of total.",
        "materialization_bucket": "core",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_041",
        "workload_id": "tpch_qgen",
        "pattern_name": "max_aggregate_winner_selection",
        "pattern_description": "Aggregate by group and keep only the winner(s) with the maximum aggregate value.",
        "primary_family": "subgroup_structure",
        "secondary_family": "tail_rarity_structure",
        "evidence_url": "https://raw.githubusercontent.com/electrum/tpch-dbgen/master/queries/15.sql",
        "evidence_snippet": "TPC-H Q15 selects supplier rows whose total revenue equals the maximum total revenue.",
        "single_table_portable": "partial",
        "portability_notes": "Best when one group axis and one numeric measure represent the ranked entity and its value contribution.",
        "confidence": "high",
        "source_query_id": "tpch_q15",
        "template_id": "tpl_tpch_max_aggregate_winner",
        "template_name": "Max Aggregate Winner Selection",
        "template_kind": "max_aggregate_winner",
        "status_hint": "ready",
        "template_notes": "Distinct from ordinary top-k because it encodes winner-only selection after grouped aggregation.",
        "materialization_bucket": "core",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_042",
        "workload_id": "tpch_qgen",
        "pattern_name": "thresholded_group_ranking",
        "pattern_description": "Rank only those groups whose aggregated value exceeds an absolute threshold.",
        "primary_family": "tail_rarity_structure",
        "secondary_family": "subgroup_structure",
        "evidence_url": "https://raw.githubusercontent.com/electrum/tpch-dbgen/master/queries/18.sql",
        "evidence_snippet": "TPC-H Q18 retains only orders above a large quantity threshold and then ranks them.",
        "single_table_portable": "partial",
        "portability_notes": "Portable when the dataset has one group axis and a numeric measure that can be aggregated then thresholded.",
        "confidence": "high",
        "source_query_id": "tpch_q18",
        "template_id": "tpl_tpch_thresholded_group_ranking",
        "template_name": "Thresholded Group Ranking",
        "template_kind": "thresholded_group_ranking",
        "status_hint": "ready",
        "template_notes": "Separates true large-segment ranking from ordinary support guards by thresholding the aggregate itself.",
        "materialization_bucket": "core",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_043",
        "workload_id": "tpcds_altinity_queries",
        "pattern_name": "subgroup_baseline_outlier",
        "pattern_description": "Flag entities whose aggregate value is extreme relative to the baseline of their own subgroup.",
        "primary_family": "tail_rarity_structure",
        "secondary_family": "conditional_dependency_structure",
        "evidence_url": "https://github.com/Altinity/tpc-ds/blob/master/queries/query_1.sql",
        "evidence_snippet": "TPC-DS Q1 filters customers whose total returns exceed 1.2x the average return of their store.",
        "single_table_portable": "partial",
        "portability_notes": "Portable when entity rows and subgroup identifiers can be folded into one table and the measure can be aggregated before comparing to subgroup baseline.",
        "confidence": "high",
        "source_query_id": "tpcds_alt_q1",
        "template_id": "tpl_tpcds_subgroup_baseline_outlier",
        "template_name": "Subgroup Baseline Outlier",
        "template_kind": "subgroup_baseline_outlier",
        "status_hint": "ready",
        "template_notes": "High-value because it captures rarity relative to a local subgroup baseline, not just global magnitude.",
        "materialization_bucket": "core",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_044",
        "workload_id": "tpcds_altinity_queries",
        "pattern_name": "baseline_gated_extreme_ranking",
        "pattern_description": "Apply a subgroup baseline gate before ranking items or entities by an extreme aggregate outcome.",
        "primary_family": "conditional_dependency_structure",
        "secondary_family": "tail_rarity_structure",
        "evidence_url": "https://github.com/Altinity/tpc-ds/blob/master/queries/query_44.sql",
        "evidence_snippet": "TPC-DS Q44 ranks items only after requiring average net profit above a subgroup baseline floor.",
        "single_table_portable": "partial",
        "portability_notes": "Requires an entity role, a subgroup role, and a numeric measure. Best when we want rankable extremes but also a baseline floor.",
        "confidence": "high",
        "source_query_id": "tpcds_alt_q44",
        "template_id": "tpl_tpcds_baseline_gated_extreme_ranking",
        "template_name": "Baseline-Gated Extreme Ranking",
        "template_kind": "baseline_gated_extreme_ranking",
        "status_hint": "ready",
        "template_notes": "Distinct from plain top-k because the ranking only happens after a relative baseline gate is cleared.",
        "materialization_bucket": "core",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_045",
        "workload_id": "bigquery_approx_aggregate_docs",
        "pattern_name": "weighted_topk_sum",
        "pattern_description": "Rank groups by aggregated weighted mass rather than raw frequency alone.",
        "primary_family": "subgroup_structure",
        "secondary_family": "tail_rarity_structure",
        "evidence_url": "https://cloud.google.com/bigquery/docs/reference/standard-sql/approximate_aggregate_functions",
        "evidence_snippet": "BigQuery APPROX_TOP_SUM returns the top elements ordered by approximate weighted sum rather than count.",
        "single_table_portable": "yes",
        "portability_notes": "Portable whenever a groupable dimension and a numeric weight-like measure exist; exact SQL can fall back to SUM with GROUP BY and LIMIT.",
        "confidence": "high",
        "source_query_id": "bigquery_approx_top_sum",
        "template_id": "tpl_tail_weighted_topk_sum",
        "template_name": "Weighted Top-k Sum",
        "template_kind": "weighted_topk_sum",
        "status_hint": "ready",
        "template_notes": "Materialized as a canonical family rather than an engine-specific function variant, with BigQuery and ClickHouse as independent public evidence sources.",
        "materialization_bucket": "core",
        "template_flags": "",
        "additional_source_query_ids": "clickhouse_approx_top_sum",
    },
    {
        "mapping_id": "map_046",
        "workload_id": "bigquery_approx_aggregate_docs",
        "pattern_name": "grouped_percentile_point",
        "pattern_description": "Return a percentile point such as p95 or p99 for each subgroup rather than the raw tail rows themselves.",
        "primary_family": "tail_rarity_structure",
        "secondary_family": "subgroup_structure",
        "evidence_url": "https://cloud.google.com/bigquery/docs/reference/standard-sql/approximate_aggregate_functions",
        "evidence_snippet": "BigQuery APPROX_QUANTILES exposes direct percentile extraction from aggregated boundaries, and the same family appears across Trino, Snowflake, ClickHouse, Druid, and Pinot.",
        "single_table_portable": "yes",
        "portability_notes": "Portable on numeric-measure datasets; exact or approximate percentile syntax will vary by engine.",
        "confidence": "high",
        "source_query_id": "bigquery_approx_quantiles",
        "template_id": "tpl_grouped_percentile_point",
        "template_name": "Grouped Percentile Point",
        "template_kind": "grouped_percentile_point",
        "status_hint": "ready",
        "template_notes": "Canonical percentile-point family added so the library can represent p95/p99 style tail monitoring without returning full quantile slices.",
        "materialization_bucket": "core",
        "template_flags": "optional;dialect_sensitive",
        "additional_source_query_ids": "trino_approx_percentile;snowflake_percentile_cont;clickhouse_quantile;druid_approx_quantile_ds;pinot_percentile",
    },
    {
        "mapping_id": "map_047",
        "workload_id": "clickhouse_aggregate_docs",
        "pattern_name": "conditional_group_quantiles",
        "pattern_description": "Compute subgroup percentiles conditioned on a success/failure or other low-cardinality state.",
        "primary_family": "conditional_dependency_structure",
        "secondary_family": "tail_rarity_structure",
        "evidence_url": "https://clickhouse.com/docs/examples/aggregate-function-combinators/quantilesTimingIf",
        "evidence_snippet": "ClickHouse quantilesTimingIf computes response-time quantiles only for rows satisfying a condition such as successful requests.",
        "single_table_portable": "yes",
        "portability_notes": "Portable when a numeric measure, one subgroup dimension, and one low-cardinality condition column exist; engine syntax is more specialized than ordinary GROUP BY queries.",
        "confidence": "high",
        "source_query_id": "clickhouse_quantiles_timing_if",
        "template_id": "tpl_conditional_group_quantiles",
        "template_name": "Conditional Group Quantiles",
        "template_kind": "conditional_group_quantiles",
        "status_hint": "ready",
        "template_notes": "Kept optional because it is highly valuable for observability-style tails but more dialect-sensitive than the rest of the core library.",
        "materialization_bucket": "core",
        "template_flags": "optional;dialect_sensitive",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_048",
        "workload_id": "druid_sql_functions",
        "pattern_name": "threshold_rarity_cdf",
        "pattern_description": "Estimate how rare a threshold is inside a distribution by reporting the empirical CDF at that threshold.",
        "primary_family": "tail_rarity_structure",
        "secondary_family": "conditional_dependency_structure",
        "evidence_url": "https://druid.apache.org/docs/latest/querying/sql-functions/",
        "evidence_snippet": "Druid DS_RANK answers how much of the distribution lies below a chosen threshold, which directly encodes rarity at threshold T.",
        "single_table_portable": "yes",
        "portability_notes": "Portable whenever a numeric measure exists; exact SQL can use CASE/AVG or window CDF forms when sketch functions are unavailable.",
        "confidence": "high",
        "source_query_id": "druid_ds_rank",
        "template_id": "tpl_threshold_rarity_cdf",
        "template_name": "Threshold Rarity CDF",
        "template_kind": "threshold_rarity_cdf",
        "status_hint": "ready",
        "template_notes": "Added because it answers a different question from percentile-point queries: not 'what is p99?' but 'how rare is threshold T?'",
        "materialization_bucket": "core",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_049",
        "workload_id": "tpcds_altinity_queries",
        "pattern_name": "tail_drift_ratio",
        "pattern_description": "Compare current-period to prior-period subgroup counts and flag material tail drift.",
        "primary_family": "conditional_dependency_structure",
        "secondary_family": "tail_rarity_structure",
        "evidence_url": "https://github.com/Altinity/tpc-ds/blob/master/queries/query_75.sql",
        "evidence_snippet": "TPC-DS Q75 flags segments whose current sales count falls below 90% of the prior year baseline.",
        "single_table_portable": "partial",
        "portability_notes": "Excellent drift pattern, but it depends on a real temporal or period column and should stay outside the default core candidate pool.",
        "confidence": "high",
        "source_query_id": "tpcds_alt_q75",
        "template_id": "tpl_tail_drift_ratio",
        "template_name": "Tail Drift Ratio",
        "template_kind": "tail_drift_ratio",
        "status_hint": "ready",
        "template_notes": "Explicitly kept in the extension bucket because most current benchmark datasets lack real temporal semantics.",
        "materialization_bucket": "extension",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_050",
        "workload_id": "tpcds_altinity_queries",
        "pattern_name": "one_vs_rest_concentration_ratio",
        "pattern_description": "Measure how strongly one subgroup dominates relative to the rest of the mix for the same entity-period slice.",
        "primary_family": "conditional_dependency_structure",
        "secondary_family": "tail_rarity_structure",
        "evidence_url": "https://github.com/Altinity/tpc-ds/blob/master/queries/query_78.sql",
        "evidence_snippet": "TPC-DS Q78 computes a channel ratio against the rest of the channels after removing returns.",
        "single_table_portable": "partial",
        "portability_notes": "Very valuable, but current library already has within-group share coverage and this pattern depends more heavily on explicit one-vs-rest semantics.",
        "confidence": "high",
        "source_query_id": "tpcds_alt_q78",
        "template_id": "",
        "template_name": "",
        "template_kind": "",
        "status_hint": "",
        "template_notes": "Retained as prior-only evidence for a possible second-wave channel-vs-rest skew template.",
        "materialization_bucket": "prior_only",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_051",
        "workload_id": "tpch_qgen",
        "pattern_name": "high_value_inactive_segment",
        "pattern_description": "Identify rare entities whose value is above baseline while downstream activity is absent.",
        "primary_family": "tail_rarity_structure",
        "secondary_family": "conditional_dependency_structure",
        "evidence_url": "https://raw.githubusercontent.com/electrum/tpch-dbgen/master/queries/22.sql",
        "evidence_snippet": "TPC-H Q22 filters to above-average account balances with absent downstream activity.",
        "single_table_portable": "partial",
        "portability_notes": "Needs both value semantics and a reliable activity-absence signal in the same table; kept as prior-only until more datasets support that combination cleanly.",
        "confidence": "high",
        "source_query_id": "tpch_q22",
        "template_id": "",
        "template_name": "",
        "template_kind": "",
        "status_hint": "",
        "template_notes": "Strong paper evidence for rare dormant high-value segments, but still too semantically specific for the current core library.",
        "materialization_bucket": "prior_only",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
    {
        "mapping_id": "map_052",
        "workload_id": "trino_aggregate_docs",
        "pattern_name": "weighted_percentile",
        "pattern_description": "Locate percentile points after reweighting the distribution by support or exposure.",
        "primary_family": "tail_rarity_structure",
        "secondary_family": "conditional_dependency_structure",
        "evidence_url": "https://trino.io/docs/current/functions/aggregate.html",
        "evidence_snippet": "Trino weighted approx_percentile allows weights to shift where the effective tail sits inside the distribution.",
        "single_table_portable": "yes",
        "portability_notes": "Conceptually powerful, but current engines and datasets do not justify materializing it before the simpler weighted top-k family.",
        "confidence": "medium",
        "source_query_id": "trino_weighted_approx_percentile",
        "template_id": "",
        "template_name": "",
        "template_kind": "",
        "status_hint": "",
        "template_notes": "Kept as prior-only evidence for a future weighted-tail statistics wave.",
        "materialization_bucket": "prior_only",
        "template_flags": "",
        "additional_source_query_ids": "",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the restrained tail-template expansion to workload-grounding assets.")
    parser.add_argument("--catalog", default="data/workload_grounding/workload_catalog.csv")
    parser.add_argument("--source-bank", default="data/workload_grounding/source_query_bank_v1.jsonl")
    parser.add_argument("--mapping", default="data/workload_grounding/workload_to_family_mapping_v1.csv")
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def upsert_rows(rows: list[dict[str, Any]], key: str, upserts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {row[key]: row for row in rows}
    for item in upserts:
        index[item[key]] = item
    merged = list(index.values())
    merged.sort(key=lambda row: row[key])
    return merged


def ensure_mapping_fields(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        item = {field: row.get(field, "") for field in MAPPING_FIELDNAMES}
        normalized.append(item)
    return normalized


def main() -> None:
    args = parse_args()
    catalog_path = Path(args.catalog)
    source_bank_path = Path(args.source_bank)
    mapping_path = Path(args.mapping)

    catalog_rows = load_csv_rows(catalog_path)
    catalog_fieldnames = list(catalog_rows[0].keys())
    catalog_rows = upsert_rows(catalog_rows, "workload_id", WORKLOAD_UPSERTS)
    write_csv_rows(catalog_path, catalog_rows, catalog_fieldnames)

    source_rows = load_jsonl_rows(source_bank_path)
    source_rows = upsert_rows(source_rows, "source_query_id", SOURCE_UPSERTS)
    write_jsonl_rows(source_bank_path, source_rows)

    mapping_rows = ensure_mapping_fields(load_csv_rows(mapping_path))
    mapping_rows = upsert_rows(mapping_rows, "mapping_id", MAPPING_UPSERTS)
    write_csv_rows(mapping_path, mapping_rows, MAPPING_FIELDNAMES)

    print(
        json.dumps(
            {
                "catalog_path": str(catalog_path.resolve()),
                "catalog_row_count": len(catalog_rows),
                "source_bank_path": str(source_bank_path.resolve()),
                "source_bank_row_count": len(source_rows),
                "mapping_path": str(mapping_path.resolve()),
                "mapping_row_count": len(mapping_rows),
                "new_mapping_ids": [row["mapping_id"] for row in MAPPING_UPSERTS if row["mapping_id"] != "map_015"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
