# Semantic Query Scoring V8

This document summarizes the semantic scoring contract used for the current
TabQueryBench run3 query analysis files.

Current implementation fields:

- `semantic_query_score_method`: `semantic_scorer_v1_parallel`
- `score_contract_version`: `real_vs_synthetic_sql_semantic_v1_parallel`
- `legacy_query_score`: the original composite score
- `semantic_query_score`: the template-specific semantic score
- `primary_query_score_mode`: `legacy_composite` or `semantic_template`

The evaluator keeps both scoring systems side by side. By default, the public
`query_score` and aggregate `overall_score` fields use the legacy composite
score for backward compatibility. Passing the boolean CLI flag
`--use-semantic-query-score` makes `query_score`, `overall_score`, template
scores, subitem scores, and family scores use `semantic_query_score` instead.
The non-primary score remains available in its dedicated column.

## Scoring Flow

For each SQL query, the evaluator selects a semantic scorer from the template
metadata and SQL result shape. Each scorer computes one or more component scores
in `[0, 1]`, then aggregates them with a simple rule:

- `100`: one component only
- `50_50`: `0.5 * a + 0.5 * b`
- `50_25_25`: `0.5 * a + 0.25 * b + 0.25 * c`
- `one_third_each`: `(a + b + c) / 3`

The primary score is clipped to `[0, 1]`.

## Scorer Types

### `count_support_distribution`

Used for grouped-count support profiles.

Default aggregation:

```text
0.5 * support_distribution_similarity
+ 0.25 * total_mass_similarity
+ 0.25 * key_f1
```

### `distribution_cardinality_profile`

Used for support-rank and distinct-share cardinality profiles.

Default aggregation:

```text
0.5 * distribution_similarity
+ 0.25 * rank_profile_similarity
+ 0.25 * cardinality_similarity
```

### `keyed_numeric_aggregate`

Used for keyed aggregates such as grouped averages, sums, dispersions, and
percentile points.

Default aggregation:

```text
0.5 * numeric_measure_similarity
+ 0.25 * key_f1
+ 0.25 * support_or_evidence_similarity
```

When no support or evidence column is present, the evaluator records the
available components and applies the implemented fallback for that query shape.

### `rate_share_proportion`

Used for rates, shares, proportions, missingness rates, and within-group shares.

Default aggregation:

```text
0.5 * rate_similarity
+ 0.25 * direction_consistency
+ 0.25 * key_f1
```

### `ratio`

Used for grouped ratios of two conditions.

Default aggregation:

```text
0.5 * ratio_similarity
+ 0.25 * direction_or_threshold_consistency
+ 0.25 * key_f1
```

### `scalar`

Used for scalar or scalar-vector answers with no semantic key, such as marginal
missingness rates and rarity CDFs.

Default aggregation:

```text
scalar_similarity
```

### `tail_outlier_distribution`

Used for quantile tail slices and tail boundary checks.

Default aggregation:

```text
0.5 * tail_distribution_similarity
+ 0.25 * tail_boundary_similarity
+ 0.25 * tail_severity_or_direction_similarity
```

### `topk_tailk_ranking`

Used for top-k, tail-k, threshold selection, and ranked group queries.

Default aggregation:

```text
0.5 * selected_set_overlap
+ 0.25 * rank_similarity
+ 0.25 * measure_similarity
```

## Primitive Components

- `key_f1`: F1 overlap between real and synthetic semantic keys.
- `*_distribution_similarity`: distribution similarity, typically based on
  Jensen-Shannon distance or an equivalent bounded distribution comparison.
- `*_measure_similarity`: symmetric numeric similarity over aligned values.
- `*_rate_similarity`: bounded rate/proportion similarity.
- `*_mass_similarity`: similarity of total support or selected population mass.
- `*_rank_similarity`: rank-profile agreement for ordered result sets.
- `*_direction_consistency`: agreement on above/below-baseline or threshold
  direction.
- `*_cardinality_similarity`: agreement on the number of distinct semantic keys.

## Public Outputs

Run-level semantic scoring outputs are published under:

```text
Synthesizing/synthetic_data/run3/query_analysis/
```

The main query-level file is:

```text
analysis_query_scores__all_datasets.jsonl
```

Template, subitem, family, and asset-level summaries are published in the same
directory.

## Relationship To Templates

The v8 template release adds semantic metadata used by these scorers, including
result contracts, scorer hints, output roles, and role-distinctness constraints.
The merged v8 template library is available at:

```text
Query/code/data/workload_grounding_v8/template_library_v8.jsonl
```

The same 49 templates are split by family under:

```text
Query/Query_Templates/templates/core/*/*_v8.jsonl
```
