# Semantic Query Scoring V9: SPQ

Full name: **TabQueryBench Single-Primary Query Similarity v9**.

This scoring contract replaces the v8 multi-component weighted semantic score
with a single-primary-target rule:

```text
query_score = primary_semantic_similarity
```

Diagnostics such as key overlap, total support mass, direction consistency, and
measure-value agreement may still be emitted, but they do not affect the
primary semantic score.

Implementation fields:

- `semantic_query_score_method`: `spq`
- `score_contract_version`: `spq_v9_single_primary`
- `primary_query_score_mode`: `legacy_composite` or `semantic_template`
- `semantic_query_score`: the v9 single-primary semantic score
- `legacy_query_score`: the original composite score retained for comparison

The evaluator continues to keep both scoring systems side by side. Passing
`--use-semantic-query-score` makes `semantic_query_score` the primary
`query_score`/`overall_score`; otherwise the legacy composite remains primary.

## Active Scorer Types

V9 uses exactly six active scorer types.

### `scalar`

Used when the SQL answer is one primary number or a small scalar vector with no
semantic key.

Numeric scalar similarity:

```text
1 - abs(real - synthetic) / max(abs(real), abs(synthetic), eps)
```

Rate/probability scalar similarity:

```text
1 - abs(real_rate - synthetic_rate)
```

### `count_support_distribution`

Used for grouped count/support distributions. Counts are normalized over the
union key space, then scored with total variation distance:

```text
TVD = 0.5 * sum_k abs(P_real(k) - P_synthetic(k))
score = 1 - TVD
```

`key_f1` and total-mass similarity are diagnostics only.

### `rate_share_proportion`

Used for keyed rates, shares, proportions, probabilities, and missingness rates.
The primary score is a macro average over the union key space:

```text
per_key_score = 1 - abs(real_rate - synthetic_rate)
missing_or_extra_key_score = 0
score = mean(per_key_score over union keys)
```

### `ratio`

Used for positive ratios and relative comparisons. For each value:

```text
score = min(abs(real), abs(synthetic)) / max(abs(real), abs(synthetic))
```

with `0` vs `0` scoring `1`, and `0` vs positive scoring `0`. Keyed ratio
queries macro-average this value over the union key space.

### `keyed_numeric_aggregate`

Used for grouped averages, sums, medians, percentiles, dispersion statistics,
and other keyed numeric aggregates. The primary score is a macro average of the
symmetric numeric similarity over the union key space. Missing or extra keys
score `0`.

### `topk_ranking`

Used for top-k, tail-k, ranked selections, and argmax/winner-style answers. The
primary score is rank-biased overlap (RBO) with:

```text
RBO_PERSISTENCE = 0.9
```

Magnitudes attached to ranked rows are emitted only as diagnostics and do not
affect the ranking score.

## V8 Compatibility Mapping

Older template scorer hints are routed into the six v9 scorer types:

- `distribution_cardinality_profile` -> `count_support_distribution`
- `topk_tailk_ranking`, `topk_ranked_measure`, `argmax_selection`,
  `tail_topn_value_curve` -> `topk_ranking`
- `tail_outlier_distribution` is no longer an independent scorer; tail queries
  are routed by answer shape to scalar, support distribution, keyed numeric
  aggregate, or top-k ranking.
- temporal-shaped answers are routed by result shape to rate/share,
  support-distribution, or keyed numeric aggregate scoring.

## Aggregation

Run-level reporting should prefer scorer-type macro summaries when comparing
semantic categories, because v9 intentionally asks each query one primary
semantic question. Query-count-weighted averages may still be reported as a
secondary operational summary.
