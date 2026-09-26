# Query Scoring Redesign Outline

This document is a design outline for revising TabQueryBench query scoring. It is intentionally not an implementation patch. The template set is expected to change, so the goal here is to define a stable scoring architecture and metadata contract before changing scorer code.

## Current Scoring Location

The current query score used by the SQL evaluation pipeline is computed in the evaluation codebase, not inside the public `Query_Templates` release folder.

Primary scoring entry points:

- `Scoring/code/tqb_scoring/evaluation/real_panel_experiment.py`
  - `_compare_execution_results(...)` computes the current per-query score.
  - Current formula:
    - `0.45 * strict_set_score`
    - `0.20 * key_set_score`
    - `0.15 * profile_score`
    - `0.10 * row_count_score`
    - `0.10 * column_score`
- `Scoring/code/tqb_scoring/eval/analysis/runner.py`
  - Calls `_compare_execution_results(...)` for each query.
  - Writes query-level rows and averages query scores into asset/template/family summaries.
- `Query/code/tqb_query/analytics_contract.py`
  - Maps query rows to subitems/families and aggregates subitem/family rows.

Current issue:

- The same fixed weighted formula is applied to every query shape.
- `column_score` and `row_count_score` are mixed into fidelity even though they are mostly validity or diagnostic signals.
- `key_set_score` and `profile_score` can be high when aggregate values are wrong.
- Current `overall_score` is mostly a micro-average over query instances, so templates with many generated queries dominate the benchmark.

## Design Goals

1. Score the analytical answer represented by each SQL query, not the surface shape of the output table.
2. Use query-shape-specific scorers instead of one fixed formula for all templates.
3. Keep weights simple and explainable:
   - `100%`
   - `50 / 50`
   - `50 / 25 / 25`
   - `1/3 / 1/3 / 1/3`
4. Avoid arbitrary-looking weights such as `0.55 / 0.20 / 0.15 / 0.10`.
5. Keep diagnostics separate from the primary query score.
6. Avoid rewarding memorization of raw real rows.
7. Support future template changes by relying on semantic template metadata rather than fragile output-column guessing.

## Proposed Architecture

Use a two-layer scoring path:

```text
SQL execution results
  -> validity gate
  -> semantic scorer selected by template metadata
  -> query_score
  -> query diagnostics
  -> template macro average
  -> subitem/family macro average
  -> overall macro average
```

### Validity Gate

Validity checks should happen before semantic scoring. If a required condition fails, the query score should be zero or marked invalid depending on policy.

Validity checks:

- Real SQL executes successfully.
- Synthetic SQL executes successfully.
- Required output columns are present.
- Required output columns can be parsed into expected types.
- Required measure columns contain usable values.
- Real result is evaluable for the scorer type.

Signals that should move from primary score to gate/diagnostics:

- `column_score`
- SQL execution success
- required schema presence
- type parse success

`row_count_score` should usually be a diagnostic or a semantic component only when output size is part of the query answer, such as top-k size, selected set size, or filtered population mass.

## Template Metadata Contract

The evaluator should not infer semantics only from output column names. Each template should eventually carry scorer metadata.

Recommended fields:

```json
{
  "template_id": "...",
  "scorer_type": "...",
  "semantic_key_roles": ["group_col"],
  "measure_roles": ["support"],
  "support_role": "support",
  "rate_roles": ["target_rate"],
  "rank_by": "support DESC",
  "selection_mode": "topk",
  "key_match_mode": "categorical_exact",
  "primary_metric": "...",
  "diagnostic_metrics": ["..."]
}
```

Recommended `key_match_mode` values:

- `categorical_exact`
  - Exact key matching is meaningful.
- `deterministic_bucket`
  - Fixed buckets or time buckets; missing buckets should be aligned and filled with zero.
- `high_cardinality_distribution`
  - Exact IDs should not be rewarded; compare cardinality, support profile, and response distribution.
- `raw_numeric_distribution`
  - Raw values should be compared by distribution, not exact identity.
- `none`
  - Scalar or pure distribution query with no semantic key.

## Primitive Metrics

These primitives should be implemented once and reused by scorer types.

### Key F1

Use for meaningful categorical or bucket keys.

```text
precision = |K_real intersect K_syn| / |K_syn|
recall    = |K_real intersect K_syn| / |K_real|
key_f1    = harmonic_mean(precision, recall)
```

Use F1 instead of one-sided coverage because it penalizes both missing real groups and extra synthetic groups.

### Symmetric Numeric Similarity

Use for numeric values that are not bounded rates.

```text
S_num(r, s) = 1 - |r - s| / (|r| + |s| + eps)
```

Special case:

```text
r = 0 and s = 0 -> 1
```

### Rate Similarity

Use for values naturally in `[0, 1]`, such as rates, shares, proportions, CDF values, and missing rates.

```text
S_rate = 1 - mean_absolute_error(real_rates, synthetic_rates)
```

### Distribution Similarity

Use for support/count/share profiles.

Recommended:

```text
S_dist = 1 - sqrt(JSD(real_distribution, synthetic_distribution))
```

Use base-2 JSD so the distance is naturally bounded.

### Total Mass Similarity

Use when total filtered population or total support is part of the answer.

```text
S_mass = 1 - |N_real - N_syn| / (N_real + N_syn + eps)
```

### Selected Set Overlap

Use for top-k, tail-k, threshold selection, and winner queries.

Use set F1 or Jaccard when result size can vary. Use overlap at k when k is fixed.

### Rank Similarity

Use NDCG-like scoring for top-k/ranked results because high-ranked items should matter more than low-ranked items.

Spearman/Kendall can be kept as diagnostics, but they should not be the default primary metric for top-k queries.

### Direction Consistency

Use when the query asks whether a subgroup is above/below baseline or whether a trend increases/decreases.

Example:

```text
real_direction = sign(real_group_value - real_baseline)
syn_direction  = sign(syn_group_value - syn_baseline)
```

Then compute support-weighted agreement over aligned keys.

## Scorer Types

The scorer types below are intentionally few and use simple weight rules. Missingness, temporal, and cardinality can be represented as special cases of these core scorer types unless the template redesign requires a separate class.

### 1. Scalar

Use when the SQL answer is one scalar or one scalar vector with no semantic key.

Examples:

- Filtered median numeric slice
- Filtered sum in numeric band
- Marginal missing rate profile
- Threshold rarity CDF

Rules:

```text
scalar numeric: score = 100% symmetric_numeric_similarity
scalar rate:    score = 100% rate_similarity
```

Rationale:

- If a query answers one number, compare that number directly.
- Do not include key, row, or column scores in the primary score.

### 2. Count / Support Distribution

Use when the answer is a count/support profile over groups.

Examples:

- Grouped count by category
- Filtered two-dimensional group count
- Time-bucket filtered count
- Low-support group count when the emphasis is support profile
- Pairwise sparse slice count when the emphasis is low-support group support

Rule:

```text
score =
  50% support_distribution_similarity
+ 25% total_mass_similarity
+ 25% key_F1
```

Rationale:

- The distribution is the primary answer.
- Total population size matters because distribution normalization can hide scale errors.
- Key F1 checks whether the same groups are represented.

### 3. Rate / Share / Proportion

Use when the answer is a bounded rate/share/proportion over keys.

Examples:

- Two-axis target rate surface
- Grouped condition rate
- Within-group share of total
- Missing rate by subgroup
- Missingness-target interaction

Rule:

```text
score =
  50% rate_similarity
+ 25% direction_consistency
+ 25% key_F1
```

Rationale:

- Rate values are the primary answer.
- Direction matters for dependency and interaction claims.
- Key F1 guards against missing or extra groups.

### 4. Ratio

Use when the output measure is an unbounded or positive ratio rather than a bounded rate.

Examples:

- Grouped ratio of two conditions
- Tail drift ratio

Rule:

```text
score =
  50% ratio_similarity
+ 25% direction_or_threshold_consistency
+ 25% key_F1
```

Rationale:

- Ratios may exceed 1, so use symmetric numeric or log-ratio similarity, not rate MAE.
- Direction/threshold consistency captures claims like increase/decrease or above/below threshold.

### 5. Keyed Numeric Aggregate

Use when the answer is one or more numeric aggregate measures aligned by group keys.

Examples:

- Grouped numeric mean
- Grouped numeric sum
- Support-guarded group average
- Two-dimensional group average
- Conditional group quantiles
- Grouped percentile point
- Two-dimensional summary with filter
- High-cardinality response stability, if exact high-cardinality identity is not the target

Default rule:

```text
score =
  50% numeric_measure_similarity
+ 25% key_F1
+ 25% support_or_evidence_similarity
```

If there is no support/evidence measure:

```text
score =
  50% numeric_measure_similarity
+ 50% key_F1
```

Rationale:

- Numeric aggregate values are the main answer.
- Keys define alignment.
- Support/evidence should matter when the aggregate is only meaningful above sufficient support.

### 6. Top-k / Tail-k / Ranking

Use when the query selects, ranks, or thresholds entities/groups.

Examples:

- Filtered top-k group count
- Top-k groups by distinct entity coverage
- Grouped summary top-k
- Two-dimensional top-k count
- Tail target-rate extremes
- Thresholded group ranking
- Max aggregate winner selection

Default rule:

```text
score =
  50% selected_set_overlap
+ 25% rank_similarity
+ 25% measure_similarity
```

Winner-only rule:

```text
score =
  50% winner_set_match
+ 50% winning_value_similarity
```

Rationale:

- For ranking queries, the selected items are the primary answer.
- Ranking and values are supporting evidence.
- Exact full-row match is too strict and often semantically wrong.

### 7. Distribution / Cardinality Profile

Use when the query asks for cardinality, support concentration, or rank profile of a distribution.

Examples:

- Cardinality distinct share profile
- Cardinality support rank profile

Rule:

```text
score =
  50% distribution_similarity
+ 25% rank_profile_similarity
+ 25% cardinality_similarity
```

Rationale:

- The distribution is the main signal.
- Rank profile captures heavy-tail or concentration structure.
- Distinct cardinality checks whether support breadth is preserved.

### 8. Tail / Outlier Distribution

Use when the query exposes raw tail values or outlier rows where exact identity matching would reward memorization.

Examples:

- Top-N within group by measure
- Global z-score outlier scan
- Quantile tail slice

Rule:

```text
score =
  50% tail_distribution_similarity
+ 25% tail_boundary_similarity
+ 25% tail_severity_or_direction_similarity
```

Rationale:

- Synthetic data should preserve tail distribution, not copy exact rows.
- Boundary and severity capture whether extremes remain comparably extreme.
- Exact raw-row overlap should be a privacy-risk diagnostic, not a fidelity reward.

### 9. Temporal Shape Specialization

Temporal queries can use the closest scorer above plus a temporal-shape component.

Examples:

- Time-bucket filtered count
- Time-bucket group moving average
- Tail drift ratio

Rule:

```text
score =
  50% pointwise_value_similarity
+ 25% temporal_shape_similarity
+ 25% total_mass_or_direction_similarity
```

Rationale:

- Time-bucket alignment should fill missing buckets with zero when appropriate.
- The curve/trend matters, not just unordered key overlap.
- Moving averages and raw counts are related, so avoid double-counting the same signal with equal independent weights.

## Template-to-Scorer Mapping Strategy

Do not hard-code only by family. Use explicit template metadata.

Interim mapping can be stored in a new file such as:

```text
Query/Query_Templates/templates/policy/template_scoring_policy_v1.jsonl
```

Candidate fields:

```json
{
  "template_id": "tpl_clickbench_group_count",
  "scorer_type": "count_support_distribution",
  "weight_rule": "50_25_25",
  "primary_metrics": ["support_distribution_similarity"],
  "secondary_metrics": ["total_mass_similarity", "key_f1"],
  "semantic_key_roles": ["group_col"],
  "measure_roles": ["row_count"],
  "key_match_mode": "categorical_exact"
}
```

This policy file should be versioned independently from template SQL text, because template fixes and scoring fixes may move on different timelines.

## Aggregation Redesign

Current query counts differ widely across templates, so micro-averaging query rows can over-weight templates that produce more queries.

Recommended primary aggregation:

```text
query_score
  -> template_score = mean(query_score within template)
  -> subitem_score = mean(template_score within subitem)
  -> family_score = mean(subitem_score or template_score within family)
  -> overall_score = mean(family_score)
```

Recommended reporting:

- Primary: macro overall score.
- Secondary diagnostic: micro query-weighted score.
- Also report query count, valid query count, and invalid query rate.

This makes the benchmark less sensitive to accidental template/query-count imbalance.

## Output Contract

Each scored query row should include:

```json
{
  "query_score": 0.0,
  "query_score_method": "semantic_scorer_v1",
  "scorer_type": "count_support_distribution",
  "validity_status": "ok",
  "validity_failures": [],
  "primary_metric": "support_distribution_similarity",
  "component_scores": {
    "support_distribution_similarity": 0.0,
    "total_mass_similarity": 0.0,
    "key_f1": 0.0
  },
  "weight_rule": "50_25_25",
  "diagnostics": {
    "strict_set_score": 0.0,
    "old_composite_score": 0.0,
    "row_count_score": 0.0,
    "column_score": 0.0
  }
}
```

Keep old scores during migration:

- `legacy_query_score`
- `legacy_query_score_method`
- `strict_set_score`
- `semantic_query_score`
- `semantic_query_score_method`

## Implementation Plan

### Phase 0: No Behavior Change

- Add this design document.
- Audit template inventory and identify obvious degenerate templates.
- Decide scorer metadata fields.

### Phase 1: Metadata Only

- Add `template_scoring_policy_v1.jsonl`.
- Map each current template to a scorer type and semantic roles.
- Do not change scoring output yet.

### Phase 2: Parallel Semantic Scoring

- Implement semantic scorers in a new module, for example:

```text
Scoring/Scoring_Standard/standard_v1/scorer.py
```

- Keep `_compare_execution_results(...)` for legacy scoring.
- Add a parallel call that computes `semantic_query_score`.
- Write both legacy and semantic scores to query rows.

### Phase 3: Validation Run

- Run on a small set of datasets.
- Compare:
  - legacy composite score
  - strict-only score
  - semantic score
  - per-scorer component distributions
- Inspect examples where scores diverge strongly.

### Phase 4: Switch Primary Score

- After validation, make semantic score the primary `query_score`.
- Keep legacy score as a diagnostic for at least one release.
- Update analysis exports, README text, and paper tables.

## Open Questions

1. Should high-cardinality IDs ever use exact key matching?
2. Should macro aggregation average templates equally or subitems equally first?
3. How should invalid real queries be handled: removed from denominator or scored zero?
4. Should privacy-risk exact row overlap be reported for raw-tail queries?
5. How should scorer metadata be synchronized between SQLagent internals and the public `Query_Templates` release?

## Immediate Recommendation

Do not modify the scoring formula yet. First add template-level scoring metadata and fix obviously degenerate templates. Then implement semantic scoring in parallel with legacy scoring so the score change can be audited before becoming the public benchmark score.
