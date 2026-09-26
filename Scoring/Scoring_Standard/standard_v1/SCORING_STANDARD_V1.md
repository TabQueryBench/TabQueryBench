# TabQueryBench Semantic Scoring Standard v1

This is the normative scoring standard for real-versus-synthetic SQL result
comparison. It supersedes the fixed legacy composite as the semantic scoring
definition. It does not alter template selection, parameter binding, SQL
generation, or query execution.

## Scope and result contract

The scorer evaluates the analytical answer, after both SQL statements execute.
`semantic_result_contract` on the template is authoritative for `scorer_type`
and `validity_required_outputs`; a top-level `scorer_type` is only a backwards
compatible fallback. Required output aliases are compared case-insensitively.
Failure of either execution, absence of an output column, or absence of a
required output produces `semantic_query_score = 0.0` and an `invalid`
validity status. Schema/execution checks are gates, rather than weighted
fidelity components.

Score calibration is intentionally pending. The scorer currently reports its
raw weighted semantic composite; no separate normalized score field or
dataset-level min/max scaling is emitted.

## Semantic families

Families classify the analytical phenomenon to be covered. They are not a
formula selector: templates declare one of the answer/scorer types below.

| Family | Analytical target | Typical answer forms |
| --- | --- | --- |
| `subgroup_structure` | subgroup prevalence, composition, and group-level summaries | count profiles, grouped aggregates, ranked groups |
| `conditional_dependency_structure` | conditional rates, contrasts, and associations | keyed rate/share/ratio profiles |
| `tail_rarity_structure` | rare mass, extreme values, thresholded or ranked tails | top-k selections, tail distributions, numeric aggregates |
| `missingness_structure` | marginal and conditional missing-data mechanisms | scalar or keyed missing rates/support profiles |
| `cardinality_structure` | support breadth, concentration, and distinct-value structure | distribution/cardinality profiles and range summaries |

## Answer/scorer types

The standard has exactly eight core scorer types. A temporal result is a
deterministic-bucket specialization of count or keyed-numeric scoring; it is
not a ninth type.

| Type | Primary score components | Weight rule |
| --- | --- | --- |
| `scalar` | symmetric numeric similarity, or rate similarity for `[0,1]` values | 100% direct similarity |
| `count_support_distribution` | support-distribution similarity, total-mass similarity, key F1 | 50 / 25 / 25 |
| `rate_share_proportion` | rate similarity, direction consistency, key F1 | 50 / 25 / 25 |
| `ratio` | symmetric numeric (or log-ratio) similarity, direction/threshold consistency, key F1 | 50 / 25 / 25 |
| `keyed_numeric_aggregate` | aligned numeric-measure similarity, key F1, support/evidence similarity | 50 / 25 / 25; 50 / 50 without support |
| `topk_tailk_ranking` | selected-set overlap, rank similarity, aligned measure similarity | 50 / 25 / 25; winner-only is 50 / 50 |
| `distribution_cardinality_profile` | distribution similarity, rank-profile similarity, cardinality similarity | 50 / 25 / 25 |
| `tail_outlier_distribution` | tail-distribution similarity, tail-boundary similarity, tail severity/direction similarity | 50 / 25 / 25 |

Template aliases (for example `keyed_count_distribution`,
`topk_ranked_measure`, `missingness_group_rate`, and `temporal_count_curve`)
resolve to one of these eight types. Aliases remain accepted to preserve
released workloads.

## Reusable primitives

- **Key F1:** harmonic mean of exact-key precision and recall.
- **Symmetric numeric similarity:** `1 - |r-s|/(|r|+|s|+eps)`; two zeroes
  score one.
- **Rate similarity:** `1 - MAE` for values naturally bounded in `[0,1]`.
- **Distribution similarity:** `1 - sqrt(JSD_base2(P_real, P_syn))` over the
  union of support keys.
- **Total mass similarity:** symmetric numeric similarity of total support.
- **Selected-set overlap:** key-set F1 (fixed-k implementations may use
  overlap-at-k).
- **Rank similarity:** an NDCG-style score, with high real ranks weighted more.
- **Direction consistency:** agreement of above/below-baseline directions on
  aligned keys.

For deterministic buckets, missing buckets are aligned as zero-support buckets
when the template contract calls for them. Raw high-cardinality identities and
raw tail rows must not earn fidelity through exact-row copying; their
distributional scorer is used instead.

## Output schema and compatibility

Each query row must retain `legacy_score` (and the older
`legacy_query_score` alias), and add:

```json
{
  "semantic_query_score": 0.0,
  "component_scores": {"key_f1": 0.0},
  "score_version": "semantic_scoring_standard_v1"
}
```

`query_score` continues to follow the selected primary-score mode, so legacy
runs are reproducible while semantic mode uses the raw semantic composite.

Aggregation is macro-first: queries to templates, templates to subitems,
subitems to families, then the five family scores to the overall score. Report
the query-weighted micro average only as a diagnostic.
