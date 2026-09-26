# Semantic Scoring V2 (SV2) — Implementation Contract

This document is the complete specification of the TabQueryBench semantic query
scorer SV2. Implementation notes for this repository are in [`README.md`](README.md).

**Core rule.** Each query has one primary semantic scorer. The final query score
is exactly that scorer's similarity value:

```text
query_score = primary_semantic_similarity
```

SV2 removes the old distinction between scalar / ratio / keyed numeric aggregate by
merging them into one Numeric Magnitude family.

## 1. Design goals

1. **Single primary metric per query.** No weighted 50/25/25-style composition and no
   mixing of auxiliary diagnostics into the primary score.
2. **Established metrics.** Normalized sMAPE for numeric magnitude, TVD for count/support
   distributions, MAE for rates/shares/proportions, RBO for ranked answers.
3. **Uniform score direction and range.** `score ∈ [0, 1]`; `1.0` = perfect semantic
   match; `0.0` = maximal disagreement under that scorer.
4. **Answer semantics determine the scorer.** Never choose a scorer from the observed
   numeric range.
5. **Diagnostics are diagnostics only.** Key overlap, raw errors, total mass, support size,
   etc. may be emitted but never change the primary score.

## 2. Active scorer types

| ID | Scorer type | Typical answers | Primary metric | Final similarity |
| --- | --- | --- | --- | --- |
| 1 | `numeric_magnitude` | scalar numeric values, ratios, AVG/SUM/median/percentile/std, keyed numeric aggregates | normalized sMAPE | `1 - sMAPE / 2` |
| 2 | `count_support_distribution` | grouped counts, frequencies, cardinality/support distributions | TVD | `1 - TVD` |
| 3 | `rate_share_proportion` | rates, percentages, probabilities, missingness rates, shares | MAE | `1 - MAE` |
| 4 | `topk_ranking` | top-k, bottom-k, ranked selections, winner/argmax | RBO | `RBO(p=0.9)` |

All four scorers return a value in `[0, 1]`.

## 3. Global scoring contract

```text
semantic_query_score = primary_semantic_similarity
query_score          = semantic_query_score
overall_score        = semantic_query_score   # if the pipeline uses query-level overall_score
```

Recommended metadata:

```json
{
  "semantic_query_score_method": "sv2",
  "score_contract_version": "sv2_four_primary_v1",
  "scorer_type": "numeric_magnitude",
  "metric": "normalized_smape_similarity",
  "semantic_query_score": 0.8888888889,
  "query_score": 0.8888888889,
  "valid": true
}
```

Legacy scores may be retained in separate fields (e.g. `legacy_query_score`). Never blend them.

## 4. Scorer routing

### 4.1 Routing is semantic, not value-based

Priority:

1. explicit `scorer_type` in query/template metadata;
2. explicit semantic answer type / scorer hint;
3. compatibility mapping from old scorer names;
4. otherwise fail routing explicitly.

Never route with rules such as `if 0 <= value <= 1: rate_share_proportion` — a ratio,
mean, coefficient, or normalized aggregate can also lie in `[0, 1]`.

### 4.2 Canonical routing examples

- **`numeric_magnitude`**: scalar AVG / SUM / median / percentile / standard deviation /
  variance-like output, numeric ratio, relative comparison, keyed AVG / SUM / median /
  percentile / dispersion, fixed-length scalar vector without semantic keys.
- **`count_support_distribution`**: the answer is a distribution of count/support mass over
  keys (e.g. `SELECT category, COUNT(*) FROM t GROUP BY category`); the question is whether
  the distribution of support mass across keys is preserved.
- **`rate_share_proportion`**: bounded rate-like quantities with a fixed semantic scale —
  probability, rate, percentage, share, fraction, missingness rate, prevalence,
  conditional proportion; canonical internal range `[0, 1]`. A scalar rate is still a rate.
- **`topk_ranking`**: ordering/selection identity is the semantic target — top-k, bottom-k,
  tail-k, ranked categories, argmax/winner, argmin/loser. Attached magnitudes are
  diagnostics only.

## 5. Canonical answer representation

- **Scalar numeric**: `real = 100.0`, `syn = 80.0`; internally one implicit key `{"__scalar__": value}`.
- **Unkeyed numeric vector**: positions are implicit keys `__pos_0__, __pos_1__, …`. Do not sort values.
- **Keyed numeric/rate answer**: `{("A",): 100.0, ("B",): 50.0}`. Compound keys stay tuples or
  another collision-safe structure; never unsafe string concatenation.
- **Count distribution**: `{("A",): 50, ("B",): 30, ("C",): 20}`.
- **Ranking**: ordered identities, e.g. `["A", "B", "C"]` or compound `[("US", "A"), ("CA", "C")]`.

## 6. Shared key-alignment rules

Keyed answers use the union key space `K = keys(real) ∪ keys(syn)`; never only the intersection.

### 6.1 Missing/extra keys

For `numeric_magnitude` and `rate_share_proportion`: a key present on both sides gets the
normal per-key similarity; a key present on one side gets per-key similarity `0`.
A missing key is not the value zero (`key exists with value 0 != key absent`).

### 6.2 Duplicate keys

- `count_support_distribution`: duplicate keys are summed (`A→10, A→20` becomes `A→30`).
- `numeric_magnitude`, `rate_share_proportion`: duplicate semantic keys are invalid unless an
  upstream query-specific rule resolves them; never average silently.
- `topk_ranking`: ranking identities must be unique; duplicates are invalid.

## 7. Scorer 1 — Numeric Magnitude

Purpose: one or more numeric magnitudes whose meaningful error is relative to scale.
Replaces old `scalar`, `ratio`, and `keyed_numeric_aggregate` for non-rate numeric semantics.

**Metric.** For one pair, `sMAPE(r, s) = 2|r − s| / (|r| + |s|)`, and

```text
S_num(r, s) = 1 − sMAPE(r, s) / 2 = 1 − |r − s| / (|r| + |s|)
```

which lies in `[0, 1]` for all finite real `r, s` with a nonzero denominator.

**Zero handling** (never add epsilon to the denominator):

```text
r = 0, s = 0    -> 1
r = 0, s != 0   -> 0
r != 0, s = 0   -> 0
```

A tiny tolerance (`ZERO_TOL = 1e-12`) may be used only to classify floating-point zeros.

**Scalar:** `S = S_num(r, s)`; e.g. `real = 100, syn = 80 → 1 − 20/180 = 0.888888…`.

**Keyed:** `S_k = S_num(r_k, s_k)` for matched keys, `S_k = 0` for missing/extra keys, and
`S = (1/|K|) Σ_{k∈K} S_k` (macro mean, equal weight per key; not support-weighted).

**Signed values** are allowed: `real = −10, syn = −8 → 0.888888…`; `real = 1, syn = −1 → 0`.

```python
def normalized_smape_similarity(r, s, zero_tol=1e-12):
    if not is_finite_number(r) or not is_finite_number(s):
        raise InvalidScoreInput
    r_zero, s_zero = abs(r) <= zero_tol, abs(s) <= zero_tol
    if r_zero and s_zero:
        return 1.0
    if r_zero != s_zero:
        return 0.0
    return checked_unit_interval(1.0 - abs(r - s) / (abs(r) + abs(s)))
```

## 8. Scorer 2 — Count / Support Distribution

Purpose: the shape of count/support mass over semantic keys, not the absolute total count.

**Normalize:** `P_r(k) = c_r(k) / Σ_j c_r(j)`, `P_s(k) = c_s(k) / Σ_j c_s(j)` over `K = K_r ∪ K_s`,
absent keys have probability 0.

**Metric:** `D_TV = ½ Σ_{k∈K} |P_r(k) − P_s(k)|`, `S = 1 − D_TV`.

Example: real `A 50, B 30, C 20`, synthetic `A 40, B 40, C 20` → `TVD = 0.1`, `S = 0.9`.

**Counts** must be finite, non-negative, and numeric; reject negative, NaN, ±Inf, SQL NULL,
and non-numeric values. Floating weights are allowed when the query returns weighted support.

**Zero totals:**

```text
T_real = 0 and T_syn = 0  -> 1
T_real = 0 and T_syn > 0  -> 0
T_real > 0 and T_syn = 0  -> 0
otherwise                 -> 1 − TVD
```

**Total count is not part of the score:** `A=50, B=50` vs `A=500, B=500` scores `1`.
Total mass may be emitted as a diagnostic only.

## 9. Scorer 3 — Rate / Share / Proportion

Purpose: values with a fixed bounded semantic scale `[0, 1]`; the meaningful error is an
absolute percentage-point error.

**Metric:** matched key `S_k = 1 − |r_k − s_k|`; missing/extra key `S_k = 0`;
`S = (1/|K|) Σ_{k∈K} S_k`, which equals `1 − MAE` when all keys match.
Scalar rate: `real = 0.30, syn = 0.25 → 0.95`.
Keyed example: `A 0.20/0.25, B 0.60/0.50, C 0.90/0.90 → (0.95 + 0.90 + 1.00)/3 = 0.95`.

**Range validation:** canonical values satisfy `0 ≤ rate ≤ 1`; values outside only by
floating noise (`RANGE_TOL = 1e-12`) may be clamped; material violations are rejected.
Never silently convert `35 → 0.35`; scale conversion must be explicitly declared by
template/query metadata and deterministic.

**Why not sMAPE:** `0.10 → 0.20` and `0.80 → 0.90` both differ by 10 percentage points;
MAE treats them equally, sMAPE would not.

## 10. Scorer 4 — Top-k Ranking

Purpose: an ordered list of identities (top-k, bottom-k, tail-k, ranked selection,
argmax/winner, argmin/loser). The score depends on item overlap, rank position,
top-weighting, and possibly different item sets. Attached magnitudes are diagnostics.

**Metric:** finite extrapolated RBO with `RBO_PERSISTENCE = 0.9`.

1. **Depth.** If the query/template has a requested depth `k`, `D = k`; otherwise
   `D = max(len(real), len(syn))`. Truncate longer lists to `D`; pad shorter lists to `D`
   with side-specific unique sentinel identities that never match any item or each other.
2. **Prefix agreement.** `X_d = |R_{1:d} ∩ S_{1:d}|`, `A_d = X_d / d`.
3. **Extrapolated RBO.**

```text
RBO_EXT = (1 − p) Σ_{d=1..D} p^(d−1) A_d + p^D A_D
```

Identical rankings score `1`; fully disjoint rankings score `0`.

**Identities, not measures:** rows `A 100, B 90, C 80` give ranking input `["A", "B", "C"]`.

**Ties:** official scoring must be deterministic. Queries that can tie should use a
deterministic tie-break (e.g. `ORDER BY score DESC, category ASC`). Engine incidental row
order must not determine RBO; if deterministic ordering cannot be established, the query is
invalid for official scoring.

## 11. NULL, NaN, Inf, and invalid values

- **SQL NULL** for numeric/rate values, key present on both sides:
  `NULL vs NULL → 1`, `NULL vs finite → 0`, `finite vs NULL → 0`.
  A key absent from one result remains unmatched and scores `0`.
- **NaN and ±Inf** are invalid scorer inputs, never SQL NULL (`math.isfinite` must hold).
- **Parse/type errors:** `valid = false`, `score = null`. Never assign `0`, which means valid
  maximal disagreement. Execution and parsing failures stay distinguishable from the metric.

## 12. Score range enforcement

Every successful scorer returns `0 ≤ score ≤ 1`, enforced by a postcondition checker with
`SCORE_TOL = 1e-12`: values outside the tolerance raise; tiny drift is clamped.
Clamping must never hide formula or parsing bugs.

## 13. Empty-answer policy

- Keyed numeric/rate, both empty: `1.0`.
- Keyed numeric/rate, one empty: unmatched keys score `0` (e.g. `{A: 1.0}` vs `{}` → `0.0`).
- Count distributions: zero-total rules of §8.
- Rankings: `[] vs [] → 1.0`; `[] vs non-empty →` RBO under the padding contract, normally `0.0`.

## 14. Diagnostics

Diagnostics may be emitted but never alter the primary score.

- Common: `real_key_count`, `synthetic_key_count`, `union_key_count`, `intersection_key_count`,
  `key_precision`, `key_recall`, `key_f1`.
- Numeric: `per_key_similarity`, `per_key_abs_error`, `mean_abs_error`, `matched_key_count`,
  `unmatched_key_count`.
- Count: `real_total_mass`, `synthetic_total_mass`, support sizes, `raw_tvd`, `key_f1`
  (total mass does not affect the TVD score).
- Rate: `raw_mae`, `mean_percentage_point_error`, `max_absolute_rate_error`, `key_f1`.
- Ranking: `requested_depth`, list lengths, `prefix_overlap_by_depth`, `raw_rbo`, `top1_match`,
  `set_overlap_at_k`; attached row magnitudes.

## 15. Output object

```json
{
  "valid": true,
  "semantic_query_score_method": "sv2",
  "score_contract_version": "sv2_four_primary_v1",
  "scorer_type": "numeric_magnitude",
  "metric": "normalized_smape_similarity",
  "semantic_query_score": 0.9454191033,
  "query_score": 0.9454191033,
  "diagnostics": {"real_key_count": 3, "synthetic_key_count": 3, "union_key_count": 3, "matched_key_count": 3}
}
```

Invalid:

```json
{
  "valid": false,
  "semantic_query_score_method": "sv2",
  "score_contract_version": "sv2_four_primary_v1",
  "scorer_type": "numeric_magnitude",
  "metric": "normalized_smape_similarity",
  "semantic_query_score": null,
  "query_score": null,
  "error_code": "NON_FINITE_NUMERIC_VALUE",
  "error_message": "Synthetic answer contains NaN."
}
```

## 16. Old V9 → SV2 compatibility mapping

| Old scorer / semantic type | SV2 scorer |
| --- | --- |
| `scalar` with ordinary numeric magnitude | `numeric_magnitude` |
| `scalar` with rate/probability semantics | `rate_share_proportion` |
| `ratio` | `numeric_magnitude` |
| `keyed_numeric_aggregate` | `numeric_magnitude` |
| `count_support_distribution` | `count_support_distribution` |
| `rate_share_proportion` | `rate_share_proportion` |
| `topk_ranking` | `topk_ranking` |

| Older hint | SV2 scorer |
| --- | --- |
| `distribution_cardinality_profile` | `count_support_distribution` |
| `topk_tailk_ranking`, `topk_ranked_measure`, `argmax_selection` | `topk_ranking` |
| `tail_topn_value_curve` | by semantic target: ranking if identity/order is primary, otherwise numeric magnitude |
| `tail_outlier_distribution` | by actual answer semantics |
| temporal rate / count distribution / numeric aggregate | `rate_share_proportion` / `count_support_distribution` / `numeric_magnitude` |

Do not route only from an old class name when the actual answer semantics are known.

## 17. No hidden weighting

No primary-score weighting of any kind. Numeric magnitude → normalized sMAPE only; count
distribution → `1 − TVD` only; rate → `1 − MAE` only; ranking → RBO only.

## 18. Query-level vs run-level aggregation

SV2 defines the query-level score. For reporting across queries:

1. **Scorer-type macro summary (preferred):** `M_t = mean{S_q : q uses scorer type t}` for each type.
2. **Query-count-weighted overall mean (secondary):** `(1/N) Σ_q S_q`.
3. **Optional macro over scorer types:** `(1/|T|) Σ_{t∈T} M_t` over scorer types present.

Name the aggregation explicitly; never silently substitute one for another.

## 19. Invalid-query accounting

Never convert invalid inputs into semantic zero. Report valid and invalid query counts, the
valid fraction, and invalid reason counts (`n_queries_total`, `n_queries_valid`,
`n_queries_invalid`, `valid_query_fraction`). SV2 does not impose a validity threshold.

## 20–23. Reference structure, API, router, nullable helper

- Module layout: `types`, `routing`, `canonicalize`, `common`, one module per scorer,
  `diagnostics`, `aggregate`, and tests.
- API: `score_query(real_answer, synthetic_answer, scorer_type, scorer_config=None) -> dict`.
- Router: explicit `sv2_scorer_type` first; old names mapped as in §16 (`scalar` uses
  `semantic_value_type` to select rate vs numeric); otherwise raise `UnresolvedScorerType`.
  Ambiguous old hints (tail/temporal shapes) require explicit semantic metadata.
- `compare_nullable_numeric(real, syn, comparator)`: NULL vs NULL → 1, NULL vs value → 0,
  otherwise the comparator; NaN/Inf must not pass as NULL.

## 24. Minimum unit tests

- **Numeric:** `100/100 → 1`; `100/80 → 0.8888888889`; `2/3 → 0.8`; `0/0 → 1`; `0/5 → 0`;
  `−10/−8 → 0.8888888889`; `1/−1 → 0`; keyed `A 100/90, B 50/40, C 20/20 → 0.9454191033`;
  missing key `A 10/10, B 20/– → 0.5`.
- **Count:** `A=50,B=50` vs `A=5,B=5 → 1`; shifted mass `→ 0.9`; disjoint support `→ 0`;
  both zero total (`{}` vs `{}` or `A=0` vs `A=0`) `→ 1`; one zero total `→ 0`.
- **Rate:** scalar `0.30/0.25 → 0.95`; keyed `→ 0.95`; missing key `→ 0.5`; `1.2` vs `0.8`
  → invalid unless metadata declares a transformation.
- **RBO:** identical `→ 1`; disjoint `→ 0`; winner same `→ 1`; winner different `→ 0`;
  unequal lengths compared against a manual computation of the §10 formula.

## 25. Property tests

All scorers bounded in `[0, 1]`; identity `score(x, x) = 1`; symmetry for all four scorers;
disjoint non-empty count supports score `0`; `score(x, −x) = 0` for nonzero finite `x`;
scalar rate `score(r, s) = 1 − |r − s|`.

## 26. Numerical reproducibility

Compute in double precision, do not round intermediates, clamp only tiny drift at the final
check, store full-precision results, and round only for presentation (recommended 4 decimals).

## 27. Metric names

| Scorer | Paper name | Metric ID |
| --- | --- | --- |
| Numeric Magnitude | normalized sMAPE similarity | `normalized_smape_similarity` |
| Count / Support Distribution | complement of TVD | `one_minus_tvd` |
| Rate / Share / Proportion | complement of MAE | `one_minus_mae` |
| Top-k Ranking | Rank-Biased Overlap | `rbo_p09` |

Recommended wording: *SV2 routes each query answer to a single established metric according
to its semantic answer space: normalized sMAPE for magnitude-valued numeric answers, total
variation distance for count distributions, mean absolute error for bounded rates and
proportions, and rank-biased overlap for ranked answers. All metrics are represented as
similarities in [0,1], with larger values indicating better fidelity.*

## 28. Semantic distinction

- Numeric Magnitude — how close are the magnitudes relative to their scale? (normalized sMAPE)
- Count / Support Distribution — how much support mass moved between categories? (TVD)
- Rate / Share / Proportion — by how many percentage points do bounded rates differ? (MAE)
- Ranking — how similar are the ordered top results, emphasizing higher ranks? (RBO)

## 29. Implementation checklist

Four and only four active scorer types; old scalar/ratio/keyed numeric route to numeric
magnitude unless the scalar is a rate; exact normalized sMAPE; normalized counts with
`1 − TVD`; `1 − MAE` for rates; finite extrapolated RBO with `p = 0.9` and §10 depth/padding;
union key space; missing/extra keys score 0; duplicate-key rules of §6.2; `0 vs 0 → 1`;
`0 vs nonzero → 0`; signed magnitudes; rates validated in `[0, 1]`; SQL NULL rules of §11;
NaN/Inf invalid; invalid inputs return `valid = false`; every valid score checked in
`[0, 1]`; diagnostics never affect the score; no composite weighting; fixed and property
tests pass; outputs carry scorer type, metric ID, contract version, score, validity, and
diagnostics; run-level reports distinguish scorer-type macro summaries from query-weighted
summaries.

## 30–31. Dispatcher and final definition

```text
Numeric Magnitude            -> 1 − sMAPE / 2
Count / Support Distribution -> 1 − TVD
Rate / Share / Proportion    -> 1 − MAE
Top-k Ranking                -> RBO_{p=0.9}
```

One query → one semantic answer type → one primary established metric → one score in `[0, 1]`.
