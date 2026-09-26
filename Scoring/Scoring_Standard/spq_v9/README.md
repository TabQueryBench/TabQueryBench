# SPQ v9 — single-primary score

**SPQ** = TabQueryBench Single-Primary Query Similarity v9. One query yields
exactly one primary similarity in `[0, 1]`:

```text
query_score = primary_semantic_similarity
```

Component values are diagnostics only and never mixed into the score.

There are two implementations with the same identifiers
(`score_contract_version = spq_v9_single_primary`, method `spq`). They are not
numerically equivalent, so always record which one produced a result.

| Registry name | File | Origin | Used for |
| --- | --- | --- | --- |
| `spq_v9` (default semantic standard) | `scorer.py` | repository implementation (commits `a3c3dd7`, `1f4954d`); definition in [`SEMANTIC_SCORING_V9_SINGLE_PRIMARY.md`](SEMANTIC_SCORING_V9_SINGLE_PRIMARY.md); tests in `Scoring/code/tests/test_query_semantic_scoring_v9.py` | analysis runner default semantic score (`--use-semantic-query-score`, or `TQB_SCORING_MODE=spq_v9`) |
| `spq_v9_sqlagent_batch` | `sqlagent_batch_20260914/scorer.py` | `/mnt/nas/jialinzhang/SQLagent/src/evaluation/query_spq_v9.py` (original sha256 `6a0604e9…e4ec1`); helper import repointed to `semantic_v8` | the `semantic_spq_v9_batch_20260914*` runs (`TQB_SCORING_MODE=spq_v9_sqlagent_batch`) |

On a sample of 240 real V8 queries (n18/arf, c10/tvae, m1/ctgan) the two
implementations agreed on 109 queries and differed on 131; 16 of the
differences route the query to a different scorer type (`topk_ranking` in the
repository implementation vs `keyed_numeric_aggregate` / `ratio` in the SQLagent
variant). The stored scores of the 20260914c batch match the SQLagent variant.

## Repository implementation (`spq_v9`)

Six active scorer types (see the definition document for formulas):
`scalar`, `count_support_distribution` (1 − TVD over union keys),
`rate_share_proportion` (mean `1 - |r - s|` over union keys, missing keys 0),
`ratio` (`min/max` value similarity), `keyed_numeric_aggregate` (mean symmetric
numeric similarity over union keys), and `topk_ranking` (RBO, p = 0.9).
V8 scorer hints are routed into these types (`distribution_cardinality_profile`
→ support distribution, `topk_tailk_ranking` etc. → ranking; tail and temporal
answers are routed by result shape).

## SQLagent batch variant (`spq_v9_sqlagent_batch`)

1. **Validity gate.** If either execution fails, or either result has no columns,
   the score is `0.0` with `validity_status = invalid`.
2. **Scorer type.** Declared `scorer_type`, otherwise the `semantic_v8` policy:
   `scalar`/`tail_outlier_distribution` → `scalar`;
   `count_support_distribution`/`distribution_cardinality_profile` → `count_support_distribution`;
   `rate_share_proportion`; `ratio`; `topk_tailk_ranking` → `topk_ranking`;
   anything else → `keyed_numeric_aggregate`.
3. **Keys.** Result columns named in the SQL `GROUP BY` clause, falling back to the
   `semantic_v8` policy key columns; a key column that is at least 80% numeric is
   matched after numeric normalization.
4. **Primary similarity.** `scalar`: `semantic_v8` scalar similarity;
   `rate_share_proportion`: mean `1 - |r - s|` of the first measure over union keys;
   `count_support_distribution`: 1 − TVD of support distributions;
   `ratio` and `keyed_numeric_aggregate`: mean symmetric numeric similarity over
   union keys; `topk_ranking`: RBO (p = 0.9). Missing keys score 0.

## Results

- `Scoring/results/spq_v8_dataevolve_train49/` — scoring-mode comparison on the
  DataEvolve train49 assets (from commit `1f4954d`).
- `Scoring/results/analysis/runs/semantic_spq_v9_batch_20260914c/` — SQLagent
  variant batch; per-(dataset, model) scores in `source_runs/` (see its README).
  Known state: all 588 tasks returned 0, the merge step failed
  (`Duplicate dataset_id across source runs: n18`), 36 c1/m3/n13 tasks scored 0
  queries (missing V8 SQL source ids), 15 tasks had no synthetic data; SQL source V8.
