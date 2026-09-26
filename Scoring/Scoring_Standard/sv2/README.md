# SV2 — Semantic Scoring V2

- Contract: [`SEMANTIC_SCORING_SV2.md`](SEMANTIC_SCORING_SV2.md)
- `semantic_query_score_method`: `sv2`; `score_contract_version` / `score_version`: `sv2_four_primary_v1`
- Metrics: `normalized_smape_similarity`, `one_minus_tvd`, `one_minus_mae`, `rbo_p09`
- Tests: `Scoring/code/tests/test_sv2_scoring.py`

## Layout

```text
sv2/
  SEMANTIC_SCORING_SV2.md       contract
  template_routing_sv2.json     explicit per-template routing (49 V8/V9 templates)
  common.py  types.py  diagnostics.py
  routing.py                    semantic routing (explicit > template table > compatibility names > fail)
  canonicalize.py               raw answers / SQL result tables -> canonical answers
  numeric_magnitude.py  count_support_distribution.py  rate_share_proportion.py  topk_ranking.py
  api.py                        score_query(), score_execution_results()
  aggregate.py                  run-level summaries (+ CLI)
  scorer.py                     analysis-runner entry point
```

## Usage

```python
from tqb_scoring.standards.sv2 import score_query

score_query({"A": 100, "B": 50}, {"A": 90, "B": 40}, scorer_type="numeric_magnitude")["query_score"]  # 0.9181286549
```

Recompute query scores with the analysis runner:

```bash
TQB_SCORING_MODE=sv2 PYTHONPATH=Scoring/code \
  python3 -m tqb_scoring.eval.analysis.runner --run-tag sv2_qv9_<date> --sql-source-version v9 ...
```

With `TQB_SCORING_MODE=sv2` the runner fills `semantic_query_score` with SV2, uses it as
`query_score` (full precision), keeps `legacy_query_score` separately, and writes
`semantic_valid`, `semantic_error_code`, and `semantic_metric` on every query row.
Invalid queries have `query_score = null` and are excluded from template/subitem/family/asset
means; asset rows report `valid_query_count` and `invalid_query_count`. The run writes
`summaries/analysis_sv2_summary__all_datasets.csv` plus `_by_model` and `_by_dataset_model`
variants (scorer-type means, query-weighted mean, macro-over-scorer-types mean, invalid
reason counts), and `manifest.json` gets `sv2_summary`.

Summaries can be rebuilt from any query-score JSONL:

```bash
PYTHONPATH=Scoring/code python3 -m tqb_scoring.standards.sv2.aggregate \
  <run>/summaries/analysis_query_scores__all_datasets.jsonl --group-by model_id --output sv2_by_model.csv
```

## Implementation decisions

These follow the contract; where the contract leaves a choice to the implementation, the
choice is recorded here.

1. **Routing table.** The V8/V9 `semantic_result_contract.primary_measure` often does not
   name an actual output column (e.g. `tpl_h2o_group_sum` outputs `total_measure`, and
   `tpl_m4_group_dispersion_rank` renders `measure_variance`). `template_routing_sv2.json`
   therefore declares, per template: scorer type, answer shape, value column, diagnostic
   columns, ranking column/direction, ranking depth rule, and rate scale. Output columns
   were verified by executing V8 and V9 release queries for every template and variant on
   real data. Key columns are all result columns that are not declared value, ranking, or
   diagnostic columns. Queries without a table entry fall back to the contract's
   `primary_measure`; if it is not a result column the query is invalid
   (`UNRESOLVED_VALUE_COLUMN`).
2. **Semantic choices in the table.**
   - `tpl_tpcds_within_group_share` returns `SUM * 100.0 / group total` and is declared with
     `rate_scale = percent` (explicit ÷100).
   - `tpl_h2o_topn_within_group` (`tail_topn_value_curve`) is numeric magnitude keyed by
     `(group_value, measure_rank)`: the target is the value curve at each within-group rank.
   - `tpl_m4_global_zscore_outliers` scores `outlier_rate` (rate);
     `tpl_m4_quantile_tail_slice` scores `mean_tail_measure` (numeric).
   - `tpl_tail_drift_ratio` scores `drift_ratio` (numeric); counts are diagnostics.
   - `tpl_h2o_two_dimensional_robust_summary` scores `median_measure`;
     `tpl_tpch_two_dimensional_summary` scores `avg_measure`;
     `tpl_tpcds_subgroup_baseline_outlier` scores `entity_measure`.
   - Count templates that use `LIMIT` (top-k/tail-k counts) stay count distributions over the
     returned rows, as their template contracts declare.
   - Ranking templates take the requested depth from the trailing `LIMIT n` of the rendered
     SQL; `tpl_tpch_max_aggregate_winner` and `tpl_tpch_relative_total_threshold` have no
     requested depth (`D = max list length`).
3. **Key canonicalization.** Key cells are type-aware: numbers and numeric text share one
   representation (integers exactly, other values with 15 significant digits), so a group
   value stored as INTEGER in one table and TEXT/REAL in the other still aligns. Other text
   is compared exactly. If this normalization would merge two distinct raw keys inside
   either result (e.g. `0` and `'0'` as separate SQLite groups), both sides of that query use
   exact type-tagged keys instead (`key_match_mode = exact_typed`); remaining duplicates are
   invalid.
4. **Ranking depth.** `D` is the requested depth `k` capped at the longer observed list. The
   contract pads both lists when both are shorter than `k`, which would make identical
   rankings score below 1 and contradict its own identity guarantee (§10.3, §25.2); the cap
   keeps truncation to `k` and padding of the shorter list otherwise unchanged.
5. **Ranking ties.** Rows are ordered by the ranking column in the declared direction and then
   by identity ascending, which reproduces the SQL order with a deterministic tie-break.
   Ties at a SQL `LIMIT` boundary (which tied rows were cut) cannot be recovered from the
   result; `*_tie_reordered_positions` diagnostics report reordering inside the returned rows.
6. **Execution failures** of the real or synthetic query are invalid
   (`REAL_EXECUTION_FAILED`, `SYNTHETIC_EXECUTION_FAILED`), not zero, per contract §11.
   Other invalid codes: `COLUMN_MISMATCH`, `MISSING_OUTPUT_COLUMN`, `UNEXPECTED_OUTPUT_COLUMNS`,
   `SCALAR_ROW_COUNT`, `DUPLICATE_KEY`, `DUPLICATE_RANKING_IDENTITY`, `NON_NUMERIC_VALUE`,
   `NON_FINITE_NUMERIC_VALUE`, `NULL_COUNT`, `NEGATIVE_COUNT`, `RATE_OUT_OF_RANGE`,
   `UNRESOLVED_SCORER_TYPE`.
7. **Numerics.** float64 with `math.fsum`; `ZERO_TOL = RANGE_TOL = SCORE_TOL = 1e-12`.
