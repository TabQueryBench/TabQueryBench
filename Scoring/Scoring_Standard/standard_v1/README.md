# Semantic Scoring Standard v1

- `score_version` / `semantic_query_score_method`: `semantic_scoring_standard_v1`
- Definition: [`SCORING_STANDARD_V1.md`](SCORING_STANDARD_V1.md)
- Implementation: `scorer.py` (the pre-SPQ `code/src/evaluation/query_semantic_scoring.py`,
  which commit `a3c3dd7` replaced in place with SPQ v9).
- Select with `TQB_SCORING_MODE=standard_v1`; with `--use-semantic-query-score` it becomes
  the primary `query_score` (`primary_query_score_mode = semantic_template`).

Compared with `semantic_v8`, the template's `semantic_result_contract` is
authoritative for `scorer_type` and required outputs, missing required outputs
are a validity gate (score `0.0`, `validity_status = invalid`), and every row
records `score_version` and `component_scores`.

## Results

The "SPQ v8" runs use this standard (`score_version = semantic_scoring_standard_v1`,
`primary_query_score_mode = semantic_template`) on V8 SQL:

- `Scoring/results/analysis/runs/spq_v8_all_synthetic_models_20260913` (+ `final49_tables`, `partial47_tables`)
- `Scoring/results/analysis/runs/spq_v8_c13_*_rescue_20260913`, `spq_v8_partial47_id_mappings_20260913`
- Review bundle: `Scoring/results/analysis/packages/tabquerybench_spq_v8_review_bundle_20260914/` (and `.zip`)
