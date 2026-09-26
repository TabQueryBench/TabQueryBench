# Legacy composite score

- `query_score_method`: `composite_key_profile_rowcount_column`
- `score_contract_version`: `real_vs_synthetic_sql_result_v2`
- Implementation: `_compare_execution_results` in
  `Scoring/code/tqb_scoring/evaluation/real_panel_experiment.py`, exposed here by
  `scorer.py` as `compare_execution_results`. It stays in that module because the
  analysis runner computes it for every query as `legacy_query_score`.

## Definition

If either SQL execution fails the score is `0.0` (`real_query_failed` or
`synthetic_query_failed`). Otherwise, with real result `R` and synthetic result `S`:

| Component | Weight | Definition |
| --- | --- | --- |
| `strict_set_score` | 0.45 | weighted Jaccard of full-row multisets |
| `key_set_score` | 0.20 | weighted Jaccard of rows projected onto key columns |
| `profile_score` | 0.15 | mean per-column weighted Jaccard of value multisets over the key columns |
| `row_count_score` | 0.10 | `1 - |n_R - n_S| / max(1, n_R, n_S)` |
| `column_score` | 0.10 | Jaccard of column-name sets |

`score = clip(Σ weight × component, 0, 1)`. Key columns come from an explicit
SQL result-role annotation when available, otherwise from a column-name regex
that separates measure columns from key columns.

## Results

- Per-dataset legacy scores of agent-generated releases:
  `Query/Queries/V5-gpt-5.5-full`, `V6-gpt-5.4-full`, `V7-gpt-5.4-mini-full` (`*/analysis/*_scores.csv`).
- Versioned final bundles (not copied into this repository):
  `/mnt/nas/jialinzhang/SQLagent/Evaluation/analysis/final/v1 … v8`.
- Every later run also reports `legacy_query_score` per query.
