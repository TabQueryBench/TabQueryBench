# Scoring_Standard

Each subfolder is one version of the query scoring standard: its definition
documents and its `scorer.py` implementation. `tqb_scoring.standards` loads the
implementation directly from these folders, so this directory is the single
source of truth for scorer code.

| Registry name | `score_version` / method | Primary score rule | Selected by | Main results |
| --- | --- | --- | --- | --- |
| [`legacy_composite`](legacy_composite/README.md) | `composite_key_profile_rowcount_column` | fixed weighted composite of set, key, profile, row-count, column scores | default `query_score` (no flag) | V5–V7 releases, SQLagent `Evaluation/analysis/final/v1…v8` |
| [`semantic_v8`](semantic_v8/README.md) | `semantic_scorer_v1_parallel` | template-specific scorer, 100 / 50_50 / 50_25_25 / one_third_each weights | `TQB_SCORING_MODE=semantic_v8` | `Synthesizing/synthetic_data/run3/query_analysis` |
| [`standard_v1`](standard_v1/README.md) | `semantic_scoring_standard_v1` | Semantic Scoring Standard v1 composite, `semantic_result_contract` authoritative | `TQB_SCORING_MODE=standard_v1` | `results/analysis/runs/spq_v8_*`, `results/analysis/packages/tabquerybench_spq_v8_review_bundle_20260914` |
| [`spq_v9`](spq_v9/README.md) | `spq_v9_single_primary` / `spq` | one primary similarity per query (repository implementation) | default semantic standard (`--use-semantic-query-score`) or `TQB_SCORING_MODE=spq_v9` | `results/spq_v8_dataevolve_train49` |
| [`spq_v9_sqlagent_batch`](spq_v9/README.md) | `spq_v9_single_primary` / `spq` | one primary similarity per query (SQLagent variant) | `TQB_SCORING_MODE=spq_v9_sqlagent_batch` | `results/analysis/runs/semantic_spq_v9_batch_20260914*` |
| [`sv2`](sv2/README.md) | `sv2_four_primary_v1` / `sv2` | one established metric per semantic answer type: normalized sMAPE (numeric), 1 − TVD (count), 1 − MAE (rate), RBO p=0.9 (ranking); invalid inputs are null, not 0 | `TQB_SCORING_MODE=sv2` | new runs `results/analysis/runs/sv2_*` |

Lineage: `legacy_composite` → `semantic_v8` → `standard_v1` → `spq_v9` → `sv2`.
The SQLagent SPQ v9 variant reuses helper functions from `semantic_v8`.

In the analysis runner, `TQB_SCORING_MODE` selects which semantic standard fills
`semantic_query_score`; `--use-semantic-query-score` (or a `spq_v9*` / `sv2` mode) makes
it the primary `query_score`. `legacy_query_score` is always reported.

## Using a standard

```python
from tqb_scoring.standards import load_standard

spq = load_standard("spq_v9")
score, detail = spq.compare_semantic_execution_results(real_exec, synthetic_exec, query=query_row, legacy_detail=legacy_detail)
```

## Adding a standard

1. Create `<version>/` with a definition document and `scorer.py` exposing
   `compare_semantic_execution_results(real_exec, syn_exec, *, query=None, legacy_detail=None)`.
2. Register it in `STANDARDS` in `Scoring/code/tqb_scoring/standards/__init__.py`.
3. Select it with `TQB_SCORING_MODE=<version>` and store its runs under
   `Scoring/results/analysis/runs/<version>_*`.
