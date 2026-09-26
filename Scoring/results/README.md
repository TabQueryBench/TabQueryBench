# Scoring results

Outputs of `tqb_scoring` (formerly `code/Evaluation/`). Runners write to
`results/<task>/`; a local `code/Evaluation` symlink keeps old paths working.

```text
results/
  analysis/
    LATEST_RUN.json
    runs/
      spq_v8_all_synthetic_models_20260913/        standard_v1 on V8 SQL, all models (final49/partial47 tables)
      spq_v8_c13_*_rescue_20260913/                per-model c13 rescue runs
      spq_v8_partial47_id_mappings_20260913/       c1/m3/n13 → c20/m12/n20 id-mapping runs
      semantic_spq_v9_batch_20260914{,b,c}/        SPQ v9 orchestration logs and status files
        (20260914c) source_runs/                   SPQ v9 per-(dataset, model) scores copied from SQLagent
      sv2_v8_<dataset>_<model>_20260916/            SV2 on V8 SQL, one run per (dataset, model), 588 runs (tracked; cache/ excluded)
      sv2_v8_batch_20260916/                       SV2 V8 batch: overall / model / dataset-model summaries, batch_status.json
        paper_check/                               leaderboards, per-template/family/facet scores, coverage, cost vs fidelity
    packages/
      tabquerybench_spq_v8_review_bundle_20260914/ (+ .zip)   SPQ v8 review bundle
  spq_v8_dataevolve_train49/                     legacy vs SPQ scoring-mode comparison on DataEvolve train49 (tracked)
  tail_threshold_v2/
    runs/tail_10pct_to_0_1pct_20260914/            tail threshold sweep (10% → 0.1%)
    final/  LATEST_RUN.json
```

## Scores stored elsewhere

| Standard | Location |
| --- | --- |
| `legacy_composite` | `Query/Queries/V5…V7/*/analysis/*_scores.csv`; `/mnt/nas/jialinzhang/SQLagent/Evaluation/analysis/final/v1…v8` |
| `semantic_v8` | `Synthesizing/synthetic_data/run3/query_analysis/` |
| `spq_v9` raw runs with caches | `/mnt/nas/jialinzhang/SQLagent/Evaluation/analysis/runs/semantic_spq_v9_*` |
| `sv2` SQLite caches (not uploaded) | `results/analysis/runs/sv2_v8_*_20260916/cache/` on the NAS, rebuilt on demand |
