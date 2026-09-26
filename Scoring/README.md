# Scoring

Scoring standards, the evaluation code that executes benchmark SQL on real and
synthetic data, and the resulting scores.

```text
Scoring/
  Scoring_Standard/     versioned scoring definitions and implementations
    legacy_composite/   semantic_v8/   standard_v1/   spq_v9/ (+ sqlagent_batch_20260914/)   sv2/
  code/
    tqb_scoring/
      standards/        registry that loads Scoring_Standard/<version>/scorer.py
      eval/             analysis runner and downstream evaluation tasks
                        (distance, tail_threshold, query_fivepart_breakdown, …)
      evaluation/       benchmark self-evaluation, legacy composite scorer,
                        query_semantic_scoring (compatibility module for the default standard, spq_v9)
      paths.py          repository locations used by scoring code
    scripts/            evaluation launchers, figures, and tables
    tests/              scoring standard tests
  results/              scoring outputs (formerly code/Evaluation); see results/README.md
```

## Running the analysis

```bash
# default semantic standard (spq_v9) as primary score
PYTHONPATH=Scoring/code python3 -m tqb_scoring.eval.analysis.runner --use-semantic-query-score ...
# SV2 as primary score (invalid queries are null and excluded from means; writes sv2 summaries)
TQB_SCORING_MODE=sv2 PYTHONPATH=Scoring/code python3 -m tqb_scoring.eval.analysis.runner ...
# another standard, e.g. standard_v1, as primary score
TQB_SCORING_MODE=standard_v1 PYTHONPATH=Scoring/code python3 -m tqb_scoring.eval.analysis.runner --use-semantic-query-score ...
# tests
PYTHONPATH=Scoring/code python3 -m unittest discover -s Scoring/code/tests
```

Runs of the same dataset otherwise each materialize an identical real-train SQLite
database. `--shared-real-cache-root DIR` (or `EVAL_SHARED_REAL_CACHE_ROOT`) materializes it
once per dataset under `DIR/real_sqlite/`, guarded by a lock file, and every later run reuses
it — worth setting when scoring one dataset across many models in parallel.

Every query row always carries `legacy_query_score` alongside the semantic
score, so runs remain comparable across standards. Outputs default to
`Scoring/results/<task>/` (override with `EVAL_OUTPUT_ROOT` or
`TQB_SCORING_RESULTS_ROOT`); query SQL and logs are read from `Query/code`.

`tqb_scoring` imports `tqb_query`; when only `Scoring/code` is on `sys.path`
the package adds `Query/code` automatically. Install the dependencies from
`Query/code/requirements.txt`.
