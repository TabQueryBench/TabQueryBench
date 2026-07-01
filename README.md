## TabQueryBench Code Snapshot

This directory contains the public code snapshot for the TabQueryBench release.

### Layout

- `src/agent/`: SQL agent execution logic.
- `src/benchmark/`: benchmark construction, contracts, validation, and execution helpers.
- `src/data/`: dataset bundle and layout utilities.
- `src/db/`: CSV-to-SQLite materialization helpers.
- `src/workload_grounding/`: query inventory, grounding, and question set construction.
- `src/eval/`: SQL evaluation, analysis, family breakdowns, and reporting code.
- `src/evaluation/`: auxiliary evaluation utilities used by experiment scripts.
- `src/usage/`: token usage and pricing utilities.
- `scripts/`: command-line entrypoints, packaging helpers, audits, and figure builders.
- `tests/`: lightweight validation scripts.

### Recommended Entry Points

- `scripts/run_sql_agent.py`: run SQL generation against a dataset/query inventory.
- `scripts/run_benchmark_agent.py`: run benchmark construction and agent execution.
- `scripts/run_benchmark_evaluation.py`: run benchmark-side evaluation workflows.
- `scripts/run_subitem_workload_v2.py`: run the v2 subitem workload pipeline.
- `scripts/run_subitem_workload_inventory_dir.py`: run a prepared inventory directory.
- `scripts/run_tail_threshold.py`: run the tail-threshold evaluation pipeline.

### Environment

Install the base Python dependencies with:

```bash
pip install -r requirements.txt
```

This is a release snapshot rather than a fully packaged Python project, so some scripts assume repository-relative paths and local data directories.

### Scope

The public `code/` directory is intended to document and reproduce the SQL/query and evaluation pipelines at a script level. It does not currently include a full packaging layer, CI setup, or an exhaustive automated test suite.
