# TabQueryBench

This directory contains the query benchmark portion of the TabQueryBench public code release.

The code in this directory is intended to reproduce the SQL/query generation, workload grounding, and evaluation pipelines used in the TabQueryBench release. The synthetic model training and sampling code is separated under `../synthetic_generation/`.

## Related Hugging Face Repositories

### Main public data repository

- [TabQueryBench2026/TabQueryBench](https://huggingface.co/datasets/TabQueryBench2026/TabQueryBench/tree/main)

This repository contains the public non-code assets, including:

- `raw_data/`: released raw tabular datasets
- `synthetic_data/`: released synthetic tabular outputs
- `Query/`: query artifacts organized by dataset
- `Query_Templates/`: template library and supporting materials

### Docker images

- [TabQueryBench2026/TabSyn-Docker](https://huggingface.co/datasets/TabQueryBench2026/TabSyn-Docker/tree/main)

Use the Docker repository if you want prebuilt container assets for the tabular synthesis stack.

## Directory Layout

- `src/agent/`: SQL agent execution logic
- `src/benchmark/`: benchmark construction, contracts, planning, validation, and SQL execution helpers
- `src/data/`: dataset bundle and layout utilities
- `src/db/`: CSV-to-SQLite materialization helpers
- `src/workload_grounding/`: query inventory, grounding, and question set construction
- `src/eval/`: SQL evaluation, analysis, family breakdowns, validation, and reporting code
- `src/evaluation/`: auxiliary evaluation utilities used by experiment scripts
- `src/usage/`: token usage and pricing utilities
- `scripts/`: command-line entrypoints, packaging helpers, audits, repairs, and figure builders
- `tests/`: lightweight validation scripts

## Installation

Install the base Python dependencies with:

```bash
pip install -r query_benchmark/requirements.txt
```

This repository is a release snapshot rather than a fully packaged Python project. Some scripts assume repository-relative paths and locally mounted data directories.

## Typical Usage

Representative entrypoints include:

- `scripts/run_sql_agent.py`: run SQL generation against a dataset/query inventory
- `scripts/run_benchmark_agent.py`: run benchmark construction and agent execution
- `scripts/run_benchmark_evaluation.py`: run benchmark-side evaluation workflows
- `scripts/run_subitem_workload_v2.py`: run the v2 subitem workload pipeline
- `scripts/run_subitem_workload_inventory_dir.py`: run a prepared inventory directory
- `scripts/run_tail_threshold.py`: run the tail-threshold evaluation pipeline

Run from the repository root with:

```bash
PYTHONPATH=query_benchmark python query_benchmark/scripts/run_benchmark_agent.py
```

## Scope of This GitHub Repository

This GitHub repository is intentionally code-focused. It does not attempt to mirror all released data assets. For datasets, synthetic outputs, query artifacts, and template libraries, use the linked Hugging Face repositories above.
