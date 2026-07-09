# TabQueryBench

This repository contains the public code release for TabQueryBench.

The code is split into two top-level components:

- `query_benchmark/`: SQL/query generation, workload grounding, benchmark construction, and evaluation code.
- `synthetic_generation/`: training and sampling code for the 11 tabular synthetic data generation models used by TabQueryBench.

Large datasets, generated synthetic outputs, query artifacts, template libraries, model weights, and run logs are not stored in this GitHub repository. They are released through the linked Hugging Face repositories.

## Related Hugging Face Repositories

### Main Public Data Repository

- [TabQueryBench2026/TabQueryBench](https://huggingface.co/datasets/TabQueryBench2026/TabQueryBench/tree/main)

This repository contains the public non-code assets, including:

- `raw_data/`: released raw tabular datasets
- `synthetic_data/`: released synthetic tabular outputs
- `Query/`: query artifacts organized by dataset
- `Query_Templates/`: template library and supporting materials

### Docker Images

- [TabQueryBench2026/TabSyn-Docker](https://huggingface.co/datasets/TabQueryBench2026/TabSyn-Docker/tree/main)

Use the Docker repository if you want prebuilt container assets for the tabular synthesis stack. The synthetic generation code also supports overriding every model image through `BENCHMARK_*_IMAGE` environment variables.

## Repository Layout

```text
TabQueryBench/
  query_benchmark/
    src/        # SQL agent, benchmark construction, grounding, and evaluation packages
    scripts/    # query benchmark and evaluation entrypoints
    tests/      # lightweight validation scripts

  synthetic_generation/
    src/core/   # unified train/generate runner
    src/models/ # 11 model adapters, shared postprocessing, and vendored model snapshots
    synthetic_benchmark/ # vendored upstream sources used by selected adapters
```

## Typical Entry Points

Query benchmark:

```bash
PYTHONPATH=query_benchmark python query_benchmark/scripts/run_benchmark_agent.py
PYTHONPATH=query_benchmark python query_benchmark/scripts/run_benchmark_evaluation.py
```

Synthetic generation:

```bash
PYTHONPATH=synthetic_generation/src python -m core.runner.runner --model ctgan --dataset c1 --train --generate
```

See the component READMEs for more details:

- [query_benchmark/README.md](query_benchmark/README.md)
- [synthetic_generation/README.md](synthetic_generation/README.md)
