# Synthetic Generation

This directory contains the TabQueryBench training and sampling code for 11 tabular synthetic data generation models.

## Models

- `arf`
- `bayesnet`
- `ctgan`
- `forestdiffusion`
- `realtabformer`
- `tabbyflow`
- `tabddpm`
- `tabdiff`
- `tabpfgen`
- `tabsyn`
- `tvae`

## Layout

```text
synthetic_generation/
  src/core/runner/       # unified train/generate runner
  src/models/shared/     # shared adapter utilities and schema postprocessing
  src/models/<model>/    # model-specific adapter and upstream source snapshot
  synthetic_benchmark/   # vendored upstream source trees required by selected models
```

Each model adapter exposes a common `train()` and `generate()` interface. The shared adapter layer performs final generated CSV validation and postprocessing, including:

- enforcing the requested row count
- restoring training column order
- enforcing numeric finite values
- rounding integer columns back to integer dtype
- validating categorical-like values against the training domain
- failing generation instead of falling back to raw training data

## Typical Usage

Run from the repository root:

```bash
PYTHONPATH=synthetic_generation/src python -m core.runner.runner --model ctgan --dataset c1 --train --generate
```

Generation-only with a previous model directory:

```bash
PYTHONPATH=synthetic_generation/src python -m core.runner.runner --model ctgan --dataset c1 --generate --model-dir /path/to/model_dir
```

## Data and Output Paths

The runner resolves paths from environment variables when provided:

- `BENCHMARK_NEW_DATASET_ROOT`: root for released tabular datasets
- `BENCHMARK_PREPROCESSING_ROOT`: root for preprocessing outputs
- `BENCHMARK_OUTPUT_ROOT`: output root for model runs and generated CSVs
- `BENCHMARK_SYNTHETIC_ROOT`: override for vendored upstream source root

Without overrides, paths are resolved relative to this `synthetic_generation/` directory.

## Docker Images

Model image names are configured in `src/core/runner/docker_images.json`. Every image can be overridden by an environment variable, for example:

```bash
export BENCHMARK_CTGAN_IMAGE=your-ctgan-image:tag
```

Prebuilt container assets are released separately on Hugging Face.
