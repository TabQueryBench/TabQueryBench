# Synthetic Generation

Training and sampling code for 11 tabular synthetic data generation models, plus the
pipeline that takes a new dataset from a raw CSV and a profile to synthetic CSVs.

## Models

`arf`, `bayesnet`, `ctgan`, `forestdiffusion`, `realtabformer`, `tabbyflow`, `tabddpm`,
`tabdiff`, `tabpfgen`, `tabsyn`, `tvae` (images in `src/core/runner/docker_images.json`;
`tvae` uses the ctgan image, `tabdiff`/`tabbyflow` share one image).

## New dataset → synthetic data

A dataset package is a directory with the raw CSV and a required `profile.yaml`
(column types, missing tokens, temporal formats, target). Spec: [`docs/DATASET_PROFILE.md`](docs/DATASET_PROFILE.md).

```bash
cd Synthesizing/code
export PYTHONPATH=src

# 1. (optional) draft a profile, then review every entry
python -m core.prepare draft --data my_ds/my_ds.csv --dataset-id my_ds --target label -o my_ds/profile.yaml

# 2. validate, clean, normalize dates, split, build metadata  -> DatasetNew/my_ds/
python -m core.prepare run my_ds            # or a parent directory containing many dataset packages

# 3. check docker, images, GPUs, model weights
python -m core.doctor

# 4. quick end-to-end check on all 11 models, then the real run
python -m core.runner.batch --datasets my_ds --models all --preset smoke --gpus 0,1
python -m core.runner.batch --datasets my_ds --models all --gpus 0,1 [--max-train-rows 50000]
```

Single run:

```bash
python -m core.runner.runner --model ctgan --dataset my_ds --train --generate [--preset smoke] [--max-train-rows N] [--num-rows N]
python -m core.runner.runner --model ctgan --dataset my_ds --generate --model-dir /path/to/run_dir
```

Outputs: `output-Benchmark-trainonly-v1/<dataset>/<model>/<run>/` with the synthetic CSV (same
schema and formats as the prepared table), `runtime_result.json`, `run_config.json`,
`postprocess_run_status.json`, `model_view_restore.json`. Batches also write
`output-Benchmark-trainonly-v1/_batch/<batch_id>/batch_status.csv` and per-job logs.

## What the pipeline handles

- **Preparation** (`src/core/prepare/`): profile validation against the data, exact missing tokens,
  type coercion with invalid-value reporting, ISO 8601 date/datetime/time normalization,
  deterministic (stratified) train/val/test split, `field_registry.json`, `prepare_report.json`
  with warnings (all-missing, constant, high-cardinality columns, large tables).
- **Model view** (`src/core/runner/staging/model_view.py`): every adapter receives the same simple
  table — temporal columns as integers, all-missing/constant columns removed, sanitized column
  names, NaN-looking category values escaped, id/text as categorical — and the view is inverted on
  the generated CSV.
- **Training-row cap**: `--max-train-rows` / `generation.max_train_rows` / `BENCHMARK_MAX_TRAIN_ROWS`,
  stratified for classification. Off by default. Generation size defaults to the full train split.
- **Postprocess** (`src/models/shared/`): row count, column order, finite numbers, integer rounding,
  categorical domain restoration; a run fails instead of publishing malformed data. Accepted outputs also
  get non-blocking `quality_warnings` (numeric values piled on the training min/max, constant columns,
  categorical TVD > 0.5, missing-rate drift > 0.25), shown in `runtime_result.json` and the batch table.
- **Batch** (`src/core/runner/batch.py`): GPU/CPU slot scheduling, per-job logs, `--skip-done`, timeouts,
  automatic retry on CUDA out-of-memory.
- **Presets** (`src/core/runner/model_presets.json`): `smoke` (a few minutes per model) and `default`.

## Layout

```text
Synthesizing/code/
  docs/DATASET_PROFILE.md
  src/core/prepare/          # profile.yaml + CSV -> prepared dataset
  src/core/runner/           # runner, batch scheduler, staging/model view, presets, docker image index
  src/core/doctor.py         # environment and readiness checks
  src/models/shared/         # base adapter (docker), postprocess
  src/models/<model>/        # model adapter and upstream source snapshot
  synthetic_benchmark/       # vendored upstream source trees mounted into containers
  scripts/                   # historical one-off jobs for the original 49 datasets (hard-coded paths)
```

## Paths and environment

| Variable | Default |
|---|---|
| `BENCHMARK_NEW_DATASET_ROOT` | `Synthesizing/code/DatasetNew` (where `core.prepare run` writes) |
| `BENCHMARK_PREPROCESSING_ROOT` | same as the dataset root |
| `BENCHMARK_OUTPUT_ROOT` | `Synthesizing/code/output-Benchmark-trainonly-v1` |
| `BENCHMARK_<MODEL>_IMAGE` | image from `docker_images.json` |
| `BENCHMARK_<MODEL>_GPUS` | `all` (batch sets `device=N`) |
| `BENCHMARK_MAX_TRAIN_ROWS` | unset (no cap) |
| `BENCHMARK_DOCKER_EXTRA_MOUNTS` | extra host paths to mount (os.pathsep-separated) |

Dataset, preprocessing and output roots outside the code directory are mounted into containers at
the same absolute path automatically. Model-specific variables (`CTGAN_*`, `TABSYN_*`, `TABPFGEN_*`, ...)
are documented in each adapter.

## Docker images

Image names come from `src/core/runner/docker_images.json` (`tabquerybench/<model>:latest`); override
with `BENCHMARK_<MODEL>_IMAGE`. Prebuilt image archives are released separately on Hugging Face.
TabPFGen uses TabPFN v2 weights by default (public; `python -m models.tabpfgen.adapter.tabpfgen_adapter download --version v2`).
