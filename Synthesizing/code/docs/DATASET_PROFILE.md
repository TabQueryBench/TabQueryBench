# Dataset profile (`profile.yaml`) — `tqb_dataset_profile_v1`

Every new dataset is delivered as a directory with the raw table and a `profile.yaml`:

```text
my_dataset/
  my_dataset.csv
  profile.yaml
```

The profile is **required**. It is the single source of truth for column types, missing
values, temporal formats and the target; nothing is guessed at run time.

```bash
cd Synthesizing/code
export PYTHONPATH=src
python -m core.prepare draft --data my_dataset/my_dataset.csv --dataset-id my_dataset --target label -o my_dataset/profile.yaml  # optional starting point, review it
python -m core.prepare check my_dataset
python -m core.prepare run my_dataset            # -> DatasetNew/my_dataset/
python -m core.doctor
python -m core.runner.batch --datasets my_dataset --models all --preset smoke --gpus 0,1
python -m core.runner.batch --datasets my_dataset --models all --gpus 0,1
```

## Example

```yaml
schema_version: tqb_dataset_profile_v1
dataset_id: 1117_orcamaster2010_csv          # [A-Za-z0-9][A-Za-z0-9_.-]*; output directory name
description: Southern resident orca sightings
source: SQLShare Release 1, [1117].[OrcaMaster2010.csv]
data_file: 1117_orcamaster2010_csv.csv       # relative to profile.yaml
format: {delimiter: ",", encoding: utf-8}    # optional (these are the defaults)
missing_tokens: ["", "NULL"]                 # exact cell values (after trimming) that mean missing
on_invalid: error                            # error (default) | null
target: {column: Pod, task_type: classification}
split: {train: 0.8, val: 0.1, test: 0.1, seed: 42, stratify: true}
generation: {max_train_rows: null, num_rows: train}
columns:                                     # every data column, exactly once
  - {name: SightDate, type: date, format: "%m/%d/%Y"}
  - {name: Time2, type: time, format: "%H%M", pad: 4, on_invalid: "null"}
  - {name: Pod, type: categorical}
  - {name: Lat, type: continuous}
  - {name: Quadrant, type: integer}
  - {name: Notes, type: text, drop: true}
```

## Top-level keys

| Key | Required | Meaning |
|---|---|---|
| `schema_version` | yes | `tqb_dataset_profile_v1` |
| `dataset_id` | yes | Id used by the runner and as output directory name |
| `data_file` | yes | CSV path relative to the profile |
| `target.column` | yes | Exactly one target (several models need X/y) |
| `target.task_type` | yes | `classification` or `regression` |
| `columns` | yes | One entry per data column (see below); order does not matter |
| `format.delimiter` / `format.encoding` | no | Default `,` / `utf-8` (BOM tolerated) |
| `missing_tokens` | no | Default `[""]`. Matched exactly and case-sensitively after trimming whitespace. Words like `Unknown` are real values unless listed |
| `on_invalid` | no | What to do with a non-missing value that does not fit the column type: `error` (stop and report examples) or `null` (turn into missing and count it in the report) |
| `split` | no | Ratios (sum to 1), seed, and `stratify` (classification only; classes with < 3 rows go to train) |
| `generation.max_train_rows` | no | Cap on training rows (stratified sample for classification). **Default `null` = off.** Overridden by `--max-train-rows` (`0` forces off) |
| `generation.num_rows` | no | Rows to generate: `train` (size of the train split, default) or an integer. Overridden by `--num-rows` |
| `description`, `source`, `notes` | no | Free text |

## Column entries

| Key | Meaning |
|---|---|
| `name` | Column header in the CSV |
| `type` | `continuous`, `integer`, `categorical`, `ordinal`, `boolean`, `id`, `text`, `date`, `datetime`, `time` |
| `format` | Required for `date` / `datetime` / `time`: a `strptime` pattern (`%m/%d/%Y %I:%M:%S %p`, `%Y_%j`, ...) or `iso8601`, `unix_s`, `unix_ms` |
| `pad` | Left-pad with zeros to this width before parsing (numbers that lost leading zeros, e.g. `80813` → `080813` for `%d%m%y`) |
| `order` | `ordinal` only: ordered list of levels |
| `missing_tokens` | Per-column override of the dataset-level list |
| `on_invalid` | Per-column override (`error` / `null`) |
| `drop` | `true` to exclude the column from prepared data and generation |
| `description` | Free text |

### Type semantics

| Type | Prepared value | Model sees | Generated value |
|---|---|---|---|
| `continuous` | float | float | float, finite (range warning outside real min/max) |
| `integer` | integer | integer | rounded integer |
| `categorical`, `boolean`, `ordinal` | original string | category | a value from the real domain |
| `id`, `text` | original string | category (high cardinality warned) | a value seen in training |
| `date` | `YYYY-MM-DD` | days since 1970-01-01 | `YYYY-MM-DD`, clipped to the real range |
| `datetime` | `YYYY-MM-DD HH:MM:SS` (timezones converted to UTC) | seconds since 1970-01-01 | `YYYY-MM-DD HH:MM:SS` |
| `time` | `HH:MM:SS` | seconds since midnight | `HH:MM:SS` |

Missing values are written as empty cells everywhere and are preserved through generation.

## What `core.prepare run` produces

```text
DatasetNew/<dataset_id>/
  <id>-main.csv  <id>-train.csv  <id>-val.csv  <id>-test.csv
  metadata_core/field_registry.json      # consumed by staging and postprocess
  metadata_core/dataset_semantics.yaml
  metadata_core/profile.yaml             # copy of the input profile
  prepare_report.json                    # per-column missing/invalid counts, warnings
```

`main.csv` is the full cleaned table (rows with a missing target are kept there but excluded
from train/val/test). Warnings flag entirely-missing columns, constant columns, high-cardinality
categorical columns, many-class targets and large tables.

## What happens at run time (no per-model work needed)

Staging builds a **model view** of the prepared data so all 11 adapters receive the same simple
table, and inverts it after generation:

1. temporal columns → integers, decoded back to ISO 8601;
2. entirely-missing and constant columns removed, re-inserted afterwards;
3. column names sanitized (`OCEAN.TEMP` → `OCEAN_TEMP`), restored afterwards;
4. category values that pandas would read as NaN (`NA`, `None`, ...) escaped, restored afterwards;
5. `id` / `text` presented as categorical;
6. optional training-row cap (`max_train_rows`, off by default).

Each run directory contains `staged/public/model_view_spec.json`, the model postprocess status
(`postprocess_run_status.json`) and `model_view_restore.json`; the final synthetic CSV has exactly
the prepared schema.
