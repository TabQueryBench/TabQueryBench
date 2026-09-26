# synthetic_data

`synthetic_data` is the public release view of valid synthetic generation assets.

## Structure

```text
synthetic_data/
  README.md
  INDEX.csv
  INDEX.md
  main/
    <dataset_id>/
      <model_id>/
        <run_id>/
```

## Included Sources

- `main/` is built from the canonical server source `/data/jialinzhang/TabQueryBench/SynDataSuccess/main`.
- Unique supplemental runs are included from `timecost` and `5090-Success` when they do not duplicate an existing dataset/model/run.
- `hyper_parameter_tuning/` is intentionally excluded from this public release.

## Dedup Rule

When the same dataset/model/run exists in multiple source roots, only one public copy is retained.
Preferred source precedence is:
1. `main`
2. `timecost`
3. `success_5090`

All public run entries are recorded in `INDEX.csv`.

## Current Snapshot

- datasets: `0`
- models: `0`
- runs: `0`
