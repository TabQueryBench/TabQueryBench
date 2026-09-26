# Synthesizing

Real tabular datasets, synthetic data generation code, and the generated
synthetic data.

```text
Synthesizing/
  code/                 generation code (formerly synthetic_generation/)
    src/core/runner/    unified train/generate runner
    src/models/         model adapters (arf, bayesnet, ctgan, forestdiffusion, realtabformer,
                        tabbyflow, tabddpm, tabdiff, tabpfgen, tabsyn, tvae)
    synthetic_benchmark/  vendored upstream sources
    scripts/            data preparation, repair, import, audit, and HF packaging scripts
  raw_data/tabular_datasets/<dataset_id>/   real splits, metadata, SQLite cache
  synthetic_data/       generated synthetic data
    main/<dataset>/<model>/<run>/            canonical synthetic runs used for scoring
    run2/, run3/        additional generation rounds
    csv/                flattened CSV view (identical copies also under run3/csv and synthetic_data/run3/csv)
    query_analysis/     semantic v8 run3 query scores (identical copies under run3/ and run3/query_analysis)
    tail_threshold_runs/  tail-threshold packaging logs and archive
  release_manifests/    Hugging Face release manifests
```

Run the generator from the repository root:

```bash
PYTHONPATH=Synthesizing/code/src python -m core.runner.runner --model ctgan --dataset c1 --train --generate
```

See `code/README.md` for environment variables and Docker images, and
`synthetic_data/README.md` for the release layout. Many scripts in
`code/scripts/` are historical one-off jobs with hard-coded server paths.
