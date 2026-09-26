# TabQueryBench pipeline overview

This document defines the public, local pipeline for a user-provided CSV.  It
also records the compatibility boundary between query bundles, SQL execution,
reference data, scoring, and the repository browser.

## Flow

```text
custom CSV + optional metadata
  -> Query API (`generate_query_bundle`)
  -> SQLite execution (real/reference and synthetic data)
  -> semantic + legacy scoring
  -> JSON/JSONL/CSV artifacts
  -> local website browser
```

The query API is deliberately offline: it uses the checked-in V8 template
registry and SQLite only.  The generated bundle records zero network and LLM
calls, plus hashes of the input CSV and template registry.

## Public artifact contracts

| Boundary | Source of truth | Required compatibility fields |
| --- | --- | --- |
| Query bundle | `tabquerybench.query_bundle.v1` | top-level `schema_version`, `manifest`, `dataset`, `queries`; each query has `query_id`, `sql`, `output`, `provenance`, and `semantic_result_contract` |
| Execution timing | `ExecutionResult` | `exec_ok`, `exec_engine`, `exec_row_count`, `exec_timed_out`, `exec_error`, `exec_started_at`, `exec_ended_at`, `exec_elapsed_ms` (with `real_` or `synthetic_` prefixes in score rows) |
| Reference | real CSV materialized as SQLite | `real_reference_split`, `real_source_dataset_id`, `real_source_path`, `real_source_exists`; scoring compares the synthetic result with this real baseline |
| Scoring standard | `semantic_scoring_standard_v1` | raw `semantic_query_score`, `score_version`, `scorer_type`, `component_scores`, `validity_status`, `validity_failures` |
| Published V8 release | `Query/Queries/V8-template-full/release_manifest.json` plus every `*/analysis/dataset_manifest.csv` | shared query version, release directory, generator, round ID, query count, and one common dataset-manifest header |
| Website | `website/server.mjs` | `GET /api/inventory` returns `files` and `counts`; `GET /api/preview?path=…` is repo-root confined and previews text files only |

`semantic_query_score_method` is retained as a compatibility field alongside
the standard `score_version`; historical V8 artifacts use
`semantic_scorer_v1_parallel`. Score calibration and any separate normalized
score are intentionally not part of this release.

Timing is observability data, not a score component.  It may be aggregated in
run manifests but must not alter `query_score`, `legacy_query_score`, or
`semantic_normalized_score`.

## Generate a custom bundle

From repository root:

```bash
PYTHONPATH=Query/code python3 Query/code/scripts/generate_query_bundle.py \
  --csv /path/to/input.csv \
  --metadata /path/to/metadata.json \
  --output-dir /path/to/output_bundle \
  --max-queries 25
```

`metadata.json` is optional.  When provided, it may set `dataset_id`, a simple
SQLite `table_name`, and `roles` (template role to CSV-column mappings).  The
API profiles the CSV, binds compatible V8 templates, executes each candidate
against SQLite, and retains only non-empty executable queries.

## Smoke-test gate

The release gate is:

1. All 49 V8 template rows have a valid `semantic_result_contract`; every
   release dataset manifest has the same header and agrees with the release
   manifest.
2. A custom CSV produces a non-empty `query_bundle.json`, valid `manifest.json`,
   and executable SQL.
3. The same SQL runs against reference and synthetic SQLite inputs, emits
   timing fields, and produces a valid raw semantic score.
4. The website starts and can inventory the repository and preview the bundle
   JSON without exposing paths outside the repository root.

Run the unit checks with:

```bash
PYTHONPATH=Query/code python3 -m unittest discover -s Query/code/tests -v
```

Run the browser locally with `npm --prefix website run dev`, then open
`http://127.0.0.1:4173`.  The browser is a local repository explorer; it is
not a hosted public website and does not alter release files.
