# TabQueryBench query-bundle API

Generate executable, template-grounded SQL query bundles for a CSV without using the
benchmark scoring pipeline or the website.

## Quick start

From the repository root (the command adds `Query/code` to `sys.path` itself):

```bash
python Query/code/scripts/generate_query_bundle.py \
  --csv /path/to/orders.csv \
  --metadata /path/to/orders.metadata.json \
  --output-dir /path/to/orders-query-bundle
```

The command uses `Query/code/data/workload_grounding_v8/template_library_v8.jsonl` by
default. Each registry template is bound to the input schema and executed in SQLite;
only non-empty, executable queries are retained. No network or LLM calls are made.

Metadata is optional. When present, it may select stable role bindings rather than
relying on inference:

```json
{
  "dataset_id": "orders_2026",
  "table_name": "orders",
  "roles": {
    "group_col": "region",
    "group_col_2": "channel",
    "measure_col": "revenue",
    "entity_col": "customer_id",
    "predicate_col": "status",
    "condition_col": "returned",
    "missing_col": "coupon_code"
  }
}
```

`roles` values must be CSV column names. Omitted roles are inferred from CSV types,
cardinality, and observed values. A generated `input.sqlite` is included only to make
the generated SQL immediately runnable; it is not part of the logical bundle format.

## Python API

```python
from pathlib import Path
from tqb_query.workload_grounding import generate_query_bundle

result = generate_query_bundle(
    csv_path=Path("orders.csv"),
    metadata_path=Path("orders.metadata.json"),  # optional
    output_dir=Path("out/orders"),
    max_queries=25,
)
print(result["bundle_path"])
```

## Output contract

The output directory contains:

- `query_bundle.json` — canonical bundle (`tabquerybench.query_bundle.v1`)
- `manifest.json` — copied manifest for automation
- `selected_queries.sql` — accepted SQL, each tagged with `template_id`
- `input.sqlite` — SQLite materialization used to test the SQL

The machine-readable schema is [schemas/query_bundle_v1.schema.json](schemas/query_bundle_v1.schema.json).
The manifest records input and registry SHA-256s, SQLite version/table/row count,
zeroed usage counters, timings, and runtime environment. Each query carries its
template ID, source provenance, template hash, actual role bindings, semantic result
contract, output column names, and up to five preview rows. Templates that cannot be
bound, execute, or return rows appear in `skipped_templates` with a reason.

To use another compatible registry, pass `--template-library PATH`. The CLI exits
non-zero for missing inputs, invalid metadata, empty CSVs, or invalid headers.

## AI-grounded query generation service

The service accepts an arbitrary single-table CSV, builds a typed dataset
profile, considers every structurally applicable template, and asks a configured
chat model to select only the template bindings (columns, observed values, and
bounded parameters). The model cannot select or rewrite templates. Bindings are
validated before deterministic SQL rendering, and only read-only, executable,
non-empty queries are included in the result.

Configure a model supported by LangChain, then start the API:

```bash
export QUERY_GENERATION_PROVIDER=langchain
export QUERY_GENERATION_MODEL=openai:gpt-5.4
python Query/code/scripts/serve_query_generation_api.py
```

For local development with an installed Claude CLI, use
`QUERY_GENERATION_PROVIDER=claude-cli` and set `QUERY_GENERATION_MODEL` to the
desired CLI model name. This adapter disables Claude tools and does not invoke a
shell; hosted deployments should use the direct API-backed provider.

Create a job with `POST /v1/query-generation-jobs` using multipart form data.
The `dataset_file` field is required; optional fields include `dataset_id`,
`target_column`, `excluded_columns`, and `bindings_per_template`. Poll
`GET /v1/query-generation-jobs/{job_id}` and retrieve the completed bundle from
`GET /v1/query-generation-jobs/{job_id}/result`.

Every job uses an isolated working directory. The public bundle contains hashes
and model provenance but no server filesystem paths or raw model responses.
The built-in job manager is intentionally single-process. Run one API process per
job directory; use an external durable queue before scaling to multiple API/worker
processes. Set request-body limits at the reverse proxy in addition to the
application's upload, row, column, timeout, and pending-job limits.
