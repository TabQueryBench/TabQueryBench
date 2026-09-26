# Query_Templates

`Query_Templates` is the public release view of the template library and supporting metadata used to construct SQL/query workloads.

## Generate a bundle for your CSV

The public v8 registry can be used without the benchmark scorer. See
[`code/README.md`](../code/README.md) for the `generate_query_bundle.py` CLI,
Python API, metadata format, and the versioned output schema. The generator executes
candidate SQL locally in SQLite and records all bindings, source template provenance,
input hashes, timing, and usage in its manifest.

## Recommended Reading Order

1. `templates/core/`
   Core template library, split by query family.
2. `../../Scoring/Scoring_Standard/`
   Versioned scoring standards (legacy composite, semantic v8, standard v1, SPQ v9, sv2).
3. `templates/extensions/`
   Optional extension templates.
5. `templates/policy/`
   Can-vary / must-fix policy rules.
6. `mappings/`
   Workload and family mapping tables.
7. `evidence/`
   Template derivation evidence.
8. `candidate_specs/` and `reports/`
   Agent shortlist files and supporting reports.

## Directory Layout

- `templates/core/subgroup/`
  Core subgroup templates. Current release file:
  `template_library_subgroup_v1.jsonl`; semantic v8 file:
  `template_library_subgroup_v8.jsonl`.
- `templates/core/conditional/`
  Core conditional templates. Current release file:
  `template_library_conditional_v1.jsonl`; semantic v8 file:
  `template_library_conditional_v8.jsonl`.
- `templates/core/tail/`
  Core tail templates. Current release file:
  `template_library_tail_v1.jsonl`; semantic v8 file:
  `template_library_tail_v8.jsonl`.
- `templates/core/cardinality/`
  Core cardinality templates covering discrete support and continuous range preservation.
  Current release file: `template_library_cardinality_v1.jsonl` with:
  - `tpl_cardinality_discrete_support_v1`
  - `tpl_cardinality_continuous_range_v1`
  Semantic v8 file: `template_library_cardinality_v8.jsonl`.
- `templates/core/missing/`
  Core missingness templates covering marginal missing-rate consistency and broad
  state-conditioned missingness. Current release file:
  `template_library_missing_v1.jsonl` with:
  - `tpl_missing_rate_marginal_v1`
  - `tpl_missing_rate_by_discrete_state_v1`
  - `tpl_missing_rate_by_continuous_bucket_v1`
  Semantic v8 file: `template_library_missing_v8.jsonl`.
- `templates/extensions/`
  Extension template library.
- `templates/policy/`
  Template policy rules.
- `mappings/`
  Workload catalog, workload-to-family mapping, and source query bank.
- `evidence/`
  Evidence tables showing how templates were derived.
- `candidate_specs/`
  Curated agent candidate template sets.
- `reports/portability/`
  Portability reports.
- `reports/selection/`
  Template-selection summary reports.

## Excluded Content

This folder intentionally excludes:

- dataset-specific inventories
- full question inventories
- runtime execution outputs
- provenance-only records
- internal workflow logs

Those materials belong to the public `Query/Queries/` release or to internal workflow artifacts rather than the template library itself.

## Family Coverage In This Release

- `subgroup`
  Released as a dedicated family-specific core library. The v8 semantic release
  contains 16 subgroup templates.
- `conditional`
  Released as a dedicated family-specific core library. The v8 semantic release
  contains 15 conditional templates.
- `tail`
  Released as a dedicated family-specific core library. The v8 semantic release
  contains 11 tail/rareness templates.
- `cardinality`
  Released as a dedicated family-specific core library with 2 v1 templates and
  4 v8 semantic templates.
- `missingness`
  Released as a dedicated family-specific core library with 3 templates.

## Semantic V8 Release

The v8 template release adds explicit semantic metadata to each template,
including result contracts, scorer type hints, key and measure output roles,
and role-distinctness constraints. These fields are intended to make query
generation and evaluation less dependent on column-name guessing.

The merged v8 library is available at
`Query/code/data/workload_grounding_v8/template_library_v8.jsonl`; the same 49
templates are also split by family under `templates/core/*/*_v8.jsonl`.
Dataset-specific inventories, query registries, SQL files, and runtime outputs
remain outside `Query_Templates` and belong in the public `Query/Queries/` release.

The v8 semantic scoring contract is summarized in
`Scoring/Scoring_Standard/semantic_v8/SEMANTIC_SCORING_V8.md`. Current run3 query analysis outputs that use this
contract are published under `Synthesizing/synthetic_data/run3/query_analysis/`.
The evaluator writes both legacy and semantic query scores; use
`--use-semantic-query-score` to make the semantic score the primary
`query_score`/`overall_score` while retaining the legacy columns for comparison.

## Semantic V9 Contract

The v9 scoring contract is named **SPQ** (**TabQueryBench Single-Primary
Query Similarity v9**). It replaces v8 weighted semantic blends with a single
primary similarity per query:

```text
query_score = primary_semantic_similarity
```

The six active scorer types are `scalar`, `count_support_distribution`,
`rate_share_proportion`, `ratio`, `keyed_numeric_aggregate`, and
`topk_ranking`. Diagnostic scores remain available but do not affect the
primary score. See `Scoring/Scoring_Standard/spq_v9/SEMANTIC_SCORING_V9_SINGLE_PRIMARY.md`.
