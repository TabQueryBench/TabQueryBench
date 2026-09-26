# Query

Query templates, the source material they were derived from, the released
benchmark queries, query generation code, and query generation runs.

```text
Query/
  Query_Templates/      template library, policies, mappings, derivation evidence
  Reference/            third-party source SQL used to derive the templates
  Queries/              released queries, one folder per version
    V2-gpt-5.4-partial … V7-gpt-5.4-mini-full   agent-generated releases
    V8-template-full, V9-template-full          template-engine releases
      <dataset>/sql/      grounding inventories, registries, SQL files
      <dataset>/analysis/ per-dataset query counts (V5–V7 also contain legacy scores)
  code/
    tqb_query/          Python package: template binding, SQL execution, workload grounding,
                        subitem workload inventories, query generation API
    scripts/  tests/  schemas/  README.md  requirements.txt
    data/workload_grounding_v8|v9|v10/   template libraries, inventories, registries
    logs/               query generation runs (v8 → SQLagent symlink, v9, v10)
    Evaluation/         query generation evaluation outputs (subitem_workload_v9, v10)
  tools/                V8/V9 materialization, audit, export, and release scripts
  reports/              V8 query audits and materialization report
  exports/              template question/SQL exports (zip and tar.gz)
  experiments/          SQL/question consistency study
  docs/                 V10 agent binding notes
```

Common commands (from the repository root):

```bash
PYTHONPATH=Query/code python3 -m unittest discover -s Query/code/tests
python3 Query/code/scripts/generate_query_bundle.py --csv data.csv --output-dir out/
```

The runners resolve their working directories relative to `Query/code`
(`tqb_query.config.settings.PROJECT_ROOT`). Real datasets are read from
`Synthesizing/raw_data/tabular_datasets`.
