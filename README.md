# TabQueryBench

TabQueryBench evaluates tabular synthetic data by running workload-grounded SQL
queries on real and synthetic tables and scoring how well the synthetic answers
match the real ones. The repository is organized into three parts; each part
keeps its own code inside its folder.

| Part | Contents |
| --- | --- |
| [`Synthesizing/`](Synthesizing/README.md) | Real datasets, synthetic data generation code, generated synthetic data |
| [`Query/`](Query/README.md) | Query templates, template sources, released queries (V2–V9), query generation code, runs, and audits |
| [`Scoring/`](Scoring/README.md) | Scoring standards (legacy → semantic → single-primary SPQ → sv2), scoring/evaluation code, and scoring results |

Shared, cross-cutting material:

- `docs/PIPELINE_OVERVIEW.md` — end-to-end pipeline and artifact contracts.
- `website/` — local repository browser (`npm --prefix website run dev`).

## Python packages

| Package | Location | Depends on |
| --- | --- | --- |
| `tqb_query` | `Query/code/tqb_query` | — |
| `tqb_scoring` | `Scoring/code/tqb_scoring` | `tqb_query` (added to `sys.path` automatically) |
| generation runner (`core`, `models`) | `Synthesizing/code/src` | — |

```bash
# query generation / query bundle API
PYTHONPATH=Query/code python3 -m unittest discover -s Query/code/tests
# scoring
PYTHONPATH=Scoring/code python3 -m tqb_scoring.eval.analysis.runner --help
```

Before the reorganization the code lived in a single `code/src` package
(`src.*`). Module paths map as follows: `src.eval.subitem_workload_v2`,
`src.eval.analytics_contract`, `src.eval.token_usage_v1` and
`src.{agent,benchmark,config,data,db,logging,usage,query_generation,workload_grounding}`
→ `tqb_query.*`; `src.eval.*` and `src.evaluation.*` → `tqb_scoring.eval.*` and
`tqb_scoring.evaluation.*`.

## Local compatibility links

Existing external jobs (for example SQLagent evaluation and scripts under
`/tmp`) reference the old paths. The following untracked symlinks are kept
locally and ignored by git:

- `raw_data` → `Synthesizing/raw_data`
- `synthetic_data` → `Synthesizing/synthetic_data`
- `code/Evaluation` → `Scoring/results`
- `code/data` → `Query/code/data`
- `code/logs` → `Query/code/logs`
