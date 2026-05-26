# subitem_workload_v2

This package holds the agreed design-time contracts for the new v2 subitem
workload line. It is intentionally separated from the legacy loader path so the
future regeneration work can proceed without mixing old and new SQL artifacts.

Current scope:

- template-to-family and template-to-subitem contract tables
- deterministic family enumeration rules
- v2 query registry field definitions
- CSV/Markdown export helper for review and downstream implementation

Expected neighboring artifact roots:

- `data/workload_grounding_v2/contracts/`
- `logs/subitem_workload_v2/`
- `Evaluation/subitem_workload_v2/final/`
