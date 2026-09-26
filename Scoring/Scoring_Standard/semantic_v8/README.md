# Semantic scoring V8

- `semantic_query_score_method`: `semantic_scorer_v1_parallel`
- `score_contract_version`: `real_vs_synthetic_sql_semantic_v1_parallel`
- Definition: [`SEMANTIC_SCORING_V8.md`](SEMANTIC_SCORING_V8.md); design notes:
  [`SCORING_REDESIGN_OUTLINE.md`](SCORING_REDESIGN_OUTLINE.md)
- Implementation: `scorer.py`, copied unchanged from
  `/mnt/nas/jialinzhang/SQLagent/src/evaluation/query_semantic_scoring.py`
  (sha256 `219ab011ecd7fdd3305445098ec0447089e996b5c19476edd5836645321cf7c4`).
- Template scorer hints: `Query/Query_Templates/templates/policy/template_scoring_policy_v1.jsonl`
  and the `semantic_result_contract` fields in the V8 template library.

Each query is routed to a scorer type (`scalar`, `count_support_distribution`,
`distribution_cardinality_profile`, `rate_share_proportion`, `ratio`,
`keyed_numeric_aggregate`, `topk_tailk_ranking`, `tail_outlier_distribution`,
`temporal_shape`); component scores are combined with the template's weight
rule (`100`, `50_50`, `50_25_25`, `one_third_each`).

`spq_v9` imports helper functions from this implementation.

## Results

- V8 run3 query analysis: `Synthesizing/synthetic_data/run3/query_analysis/`
  (identical copies in `Synthesizing/synthetic_data/query_analysis/` and
  `Synthesizing/synthetic_data/synthetic_data/run3/query_analysis/`).
