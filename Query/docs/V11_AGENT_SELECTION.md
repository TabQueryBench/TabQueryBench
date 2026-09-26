# V11 agent-selected, agent-bound query grounding

V11 extends [V10](V10_AGENT_BINDING.md). In V10 the template set is fixed by rule (`all_applicable_dense`: every applicable template that V7 accepted) and the model only binds placeholders. In V11 the model first **chooses which templates to ground**, then binds each chosen template exactly as in V10.

## Template selection (`agent_selected_rule_count`)

1. Candidates: every core agent template (`subgroup_structure`, `conditional_dependency_structure`, `tail_rarity_structure`) whose roles can be bound on the dataset profile. Unlike V10, candidates are not filtered to the V7 accepted set.
2. Count rule, reused from the V2 CLI template-selection line (`Query/code/tqb_query/subitem_workload_v2/inventory.py`): `AGENT_TEMPLATE_MIN = 10`, `AGENT_TEMPLATE_TARGET = 12`, capped by the candidate count, i.e. `min = min(10, n)` and `target = max(min, min(12, n))`.
3. The model receives the dataset summary, every candidate (id, name, intent, SQL skeleton, family, roles, subitems), the min/target, and `AGENT_FAMILY_TEMPLATE_MINIMUMS` (2/4/4) as coverage guidance. It returns `{"selected_template_ids": [...]}`.
4. Validation keeps the model's order, drops invented and duplicate ids, and truncates beyond `target`. If fewer than `min` valid ids remain, the rule-selected order backfills up to `min` (mode `rule_backfill`) and the inventory records an `agent_selected_templates_below_minimum` deficit. V2 backfilled to `target`; V11 backfills only to `min` so the model's choice stays dominant.
5. The raw choice and every correction are stored in `planner_usage_summary.template_selection`. Selected templates carry `selection_mode = agent_selected | rule_backfill`.

Quota, authentication, and exhausted-retry errors during selection or binding stop the run without writing the dataset inventory, so `--resume` redoes it.

## Versions

V11 reuses the V10 slot table (see [V10 version allocation](V10_AGENT_BINDING.md#version-allocation)): `v11.x.y_<model>` is the same model as `v10.x.y_<model>`.

| Vendor | Slots |
|---|---|
| 1 Anthropic | `v11.1.1_claude-opus-5`, `v11.1.2_claude-sonnet-4-6`, `v11.1.3_claude-fable-5-1`, `v11.1.4_claude-haiku-4-5` |
| 2 Z.AI | `v11.2.1_glm-5.3-flash`, `v11.2.2_glm-5.3` |
| 3 OpenAI | `v11.3.1_gpt-5.4`, `v11.3.2_gpt-5.5`, `v11.3.3_gpt-5.6-sol`, `v11.3.4_gpt-5.6-terra`, `v11.3.5_gpt-5.6-luna` |

Repeat runs add `-<run>`: glm-5.3 has been grounded as `v11.2.2_glm-5.3`, `v11.2.2-2_glm-5.3` and `v11.2.2-3_glm-5.3`, and glm-5.3-flash as `v11.2.1_glm-5.3-flash`, `v11.2.1-2_glm-5.3-flash` and `v11.2.1-3_glm-5.3-flash`. `code/scripts/run_v11_glm_repeats_tmux.sh "glm:2 flash:2"` launches repeats; the other launchers take `RUN=2`.

Earlier names map as follows: `v11.5.2` → `v11.2.2_glm-5.3`, `v11.5.3` → `v11.2.1_glm-5.3-flash`, `v11.6.2` / `v11.7.2` → `v11.2.2-2` / `-3_glm-5.3`, `v11.6.3` / `v11.7.3` → `v11.2.1-2` / `-3_glm-5.3-flash`; the superseded `v11.3.0` execution was also glm-5.3. `v11.1.2_claude-sonnet-5` holds the sonnet-5 attempts made before slot 1.2 moved to sonnet-4.6.

**A GLM Coding Plan key reaches only these two models.** The coding endpoint silently routes older model ids to them — `glm-5 / 5.1 / 5.2` are served by `glm-5.3`, and `glm-4.5 / 4.5-air / 4.6 / 4.7 / 5-turbo` by `glm-5.3-flash` — and the response's `model` field reports the model that actually answered. Always check that field before trusting a run's provenance. `glm-5.2` therefore has no slot. Artifacts go under `workload_grounding_v11/variants/<version>` and `logs/subitem_workload_v11/variants/<version>`; a published release goes to `Queries/V<version>-full`, e.g. `Queries/V11.2.2_glm-5.3-full`.

## Commands

Run from `Query/`:

```bash
PYTHONPATH=code python3 code/scripts/build_subitem_workload_v2_inventory.py \
  --line-version v11 \
  --planner-kind agent-select-bind \
  --planner-model glm \
  --dataset-ids c2 \
  --data-root ../Synthesizing/raw_data/tabular_datasets \
  --agent-bind-problems-per-template 1 \
  --resume

PYTHONPATH=code python3 code/scripts/run_subitem_workload_v2.py \
  --line-version v11 --model glm --dataset-ids c2 \
  --data-root ../Synthesizing/raw_data/tabular_datasets --engine template
```

Provider routing (`--ai-cli-preset auto`) is the same as V10: GPT → Codex CLI, Claude → Claude CLI, GLM → Z.AI API.
