# V10 agent-bound query grounding

V10 keeps each template SQL skeleton immutable. The model receives the skeleton, the complete dataset schema/profile, observed value candidates, numeric candidates, and template constraints. It may return only:

```json
{"bindings": {"group_col": "...", "target_col": "...", "target_value": "..."}}
```

The pipeline strictly validates the response and rejects missing, extra, invented, or constraint-breaking bindings. Accepted bindings are rendered into SQL deterministically and executed through the existing template runner. Raw model bindings, validation results, model/version provenance, prompts, responses, and usage are retained for comparison.

## Version allocation

V10 and V11 share one slot table, so `v10.1.1_claude-opus-5` and `v11.1.1_claude-opus-5` are the same model. A version reads `<line>.<vendor>.<model>[-<run>]_<model name>`: the vendor digit is 1 Anthropic, 2 Z.AI, 3 OpenAI, and the model name is written into the version so every directory, file and score row names the model that produced it.

| Model (alias) | V10 version | Provider |
|---|---|---|
| `claude-opus-5` (`opus5`) | `v10.1.1_claude-opus-5` | Claude CLI |
| `claude-sonnet-4-6` (`sonnet`) | `v10.1.2_claude-sonnet-4-6` | Claude CLI |
| `claude-fable-5-1` (`fable`) | `v10.1.3_claude-fable-5-1` | Claude CLI |
| `claude-haiku-4-5` (`haiku`) | `v10.1.4_claude-haiku-4-5` | Claude CLI |
| `glm-5.3-flash` (`flash`) | `v10.2.1_glm-5.3-flash` | Z.AI API |
| `glm-5.3` (`glm`) | `v10.2.2_glm-5.3` | Z.AI API |
| `gpt-5.4` | `v10.3.1_gpt-5.4` | Codex CLI |
| `gpt-5.5` | `v10.3.2_gpt-5.5` | Codex CLI |
| `gpt-5.6-sol` | `v10.3.3_gpt-5.6-sol` | Codex CLI |
| `gpt-5.6-terra` | `v10.3.4_gpt-5.6-terra` | Codex CLI |
| `gpt-5.6-luna` | `v10.3.5_gpt-5.6-luna` | Codex CLI |

A repeat run of the same model adds `-<run>` after the slot: the second glm-5.3 run is `v10.2.2-2_glm-5.3`, the third `v10.2.2-3_glm-5.3`. The first run carries no suffix. Pass the run with `--grounding-version v10.2.2-2` (the model part may be left off) or `RUN=2` for the tmux launchers.

The existing `v10.1.2_claude-sonnet-5` artifacts were produced by `claude-sonnet-5` before slot 1.2 moved to `claude-sonnet-4-6`. They keep that name (`LEGACY_MODEL_SLOTS`), so they never mix with sonnet-4.6 runs.

The table is `MODEL_VERSION_SLOTS` in `Query/code/tqb_query/workload_grounding/v10_versions.py`; a version whose model does not own its slot is rejected. Model artifacts are isolated under `variants/<version>`.

## Generate bindings

Start with one dataset and one binding per applicable template:

Run the commands below from the `Query/` directory.

```bash
PYTHONPATH=code python3 code/scripts/build_subitem_workload_v2_inventory.py \
  --line-version v10 \
  --planner-kind agent-bind \
  --planner-model gpt-5.5 \
  --dataset-ids c2 \
  --data-root ../Synthesizing/raw_data/tabular_datasets \
  --agent-bind-problems-per-template 1
```

`--ai-cli-preset auto` is the default for this command: GPT models route to Codex and Claude models route to Claude CLI. Use the same command with `opus5`, `sonnet`, `fable`, or `haiku`; the output directory and metadata resolve to the corresponding version automatically. Omitting `--dataset-ids` selects the complete V10 dataset set.

### GLM-5.3 via Z.AI

`--planner-model glm` routes to `ZAIProblemPlanner`, which calls `https://api.z.ai/api/paas/v4/chat/completions` with model `glm-5.3` through the OpenAI Python SDK (`pip install openai`). The client lives in `Query/code/tqb_query/llm/zai.py`.

- Endpoint: `ZAI_API_BASE=coding` (default, `https://api.z.ai/api/coding/paas/v4/`, GLM Coding Plan quota) or `ZAI_API_BASE=general` (`https://api.z.ai/api/paas/v4/`, pay-as-you-go balance). A GLM Coding Plan key has no general-endpoint balance and gets `429 code=1113 Insufficient balance` for `glm-5.3` there; it must use `coding`. Z.AI restricts Coding Plan quota to supported coding tools, so check the plan's usage policy before batch runs.
- The key is read from `ZAI_API_KEY`; if unset, from `ZAI_API_KEY_FILE` (default `~/.config/zai/env`, containing `export ZAI_API_KEY=...`). Never commit or log the key.
- Requests use `response_format=json_object`, `temperature=0.2` (`ZAI_PLANNER_TEMPERATURE`), non-streaming. GLM-5.3 always thinks and rejects `thinking.type=disabled`; set depth with `ZAI_REASONING_EFFORT=low|high|max` (unset = server default `max`).
- Rate limits/overload (HTTP 429 codes 1302/1305, 5xx, timeouts) are retried with backoff up to `ZAI_MAX_ATTEMPTS` (default 5). Balance/plan exhaustion (1113, 1308-1321) stops the run as a quota error; auth failures and exhausted retries also stop it, so `--resume` redoes the unfinished dataset. HTTP 400 rejections are recorded as a failed binding for that template only.

## Render and execute accepted bindings

```bash
PYTHONPATH=code python3 code/scripts/run_subitem_workload_v2.py \
  --line-version v10 \
  --model gpt-5.5 \
  --dataset-ids c2 \
  --data-root ../Synthesizing/raw_data/tabular_datasets \
  --engine template
```

The model resolves to its version (`v10.3.2_gpt-5.5` here); add `--grounding-version v10.3.2-2` to execute a repeat run. Registries, SQL files and evaluation provenance stay isolated per version.
