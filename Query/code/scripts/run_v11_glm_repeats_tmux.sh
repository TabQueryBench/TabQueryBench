#!/usr/bin/env bash
# Launch repeat V11 grounding runs for the Z.AI GLM models, several runs at once.
#
# Each run is one (model, run number) pair and is split into SHARDS tmux sessions that each
# ground a slice of the 49 datasets, so a run finishes in roughly 1/SHARDS of the time.
# Total concurrent Z.AI calls is RUNS x SHARDS. On this plan 8 held for hours without a
# rate-limit error, while 12 drew HTTP 429 code 1302 within the first minute.
#
#   bash run_v11_glm_repeats_tmux.sh "glm:2 glm:3 flash:2 flash:3"
#
# Each spec is model:run; glm:2 grounds v11.2.2-2_glm-5.3 (see v10_versions.py).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SPECS="${1:?usage: run_v11_glm_repeats_tmux.sh \"model:run model:run ...\"}"
SHARDS="${SHARDS:-2}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../Synthesizing/raw_data/tabular_datasets}"
LAUNCH_ID="${LAUNCH_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/code/logs/v11_glm_launches/repeats_$LAUNCH_ID}"

mkdir -p "$LOG_ROOT"
echo "[plan] shards per run=$SHARDS, log root=$LOG_ROOT"

for spec in $SPECS; do
  model="${spec%%:*}"
  run="${spec##*:}"
  version="$(PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
from tqb_query.workload_grounding.v10_versions import format_grounding_version
print(format_grounding_version('v11', '$model', int('$run')))
")"

  mapfile -t pending < <(PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
import os
from tqb_query.subitem_workload_v2.paths import default_dataset_ids_for_line_version
done = set()
d = '$PROJECT_ROOT/code/data/workload_grounding_v11/variants/$version/inventories'
if os.path.isdir(d):
    done = {f.split('_inventory_')[0] for f in os.listdir(d) if '_inventory_' in f}
print('\n'.join(x for x in default_dataset_ids_for_line_version('v11') if x not in done))
")
  if (( ${#pending[@]} == 0 )); then
    echo "[skip] $model $version already complete"
    continue
  fi

  for (( shard=0; shard<SHARDS; shard++ )); do
    slice=()
    for (( i=shard; i<${#pending[@]}; i+=SHARDS )); do
      slice+=("${pending[$i]}")
    done
    (( ${#slice[@]} == 0 )) && continue
    ids="$(IFS=,; echo "${slice[*]}")"
    session="v11_${version//./_}_s${shard}"
    log="$LOG_ROOT/${version}_s${shard}.log"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "[skip] session exists: $session"
      continue
    fi
    tmux new-session -d -s "$session" "cd $(printf '%q' "$PROJECT_ROOT") && \
PYTHONUNBUFFERED=1 PYTHONPATH=code nice -n 10 python3 code/scripts/build_subitem_workload_v2_inventory.py \
  --line-version v11 --planner-kind agent-select-bind --planner-model $(printf '%q' "$model") \
  --grounding-version $(printf '%q' "$version") --dataset-ids $(printf '%q' "$ids") \
  --data-root $(printf '%q' "$DATA_ROOT") --agent-bind-problems-per-template 1 --resume \
  > $(printf '%q' "$log") 2>&1; echo \$? > $(printf '%q' "${log%.log}.status")"
    echo "[launched] $session: ${#slice[@]} datasets -> $log"
  done
done

echo "[status] tmux ls | grep v11_"
