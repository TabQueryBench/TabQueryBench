#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../Synthesizing/raw_data/tabular_datasets}"
DATASET_IDS="${DATASET_IDS:-}"
PROBLEMS_PER_TEMPLATE="${PROBLEMS_PER_TEMPLATE:-1}"
LAUNCH_ID="${LAUNCH_ID:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/code/logs/v10_claude_launches/$LAUNCH_ID}"

mkdir -p "$LOG_ROOT"

MODELS=(
  "opus5:v10.1.1"
  "sonnet:v10.1.2"
  "fable:v10.1.3"
  "haiku:v10.1.4"
)

for spec in "${MODELS[@]}"; do
  model="${spec%%:*}"
  version="${spec##*:}"
  session="v10_${version//./_}_${model}"
  log_path="$LOG_ROOT/${version}_${model}.log"
  status_path="$LOG_ROOT/${version}_${model}.status"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] tmux session already exists: $session"
    continue
  fi

  # A previous failed/restarted session may have left a terminal status file.
  # Its absence means the current detached job is still running.
  rm -f "$status_path"

  printf -v dataset_arg '%q' "$DATASET_IDS"
  printf -v project_arg '%q' "$PROJECT_ROOT"
  printf -v data_arg '%q' "$DATA_ROOT"
  printf -v log_arg '%q' "$log_path"
  printf -v status_arg '%q' "$status_path"
  printf -v launch_arg '%q' "$LAUNCH_ID"
  printf -v model_arg '%q' "$model"
  printf -v version_arg '%q' "$version"
  printf -v count_arg '%q' "$PROBLEMS_PER_TEMPLATE"

  command="cd $project_arg && { "
  command+="trap 'rc=130; echo \"[interrupted] signal received at \$(date -Is)\"; echo \"\$rc\" > $status_arg; exit \$rc' INT TERM HUP; "
  command+="echo \"[start] launch=$launch_arg model=$model_arg version=$version_arg at=\$(date -Is)\"; "
  command+="PYTHONUNBUFFERED=1 PYTHONPATH=code python3 code/scripts/build_subitem_workload_v2_inventory.py "
  command+="--line-version v10 --grounding-version $version_arg --planner-kind agent-bind "
  command+="--planner-model $model_arg --ai-cli-preset auto --data-root $data_arg "
  command+="--agent-bind-problems-per-template $count_arg"
  command+=" --resume"
  if [[ -n "$DATASET_IDS" ]]; then
    command+=" --dataset-ids $dataset_arg"
  fi
  command+="; rc=\$?; echo \"[end] rc=\$rc at=\$(date -Is)\"; echo \"\$rc\" > $status_arg; exit \$rc; "
  command+="} >> $log_arg 2>&1"

  tmux new-session -d -s "$session" "$command"
  echo "[launched] session=$session model=$model version=$version log=$log_path"
done

echo "[launch-root] $LOG_ROOT"
