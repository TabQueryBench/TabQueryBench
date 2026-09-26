#!/usr/bin/env bash
# Ground a V11 line with a Claude model, several datasets at a time, without draining the weekly quota.
#
# One process per dataset, PARALLEL of them per round. Measured on claude-haiku-4-5: 8 concurrent
# CLI calls run ~8.7x a single call with no failures, while 16 starts returning empty output.
# The weekly quota is read once per round, between rounds, so the guard stays a single decision point
# (a round can overshoot by at most PARALLEL datasets, about 1.4% of the weekly quota).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:-haiku}"
PARALLEL="${PARALLEL:-8}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../Synthesizing/raw_data/tabular_datasets}"
WEEK_STOP_PERCENT="${WEEK_STOP_PERCENT:-90}"
SESSION_WAIT_SECONDS="${SESSION_WAIT_SECONDS:-1800}"
MAX_ROUND_RETRIES="${MAX_ROUND_RETRIES:-3}"

GROUNDING_VERSION="$(PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
from tqb_query.workload_grounding.v10_versions import format_grounding_version, resolve_model_version
print(resolve_model_version('$MODEL', line_family='v11', grounding_version=format_grounding_version('v11', '$MODEL', int('${RUN:-1}'))).grounding_version)
")"
LAUNCH_ID="${LAUNCH_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/code/logs/v11_claude_launches/${GROUNDING_VERSION}_p${PARALLEL}_$LAUNCH_ID}"
SESSION="v11_${GROUNDING_VERSION//./_}_p${PARALLEL}"
INVENTORY_DIR="$PROJECT_ROOT/code/data/workload_grounding_v11/variants/$GROUNDING_VERSION/inventories"
LOG_PATH="$LOG_ROOT/${GROUNDING_VERSION}.log"
STATUS_PATH="$LOG_ROOT/${GROUNDING_VERSION}.status"

remaining_datasets() {
  PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
import os
from tqb_query.subitem_workload_v2.paths import default_dataset_ids_for_line_version
done = set()
d = '$INVENTORY_DIR'
if os.path.isdir(d):
    done = {f.split('_inventory_')[0] for f in os.listdir(d) if '_inventory_' in f}
print(' '.join(x for x in default_dataset_ids_for_line_version('v11') if x not in done))
"
}

weekly_used_percent() {
  local dir="$LOG_ROOT/usage/$(date +%H%M%S)"
  mkdir -p "$dir"
  if ! timeout 180 python3 "$PROJECT_ROOT/code/scripts/capture_claude_usage.py" --output-dir "$dir" > "$dir/capture.log" 2>&1; then
    echo -1
    return
  fi
  python3 -c "
import glob, json, sys
files = sorted(glob.glob('$dir/*summary.json'))
if not files:
    print(-1); sys.exit()
print(int(((json.load(open(files[-1])).get('week_all_models') or {}).get('used_percent')) or -1))
"
}

run_inner() {
  cd "$PROJECT_ROOT"
  trap 'echo "[interrupted] at $(date -Is)"; echo 130 > "$STATUS_PATH"; exit 130' INT TERM HUP
  local round=0 retries=0 used
  while true; do
    read -r -a pending <<< "$(remaining_datasets)"
    if (( ${#pending[@]} == 0 )); then
      echo "[done] all datasets grounded at $(date -Is)"
      echo 0 > "$STATUS_PATH"
      return 0
    fi

    used="$(weekly_used_percent)"
    if (( used < 0 )); then
      echo "[warn] could not read Claude usage; stopping to stay safe at $(date -Is)"
      echo 3 > "$STATUS_PATH"
      return 3
    fi
    echo "[quota] weekly used=${used}% (stop at ${WEEK_STOP_PERCENT}%), remaining=${#pending[@]} at $(date -Is)"
    if (( used >= WEEK_STOP_PERCENT )); then
      echo "[stop] weekly quota below $((100 - WEEK_STOP_PERCENT))% left; ${#pending[@]} datasets still pending"
      echo 2 > "$STATUS_PATH"
      return 2
    fi

    round=$((round + 1))
    local group=("${pending[@]:0:$PARALLEL}")
    local round_dir="$LOG_ROOT/round_${round}"
    mkdir -p "$round_dir"
    echo "[round $round] starting ${#group[@]} datasets in parallel: ${group[*]} at $(date -Is)"
    local started epoch_start
    epoch_start=$(date +%s)
    local pids=()
    for ds in "${group[@]}"; do
      PYTHONUNBUFFERED=1 PYTHONPATH=code nice -n 10 ionice -c3 python3 code/scripts/build_subitem_workload_v2_inventory.py \
        --line-version v11 --planner-kind agent-select-bind --planner-model "$MODEL" --grounding-version "$GROUNDING_VERSION" \
        --data-root "$DATA_ROOT" --agent-bind-problems-per-template 1 --resume --dataset-ids "$ds" \
        > "$round_dir/$ds.log" 2>&1 &
      pids+=($!)
    done
    local failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=$((failed + 1))
    done
    local finished
    finished=$(( $(date +%s) - epoch_start ))
    read -r -a still_pending <<< "$(remaining_datasets)"
    echo "[round $round] finished in ${finished}s, failures=$failed, remaining=${#still_pending[@]} at $(date -Is)"

    if (( failed == 0 )); then
      retries=0
      continue
    fi
    if grep -lq "PlannerQuotaError" "$round_dir"/*.log 2>/dev/null; then
      echo "[session-limit] waiting ${SESSION_WAIT_SECONDS}s before retrying the round"
      sleep "$SESSION_WAIT_SECONDS"
      continue
    fi
    retries=$((retries + 1))
    for log in "$round_dir"/*.log; do
      grep -E '^[A-Za-z_.]+(Error|Exception)' "$log" | tail -1 | sed "s|^|  $(basename "$log" .log): |" || true
    done
    if (( retries > MAX_ROUND_RETRIES )); then
      echo "[abort] $retries consecutive failing rounds"
      echo 1 > "$STATUS_PATH"
      return 1
    fi
    echo "[retry] failing round $retries/$MAX_ROUND_RETRIES, waiting 120s"
    sleep 120
  done
}

if [[ "${1:-}" == "--inner" ]]; then
  run_inner
  exit $?
fi

mkdir -p "$LOG_ROOT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[skip] tmux session already exists: $SESSION"
  exit 0
fi
rm -f "$STATUS_PATH"
printf -v inner_cmd 'MODEL=%q PARALLEL=%q LOG_ROOT=%q DATA_ROOT=%q WEEK_STOP_PERCENT=%q SESSION_WAIT_SECONDS=%q MAX_ROUND_RETRIES=%q bash %q --inner >> %q 2>&1' \
  "$MODEL" "$PARALLEL" "$LOG_ROOT" "$DATA_ROOT" "$WEEK_STOP_PERCENT" "$SESSION_WAIT_SECONDS" "$MAX_ROUND_RETRIES" \
  "$PROJECT_ROOT/code/scripts/run_v11_claude_parallel_tmux.sh" "$LOG_PATH"
tmux new-session -d -s "$SESSION" "$inner_cmd"
echo "[launched] model=$MODEL version=$GROUNDING_VERSION parallel=$PARALLEL session=$SESSION"
echo "[log]      $LOG_PATH"
echo "[guard]    stops when weekly usage reaches ${WEEK_STOP_PERCENT}%"
