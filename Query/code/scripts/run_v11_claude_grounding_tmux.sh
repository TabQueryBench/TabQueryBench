#!/usr/bin/env bash
# Run V11 (agent-select-bind) with a Claude model, stopping before the weekly subscription quota runs out.
#
# Datasets are grounded in small batches so the weekly quota can be checked between them: once
# `Current week (all models)` reaches WEEK_STOP_PERCENT, the job stops instead of draining the quota.
# A 5-hour session limit is different: that one refills on its own, so the job waits and retries.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:-haiku}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../Synthesizing/raw_data/tabular_datasets}"
BATCH_SIZE="${BATCH_SIZE:-3}"
WEEK_STOP_PERCENT="${WEEK_STOP_PERCENT:-90}"   # stop when less than 10% of the weekly quota is left
SESSION_WAIT_SECONDS="${SESSION_WAIT_SECONDS:-1800}"
MAX_ERROR_RETRIES="${MAX_ERROR_RETRIES:-3}"

GROUNDING_VERSION="$(PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
from tqb_query.workload_grounding.v10_versions import format_grounding_version, resolve_model_version
print(resolve_model_version('$MODEL', line_family='v11', grounding_version=format_grounding_version('v11', '$MODEL', int('${RUN:-1}'))).grounding_version)
")"
LAUNCH_ID="${LAUNCH_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/code/logs/v11_claude_launches/${GROUNDING_VERSION}_$LAUNCH_ID}"
SESSION="v11_${GROUNDING_VERSION//./_}"
INVENTORY_DIR="$PROJECT_ROOT/code/data/workload_grounding_v11/variants/$GROUNDING_VERSION/inventories"
LOG_PATH="$LOG_ROOT/${GROUNDING_VERSION}.log"
STATUS_PATH="$LOG_ROOT/${GROUNDING_VERSION}.status"
USAGE_DIR="$LOG_ROOT/usage"

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

# Prints the weekly percentage already used, or -1 when the usage probe fails.
weekly_used_percent() {
  local dir="$USAGE_DIR/$(date +%H%M%S)"
  mkdir -p "$dir"
  if ! timeout 180 python3 "$PROJECT_ROOT/code/scripts/capture_claude_usage.py" --output-dir "$dir" > "$dir/capture.log" 2>&1; then
    echo -1
    return
  fi
  PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
import glob, json, sys
files = sorted(glob.glob('$dir/*summary.json'))
if not files:
    print(-1); sys.exit()
d = json.load(open(files[-1]))
print(int(((d.get('week_all_models') or {}).get('used_percent')) or -1))
"
}

run_inner() {
  cd "$PROJECT_ROOT"
  trap 'echo "[interrupted] at $(date -Is)"; echo 130 > "$STATUS_PATH"; exit 130' INT TERM HUP
  local errors=0 rc batch used
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
    echo "[quota] weekly used=${used}% (stop at ${WEEK_STOP_PERCENT}%), remaining datasets=${#pending[@]} at $(date -Is)"
    if (( used >= WEEK_STOP_PERCENT )); then
      echo "[stop] weekly quota below $((100 - WEEK_STOP_PERCENT))% left; ${#pending[@]} datasets still pending"
      echo 2 > "$STATUS_PATH"
      return 2
    fi

    batch="$(printf '%s,' "${pending[@]:0:$BATCH_SIZE}")"
    batch="${batch%,}"
    echo "[start] model=$MODEL version=$GROUNDING_VERSION batch=$batch at=$(date -Is)"
    set +e
    PYTHONUNBUFFERED=1 PYTHONPATH=code nice -n 10 ionice -c3 python3 code/scripts/build_subitem_workload_v2_inventory.py \
      --line-version v11 --planner-kind agent-select-bind --planner-model "$MODEL" --grounding-version "$GROUNDING_VERSION" \
      --data-root "$DATA_ROOT" --agent-bind-problems-per-template 1 --resume --dataset-ids "$batch" \
      > "$LOG_ROOT/batch_$(date +%H%M%S).log" 2>&1
    rc=$?
    set -e
    echo "[end] batch=$batch rc=$rc at=$(date -Is)"
    if (( rc == 0 )); then
      errors=0
      continue
    fi

    local last_log last_error
    last_log="$(ls -t "$LOG_ROOT"/batch_*.log | head -1)"
    last_error="$(grep -E '^[A-Za-z_.]+(Error|Exception)' "$last_log" | tail -n 1 || true)"
    cat "$last_log" | tail -5
    if [[ "$last_error" == *PlannerQuotaError* ]]; then
      # Session limits refill within hours; the weekly guard above decides whether waiting is worth it.
      echo "[session-limit] waiting ${SESSION_WAIT_SECONDS}s: $last_error"
      sleep "$SESSION_WAIT_SECONDS"
      continue
    fi
    errors=$((errors + 1))
    if (( errors > MAX_ERROR_RETRIES )); then
      echo "[abort] $errors consecutive failures: $last_error"
      echo "$rc" > "$STATUS_PATH"
      return "$rc"
    fi
    echo "[retry] failure $errors/$MAX_ERROR_RETRIES: $last_error"
    sleep 120
  done
}

if [[ "${1:-}" == "--inner" ]]; then
  run_inner
  exit $?
fi

mkdir -p "$LOG_ROOT" "$USAGE_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[skip] tmux session already exists: $SESSION"
  exit 0
fi
rm -f "$STATUS_PATH"
printf -v inner_cmd 'MODEL=%q LOG_ROOT=%q DATA_ROOT=%q BATCH_SIZE=%q WEEK_STOP_PERCENT=%q SESSION_WAIT_SECONDS=%q MAX_ERROR_RETRIES=%q bash %q --inner >> %q 2>&1' \
  "$MODEL" "$LOG_ROOT" "$DATA_ROOT" "$BATCH_SIZE" "$WEEK_STOP_PERCENT" "$SESSION_WAIT_SECONDS" "$MAX_ERROR_RETRIES" \
  "$PROJECT_ROOT/code/scripts/run_v11_claude_grounding_tmux.sh" "$LOG_PATH"
tmux new-session -d -s "$SESSION" "$inner_cmd"
echo "[launched] model=$MODEL version=$GROUNDING_VERSION session=$SESSION log=$LOG_PATH status=$STATUS_PATH"
echo "[guard] stops when weekly usage reaches ${WEEK_STOP_PERCENT}% (less than $((100 - WEEK_STOP_PERCENT))% left)"
