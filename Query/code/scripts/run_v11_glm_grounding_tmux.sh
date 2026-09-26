#!/usr/bin/env bash
# Run V11 (agent-select-bind) with one Z.AI GLM model over the V11 dataset set in a detached tmux session.
# The job re-runs with --resume until every inventory exists: quota/rate-limit stops wait and continue,
# other failures retry a bounded number of times (the counter resets whenever a dataset completes).
#
# MODEL selects the planner model (glm / glm-5.3 -> v11.2.2_glm-5.3, flash / glm-5.3-flash -> v11.2.1_glm-5.3-flash);
# RUN=2 grounds the second run (v11.2.2-2_glm-5.3), and so on.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:-glm}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../Synthesizing/raw_data/tabular_datasets}"
DATASET_IDS="${DATASET_IDS:-}"
QUOTA_WAIT_SECONDS="${QUOTA_WAIT_SECONDS:-1800}"
ERROR_WAIT_SECONDS="${ERROR_WAIT_SECONDS:-600}"
MAX_ERROR_RETRIES="${MAX_ERROR_RETRIES:-5}"

GROUNDING_VERSION="$(PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
from tqb_query.workload_grounding.v10_versions import format_grounding_version, resolve_model_version
print(resolve_model_version('$MODEL', line_family='v11', grounding_version=format_grounding_version('v11', '$MODEL', int('${RUN:-1}'))).grounding_version)
")"
LAUNCH_ID="${LAUNCH_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/code/logs/v11_glm_launches/${GROUNDING_VERSION}_$LAUNCH_ID}"
SESSION="v11_${GROUNDING_VERSION//./_}"
INVENTORY_DIR="$PROJECT_ROOT/code/data/workload_grounding_v11/variants/$GROUNDING_VERSION/inventories"
LOG_PATH="$LOG_ROOT/${GROUNDING_VERSION}.log"
STATUS_PATH="$LOG_ROOT/${GROUNDING_VERSION}.status"

inventory_count() {
  # A missing directory (first run of a version) must read as a plain 0: a failing find under
  # set -e/pipefail would abort the script, and a fallback echo would emit two lines and break (( )).
  if [[ -d "$INVENTORY_DIR" ]]; then
    find "$INVENTORY_DIR" -maxdepth 1 -name "*_inventory_${GROUNDING_VERSION}.json" | wc -l
  else
    echo 0
  fi
}

run_inner() {
  cd "$PROJECT_ROOT"
  trap 'echo "[interrupted] at $(date -Is)"; echo 130 > "$STATUS_PATH"; exit 130' INT TERM HUP
  local attempt=0 errors=0 rc before after last_error
  local dataset_args=()
  [[ -n "$DATASET_IDS" ]] && dataset_args=(--dataset-ids "$DATASET_IDS")
  while true; do
    attempt=$((attempt + 1))
    before="$(inventory_count)"
    echo "[start] model=$MODEL version=$GROUNDING_VERSION attempt=$attempt inventories=$before at=$(date -Is)"
    set +e
    PYTHONUNBUFFERED=1 PYTHONPATH=code nice -n 10 ionice -c3 python3 code/scripts/build_subitem_workload_v2_inventory.py \
      --line-version v11 --planner-kind agent-select-bind --planner-model "$MODEL" --grounding-version "$GROUNDING_VERSION" \
      --data-root "$DATA_ROOT" --agent-bind-problems-per-template 1 --resume "${dataset_args[@]}" \
      > "$LOG_ROOT/attempt_${attempt}.log" 2>&1
    rc=$?
    set -e
    cat "$LOG_ROOT/attempt_${attempt}.log"
    after="$(inventory_count)"
    echo "[end] attempt=$attempt rc=$rc inventories=$after at=$(date -Is)"
    if (( rc == 0 )); then
      echo 0 > "$STATUS_PATH"
      return 0
    fi
    (( after > before )) && errors=0
    last_error="$(grep -E '^[A-Za-z_.]+(Error|Exception)' "$LOG_ROOT/attempt_${attempt}.log" | tail -n 1 || true)"
    if [[ "$last_error" == *PlannerQuotaError* ]]; then
      echo "[quota] waiting ${QUOTA_WAIT_SECONDS}s: $last_error"
      sleep "$QUOTA_WAIT_SECONDS"
      continue
    fi
    errors=$((errors + 1))
    if (( errors > MAX_ERROR_RETRIES )); then
      echo "[abort] $errors consecutive failures without progress: $last_error"
      echo "$rc" > "$STATUS_PATH"
      return "$rc"
    fi
    echo "[retry] failure $errors/$MAX_ERROR_RETRIES, waiting ${ERROR_WAIT_SECONDS}s: $last_error"
    sleep "$ERROR_WAIT_SECONDS"
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
printf -v inner_cmd 'MODEL=%q LOG_ROOT=%q DATA_ROOT=%q DATASET_IDS=%q QUOTA_WAIT_SECONDS=%q ERROR_WAIT_SECONDS=%q MAX_ERROR_RETRIES=%q bash %q --inner >> %q 2>&1' \
  "$MODEL" "$LOG_ROOT" "$DATA_ROOT" "$DATASET_IDS" "$QUOTA_WAIT_SECONDS" "$ERROR_WAIT_SECONDS" "$MAX_ERROR_RETRIES" \
  "$PROJECT_ROOT/code/scripts/run_v11_glm_grounding_tmux.sh" "$LOG_PATH"
tmux new-session -d -s "$SESSION" "$inner_cmd"
echo "[launched] model=$MODEL version=$GROUNDING_VERSION session=$SESSION log=$LOG_PATH status=$STATUS_PATH"
