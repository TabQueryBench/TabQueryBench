#!/usr/bin/env bash
# Execute a V11 inventory (render + run the SQL) with several shards in parallel.
#
# All shards share one --run-id so their rows land in the same registry; the runner guards that
# file with flock, and shards split a dataset's items by index, so parallel writes are safe.
#
#   MODEL=glm bash run_v11_exec_shards_tmux.sh                     # start 4 shards now
#   MODEL=glm RUN=2 bash run_v11_exec_shards_tmux.sh               # execute the second run, v11.2.2-2_glm-5.3
#   MODEL=flash WAIT_FOR_SESSIONS="v11_2_1_glm-5_3-flash_s0" bash ... # start once those sessions end
#   MODEL=glm bash run_v11_exec_shards_tmux.sh --status            # per-shard progress
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:-glm}"
SHARD_COUNT="${SHARD_COUNT:-4}"
SQL_TIMEOUT_MS="${SQL_TIMEOUT_MS:-60000}"     # 10s (the default) lost whole datasets under NAS contention
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/../Synthesizing/raw_data/tabular_datasets}"
DATASET_IDS="${DATASET_IDS:-}"
WAIT_FOR_SESSIONS="${WAIT_FOR_SESSIONS:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
FORCE="${FORCE:-0}"

GROUNDING_VERSION="$(PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
from tqb_query.workload_grounding.v10_versions import format_grounding_version, resolve_model_version
print(resolve_model_version('$MODEL', line_family='v11', grounding_version=format_grounding_version('v11', '$MODEL', int('${RUN:-1}'))).grounding_version)
")"
INVENTORY_DIR="$PROJECT_ROOT/code/data/workload_grounding_v11/variants/$GROUNDING_VERSION/inventories"
SESSION_PREFIX="v11_exec_${GROUNDING_VERSION//./_}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/code/logs/v11_exec_launches/${GROUNDING_VERSION}_$(date +%Y%m%d_%H%M%S)}"

inventory_count() {
  if [[ -d "$INVENTORY_DIR" ]]; then
    find "$INVENTORY_DIR" -maxdepth 1 -name "*_inventory_${GROUNDING_VERSION}.json" | wc -l
  else
    echo 0
  fi
}

if [[ "${1:-}" == "--status" ]]; then
  echo "version=$GROUNDING_VERSION inventories=$(inventory_count)/49"
  latest_log_root="$(ls -dt "$PROJECT_ROOT/code/logs/v11_exec_launches/${GROUNDING_VERSION}"_* 2>/dev/null | head -1 || true)"
  if [[ -z "$latest_log_root" ]]; then
    echo "no execution run launched yet"
    exit 0
  fi
  echo "log_root=$latest_log_root"
  for i in $(seq 0 $((SHARD_COUNT - 1))); do
    state="running"
    tmux has-session -t "${SESSION_PREFIX}_s$i" 2>/dev/null || state="ended"
    status_line=""
    [[ -f "$latest_log_root/shard$i.status" ]] && status_line=" exit=$(cat "$latest_log_root/shard$i.status")"
    printf "shard %d [%s]%s: %s\n" "$i" "$state" "$status_line" \
      "$(grep -E '(done items=|FAILED )' "$latest_log_root/shard$i.log" 2>/dev/null | tail -1 | cut -c1-140)"
  done
  registry="$(ls -t "$PROJECT_ROOT/code/data/workload_grounding_v11/variants/$GROUNDING_VERSION/registries"/*.jsonl 2>/dev/null | head -1 || true)"
  [[ -n "$registry" ]] && echo "registry rows=$(wc -l < "$registry") file=$(basename "$registry")"
  exit 0
fi

launch_shards() {
  local run_id="$1"
  mkdir -p "$LOG_ROOT"
  echo "$run_id" > "$LOG_ROOT/run_id.txt"
  local dataset_args=()
  [[ -n "$DATASET_IDS" ]] && dataset_args=(--dataset-ids "$DATASET_IDS")
  for i in $(seq 0 $((SHARD_COUNT - 1))); do
    local session="${SESSION_PREFIX}_s$i"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "[skip] session already exists: $session"
      continue
    fi
    tmux new-session -d -s "$session" "cd $(printf '%q' "$PROJECT_ROOT") && \
PYTHONUNBUFFERED=1 PYTHONPATH=code nice -n 10 ionice -c3 python3 code/scripts/run_subitem_workload_v2.py \
  --line-version v11 --model $(printf '%q' "$MODEL") --grounding-version $(printf '%q' "$GROUNDING_VERSION") --engine template \
  --run-id $(printf '%q' "$run_id") --data-root $(printf '%q' "$DATA_ROOT") \
  --sql-timeout-ms $SQL_TIMEOUT_MS --shard-index $i --shard-count $SHARD_COUNT \
  ${dataset_args[*]:-} > $(printf '%q' "$LOG_ROOT/shard$i.log") 2>&1; \
echo \$? > $(printf '%q' "$LOG_ROOT/shard$i.status")"
    echo "[launched] $session -> $LOG_ROOT/shard$i.log"
  done
  echo "[run-id] $run_id (shared by all $SHARD_COUNT shards)"
  echo "[watch]  tail -f $LOG_ROOT/shard0.log"
  echo "[status] MODEL=$MODEL RUN=${RUN:-1} bash $PROJECT_ROOT/code/scripts/run_v11_exec_shards_tmux.sh --status"
}

if [[ "${1:-}" == "--wait-then-launch" ]]; then
  while :; do
    pending=""
    for session in $WAIT_FOR_SESSIONS; do
      tmux has-session -t "$session" 2>/dev/null && pending="$pending $session"
    done
    [[ -z "$pending" ]] && break
    echo "[wait] still running:$pending at $(date -Is)"
    sleep "$WAIT_POLL_SECONDS"
  done
  echo "[wait] prerequisites finished at $(date -Is); inventories=$(inventory_count)/49"
  if [[ "$(inventory_count)" -lt 49 && "$FORCE" != "1" ]]; then
    echo "[abort] only $(inventory_count)/49 inventories present; set FORCE=1 to run anyway"
    exit 2
  fi
  launch_shards "subitem_workload_${GROUNDING_VERSION}_$(date -u +%Y%m%d_%H%M%S)"
  exit 0
fi

have="$(inventory_count)"
if [[ -n "$WAIT_FOR_SESSIONS" ]]; then
  wait_session="${SESSION_PREFIX}_wait"
  tmux has-session -t "$wait_session" 2>/dev/null && { echo "[skip] waiter already running: $wait_session"; exit 0; }
  mkdir -p "$LOG_ROOT"
  tmux new-session -d -s "$wait_session" "MODEL=$(printf '%q' "$MODEL") RUN=$(printf '%q' "${RUN:-1}") SHARD_COUNT=$SHARD_COUNT \
SQL_TIMEOUT_MS=$SQL_TIMEOUT_MS DATA_ROOT=$(printf '%q' "$DATA_ROOT") DATASET_IDS=$(printf '%q' "$DATASET_IDS") \
WAIT_FOR_SESSIONS=$(printf '%q' "$WAIT_FOR_SESSIONS") WAIT_POLL_SECONDS=$WAIT_POLL_SECONDS FORCE=$FORCE \
LOG_ROOT=$(printf '%q' "$LOG_ROOT") bash $(printf '%q' "$PROJECT_ROOT/code/scripts/run_v11_exec_shards_tmux.sh") \
--wait-then-launch >> $(printf '%q' "$LOG_ROOT/wait.log") 2>&1"
  echo "[waiting] session=$wait_session waits for:$WAIT_FOR_SESSIONS then starts $SHARD_COUNT shards"
  echo "[log]     $LOG_ROOT/wait.log"
  exit 0
fi

if [[ "$have" -lt 49 && -z "$DATASET_IDS" && "$FORCE" != "1" ]]; then
  echo "[abort] only $have/49 inventories for $GROUNDING_VERSION; set FORCE=1 or pass DATASET_IDS"
  exit 2
fi
launch_shards "${RUN_ID:-subitem_workload_${GROUNDING_VERSION}_$(date -u +%Y%m%d_%H%M%S)}"
