#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_PATH="${1:-$PROJECT_ROOT/code/logs/v10_claude_launches/v10_stop_$(date +%Y%m%d_%H%M%S).log}"
GRACE_SECONDS="${GRACE_SECONDS:-20}"

SESSIONS=(
  "v10_v10_1_1_opus5"
  "v10_v10_1_2_sonnet"
  "v10_v10_1_3_fable"
  "v10_v10_1_4_haiku"
)

mkdir -p "$(dirname "$LOG_PATH")"

{
  echo "[stop] requested at $(date -Is)"

  for session in "${SESSIONS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "[stop] sending Ctrl-C to $session"
      tmux send-keys -t "$session" C-c
    else
      echo "[stop] already absent: $session"
    fi
  done

  deadline=$((SECONDS + GRACE_SECONDS))
  while (( SECONDS < deadline )); do
    alive=0
    for session in "${SESSIONS[@]}"; do
      if tmux has-session -t "$session" 2>/dev/null; then
        alive=1
        break
      fi
    done
    (( alive == 0 )) && break
    sleep 1
  done

  for session in "${SESSIONS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "[stop] grace period expired; killing tmux session $session"
      tmux kill-session -t "$session"
    fi
  done

  # Only match this V10 grounding entry point. This deliberately leaves other
  # Claude workloads on the server untouched.
  mapfile -t orphan_pids < <(
    pgrep -f 'python3 code/scripts/build_subitem_workload_v2_inventory.py .*--line-version v10 .*--planner-kind agent-bind' || true
  )
  if (( ${#orphan_pids[@]} > 0 )); then
    echo "[stop] terminating orphan V10 worker pids: ${orphan_pids[*]}"
    kill -TERM "${orphan_pids[@]}" 2>/dev/null || true
    sleep 3
    for pid in "${orphan_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        echo "[stop] force-killing orphan V10 worker pid: $pid"
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
  fi

  echo "[stop] completed at $(date -Is)"
} >>"$LOG_PATH" 2>&1

echo "[stop-log] $LOG_PATH"
