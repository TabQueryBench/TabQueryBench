#!/usr/bin/env bash
# ForestDiffusion: fixed dataset list, serial train+generate, one email per dataset.
# NUM_ROWS=0 means generate as many rows as the training split.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

export BENCHMARK_FORESTDIFFUSION_GPUS="${BENCHMARK_FORESTDIFFUSION_GPUS:-device=0}"
export BENCHMARK_FORESTDIFFUSION_IMAGE="${BENCHMARK_FORESTDIFFUSION_IMAGE:-ghcr.io/fredzjl/synthetic-benchmark:forestdiffusion}"

NOTIFY_TO="${NOTIFY_TO:-1694362889@qq.com}"
NOTIFY_URL="${NOTIFY_URL:-http://127.0.0.1:18765/notify}"
TS_TAG="${TS_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$ROOT/logs/forestdiffusion_c3_c6_c15_c19_serial_gpu0_${TS_TAG}.log}"
TIMING_CSV="${TIMING_CSV:-$ROOT/logs/forestdiffusion_c3_c6_c15_c19_serial_gpu0_timings_${TS_TAG}.csv}"
NUM_ROWS="${NUM_ROWS:-0}"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$TIMING_CSV")"
exec > >(tee -a "$LOG_FILE") 2>&1

declare -a DATASETS=(c3 c6 c15 c16 c17 c18 c19)
N_TASKS="${#DATASETS[@]}"

echo "[$(date -Iseconds)] batch_start forestdiffusion serial gpu=$BENCHMARK_FORESTDIFFUSION_GPUS n=$N_TASKS num_rows=$NUM_ROWS"
echo "[$(date -Iseconds)] datasets=${DATASETS[*]}"
echo "dataset,seconds,exit_code" > "$TIMING_CSV"

send_notify() {
  local ds="$1"
  local status="$2"
  local code="$3"
  local dur="$4"
  export NOTIFY_TO NOTIFY_URL LOG_FILE TIMING_CSV
  DATASET="$ds" STATUS="$status" CODE="$code" DURATION="$dur" "$PY" -c '
import json, os, urllib.request

ds = os.environ["DATASET"]
status = os.environ["STATUS"]
code = int(os.environ["CODE"])
dur = os.environ["DURATION"]
log_file = os.environ["LOG_FILE"]
timing_csv = os.environ["TIMING_CSV"]
subject = f"5090-{ds}-forestdiffusion-{status}"
payload = {
    "to": os.environ["NOTIFY_TO"],
    "subject": subject,
    "pid": str(os.getpid()),
    "phase": f"forestdiffusion {ds} {status}",
    "note": (
        f"forestdiffusion train+generate serial on GPU0 for dataset {ds}; "
        f"status={status}; exit={code}; wall_seconds={dur}; "
        f"log={log_file}; timing={timing_csv}"
    ),
    "job_finished": True,
    "exit_code": code,
}
req = urllib.request.Request(
    os.environ["NOTIFY_URL"],
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as r:
    print("notify:", r.read().decode()[:400])
' || echo "[warn] notify failed for $ds"
}

ec=0
idx=0
for ds in "${DATASETS[@]}"; do
  idx=$((idx + 1))
  t0=$(date +%s)
  echo "[$(date -Iseconds)] === [$idx/$N_TASKS] forestdiffusion train+generate $ds ==="
  if "$PY" -m src.SpecificModels.runner \
    --model forestdiffusion \
    --dataset "$ds" \
    --dataset-source auto \
    --train \
    --generate \
    --num-rows "$NUM_ROWS" \
    --no-stats; then
    one_ec=0
    status="done"
    echo "[$(date -Iseconds)] OK dataset=$ds"
  else
    one_ec=1
    status="fail"
    ec=1
    echo "[$(date -Iseconds)] FAILED dataset=$ds"
  fi
  t1=$(date +%s)
  dur=$((t1 - t0))
  echo "$ds,$dur,$one_ec" >> "$TIMING_CSV"
  echo "[timing] dataset=$ds wall_seconds=$dur exit_code=$one_ec"
  send_notify "$ds" "$status" "$one_ec" "$dur"
done

echo "[$(date -Iseconds)] batch_end exit=$ec"
exit "$ec"
