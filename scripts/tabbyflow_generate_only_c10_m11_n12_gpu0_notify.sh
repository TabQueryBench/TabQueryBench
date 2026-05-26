#!/usr/bin/env bash
# TabbyFlow: 指定数据集在 GPU0 上串行 generate-only；每个数据集完成后单独发一封邮件。
# 生成数量使用 --num-rows 0 自动对齐训练集行数。
# 邮件主题示例：5090-c10-tabbyflow-done / 5090-c10-tabbyflow-fail

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

export BENCHMARK_TABBYFLOW_GPUS="${BENCHMARK_TABBYFLOW_GPUS:-device=0}"
export BENCHMARK_TABBYFLOW_IMAGE="${BENCHMARK_TABBYFLOW_IMAGE:-ghcr.io/fredzjl/synthetic-benchmark:tabdiff-tabbyflow}"

NOTIFY_TO="${NOTIFY_TO:-1694362889@qq.com}"
NOTIFY_URL="${NOTIFY_URL:-http://127.0.0.1:18765/notify}"
TS_TAG="${TS_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$ROOT/logs/tabbyflow_generate_only_c10_m11_n12_gpu0_${TS_TAG}.log}"
TIMING_CSV="${TIMING_CSV:-$ROOT/logs/tabbyflow_generate_only_c10_m11_n12_gpu0_timings_${TS_TAG}.csv}"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$TIMING_CSV")"
exec > >(tee -a "$LOG_FILE") 2>&1

declare -a DATASETS=(c10 m11 n12)
N_TASKS="${#DATASETS[@]}"

echo "[$(date -Iseconds)] batch_start tabbyflow generate-only serial gpu=$BENCHMARK_TABBYFLOW_GPUS n=$N_TASKS"
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
subject = f"5090-{ds}-tabbyflow-{status}"
payload = {
    "to": os.environ["NOTIFY_TO"],
    "subject": subject,
    "pid": str(os.getpid()),
    "phase": f"tabbyflow {ds} {status}",
    "note": (
        f"tabbyflow generate-only serial on GPU0 for dataset {ds}; "
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
  echo "[$(date -Iseconds)] === [$idx/$N_TASKS] tabbyflow generate-only $ds ==="
  if "$PY" -m src.SpecificModels.runner \
    --model tabbyflow \
    --dataset "$ds" \
    --dataset-source auto \
    --generate \
    --num-rows 0 \
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
