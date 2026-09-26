#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-hku172}"
REMOTE_BASE="${REMOTE_BASE:-/data/jialinzhang/SQLagent}"
REMOTE_WORK="$REMOTE_BASE/tmp/rebuild_missing_cardinality_20260519"
REMOTE_DATA_ROOT="$REMOTE_WORK/datasets"
LOCAL_BASE="$(cd "$(dirname "$0")/.." && pwd)"
SSH_OPTS=(
  -o ConnectTimeout=10
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

echo "[deploy] syncing patched files to ${REMOTE_HOST}:${REMOTE_BASE}"
rsync -a -e "$RSYNC_SSH" \
  "$LOCAL_BASE/src/eval/subitem_workload_v2/inventory.py" \
  "$LOCAL_BASE/scripts/build_subitem_workload_v2_inventory.py" \
  "$LOCAL_BASE/scripts/run_subitem_workload_v2.py" \
  "${REMOTE_HOST}:${REMOTE_BASE}/tmp/rebuild_missing_cardinality_20260519_sync/"

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_WORK' '$REMOTE_DATA_ROOT'"

echo "[deploy] syncing complete benchmark dataset bundles to ${REMOTE_HOST}:${REMOTE_DATA_ROOT}"
python3 - <<'PY' > /tmp/rebuild_missing_cardinality_datasets_20260519.txt
from pathlib import Path

base = Path("/Users/jialinzhang/Documents/HKUNAISS/SyntheticNips/SQLagent/data")
for child in sorted(base.iterdir()):
    if not child.is_dir():
        continue
    name = child.name
    if name in {"c21", "n13"}:
        continue
    if name and name[0] in {"c", "m", "n"} and name[1:].isdigit():
        print(name)
PY

while IFS= read -r ds; do
  [ -n "$ds" ] || continue
  rsync -a -e "$RSYNC_SSH" \
    --include='*/' \
    --include='*-main.csv' \
    --include='*-train.csv' \
    --include='*-val.csv' \
    --include='*-test.csv' \
    --include='contracts/***' \
    --include='metadata/***' \
    --include='metadata_core/***' \
    --include='metadata_optional/***' \
    --include='source/***' \
    --exclude='*' \
    "$LOCAL_BASE/data/$ds/" "${REMOTE_HOST}:${REMOTE_DATA_ROOT}/$ds/"
done < /tmp/rebuild_missing_cardinality_datasets_20260519.txt

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail

BASE=/data/jialinzhang/SQLagent
WORK=$BASE/tmp/rebuild_missing_cardinality_20260519
LOGDIR=$WORK/logs
SYNC_DIR=$BASE/tmp/rebuild_missing_cardinality_20260519_sync
DATA_ROOT=$WORK/datasets
mkdir -p "$WORK" "$LOGDIR" "$SYNC_DIR" "$DATA_ROOT"

# Install synced files into the server workspace.
install -m 0644 "$SYNC_DIR/inventory.py" "$BASE/src/eval/subitem_workload_v2/inventory.py"
install -m 0755 "$SYNC_DIR/build_subitem_workload_v2_inventory.py" "$BASE/scripts/build_subitem_workload_v2_inventory.py"
install -m 0755 "$SYNC_DIR/run_subitem_workload_v2.py" "$BASE/scripts/run_subitem_workload_v2.py"

cat > "$WORK/rebuild_official_v2_missing_cardinality.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

BASE=/data/jialinzhang/SQLagent
WORK=$BASE/tmp/rebuild_missing_cardinality_20260519
LOGDIR=$WORK/logs
DATA_ROOT=$WORK/datasets
mkdir -p "$WORK" "$LOGDIR" "$DATA_ROOT"
cd "$BASE"

OFFICIAL_RUN_IDS=(
  v2_cli_20260502_081223_a
  v2_cli_20260502_081223_b
  v2_cli_20260502_081223_c
  v2_cli_20260502_081223_d
  v2_cli_20260502_081223_e
  v2_cli_20260502_081223_f
  subitem_workload_v2_failed_rerun_20260502_173341
)

echo "[rebuild] started $(date -Is)"
echo "[rebuild] staged data root: $DATA_ROOT"

python3 - <<'PY' > "$WORK/run_groups.json"
import csv
import json
from pathlib import Path

base = Path("/data/jialinzhang/SQLagent/data/workload_grounding_v2/registries")
backup_root = Path("/data/jialinzhang/SQLagent/tmp/rebuild_missing_cardinality_20260519/backups")
run_ids = [
    "v2_cli_20260502_081223_a",
    "v2_cli_20260502_081223_b",
    "v2_cli_20260502_081223_c",
    "v2_cli_20260502_081223_d",
    "v2_cli_20260502_081223_e",
    "v2_cli_20260502_081223_f",
    "subitem_workload_v2_failed_rerun_20260502_173341",
]

latest_backup = None
if backup_root.exists():
    backups = sorted([p for p in backup_root.iterdir() if p.is_dir()])
    if backups:
        latest_backup = backups[-1]

mapping: dict[str, list[str]] = {}
for run_id in run_ids:
    csv_path = base / f"{run_id}_query_registry_v2.csv"
    if not csv_path.exists() and latest_backup is not None:
        fallback = latest_backup / "registries" / f"{run_id}_query_registry_v2.csv"
        if fallback.exists():
            csv_path = fallback
    datasets = set()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                ds = str(row.get("dataset_id") or "").strip()
                if ds and ds not in {"c21", "n13"}:
                    datasets.add(ds)
    mapping[run_id] = sorted(datasets, key=lambda s: (s[0], int(s[1:])))

print(json.dumps(mapping, ensure_ascii=False, indent=2))
PY

DATASET_IDS="$(python3 - <<'PY'
import json
from pathlib import Path

mapping = json.loads(Path('/data/jialinzhang/SQLagent/tmp/rebuild_missing_cardinality_20260519/run_groups.json').read_text(encoding='utf-8'))
datasets = sorted({ds for values in mapping.values() for ds in values}, key=lambda s: (s[0], int(s[1:])))
print(",".join(datasets))
PY
)"

echo "[rebuild] rebuilding inventories for: $DATASET_IDS"
python3 scripts/build_subitem_workload_v2_inventory.py \
  --line-version v2 \
  --dataset-ids "$DATASET_IDS" \
  --data-root "$DATA_ROOT" \
  --planner-kind rule \
  | tee "$LOGDIR/build_inventory.log"

python3 - <<'PY' > "$WORK/inventory_sanity.tsv"
import json
from pathlib import Path

base = Path('/data/jialinzhang/SQLagent/data/workload_grounding_v2/inventories')
for ds in ['n8', 'c12', 'n18', 'c15']:
    path = base / f'{ds}_inventory_v2.json'
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding='utf-8'))
    print(f"{ds}\tproblem_count={data.get('problem_count')}\tagent_problem_count={data.get('agent_problem_count')}\tdeterministic_problem_count={data.get('deterministic_problem_count')}")
PY

BACKUP_TAG="before_missing_cardinality_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK/backups/$BACKUP_TAG/runs" "$WORK/backups/$BACKUP_TAG/registries"
cp "$WORK/run_groups.json" "$WORK/backups/$BACKUP_TAG/run_groups.json"

for run_id in "${OFFICIAL_RUN_IDS[@]}"; do
  if [ -d "$BASE/logs/subitem_workload_v2/runs/$run_id" ]; then
    mv "$BASE/logs/subitem_workload_v2/runs/$run_id" "$WORK/backups/$BACKUP_TAG/runs/$run_id"
  fi
  for ext in csv jsonl; do
    f="$BASE/data/workload_grounding_v2/registries/${run_id}_query_registry_v2.$ext"
    if [ -f "$f" ]; then
      mv "$f" "$WORK/backups/$BACKUP_TAG/registries/"
    fi
  done
done

python3 - <<'PY' > "$WORK/run_plan.tsv"
import json
from pathlib import Path

mapping = json.loads(Path('/data/jialinzhang/SQLagent/tmp/rebuild_missing_cardinality_20260519/run_groups.json').read_text(encoding='utf-8'))
for run_id, datasets in mapping.items():
    if datasets:
        print(f"{run_id}\t{','.join(datasets)}")
PY

while IFS=$'\t' read -r run_id dataset_ids; do
  [ -n "$run_id" ] || continue
  echo "[rebuild] rerunning $run_id on $dataset_ids"
  python3 scripts/run_subitem_workload_v2.py \
    --line-version v2 \
    --dataset-ids "$dataset_ids" \
    --engine cli \
    --model gpt-5.4 \
    --run-id "$run_id" \
    --ai-cli-preset codex \
    --ai-cli-timeout-seconds 120 \
    --ai-cli-retries 1 \
    --ai-cli-answer-mode local \
    --row-limit 50 \
    --sql-timeout-ms 10000 \
    | tee -a "$LOGDIR/run_${run_id}.log"
done < "$WORK/run_plan.tsv"

echo "[rebuild] rerunning official current-success analysis"
python3 "$BASE/tmp/official_sql_recompute_20260519/run_official_analysis_32way.py" \
  | tee "$LOGDIR/official_analysis.log"

echo "[rebuild] finished $(date -Is)"
SH

chmod +x "$WORK/rebuild_official_v2_missing_cardinality.sh"

cat > "$WORK/launch_when_idle.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

BASE=/data/jialinzhang/SQLagent
WORK=$BASE/tmp/rebuild_missing_cardinality_20260519
mkdir -p "$WORK"
LOG="$WORK/launch_when_idle.log"

{
  echo "[launcher] started $(date -Is)"
} >> "$LOG"

while pgrep -af "$BASE/tmp/official_sql_recompute_20260519/run_official_analysis_32way.py" >/dev/null 2>&1; do
  echo "[launcher] official analysis still running, wait 300s $(date -Is)" >> "$LOG"
  sleep 300
done

echo "[launcher] launching rebuild $(date -Is)" >> "$LOG"
bash "$WORK/rebuild_official_v2_missing_cardinality.sh" >> "$WORK/rebuild_run.log" 2>&1
echo "[launcher] rebuild script exited $(date -Is)" >> "$LOG"
SH

chmod +x "$WORK/launch_when_idle.sh"

nohup bash "$WORK/launch_when_idle.sh" > "$WORK/nohup.out" 2>&1 < /dev/null &
echo $! > "$WORK/launch_when_idle.pid"
echo "[deploy] immediate launcher pid=$(cat "$WORK/launch_when_idle.pid")"
REMOTE

echo "[deploy] complete"
