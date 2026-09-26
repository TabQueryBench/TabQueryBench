#!/usr/bin/env bash
# Execute, publish and SV2-score repeat runs of V11 grounding, one tmux session end to end.
#
#   bash run_v11_repeat_pipeline.sh                          # glm and flash, runs 2 and 3
#   PAIRS="glm:2,flash:2 glm:3,flash:3" bash run_v11_repeat_pipeline.sh
#
# PAIRS is a space-separated list of groups; the versions inside a group execute at the
# same time (4 shards each), groups run one after another. Scoring then runs two lanes in
# parallel. Any failed step stops the pipeline so nothing is published or scored from a
# partial execution.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
Q="$ROOT/Query"
PAIRS="${PAIRS:-glm:2,flash:2 glm:3,flash:3}"
SCORE_WORKERS="${SCORE_WORKERS:-6}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-$Q/code/logs/v11_repeat_pipeline_$STAMP}"
mkdir -p "$LOG"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG/pipeline.log"; }
die() { log "STOP: $*"; exit 1; }

version_of() {  # model run -> full version name
  PYTHONPATH="$Q/code" python3 -c "
from tqb_query.workload_grounding.v10_versions import format_grounding_version
print(format_grounding_version('v11', '$1', int('$2')))"
}

latest_registry() {
  ls -t "$Q/code/data/workload_grounding_v11/variants/$1/registries/"*_query_registry_"$1".jsonl 2>/dev/null \
    | grep -vE 'discarded|pre_reconcile|\.bak' | head -1
}

check_execution() {  # version exec_log_root
  local v="$1" root="$2" bad
  bad="$(grep -Lx 0 "$root"/shard*.status 2>/dev/null)"
  [[ -z "$bad" && -n "$(ls "$root"/shard*.status 2>/dev/null)" ]] || die "$v: shard exit codes: $(cat "$root"/shard*.status 2>/dev/null | tr '\n' ' ')"
  local report rc
  report="$(python3 - "$Q" "$v" "$(latest_registry "$v")" <<'EOF'
import glob, json, sys
q, v, registry = sys.argv[1:]
expected = set()
for f in glob.glob(f"{q}/code/data/workload_grounding_v11/variants/{v}/inventories/*_inventory_{v}.json"):
    expected |= {item["query_record_id"] for item in json.load(open(f)).get("items") or []}
rows = [json.loads(line) for line in open(registry)]
ids = {row["query_record_id"] for row in rows}
ok = sum(1 for row in rows if row.get("exec_ok_real"))
print(f"  {v}: expected={len(expected)} registry_rows={len(rows)} unique={len(ids)} "
      f"missing={len(expected - ids)} extra={len(ids - expected)} exec_ok_real={ok}")
sys.exit(1 if expected - ids else 0)
EOF
)"
  rc=$?
  log "$report"
  [[ $rc == 0 ]] || die "$v: registry does not cover every inventory item"
}

# ---- 1. execute, group by group -------------------------------------------------------
for group in $PAIRS; do
  roots=()
  for spec in ${group//,/ }; do
    model="${spec%%:*}"; run="${spec##*:}"; v="$(version_of "$model" "$run")"
    root="$LOG/exec_$v"
    log "exec start $v"
    MODEL="$model" RUN="$run" LOG_ROOT="$root" bash "$Q/code/scripts/run_v11_exec_shards_tmux.sh" >> "$LOG/pipeline.log" 2>&1 \
      || die "$v: could not launch execution"
    roots+=("$v=$root")
  done
  while tmux ls -F '#S' 2>/dev/null | grep -q '^v11_exec_'; do sleep 60; done
  for pair in "${roots[@]}"; do
    check_execution "${pair%%=*}" "${pair#*=}"
    log "exec done ${pair%%=*}"
  done
done

# ---- 2. reconcile and publish -----------------------------------------------------------
VERSIONS=()
for group in $PAIRS; do
  for spec in ${group//,/ }; do VERSIONS+=("$(version_of "${spec%%:*}" "${spec##*:}")"); done
done
cd "$Q" || die "no Query dir"
for v in "${VERSIONS[@]}"; do
  python3 tools/reconcile_v11_registry.py --grounding-version "$v" >> "$LOG/pipeline.log" 2>&1 || die "$v: reconcile failed"
  rid="$(python3 -c "import json;print(json.loads(open('$(latest_registry "$v")').readline())['round_id'])")"
  python3 tools/publish_v11_release.py --grounding-version "$v" --run-id "$rid" >> "$LOG/pipeline.log" 2>&1 \
    || die "$v: publish failed"
  log "published $v -> Queries/V${v#v}-full (run $rid)"
done

# ---- 3. SV2 scoring, two lanes ----------------------------------------------------------
score() {  # version
  local v="$1" tag="sv2_$1_$(date +%Y%m%d_%H%M%S)" rc
  log "score start $v -> $tag"
  ( cd "$ROOT/Scoring/code" && TQB_SCORING_MODE=sv2 PYTHONUNBUFFERED=1 PYTHONPATH=. \
      EVAL_REAL_DATA_ROOT="$ROOT/Synthesizing/raw_data/tabular_datasets" \
      EVAL_TABQUERYBENCH_MAIN_ROOT="$ROOT/Synthesizing/synthetic_data/main" \
      nice -n 10 python3 -m tqb_scoring.eval.analysis.runner \
        --sql-source-version "$v" --engines template --root-names TabQueryBench-SynDataSuccess-main \
        --latest-only --max-workers "$SCORE_WORKERS" --shared-real-cache-root "$ROOT/Scoring/cache/shared_real" \
        --run-tag "$tag" > "$ROOT/Scoring/results/logs/$tag.log" 2>&1 )
  rc=$?
  log "score done $v rc=$rc tag=$tag"
  return $rc
}
lane() { for v in "$@"; do score "$v" || return 1; done; }

lane_a=(); lane_b=()
for i in "${!VERSIONS[@]}"; do
  if (( i % 2 == 0 )); then lane_a+=("${VERSIONS[$i]}"); else lane_b+=("${VERSIONS[$i]}"); fi
done
lane "${lane_a[@]}" & pid_a=$!
lane "${lane_b[@]}" & pid_b=$!
wait $pid_a; rc_a=$?
wait $pid_b; rc_b=$?
[[ $rc_a == 0 && $rc_b == 0 ]] || die "scoring failed (lane rc $rc_a/$rc_b)"
log "ALL DONE: ${VERSIONS[*]}"
