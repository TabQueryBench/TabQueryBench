#!/usr/bin/env python3
"""Produce the V2 final checkpoint comparing baseline artifacts against V2 outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404")
DEFAULT_OUTPUT_DIR = Path("logs/sql_high_corpus_build_20260404/v2_refinement/final_v2")
DEFAULT_BASELINE_FINAL_INDEX = Path("logs/sql_high_corpus_build_20260404/final/final_index.csv")
DEFAULT_BASELINE_MORNING_TABLE = Path(
    "logs/sql_high_corpus_build_20260404/final/morning_review_dataset_table.csv"
)
DEFAULT_FINAL_INDEX_V2 = Path("logs/sql_high_corpus_build_20260404/v2_refinement/final_v2/final_index_v2.csv")
DEFAULT_GATE_CSV = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/gate/dataset_readiness_v2.csv"
)
DEFAULT_EXECUTE_CSV = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/execute/sql_executability_v2.csv"
)
DEFAULT_DEDUP_CSV = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/dedup/master_sql_inventory_dedup_v2.csv"
)

DELTA_FIELDNAMES = [
    "own_id",
    "dataset_name",
    "baseline_strict_sql_count",
    "v2_strict_sql_count",
    "strict_delta",
    "baseline_collision_risk_sql_count",
    "v2_collision_risk_sql_count",
    "collision_delta",
    "baseline_readiness",
    "v2_readiness_label",
    "baseline_top_strict_count",
    "v2_top_strict_count",
    "baseline_seed_count",
    "v2_seed_count",
    "recommended_decision",
]

READINESS_PRIORITY = {
    "READY": 0,
    "READY_WITH_WARNINGS": 1,
    "NOT_READY": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final V2 delta memo and readiness checkpoint."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline-final-index", type=Path, default=DEFAULT_BASELINE_FINAL_INDEX)
    parser.add_argument("--baseline-morning-table", type=Path, default=DEFAULT_BASELINE_MORNING_TABLE)
    parser.add_argument("--final-index-v2", type=Path, default=DEFAULT_FINAL_INDEX_V2)
    parser.add_argument("--gate-csv", type=Path, default=DEFAULT_GATE_CSV)
    parser.add_argument("--execute-csv", type=Path, default=DEFAULT_EXECUTE_CSV)
    parser.add_argument("--dedup-csv", type=Path, default=DEFAULT_DEDUP_CSV)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_int(value: str | None) -> int:
    text = (value or "").strip()
    return int(text) if text else 0


def to_float(value: str | None) -> float:
    text = (value or "").strip()
    return float(text) if text else 0.0


def recommended_decision(v2_readiness_label: str) -> str:
    if v2_readiness_label == "READY":
        return "GO"
    if v2_readiness_label == "READY_WITH_WARNINGS":
        return "GO_WITH_WARNINGS"
    return "HOLD"


def baseline_ready_value(baseline_row: dict[str, str], baseline_morning_row: dict[str, str]) -> str:
    final_value = (baseline_row.get("question_taxonomy_readiness") or "").strip()
    if final_value:
        return final_value
    return (baseline_morning_row.get("readiness_for_question_taxonomy") or "").strip() or "no"


def readiness_transition_score(baseline_readiness: str, v2_readiness: str) -> int:
    baseline_yes = baseline_readiness.lower() == "yes"
    if baseline_yes and v2_readiness == "READY":
        return 2
    if baseline_yes and v2_readiness == "READY_WITH_WARNINGS":
        return 1
    if baseline_yes and v2_readiness == "NOT_READY":
        return -2
    if not baseline_yes and v2_readiness == "READY":
        return 3
    if not baseline_yes and v2_readiness == "READY_WITH_WARNINGS":
        return 2
    return 0


def improvement_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        -readiness_transition_score(row["baseline_readiness"], row["v2_readiness_label"]),
        -to_int(row["strict_delta"]),
        to_int(row["collision_delta"]),
        row["own_id"],
    )


def degradation_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        to_int(row["strict_delta"]),
        -to_int(row["collision_delta"]),
        READINESS_PRIORITY.get(row["v2_readiness_label"], 9),
        row["own_id"],
    )


def compute_execute_summary(execute_rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in execute_rows:
        own_id = (row.get("own_id") or "").strip()
        dataset_summary = summary.setdefault(own_id, {"pass": 0, "fail": 0, "unknown": 0})
        dataset_summary[(row.get("executable_status_v2") or "unknown").strip() or "unknown"] += 1
    return summary


def compute_dedup_summary(dedup_rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in dedup_rows:
        own_id = (row.get("own_id") or "").strip()
        dataset_summary = summary.setdefault(own_id, {"total": 0, "primary": 0, "dropped": 0})
        dataset_summary["total"] += 1
        if (row.get("is_primary_canonical") or "").strip() == "yes":
            dataset_summary["primary"] += 1
        else:
            dataset_summary["dropped"] += 1
    return summary


def render_markdown(
    *,
    now_utc: str,
    baseline_final_index: Path,
    baseline_morning_table: Path,
    final_index_v2: Path,
    gate_csv: Path,
    execute_csv: Path,
    dedup_csv: Path,
    rows: list[dict[str, str]],
    hold_rows: list[dict[str, str]],
    safe_rows: list[dict[str, str]],
    warning_rows: list[dict[str, str]],
    top_improved: list[dict[str, str]],
    top_degraded: list[dict[str, str]],
    dedup_summary: dict[str, dict[str, int]],
    execute_summary: dict[str, dict[str, int]],
) -> str:
    strict_delta_total = sum(to_int(row["strict_delta"]) for row in rows)
    collision_delta_total = sum(to_int(row["collision_delta"]) for row in rows)
    dedup_dropped_total = sum(dataset["dropped"] for dataset in dedup_summary.values())
    pass_total = sum(dataset.get("pass", 0) for dataset in execute_summary.values())
    fail_total = sum(dataset.get("fail", 0) for dataset in execute_summary.values())
    unknown_total = sum(dataset.get("unknown", 0) for dataset in execute_summary.values())
    baseline_ready_count = sum(1 for row in rows if row["baseline_readiness"].lower() == "yes")
    v2_ready_count = sum(1 for row in rows if row["v2_readiness_label"] == "READY")
    v2_warning_count = sum(1 for row in rows if row["v2_readiness_label"] == "READY_WITH_WARNINGS")
    v2_hold_count = sum(1 for row in rows if row["v2_readiness_label"] == "NOT_READY")

    lines = [
        "# Delta Baseline vs V2",
        "",
        f"- Generated at UTC: `{now_utc}`",
        f"- Baseline final index: `{baseline_final_index.resolve()}`",
        f"- Baseline morning review table: `{baseline_morning_table.resolve()}`",
        f"- V2 final index: `{final_index_v2.resolve()}`",
        f"- V2 gate CSV: `{gate_csv.resolve()}`",
        f"- V2 execute CSV: `{execute_csv.resolve()}`",
        f"- V2 dedup CSV: `{dedup_csv.resolve()}`",
        "",
        "## Global Delta Summary",
        "",
        f"- Baseline ready datasets: `{baseline_ready_count}`",
        f"- V2 ready datasets: `{v2_ready_count}`",
        f"- V2 ready-with-warnings datasets: `{v2_warning_count}`",
        f"- V2 hold datasets: `{v2_hold_count}`",
        f"- Aggregate strict count delta: `{strict_delta_total}`",
        f"- Aggregate collision count delta: `{collision_delta_total}`",
        f"- Aggregate dedup reduction in V2: `{dedup_dropped_total}` rows removed from canonical selection",
        f"- V2 executability evidence across all rows: `pass={pass_total}`, `fail={fail_total}`, `unknown={unknown_total}`",
        "",
        "## Top Improved Datasets",
        "",
    ]

    improved_written = 0
    for row in top_improved:
        if readiness_transition_score(row["baseline_readiness"], row["v2_readiness_label"]) <= 0 and to_int(row["strict_delta"]) <= 0:
            continue
        improved_written += 1
        lines.append(
            f"- `{row['own_id']}` {row['dataset_name']}: baseline_readiness=`{row['baseline_readiness']}`,"
            f" v2=`{row['v2_readiness_label']}`, strict_delta=`{row['strict_delta']}`,"
            f" collision_delta=`{row['collision_delta']}`"
        )
    if improved_written == 0:
        lines.append("- No dataset showed a positive readiness or strict-count gain versus baseline.")

    lines.extend(["", "## Top Degraded Datasets", ""])
    degraded_written = 0
    for row in top_degraded:
        if to_int(row["strict_delta"]) >= 0 and to_int(row["collision_delta"]) <= 0:
            continue
        degraded_written += 1
        lines.append(
            f"- `{row['own_id']}` {row['dataset_name']}: baseline_readiness=`{row['baseline_readiness']}`,"
            f" v2=`{row['v2_readiness_label']}`, strict_delta=`{row['strict_delta']}`,"
            f" collision_delta=`{row['collision_delta']}`"
        )
    if degraded_written == 0:
        lines.append("- No materially degraded dataset was detected.")

    lines.extend(["", "## Datasets Now Safe For Taxonomy", ""])
    if safe_rows:
        for row in safe_rows:
            lines.append(
                f"- `{row['own_id']}` {row['dataset_name']}: GO, v2_strict=`{row['v2_strict_sql_count']}`,"
                f" v2_top_strict=`{row['v2_top_strict_count']}`, v2_seeds=`{row['v2_seed_count']}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Datasets Safe With Warnings Only", ""])
    if warning_rows:
        for row in warning_rows:
            lines.append(
                f"- `{row['own_id']}` {row['dataset_name']}: GO_WITH_WARNINGS,"
                f" v2_strict=`{row['v2_strict_sql_count']}`, next=`{row['recommended_next_action']}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Do-Not-Use-Yet", ""])
    if hold_rows:
        for row in hold_rows:
            lines.append(
                f"- `{row['own_id']}` {row['dataset_name']}: HOLD,"
                f" v2_strict=`{row['v2_strict_sql_count']}`, collision=`{row['v2_collision_risk_sql_count']}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Concrete Next Actions For HOLD Datasets", ""])
    if hold_rows:
        for row in hold_rows:
            lines.append(f"- `{row['own_id']}` {row['dataset_name']}: {row['recommended_next_action']}")
    else:
        lines.append("- None")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (output_root / "final_v2").resolve()
    baseline_final_index_path = args.baseline_final_index.resolve()
    baseline_morning_table_path = args.baseline_morning_table.resolve()
    final_index_v2_path = args.final_index_v2.resolve()
    gate_csv_path = args.gate_csv.resolve()
    execute_csv_path = args.execute_csv.resolve()
    dedup_csv_path = args.dedup_csv.resolve()

    baseline_rows = read_csv_rows(baseline_final_index_path)
    baseline_morning_rows = read_csv_rows(baseline_morning_table_path)
    v2_rows = read_csv_rows(final_index_v2_path)
    gate_rows = read_csv_rows(gate_csv_path)
    execute_rows = read_csv_rows(execute_csv_path)
    dedup_rows = read_csv_rows(dedup_csv_path)

    baseline_by_id = {(row.get("own_id") or "").strip(): row for row in baseline_rows}
    baseline_morning_by_id = {(row.get("own_id") or "").strip(): row for row in baseline_morning_rows}
    v2_by_id = {(row.get("own_id") or "").strip(): row for row in v2_rows}
    gate_by_id = {(row.get("own_id") or "").strip(): row for row in gate_rows}

    ordered_own_ids = [(row.get("own_id") or "").strip() for row in v2_rows]
    delta_rows: list[dict[str, str]] = []
    enriched_rows: list[dict[str, str]] = []

    for own_id in ordered_own_ids:
        baseline_row = baseline_by_id.get(own_id, {})
        baseline_morning_row = baseline_morning_by_id.get(own_id, {})
        v2_row = v2_by_id.get(own_id, {})
        gate_row = gate_by_id.get(own_id, {})

        baseline_strict = to_int(baseline_row.get("strict_sql_count"))
        v2_strict = to_int(v2_row.get("strict_keep_count_v2"))
        baseline_collision = to_int(baseline_row.get("collision_risk_sql_count"))
        v2_collision = to_int(gate_row.get("collision_primary_count_v2"))
        baseline_readiness = baseline_ready_value(baseline_row, baseline_morning_row)
        v2_readiness = (v2_row.get("readiness_label_v2") or "NOT_READY").strip()
        row = {
            "own_id": own_id,
            "dataset_name": (v2_row.get("dataset_name") or baseline_row.get("dataset_name") or "").strip(),
            "baseline_strict_sql_count": str(baseline_strict),
            "v2_strict_sql_count": str(v2_strict),
            "strict_delta": str(v2_strict - baseline_strict),
            "baseline_collision_risk_sql_count": str(baseline_collision),
            "v2_collision_risk_sql_count": str(v2_collision),
            "collision_delta": str(v2_collision - baseline_collision),
            "baseline_readiness": baseline_readiness,
            "v2_readiness_label": v2_readiness,
            "baseline_top_strict_count": str(to_int(baseline_row.get("top_strict_sql_count"))),
            "v2_top_strict_count": str(to_int(v2_row.get("top_strict_sql_count_v2"))),
            "baseline_seed_count": str(to_int(baseline_row.get("question_seed_count"))),
            "v2_seed_count": str(to_int(v2_row.get("question_seed_count_v2"))),
            "recommended_decision": recommended_decision(v2_readiness),
            "recommended_next_action": (v2_row.get("recommended_next_action") or "").strip(),
        }
        delta_rows.append({key: row[key] for key in DELTA_FIELDNAMES})
        enriched_rows.append(row)

    enriched_rows.sort(
        key=lambda row: (
            READINESS_PRIORITY.get(row["v2_readiness_label"], 9),
            -to_int(row["v2_strict_sql_count"]),
            row["own_id"],
        )
    )
    delta_rows = [{key: row[key] for key in DELTA_FIELDNAMES} for row in enriched_rows]

    delta_csv_path = output_dir / "delta_baseline_vs_v2.csv"
    delta_md_path = output_dir / "delta_baseline_vs_v2.md"
    checkpoint_path = output_dir / "checkpoint_final_status.json"

    write_csv(delta_csv_path, DELTA_FIELDNAMES, delta_rows)

    safe_rows = [row for row in enriched_rows if row["recommended_decision"] == "GO"]
    warning_rows = [row for row in enriched_rows if row["recommended_decision"] == "GO_WITH_WARNINGS"]
    hold_rows = [row for row in enriched_rows if row["recommended_decision"] == "HOLD"]
    top_improved = sorted(enriched_rows, key=improvement_sort_key)[:10]
    top_degraded = sorted(enriched_rows, key=degradation_sort_key)[:10]
    dedup_summary = compute_dedup_summary(dedup_rows)
    execute_summary = compute_execute_summary(execute_rows)

    write_text(
        delta_md_path,
        render_markdown(
            now_utc=utc_now_iso(),
            baseline_final_index=baseline_final_index_path,
            baseline_morning_table=baseline_morning_table_path,
            final_index_v2=final_index_v2_path,
            gate_csv=gate_csv_path,
            execute_csv=execute_csv_path,
            dedup_csv=dedup_csv_path,
            rows=enriched_rows,
            hold_rows=hold_rows,
            safe_rows=safe_rows,
            warning_rows=warning_rows,
            top_improved=top_improved,
            top_degraded=top_degraded,
            dedup_summary=dedup_summary,
            execute_summary=execute_summary,
        ),
    )

    required_stage_outputs = [
        output_dir / "final_index_v2.csv",
        output_dir / "final_overview_v2.md",
        output_dir / "morning_review_dataset_table_v2.csv",
        output_dir / "morning_review_risks_v2.md",
        output_dir / "run_manifest_v2_phase4.json",
        delta_csv_path,
        delta_md_path,
    ]
    missing_artifacts = [str(path) for path in required_stage_outputs if not path.exists()]
    final_index_dataset_count = len(v2_rows)
    notes: list[str] = [
        f"final_index_v2 dataset count = {final_index_dataset_count}",
        f"GO datasets = {len(safe_rows)}",
        f"GO_WITH_WARNINGS datasets = {len(warning_rows)}",
        f"HOLD datasets = {len(hold_rows)}",
    ]
    if missing_artifacts:
        notes.append("Missing required artifacts:")
        notes.extend(missing_artifacts)
    if final_index_dataset_count != 26:
        notes.append(f"Expected 26 datasets in final_index_v2 but found {final_index_dataset_count}.")

    status = "PASS" if not missing_artifacts and final_index_dataset_count == 26 else "FAIL"
    checkpoint_payload = {
        "status": status,
        "ready_for_human_review": status == "PASS",
        "v2_ready_count": len(safe_rows),
        "v2_ready_with_warnings_count": len(warning_rows),
        "v2_hold_count": len(hold_rows),
        "notes": notes,
        "inputs": {
            "baseline_final_index": str(baseline_final_index_path),
            "baseline_final_index_sha256": sha256_file(baseline_final_index_path),
            "baseline_morning_review_table": str(baseline_morning_table_path),
            "baseline_morning_review_table_sha256": sha256_file(baseline_morning_table_path),
            "v2_final_index": str(final_index_v2_path),
            "v2_final_index_sha256": sha256_file(final_index_v2_path),
            "gate_csv": str(gate_csv_path),
            "gate_csv_sha256": sha256_file(gate_csv_path),
            "execute_csv": str(execute_csv_path),
            "execute_csv_sha256": sha256_file(execute_csv_path),
            "dedup_csv": str(dedup_csv_path),
            "dedup_csv_sha256": sha256_file(dedup_csv_path),
        },
        "outputs": {
            "delta_baseline_vs_v2_csv": str(delta_csv_path),
            "delta_baseline_vs_v2_md": str(delta_md_path),
            "checkpoint_final_status_json": str(checkpoint_path),
        },
    }
    write_json(checkpoint_path, checkpoint_payload)

    print(f"FINAL_STATUS\t{status}")
    print(f"V2_READY\t{len(safe_rows)}")
    print(f"V2_READY_WITH_WARNINGS\t{len(warning_rows)}")
    print(f"V2_HOLD\t{len(hold_rows)}")


if __name__ == "__main__":
    main()
