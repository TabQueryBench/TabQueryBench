#!/usr/bin/env python3
"""Apply strict V2 readiness gating for question taxonomy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RECLASSIFY = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/reclassify/master_sql_inventory_reclassified_v2.csv"
)
DEFAULT_DEDUP = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/dedup/master_sql_inventory_dedup_v2.csv"
)
DEFAULT_EXECUTE = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/execute/sql_executability_v2.csv"
)
DEFAULT_BASELINE_TABLE = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/baseline/baseline_dataset_table.csv"
)
DEFAULT_SCOPE = Path("logs/sql_high_corpus_build_20260404/scope/high_datasets.csv")
DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404/v2_refinement")

CSV_FIELDNAMES = [
    "own_id",
    "dataset_id",
    "dataset_name",
    "reclassify_row_count_v2",
    "dedup_row_count_v2",
    "execute_row_count_v2",
    "phase_row_count_consistent_v2",
    "primary_sql_rows_v2",
    "duplicate_dropped_count_v2",
    "keep_candidate_primary_rows_v2",
    "strict_keep_count_v2",
    "weak_keep_count_v2",
    "collision_primary_count_v2",
    "strict_keep_ratio_v2",
    "collision_ratio_v2",
    "duplicate_burden_v2",
    "strict_pass_count_v2",
    "strict_fail_count_v2",
    "strict_unknown_count_v2",
    "executability_pass_ratio_v2",
    "source_credibility_score_v2",
    "source_credibility_tier_mix_v2",
    "readiness_label_v2",
    "gate_reason_codes_v2",
    "recommended_next_action",
]

CREDIBILITY_SCORES = {
    "tier_1_official": 1.00,
    "tier_2_primary_code": 0.85,
    "tier_3_secondary_explanatory": 0.60,
    "tier_4_low_trust": 0.30,
    "": 0.00,
}

READY_THRESHOLDS = {
    "strict_keep_count_v2": 20,
    "strict_keep_ratio_v2": 0.85,
    "collision_ratio_v2": 0.10,
    "duplicate_burden_v2": 0.35,
    "executability_pass_ratio_v2": 0.60,
    "source_credibility_score_v2": 0.80,
}

WARNING_THRESHOLDS = {
    "strict_keep_count_v2": 10,
    "strict_keep_ratio_v2": 0.40,
    "collision_ratio_v2": 0.50,
    "duplicate_burden_v2": 0.50,
    "executability_pass_ratio_v2": 0.50,
    "source_credibility_score_v2": 0.80,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply strict V2 dataset readiness gating.")
    parser.add_argument("--reclassify", type=Path, default=DEFAULT_RECLASSIFY)
    parser.add_argument("--dedup", type=Path, default=DEFAULT_DEDUP)
    parser.add_argument("--execute", type=Path, default=DEFAULT_EXECUTE)
    parser.add_argument("--baseline-table", type=Path, default=DEFAULT_BASELINE_TABLE)
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def format_ratio(value: float) -> str:
    return f"{value:.6f}"


def load_dataset_roster(baseline_table: Path, scope_csv: Path) -> list[dict[str, str]]:
    if baseline_table.exists():
        rows = read_csv_rows(baseline_table)
        if rows:
            return rows
    return read_csv_rows(scope_csv)


def tier_mix(rows: list[dict[str, str]]) -> str:
    counts = Counter(row.get("v2_source_credibility_tier", "") for row in rows if row.get("v2_source_credibility_tier", ""))
    if not counts:
        return ""
    return "; ".join(f"{tier}:{counts[tier]}" for tier in sorted(counts))


def credibility_score(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    total = sum(CREDIBILITY_SCORES.get(row.get("v2_source_credibility_tier", ""), 0.0) for row in rows)
    return total / len(rows)


def label_reason_codes(metrics: dict[str, float], row_counts: dict[str, int]) -> list[str]:
    reasons: list[str] = []
    if row_counts["dedup_row_count_v2"] == 0:
        return ["no_sql_rows_v2"]
    if metrics["strict_keep_count_v2"] == 0:
        reasons.append("no_strict_kept_sql")
    elif metrics["strict_keep_count_v2"] < WARNING_THRESHOLDS["strict_keep_count_v2"]:
        reasons.append("strict_kept_sql_too_sparse")
    if metrics["strict_keep_ratio_v2"] < WARNING_THRESHOLDS["strict_keep_ratio_v2"]:
        reasons.append("low_strict_keep_purity")
    if metrics["collision_ratio_v2"] > WARNING_THRESHOLDS["collision_ratio_v2"]:
        reasons.append("high_collision_contamination")
    elif metrics["collision_ratio_v2"] > READY_THRESHOLDS["collision_ratio_v2"]:
        reasons.append("moderate_collision_contamination")
    if metrics["duplicate_burden_v2"] > WARNING_THRESHOLDS["duplicate_burden_v2"]:
        reasons.append("high_duplicate_burden")
    elif metrics["duplicate_burden_v2"] > READY_THRESHOLDS["duplicate_burden_v2"]:
        reasons.append("moderate_duplicate_burden")
    if metrics["executability_pass_ratio_v2"] < WARNING_THRESHOLDS["executability_pass_ratio_v2"]:
        reasons.append("low_strict_executability_pass_ratio")
    elif metrics["executability_pass_ratio_v2"] < READY_THRESHOLDS["executability_pass_ratio_v2"]:
        reasons.append("borderline_strict_executability_pass_ratio")
    if metrics["source_credibility_score_v2"] < WARNING_THRESHOLDS["source_credibility_score_v2"]:
        reasons.append("low_source_credibility")
    return reasons


def readiness_label(metrics: dict[str, float], row_counts: dict[str, int]) -> str:
    if row_counts["dedup_row_count_v2"] == 0:
        return "NOT_READY"

    if (
        metrics["strict_keep_count_v2"] >= READY_THRESHOLDS["strict_keep_count_v2"]
        and metrics["strict_keep_ratio_v2"] >= READY_THRESHOLDS["strict_keep_ratio_v2"]
        and metrics["collision_ratio_v2"] <= READY_THRESHOLDS["collision_ratio_v2"]
        and metrics["duplicate_burden_v2"] <= READY_THRESHOLDS["duplicate_burden_v2"]
        and metrics["executability_pass_ratio_v2"] >= READY_THRESHOLDS["executability_pass_ratio_v2"]
        and metrics["source_credibility_score_v2"] >= READY_THRESHOLDS["source_credibility_score_v2"]
    ):
        return "READY"

    if (
        metrics["strict_keep_count_v2"] >= WARNING_THRESHOLDS["strict_keep_count_v2"]
        and metrics["strict_keep_ratio_v2"] >= WARNING_THRESHOLDS["strict_keep_ratio_v2"]
        and metrics["collision_ratio_v2"] <= WARNING_THRESHOLDS["collision_ratio_v2"]
        and metrics["duplicate_burden_v2"] <= WARNING_THRESHOLDS["duplicate_burden_v2"]
        and metrics["executability_pass_ratio_v2"] >= WARNING_THRESHOLDS["executability_pass_ratio_v2"]
        and metrics["source_credibility_score_v2"] >= WARNING_THRESHOLDS["source_credibility_score_v2"]
    ):
        return "READY_WITH_WARNINGS"

    return "NOT_READY"


def recommended_action(label: str, metrics: dict[str, float], row_counts: dict[str, int]) -> str:
    if row_counts["dedup_row_count_v2"] == 0:
        return "Run targeted source search plus SQL extraction for this dataset; there are no V2 SQL rows to gate."
    if metrics["strict_keep_count_v2"] == 0 and row_counts["keep_candidate_primary_rows_v2"] == 0:
        if metrics["collision_ratio_v2"] >= 0.50:
            return "Recollect benchmark-specific SQL and replace collision-risk inventory before rerunning V2 gating."
        return "Collect benchmark-specific strict SQL sources; current inventory has no kept taxonomy-ready rows."
    if metrics["strict_keep_count_v2"] == 0:
        if metrics["collision_ratio_v2"] >= 0.50:
            return "Replace collision-risk retained rows with strict benchmark-matched SQL before taxonomy."
        return "Promote weak kept rows to strict only with explicit schema or benchmark evidence, or recollect stricter sources."
    if metrics["strict_keep_count_v2"] < WARNING_THRESHOLDS["strict_keep_count_v2"]:
        return "Expand the strict kept SQL core to at least 10 canonical rows before taxonomy drafting."
    if metrics["collision_ratio_v2"] > WARNING_THRESHOLDS["collision_ratio_v2"]:
        return "Reduce collision-risk contamination with targeted source replacement before question taxonomy."
    if metrics["duplicate_burden_v2"] > WARNING_THRESHOLDS["duplicate_burden_v2"]:
        return "Add more unique strict SQL patterns; duplicate burden is too high for stable taxonomy seeding."
    if metrics["executability_pass_ratio_v2"] < WARNING_THRESHOLDS["executability_pass_ratio_v2"]:
        return "Repair or replace non-passing strict SQL with cleaner SQLite-portable variants before taxonomy."
    if metrics["source_credibility_score_v2"] < WARNING_THRESHOLDS["source_credibility_score_v2"]:
        return "Replace low-trust SQL sources with official or primary-code evidence before taxonomy."
    if label == "READY_WITH_WARNINGS":
        if metrics["collision_ratio_v2"] > READY_THRESHOLDS["collision_ratio_v2"] or metrics["strict_keep_ratio_v2"] < READY_THRESHOLDS["strict_keep_ratio_v2"]:
            return "Proceed only with manual review of weak or collision spillover around the strict core."
        return "Proceed with taxonomy drafting, but manually review strict rows that did not pass lightweight executability checks."
    return "Proceed to question taxonomy using the strict kept primary SQL set."


def render_report(
    now_utc: str,
    reclassify_path: Path,
    dedup_path: Path,
    execute_path: Path,
    rows: list[dict[str, str]],
    label_counts: Counter[str],
    phase_consistency_ok: bool,
) -> str:
    ready_rows = [row for row in rows if row["readiness_label_v2"] == "READY"]
    warning_rows = [row for row in rows if row["readiness_label_v2"] == "READY_WITH_WARNINGS"]
    not_ready_rows = [row for row in rows if row["readiness_label_v2"] == "NOT_READY"]

    lines = [
        "# V2 Readiness Gate Report",
        "",
        f"- Generated at UTC: `{now_utc}`",
        f"- Phase 1 input: `{reclassify_path.resolve()}`",
        f"- Phase 2 input: `{dedup_path.resolve()}`",
        f"- Phase 3 input: `{execute_path.resolve()}`",
        f"- Phase row-count consistency across inputs: `{phase_consistency_ok}`",
        "",
        "## Gate Dimensions",
        "",
        "- `strict_keep_count_v2`: primary canonical rows with `v2_keep_candidate=yes` and `v2_specificity_label=strict`.",
        "- `strict_keep_ratio_v2`: `strict_keep_count_v2 / keep_candidate_primary_rows_v2`.",
        "- `collision_ratio_v2`: primary canonical collision-risk rows divided by all primary canonical rows.",
        "- `duplicate_burden_v2`: `(dedup_row_count_v2 - primary_sql_rows_v2) / dedup_row_count_v2`.",
        "- `executability_pass_ratio_v2`: strict kept rows with `executable_status_v2=pass` divided by `strict_keep_count_v2`.",
        "- `source_credibility_score_v2`: mean source-tier score across kept primary rows using tier1=1.00, tier2=0.85, tier3=0.60, tier4=0.30.",
        "",
        "## Thresholds",
        "",
        "- `READY`: strict_keep_count>=20, strict_keep_ratio>=0.85, collision_ratio<=0.10, duplicate_burden<=0.35, executability_pass_ratio>=0.60, source_credibility_score>=0.80.",
        "- `READY_WITH_WARNINGS`: strict_keep_count>=10, strict_keep_ratio>=0.40, collision_ratio<=0.50, duplicate_burden<=0.50, executability_pass_ratio>=0.50, source_credibility_score>=0.80.",
        "- `NOT_READY`: anything else, including datasets with zero V2 SQL rows.",
        "",
        "## Label Summary",
        "",
        f"- Total datasets evaluated: {len(rows)}",
        f"- `READY`: {label_counts['READY']}",
        f"- `READY_WITH_WARNINGS`: {label_counts['READY_WITH_WARNINGS']}",
        f"- `NOT_READY`: {label_counts['NOT_READY']}",
        "",
        "## Ready Datasets",
        "",
    ]

    if ready_rows:
        for row in sorted(ready_rows, key=lambda item: (-int(item["strict_keep_count_v2"]), item["own_id"])):
            lines.append(
                "- `{own_id}` {dataset_name}: strict_keep={strict_keep_count_v2}, pass_ratio={executability_pass_ratio_v2}, duplicate_burden={duplicate_burden_v2}".format(
                    **row
                )
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Ready With Warnings", ""])
    if warning_rows:
        for row in sorted(warning_rows, key=lambda item: (-int(item["strict_keep_count_v2"]), item["own_id"])):
            lines.append(
                "- `{own_id}` {dataset_name}: reasons=`{gate_reason_codes_v2}`; next=`{recommended_next_action}`".format(
                    **row
                )
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Not Ready Highlights", ""])
    for row in sorted(not_ready_rows, key=lambda item: (-int(item["strict_keep_count_v2"]), item["own_id"]))[:12]:
        lines.append(
            "- `{own_id}` {dataset_name}: reasons=`{gate_reason_codes_v2}`; next=`{recommended_next_action}`".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Full Gate Table",
            "",
            "| own_id | dataset_name | label | strict_keep | strict_ratio | collision_ratio | duplicate_burden | exec_pass_ratio | credibility | next_action |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {own_id} | {dataset_name} | {readiness_label_v2} | {strict_keep_count_v2} | {strict_keep_ratio_v2} | "
            "{collision_ratio_v2} | {duplicate_burden_v2} | {executability_pass_ratio_v2} | {source_credibility_score_v2} | {recommended_next_action} |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reclassify_path = args.reclassify.resolve()
    dedup_path = args.dedup.resolve()
    execute_path = args.execute.resolve()
    baseline_table = args.baseline_table.resolve()
    scope_csv = args.scope_csv.resolve()
    output_dir = args.output_root.resolve() / "gate"
    csv_path = output_dir / "dataset_readiness_v2.csv"
    report_path = output_dir / "readiness_gate_report_v2.md"
    status_path = output_dir / "checkpoint2_status.json"

    now_utc = utc_now_iso()
    roster = load_dataset_roster(baseline_table, scope_csv)
    roster_by_id = {row["own_id"]: row for row in roster}
    ordered_ids = [row["own_id"] for row in roster]

    reclassify_rows = read_csv_rows(reclassify_path)
    dedup_rows = read_csv_rows(dedup_path)
    execute_rows = read_csv_rows(execute_path)

    reclassify_by = defaultdict(list)
    dedup_by = defaultdict(list)
    execute_by = defaultdict(list)
    for row in reclassify_rows:
        reclassify_by[row["own_id"]].append(row)
    for row in dedup_rows:
        dedup_by[row["own_id"]].append(row)
    for row in execute_rows:
        execute_by[row["own_id"]].append(row)

    output_rows: list[dict[str, str]] = []
    label_counts: Counter[str] = Counter()
    phase_consistency_ok = True
    inconsistent_datasets: list[dict[str, Any]] = []

    for own_id in ordered_ids:
        meta = roster_by_id[own_id]
        dataset_id = meta.get("dataset_id", "")
        dataset_name = meta.get("dataset_name", "")

        phase1 = reclassify_by.get(own_id, [])
        phase2 = dedup_by.get(own_id, [])
        phase3 = execute_by.get(own_id, [])

        reclassify_count = len(phase1)
        dedup_count = len(phase2)
        execute_count = len(phase3)
        consistent = reclassify_count == dedup_count == execute_count
        if not consistent:
            phase_consistency_ok = False
            inconsistent_datasets.append(
                {
                    "own_id": own_id,
                    "dataset_name": dataset_name,
                    "reclassify_row_count_v2": reclassify_count,
                    "dedup_row_count_v2": dedup_count,
                    "execute_row_count_v2": execute_count,
                }
            )

        primary = [row for row in phase2 if row.get("is_primary_canonical") == "yes"]
        keep_primary = [row for row in primary if row.get("v2_keep_candidate") == "yes"]
        strict_keep = [row for row in keep_primary if row.get("v2_specificity_label") == "strict"]
        weak_keep = [row for row in keep_primary if row.get("v2_specificity_label") == "weak"]
        collision_primary = [row for row in primary if row.get("v2_specificity_label") == "collision_risk"]
        strict_execute = [
            row
            for row in phase3
            if row.get("is_primary_canonical") == "yes"
            and row.get("v2_keep_candidate") == "yes"
            and row.get("v2_specificity_label") == "strict"
        ]
        strict_pass_count = sum(1 for row in strict_execute if row.get("executable_status_v2") == "pass")
        strict_fail_count = sum(1 for row in strict_execute if row.get("executable_status_v2") == "fail")
        strict_unknown_count = sum(1 for row in strict_execute if row.get("executable_status_v2") == "unknown")

        row_counts = {
            "reclassify_row_count_v2": reclassify_count,
            "dedup_row_count_v2": dedup_count,
            "execute_row_count_v2": execute_count,
            "primary_sql_rows_v2": len(primary),
            "duplicate_dropped_count_v2": max(dedup_count - len(primary), 0),
            "keep_candidate_primary_rows_v2": len(keep_primary),
        }
        metrics = {
            "strict_keep_count_v2": len(strict_keep),
            "weak_keep_count_v2": len(weak_keep),
            "collision_primary_count_v2": len(collision_primary),
            "strict_keep_ratio_v2": (
                len(strict_keep) / len(keep_primary) if keep_primary else 0.0
            ),
            "collision_ratio_v2": (
                len(collision_primary) / len(primary) if primary else 0.0
            ),
            "duplicate_burden_v2": (
                (dedup_count - len(primary)) / dedup_count if dedup_count else 0.0
            ),
            "strict_pass_count_v2": strict_pass_count,
            "strict_fail_count_v2": strict_fail_count,
            "strict_unknown_count_v2": strict_unknown_count,
            "executability_pass_ratio_v2": (
                strict_pass_count / len(strict_execute) if strict_execute else 0.0
            ),
            "source_credibility_score_v2": credibility_score(keep_primary),
        }

        label = readiness_label(metrics, row_counts)
        reasons = label_reason_codes(metrics, row_counts)
        if label == "READY":
            reasons = ["ready_thresholds_met"]
        elif label == "READY_WITH_WARNINGS" and not reasons:
            reasons = ["warning_thresholds_met"]
        action = recommended_action(label, metrics, row_counts)

        output_row = {
            "own_id": own_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "reclassify_row_count_v2": str(reclassify_count),
            "dedup_row_count_v2": str(dedup_count),
            "execute_row_count_v2": str(execute_count),
            "phase_row_count_consistent_v2": "yes" if consistent else "no",
            "primary_sql_rows_v2": str(row_counts["primary_sql_rows_v2"]),
            "duplicate_dropped_count_v2": str(row_counts["duplicate_dropped_count_v2"]),
            "keep_candidate_primary_rows_v2": str(row_counts["keep_candidate_primary_rows_v2"]),
            "strict_keep_count_v2": str(metrics["strict_keep_count_v2"]),
            "weak_keep_count_v2": str(metrics["weak_keep_count_v2"]),
            "collision_primary_count_v2": str(metrics["collision_primary_count_v2"]),
            "strict_keep_ratio_v2": format_ratio(metrics["strict_keep_ratio_v2"]),
            "collision_ratio_v2": format_ratio(metrics["collision_ratio_v2"]),
            "duplicate_burden_v2": format_ratio(metrics["duplicate_burden_v2"]),
            "strict_pass_count_v2": str(metrics["strict_pass_count_v2"]),
            "strict_fail_count_v2": str(metrics["strict_fail_count_v2"]),
            "strict_unknown_count_v2": str(metrics["strict_unknown_count_v2"]),
            "executability_pass_ratio_v2": format_ratio(metrics["executability_pass_ratio_v2"]),
            "source_credibility_score_v2": format_ratio(metrics["source_credibility_score_v2"]),
            "source_credibility_tier_mix_v2": tier_mix(keep_primary),
            "readiness_label_v2": label,
            "gate_reason_codes_v2": ",".join(reasons),
            "recommended_next_action": action,
        }
        output_rows.append(output_row)
        label_counts[label] += 1

    write_csv(csv_path, CSV_FIELDNAMES, output_rows)

    report_text = render_report(
        now_utc=now_utc,
        reclassify_path=reclassify_path,
        dedup_path=dedup_path,
        execute_path=execute_path,
        rows=output_rows,
        label_counts=label_counts,
        phase_consistency_ok=phase_consistency_ok,
    )
    write_text(report_path, report_text)

    status_payload = {
        "checkpoint": "2",
        "phase_name": "v2_dataset_readiness_gate",
        "generated_at_utc": now_utc,
        "status": "PASS" if phase_consistency_ok and len(output_rows) == len(ordered_ids) else "FAIL",
        "summary": {
            "dataset_count": len(output_rows),
            "ready_count": label_counts["READY"],
            "ready_with_warnings_count": label_counts["READY_WITH_WARNINGS"],
            "not_ready_count": label_counts["NOT_READY"],
            "phase_row_count_consistency_ok": phase_consistency_ok,
        },
        "thresholds": {
            "ready": READY_THRESHOLDS,
            "ready_with_warnings": WARNING_THRESHOLDS,
        },
        "ready_datasets": [
            {
                "own_id": row["own_id"],
                "dataset_name": row["dataset_name"],
                "strict_keep_count_v2": int(row["strict_keep_count_v2"]),
                "executability_pass_ratio_v2": float(row["executability_pass_ratio_v2"]),
            }
            for row in output_rows
            if row["readiness_label_v2"] == "READY"
        ],
        "ready_with_warnings_datasets": [
            {
                "own_id": row["own_id"],
                "dataset_name": row["dataset_name"],
                "gate_reason_codes_v2": row["gate_reason_codes_v2"],
                "recommended_next_action": row["recommended_next_action"],
            }
            for row in output_rows
            if row["readiness_label_v2"] == "READY_WITH_WARNINGS"
        ],
        "inconsistent_datasets": inconsistent_datasets,
        "input": {
            "reclassify_path": str(reclassify_path),
            "reclassify_sha256": sha256_file(reclassify_path),
            "dedup_path": str(dedup_path),
            "dedup_sha256": sha256_file(dedup_path),
            "execute_path": str(execute_path),
            "execute_sha256": sha256_file(execute_path),
        },
        "outputs": {
            "dataset_readiness_v2_csv": str(csv_path),
            "readiness_gate_report_v2_md": str(report_path),
            "checkpoint2_status_json": str(status_path),
        },
    }
    write_json(status_path, status_payload)

    for row in output_rows:
        print(
            f"{row['own_id']}\t{row['readiness_label_v2']}\tstrict_keep={row['strict_keep_count_v2']}\tpass_ratio={row['executability_pass_ratio_v2']}"
        )


if __name__ == "__main__":
    main()
