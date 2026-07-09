#!/usr/bin/env python3
"""Generate a human-audit package for V2 Phase 1 SQL reclassification results."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_phase_c_sql_inventory import (  # noqa: E402
    leading_sql_candidate,
    normalize_url_root,
)


DEFAULT_INPUT = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/reclassify/master_sql_inventory_reclassified_v2.csv"
)
DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404/v2_refinement")
FOCUS_DATASETS = ["c5", "c7", "c2", "n16", "c13", "c19", "m10"]
TOP_STRICT_LOSS_LIMIT = 10
MIN_SAMPLE_ROWS = 30
FOCUS_SAMPLE_LIMIT = 5
OTHER_TOP_SAMPLE_LIMIT = 3
MASS_RELABEL_MIN_STRICT_OLD = 10
MASS_RELABEL_MIN_STRICT_LOSS_COUNT = 20
MASS_RELABEL_MIN_STRICT_LOSS_RATIO = 0.50
EXPLAINED_REASON_TOP3_COVERAGE_THRESHOLD = 0.80
CSV_FIELDNAMES = [
    "own_id",
    "dataset_name",
    "total_rows",
    "old_strict_count",
    "old_weak_count",
    "old_collision_risk_count",
    "new_strict_count",
    "new_weak_count",
    "new_collision_risk_count",
    "new_reject_non_sql_count",
    "strict_to_strict_count",
    "strict_to_weak_count",
    "strict_to_collision_count",
    "strict_to_reject_count",
    "strict_loss_count",
    "strict_loss_ratio",
    "label_change_count",
    "missing_reason_code_count",
    "missing_reason_text_count",
    "mass_relabeling_trigger",
    "explained_mass_relabeling",
    "strict_loss_top_reason_codes",
    "dominant_changed_reason_codes",
    "dominant_source_roots",
    "focus_review_required",
    "audit_status",
    "audit_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Phase 1 V2 reclassification outputs with emphasis on strict-label "
            "losses, risky datasets, and human-review samples."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
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


def to_int(value: str | int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = value.strip()
    if not text:
        return 0
    return int(text)


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def reason_summary(counter: Counter[str], top_n: int = 3) -> str:
    items = [(key, count) for key, count in counter.items() if key]
    if not items:
        return ""
    items.sort(key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{key}={count}" for key, count in items[:top_n])


def source_root_summary(counter: Counter[str], top_n: int = 3) -> str:
    items = [(key, count) for key, count in counter.items() if key]
    if not items:
        return ""
    items.sort(key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{key}={count}" for key, count in items[:top_n])


def sql_snippet(sql_text: str, limit: int = 160) -> str:
    candidate = leading_sql_candidate(sql_text or "")
    candidate = " ".join(candidate.split())
    if len(candidate) <= limit:
        return candidate
    return candidate[: limit - 3] + "..."


def strict_loss_ratio_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["strict_loss_ratio"]),
        -int(row["strict_loss_count"]),
        -int(row["old_strict_count"]),
        row["own_id"],
    )


def row_sample_priority(row: dict[str, str]) -> tuple[Any, ...]:
    old_label = (row.get("dataset_specificity_label") or "").strip()
    new_label = (row.get("v2_specificity_label") or "").strip()
    return (
        0 if old_label == "strict" and new_label != "strict" else 1,
        0 if old_label != new_label else 1,
        0 if new_label == "reject_non_sql" else 1,
        0 if new_label == "collision_risk" else 1,
        0 if (row.get("v2_keep_candidate") or "").strip() == "no" else 1,
        (row.get("v2_specificity_reason_code") or "").strip(),
        (row.get("sql_item_id") or "").strip(),
    )


def build_dataset_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("own_id") or "").strip()].append(row)

    dataset_summaries: list[dict[str, Any]] = []
    for own_id, dataset_rows in grouped.items():
        dataset_name = (dataset_rows[0].get("dataset_name") or "").strip()
        old_counts = Counter((row.get("dataset_specificity_label") or "").strip() or "unknown" for row in dataset_rows)
        new_counts = Counter((row.get("v2_specificity_label") or "").strip() or "unknown" for row in dataset_rows)
        transitions = Counter(
            (
                (row.get("dataset_specificity_label") or "").strip() or "unknown",
                (row.get("v2_specificity_label") or "").strip() or "unknown",
            )
            for row in dataset_rows
        )
        changed_rows = [
            row
            for row in dataset_rows
            if (row.get("dataset_specificity_label") or "").strip()
            != (row.get("v2_specificity_label") or "").strip()
        ]
        strict_loss_rows = [
            row
            for row in dataset_rows
            if (row.get("dataset_specificity_label") or "").strip() == "strict"
            and (row.get("v2_specificity_label") or "").strip() != "strict"
        ]
        strict_old_count = old_counts.get("strict", 0)
        strict_to_strict = transitions.get(("strict", "strict"), 0)
        strict_to_weak = transitions.get(("strict", "weak"), 0)
        strict_to_collision = transitions.get(("strict", "collision_risk"), 0)
        strict_to_reject = transitions.get(("strict", "reject_non_sql"), 0)
        strict_loss_count = strict_to_weak + strict_to_collision + strict_to_reject
        strict_loss_ratio = ratio(strict_loss_count, strict_old_count)
        changed_reason_counter = Counter(
            (row.get("v2_specificity_reason_code") or "").strip()
            for row in changed_rows
            if (row.get("v2_specificity_reason_code") or "").strip()
        )
        strict_loss_reason_counter = Counter(
            (row.get("v2_specificity_reason_code") or "").strip()
            for row in strict_loss_rows
            if (row.get("v2_specificity_reason_code") or "").strip()
        )
        source_root_counter = Counter(
            normalize_url_root((row.get("source_url") or "").strip())
            for row in dataset_rows
            if (row.get("source_url") or "").strip()
        )
        missing_reason_code_count = sum(1 for row in dataset_rows if not (row.get("v2_specificity_reason_code") or "").strip())
        missing_reason_text_count = sum(1 for row in dataset_rows if not (row.get("v2_specificity_reason_text") or "").strip())
        missing_strict_loss_reason_code_count = sum(
            1 for row in strict_loss_rows if not (row.get("v2_specificity_reason_code") or "").strip()
        )
        missing_strict_loss_reason_text_count = sum(
            1 for row in strict_loss_rows if not (row.get("v2_specificity_reason_text") or "").strip()
        )

        top3_reason_coverage = 1.0
        if strict_loss_count > 0:
            top3_reason_coverage = ratio(
                sum(count for _, count in strict_loss_reason_counter.most_common(3)),
                strict_loss_count,
            )

        mass_relabeling_trigger = (
            strict_old_count >= MASS_RELABEL_MIN_STRICT_OLD
            and (
                strict_loss_count >= MASS_RELABEL_MIN_STRICT_LOSS_COUNT
                or strict_loss_ratio >= MASS_RELABEL_MIN_STRICT_LOSS_RATIO
            )
        )
        explained_mass_relabeling = (
            not mass_relabeling_trigger
            or (
                missing_strict_loss_reason_code_count == 0
                and missing_strict_loss_reason_text_count == 0
                and top3_reason_coverage >= EXPLAINED_REASON_TOP3_COVERAGE_THRESHOLD
            )
        )

        focus_required = own_id in FOCUS_DATASETS
        audit_status = "PASS"
        if missing_reason_code_count > 0 or missing_reason_text_count > 0 or not explained_mass_relabeling:
            audit_status = "FAIL"

        if strict_old_count == 0:
            note = "No baseline strict rows; audit focuses on non-strict relabeling stability."
        elif strict_loss_count == 0:
            note = "No strict loss detected."
        elif explained_mass_relabeling:
            note = (
                f"Strict loss is explained by populated reasons; top reason coverage={top3_reason_coverage:.3f}."
            )
        else:
            note = (
                f"Strict loss is not sufficiently explained; top reason coverage={top3_reason_coverage:.3f}."
            )

        dataset_summaries.append(
            {
                "own_id": own_id,
                "dataset_name": dataset_name,
                "total_rows": len(dataset_rows),
                "old_strict_count": strict_old_count,
                "old_weak_count": old_counts.get("weak", 0),
                "old_collision_risk_count": old_counts.get("collision_risk", 0),
                "new_strict_count": new_counts.get("strict", 0),
                "new_weak_count": new_counts.get("weak", 0),
                "new_collision_risk_count": new_counts.get("collision_risk", 0),
                "new_reject_non_sql_count": new_counts.get("reject_non_sql", 0),
                "strict_to_strict_count": strict_to_strict,
                "strict_to_weak_count": strict_to_weak,
                "strict_to_collision_count": strict_to_collision,
                "strict_to_reject_count": strict_to_reject,
                "strict_loss_count": strict_loss_count,
                "strict_loss_ratio": round(strict_loss_ratio, 6),
                "label_change_count": len(changed_rows),
                "missing_reason_code_count": missing_reason_code_count,
                "missing_reason_text_count": missing_reason_text_count,
                "mass_relabeling_trigger": "yes" if mass_relabeling_trigger else "no",
                "explained_mass_relabeling": "yes" if explained_mass_relabeling else "no",
                "strict_loss_top_reason_codes": reason_summary(strict_loss_reason_counter),
                "dominant_changed_reason_codes": reason_summary(changed_reason_counter),
                "dominant_source_roots": source_root_summary(source_root_counter),
                "focus_review_required": "yes" if focus_required else "no",
                "audit_status": audit_status,
                "audit_note": note,
                "_strict_loss_reason_counter": strict_loss_reason_counter,
                "_changed_reason_counter": changed_reason_counter,
                "_source_root_counter": source_root_counter,
                "_dataset_rows": dataset_rows,
            }
        )

    dataset_summaries.sort(key=strict_loss_ratio_key)
    return dataset_summaries


def select_sample_rows(
    dataset_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_dataset = {summary["own_id"]: summary for summary in dataset_summaries}
    selected_ids: set[str] = set()
    samples: list[dict[str, Any]] = []

    def take_rows(own_id: str, limit: int) -> None:
        summary = by_dataset.get(own_id)
        if not summary:
            return
        rows = sorted(summary["_dataset_rows"], key=row_sample_priority)
        for row in rows:
            sql_item_id = (row.get("sql_item_id") or "").strip()
            if sql_item_id in selected_ids:
                continue
            samples.append(
                {
                    "own_id": own_id,
                    "dataset_name": summary["dataset_name"],
                    "sql_item_id": sql_item_id,
                    "source_url": (row.get("source_url") or "").strip(),
                    "old_label": (row.get("dataset_specificity_label") or "").strip(),
                    "new_label": (row.get("v2_specificity_label") or "").strip(),
                    "reason_code": (row.get("v2_specificity_reason_code") or "").strip(),
                    "reason_text": (row.get("v2_specificity_reason_text") or "").strip(),
                    "sql_snippet": sql_snippet(row.get("sql_text_raw") or ""),
                }
            )
            selected_ids.add(sql_item_id)
            if sum(1 for sample in samples if sample["own_id"] == own_id) >= limit:
                break

    for own_id in FOCUS_DATASETS:
        take_rows(own_id, FOCUS_SAMPLE_LIMIT)

    top_loss_ids = [
        summary["own_id"]
        for summary in dataset_summaries
        if summary["old_strict_count"] > 0
    ][:TOP_STRICT_LOSS_LIMIT]
    for own_id in top_loss_ids:
        if own_id in FOCUS_DATASETS:
            continue
        take_rows(own_id, OTHER_TOP_SAMPLE_LIMIT)

    if len(samples) < MIN_SAMPLE_ROWS:
        remaining_rows: list[dict[str, str]] = []
        for summary in dataset_summaries:
            remaining_rows.extend(summary["_dataset_rows"])
        for row in sorted(remaining_rows, key=row_sample_priority):
            sql_item_id = (row.get("sql_item_id") or "").strip()
            if sql_item_id in selected_ids:
                continue
            own_id = (row.get("own_id") or "").strip()
            samples.append(
                {
                    "own_id": own_id,
                    "dataset_name": (row.get("dataset_name") or "").strip(),
                    "sql_item_id": sql_item_id,
                    "source_url": (row.get("source_url") or "").strip(),
                    "old_label": (row.get("dataset_specificity_label") or "").strip(),
                    "new_label": (row.get("v2_specificity_label") or "").strip(),
                    "reason_code": (row.get("v2_specificity_reason_code") or "").strip(),
                    "reason_text": (row.get("v2_specificity_reason_text") or "").strip(),
                    "sql_snippet": sql_snippet(row.get("sql_text_raw") or ""),
                }
            )
            selected_ids.add(sql_item_id)
            if len(samples) >= MIN_SAMPLE_ROWS:
                break

    for index, row in enumerate(samples, start=1):
        row["sample_rank"] = index
    return samples


def explicit_review_lines(dataset_summaries: list[dict[str, Any]]) -> list[str]:
    by_dataset = {summary["own_id"]: summary for summary in dataset_summaries}
    lines: list[str] = []
    for own_id in FOCUS_DATASETS:
        summary = by_dataset.get(own_id)
        if not summary:
            lines.append(f"- {own_id}: No rows found in the reclassified inventory.")
            continue
        lines.append(
            f"- {own_id} - {summary['dataset_name']}: "
            f"strict transitions = strict->strict {summary['strict_to_strict_count']}, "
            f"strict->weak {summary['strict_to_weak_count']}, "
            f"strict->collision {summary['strict_to_collision_count']}, "
            f"strict->reject {summary['strict_to_reject_count']}. "
            f"Audit status={summary['audit_status']}. "
            f"Reasons={summary['strict_loss_top_reason_codes'] or summary['dominant_changed_reason_codes'] or 'none'}. "
            f"Source roots={summary['dominant_source_roots'] or 'none'}. "
            f"Note={summary['audit_note']}"
        )
    return lines


def build_markdown(
    *,
    input_path: Path,
    dataset_summaries: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    overall_status: str,
    missing_reason_code_rows: int,
    missing_reason_text_rows: int,
    unexplained_mass_datasets: list[str],
) -> str:
    strict_loss_ranked = [
        summary for summary in dataset_summaries if summary["old_strict_count"] > 0
    ][:TOP_STRICT_LOSS_LIMIT]

    lines = [
        "# Checkpoint 1 Reclassification Audit",
        "",
        f"- Generated at UTC: `{utc_now_iso()}`",
        f"- Input inventory: `{input_path.resolve()}`",
        f"- Overall status: `{overall_status}`",
        "- Pass criteria: no dataset has unexplained mass relabeling, and reasons are populated and non-empty.",
        "",
        "## Pass-Criteria Check",
        "",
        f"- Rows with missing `v2_specificity_reason_code`: {missing_reason_code_rows}",
        f"- Rows with missing `v2_specificity_reason_text`: {missing_reason_text_rows}",
        f"- Datasets with unexplained mass relabeling: {len(unexplained_mass_datasets)}",
        f"- Unexplained mass relabeling dataset ids: {', '.join(unexplained_mass_datasets) if unexplained_mass_datasets else 'none'}",
        "",
        "## Top 10 Datasets by Strict Loss Ratio",
        "",
        "| Rank | own_id | dataset_name | old_strict | strict->strict | strict->weak | strict->collision | strict->reject | strict_loss_ratio | mass_relabeling_trigger | explained_mass_relabeling | top_reason_codes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for rank, summary in enumerate(strict_loss_ranked, start=1):
        lines.append(
            f"| {rank} | {summary['own_id']} | {summary['dataset_name']} | {summary['old_strict_count']} | "
            f"{summary['strict_to_strict_count']} | {summary['strict_to_weak_count']} | "
            f"{summary['strict_to_collision_count']} | {summary['strict_to_reject_count']} | "
            f"{summary['strict_loss_ratio']:.3f} | {summary['mass_relabeling_trigger']} | "
            f"{summary['explained_mass_relabeling']} | {summary['strict_loss_top_reason_codes'] or 'none'} |"
        )

    lines.extend(
        [
            "",
            "## Explicit Reviews",
            "",
            *explicit_review_lines(dataset_summaries),
            "",
            "## Sampled Rows",
            "",
            f"- Sample count: {len(sample_rows)}",
            "",
            "| Rank | own_id | sql_item_id | old_label | new_label | reason_code | source_url | sql_snippet |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in sample_rows:
        lines.append(
            f"| {sample['sample_rank']} | {sample['own_id']} | {sample['sql_item_id']} | "
            f"{sample['old_label']} | {sample['new_label']} | {sample['reason_code']} | "
            f"{sample['source_url']} | {sample['sql_snippet']} |"
        )
    return "\n".join(lines)


def build_status_payload(
    *,
    args: argparse.Namespace,
    dataset_summaries: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    output_paths: list[Path],
    overall_status: str,
    missing_reason_code_rows: int,
    missing_reason_text_rows: int,
    unexplained_mass_datasets: list[str],
) -> dict[str, Any]:
    strict_loss_ranked = [
        summary for summary in dataset_summaries if summary["old_strict_count"] > 0
    ][:TOP_STRICT_LOSS_LIMIT]
    return {
        "checkpoint": "1",
        "phase_name": "v2_phase1_reclassification_human_audit",
        "generated_at_utc": utc_now_iso(),
        "status": overall_status,
        "pass_criteria": {
            "no_dataset_has_unexplained_mass_relabeling": len(unexplained_mass_datasets) == 0,
            "reason_codes_populated_and_nonempty": missing_reason_code_rows == 0,
            "reason_text_populated_and_nonempty": missing_reason_text_rows == 0,
        },
        "summary": {
            "dataset_count_in_inventory": len(dataset_summaries),
            "sample_row_count": len(sample_rows),
            "missing_reason_code_rows": missing_reason_code_rows,
            "missing_reason_text_rows": missing_reason_text_rows,
            "unexplained_mass_relabeling_dataset_count": len(unexplained_mass_datasets),
        },
        "unexplained_mass_relabeling_datasets": unexplained_mass_datasets,
        "top_strict_loss_ratio_datasets": [
            {
                "own_id": summary["own_id"],
                "dataset_name": summary["dataset_name"],
                "strict_loss_ratio": summary["strict_loss_ratio"],
                "strict_to_strict_count": summary["strict_to_strict_count"],
                "strict_to_weak_count": summary["strict_to_weak_count"],
                "strict_to_collision_count": summary["strict_to_collision_count"],
                "strict_to_reject_count": summary["strict_to_reject_count"],
                "explained_mass_relabeling": summary["explained_mass_relabeling"],
            }
            for summary in strict_loss_ranked
        ],
        "explicit_review_datasets": FOCUS_DATASETS,
        "input": {
            "reclassified_inventory_path": str(args.input.resolve()),
            "reclassified_inventory_sha256": sha256_file(args.input),
        },
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / "audits"
    audit_csv_path = output_dir / "checkpoint1_reclass_audit.csv"
    audit_md_path = output_dir / "checkpoint1_reclass_audit.md"
    status_path = output_dir / "checkpoint1_status.json"

    rows = read_csv_rows(args.input)
    dataset_summaries = build_dataset_summary(rows)
    sample_rows = select_sample_rows(dataset_summaries)

    missing_reason_code_rows = sum(
        1 for row in rows if not (row.get("v2_specificity_reason_code") or "").strip()
    )
    missing_reason_text_rows = sum(
        1 for row in rows if not (row.get("v2_specificity_reason_text") or "").strip()
    )
    unexplained_mass_datasets = [
        summary["own_id"]
        for summary in dataset_summaries
        if summary["mass_relabeling_trigger"] == "yes"
        and summary["explained_mass_relabeling"] != "yes"
    ]
    overall_status = (
        "PASS"
        if missing_reason_code_rows == 0
        and missing_reason_text_rows == 0
        and not unexplained_mass_datasets
        else "FAIL"
    )

    write_csv(
        audit_csv_path,
        CSV_FIELDNAMES,
        [{field: summary.get(field, "") for field in CSV_FIELDNAMES} for summary in dataset_summaries],
    )
    write_text(
        audit_md_path,
        build_markdown(
            input_path=args.input,
            dataset_summaries=dataset_summaries,
            sample_rows=sample_rows,
            overall_status=overall_status,
            missing_reason_code_rows=missing_reason_code_rows,
            missing_reason_text_rows=missing_reason_text_rows,
            unexplained_mass_datasets=unexplained_mass_datasets,
        ),
    )
    status_payload = build_status_payload(
        args=args,
        dataset_summaries=dataset_summaries,
        sample_rows=sample_rows,
        output_paths=[audit_csv_path, audit_md_path],
        overall_status=overall_status,
        missing_reason_code_rows=missing_reason_code_rows,
        missing_reason_text_rows=missing_reason_text_rows,
        unexplained_mass_datasets=unexplained_mass_datasets,
    )
    write_json(status_path, status_payload)
    status_payload["outputs"] = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in [audit_csv_path, audit_md_path, status_path]
    ]
    write_json(status_path, status_payload)

    print(str(audit_csv_path.resolve()))
    print(str(audit_md_path.resolve()))
    print(str(status_path.resolve()))
    print(overall_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
