"""Coverage gate and reporting helpers for the v2 workload line."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .contract_spec import CORE_AGENT_SUBITEMS, DETERMINISTIC_SUBITEMS
from .paths import V2_EVALUATION_FINAL_DIR, ensure_v2_dirs
from .registry import load_registry_rows


_TOKEN_RE = re.compile(r"([A-Za-z]+)(\d+)$")


def natural_sort_key(value: str) -> tuple[Any, ...]:
    match = _TOKEN_RE.match(value)
    if match:
        return (match.group(1), int(match.group(2)))
    return (value, 0)


def _load_inventory_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize_coverage(
    *,
    inventory_paths: list[Path],
    registry_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_v2_dirs()
    output_root = output_dir or V2_EVALUATION_FINAL_DIR
    inventories = [_load_inventory_payload(path) for path in inventory_paths]
    dataset_ids = sorted([payload["dataset_id"] for payload in inventories], key=natural_sort_key)
    registry_rows = load_registry_rows(registry_path)

    accepted_rows = [row for row in registry_rows if bool(row.get("accepted_for_eval"))]
    by_dataset_subitem: dict[tuple[str, str], int] = {}
    for row in accepted_rows:
        key = (str(row.get("dataset_id")), str(row.get("canonical_subitem_id")))
        by_dataset_subitem[key] = by_dataset_subitem.get(key, 0) + 1

    dataset_rows: list[dict[str, Any]] = []
    deficit_rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        for subitem_id in CORE_AGENT_SUBITEMS:
            count = by_dataset_subitem.get((dataset_id, subitem_id), 0)
            passed = count >= 5
            dataset_rows.append(
                {
                    "dataset_id": dataset_id,
                    "canonical_subitem_id": subitem_id,
                    "accepted_sql_count": count,
                    "coverage_target_min": 5,
                    "passed": "yes" if passed else "no",
                    "realization_mode": "agent",
                }
            )
            if not passed:
                deficit_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "canonical_subitem_id": subitem_id,
                        "accepted_sql_count": count,
                        "coverage_target_min": 5,
                        "deficit": 5 - count,
                    }
                )

        planned_counts: dict[str, int] = {}
        payload = next(item for item in inventories if item["dataset_id"] == dataset_id)
        for item in payload.get("items") or []:
            subitem_id = str(item.get("canonical_subitem_id") or "")
            if subitem_id in DETERMINISTIC_SUBITEMS:
                planned_counts[subitem_id] = planned_counts.get(subitem_id, 0) + 1
        for subitem_id in DETERMINISTIC_SUBITEMS:
            planned = planned_counts.get(subitem_id, 0)
            count = by_dataset_subitem.get((dataset_id, subitem_id), 0)
            passed = count == planned
            dataset_rows.append(
                {
                    "dataset_id": dataset_id,
                    "canonical_subitem_id": subitem_id,
                    "accepted_sql_count": count,
                    "coverage_target_min": "enumerate_all_applicable",
                    "planned_sql_count": planned,
                    "passed": "yes" if passed else "no",
                    "realization_mode": "deterministic",
                }
            )
            if planned and not passed:
                deficit_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "canonical_subitem_id": subitem_id,
                        "accepted_sql_count": count,
                        "coverage_target_min": planned,
                        "deficit": planned - count,
                    }
                )

    source_counts: dict[str, int] = {}
    for row in registry_rows:
        key = str(row.get("subitem_inference_source") or "explicit")
        source_counts[key] = source_counts.get(key, 0) + 1

    _write_csv(output_root / "dataset_subitem_sql_counts.csv", dataset_rows)
    _write_csv(output_root / "coverage_deficits.csv", deficit_rows)
    summary = {
        "registry_path": str(registry_path.resolve()),
        "dataset_count": len(dataset_ids),
        "accepted_query_count": len(accepted_rows),
        "registered_query_count": len(registry_rows),
        "dataset_subitem_count_rows": len(dataset_rows),
        "deficit_count": len(deficit_rows),
        "subitem_inference_source_counts": source_counts,
    }
    (output_root / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
