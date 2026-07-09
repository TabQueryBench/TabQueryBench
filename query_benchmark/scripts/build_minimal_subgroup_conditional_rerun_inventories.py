#!/usr/bin/env python3
"""Build minimal rerun inventories for subgroup/conditional coverage repairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.subitem_workload_v2.paths import dataset_inventory_path, normalize_line_version


CONDITIONAL_FAMILY = "conditional_dependency_structure"
SUBGROUP_FAMILY = "subgroup_structure"

SCOPE_TO_SUBITEMS: dict[tuple[str, str], set[str] | None] = {
    ("conditional", "all conditional subitems"): None,
    ("conditional", "slice_level_consistency only"): {"slice_level_consistency"},
    ("subgroup", "all subgroup subitems"): None,
    ("subgroup", "subgroup_size_stability only"): {"subgroup_size_stability"},
}

FAMILY_TO_FAMILY_ID = {
    "conditional": CONDITIONAL_FAMILY,
    "subgroup": SUBGROUP_FAMILY,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minimal subgroup/conditional rerun inventories.")
    parser.add_argument(
        "--plan-csv",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "minimal_subgroup_conditional_rerun_plan_20260526.csv",
        help="CSV plan file with dataset/family/rerun_scope rows.",
    )
    parser.add_argument(
        "--line-version",
        type=str,
        choices=["v2", "v3", "v4"],
        default="v2",
        help="Workload line version.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "minimal_subgroup_conditional_rerun_20260526",
        help="Directory for filtered rerun inventories.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_plan_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _filter_items(
    items: list[dict[str, Any]],
    *,
    family: str,
    rerun_scope: str,
    target_query_count: int,
) -> list[dict[str, Any]]:
    family_id = FAMILY_TO_FAMILY_ID[family]
    target_subitems = SCOPE_TO_SUBITEMS.get((family, rerun_scope))
    chosen: list[dict[str, Any]] = []
    for item in items:
        if str(item.get("family_id") or "") != family_id:
            continue
        subitem = str(item.get("canonical_subitem_id") or "")
        if target_subitems is not None and subitem not in target_subitems:
            continue
        chosen.append(dict(item))
    chosen.sort(key=lambda item: str(item.get("query_record_id") or ""))
    if target_query_count > 0:
        return chosen[:target_query_count]
    return chosen


def main() -> None:
    args = parse_args()
    line_version = normalize_line_version(args.line_version)
    plan_rows = _load_plan_rows(args.plan_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in plan_rows:
        by_dataset[str(row["dataset"]).strip()].append(row)

    summary_rows: list[dict[str, Any]] = []
    inventory_paths: list[str] = []

    for dataset_id in sorted(by_dataset):
        source_path = dataset_inventory_path(dataset_id, line_version=line_version)
        source_payload = _load_json(source_path)
        selected_items: list[dict[str, Any]] = []
        selection_notes: list[dict[str, Any]] = []

        for row in by_dataset[dataset_id]:
            family = str(row["family"]).strip()
            rerun_scope = str(row["rerun_scope"]).strip()
            target_query_count = int(row["target_query_count"])
            chosen = _filter_items(
                source_payload.get("items") or [],
                family=family,
                rerun_scope=rerun_scope,
                target_query_count=target_query_count,
            )
            selected_items.extend(chosen)
            selection_notes.append(
                {
                    "family": family,
                    "rerun_scope": rerun_scope,
                    "target_query_count": target_query_count,
                    "selected_query_count": len(chosen),
                    "reason": row["reason"],
                }
            )

        selected_items.sort(key=lambda item: str(item.get("query_record_id") or ""))
        payload = {
            "dataset_id": dataset_id,
            "inventory_version": f"minimal_subgroup_conditional_rerun_{line_version}",
            "line_version": line_version,
            "planner_kind": "manual_repair_plan_filter",
            "source_inventory_path": str(source_path.resolve()),
            "problem_count": len(selected_items),
            "selected_scope_count": len(selection_notes),
            "selection_notes": selection_notes,
            "items": selected_items,
        }
        out_path = args.output_dir / f"{dataset_id}_minimal_rerun_inventory_{line_version}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        inventory_paths.append(str(out_path.resolve()))
        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "inventory_path": str(out_path.resolve()),
                "problem_count": len(selected_items),
                "scope_count": len(selection_notes),
                "selected_details": selection_notes,
            }
        )

    summary = {
        "line_version": line_version,
        "plan_csv": str(args.plan_csv.resolve()),
        "dataset_count": len(summary_rows),
        "inventory_paths": inventory_paths,
        "datasets": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "dataset_count": len(summary_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
