"""Deficit-driven follow-up planning for the v2 workload line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqb_query.benchmark.canonical_sql import stable_hash

from .contract_spec import CORE_AGENT_SUBITEMS, DETERMINISTIC_SUBITEMS
from .registry import load_registry_rows


def _inventory_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _accepted_counts(registry_rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in registry_rows:
        if not bool(row.get("accepted_for_eval")):
            continue
        key = (str(row.get("dataset_id") or ""), str(row.get("canonical_subitem_id") or ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_deficit_round_plan(
    *,
    inventory_paths: list[Path],
    registry_path: Path,
    round_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_rows = load_registry_rows(registry_path)
    accepted_counts = _accepted_counts(registry_rows)
    accepted_query_ids = {
        str(row.get("query_record_id") or "")
        for row in registry_rows
        if bool(row.get("accepted_for_eval"))
    }

    summary: dict[str, Any] = {
        "round_id": round_id,
        "registry_path": str(registry_path.resolve()),
        "inventories": {},
        "deficits": [],
    }

    for inventory_path in inventory_paths:
        payload = _inventory_payload(inventory_path)
        dataset_id = str(payload["dataset_id"])
        items = [dict(item) for item in (payload.get("items") or [])]
        dataset_deficit_items: list[dict[str, Any]] = []

        for subitem_id in CORE_AGENT_SUBITEMS:
            current = accepted_counts.get((dataset_id, subitem_id), 0)
            deficit = max(0, 5 - current)
            if deficit <= 0:
                continue
            candidates = [
                item
                for item in items
                if item.get("canonical_subitem_id") == subitem_id and item.get("realization_mode") == "agent"
            ]
            for index in range(deficit):
                source = candidates[index % len(candidates)] if candidates else None
                if source is None:
                    break
                cloned = dict(source)
                seed = json.dumps(
                    {
                        "source_query_record_id": cloned.get("query_record_id"),
                        "round_id": round_id,
                        "clone_index": index,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                digest = stable_hash(seed)[:12]
                cloned["query_record_id"] = f"{cloned['query_record_id']}_def_{digest}"
                cloned["problem_id"] = f"{cloned['problem_id']}_def_{digest}"
                cloned["notes"] = list(cloned.get("notes") or []) + [f"deficit_round={round_id}"]
                dataset_deficit_items.append(cloned)
                summary["deficits"].append(
                    {
                        "dataset_id": dataset_id,
                        "canonical_subitem_id": subitem_id,
                        "needed": deficit,
                        "planned_query_record_id": cloned["query_record_id"],
                    }
                )

        for subitem_id in DETERMINISTIC_SUBITEMS:
            planned_items = [
                item
                for item in items
                if item.get("canonical_subitem_id") == subitem_id and item.get("realization_mode") == "deterministic"
            ]
            missing_items = [
                item
                for item in planned_items
                if str(item.get("query_record_id") or "") not in accepted_query_ids
            ]
            dataset_deficit_items.extend(missing_items)
            for item in missing_items:
                summary["deficits"].append(
                    {
                        "dataset_id": dataset_id,
                        "canonical_subitem_id": subitem_id,
                        "needed": len(missing_items),
                        "planned_query_record_id": item["query_record_id"],
                    }
                )

        dataset_payload = {
            "dataset_id": dataset_id,
            "round_id": round_id,
            "source_inventory_path": str(inventory_path.resolve()),
            "problem_count": len(dataset_deficit_items),
            "items": dataset_deficit_items,
        }
        dataset_output = output_dir / f"{dataset_id}_deficit_inventory_v2.json"
        dataset_output.write_text(json.dumps(dataset_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["inventories"][dataset_id] = {
            "path": str(dataset_output.resolve()),
            "problem_count": len(dataset_deficit_items),
        }

    (output_dir / "deficit_round_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
