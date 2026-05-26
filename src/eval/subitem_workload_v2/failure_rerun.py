from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .paths import rerun_dir


def load_failed_question_records(run_roots: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run_root in run_roots:
        if not run_root.exists():
            continue
        for manifest_path in run_root.rglob("run_manifest.json"):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("status") != "failed":
                continue
            question_record = payload.get("question_record") or {}
            query_record_id = str(question_record.get("query_record_id") or "")
            if not query_record_id or query_record_id in seen:
                continue
            error_text = str(payload.get("error") or "")
            record = dict(question_record)
            record["rerun_source_run_id"] = payload.get("run_id", "")
            record["rerun_source_manifest_path"] = str(manifest_path.resolve())
            record["rerun_source_error"] = error_text
            records.append(record)
            seen.add(query_record_id)
    return records


def write_failed_rerun_inventories(*, run_roots: list[Path], rerun_tag: str, line_version: str = "v2") -> list[Path]:
    records = load_failed_question_records(run_roots)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        dataset_id = str(record.get("dataset_id") or "")
        if dataset_id:
            by_dataset[dataset_id].append(record)

    output_dir = rerun_dir(line_version) / rerun_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_paths: list[Path] = []
    for dataset_id, items in sorted(by_dataset.items()):
        payload = {
            "dataset_id": dataset_id,
            "inventory_version": f"subitem_workload_{line_version}_failed_rerun",
            "line_version": line_version,
            "planner_kind": "failed_manifest_extract",
            "selected_template_count": 0,
            "selected_agent_template_count": 0,
            "selected_deterministic_template_count": 0,
            "problem_count": len(items),
            "agent_problem_count": len(items),
            "deterministic_problem_count": 0,
            "coverage_policy": {
                "rerun_mode": "failed_only",
                "source_run_roots": [str(path.resolve()) for path in run_roots],
            },
            "items": items,
        }
        inventory_path = output_dir / f"{dataset_id}_failed_rerun_inventory_{line_version}.json"
        inventory_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        inventory_paths.append(inventory_path)
    summary_path = output_dir / "failed_rerun_summary.json"
    summary_payload = {
        "rerun_tag": rerun_tag,
        "dataset_count": len(inventory_paths),
        "failed_query_count": len(records),
        "inventory_paths": [str(path.resolve()) for path in inventory_paths],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return inventory_paths
