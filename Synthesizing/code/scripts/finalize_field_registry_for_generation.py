#!/usr/bin/env python3
"""Finalize AI-reviewed field registries for generation and postprocessing.

This script makes dataset-level target semantics explicit without forcing every
dataset to have an official single target.  Models that require an X/y layout
use ``adapter_primary_target``; dataset semantics remain faithful to the source.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLICIES = {
    "round_integer",
    "restore_domain",
    "strip_whitespace",
    "warn_range",
    "clip_range",
    "reject_range",
    "warn_null_rate",
    "reject_all_null",
    "preserve_missing_sentinel",
}


CURATED_TARGET_POLICY: dict[str, dict[str, Any]] = {
    "c13": {"kind": "official_single_target", "official_target_column": "iClass", "adapter_primary_target": "iClass", "task_type": "classification", "source_note": "Census/KDD-style supervised target retained as iClass."},
    "c16": {"kind": "no_official_target", "official_target_column": None, "adapter_primary_target": "EYE", "task_type": "classification", "source_note": "Comic character table has no canonical supervised target; EYE is adapter-only."},
    "c17": {"kind": "no_official_target", "official_target_column": None, "adapter_primary_target": "type", "task_type": "classification", "source_note": "Netflix catalog table has no canonical target; type is adapter-only."},
    "c18": {"kind": "no_official_target", "official_target_column": None, "adapter_primary_target": "points", "task_type": "regression", "source_note": "Wine review table is EDA-style; points is adapter-only numeric target."},
    "c19": {"kind": "no_official_target", "official_target_column": None, "adapter_primary_target": "category_id", "task_type": "classification", "source_note": "YouTube trending table has no canonical target; category_id is adapter-only."},
    "c3": {"kind": "official_single_target", "official_target_column": "EI", "adapter_primary_target": "EI", "task_type": "classification", "source_note": "Splice-junction class label."},
    "c6": {"kind": "official_single_target", "official_target_column": "Type of Answer", "adapter_primary_target": "Type of Answer", "task_type": "classification", "source_note": "Student answer type label."},
    "c9": {"kind": "official_single_target", "official_target_column": "ACTION", "adapter_primary_target": "ACTION", "task_type": "classification", "source_note": "Amazon employee access challenge label is ACTION."},
    "m1": {"kind": "official_single_target", "official_target_column": "Response_Quality", "adapter_primary_target": "Response_Quality", "task_type": "classification", "source_note": "Survey response quality label."},
    "m2": {"kind": "official_single_target", "official_target_column": "is_claim", "adapter_primary_target": "is_claim", "task_type": "classification", "source_note": "Vehicle insurance claim label."},
    "m3": {"kind": "official_single_target", "official_target_column": "is_canceled", "adapter_primary_target": "is_canceled", "task_type": "classification", "source_note": "Hotel booking cancellation label."},
    "m4": {"kind": "official_single_target", "official_target_column": "charges", "adapter_primary_target": "charges", "task_type": "regression", "source_note": "Medical insurance cost target."},
    "m6": {"kind": "official_single_target", "official_target_column": "Revenue", "adapter_primary_target": "Revenue", "task_type": "classification", "source_note": "Online shoppers purchasing intention label."},
    "m7": {"kind": "official_single_target", "official_target_column": "stroke", "adapter_primary_target": "stroke", "task_type": "classification", "source_note": "Stroke prediction label."},
    "m8": {"kind": "official_single_target", "official_target_column": "y", "adapter_primary_target": "y", "task_type": "classification", "source_note": "Bank marketing subscription label."},
    "m11": {"kind": "official_single_target", "official_target_column": "Response", "adapter_primary_target": "Response", "task_type": "classification", "source_note": "Health insurance cross-sell label is Response."},
    "n2": {"kind": "official_single_target", "official_target_column": "sound_pressure_level", "adapter_primary_target": "sound_pressure_level", "task_type": "regression", "source_note": "Airfoil self-noise output variable."},
    "n3": {"kind": "official_single_target", "official_target_column": "quality", "adapter_primary_target": "quality", "task_type": "classification", "source_note": "Wine quality score label."},
    "n5": {"kind": "official_single_target", "official_target_column": "critical_temp", "adapter_primary_target": "critical_temp", "task_type": "regression", "source_note": "Superconductivity critical temperature target."},
    "n6": {"kind": "official_single_target", "official_target_column": "y", "adapter_primary_target": "y", "task_type": "classification", "source_note": "Dataset-provided y label."},
    "n7": {"kind": "official_multilabel", "official_target_column": None, "label_columns": ["Family", "Genus", "Species"], "adapter_primary_target": "Family", "task_type": "classification", "source_note": "Anuran calls has hierarchical Family/Genus/Species labels; Family is adapter-only primary target."},
    "n8": {"kind": "official_multilabel", "official_target_column": None, "label_columns": ["label_1", "label_2", "label_3"], "adapter_primary_target": "label_2", "task_type": "classification", "source_note": "Dataset contains three label columns; label_2 preserves historical adapter target."},
    "n11": {"kind": "official_single_target", "official_target_column": "g", "adapter_primary_target": "g", "task_type": "classification", "source_note": "Dataset-provided g label."},
    "n13": {"kind": "official_single_target", "official_target_column": "PE", "adapter_primary_target": "PE", "task_type": "regression", "source_note": "Combined-cycle power plant energy output target."},
    "n15": {"kind": "official_single_target", "official_target_column": "CARAVAN", "adapter_primary_target": "CARAVAN", "task_type": "classification", "source_note": "Caravan insurance purchase label."},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def infer_task_type(field: dict[str, Any] | None) -> str:
    if not field:
        return "classification"
    semantic = str(field.get("semantic_type") or "").lower()
    if semantic in {"continuous"}:
        return "regression"
    if semantic in {"integer"}:
        unique_count = int(field.get("unique_count") or 0)
        return "classification" if 0 < unique_count <= 20 else "regression"
    return "classification"


def ensure_policy(field: dict[str, Any]) -> None:
    semantic = str(field.get("semantic_type") or "").lower()
    policies = {str(item) for item in field.get("postprocess_policy") or [] if str(item) in POLICIES}
    policies.update({"strip_whitespace", "warn_null_rate", "warn_range"})
    if field.get("nullable") or float(field.get("raw_null_rate") or 0.0) > 0:
        policies.add("preserve_missing_sentinel")
    if semantic == "integer" or field.get("integer_like"):
        policies.add("round_integer")
    if semantic in {"boolean", "categorical", "ordinal"}:
        policies.add("restore_domain")
    if semantic == "id":
        policies.add("restore_domain")
        if field.get("integer_like"):
            policies.add("round_integer")
    if semantic in {"boolean"} and "reject_range" not in policies:
        policies.add("restore_domain")
    field["postprocess_policy"] = sorted(policies)


def choose_default_policy(registry: dict[str, Any]) -> dict[str, Any]:
    fields = registry.get("fields") or []
    targets = [
        f
        for f in fields
        if str(f.get("role") or "").lower() == "target" or bool(f.get("use_as_target"))
    ]
    if len(targets) == 1:
        target = str(targets[0]["name"])
        return {
            "kind": "official_single_target",
            "official_target_column": target,
            "adapter_primary_target": target,
            "task_type": registry.get("task_type") if registry.get("task_type") != "unknown" else infer_task_type(targets[0]),
            "source_note": "Inherited from AI-reviewed field registry.",
        }
    # Fail closed at finalization time: no silent last-column target fallback.
    return {
        "kind": "needs_target_review",
        "official_target_column": None,
        "adapter_primary_target": None,
        "task_type": "unknown",
        "source_note": "No curated target policy and no single AI-reviewed target.",
    }


def write_semantics(path: Path, registry: dict[str, Any]) -> None:
    target_policy = registry.get("target_policy") or {}
    lines = [
        f"dataset_id: {registry.get('dataset_id')}",
        f"target_policy_kind: {target_policy.get('kind')}",
        f"official_target_column: {json.dumps(target_policy.get('official_target_column'), ensure_ascii=False)}",
        "label_columns:",
    ]
    for col in target_policy.get("label_columns") or []:
        lines.append(f"  - {json.dumps(col, ensure_ascii=False)}")
    lines.extend(
        [
            f"adapter_primary_target: {json.dumps(target_policy.get('adapter_primary_target'), ensure_ascii=False)}",
            f"task_type: {registry.get('task_type')}",
            f"row_count: {registry.get('expected_rows')}",
            "notes:",
            "  - Finalized for metadata-only generation and automatic postprocessing.",
            f"  - {str(target_policy.get('source_note') or '').replace(chr(10), ' ')}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_one(registry_path: Path) -> dict[str, Any]:
    registry = read_json(registry_path)
    dataset_id = str(registry.get("dataset_id") or registry_path.parents[1].name)
    fields = registry.get("fields") or []
    field_by_name = {str(field.get("name")): field for field in fields if field.get("name")}
    policy = dict(CURATED_TARGET_POLICY.get(dataset_id) or choose_default_policy(registry))

    label_columns = list(policy.get("label_columns") or [])
    official_target = policy.get("official_target_column")
    if official_target:
        label_columns = [str(official_target)]
    adapter_primary = policy.get("adapter_primary_target")

    missing_policy_cols = [
        col
        for col in ([official_target] if official_target else []) + label_columns + ([adapter_primary] if adapter_primary else [])
        if col and col not in field_by_name
    ]
    if missing_policy_cols:
        raise ValueError(f"{dataset_id}: target policy references missing columns: {missing_policy_cols}")

    for field in fields:
        name = str(field.get("name"))
        if official_target and name == official_target:
            field["role"] = "target"
            field["use_as_target"] = True
            field["is_label_column"] = True
        elif name in label_columns:
            field["role"] = "label"
            field["use_as_target"] = False
            field["is_label_column"] = True
        else:
            field["role"] = "feature"
            field["use_as_target"] = False
            field["is_label_column"] = False
        field["adapter_role"] = "target" if adapter_primary and name == adapter_primary else "feature"
        ensure_policy(field)

    if policy.get("task_type") in {None, "", "unknown"} and adapter_primary in field_by_name:
        policy["task_type"] = infer_task_type(field_by_name[adapter_primary])
    policy.setdefault("label_columns", label_columns)
    policy["label_columns"] = label_columns

    registry["schema_version"] = "dataset_understanding_field_registry_v2"
    registry["finalized_at"] = datetime.now(timezone.utc).isoformat()
    registry["finalized_by"] = "code/scripts/finalize_field_registry_for_generation.py"
    registry["target_policy"] = policy
    registry["official_target_column"] = official_target
    registry["label_columns"] = label_columns
    registry["adapter_primary_target"] = adapter_primary
    registry["target_column"] = official_target
    registry["task_type"] = str(policy.get("task_type") or "unknown")
    registry["field_count"] = len(fields)

    write_json(registry_path, registry)
    write_semantics(registry_path.parent / "dataset_semantics.yaml", registry)
    return {
        "dataset_id": dataset_id,
        "kind": policy.get("kind"),
        "official_target_column": official_target,
        "label_columns": label_columns,
        "adapter_primary_target": adapter_primary,
        "task_type": registry["task_type"],
        "field_count": len(fields),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("raw_data/tabular_datasets"))
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--report", type=Path, default=Path("tmp/finalized_field_registry_report.json"))
    args = parser.parse_args()

    root = args.data_root.resolve()
    selected = set(args.dataset or [])
    paths = sorted(root.glob("*/metadata_core/field_registry.json"))
    if selected:
        paths = [path for path in paths if path.parents[1].name in selected]
    rows = [finalize_one(path) for path in paths]
    problems = [
        row
        for row in rows
        if not row.get("adapter_primary_target") or row.get("task_type") in {None, "", "unknown"}
    ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(rows),
        "problem_count": len(problems),
        "problems": problems,
        "datasets": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_count": len(rows), "problem_count": len(problems), "report": str(args.report)}, indent=2))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
