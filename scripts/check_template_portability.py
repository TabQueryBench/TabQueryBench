#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bundle import load_dataset_bundle

MISSING_TOKENS = {"", "null", "NULL", "None", "none", "nan", "NaN", "N/A", "n/a", "<null>"}


@dataclass
class ColumnProfile:
    name: str
    missing_count: int
    unique_count: int
    unique_ratio: float
    numeric_ratio: float
    avg_length: float
    declared_type: str
    semantic_type: str
    use_for_groupby: bool
    use_for_predicate: bool
    role: str
    ordered: bool


@dataclass
class DatasetProfile:
    dataset_id: str
    row_count: int
    target_column: str | None
    task_type: str | None
    columns: dict[str, ColumnProfile]


ROLE_ORDER = [
    "group_col",
    "group_col_2",
    "time_col",
    "measure_col",
    "target_col",
    "predicate_col",
    "condition_col",
    "entity_col",
    "item_col",
    "key_col",
    "key_col_2",
    "missing_col",
    "text_col",
    "band_col",
    "condition_col_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static portability check for workload-grounded templates.")
    parser.add_argument(
        "--template-library",
        default="data/workload_grounding/template_library_v1.jsonl",
        help="Path to template library JSONL.",
    )
    parser.add_argument(
        "--output",
        default="data/workload_grounding/template_portability_report_v1.csv",
        help="Output CSV path for portability report.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing dataset folders.",
    )
    parser.add_argument(
        "--dataset-ids",
        default="c2,m4,n1",
        help="Comma-separated dataset ids to check.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. If provided and a manifest exists, the script updates it.",
    )
    parser.add_argument(
        "--logs-root",
        default="logs/workload_grounding",
        help="Root directory for run manifests.",
    )
    parser.add_argument(
        "--manifest-output-key",
        default="template_portability_report",
        help="Manifest output key to update for this portability run.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset_profile(data_root: Path, dataset_id: str) -> DatasetProfile:
    bundle = load_dataset_bundle(dataset_id=dataset_id, data_root=data_root, strict=True)
    field_registry = bundle.field_registry
    semantics = bundle.dataset_semantics
    contract_columns = {
        str(column.get("name") or "").strip(): column
        for column in (bundle.dataset_contract.get("columns") or [])
        if str(column.get("name") or "").strip()
    }
    row_count = int(
        (bundle.dataset_contract.get("row_counts") or {}).get("main")
        or (bundle.dataset_profile.get("summary") or {}).get("n_rows")
        or 0
    )

    metadata_fields = {
        str(field.get("name") or "").strip(): field
        for field in (field_registry.get("fields") or [])
        if str(field.get("name") or "").strip()
    }
    columns: dict[str, ColumnProfile] = {}
    for name, meta in metadata_fields.items():
        contract_column = contract_columns.get(name, {})
        stats = contract_column.get("profile_stats") or {}
        missing_rate = float(stats.get("missing_rate") or 0.0)
        missing_count = int(round(missing_rate * row_count))
        unique_count = int(stats.get("unique_count") or 0)
        unique_ratio = float(stats.get("unique_ratio") or (unique_count / max(1, row_count)))
        example_values = [str(value) for value in (stats.get("example_values") or []) if value is not None]
        lengths = [len(value) for value in example_values]
        declared_type = str(meta.get("declared_type", "unknown"))
        semantic_type = str(meta.get("semantic_type", "unknown"))
        numeric_ratio = (
            1.0
            if "numeric" in f"{declared_type} {semantic_type}".lower() or declared_type == "boolean"
            else 0.0
        )
        meta = metadata_fields.get(name, {})
        columns[name] = ColumnProfile(
            name=name,
            missing_count=missing_count,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            numeric_ratio=numeric_ratio,
            avg_length=mean(lengths) if lengths else 0.0,
            declared_type=str(meta.get("declared_type", "unknown")),
            semantic_type=str(meta.get("semantic_type", "unknown")),
            use_for_groupby=bool(meta.get("use_for_groupby", False)),
            use_for_predicate=bool(meta.get("use_for_predicate", True)),
            role=str(meta.get("role", "unknown")),
            ordered=bool(meta.get("ordered", False)),
        )
    return DatasetProfile(
        dataset_id=dataset_id,
        row_count=row_count,
        target_column=semantics.get("target_column"),
        task_type=semantics.get("task_type"),
        columns=columns,
    )


def is_numeric(profile: ColumnProfile) -> bool:
    return profile.numeric_ratio >= 0.95 or profile.semantic_type.startswith("numeric") or profile.declared_type == "numeric"


def is_groupable(profile: ColumnProfile) -> bool:
    if profile.role == "target" and profile.unique_count <= 10:
        return True
    if profile.use_for_groupby:
        return True
    return is_numeric(profile) and profile.unique_count <= 20


def is_binary_or_low_cardinality(profile: ColumnProfile) -> bool:
    return profile.unique_count <= 5


def is_high_cardinality(profile: ColumnProfile) -> bool:
    return profile.unique_count >= 20 or profile.unique_ratio >= 0.2


def is_text_like(profile: ColumnProfile) -> bool:
    if is_numeric(profile):
        return False
    return profile.avg_length >= 4


def ordered_or_numeric(profile: ColumnProfile) -> bool:
    return profile.ordered or is_numeric(profile)


def is_temporal(profile: ColumnProfile) -> bool:
    semantic = (profile.semantic_type or "").lower()
    declared = (profile.declared_type or "").lower()
    name = (profile.name or "").lower()
    return (
        semantic in {"datetime", "date", "timestamp", "temporal"}
        or declared in {"datetime", "date", "timestamp"}
        or "time" in name
        or "date" in name
    )


def choose_candidate(role: str, dataset: DatasetProfile, used: set[str], template: dict[str, Any]) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    columns = dataset.columns
    all_profiles = list(columns.values())
    target_profile = columns.get(dataset.target_column) if dataset.target_column else None
    groupable = [p for p in all_profiles if is_groupable(p) and p.name not in used]
    temporal_cols = [p for p in all_profiles if is_temporal(p) and p.name not in used]
    numeric_cols = [p for p in all_profiles if is_numeric(p) and p.name not in used]
    low_card = [p for p in all_profiles if is_binary_or_low_cardinality(p) and p.name not in used]
    high_card = [p for p in all_profiles if is_high_cardinality(p) and p.name not in used]
    text_like = [p for p in all_profiles if is_text_like(p) and p.name not in used]
    missing_native = [p for p in all_profiles if p.missing_count > 0 and p.name not in used]
    filterable = [p for p in all_profiles if p.use_for_predicate and p.name not in used]

    if role == "target_col":
        if dataset.target_column and dataset.target_column not in used:
            return dataset.target_column, warnings
        warnings.append("target_col:fallback_first_column")
        return next(iter(columns)), warnings
    if role == "group_col":
        if groupable:
            return groupable[0].name, warnings
        if (
            dataset.task_type == "classification"
            and dataset.target_column
            and dataset.target_column not in used
            and target_profile is not None
            and target_profile.unique_count <= 10
        ):
            warnings.append("group_col:used_classification_target_fallback")
            return dataset.target_column, warnings
        return None, warnings
    if role == "group_col_2":
        if groupable:
            return groupable[0].name, warnings
        return None, warnings
    if role == "time_col":
        if temporal_cols:
            return temporal_cols[0].name, warnings
        return None, warnings
    if role == "measure_col":
        if dataset.target_column and dataset.target_column in columns:
            target_profile = columns[dataset.target_column]
            if is_numeric(target_profile) and dataset.target_column not in used:
                return dataset.target_column, warnings
        if numeric_cols:
            return numeric_cols[0].name, warnings
        return None, warnings
    if role == "predicate_col":
        if filterable:
            return filterable[0].name, warnings
        return None, warnings
    if role == "condition_col":
        if low_card:
            return low_card[0].name, warnings
        if filterable:
            warnings.append("condition_col:used_high_cardinality_fallback")
            return filterable[0].name, warnings
        return None, warnings
    if role == "condition_col_2":
        if low_card:
            return low_card[0].name, warnings
        if filterable:
            warnings.append("condition_col_2:used_high_cardinality_fallback")
            return filterable[0].name, warnings
        return None, warnings
    if role == "entity_col":
        if high_card:
            return high_card[0].name, warnings
        if dataset.target_column and dataset.target_column not in used:
            warnings.append("entity_col:no_high_cardinality_used_target_fallback")
            return dataset.target_column, warnings
        return None, warnings
    if role == "item_col":
        if groupable:
            return groupable[0].name, warnings
        if high_card:
            warnings.append("item_col:used_high_cardinality_fallback")
            return high_card[0].name, warnings
        return None, warnings
    if role == "key_col":
        if high_card:
            return high_card[0].name, warnings
        if groupable:
            warnings.append("key_col:no_high_cardinality_used_groupable_fallback")
            return groupable[0].name, warnings
        return None, warnings
    if role == "key_col_2":
        if high_card:
            return high_card[0].name, warnings
        if groupable:
            warnings.append("key_col_2:no_high_cardinality_used_groupable_fallback")
            return groupable[0].name, warnings
        return None, warnings
    if role == "missing_col":
        if template.get("primary_family") == "missing_introduction_validation":
            preferred = [p for p in all_profiles if p.name not in used and p.role == "feature"]
            if preferred:
                if preferred[0].missing_count == 0:
                    warnings.append("missing_col:synthetic_injection_assumption")
                return preferred[0].name, warnings
        if missing_native:
            return missing_native[0].name, warnings
        return None, warnings
    if role == "text_col":
        if text_like:
            return text_like[0].name, warnings
        return None, warnings
    if role == "band_col":
        if numeric_cols:
            return numeric_cols[0].name, warnings
        return None, warnings
    return None, warnings


def evaluate_constraints(template: dict[str, Any], bound: dict[str, str], dataset: DatasetProfile) -> list[str]:
    warnings: list[str] = []
    columns = dataset.columns
    for constraint in template.get("constraints", []):
        if constraint == "group_col:groupable" and not is_groupable(columns[bound["group_col"]]):
            warnings.append("constraint_failed:group_col_not_groupable")
        elif constraint == "time_col:temporal" and not is_temporal(columns[bound["time_col"]]):
            warnings.append("constraint_failed:time_col_not_temporal")
        elif constraint == "group_col_2:groupable_distinct_from_group_col":
            if bound.get("group_col_2") == bound.get("group_col"):
                warnings.append("constraint_failed:group_col_2_not_distinct")
            elif not is_groupable(columns[bound["group_col_2"]]):
                warnings.append("constraint_failed:group_col_2_not_groupable")
        elif constraint == "measure_col:numeric" and not is_numeric(columns[bound["measure_col"]]):
            warnings.append("constraint_failed:measure_col_not_numeric")
        elif constraint == "measure_col:ordered_or_numeric" and not ordered_or_numeric(columns[bound["measure_col"]]):
            warnings.append("constraint_failed:measure_col_not_ordered_or_numeric")
        elif constraint == "predicate_col:ordered_or_numeric_preferred" and not ordered_or_numeric(columns[bound["predicate_col"]]):
            warnings.append("constraint_soft:predicate_col_not_ordered")
        elif constraint == "condition_col:binary_or_low_cardinality_preferred" and not is_binary_or_low_cardinality(columns[bound["condition_col"]]):
            warnings.append("constraint_soft:condition_col_not_low_cardinality")
        elif constraint == "target_col:categorical_or_binary":
            target_profile = columns[bound["target_col"]]
            if is_numeric(target_profile) and not (dataset.task_type == "classification" and target_profile.unique_count <= 10):
                warnings.append("constraint_failed:target_col_not_categorical")
        elif constraint == "entity_col:high_cardinality_preferred" and not is_high_cardinality(columns[bound["entity_col"]]):
            warnings.append("constraint_soft:entity_col_not_high_cardinality")
        elif constraint == "text_col:text_like" and not is_text_like(columns[bound["text_col"]]):
            warnings.append("constraint_failed:text_col_not_text_like")
        elif constraint == "item_col:groupable_or_high_cardinality":
            profile = columns[bound["item_col"]]
            if not (is_groupable(profile) or is_high_cardinality(profile)):
                warnings.append("constraint_failed:item_col_not_groupable_or_high_cardinality")
        elif constraint == "band_col:ordered_or_numeric" and not ordered_or_numeric(columns[bound["band_col"]]):
            warnings.append("constraint_failed:band_col_not_ordered_or_numeric")
        elif constraint == "key_col_2:distinct_from_key_col" and bound.get("key_col") == bound.get("key_col_2"):
            warnings.append("constraint_failed:key_col_2_not_distinct")
        elif constraint == "condition_col_2:distinct_from_condition_col" and bound.get("condition_col") == bound.get("condition_col_2"):
            warnings.append("constraint_failed:condition_col_2_not_distinct")
        elif constraint == "domain_rule_required":
            warnings.append("constraint_soft:domain_rule_not_in_metadata")
    return warnings


def classify_portability(missing_roles: list[str], warnings: list[str], template: dict[str, Any]) -> tuple[str, str, str]:
    if missing_roles:
        return "no", ";".join(missing_roles), "required roles unavailable"
    if template.get("status") == "blocked":
        return "partial", "", "template marked blocked or placeholder"
    if any(warning.startswith("constraint_failed") for warning in warnings):
        return "partial", "", "; ".join(warnings)
    if warnings:
        return "partial", "", "; ".join(warnings)
    return "yes", "", ""


def main() -> None:
    args = parse_args()
    template_library_path = Path(args.template_library)
    output_path = Path(args.output)
    data_root = Path(args.data_root)
    dataset_ids = [item.strip() for item in args.dataset_ids.split(",") if item.strip()]
    templates = load_jsonl(template_library_path)
    datasets = {dataset_id: load_dataset_profile(data_root, dataset_id) for dataset_id in dataset_ids}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "template_id",
        "dataset_id",
        "portable",
        "missing_required_roles",
        "binding_example",
        "failure_reason",
        "review_flag",
    ]
    rows: list[dict[str, str]] = []

    for template in templates:
        for dataset_id, dataset in datasets.items():
            bound: dict[str, str] = {"table": dataset_id}
            missing_roles: list[str] = []
            warnings: list[str] = []
            used: set[str] = set()
            for role in template.get("required_roles", []):
                choice, role_warnings = choose_candidate(role, dataset, used, template)
                warnings.extend(role_warnings)
                if choice is None:
                    missing_roles.append(role)
                else:
                    bound[role] = choice
                    used.add(choice)
            if not missing_roles:
                warnings.extend(evaluate_constraints(template, bound, dataset))
            portable, missing_required_roles, failure_reason = classify_portability(missing_roles, warnings, template)
            review_flag = "yes" if portable != "yes" or template.get("status") != "ready" else "no"
            rows.append(
                {
                    "template_id": template["template_id"],
                    "dataset_id": dataset_id,
                    "portable": portable,
                    "missing_required_roles": missing_required_roles,
                    "binding_example": json.dumps(bound, ensure_ascii=False, sort_keys=True),
                    "failure_reason": failure_reason,
                    "review_flag": review_flag,
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.run_id:
        manifest_path = Path(args.logs_root) / args.run_id / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"run_id": args.run_id}
        summary: dict[str, dict[str, int]] = {}
        for row in rows:
            summary.setdefault(row["dataset_id"], {"yes": 0, "partial": 0, "no": 0})
            summary[row["dataset_id"]][row["portable"]] += 1
        manifest.setdefault("outputs", {})[args.manifest_output_key] = {
            "path": str(output_path.resolve()),
            "row_count": len(rows),
            "dataset_summary": summary,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_path": str(output_path.resolve()),
        "row_count": len(rows),
        "dataset_ids": dataset_ids,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
