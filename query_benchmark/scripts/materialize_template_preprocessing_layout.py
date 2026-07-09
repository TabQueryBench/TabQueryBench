#!/usr/bin/env python3
"""Materialize canonical template-preprocessing layout and fill missing dataset metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import DATA_DIR
from src.data.layout import dataset_asset_candidates

CORE_METADATA_FILES = (
    "dataset_description",
    "dataset_semantics",
    "field_registry",
    "query_policy",
    "validation_policy",
)
OPTIONAL_METADATA_FILES = (
    "family_applicability",
    "risk_register",
    "uncertainty_register",
)
IGNORED_DATASET_DIRS = {"artifacts", "workload_grounding", "SynData"}
FIVE_FIXED_FAMILIES = [
    "subgroup_structure",
    "conditional_dependency_structure",
    "tail_rarity_structure",
    "missingness_structure",
    "cardinality_structure",
]
WORKLOAD_GROUNDING_LAYOUT = {
    "library/core/template_library_v1.jsonl": "template_library_v1.jsonl",
    "library/core/source_query_bank_v1.jsonl": "source_query_bank_v1.jsonl",
    "library/core/workload_catalog.csv": "workload_catalog.csv",
    "library/core/workload_to_family_mapping_v1.csv": "workload_to_family_mapping_v1.csv",
    "library/extensions/template_library_extensions_v1.jsonl": "template_library_extensions_v1.jsonl",
    "policy/template_policy_v1.jsonl": "template_policy_v1.jsonl",
    "provenance/template_derivation_evidence_v1.csv": "template_derivation_evidence_v1.csv",
    "provenance/template_derivation_evidence_v1.jsonl": "template_derivation_evidence_v1.jsonl",
    "runtime_specs/agent_candidate_spec_all_core_v1.json": "agent_candidate_spec_all_core_v1.json",
    "runtime_specs/agent_candidate_spec_top10_v1.json": "agent_candidate_spec_top10_v1.json",
    "runtime_specs/agent_candidate_spec_top10_plus5_v1.json": "agent_candidate_spec_top10_plus5_v1.json",
    "portability/global/template_portability_report_v1.csv": "template_portability_report_v1.csv",
    "portability/extensions/template_extension_portability_report_v1.csv": "template_extension_portability_report_v1.csv",
    "inventories/full_question_inventory_v1": "full_question_inventory_v1",
    "inventories/full_question_inventory_v2_policy_gpt54": "full_question_inventory_v2_policy_gpt54",
    "reports/preprocessing_shadow_v1": "preprocessing_shadow_v1",
    "reports/policyfull54_comparison_summary_v1.json": "policyfull54_comparison_summary_v1.json",
    "reports/top10_research_summary_v1.json": "top10_research_summary_v1.json",
    "reports/top10_vs_all_core_question_panel_v1.json": "top10_vs_all_core_question_panel_v1.json",
    "reports/top10_vs_all_core_summary_v1.json": "top10_vs_all_core_summary_v1.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize canonical dataset/template preprocessing layout and fill missing metadata bundles.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_DIR,
        help="Root directory containing dataset folders.",
    )
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default="",
        help="Optional comma-separated dataset ids. Defaults to every dataset under data root.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DATA_DIR / "workload_grounding" / "reports" / "preprocessing_layout_v2",
        help="Directory where migration reports are written.",
    )
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _dump_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _list_dataset_ids(data_root: Path) -> list[str]:
    dataset_ids: list[str] = []
    for path in sorted(data_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name in IGNORED_DATASET_DIRS:
            continue
        dataset_ids.append(path.name)
    return dataset_ids


def _is_numeric_token(value: Any) -> bool:
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _is_boolean_token(value: Any) -> bool:
    token = str(value).strip().lower()
    if not token:
        return False
    return token in {"0", "1", "0.0", "1.0", "true", "false", "t", "f", "yes", "no", "y", "n"}


def _values_fit_boolean_domain(values: list[Any]) -> bool:
    cleaned = [value for value in values if value is not None and str(value).strip() != ""]
    return bool(cleaned) and all(_is_boolean_token(value) for value in cleaned)


def _load_semantic_type_overrides(dataset_dir: Path) -> dict[str, str]:
    overrides_path = dataset_dir / "metadata" / "contract_overrides.json"
    if not overrides_path.exists():
        return {}
    try:
        payload = _load_json(overrides_path)
    except Exception:
        return {}
    overrides = payload.get("overrides") or {}
    semantic_type_overrides = overrides.get("semantic_type_overrides") or {}
    return {
        str(name): str(value)
        for name, value in semantic_type_overrides.items()
        if str(name).strip() and str(value).strip()
    }


def _apply_semantic_type_overrides(
    contract: dict[str, Any],
    profile: dict[str, Any],
    semantic_type_overrides: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not semantic_type_overrides:
        return contract, profile
    for column in contract.get("columns") or []:
        name = str(column.get("name") or "")
        override = semantic_type_overrides.get(name)
        if override:
            column["semantic_type"] = override
    for name, column_profile in (profile.get("column_profiles") or {}).items():
        override = semantic_type_overrides.get(str(name))
        if not override:
            continue
        column_profile["inferred_type"] = "numerical" if override == "numeric" else override
    return contract, profile


def _humanize_identifier(name: str) -> str:
    cleaned = name.replace("_", " ").replace("-", " ").strip()
    if not cleaned:
        return "field"
    return " ".join(part for part in cleaned.split() if part)


def _infer_dataset_name(dataset_id: str, source_info: dict[str, Any]) -> str:
    source_url = str(source_info.get("source_url") or "").strip()
    if source_url:
        parsed = urlparse(source_url)
        parts = [part for part in parsed.path.split("/") if part]
        if parts:
            tail = unquote(parts[-1]).replace("+", " ").replace("-", " ").replace("_", " ").strip()
            if tail and not tail.isdigit():
                return tail.title()
    return dataset_id.upper() if dataset_id.isalnum() else dataset_id


def _infer_description(column_name: str, role: str, semantic_type: str) -> str:
    label = _humanize_identifier(column_name)
    if role == "target":
        return f"Target field for {label}."
    if "identifier" in semantic_type:
        return f"Identifier-like field for {label}."
    if "numeric" in semantic_type:
        return f"Numeric field for {label}."
    if "categorical" in semantic_type or "binary" in semantic_type:
        return f"Categorical field for {label}."
    if "text" in semantic_type:
        return f"Text field for {label}."
    return f"Schema field for {label}."


def _support_thresholds(row_count: int, *, sparse_like: bool = False) -> dict[str, Any]:
    if row_count <= 5_000:
        absolute_min = 20
    elif row_count <= 25_000:
        absolute_min = 25
    elif row_count <= 100_000:
        absolute_min = 30
    else:
        absolute_min = 40
    subgroup_min = absolute_min + 5
    thresholds: dict[str, Any] = {
        "absolute_min_rows": absolute_min,
        "subgroup_min_rows": subgroup_min,
        "predicate_min_selectivity": 0.005 if sparse_like or row_count >= 100_000 else 0.01,
    }
    return thresholds


def _relative_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target_path, start=link_path.parent)
    if link_path.is_symlink():
        current = os.readlink(link_path)
        if current == relative_target:
            return
        link_path.unlink()
    elif link_path.exists():
        link_path.unlink()
    link_path.symlink_to(relative_target)


def _canonical_metadata_path(dataset_dir: Path, asset_name: str) -> Path:
    mapping = {
        "dataset_description": dataset_dir / "metadata_core" / "dataset_description.txt",
        "dataset_semantics": dataset_dir / "metadata_core" / "dataset_semantics.yaml",
        "field_registry": dataset_dir / "metadata_core" / "field_registry.json",
        "query_policy": dataset_dir / "metadata_core" / "query_policy.yaml",
        "validation_policy": dataset_dir / "metadata_core" / "validation_policy.yaml",
        "family_applicability": dataset_dir / "metadata_optional" / "family_applicability.json",
        "risk_register": dataset_dir / "metadata_optional" / "risk_register.json",
        "uncertainty_register": dataset_dir / "metadata_optional" / "uncertainty_register.json",
    }
    return mapping[asset_name]


def _legacy_metadata_path(dataset_dir: Path, asset_name: str) -> Path:
    suffix = {
        "dataset_description": "dataset_description.txt",
        "dataset_semantics": "dataset_semantics.yaml",
        "field_registry": "field_registry.json",
        "query_policy": "query_policy.yaml",
        "validation_policy": "validation_policy.yaml",
        "family_applicability": "family_applicability.json",
        "risk_register": "risk_register.json",
        "uncertainty_register": "uncertainty_register.json",
    }
    return dataset_dir / "metadata" / suffix[asset_name]


def _resolved_existing_asset(dataset_id: str, asset_name: str, data_root: Path) -> Path | None:
    candidates = dataset_asset_candidates(dataset_id, asset_name, data_root)
    for path in candidates:
        if path.exists():
            return path
    return None


def _normalize_field_specs(
    *,
    dataset_id: str,
    contract: dict[str, Any],
    profile: dict[str, Any],
    source_info: dict[str, Any],
) -> dict[str, Any]:
    columns = contract.get("columns") or []
    task_type = str(profile.get("task_type") or contract.get("task_type") or "unknown")
    target_column = str(profile.get("target_column") or contract.get("target_column") or "")
    row_count = int(
        (contract.get("row_counts") or {}).get("main")
        or (profile.get("summary") or {}).get("n_rows")
        or 0
    )
    field_rows: list[dict[str, Any]] = []

    group_candidates: list[str] = []
    measure_candidates: list[str] = []
    predicate_candidates: list[str] = []
    entity_candidates: list[str] = []
    item_candidates: list[str] = []
    missing_fields: list[str] = []
    high_cardinality_fields: list[str] = []
    id_fields: list[str] = []
    low_cardinality_fields: list[str] = []
    numeric_backfill_candidates: list[tuple[int, str]] = []
    sparse_like = False

    for column in columns:
        name = str(column.get("name") or "").strip()
        if not name:
            continue

        base_role = str(column.get("role") or "feature")
        is_target = name == target_column or base_role == "target"
        stats = column.get("profile_stats") or {}
        unique_count = int(stats.get("unique_count") or 0)
        unique_ratio = float(stats.get("unique_ratio") or 0.0)
        missing_rate = float(stats.get("missing_rate") or 0.0)
        example_values = [value for value in (stats.get("example_values") or []) if value is not None]
        avg_sample_len = (
            sum(len(str(value)) for value in example_values) / len(example_values)
            if example_values
            else 0.0
        )
        source_semantic = str(column.get("semantic_type") or "").lower()
        lower_name = name.lower()
        examples_fit_boolean = _values_fit_boolean_domain(example_values)
        has_numeric_examples = bool(example_values) and all(_is_numeric_token(value) for value in example_values)
        is_numeric = (
            "numeric" in source_semantic
            or source_semantic in {"integer", "float"}
            or ("boolean" in source_semantic and not examples_fit_boolean and has_numeric_examples)
        )
        is_boolean = "boolean" in source_semantic and examples_fit_boolean
        is_categorical = "categorical" in source_semantic or is_boolean
        is_text = "text" in source_semantic
        looks_id_name = lower_name.endswith("_id") or lower_name == "id" or lower_name.endswith("id")
        is_identifier_like = (
            not is_target
            and (
                looks_id_name
                or (unique_ratio >= 0.95 and unique_count >= max(50, int(row_count * 0.2)))
                or (is_text and avg_sample_len >= 18 and unique_ratio >= 0.8)
            )
        )
        moderate_discrete_numeric = is_numeric and not is_target and unique_count > 0 and (
            unique_count <= 32 or unique_ratio <= 0.02
        )
        high_cardinality = (
            unique_count >= max(20, min(200, int(max(row_count, 1) * 0.05)))
            or unique_ratio >= 0.2
        )
        if is_text and unique_ratio >= 0.5 and avg_sample_len >= 12:
            high_cardinality = True

        if is_identifier_like:
            declared_type = "id"
            semantic_type = "identifier_numeric" if is_numeric else "identifier_string"
        elif is_target:
            if task_type == "classification":
                if is_boolean or unique_count <= 2:
                    declared_type = "boolean" if is_boolean else "categorical"
                    semantic_type = "binary_target"
                else:
                    declared_type = "categorical" if not is_numeric else "categorical"
                    semantic_type = "categorical_target"
            elif is_numeric:
                declared_type = "numeric"
                semantic_type = "numeric_target"
            else:
                declared_type = "categorical"
                semantic_type = "target"
        elif is_boolean:
            declared_type = "boolean"
            semantic_type = "categorical_binary"
        elif is_categorical:
            declared_type = "categorical"
            if example_values and all(_is_numeric_token(value) for value in example_values):
                semantic_type = "categorical_ordinal_count_like"
            else:
                semantic_type = "categorical_nominal"
        elif is_numeric:
            declared_type = "numeric"
            semantic_type = "numeric_discrete" if moderate_discrete_numeric else "numeric"
        elif is_text and unique_count <= 256 and avg_sample_len <= 40:
            declared_type = "categorical"
            semantic_type = "categorical_nominal"
        elif is_text:
            declared_type = "text"
            semantic_type = "free_text"
        else:
            declared_type = "categorical"
            semantic_type = "categorical_nominal"

        ordered = False
        value_order: list[Any] = []
        if declared_type == "categorical" and example_values and len(example_values) <= 16:
            ordered = all(_is_numeric_token(value) for value in example_values)
            if ordered:
                value_order = [value for value in sorted({str(v) for v in example_values}, key=lambda item: float(item))]
                semantic_type = (
                    "categorical_ordinal_target"
                    if is_target and task_type == "classification"
                    else "categorical_ordinal_count_like"
                )
            else:
                deduped = []
                seen = set()
                for value in example_values:
                    marker = str(value)
                    if marker in seen:
                        continue
                    seen.add(marker)
                    deduped.append(value)
                value_order = deduped
        elif is_target and task_type == "classification" and example_values and len(example_values) <= 16:
            value_order = list(dict.fromkeys(example_values))

        use_for_groupby = False
        if not is_identifier_like:
            if is_target and task_type == "classification" and unique_count <= 20:
                use_for_groupby = True
            elif not is_target:
                if declared_type in {"categorical", "boolean"} and unique_count <= 256:
                    use_for_groupby = True
                elif moderate_discrete_numeric:
                    use_for_groupby = True
        use_for_predicate = not is_identifier_like and declared_type != "id"
        if declared_type == "text" and high_cardinality:
            use_for_predicate = False
        use_as_target = is_target

        field_tags: list[str] = []
        risk_tags: list[str] = []
        probe_hints: list[str] = []

        if is_identifier_like:
            field_tags.extend(["identifier", "probe_exclude"])
            id_fields.append(name)
            if high_cardinality:
                field_tags.append("high_cardinality_candidate")
        else:
            if use_for_groupby:
                field_tags.append("subgroup_candidate")
                probe_hints.append("allow_groupby")
            if use_for_predicate:
                field_tags.append("condition_candidate")
            if declared_type == "numeric" and not use_as_target:
                field_tags.append("measure")
                probe_hints.append("numeric_thresholds")
            if high_cardinality and not use_as_target:
                field_tags.append("high_cardinality_candidate")
            if declared_type == "text":
                field_tags.append("text_exclude")
            if use_as_target:
                field_tags.append("target_candidate")
        if missing_rate > 0.0:
            risk_tags.append("missingness_present")
            field_tags.append("missingness_candidate")
            missing_fields.append(name)
        if high_cardinality and not is_identifier_like:
            high_cardinality_fields.append(name)
        if moderate_discrete_numeric:
            probe_hints.append("discrete_numeric_ok")
        elif declared_type == "numeric":
            probe_hints.append("bin_before_groupby")
        if use_as_target:
            risk_tags.append("support_instability_risk")
            if task_type == "classification":
                probe_hints.append("target_coupled_allowed")
            else:
                probe_hints.append("tail_sensitive")
        if use_for_groupby:
            group_candidates.append(name)
        elif declared_type == "numeric" and not use_as_target and unique_count > 0 and unique_count <= 1024:
            numeric_backfill_candidates.append((unique_count, name))
        if declared_type == "numeric" or use_as_target and semantic_type == "numeric_target":
            measure_candidates.append(name)
        if use_for_predicate:
            predicate_candidates.append(name)
        if high_cardinality and not is_identifier_like and not use_as_target:
            entity_candidates.append(name)
            item_candidates.append(name)
        if use_for_groupby and unique_count <= 16:
            low_cardinality_fields.append(name)
        if declared_type == "numeric" and unique_ratio <= 0.02 and unique_count >= 20:
            sparse_like = True

        field_rows.append(
            {
                "name": name,
                "declared_type": declared_type,
                "semantic_type": semantic_type,
                "role": "target" if use_as_target else "feature",
                "description": _infer_description(name, "target" if use_as_target else "feature", semantic_type),
                "ordered": ordered,
                "value_order": value_order,
                "field_tags": list(dict.fromkeys(field_tags)),
                "risk_tags": list(dict.fromkeys(risk_tags)),
                "use_for_groupby": use_for_groupby,
                "use_for_predicate": use_for_predicate,
                "use_as_target": use_as_target,
                "probe_eligibility_hints": list(dict.fromkeys(probe_hints or ["default_allow"])),
                "confidence": 0.82 if not use_as_target else 0.9,
                "evidence": [
                    {
                        "type": "contract",
                        "path": f"data/artifacts/data_core/tabular/{dataset_id}/{dataset_id}-dataset_contract_v1.json",
                        "detail": f"role={base_role}, semantic_type={column.get('semantic_type')}",
                    },
                    {
                        "type": "profile",
                        "path": f"data/artifacts/data_core/tabular/{dataset_id}/{dataset_id}-dataset_profile.json",
                        "detail": f"unique_count={unique_count}, missing_rate={missing_rate}",
                    },
                    {
                        "type": "source",
                        "url": source_info.get("source_url"),
                        "detail": "Referenced in source_info.json.",
                    },
                ],
            }
        )

    if not group_candidates and numeric_backfill_candidates:
        backfilled_names = [name for _count, name in sorted(numeric_backfill_candidates)[:2]]
        for field in field_rows:
            if field["name"] not in backfilled_names:
                continue
            field["use_for_groupby"] = True
            field["field_tags"] = list(dict.fromkeys([*field["field_tags"], "subgroup_candidate"]))
            field["probe_eligibility_hints"] = list(
                dict.fromkeys([*field["probe_eligibility_hints"], "allow_groupby_numeric_backfill"])
            )
            group_candidates.append(field["name"])
            if field["name"] not in low_cardinality_fields:
                low_cardinality_fields.append(field["name"])

    return {
        "task_type": task_type,
        "target_column": target_column,
        "row_count": row_count,
        "fields": field_rows,
        "group_candidates": list(dict.fromkeys(group_candidates)),
        "measure_candidates": list(dict.fromkeys(measure_candidates)),
        "predicate_candidates": list(dict.fromkeys(predicate_candidates)),
        "entity_candidates": list(dict.fromkeys(entity_candidates)),
        "item_candidates": list(dict.fromkeys(item_candidates)),
        "missing_fields": list(dict.fromkeys(missing_fields)),
        "high_cardinality_fields": list(dict.fromkeys(high_cardinality_fields)),
        "id_fields": list(dict.fromkeys(id_fields)),
        "low_cardinality_fields": list(dict.fromkeys(low_cardinality_fields)),
        "sparse_like": sparse_like,
    }


def _choose_useful_field_combinations(field_spec: dict[str, Any]) -> list[list[str]]:
    combos: list[list[str]] = []
    group_candidates = field_spec["group_candidates"]
    measure_candidates = [name for name in field_spec["measure_candidates"] if name != field_spec["target_column"]]
    predicate_candidates = [name for name in field_spec["predicate_candidates"] if name != field_spec["target_column"]]
    target = field_spec["target_column"]

    if len(group_candidates) >= 2 and target:
        combos.append([group_candidates[0], group_candidates[1], target])
    if group_candidates and measure_candidates:
        combos.append([group_candidates[0], measure_candidates[0], target or measure_candidates[0]])
    if predicate_candidates and group_candidates and target:
        combos.append([group_candidates[0], predicate_candidates[0], target])
    if predicate_candidates and len(predicate_candidates) >= 2 and target:
        combos.append([predicate_candidates[0], predicate_candidates[1], target])

    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for combo in combos:
        cleaned = [value for value in combo if value]
        marker = tuple(cleaned)
        if len(cleaned) < 2 or marker in seen:
            continue
        seen.add(marker)
        unique.append(cleaned)
    return unique[:3]


def _build_dataset_description(dataset_name: str, source_info: dict[str, Any], field_spec: dict[str, Any], column_count: int) -> str:
    task_type = field_spec["task_type"]
    target_column = field_spec["target_column"]
    row_count = field_spec["row_count"]
    source_name = source_info.get("source_name") or "public source"
    source_type = source_info.get("source_type") or "unknown"
    return (
        f"{dataset_name} is a tabular {task_type} dataset with {row_count} rows and {column_count} columns. "
        f"It is sourced from {source_name} ({source_type}). "
        f"One row represents a single observational record with target column `{target_column}`."
    )


def _build_dataset_semantics(
    *,
    dataset_id: str,
    dataset_name: str,
    source_info: dict[str, Any],
    field_spec: dict[str, Any],
    column_count: int,
) -> dict[str, Any]:
    structural_tags = ["has_target_candidate"]
    if field_spec["high_cardinality_fields"]:
        structural_tags.append("has_high_cardinality_features")
    if field_spec["missing_fields"]:
        structural_tags.append("has_missingness")
    if field_spec["group_candidates"]:
        structural_tags.append("has_subgroup_axes")

    risk_tags: list[str] = []
    if field_spec["missing_fields"]:
        risk_tags.append("missingness_risk")
    if field_spec["high_cardinality_fields"]:
        risk_tags.append("cardinality_skew_risk")
    if field_spec["task_type"] == "classification":
        risk_tags.append("support_instability_risk")
    if field_spec["sparse_like"]:
        risk_tags.append("high_sparsity")

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "source_url": source_info.get("source_url"),
        "row_semantics": (
            f"One row is one tabular observation with {column_count - 1 if column_count > 0 else column_count} "
            f"feature columns and target `{field_spec['target_column']}`."
        ),
        "task_type": field_spec["task_type"],
        "target_column": field_spec["target_column"],
        "dataset_structural_tags": structural_tags,
        "dataset_risk_tags": risk_tags,
        "notes": [
            "This metadata package was materialized from dataset contract/profile artifacts to support template grounding.",
            "Canonical preprocessing assets now live under metadata_core/ and metadata_optional/ with legacy metadata/ compatibility links.",
        ],
        "uncertainties": [
            "Some semantics are heuristic because only schema/profile/source artifacts are locally cached for this dataset."
        ],
    }


def _build_family_applicability(dataset_id: str, field_spec: dict[str, Any]) -> dict[str, Any]:
    group_candidates = field_spec["group_candidates"]
    predicate_candidates = field_spec["predicate_candidates"]
    measure_candidates = field_spec["measure_candidates"]
    missing_fields = field_spec["missing_fields"]
    high_cardinality_fields = field_spec["high_cardinality_fields"]
    task_type = field_spec["task_type"]

    subgroup_status = "applicable" if group_candidates else "likely_not_applicable"
    conditional_status = "applicable" if predicate_candidates and (group_candidates or measure_candidates) else "likely_not_applicable"
    if task_type == "classification":
        tail_status = "likely_applicable"
    elif measure_candidates:
        tail_status = "applicable"
    else:
        tail_status = "uncertain"
    missingness_status = "applicable" if missing_fields else "likely_not_applicable"
    cardinality_status = "applicable" if high_cardinality_fields else "likely_not_applicable"

    families = [
        {
            "family_name": "subgroup_structure",
            "status": subgroup_status,
            "candidate_fields": group_candidates[:8],
            "candidate_axes": [f"{name}_subgroup" for name in group_candidates[:3]],
            "reason": "Dataset exposes one or more fields that are safe to use as subgroup axes." if group_candidates else "No stable subgroup axis was detected from current metadata.",
            "anticipated_failure_modes": ["subgroup_support_fragmentation"],
            "notes": "Prefer low-cardinality categorical or discrete numeric fields for group analyses.",
        },
        {
            "family_name": "conditional_dependency_structure",
            "status": conditional_status,
            "candidate_fields": (predicate_candidates[:4] + measure_candidates[:2])[:8],
            "candidate_axes": [f"{predicate_candidates[0]}_conditioned_pattern"] if predicate_candidates else [],
            "reason": "There are predicate-eligible fields plus target/measure fields for dependency checks." if conditional_status == "applicable" else "Conditional structure is weak under current field typing.",
            "anticipated_failure_modes": ["over_fragmented_conditions"],
            "notes": "Use support guards when combining multiple conditions.",
        },
        {
            "family_name": "tail_rarity_structure",
            "status": tail_status,
            "candidate_fields": (measure_candidates[:4] or [field_spec["target_column"]])[:8],
            "candidate_axes": [f"{(measure_candidates or [field_spec['target_column']])[0]}_tail"] if (measure_candidates or [field_spec["target_column"]]) else [],
            "reason": "Numeric measures or classification rarity permit tail-oriented probes.",
            "anticipated_failure_modes": ["extreme_support_instability"],
            "notes": "Tail queries should share the same support thresholds as validation policy.",
        },
        {
            "family_name": "missingness_structure",
            "status": missingness_status,
            "candidate_fields": missing_fields[:8],
            "candidate_axes": [f"{name}_missingness" for name in missing_fields[:3]],
            "reason": "Observed missing values can support missingness-aware templates." if missing_fields else "No missing values were detected in current artifacts.",
            "anticipated_failure_modes": ["missing_token_mismatch"] if missing_fields else ["none_expected"],
            "notes": "Use null checks only when the field-level contract reports missingness.",
        },
        {
            "family_name": "cardinality_structure",
            "status": cardinality_status,
            "candidate_fields": high_cardinality_fields[:8],
            "candidate_axes": [f"{name}_heavy_hitter" for name in high_cardinality_fields[:3]],
            "reason": "High-cardinality fields can support heavy-hitter and concentration style probes." if high_cardinality_fields else "No obvious high-cardinality field was detected.",
            "anticipated_failure_modes": ["identifier_leakage"] if high_cardinality_fields else ["none_expected"],
            "notes": "Exclude identifier-like fields from direct semantic interpretation.",
        },
    ]
    return {
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "family_review_basis": "materialize_template_preprocessing_layout.py",
        "families": families,
    }


def _build_query_policy(dataset_id: str, field_spec: dict[str, Any]) -> dict[str, Any]:
    preferred: list[str] = []
    if field_spec["group_candidates"]:
        preferred.append("subgroup_structure")
    if field_spec["predicate_candidates"]:
        preferred.append("conditional_dependency_structure")
    if field_spec["measure_candidates"] or field_spec["task_type"] == "classification":
        preferred.append("tail_rarity_structure")
    preferred = preferred[:3]

    discouraged: list[str] = []
    if not field_spec["missing_fields"]:
        discouraged.append("missingness_structure")
    if not field_spec["high_cardinality_fields"]:
        discouraged.append("cardinality_structure")

    preferred_roles = []
    if field_spec["group_candidates"]:
        preferred_roles.append("subgroup_candidate")
    if field_spec["predicate_candidates"]:
        preferred_roles.append("condition_candidate")
    if field_spec["measure_candidates"]:
        preferred_roles.append("measure")
    preferred_roles.append("target_candidate")

    return {
        "dataset_id": dataset_id,
        "preferred_families": preferred,
        "discouraged_families": discouraged,
        "preferred_field_roles": list(dict.fromkeys(preferred_roles)),
        "discouraged_fields": field_spec["id_fields"][:8],
        "useful_field_combinations": _choose_useful_field_combinations(field_spec),
        "target_coupled_opportunities": [
            f"Anchor template bindings on target `{field_spec['target_column']}` and the strongest subgroup/condition candidates."
        ],
        "notes": [
            "Policy is intentionally abstract and dataset-specific column grounding happens later at runtime.",
            "Identifier-like fields are excluded from direct semantic template binding.",
        ],
    }


def _build_validation_policy(dataset_id: str, field_spec: dict[str, Any]) -> dict[str, Any]:
    thresholds = _support_thresholds(field_spec["row_count"], sparse_like=field_spec["sparse_like"])
    if field_spec["task_type"] == "classification":
        thresholds["target_class_min_rows"] = thresholds["absolute_min_rows"]
    else:
        thresholds["target_tail_min_rows"] = thresholds["absolute_min_rows"]

    support_families = [family for family in ["subgroup_structure", "conditional_dependency_structure", "tail_rarity_structure"]]
    validator_families = ["support_guard_validator", "consistency_validator"]
    if field_spec["task_type"] == "classification":
        validator_families.append("class_support_validator")
    else:
        validator_families.append("numeric_tail_validator")
    if field_spec["missing_fields"]:
        validator_families.append("missingness_validator")

    caution_fields = list(
        dict.fromkeys(
            [field_spec["target_column"], *field_spec["high_cardinality_fields"][:3], *field_spec["missing_fields"][:3]]
        )
    )

    return {
        "dataset_id": dataset_id,
        "minimum_support_thresholds": thresholds,
        "support_sensitive_families": support_families,
        "recommended_validator_families": validator_families,
        "warning_conditions": [
            "Warn when subgroup support falls below configured minimums.",
            "Warn when predicates target ultra-thin slices without a paired verification SQL.",
        ],
        "fields_requiring_caution": [field for field in caution_fields if field],
        "notes": [
            "Thresholds are generic preprocessing defaults and can be tightened in dataset-specific experiments.",
        ],
    }


def _build_risk_register(dataset_id: str, field_spec: dict[str, Any]) -> dict[str, Any]:
    risks: list[dict[str, Any]] = []
    if field_spec["task_type"] == "classification":
        risks.append(
            {
                "risk_id": f"{dataset_id}_risk_target_support",
                "scope": "dataset",
                "description": "Classification targets can become support-fragile under multi-field conditioning.",
                "severity": "high",
                "evidence": [f"target_column={field_spec['target_column']}"],
                "suggested_handling": "Always pair generated SQL with explicit support-check validation SQL.",
            }
        )
    if field_spec["high_cardinality_fields"]:
        risks.append(
            {
                "risk_id": f"{dataset_id}_risk_high_cardinality",
                "scope": "field",
                "description": "High-cardinality fields can create brittle heavy-hitter or distinct-count behavior.",
                "severity": "medium",
                "evidence": [f"high_cardinality_fields={field_spec['high_cardinality_fields'][:5]}"],
                "suggested_handling": "Prefer semantic entity roles and avoid identifier leakage.",
            }
        )
    if field_spec["missing_fields"]:
        risks.append(
            {
                "risk_id": f"{dataset_id}_risk_missingness",
                "scope": "field",
                "description": "Missing-value handling may change support and predicate behavior.",
                "severity": "medium",
                "evidence": [f"missing_fields={field_spec['missing_fields'][:5]}"],
                "suggested_handling": "Keep null-aware validation active for the affected columns.",
            }
        )
    if not risks:
        risks.append(
            {
                "risk_id": f"{dataset_id}_risk_generic_support",
                "scope": "dataset",
                "description": "Even clean tabular datasets can fragment under aggressive template binding.",
                "severity": "medium",
                "evidence": ["heuristic preprocessing package"],
                "suggested_handling": "Use minimum-support checks before retaining any template/problem instance.",
            }
        )
    return {
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "risks": risks,
    }


def _build_uncertainty_register(dataset_id: str, source_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "uncertainties": [
            {
                "item": f"{dataset_id}_u1",
                "scope": "dataset",
                "uncertainty_type": "heuristic_metadata_materialization",
                "description": "This preprocessing package was synthesized from contract/profile/source artifacts rather than fully manual semantic annotation.",
                "why_uncertain": "Local cache does not contain a complete hand-authored semantics memo for this dataset.",
                "possible_resolutions": [
                    "Promote this dataset to a manually reviewed metadata package if it becomes a focus dataset."
                ],
            },
            {
                "item": f"{dataset_id}_u2",
                "scope": "source",
                "uncertainty_type": "source_context_depth",
                "description": f"Source context is limited to locally cached source_info fields from {source_info.get('source_type') or 'unknown'} metadata.",
                "why_uncertain": "Full dataset card / benchmark narrative is not guaranteed to be cached locally.",
                "possible_resolutions": [
                    "Attach fuller citation and source-card context under source/ when needed for paper-ready writeups."
                ],
            },
        ],
    }


def _write_or_migrate_metadata_asset(
    *,
    dataset_dir: Path,
    asset_name: str,
    generator_payload: Any | None,
    action_log: list[str],
    force_rewrite: bool = False,
) -> str:
    canonical_path = _canonical_metadata_path(dataset_dir, asset_name)
    legacy_path = _legacy_metadata_path(dataset_dir, asset_name)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)

    if force_rewrite and canonical_path.exists() and generator_payload is not None:
        if canonical_path.suffix == ".json":
            _dump_json(canonical_path, generator_payload)
        elif canonical_path.suffix == ".yaml":
            _dump_yaml(canonical_path, generator_payload)
        else:
            canonical_path.write_text(str(generator_payload).strip() + "\n", encoding="utf-8")
        action_log.append(f"refreshed:{asset_name}")
        status = "refreshed"
    elif not canonical_path.exists():
        if legacy_path.exists() and not legacy_path.is_symlink():
            shutil.move(str(legacy_path), str(canonical_path))
            action_log.append(f"migrated:{asset_name}")
            status = "migrated"
        elif generator_payload is not None:
            if canonical_path.suffix == ".json":
                _dump_json(canonical_path, generator_payload)
            elif canonical_path.suffix == ".yaml":
                _dump_yaml(canonical_path, generator_payload)
            else:
                canonical_path.write_text(str(generator_payload).strip() + "\n", encoding="utf-8")
            action_log.append(f"generated:{asset_name}")
            status = "generated"
        else:
            status = "missing"
    else:
        status = "existing"

    if canonical_path.exists():
        _relative_symlink(legacy_path, canonical_path)
        action_log.append(f"linked_legacy:{asset_name}")
    return status


def _dataset_metadata_is_script_managed(dataset_dir: Path) -> bool:
    semantics_path = dataset_dir / "metadata_core" / "dataset_semantics.yaml"
    family_path = dataset_dir / "metadata_optional" / "family_applicability.json"
    if semantics_path.exists():
        try:
            payload = yaml.safe_load(semantics_path.read_text(encoding="utf-8")) or {}
        except Exception:
            payload = {}
        notes = payload.get("notes") or []
        if any("materialized from dataset contract/profile artifacts" in str(note) for note in notes):
            return True
    if family_path.exists():
        try:
            payload = _load_json(family_path)
        except Exception:
            payload = {}
        if payload.get("family_review_basis") == "materialize_template_preprocessing_layout.py":
            return True
    return False


def _materialize_dataset(dataset_id: str, data_root: Path) -> dict[str, Any]:
    dataset_dir = data_root / dataset_id
    metadata_dir = dataset_dir / "metadata"
    source_dir = dataset_dir / "source"
    contracts_dir = dataset_dir / "contracts"
    metadata_core_dir = dataset_dir / "metadata_core"
    metadata_optional_dir = dataset_dir / "metadata_optional"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir.mkdir(parents=True, exist_ok=True)
    metadata_core_dir.mkdir(parents=True, exist_ok=True)
    metadata_optional_dir.mkdir(parents=True, exist_ok=True)

    contract_path = _resolved_existing_asset(dataset_id, "dataset_contract", data_root)
    profile_path = _resolved_existing_asset(dataset_id, "dataset_profile", data_root)
    source_info_path = _resolved_existing_asset(dataset_id, "source_info", data_root)
    if contract_path is None or profile_path is None or source_info_path is None:
        raise FileNotFoundError(
            f"{dataset_id}: missing contract/profile/source_info required for preprocessing materialization"
        )

    contract = _load_json(contract_path)
    profile = _load_json(profile_path)
    source_info = _load_json(source_info_path)
    semantic_type_overrides = _load_semantic_type_overrides(dataset_dir)
    contract, profile = _apply_semantic_type_overrides(contract, profile, semantic_type_overrides)
    column_count = int(contract.get("column_count") or len(contract.get("columns") or []))
    dataset_name = _infer_dataset_name(dataset_id, source_info)
    field_spec = _normalize_field_specs(
        dataset_id=dataset_id,
        contract=contract,
        profile=profile,
        source_info=source_info,
    )
    refresh_script_managed = _dataset_metadata_is_script_managed(dataset_dir)

    action_log: list[str] = []
    metadata_payloads = {
        "dataset_description": _build_dataset_description(dataset_name, source_info, field_spec, column_count),
        "dataset_semantics": _build_dataset_semantics(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            source_info=source_info,
            field_spec=field_spec,
            column_count=column_count,
        ),
        "field_registry": {
            "dataset_id": dataset_id,
            "generated_at": _now_iso(),
            "fields": field_spec["fields"],
        },
        "query_policy": _build_query_policy(dataset_id, field_spec),
        "validation_policy": _build_validation_policy(dataset_id, field_spec),
        "family_applicability": _build_family_applicability(dataset_id, field_spec),
        "risk_register": _build_risk_register(dataset_id, field_spec),
        "uncertainty_register": _build_uncertainty_register(dataset_id, source_info),
    }

    asset_status: dict[str, str] = {}
    for asset_name in [*CORE_METADATA_FILES, *OPTIONAL_METADATA_FILES]:
        asset_status[asset_name] = _write_or_migrate_metadata_asset(
            dataset_dir=dataset_dir,
            asset_name=asset_name,
            generator_payload=metadata_payloads[asset_name],
            action_log=action_log,
            force_rewrite=refresh_script_managed,
        )

    for canonical_name, asset_name in (
        ("dataset_profile.json", "dataset_profile"),
        ("dataset_contract_v1.json", "dataset_contract"),
    ):
        canonical_contract_path = contracts_dir / canonical_name
        resolved = _resolved_existing_asset(dataset_id, asset_name, data_root)
        if resolved and canonical_contract_path.resolve() != resolved.resolve() if canonical_contract_path.exists() else True:
            _relative_symlink(canonical_contract_path, resolved)
            action_log.append(f"linked_contract:{canonical_name}")

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "task_type": field_spec["task_type"],
        "target_column": field_spec["target_column"],
        "row_count": field_spec["row_count"],
        "column_count": column_count,
        "group_candidate_count": len(field_spec["group_candidates"]),
        "measure_candidate_count": len(field_spec["measure_candidates"]),
        "predicate_candidate_count": len(field_spec["predicate_candidates"]),
        "asset_status": asset_status,
        "actions": action_log,
    }


def _materialize_workload_grounding_layout(data_root: Path) -> list[str]:
    wg_root = data_root / "workload_grounding"
    wg_root.mkdir(parents=True, exist_ok=True)
    actions: list[str] = []
    for canonical_rel, legacy_rel in WORKLOAD_GROUNDING_LAYOUT.items():
        canonical_path = wg_root / canonical_rel
        legacy_path = wg_root / legacy_rel
        if not legacy_path.exists():
            continue
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        _relative_symlink(canonical_path, legacy_path)
        actions.append(f"linked:{canonical_rel}")

    readme_path = wg_root / "STRUCTURE_V2.md"
    lines = [
        "# Workload Grounding Structure V2",
        "",
        "This directory now exposes a canonical layered layout while retaining legacy root-level paths for compatibility.",
        "",
        "## Canonical subdirectories",
        "- `library/core/`: primary template library and workload provenance tables",
        "- `library/extensions/`: optional extension templates",
        "- `policy/`: template-level policy assets",
        "- `provenance/`: evidence tables and provenance ledgers",
        "- `runtime_specs/`: runtime shortlist specs consumed by agent/inventory scripts",
        "- `portability/`: static portability reports",
        "- `inventories/`: generated question inventories",
        "- `reports/`: evaluation and migration reports",
        "",
        "Legacy root-level files are preserved and the canonical paths are materialized as symlinks.",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    actions.append("wrote:STRUCTURE_V2.md")
    return actions


def _write_report(report_dir: Path, dataset_rows: list[dict[str, Any]], wg_actions: list[str]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": _now_iso(),
        "dataset_count": len(dataset_rows),
        "datasets": dataset_rows,
        "workload_grounding_actions": wg_actions,
    }
    _dump_json(report_dir / "materialization_summary.json", summary)

    csv_path = report_dir / "materialization_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_id",
                "dataset_name",
                "task_type",
                "target_column",
                "row_count",
                "column_count",
                "group_candidate_count",
                "measure_candidate_count",
                "predicate_candidate_count",
                "generated_assets",
                "migrated_assets",
                "refreshed_assets",
                "existing_assets",
            ],
        )
        writer.writeheader()
        for row in dataset_rows:
            status_counter = Counter(row["asset_status"].values())
            writer.writerow(
                {
                    "dataset_id": row["dataset_id"],
                    "dataset_name": row["dataset_name"],
                    "task_type": row["task_type"],
                    "target_column": row["target_column"],
                    "row_count": row["row_count"],
                    "column_count": row["column_count"],
                    "group_candidate_count": row["group_candidate_count"],
                    "measure_candidate_count": row["measure_candidate_count"],
                    "predicate_candidate_count": row["predicate_candidate_count"],
                    "generated_assets": status_counter.get("generated", 0),
                    "migrated_assets": status_counter.get("migrated", 0),
                    "refreshed_assets": status_counter.get("refreshed", 0),
                    "existing_assets": status_counter.get("existing", 0),
                }
            )


def main() -> None:
    args = parse_args()
    dataset_ids = (
        [item.strip() for item in args.dataset_ids.split(",") if item.strip()]
        if args.dataset_ids.strip()
        else _list_dataset_ids(args.data_root)
    )

    dataset_rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        row = _materialize_dataset(dataset_id, args.data_root)
        dataset_rows.append(row)
        status_counter = Counter(row["asset_status"].values())
        print(
            f"[materialize] dataset_id={dataset_id} generated={status_counter.get('generated', 0)} "
            f"migrated={status_counter.get('migrated', 0)} existing={status_counter.get('existing', 0)} "
            f"group_candidates={row['group_candidate_count']}"
        )

    wg_actions = _materialize_workload_grounding_layout(args.data_root)
    _write_report(args.report_dir, dataset_rows, wg_actions)
    print(f"[materialize] report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
