"""Inventory builder for the isolated v2 workload line."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from src.benchmark.canonical_sql import stable_hash
from src.config.settings import DATA_DIR

from .catalog import build_template_library_rows
from .contract_spec import (
    CORE_AGENT_SUBITEMS,
    DETERMINISTIC_SUBITEMS,
    SUBITEM_TO_FAMILY,
    default_facet_ids_for_subitem,
)
from .dataset_profile import DatasetRoleProfile, load_dataset_role_profile
from .paths import combined_inventory_path, dataset_inventory_path, ensure_line_dirs


PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
AGENT_TEMPLATE_MIN = 10
AGENT_TEMPLATE_TARGET = 12
AGENT_PROBLEMS_PER_TEMPLATE_MIN = 4
AGENT_PROBLEMS_PER_TEMPLATE_MAX = 12
AGENT_FAMILY_TEMPLATE_MINIMUMS: dict[str, int] = {
    "subgroup_structure": 2,
    "conditional_dependency_structure": 4,
    "tail_rarity_structure": 4,
}
CORE_AGENT_FAMILIES = tuple(AGENT_FAMILY_TEMPLATE_MINIMUMS.keys())
TEMPLATE_PRIORITY_ORDER = {"primary": 0, "support": 1, "review": 2, "deterministic": 3}


@dataclass(frozen=True)
class V2InventoryItem:
    query_record_id: str
    problem_id: str
    dataset_id: str
    template_id: str
    template_name: str
    family_id: str
    canonical_subitem_id: str
    intended_facet_id: str
    variant_semantic_role: str
    subitem_assignment_source: str
    source_kind: str
    realization_mode: str
    gate_priority: str
    extended_family: bool
    question: str
    bindings: dict[str, Any]
    binding_roles: list[str]
    coverage_target_min: str
    runtime_sql_skeleton: str | None = None
    notes: list[str] | None = None
    template_selection_mode: str = ""
    selected_template_rank: int = 0
    problem_index_within_template: int = 0
    sql_variant_index: int = 1
    sql_variant_total: int = 1


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _template_rows_by_id() -> dict[str, dict[str, Any]]:
    return {row["template_id"]: row for row in build_template_library_rows()}


def _template_priority_rank(row: dict[str, Any]) -> int:
    return TEMPLATE_PRIORITY_ORDER.get(str(row.get("gate_priority")), 9)


def _template_rows_for_subitem(*, subitem_id: str, realization_mode: str) -> list[dict[str, Any]]:
    rows = []
    for row in build_template_library_rows():
        if row.get("realization_mode") != realization_mode:
            continue
        if subitem_id not in (row.get("supported_canonical_subitem_ids") or []):
            continue
        rows.append(row)
    rows.sort(
        key=lambda item: (
            _template_priority_rank(item),
            len(item.get("supported_canonical_subitem_ids") or []),
            str(item.get("template_id")),
        )
    )
    return rows


def _agent_template_rows() -> list[dict[str, Any]]:
    rows = [
        row
        for row in build_template_library_rows()
        if str(row.get("realization_mode")) == "agent"
        and str(row.get("family_id")) in CORE_AGENT_FAMILIES
    ]
    rows.sort(
        key=lambda item: (
            str(item.get("family_id")),
            _template_priority_rank(item),
            len(item.get("supported_canonical_subitem_ids") or []),
            str(item.get("template_id")),
        )
    )
    return rows


def _role_values(stats: Any) -> list[Any]:
    values = [value for value, _count in (stats.top_values or []) if value is not None]
    return values or ["unknown"]


def _choose_column(candidates: list[str], index: int, *, avoid: set[str] | None = None) -> str | None:
    avoid = avoid or set()
    usable = [value for value in candidates if value not in avoid]
    if not usable:
        return None
    return usable[index % len(usable)]


def _choose_group_pair(profile: DatasetRoleProfile, index: int) -> tuple[str | None, str | None]:
    pairs = list(combinations(profile.groupable_cols, 2))
    if not pairs:
        first = _choose_column(list(profile.groupable_cols), index)
        return first, None
    first, second = pairs[index % len(pairs)]
    return first, second


def _predicate_binding(profile: DatasetRoleProfile, index: int) -> dict[str, Any] | None:
    if not profile.filterable_cols:
        return None
    col = profile.filterable_cols[index % len(profile.filterable_cols)]
    stats = profile.field_stats[col]
    if stats.is_numeric and stats.q75 is not None:
        return {
            "predicate_col": col,
            "predicate_op": ">=",
            "predicate_value": round(float(stats.q75), 6),
        }
    values = _role_values(stats)
    return {
        "predicate_col": col,
        "predicate_op": "=",
        "predicate_value": values[index % len(values)],
    }


def _condition_values(profile: DatasetRoleProfile, condition_col: str) -> tuple[Any, Any]:
    values = _role_values(profile.field_stats[condition_col])
    if len(values) == 1:
        return values[0], values[0]
    return values[0], values[1]


def _binding_from_template(
    row: dict[str, Any],
    profile: DatasetRoleProfile,
    *,
    index: int,
) -> dict[str, Any] | None:
    placeholders = set(PLACEHOLDER_RE.findall(str(row.get("sql_skeleton") or "")))
    bindings: dict[str, Any] = {}

    group_col = _choose_column(list(profile.groupable_cols), index)
    group_pair = _choose_group_pair(profile, index)
    measure_col = _choose_column(list(profile.numeric_cols), index)
    target_col = profile.target_column or _choose_column(list(profile.condition_cols), index)
    condition_col = _choose_column(list(profile.condition_cols), index)
    predicate = _predicate_binding(profile, index)
    missing_col = _choose_column(list(profile.missing_cols), index)
    key_col = _choose_column(list(profile.high_card_cols), index)
    entity_col = _choose_column(list(profile.high_card_cols or profile.groupable_cols), index)
    item_col = _choose_column(list(profile.high_card_cols or profile.groupable_cols), index + 1)
    time_col = _choose_column(list(profile.temporal_cols), index)
    band_col = _choose_column(list(profile.numeric_cols), index)

    if "group_col" in placeholders:
        if not group_col:
            return None
        bindings["group_col"] = group_col
    if "group_col_2" in placeholders:
        first, second = group_pair
        if not first or not second:
            return None
        bindings["group_col"] = first
        bindings["group_col_2"] = second
    if "measure_col" in placeholders:
        if not measure_col:
            return None
        bindings["measure_col"] = measure_col
    if "target_col" in placeholders:
        if not target_col:
            return None
        bindings["target_col"] = target_col
        target_values = _role_values(profile.field_stats[target_col])
        bindings["target_value"] = target_values[index % max(1, len(target_values))]
    if "condition_col" in placeholders:
        if not condition_col:
            return None
        bindings["condition_col"] = condition_col
        positive, negative = _condition_values(profile, condition_col)
        bindings["condition_value"] = positive
        bindings["positive_value"] = positive
        bindings["negative_value"] = negative
    if "predicate_col" in placeholders:
        if not predicate:
            return None
        bindings.update(predicate)
    if "missing_col" in placeholders:
        if not missing_col:
            return None
        bindings["missing_col"] = missing_col
    if "key_col" in placeholders:
        if not key_col:
            return None
        bindings["key_col"] = key_col
    if "entity_col" in placeholders:
        if not entity_col:
            return None
        bindings["entity_col"] = entity_col
    if "item_col" in placeholders:
        if not item_col:
            return None
        bindings["item_col"] = item_col
    if "time_col" in placeholders:
        if not time_col:
            return None
        bindings["time_col"] = time_col
    if "band_col" in placeholders:
        if not band_col:
            return None
        stats = profile.field_stats[band_col]
        if stats.q33 is None or stats.q66 is None:
            return None
        bindings["band_col"] = band_col
        bindings["band_cut_1"] = round(float(stats.q33), 6)
        bindings["band_cut_2"] = round(float(stats.q66), 6)
        bindings["lower_bound"] = round(float(stats.q33), 6)
        bindings["upper_bound"] = round(float(stats.q66), 6)

    bindings.setdefault("top_k", 10 + (index % 5))
    bindings.setdefault("top_n", 3 + (index % 4))
    bindings.setdefault("num_tiles", 10)
    bindings.setdefault("percentile_value", 0.95 if index % 2 == 0 else 0.9)
    bindings.setdefault("z_threshold", 2.0)
    bindings.setdefault("fraction_threshold", 0.1)
    bindings.setdefault("baseline_multiplier", 1.5)
    bindings.setdefault("baseline_fraction", 0.1)
    bindings.setdefault("min_group_size", 5)
    bindings.setdefault("min_support", 5)
    bindings.setdefault(
        "measure_threshold",
        round(float(profile.field_stats[measure_col].q75), 6)
        if measure_col and profile.field_stats[measure_col].q75 is not None
        else 0.0,
    )
    bindings.setdefault("time_grain", "month")
    bindings.setdefault("lookback_rows", 3)
    bindings.setdefault("current_period_start", "'2024-01-01'")
    bindings.setdefault("current_period_end", "'2024-04-01'")
    bindings.setdefault("previous_period_start", "'2023-10-01'")
    bindings.setdefault("previous_period_end", "'2024-01-01'")
    bindings.setdefault("drift_ratio_threshold", 0.8)
    return bindings


def _question_text(
    *,
    row: dict[str, Any],
    subitem_id: str,
    bindings: dict[str, Any],
    variant_role: str,
) -> str:
    key_bits = []
    for key in (
        "group_col",
        "group_col_2",
        "measure_col",
        "condition_col",
        "target_col",
        "missing_col",
        "key_col",
    ):
        if key in bindings:
            key_bits.append(f"{key}={bindings[key]}")
    detail = ", ".join(key_bits) if key_bits else "default bindings"
    return (
        f"Use template {row['template_name']} to probe {subitem_id} "
        f"with semantic role {variant_role}. Focus on {detail}."
    )


def _problem_digest(
    *,
    dataset_id: str,
    row: dict[str, Any],
    subitem_id: str,
    facet_id: str,
    variant_role: str,
    base_bindings: dict[str, Any],
    problem_index: int,
) -> str:
    return stable_hash(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "template_id": row["template_id"],
                "subitem_id": subitem_id,
                "facet_id": facet_id,
                "variant_role": variant_role,
                "bindings": base_bindings,
                "problem_index": problem_index,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )[:16]


def _query_digest(
    *,
    dataset_id: str,
    row: dict[str, Any],
    problem_digest: str,
    facet_id: str,
    variant_role: str,
    bindings: dict[str, Any],
    variant_index: int,
) -> str:
    return stable_hash(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "template_id": row["template_id"],
                "problem_digest": problem_digest,
                "facet_id": facet_id,
                "variant_role": variant_role,
                "bindings": bindings,
                "variant_index": variant_index,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )[:16]


def _inventory_item(
    *,
    dataset_id: str,
    row: dict[str, Any],
    subitem_id: str,
    facet_id: str,
    variant_role: str,
    base_bindings: dict[str, Any],
    query_bindings: dict[str, Any],
    problem_index: int,
    variant_index: int,
    variant_total: int,
    selected_template_rank: int,
    template_selection_mode: str,
    extra_notes: list[str] | None = None,
) -> V2InventoryItem:
    problem_digest = _problem_digest(
        dataset_id=dataset_id,
        row=row,
        subitem_id=subitem_id,
        facet_id=facet_id,
        variant_role=variant_role,
        base_bindings=base_bindings,
        problem_index=problem_index,
    )
    query_digest = _query_digest(
        dataset_id=dataset_id,
        row=row,
        problem_digest=problem_digest,
        facet_id=facet_id,
        variant_role=variant_role,
        bindings=query_bindings,
        variant_index=variant_index,
    )
    notes = [
        f"default_facets={','.join(default_facet_ids_for_subitem(subitem_id))}",
        f"template_selection_mode={template_selection_mode}",
        f"problem_index_within_template={problem_index + 1}",
        f"sql_variant_index={variant_index + 1}/{variant_total}",
    ]
    if extra_notes:
        notes.extend(extra_notes)
    return V2InventoryItem(
        query_record_id=f"v2q_{dataset_id}_{query_digest}",
        problem_id=f"v2p_{dataset_id}_{problem_digest}",
        dataset_id=dataset_id,
        template_id=str(row["template_id"]),
        template_name=str(row["template_name"]),
        family_id=str(row["family_id"]),
        canonical_subitem_id=subitem_id,
        intended_facet_id=facet_id,
        variant_semantic_role=variant_role,
        subitem_assignment_source="template_fixed"
        if row.get("realization_mode") == "deterministic"
        else "planner_selected",
        source_kind=str(row["realization_mode"]),
        realization_mode=str(row["realization_mode"]),
        gate_priority=str(row["gate_priority"]),
        extended_family=bool(row.get("extended_family")),
        question=_question_text(row=row, subitem_id=subitem_id, bindings=query_bindings, variant_role=variant_role),
        bindings=query_bindings,
        binding_roles=list(row.get("binding_roles") or []),
        coverage_target_min="enumerate_all_applicable" if row.get("realization_mode") == "deterministic" else "5",
        runtime_sql_skeleton=str(row.get("sql_skeleton") or ""),
        notes=notes,
        template_selection_mode=template_selection_mode,
        selected_template_rank=selected_template_rank,
        problem_index_within_template=problem_index + 1,
        sql_variant_index=variant_index + 1,
        sql_variant_total=variant_total,
    )


def _template_binding_possible(row: dict[str, Any], profile: DatasetRoleProfile) -> bool:
    for index in range(AGENT_PROBLEMS_PER_TEMPLATE_MAX):
        if _binding_from_template(row, profile, index=index) is not None:
            return True
    return False


def _candidate_template_summary(row: dict[str, Any], profile: DatasetRoleProfile) -> dict[str, Any]:
    return {
        "template_id": row["template_id"],
        "template_name": row["template_name"],
        "family_id": row["family_id"],
        "gate_priority": row["gate_priority"],
        "binding_roles": list(row.get("binding_roles") or []),
        "supported_canonical_subitem_ids": list(row.get("supported_canonical_subitem_ids") or []),
        "allowed_variant_roles": list(row.get("allowed_variant_roles") or []),
        "dataset_fit": {
            "has_groupable_cols": bool(profile.groupable_cols),
            "has_numeric_cols": bool(profile.numeric_cols),
            "has_condition_cols": bool(profile.condition_cols),
            "has_temporal_cols": bool(profile.temporal_cols),
            "has_high_card_cols": bool(profile.high_card_cols),
        },
    }


def _specialized_template_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _template_priority_rank(row),
        len(row.get("supported_canonical_subitem_ids") or []),
        len(row.get("binding_roles") or []),
        str(row.get("template_id")),
    )


def _fill_template_ids_for_family(
    *,
    family_id: str,
    minimum: int,
    candidate_rows: list[dict[str, Any]],
    selected_ids: list[str],
) -> None:
    family_count = sum(1 for template_id in selected_ids if next(row for row in candidate_rows if row["template_id"] == template_id)["family_id"] == family_id)
    if family_count >= minimum:
        return
    family_rows = [row for row in candidate_rows if str(row.get("family_id")) == family_id]
    family_rows.sort(key=_specialized_template_key)
    for row in family_rows:
        if row["template_id"] in selected_ids:
            continue
        selected_ids.append(str(row["template_id"]))
        family_count += 1
        if family_count >= minimum:
            return


def _rule_selected_template_ids(
    *,
    candidate_rows: list[dict[str, Any]],
    min_templates: int,
    target_templates: int,
) -> list[str]:
    row_lookup = {str(row["template_id"]): row for row in candidate_rows}
    selected_ids: list[str] = []

    for subitem_id in CORE_AGENT_SUBITEMS:
        candidates = [
            row
            for row in candidate_rows
            if subitem_id in (row.get("supported_canonical_subitem_ids") or [])
        ]
        candidates.sort(key=_specialized_template_key)
        for row in candidates:
            template_id = str(row["template_id"])
            if template_id in selected_ids:
                continue
            selected_ids.append(template_id)
            break

    for family_id, minimum in AGENT_FAMILY_TEMPLATE_MINIMUMS.items():
        _fill_template_ids_for_family(
            family_id=family_id,
            minimum=minimum,
            candidate_rows=candidate_rows,
            selected_ids=selected_ids,
        )

    effective_target = max(min_templates, min(target_templates, len(candidate_rows)))
    remaining_rows = [row_lookup[template_id] for template_id in row_lookup if template_id not in selected_ids]
    remaining_rows.sort(
        key=lambda row: (
            _template_priority_rank(row),
            -len(row.get("supported_canonical_subitem_ids") or []),
            len(row.get("binding_roles") or []),
            str(row.get("template_id")),
        )
    )
    for row in remaining_rows:
        if len(selected_ids) >= effective_target:
            break
        selected_ids.append(str(row["template_id"]))
    return selected_ids


def _select_agent_templates(
    *,
    dataset_id: str,
    profile: DatasetRoleProfile,
    planner_kind: str,
    planner_model: str,
    ai_cli_preset: str,
    ai_cli_command: str,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    applicable_rows = [row for row in _agent_template_rows() if _template_binding_possible(row, profile)]
    deficits: list[dict[str, Any]] = []
    planner_usage_summary: dict[str, Any] = {
        "planner_kind": planner_kind,
        "model": planner_model if planner_kind == "cli" else "",
        "calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "usage_source": "none" if planner_kind == "rule" else "ai_cli_json_usage",
    }
    if not applicable_rows:
        return [], {}, [
            {
                "dataset_id": dataset_id,
                "reason": "no_applicable_agent_templates",
            }
        ], planner_usage_summary

    min_templates = min(AGENT_TEMPLATE_MIN, len(applicable_rows))
    target_templates = max(min_templates, min(AGENT_TEMPLATE_TARGET, len(applicable_rows)))
    rule_selected_ids = _rule_selected_template_ids(
        candidate_rows=applicable_rows,
        min_templates=min_templates,
        target_templates=target_templates,
    )

    selection_modes: dict[str, str] = {}
    selected_ids: list[str] = []
    row_lookup = {str(row["template_id"]): row for row in applicable_rows}

    if planner_kind == "cli":
        try:
            from src.workload_grounding.problem_planner import CLIProblemPlanner

            planner = CLIProblemPlanner(
                model_name=planner_model,
                dataset_id=dataset_id,
                run_id=f"v2_inventory_{dataset_id}",
                project_root=Path.cwd(),
                ai_cli_preset=ai_cli_preset,
                ai_cli_command=ai_cli_command,
            )
            planner_selected_ids = planner.select_templates(
                dataset_id=dataset_id,
                dataset_summary=profile.summary(),
                candidates=[_candidate_template_summary(row, profile) for row in applicable_rows],
                min_templates=min_templates,
                target_templates=target_templates,
            )
            planner_usage_summary = {
                **planner_usage_summary,
                **dict(planner.summary),
                "usage_source": "ai_cli_json_usage",
            }
            for template_id in planner_selected_ids:
                if template_id not in row_lookup or template_id in selected_ids:
                    continue
                selected_ids.append(template_id)
                selection_modes[template_id] = "cli"
                if len(selected_ids) >= target_templates:
                    break
        except Exception as exc:  # noqa: BLE001
            deficits.append(
                {
                    "dataset_id": dataset_id,
                    "reason": "agent_template_selection_failed",
                    "planner_kind": planner_kind,
                    "error": str(exc),
                }
            )

    for template_id in rule_selected_ids:
        if len(selected_ids) >= target_templates:
            break
        if template_id in selected_ids:
            continue
        selected_ids.append(template_id)
        selection_modes[template_id] = "rule" if planner_kind == "rule" else "rule_backfill"

    if len(selected_ids) < AGENT_TEMPLATE_MIN:
        deficits.append(
            {
                "dataset_id": dataset_id,
                "reason": "insufficient_agent_templates_for_minimum",
                "available_agent_template_count": len(applicable_rows),
                "selected_agent_template_count": len(selected_ids),
                "required_agent_template_count": AGENT_TEMPLATE_MIN,
            }
        )

    selected_rows = [row_lookup[template_id] for template_id in selected_ids]
    return selected_rows, selection_modes, deficits, planner_usage_summary


def _problem_count_for_template(row: dict[str, Any], profile: DatasetRoleProfile) -> int:
    count = AGENT_PROBLEMS_PER_TEMPLATE_MIN
    if str(row.get("gate_priority")) == "primary":
        count += 2
    elif str(row.get("gate_priority")) == "support":
        count += 1
    if len(row.get("supported_canonical_subitem_ids") or []) == 1:
        count += 1
    if len(row.get("allowed_variant_roles") or []) > 1:
        count += 1
    if len(row.get("binding_roles") or []) >= 3:
        count += 1
    if len(profile.groupable_cols) >= 6:
        count += 1
    if len(profile.numeric_cols) >= 4 and any(
        role in {"measure_col", "band_col"} for role in (row.get("binding_roles") or [])
    ):
        count += 1
    if len(profile.condition_cols) >= 4 and any(
        role in {"condition_col", "target_col"} for role in (row.get("binding_roles") or [])
    ):
        count += 1
    return min(AGENT_PROBLEMS_PER_TEMPLATE_MAX, max(AGENT_PROBLEMS_PER_TEMPLATE_MIN, count))


def _variant_count_for_problem(row: dict[str, Any], subitem_id: str, problem_index: int) -> int:
    role_count = len(row.get("allowed_variant_roles") or [])
    facet_count = len(default_facet_ids_for_subitem(subitem_id))
    if role_count <= 1 and facet_count <= 1:
        return 1
    if str(row.get("gate_priority")) == "primary":
        return 2
    return 2 if problem_index % 2 == 0 else 1


def _variantized_bindings(
    *,
    base_bindings: dict[str, Any],
    row: dict[str, Any],
    profile: DatasetRoleProfile,
    problem_index: int,
    variant_index: int,
) -> dict[str, Any]:
    bindings = dict(base_bindings)
    if variant_index == 0:
        return bindings

    if "top_k" in bindings:
        bindings["top_k"] = min(25, int(bindings["top_k"]) + 5)
    if "top_n" in bindings:
        bindings["top_n"] = min(10, int(bindings["top_n"]) + 1)
    if "percentile_value" in bindings:
        current = float(bindings["percentile_value"])
        bindings["percentile_value"] = 0.9 if current >= 0.95 else 0.95
    if "fraction_threshold" in bindings:
        bindings["fraction_threshold"] = round(max(0.05, float(bindings["fraction_threshold"]) / 2.0), 4)
    if "baseline_multiplier" in bindings:
        bindings["baseline_multiplier"] = round(float(bindings["baseline_multiplier"]) + 0.25, 4)
    if "min_support" in bindings:
        bindings["min_support"] = max(3, int(bindings["min_support"]) - 1)
    if "predicate_col" in bindings:
        predicate = _predicate_binding(profile, problem_index + variant_index + 3)
        if predicate is not None:
            bindings.update(predicate)
    if "condition_col" in bindings and "condition_value" in bindings:
        positive, negative = _condition_values(profile, str(bindings["condition_col"]))
        if positive != negative:
            current_value = bindings["condition_value"]
            bindings["condition_value"] = negative if current_value == positive else positive
            bindings["positive_value"] = positive
            bindings["negative_value"] = negative
    if "measure_col" in bindings and "measure_threshold" in bindings:
        stats = profile.field_stats.get(str(bindings["measure_col"]))
        if stats is not None and stats.q66 is not None:
            bindings["measure_threshold"] = round(float(stats.q66), 6)
    return bindings


def _agent_items_for_dataset(
    dataset_id: str,
    profile: DatasetRoleProfile,
    *,
    planner_kind: str,
    planner_model: str,
    ai_cli_preset: str,
    ai_cli_command: str,
) -> tuple[list[V2InventoryItem], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    items: list[V2InventoryItem] = []
    deficits: list[dict[str, Any]] = []
    selected_rows, selection_modes, selection_deficits, planner_usage_summary = _select_agent_templates(
        dataset_id=dataset_id,
        profile=profile,
        planner_kind=planner_kind,
        planner_model=planner_model,
        ai_cli_preset=ai_cli_preset,
        ai_cli_command=ai_cli_command,
    )
    deficits.extend(selection_deficits)

    template_summaries: list[dict[str, Any]] = []
    subitem_counts: dict[str, int] = {subitem_id: 0 for subitem_id in CORE_AGENT_SUBITEMS}

    for template_rank, row in enumerate(selected_rows, start=1):
        selection_mode = selection_modes.get(str(row["template_id"]), planner_kind)
        planned_problem_count = _problem_count_for_template(row, profile)
        template_summaries.append(
            {
                **dict(row),
                "selection_mode": selection_mode,
                "selected_template_rank": template_rank,
                "planned_problem_count": planned_problem_count,
                "sql_variant_policy": "1-2",
            }
        )

        supported_subitems = list(row.get("supported_canonical_subitem_ids") or [])
        role_options = list(row.get("allowed_variant_roles") or ["count_distribution"])
        if not supported_subitems:
            deficits.append(
                {
                    "dataset_id": dataset_id,
                    "template_id": row["template_id"],
                    "reason": "template_has_no_supported_subitems",
                }
            )
            continue

        for problem_index in range(planned_problem_count):
            binding_index = (template_rank - 1) * AGENT_PROBLEMS_PER_TEMPLATE_MAX + problem_index
            base_bindings = _binding_from_template(row, profile, index=binding_index)
            if base_bindings is None:
                deficits.append(
                    {
                        "dataset_id": dataset_id,
                        "template_id": row["template_id"],
                        "reason": "binding_generation_failed",
                        "problem_index_within_template": problem_index + 1,
                    }
                )
                continue
            subitem_id = supported_subitems[problem_index % len(supported_subitems)]
            facet_options = list(default_facet_ids_for_subitem(subitem_id)) or [f"{SUBITEM_TO_FAMILY[subitem_id]}_general"]
            variant_total = _variant_count_for_problem(row, subitem_id, problem_index)
            for variant_index in range(variant_total):
                variant_role = role_options[(problem_index + variant_index) % len(role_options)]
                facet_id = facet_options[(problem_index + variant_index) % len(facet_options)]
                query_bindings = _variantized_bindings(
                    base_bindings=base_bindings,
                    row=row,
                    profile=profile,
                    problem_index=binding_index,
                    variant_index=variant_index,
                )
                items.append(
                    _inventory_item(
                        dataset_id=dataset_id,
                        row=row,
                        subitem_id=subitem_id,
                        facet_id=facet_id,
                        variant_role=variant_role,
                        base_bindings=base_bindings,
                        query_bindings=query_bindings,
                        problem_index=problem_index,
                        variant_index=variant_index,
                        variant_total=variant_total,
                        selected_template_rank=template_rank,
                        template_selection_mode=selection_mode,
                        extra_notes=[f"binding_index={binding_index}"],
                    )
                )
                subitem_counts[subitem_id] = subitem_counts.get(subitem_id, 0) + 1

    for subitem_id in CORE_AGENT_SUBITEMS:
        if subitem_counts.get(subitem_id, 0) >= 5:
            continue
        deficits.append(
            {
                "dataset_id": dataset_id,
                "canonical_subitem_id": subitem_id,
                "reason": "planned_agent_sql_below_minimum",
                "planned_agent_sql_count": subitem_counts.get(subitem_id, 0),
                "required_agent_sql_count": 5,
            }
        )
    return items, deficits, template_summaries, planner_usage_summary


def _deterministic_items_for_dataset(dataset_id: str, profile: DatasetRoleProfile) -> list[V2InventoryItem]:
    items: list[V2InventoryItem] = []

    for row in _template_rows_for_subitem(
        subitem_id="marginal_missing_rate_consistency",
        realization_mode="deterministic",
    ):
        for index, missing_col in enumerate(profile.missing_cols):
            bindings = {"missing_col": missing_col}
            items.append(
                _inventory_item(
                    dataset_id=dataset_id,
                    row=row,
                    subitem_id="marginal_missing_rate_consistency",
                    facet_id=default_facet_ids_for_subitem("marginal_missing_rate_consistency")[0],
                    variant_role=list(row.get("allowed_variant_roles") or ["missing_indicator_view"])[0],
                    base_bindings=bindings,
                    query_bindings=bindings,
                    problem_index=index,
                    variant_index=0,
                    variant_total=1,
                    selected_template_rank=0,
                    template_selection_mode="deterministic",
                )
            )

    subgroup_templates = _template_rows_for_subitem(
        subitem_id="co_missingness_pattern_consistency",
        realization_mode="deterministic",
    )
    for row in subgroup_templates:
        if row["template_id"] == "tpl_missing_rate_by_subgroup":
            pairs = [
                (missing_col, group_col)
                for missing_col in profile.missing_cols
                for group_col in profile.groupable_cols[:10]
                if missing_col != group_col
            ]
            for index, (missing_col, group_col) in enumerate(pairs):
                bindings = {"missing_col": missing_col, "group_col": group_col}
                items.append(
                    _inventory_item(
                        dataset_id=dataset_id,
                        row=row,
                        subitem_id="co_missingness_pattern_consistency",
                        facet_id="missing_rate_by_subgroup",
                        variant_role="missing_rate_by_subgroup",
                        base_bindings=bindings,
                        query_bindings=bindings,
                        problem_index=index,
                        variant_index=0,
                        variant_total=1,
                        selected_template_rank=0,
                        template_selection_mode="deterministic",
                    )
                )
        elif row["template_id"] == "tpl_missing_target_interaction":
            context_seed = ([profile.target_column] if profile.target_column else []) + list(profile.condition_cols[:10])
            context_cols = _unique(context_seed)
            pairs = [
                (missing_col, target_col)
                for missing_col in profile.missing_cols
                for target_col in context_cols
                if target_col and missing_col != target_col
            ]
            for index, (missing_col, target_col) in enumerate(pairs):
                bindings = {"missing_col": missing_col, "target_col": target_col}
                items.append(
                    _inventory_item(
                        dataset_id=dataset_id,
                        row=row,
                        subitem_id="co_missingness_pattern_consistency",
                        facet_id="missing_target_interaction",
                        variant_role="missing_target_interaction",
                        base_bindings=bindings,
                        query_bindings=bindings,
                        problem_index=index,
                        variant_index=0,
                        variant_total=1,
                        selected_template_rank=0,
                        template_selection_mode="deterministic",
                    )
                )

    for row in _template_rows_for_subitem(
        subitem_id="support_rank_profile_consistency",
        realization_mode="deterministic",
    ):
        for index, group_col in enumerate(profile.groupable_cols):
            variant_role = list(row.get("allowed_variant_roles") or ["count_distribution"])[0]
            facet_options = list(default_facet_ids_for_subitem("support_rank_profile_consistency"))
            facet_id = facet_options[index % len(facet_options)]
            bindings = {"group_col": group_col}
            items.append(
                _inventory_item(
                    dataset_id=dataset_id,
                    row=row,
                    subitem_id="support_rank_profile_consistency",
                    facet_id=facet_id,
                    variant_role=variant_role,
                    base_bindings=bindings,
                    query_bindings=bindings,
                    problem_index=index,
                    variant_index=0,
                    variant_total=1,
                    selected_template_rank=0,
                    template_selection_mode="deterministic",
                )
            )

    for row in _template_rows_for_subitem(
        subitem_id="high_cardinality_response_stability",
        realization_mode="deterministic",
    ):
        combos = [
            (key_col, measure_col)
            for key_col in profile.high_card_cols[:20]
            for measure_col in profile.numeric_cols[:4]
            if key_col != measure_col
        ]
        for index, (key_col, measure_col) in enumerate(combos):
            bindings = {"key_col": key_col, "measure_col": measure_col, "min_support": 5}
            items.append(
                _inventory_item(
                    dataset_id=dataset_id,
                    row=row,
                    subitem_id="high_cardinality_response_stability",
                    facet_id=default_facet_ids_for_subitem("high_cardinality_response_stability")[0],
                    variant_role="focused_target_view",
                    base_bindings=bindings,
                    query_bindings=bindings,
                    problem_index=index,
                    variant_index=0,
                    variant_total=1,
                    selected_template_rank=0,
                    template_selection_mode="deterministic",
                )
            )

    return items


def build_dataset_inventory(
    dataset_id: str,
    *,
    line_version: str = "v2",
    data_root: Path = DATA_DIR,
    use_cache: bool = True,
    planner_kind: str = "rule",
    planner_model: str = "gpt-5.4",
    ai_cli_preset: str = "codex",
    ai_cli_command: str = "",
) -> dict[str, Any]:
    ensure_line_dirs(line_version)
    profile = load_dataset_role_profile(dataset_id, data_root=data_root, use_cache=use_cache)
    agent_items, deficits, selected_agent_templates, planner_usage_summary = _agent_items_for_dataset(
        dataset_id,
        profile,
        planner_kind=planner_kind,
        planner_model=planner_model,
        ai_cli_preset=ai_cli_preset,
        ai_cli_command=ai_cli_command,
    )
    deterministic_items = _deterministic_items_for_dataset(dataset_id, profile)
    all_items = agent_items + deterministic_items

    selected_deterministic_template_ids = _unique(item.template_id for item in deterministic_items)
    template_lookup = _template_rows_by_id()
    selected_deterministic_templates = [
        dict(template_lookup[template_id]) for template_id in selected_deterministic_template_ids
    ]
    selected_templates = selected_agent_templates + selected_deterministic_templates

    payload = {
        "dataset_id": dataset_id,
        "inventory_version": f"subitem_workload_{line_version}",
        "planner_kind": planner_kind,
        "line_version": line_version,
        "planner_usage_summary": planner_usage_summary,
        "role_profile_summary": profile.summary(),
        "selected_template_count": len(selected_templates),
        "selected_agent_template_count": len(selected_agent_templates),
        "selected_deterministic_template_count": len(selected_deterministic_templates),
        "problem_count": len(all_items),
        "agent_problem_count": len(agent_items),
        "deterministic_problem_count": len(deterministic_items),
        "coverage_policy": {
            "agent_template_families": list(CORE_AGENT_FAMILIES),
            "agent_selected_template_min": AGENT_TEMPLATE_MIN,
            "agent_selected_template_target": AGENT_TEMPLATE_TARGET,
            "agent_problem_count_per_template_min": AGENT_PROBLEMS_PER_TEMPLATE_MIN,
            "agent_problem_count_per_template_max": AGENT_PROBLEMS_PER_TEMPLATE_MAX,
            "agent_sql_variants_per_problem": "1-2",
            "agent_dataset_subitem_min_sql": 5,
            "deterministic_policy": "enumerate_all_applicable",
        },
        "selected_agent_templates": selected_agent_templates,
        "selected_deterministic_templates": selected_deterministic_templates,
        "selected_templates": selected_templates,
        "items": [asdict(item) for item in all_items],
        "deficits": deficits,
    }
    output_path = dataset_inventory_path(dataset_id, line_version=line_version)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def build_inventories_for_datasets(
    dataset_ids: list[str],
    *,
    line_version: str = "v2",
    data_root: Path = DATA_DIR,
    use_cache: bool = True,
    planner_kind: str = "rule",
    planner_model: str = "gpt-5.4",
    ai_cli_preset: str = "codex",
    ai_cli_command: str = "",
) -> dict[str, Any]:
    ensure_line_dirs(line_version)
    inventories = {
        dataset_id: build_dataset_inventory(
            dataset_id,
            line_version=line_version,
            data_root=data_root,
            use_cache=use_cache,
            planner_kind=planner_kind,
            planner_model=planner_model,
            ai_cli_preset=ai_cli_preset,
            ai_cli_command=ai_cli_command,
        )
        for dataset_id in dataset_ids
    }
    summary = {
        "inventory_version": f"subitem_workload_{line_version}",
        "line_version": line_version,
        "dataset_ids": dataset_ids,
        "planner_kind": planner_kind,
        "inventories": {
            dataset_id: {
                "path": str(dataset_inventory_path(dataset_id, line_version=line_version).resolve()),
                "problem_count": payload["problem_count"],
                "agent_problem_count": payload["agent_problem_count"],
                "deterministic_problem_count": payload["deterministic_problem_count"],
                "selected_template_count": payload["selected_template_count"],
                "selected_agent_template_count": payload["selected_agent_template_count"],
                "selected_deterministic_template_count": payload["selected_deterministic_template_count"],
                "deficit_count": len(payload["deficits"]),
            }
            for dataset_id, payload in inventories.items()
        },
    }
    combined_inventory_path(line_version=line_version).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
