"""Inventory builder for the isolated v2 workload line."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from tqb_query.benchmark.canonical_sql import stable_hash
from tqb_query.config.settings import DATA_DIR, PROJECT_ROOT
from tqb_query.logging.run_artifacts import RunArtifactWriter
from tqb_query.workload_grounding.agent_binding import build_agent_binding_request, validate_agent_bindings
from tqb_query.workload_grounding.v10_versions import V10ModelVersion, resolve_model_version

from .catalog import build_template_library_rows
from .contract_spec import (
    CORE_AGENT_SUBITEMS,
    DETERMINISTIC_SUBITEMS,
    SUBITEM_TO_FAMILY,
    default_facet_ids_for_subitem,
)
from .dataset_profile import DatasetRoleProfile, load_dataset_role_profile
from .paths import combined_inventory_path, dataset_inventory_path, ensure_line_dirs, line_version_family, logs_root


PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
AGENT_TEMPLATE_MIN = 10
AGENT_TEMPLATE_TARGET = 12
AGENT_PROBLEMS_PER_TEMPLATE_MIN = 4
AGENT_PROBLEMS_PER_TEMPLATE_MAX = 12
V5_AGENT_PROBLEMS_PER_TEMPLATE = 1
DETERMINISTIC_PROBLEMS_PER_TEMPLATE_MAX = 12
AGENT_FAMILY_TEMPLATE_MINIMUMS: dict[str, int] = {
    "subgroup_structure": 2,
    "conditional_dependency_structure": 4,
    "tail_rarity_structure": 4,
}
CORE_AGENT_FAMILIES = tuple(AGENT_FAMILY_TEMPLATE_MINIMUMS.keys())
TEMPLATE_PRIORITY_ORDER = {"primary": 0, "support": 1, "review": 2, "deterministic": 3}
ALL_APPLICABLE_MINIMAL_LINE_VERSIONS = {"v5", "v6", "v7"}
ALL_APPLICABLE_DENSE_LINE_VERSIONS = {"v9", "v10"}
# V11: the model picks AGENT_TEMPLATE_MIN..AGENT_TEMPLATE_TARGET templates from all applicable ones.
AGENT_SELECTED_LINE_VERSIONS = {"v11"}
AGENT_PLANNER_LINE_FAMILIES = {"agent-bind": "v10", "agent-select-bind": "v11"}
V7_ROLE_OVERLAP_COMPATIBILITY = {
    ("c17", "tpl_m4_binned_numeric_group_avg"),
    ("c17", "tpl_tpch_filtered_sum_band"),
    ("c3", "tpl_clickbench_filtered_distinct_topk"),
}


def selection_policy_for_line_version(line_version: str) -> str:
    """Return the explicit template-coverage policy for a workload version."""
    line_version = line_version_family(line_version)
    if line_version in ALL_APPLICABLE_MINIMAL_LINE_VERSIONS:
        return "all_applicable_minimal"
    if line_version in ALL_APPLICABLE_DENSE_LINE_VERSIONS:
        return "all_applicable_dense"
    if line_version in AGENT_SELECTED_LINE_VERSIONS:
        return "agent_selected_rule_count"
    return "targeted_subitem_coverage"


def _resolve_agent_line_version(
    *,
    planner_kind: str,
    planner_model: str,
    line_version: str,
    grounding_version: str,
) -> V10ModelVersion:
    family = AGENT_PLANNER_LINE_FAMILIES[planner_kind]
    if line_version_family(line_version) != family:
        raise ValueError(f"planner_kind={planner_kind} requires line_version {family} or {family}.x.y")
    requested = grounding_version or (line_version if line_version != family else "")
    return resolve_model_version(planner_model, line_family=family, grounding_version=requested)


@lru_cache(maxsize=None)
def _v7_reference_template_ids(dataset_id: str) -> frozenset[str]:
    registry_paths = sorted(
        (PROJECT_ROOT.parent / "Queries" / "V7-gpt-5.4-mini-full" / dataset_id / "sql" / "grounding" / "registries").glob(
            "*_query_registry_v7.jsonl"
        )
    )
    template_ids: set[str] = set()
    for path in registry_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("source_kind")) == "agent" and bool(row.get("accepted_for_eval")):
                    template_ids.add(str(row["template_id"]))
    if not template_ids:
        raise FileNotFoundError(f"V7 applicability reference registry not found for dataset {dataset_id}")
    return frozenset(template_ids)


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
    semantic_result_contract: dict[str, Any] | None = None
    role_constraints: dict[str, Any] | None = None
    runtime_sql_skeleton: str | None = None
    notes: list[str] | None = None
    template_selection_mode: str = ""
    selected_template_rank: int = 0
    problem_index_within_template: int = 0
    sql_variant_index: int = 1
    sql_variant_total: int = 1
    raw_agent_bindings: dict[str, Any] | None = None
    binding_validation: dict[str, Any] | None = None
    binding_request_case_id: str = ""
    model_provenance: dict[str, Any] | None = None


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _cap_ordered[T](values: list[T], limit: int = DETERMINISTIC_PROBLEMS_PER_TEMPLATE_MAX) -> list[T]:
    if limit <= 0 or len(values) <= limit:
        return list(values)
    n = len(values)
    indices = []
    seen: set[int] = set()
    for i in range(limit):
        idx = (i * n) // limit
        if idx >= n:
            idx = n - 1
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    if len(indices) < limit:
        for idx in range(n):
            if idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
            if len(indices) >= limit:
                break
    return [values[idx] for idx in indices]


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


def _choose_group_pair(
    profile: DatasetRoleProfile,
    index: int,
    *,
    avoid: set[str] | None = None,
) -> tuple[str | None, str | None]:
    avoid = avoid or set()
    candidates = [column for column in profile.groupable_cols if column not in avoid]
    pairs = list(combinations(candidates, 2))
    if not pairs:
        first = _choose_column(candidates, index)
        return first, None
    first, second = pairs[index % len(pairs)]
    return first, second


def _predicate_binding(
    profile: DatasetRoleProfile,
    index: int,
    *,
    avoid: set[str] | None = None,
) -> dict[str, Any] | None:
    avoid = avoid or set()
    candidates = [column for column in profile.filterable_cols if column not in avoid]
    if not candidates:
        return None
    col = candidates[index % len(candidates)]
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


def _validate_template_binding(row: dict[str, Any], bindings: dict[str, Any]) -> bool:
    constraints = row.get("role_constraints") or {}
    if not isinstance(constraints, dict):
        return True

    for role_pair in constraints.get("distinct_roles") or []:
        if not isinstance(role_pair, (list, tuple)) or len(role_pair) != 2:
            continue
        left, right = str(role_pair[0]), str(role_pair[1])
        if left in bindings and right in bindings and str(bindings[left]) == str(bindings[right]):
            return False

    if constraints.get("forbid_distinct_self_count"):
        group_cols = {str(bindings[key]) for key in ("group_col", "group_col_2") if key in bindings}
        entity_col = str(bindings.get("entity_col") or "")
        if entity_col and entity_col in group_cols:
            return False

    return True


def _binding_from_template(
    row: dict[str, Any],
    profile: DatasetRoleProfile,
    *,
    index: int,
) -> dict[str, Any] | None:
    placeholders = set(PLACEHOLDER_RE.findall(str(row.get("sql_skeleton") or "")))
    bindings: dict[str, Any] = {}
    template_id = str(row.get("template_id") or "")
    allow_v7_role_overlap = (profile.dataset_id, template_id) in V7_ROLE_OVERLAP_COMPATIBILITY

    # Distinct-entity templates need to reserve a high-cardinality column for
    # ``entity_col``.  Choosing it as the grouping column first made otherwise
    # valid V7 combinations (notably c13 and n15) appear inapplicable.
    reserved_entity_col = (
        _choose_column(list(profile.high_card_cols), index)
        if "entity_col" in placeholders
        else None
    )
    entity_reserve = {reserved_entity_col} if reserved_entity_col else set()
    group_col = _choose_column(list(profile.groupable_cols), index, avoid=entity_reserve)
    pair_avoid = set(entity_reserve)
    if "target_col" in placeholders and profile.target_column:
        pair_avoid.add(profile.target_column)
    group_pair = _choose_group_pair(profile, index, avoid=pair_avoid)
    active_group_cols = (
        tuple(value for value in group_pair if value)
        if "group_col_2" in placeholders
        else tuple(value for value in (group_col,) if value)
    )
    band_col = _choose_column(list(profile.numeric_cols), index) if "band_col" in placeholders else None
    avoid_for_measure = set(active_group_cols)
    if band_col:
        avoid_for_measure.add(band_col)
    measure_col = _choose_column(list(profile.numeric_cols), index, avoid=avoid_for_measure)
    if measure_col is None and allow_v7_role_overlap and band_col is not None:
        measure_col = band_col
    target_col = profile.target_column or _choose_column(list(profile.condition_cols), index)
    condition_col = _choose_column(list(profile.condition_cols), index, avoid=set(active_group_cols))
    missing_col = _choose_column(list(profile.missing_cols), index)
    key_col = _choose_column(list(profile.high_card_cols), index, avoid={measure_col or "", target_col or ""})
    entity_col = reserved_entity_col or _choose_column(
        list(profile.high_card_cols or profile.groupable_cols),
        index,
        avoid=set(group_pair),
    )
    item_col = _choose_column(list(profile.high_card_cols or profile.groupable_cols), index + 1)
    time_col = _choose_column(list(profile.temporal_cols), index)

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
    if "predicate_col" in placeholders:
        predicate_avoid: set[str] = set()
        constraints = row.get("role_constraints") or {}
        for role_pair in constraints.get("distinct_roles") or []:
            if not isinstance(role_pair, (list, tuple)) or len(role_pair) != 2:
                continue
            left, right = str(role_pair[0]), str(role_pair[1])
            if left == "predicate_col" and right in bindings:
                predicate_avoid.add(str(bindings[right]))
            elif right == "predicate_col" and left in bindings:
                predicate_avoid.add(str(bindings[left]))
        predicate = _predicate_binding(profile, index, avoid=predicate_avoid)
        if predicate is None and allow_v7_role_overlap:
            predicate = _predicate_binding(profile, index)
        if not predicate:
            return None
        bindings.update(predicate)

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
    if not allow_v7_role_overlap and not _validate_template_binding(row, bindings):
        return None
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
    coverage_target_min: str | None = None,
    extra_notes: list[str] | None = None,
    source_kind: str | None = None,
    realization_mode: str | None = None,
    raw_agent_bindings: dict[str, Any] | None = None,
    binding_validation: dict[str, Any] | None = None,
    binding_request_case_id: str = "",
    model_provenance: dict[str, Any] | None = None,
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
        source_kind=source_kind or str(row["realization_mode"]),
        realization_mode=realization_mode or str(row["realization_mode"]),
        gate_priority=str(row["gate_priority"]),
        extended_family=bool(row.get("extended_family")),
        question=_question_text(row=row, subitem_id=subitem_id, bindings=query_bindings, variant_role=variant_role),
        bindings=query_bindings,
        binding_roles=list(row.get("binding_roles") or []),
        semantic_result_contract=dict(row.get("semantic_result_contract") or {}),
        role_constraints=dict(row.get("role_constraints") or {}),
        coverage_target_min=coverage_target_min
        or ("enumerate_all_applicable" if row.get("realization_mode") == "deterministic" else "5"),
        runtime_sql_skeleton=str(row.get("sql_skeleton") or ""),
        notes=notes,
        template_selection_mode=template_selection_mode,
        selected_template_rank=selected_template_rank,
        problem_index_within_template=problem_index + 1,
        sql_variant_index=variant_index + 1,
        sql_variant_total=variant_total,
        raw_agent_bindings=raw_agent_bindings,
        binding_validation=binding_validation,
        binding_request_case_id=binding_request_case_id,
        model_provenance=model_provenance,
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
    selection_policy: str = "targeted_subitem_coverage",
    agent_planner: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    applicable_rows = [row for row in _agent_template_rows() if _template_binding_possible(row, profile)]
    if selection_policy == "all_applicable_dense":
        v7_template_ids = _v7_reference_template_ids(dataset_id)
        applicable_rows = [row for row in applicable_rows if str(row["template_id"]) in v7_template_ids]
    deficits: list[dict[str, Any]] = []
    planner_usage_summary: dict[str, Any] = {
        "planner_kind": planner_kind,
        "selection_policy": selection_policy,
        "model": planner_model if planner_kind == "cli" or planner_kind in AGENT_PLANNER_LINE_FAMILIES else "",
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

    if selection_policy in {"all_applicable_minimal", "all_applicable_dense"}:
        selection_modes = {str(row["template_id"]): selection_policy for row in applicable_rows}
        planner_usage_summary["target_templates"] = len(applicable_rows)
        planner_usage_summary["selected_agent_template_count"] = len(applicable_rows)
        planner_usage_summary["applicable_agent_template_count"] = len(applicable_rows)
        return applicable_rows, selection_modes, deficits, planner_usage_summary

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

    if selection_policy == "agent_selected_rule_count":
        if agent_planner is None:
            raise ValueError("selection_policy=agent_selected_rule_count requires an agent planner")
        # Planner errors (quota, auth) propagate so --resume redoes the dataset instead of saving a partial inventory.
        raw_ids = agent_planner.select_templates_for_binding(
            dataset_id=dataset_id,
            dataset_summary=profile.summary(),
            candidates=[
                {
                    **_candidate_template_summary(row, profile),
                    "intent": row.get("intent", ""),
                    "sql_skeleton": row.get("sql_skeleton", ""),
                }
                for row in applicable_rows
            ],
            min_templates=min_templates,
            target_templates=target_templates,
            family_minimums=dict(AGENT_FAMILY_TEMPLATE_MINIMUMS),
        )
        invalid_ids: list[str] = []
        duplicate_ids: list[str] = []
        truncated_ids: list[str] = []
        for value in raw_ids:
            template_id = str(value)
            if template_id not in row_lookup:
                invalid_ids.append(template_id)
            elif template_id in selected_ids:
                duplicate_ids.append(template_id)
            elif len(selected_ids) >= target_templates:
                truncated_ids.append(template_id)
            else:
                selected_ids.append(template_id)
                selection_modes[template_id] = "agent_selected"
        agent_selected_ids = list(selected_ids)
        backfilled_ids: list[str] = []
        for template_id in rule_selected_ids:
            if len(selected_ids) >= min_templates:
                break
            if template_id in selected_ids:
                continue
            selected_ids.append(template_id)
            selection_modes[template_id] = "rule_backfill"
            backfilled_ids.append(template_id)
        if backfilled_ids:
            deficits.append(
                {
                    "dataset_id": dataset_id,
                    "reason": "agent_selected_templates_below_minimum",
                    "agent_selected_template_count": len(agent_selected_ids),
                    "required_agent_template_count": min_templates,
                    "rule_backfilled_template_ids": backfilled_ids,
                }
            )
        if len(applicable_rows) < AGENT_TEMPLATE_MIN:
            deficits.append(
                {
                    "dataset_id": dataset_id,
                    "reason": "insufficient_agent_templates_for_minimum",
                    "available_agent_template_count": len(applicable_rows),
                    "selected_agent_template_count": len(selected_ids),
                    "required_agent_template_count": AGENT_TEMPLATE_MIN,
                }
            )
        planner_usage_summary["applicable_agent_template_count"] = len(applicable_rows)
        planner_usage_summary["selected_agent_template_count"] = len(selected_ids)
        planner_usage_summary["template_selection"] = {
            "candidate_template_ids": list(row_lookup),
            "min_templates": min_templates,
            "target_templates": target_templates,
            "raw_selected_template_ids": [str(value) for value in raw_ids],
            "agent_selected_template_ids": agent_selected_ids,
            "invalid_template_ids": invalid_ids,
            "duplicate_template_ids": duplicate_ids,
            "truncated_template_ids": truncated_ids,
            "rule_backfilled_template_ids": backfilled_ids,
        }
        return [row_lookup[template_id] for template_id in selected_ids], selection_modes, deficits, planner_usage_summary

    if planner_kind == "cli":
        try:
            from tqb_query.workload_grounding.problem_planner import CLIProblemPlanner

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
    selection_policy: str = "targeted_subitem_coverage",
    grounding_version: str = "",
    agent_bind_problems_per_template: int = 1,
) -> tuple[list[V2InventoryItem], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    items: list[V2InventoryItem] = []
    deficits: list[dict[str, Any]] = []

    binding_planner = None
    model_provenance: dict[str, Any] | None = None
    if planner_kind in AGENT_PLANNER_LINE_FAMILIES:
        from tqb_query.workload_grounding.problem_planner import CLIProblemPlanner, ZAIProblemPlanner

        resolved = resolve_model_version(
            planner_model,
            line_family=AGENT_PLANNER_LINE_FAMILIES[planner_kind],
            grounding_version=grounding_version,
        )
        model_provenance = resolved.as_dict()
        effective_cli_preset = ai_cli_preset
        if effective_cli_preset == "auto":
            if resolved.resolved_model.startswith("claude-"):
                effective_cli_preset = "claude"
            elif resolved.resolved_model.startswith("glm-"):
                effective_cli_preset = "zai"
            else:
                effective_cli_preset = "codex"
        model_provenance["ai_cli_preset"] = effective_cli_preset
        planner_cls = ZAIProblemPlanner if effective_cli_preset == "zai" else CLIProblemPlanner
        binding_planner = planner_cls(
            model_name=resolved.resolved_model,
            dataset_id=dataset_id,
            run_id=f"{resolved.grounding_version}_inventory_{dataset_id}",
            project_root=Path.cwd(),
            ai_cli_preset=effective_cli_preset,
            ai_cli_command=ai_cli_command,
            artifact_writer=RunArtifactWriter(
                logs_root(resolved.grounding_version) / "planner_runs",
                f"{resolved.grounding_version}_inventory_{dataset_id}",
            ),
        )

    selected_rows, selection_modes, selection_deficits, planner_usage_summary = _select_agent_templates(
        dataset_id=dataset_id,
        profile=profile,
        planner_kind=planner_kind,
        planner_model=planner_model,
        ai_cli_preset=ai_cli_preset,
        ai_cli_command=ai_cli_command,
        selection_policy=selection_policy,
        agent_planner=binding_planner,
    )
    deficits.extend(selection_deficits)

    template_summaries: list[dict[str, Any]] = []
    subitem_counts: dict[str, int] = {subitem_id: 0 for subitem_id in CORE_AGENT_SUBITEMS}

    for template_rank, row in enumerate(selected_rows, start=1):
        selection_mode = selection_modes.get(str(row["template_id"]), planner_kind)
        if planner_kind in AGENT_PLANNER_LINE_FAMILIES:
            planned_problem_count = max(1, min(9, int(agent_bind_problems_per_template)))
        else:
            planned_problem_count = (
                V5_AGENT_PROBLEMS_PER_TEMPLATE
                if selection_policy == "all_applicable_minimal"
                else _problem_count_for_template(row, profile)
            )
        template_summaries.append(
            {
                **dict(row),
                "selection_mode": selection_mode,
                "selected_template_rank": template_rank,
                "planned_problem_count": planned_problem_count,
                "sql_variant_policy": "1"
                if planner_kind in AGENT_PLANNER_LINE_FAMILIES or selection_policy == "all_applicable_minimal"
                else "1-2",
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
            raw_agent_bindings: dict[str, Any] | None = None
            validation_payload: dict[str, Any] | None = None
            binding_request_case_id = ""
            if binding_planner is not None:
                request = build_agent_binding_request(
                    dataset_id=dataset_id,
                    template_row=row,
                    profile=profile,
                    grounding_index=binding_index,
                )
                binding_request_case_id = str(request["case_id"])
                try:
                    raw_agent_bindings = binding_planner.bind_template_placeholders(request)
                    validation = validate_agent_bindings(raw_agent_bindings, request)
                    validation_payload = validation.as_dict()
                    base_bindings = validation.bindings if validation.valid else None
                except Exception as exc:  # noqa: BLE001
                    from tqb_query.workload_grounding.problem_planner import PlannerQuotaError

                    if isinstance(exc, PlannerQuotaError):
                        raise
                    base_bindings = None
                    validation_payload = {
                        "valid": False,
                        "bindings": {},
                        "issues": [{"code": "agent_binding_call_failed", "role": "", "detail": str(exc)}],
                    }
            else:
                base_bindings = _binding_from_template(row, profile, index=binding_index)
            if base_bindings is None:
                deficits.append(
                    {
                        "dataset_id": dataset_id,
                        "template_id": row["template_id"],
                        "reason": "agent_binding_rejected" if binding_planner is not None else "binding_generation_failed",
                        "problem_index_within_template": problem_index + 1,
                        "binding_request_case_id": binding_request_case_id,
                        "binding_validation": validation_payload,
                    }
                )
                continue
            subitem_id = supported_subitems[problem_index % len(supported_subitems)]
            facet_options = list(default_facet_ids_for_subitem(subitem_id)) or [f"{SUBITEM_TO_FAMILY[subitem_id]}_general"]
            variant_total = 1 if binding_planner is not None or selection_policy == "all_applicable_minimal" else _variant_count_for_problem(row, subitem_id, problem_index)
            for variant_index in range(variant_total):
                variant_role = role_options[(problem_index + variant_index) % len(role_options)]
                facet_id = facet_options[(problem_index + variant_index) % len(facet_options)]
                query_bindings = dict(base_bindings) if binding_planner is not None else _variantized_bindings(
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
                        coverage_target_min="1" if selection_policy == "all_applicable_minimal" else None,
                        extra_notes=[f"binding_index={binding_index}"],
                        source_kind="agent_bind" if binding_planner is not None else None,
                        realization_mode="agent_bind" if binding_planner is not None else None,
                        raw_agent_bindings=raw_agent_bindings,
                        binding_validation=validation_payload,
                        binding_request_case_id=binding_request_case_id,
                        model_provenance=model_provenance,
                    )
                )
                subitem_counts[subitem_id] = subitem_counts.get(subitem_id, 0) + 1

    if selection_policy != "all_applicable_minimal":
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
    if binding_planner is not None:
        planner_usage_summary = {
            **planner_usage_summary,
            **dict(binding_planner.summary),
            "planner_kind": planner_kind,
            "grounding_version": grounding_version,
            "model_provenance": model_provenance,
            "usage_source": "ai_cli_json_usage",
        }
    return items, deficits, template_summaries, planner_usage_summary


def _deterministic_items_for_dataset(dataset_id: str, profile: DatasetRoleProfile) -> list[V2InventoryItem]:
    items: list[V2InventoryItem] = []

    for row in _template_rows_for_subitem(
        subitem_id="marginal_missing_rate_consistency",
        realization_mode="deterministic",
    ):
        for index, missing_col in enumerate(_cap_ordered(list(profile.missing_cols))):
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
            for index, (missing_col, group_col) in enumerate(_cap_ordered(pairs)):
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
            for index, (missing_col, target_col) in enumerate(_cap_ordered(pairs)):
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
        if str(row.get("template_id")) == "tpl_cardinality_continuous_range_envelope":
            for index, measure_col in enumerate(_cap_ordered(list(profile.continuous_numeric_cols))):
                bindings = {"measure_col": measure_col}
                items.append(
                    _inventory_item(
                        dataset_id=dataset_id,
                        row=row,
                        subitem_id="support_rank_profile_consistency",
                        facet_id="continuous_range_envelope",
                        variant_role="range_envelope_view",
                        base_bindings=bindings,
                        query_bindings=bindings,
                        problem_index=index,
                        variant_index=0,
                        variant_total=1,
                        selected_template_rank=0,
                        template_selection_mode="deterministic",
                    )
                )
            continue

        for index, group_col in enumerate(_cap_ordered(list(profile.groupable_cols))):
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
        for index, (key_col, measure_col) in enumerate(_cap_ordered(combos)):
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
    grounding_version: str = "",
    agent_bind_problems_per_template: int = 1,
) -> dict[str, Any]:
    if planner_kind not in {"rule", "cli", *AGENT_PLANNER_LINE_FAMILIES}:
        raise ValueError(f"Unsupported planner kind: {planner_kind}")
    model_provenance: dict[str, Any] | None = None
    if planner_kind in AGENT_PLANNER_LINE_FAMILIES:
        resolved = _resolve_agent_line_version(
            planner_kind=planner_kind,
            planner_model=planner_model,
            line_version=line_version,
            grounding_version=grounding_version,
        )
        model_provenance = resolved.as_dict()
        grounding_version = resolved.grounding_version
        line_version = grounding_version
    ensure_line_dirs(line_version)
    profile = load_dataset_role_profile(dataset_id, data_root=data_root, use_cache=use_cache)
    selection_policy = selection_policy_for_line_version(line_version)
    agent_items, deficits, selected_agent_templates, planner_usage_summary = _agent_items_for_dataset(
        dataset_id,
        profile,
        planner_kind=planner_kind,
        planner_model=planner_model,
        ai_cli_preset=ai_cli_preset,
        ai_cli_command=ai_cli_command,
        selection_policy=selection_policy,
        grounding_version=grounding_version,
        agent_bind_problems_per_template=agent_bind_problems_per_template,
    )
    deterministic_items = _deterministic_items_for_dataset(dataset_id, profile)
    all_items = agent_items + deterministic_items
    family = line_version_family(line_version)
    if family in AGENT_PLANNER_LINE_FAMILIES.values():
        all_items = [
            replace(
                item,
                query_record_id=item.query_record_id.replace("v2q_", f"{family}q_", 1),
                problem_id=item.problem_id.replace("v2p_", f"{family}p_", 1),
            )
            for item in all_items
        ]
        agent_items = all_items[: len(agent_items)]
        deterministic_items = all_items[len(agent_items) :]

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
        "line_version_family": line_version_family(line_version),
        "grounding_version": grounding_version,
        "model_provenance": model_provenance,
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
            "agent_selected_template_min": "all_applicable"
            if selection_policy in {"all_applicable_minimal", "all_applicable_dense"}
            else AGENT_TEMPLATE_MIN,
            "agent_selected_template_target": "all_applicable"
            if selection_policy in {"all_applicable_minimal", "all_applicable_dense"}
            else AGENT_TEMPLATE_TARGET,
            "agent_problem_count_per_template_min": V5_AGENT_PROBLEMS_PER_TEMPLATE
            if selection_policy == "all_applicable_minimal"
            else AGENT_PROBLEMS_PER_TEMPLATE_MIN,
            "agent_problem_count_per_template_max": V5_AGENT_PROBLEMS_PER_TEMPLATE
            if selection_policy == "all_applicable_minimal"
            else AGENT_PROBLEMS_PER_TEMPLATE_MAX,
            "agent_sql_variants_per_problem": "1" if selection_policy == "all_applicable_minimal" else "1-2",
            "agent_dataset_subitem_min_sql": f"not_enforced_for_template_coverage_{line_version}"
            if selection_policy == "all_applicable_minimal"
            else 5,
            "deterministic_problem_count_per_template_max": DETERMINISTIC_PROBLEMS_PER_TEMPLATE_MAX,
            "deterministic_policy": "capped_evenly_spaced_selection",
            "selection_policy": selection_policy,
            "v5_agent_problem_count_per_template": V5_AGENT_PROBLEMS_PER_TEMPLATE
            if selection_policy == "all_applicable_minimal"
            else "",
            "v6_agent_problem_count_per_template": V5_AGENT_PROBLEMS_PER_TEMPLATE
            if line_version == "v6" and selection_policy == "all_applicable_minimal"
            else "",
            "v7_agent_problem_count_per_template": V5_AGENT_PROBLEMS_PER_TEMPLATE
            if line_version == "v7" and selection_policy == "all_applicable_minimal"
            else "",
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
    grounding_version: str = "",
    agent_bind_problems_per_template: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    if planner_kind in AGENT_PLANNER_LINE_FAMILIES:
        resolved = _resolve_agent_line_version(
            planner_kind=planner_kind,
            planner_model=planner_model,
            line_version=line_version,
            grounding_version=grounding_version,
        )
        grounding_version = resolved.grounding_version
        line_version = grounding_version
    ensure_line_dirs(line_version)
    inventories: dict[str, dict[str, Any]] = {}
    for dataset_id in dataset_ids:
        existing_path = dataset_inventory_path(dataset_id, line_version=line_version)
        if resume and existing_path.exists():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if str(existing.get("grounding_version") or "") == grounding_version:
                inventories[dataset_id] = existing
                continue
        inventories[dataset_id] = build_dataset_inventory(
            dataset_id,
            line_version=line_version,
            data_root=data_root,
            use_cache=use_cache,
            planner_kind=planner_kind,
            planner_model=planner_model,
            ai_cli_preset=ai_cli_preset,
            ai_cli_command=ai_cli_command,
            grounding_version=grounding_version,
            agent_bind_problems_per_template=agent_bind_problems_per_template,
        )
    summary = {
        "inventory_version": f"subitem_workload_{line_version}",
        "line_version": line_version,
        "line_version_family": line_version_family(line_version),
        "grounding_version": grounding_version,
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


def rebuild_inventory_summary(
    dataset_ids: list[str],
    *,
    line_version: str = "v2",
    planner_kind: str = "rule",
) -> dict[str, Any]:
    """Rebuild the combined summary from existing per-dataset inventories."""
    inventories = {
        dataset_id: json.loads(
            dataset_inventory_path(dataset_id, line_version=line_version).read_text(encoding="utf-8")
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
