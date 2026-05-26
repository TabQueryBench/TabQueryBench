"""Static and operational dataset understanding builders."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.benchmark.models import FIVE_FIXED_FAMILIES, OperationalUnderstanding, ProbeResult, StaticDatasetUnderstanding
from src.data.bundle import DatasetBundle


def _family_status_map(family_applicability: dict[str, Any]) -> dict[str, str]:
    result = {family: "uncertain" for family in FIVE_FIXED_FAMILIES}
    families = family_applicability.get("families") or []
    for item in families:
        family_name = item.get("family_name")
        status = item.get("status")
        if isinstance(family_name, str) and isinstance(status, str) and family_name in result:
            result[family_name] = status
    return result


def build_static_understanding(bundle: DatasetBundle) -> StaticDatasetUnderstanding:
    semantics = bundle.dataset_semantics or {}
    contract = bundle.dataset_contract or {}
    profile = bundle.dataset_profile or {}
    field_registry = bundle.field_registry or {}
    query_policy = bundle.query_policy or {}
    validation_policy = bundle.validation_policy or {}

    dataset_name = semantics.get("dataset_name") or bundle.dataset_id
    task_type = semantics.get("task_type") or contract.get("task_type") or profile.get("task_type") or "unknown"
    row_semantics = semantics.get("row_semantics") or bundle.dataset_description
    target_column = (
        semantics.get("target_column")
        or contract.get("target_column")
        or profile.get("target_column")
        or ""
    )

    fields = field_registry.get("fields") or []
    field_roles: dict[str, str] = {}
    ordered_fields: dict[str, list[str]] = {}
    target_labels: list[str] = []
    key_fields: list[str] = []

    for field in fields:
        name = field.get("name")
        if not isinstance(name, str):
            continue
        field_roles[name] = str(field.get("role") or "")
        if field.get("ordered"):
            value_order = field.get("value_order") or []
            ordered_fields[name] = [str(v) for v in value_order if isinstance(v, (str, int, float))]
        if field.get("use_as_target"):
            target_labels = [str(v) for v in (field.get("value_order") or [])]
        tags = field.get("field_tags") or []
        if "subgroup_candidate" in tags or "condition_candidate" in tags:
            key_fields.append(name)

    if not target_labels and target_column in ordered_fields:
        target_labels = ordered_fields[target_column]

    family_summary = _family_status_map(bundle.family_applicability)
    policy_summary = {
        "preferred_families": query_policy.get("preferred_families") or [],
        "discouraged_families": query_policy.get("discouraged_families") or [],
        "useful_field_combinations": query_policy.get("useful_field_combinations") or [],
        "minimum_support_thresholds": validation_policy.get("minimum_support_thresholds") or {},
        "fields_requiring_caution": validation_policy.get("fields_requiring_caution") or [],
    }

    risk_summary = list((bundle.risk_register or {}).get("risks") or [])
    uncertainty_summary = list((bundle.uncertainty_register or {}).get("uncertainties") or [])

    if not key_fields and fields:
        key_fields = [f.get("name") for f in fields if isinstance(f.get("name"), str)][:6]

    return StaticDatasetUnderstanding(
        dataset_id=bundle.dataset_id,
        dataset_name=dataset_name,
        task_type=str(task_type),
        row_semantics=str(row_semantics),
        target_column=str(target_column),
        target_labels=target_labels,
        field_roles=field_roles,
        ordered_fields=ordered_fields,
        family_applicability_summary=family_summary,
        policy_summary=policy_summary,
        risk_summary=risk_summary,
        uncertainty_summary=uncertainty_summary,
        key_fields=key_fields,
    )


def build_operational_understanding(
    static_understanding: StaticDatasetUnderstanding,
    probes: list[ProbeResult],
) -> OperationalUnderstanding:
    status_weight = {
        "applicable": 1.0,
        "likely_applicable": 0.75,
        "uncertain": 0.5,
        "likely_not_applicable": 0.2,
    }

    family_scores = {
        family: status_weight.get(static_understanding.family_applicability_summary.get(family, "uncertain"), 0.5)
        for family in FIVE_FIXED_FAMILIES
    }

    low_support_signals: list[str] = []
    triviality_signals: list[str] = []
    promising_field_combinations: list[list[str]] = []
    notes: list[str] = []

    for probe in probes:
        if probe.error:
            notes.append(f"probe_error:{probe.probe_id}")
            continue

        if probe.probe_id == "target_distribution":
            for row in probe.rows:
                if len(row) < 3:
                    continue
                label = row[0]
                count = row[1]
                pct = row[2]
                try:
                    count_int = int(count)
                    pct_float = float(pct)
                except (TypeError, ValueError):
                    continue
                if count_int < 20 or pct_float < 5.0:
                    low_support_signals.append(f"target_tail:{label}:{count_int}")
                    family_scores["tail_rarity_structure"] += 0.2
                if pct_float > 80:
                    triviality_signals.append(f"target_dominant:{label}:{pct_float:.2f}")

        if probe.probe_type == "pair_target_support":
            support_values: list[int] = []
            for row in probe.rows:
                if len(row) < 4:
                    continue
                try:
                    support_values.append(int(row[3]))
                except (TypeError, ValueError):
                    continue
            if support_values:
                if max(support_values) >= 30:
                    meta = probe.description.replace("Pair-target support for ", "")
                    fields = [x.strip() for x in meta.split(" and ")]
                    if len(fields) >= 2:
                        promising_field_combinations.append(fields[:2] + [static_understanding.target_column])
                        family_scores["conditional_dependency_structure"] += 0.05
                        family_scores["subgroup_structure"] += 0.05
                if min(support_values) < 10:
                    low_support_signals.append(f"pair_low_support:{probe.probe_id}")

        if probe.probe_type == "ordered_values":
            if probe.row_count <= 1:
                triviality_signals.append(f"ordered_low_variation:{probe.probe_id}")

    if static_understanding.family_applicability_summary.get("missingness_structure") == "likely_not_applicable":
        notes.append("missingness_low_priority:no_missingness_detected")
    if static_understanding.family_applicability_summary.get("cardinality_structure") == "likely_not_applicable":
        notes.append("cardinality_low_priority:low_cardinality_dataset")

    for family in list(family_scores.keys()):
        family_scores[family] = round(max(0.0, min(1.5, family_scores[family])), 3)

    family_priority_order = sorted(FIVE_FIXED_FAMILIES, key=lambda family: family_scores.get(family, 0.0), reverse=True)

    unique_combinations: list[list[str]] = []
    seen = set()
    for combo in promising_field_combinations:
        key = tuple(combo)
        if key in seen:
            continue
        seen.add(key)
        unique_combinations.append(combo)

    return OperationalUnderstanding(
        dataset_id=static_understanding.dataset_id,
        family_scores=family_scores,
        family_priority_order=family_priority_order,
        promising_field_combinations=unique_combinations,
        low_support_signals=sorted(set(low_support_signals)),
        triviality_signals=sorted(set(triviality_signals)),
        notes=notes,
    )


def clone_operational_understanding(operational: OperationalUnderstanding) -> OperationalUnderstanding:
    payload = deepcopy(operational.to_dict())
    return OperationalUnderstanding(**payload)


def update_operational_with_validation_feedback(
    operational: OperationalUnderstanding,
    family: str,
    reason_codes: list[str],
) -> OperationalUnderstanding:
    updated = clone_operational_understanding(operational)

    fail_codes = [code for code in reason_codes if code.startswith("VAL_") or code.startswith("EXEC_")]
    if fail_codes:
        updated.family_scores[family] = round(max(0.0, updated.family_scores.get(family, 0.5) - 0.05), 3)
        updated.updates_from_validation.append(
            f"family={family}:deprioritize_due_to={'|'.join(sorted(set(fail_codes)))}"
        )

    if any(code == "VAL_SANITY_TRIVIAL" for code in reason_codes):
        updated.triviality_signals.append(f"validation_trivial:{family}")

    if any(code == "VAL_EXEC_LOW_SUPPORT" for code in reason_codes):
        updated.low_support_signals.append(f"validation_low_support:{family}")

    updated.family_priority_order = sorted(
        FIVE_FIXED_FAMILIES,
        key=lambda item: updated.family_scores.get(item, 0.0),
        reverse=True,
    )
    updated.low_support_signals = sorted(set(updated.low_support_signals))
    updated.triviality_signals = sorted(set(updated.triviality_signals))
    return updated
