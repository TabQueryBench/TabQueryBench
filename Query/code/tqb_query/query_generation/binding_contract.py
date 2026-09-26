"""Self-contained contracts for model-driven query-template binding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Mapping


PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
ALLOWED_OPERATORS = ("=", "!=", "<>", "<", "<=", ">", ">=", "LIKE")
COLUMN_ROLES = {
    "group_col", "group_col_2", "measure_col", "target_col", "condition_col",
    "missing_col", "key_col", "entity_col", "item_col", "time_col", "band_col",
    "predicate_col",
}

STATIC_PARAMETER_CANDIDATES: dict[str, list[Any]] = {
    "top_k": [5, 10, 15, 20],
    "top_n": [3, 5, 10],
    "num_tiles": [4, 5, 10],
    "percentile_value": [0.9, 0.95, 0.99],
    "z_threshold": [2.0, 2.5, 3.0],
    "fraction_threshold": [0.05, 0.1, 0.2],
    "baseline_multiplier": [1.25, 1.5, 2.0],
    "baseline_fraction": [0.05, 0.1, 0.2],
    "min_group_size": [3, 5, 10, 20],
    "min_support": [3, 5, 10, 20],
    "time_grain": ["day", "week", "month", "quarter", "year"],
    "lookback_rows": [1, 2, 3, 6, 11],
    "drift_ratio_threshold": [0.5, 0.8, 0.9, 1.0],
}


@dataclass(frozen=True)
class BindingValidationIssue:
    code: str
    role: str = ""
    detail: str = ""


@dataclass(frozen=True)
class BindingValidationResult:
    valid: bool
    bindings: dict[str, Any]
    issues: tuple[BindingValidationIssue, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "bindings": dict(self.bindings),
            "issues": [asdict(issue) for issue in self.issues],
        }


def _unique(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _observed_values(stats: Any) -> list[Any]:
    return _unique([value for value, _count in (stats.top_values or []) if value is not None])


def _numeric_candidates(stats: Any) -> list[float]:
    values = []
    for value in (stats.min_value, stats.q33, stats.q50, stats.q66, stats.q75, stats.max_value):
        if value is not None:
            values.append(round(float(value), 6))
    return _unique(values)


def _role_options(profile: Any) -> dict[str, list[str]]:
    groupable = list(profile.groupable_cols)
    high_or_groupable = list(profile.high_card_cols) or groupable
    target_options = [profile.target_column] if profile.target_column else []
    target_options.extend(name for name in profile.condition_cols if name not in target_options)
    return {
        "group_col": groupable,
        "group_col_2": groupable,
        "measure_col": list(profile.numeric_cols),
        "target_col": target_options,
        "condition_col": list(profile.condition_cols),
        "missing_col": list(profile.missing_cols),
        "key_col": list(profile.high_card_cols),
        "entity_col": high_or_groupable,
        "item_col": high_or_groupable,
        "time_col": list(profile.temporal_cols),
        "band_col": list(profile.numeric_cols),
        "predicate_col": list(profile.filterable_cols),
    }


def build_agent_binding_request(
    *, dataset_id: str, template_row: Mapping[str, Any], profile: Any, grounding_index: int
) -> dict[str, Any]:
    skeleton = str(template_row.get("sql_skeleton") or "").strip()
    placeholders = list(dict.fromkeys(PLACEHOLDER_RE.findall(skeleton)))
    agent_placeholders = [name for name in placeholders if name != "table"]
    role_options = _role_options(profile)
    schema = []
    value_candidates: dict[str, list[Any]] = {}
    numeric_candidates: dict[str, list[float]] = {}
    for name, stats in profile.field_stats.items():
        observed = _observed_values(stats)
        numeric = _numeric_candidates(stats) if stats.is_numeric else []
        value_candidates[name] = observed
        numeric_candidates[name] = numeric
        schema.append({
            "name": name,
            "declared_type": stats.declared_type,
            "semantic_type": stats.semantic_type,
            "field_role": stats.field_role,
            "field_tags": list(stats.field_tags),
            "is_numeric": bool(stats.is_numeric),
            "is_categorical": bool(stats.is_categorical),
            "distinct_count": int(stats.distinct_count),
            "top_values": [
                {"value": value, "count": int(count)}
                for value, count in (stats.top_values or []) if value is not None
            ],
            "min_value": stats.min_value,
            "max_value": stats.max_value,
            "q33": stats.q33,
            "q50": stats.q50,
            "q66": stats.q66,
            "q75": stats.q75,
            "eligible_roles": sorted(role for role, options in role_options.items() if name in options),
        })

    predicate_candidates = []
    for column in profile.filterable_cols:
        stats = profile.field_stats[column]
        if stats.is_numeric:
            predicate_candidates.extend(
                {"column": column, "operator": ">=", "value": value}
                for value in _numeric_candidates(stats)
            )
        else:
            predicate_candidates.extend(
                {"column": column, "operator": "=", "value": value}
                for value in _observed_values(stats)
            )

    return {
        "contract_version": "agent_template_binding_v10",
        "case_id": f"{dataset_id}::{template_row.get('template_id')}::{grounding_index}",
        "dataset": {
            "dataset_id": dataset_id,
            "table_name": profile.sqlite_result.table_name,
            "row_count": int(profile.row_count),
            "target_column": profile.target_column,
            "schema": schema,
        },
        "template": {
            "template_id": template_row.get("template_id"),
            "template_name": template_row.get("template_name"),
            "intent": template_row.get("intent"),
            "sql_skeleton": skeleton,
            "required_roles": list(template_row.get("required_roles") or []),
            "constraints": list(template_row.get("constraints") or []),
            "role_constraints": dict(template_row.get("role_constraints") or {}),
        },
        "allowed_bindings": {
            "column_roles": {
                role: role_options.get(role, [])
                for role in agent_placeholders if role in COLUMN_ROLES
            },
            "observed_values_by_column": value_candidates,
            "numeric_candidates_by_column": numeric_candidates,
            "predicate_candidates": predicate_candidates,
            "parameter_candidates": {
                role: STATIC_PARAMETER_CANDIDATES[role]
                for role in agent_placeholders if role in STATIC_PARAMETER_CANDIDATES
            },
            "allowed_operators": list(ALLOWED_OPERATORS),
        },
        "output_contract": {
            "expected_binding_keys": agent_placeholders,
            "json_shape": {"bindings": {role: "selected value" for role in agent_placeholders}},
        },
    }


def _contains(values: list[Any], candidate: Any) -> bool:
    key = json.dumps(candidate, sort_keys=True, ensure_ascii=False, default=str)
    return any(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str) == key
        for value in values
    )


def validate_agent_bindings(raw_bindings: Any, request: Mapping[str, Any]) -> BindingValidationResult:
    issues: list[BindingValidationIssue] = []
    if not isinstance(raw_bindings, dict):
        return BindingValidationResult(False, {}, (BindingValidationIssue("bindings_not_object"),))
    bindings = dict(raw_bindings)
    expected = list((request.get("output_contract") or {}).get("expected_binding_keys") or [])
    expected_set = set(expected)
    for role in sorted(expected_set - set(bindings)):
        issues.append(BindingValidationIssue("missing_required_placeholder", role))
    for role in sorted(set(bindings) - expected_set):
        issues.append(BindingValidationIssue("unknown_binding_key", role))

    allowed = request.get("allowed_bindings") or {}
    column_roles = allowed.get("column_roles") or {}
    for role, options in column_roles.items():
        if role in bindings and bindings[role] not in options:
            issues.append(BindingValidationIssue("column_not_allowed_for_role", role, str(bindings[role])))

    operators = list(allowed.get("allowed_operators") or [])
    if "predicate_op" in bindings and bindings["predicate_op"] not in operators:
        issues.append(BindingValidationIssue("operator_not_allowed", "predicate_op", str(bindings["predicate_op"])))
    if all(key in bindings for key in ("predicate_col", "predicate_op", "predicate_value")):
        candidate = {
            "column": bindings["predicate_col"],
            "operator": bindings["predicate_op"],
            "value": bindings["predicate_value"],
        }
        if candidate not in (allowed.get("predicate_candidates") or []):
            issues.append(BindingValidationIssue("predicate_not_in_candidates", "predicate_value"))

    observed = allowed.get("observed_values_by_column") or {}
    for value_role, column_role in (
        ("condition_value", "condition_col"),
        ("positive_value", "condition_col"),
        ("negative_value", "condition_col"),
        ("target_value", "target_col"),
    ):
        if value_role in bindings and column_role in bindings:
            candidates = list(observed.get(str(bindings[column_role])) or [])
            if not _contains(candidates, bindings[value_role]):
                issues.append(BindingValidationIssue("value_not_observed", value_role, str(bindings[value_role])))

    numeric_by_column = allowed.get("numeric_candidates_by_column") or {}
    for value_role, column_role in (
        ("band_cut_1", "band_col"),
        ("band_cut_2", "band_col"),
        ("lower_bound", "band_col"),
        ("upper_bound", "band_col"),
        ("measure_threshold", "measure_col"),
    ):
        if value_role in bindings and column_role in bindings:
            candidates = list(numeric_by_column.get(str(bindings[column_role])) or [])
            if not _contains(candidates, bindings[value_role]):
                issues.append(BindingValidationIssue("numeric_value_not_in_candidates", value_role, str(bindings[value_role])))

    for role, candidates in (allowed.get("parameter_candidates") or {}).items():
        if role in bindings and not _contains(list(candidates), bindings[role]):
            issues.append(BindingValidationIssue("parameter_not_in_candidates", role, str(bindings[role])))

    constraints = (request.get("template") or {}).get("role_constraints") or {}
    for pair in constraints.get("distinct_roles") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            left, right = str(pair[0]), str(pair[1])
            if left in bindings and right in bindings and bindings[left] == bindings[right]:
                issues.append(BindingValidationIssue("distinct_roles_violation", f"{left},{right}"))
    if constraints.get("forbid_distinct_self_count"):
        entity = bindings.get("entity_col")
        if entity is not None and entity in {bindings.get("group_col"), bindings.get("group_col_2")}:
            issues.append(BindingValidationIssue("forbid_distinct_self_count", "entity_col"))

    return BindingValidationResult(not issues, bindings, tuple(issues))
