"""Build template candidate pools and dataset-level problem inventories."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.benchmark.canonical_sql import stable_hash
from src.config.settings import DATA_DIR
from src.data.bundle import load_dataset_bundle
from src.db.csv_sqlite import materialize_dataset_to_sqlite
from src.workload_grounding.queryset_builder import FieldStats, build_field_stats
from src.usage.logger import UsageCSVLogger

if TYPE_CHECKING:
    from src.logging.run_artifacts import RunArtifactWriter
    from src.workload_grounding.problem_planner import CLIProblemPlanner, LLMProblemPlanner


POSITIVE_VALUE_HINTS = ["vgood", "good", "acc", "yes", "true", "1", "positive"]
NEGATIVE_VALUE_HINTS = ["unacc", "no", "false", "0", "negative"]

RATE_RATIO_IDS = {
    "tpl_m4_group_condition_rate",
    "tpl_m4_group_ratio_two_conditions",
    "tpl_conditional_group_quantiles",
}
ITEM_HEAVY_IDS = {
    "tpl_tpcds_within_group_share",
    "tpl_tpcds_subgroup_baseline_outlier",
    "tpl_tpcds_baseline_gated_extreme_ranking",
}
COUNT_AGGREGATE_FALLBACK_IDS = {
    "tpl_h2o_group_sum",
    "tpl_h2o_two_dimensional_group_sum",
    "tpl_tpcds_topk_group_sum",
    "tpl_tpcds_within_group_share",
    "tpl_tpch_relative_total_threshold",
    "tpl_tpch_max_aggregate_winner",
    "tpl_tpch_thresholded_group_ranking",
    "tpl_tail_weighted_topk_sum",
}

TWO_DIMENSIONAL_IDS = {
    "tpl_c2_two_dim_target_rate",
    "tpl_c2_filtered_group_count_2d",
    "tpl_tpch_two_dimensional_summary",
    "tpl_clickbench_two_dimensional_topk_count",
    "tpl_m4_two_dimensional_group_avg",
    "tpl_h2o_two_dimensional_group_sum",
    "tpl_h2o_two_dimensional_robust_summary",
}
PERCENTILE_IDS = {
    "tpl_grouped_percentile_point",
    "tpl_conditional_group_quantiles",
    "tpl_m4_quantile_tail_slice",
}
THRESHOLD_IDS = {
    "tpl_threshold_rarity_cdf",
    "tpl_tpch_thresholded_group_ranking",
    "tpl_tpch_relative_total_threshold",
}
BASELINE_IDS = {
    "tpl_tpcds_subgroup_baseline_outlier",
    "tpl_tpcds_baseline_gated_extreme_ranking",
}
OUTLIER_IDS = {
    "tpl_m4_global_zscore_outliers",
}

PORTABILITY_SCORE = {"yes": 2, "partial": 1, "adaptive": 1, "no": 0}
FAMILY_PRIORITY = {
    "subgroup_structure": 2,
    "conditional_dependency_structure": 1,
    "tail_rarity_structure": 1,
}

TEMPLATE_LIBRARY_PATH = DATA_DIR / "workload_grounding" / "template_library_v1.jsonl"
TEMPLATE_EXTENSION_LIBRARY_PATH = DATA_DIR / "workload_grounding" / "template_library_extensions_v1.jsonl"
TEMPLATE_POLICY_PATH = DATA_DIR / "workload_grounding" / "template_policy_v1.jsonl"


@dataclass
class QuestionInventoryItem:
    question_id: str
    dataset_id: str
    template_id: str
    template_name: str
    question: str
    bindings: dict[str, Any]
    portability: str
    failure_reason: str
    review_flag: str
    source_workload_id: str
    primary_family: str
    activation_tier: str
    dialect_sensitive: bool
    rank: int | None
    notes: list[str]
    problem_index_within_template: int
    variation_axes: list[str]
    can_vary: list[str]
    must_fix: list[str]
    expected_sql_count: int
    runtime_sql_skeleton: str | None = None


@dataclass
class TemplatePlanRecord:
    template_id: str
    template_name: str
    source_workload_id: str
    primary_family: str
    activation_tier: str
    dialect_sensitive: bool
    portability: str
    portability_reason: str
    review_flag: str
    rank: int | None
    can_vary: list[str]
    must_fix: list[str]
    base_bindings: dict[str, Any]
    selected_reason: str
    target_problem_min: int
    target_problem_max: int
    generated_problem_count: int
    candidate_problem_count: int
    loop_stats: dict[str, int]
    problems: list[QuestionInventoryItem]
    runtime_sql_skeleton: str | None = None
    selection_mode: str = "heuristic"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[obj["template_id"]] = obj
    return rows


def _validate_policy_list(*, template_id: str, field_name: str, values: Any) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"invalid template policy row for {template_id}: `{field_name}` must be a list of non-empty strings")
    normalized = _unique_preserve_order([value.strip() for value in values])
    if len(normalized) != len(values):
        raise ValueError(f"invalid template policy row for {template_id}: `{field_name}` contains duplicates")
    return normalized


@lru_cache(maxsize=1)
def _known_template_ids() -> set[str]:
    template_ids = set(_load_jsonl_by_id(TEMPLATE_LIBRARY_PATH))
    if TEMPLATE_EXTENSION_LIBRARY_PATH.exists():
        template_ids.update(_load_jsonl_by_id(TEMPLATE_EXTENSION_LIBRARY_PATH))
    return template_ids


@lru_cache(maxsize=1)
def _load_template_policy_lookup() -> dict[str, dict[str, list[str]]]:
    if not TEMPLATE_POLICY_PATH.exists():
        return {}

    rows: dict[str, dict[str, list[str]]] = {}
    with TEMPLATE_POLICY_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to parse {TEMPLATE_POLICY_PATH} line {line_number}") from exc

            template_id = row.get("template_id")
            if not isinstance(template_id, str) or not template_id.strip():
                raise ValueError(f"invalid template policy row on line {line_number}: missing `template_id`")
            template_id = template_id.strip()
            if template_id in rows:
                raise ValueError(f"duplicate template policy row for {template_id}")

            rows[template_id] = {
                "can_vary": _validate_policy_list(
                    template_id=template_id,
                    field_name="can_vary",
                    values=row.get("can_vary"),
                ),
                "must_fix": _validate_policy_list(
                    template_id=template_id,
                    field_name="must_fix",
                    values=row.get("must_fix"),
                ),
            }

    known_template_ids = _known_template_ids()
    missing = sorted(known_template_ids - set(rows))
    extras = sorted(set(rows) - known_template_ids)
    if missing or extras:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extras:
            details.append(f"extras={extras}")
        raise ValueError(
            "template policy coverage mismatch for "
            f"{TEMPLATE_POLICY_PATH}: " + "; ".join(details)
        )
    return rows


def _load_portability_rows(path: Path, dataset_id: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset_id"] != dataset_id:
                continue
            rows[row["template_id"]] = row
    return rows


def _humanize(name: str | None) -> str:
    if not name:
        return "value"
    return str(name).replace("_", " ")


def _stringify_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _q_or_default(value: float | None, fallback: float) -> float:
    return float(value) if value is not None else float(fallback)


def _percentile_label(value: Any) -> str:
    try:
        frac = float(value)
    except (TypeError, ValueError):
        frac = 0.95
    return f"p{int(round(frac * 100))}"


def _is_low_cardinality(stats: FieldStats | None) -> bool:
    if stats is None:
        return False
    return stats.distinct_count <= 5


def _is_binaryish(stats: FieldStats | None) -> bool:
    if stats is None:
        return False
    return stats.distinct_count <= 2


def _is_high_cardinality(stats: FieldStats | None) -> bool:
    if stats is None:
        return False
    return stats.distinct_count >= 20


def _top_non_null_values(stats: FieldStats | None) -> list[Any]:
    if stats is None:
        return []
    return [value for value, _count in stats.top_values if value is not None]


def _pick_positive_value(stats: FieldStats | None) -> Any:
    values = _top_non_null_values(stats)
    lowered = {str(v).lower(): v for v in values}
    for key in POSITIVE_VALUE_HINTS:
        if key in lowered:
            return lowered[key]
    return values[0] if values else None


def _pick_negative_value(stats: FieldStats | None) -> Any:
    values = _top_non_null_values(stats)
    lowered = {str(v).lower(): v for v in values}
    for key in NEGATIVE_VALUE_HINTS:
        if key in lowered:
            return lowered[key]
    if len(values) >= 2:
        return values[1]
    return values[0] if values else None


def _unique_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(value)
    return unique


def _pick_numeric_measure(
    field_stats: dict[str, FieldStats],
    *,
    target_column: str | None,
    exclude: set[str],
    current: str | None = None,
) -> str | None:
    def good_measure(col: str | None) -> bool:
        if not col or col in exclude:
            return False
        stats = field_stats.get(col)
        if stats is None or not stats.is_numeric:
            return False
        if target_column and col == target_column and _is_low_cardinality(stats):
            return False
        return True

    if good_measure(current):
        return current

    non_target_numeric = [
        stats.name
        for stats in field_stats.values()
        if stats.is_numeric and stats.name not in exclude and stats.name != target_column
    ]
    if non_target_numeric:
        non_target_numeric.sort(key=lambda name: (-field_stats[name].distinct_count, name))
        return non_target_numeric[0]

    fallback_numeric = [
        stats.name
        for stats in field_stats.values()
        if stats.is_numeric and stats.name not in exclude
    ]
    if fallback_numeric:
        fallback_numeric.sort(key=lambda name: (-field_stats[name].distinct_count, name))
        return fallback_numeric[0]
    return None


def _pick_groupable(
    field_stats: dict[str, FieldStats],
    *,
    exclude: set[str],
    current: str | None = None,
    target_column: str | None = None,
) -> str | None:
    def good_group(col: str | None) -> bool:
        if not col or col in exclude:
            return False
        stats = field_stats.get(col)
        if stats is None:
            return False
        return (
            stats.use_for_groupby
            or stats.is_categorical
            or stats.distinct_count <= 8
            or (stats.is_numeric and stats.distinct_count <= 20)
        )

    if good_group(current):
        return current

    candidates: list[str] = []
    if target_column and good_group(target_column):
        candidates.append(target_column)
    candidates.extend(
        stats.name
        for stats in field_stats.values()
        if stats.name not in exclude
        and (
            stats.use_for_groupby
            or stats.is_categorical
            or stats.distinct_count <= 8
            or (stats.is_numeric and stats.distinct_count <= 20)
        )
    )
    candidates = _unique_preserve_order(candidates)
    candidates.sort(key=lambda name: (field_stats[name].distinct_count, name))
    return candidates[0] if candidates else None


def _pick_item_col(
    field_stats: dict[str, FieldStats],
    *,
    group_col: str | None,
    exclude: set[str],
    current: str | None = None,
) -> str | None:
    if current and current not in exclude:
        return current
    candidates = [
        stats.name
        for stats in field_stats.values()
        if stats.name not in exclude
        and stats.name != group_col
        and (stats.use_for_groupby or stats.is_categorical or _is_high_cardinality(stats))
    ]
    candidates.sort(key=lambda name: (-field_stats[name].distinct_count, name))
    return candidates[0] if candidates else None


def _pick_predicate_value(stats: FieldStats | None) -> tuple[str, Any]:
    if stats is None:
        return "=", None
    if stats.is_numeric:
        threshold = _q_or_default(stats.q75, stats.q50 if stats.q50 is not None else 0.0)
        if threshold <= 0 and (stats.max_value or 0) > 0:
            return ">", 0
        return ">=", round(threshold, 4)
    return "=", _pick_positive_value(stats)


def _pick_band_bounds(stats: FieldStats | None) -> tuple[float, float]:
    if stats is None:
        return 0.0, 1.0
    lower = _q_or_default(stats.q33, stats.min_value if stats.min_value is not None else 0.0)
    upper = _q_or_default(stats.q66, stats.max_value if stats.max_value is not None else lower + 1.0)
    if upper <= lower:
        upper = lower + 1.0
    return round(lower, 4), round(upper, 4)


def _field_candidates_by_kind(
    field_stats: dict[str, FieldStats],
    *,
    target_column: str | None,
) -> dict[str, list[str]]:
    groupable = _unique_preserve_order(
        ([target_column] if target_column else [])
        + [
            stats.name
            for stats in field_stats.values()
            if stats.use_for_groupby or stats.is_categorical or stats.distinct_count <= 8 or (stats.is_numeric and stats.distinct_count <= 20)
        ]
    )
    numeric = sorted(
        [stats.name for stats in field_stats.values() if stats.is_numeric and stats.name != target_column],
        key=lambda name: (-field_stats[name].distinct_count, name),
    )
    low_card = sorted(
        [stats.name for stats in field_stats.values() if _is_low_cardinality(stats)],
        key=lambda name: (field_stats[name].distinct_count, name),
    )
    high_card = sorted(
        [stats.name for stats in field_stats.values() if _is_high_cardinality(stats)],
        key=lambda name: (-field_stats[name].distinct_count, name),
    )
    return {
        "groupable": groupable,
        "numeric": numeric,
        "low_card": low_card,
        "high_card": high_card,
    }


def _candidate_group_cols(
    field_stats: dict[str, FieldStats],
    *,
    current: str | None,
    target_column: str | None,
    exclude: set[str] | None = None,
    limit: int = 6,
) -> list[str]:
    exclude = exclude or set()
    pools = _field_candidates_by_kind(field_stats, target_column=target_column)
    ordered = [current] if current else []
    ordered.extend(pools["groupable"])
    ordered = [name for name in _unique_preserve_order(ordered) if name and name not in exclude]
    return ordered[:limit]


def _candidate_group_pairs(
    field_stats: dict[str, FieldStats],
    *,
    current_pair: tuple[str | None, str | None],
    target_column: str | None,
    limit: int = 6,
) -> list[tuple[str, str]]:
    base_group, base_group_2 = current_pair
    group_cols = _candidate_group_cols(field_stats, current=base_group, target_column=target_column, limit=6)
    pair_candidates: list[tuple[str, str]] = []
    if base_group and base_group_2 and base_group != base_group_2:
        pair_candidates.append((base_group, base_group_2))
    for first, second in combinations(group_cols, 2):
        pair_candidates.append((first, second))
    return _unique_preserve_order(pair_candidates)[:limit]


def _candidate_measure_cols(
    field_stats: dict[str, FieldStats],
    *,
    current: str | None,
    target_column: str | None,
    exclude: set[str] | None = None,
    limit: int = 3,
) -> list[str]:
    exclude = exclude or set()
    ordered = [current] if current else []
    ordered.extend(
        [
            stats.name
            for stats in field_stats.values()
            if stats.is_numeric and stats.name not in exclude and stats.name != target_column
        ]
    )
    ordered = [name for name in _unique_preserve_order(ordered) if name and name not in exclude]
    ordered.sort(key=lambda name: (-field_stats[name].distinct_count, name))
    if current and current in ordered:
        ordered.remove(current)
        ordered.insert(0, current)
    return ordered[:limit]


def _candidate_predicates(
    field_stats: dict[str, FieldStats],
    *,
    current_col: str | None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []

    def add_option(col: str) -> None:
        stats = field_stats.get(col)
        if stats is None:
            return
        if stats.is_numeric:
            if stats.q75 is not None:
                options.append({"predicate_col": col, "predicate_op": ">=", "predicate_value": round(stats.q75, 4)})
            if (stats.max_value or 0) > 0:
                options.append({"predicate_col": col, "predicate_op": ">", "predicate_value": 0})
        else:
            values = _top_non_null_values(stats)
            positive = _pick_positive_value(stats)
            if positive is not None:
                options.append({"predicate_col": col, "predicate_op": "=", "predicate_value": positive})
            for value in values[:2]:
                options.append({"predicate_col": col, "predicate_op": "=", "predicate_value": value})

    if current_col:
        add_option(current_col)
    for stats in sorted(field_stats.values(), key=lambda row: (row.distinct_count, row.name)):
        if stats.name == current_col:
            continue
        if stats.is_numeric or _is_low_cardinality(stats):
            add_option(stats.name)
    return _unique_preserve_order(options)[:limit]


def _candidate_conditions(
    field_stats: dict[str, FieldStats],
    *,
    current_col: str | None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []

    def add_option(col: str) -> None:
        stats = field_stats.get(col)
        if stats is None or not (_is_low_cardinality(stats) or _is_binaryish(stats)):
            return
        positive = _pick_positive_value(stats)
        negative = _pick_negative_value(stats)
        if positive is None:
            return
        options.append(
            {
                "condition_col": col,
                "condition_value": positive,
                "positive_value": positive,
                "negative_value": negative,
            }
        )

    if current_col:
        add_option(current_col)
    for stats in sorted(field_stats.values(), key=lambda row: (row.distinct_count, row.name)):
        if stats.name == current_col:
            continue
        add_option(stats.name)
    return _unique_preserve_order(options)[:limit]


def _candidate_item_cols(
    field_stats: dict[str, FieldStats],
    *,
    current: str | None,
    group_col: str | None,
    limit: int = 4,
) -> list[str]:
    ordered = [current] if current else []
    ordered.extend(
        [
            stats.name
            for stats in field_stats.values()
            if stats.name != group_col and (stats.use_for_groupby or stats.is_categorical or _is_high_cardinality(stats))
        ]
    )
    ordered = [name for name in _unique_preserve_order(ordered) if name and name != group_col]
    return ordered[:limit]


def _candidate_band_settings(
    field_stats: dict[str, FieldStats],
    *,
    current: str | None,
    target_column: str | None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for band_col in _candidate_measure_cols(field_stats, current=current, target_column=target_column, limit=limit):
        lower, upper = _pick_band_bounds(field_stats.get(band_col))
        settings.append(
            {
                "band_col": band_col,
                "lower_bound": lower,
                "upper_bound": upper,
                "band_cut_1": lower,
                "band_cut_2": upper,
            }
        )
    return _unique_preserve_order(settings)[:limit]


def _candidate_threshold_values(stats: FieldStats | None, *, row_count: int) -> list[float]:
    if stats is None or not stats.is_numeric:
        return [float(max(5, row_count // 20))]
    values: list[float] = []
    if stats.q66 is not None:
        values.append(round(stats.q66, 4))
    if stats.q75 is not None:
        values.append(round(stats.q75, 4))
    if stats.q75 is not None and stats.max_value is not None and stats.max_value > stats.q75:
        values.append(round((stats.q75 + stats.max_value) / 2.0, 4))
    if not values and stats.q50 is not None:
        values.append(round(stats.q50, 4))
    return _unique_preserve_order(values)[:3]


def _candidate_fraction_values() -> list[float]:
    return [0.05, 0.10, 0.20]


def _candidate_percentile_values() -> list[float]:
    return [0.90, 0.95, 0.99]


def _candidate_num_tiles() -> list[int]:
    return [4, 5, 10]


def _candidate_z_thresholds() -> list[float]:
    return [2.0, 2.5, 3.0]


def _candidate_support_values(row_count: int) -> list[int]:
    values = [max(10, row_count // 100), max(20, row_count // 50), max(30, row_count // 25)]
    return _unique_preserve_order(values)


def _candidate_baseline_values() -> list[float]:
    return [1.25, 1.50, 2.0]


def _supports_adaptive_count_fallback(
    *,
    template_id: str,
    portability_row: dict[str, Any],
    field_stats: dict[str, FieldStats],
) -> bool:
    if template_id not in COUNT_AGGREGATE_FALLBACK_IDS:
        return False
    if portability_row.get("portable") != "no":
        return False
    if any(stats.is_numeric for stats in field_stats.values()):
        return False
    try:
        raw_bindings = json.loads(portability_row.get("binding_example") or "{}")
    except Exception:
        return False
    return bool(raw_bindings.get("group_col"))


def _maybe_exclude_partial(
    template_id: str,
    portability_row: dict[str, Any],
    *,
    field_stats: dict[str, FieldStats],
) -> str | None:
    if _supports_adaptive_count_fallback(
        template_id=template_id,
        portability_row=portability_row,
        field_stats=field_stats,
    ):
        return None

    failure_reason = portability_row.get("failure_reason") or ""
    if portability_row.get("portable") == "no":
        return "portable=no"
    if "constraint_failed:" in failure_reason:
        return "constraint_failed"
    if template_id in RATE_RATIO_IDS and "condition_col:used_high_cardinality_fallback" in failure_reason:
        return "condition_col_high_cardinality_fallback"
    if template_id in ITEM_HEAVY_IDS and "item_col:used_high_cardinality_fallback" in failure_reason:
        return "item_col_high_cardinality_fallback"
    return None


def _prepare_bindings(
    *,
    template_id: str,
    raw_bindings: dict[str, Any],
    field_stats: dict[str, FieldStats],
    target_column: str | None,
    row_count: int,
) -> tuple[dict[str, Any], list[str]]:
    bindings = {k: v for k, v in raw_bindings.items() if k != "table"}
    notes: list[str] = []

    group_col = bindings.get("group_col")
    group_col_2 = bindings.get("group_col_2")
    item_col = bindings.get("item_col")
    measure_col = bindings.get("measure_col")
    predicate_col = bindings.get("predicate_col")
    condition_col = bindings.get("condition_col")
    target_col = bindings.get("target_col") or target_column
    band_col = bindings.get("band_col")

    if group_col:
        bindings["group_col"] = _pick_groupable(
            field_stats,
            exclude=set(),
            current=group_col,
            target_column=target_column,
        ) or group_col
    if group_col_2:
        bindings["group_col_2"] = _pick_groupable(
            field_stats,
            exclude={bindings.get("group_col")} if bindings.get("group_col") else set(),
            current=group_col_2,
            target_column=target_column,
        ) or group_col_2
    if item_col:
        bindings["item_col"] = _pick_item_col(
            field_stats,
            group_col=bindings.get("group_col"),
            exclude={bindings.get("group_col")} if bindings.get("group_col") else set(),
            current=item_col,
        ) or item_col

    if measure_col or template_id in {
        "tpl_m4_group_avg_numeric",
        "tpl_m4_median_filtered_numeric",
        "tpl_m4_support_guarded_group_avg",
        "tpl_tpcds_topk_group_sum",
        "tpl_m4_group_dispersion_rank",
        "tpl_m4_global_zscore_outliers",
        "tpl_m4_binned_numeric_group_avg",
        "tpl_m4_two_dimensional_group_avg",
        "tpl_h2o_group_sum",
        "tpl_h2o_two_dimensional_group_sum",
        "tpl_h2o_two_dimensional_robust_summary",
        "tpl_h2o_topn_within_group",
        "tpl_tpch_filtered_sum_band",
        "tpl_tpch_relative_total_threshold",
        "tpl_tpch_max_aggregate_winner",
        "tpl_tpch_thresholded_group_ranking",
        "tpl_tpcds_subgroup_baseline_outlier",
        "tpl_tpcds_baseline_gated_extreme_ranking",
        "tpl_tail_weighted_topk_sum",
        "tpl_grouped_percentile_point",
        "tpl_conditional_group_quantiles",
        "tpl_threshold_rarity_cdf",
        "tpl_m4_quantile_tail_slice",
    }:
        exclude = {bindings.get("group_col"), bindings.get("group_col_2"), bindings.get("item_col"), bindings.get("band_col")}
        exclude = {value for value in exclude if value}
        refined_measure = _pick_numeric_measure(
            field_stats,
            target_column=target_column,
            exclude=exclude,
            current=measure_col,
        )
        if refined_measure and refined_measure != measure_col:
            notes.append(f"measure_col_refined:{measure_col}->{refined_measure}")
        if refined_measure:
            bindings["measure_col"] = refined_measure
        elif template_id in COUNT_AGGREGATE_FALLBACK_IDS:
            bindings["aggregate_measure_mode"] = "count_rows"
            bindings.pop("measure_col", None)
            notes.append("adaptive_count_aggregate_fallback")

    if band_col:
        current_band = field_stats.get(str(band_col))
        if current_band is None or not current_band.is_numeric:
            candidate = _pick_numeric_measure(field_stats, target_column=target_column, exclude=set(), current=None)
            if candidate:
                bindings["band_col"] = candidate
                band_col = candidate
                notes.append("band_col_refined")
        lower_bound, upper_bound = _pick_band_bounds(field_stats.get(str(bindings.get("band_col"))))
        bindings["lower_bound"] = lower_bound
        bindings["upper_bound"] = upper_bound
        bindings["band_cut_1"] = lower_bound
        bindings["band_cut_2"] = upper_bound

    if predicate_col:
        op, value = _pick_predicate_value(field_stats.get(str(predicate_col)))
        bindings["predicate_op"] = op
        bindings["predicate_value"] = value

    if condition_col:
        stats = field_stats.get(str(condition_col))
        bindings["condition_value"] = _pick_positive_value(stats)
        positive = _pick_positive_value(stats)
        negative = _pick_negative_value(stats)
        if positive is not None:
            bindings["positive_value"] = positive
        if negative is not None:
            bindings["negative_value"] = negative

    if target_col:
        bindings["target_col"] = target_col
        stats = field_stats.get(str(target_col))
        bindings["target_value"] = _pick_positive_value(stats)

    bindings.setdefault("top_k", 5)
    bindings.setdefault("top_n", 3)
    bindings.setdefault("percentile_value", 0.95)
    bindings.setdefault("num_tiles", 4)
    bindings.setdefault("z_threshold", 2.0)
    bindings.setdefault("fraction_threshold", 0.10)
    bindings.setdefault("baseline_multiplier", 1.50)
    bindings.setdefault("baseline_fraction", 1.20)
    bindings.setdefault("min_support", max(10, row_count // 100))
    bindings.setdefault("min_group_size", max(20, row_count // 50))

    measure_stats = field_stats.get(str(bindings.get("measure_col"))) if bindings.get("measure_col") else None
    if measure_stats and measure_stats.is_numeric:
        rarity_threshold = _q_or_default(
            measure_stats.q75,
            measure_stats.q50 if measure_stats.q50 is not None else 1.0,
        )
        grouped_threshold = round(rarity_threshold * max(3, row_count // 40), 4)
        if template_id == "tpl_threshold_rarity_cdf":
            bindings.setdefault("measure_threshold", round(rarity_threshold, 4))
        else:
            bindings.setdefault("measure_threshold", grouped_threshold)
    else:
        bindings.setdefault("measure_threshold", max(5, row_count // 20))

    return bindings, notes


def _build_question(template_id: str, bindings: dict[str, Any]) -> str:
    g = _humanize(bindings.get("group_col"))
    g2 = _humanize(bindings.get("group_col_2"))
    item = _humanize(bindings.get("item_col"))
    measure = _humanize(bindings.get("measure_col"))
    predicate = _humanize(bindings.get("predicate_col"))
    condition = _humanize(bindings.get("condition_col"))
    entity = _humanize(bindings.get("entity_col"))
    target = _humanize(bindings.get("target_col"))
    band = _humanize(bindings.get("band_col"))
    pred_val = _stringify_value(bindings.get("predicate_value"))
    cond_val = _stringify_value(bindings.get("condition_value"))
    pos_val = _stringify_value(bindings.get("positive_value"))
    neg_val = _stringify_value(bindings.get("negative_value"))
    target_val = _stringify_value(bindings.get("target_value"))
    percentile_label = _percentile_label(bindings.get("percentile_value", 0.95))
    measure_display = "row count" if bindings.get("aggregate_measure_mode") == "count_rows" else measure

    questions = {
        "tpl_clickbench_group_count": f"How is the dataset distributed across {g}?",
        "tpl_clickbench_group_distinct_topk": f"Which {g} groups cover the most distinct {entity} values?",
        "tpl_c2_two_dim_target_rate": f"Across {g} and {g2}, where is {target} most likely to be {target_val}?",
        "tpl_c2_filtered_group_count_2d": f"Among rows where {predicate} {bindings.get('predicate_op', '=')} {pred_val}, which {g}-by-{g2} combinations are most common?",
        "tpl_m4_group_avg_numeric": f"How does average {measure} vary across {g}?",
        "tpl_m4_group_condition_rate": f"Which {g} groups most often have {condition} = {cond_val}?",
        "tpl_m4_median_filtered_numeric": f"What is the median {measure} among rows where {predicate} {bindings.get('predicate_op', '=')} {pred_val}?",
        "tpl_m4_support_guarded_group_avg": f"How does average {measure} vary across {g} after excluding groups with fewer than {bindings.get('min_group_size')} rows?",
        "tpl_m4_group_ratio_two_conditions": f"How does the ratio of {condition} = {pos_val} to {condition} = {neg_val} vary across {g}?",
        "tpl_tpch_two_dimensional_summary": f"How do total and average {measure} vary across {g} and {g2} when {predicate} {bindings.get('predicate_op', '=')} {pred_val}?",
        "tpl_tpch_filtered_sum_band": f"What is the total {measure} for rows where {band} falls between {bindings.get('lower_bound')} and {bindings.get('upper_bound')}?",
        "tpl_tpcds_topk_group_sum": f"Which {g} groups have the highest total {measure_display} among rows where {predicate} {bindings.get('predicate_op', '=')} {pred_val}?",
        "tpl_tpcds_within_group_share": f"Within each {g}, which {item} values contribute the largest share of total {measure_display}?",
        "tpl_clickbench_group_summary_topk": f"Which {g} groups lead on support and average {measure}?",
        "tpl_clickbench_filtered_topk_group_count": f"Among rows where {predicate} {bindings.get('predicate_op', '=')} {pred_val}, which {g} groups are most common?",
        "tpl_clickbench_two_dimensional_topk_count": f"Which {g}-by-{g2} combinations are most common?",
        "tpl_m4_window_partition_avg": f"Which {g} groups have the highest partition-level average {measure}?",
        "tpl_m4_quantile_tail_slice": f"Which {measure} values fall into the top {bindings.get('num_tiles')} tail bucket of the distribution?",
        "tpl_m4_group_dispersion_rank": f"Which {g} groups show the largest dispersion in {measure}?",
        "tpl_m4_global_zscore_outliers": f"Which rows are global outliers on {measure} using a z-score threshold above {bindings.get('z_threshold')}?",
        "tpl_m4_binned_numeric_group_avg": f"How does average {measure} vary across {band} bands?",
        "tpl_m4_two_dimensional_group_avg": f"How does average {measure} vary across {g} and {g2} combinations?",
        "tpl_clickbench_filtered_distinct_topk": f"Among rows where {predicate} {bindings.get('predicate_op', '=')} {pred_val}, which {g} groups cover the most distinct {entity} values?",
        "tpl_h2o_group_sum": f"Which {g} groups contribute the highest total {measure_display}?",
        "tpl_h2o_two_dimensional_group_sum": f"Which {g} and {g2} combinations contribute the highest total {measure_display}?",
        "tpl_h2o_two_dimensional_robust_summary": f"Across {g} and {g2}, which combinations have the highest median {measure} and the largest spread?",
        "tpl_h2o_topn_within_group": f"What are the top {bindings.get('top_n')} {measure} values within each {g} group?",
        "tpl_tpch_relative_total_threshold": f"Which {g} groups contribute more than {int(float(bindings.get('fraction_threshold', 0.1)) * 100)}% of total {measure_display}?",
        "tpl_tpch_max_aggregate_winner": (
            f"Which {g} group has the single highest total {measure}?"
            if bindings.get("aggregate_measure_mode") != "count_rows"
            else f"Which {g} group has the single highest row count?"
        ),
        "tpl_tpch_thresholded_group_ranking": (
            f"Which {g} groups exceed a total {measure} threshold of {bindings.get('measure_threshold')} and rank highest?"
            if bindings.get("aggregate_measure_mode") != "count_rows"
            else f"Which {g} groups exceed a row-count threshold of {bindings.get('measure_threshold')} and rank highest?"
        ),
        "tpl_tpcds_subgroup_baseline_outlier": f"Within each {g}, which {item} values are unusually high on {measure} relative to their subgroup baseline?",
        "tpl_tpcds_baseline_gated_extreme_ranking": f"Within each {g}, which {item} values remain extreme on {measure} after passing a subgroup baseline gate?",
        "tpl_tail_weighted_topk_sum": f"Which {g} groups have the largest weighted total {measure} after requiring at least {bindings.get('min_support')} rows?",
        "tpl_grouped_percentile_point": f"What is the {percentile_label} of {measure} for each {g}?",
        "tpl_conditional_group_quantiles": f"For each {g}, what is the {percentile_label} of {measure} among rows where {condition} = {cond_val}?",
        "tpl_threshold_rarity_cdf": f"How rare is it for {measure_display} to be above {bindings.get('measure_threshold')}?",
    }
    return questions.get(template_id, f"Analyze {template_id} on this dataset.")


def _infer_template_policy(template_id: str, template: dict[str, Any]) -> tuple[list[str], list[str]]:
    required_roles = set(template.get("required_roles") or [])
    can_vary: list[str] = []

    if template_id in TWO_DIMENSIONAL_IDS:
        can_vary.append("group_pair")
    elif "group_col" in required_roles:
        can_vary.append("group_col")

    if "measure_col" in required_roles:
        can_vary.append("measure_col")
    if "predicate_col" in required_roles:
        can_vary.append("predicate")
    if "condition_col" in required_roles:
        can_vary.append("condition")
    if "item_col" in required_roles:
        can_vary.append("item_col")
    if "band_col" in required_roles:
        can_vary.append("band")

    if template_id in PERCENTILE_IDS:
        can_vary.append("percentile_level")
    if template_id in THRESHOLD_IDS:
        can_vary.append("threshold")
    if template_id in OUTLIER_IDS:
        can_vary.append("threshold")
    if template_id == "tpl_m4_quantile_tail_slice":
        can_vary.append("num_tiles")
    if template_id in {"tpl_m4_support_guarded_group_avg", "tpl_tail_weighted_topk_sum"}:
        can_vary.append("support_gate")
    if template_id in BASELINE_IDS:
        can_vary.append("baseline_factor")

    must_fix = [
        "single_table_only",
        "template_intent",
        "canonical_query_shape",
        "required_role_structure_except_can_vary",
    ]
    if template_id in TWO_DIMENSIONAL_IDS:
        must_fix.append("two_dimensional_grouping")
    if template_id in RATE_RATIO_IDS:
        must_fix.append("rate_or_ratio_semantics")
    if template_id in PERCENTILE_IDS:
        must_fix.append("percentile_tail_semantics")
    if template_id in THRESHOLD_IDS or template_id in OUTLIER_IDS or template_id in BASELINE_IDS:
        must_fix.append("tail_or_threshold_semantics")
    if template_id in COUNT_AGGREGATE_FALLBACK_IDS:
        must_fix.append("winner_selection_semantics")

    return _unique_preserve_order(can_vary), _unique_preserve_order(must_fix)


def _resolve_template_policy(template_id: str, template: dict[str, Any]) -> tuple[list[str], list[str]]:
    policy_lookup = _load_template_policy_lookup()
    if template_id in policy_lookup:
        row = policy_lookup[template_id]
        return list(row["can_vary"]), list(row["must_fix"])
    return _infer_template_policy(template_id, template)


def _reset_derived_fields(raw_bindings: dict[str, Any], axes: list[str]) -> None:
    if "predicate" in axes:
        raw_bindings.pop("predicate_op", None)
        raw_bindings.pop("predicate_value", None)
    if "condition" in axes:
        raw_bindings.pop("condition_value", None)
        raw_bindings.pop("positive_value", None)
        raw_bindings.pop("negative_value", None)
    if "measure_col" in axes or "threshold" in axes:
        raw_bindings.pop("measure_threshold", None)
    if "band" in axes or "measure_col" in axes:
        raw_bindings.pop("lower_bound", None)
        raw_bindings.pop("upper_bound", None)
        raw_bindings.pop("band_cut_1", None)
        raw_bindings.pop("band_cut_2", None)
    if "percentile_level" in axes:
        raw_bindings.pop("percentile_value", None)
    if "support_gate" in axes:
        raw_bindings.pop("min_support", None)
        raw_bindings.pop("min_group_size", None)
    if "baseline_factor" in axes:
        raw_bindings.pop("baseline_multiplier", None)
        raw_bindings.pop("baseline_fraction", None)


def _validate_problem_bindings(
    *,
    template_id: str,
    template: dict[str, Any],
    bindings: dict[str, Any],
) -> str | None:
    required_roles = list(template.get("required_roles") or [])
    for role in required_roles:
        if role == "measure_col" and template_id in COUNT_AGGREGATE_FALLBACK_IDS and bindings.get("aggregate_measure_mode") == "count_rows":
            continue
        value = bindings.get(role)
        if value in {None, ""}:
            return f"missing_required_role:{role}"
    if template_id in TWO_DIMENSIONAL_IDS and bindings.get("group_col") == bindings.get("group_col_2"):
        return "duplicate_group_pair"
    if bindings.get("item_col") and bindings.get("item_col") == bindings.get("group_col"):
        return "item_equals_group"
    return None


def _adaptive_runtime_sql_skeleton(template_id: str, bindings: dict[str, Any]) -> str | None:
    if template_id == "tpl_tpch_max_aggregate_winner" and bindings.get("aggregate_measure_mode") == "count_rows":
        return """WITH grouped AS (
    SELECT {group_col}, COUNT(*) AS total_rows
    FROM {table}
    GROUP BY {group_col}
)
SELECT {group_col}, total_rows
FROM grouped
WHERE total_rows = (SELECT MAX(total_rows) FROM grouped)
ORDER BY {group_col};"""
    return None


def _template_priority(spec_item: dict[str, Any]) -> str:
    return str(spec_item.get("priority") or "p1")


def _template_score(
    *,
    portability: str,
    problem_count: int,
    primary_family: str,
    dialect_sensitive: bool,
    rank: int | None,
) -> tuple[int, int, int, int]:
    return (
        PORTABILITY_SCORE.get(portability, 0),
        min(problem_count, 12),
        FAMILY_PRIORITY.get(primary_family, 0),
        -(rank if rank is not None else 999),
    )


def _dataset_summary_for_planner(
    *,
    dataset_id: str,
    field_stats: dict[str, FieldStats],
    target_column: str | None,
    row_count: int,
) -> dict[str, Any]:
    groupable = []
    numeric = []
    low_cardinality = []
    high_cardinality = []
    for col, stats in field_stats.items():
        if stats.is_numeric:
            numeric.append(col)
        else:
            groupable.append(col)
        if _is_low_cardinality(stats):
            low_cardinality.append(col)
        if _is_high_cardinality(stats):
            high_cardinality.append(col)
    return {
        "dataset_id": dataset_id,
        "row_count": row_count,
        "target_column": target_column,
        "groupable_columns": groupable[:8],
        "numeric_columns": numeric[:8],
        "low_cardinality_columns": low_cardinality[:8],
        "high_cardinality_columns": high_cardinality[:8],
        "field_cardinality_summary": {
            col: {
                "is_numeric": stats.is_numeric,
                "distinct_count": stats.distinct_count,
                "top_values": [_stringify_value(v) for v, _count in stats.top_values[:3]],
            }
            for col, stats in list(field_stats.items())[:20]
        },
    }


def _target_column_from_bundle(bundle: Any) -> str | None:
    target_column = (
        str(
            bundle.dataset_semantics.get("target_column")
            or bundle.dataset_contract.get("target_column")
            or bundle.dataset_profile.get("target_column")
            or ""
        )
        or None
    )
    return target_column


def _load_inventory_context(*, dataset_id: str, data_root: Path) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_id=dataset_id, data_root=data_root, strict=True)
    sqlite_result = materialize_dataset_to_sqlite(bundle=bundle, use_cache=True)
    field_stats = build_field_stats(bundle, sqlite_result.table_name, sqlite_result.db_path)
    target_column = _target_column_from_bundle(bundle)
    dataset_summary = _dataset_summary_for_planner(
        dataset_id=dataset_id,
        field_stats=field_stats,
        target_column=target_column,
        row_count=sqlite_result.row_count,
    )
    return {
        "bundle": bundle,
        "sqlite_result": sqlite_result,
        "field_stats": field_stats,
        "target_column": target_column,
        "dataset_summary": dataset_summary,
    }


def _adaptive_inventory_thresholds(
    *,
    field_stats: dict[str, FieldStats],
    min_templates: int,
    target_templates: int,
    min_problems_per_template: int,
    max_problems_per_template: int,
) -> dict[str, Any]:
    total_fields = len(field_stats)
    numeric_count = sum(1 for stats in field_stats.values() if stats.is_numeric)
    groupby_count = sum(1 for stats in field_stats.values() if stats.use_for_groupby)
    predicate_count = sum(1 for stats in field_stats.values() if stats.use_for_predicate)
    target_count = sum(1 for stats in field_stats.values() if stats.use_as_target)
    identifier_like_count = sum(
        1
        for stats in field_stats.values()
        if stats.field_role == "identifier" or "identifier" in stats.field_tags
    )

    policy = {
        "triggered": False,
        "reason": "",
        "requested_min_templates": min_templates,
        "requested_target_templates": target_templates,
        "requested_min_problems_per_template": min_problems_per_template,
        "requested_max_problems_per_template": max_problems_per_template,
        "effective_min_templates": min_templates,
        "effective_target_templates": target_templates,
        "effective_min_problems_per_template": min_problems_per_template,
        "effective_max_problems_per_template": max_problems_per_template,
        "schema_signals": {
            "total_fields": total_fields,
            "numeric_count": numeric_count,
            "groupby_count": groupby_count,
            "predicate_count": predicate_count,
            "target_count": target_count,
            "identifier_like_count": identifier_like_count,
        },
    }

    # Extremely compact schemas such as c3 have almost no stable non-target roles.
    # Relax the template/problem minimums so they can still produce a small but valid
    # inventory instead of collapsing to an empty question set.
    if total_fields <= 4 and numeric_count == 0 and groupby_count == 0:
        effective_min_templates = max(1, min(min_templates, 4))
        effective_target_templates = max(
            effective_min_templates,
            min(target_templates, 6),
        )
        policy.update(
            {
                "triggered": True,
                "reason": "compact_schema_without_numeric_or_groupby_roles",
                "effective_min_templates": effective_min_templates,
                "effective_target_templates": effective_target_templates,
                "effective_min_problems_per_template": 1,
            }
        )
    return policy


def build_template_candidate_pool(
    *,
    dataset_id: str,
    spec_path: Path,
    spec_bucket: str,
    core_library_path: Path,
    portability_report_path: Path,
    data_root: Path = DATA_DIR,
    min_templates: int = 10,
) -> dict[str, Any]:
    spec = _load_json(spec_path)
    templates = _load_jsonl_by_id(core_library_path)
    portability = _load_portability_rows(portability_report_path, dataset_id)

    context = _load_inventory_context(dataset_id=dataset_id, data_root=data_root)
    bundle = context["bundle"]
    sqlite_result = context["sqlite_result"]
    field_stats = context["field_stats"]
    target_column = context["target_column"]
    dataset_summary = context["dataset_summary"]

    candidate_rows: list[dict[str, Any]] = []

    for rank, spec_item in enumerate(spec.get(spec_bucket, []), start=1):
        template_id = spec_item["template_id"]
        template = templates.get(template_id)
        portability_row = portability.get(template_id)

        row: dict[str, Any] = {
            "template_id": template_id,
            "template_name": spec_item.get("template_name") or (template or {}).get("template_name") or template_id,
            "source_workload_id": spec_item.get("source_workload_id") or (template or {}).get("source_workload_id") or "",
            "primary_family": spec_item.get("primary_family") or (template or {}).get("primary_family") or "",
            "activation_tier": spec_item.get("activation_tier") or "",
            "dialect_sensitive": bool(spec_item.get("dialect_sensitive", False)),
            "rank": rank,
            "required_roles": list(spec_item.get("required_roles") or (template or {}).get("required_roles") or []),
            "constraints": list(spec_item.get("constraints") or []),
            "portability": "",
            "portability_reason": "",
            "failure_reason": "",
            "missing_required_roles": [],
            "review_flag": "no",
            "can_vary": [],
            "must_fix": [],
            "base_bindings": {},
            "binding_notes": [],
            "runtime_sql_skeleton": None,
            "adaptive_count_fallback": False,
            "screening_status": "",
            "screening_reason": "",
        }

        if template is None:
            row.update(
                {
                    "screening_status": "template_missing",
                    "screening_reason": "template_missing",
                }
            )
            candidate_rows.append(row)
            continue

        can_vary, must_fix = _resolve_template_policy(template_id, template)
        row["can_vary"] = can_vary
        row["must_fix"] = must_fix

        if portability_row is None:
            row.update(
                {
                    "screening_status": "no_portability_row",
                    "screening_reason": "no_portability_row",
                }
            )
            candidate_rows.append(row)
            continue

        portable = str(portability_row.get("portable") or "")
        failure_reason = str(portability_row.get("failure_reason") or "")
        review_flag = str(portability_row.get("review_flag") or "no")
        missing_required_roles = [
            value.strip()
            for value in str(portability_row.get("missing_required_roles") or "").split(",")
            if value.strip()
        ]
        adaptive_count_fallback = _supports_adaptive_count_fallback(
            template_id=template_id,
            portability_row=portability_row,
            field_stats=field_stats,
        )

        base_bindings: dict[str, Any] = {}
        binding_notes: list[str] = []
        binding_parse_failed = False
        try:
            raw_bindings = json.loads(portability_row.get("binding_example") or "{}")
            if not isinstance(raw_bindings, dict):
                raise ValueError("binding_example must decode to an object")
            base_bindings, binding_notes = _prepare_bindings(
                template_id=template_id,
                raw_bindings=raw_bindings,
                field_stats=field_stats,
                target_column=target_column,
                row_count=sqlite_result.row_count,
            )
        except Exception:
            binding_parse_failed = True
            binding_notes = ["binding_example_parse_failed"]

        exclusion = _maybe_exclude_partial(
            template_id,
            portability_row,
            field_stats=field_stats,
        )

        effective_portability = "adaptive" if adaptive_count_fallback else portable
        portability_reason = "adaptive_count_aggregate_fallback" if adaptive_count_fallback else failure_reason
        screening_status = "eligible"
        screening_reason = "eligible"
        if exclusion:
            screening_status = "excluded"
            screening_reason = exclusion
        elif binding_parse_failed:
            screening_status = "excluded"
            screening_reason = "binding_example_parse_failed"

        row.update(
            {
                "portability": effective_portability,
                "portability_reason": portability_reason,
                "failure_reason": failure_reason,
                "missing_required_roles": missing_required_roles,
                "review_flag": review_flag,
                "base_bindings": base_bindings,
                "binding_notes": binding_notes,
                "runtime_sql_skeleton": _adaptive_runtime_sql_skeleton(template_id, base_bindings)
                if adaptive_count_fallback
                else None,
                "adaptive_count_fallback": adaptive_count_fallback,
                "screening_status": screening_status,
                "screening_reason": screening_reason,
            }
        )
        candidate_rows.append(row)

    screening_status_counts = Counter(str(row["screening_status"]) for row in candidate_rows)
    portability_counts = Counter(str(row["portability"]) for row in candidate_rows if row["portability"])
    eligible_rows = [row for row in candidate_rows if row["screening_status"] == "eligible"]

    return {
        "dataset_id": dataset_id,
        "row_count": sqlite_result.row_count,
        "main_csv_path": str(bundle.main_csv_path),
        "candidate_pool_count": len(spec.get(spec_bucket, [])),
        "screened_template_count": len(candidate_rows),
        "eligible_template_count": len(eligible_rows),
        "eligible_template_ids": [row["template_id"] for row in eligible_rows],
        "review_candidate_count": sum(
            1 for row in eligible_rows if str(row.get("review_flag") or "no").lower() == "yes"
        ),
        "screening_status_counts": dict(screening_status_counts),
        "portability_counts": dict(portability_counts),
        "agent_selection_min_templates": min_templates,
        "agent_selection_ready": len(eligible_rows) >= min_templates,
        "agent_selection_gap": max(0, min_templates - len(eligible_rows)),
        "preprocessing_policy": {
            "mode": "candidate_pool_only",
            "final_template_selection_deferred_to_agent": True,
            "final_problem_generation_deferred_to_agent": True,
            "screening_uses_portability_and_binding_validation": True,
            "policy_fields_available": ["can_vary", "must_fix"],
        },
        "dataset_summary": dataset_summary,
        "templates": candidate_rows,
    }


def _template_candidates_for_planner(template_plans: list[TemplatePlanRecord]) -> list[dict[str, Any]]:
    return [
        {
            "template_id": plan.template_id,
            "template_name": plan.template_name,
            "primary_family": plan.primary_family,
            "portability": plan.portability,
            "dialect_sensitive": plan.dialect_sensitive,
            "generated_problem_count": plan.generated_problem_count,
            "can_vary": plan.can_vary,
            "must_fix": plan.must_fix,
            "rank": plan.rank,
        }
        for plan in template_plans
    ]


def _problem_candidates_for_planner(plan: TemplatePlanRecord) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in plan.problems:
        candidates.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "variation_axes": item.variation_axes,
                "bindings": item.bindings,
                "can_vary": item.can_vary,
                "must_fix": item.must_fix,
                "dialect_sensitive": item.dialect_sensitive,
            }
        )
    return candidates


def _apply_planner_template_selection(
    *,
    planner: LLMProblemPlanner,
    dataset_id: str,
    dataset_summary: dict[str, Any],
    template_plans: list[TemplatePlanRecord],
    min_templates: int,
    target_templates: int,
    fallback: list[TemplatePlanRecord],
) -> list[TemplatePlanRecord]:
    ai_ids = planner.select_templates(
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
        candidates=_template_candidates_for_planner(template_plans),
        min_templates=min_templates,
        target_templates=target_templates,
    )
    by_id = {plan.template_id: plan for plan in template_plans}
    selected: list[TemplatePlanRecord] = []
    selected_ids: set[str] = set()
    for template_id in ai_ids:
        plan = by_id.get(template_id)
        if plan is None or template_id in selected_ids:
            continue
        updated = TemplatePlanRecord(**{**plan.__dict__, "selection_mode": "llm_selected"})
        selected.append(updated)
        selected_ids.add(template_id)
        if len(selected) >= target_templates:
            break

    for plan in fallback:
        if len(selected) >= min_templates:
            break
        if plan.template_id in selected_ids:
            continue
        updated = TemplatePlanRecord(**{**plan.__dict__, "selection_mode": "heuristic_backfill"})
        selected.append(updated)
        selected_ids.add(plan.template_id)

    if len(selected) < min_templates:
        remaining = [
            plan for plan in template_plans if plan.template_id not in selected_ids
        ]
        remaining.sort(
            key=lambda plan: _template_score(
                portability=plan.portability,
                problem_count=plan.generated_problem_count,
                primary_family=plan.primary_family,
                dialect_sensitive=plan.dialect_sensitive,
                rank=plan.rank,
            ),
            reverse=True,
        )
        for plan in remaining:
            if len(selected) >= min_templates:
                break
            updated = TemplatePlanRecord(**{**plan.__dict__, "selection_mode": "heuristic_backfill"})
            selected.append(updated)
            selected_ids.add(plan.template_id)
    return selected


def _clone_problem_item(
    item: QuestionInventoryItem,
    *,
    problem_index_within_template: int | None = None,
    expected_sql_count: int | None = None,
) -> QuestionInventoryItem:
    return QuestionInventoryItem(
        question_id=item.question_id,
        dataset_id=item.dataset_id,
        template_id=item.template_id,
        template_name=item.template_name,
        question=item.question,
        bindings=item.bindings,
        portability=item.portability,
        failure_reason=item.failure_reason,
        review_flag=item.review_flag,
        source_workload_id=item.source_workload_id,
        primary_family=item.primary_family,
        activation_tier=item.activation_tier,
        dialect_sensitive=item.dialect_sensitive,
        rank=item.rank,
        notes=item.notes,
        problem_index_within_template=problem_index_within_template or item.problem_index_within_template,
        variation_axes=item.variation_axes,
        can_vary=item.can_vary,
        must_fix=item.must_fix,
        expected_sql_count=expected_sql_count if expected_sql_count is not None else item.expected_sql_count,
        runtime_sql_skeleton=item.runtime_sql_skeleton,
    )


def _reindex_selected_problems(problems: list[QuestionInventoryItem]) -> list[QuestionInventoryItem]:
    reindexed: list[QuestionInventoryItem] = []
    for idx, item in enumerate(problems, start=1):
        reindexed.append(_clone_problem_item(item, problem_index_within_template=idx))
    return reindexed


def _apply_planner_problem_selection(
    *,
    planner: LLMProblemPlanner,
    dataset_id: str,
    plan: TemplatePlanRecord,
    min_problems: int,
    max_problems: int,
) -> TemplatePlanRecord:
    ai_ids = planner.select_problem_ids(
        dataset_id=dataset_id,
        template_summary={
            "template_id": plan.template_id,
            "template_name": plan.template_name,
            "primary_family": plan.primary_family,
            "can_vary": plan.can_vary,
            "must_fix": plan.must_fix,
            "base_bindings": plan.base_bindings,
        },
        candidate_items=_problem_candidates_for_planner(plan),
        min_problems=min_problems,
        max_problems=max_problems,
    )
    by_id = {item.question_id: item for item in plan.problems}
    selected: list[QuestionInventoryItem] = []
    selected_ids: set[str] = set()
    for question_id in ai_ids:
        item = by_id.get(question_id)
        if item is None or question_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(question_id)
        if len(selected) >= max_problems:
            break

    if len(selected) < min_problems:
        for item in plan.problems:
            if len(selected) >= min_problems:
                break
            if item.question_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item.question_id)

    selected = _reindex_selected_problems(selected[:max_problems])
    updated_loop_stats = dict(plan.loop_stats)
    updated_loop_stats["llm_selected_problems"] = len(ai_ids)
    updated_loop_stats["final_selected_problems"] = len(selected)
    return TemplatePlanRecord(
        template_id=plan.template_id,
        template_name=plan.template_name,
        source_workload_id=plan.source_workload_id,
        primary_family=plan.primary_family,
        activation_tier=plan.activation_tier,
        dialect_sensitive=plan.dialect_sensitive,
        portability=plan.portability,
        portability_reason=plan.portability_reason,
        review_flag=plan.review_flag,
        rank=plan.rank,
        can_vary=plan.can_vary,
        must_fix=plan.must_fix,
        base_bindings=plan.base_bindings,
        selected_reason=plan.selected_reason,
        target_problem_min=plan.target_problem_min,
        target_problem_max=plan.target_problem_max,
        generated_problem_count=len(selected),
        candidate_problem_count=plan.candidate_problem_count,
        loop_stats=updated_loop_stats,
        problems=selected,
        runtime_sql_skeleton=plan.runtime_sql_skeleton,
        selection_mode="llm_selected",
    )


def _materialize_problem_item(
    *,
    dataset_id: str,
    template: dict[str, Any],
    template_id: str,
    template_name: str,
    raw_bindings: dict[str, Any],
    field_stats: dict[str, FieldStats],
    target_column: str | None,
    row_count: int,
    portability: str,
    failure_reason: str,
    review_flag: str,
    rank: int | None,
    can_vary: list[str],
    must_fix: list[str],
    variation_axes: list[str],
    problem_index_within_template: int,
    expected_sql_count: int = 2,
) -> tuple[QuestionInventoryItem | None, str | None]:
    prepared_bindings, notes = _prepare_bindings(
        template_id=template_id,
        raw_bindings=raw_bindings,
        field_stats=field_stats,
        target_column=target_column,
        row_count=row_count,
    )
    validation_error = _validate_problem_bindings(template_id=template_id, template=template, bindings=prepared_bindings)
    if validation_error:
        return None, validation_error

    question = _build_question(template_id, prepared_bindings)
    runtime_sql_skeleton = _adaptive_runtime_sql_skeleton(template_id, prepared_bindings)
    question_id = f"fq_{dataset_id}_{stable_hash(f'{dataset_id}|{template_id}|{question}', 10)}"

    return (
        QuestionInventoryItem(
            question_id=question_id,
            dataset_id=dataset_id,
            template_id=template_id,
            template_name=template_name,
            question=question,
            bindings=prepared_bindings,
            portability=portability,
            failure_reason=failure_reason,
            review_flag=review_flag,
            source_workload_id=str(template.get("source_workload_id") or "unknown"),
            primary_family=str(template.get("primary_family") or "unknown"),
            activation_tier=str(template.get("activation_tier") or "core"),
            dialect_sensitive=bool(template.get("dialect_sensitive", False)),
            rank=rank,
            notes=notes,
            problem_index_within_template=problem_index_within_template,
            variation_axes=variation_axes,
            can_vary=can_vary,
            must_fix=must_fix,
            expected_sql_count=expected_sql_count,
            runtime_sql_skeleton=runtime_sql_skeleton,
        ),
        None,
    )


def _generate_problem_items_for_template(
    *,
    dataset_id: str,
    template_id: str,
    template: dict[str, Any],
    spec_item: dict[str, Any],
    portability_row: dict[str, Any],
    field_stats: dict[str, FieldStats],
    target_column: str | None,
    row_count: int,
    min_problems: int,
    max_problems: int,
    candidate_problem_cap: int,
) -> tuple[TemplatePlanRecord | None, dict[str, Any] | None]:
    try:
        base_raw_bindings = json.loads(portability_row.get("binding_example") or "{}")
    except Exception:
        return None, {"template_id": template_id, "reason": "binding_example_parse_failed"}

    can_vary, must_fix = _resolve_template_policy(template_id, template)
    template_name = str(template.get("template_name") or template_id)
    portability = portability_row.get("portable") or "no"
    if _supports_adaptive_count_fallback(
        template_id=template_id,
        portability_row=portability_row,
        field_stats=field_stats,
    ):
        portability = "adaptive"

    base_item, base_error = _materialize_problem_item(
        dataset_id=dataset_id,
        template=template,
        template_id=template_id,
        template_name=template_name,
        raw_bindings=base_raw_bindings,
        field_stats=field_stats,
        target_column=target_column,
        row_count=row_count,
        portability=portability,
        failure_reason=portability_row.get("failure_reason", ""),
        review_flag=portability_row.get("review_flag", ""),
        rank=spec_item.get("rank"),
        can_vary=can_vary,
        must_fix=must_fix,
        variation_axes=[],
        problem_index_within_template=1,
    )
    if base_item is None:
        return None, {"template_id": template_id, "reason": base_error or "base_problem_invalid"}

    loop_stats = Counter(
        {
            "attempted_candidates": 0,
            "accepted_problems": 0,
            "rejected_duplicates": 0,
            "rejected_invalid": 0,
        }
    )
    problems: list[QuestionInventoryItem] = []
    seen_questions: set[str] = set()

    def try_add(raw_bindings: dict[str, Any], variation_axes: list[str]) -> None:
        if len(problems) >= candidate_problem_cap:
            return
        loop_stats["attempted_candidates"] += 1
        item, error = _materialize_problem_item(
            dataset_id=dataset_id,
            template=template,
            template_id=template_id,
            template_name=template_name,
            raw_bindings=raw_bindings,
            field_stats=field_stats,
            target_column=target_column,
            row_count=row_count,
            portability=portability,
            failure_reason=portability_row.get("failure_reason", ""),
            review_flag=portability_row.get("review_flag", ""),
            rank=spec_item.get("rank"),
            can_vary=can_vary,
            must_fix=must_fix,
            variation_axes=variation_axes,
            problem_index_within_template=len(problems) + 1,
        )
        if item is None:
            loop_stats["rejected_invalid"] += 1
            return
        question_key = item.question.strip().lower()
        if question_key in seen_questions:
            loop_stats["rejected_duplicates"] += 1
            return
        seen_questions.add(question_key)
        problems.append(item)
        loop_stats["accepted_problems"] += 1

    try_add(base_raw_bindings, [])

    current_bindings = base_item.bindings
    group_candidates = _candidate_group_cols(
        field_stats,
        current=current_bindings.get("group_col"),
        target_column=target_column,
    )
    pair_candidates = _candidate_group_pairs(
        field_stats,
        current_pair=(current_bindings.get("group_col"), current_bindings.get("group_col_2")),
        target_column=target_column,
    )
    measure_candidates = _candidate_measure_cols(
        field_stats,
        current=current_bindings.get("measure_col"),
        target_column=target_column,
        exclude={current_bindings.get("group_col"), current_bindings.get("group_col_2"), current_bindings.get("item_col"), current_bindings.get("band_col")} - {None},
    )
    predicate_candidates = _candidate_predicates(field_stats, current_col=current_bindings.get("predicate_col"))
    condition_candidates = _candidate_conditions(field_stats, current_col=current_bindings.get("condition_col"))
    item_candidates = _candidate_item_cols(
        field_stats,
        current=current_bindings.get("item_col"),
        group_col=current_bindings.get("group_col"),
    )
    band_candidates = _candidate_band_settings(
        field_stats,
        current=current_bindings.get("band_col"),
        target_column=target_column,
    )
    threshold_candidates = _candidate_threshold_values(
        field_stats.get(str(current_bindings.get("measure_col"))) if current_bindings.get("measure_col") else None,
        row_count=row_count,
    )

    def add_axis_variations() -> None:
        if "group_pair" in can_vary:
            for first, second in pair_candidates:
                raw = dict(base_raw_bindings)
                raw["group_col"] = first
                raw["group_col_2"] = second
                _reset_derived_fields(raw, ["group_pair"])
                try_add(raw, ["group_pair"])
        if "group_col" in can_vary:
            for group_col in group_candidates:
                raw = dict(base_raw_bindings)
                raw["group_col"] = group_col
                _reset_derived_fields(raw, ["group_col"])
                try_add(raw, ["group_col"])
        if "measure_col" in can_vary:
            for measure_col in measure_candidates:
                raw = dict(base_raw_bindings)
                raw["measure_col"] = measure_col
                _reset_derived_fields(raw, ["measure_col"])
                try_add(raw, ["measure_col"])
        if "predicate" in can_vary:
            for predicate in predicate_candidates:
                raw = dict(base_raw_bindings)
                raw.update(predicate)
                _reset_derived_fields(raw, ["predicate"])
                raw.update(predicate)
                try_add(raw, ["predicate"])
        if "condition" in can_vary:
            for condition in condition_candidates:
                raw = dict(base_raw_bindings)
                raw.update(condition)
                _reset_derived_fields(raw, ["condition"])
                raw.update(condition)
                try_add(raw, ["condition"])
        if "item_col" in can_vary:
            for item_col in item_candidates:
                raw = dict(base_raw_bindings)
                raw["item_col"] = item_col
                _reset_derived_fields(raw, ["item_col"])
                try_add(raw, ["item_col"])
        if "band" in can_vary:
            for band_setting in band_candidates:
                raw = dict(base_raw_bindings)
                raw.update(band_setting)
                _reset_derived_fields(raw, ["band"])
                raw.update(band_setting)
                try_add(raw, ["band"])
        if "percentile_level" in can_vary:
            for percentile_value in _candidate_percentile_values():
                raw = dict(base_raw_bindings)
                raw["percentile_value"] = percentile_value
                _reset_derived_fields(raw, ["percentile_level"])
                raw["percentile_value"] = percentile_value
                try_add(raw, ["percentile_level"])
        if "threshold" in can_vary:
            for threshold in threshold_candidates:
                raw = dict(base_raw_bindings)
                raw["measure_threshold"] = threshold
                _reset_derived_fields(raw, ["threshold"])
                raw["measure_threshold"] = threshold
                try_add(raw, ["threshold"])
        if "num_tiles" in can_vary:
            for num_tiles in _candidate_num_tiles():
                raw = dict(base_raw_bindings)
                raw["num_tiles"] = num_tiles
                _reset_derived_fields(raw, ["num_tiles"])
                raw["num_tiles"] = num_tiles
                try_add(raw, ["num_tiles"])
        if "support_gate" in can_vary:
            for support_value in _candidate_support_values(row_count):
                raw = dict(base_raw_bindings)
                raw["min_support"] = support_value
                raw["min_group_size"] = support_value
                _reset_derived_fields(raw, ["support_gate"])
                raw["min_support"] = support_value
                raw["min_group_size"] = support_value
                try_add(raw, ["support_gate"])
        if "baseline_factor" in can_vary:
            for baseline_value in _candidate_baseline_values():
                raw = dict(base_raw_bindings)
                raw["baseline_multiplier"] = baseline_value
                raw["baseline_fraction"] = baseline_value
                _reset_derived_fields(raw, ["baseline_factor"])
                raw["baseline_multiplier"] = baseline_value
                raw["baseline_fraction"] = baseline_value
                try_add(raw, ["baseline_factor"])

    def add_pairwise_variations() -> None:
        if len(problems) >= max_problems:
            return
        if "group_col" in can_vary and "measure_col" in can_vary:
            for group_col in group_candidates[:4]:
                for measure_col in measure_candidates[:3]:
                    raw = dict(base_raw_bindings)
                    raw["group_col"] = group_col
                    raw["measure_col"] = measure_col
                    _reset_derived_fields(raw, ["group_col", "measure_col"])
                    raw["group_col"] = group_col
                    raw["measure_col"] = measure_col
                    try_add(raw, ["group_col", "measure_col"])
                    if len(problems) >= max_problems:
                        return
        if "group_col" in can_vary and "predicate" in can_vary:
            for group_col in group_candidates[:3]:
                for predicate in predicate_candidates[:3]:
                    raw = dict(base_raw_bindings)
                    raw["group_col"] = group_col
                    raw.update(predicate)
                    _reset_derived_fields(raw, ["group_col", "predicate"])
                    raw["group_col"] = group_col
                    raw.update(predicate)
                    try_add(raw, ["group_col", "predicate"])
                    if len(problems) >= max_problems:
                        return
        if "group_col" in can_vary and "condition" in can_vary:
            for group_col in group_candidates[:3]:
                for condition in condition_candidates[:3]:
                    raw = dict(base_raw_bindings)
                    raw["group_col"] = group_col
                    raw.update(condition)
                    _reset_derived_fields(raw, ["group_col", "condition"])
                    raw["group_col"] = group_col
                    raw.update(condition)
                    try_add(raw, ["group_col", "condition"])
                    if len(problems) >= max_problems:
                        return
        if "group_pair" in can_vary and "measure_col" in can_vary:
            for first, second in pair_candidates[:3]:
                for measure_col in measure_candidates[:2]:
                    raw = dict(base_raw_bindings)
                    raw["group_col"] = first
                    raw["group_col_2"] = second
                    raw["measure_col"] = measure_col
                    _reset_derived_fields(raw, ["group_pair", "measure_col"])
                    raw["group_col"] = first
                    raw["group_col_2"] = second
                    raw["measure_col"] = measure_col
                    try_add(raw, ["group_pair", "measure_col"])
                    if len(problems) >= max_problems:
                        return
        if "measure_col" in can_vary and "threshold" in can_vary:
            for measure_col in measure_candidates[:3]:
                measure_stats = field_stats.get(measure_col)
                for threshold in _candidate_threshold_values(measure_stats, row_count=row_count)[:3]:
                    raw = dict(base_raw_bindings)
                    raw["measure_col"] = measure_col
                    raw["measure_threshold"] = threshold
                    _reset_derived_fields(raw, ["measure_col", "threshold"])
                    raw["measure_col"] = measure_col
                    raw["measure_threshold"] = threshold
                    try_add(raw, ["measure_col", "threshold"])
                    if len(problems) >= max_problems:
                        return

    add_axis_variations()
    add_pairwise_variations()

    if len(problems) < min_problems:
        return None, {
            "template_id": template_id,
            "reason": "insufficient_problem_count",
            "generated_problem_count": len(problems),
            "required_min_problem_count": min_problems,
        }

    selected_reason = (
        f"selected_for_dataset_problem_inventory; portability={portability}; "
        f"generated_problems={len(problems)}"
    )
    base_runtime_sql_skeleton = problems[0].runtime_sql_skeleton if problems else None
    return (
        TemplatePlanRecord(
            template_id=template_id,
            template_name=template_name,
            source_workload_id=str(template.get("source_workload_id") or "unknown"),
            primary_family=str(template.get("primary_family") or "unknown"),
            activation_tier=str(template.get("activation_tier") or "core"),
            dialect_sensitive=bool(template.get("dialect_sensitive", False)),
            portability=portability,
            portability_reason=portability_row.get("failure_reason", ""),
            review_flag=portability_row.get("review_flag", ""),
            rank=spec_item.get("rank"),
            can_vary=can_vary,
            must_fix=must_fix,
            base_bindings=base_item.bindings,
            selected_reason=selected_reason,
            target_problem_min=min_problems,
            target_problem_max=max_problems,
            generated_problem_count=min(len(problems), max_problems),
            candidate_problem_count=len(problems),
            loop_stats=dict(loop_stats),
            problems=problems[:max_problems],
            runtime_sql_skeleton=base_runtime_sql_skeleton,
        ),
        None,
    )


def _family_template_targets(
    *,
    available_by_family: dict[str, list[TemplatePlanRecord]],
    min_templates: int,
    target_templates: int,
    has_numeric: bool,
) -> dict[str, int]:
    targets: dict[str, int] = {}
    if has_numeric:
        requested = {
            "subgroup_structure": 4,
            "conditional_dependency_structure": 3,
            "tail_rarity_structure": 3,
        }
    else:
        requested = {
            "subgroup_structure": 6,
            "conditional_dependency_structure": 4,
            "tail_rarity_structure": 0,
        }
    for family, wanted in requested.items():
        available = len(available_by_family.get(family, []))
        targets[family] = min(available, wanted)

    total = sum(targets.values())
    if total < min_templates:
        leftovers = {
            family: len(available_by_family.get(family, [])) - targets.get(family, 0)
            for family in available_by_family
        }
        for family in sorted(leftovers, key=lambda key: (-leftovers[key], key)):
            while total < min_templates and leftovers[family] > 0:
                targets[family] = targets.get(family, 0) + 1
                leftovers[family] -= 1
                total += 1

    if total > target_templates:
        for family in sorted(targets, key=lambda key: (targets[key], key), reverse=True):
            while total > target_templates and targets[family] > 0:
                targets[family] -= 1
                total -= 1

    return targets


def _select_template_plans(
    *,
    template_plans: list[TemplatePlanRecord],
    min_templates: int,
    target_templates: int,
    has_numeric: bool,
) -> list[TemplatePlanRecord]:
    by_family: dict[str, list[TemplatePlanRecord]] = defaultdict(list)
    for plan in template_plans:
        by_family[plan.primary_family].append(plan)
    for family in by_family:
        by_family[family].sort(
            key=lambda plan: _template_score(
                portability=plan.portability,
                problem_count=plan.generated_problem_count,
                primary_family=plan.primary_family,
                dialect_sensitive=plan.dialect_sensitive,
                rank=plan.rank,
            ),
            reverse=True,
        )

    family_targets = _family_template_targets(
        available_by_family=by_family,
        min_templates=min_templates,
        target_templates=target_templates,
        has_numeric=has_numeric,
    )

    selected: list[TemplatePlanRecord] = []
    selected_ids: set[str] = set()
    for family, target in family_targets.items():
        for plan in by_family.get(family, [])[:target]:
            if plan.template_id in selected_ids:
                continue
            selected.append(plan)
            selected_ids.add(plan.template_id)

    remaining = [
        plan
        for plan in sorted(
            template_plans,
            key=lambda plan: _template_score(
                portability=plan.portability,
                problem_count=plan.generated_problem_count,
                primary_family=plan.primary_family,
                dialect_sensitive=plan.dialect_sensitive,
                rank=plan.rank,
            ),
            reverse=True,
        )
        if plan.template_id not in selected_ids
    ]
    for plan in remaining:
        if len(selected) >= target_templates:
            break
        selected.append(plan)
        selected_ids.add(plan.template_id)

    if len(selected) < min_templates:
        for plan in remaining:
            if len(selected) >= min_templates:
                break
            if plan.template_id in selected_ids:
                continue
            selected.append(plan)
            selected_ids.add(plan.template_id)

    return selected


def build_full_question_inventory(
    *,
    dataset_id: str,
    spec_path: Path,
    spec_bucket: str,
    core_library_path: Path,
    portability_report_path: Path,
    data_root: Path = DATA_DIR,
    min_templates: int = 10,
    target_templates: int = 12,
    min_problems_per_template: int = 4,
    max_problems_per_template: int = 12,
    planner_model: str | None = None,
    planner_run_id: str = "",
    usage_logger: UsageCSVLogger | None = None,
    pricing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = _load_json(spec_path)
    templates = _load_jsonl_by_id(core_library_path)
    portability = _load_portability_rows(portability_report_path, dataset_id)

    context = _load_inventory_context(dataset_id=dataset_id, data_root=data_root)
    sqlite_result = context["sqlite_result"]
    field_stats = context["field_stats"]
    target_column = context["target_column"]
    has_numeric = any(stats.is_numeric for stats in field_stats.values())
    dataset_summary = context["dataset_summary"]
    adaptive_policy = _adaptive_inventory_thresholds(
        field_stats=field_stats,
        min_templates=min_templates,
        target_templates=target_templates,
        min_problems_per_template=min_problems_per_template,
        max_problems_per_template=max_problems_per_template,
    )
    min_templates = int(adaptive_policy["effective_min_templates"])
    target_templates = int(adaptive_policy["effective_target_templates"])
    min_problems_per_template = int(adaptive_policy["effective_min_problems_per_template"])
    max_problems_per_template = int(adaptive_policy["effective_max_problems_per_template"])
    planner = None
    if planner_model:
        from src.workload_grounding.problem_planner import LLMProblemPlanner

        planner = LLMProblemPlanner(
            model_name=planner_model,
            dataset_id=dataset_id,
            run_id=planner_run_id or f"{dataset_id}_planner",
            usage_logger=usage_logger,
            pricing_config=pricing_config,
        )

    template_plans: list[TemplatePlanRecord] = []
    skipped: list[dict[str, Any]] = []

    for rank, spec_item in enumerate(spec.get(spec_bucket, []), start=1):
        template_id = spec_item["template_id"]
        template = templates.get(template_id)
        portability_row = portability.get(template_id)
        if template is None:
            skipped.append({"template_id": template_id, "reason": "template_missing"})
            continue
        if portability_row is None:
            skipped.append({"template_id": template_id, "reason": "no_portability_row"})
            continue

        exclusion = _maybe_exclude_partial(template_id, portability_row, field_stats=field_stats)
        if exclusion:
            skipped.append({"template_id": template_id, "reason": exclusion})
            continue

        spec_item = dict(spec_item)
        spec_item["rank"] = rank
        plan, plan_error = _generate_problem_items_for_template(
            dataset_id=dataset_id,
            template_id=template_id,
            template=template,
            spec_item=spec_item,
            portability_row=portability_row,
            field_stats=field_stats,
            target_column=target_column,
            row_count=sqlite_result.row_count,
            min_problems=min_problems_per_template,
            max_problems=max_problems_per_template,
            candidate_problem_cap=max(max_problems_per_template * 3, 24),
        )
        if plan is None:
            skipped.append(plan_error or {"template_id": template_id, "reason": "problem_generation_failed"})
            continue
        if planner is not None:
            plan = _apply_planner_problem_selection(
                planner=planner,
                dataset_id=dataset_id,
                plan=plan,
                min_problems=min_problems_per_template,
                max_problems=max_problems_per_template,
            )
        template_plans.append(plan)

    fallback_selected = _select_template_plans(
        template_plans=template_plans,
        min_templates=min_templates,
        target_templates=target_templates,
        has_numeric=has_numeric,
    )
    if planner is not None:
        selected_templates = _apply_planner_template_selection(
            planner=planner,
            dataset_id=dataset_id,
            dataset_summary=dataset_summary,
            template_plans=template_plans,
            min_templates=min_templates,
            target_templates=target_templates,
            fallback=fallback_selected,
        )
    else:
        selected_templates = fallback_selected

    items: list[QuestionInventoryItem] = []
    for plan in selected_templates:
        items.extend(plan.problems)

    family_counts = Counter(item.primary_family for item in items)
    template_problem_counts = {plan.template_id: plan.generated_problem_count for plan in selected_templates}
    selected_template_ids = [plan.template_id for plan in selected_templates]

    return {
        "dataset_id": dataset_id,
        "row_count": sqlite_result.row_count,
        "candidate_pool_count": len(spec.get(spec_bucket, [])),
        "template_candidate_count": len(template_plans),
        "selected_template_count": len(selected_templates),
        "problem_count": len(items),
        "inventory_count": len(items),
        "family_counts": dict(family_counts),
        "template_ids": selected_template_ids,
        "problem_generation_policy": {
            "min_templates": min_templates,
            "target_templates": target_templates,
            "min_problems_per_template": min_problems_per_template,
            "max_problems_per_template": max_problems_per_template,
            "expected_sql_per_problem": 2,
            "planner_model": planner_model or "",
            "planner_mode": "llm_constrained_selection" if planner_model else "heuristic_fallback",
            "policy_fields_planned": ["can_vary", "must_fix"],
            "policy_materialization_status": "runtime_inferred_placeholder_until_template_preprocessing",
            "adaptive_threshold_policy": adaptive_policy,
        },
        "selected_templates": [
            {
                "template_id": plan.template_id,
                "template_name": plan.template_name,
                "source_workload_id": plan.source_workload_id,
                "primary_family": plan.primary_family,
                "activation_tier": plan.activation_tier,
                "dialect_sensitive": plan.dialect_sensitive,
                "portability": plan.portability,
                "portability_reason": plan.portability_reason,
                "review_flag": plan.review_flag,
                "rank": plan.rank,
                "can_vary": plan.can_vary,
                "must_fix": plan.must_fix,
                "base_bindings": plan.base_bindings,
                "selected_reason": plan.selected_reason,
                "target_problem_min": plan.target_problem_min,
                "target_problem_max": plan.target_problem_max,
                "generated_problem_count": plan.generated_problem_count,
                "candidate_problem_count": plan.candidate_problem_count,
                "loop_stats": plan.loop_stats,
                "runtime_sql_skeleton": plan.runtime_sql_skeleton,
                "selection_mode": plan.selection_mode,
            }
            for plan in selected_templates
        ],
        "problem_counts_by_template": template_problem_counts,
        "items": [
            {
                "question_id": item.question_id,
                "dataset_id": item.dataset_id,
                "template_id": item.template_id,
                "template_name": item.template_name,
                "question": item.question,
                "bindings": item.bindings,
                "portability": item.portability,
                "failure_reason": item.failure_reason,
                "review_flag": item.review_flag,
                "source_workload_id": item.source_workload_id,
                "primary_family": item.primary_family,
                "activation_tier": item.activation_tier,
                "dialect_sensitive": item.dialect_sensitive,
                "rank": item.rank,
                "notes": item.notes,
                "problem_index_within_template": item.problem_index_within_template,
                "variation_axes": item.variation_axes,
                "can_vary": item.can_vary,
                "must_fix": item.must_fix,
                "expected_sql_count": item.expected_sql_count,
                "runtime_sql_skeleton": item.runtime_sql_skeleton,
            }
            for item in items
        ],
        "skipped": skipped,
    }


def _normalize_variation_axes(
    values: list[str] | None,
    *,
    can_vary: list[str],
    base_bindings: dict[str, Any],
    bindings: dict[str, Any],
) -> list[str]:
    normalized = _unique_preserve_order([str(value).strip() for value in (values or []) if str(value).strip()])
    filtered = [value for value in normalized if value in can_vary]
    if filtered:
        return filtered

    inferred: list[str] = []
    for axis in can_vary:
        if base_bindings.get(axis) != bindings.get(axis):
            inferred.append(axis)
    return inferred


def _template_summary_for_cli_problem_generation(
    *,
    plan: TemplatePlanRecord,
    template: dict[str, Any],
) -> dict[str, Any]:
    return {
        "template_id": plan.template_id,
        "template_name": plan.template_name,
        "primary_family": plan.primary_family,
        "source_workload_id": plan.source_workload_id,
        "portability": plan.portability,
        "dialect_sensitive": plan.dialect_sensitive,
        "intent": template.get("intent"),
        "required_roles": list(template.get("required_roles") or []),
        "optional_roles": list(template.get("optional_roles") or []),
        "constraints": list(template.get("constraints") or []),
        "sql_skeleton": template.get("sql_skeleton"),
        "can_vary": plan.can_vary,
        "must_fix": plan.must_fix,
        "base_bindings": plan.base_bindings,
    }


def build_cli_all_question_inventory(
    *,
    dataset_id: str,
    spec_path: Path,
    spec_bucket: str,
    core_library_path: Path,
    portability_report_path: Path,
    planner_model: str,
    project_root: Path,
    data_root: Path = DATA_DIR,
    min_templates: int = 10,
    target_templates: int = 12,
    min_problems_per_template: int = 4,
    max_problems_per_template: int = 12,
    ai_cli_preset: str = "codex",
    ai_cli_command: str = "",
    planner_timeout_seconds: int = 420,
    planner_invoke_retries: int = 2,
    planner_run_id: str = "",
    usage_logger: UsageCSVLogger | None = None,
    pricing_config: dict[str, Any] | None = None,
    artifact_writer: RunArtifactWriter | None = None,
) -> dict[str, Any]:
    if not planner_model.strip():
        raise ValueError("cli-all inventory generation requires a non-empty planner_model")

    spec = _load_json(spec_path)
    templates = _load_jsonl_by_id(core_library_path)
    portability = _load_portability_rows(portability_report_path, dataset_id)

    context = _load_inventory_context(dataset_id=dataset_id, data_root=data_root)
    sqlite_result = context["sqlite_result"]
    field_stats = context["field_stats"]
    target_column = context["target_column"]
    has_numeric = any(stats.is_numeric for stats in field_stats.values())
    dataset_summary = context["dataset_summary"]
    adaptive_policy = _adaptive_inventory_thresholds(
        field_stats=field_stats,
        min_templates=min_templates,
        target_templates=target_templates,
        min_problems_per_template=min_problems_per_template,
        max_problems_per_template=max_problems_per_template,
    )
    min_templates = int(adaptive_policy["effective_min_templates"])
    target_templates = int(adaptive_policy["effective_target_templates"])
    min_problems_per_template = int(adaptive_policy["effective_min_problems_per_template"])
    max_problems_per_template = int(adaptive_policy["effective_max_problems_per_template"])

    from src.workload_grounding.problem_planner import CLIProblemPlanner

    planner = CLIProblemPlanner(
        model_name=planner_model,
        dataset_id=dataset_id,
        run_id=planner_run_id or f"{dataset_id}_cli_all_planner",
        project_root=project_root,
        ai_cli_preset=ai_cli_preset,
        ai_cli_command=ai_cli_command,
        usage_logger=usage_logger,
        pricing_config=pricing_config,
        artifact_writer=artifact_writer,
        request_timeout_seconds=planner_timeout_seconds,
        invoke_retries=max(1, planner_invoke_retries),
    )

    template_plans: list[TemplatePlanRecord] = []
    skipped: list[dict[str, Any]] = []

    for rank, spec_item in enumerate(spec.get(spec_bucket, []), start=1):
        template_id = spec_item["template_id"]
        template = templates.get(template_id)
        portability_row = portability.get(template_id)
        if template is None:
            skipped.append({"template_id": template_id, "reason": "template_missing"})
            continue
        if portability_row is None:
            skipped.append({"template_id": template_id, "reason": "no_portability_row"})
            continue

        exclusion = _maybe_exclude_partial(template_id, portability_row, field_stats=field_stats)
        if exclusion:
            skipped.append({"template_id": template_id, "reason": exclusion})
            continue

        spec_item = dict(spec_item)
        spec_item["rank"] = rank
        plan, plan_error = _generate_problem_items_for_template(
            dataset_id=dataset_id,
            template_id=template_id,
            template=template,
            spec_item=spec_item,
            portability_row=portability_row,
            field_stats=field_stats,
            target_column=target_column,
            row_count=sqlite_result.row_count,
            min_problems=min_problems_per_template,
            max_problems=max_problems_per_template,
            candidate_problem_cap=max(max_problems_per_template * 3, 24),
        )
        if plan is None:
            skipped.append(plan_error or {"template_id": template_id, "reason": "problem_generation_failed"})
            continue
        template_plans.append(plan)

    fallback_selected = _select_template_plans(
        template_plans=template_plans,
        min_templates=min_templates,
        target_templates=target_templates,
        has_numeric=has_numeric,
    )
    selected_templates = _apply_planner_template_selection(
        planner=planner,
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
        template_plans=template_plans,
        min_templates=min_templates,
        target_templates=target_templates,
        fallback=fallback_selected,
    )

    finalized_templates: list[TemplatePlanRecord] = []
    invalid_problem_records: list[dict[str, Any]] = []
    for plan in selected_templates:
        template = templates.get(plan.template_id)
        if template is None:
            continue
        generated_specs = planner.generate_problems(
            dataset_id=dataset_id,
            dataset_summary=dataset_summary,
            template_summary=_template_summary_for_cli_problem_generation(plan=plan, template=template),
            reference_items=_problem_candidates_for_planner(plan),
            min_problems=min_problems_per_template,
            max_problems=max_problems_per_template,
        )
        selected_problems: list[QuestionInventoryItem] = []
        selected_ids: set[str] = set()
        invalid_count = 0
        for raw_problem in generated_specs:
            bindings = raw_problem.get("bindings")
            if not isinstance(bindings, dict):
                invalid_problem_records.append(
                    {
                        "template_id": plan.template_id,
                        "reason": "bindings_missing_or_invalid",
                        "raw_problem": raw_problem,
                    }
                )
                invalid_count += 1
                continue
            variation_axes = _normalize_variation_axes(
                raw_problem.get("variation_axes"),
                can_vary=plan.can_vary,
                base_bindings=plan.base_bindings,
                bindings=bindings,
            )
            item, error = _materialize_problem_item(
                dataset_id=dataset_id,
                template=template,
                template_id=plan.template_id,
                template_name=plan.template_name,
                raw_bindings=bindings,
                field_stats=field_stats,
                target_column=target_column,
                row_count=sqlite_result.row_count,
                portability=plan.portability,
                failure_reason=plan.portability_reason,
                review_flag=plan.review_flag,
                rank=plan.rank,
                can_vary=plan.can_vary,
                must_fix=plan.must_fix,
                variation_axes=variation_axes,
                problem_index_within_template=len(selected_problems) + 1,
                expected_sql_count=1,
            )
            if item is None:
                invalid_problem_records.append(
                    {
                        "template_id": plan.template_id,
                        "reason": error or "materialization_failed",
                        "raw_problem": raw_problem,
                    }
                )
                invalid_count += 1
                continue
            if item.question_id in selected_ids:
                invalid_problem_records.append(
                    {
                        "template_id": plan.template_id,
                        "reason": "duplicate_question_id",
                        "question_id": item.question_id,
                        "raw_problem": raw_problem,
                    }
                )
                invalid_count += 1
                continue
            selected_problems.append(item)
            selected_ids.add(item.question_id)
            if len(selected_problems) >= max_problems_per_template:
                break

        backfill_count = 0
        if len(selected_problems) < min_problems_per_template:
            for reference_item in plan.problems:
                if len(selected_problems) >= min_problems_per_template:
                    break
                cloned = _clone_problem_item(
                    reference_item,
                    problem_index_within_template=len(selected_problems) + 1,
                    expected_sql_count=1,
                )
                if cloned.question_id in selected_ids:
                    continue
                selected_problems.append(cloned)
                selected_ids.add(cloned.question_id)
                backfill_count += 1

        finalized_templates.append(
            TemplatePlanRecord(
                template_id=plan.template_id,
                template_name=plan.template_name,
                source_workload_id=plan.source_workload_id,
                primary_family=plan.primary_family,
                activation_tier=plan.activation_tier,
                dialect_sensitive=plan.dialect_sensitive,
                portability=plan.portability,
                portability_reason=plan.portability_reason,
                review_flag=plan.review_flag,
                rank=plan.rank,
                can_vary=plan.can_vary,
                must_fix=plan.must_fix,
                base_bindings=plan.base_bindings,
                selected_reason=plan.selected_reason,
                target_problem_min=min_problems_per_template,
                target_problem_max=max_problems_per_template,
                generated_problem_count=len(selected_problems),
                candidate_problem_count=max(plan.candidate_problem_count, len(generated_specs)),
                loop_stats={
                    **dict(plan.loop_stats),
                    "cli_generated_problem_candidates": len(generated_specs),
                    "cli_generated_problem_invalid": invalid_count,
                    "cli_generated_problem_valid": len(selected_problems) - backfill_count,
                    "cli_generated_problem_backfill": backfill_count,
                    "final_selected_problems": len(selected_problems),
                },
                problems=selected_problems[:max_problems_per_template],
                runtime_sql_skeleton=plan.runtime_sql_skeleton,
                selection_mode=f"{plan.selection_mode}+cli_generated_problems",
            )
        )

    items: list[QuestionInventoryItem] = []
    for plan in finalized_templates:
        items.extend(plan.problems)

    family_counts = Counter(item.primary_family for item in items)
    template_problem_counts = {plan.template_id: plan.generated_problem_count for plan in finalized_templates}
    selected_template_ids = [plan.template_id for plan in finalized_templates]

    return {
        "dataset_id": dataset_id,
        "row_count": sqlite_result.row_count,
        "candidate_pool_count": len(spec.get(spec_bucket, [])),
        "template_candidate_count": len(template_plans),
        "selected_template_count": len(finalized_templates),
        "problem_count": len(items),
        "inventory_count": len(items),
        "family_counts": dict(family_counts),
        "template_ids": selected_template_ids,
        "problem_generation_policy": {
            "min_templates": min_templates,
            "target_templates": target_templates,
            "min_problems_per_template": min_problems_per_template,
            "max_problems_per_template": max_problems_per_template,
            "expected_sql_per_problem": 1,
            "planner_model": planner_model,
            "planner_mode": "cli_all_ai_generation_with_heuristic_reference_backfill",
            "planner_preset": ai_cli_preset,
            "policy_fields_planned": ["can_vary", "must_fix"],
            "policy_materialization_status": "validated_by_python_after_cli_generation",
            "adaptive_threshold_policy": adaptive_policy,
        },
        "planner_summary": planner.summary,
        "selected_templates": [
            {
                "template_id": plan.template_id,
                "template_name": plan.template_name,
                "source_workload_id": plan.source_workload_id,
                "primary_family": plan.primary_family,
                "activation_tier": plan.activation_tier,
                "dialect_sensitive": plan.dialect_sensitive,
                "portability": plan.portability,
                "portability_reason": plan.portability_reason,
                "review_flag": plan.review_flag,
                "rank": plan.rank,
                "can_vary": plan.can_vary,
                "must_fix": plan.must_fix,
                "base_bindings": plan.base_bindings,
                "selected_reason": plan.selected_reason,
                "target_problem_min": plan.target_problem_min,
                "target_problem_max": plan.target_problem_max,
                "generated_problem_count": plan.generated_problem_count,
                "candidate_problem_count": plan.candidate_problem_count,
                "loop_stats": plan.loop_stats,
                "runtime_sql_skeleton": plan.runtime_sql_skeleton,
                "selection_mode": plan.selection_mode,
            }
            for plan in finalized_templates
        ],
        "problem_counts_by_template": template_problem_counts,
        "items": [
            {
                "question_id": item.question_id,
                "dataset_id": item.dataset_id,
                "template_id": item.template_id,
                "template_name": item.template_name,
                "question": item.question,
                "bindings": item.bindings,
                "portability": item.portability,
                "failure_reason": item.failure_reason,
                "review_flag": item.review_flag,
                "source_workload_id": item.source_workload_id,
                "primary_family": item.primary_family,
                "activation_tier": item.activation_tier,
                "dialect_sensitive": item.dialect_sensitive,
                "rank": item.rank,
                "notes": item.notes,
                "problem_index_within_template": item.problem_index_within_template,
                "variation_axes": item.variation_axes,
                "can_vary": item.can_vary,
                "must_fix": item.must_fix,
                "expected_sql_count": item.expected_sql_count,
                "runtime_sql_skeleton": item.runtime_sql_skeleton,
            }
            for item in items
        ],
        "invalid_problem_records": invalid_problem_records,
        "skipped": skipped,
    }
