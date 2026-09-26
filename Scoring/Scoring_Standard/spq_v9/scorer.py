"""Single-primary semantic query scoring for real-vs-synthetic SQL outputs.

This module is intentionally additive: it does not replace the legacy
``query_score`` contract. Callers can attach ``semantic_query_score`` and the
component diagnostics to query rows while keeping legacy scoring unchanged.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any


EPS = 1e-12
SEMANTIC_QUERY_SCORE_METHOD = "spq"
SCORE_CONTRACT_VERSION = "spq_v9_single_primary"
RBO_PERSISTENCE = 0.9

SUPPORT_NAMES = {
    "count",
    "cnt",
    "row_count",
    "support",
    "event_count",
    "total_rows",
    "missing_rows",
    "current_count",
    "previous_count",
    "non_null_rows",
}
RATE_TOKENS = ("rate", "ratio", "share", "pct", "percent", "prob", "cdf")
NUMERIC_TOKENS = (
    "avg",
    "mean",
    "sum",
    "total",
    "min",
    "max",
    "median",
    "percentile",
    "quantile",
    "std",
    "var",
    "measure",
    "value",
    "score",
)


@dataclass(frozen=True)
class ScoringPolicy:
    scorer_type: str
    key_columns: tuple[str, ...] = ()
    measure_columns: tuple[str, ...] = ()
    support_column: str = ""
    rate_columns: tuple[str, ...] = ()
    selection_mode: str = ""
    key_match_mode: str = "categorical_exact"


def _clip01(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _cell(value: Any) -> str:
    if value is None:
        return "<NULL>"
    return str(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:  # noqa: BLE001
        return None


def _name_has(name: str, tokens: tuple[str, ...] | set[str]) -> bool:
    lower = str(name or "").lower()
    return any(token in lower for token in tokens)


def _column_indices(columns: list[str], wanted: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    remaining: dict[str, list[int]] = {}
    for idx, col in enumerate(columns):
        remaining.setdefault(str(col), []).append(idx)
    for name in wanted:
        options = remaining.get(str(name)) or []
        if options:
            out.append(options.pop(0))
    return out


def _infer_measure_columns(columns: list[str]) -> tuple[str, ...]:
    measures = []
    for col in columns:
        lower = str(col).lower()
        if lower in SUPPORT_NAMES or _name_has(lower, RATE_TOKENS) or _name_has(lower, NUMERIC_TOKENS):
            measures.append(str(col))
    return tuple(measures)


def _infer_support_column(columns: list[str]) -> str:
    for col in columns:
        lower = str(col).lower()
        if lower in SUPPORT_NAMES or any(token in lower for token in ("count", "support", "rows")):
            return str(col)
    return ""


def _infer_rate_columns(columns: list[str]) -> tuple[str, ...]:
    return tuple(str(col) for col in columns if _name_has(str(col), RATE_TOKENS))


def _infer_key_columns(columns: list[str], measure_columns: tuple[str, ...]) -> tuple[str, ...]:
    measure_set = set(measure_columns)
    keys = tuple(str(col) for col in columns if str(col) not in measure_set)
    if keys:
        return keys
    if len(columns) >= 2:
        return tuple(str(col) for col in columns[:-1])
    return ()


def _policy_from_query(query: dict[str, Any], real_columns: list[str]) -> ScoringPolicy:
    template_id = str(query.get("template_id") or "")
    template_name = str(query.get("template_name") or "").lower()
    sql = str(query.get("sql") or "").lower()
    contract = query.get("semantic_result_contract")
    contract = contract if isinstance(contract, dict) else {}
    declared_scorer = str(contract.get("scorer_type") or query.get("scorer_type") or "").strip()
    measure_cols = _infer_measure_columns(real_columns)
    support_col = _infer_support_column(real_columns)
    rate_cols = _infer_rate_columns(real_columns)
    key_cols = _infer_key_columns(real_columns, measure_cols)

    def policy(
        scorer_type: str,
        *,
        selection_mode: str = "",
        key_match_mode: str = "categorical_exact",
    ) -> ScoringPolicy:
        return ScoringPolicy(
            scorer_type=scorer_type,
            key_columns=key_cols,
            measure_columns=measure_cols,
            support_column=support_col,
            rate_columns=rate_cols,
            selection_mode=selection_mode,
            key_match_mode=key_match_mode,
        )

    scorer_aliases = {
        "scalar": ("scalar", ""),
        "scalar_rate": ("scalar", ""),
        "scalar_numeric": ("scalar", ""),
        "count_support_distribution": ("count_support_distribution", ""),
        "distribution_cardinality_profile": ("count_support_distribution", ""),
        "keyed_count_distribution": ("count_support_distribution", ""),
        "rate_share_proportion": ("rate_share_proportion", ""),
        "keyed_rate_profile": ("rate_share_proportion", ""),
        "missingness_group_rate": ("rate_share_proportion", ""),
        "missingness_interaction_rate": ("rate_share_proportion", ""),
        "ratio": ("ratio", ""),
        "keyed_ratio_profile": ("ratio", ""),
        "keyed_numeric_aggregate": ("keyed_numeric_aggregate", ""),
        "generic_keyed_result": ("keyed_numeric_aggregate", ""),
        "topk_ranking": ("topk_ranking", "topk"),
        "topk_tailk_ranking": ("topk_ranking", "topk"),
        "topk_ranked_measure": ("topk_ranking", "topk"),
        "argmax_selection": ("topk_ranking", "winner"),
        "tail_topn_value_curve": ("topk_ranking", "topk"),
        "missingness_scalar_rate": ("scalar", ""),
    }
    if declared_scorer in scorer_aliases:
        scorer_type, selection_mode = scorer_aliases[declared_scorer]
        return policy(scorer_type, selection_mode=selection_mode)

    if "zscore" in template_id or "outlier" in template_name or "quantile_tail" in template_id:
        if "top" in template_id or "rank" in template_name or "limit" in sql:
            return policy("topk_ranking", selection_mode="topk", key_match_mode="raw_numeric_distribution")
        if support_col and len(measure_cols) == 1 and measure_cols[0] == support_col:
            return policy("count_support_distribution", key_match_mode="raw_numeric_distribution")
        if not key_cols or len(real_columns) == 1:
            return policy("scalar", key_match_mode="none")
        return policy("keyed_numeric_aggregate", key_match_mode="raw_numeric_distribution")
    if "time_bucket" in template_id or "moving_avg" in template_id or "drift" in template_id:
        if rate_cols:
            return policy("rate_share_proportion", selection_mode="time_bucket", key_match_mode="deterministic_bucket")
        if support_col and len(measure_cols) == 1 and measure_cols[0] == support_col:
            return policy("count_support_distribution", selection_mode="time_bucket", key_match_mode="deterministic_bucket")
        return policy("keyed_numeric_aggregate", selection_mode="time_bucket", key_match_mode="deterministic_bucket")
    if "topk" in template_id or "top-k" in template_name or "ranking" in template_name or "rank" in template_name or "limit" in sql:
        if "winner" in template_id or "winner" in template_name:
            return policy("topk_ranking", selection_mode="winner")
        return policy("topk_ranking", selection_mode="topk")
    if "cardinality" in template_id or "support_rank_profile" in template_id or "distinct_share_profile" in template_id:
        if len(real_columns) <= 3 and ("min" in ",".join(real_columns).lower() and "max" in ",".join(real_columns).lower()):
            return policy("scalar", key_match_mode="none")
        return policy("count_support_distribution")
    if rate_cols:
        if any("ratio" in col.lower() for col in rate_cols):
            return policy("ratio")
        if not key_cols:
            return policy("scalar", key_match_mode="none")
        return policy("rate_share_proportion")
    if not key_cols or len(real_columns) == 1:
        return policy("scalar", key_match_mode="none")
    if support_col and len(measure_cols) == 1 and measure_cols[0] == support_col:
        return policy("count_support_distribution")
    if "count(" in sql and support_col and not any(_name_has(col, NUMERIC_TOKENS) for col in measure_cols if col != support_col):
        return policy("count_support_distribution")
    if support_col:
        return policy("keyed_numeric_aggregate")
    return policy("keyed_numeric_aggregate")


def _missing_required_outputs(
    columns: list[str],
    query: dict[str, Any],
    *,
    key_columns_present: bool = False,
) -> list[str]:
    contract = query.get("semantic_result_contract")
    if not isinstance(contract, dict):
        return []
    available = {str(column).lower() for column in columns}
    key_roles = {str(role).lower() for role in contract.get("key_roles") or []}
    missing: list[str] = []
    for name in contract.get("validity_required_outputs") or []:
        if str(name).lower() in key_roles:
            if not key_columns_present:
                missing.append(str(name))
        elif str(name).lower() not in available:
            missing.append(str(name))
    return missing


def _key_tuple(row: list[Any], indices: list[int]) -> tuple[str, ...]:
    return tuple(_cell(row[idx]) if idx < len(row) else "<MISSING_COL>" for idx in indices)


def _key_set(rows: list[list[Any]], key_indices: list[int]) -> set[tuple[str, ...]]:
    if not key_indices:
        return set()
    return {_key_tuple(row, key_indices) for row in rows}


def _f1(real_items: set[Any], syn_items: set[Any]) -> float:
    if not real_items and not syn_items:
        return 1.0
    if not real_items or not syn_items:
        return 0.0
    inter = len(real_items & syn_items)
    precision = inter / max(1, len(syn_items))
    recall = inter / max(1, len(real_items))
    if precision + recall <= 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _symmetric_numeric_similarity(real: float | None, syn: float | None) -> float:
    if real is None or syn is None:
        return 0.0
    if abs(real) <= EPS and abs(syn) <= EPS:
        return 1.0
    return _clip01(1.0 - (abs(real - syn) / max(abs(real), abs(syn), EPS)))


def _distribution_from_counter(counter: Counter) -> dict[Any, float]:
    total = float(sum(max(0.0, float(value)) for value in counter.values()))
    if total <= EPS:
        return {}
    return {key: max(0.0, float(value)) / total for key, value in counter.items()}


def _tvd_similarity(real_dist: dict[Any, float], syn_dist: dict[Any, float]) -> float:
    keys = set(real_dist) | set(syn_dist)
    if not keys:
        return 1.0
    tvd = 0.5 * sum(abs(float(real_dist.get(key, 0.0)) - float(syn_dist.get(key, 0.0))) for key in keys)
    return _clip01(1.0 - tvd)


def _measure_maps(
    rows: list[list[Any]],
    key_indices: list[int],
    measure_indices: list[int],
) -> dict[tuple[str, ...], list[float]]:
    out: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        key = _key_tuple(row, key_indices) if key_indices else ("__scalar__",)
        values = []
        for idx in measure_indices:
            values.append(_to_float(row[idx]) if idx < len(row) else None)
        numeric = [value for value in values if value is not None]
        if numeric:
            out[key] = numeric
    return out


def _key_f1(real_rows: list[list[Any]], syn_rows: list[list[Any]], key_indices: list[int]) -> float:
    if not key_indices:
        return 1.0
    return _f1(_key_set(real_rows, key_indices), _key_set(syn_rows, key_indices))


def _support_distribution_similarity(
    real_rows: list[list[Any]],
    syn_rows: list[list[Any]],
    key_indices: list[int],
    support_idx: int | None,
) -> float:
    if support_idx is None:
        return _key_f1(real_rows, syn_rows, key_indices)
    real_counter: Counter = Counter()
    syn_counter: Counter = Counter()
    for row in real_rows:
        key = _key_tuple(row, key_indices) if key_indices else ("__total__",)
        real_counter[key] += max(0.0, _to_float(row[support_idx]) or 0.0)
    for row in syn_rows:
        key = _key_tuple(row, key_indices) if key_indices else ("__total__",)
        syn_counter[key] += max(0.0, _to_float(row[support_idx]) or 0.0)
    return _tvd_similarity(_distribution_from_counter(real_counter), _distribution_from_counter(syn_counter))


def _mass_similarity(
    real_rows: list[list[Any]],
    syn_rows: list[list[Any]],
    support_idx: int | None,
) -> float:
    if support_idx is None:
        return _symmetric_numeric_similarity(float(len(real_rows)), float(len(syn_rows)))
    real_total = sum(max(0.0, _to_float(row[support_idx]) or 0.0) for row in real_rows if support_idx < len(row))
    syn_total = sum(max(0.0, _to_float(row[support_idx]) or 0.0) for row in syn_rows if support_idx < len(row))
    return _symmetric_numeric_similarity(real_total, syn_total)


def _aligned_numeric_similarity(
    real_rows: list[list[Any]],
    syn_rows: list[list[Any]],
    key_indices: list[int],
    measure_indices: list[int],
) -> float:
    real_map = _measure_maps(real_rows, key_indices, measure_indices)
    syn_map = _measure_maps(syn_rows, key_indices, measure_indices)
    keys = set(real_map) | set(syn_map)
    if not keys:
        return 0.0 if (real_map or syn_map) else 1.0
    values: list[float] = []
    for key in keys:
        if key not in real_map or key not in syn_map:
            values.append(0.0)
            continue
        real_values = real_map[key]
        syn_values = syn_map[key]
        for idx in range(max(len(real_values), len(syn_values))):
            real_value = real_values[idx] if idx < len(real_values) else None
            syn_value = syn_values[idx] if idx < len(syn_values) else None
            values.append(_symmetric_numeric_similarity(real_value, syn_value))
    return sum(values) / len(values) if values else 0.0


def _rate_similarity(
    real_rows: list[list[Any]],
    syn_rows: list[list[Any]],
    key_indices: list[int],
    rate_indices: list[int],
) -> float:
    real_map = _measure_maps(real_rows, key_indices, rate_indices)
    syn_map = _measure_maps(syn_rows, key_indices, rate_indices)
    keys = set(real_map) | set(syn_map)
    if not keys:
        return 0.0 if (real_map or syn_map) else 1.0
    scores: list[float] = []
    for key in keys:
        if key not in real_map or key not in syn_map:
            scores.append(0.0)
            continue
        for idx in range(max(len(real_map[key]), len(syn_map[key]))):
            if idx >= len(real_map[key]) or idx >= len(syn_map[key]):
                scores.append(0.0)
            else:
                scores.append(_clip01(1.0 - abs(real_map[key][idx] - syn_map[key][idx])))
    return sum(scores) / len(scores) if scores else 0.0


def _ratio_value_similarity(real: float | None, syn: float | None) -> float:
    if real is None or syn is None:
        return 0.0
    real_abs = abs(real)
    syn_abs = abs(syn)
    if real_abs <= EPS and syn_abs <= EPS:
        return 1.0
    if real_abs <= EPS or syn_abs <= EPS:
        return 0.0
    return _clip01(min(real_abs, syn_abs) / max(real_abs, syn_abs))


def _ratio_similarity(
    real_rows: list[list[Any]],
    syn_rows: list[list[Any]],
    key_indices: list[int],
    measure_indices: list[int],
) -> float:
    real_map = _measure_maps(real_rows, key_indices, measure_indices)
    syn_map = _measure_maps(syn_rows, key_indices, measure_indices)
    keys = set(real_map) | set(syn_map)
    if not keys:
        return 0.0 if (real_map or syn_map) else 1.0
    scores: list[float] = []
    for key in keys:
        if key not in real_map or key not in syn_map:
            scores.append(0.0)
            continue
        for idx in range(max(len(real_map[key]), len(syn_map[key]))):
            real_value = real_map[key][idx] if idx < len(real_map[key]) else None
            syn_value = syn_map[key][idx] if idx < len(syn_map[key]) else None
            scores.append(_ratio_value_similarity(real_value, syn_value))
    return sum(scores) / len(scores) if scores else 0.0


def _direction_consistency(
    real_rows: list[list[Any]],
    syn_rows: list[list[Any]],
    key_indices: list[int],
    measure_indices: list[int],
) -> float:
    real_map = _measure_maps(real_rows, key_indices, measure_indices[:1])
    syn_map = _measure_maps(syn_rows, key_indices, measure_indices[:1])
    keys = sorted(set(real_map) & set(syn_map))
    if not keys:
        return 0.0 if (real_map or syn_map) else 1.0
    real_values = [real_map[key][0] for key in keys if real_map.get(key)]
    syn_values = [syn_map[key][0] for key in keys if syn_map.get(key)]
    if not real_values or not syn_values:
        return 0.0
    real_base = sum(real_values) / len(real_values)
    syn_base = sum(syn_values) / len(syn_values)
    agree = 0
    total = 0
    for key in keys:
        if not real_map.get(key) or not syn_map.get(key):
            continue
        real_dir = 1 if real_map[key][0] > real_base else (-1 if real_map[key][0] < real_base else 0)
        syn_dir = 1 if syn_map[key][0] > syn_base else (-1 if syn_map[key][0] < syn_base else 0)
        agree += 1 if real_dir == syn_dir else 0
        total += 1
    return agree / total if total else 0.0


def _ranking_items(rows: list[list[Any]], key_indices: list[int]) -> list[tuple[str, ...]]:
    if key_indices:
        return [_key_tuple(row, key_indices) for row in rows]
    return [tuple(_cell(value) for value in row) for row in rows]


def _rbo_similarity(
    real_rows: list[list[Any]],
    syn_rows: list[list[Any]],
    key_indices: list[int],
    *,
    persistence: float = RBO_PERSISTENCE,
) -> float:
    real_order = _ranking_items(real_rows, key_indices)
    syn_order = _ranking_items(syn_rows, key_indices)
    depth = max(len(real_order), len(syn_order))
    if depth == 0:
        return 1.0
    if not real_order or not syn_order:
        return 0.0
    seen_real: set[tuple[str, ...]] = set()
    seen_syn: set[tuple[str, ...]] = set()
    weighted_overlap = 0.0
    overlap_at_depth = 0.0
    for idx in range(depth):
        if idx < len(real_order):
            seen_real.add(real_order[idx])
        if idx < len(syn_order):
            seen_syn.add(syn_order[idx])
        overlap_at_depth = len(seen_real & seen_syn) / float(idx + 1)
        weighted_overlap += (persistence**idx) * overlap_at_depth
    score = ((1.0 - persistence) * weighted_overlap) + ((persistence**depth) * overlap_at_depth)
    return _clip01(score)


def _selected_overlap(real_rows: list[list[Any]], syn_rows: list[list[Any]], key_indices: list[int]) -> float:
    if not key_indices:
        return _symmetric_numeric_similarity(float(len(real_rows)), float(len(syn_rows)))
    return _f1(_key_set(real_rows, key_indices), _key_set(syn_rows, key_indices))


def _scalar_similarity(real_rows: list[list[Any]], syn_rows: list[list[Any]], measure_indices: list[int], rate: bool) -> float:
    if not measure_indices:
        return 0.0
    real_values = [_to_float(row[idx]) for row in real_rows for idx in measure_indices if idx < len(row)]
    syn_values = [_to_float(row[idx]) for row in syn_rows for idx in measure_indices if idx < len(row)]
    real_numeric = [value for value in real_values if value is not None]
    syn_numeric = [value for value in syn_values if value is not None]
    if not real_numeric or not syn_numeric:
        return 0.0
    scores: list[float] = []
    for idx in range(min(len(real_numeric), len(syn_numeric))):
        if rate:
            scores.append(_clip01(1.0 - abs(real_numeric[idx] - syn_numeric[idx])))
        else:
            scores.append(_symmetric_numeric_similarity(real_numeric[idx], syn_numeric[idx]))
    return sum(scores) / len(scores) if scores else 0.0


def _scalar_component_scores(
    real_rows: list[list[Any]],
    syn_rows: list[list[Any]],
    measure_indices: list[int],
    columns: list[str],
    *,
    rate: bool,
) -> dict[str, float]:
    components: dict[str, float] = {}
    for idx in measure_indices:
        real_values = [_to_float(row[idx]) for row in real_rows if idx < len(row)]
        syn_values = [_to_float(row[idx]) for row in syn_rows if idx < len(row)]
        real_numeric = [value for value in real_values if value is not None]
        syn_numeric = [value for value in syn_values if value is not None]
        if not real_numeric or not syn_numeric:
            score = 0.0
        elif rate:
            score = _clip01(1.0 - abs(real_numeric[0] - syn_numeric[0]))
        else:
            score = _symmetric_numeric_similarity(real_numeric[0], syn_numeric[0])
        name = str(columns[idx]) if idx < len(columns) else f"measure_{idx}"
        components[f"{name}_similarity"] = score
    if not components:
        components["scalar_similarity"] = 0.0
    return components


def _quantiles(values: list[float], probs: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    if len(ordered) == 1:
        return [ordered[0] for _ in probs]
    out: list[float] = []
    for prob in probs:
        pos = prob * (len(ordered) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.append(ordered[lo])
        else:
            frac = pos - lo
            out.append((ordered[lo] * (1.0 - frac)) + (ordered[hi] * frac))
    return out


def _tail_distribution_similarity(
    real_rows: list[list[Any]],
    syn_rows: list[list[Any]],
    measure_indices: list[int],
) -> float:
    if not measure_indices:
        return 0.0
    scores: list[float] = []
    for idx in measure_indices:
        real_values = [_to_float(row[idx]) for row in real_rows if idx < len(row)]
        syn_values = [_to_float(row[idx]) for row in syn_rows if idx < len(row)]
        real_numeric = [value for value in real_values if value is not None]
        syn_numeric = [value for value in syn_values if value is not None]
        real_q = _quantiles(real_numeric)
        syn_q = _quantiles(syn_numeric)
        for q_idx in range(min(len(real_q), len(syn_q))):
            scores.append(_symmetric_numeric_similarity(real_q[q_idx], syn_q[q_idx]))
    return sum(scores) / len(scores) if scores else 0.0


def _tail_boundary_similarity(
    real_rows: list[list[Any]],
    syn_rows: list[list[Any]],
    measure_indices: list[int],
) -> float:
    if not measure_indices:
        return 0.0
    idx = measure_indices[0]
    real_values = [_to_float(row[idx]) for row in real_rows if idx < len(row)]
    syn_values = [_to_float(row[idx]) for row in syn_rows if idx < len(row)]
    real_numeric = [value for value in real_values if value is not None]
    syn_numeric = [value for value in syn_values if value is not None]
    if not real_numeric or not syn_numeric:
        return 0.0
    return _symmetric_numeric_similarity(min(real_numeric), min(syn_numeric))


def compare_semantic_execution_results(
    real_exec: Any,
    syn_exec: Any,
    *,
    query: dict[str, Any] | None = None,
    legacy_detail: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    query = query or {}
    if not getattr(real_exec, "ok", False):
        return 0.0, {
            "semantic_query_score_method": SEMANTIC_QUERY_SCORE_METHOD,
            "validity_status": "invalid",
            "validity_failures": ["real_query_failed"],
            "scorer_type": "invalid",
            "component_scores": {},
        }
    if not getattr(syn_exec, "ok", False):
        return 0.0, {
            "semantic_query_score_method": SEMANTIC_QUERY_SCORE_METHOD,
            "validity_status": "invalid",
            "validity_failures": ["synthetic_query_failed"],
            "scorer_type": "invalid",
            "component_scores": {},
        }

    real_columns = [str(col) for col in getattr(real_exec, "columns", [])]
    syn_columns = [str(col) for col in getattr(syn_exec, "columns", [])]
    if not real_columns or not syn_columns:
        return 0.0, {
            "semantic_query_score_method": SEMANTIC_QUERY_SCORE_METHOD,
            "validity_status": "invalid",
            "validity_failures": ["missing_columns"],
            "scorer_type": "invalid",
            "component_scores": {},
        }

    policy = _policy_from_query(query, real_columns)
    key_indices = _column_indices(real_columns, policy.key_columns)
    measure_indices = _column_indices(real_columns, policy.measure_columns)
    support_indices = _column_indices(real_columns, (policy.support_column,)) if policy.support_column else []
    rate_indices = _column_indices(real_columns, policy.rate_columns)
    if not measure_indices:
        measure_indices = [idx for idx in range(len(real_columns)) if idx not in set(key_indices)]
    support_idx = support_indices[0] if support_indices else None
    required_failures = _missing_required_outputs(real_columns, query, key_columns_present=bool(key_indices))
    required_failures.extend(
        f"synthetic_missing:{name}"
        for name in _missing_required_outputs(syn_columns, query, key_columns_present=bool(key_indices))
    )
    if required_failures:
        return 0.0, {
            "semantic_query_score_method": SEMANTIC_QUERY_SCORE_METHOD,
            "semantic_normalized_score": 0.0,
            "score_version": SCORE_CONTRACT_VERSION,
            "score_contract_version": SCORE_CONTRACT_VERSION,
            "validity_status": "invalid",
            "validity_failures": [f"required_output_missing:{name}" for name in required_failures],
            "scorer_type": "invalid",
            "component_scores": {},
        }
    real_rows = list(getattr(real_exec, "rows", []) or [])
    syn_rows = list(getattr(syn_exec, "rows", []) or [])

    components: dict[str, float] = {}
    primary_metric = ""
    scorer = policy.scorer_type
    if scorer == "scalar":
        is_rate = bool(rate_indices) or any("rate" in col.lower() or "cdf" in col.lower() for col in policy.measure_columns)
        components["scalar_similarity"] = _scalar_similarity(real_rows, syn_rows, measure_indices, is_rate)
        primary_metric = "scalar_similarity"
    elif scorer == "count_support_distribution":
        components["support_distribution_similarity"] = _support_distribution_similarity(real_rows, syn_rows, key_indices, support_idx)
        primary_metric = "support_distribution_similarity"
        components["total_mass_similarity"] = _mass_similarity(real_rows, syn_rows, support_idx)
        components["key_f1"] = _key_f1(real_rows, syn_rows, key_indices)
    elif scorer == "rate_share_proportion":
        active_rate_indices = rate_indices or measure_indices
        components["rate_similarity"] = _rate_similarity(real_rows, syn_rows, key_indices, active_rate_indices)
        primary_metric = "rate_similarity"
        components["direction_consistency"] = _direction_consistency(real_rows, syn_rows, key_indices, active_rate_indices)
        components["key_f1"] = _key_f1(real_rows, syn_rows, key_indices)
    elif scorer == "ratio":
        components["ratio_similarity"] = _ratio_similarity(real_rows, syn_rows, key_indices, measure_indices)
        primary_metric = "ratio_similarity"
        components["direction_or_threshold_consistency"] = _direction_consistency(real_rows, syn_rows, key_indices, measure_indices)
        components["key_f1"] = _key_f1(real_rows, syn_rows, key_indices)
    elif scorer == "keyed_numeric_aggregate":
        components["numeric_measure_similarity"] = _aligned_numeric_similarity(real_rows, syn_rows, key_indices, measure_indices)
        primary_metric = "numeric_measure_similarity"
        components["key_f1"] = _key_f1(real_rows, syn_rows, key_indices)
        if support_idx is not None:
            components["support_or_evidence_similarity"] = _mass_similarity(real_rows, syn_rows, support_idx)
    elif scorer == "topk_ranking":
        components["rbo_similarity"] = _rbo_similarity(real_rows, syn_rows, key_indices)
        primary_metric = "rbo_similarity"
        components["selected_set_overlap"] = _selected_overlap(real_rows, syn_rows, key_indices)
        components["measure_similarity"] = _aligned_numeric_similarity(real_rows, syn_rows, key_indices, measure_indices)
    else:
        components["strict_set_fallback"] = float((legacy_detail or {}).get("strict_set_score") or (legacy_detail or {}).get("set_score") or 0.0)
        primary_metric = "strict_set_fallback"

    score = _clip01(components.get(primary_metric, 0.0))
    detail = {
        "semantic_query_score_method": SEMANTIC_QUERY_SCORE_METHOD,
        "semantic_normalized_score": round(score, 6),
        "score_version": SCORE_CONTRACT_VERSION,
        "score_contract_version": SCORE_CONTRACT_VERSION,
        "validity_status": "ok",
        "validity_failures": [],
        "scorer_type": scorer,
        "primary_metric": primary_metric,
        "primary_target": scorer,
        "rbo_persistence": RBO_PERSISTENCE if scorer == "topk_ranking" else None,
        "component_scores": {key: round(_clip01(value), 6) for key, value in components.items()},
        "semantic_key_columns": [real_columns[idx] for idx in key_indices if idx < len(real_columns)],
        "semantic_measure_columns": [real_columns[idx] for idx in measure_indices if idx < len(real_columns)],
        "semantic_support_column": real_columns[support_idx] if support_idx is not None and support_idx < len(real_columns) else "",
        "selection_mode": policy.selection_mode,
        "key_match_mode": policy.key_match_mode,
        "legacy_query_score_method": (legacy_detail or {}).get("query_score_method"),
    }
    return score, detail
