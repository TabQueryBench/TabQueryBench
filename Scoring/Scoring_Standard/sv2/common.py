"""Shared constants, errors, and numeric helpers for SV2 scoring."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any

SCORE_METHOD = "sv2"
CONTRACT_VERSION = "sv2_four_primary_v1"

NUMERIC_MAGNITUDE = "numeric_magnitude"
COUNT_SUPPORT_DISTRIBUTION = "count_support_distribution"
RATE_SHARE_PROPORTION = "rate_share_proportion"
TOPK_RANKING = "topk_ranking"
SCORER_TYPES = (NUMERIC_MAGNITUDE, COUNT_SUPPORT_DISTRIBUTION, RATE_SHARE_PROPORTION, TOPK_RANKING)

METRIC_IDS = {
    NUMERIC_MAGNITUDE: "normalized_smape_similarity",
    COUNT_SUPPORT_DISTRIBUTION: "one_minus_tvd",
    RATE_SHARE_PROPORTION: "one_minus_mae",
    TOPK_RANKING: "rbo_p09",
}

# Used only to classify floating-point zeros; never added to a denominator.
ZERO_TOL = 1e-12
# Floating-point noise allowed around the [0, 1] rate range before rejecting a value.
RANGE_TOL = 1e-12
# Floating-point drift allowed around the final [0, 1] score before raising.
SCORE_TOL = 1e-12
RBO_PERSISTENCE = 0.9

SCALAR_KEY: tuple[str] = ("__scalar__",)


class ScoringError(Exception):
    """A scorer input that SV2 cannot evaluate (reported as valid=false, never as a zero score)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ScoreContractViolation(ScoringError):
    def __init__(self, value: float) -> None:
        super().__init__("SCORE_OUT_OF_RANGE", f"Score {value!r} is outside [0, 1].")


def metric_id_for(scorer_type: str | None) -> str | None:
    return METRIC_IDS.get(scorer_type or "")


def checked_unit_interval(value: float) -> float:
    """Postcondition for every scorer: clamp tiny drift, raise on real violations."""
    x = float(value)
    if not math.isfinite(x) or x < -SCORE_TOL or x > 1.0 + SCORE_TOL:
        raise ScoreContractViolation(x)
    return min(1.0, max(0.0, x))


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require_finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringError("NON_NUMERIC_VALUE", f"{context}: expected a number, got {value!r}.")
    if not math.isfinite(value):
        raise ScoringError("NON_FINITE_NUMERIC_VALUE", f"{context}: non-finite value {value!r}.")
    return float(value)


def compare_nullable_numeric(real_value: Any, syn_value: Any, comparator: Callable[[Any, Any], float]) -> float:
    """SQL NULL vs NULL agrees (1); NULL vs a number disagrees (0). NaN/Inf are not NULL."""
    real_null = real_value is None
    syn_null = syn_value is None
    if real_null and syn_null:
        return 1.0
    if real_null != syn_null:
        return 0.0
    return comparator(real_value, syn_value)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return math.fsum(items) / len(items)


def sorted_keys(keys: Iterable[Hashable]) -> list[Hashable]:
    return sorted(keys, key=repr)


def validate_nullable_numbers(answer: Mapping[Hashable, Any], *, side: str) -> None:
    for key, value in answer.items():
        if value is not None:
            require_finite_number(value, context=f"{side} value for key {key!r}")


def score_union_keyed(
    real_map: Mapping[Hashable, Any],
    syn_map: Mapping[Hashable, Any],
    similarity: Callable[[Any, Any], float],
) -> tuple[float, dict[str, Any]]:
    """Macro mean of per-key similarity over the union key space; unmatched keys score 0."""
    from .diagnostics import MAX_PER_KEY_DIAGNOSTICS, format_key, key_overlap

    keys = sorted_keys(set(real_map) | set(syn_map))
    diagnostics: dict[str, Any] = key_overlap(real_map.keys(), syn_map.keys())
    if not keys:
        diagnostics.update(
            {"matched_key_count": 0, "unmatched_key_count": 0, "mean_abs_error": None, "max_abs_error": None, "per_key": []}
        )
        return 1.0, diagnostics

    scores: list[float] = []
    abs_errors: list[float] = []
    per_key: list[dict[str, Any]] = []
    for key in keys:
        if key not in real_map or key not in syn_map:
            per_key_score = 0.0
            abs_error = None
        else:
            real_value, syn_value = real_map[key], syn_map[key]
            per_key_score = compare_nullable_numeric(real_value, syn_value, similarity)
            abs_error = abs(float(real_value) - float(syn_value)) if real_value is not None and syn_value is not None else None
        scores.append(per_key_score)
        if abs_error is not None:
            abs_errors.append(abs_error)
        if len(per_key) < MAX_PER_KEY_DIAGNOSTICS:
            per_key.append({"key": format_key(key), "similarity": per_key_score, "abs_error": abs_error})

    matched = sum(1 for key in keys if key in real_map and key in syn_map)
    diagnostics.update(
        {
            "matched_key_count": matched,
            "unmatched_key_count": len(keys) - matched,
            "mean_abs_error": mean(abs_errors) if abs_errors else None,
            "max_abs_error": max(abs_errors) if abs_errors else None,
            "per_key": per_key,
            "per_key_truncated": len(keys) > MAX_PER_KEY_DIAGNOSTICS,
        }
    )
    return checked_unit_interval(mean(scores)), diagnostics
