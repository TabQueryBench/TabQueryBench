"""Scorer 3 — Rate / Share / Proportion: 1 - mean absolute error on the [0, 1] scale."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

from .common import RANGE_TOL, ScoringError, checked_unit_interval, require_finite_number, score_union_keyed

RATE_SCALES = {"unit": 1.0, "percent": 100.0}


def validate_and_normalize_rate(value: Any, *, scale: str = "unit") -> float:
    """Convert an explicitly declared scale to [0, 1] and reject material range violations."""
    if scale not in RATE_SCALES:
        raise ScoringError("UNKNOWN_RATE_SCALE", f"Unknown rate scale {scale!r}.")
    rate = require_finite_number(value, context="rate value") / RATE_SCALES[scale]
    if rate < -RANGE_TOL or rate > 1.0 + RANGE_TOL:
        raise ScoringError("RATE_OUT_OF_RANGE", f"Rate {value!r} (scale={scale}) is outside [0, 1].")
    return min(1.0, max(0.0, rate))


def rate_similarity(real: float, syn: float) -> float:
    return checked_unit_interval(1.0 - abs(validate_and_normalize_rate(real) - validate_and_normalize_rate(syn)))


def _normalize_map(answer: Mapping[Hashable, Any], scale: str) -> dict[Hashable, float | None]:
    return {key: None if value is None else validate_and_normalize_rate(value, scale=scale) for key, value in answer.items()}


def score_rate_share_proportion(
    real_map: Mapping[Hashable, Any],
    syn_map: Mapping[Hashable, Any],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    scale = str((metadata or {}).get("rate_scale") or "unit")
    real = _normalize_map(real_map, scale)
    syn = _normalize_map(syn_map, scale)
    score, diagnostics = score_union_keyed(real, syn, rate_similarity)
    diagnostics["rate_scale"] = scale
    diagnostics["raw_mae"] = diagnostics.pop("mean_abs_error")
    diagnostics["max_absolute_rate_error"] = diagnostics.pop("max_abs_error")
    diagnostics["mean_percentage_point_error"] = (
        diagnostics["raw_mae"] * 100.0 if diagnostics["raw_mae"] is not None else None
    )
    return score, diagnostics
