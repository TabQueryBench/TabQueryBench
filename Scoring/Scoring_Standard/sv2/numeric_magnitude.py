"""Scorer 1 — Numeric Magnitude: normalized sMAPE similarity."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

from .common import (
    ZERO_TOL,
    ScoringError,
    checked_unit_interval,
    is_finite_number,
    score_union_keyed,
    validate_nullable_numbers,
)


def normalized_smape_similarity(real: float, syn: float, zero_tol: float = ZERO_TOL) -> float:
    """1 - |r - s| / (|r| + |s|), with 0 vs 0 -> 1 and 0 vs nonzero -> 0."""
    if not is_finite_number(real) or not is_finite_number(syn):
        raise ScoringError("NON_FINITE_NUMERIC_VALUE", f"Numeric magnitude requires finite values, got {real!r} and {syn!r}.")
    r = float(real)
    s = float(syn)
    r_zero = abs(r) <= zero_tol
    s_zero = abs(s) <= zero_tol
    if r_zero and s_zero:
        return 1.0
    if r_zero != s_zero:
        return 0.0
    return checked_unit_interval(1.0 - abs(r - s) / (abs(r) + abs(s)))


def score_numeric_magnitude(
    real_map: Mapping[Hashable, Any],
    syn_map: Mapping[Hashable, Any],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    validate_nullable_numbers(real_map, side="real")
    validate_nullable_numbers(syn_map, side="synthetic")
    return score_union_keyed(real_map, syn_map, normalized_smape_similarity)
