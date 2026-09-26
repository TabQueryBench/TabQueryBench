"""Scorer 2 — Count / Support Distribution: 1 - total variation distance."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

from .common import ScoringError, checked_unit_interval, require_finite_number, sorted_keys
from .diagnostics import key_overlap


def _entries(counts: Mapping[Hashable, Any] | Iterable[tuple[Hashable, Any]]) -> Iterable[tuple[Hashable, Any]]:
    return counts.items() if isinstance(counts, Mapping) else counts


def validate_count(value: Any, *, side: str, key: Hashable) -> float:
    if value is None:
        raise ScoringError("NULL_COUNT", f"{side} count for key {key!r} is SQL NULL.")
    count = require_finite_number(value, context=f"{side} count for key {key!r}")
    if count < 0:
        raise ScoringError("NEGATIVE_COUNT", f"{side} count for key {key!r} is negative ({count!r}).")
    return count


def aggregate_duplicate_counts(
    counts: Mapping[Hashable, Any] | Iterable[tuple[Hashable, Any]], *, side: str = "answer"
) -> dict[Hashable, float]:
    """Validate every entry and sum counts of duplicate keys."""
    parts: dict[Hashable, list[float]] = {}
    for key, value in _entries(counts):
        parts.setdefault(key, []).append(validate_count(value, side=side, key=key))
    return {key: math.fsum(values) for key, values in parts.items()}


def score_count_support_distribution(
    real_counts: Mapping[Hashable, Any] | Iterable[tuple[Hashable, Any]],
    syn_counts: Mapping[Hashable, Any] | Iterable[tuple[Hashable, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    real = aggregate_duplicate_counts(real_counts, side="real")
    syn = aggregate_duplicate_counts(syn_counts, side="synthetic")
    real_total = math.fsum(real.values())
    syn_total = math.fsum(syn.values())

    diagnostics: dict[str, Any] = {
        **key_overlap(real.keys(), syn.keys()),
        "real_total_mass": real_total,
        "synthetic_total_mass": syn_total,
        "real_support_size": len(real),
        "synthetic_support_size": len(syn),
        "union_support_size": len(set(real) | set(syn)),
    }

    if real_total == 0 and syn_total == 0:
        diagnostics["raw_tvd"] = 0.0
        return 1.0, diagnostics
    if real_total == 0 or syn_total == 0:
        diagnostics["raw_tvd"] = 1.0
        return 0.0, diagnostics

    keys = sorted_keys(set(real) | set(syn))
    tvd = 0.5 * math.fsum(abs(real.get(key, 0.0) / real_total - syn.get(key, 0.0) / syn_total) for key in keys)
    diagnostics["raw_tvd"] = tvd
    return checked_unit_interval(1.0 - tvd), diagnostics
