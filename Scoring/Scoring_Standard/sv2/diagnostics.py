"""Diagnostics shared by SV2 scorers. Diagnostics never change the primary score."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

MAX_PER_KEY_DIAGNOSTICS = 50


def format_key(key: Any) -> Any:
    """JSON-friendly rendering of a canonical key or ranking identity."""
    if isinstance(key, tuple):
        return [format_key(part) for part in key]
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key
    return repr(key)


def key_overlap(real_keys: Iterable[Hashable], syn_keys: Iterable[Hashable]) -> dict[str, Any]:
    real = set(real_keys)
    syn = set(syn_keys)
    intersection = len(real & syn)
    precision = intersection / len(syn) if syn else (1.0 if not real else 0.0)
    recall = intersection / len(real) if real else (1.0 if not syn else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "real_key_count": len(real),
        "synthetic_key_count": len(syn),
        "union_key_count": len(real | syn),
        "intersection_key_count": intersection,
        "key_precision": precision,
        "key_recall": recall,
        "key_f1": f1,
    }
