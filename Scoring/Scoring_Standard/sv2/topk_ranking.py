"""Scorer 4 — Top-k Ranking: finite extrapolated rank-biased overlap (p = 0.9)."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from .common import RBO_PERSISTENCE, ScoringError, checked_unit_interval
from .diagnostics import MAX_PER_KEY_DIAGNOSTICS, format_key


class _Padding:
    """Side-specific sentinel identity; equal only to itself, so it never matches a real item."""

    __slots__ = ("side", "position")

    def __init__(self, side: str, position: int) -> None:
        self.side = side
        self.position = position

    def __repr__(self) -> str:
        return f"__{self.side.upper()}_PAD_{self.position}__"


def validate_unique_ranking_identities(ranking: Sequence[Hashable], *, side: str) -> None:
    seen: set[Hashable] = set()
    for item in ranking:
        if item in seen:
            raise ScoringError("DUPLICATE_RANKING_IDENTITY", f"{side} ranking repeats identity {item!r}.")
        seen.add(item)


def truncate_and_pad(ranking: Sequence[Hashable], depth: int, *, side: str) -> list[Hashable]:
    items = list(ranking[:depth])
    items.extend(_Padding(side, position) for position in range(len(items) + 1, depth + 1))
    return items


def _resolve_depth(real: Sequence[Hashable], syn: Sequence[Hashable], requested_depth: Any) -> int:
    """D = requested depth k, capped at the longer observed list.

    When both answers are shorter than k (e.g. fewer than k groups exist), padding both sides
    would make identical rankings score below 1, contradicting the identity guarantee; the
    cap keeps truncation to k and padding of the shorter list unchanged otherwise.
    """
    longest = max(len(real), len(syn))
    if requested_depth is None:
        return longest
    if isinstance(requested_depth, bool) or not isinstance(requested_depth, int) or requested_depth < 0:
        raise ScoringError("INVALID_REQUESTED_DEPTH", f"Requested ranking depth must be a non-negative int, got {requested_depth!r}.")
    return min(requested_depth, longest)


def _prefix_agreements(real: list[Hashable], syn: list[Hashable]) -> list[float]:
    real_prefix: set[Hashable] = set()
    syn_prefix: set[Hashable] = set()
    overlap = 0
    agreements: list[float] = []
    for depth, (real_item, syn_item) in enumerate(zip(real, syn), start=1):
        if real_item == syn_item:
            overlap += 1
        else:
            overlap += int(real_item in syn_prefix) + int(syn_item in real_prefix)
        real_prefix.add(real_item)
        syn_prefix.add(syn_item)
        agreements.append(overlap / depth)
    return agreements


def rbo_ext(
    real_rank: Sequence[Hashable],
    syn_rank: Sequence[Hashable],
    p: float = RBO_PERSISTENCE,
    requested_depth: int | None = None,
) -> float:
    validate_unique_ranking_identities(real_rank, side="real")
    validate_unique_ranking_identities(syn_rank, side="synthetic")
    depth = _resolve_depth(real_rank, syn_rank, requested_depth)
    if depth == 0:
        return 1.0
    agreements = _prefix_agreements(
        truncate_and_pad(real_rank, depth, side="real"),
        truncate_and_pad(syn_rank, depth, side="syn"),
    )
    weighted_sum = math.fsum((1.0 - p) * (p ** (d - 1)) * agreements[d - 1] for d in range(1, depth + 1))
    return checked_unit_interval(weighted_sum + (p**depth) * agreements[-1])


def score_topk_ranking(
    real_rank: Sequence[Hashable],
    syn_rank: Sequence[Hashable],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    requested_depth = (metadata or {}).get("requested_depth")
    score = rbo_ext(real_rank, syn_rank, RBO_PERSISTENCE, requested_depth)
    depth = _resolve_depth(real_rank, syn_rank, requested_depth)
    real = list(real_rank[:depth])
    syn = list(syn_rank[:depth])
    agreements = (
        _prefix_agreements(truncate_and_pad(real, depth, side="real"), truncate_and_pad(syn, depth, side="syn"))
        if depth
        else []
    )
    diagnostics = {
        "requested_depth": requested_depth,
        "comparison_depth": depth,
        "real_list_length": len(real_rank),
        "synthetic_list_length": len(syn_rank),
        "raw_rbo": score,
        "top1_match": bool(real and syn and real[0] == syn[0]),
        "set_overlap_at_k": len(set(real) & set(syn)),
        "prefix_overlap_by_depth": agreements[:MAX_PER_KEY_DIAGNOSTICS],
        "real_top": [format_key(item) for item in real[:10]],
        "synthetic_top": [format_key(item) for item in syn[:10]],
    }
    return score, diagnostics
