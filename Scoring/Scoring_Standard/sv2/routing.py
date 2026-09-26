"""Semantic scorer routing for SV2 (never value-based)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from .common import (
    COUNT_SUPPORT_DISTRIBUTION,
    NUMERIC_MAGNITUDE,
    RATE_SHARE_PROPORTION,
    SCORER_TYPES,
    TOPK_RANKING,
    ScoringError,
)

TEMPLATE_ROUTING_PATH = Path(__file__).with_name("template_routing_sv2.json")

RATE_SEMANTIC_VALUE_TYPES = frozenset(
    {"rate", "share", "proportion", "probability", "percentage", "fraction", "prevalence", "missingness_rate"}
)

# Old scorer names whose SV2 target is unambiguous.
COMPATIBILITY_MAP = {
    "ratio": NUMERIC_MAGNITUDE,
    "keyed_numeric_aggregate": NUMERIC_MAGNITUDE,
    "scalar_numeric": NUMERIC_MAGNITUDE,
    "keyed_ratio_profile": NUMERIC_MAGNITUDE,
    "count_support_distribution": COUNT_SUPPORT_DISTRIBUTION,
    "distribution_cardinality_profile": COUNT_SUPPORT_DISTRIBUTION,
    "keyed_count_distribution": COUNT_SUPPORT_DISTRIBUTION,
    "temporal_count_curve": COUNT_SUPPORT_DISTRIBUTION,
    "rate_share_proportion": RATE_SHARE_PROPORTION,
    "keyed_rate_profile": RATE_SHARE_PROPORTION,
    "scalar_rate": RATE_SHARE_PROPORTION,
    "missingness_scalar_rate": RATE_SHARE_PROPORTION,
    "missingness_group_rate": RATE_SHARE_PROPORTION,
    "missingness_interaction_rate": RATE_SHARE_PROPORTION,
    "topk_ranking": TOPK_RANKING,
    "topk_tailk_ranking": TOPK_RANKING,
    "topk_ranked_measure": TOPK_RANKING,
    "argmax_selection": TOPK_RANKING,
}
# Old hints that need explicit semantic metadata (template routing) instead of a name-based guess.
AMBIGUOUS_OLD_SCORERS = frozenset(
    {
        "tail_topn_value_curve",
        "tail_outlier_distribution",
        "tail_outlier_summary",
        "tail_distribution_slice",
        "temporal_drift",
        "temporal_shape",
        "generic_keyed_result",
    }
)


@lru_cache(maxsize=1)
def load_template_routing() -> dict[str, dict[str, Any]]:
    payload = json.loads(TEMPLATE_ROUTING_PATH.read_text(encoding="utf-8"))
    templates = payload["templates"]
    for template_id, entry in templates.items():
        if entry.get("scorer_type") not in SCORER_TYPES:
            raise ValueError(f"Invalid SV2 scorer_type for {template_id}: {entry.get('scorer_type')!r}")
    return templates


def template_routing_entry(template_id: str | None) -> dict[str, Any] | None:
    return load_template_routing().get(str(template_id or ""))


def _contract(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = metadata.get("semantic_result_contract")
    return contract if isinstance(contract, Mapping) else {}


def resolve_sv2_scorer(metadata: Mapping[str, Any]) -> tuple[str, str]:
    """Return (scorer_type, routing_source) following the SV2 routing priority."""
    contract = _contract(metadata)

    explicit = metadata.get("sv2_scorer_type") or contract.get("sv2_scorer_type")
    if explicit is not None:
        if explicit not in SCORER_TYPES:
            raise ScoringError("INVALID_SV2_SCORER_TYPE", f"Unknown sv2_scorer_type {explicit!r}.")
        return str(explicit), "explicit_sv2_scorer_type"

    entry = template_routing_entry(metadata.get("template_id"))
    if entry is not None:
        return str(entry["scorer_type"]), "template_routing"

    old = str(metadata.get("old_scorer_type") or contract.get("scorer_type") or metadata.get("scorer_type") or "").strip()
    if old == "scalar":
        semantic_value_type = str(metadata.get("semantic_value_type") or contract.get("semantic_value_type") or "")
        scorer = RATE_SHARE_PROPORTION if semantic_value_type in RATE_SEMANTIC_VALUE_TYPES else NUMERIC_MAGNITUDE
        return scorer, "compatibility_mapping"
    if old in COMPATIBILITY_MAP:
        return COMPATIBILITY_MAP[old], "compatibility_mapping"

    reason = "ambiguous legacy scorer hint" if old in AMBIGUOUS_OLD_SCORERS else "no semantic routing metadata"
    raise ScoringError(
        "UNRESOLVED_SCORER_TYPE",
        f"Cannot route template {metadata.get('template_id')!r} (scorer hint {old or None!r}): {reason}.",
    )
