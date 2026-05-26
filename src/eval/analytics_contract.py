"""Canonical analytics family/sub-item contract helpers.

This module centralizes how query-level evidence is mapped onto the
frozen analytics sub-item contract defined in
`doc/analytics_family_subitem_contract_v1.md`.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Any, Mapping

ANALYTICS_CONTRACT_VERSION = "analytics_family_subitem_contract_v1"

CANONICAL_ANALYTICS_SUBITEMS: dict[str, list[str]] = {
    "subgroup_structure": [
        "internal_profile_stability",
        "subgroup_size_stability",
    ],
    "conditional_dependency_structure": [
        "dependency_strength_similarity",
        "direction_consistency",
        "slice_level_consistency",
    ],
    "tail_rarity_structure": [
        "tail_set_consistency",
        "tail_mass_similarity",
        "tail_concentration_consistency",
    ],
    "missingness_structure": [
        "marginal_missing_rate_consistency",
        "co_missingness_pattern_consistency",
    ],
}

_FACET_TO_SUBITEM: dict[str, dict[str, str]] = {
    "subgroup_structure": {
        "subgroup_distribution_shift": "internal_profile_stability",
        "subgroup_rank_order": "internal_profile_stability",
        "subgroup_conditional_contrast": "internal_profile_stability",
    },
    "conditional_dependency_structure": {
        "pairwise_conditional_dependency": "dependency_strength_similarity",
        "conditional_rate_shift": "direction_consistency",
        "conditional_interaction_hotspots": "slice_level_consistency",
    },
    "tail_rarity_structure": {
        "rare_target_concentration": "tail_concentration_consistency",
        "low_support_extremes": "tail_set_consistency",
        "tail_ranked_signal": "tail_concentration_consistency",
    },
    "missingness_structure": {
        "missing_indicator_distribution": "marginal_missing_rate_consistency",
        "missing_target_interaction": "co_missingness_pattern_consistency",
        "missing_rate_by_subgroup": "co_missingness_pattern_consistency",
    },
}

_ROLE_ALIASES = {
    "group_count": "count_distribution",
    "filtered_group_count_topk": "filtered_stable_view",
    "group_condition_rate": "within_group_proportion",
    "group_ratio_two_conditions": "within_group_proportion",
    "group_sum": "collapsed_target_view",
    "group_avg_numeric": "collapsed_target_view",
    "support_guarded_group_avg": "filtered_stable_view",
    "binned_numeric_group_avg": "collapsed_target_view",
    "two_dimensional_group_avg": "collapsed_target_view",
}

_ROLE_TO_SUBITEM: dict[str, dict[str, str]] = {
    "subgroup_structure": {
        "count_distribution": "subgroup_size_stability",
        "filtered_stable_view": "subgroup_size_stability",
        "within_group_proportion": "internal_profile_stability",
        "collapsed_target_view": "internal_profile_stability",
        "ranked_signal_view": "internal_profile_stability",
        "focused_target_view": "internal_profile_stability",
        "contrastive_conditional_view": "internal_profile_stability",
        "rare_extreme_view": "internal_profile_stability",
    },
    "conditional_dependency_structure": {
        "within_group_proportion": "dependency_strength_similarity",
        "collapsed_target_view": "dependency_strength_similarity",
        "count_distribution": "slice_level_consistency",
        "filtered_stable_view": "slice_level_consistency",
        "ranked_signal_view": "direction_consistency",
        "focused_target_view": "direction_consistency",
        "contrastive_conditional_view": "direction_consistency",
        "rare_extreme_view": "direction_consistency",
    },
    "tail_rarity_structure": {
        "rare_extreme_view": "tail_set_consistency",
        "count_distribution": "tail_mass_similarity",
        "filtered_stable_view": "tail_mass_similarity",
        "within_group_proportion": "tail_concentration_consistency",
        "focused_target_view": "tail_concentration_consistency",
        "contrastive_conditional_view": "tail_concentration_consistency",
        "ranked_signal_view": "tail_concentration_consistency",
        "collapsed_target_view": "tail_concentration_consistency",
    },
    "missingness_structure": {
        "missing_indicator_view": "marginal_missing_rate_consistency",
        "missing_ranked_view": "marginal_missing_rate_consistency",
        "filtered_stable_view": "marginal_missing_rate_consistency",
        "count_distribution": "marginal_missing_rate_consistency",
        "missing_target_interaction": "co_missingness_pattern_consistency",
        "missing_rate_by_subgroup": "co_missingness_pattern_consistency",
        "focused_target_view": "co_missingness_pattern_consistency",
        "contrastive_conditional_view": "co_missingness_pattern_consistency",
        "rare_extreme_view": "co_missingness_pattern_consistency",
        "within_group_proportion": "co_missingness_pattern_consistency",
    },
}

_RATE_RE = re.compile(r"(rate|ratio|proportion|share|pct|percent|bucket_rate|global_rate|within_group_rate|focus_rate)", re.IGNORECASE)
_COUNT_RE = re.compile(r"(count|support|total|freq|frequency)", re.IGNORECASE)
_RANK_RE = re.compile(r"(rank|ranked|order|top|highest|lowest|strongest|weakest|focus)", re.IGNORECASE)
_TAIL_RE = re.compile(r"(tail|rare|extreme|low[\s\-_]?support|outlier)", re.IGNORECASE)
_CONCENTRATION_RE = re.compile(r"(concentrat|dominant|heavy|share|focus)", re.IGNORECASE)
_MISSING_RE = re.compile(r"(missing|null|not_missing)", re.IGNORECASE)
_PAIRWISE_RE = re.compile(r"(pairwise|co[\s\-_]?missing|joint|interaction|subgroup)", re.IGNORECASE)


def canonical_subitem_score_field(family_id: str, subitem_id: str) -> str:
    return f"{family_id}__{subitem_id}_score"


def all_canonical_subitem_score_fields() -> list[str]:
    fields: list[str] = []
    for family_id, subitems in CANONICAL_ANALYTICS_SUBITEMS.items():
        for subitem_id in subitems:
            fields.append(canonical_subitem_score_field(family_id, subitem_id))
    return fields


def normalize_variant_semantic_role(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return _ROLE_ALIASES.get(text, text)


def _normalize_family(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_facet(value: Any) -> str:
    return str(value or "").strip().lower()


def _text_blob(query_row: Mapping[str, Any]) -> str:
    parts = [
        query_row.get("question"),
        query_row.get("research_question"),
        query_row.get("expected_output_shape"),
        query_row.get("template_name"),
        query_row.get("template_id"),
        query_row.get("variant_semantic_role"),
        query_row.get("intended_facet_id"),
        query_row.get("intended_structure_claim"),
        query_row.get("sql"),
    ]
    return " ".join(str(item or "") for item in parts).strip().lower()


def infer_canonical_subitem(query_row: Mapping[str, Any]) -> dict[str, Any]:
    family_id = _normalize_family(query_row.get("family_id") or query_row.get("family"))
    if family_id not in CANONICAL_ANALYTICS_SUBITEMS:
        return {
            "family_id": family_id,
            "canonical_subitem_id": "",
            "contract_version": ANALYTICS_CONTRACT_VERSION,
            "normalized_variant_semantic_role": normalize_variant_semantic_role(query_row.get("variant_semantic_role")),
            "normalized_intended_facet_id": _normalize_facet(query_row.get("intended_facet_id")),
            "subitem_inference_source": "non_analytics_family",
            "subitem_inference_note": "family_not_in_canonical_contract",
        }

    normalized_role = normalize_variant_semantic_role(query_row.get("variant_semantic_role"))
    normalized_facet = _normalize_facet(query_row.get("intended_facet_id"))
    text_blob = _text_blob(query_row)
    sql_text = str(query_row.get("sql") or "").lower()
    explicit_subitem_id = str(query_row.get("canonical_subitem_id") or "").strip()

    if explicit_subitem_id and explicit_subitem_id in CANONICAL_ANALYTICS_SUBITEMS.get(family_id, []):
        return {
            "family_id": family_id,
            "canonical_subitem_id": explicit_subitem_id,
            "contract_version": ANALYTICS_CONTRACT_VERSION,
            "normalized_variant_semantic_role": normalized_role,
            "normalized_intended_facet_id": normalized_facet,
            "subitem_inference_source": "explicit",
            "subitem_inference_note": "canonical_subitem_id",
        }

    if normalized_facet in _FACET_TO_SUBITEM.get(family_id, {}):
        return {
            "family_id": family_id,
            "canonical_subitem_id": _FACET_TO_SUBITEM[family_id][normalized_facet],
            "contract_version": ANALYTICS_CONTRACT_VERSION,
            "normalized_variant_semantic_role": normalized_role,
            "normalized_intended_facet_id": normalized_facet,
            "subitem_inference_source": "facet",
            "subitem_inference_note": normalized_facet,
        }

    if normalized_role in _ROLE_TO_SUBITEM.get(family_id, {}):
        return {
            "family_id": family_id,
            "canonical_subitem_id": _ROLE_TO_SUBITEM[family_id][normalized_role],
            "contract_version": ANALYTICS_CONTRACT_VERSION,
            "normalized_variant_semantic_role": normalized_role,
            "normalized_intended_facet_id": normalized_facet,
            "subitem_inference_source": "role",
            "subitem_inference_note": normalized_role,
        }

    if family_id == "subgroup_structure":
        if _RATE_RE.search(text_blob) or _RANK_RE.search(text_blob):
            subitem_id = "internal_profile_stability"
            note = "heuristic_rate_or_rank"
        elif _COUNT_RE.search(text_blob) or "count(" in sql_text:
            subitem_id = "subgroup_size_stability"
            note = "heuristic_count_or_support"
        else:
            subitem_id = "internal_profile_stability"
            note = "heuristic_family_default"
    elif family_id == "conditional_dependency_structure":
        if "contrast" in text_blob or _RANK_RE.search(text_blob):
            subitem_id = "direction_consistency"
            note = "heuristic_directional_signal"
        elif _COUNT_RE.search(text_blob) and not _RATE_RE.search(text_blob):
            subitem_id = "slice_level_consistency"
            note = "heuristic_slice_support"
        else:
            subitem_id = "dependency_strength_similarity"
            note = "heuristic_dependency_strength"
    elif family_id == "tail_rarity_structure":
        if _TAIL_RE.search(text_blob) and ("support asc" in sql_text or "order by support asc" in sql_text):
            subitem_id = "tail_set_consistency"
            note = "heuristic_tail_membership"
        elif _CONCENTRATION_RE.search(text_blob) and (_RANK_RE.search(text_blob) or "focus_rate" in sql_text):
            subitem_id = "tail_concentration_consistency"
            note = "heuristic_tail_concentration"
        elif (_RATE_RE.search(text_blob) or "focus_rate" in sql_text) and ("group by" in sql_text or "partition by" in sql_text):
            subitem_id = "tail_concentration_consistency"
            note = "heuristic_tail_concentration_from_rate_view"
        else:
            subitem_id = "tail_mass_similarity"
            note = "heuristic_tail_mass"
    else:  # missingness_structure
        if _PAIRWISE_RE.search(text_blob) or "missing_rate" in sql_text or "group by" in sql_text and _MISSING_RE.search(text_blob):
            subitem_id = "co_missingness_pattern_consistency"
            note = "heuristic_missing_structure"
        else:
            subitem_id = "marginal_missing_rate_consistency"
            note = "heuristic_missing_marginal"

    return {
        "family_id": family_id,
        "canonical_subitem_id": subitem_id,
        "contract_version": ANALYTICS_CONTRACT_VERSION,
        "normalized_variant_semantic_role": normalized_role,
        "normalized_intended_facet_id": normalized_facet,
        "subitem_inference_source": "heuristic",
        "subitem_inference_note": note,
    }


def annotate_query_row_with_contract(query_row: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(query_row)
    annotated.update(infer_canonical_subitem(query_row))
    return annotated


def build_subitem_and_family_rows(
    *,
    query_rows: list[dict[str, Any]],
    context_fields: Mapping[str, Any],
    score_field: str = "query_score",
    missingness_applicable: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subitem_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []

    by_family_subitem: dict[tuple[str, str], list[dict[str, Any]]] = {}
    family_query_counts: dict[str, int] = {}
    for row in query_rows:
        family_id = _normalize_family(row.get("family_id"))
        subitem_id = str(row.get("canonical_subitem_id") or "")
        if family_id not in CANONICAL_ANALYTICS_SUBITEMS or not subitem_id:
            continue
        if family_id == "missingness_structure" and not missingness_applicable:
            continue
        by_family_subitem.setdefault((family_id, subitem_id), []).append(row)
        family_query_counts[family_id] = family_query_counts.get(family_id, 0) + 1

    for family_id, subitems in CANONICAL_ANALYTICS_SUBITEMS.items():
        active_scores: list[float] = []
        for order_index, subitem_id in enumerate(subitems, start=1):
            applicable = not (family_id == "missingness_structure" and not missingness_applicable)
            rows = by_family_subitem.get((family_id, subitem_id), [])
            score_values = [
                float(item.get(score_field))
                for item in rows
                if item.get(score_field) is not None
            ]
            score = mean(score_values) if score_values else None
            if applicable and score is not None:
                active_scores.append(float(score))
            inference_sources = sorted({str(item.get("subitem_inference_source") or "") for item in rows if item.get("subitem_inference_source")})
            subitem_rows.append(
                {
                    **context_fields,
                    "family_id": family_id,
                    "subitem_id": subitem_id,
                    "subitem_order": order_index,
                    "subitem_score": round(float(score), 6) if score is not None else None,
                    "query_count": len(rows),
                    "subitem_applicable": applicable,
                    "subitem_inference_sources": ",".join(inference_sources),
                    "contract_version": ANALYTICS_CONTRACT_VERSION,
                }
            )

        family_score = mean(active_scores) if active_scores else None
        family_rows.append(
            {
                **context_fields,
                "family_id": family_id,
                "family_score": round(float(family_score), 6) if family_score is not None else None,
                "query_count": family_query_counts.get(family_id, 0),
                "active_subitem_count": len(active_scores),
                "subitem_count": len(subitems),
                "contract_version": ANALYTICS_CONTRACT_VERSION,
            }
        )

    return subitem_rows, family_rows
