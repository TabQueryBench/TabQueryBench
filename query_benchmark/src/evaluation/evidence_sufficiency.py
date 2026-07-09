"""Evidence sufficiency evaluation (family-level + within-question diversity)."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    if max_entropy <= 0:
        return 0.0
    return max(0.0, min(1.0, entropy / max_entropy))


def _question_signature(question: dict[str, Any]) -> str:
    related = question.get("related_columns") or question.get("related_fields") or []
    related_norm = tuple(sorted(str(item) for item in related if item))
    payload = "|".join(
        [
            str(question.get("family_id") or question.get("family") or ""),
            str(question.get("intent") or ""),
            str(question.get("comparator_type") or ""),
            str(question.get("intended_facet_id") or ""),
            str(related_norm),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _bundle_map_by_question(question_bundles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for bundle in question_bundles:
        rq = bundle.get("research_question") if isinstance(bundle, dict) else None
        if not isinstance(rq, dict):
            continue
        qid = str(rq.get("question_id") or "")
        if not qid:
            continue
        mapping[qid] = bundle
    return mapping


def _diversity_map(bundle_diversity_records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_bundle: dict[str, dict[str, Any]] = {}
    by_question: dict[str, dict[str, Any]] = {}
    for row in bundle_diversity_records:
        if not isinstance(row, dict):
            continue
        bundle_id = str(row.get("bundle_id") or "")
        question_id = str(row.get("question_id") or "")
        if bundle_id:
            by_bundle[bundle_id] = row
        if question_id:
            by_question[question_id] = row
    return by_bundle, by_question


def _pairwise_metrics_from_record(
    diversity_record: dict[str, Any],
    near_duplicate_jaccard_threshold: float,
) -> tuple[float, float, int, int]:
    pairwise = diversity_record.get("pairwise_signals") if isinstance(diversity_record, dict) else []
    if not isinstance(pairwise, list):
        pairwise = []

    structural_diversities: list[float] = []
    near_dup_count = 0
    for row in pairwise:
        if not isinstance(row, dict):
            continue
        jaccard = _to_float(row.get("jaccard_similarity"), default=1.0)
        structural_diversities.append(max(0.0, 1.0 - jaccard))
        same_sig = bool(row.get("same_information_signature"))
        if same_sig or jaccard >= near_duplicate_jaccard_threshold:
            near_dup_count += 1

    total_pairs = len(structural_diversities)
    structural_diversity_summary = sum(structural_diversities) / total_pairs if total_pairs else 0.0
    near_dup_ratio = near_dup_count / total_pairs if total_pairs else 0.0
    return structural_diversity_summary, near_dup_ratio, near_dup_count, total_pairs


def _canonical_hash_dup_ratio(query_specs: list[dict[str, Any]]) -> float:
    hashes = [str(item.get("canonical_sql_hash") or "") for item in query_specs if item.get("canonical_sql_hash")]
    if len(hashes) <= 1:
        return 0.0
    total_pairs = len(hashes) * (len(hashes) - 1) // 2
    if total_pairs <= 0:
        return 0.0
    counter = Counter(hashes)
    dup_pairs = sum(count * (count - 1) // 2 for count in counter.values() if count > 1)
    return dup_pairs / total_pairs


def _group_query_specs_by_question(query_specs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_specs:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("question_id") or "")
        if not qid:
            qid = str(row.get("stable_question_id") or "")
        if not qid:
            continue
        mapping[qid].append(row)
    return mapping


def evaluate_evidence_sufficiency(
    *,
    query_specs: list[dict[str, Any]],
    question_bundles: list[dict[str, Any]],
    bundle_diversity_records: list[dict[str, Any]],
    family_facet_catalog: dict[str, Any],
    near_duplicate_jaccard_threshold: float = 0.92,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_question_specs = _group_query_specs_by_question(query_specs)
    bundle_by_question = _bundle_map_by_question(question_bundles)
    _, diversity_by_question = _diversity_map(bundle_diversity_records)

    question_rows: list[dict[str, Any]] = []
    family_question_scores: dict[str, list[float]] = defaultdict(list)
    family_question_signatures: dict[str, set[str]] = defaultdict(set)
    family_question_facet_ids: dict[str, list[str]] = defaultdict(list)

    all_question_ids = set(by_question_specs.keys()) | set(bundle_by_question.keys())
    for question_id in sorted(all_question_ids):
        specs = by_question_specs.get(question_id, [])
        bundle = bundle_by_question.get(question_id, {})
        rq = bundle.get("research_question") if isinstance(bundle, dict) else {}
        if not isinstance(rq, dict):
            rq = {}

        family_id = str(
            rq.get("family_id")
            or rq.get("family")
            or (specs[0].get("family_id") if specs else "")
            or (specs[0].get("family") if specs else "unknown")
        )
        intended_facet_id = str(rq.get("intended_facet_id") or (specs[0].get("intended_facet_id") if specs else "unknown"))

        roles = [str(item.get("variant_semantic_role") or "unknown") for item in specs]
        unique_roles = len(set(roles))
        role_diversity = unique_roles / max(1, len(specs))

        diversity_record = diversity_by_question.get(question_id, {})
        structural_diversity, near_dup_ratio_pairwise, near_dup_count, pair_count = _pairwise_metrics_from_record(
            diversity_record,
            near_duplicate_jaccard_threshold=near_duplicate_jaccard_threshold,
        )
        hash_dup_ratio = _canonical_hash_dup_ratio(specs)
        near_dup_ratio = max(near_dup_ratio_pairwise, hash_dup_ratio)

        bundle_div_score = _to_float(diversity_record.get("bundle_diversity_score"), default=0.0)
        if bundle_div_score <= 0.0:
            bundle_div_score = _to_float(
                (bundle.get("bundle_quality") or {}).get("semantic_diversity_score"),
                default=0.0,
            )

        bundle_novelty_score = _to_float(diversity_record.get("bundle_novelty_score"), default=0.0)
        if bundle_novelty_score <= 0.0:
            bundle_novelty_score = _to_float(
                (bundle.get("bundle_quality") or {}).get("informational_novelty_score"),
                default=0.0,
            )

        # v0.1 explicit formula:
        # question_score = 0.30*structural_diversity + 0.25*role_diversity
        #                + 0.20*bundle_diversity + 0.25*bundle_novelty
        #                - 0.25*near_duplicate_ratio
        question_score = (
            0.30 * structural_diversity
            + 0.25 * role_diversity
            + 0.20 * bundle_div_score
            + 0.25 * bundle_novelty_score
            - 0.25 * near_dup_ratio
        )
        question_score = max(0.0, min(1.0, question_score))

        question_row = {
            "question_id": question_id,
            "stable_question_id": str(rq.get("stable_question_id") or (specs[0].get("stable_question_id") if specs else "")),
            "family_id": family_id,
            "intended_facet_id": intended_facet_id or "unknown",
            "query_count": len(specs),
            "role_diversity": round(role_diversity, 6),
            "structural_diversity": round(structural_diversity, 6),
            "near_duplicate_ratio": round(near_dup_ratio, 6),
            "near_duplicate_count": near_dup_count,
            "pair_count": pair_count,
            "bundle_diversity_score": round(bundle_div_score, 6),
            "bundle_novelty_score": round(bundle_novelty_score, 6),
            "question_evidence_score": round(question_score, 6),
            "variant_roles": sorted(set(roles)),
            "diversity_intent_tags": sorted(
                set(str(item.get("diversity_intent_tag") or "unknown") for item in specs)
            ),
        }
        question_rows.append(question_row)

        family_question_scores[family_id].append(question_score)
        family_question_facet_ids[family_id].append(question_row["intended_facet_id"])
        family_question_signatures[family_id].add(_question_signature(rq if rq else question_row))

    families = (family_facet_catalog or {}).get("families") if isinstance(family_facet_catalog, dict) else {}
    if not isinstance(families, dict):
        families = {}

    family_rows: list[dict[str, Any]] = []
    for family_id, facet_defs in families.items():
        if not isinstance(facet_defs, list):
            facet_defs = []
        required_facets = [str(item.get("facet_id")) for item in facet_defs if isinstance(item, dict) and item.get("facet_id")]
        required_set = set(required_facets)

        question_scores = family_question_scores.get(family_id, [])
        observed_facets = [facet for facet in family_question_facet_ids.get(family_id, []) if facet]
        covered_set = set(observed_facets) & required_set if required_set else set(observed_facets)

        coverage_ratio = (len(covered_set) / len(required_set)) if required_set else (1.0 if question_scores else 0.0)

        facet_counter = Counter(observed_facets)
        if required_facets:
            distribution_counts = [facet_counter.get(facet, 0) for facet in required_facets]
        else:
            distribution_counts = list(facet_counter.values())
        facet_balance = _normalized_entropy(distribution_counts)

        distinct_angle_ratio = 0.0
        question_count = len(question_scores)
        if question_count > 0:
            distinct_angle_ratio = len(family_question_signatures.get(family_id, set())) / question_count

        avg_question_score = sum(question_scores) / question_count if question_count else 0.0

        # v0.1 explicit formula:
        # family_score = 0.45*facet_coverage + 0.20*facet_balance
        #              + 0.15*distinct_angle_ratio + 0.20*avg_question_score
        family_score = (
            0.45 * coverage_ratio
            + 0.20 * facet_balance
            + 0.15 * distinct_angle_ratio
            + 0.20 * avg_question_score
        )
        family_score = max(0.0, min(1.0, family_score))

        family_rows.append(
            {
                "family_id": family_id,
                "question_count": question_count,
                "required_facet_count": len(required_set),
                "covered_facet_count": len(covered_set),
                "covered_facets": sorted(covered_set),
                "missing_facets": sorted(required_set - covered_set),
                "facet_coverage_ratio": round(coverage_ratio, 6),
                "facet_balance_score": round(facet_balance, 6),
                "distinct_question_angle_ratio": round(distinct_angle_ratio, 6),
                "avg_question_evidence_score": round(avg_question_score, 6),
                "family_evidence_sufficient_score": round(family_score, 6),
                "facet_distribution": dict(facet_counter),
            }
        )

    # Include families that appear in data but not in catalog.
    families_in_questions = set(family_question_scores.keys())
    catalog_families = set(families.keys())
    for family_id in sorted(families_in_questions - catalog_families):
        question_scores = family_question_scores.get(family_id, [])
        question_count = len(question_scores)
        avg_question_score = sum(question_scores) / question_count if question_count else 0.0
        family_rows.append(
            {
                "family_id": family_id,
                "question_count": question_count,
                "required_facet_count": 0,
                "covered_facet_count": 0,
                "covered_facets": [],
                "missing_facets": [],
                "facet_coverage_ratio": 0.0,
                "facet_balance_score": 0.0,
                "distinct_question_angle_ratio": round(
                    len(family_question_signatures.get(family_id, set())) / max(1, question_count),
                    6,
                ),
                "avg_question_evidence_score": round(avg_question_score, 6),
                "family_evidence_sufficient_score": round(0.20 * avg_question_score, 6),
                "facet_distribution": {},
                "notes": ["family_missing_in_catalog_v0_1"],
            }
        )

    family_rows_sorted = sorted(family_rows, key=lambda x: x["family_id"])

    total_questions = len(question_rows)
    weighted_sum = sum(row["family_evidence_sufficient_score"] * row["question_count"] for row in family_rows_sorted)
    workload_score = weighted_sum / max(1, total_questions)

    weak_families = [
        row["family_id"]
        for row in family_rows_sorted
        if row["family_evidence_sufficient_score"] < 0.45
    ]

    report = {
        "contract_version": "evidence_sufficiency_report_v0_1",
        "formulas": {
            "question_evidence_score": {
                "definition": "0.30*structural_diversity + 0.25*role_diversity + 0.20*bundle_diversity + 0.25*bundle_novelty - 0.25*near_duplicate_ratio",
                "range": "[0,1]",
            },
            "family_evidence_sufficient_score": {
                "definition": "0.45*facet_coverage + 0.20*facet_balance + 0.15*distinct_question_angle_ratio + 0.20*avg_question_evidence_score",
                "range": "[0,1]",
            },
        },
        "config": {
            "near_duplicate_jaccard_threshold": near_duplicate_jaccard_threshold,
        },
        "summary": {
            "question_count": total_questions,
            "family_count": len(family_rows_sorted),
            "workload_evidence_sufficient_score": round(workload_score, 6),
            "weak_families": weak_families,
        },
        "by_family": family_rows_sorted,
    }

    return report, question_rows
