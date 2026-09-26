"""Set-level curation for benchmark package selection at question-bundle level."""

from __future__ import annotations

from collections import Counter

from tqb_query.benchmark.models import FIVE_FIXED_FAMILIES, QuestionBundleRecord, SetCurationResult


def _bundle_quality(bundle: QuestionBundleRecord) -> dict:
    return bundle.bundle_quality or {}


def _variant_quality_score(bundle: QuestionBundleRecord) -> float:
    accepted = bundle.accepted_variants()
    if not accepted:
        return 0.0

    score = 1.0 * (len(accepted) / max(1, len(bundle.variants)))
    warning_penalty = 0.0
    for variant in accepted:
        warning_codes = (
            variant.validation.static_validation.reason_codes
            + variant.validation.execution_validation.reason_codes
            + variant.validation.sanity_validation.reason_codes
        )
        for code in warning_codes:
            if code.endswith("WARNING"):
                warning_penalty += 0.01
            if code in {
                "VAL_EXEC_LOW_SUPPORT",
                "VAL_NO_NEW_INFORMATION",
                "VAL_BUNDLE_INFORMATION_PENALTY",
                "VAL_SANITY_TRIVIAL",
                "VAL_SANITY_QUESTION_MISMATCH",
                "VAL_SANITY_RQ_SQL_MISMATCH",
            }:
                warning_penalty += 0.05

    bundle_penalty = 0.0
    if not bundle.bundle_validation.passed:
        bundle_penalty += 0.2
    for code in bundle.bundle_validation.reason_codes:
        if code in {"BUNDLE_PASS_COUNT_INSUFFICIENT", "BUNDLE_VARIANTS_IDENTICAL", "BUNDLE_VARIANTS_TOO_DIVERSE"}:
            bundle_penalty += 0.1
        if code in {
            "BUNDLE_PSEUDO_DIVERSITY",
            "BUNDLE_SEMANTIC_DIVERSITY_WEAK",
            "BUNDLE_COHERENCE_WEAK",
            "BUNDLE_INFORMATIONAL_NOVELTY_WEAK",
        }:
            bundle_penalty += 0.2

    quality = _bundle_quality(bundle)
    semantic_diversity_score = float(quality.get("semantic_diversity_score", 0.0))
    novelty_score = float(quality.get("informational_novelty_score", 0.0))
    no_new_info_ratio = float(quality.get("no_new_information_ratio", 0.0))
    pseudo_flags = quality.get("pseudo_diversity_flags") or []
    if semantic_diversity_score < 0.6:
        bundle_penalty += 0.2
    if novelty_score < 0.6:
        bundle_penalty += 0.2
    if no_new_info_ratio >= 0.38:
        bundle_penalty += 0.2
    if pseudo_flags:
        bundle_penalty += 0.2

    return max(0.0, score - warning_penalty - bundle_penalty)


def _bundle_signature(bundle: QuestionBundleRecord) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    accepted = bundle.accepted_variants()
    if accepted:
        spec = accepted[0].query_spec
    else:
        spec = bundle.variants[0].query_spec
    return (
        bundle.family,
        tuple(spec.subgroup_columns),
        tuple(spec.feature_columns),
    )


def curate_question_bundles(
    *,
    bundle_pool: list[QuestionBundleRecord],
    min_questions: int,
    max_questions: int,
    required_families: list[str],
) -> SetCurationResult:
    selected: list[QuestionBundleRecord] = []
    rejected_bundle_ids: list[str] = []
    notes: list[str] = []
    rejection_reason_counts: Counter[str] = Counter()
    quality_thresholds = {
        "semantic_diversity_min": 0.45,
        "informational_novelty_min": 0.45,
        "no_new_information_ratio_max": 0.5,
    }

    quality_filtered_pool: list[QuestionBundleRecord] = []
    for bundle in bundle_pool:
        if not bundle.accepted_local:
            continue
        quality = _bundle_quality(bundle)
        pseudo_flags = quality.get("pseudo_diversity_flags") or []
        semantic_diversity_score = float(quality.get("semantic_diversity_score", 0.0))
        novelty_score = float(quality.get("informational_novelty_score", 0.0))
        no_new_info_ratio = float(quality.get("no_new_information_ratio", 0.0))
        bundle_reason_codes = set(bundle.bundle_validation.reason_codes)
        if "BUNDLE_INFORMATIONAL_NOVELTY_WEAK" in bundle_reason_codes:
            rejected_bundle_ids.append(bundle.bundle_id)
            rejection_reason_counts["informational_novelty_weak"] += 1
            notes.append(f"bundle_quality_drop:{bundle.bundle_id}:informational_novelty_weak")
            continue
        if (
            semantic_diversity_score < quality_thresholds["semantic_diversity_min"]
            or novelty_score < quality_thresholds["informational_novelty_min"]
        ):
            rejected_bundle_ids.append(bundle.bundle_id)
            rejection_reason_counts["low_semantic_or_novelty"] += 1
            notes.append(f"bundle_quality_drop:{bundle.bundle_id}:low_semantic_or_novelty")
            continue
        if no_new_info_ratio >= quality_thresholds["no_new_information_ratio_max"]:
            rejected_bundle_ids.append(bundle.bundle_id)
            rejection_reason_counts["too_many_no_new_information"] += 1
            notes.append(f"bundle_quality_drop:{bundle.bundle_id}:too_many_no_new_information")
            continue
        if pseudo_flags and "degenerate_statistic" in pseudo_flags:
            rejected_bundle_ids.append(bundle.bundle_id)
            rejection_reason_counts["degenerate_statistic"] += 1
            notes.append(f"bundle_quality_drop:{bundle.bundle_id}:degenerate_statistic")
            continue
        quality_filtered_pool.append(bundle)

    pool = quality_filtered_pool
    if not pool:
        achieved = {family: 0 for family in FIVE_FIXED_FAMILIES}
        audit_v2 = {
            "contract_version": "set_curation_audit_v2",
            "required_family_coverage": {family: 1 if family in required_families else 0 for family in FIVE_FIXED_FAMILIES},
            "achieved_family_coverage": achieved,
            "selected_bundle_ids": [],
            "rejected_bundle_ids": sorted(set(rejected_bundle_ids)),
            "rejection_reason_counts": dict(rejection_reason_counts),
            "quality_thresholds": quality_thresholds,
            "key_notes": notes,
        }
        return SetCurationResult(
            selected_bundle_ids=[],
            family_coverage=achieved,
            notes=["No locally accepted question bundles available for set-level curation."],
            rejected_bundle_ids=[],
            audit_v2=audit_v2,
        )

    # First pass: ensure required family coverage when possible.
    for family in required_families:
        options = [bundle for bundle in pool if bundle.family == family]
        if not options:
            notes.append(f"family_gap:{family}")
            continue
        best = sorted(options, key=_variant_quality_score, reverse=True)[0]
        if best.bundle_id not in {item.bundle_id for item in selected}:
            selected.append(best)

    selected_ids = {item.bundle_id for item in selected}
    remaining = [bundle for bundle in sorted(pool, key=_variant_quality_score, reverse=True) if bundle.bundle_id not in selected_ids]

    field_counter: Counter[str] = Counter()
    for bundle in selected:
        accepted = bundle.accepted_variants()
        spec = accepted[0].query_spec if accepted else bundle.variants[0].query_spec
        field_counter.update(spec.subgroup_columns)
        field_counter.update(spec.feature_columns)

    selected_signatures = {_bundle_signature(bundle) for bundle in selected}

    for bundle in remaining:
        if len(selected) >= max_questions:
            rejected_bundle_ids.append(bundle.bundle_id)
            rejection_reason_counts["max_questions_reached"] += 1
            continue

        signature = _bundle_signature(bundle)
        if signature in selected_signatures and len(selected) >= min_questions:
            rejected_bundle_ids.append(bundle.bundle_id)
            rejection_reason_counts["redundant_bundle"] += 1
            notes.append(f"redundant_bundle_drop:{bundle.bundle_id}")
            continue

        accepted = bundle.accepted_variants()
        spec = accepted[0].query_spec if accepted else bundle.variants[0].query_spec
        candidate_fields = spec.subgroup_columns + spec.feature_columns
        imbalance_penalty = sum(field_counter.get(field, 0) for field in candidate_fields)
        if (
            imbalance_penalty >= 8
            and len(selected) >= max(min_questions, len(required_families))
        ):
            rejected_bundle_ids.append(bundle.bundle_id)
            rejection_reason_counts["field_balance_penalty"] += 1
            notes.append(f"field_balance_bundle_drop:{bundle.bundle_id}")
            continue

        selected.append(bundle)
        selected_signatures.add(signature)
        field_counter.update(candidate_fields)

    selected = selected[:max_questions]
    coverage_counter = Counter(bundle.family for bundle in selected)
    family_coverage = {family: coverage_counter.get(family, 0) for family in FIVE_FIXED_FAMILIES}

    selected_with_5 = sum(1 for bundle in selected if bundle.accepted_variant_count() >= 5)
    notes.append(f"selected_bundles_with_5_pass={selected_with_5}/{len(selected)}")
    notes.append(f"diversity:distinct_field_count={len(field_counter)}")
    notes.append(f"quality_pool_size={len(pool)}")

    if len(selected) < min_questions:
        notes.append(
            f"shortfall:selected_questions={len(selected)}<min_questions={min_questions};accepted_bundle_pool={len(pool)}"
        )

    for family in required_families:
        if family_coverage.get(family, 0) == 0:
            notes.append(f"required_family_missing:{family}")

    audit_v2 = {
        "contract_version": "set_curation_audit_v2",
        "required_family_coverage": {family: 1 if family in required_families else 0 for family in FIVE_FIXED_FAMILIES},
        "achieved_family_coverage": family_coverage,
        "selected_bundle_ids": [bundle.bundle_id for bundle in selected],
        "rejected_bundle_ids": sorted(set(rejected_bundle_ids)),
        "rejection_reason_counts": dict(rejection_reason_counts),
        "quality_thresholds": quality_thresholds,
        "key_notes": notes,
    }

    return SetCurationResult(
        selected_bundle_ids=[bundle.bundle_id for bundle in selected],
        family_coverage=family_coverage,
        notes=notes,
        rejected_bundle_ids=sorted(set(rejected_bundle_ids)),
        audit_v2=audit_v2,
    )
