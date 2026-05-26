"""End-to-end benchmark construction pipeline (question-bundle mode)."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.benchmark.curation import curate_question_bundles
from src.benchmark.facets import catalog_summary, load_family_facet_catalog
from src.benchmark.llm_runtime import BenchmarkLLMRuntime
from src.benchmark.models import (
    CandidateRecord,
    FIVE_FIXED_FAMILIES,
    QuerySpec,
    QuestionBundleRecord,
    ResearchQuestion,
)
from src.benchmark.planning import build_family_plan, generate_research_questions_for_family
from src.benchmark.probing import run_exploratory_sql_probes
from src.benchmark.realization import (
    realize_query_spec_variants,
    repair_queryspec_level,
    repair_sql_level,
)
from src.benchmark.sql_exemplars import extract_csv_columns, load_sql_exemplar_repository
from src.benchmark.sql_exec import execute_sql
from src.benchmark.understanding import (
    build_operational_understanding,
    build_static_understanding,
    update_operational_with_validation_feedback,
)
from src.benchmark.validation import run_basic_validation, run_bundle_similarity_validation
from src.benchmark.validation import build_query_execution_summary_v2
from src.config.settings import FAMILY_FACET_CATALOG_PATH
from src.data.bundle import DatasetBundle
from src.db.csv_sqlite import SqliteMaterializationResult
from src.logging.run_artifacts import RunArtifactWriter


def _collect_failure_reason_codes(query_spec: QuerySpec, validation) -> list[str]:
    reason_codes = []
    reason_codes.extend(validation.static_validation.reason_codes)
    reason_codes.extend(validation.execution_validation.reason_codes)
    reason_codes.extend(validation.sanity_validation.reason_codes)
    reason_codes.extend(query_spec.reason_codes)
    return list(dict.fromkeys(reason_codes))


def _required_families(static_understanding) -> list[str]:
    required: list[str] = []
    for family in FIVE_FIXED_FAMILIES:
        status = static_understanding.family_applicability_summary.get(family, "uncertain")
        if status == "likely_not_applicable":
            continue
        required.append(family)
    return required or list(FIVE_FIXED_FAMILIES)


def _safe_dataset_columns(bundle: DatasetBundle) -> list[str]:
    try:
        columns = extract_csv_columns(bundle.main_csv_path)
        if columns:
            return columns
    except Exception:
        return []
    return []


def _build_single_variant_candidate(
    *,
    llm_runtime: BenchmarkLLMRuntime,
    static_understanding,
    table_name: str,
    db_path: Path,
    query_spec: QuerySpec,
    max_repairs: int,
    bundle_id: str,
    variant_index: int,
    artifact_writer: RunArtifactWriter,
) -> tuple[CandidateRecord, QuerySpec, list[dict[str, Any]]]:
    current_spec = query_spec

    attempts_log: list[dict[str, Any]] = []
    validation = None
    execution = None
    accepted_local = False

    for attempt_index in range(max_repairs + 1):
        execution = execute_sql(db_path=db_path, sql=current_spec.sql)
        validation = run_basic_validation(
            llm_runtime=llm_runtime,
            static_understanding=static_understanding,
            query_spec=current_spec,
            execution_result=execution,
            table_name=table_name,
        )

        failure_reason_codes = _collect_failure_reason_codes(current_spec, validation)
        attempts_log.append(
            {
                "attempt_index": attempt_index,
                "query_id": current_spec.query_id,
                "sql": current_spec.sql,
                "execution_ok": execution.ok,
                "execution_error": execution.error,
                "validation": validation.to_dict(),
                "failure_reason_codes": failure_reason_codes,
            }
        )

        artifact_writer.append_trace(
            {
                "event_type": "variant_attempt",
                "bundle_id": bundle_id,
                "variant_index": variant_index,
                "query_id": current_spec.query_id,
                "research_question": current_spec.research_question,
                "attempt_index": attempt_index,
                "family": current_spec.family,
                "execution_ok": execution.ok,
                "validation_passed": validation.overall_passed,
                "failure_reason_codes": failure_reason_codes,
            }
        )

        if validation.overall_passed:
            accepted_local = True
            current_spec.status = "accepted_local"
            current_spec.reason_codes.append("ACCEPT_LOCAL_VALIDATION_PASS")
            break

        if attempt_index >= max_repairs:
            current_spec.status = "rejected_local"
            break

        if attempt_index == 0:
            current_spec = repair_sql_level(
                llm_runtime=llm_runtime,
                static_understanding=static_understanding,
                query_spec=current_spec,
                table_name=table_name,
                failure_reason_codes=failure_reason_codes,
                execution_error=execution.error or "",
            )
        else:
            current_spec = repair_queryspec_level(current_spec, failure_reason_codes=failure_reason_codes)
            current_spec = repair_sql_level(
                llm_runtime=llm_runtime,
                static_understanding=static_understanding,
                query_spec=current_spec,
                table_name=table_name,
                failure_reason_codes=failure_reason_codes,
                execution_error=execution.error or "",
            )

    assert validation is not None
    assert execution is not None

    rejected_reason_codes = []
    if not accepted_local:
        rejected_reason_codes = _collect_failure_reason_codes(current_spec, validation)
        current_spec.reason_codes.append("REJECT_LOCAL_VALIDATION_FAIL")

    candidate = CandidateRecord(
        query_spec=current_spec,
        validation=validation,
        execution=execution,
        accepted_local=accepted_local,
        rejected_reason_codes=rejected_reason_codes,
        provenance={
            "bundle_id": bundle_id,
            "variant_index": variant_index,
            "attempts": attempts_log,
            "final_attempt_count": len(attempts_log),
        },
    )

    return candidate, current_spec, attempts_log


def _build_question_bundle(
    *,
    llm_runtime: BenchmarkLLMRuntime,
    static_understanding,
    operational_understanding,
    table_name: str,
    db_path: Path,
    research_question: ResearchQuestion,
    queries_per_question: int,
    min_pass_variants: int,
    max_repairs: int,
    artifact_writer: RunArtifactWriter,
    sql_exemplar_repo=None,
    available_columns: list[str] | None = None,
    exemplar_max_candidates_per_role: int = 4,
) -> tuple[QuestionBundleRecord, list[QuerySpec], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Any]:
    bundle_id = f"qb_{research_question.family}_{uuid4().hex[:8]}"

    query_specs = realize_query_spec_variants(
        llm_runtime=llm_runtime,
        static_understanding=static_understanding,
        research_question=research_question,
        table_name=table_name,
        num_variants=queries_per_question,
        sql_exemplar_repo=sql_exemplar_repo,
        available_columns=available_columns,
        exemplar_max_candidates_per_role=exemplar_max_candidates_per_role,
    )

    variants: list[CandidateRecord] = []
    final_specs: list[QuerySpec] = []
    validation_entries: list[dict[str, Any]] = []
    execution_summaries: list[dict[str, Any]] = []
    updated_operational = operational_understanding

    for variant_index, spec in enumerate(query_specs, start=1):
        candidate, final_spec, attempts_log = _build_single_variant_candidate(
            llm_runtime=llm_runtime,
            static_understanding=static_understanding,
            table_name=table_name,
            db_path=db_path,
            query_spec=spec,
            max_repairs=max_repairs,
            bundle_id=bundle_id,
            variant_index=variant_index,
            artifact_writer=artifact_writer,
        )

        variants.append(candidate)
        final_specs.append(final_spec)
        validation_entries.append(
            {
                "bundle_id": bundle_id,
                "question_id": research_question.question_id,
                "query_id": final_spec.query_id,
                "variant_index": variant_index,
                "variant_semantic_role": final_spec.variant_semantic_role,
                "research_question": final_spec.research_question,
                "family": final_spec.family,
                "accepted_local": candidate.accepted_local,
                "validation": candidate.validation.to_dict(),
                "attempts": attempts_log,
            }
        )
        execution_summaries.append(
            build_query_execution_summary_v2(
                query_spec=final_spec,
                execution_result=candidate.execution,
                validation_result=candidate.validation,
            )
        )

        updated_operational = update_operational_with_validation_feedback(
            updated_operational,
            family=final_spec.family,
            reason_codes=candidate.rejected_reason_codes,
        )

    bundle_validation, bundle_quality = run_bundle_similarity_validation(
        variants=variants,
        required_min_pass=min_pass_variants,
    )

    bundle_diversity_record = {
        "contract_version": "bundle_diversity_matrix_v2",
        "bundle_id": bundle_id,
        "question_id": research_question.question_id,
        "stable_question_id": research_question.stable_question_id,
        "family_id": research_question.family_id or research_question.family,
        "intended_facet_id": research_question.intended_facet_id or "unknown",
        "variants": [
            {
                "query_id": item.query_spec.query_id,
                "stable_query_id": item.query_spec.stable_query_id,
                "variant_id": item.query_spec.variant_id or item.query_spec.query_id,
                "variant_semantic_role": item.query_spec.variant_semantic_role,
                "accepted_local": item.accepted_local,
                "canonical_sql_hash": item.query_spec.canonical_sql_hash,
            }
            for item in variants
        ],
        "pairwise_signals": list(bundle_quality.get("pairwise_diversity_signals") or []),
        "bundle_diversity_score": bundle_quality.get("bundle_diversity_score"),
        "bundle_novelty_score": bundle_quality.get("bundle_novelty_score"),
        "bundle_reason_codes": bundle_quality.get("bundle_reason_codes") or bundle_validation.reason_codes,
        "pseudo_diversity_flags": bundle_quality.get("pseudo_diversity_flags") or [],
    }

    accepted_local = bundle_validation.passed
    rejected_reason_codes: list[str] = []
    if not accepted_local:
        rejected_reason_codes.extend(bundle_validation.reason_codes)
        for variant in variants:
            rejected_reason_codes.extend(variant.rejected_reason_codes)
        rejected_reason_codes = list(dict.fromkeys(rejected_reason_codes))

    bundle_record = QuestionBundleRecord(
        bundle_id=bundle_id,
        research_question=research_question,
        family=research_question.family,
        variants=variants,
        bundle_validation=bundle_validation,
        bundle_quality=bundle_quality,
        accepted_local=accepted_local,
        rejected_reason_codes=rejected_reason_codes,
        provenance={
            "question_id": research_question.question_id,
            "queries_per_question": queries_per_question,
            "min_pass_variants": min_pass_variants,
        },
    )

    artifact_writer.append_trace(
        {
            "event_type": "bundle_completed",
            "bundle_id": bundle_id,
            "question_id": research_question.question_id,
            "family": research_question.family,
            "accepted_variants": bundle_record.accepted_variant_count(),
            "total_variants": len(bundle_record.variants),
            "bundle_accepted": bundle_record.accepted_local,
            "bundle_reason_codes": bundle_validation.reason_codes,
            "bundle_quality": bundle_quality,
        }
    )

    return (
        bundle_record,
        final_specs,
        validation_entries,
        execution_summaries,
        bundle_diversity_record,
        updated_operational,
    )


def run_benchmark_construction_v1(
    *,
    bundle: DatasetBundle,
    sqlite_result: SqliteMaterializationResult,
    llm_runtime: BenchmarkLLMRuntime,
    artifact_writer: RunArtifactWriter,
    min_questions: int,
    max_questions: int,
    target_questions: int,
    queries_per_question: int,
    min_pass_variants: int,
    max_family_rounds: int,
    max_repairs: int,
    verbose: bool,
    family_facet_catalog: dict[str, Any] | None = None,
    enable_sql_exemplars: bool = True,
    sql_exemplar_pool_path: Path | None = None,
    exemplar_max_candidates_per_role: int = 4,
) -> dict[str, Any]:
    facet_catalog = family_facet_catalog or load_family_facet_catalog(FAMILY_FACET_CATALOG_PATH)
    static_understanding = build_static_understanding(bundle)
    available_columns = _safe_dataset_columns(bundle)
    sql_exemplar_repo = None
    exemplar_summary: dict[str, Any] = {
        "enabled": enable_sql_exemplars,
        "pool_path": (str(sql_exemplar_pool_path) if sql_exemplar_pool_path else ""),
        "loaded": False,
        "error": "",
    }
    if enable_sql_exemplars and sql_exemplar_pool_path is not None:
        sql_exemplar_repo = load_sql_exemplar_repository(sql_exemplar_pool_path)
        if sql_exemplar_repo is not None:
            exemplar_summary["loaded"] = True
            exemplar_summary["repo"] = sql_exemplar_repo.summary()
        else:
            exemplar_summary["error"] = "failed_to_load_or_missing_pool"

    useful_field_combinations = [
        combo
        for combo in (static_understanding.policy_summary.get("useful_field_combinations") or [])
        if isinstance(combo, list)
    ]

    probes = run_exploratory_sql_probes(
        db_path=sqlite_result.db_path,
        table_name=sqlite_result.table_name,
        static_understanding=static_understanding,
        useful_field_combinations=useful_field_combinations,
    )
    operational = build_operational_understanding(static_understanding, probes)

    artifact_writer.write_json("static_understanding.json", static_understanding.to_dict())
    artifact_writer.write_json("operational_understanding.json", operational.to_dict())
    artifact_writer.write_json("sql_exemplar_context.json", exemplar_summary)
    artifact_writer.write_json(
        "family_facet_catalog.json",
        {
            "contract_version": "family_facet_catalog_v0_1",
            "catalog": facet_catalog,
            "summary": catalog_summary(facet_catalog),
        },
    )
    artifact_writer.write_json(
        "exploratory_probes.json",
        {
            "probe_count": len(probes),
            "probes": [probe.to_dict() for probe in probes],
        },
    )

    question_budget = max(min_questions, min(max_questions, target_questions))
    required_families = _required_families(static_understanding)

    all_research_questions: list[ResearchQuestion] = []
    all_query_specs: list[QuerySpec] = []
    all_validations: list[dict[str, Any]] = []
    execution_summaries_v2: list[dict[str, Any]] = []
    bundle_diversity_records_v2: list[dict[str, Any]] = []
    variant_pool: list[CandidateRecord] = []
    question_bundle_pool: list[QuestionBundleRecord] = []
    family_plans: list[dict[str, Any]] = []

    missing_required_families = list(required_families)

    for round_index in range(max_family_rounds):
        remaining_budget = max(0, question_budget - len(all_research_questions))
        if remaining_budget <= 0:
            artifact_writer.append_trace(
                {
                    "event_type": "question_budget_exhausted",
                    "round_index": round_index,
                    "question_budget": question_budget,
                    "generated_research_questions": len(all_research_questions),
                }
            )
            break

        rounds_left = max(1, max_family_rounds - round_index)
        round_budget = max(len(required_families), (remaining_budget + rounds_left - 1) // rounds_left)
        focus = missing_required_families if (round_index > 0 and missing_required_families) else None

        plan = build_family_plan(
            static_understanding=static_understanding,
            operational_understanding=operational,
            round_index=round_index,
            max_questions=round_budget,
            focus_families=focus,
        )
        family_plans.append(plan.to_dict())

        artifact_writer.append_trace(
            {
                "event_type": "family_plan",
                "round_index": round_index,
                "focus_families": focus,
                "attempts_by_family": plan.attempts_by_family,
            }
        )

        stop_round = False
        for family, attempts in plan.attempts_by_family.items():
            if attempts <= 0:
                continue

            remaining_budget = max(0, question_budget - len(all_research_questions))
            if remaining_budget <= 0:
                stop_round = True
                break

            to_generate = min(attempts, remaining_budget)
            questions = generate_research_questions_for_family(
                llm_runtime=llm_runtime,
                static_understanding=static_understanding,
                operational_understanding=operational,
                family=family,
                family_facet_catalog=facet_catalog,
                num_questions=to_generate,
            )
            all_research_questions.extend(questions)

            for question in questions:
                (
                    bundle_record,
                    final_specs,
                    validation_entries,
                    execution_summaries,
                    bundle_diversity_record,
                    operational,
                ) = _build_question_bundle(
                    llm_runtime=llm_runtime,
                    static_understanding=static_understanding,
                    operational_understanding=operational,
                    table_name=sqlite_result.table_name,
                    db_path=sqlite_result.db_path,
                    research_question=question,
                    queries_per_question=queries_per_question,
                    min_pass_variants=min_pass_variants,
                    max_repairs=max_repairs,
                    artifact_writer=artifact_writer,
                    sql_exemplar_repo=sql_exemplar_repo,
                    available_columns=available_columns,
                    exemplar_max_candidates_per_role=exemplar_max_candidates_per_role,
                )

                question_bundle_pool.append(bundle_record)
                variant_pool.extend(bundle_record.variants)
                all_query_specs.extend(final_specs)
                all_validations.extend(validation_entries)
                execution_summaries_v2.extend(execution_summaries)
                bundle_diversity_records_v2.append(bundle_diversity_record)

                if verbose:
                    print(
                        f"[bundle] family={bundle_record.family} bundle_id={bundle_record.bundle_id} "
                        f"accepted={bundle_record.accepted_local} accepted_variants={bundle_record.accepted_variant_count()}/{len(bundle_record.variants)}"
                    )

        if stop_round:
            break

        curation_preview = curate_question_bundles(
            bundle_pool=question_bundle_pool,
            min_questions=min_questions,
            max_questions=max_questions,
            required_families=required_families,
        )
        missing_required_families = [
            family for family in required_families if curation_preview.family_coverage.get(family, 0) == 0
        ]
        preview_selected_count = len(curation_preview.selected_bundle_ids)

        artifact_writer.append_trace(
            {
                "event_type": "outer_loop_round_summary",
                "round_index": round_index,
                "missing_required_families_after_round": missing_required_families,
                "preview_family_coverage": curation_preview.family_coverage,
                "preview_selected_questions": preview_selected_count,
                "min_questions": min_questions,
                "max_questions": max_questions,
                "remaining_question_budget_after_round": max(0, question_budget - len(all_research_questions)),
            }
        )

    final_curation = curate_question_bundles(
        bundle_pool=question_bundle_pool,
        min_questions=min_questions,
        max_questions=max_questions,
        required_families=required_families,
    )

    # Persist intermediate artifacts.
    artifact_writer.write_json("family_plans.json", {"plans": family_plans})
    artifact_writer.write_jsonl("research_questions.jsonl", [rq.to_dict() for rq in all_research_questions])
    artifact_writer.write_jsonl("query_specs.jsonl", [spec.to_dict() for spec in all_query_specs])
    artifact_writer.write_jsonl("validation_results.jsonl", all_validations)
    artifact_writer.write_jsonl("query_execution_summaries_v2.jsonl", execution_summaries_v2)
    artifact_writer.write_jsonl("bundle_diversity_matrix_v2.jsonl", bundle_diversity_records_v2)
    artifact_writer.write_json("candidate_pool.json", {"variants": [item.to_dict() for item in variant_pool]})
    artifact_writer.write_json("question_bundle_pool.json", {"bundles": [item.to_dict() for item in question_bundle_pool]})
    artifact_writer.write_json("set_level_curation.json", final_curation.to_dict())
    artifact_writer.write_json("set_curation_audit_v2.json", final_curation.audit_v2)

    selected_bundle_id_set = set(final_curation.selected_bundle_ids)
    selected_bundles = [item for item in question_bundle_pool if item.bundle_id in selected_bundle_id_set]

    benchmark_package_dir = artifact_writer.run_dir / "benchmark_package"
    benchmark_package_dir.mkdir(parents=True, exist_ok=True)

    artifact_writer.write_json("benchmark_package/question_bundles.json", {"bundles": [item.to_dict() for item in selected_bundles]})
    artifact_writer.write_json("benchmark_package/curation_report.json", final_curation.to_dict())
    artifact_writer.write_json("benchmark_package/set_curation_audit_v2.json", final_curation.audit_v2)
    selected_variants = [
        variant
        for bundle_item in selected_bundles
        for variant in bundle_item.variants
        if variant.accepted_local
    ]
    selected_query_id_set = {variant.query_spec.query_id for variant in selected_variants}
    artifact_writer.write_json(
        "benchmark_package/queryspecs.json",
        {
            "queryspecs": [
                variant.query_spec.to_dict() for variant in selected_variants
            ]
        },
    )
    artifact_writer.write_json(
        "benchmark_package/query_execution_summaries_v2.json",
        {
            "summaries": [
                item for item in execution_summaries_v2 if item.get("query_id") in selected_query_id_set
            ]
        },
    )
    artifact_writer.write_json(
        "benchmark_package/bundle_diversity_matrix_v2.json",
        {
            "bundles": [
                item for item in bundle_diversity_records_v2 if item.get("bundle_id") in selected_bundle_id_set
            ]
        },
    )

    sql_lines: list[str] = []
    for bundle_item in selected_bundles:
        quality = bundle_item.bundle_quality or {}
        semantic_score = quality.get("semantic_diversity_score", "na")
        novelty_score = quality.get("informational_novelty_score", "na")
        sql_lines.append(
            f"-- {bundle_item.bundle_id} | {bundle_item.family} | semantic_diversity_score={semantic_score} "
            f"| novelty_score={novelty_score} | question={bundle_item.research_question.question}"
        )
        emitted = 0
        for idx, variant in enumerate(bundle_item.variants, start=1):
            if not variant.accepted_local:
                continue
            sql_lines.append(
                f"-- variant_{idx} | query_id={variant.query_spec.query_id} | role={variant.query_spec.variant_semantic_role} "
                f"| origin={variant.query_spec.sql_origin_mode} "
                f"| accepted={variant.accepted_local}"
            )
            if variant.query_spec.exemplar_sql_item_id:
                sql_lines.append(
                    f"-- exemplar={variant.query_spec.exemplar_sql_item_id} "
                    f"| source_dataset={variant.query_spec.exemplar_own_id} "
                    f"| source_url={variant.query_spec.exemplar_source_url}"
                )
            sql_lines.append(variant.query_spec.sql.rstrip(";") + ";")
            emitted += 1
        if emitted == 0:
            sql_lines.append("-- no locally accepted variants retained for this bundle")
        sql_lines.append("")
    artifact_writer.write_text("benchmark_package/selected_sql.sql", "\n".join(sql_lines).rstrip() + "\n")

    local_pass_variants = sum(1 for item in variant_pool if item.accepted_local)
    local_pass_bundles = sum(1 for item in question_bundle_pool if item.accepted_local)
    family_counter = Counter(item.family for item in selected_bundles)
    origin_counter = Counter(
        variant.query_spec.sql_origin_mode
        for bundle_item in selected_bundles
        for variant in bundle_item.variants
        if variant.accepted_local
    )

    shortfall_reasons: list[str] = []
    final_selected_query_variants = len(selected_variants)
    target_min_queries = 100
    if len(selected_bundles) < min_questions:
        shortfall_reasons.append(f"final_selected_questions={len(selected_bundles)} < min_questions={min_questions}")
        if local_pass_bundles < min_questions:
            shortfall_reasons.append(f"accepted_bundle_pool={local_pass_bundles} < min_questions={min_questions}")
        if len(all_research_questions) < question_budget:
            shortfall_reasons.append(
                f"question_budget_not_fully_used:generated_research_questions={len(all_research_questions)} "
                f"question_budget={question_budget}"
            )

        rejection_counter = Counter()
        for item in question_bundle_pool:
            if item.accepted_local:
                continue
            for code in item.rejected_reason_codes:
                rejection_counter[code] += 1
        if rejection_counter:
            top_reasons = rejection_counter.most_common(10)
            shortfall_reasons.append(
                "top_bundle_rejection_reason_codes=" + ",".join(f"{code}:{count}" for code, count in top_reasons)
            )

        for family in required_families:
            if family_counter.get(family, 0) == 0:
                shortfall_reasons.append(f"required_family_coverage_zero:{family}")

    if final_selected_query_variants < target_min_queries:
        shortfall_reasons.append(
            f"final_selected_query_variants={final_selected_query_variants} < target_min_queries={target_min_queries}"
        )
        if local_pass_variants < target_min_queries:
            shortfall_reasons.append(
                f"local_pass_variants={local_pass_variants} < target_min_queries={target_min_queries}"
            )
        if len(selected_bundles) * queries_per_question < target_min_queries:
            shortfall_reasons.append(
                f"selected_bundle_capacity={len(selected_bundles) * queries_per_question} < target_min_queries={target_min_queries}"
            )
        weak_novelty_selected = sum(
            1
            for item in selected_bundles
            if "BUNDLE_INFORMATIONAL_NOVELTY_WEAK" in item.bundle_validation.reason_codes
            or float((item.bundle_quality or {}).get("no_new_information_ratio", 0.0)) >= 0.38
        )
        if weak_novelty_selected:
            shortfall_reasons.append(f"selected_bundles_with_weak_novelty={weak_novelty_selected}")
        quality_drop_count = sum(1 for note in final_curation.notes if note.startswith("bundle_quality_drop:"))
        if quality_drop_count:
            shortfall_reasons.append(f"quality_filtered_bundle_count={quality_drop_count}")

    artifact_writer.write_json(
        "benchmark_package/shortfall_reasons.json",
        {
            "min_questions": min_questions,
            "max_questions": max_questions,
            "target_questions": question_budget,
            "final_selected_questions": len(selected_bundles),
            "target_min_queries": target_min_queries,
            "final_selected_query_variants": final_selected_query_variants,
            "shortfall_reasons": shortfall_reasons,
        },
    )

    package_summary = {
        "dataset_id": bundle.dataset_id,
        "run_id": artifact_writer.run_id,
        "min_questions": min_questions,
        "max_questions": max_questions,
        "target_questions": question_budget,
        "queries_per_question": queries_per_question,
        "min_pass_variants": min_pass_variants,
        "generated_research_questions": len(all_research_questions),
        "generated_query_variants": len(variant_pool),
        "local_pass_variants": local_pass_variants,
        "local_pass_question_bundles": local_pass_bundles,
        "final_selected_questions": len(selected_bundles),
        "final_selected_query_variants": final_selected_query_variants,
        "target_min_queries": target_min_queries,
        "required_families": required_families,
        "final_family_coverage": {family: family_counter.get(family, 0) for family in FIVE_FIXED_FAMILIES},
        "families_still_weak": [family for family in required_families if family_counter.get(family, 0) == 0],
        "shortfall_reasons": shortfall_reasons,
        "usage_summary": llm_runtime.summary,
        "sql_origin_distribution": dict(origin_counter),
        "sql_exemplar_context": exemplar_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact_writer.write_json("benchmark_package/package_summary.json", package_summary)

    return {
        "static_understanding": static_understanding.to_dict(),
        "operational_understanding": operational.to_dict(),
        "probe_count": len(probes),
        "research_question_count": len(all_research_questions),
        "query_variant_count": len(variant_pool),
        "local_pass_variant_count": local_pass_variants,
        "question_bundle_count": len(question_bundle_pool),
        "local_pass_bundle_count": local_pass_bundles,
        "final_selected_question_count": len(selected_bundles),
        "final_selected_query_variant_count": final_selected_query_variants,
        "final_family_coverage": package_summary["final_family_coverage"],
        "shortfall_reasons": shortfall_reasons,
        "set_level_curation": final_curation.to_dict(),
        "set_curation_audit_v2": final_curation.audit_v2,
        "usage_summary": llm_runtime.summary,
    }
