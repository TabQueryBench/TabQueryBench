"""Stage-B benchmark self-evaluation pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmark.models import FIVE_FIXED_FAMILIES
from src.evaluation.acr import evaluate_acr
from src.evaluation.cir import evaluate_cir
from src.evaluation.io import EvaluationContext, write_json, write_jsonl
from src.evaluation.perturbations import build_perturbation_substrate
from src.evaluation.rank_stability import evaluate_rank_stability
from src.evaluation.visualization import generate_standard_evaluation_figures


def _discover_score_table(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "evaluation" / "model_scores.csv",
        run_dir / "evaluation" / "model_scores.json",
        run_dir / "model_scores.csv",
        run_dir / "model_scores.json",
        run_dir / "benchmark_package" / "model_scores.csv",
        run_dir / "benchmark_package" / "model_scores.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_build_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "build_manifest_v2.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _derive_rank_overall_score(rank_report: dict[str, Any]) -> float | None:
    top = rank_report.get("summary")
    if isinstance(top, dict) and top.get("rank_stability_score") is not None:
        return float(top.get("rank_stability_score") or 0.0)

    domains = rank_report.get("domains")
    if not isinstance(domains, dict):
        return None
    overall = domains.get("overall")
    if not isinstance(overall, dict):
        return None
    summary = overall.get("summary")
    if not isinstance(summary, dict):
        return None

    tau = float(summary.get("avg_kendall_tau") or 0.0)
    rho = float(summary.get("avg_spearman_rho") or 0.0)
    champion = float(summary.get("champion_retention_rate") or 0.0)
    topk = float(summary.get("avg_top_k_overlap") or 0.0)
    reversal = float(summary.get("avg_pairwise_reversal_ratio") or 0.0)
    return (tau + rho + champion + topk + (1.0 - reversal)) / 5.0


def _resolve_scored_builds(
    *,
    primary_context: EvaluationContext,
    compare_run_dirs: list[Path],
    score_table_overrides: dict[str, Path],
) -> list[dict[str, Any]]:
    run_dirs = [primary_context.run_dir] + [item.resolve() for item in compare_run_dirs]
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        key = str(run_dir)
        if key in seen:
            continue
        seen.add(key)

        run_id = run_dir.name
        build_manifest = _load_build_manifest(run_dir)

        score_path = score_table_overrides.get(run_id)
        if score_path is None:
            score_path = score_table_overrides.get(str(run_dir))
        if score_path is None:
            score_path = _discover_score_table(run_dir)

        if score_path is None:
            continue

        query_score_path = score_path.parent / "query_scores.jsonl"
        query_score_str = str(query_score_path.resolve()) if query_score_path.exists() else ""

        rows.append(
            {
                "run_id": run_id,
                "build_id": str(build_manifest.get("build_id") or ""),
                "score_table_path": str(score_path.resolve()),
                "query_score_path": query_score_str,
                "build_manifest_v2": build_manifest,
            }
        )

    return rows


def run_evaluation_step2_v0_1(
    *,
    context: EvaluationContext,
    output_dir: Path,
    compare_run_dirs: list[Path],
    score_table_overrides: dict[str, Path],
    perturb_intensities: list[float],
    perturb_repeats: int,
    perturb_seed: int,
    max_eval_queries: int | None,
    include_null_variant: bool,
    include_boot_variant: bool,
    top_k: int,
    # legacy args kept for compatibility; no longer headline metrics in CIR+ACR protocol.
    near_duplicate_jaccard_threshold: float = 0.92,  # noqa: ARG001
    alignment_pass_threshold: float = 0.45,  # noqa: ARG001
    high_contamination_threshold: float = 0.8,  # noqa: ARG001
    cir_lambda: float = 0.7,
    cir_query_floor_threshold: float = 0.15,
    cir_question_floor_cap: float = 0.60,
    cir_question_trim_ratio: float = 0.20,
    acr_support_min_ratio: float = 0.20,
    acr_support_min_abs: float = 3.0,
    acr_support_weight_clip_min: float = 0.25,
    acr_support_weight_clip_max: float = 1.0,
    acr_min_evaluable_valid_rewrites: int = 2,
    acr_confidence_valid_rewrites: int = 4,
    acr_query_floor_threshold: float = 0.15,
    acr_question_floor_cap: float = 0.60,
    acr_question_trim_ratio: float = 0.20,
    rs_workload_weight: float = 0.75,
    rs_query_weight: float = 0.25,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    # 1) Perturbation substrate (for CIR)
    perturb_output_dir = output_dir / "perturbations"
    perturb_manifest = build_perturbation_substrate(
        base_db_path=context.db_path,
        table_name=context.table_name,
        static_understanding=context.static_understanding,
        output_dir=perturb_output_dir,
        intensities=perturb_intensities,
        repeats=perturb_repeats,
        base_seed=perturb_seed,
        enabled_families=list(FIVE_FIXED_FAMILIES),
        include_null=include_null_variant,
        include_boot=include_boot_variant,
    )
    write_json(output_dir / "perturbation_manifest.json", perturb_manifest)

    # 2) CIR
    cir_report, cir_by_query, cir_exec_trace = evaluate_cir(
        query_specs=context.query_specs,
        perturbation_manifest=perturb_manifest,
        max_eval_queries=max_eval_queries,
        cir_lambda=cir_lambda,
        merge_dependency_bucket=True,
        include_cardinality=False,
        include_missingness=None,
        query_floor_threshold=cir_query_floor_threshold,
        question_floor_cap=cir_question_floor_cap,
        question_trim_ratio=cir_question_trim_ratio,
    )
    write_json(output_dir / "cir_report.json", cir_report)
    write_jsonl(output_dir / "cir_by_query.jsonl", cir_by_query)
    write_jsonl(output_dir / "cir_execution_trace.jsonl", cir_exec_trace)

    # 3) ACR
    acr_report, acr_by_query, acr_rewrite_trace, refinement_catalog = evaluate_acr(
        query_specs=context.query_specs,
        db_path=str(context.db_path),
        table_name=context.table_name,
        static_understanding=context.static_understanding,
        max_eval_queries=max_eval_queries,
        support_min_ratio=acr_support_min_ratio,
        support_min_abs=acr_support_min_abs,
        support_weight_clip_min=acr_support_weight_clip_min,
        support_weight_clip_max=acr_support_weight_clip_max,
        min_evaluable_valid_rewrites=acr_min_evaluable_valid_rewrites,
        confidence_valid_rewrites=acr_confidence_valid_rewrites,
        question_trim_ratio=acr_question_trim_ratio,
        query_floor_threshold=acr_query_floor_threshold,
        question_floor_cap=acr_question_floor_cap,
    )
    write_json(output_dir / "acr_report.json", acr_report)
    write_jsonl(output_dir / "acr_by_query.jsonl", acr_by_query)
    write_jsonl(output_dir / "acr_rewrite_trace.jsonl", acr_rewrite_trace)
    write_json(output_dir / "acr_refinement_catalog.json", refinement_catalog)

    # 4) Rank Stability (works with externally provided model score tables)
    scored_builds = _resolve_scored_builds(
        primary_context=context,
        compare_run_dirs=compare_run_dirs,
        score_table_overrides=score_table_overrides,
    )
    rank_report = evaluate_rank_stability(
        scored_builds=scored_builds,
        top_k=top_k,
        rs_workload_weight=rs_workload_weight,
        rs_query_weight=rs_query_weight,
    )
    write_json(output_dir / "rank_stability_report.json", rank_report)
    rank_overall_score = _derive_rank_overall_score(rank_report)
    rank_overall_summary = (
        rank_report.get("domains", {}).get("overall", {}).get("summary", {})
        if isinstance(rank_report.get("domains"), dict)
        else {}
    )
    cir_raw = cir_report.get("summary", {}).get("workload_cir_score")
    cir_eff = cir_report.get("summary", {}).get("workload_cir_effective_score")
    acr_raw = acr_report.get("summary", {}).get("workload_acr_score")
    acr_eff = acr_report.get("summary", {}).get("workload_acr_effective_score")
    qe_total_raw = None
    qe_total_effective = None
    if rank_overall_score is not None:
        try:
            qe_total_raw = (float(rank_overall_score) + float(cir_raw) + float(acr_raw)) / 3.0
            qe_total_effective = (float(rank_overall_score) + float(cir_eff) + float(acr_eff)) / 3.0
        except (TypeError, ValueError):
            qe_total_raw = None
            qe_total_effective = None

    visualization_manifest: dict[str, Any] = {"status": "skipped", "reason": "not_generated"}
    try:
        visualization_manifest = generate_standard_evaluation_figures(output_dir)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"visualization_generation_failed:{exc}")
        visualization_manifest = {"status": "error", "error": str(exc)}
    write_json(output_dir / "visualization_manifest.json", visualization_manifest)

    summary = {
        "contract_version": "evaluation_summary_v0_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "run_dir": str(context.run_dir),
            "dataset_id": context.dataset_id,
            "db_path": str(context.db_path),
            "table_name": context.table_name,
            "query_count": len(context.query_specs),
            "question_bundle_count": len(context.question_bundles),
        },
        "config": {
            "perturb_intensities": perturb_intensities,
            "perturb_repeats": perturb_repeats,
            "perturb_seed": perturb_seed,
            "max_eval_queries": max_eval_queries,
            "include_null_variant": include_null_variant,
            "include_boot_variant": include_boot_variant,
            "top_k": top_k,
            "cir_lambda": cir_lambda,
            "cir_query_floor_threshold": cir_query_floor_threshold,
            "cir_question_floor_cap": cir_question_floor_cap,
            "cir_question_trim_ratio": cir_question_trim_ratio,
            "acr_support_min_ratio": acr_support_min_ratio,
            "acr_support_min_abs": acr_support_min_abs,
            "acr_support_weight_clip_min": acr_support_weight_clip_min,
            "acr_support_weight_clip_max": acr_support_weight_clip_max,
            "acr_min_evaluable_valid_rewrites": acr_min_evaluable_valid_rewrites,
            "acr_confidence_valid_rewrites": acr_confidence_valid_rewrites,
            "acr_query_floor_threshold": acr_query_floor_threshold,
            "acr_question_floor_cap": acr_question_floor_cap,
            "acr_question_trim_ratio": acr_question_trim_ratio,
            "rs_workload_weight": rs_workload_weight,
            "rs_query_weight": rs_query_weight,
        },
        "outputs": {
            "cir_report": "cir_report.json",
            "acr_report": "acr_report.json",
            "rank_stability_report": "rank_stability_report.json",
            "perturbation_manifest": "perturbation_manifest.json",
            "acr_refinement_catalog": "acr_refinement_catalog.json",
            "visualization_manifest": "visualization_manifest.json",
        },
        "readiness_snapshot": {
            "rank_stability": rank_report.get("summary", {}).get("status"),
            "rank_stability_overall_score": (round(rank_overall_score, 6) if rank_overall_score is not None else None),
            "rs_workload_score": rank_overall_summary.get("rs_workload_score"),
            "rs_query_score": rank_overall_summary.get("rs_query_score"),
            "rs_query_status": rank_overall_summary.get("rs_query_status"),
            "cir": cir_report.get("summary", {}).get("workload_cir_score"),
            "cir_effective": cir_report.get("summary", {}).get("workload_cir_effective_score"),
            "cir_evaluable_query_ratio": cir_report.get("summary", {}).get("evaluable_query_ratio"),
            "acr": acr_report.get("summary", {}).get("workload_acr_score"),
            "acr_effective": acr_report.get("summary", {}).get("workload_acr_effective_score"),
            "acr_evaluable_query_ratio": acr_report.get("summary", {}).get("evaluable_query_ratio"),
            "qe_raw": (round(qe_total_raw, 6) if qe_total_raw is not None else None),
            "qe_effective": (round(qe_total_effective, 6) if qe_total_effective is not None else None),
            # legacy aliases (kept for compatibility with existing scripts/slides)
            "qe_total_raw": (round(qe_total_raw, 6) if qe_total_raw is not None else None),
            "qe_total_effective": (round(qe_total_effective, 6) if qe_total_effective is not None else None),
        },
        "warnings": warnings,
    }
    write_json(output_dir / "evaluation_summary.json", summary)

    return {
        "summary": summary,
        "reports": {
            "cir": cir_report,
            "acr": acr_report,
            "rank_stability": rank_report,
        },
    }
