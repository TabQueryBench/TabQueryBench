#!/usr/bin/env python3
"""CLI entrypoint for STEP 2 benchmark self-evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_scoring.evaluation.io import load_evaluation_context
from tqb_scoring.evaluation.pipeline import run_evaluation_step2_v0_1


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def _parse_score_table_overrides(items: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --score-table-map item: {item}. Use <run_id_or_run_dir>=<path>.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid --score-table-map item: {item}. Use <run_id_or_run_dir>=<path>.")
        mapping[key] = Path(value).expanduser().resolve()
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STEP 2 benchmark self-evaluation.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Benchmark-construction run directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Evaluation output directory. Default: <run-dir>/evaluation")
    parser.add_argument("--compare-run-dir", type=Path, action="append", default=[], help="Additional run dirs for rank stability comparisons.")
    parser.add_argument(
        "--score-table-map",
        action="append",
        default=[],
        help="Override score table path mapping in format <run_id_or_run_dir>=<path>.",
    )

    parser.add_argument("--perturb-intensities", type=str, default="0.3,0.6", help="Comma-separated perturbation intensity grid.")
    parser.add_argument("--perturb-repeats", type=int, default=2, help="Perturbation repeat count per intensity.")
    parser.add_argument("--perturb-seed", type=int, default=42, help="Base seed for deterministic perturbations.")
    parser.add_argument("--max-eval-queries", type=int, default=0, help="Max number of queries to evaluate (0 means all).")

    parser.add_argument("--top-k", type=int, default=3, help="Top-k for rank stability overlap metric.")
    parser.add_argument("--rs-workload-weight", type=float, default=0.75)
    parser.add_argument("--rs-query-weight", type=float, default=0.25)
    parser.add_argument("--cir-lambda", type=float, default=0.7)
    parser.add_argument("--cir-query-floor-threshold", type=float, default=0.15)
    parser.add_argument("--cir-question-floor-cap", type=float, default=0.60)
    parser.add_argument("--cir-question-trim-ratio", type=float, default=0.20)

    parser.add_argument("--acr-support-min-ratio", type=float, default=0.20)
    parser.add_argument("--acr-support-min-abs", type=float, default=3.0)
    parser.add_argument("--acr-support-weight-clip-min", type=float, default=0.25)
    parser.add_argument("--acr-support-weight-clip-max", type=float, default=1.0)
    parser.add_argument("--acr-min-evaluable-valid-rewrites", type=int, default=2)
    parser.add_argument("--acr-confidence-valid-rewrites", type=int, default=4)
    parser.add_argument("--acr-query-floor-threshold", type=float, default=0.15)
    parser.add_argument("--acr-question-floor-cap", type=float, default=0.60)
    parser.add_argument("--acr-question-trim-ratio", type=float, default=0.20)

    # Legacy args retained for compatibility with older scripts.
    parser.add_argument("--near-duplicate-jaccard-threshold", type=float, default=0.92)
    parser.add_argument("--alignment-pass-threshold", type=float, default=0.45)
    parser.add_argument("--high-contamination-threshold", type=float, default=0.8)

    parser.add_argument(
        "--include-null-variant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include null perturbation variants.",
    )
    parser.add_argument(
        "--include-boot-variant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include bootstrap perturbation variants.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (run_dir / "evaluation")

    context = load_evaluation_context(run_dir)
    if not context.db_path.exists():
        raise FileNotFoundError(
            f"SQLite DB path not found from run artifacts: {context.db_path}. "
            "Please ensure the source run contains a valid run_manifest sqlite.db_path."
        )
    if not context.table_name:
        raise ValueError("table_name missing in run artifacts; cannot execute perturbation evaluation.")

    perturb_intensities = _parse_float_list(args.perturb_intensities)
    if not perturb_intensities:
        raise ValueError("--perturb-intensities must provide at least one value.")

    score_table_overrides = _parse_score_table_overrides(args.score_table_map)

    result = run_evaluation_step2_v0_1(
        context=context,
        output_dir=output_dir,
        compare_run_dirs=[item.expanduser().resolve() for item in args.compare_run_dir],
        score_table_overrides=score_table_overrides,
        perturb_intensities=perturb_intensities,
        perturb_repeats=args.perturb_repeats,
        perturb_seed=args.perturb_seed,
        max_eval_queries=(None if args.max_eval_queries <= 0 else args.max_eval_queries),
        include_null_variant=args.include_null_variant,
        include_boot_variant=args.include_boot_variant,
        top_k=args.top_k,
        near_duplicate_jaccard_threshold=args.near_duplicate_jaccard_threshold,
        alignment_pass_threshold=args.alignment_pass_threshold,
        high_contamination_threshold=args.high_contamination_threshold,
        cir_lambda=args.cir_lambda,
        cir_query_floor_threshold=args.cir_query_floor_threshold,
        cir_question_floor_cap=args.cir_question_floor_cap,
        cir_question_trim_ratio=args.cir_question_trim_ratio,
        acr_support_min_ratio=args.acr_support_min_ratio,
        acr_support_min_abs=args.acr_support_min_abs,
        acr_support_weight_clip_min=args.acr_support_weight_clip_min,
        acr_support_weight_clip_max=args.acr_support_weight_clip_max,
        acr_min_evaluable_valid_rewrites=args.acr_min_evaluable_valid_rewrites,
        acr_confidence_valid_rewrites=args.acr_confidence_valid_rewrites,
        acr_query_floor_threshold=args.acr_query_floor_threshold,
        acr_question_floor_cap=args.acr_question_floor_cap,
        acr_question_trim_ratio=args.acr_question_trim_ratio,
        rs_workload_weight=args.rs_workload_weight,
        rs_query_weight=args.rs_query_weight,
    )

    if args.verbose:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    else:
        compact = {
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "rank_stability_status": result["reports"]["rank_stability"].get("summary", {}).get("status"),
            "rank_stability_overall_score": result["summary"].get("readiness_snapshot", {}).get("rank_stability_overall_score"),
            "rs_workload_score": result["summary"].get("readiness_snapshot", {}).get("rs_workload_score"),
            "rs_query_score": result["summary"].get("readiness_snapshot", {}).get("rs_query_score"),
            "rs_query_status": result["summary"].get("readiness_snapshot", {}).get("rs_query_status"),
            "workload_cir_score": result["reports"]["cir"].get("summary", {}).get("workload_cir_score"),
            "workload_cir_effective_score": result["reports"]["cir"].get("summary", {}).get("workload_cir_effective_score"),
            "cir_evaluable_query_ratio": result["reports"]["cir"].get("summary", {}).get("evaluable_query_ratio"),
            "workload_acr_score": result["reports"]["acr"].get("summary", {}).get("workload_acr_score"),
            "workload_acr_effective_score": result["reports"]["acr"].get("summary", {}).get("workload_acr_effective_score"),
            "acr_evaluable_query_ratio": result["reports"]["acr"].get("summary", {}).get("evaluable_query_ratio"),
            "qe_raw": result["summary"].get("readiness_snapshot", {}).get("qe_raw"),
            "qe_effective": result["summary"].get("readiness_snapshot", {}).get("qe_effective"),
        }
        print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
