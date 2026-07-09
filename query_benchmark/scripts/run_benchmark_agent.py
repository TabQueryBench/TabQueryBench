#!/usr/bin/env python3
"""CLI entrypoint for benchmark construction system v1 (question-bundle mode)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.contracts import (
    build_manifest_v2,
    dataset_fingerprint,
    finalize_build_manifest_v2,
    git_revision,
)
from src.benchmark.facets import catalog_summary, load_family_facet_catalog
from src.benchmark.llm_runtime import BenchmarkLLMRuntime
from src.benchmark.pipeline import run_benchmark_construction_v1
from src.config.settings import (
    DATA_DIR,
    DEFAULT_USAGE_CSV_PATH,
    DEFAULT_SQL_EXEMPLAR_POOL_PATH,
    FAMILY_FACET_CATALOG_PATH,
    MODEL_PRICING_CONFIG_PATH,
    PROJECT_ROOT as SETTINGS_PROJECT_ROOT,
    RUNS_DIR,
    ensure_runtime_dirs,
)
from src.data.bundle import load_dataset_bundle
from src.db.csv_sqlite import materialize_dataset_to_sqlite
from src.logging.run_artifacts import RunArtifactWriter
from src.usage.logger import UsageCSVLogger
from src.usage.pricing import load_pricing_config


def build_run_id(dataset_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset_id}_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark construction agent v1 (dataset -> benchmark package).")
    parser.add_argument("--dataset-id", type=str, default="c2", help="Dataset ID under data root.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Must be gpt-4.1-mini for benchmark v1.")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR, help="Root directory containing datasets.")
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached SQLite DB when source CSV is unchanged.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose bundle progress.")

    parser.add_argument("--min-questions", type=int, default=20, help="Minimum selected question bundles required.")
    parser.add_argument("--max-questions", type=int, default=40, help="Maximum selected question bundles kept.")
    parser.add_argument("--target-questions", type=int, default=36, help="Generation target for research questions (clamped to min/max).")
    parser.add_argument("--queries-per-question", type=int, default=8, help="Number of SQL query variants to generate per question.")
    parser.add_argument(
        "--min-pass-variants",
        type=int,
        default=5,
        help="Minimum locally passed variants required for a question bundle to be accepted.",
    )

    parser.add_argument("--max-family-rounds", type=int, default=4, help="Maximum outer regeneration rounds.")
    parser.add_argument("--max-repairs", type=int, default=2, help="Maximum repairs per query variant in inner loop.")
    parser.add_argument(
        "--enable-sql-exemplars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use preprocessed SQL exemplars as direct/adapted candidates before de-novo generation.",
    )
    parser.add_argument(
        "--sql-exemplar-pool",
        type=Path,
        default=DEFAULT_SQL_EXEMPLAR_POOL_PATH,
        help="Path to preprocessed benchmark_sql_exemplar_pool.csv.",
    )
    parser.add_argument(
        "--exemplar-max-candidates-per-role",
        type=int,
        default=4,
        help="Max exemplar candidates scanned per role before fallback.",
    )

    # Backward-compatible aliases from previous version.
    parser.add_argument("--max-candidates", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--required-min-items", type=int, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--usage-csv", type=Path, default=DEFAULT_USAGE_CSV_PATH, help="Global usage CSV path.")
    parser.add_argument(
        "--pricing-config",
        type=Path,
        default=MODEL_PRICING_CONFIG_PATH,
        help="Pricing JSON config path.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_runtime_dirs()
    args = parse_args()

    if args.max_candidates is not None:
        args.target_questions = int(args.max_candidates)
    if args.required_min_items is not None:
        args.min_questions = int(args.required_min_items)

    if args.max_questions < args.min_questions:
        raise ValueError("--max-questions must be >= --min-questions")
    if args.queries_per_question <= 0:
        raise ValueError("--queries-per-question must be > 0")
    if args.min_pass_variants <= 0 or args.min_pass_variants > args.queries_per_question:
        raise ValueError("--min-pass-variants must be in [1, queries-per-question]")
    if args.exemplar_max_candidates_per_role <= 0:
        raise ValueError("--exemplar-max-candidates-per-role must be > 0")

    usage_logger = UsageCSVLogger(args.usage_csv)
    pricing_config = load_pricing_config(args.pricing_config)

    bundle = load_dataset_bundle(dataset_id=args.dataset_id, data_root=args.data_root, strict=True)
    sqlite_result = materialize_dataset_to_sqlite(bundle=bundle, use_cache=args.use_cache)
    family_facet_catalog = load_family_facet_catalog(FAMILY_FACET_CATALOG_PATH)

    run_id = build_run_id(bundle.dataset_id)
    artifact_writer = RunArtifactWriter(RUNS_DIR, run_id)
    build_id = f"build_{bundle.dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

    generation_config = {
        "min_questions": args.min_questions,
        "max_questions": args.max_questions,
        "target_questions": args.target_questions,
        "queries_per_question": args.queries_per_question,
        "min_pass_variants": args.min_pass_variants,
        "max_family_rounds": args.max_family_rounds,
        "max_repairs": args.max_repairs,
        "enable_sql_exemplars": args.enable_sql_exemplars,
        "sql_exemplar_pool": str(args.sql_exemplar_pool),
        "exemplar_max_candidates_per_role": args.exemplar_max_candidates_per_role,
    }
    curation_config = {
        "min_questions": args.min_questions,
        "max_questions": args.max_questions,
        "required_family_policy": "required_families_from_static_understanding",
        "quality_first_selection": True,
    }
    build_manifest = build_manifest_v2(
        build_id=build_id,
        run_id=run_id,
        dataset_id=bundle.dataset_id,
        dataset_fingerprint_obj=dataset_fingerprint(bundle),
        git_info=git_revision(SETTINGS_PROJECT_ROOT),
        pipeline_version="benchmark_pipeline_v1_contract_patch_v0_1",
        llm_config={
            "model": args.model,
            "enforce_model": "gpt-4.1-mini",
        },
        generation_config=generation_config,
        curation_config=curation_config,
        prompt_info={
            "available": False,
            "reason": "LLM runtime not initialized yet",
        },
    )
    artifact_writer.write_json("build_manifest_v2.json", build_manifest)

    manifest = {
        "run_id": run_id,
        "status": "running",
        "mode": "benchmark_construction_v1_question_bundle",
        "dataset_id": bundle.dataset_id,
        "model": args.model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "loaded_files_summary": bundle.loaded_files_summary(),
        "sqlite": {
            "db_path": str(sqlite_result.db_path),
            "table_name": sqlite_result.table_name,
            "row_count": sqlite_result.row_count,
            "cache_hit": sqlite_result.cache_hit,
            "cache_manifest_path": str(sqlite_result.manifest_path),
        },
        "cli_options": {
            "data_root": str(args.data_root),
            "use_cache": args.use_cache,
            **generation_config,
            "verbose": args.verbose,
        },
        "family_facet_catalog": {
            "path": str(FAMILY_FACET_CATALOG_PATH),
            "summary": catalog_summary(family_facet_catalog),
        },
        "build_manifest_v2_path": str(artifact_writer.run_dir / "build_manifest_v2.json"),
        "artifacts_dir": str(artifact_writer.run_dir),
    }
    artifact_writer.write_manifest(manifest)
    artifact_writer.append_trace(
        {
            "event_type": "run_start",
            "run_id": run_id,
            "dataset_id": bundle.dataset_id,
            "model": args.model,
        }
    )

    print(f"[run] run_id={run_id}")
    print(f"[run] dataset_id={bundle.dataset_id}")
    print(f"[run] sqlite_db={sqlite_result.db_path} table={sqlite_result.table_name} cache_hit={sqlite_result.cache_hit}")

    try:
        llm_runtime = BenchmarkLLMRuntime(
            model_name=args.model,
            dataset_id=bundle.dataset_id,
            run_id=run_id,
            usage_logger=usage_logger,
            pricing_config=pricing_config,
            artifact_writer=artifact_writer,
            enforce_model="gpt-4.1-mini",
        )

        result = run_benchmark_construction_v1(
            bundle=bundle,
            sqlite_result=sqlite_result,
            llm_runtime=llm_runtime,
            artifact_writer=artifact_writer,
            min_questions=args.min_questions,
            max_questions=args.max_questions,
            target_questions=args.target_questions,
            queries_per_question=args.queries_per_question,
            min_pass_variants=args.min_pass_variants,
            max_family_rounds=args.max_family_rounds,
            max_repairs=args.max_repairs,
            verbose=args.verbose,
            family_facet_catalog=family_facet_catalog,
            enable_sql_exemplars=args.enable_sql_exemplars,
            sql_exemplar_pool_path=args.sql_exemplar_pool,
            exemplar_max_candidates_per_role=args.exemplar_max_candidates_per_role,
        )

        usage_summary = {
            "dataset_id": bundle.dataset_id,
            "run_id": run_id,
            **result["usage_summary"],
        }
        artifact_writer.write_usage_summary(usage_summary)

        manifest["status"] = "completed"
        manifest["summary"] = {
            "probe_count": result["probe_count"],
            "research_question_count": result["research_question_count"],
            "query_variant_count": result["query_variant_count"],
            "local_pass_variant_count": result["local_pass_variant_count"],
            "question_bundle_count": result["question_bundle_count"],
            "local_pass_bundle_count": result["local_pass_bundle_count"],
            "final_selected_question_count": result["final_selected_question_count"],
            "final_selected_query_variant_count": result["final_selected_query_variant_count"],
            "final_family_coverage": result["final_family_coverage"],
            "shortfall_reasons": result["shortfall_reasons"],
        }
        manifest["usage_summary"] = usage_summary
        build_manifest = finalize_build_manifest_v2(
            build_manifest,
            prompt_info={
                "available": True,
                "prompt_signatures": llm_runtime.summary.get("prompt_signatures", {}),
            },
            summary=manifest["summary"],
        )
        artifact_writer.write_json("build_manifest_v2.json", build_manifest)

        print("[summary]", json.dumps(manifest["summary"], ensure_ascii=False))
        print(f"[run] benchmark_package={artifact_writer.run_dir / 'benchmark_package'}")
        print(f"[run] usage_summary={usage_summary}")
    except Exception as exc:  # noqa: BLE001
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        build_manifest = finalize_build_manifest_v2(
            build_manifest,
            prompt_info={
                "available": False,
                "reason": "run_failed_before_prompt_capture",
            },
            summary={"status": "failed", "error": str(exc)},
        )
        artifact_writer.write_json("build_manifest_v2.json", build_manifest)
        artifact_writer.append_trace(
            {
                "event_type": "run_error",
                "error": str(exc),
            }
        )
        raise
    finally:
        manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        artifact_writer.write_manifest(manifest)


if __name__ == "__main__":
    main()
