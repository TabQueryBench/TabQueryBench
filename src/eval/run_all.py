"""Unified entrypoint for synthetic-data evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.appendix_tables.runner import run_appendix_table_bundle
from src.eval.analysis.runner import run_sql_analysis
from src.eval.common import (
    DEFAULT_SQL_SOURCE_VERSION,
    SQL_SOURCE_VERSION_CHOICES,
    list_dataset_ids,
    now_run_tag,
)
from src.eval.sql_cross_version_compare.runner import _parse_analysis_runs, run_sql_cross_version_compare
from src.eval.distance.runner import run_distance_evaluation
from src.eval.distance_query_scatter.runner import run_distance_query_scatter
from src.eval.sql_eval.runner import resolve_latest_analysis_run_dir, run_sql_rank_stability
from src.eval.time_cost.runner import run_time_cost_bundle
from src.eval.validation.runner import run_validation_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified synthetic-data evaluation.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["distance", "validation", "analysis", "sql_eval"],
        choices=[
            "distance",
            "validation",
            "analysis",
            "sql_eval",
            "sql_cross_version_compare",
            "appendix_tables",
            "distance_query_scatter",
            "time_cost",
        ],
        help="Tasks to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset IDs. Default: all datasets with train splits.",
    )
    parser.add_argument("--run-tag", type=str, default=None, help="Optional shared run tag.")
    parser.add_argument("--keep-all-assets", action="store_true", help="Use all synthetic assets, not just latest per model/server.")
    parser.add_argument(
        "--root-names",
        nargs="*",
        default=None,
        help=(
            "Optional synthetic root names to evaluate. "
            "Example: TabQueryBench-SynDataSuccess-main"
        ),
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["cli"],
        help="SQL-run engines to include for analysis, e.g. cli cli-all.",
    )
    parser.add_argument(
        "--sql-source-version",
        type=str,
        choices=list(SQL_SOURCE_VERSION_CHOICES),
        default=DEFAULT_SQL_SOURCE_VERSION,
        help="Which normalized SQL source line to analyze: v1 (legacy) or v2/v3/v4 (current workload lines).",
    )
    parser.add_argument(
        "--first-sql-only",
        action="store_true",
        help="Only analyze the first SQL statement from each generated_sql.sql file.",
    )
    parser.add_argument(
        "--max-sql-per-dataset",
        type=int,
        default=0,
        help="Optional cap for SQL statements per dataset (0 means all).",
    )
    parser.add_argument(
        "--analysis-row-limit",
        type=int,
        default=0,
        help="Maximum rows fetched per SQL execution during analysis (0 means fetch all rows).",
    )
    parser.add_argument(
        "--analysis-run-dir",
        type=Path,
        default=None,
        help="Optional existing analysis run dir when running sql_eval only.",
    )
    parser.add_argument(
        "--comparison-analysis-run",
        action="append",
        default=[],
        help="Versioned analysis run in the form v2=/path/to/run. Repeat for v3/v4 when running sql_cross_version_compare.",
    )
    parser.add_argument(
        "--distance-run-dir",
        type=Path,
        default=None,
        help="Optional existing distance run dir for downstream comparison tasks.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k parameter for rank-stability summaries.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Dataset-level parallel workers for distance/validation (and future task runners).",
    )
    parser.add_argument("--paper-dir", type=Path, default=None, help="Paper directory for appendix table builds.")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF compilation for appendix table builds.")
    parser.add_argument("--latex-engine", type=str, default=None, help="Explicit LaTeX engine for appendix table builds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_ids = args.datasets or list_dataset_ids()
    run_tag = args.run_tag or now_run_tag()
    latest_only = not args.keep_all_assets
    print(
        "[run_all] start"
        f" | run_tag={run_tag}"
        f" | tasks={','.join(args.tasks)}"
        f" | datasets={len(dataset_ids)}"
        f" | latest_only={latest_only}"
        f" | max_workers={args.max_workers}",
        flush=True,
    )

    results: dict[str, dict] = {}
    if "distance" in args.tasks:
        results["distance"] = run_distance_evaluation(
            run_tag=run_tag,
            datasets=dataset_ids,
            latest_only=latest_only,
            max_workers=max(1, args.max_workers),
            latex_engine=args.latex_engine,
            root_names=args.root_names,
        )
    if "validation" in args.tasks:
        results["validation"] = run_validation_evaluation(
            run_tag=run_tag,
            datasets=dataset_ids,
            latest_only=latest_only,
            max_workers=max(1, args.max_workers),
            root_names=args.root_names,
        )
    if "analysis" in args.tasks:
        results["analysis"] = run_sql_analysis(
            run_tag=run_tag,
            datasets=dataset_ids,
            latest_only=latest_only,
            engines=tuple(args.engines),
            sql_source_version=str(args.sql_source_version),
            include_all_sql_statements=not args.first_sql_only,
            max_sql_per_dataset=args.max_sql_per_dataset,
            query_row_limit=args.analysis_row_limit,
            max_workers=max(1, args.max_workers),
            latex_engine=args.latex_engine,
            root_names=args.root_names,
        )
    if "sql_eval" in args.tasks:
        analysis_run_dir = args.analysis_run_dir
        if analysis_run_dir is None:
            analysis_run_dir = results.get("analysis", {}).get("run_dir")
        if analysis_run_dir is None:
            analysis_run_dir = resolve_latest_analysis_run_dir()
        if analysis_run_dir is None:
            raise RuntimeError("sql_eval requested but no analysis run dir is available.")
        results["sql_eval"] = run_sql_rank_stability(
            run_tag=run_tag,
            analysis_run_dir=Path(analysis_run_dir),
            top_k=args.top_k,
            latex_engine=args.latex_engine,
            sql_source_version_override=str(args.sql_source_version),
        )
    if "sql_cross_version_compare" in args.tasks:
        comparison_runs = _parse_analysis_runs(list(args.comparison_analysis_run or []))
        if len(comparison_runs) < 2:
            raise RuntimeError(
                "sql_cross_version_compare requested but fewer than 2 --comparison-analysis-run values were provided."
            )
        results["sql_cross_version_compare"] = run_sql_cross_version_compare(
            run_tag=run_tag,
            analysis_runs=comparison_runs,
            top_k=args.top_k,
        )
    if "appendix_tables" in args.tasks:
        results["appendix_tables"] = run_appendix_table_bundle(
            run_tag=run_tag,
            analysis_run_dir=results.get("analysis", {}).get("run_dir"),
            validation_run_dir=results.get("validation", {}).get("run_dir"),
            paper_dir=args.paper_dir,
            compile_pdf=not args.skip_pdf,
            latex_engine=args.latex_engine,
        )
    if "distance_query_scatter" in args.tasks:
        analysis_run_dir = args.analysis_run_dir or results.get("analysis", {}).get("run_dir")
        distance_run_dir = args.distance_run_dir or results.get("distance", {}).get("run_dir")
        results["distance_query_scatter"] = run_distance_query_scatter(
            run_tag=run_tag,
            analysis_run_dir=Path(analysis_run_dir) if analysis_run_dir else None,
            distance_run_dir=Path(distance_run_dir) if distance_run_dir else None,
            compile_pdf=not args.skip_pdf,
            latex_engine=args.latex_engine,
        )
    if "time_cost" in args.tasks:
        results["time_cost"] = run_time_cost_bundle(
            run_tag=run_tag,
            analysis_run_dir=results.get("analysis", {}).get("run_dir"),
            validation_run_dir=results.get("validation", {}).get("run_dir"),
            paper_dir=args.paper_dir,
            compile_pdf=not args.skip_pdf,
            latex_engine=args.latex_engine,
        )

    payload = {
        "status": "ok",
        "run_tag": run_tag,
        "datasets": dataset_ids,
        "tasks": args.tasks,
        "max_workers": args.max_workers,
        "root_names": list(args.root_names or []),
        "requested_sql_source_version": str(args.sql_source_version),
        "resolved_sql_source_versions": {
            task: (
                result.get("manifest", {}).get("sql_source_version")
                or result.get("manifest", {}).get("analysis_sql_source_version")
                or (
                    ",".join(sorted((result.get("manifest", {}).get("versions") or {}).keys()))
                    if isinstance(result.get("manifest", {}).get("versions"), dict)
                    else None
                )
            )
            for task, result in results.items()
        },
        "results": {
            task: {
                "run_dir": str(result.get("run_dir").resolve()) if result.get("run_dir") else None,
                "manifest": result.get("manifest", {}),
            }
            for task, result in results.items()
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
