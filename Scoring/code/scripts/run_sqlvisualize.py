#!/usr/bin/env python3
"""Build analysis/validation figures under Evaluation/SQLvisualize."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_scoring.eval.common import DEFAULT_SQL_SOURCE_VERSION, SQL_SOURCE_VERSION_CHOICES
from tqb_scoring.eval.SQLvisualize.runner import run_sqlvisualize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SQL evaluation visualizations.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional output run tag.")
    parser.add_argument("--analysis-run-dir", type=Path, default=None, help="Optional existing Evaluation/analysis run dir.")
    parser.add_argument("--validation-run-dir", type=Path, default=None, help="Optional existing Evaluation/validation run dir.")
    parser.add_argument(
        "--analysis-engines",
        nargs="+",
        default=["cli-all"],
        help="Engine filter used if analysis must be rebuilt.",
    )
    parser.add_argument(
        "--analysis-sql-source-version",
        choices=list(SQL_SOURCE_VERSION_CHOICES),
        default=DEFAULT_SQL_SOURCE_VERSION,
        help="SQL source line used if analysis must be rebuilt.",
    )
    parser.add_argument(
        "--allow-analysis-rerun",
        action="store_true",
        help="Rebuild analysis if no non-empty local analysis run is available.",
    )
    parser.add_argument(
        "--force-analysis-rerun",
        action="store_true",
        help="Always rebuild analysis before plotting, even if an older non-empty run exists.",
    )
    parser.add_argument(
        "--analysis-latest-only",
        action="store_true",
        help="When rebuilding analysis, keep only latest synthetic assets per model/server.",
    )
    parser.add_argument(
        "--analysis-max-workers",
        type=int,
        default=4,
        help="Max workers for an analysis rebuild if needed.",
    )
    parser.add_argument("--latex-engine", type=str, default=None, help="Optional explicit LaTeX engine for the final report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_sqlvisualize(
        run_tag=args.run_tag,
        analysis_run_dir=args.analysis_run_dir,
        validation_run_dir=args.validation_run_dir,
        allow_analysis_rebuild=args.allow_analysis_rerun,
        force_analysis_rebuild=args.force_analysis_rerun,
        analysis_engines=tuple(args.analysis_engines),
        analysis_sql_source_version=str(args.analysis_sql_source_version),
        analysis_latest_only=args.analysis_latest_only,
        analysis_max_workers=max(1, args.analysis_max_workers),
        latex_engine=str(args.latex_engine) if args.latex_engine else None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
