"""Publish paper-ready appendix/time-cost artifacts under the requested layout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from tqb_scoring.eval.appendix_tables.runner import (
    _compile_tex,
    _detect_latex_engine,
    _model_label,
    _render_generated_section,
    run_appendix_table_bundle,
)
from tqb_scoring.eval.common import now_run_tag, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT.parent / "results"
PAPER_ROOT = PROJECT_ROOT / "Paper"

TASK_NAME = "time_cost"
FINAL_DIR_NAME = "final"


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_paper_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = []
    for path in PAPER_ROOT.rglob("main.tex"):
        if "paper_backup" in str(path).lower():
            continue
        if path.parent.name == "out":
            continue
        candidates.append(path.parent.resolve())
    if not candidates:
        raise FileNotFoundError("Could not locate active paper directory.")
    candidates.sort(key=lambda item: (len(item.parts), str(item)))
    return candidates[0]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sync_final_to_paper_figures(*, final_dir: Path, paper_dir: Path) -> Path:
    figures_dir = paper_dir / "figures" / TASK_NAME / FINAL_DIR_NAME
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(final_dir, figures_dir)
    return figures_dir


def _render_missing_time_summary_tex(rows: list[dict[str, Any]]) -> str:
    body_lines = []
    for row in rows:
        body_lines.append(
            " & ".join(
                [
                    row["model_label"],
                    str(row["missing_train_count"]),
                    str(row["missing_gen_count"]),
                ]
            )
            + r" \\"
        )
    body = "\n".join(body_lines)
    return rf"""
\begin{{table}}[t]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}lcc@{{}}}}
\toprule
Model & Missing train time & Missing gen time \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Datasets lacking audited train or generation time, summarized by model. Full lists are provided in the companion CSV files.}}
\label{{tab:timecost_missing_time_summary}}
\end{{table}}
""".strip() + "\n"


def _prepare_final_folder(
    *,
    latest_dir: Path,
    final_dir: Path,
) -> dict[str, Path]:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    latest_tables = latest_dir / "tables"
    latest_latex = latest_dir / "latex"
    latest_manifest = latest_dir / "manifest.json"

    for csv_path in latest_tables.glob("*.csv"):
        shutil.copy2(csv_path, final_dir / csv_path.name)

    component_names = [
        "model_coverage_matrix_generated.tex",
        "model_global_summary_generated.tex",
        "full_results_c_generated.tex",
        "full_results_m_generated.tex",
        "full_results_n_generated.tex",
        "standalone_appendix_full_results_generated.tex",
    ]
    for name in component_names:
        shutil.copy2(latest_latex / name, final_dir / name)

    standalone_text = (latest_latex / "standalone_appendix_tables.tex").read_text(encoding="utf-8")
    _write_text(final_dir / "TimeCost.tex", standalone_text)

    paper_input = _render_generated_section("../../Evaluation/time_cost/final")
    _write_text(final_dir / "TimeCostPaperInput.tex", paper_input)

    missing_rows = _read_csv_rows(final_dir / "model_missing_time_summary.csv")
    _write_text(final_dir / "TimeCostMissingTimeSummary.tex", _render_missing_time_summary_tex(missing_rows))

    if latest_manifest.exists():
        shutil.copy2(latest_manifest, final_dir / "manifest.json")

    return {
        "standalone_tex": final_dir / "TimeCost.tex",
        "paper_input_tex": final_dir / "TimeCostPaperInput.tex",
        "missing_time_tex": final_dir / "TimeCostMissingTimeSummary.tex",
    }


def run_time_cost_bundle(
    *,
    run_tag: str,
    analysis_run_dir: Path | None = None,
    validation_run_dir: Path | None = None,
    paper_dir: Path | None = None,
    compile_pdf: bool = True,
    latex_engine: str | None = None,
    runtime_audit_csv: Path | None = None,
    rebuild_runtime_audit: bool = False,
) -> dict[str, Any]:
    paper_dir_resolved = _resolve_paper_dir(paper_dir)
    result = run_appendix_table_bundle(
        run_tag=run_tag,
        analysis_run_dir=analysis_run_dir,
        validation_run_dir=validation_run_dir,
        paper_dir=paper_dir_resolved,
        compile_pdf=False,
        latex_engine=latex_engine,
        runtime_audit_csv=runtime_audit_csv,
        rebuild_runtime_audit=rebuild_runtime_audit,
        task_name=TASK_NAME,
    )

    latest_dir = Path(result["latest_dir"]).resolve()
    final_dir = (OUTPUT_ROOT / TASK_NAME / FINAL_DIR_NAME).resolve()
    published = _prepare_final_folder(latest_dir=latest_dir, final_dir=final_dir)
    paper_figures_dir = _sync_final_to_paper_figures(final_dir=final_dir, paper_dir=paper_dir_resolved)

    standalone_pdf: Path | None = None
    paper_pdf: Path | None = None
    engine_used: str | None = None
    if compile_pdf:
        engine = _detect_latex_engine(latex_engine)
        if engine is None:
            raise RuntimeError("No LaTeX engine found. Install tectonic/pdflatex or pass --latex-engine.")
        engine_used = engine[0]
        standalone_pdf = _compile_tex(engine, published["standalone_tex"], final_dir)
        compiled_paper_pdf = _compile_tex(engine, paper_dir_resolved / "main.tex", paper_dir_resolved / "out")
        paper_pdf = final_dir / "TimeCost_full_paper.pdf"
        shutil.copy2(compiled_paper_pdf, paper_pdf)

    manifest_path = final_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(
        {
            "published_task": TASK_NAME,
            "final_dir": str(final_dir),
            "paper_dir": str(paper_dir_resolved),
            "paper_figures_dir": str(paper_figures_dir),
            "standalone_tex": str(published["standalone_tex"]),
            "paper_input_tex": str(published["paper_input_tex"]),
            "missing_time_tex": str(published["missing_time_tex"]),
            "standalone_pdf": str(standalone_pdf) if standalone_pdf else None,
            "paper_pdf": str(paper_pdf) if paper_pdf else None,
            "compile_pdf": compile_pdf,
            "latex_engine": engine_used,
        }
    )
    write_json(manifest_path, manifest)

    return {
        "run_dir": result["run_dir"],
        "latest_dir": latest_dir,
        "final_dir": final_dir,
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the TimeCost appendix/final artifact bundle.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag.")
    parser.add_argument("--analysis-run-dir", type=Path, default=None, help="Existing analysis run dir.")
    parser.add_argument("--validation-run-dir", type=Path, default=None, help="Existing validation run dir.")
    parser.add_argument("--paper-dir", type=Path, default=None, help="Paper directory containing main.tex.")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF compilation.")
    parser.add_argument("--latex-engine", type=str, default=None, help="Explicit LaTeX engine executable.")
    parser.add_argument("--runtime-audit-csv", type=Path, default=None, help="Optional existing runtime audit CSV.")
    parser.add_argument("--rebuild-runtime-audit", action="store_true", help="Force rebuilding runtime audit from raw logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_time_cost_bundle(
        run_tag=args.run_tag or now_run_tag(),
        analysis_run_dir=args.analysis_run_dir,
        validation_run_dir=args.validation_run_dir,
        paper_dir=args.paper_dir,
        compile_pdf=not args.skip_pdf,
        latex_engine=args.latex_engine,
        runtime_audit_csv=args.runtime_audit_csv,
        rebuild_runtime_audit=args.rebuild_runtime_audit,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
