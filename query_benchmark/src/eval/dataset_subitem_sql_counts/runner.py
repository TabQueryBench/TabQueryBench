"""Build dataset x canonical-subitem SQL count tables from current evaluation SQL."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval.analytics_contract import (
    ANALYTICS_CONTRACT_VERSION,
    CANONICAL_ANALYTICS_SUBITEMS,
    annotate_query_row_with_contract,
)
from src.eval.common import (
    DEFAULT_SQL_SOURCE_VERSION,
    PROVENANCE_CONTRACT_VERSION,
    SQL_SOURCE_VERSION_CHOICES,
    list_dataset_ids,
    load_latest_sql_queries_by_dataset,
    make_task_run_dir,
    normalize_sql_source_version,
    now_run_tag,
    sql_source_description,
    sql_source_family,
    sql_source_label,
    sql_source_line_version,
    sql_source_root,
    write_csv,
    write_json,
)
from src.eval.final_outputs import copy_files, render_pdf_to_png, task_version_final_dir, write_versioned_final_readme
from src.eval.query_fivepart_breakdown.common_heatmap_palette import heatmap_hex, text_hex_for_fill

TASK_NAME = "dataset_subitem_sql_counts"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "Evaluation" / TASK_NAME
FINAL_DIR = OUTPUT_ROOT / "final"

FAMILY_DISPLAY = {
    "subgroup_structure": "Subgroup",
    "conditional_dependency_structure": "Conditional",
    "tail_rarity_structure": "Tail",
    "missingness_structure": "Missingness",
}

SUBITEM_ORDER: list[str] = [
    subitem_id
    for family_id in CANONICAL_ANALYTICS_SUBITEMS
    for subitem_id in CANONICAL_ANALYTICS_SUBITEMS[family_id]
]

TABLE_FIELDNAMES = [
    "dataset_id",
    "total_sql",
    *SUBITEM_ORDER,
]

DETAIL_FIELDNAMES = [
    "dataset_id",
    "total_sql",
    "canonical_contract_sql",
    "non_contract_sql",
    *SUBITEM_ORDER,
]

QUERY_FIELDNAMES = [
    "provenance_contract_version",
    "dataset_id",
    "question_id",
    "query_id",
    "sql_index",
    "family_id",
    "canonical_subitem_id",
    "subitem_inference_source",
    "subitem_inference_note",
    "intended_facet_id",
    "variant_semantic_role",
    "template_id",
    "template_name",
    "stable_question_id",
    "query_identity_stable_key",
    "engine",
    "model",
    "source_run_id",
    "sql_source_family",
    "sql_source_line_version",
    "sql_source_version",
    "sql_source_label",
    "sql_source_description",
    "sql_source_root",
    "sql_source_registry_root",
    "sql_source_registry_version",
    "sql_source_kind",
    "sql_source_selection_mode",
    "sql_origin_path",
    "sql_source_file_path",
    "sql_source_file_sha256",
    "sql_source_manifest_path",
    "sql_source_registry_path",
    "question",
    "sql",
]

LOCAL_TEX_COMPILERS = [
    PROJECT_ROOT / "tools" / "tectonic" / "tectonic-0.16.9" / "tectonic.exe",
    PROJECT_ROOT / "tools" / "tectonic-0.16.9" / "tectonic.exe",
    PROJECT_ROOT / "Evaluation" / "model_radar" / "_build_tools" / "tectonic" / "tectonic.exe",
]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sql_source_stats_from_rows(query_rows: list[dict[str, Any]], requested_version: str) -> dict[str, Any]:
    normalized = normalize_sql_source_version(requested_version)
    sample = query_rows[0] if query_rows else {}
    return {
        "provenance_contract_version": str(
            sample.get("provenance_contract_version") or PROVENANCE_CONTRACT_VERSION
        ),
        "sql_source_family": str(sample.get("sql_source_family") or sql_source_family(normalized)),
        "sql_source_line_version": str(sample.get("sql_source_line_version") or sql_source_line_version(normalized)),
        "sql_source_version": str(sample.get("sql_source_version") or normalized),
        "sql_source_label": str(sample.get("sql_source_label") or sql_source_label(normalized)),
        "sql_source_description": str(sample.get("sql_source_description") or sql_source_description(normalized)),
        "sql_source_root": str(sample.get("sql_source_root") or sql_source_root(normalized).resolve()),
        "sql_source_registry_root": str(sample.get("sql_source_registry_root") or ""),
    }


def _dataset_sort_key(dataset_id: str) -> tuple[str, int, str]:
    text = str(dataset_id or "").strip().lower()
    match = re.fullmatch(r"([a-zA-Z]+)(\d+)", text)
    if match:
        return match.group(1), int(match.group(2)), text
    prefix = re.match(r"[a-zA-Z]+", text)
    digits = re.search(r"(\d+)", text)
    return (
        prefix.group(0) if prefix else text,
        int(digits.group(1)) if digits else 10**9,
        text,
    )


def _latex_escape(text: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = str(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _rot_header_text(subitem_id: str) -> str:
    return _latex_escape(subitem_id)


def _blend_hex_with_white(fill_hex: str, keep_ratio: float = 0.58) -> str:
    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
    channels = [int(fill_hex[i : i + 2], 16) for i in range(0, 6, 2)]
    blended = []
    for channel in channels:
        value = int(round((255 * (1.0 - keep_ratio)) + (channel * keep_ratio)))
        blended.append(max(0, min(255, value)))
    return "".join(f"{value:02X}" for value in blended)


def _format_heatmap_count_cell(value: int, max_value: int) -> str:
    if value <= 0 or max_value <= 0:
        return str(value)
    normalized = float(value) / float(max_value)
    scaled = 0.14 + (0.68 * math.sqrt(normalized))
    fill_hex = _blend_hex_with_white(heatmap_hex(scaled), keep_ratio=0.56)
    text_hex = text_hex_for_fill(fill_hex)
    return rf"\cellcolor[HTML]{{{fill_hex}}}\textcolor[HTML]{{{text_hex}}}{{{value}}}"


def _build_header_rows() -> tuple[str, str]:
    top_cells = [
        r"\multirow{2}{*}{Dataset}",
        r"\multirow{2}{*}{Total SQL}",
    ]
    second_cells = ["", ""]
    for family_id, subitems in CANONICAL_ANALYTICS_SUBITEMS.items():
        top_cells.append(rf"\multicolumn{{{len(subitems)}}}{{c}}{{{_latex_escape(FAMILY_DISPLAY[family_id])}}}")
        second_cells.extend(rf"\rotcell{{{_rot_header_text(subitem_id)}}}" for subitem_id in subitems)
    return " & ".join(top_cells) + r" \\", " & ".join(second_cells) + r" \\"


def _build_longtable_rows(
    rows: list[dict[str, Any]],
    *,
    heatmap: bool,
) -> list[str]:
    maxima = {field: max(int(row.get(field) or 0) for row in rows) for field in TABLE_FIELDNAMES[1:]} if rows else {}
    body: list[str] = []
    for row in rows:
        cells: list[str] = [_latex_escape(row["dataset_id"])]
        for field in TABLE_FIELDNAMES[1:]:
            value = int(row.get(field) or 0)
            if heatmap:
                cells.append(_format_heatmap_count_cell(value, maxima.get(field, 0)))
            else:
                cells.append(str(value))
        body.append(" & ".join(cells) + r" \\")
    return body


def _build_tex_document(
    rows: list[dict[str, Any]],
    *,
    stats: dict[str, Any],
    heatmap: bool,
) -> str:
    column_spec = "@{}l>{\\centering\\arraybackslash}m{0.82cm}" + ">{\\centering\\arraybackslash}m{1.02cm}" * len(SUBITEM_ORDER) + "@{}"
    title = "Dataset $\\times$ canonical subitem SQL counts used by the current evaluation"
    caption = (
        "Counts of SQL statements per dataset and canonical analytics subitem. "
        "SQL source and subitem mapping are aligned with the current analysis scorer."
    )
    header_top, header_second = _build_header_rows()
    table_rows = _build_longtable_rows(rows, heatmap=heatmap)

    note_lines = [
        rf"\textbf{{Notes.}} SQL statements are loaded from \texttt{{{_latex_escape(stats['sql_source_label'])}}} ({_latex_escape(stats['sql_source_description'])}), filtered to engines={_latex_escape(','.join(stats['engine_filter']))} and expanded with \texttt{{split\_sql\_statements(...)}} when needed. This matches the SQL-loading behavior used by \texttt{{src.eval.analysis.runner}}.",
        rf"Canonical subitems are assigned with \texttt{{annotate\_query\_row\_with\_contract(...)}} from \texttt{{src.eval.analytics\_contract}}, using the same facet $\rightarrow$ role $\rightarrow$ heuristic fallback order as the formal scorer. Missing \texttt{{intended\_facet\_id}} / \texttt{{variant\_semantic\_role}} therefore reuse the current contract fallback instead of a custom rule.",
        rf"The frozen 10-subitem table intentionally excludes \texttt{{cardinality/range}} because it is not part of the README-aligned canonical analytics subitem contract. Total SQL = {stats['total_sql']}, canonical-contract SQL = {stats['canonical_contract_sql']}, out-of-contract SQL = {stats['non_contract_sql']}.",
    ]
    if heatmap:
        note_lines.append(
            r"Heatmap tint is a shallow YlGnBu-style fill normalized independently within each count column; zero-count cells stay unfilled."
        )
    else:
        note_lines.append(
            r"The main table keeps exact counts without tint so it remains appendix-friendly in single-column layout."
        )

    return "\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[margin=0.58in]{geometry}",
            r"\usepackage[table]{xcolor}",
            r"\usepackage{array}",
            r"\usepackage{booktabs}",
            r"\usepackage{longtable}",
            r"\usepackage{multirow}",
            r"\usepackage{graphicx}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{lmodern}",
            r"\setlength{\LTleft}{0pt}",
            r"\setlength{\LTright}{0pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\setlength{\tabcolsep}{2.4pt}",
            r"\newcommand{\rotcell}[1]{\rotatebox[origin=c]{65}{\parbox{2.55cm}{\centering\scriptsize\ttfamily #1}}}",
            r"\begin{document}",
            r"\thispagestyle{empty}",
            rf"\noindent\textbf{{{title}}}\\[0.25em]",
            rf"\noindent\footnotesize {caption}\\[0.8em]",
            r"\scriptsize",
            rf"\begin{{longtable}}{{{column_spec}}}",
            r"\toprule",
            header_top,
            header_second,
            r"\midrule",
            r"\endfirsthead",
            rf"\multicolumn{{{len(TABLE_FIELDNAMES)}}}{{r}}{{\footnotesize Continued from previous page}}\\",
            r"\toprule",
            header_top,
            header_second,
            r"\midrule",
            r"\endhead",
            rf"\midrule \multicolumn{{{len(TABLE_FIELDNAMES)}}}{{r}}{{\footnotesize Continued on next page}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
            *table_rows,
            r"\end{longtable}",
            r"\normalsize",
            r"\vspace{0.25em}",
            *[rf"\noindent\footnotesize {line}\\" for line in note_lines],
            r"\end{document}",
            "",
        ]
    )


def _find_latex_compiler(explicit: str | None = None) -> Path | None:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    for path in LOCAL_TEX_COMPILERS:
        if path.exists():
            candidates.append(str(path))
    for name in ["tectonic", "pdflatex", "xelatex", "latexmk"]:
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _compile_tex(tex_path: Path, *, latex_engine: str | None = None) -> tuple[Path, Path]:
    compiler = _find_latex_compiler(latex_engine)
    if compiler is None:
        raise RuntimeError("No LaTeX engine found. Install tectonic/pdflatex or pass --latex-engine.")

    pdf_path = tex_path.with_suffix(".pdf")
    log_path = tex_path.with_suffix(".compile.log")
    if compiler.name.lower().startswith("tectonic"):
        command = [str(compiler), "--outdir", str(tex_path.parent), tex_path.name]
    elif compiler.name.lower().startswith("latexmk"):
        command = [str(compiler), "-pdf", "-interaction=nonstopmode", tex_path.name]
    else:
        command = [str(compiler), "-interaction=nonstopmode", "-halt-on-error", tex_path.name]

    result = subprocess.run(
        command,
        cwd=tex_path.parent,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    _write_text(log_path, (result.stdout or "") + "\n\n" + (result.stderr or ""))
    if result.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(f"{compiler.name} failed for {tex_path.name}; see {log_path}")
    return pdf_path, log_path


def _build_dataset_tables(
    dataset_ids: list[str],
    *,
    engines: tuple[str, ...],
    sql_source_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query_rows_by_dataset = load_latest_sql_queries_by_dataset(
        dataset_ids=dataset_ids,
        engines=engines,
        include_all_statements=True,
        sql_source_version=sql_source_version,
    )

    final_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    family_totals = Counter()
    subitem_totals = Counter()
    inference_source_totals = Counter()
    missing_facet_queries = 0
    missing_role_queries = 0
    datasets_without_sql: list[str] = []
    datasets_with_non_contract_sql: list[str] = []
    datasets_per_subitem_nonzero = Counter()
    total_sql = 0
    canonical_contract_sql = 0
    non_contract_sql = 0

    for dataset_id in sorted(dataset_ids, key=_dataset_sort_key):
        counts = {subitem_id: 0 for subitem_id in SUBITEM_ORDER}
        annotated_rows: list[dict[str, Any]] = []
        dataset_canonical_count = 0
        dataset_total_sql = 0

        for row in query_rows_by_dataset.get(dataset_id, []):
            dataset_total_sql += 1
            annotated = annotate_query_row_with_contract(row)
            annotated_rows.append(annotated)
            if not str(annotated.get("intended_facet_id") or "").strip():
                missing_facet_queries += 1
            if not str(annotated.get("variant_semantic_role") or "").strip():
                missing_role_queries += 1

            family_id = str(annotated.get("family_id") or "")
            subitem_id = str(annotated.get("canonical_subitem_id") or "")
            inference_source = str(annotated.get("subitem_inference_source") or "")
            if inference_source:
                inference_source_totals[inference_source] += 1
            if family_id in CANONICAL_ANALYTICS_SUBITEMS and subitem_id:
                counts[subitem_id] += 1
                family_totals[family_id] += 1
                subitem_totals[subitem_id] += 1
                dataset_canonical_count += 1
            query_rows.append(annotated)

        if dataset_total_sql == 0:
            datasets_without_sql.append(dataset_id)
        if dataset_total_sql > dataset_canonical_count:
            datasets_with_non_contract_sql.append(dataset_id)
        for subitem_id, value in counts.items():
            if value > 0:
                datasets_per_subitem_nonzero[subitem_id] += 1

        total_sql += dataset_total_sql
        canonical_contract_sql += dataset_canonical_count
        non_contract_sql += max(0, dataset_total_sql - dataset_canonical_count)

        final_row = {
            "dataset_id": dataset_id,
            "total_sql": dataset_total_sql,
            **counts,
        }
        detail_row = {
            **final_row,
            "canonical_contract_sql": dataset_canonical_count,
            "non_contract_sql": max(0, dataset_total_sql - dataset_canonical_count),
        }
        final_rows.append(final_row)
        detail_rows.append(detail_row)

    zero_dataset_count_by_subitem = {
        subitem_id: len(dataset_ids) - int(datasets_per_subitem_nonzero.get(subitem_id, 0))
        for subitem_id in SUBITEM_ORDER
    }

    sql_source_stats = _sql_source_stats_from_rows(query_rows, sql_source_version)

    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(dataset_ids),
        "datasets_with_sql": len(dataset_ids) - len(datasets_without_sql),
        "datasets_without_sql": datasets_without_sql,
        "datasets_with_non_contract_sql": sorted(set(datasets_with_non_contract_sql), key=_dataset_sort_key),
        "total_sql": total_sql,
        "canonical_contract_sql": canonical_contract_sql,
        "non_contract_sql": non_contract_sql,
        "family_totals": dict(family_totals),
        "subitem_totals": dict(subitem_totals),
        "inference_source_totals": dict(inference_source_totals),
        "missing_intended_facet_queries": missing_facet_queries,
        "missing_variant_semantic_role_queries": missing_role_queries,
        "dataset_nonzero_count_by_subitem": dict(datasets_per_subitem_nonzero),
        "dataset_zero_count_by_subitem": zero_dataset_count_by_subitem,
        "contract_version": ANALYTICS_CONTRACT_VERSION,
        "engine_filter": list(engines),
        **sql_source_stats,
    }
    return final_rows, detail_rows, query_rows, stats


def _build_summary_note(stats: dict[str, Any]) -> str:
    family_totals = stats["family_totals"]
    inference_totals = stats["inference_source_totals"]
    no_sql_text = (
        ", ".join(stats["datasets_without_sql"])
        if stats["datasets_without_sql"]
        else "none"
    )
    out_of_contract_text = (
        ", ".join(stats["datasets_with_non_contract_sql"])
        if stats["datasets_with_non_contract_sql"]
        else "none"
    )
    subitem_zero_lines = []
    for family_id, subitems in CANONICAL_ANALYTICS_SUBITEMS.items():
        for subitem_id in subitems:
            zero_count = int(stats["dataset_zero_count_by_subitem"].get(subitem_id, 0))
            nonzero_count = int(stats["dataset_nonzero_count_by_subitem"].get(subitem_id, 0))
            subitem_zero_lines.append(
                f"- `{subitem_id}`: nonzero in {nonzero_count}/{stats['dataset_count']} datasets; zero in {zero_count}/{stats['dataset_count']} datasets."
            )

    return "\n".join(
        [
            "# Dataset x canonical subitem SQL counts",
            "",
            f"- SQL source: `{stats['sql_source_label']}` (`{stats['sql_source_version']}`) from `{stats['sql_source_root']}`. Loader description: {stats['sql_source_description']}. This mirrors the source-selection path used by `src.eval.analysis.runner.run_sql_analysis(...)`.",
            "- Subitem mapping: `annotate_query_row_with_contract(...)` from `src.eval.analytics_contract`. The mapping order is `intended_facet_id -> variant_semantic_role -> heuristic fallback`, so missing metadata reuses the same contract fallback as the main scorer.",
            f"- Datasets covered: {stats['dataset_count']} total; {stats['datasets_with_sql']} currently have at least one SQL statement in the loader. Datasets with zero SQL: {no_sql_text}.",
            f"- Total SQL counted: {stats['total_sql']}. Canonical-contract SQL: {stats['canonical_contract_sql']}. Out-of-contract SQL: {stats['non_contract_sql']}. Datasets where `Total SQL` exceeds the 10-subitem sum: {out_of_contract_text}.",
            f"- Family totals: subgroup={family_totals.get('subgroup_structure', 0)}, conditional={family_totals.get('conditional_dependency_structure', 0)}, tail={family_totals.get('tail_rarity_structure', 0)}, missingness={family_totals.get('missingness_structure', 0)}.",
            f"- Inference-source totals: facet={inference_totals.get('facet', 0)}, role={inference_totals.get('role', 0)}, heuristic={inference_totals.get('heuristic', 0)}, non_analytics_family={inference_totals.get('non_analytics_family', 0)}.",
            f"- Missing metadata handled by current fallback: missing `intended_facet_id` on {stats['missing_intended_facet_queries']} SQL statements; missing `variant_semantic_role` on {stats['missing_variant_semantic_role_queries']} SQL statements.",
            "- `cardinality/range` is intentionally excluded from this 10-subitem table because it is not part of the frozen canonical analytics subitem contract in the README / `analytics_contract.py`.",
            "",
            "## Dataset coverage by subitem",
            "",
            *subitem_zero_lines,
            "",
        ]
    )


def _write_evaluation_readme() -> None:
    readme = "\n".join(
        [
            "# dataset_subitem_sql_counts",
            "",
            "This directory stores dataset-by-canonical-subitem SQL count tables aligned with the current analysis SQL loader and analytics contract.",
            "",
            "## Rebuild",
            "",
            "From the repo root:",
            "",
            "```powershell",
            "python -m src.eval.dataset_subitem_sql_counts.runner",
            "```",
            "",
            "The main paper-facing outputs are written to `final/`.",
            "",
        ]
    )
    _write_text(OUTPUT_ROOT / "README.md", readme)


def run_dataset_subitem_sql_counts(
    *,
    run_tag: str,
    datasets: list[str] | None = None,
    engines: tuple[str, ...] = ("cli",),
    sql_source_version: str = DEFAULT_SQL_SOURCE_VERSION,
    latex_engine: str | None = None,
) -> dict[str, Any]:
    dataset_ids = sorted(datasets or list_dataset_ids(), key=_dataset_sort_key)
    run_dir = make_task_run_dir(TASK_NAME, run_tag)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    _write_evaluation_readme()
    write_versioned_final_readme(
        task_name=TASK_NAME,
        title="dataset_subitem_sql_counts final outputs",
        summary="Versioned final bundles for dataset-by-subitem SQL count inventories aligned with the analysis SQL loader.",
        notes=[
            "The root `final/` directory still keeps latest-file aliases for compatibility.",
            "Source-specific bundles live under `final/<sql_source_version>/`.",
        ],
    )
    version_final_dir = task_version_final_dir(TASK_NAME, sql_source_version)
    version_final_dir.mkdir(parents=True, exist_ok=True)

    final_rows, detail_rows, query_rows, stats = _build_dataset_tables(
        dataset_ids,
        engines=engines,
        sql_source_version=sql_source_version,
    )
    summary_note = _build_summary_note(stats)

    write_csv(raw_dir / "dataset_subitem_sql_counts_detailed.csv", detail_rows, fieldnames=DETAIL_FIELDNAMES)
    write_csv(raw_dir / "dataset_subitem_sql_query_annotations.csv", query_rows, fieldnames=QUERY_FIELDNAMES)
    write_json(raw_dir / "dataset_subitem_sql_counts_stats.json", stats)

    final_csv_path = version_final_dir / "dataset_subitem_sql_counts.csv"
    final_note_path = version_final_dir / "dataset_subitem_sql_counts_summary.md"
    main_tex_path = version_final_dir / "dataset_subitem_sql_counts_table.tex"
    heatmap_tex_path = version_final_dir / "dataset_subitem_sql_counts_heatmap.tex"

    write_csv(final_csv_path, final_rows, fieldnames=TABLE_FIELDNAMES)
    _write_text(final_note_path, summary_note)
    _write_text(main_tex_path, _build_tex_document(final_rows, stats=stats, heatmap=False))
    _write_text(heatmap_tex_path, _build_tex_document(final_rows, stats=stats, heatmap=True))

    main_pdf_path, main_log_path = _compile_tex(main_tex_path, latex_engine=latex_engine)
    heatmap_pdf_path, heatmap_log_path = _compile_tex(heatmap_tex_path, latex_engine=latex_engine)
    main_png_path = render_pdf_to_png(main_pdf_path, version_final_dir / "dataset_subitem_sql_counts_table.png", densest_page=True)
    heatmap_png_path = render_pdf_to_png(
        heatmap_pdf_path,
        version_final_dir / "dataset_subitem_sql_counts_heatmap.png",
        densest_page=True,
    )

    manifest = {
        "task": TASK_NAME,
        "run_tag": run_tag,
        "final_dir": str(version_final_dir.resolve()),
        "generated_at_utc": stats["generated_at_utc"],
        "dataset_count": stats["dataset_count"],
        "datasets_with_sql": stats["datasets_with_sql"],
        "total_sql": stats["total_sql"],
        "canonical_contract_sql": stats["canonical_contract_sql"],
        "non_contract_sql": stats["non_contract_sql"],
        "engine_filter": list(engines),
        "provenance_contract_version": stats["provenance_contract_version"],
        "sql_source_family": stats["sql_source_family"],
        "sql_source_line_version": stats["sql_source_line_version"],
        "sql_source_version": stats["sql_source_version"],
        "sql_source_label": stats["sql_source_label"],
        "sql_source_description": stats["sql_source_description"],
        "sql_source_root": stats["sql_source_root"],
        "sql_source_registry_root": stats["sql_source_registry_root"],
        "analytics_contract_version": ANALYTICS_CONTRACT_VERSION,
        "sql_loader_alignment": "load_latest_sql_queries_by_dataset / src.eval.analysis.runner defaults",
        "annotation_alignment": "annotate_query_row_with_contract",
        "final_outputs": {
            "csv": str(final_csv_path.resolve()),
            "summary_note": str(final_note_path.resolve()),
            "table_tex": str(main_tex_path.resolve()),
            "table_pdf": str(main_pdf_path.resolve()),
            "table_png": str(main_png_path.resolve()),
            "heatmap_tex": str(heatmap_tex_path.resolve()),
            "heatmap_pdf": str(heatmap_pdf_path.resolve()),
            "heatmap_png": str(heatmap_png_path.resolve()),
            "table_compile_log": str(main_log_path.resolve()),
            "heatmap_compile_log": str(heatmap_log_path.resolve()),
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(version_final_dir / "dataset_subitem_sql_counts_manifest.json", manifest)
    copy_files(
        FINAL_DIR,
        [
            final_csv_path,
            final_note_path,
            main_tex_path,
            main_pdf_path,
            main_png_path,
            main_log_path,
            heatmap_tex_path,
            heatmap_pdf_path,
            heatmap_png_path,
            heatmap_log_path,
            version_final_dir / "dataset_subitem_sql_counts_manifest.json",
        ],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dataset x canonical-subitem SQL count tables aligned with current evaluation SQL."
    )
    parser.add_argument("--run-tag", default=now_run_tag())
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--engines", nargs="*", default=["cli"])
    parser.add_argument(
        "--sql-source-version",
        choices=list(SQL_SOURCE_VERSION_CHOICES),
        default=DEFAULT_SQL_SOURCE_VERSION,
    )
    parser.add_argument("--latex-engine", default=None)
    args = parser.parse_args()

    manifest = run_dataset_subitem_sql_counts(
        run_tag=str(args.run_tag),
        datasets=list(args.datasets) if args.datasets else None,
        engines=tuple(str(item) for item in args.engines),
        sql_source_version=str(args.sql_source_version),
        latex_engine=str(args.latex_engine) if args.latex_engine else None,
    )
    print(f"[{TASK_NAME}] wrote final outputs to {FINAL_DIR.resolve()}")
    print(f"[{TASK_NAME}] total_sql={manifest['total_sql']} | datasets_with_sql={manifest['datasets_with_sql']}")


if __name__ == "__main__":
    main()
