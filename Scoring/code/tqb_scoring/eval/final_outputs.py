from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from tqb_scoring.eval.common import (
    SQL_SOURCE_VERSION_CHOICES,
    normalize_sql_source_version,
    sql_source_description,
    sql_source_label,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = PROJECT_ROOT.parent / "results"

STANDARD_MODEL_ORDER = [
    "arf",
    "bayesnet",
    "ctgan",
    "forestdiffusion",
    "realtabformer",
    "tabbyflow",
    "tabddpm",
    "tabdiff",
    "tabpfgen",
    "tabsyn",
    "tvae",
]

MODEL_ALIASES = {
    "rtf": "realtabformer",
}

LOCAL_TEX_COMPILERS = [
    PROJECT_ROOT / "tools" / "tectonic" / "tectonic-0.16.9" / "tectonic.exe",
    PROJECT_ROOT / "tools" / "tectonic-0.16.9" / "tectonic.exe",
    PROJECT_ROOT.parent / "results" / "model_radar" / "_build_tools" / "tectonic" / "tectonic.exe",
]


def task_final_root(task_name: str) -> Path:
    return EVALUATION_ROOT / task_name / "final"


def task_version_final_dir(task_name: str, sql_source_version: str) -> Path:
    version = normalize_sql_source_version(sql_source_version)
    return task_final_root(task_name) / version


def copy_files(target_dir: Path, files: Iterable[Path], aliases: Mapping[str, Path] | None = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        if not src.exists():
            continue
        dst = target_dir / src.name
        try:
            if src.resolve() == dst.resolve():
                continue
        except FileNotFoundError:
            pass
        shutil.copy2(src, dst)
    for alias_name, src in (aliases or {}).items():
        if not src.exists():
            continue
        shutil.copy2(src, target_dir / alias_name)


def write_versioned_final_readme(
    *,
    task_name: str,
    title: str,
    summary: str,
    notes: list[str] | None = None,
) -> Path:
    final_root = task_final_root(task_name)
    final_root.mkdir(parents=True, exist_ok=True)
    version_lines: list[str] = []
    for version in SQL_SOURCE_VERSION_CHOICES:
        version_lines.append(
            f"- `{version}/`: `{sql_source_label(version)}`. {sql_source_description(version)}"
        )
    lines = [
        f"# {title}",
        "",
        summary.strip(),
        "",
        "Versioned SQL bundles:",
        "",
        *version_lines,
        "",
        f"Each bundle stores a task-specific report (`.md`, `.tex`, `.pdf`, `.png`) plus the key CSV / JSON artifacts for that SQL source.",
    ]
    if notes:
        lines.extend(["", *notes])
    lines.append("")
    path = final_root / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def find_latex_compiler(explicit: str | None = None) -> Path | None:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    for path in LOCAL_TEX_COMPILERS:
        if path.exists():
            candidates.append(str(path))
    for name in ("tectonic", "pdflatex", "xelatex", "latexmk"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def compile_tex(
    tex_path: Path,
    *,
    out_dir: Path | None = None,
    latex_engine: str | None = None,
) -> tuple[Path, Path]:
    compiler = find_latex_compiler(latex_engine)
    if compiler is None:
        raise RuntimeError("No LaTeX engine found. Install tectonic/pdflatex or pass --latex-engine.")

    target_dir = out_dir or tex_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / f"{tex_path.stem}.pdf"
    log_path = target_dir / f"{tex_path.stem}.compile.log"

    compiler_name = compiler.name.lower()
    if compiler_name.startswith("tectonic"):
        command = [str(compiler), "--outdir", str(target_dir), tex_path.name]
    elif compiler_name.startswith("latexmk"):
        command = [str(compiler), "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
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
    log_path.write_text((result.stdout or "") + "\n\n" + (result.stderr or ""), encoding="utf-8")
    if result.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(f"{compiler.name} failed for {tex_path.name}; see {log_path}")
    return pdf_path, log_path


def render_pdf_to_png(
    pdf_path: Path,
    png_path: Path,
    *,
    dpi: int = 260,
    densest_page: bool = False,
) -> Path:
    try:
        import fitz  # type: ignore
    except Exception:
        import pymupdf as fitz  # type: ignore

    document = fitz.open(pdf_path)
    try:
        if len(document) <= 0:
            raise RuntimeError(f"Cannot render empty PDF: {pdf_path}")
        page_index = 0
        if densest_page and len(document) > 1:
            preview_scale = 0.75
            best_ratio = float("inf")
            for idx in range(len(document)):
                page = document.load_page(idx)
                pix = page.get_pixmap(matrix=fitz.Matrix(preview_scale, preview_scale), alpha=False)
                samples = pix.samples
                total_pixels = max(1, len(samples) // 3)
                white_pixels = 0
                for sample_idx in range(0, len(samples), 3):
                    if samples[sample_idx] > 245 and samples[sample_idx + 1] > 245 and samples[sample_idx + 2] > 245:
                        white_pixels += 1
                ratio = white_pixels / total_pixels
                if ratio < best_ratio:
                    best_ratio = ratio
                    page_index = idx
        page = document.load_page(page_index)
        scale = max(1.0, float(dpi) / 72.0)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(png_path))
    finally:
        document.close()
    return png_path


def latex_escape(value: Any) -> str:
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
    text = str(value)
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def latex_cell(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        rendered = f"{value:.6f}".rstrip("0").rstrip(".")
        return rendered or "0"
    text = str(value).strip()
    if not text:
        return "--"
    return latex_escape(text)


def build_longtable_report_tex(
    *,
    title: str,
    subtitle: str,
    intro_lines: list[str] | None = None,
    tables: list[dict[str, Any]],
) -> str:
    body: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.62in]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{4.4pt}",
        r"\begin{document}",
        r"\thispagestyle{empty}",
        rf"\noindent\textbf{{{latex_escape(title)}}}\\[0.25em]",
        rf"\noindent\footnotesize {latex_escape(subtitle)}\\[0.75em]",
    ]
    for line in (intro_lines or []):
        body.append(rf"\noindent\footnotesize {latex_escape(line)}\\")
    if intro_lines:
        body.append(r"\vspace{0.5em}")

    for table in tables:
        columns: list[tuple[str, str]] = list(table.get("columns") or [])
        rows: list[dict[str, Any]] = list(table.get("rows") or [])
        if not columns:
            continue
        heading = str(table.get("heading") or "")
        note = str(table.get("note") or "")
        widths = list(table.get("widths") or [])
        if heading:
            body.extend([rf"\section*{{{latex_escape(heading)}}}", r"\small"])
        else:
            body.append(r"\small")

        if not widths:
            if len(columns) <= 4:
                widths = ["4.2cm"] + ["2.8cm"] * (len(columns) - 1)
            elif len(columns) <= 6:
                widths = ["3.2cm"] + ["2.1cm"] * (len(columns) - 1)
            else:
                widths = ["2.7cm"] + ["1.55cm"] * (len(columns) - 1)
        if len(widths) < len(columns):
            widths.extend([widths[-1]] * (len(columns) - len(widths)))
        column_spec = "@{}" + "".join(rf">{{\raggedright\arraybackslash}}p{{{width}}}" for width in widths[: len(columns)]) + "@{}"

        body.extend(
            [
                rf"\begin{{longtable}}{{{column_spec}}}",
                r"\toprule",
                " & ".join(latex_escape(label) for _, label in columns) + r" \\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                " & ".join(latex_escape(label) for _, label in columns) + r" \\",
                r"\midrule",
                r"\endhead",
                r"\bottomrule",
                r"\endfoot",
            ]
        )
        for row in rows:
            body.append(" & ".join(latex_cell(row.get(field)) for field, _ in columns) + r" \\")
        body.extend([r"\end{longtable}", r"\normalsize"])
        if note:
            body.extend([rf"\noindent\footnotesize {latex_escape(note)}", r"\vspace{0.5em}"])
        else:
            body.append(r"\vspace{0.35em}")

    body.extend([r"\end{document}", ""])
    return "\n".join(body)


def build_image_report_tex(
    *,
    title: str,
    subtitle: str,
    intro_lines: list[str] | None = None,
    image_entries: list[dict[str, Any]],
) -> str:
    body: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.55in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\begin{document}",
        r"\thispagestyle{empty}",
        rf"\noindent\textbf{{{latex_escape(title)}}}\\[0.25em]",
        rf"\noindent\footnotesize {latex_escape(subtitle)}\\[0.75em]",
    ]
    for line in (intro_lines or []):
        body.append(rf"\noindent\footnotesize {latex_escape(line)}\\")
    if intro_lines:
        body.append(r"\vspace{0.6em}")

    for entry in image_entries:
        heading = str(entry.get("heading") or "")
        caption = str(entry.get("caption") or "")
        path = Path(str(entry.get("path") or ""))
        width = str(entry.get("width") or "0.94\\textwidth")
        if heading:
            body.append(rf"\section*{{{latex_escape(heading)}}}")
        body.append(r"\begin{center}")
        body.append(rf"\includegraphics[width={width}]{{{path.as_posix()}}}")
        body.append(r"\end{center}")
        if caption:
            body.append(rf"\noindent\footnotesize {latex_escape(caption)}\\[0.4em]")
    body.extend([r"\end{document}", ""])
    return "\n".join(body)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def version_label(version: str) -> str:
    return sql_source_label(version)


def normalize_standard_model_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    return MODEL_ALIASES.get(text, text)
