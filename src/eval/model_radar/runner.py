"""Build a clean all-model radar chart using the current README-aligned scoring sources."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.eval.common import DEFAULT_SQL_SOURCE_VERSION, resolve_requested_sql_source_version, sql_source_label
from src.eval.query_fivepart_breakdown.common_final import versioned_name

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "Evaluation" / "model_radar"
DATA_DIR = OUTPUT_ROOT / "data"
FIGURES_DIR = OUTPUT_ROOT / "figures"
FINAL_DIR = OUTPUT_ROOT / "final"
SQL_SOURCE_VERSION = resolve_requested_sql_source_version("analysis", DEFAULT_SQL_SOURCE_VERSION)

DISTANCE_LATEST = PROJECT_ROOT / "Evaluation" / "distance" / "LATEST_RUN.json"
LOCAL_TECTONIC = OUTPUT_ROOT / "_build_tools" / "tectonic" / "tectonic.exe"

MODEL_ORDER = [
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

MODEL_LABELS = {
    "arf": "ARF",
    "bayesnet": "BayesNet",
    "cdtd": "CDTD",
    "codi": "CoDi",
    "ctgan": "CTGAN",
    "forestdiffusion": "ForestDiffusion",
    "goggle": "GOGGLE",
    "realtabformer": "RealTabFormer",
    "rtf": "RealTabFormer",
    "tabbyflow": "TabbyFlow",
    "tabddpm": "TabDDPM",
    "tabdiff": "TabDiff",
    "tabpfgen": "TabPFGen",
    "tabsyn": "TabSyn",
    "tvae": "TVAE",
}

MODEL_COLORS = {
    "realtabformer": "#332288",
    "tvae": "#4477AA",
    "forestdiffusion": "#228833",
    "tabddpm": "#EE7733",
    "tabsyn": "#66CCEE",
    "tabdiff": "#AA3377",
    "ctgan": "#EE6677",
    "arf": "#777777",
    "bayesnet": "#CCBB44",
    "tabpfgen": "#009988",
    "tabbyflow": "#882255",
}

MODEL_ALIASES = {
    "rtf": "realtabformer",
}

AXIS_ORDER = [
    ("distance_score", "Distance"),
    ("subgroup_score", "Subgroup"),
    ("conditional_score", "Conditional"),
    ("tail_score", "Tail"),
    ("missingness_score", "Missing"),
    ("cardinality_score", "Cardinality"),
]

ANGLE_DEGREES = [90.0, 30.0, -30.0, -90.0, -150.0, 150.0]
AXIS_ANCHORS = ["south", "west", "west", "north", "east", "east"]


def _versioned_breakdown_csv(task_name: str, base_name: str) -> Path:
    final_root = PROJECT_ROOT / "Evaluation" / "query_fivepart_breakdown" / task_name / "final"
    candidates = [
        final_root / SQL_SOURCE_VERSION / versioned_name(base_name, SQL_SOURCE_VERSION),
        final_root / versioned_name(base_name, SQL_SOURCE_VERSION),
        final_root / base_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


SUBGROUP_CSV = _versioned_breakdown_csv("subgroup_breakdown", "model_summary.csv")
CONDITIONAL_CSV = _versioned_breakdown_csv("conditional_breakdown", "model_summary.csv")
TAIL_CSV = _versioned_breakdown_csv("tail_breakdown", "model_summary.csv")
MISSINGNESS_CSV = _versioned_breakdown_csv("missingness_breakdown", "model_summary.csv")
CARDINALITY_CSV = _versioned_breakdown_csv("cardinality", "summary_by_model.csv")


def _ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, DATA_DIR, FIGURES_DIR, FINAL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _normalize_model_id(value: Any) -> str:
    key = str(value or "").strip().lower()
    return MODEL_ALIASES.get(key, key)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _distance_run_summary_path() -> Path:
    payload = json.loads(DISTANCE_LATEST.read_text(encoding="utf-8"))
    run_dir = Path(str(payload["run_dir"]))
    return run_dir / "summaries" / "distance_summary__all_datasets.csv"


def _distance_score_from_row(row: dict[str, str]) -> float | None:
    jsd = _coerce_float(row.get("jensen_shannon_distance"))
    ksd = _coerce_float(row.get("kolmogorov_smirnov_distance"))
    tvd = _coerce_float(row.get("total_variation_distance"))
    wasserstein = _coerce_float(row.get("wasserstein_distance"))
    candidates = []
    for value in (jsd, ksd, tvd, wasserstein):
        if value is None:
            continue
        candidates.append(max(0.0, min(1.0, 1.0 - float(value))))
    if not candidates:
        return None
    return float(mean(candidates))


def _load_distance_scores() -> dict[str, dict[str, Any]]:
    path = _distance_run_summary_path()
    rows = _read_csv_rows(path)
    grouped: dict[str, list[float]] = {model_id: [] for model_id in MODEL_ORDER}
    for row in rows:
        model_id = _normalize_model_id(row.get("model_id"))
        if model_id not in grouped:
            continue
        score = _distance_score_from_row(row)
        if score is not None:
            grouped[model_id].append(score)

    output: dict[str, dict[str, Any]] = {}
    for model_id, values in grouped.items():
        if not values:
            raise ValueError(f"No distance rows found for model `{model_id}` in {path}")
        output[model_id] = {
            "distance_score": round(float(mean(values)), 6),
            "distance_dataset_count": len(values),
            "distance_source_csv": str(path.resolve()),
            "distance_formula": "mean(1-JSD, 1-KSD, 1-TVD, 1-Wasserstein)",
        }
    return output


def _load_summary_field(
    path: Path,
    *,
    model_key: str,
    score_key: str,
    output_key: str,
    count_key: str,
) -> dict[str, dict[str, Any]]:
    rows = _read_csv_rows(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        model_id = _normalize_model_id(row.get(model_key))
        if model_id not in MODEL_ORDER:
            continue
        score = _coerce_float(row.get(score_key))
        if score is None:
            continue
        count = row.get(count_key) or row.get("n_datasets") or row.get("dataset_count") or ""
        output[model_id] = {
            output_key: round(float(score), 6),
            f"{output_key}_dataset_count": str(count),
            f"{output_key}_source_csv": str(path.resolve()),
        }
    return output


def _build_summary_rows() -> list[dict[str, Any]]:
    distance = _load_distance_scores()
    subgroup = _load_summary_field(
        SUBGROUP_CSV,
        model_key="model_id",
        score_key="subgroup_structure_score__mean",
        output_key="subgroup_score",
        count_key="dataset_count",
    )
    conditional = _load_summary_field(
        CONDITIONAL_CSV,
        model_key="model_id",
        score_key="conditional_dependency_structure_score__mean",
        output_key="conditional_score",
        count_key="dataset_count",
    )
    tail = _load_summary_field(
        TAIL_CSV,
        model_key="model_id",
        score_key="tail_breakdown_score__mean",
        output_key="tail_score",
        count_key="dataset_count",
    )
    missingness = _load_summary_field(
        MISSINGNESS_CSV,
        model_key="model_id",
        score_key="missingness_structure_score__mean",
        output_key="missingness_score",
        count_key="dataset_count",
    )
    cardinality = _load_summary_field(
        CARDINALITY_CSV,
        model_key="model",
        score_key="overall_score_mean",
        output_key="cardinality_score",
        count_key="n_datasets",
    )

    summary_rows: list[dict[str, Any]] = []
    for display_order, model_id in enumerate(MODEL_ORDER, start=1):
        payload = {
            "display_order": display_order,
            "model_id": model_id,
            "model_label": MODEL_LABELS[model_id],
            "model_color": MODEL_COLORS[model_id],
        }
        for block in (distance, subgroup, conditional, tail, missingness, cardinality):
            if model_id not in block:
                raise ValueError(f"Missing summary block for model `{model_id}`.")
            payload.update(block[model_id])
        payload["radar_score_mean"] = round(
            float(
                mean(
                    float(payload[field])
                    for field, _ in AXIS_ORDER
                )
            ),
            6,
        )
        summary_rows.append(payload)
    return summary_rows


def _latex_escape(text: str) -> str:
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


def _tikz_color_name(model_id: str) -> str:
    return "model" + "".join(ch for ch in model_id.title() if ch.isalnum())


def _polygon_coords(values: list[float], radius: float) -> list[str]:
    return [f"({angle:.1f}:{value * radius:.4f})" for angle, value in zip(ANGLE_DEGREES, values)]


def _write_tikz(tex_path: Path, summary_rows: list[dict[str, Any]]) -> None:
    radius = 5.25
    ring_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    legend_x = radius + 1.55
    legend_top_y = 2.55
    legend_row_gap = 0.72
    legend_col_gap = 3.55

    lines = [
        r"\documentclass[tikz,border=5pt]{standalone}",
        r"\usepackage{xcolor}",
        r"\usepackage{tikz}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1cm,y=1cm,font=\sffamily]",
        r"\definecolor{GridLine}{HTML}{DCE2EA}",
        r"\definecolor{AxisLine}{HTML}{B6C0CB}",
        r"\definecolor{Ink}{HTML}{18222F}",
        rf"\def\RadarRadius{{{radius:.3f}}}",
    ]

    for model_id in MODEL_ORDER:
        lines.append(
            rf"\definecolor{{{_tikz_color_name(model_id)}}}{{HTML}}{{{MODEL_COLORS[model_id].lstrip('#')}}}"
        )

    lines.append("")
    for level in ring_levels:
        coords = " -- ".join(_polygon_coords([level] * len(AXIS_ORDER), radius))
        width = "0.8pt" if math.isclose(level, 1.0) else "0.45pt"
        lines.append(rf"\draw[draw=GridLine, line width={width}] {coords} -- cycle;")

    lines.append("")
    for (_, label), angle, anchor in zip(AXIS_ORDER, ANGLE_DEGREES, AXIS_ANCHORS):
        lines.append(rf"\draw[draw=AxisLine, line width=0.55pt] (0,0) -- ({angle:.1f}:{radius:.3f});")
        lines.append(
            rf"\node[anchor={anchor}, font=\bfseries\footnotesize, text=Ink] at ({angle:.1f}:{radius + 0.68:.3f}) {{{_latex_escape(label)}}};"
        )

    lines.append("")
    for row in summary_rows:
        model_id = row["model_id"]
        color_name = _tikz_color_name(model_id)
        values = [float(row[field]) for field, _ in AXIS_ORDER]
        coords = _polygon_coords(values, radius)
        polygon = " -- ".join(coords)
        lines.append(
            rf"\draw[draw={color_name}, line width=1.05pt, line join=round, opacity=0.96] {polygon} -- cycle;"
        )
        for coord in coords:
            lines.append(
                rf"\filldraw[draw={color_name}, fill=white, line width=0.8pt, opacity=0.98] {coord} circle (1.65pt);"
            )
        lines.append("")

    for index, row in enumerate(summary_rows):
        col_index = 0 if index < 6 else 1
        row_index = index if index < 6 else index - 6
        base_x = legend_x + (col_index * legend_col_gap)
        y = legend_top_y - (row_index * legend_row_gap)
        color_name = _tikz_color_name(row["model_id"])
        label = _latex_escape(str(row["model_label"]))
        lines.append(rf"\draw[draw={color_name}, line width=1.05pt] ({base_x:.3f}, {y:.3f}) -- ({base_x + 0.78:.3f}, {y:.3f});")
        lines.append(rf"\filldraw[draw={color_name}, fill=white, line width=0.8pt] ({base_x + 0.39:.3f}, {y:.3f}) circle (1.65pt);")
        lines.append(rf"\node[anchor=west, font=\footnotesize, text=Ink] at ({base_x + 1.02:.3f}, {y:.3f}) {{{label}}};")

    lines.extend(
        [
            r"\end{tikzpicture}",
            r"\end{document}",
        ]
    )

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_compiler() -> Path | None:
    candidates = [
        shutil.which("tectonic"),
        str(LOCAL_TECTONIC) if LOCAL_TECTONIC.exists() else None,
        shutil.which("pdflatex"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _compile_tikz(tex_path: Path) -> tuple[bool, Path | None, str]:
    compiler = _find_compiler()
    if compiler is None:
        return False, None, "No TeX compiler found."

    pdf_path = tex_path.with_suffix(".pdf")
    log_path = tex_path.with_suffix(".compile.log")
    if compiler.name.lower().startswith("tectonic"):
        command = [str(compiler), "--outdir", str(tex_path.parent), tex_path.name]
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
    if result.returncode == 0 and pdf_path.exists():
        return True, pdf_path, f"Compiled via {compiler.name}"
    return False, None, f"{compiler.name} returned {result.returncode}; see {log_path}"


def _render_pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 320) -> None:
    try:
        import fitz  # type: ignore
    except Exception:
        import pymupdf as fitz  # type: ignore

    document = fitz.open(pdf_path)
    try:
        page = document.load_page(0)
        scale = float(dpi) / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(png_path))
    finally:
        document.close()


def _build_report(summary_rows: list[dict[str, Any]], distance_csv: Path) -> str:
    lines = [
        "# Model Radar Report",
        "",
        f"- SQL source: `{sql_source_label(SQL_SOURCE_VERSION)}` (`{SQL_SOURCE_VERSION}`)",
        "- Figure style: clean radar + legend only.",
        "- Included models: all README-frozen paper-facing generators.",
        "- Distance source: README-aligned recomputation from raw distance run.",
        f"- Distance raw CSV: `{distance_csv.resolve()}`",
        f"- Subgroup source: `{SUBGROUP_CSV.resolve()}`",
        f"- Conditional source: `{CONDITIONAL_CSV.resolve()}`",
        f"- Tail source: `{TAIL_CSV.resolve()}`",
        f"- Missingness source: `{MISSINGNESS_CSV.resolve()}`",
        f"- Cardinality source: `{CARDINALITY_CSV.resolve()}`",
        "",
        "| Model | Distance | Subgroup | Conditional | Tail | Missing | Cardinality | Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        values = [f"{float(row[field]):.3f}" for field, _ in AXIS_ORDER]
        lines.append(
            f"| {row['model_label']} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | {values[5]} | {float(row['radar_score_mean']):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_output_readme() -> str:
    return "\n".join(
        [
            "# model_radar outputs",
            "",
            "This directory contains the cleaned all-model radar figure using the current README-aligned scoring sources.",
            "",
            "Main files:",
            "",
            "- `final/model_radar_main.tex`",
            "- `final/model_radar_main.pdf`",
            "- `final/model_radar_main.png`",
            "- `final/model_radar_summary.csv`",
            "",
        ]
    )


def _copy_final(paths: list[Path]) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, FINAL_DIR / path.name)


def run_model_radar() -> dict[str, Any]:
    _ensure_dirs()
    summary_rows = _build_summary_rows()
    distance_csv = _distance_run_summary_path()

    summary_csv = DATA_DIR / "model_radar_summary.csv"
    tex_path = FIGURES_DIR / "model_radar_main.tex"
    pdf_path = FIGURES_DIR / "model_radar_main.pdf"
    png_path = FIGURES_DIR / "model_radar_main.png"
    report_path = OUTPUT_ROOT / "analysis_report.md"
    readme_path = OUTPUT_ROOT / "README.md"
    manifest_path = OUTPUT_ROOT / "manifest.json"

    _write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "display_order",
            "model_id",
            "model_label",
            "model_color",
            "distance_score",
            "subgroup_score",
            "conditional_score",
            "tail_score",
            "missingness_score",
            "cardinality_score",
            "radar_score_mean",
            "distance_dataset_count",
            "subgroup_score_dataset_count",
            "conditional_score_dataset_count",
            "tail_score_dataset_count",
            "missingness_score_dataset_count",
            "cardinality_score_dataset_count",
            "distance_source_csv",
            "subgroup_score_source_csv",
            "conditional_score_source_csv",
            "tail_score_source_csv",
            "missingness_score_source_csv",
            "cardinality_score_source_csv",
            "distance_formula",
        ],
    )

    _write_tikz(tex_path, summary_rows)
    compile_ok, compiled_pdf, compile_note = _compile_tikz(tex_path)
    if not compile_ok or compiled_pdf is None:
        raise RuntimeError(f"Radar TeX compilation failed: {compile_note}")
    _render_pdf_to_png(compiled_pdf, png_path)

    report_path.write_text(_build_report(summary_rows, distance_csv), encoding="utf-8")
    readme_path.write_text(_build_output_readme(), encoding="utf-8")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sql_source_version": SQL_SOURCE_VERSION,
        "sql_source_label": sql_source_label(SQL_SOURCE_VERSION),
        "model_count": len(summary_rows),
        "model_ids": [row["model_id"] for row in summary_rows],
        "axis_order": [label for _, label in AXIS_ORDER],
        "distance_formula": "mean(1-JSD, 1-KSD, 1-TVD, 1-Wasserstein)",
        "distance_csv": str(distance_csv.resolve()),
        "subgroup_csv": str(SUBGROUP_CSV.resolve()),
        "conditional_csv": str(CONDITIONAL_CSV.resolve()),
        "tail_csv": str(TAIL_CSV.resolve()),
        "missingness_csv": str(MISSINGNESS_CSV.resolve()),
        "cardinality_csv": str(CARDINALITY_CSV.resolve()),
        "tex_path": str(tex_path.resolve()),
        "pdf_path": str(compiled_pdf.resolve()),
        "png_path": str(png_path.resolve()),
        "compile_note": compile_note,
    }
    _write_json(manifest_path, manifest)

    _copy_final([summary_csv, tex_path, compiled_pdf, png_path, report_path, manifest_path])
    return manifest


def main() -> None:
    manifest = run_model_radar()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
