"""Build reproducible appendix tables and PDF bundles for the paper."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.eval.common import list_dataset_ids, make_task_run_dir, now_run_tag, write_csv, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "Evaluation"
PAPER_ROOT = PROJECT_ROOT / "Paper"

EXCLUDED_MODELS = {
    "cdtd",
    "codi",
    "goggle",
}

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
    "codi": "CODI",
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

MODEL_ALIASES = {
    "rtf": "realtabformer",
}

FAMILY_COLUMNS = [
    ("subgroup_structure", "Subg."),
    ("conditional_dependency_structure", "Cond."),
    ("tail_rarity_structure", "Tail"),
    ("missingness_structure", "Miss."),
]

VALIDATION_COLUMNS = [
    ("cardinality_range_score", "Card."),
    ("missing_introduction_score", "MissIntro"),
]

GLOBAL_SUMMARY_COLUMNS = [
    ("coverage_count", "Coverage"),
    ("overall_score", "Overall"),
    ("query_success_rate", "QSR"),
    ("subgroup_structure", "Subg."),
    ("conditional_dependency_structure", "Cond."),
    ("tail_rarity_structure", "Tail"),
    ("missingness_structure", "Miss."),
    ("cardinality_range_score", "Card."),
    ("missing_introduction_score", "MissIntro"),
    ("training_time_per_1k_seconds", "Train/1k"),
    ("inference_time_per_1k_seconds", "Gen/1k"),
]


@dataclass
class RunSources:
    analysis_run_dir: Path
    validation_run_dir: Path
    paper_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _normalize_model_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    return MODEL_ALIASES.get(text, text)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _dataset_sort_key(dataset_id: str) -> tuple[str, int, str]:
    text = str(dataset_id or "").strip().lower()
    prefix = text[:1]
    digits = text[1:]
    try:
        numeric = int(digits)
    except Exception:
        numeric = 10**9
    return prefix, numeric, text


def _score_dataset_ids() -> list[str]:
    return sorted(list_dataset_ids(), key=_dataset_sort_key)


def _resolve_latest_task_run_dir(task_name: str) -> Path:
    latest_path = OUTPUT_ROOT / task_name / "LATEST_RUN.json"
    payload = _read_json(latest_path)
    run_tag = str(payload.get("run_tag") or "").strip()
    run_dir = Path(str(payload.get("run_dir") or ""))
    if run_dir and not run_dir.is_absolute():
        run_dir = (PROJECT_ROOT / run_dir).resolve()
    if run_dir and run_dir.exists():
        return run_dir.resolve()
    if run_tag:
        fallback = (OUTPUT_ROOT / task_name / "runs" / run_tag).resolve()
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Could not resolve a local run directory for task '{task_name}' from {latest_path}.")


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
        raise FileNotFoundError("Could not locate the active paper directory under Paper/.")
    candidates.sort(key=lambda item: (len(item.parts), str(item)))
    return candidates[0]


def _parse_iso_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:
        return datetime.min


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na", "<null>"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _median(values: list[float | None]) -> float | None:
    cleaned = [float(item) for item in values if item is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def _pick_richer_row(rows: list[dict[str, Any]], *, numeric_fields: list[str]) -> dict[str, Any]:
    def _score(row: dict[str, Any]) -> tuple[datetime, int]:
        ts = _parse_iso_timestamp(row.get("timestamp_utc"))
        richness = sum(1 for field in numeric_fields if _to_float(row.get(field)) is not None)
        return ts, richness

    return max(rows, key=_score)


def _collapse_rows(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    numeric_fields: list[str],
) -> dict[tuple[str, ...], dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        normalized = dict(row)
        if "model_id" in normalized:
            normalized["model_id"] = _normalize_model_id(normalized.get("model_id"))
        key = tuple(str(normalized.get(field) or "").strip().lower() for field in key_fields)
        grouped.setdefault(key, []).append(normalized)
    return {key: _pick_richer_row(items, numeric_fields=numeric_fields) for key, items in grouped.items()}


def _load_analysis_asset_rows(analysis_run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = analysis_run_dir / "summaries" / "analysis_asset_scores__all_datasets.csv"
    rows = _read_csv_rows(path)
    return _collapse_rows(
        rows,
        key_fields=("dataset_id", "model_id"),
        numeric_fields=["overall_score", "query_success_rate"],
    )


def _load_validation_rows(validation_run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = validation_run_dir / "summaries" / "validation_summary__all_datasets.csv"
    rows = _read_csv_rows(path)
    return _collapse_rows(
        rows,
        key_fields=("dataset_id", "model_id"),
        numeric_fields=[column for column, _ in VALIDATION_COLUMNS],
    )


def _load_family_rows(analysis_run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    gathered: dict[tuple[str, str], dict[str, Any]] = {}
    per_family_numeric = ["family_score"]
    family_files = sorted((analysis_run_dir / "datasets").rglob("analysis_family_scores__*.csv"))
    raw_rows: list[dict[str, Any]] = []
    for path in family_files:
        raw_rows.extend(_read_csv_rows(path))
    collapsed = _collapse_rows(
        raw_rows,
        key_fields=("dataset_id", "model_id", "family_id"),
        numeric_fields=per_family_numeric,
    )
    for (dataset_id, model_id, family_id), row in collapsed.items():
        key = (dataset_id, model_id)
        payload = gathered.setdefault(key, {"dataset_id": dataset_id, "model_id": model_id})
        payload[family_id] = _to_float(row.get("family_score"))
    return gathered


def _build_runtime_audit(run_dir: Path) -> Path:
    runtime_dir = run_dir / "runtime_probe"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "eval" / "time_score_tradeoff" / "runner.py"),
        "--output-dir",
        str(runtime_dir),
        "--output-stem",
        "appendix_runtime",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    audit_path = runtime_dir / "appendix_runtime_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(f"Runtime audit CSV missing after probe run: {audit_path}")
    return audit_path


def _resolve_runtime_audit_source(explicit: Path | None, rebuild: bool, run_dir: Path) -> Path:
    if explicit is not None:
        return explicit.resolve()
    default_cached = PROJECT_ROOT / "Evaluation" / "time_score_tradeoff" / "time_score_tradeoff_audit.csv"
    if default_cached.exists() and not rebuild:
        cached_copy = run_dir / "runtime_probe" / "appendix_runtime_audit.csv"
        cached_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(default_cached, cached_copy)
        return cached_copy
    return _build_runtime_audit(run_dir)


def _load_runtime_rows(audit_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_csv_rows(audit_path)
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        normalized["model_name"] = _normalize_model_id(normalized.get("model_name"))
        normalized_rows.append(normalized)
    return _collapse_rows(
        normalized_rows,
        key_fields=("dataset_id", "model_name"),
        numeric_fields=["training_time_per_1k_seconds", "inference_time_per_1k_seconds", "overall_score_pct"],
    )


def _coverage_code(runtime_row: dict[str, Any] | None, has_any_score: bool) -> str:
    if not has_any_score:
        return "--"
    if runtime_row is None:
        return "NR"
    train_ok = str(runtime_row.get("panel_training_status") or "").strip().lower() == "valid"
    infer_ok = str(runtime_row.get("panel_inference_status") or "").strip().lower() == "valid"
    if train_ok and infer_ok:
        return "OK"
    if not train_ok and infer_ok:
        return "NT"
    if train_ok and not infer_ok:
        return "NG"
    return "NR"


def _build_full_rows(
    analysis_rows: dict[tuple[str, str], dict[str, Any]],
    validation_rows: dict[tuple[str, str], dict[str, Any]],
    family_rows: dict[tuple[str, str], dict[str, Any]],
    runtime_rows: dict[tuple[str, str], dict[str, Any]],
    dataset_ids: list[str],
) -> list[dict[str, Any]]:
    keys = set()
    keys.update(analysis_rows.keys())
    keys.update(validation_rows.keys())
    keys.update(family_rows.keys())
    keys.update(runtime_rows.keys())
    keys = {key for key in keys if key[1] in MODEL_ORDER and key[1] not in EXCLUDED_MODELS}

    ordered_keys = sorted(keys, key=lambda item: (_dataset_sort_key(item[0]), MODEL_ORDER.index(item[1]) if item[1] in MODEL_ORDER else 10**6, item[1]))
    full_rows: list[dict[str, Any]] = []
    for dataset_id, model_id in ordered_keys:
        analysis_row = analysis_rows.get((dataset_id, model_id))
        validation_row = validation_rows.get((dataset_id, model_id))
        family_row = family_rows.get((dataset_id, model_id))
        runtime_row = runtime_rows.get((dataset_id, model_id))
        has_any_score = any(item is not None for item in [analysis_row, validation_row, family_row])
        row = {
            "dataset_id": dataset_id,
            "dataset_prefix": dataset_id[:1],
            "model_id": model_id,
            "model_label": _model_label(model_id),
            "overall_score": _to_float(analysis_row.get("overall_score")) if analysis_row else None,
            "query_success_rate": _to_float(analysis_row.get("query_success_rate")) if analysis_row else None,
            "subgroup_structure": _to_float(family_row.get("subgroup_structure")) if family_row else None,
            "conditional_dependency_structure": _to_float(family_row.get("conditional_dependency_structure")) if family_row else None,
            "tail_rarity_structure": _to_float(family_row.get("tail_rarity_structure")) if family_row else None,
            "missingness_structure": _to_float(family_row.get("missingness_structure")) if family_row else None,
            "cardinality_range_score": _to_float(validation_row.get("cardinality_range_score")) if validation_row else None,
            "missing_introduction_score": _to_float(validation_row.get("missing_introduction_score")) if validation_row else None,
            "training_time_per_1k_seconds": _to_float(runtime_row.get("training_time_per_1k_seconds")) if runtime_row else None,
            "inference_time_per_1k_seconds": _to_float(runtime_row.get("inference_time_per_1k_seconds")) if runtime_row else None,
            "coverage_code": _coverage_code(runtime_row, has_any_score),
        }
        full_rows.append(row)
    return full_rows


def _build_missing_time_summary_rows(
    runtime_rows: dict[tuple[str, str], dict[str, Any]],
    dataset_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for model_id in MODEL_ORDER:
        missing_train: list[str] = []
        missing_gen: list[str] = []
        for dataset_id in dataset_ids:
            runtime_row = runtime_rows.get((dataset_id, model_id))
            train_ok = runtime_row is not None and str(runtime_row.get("panel_training_status") or "").strip().lower() == "valid"
            gen_ok = runtime_row is not None and str(runtime_row.get("panel_inference_status") or "").strip().lower() == "valid"
            if not train_ok:
                missing_train.append(dataset_id)
                detail_rows.append(
                    {
                        "model_id": model_id,
                        "model_label": _model_label(model_id),
                        "dataset_id": dataset_id,
                        "missing_time_kind": "train",
                    }
                )
            if not gen_ok:
                missing_gen.append(dataset_id)
                detail_rows.append(
                    {
                        "model_id": model_id,
                        "model_label": _model_label(model_id),
                        "dataset_id": dataset_id,
                        "missing_time_kind": "gen",
                    }
                )

        summary_rows.append(
            {
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "missing_train_count": len(missing_train),
                "missing_train_datasets": "|".join(missing_train),
                "missing_gen_count": len(missing_gen),
                "missing_gen_datasets": "|".join(missing_gen),
            }
        )

    return summary_rows, detail_rows


def _build_global_summary_rows(full_rows: list[dict[str, Any]], dataset_ids: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in full_rows:
        grouped.setdefault(str(row["model_id"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        items = grouped.get(model_id, [])
        if not items:
            continue
        coverage_count = sum(
            1
            for row in items
            if any(
                row.get(field) is not None
                for field in [
                    "overall_score",
                    "subgroup_structure",
                    "conditional_dependency_structure",
                    "tail_rarity_structure",
                    "missingness_structure",
                    "cardinality_range_score",
                ]
            )
        )
        summary_rows.append(
            {
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "coverage_count": coverage_count,
                "coverage_text": f"{coverage_count}/{len(dataset_ids)}",
                "overall_score": _median([row.get("overall_score") for row in items]),
                "query_success_rate": _median([row.get("query_success_rate") for row in items]),
                "subgroup_structure": _median([row.get("subgroup_structure") for row in items]),
                "conditional_dependency_structure": _median([row.get("conditional_dependency_structure") for row in items]),
                "tail_rarity_structure": _median([row.get("tail_rarity_structure") for row in items]),
                "missingness_structure": _median([row.get("missingness_structure") for row in items]),
                "cardinality_range_score": _median([row.get("cardinality_range_score") for row in items]),
                "missing_introduction_score": _median([row.get("missing_introduction_score") for row in items]),
                "training_time_per_1k_seconds": _median([row.get("training_time_per_1k_seconds") for row in items]),
                "inference_time_per_1k_seconds": _median([row.get("inference_time_per_1k_seconds") for row in items]),
            }
        )
    return summary_rows


def _build_coverage_matrix_rows(full_rows: list[dict[str, Any]], dataset_ids: list[str]) -> list[dict[str, Any]]:
    by_key = {(str(row["dataset_id"]), str(row["model_id"])): str(row["coverage_code"]) for row in full_rows}
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        payload = {"dataset_id": dataset_id}
        for model_id in MODEL_ORDER:
            payload[model_id] = by_key.get((dataset_id, model_id), "--")
        rows.append(payload)
    return rows


def _latex_escape(value: str) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _format_score(value: float | None) -> str:
    if value is None:
        return r"\textit{N/A}"
    return f"{value:.3f}"


def _format_time(value: float | None) -> str:
    if value is None:
        return r"\textit{N/A}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _render_coverage_matrix_tex(rows: list[dict[str, Any]]) -> str:
    header_models = " & ".join(_latex_escape(_model_label(model_id)) for model_id in MODEL_ORDER)
    body_lines = []
    for row in rows:
        cells = [row["dataset_id"]] + [row.get(model_id, "--") for model_id in MODEL_ORDER]
        body_lines.append(" & ".join(_latex_escape(str(cell)) for cell in cells) + r" \\")
    body = "\n".join(body_lines)
    return rf"""
\begin{{landscape}}
\begin{{center}}
\scriptsize
\setlength{{\tabcolsep}}{{3.0pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{longtable}}{{@{{}}l{'c' * len(MODEL_ORDER)}@{{}}}}
\caption{{Model coverage and audited runtime availability across all datasets. Codes: \texttt{{OK}} = score plus train/gen time available, \texttt{{NT}} = missing train time, \texttt{{NG}} = missing generation time, \texttt{{NR}} = score available but no audited runtime time.}}\\
\toprule
Dataset & {header_models} \\
\midrule
\endfirsthead
\toprule
Dataset & {header_models} \\
\midrule
\endhead
{body}
\bottomrule
\end{{longtable}}
\end{{center}}
\end{{landscape}}
""".strip() + "\n"


def _render_global_summary_tex(rows: list[dict[str, Any]]) -> str:
    header = " & ".join(column_label for _, column_label in GLOBAL_SUMMARY_COLUMNS)
    body_lines = []
    for row in rows:
        cells = [row["model_label"]]
        for key, _ in GLOBAL_SUMMARY_COLUMNS:
            if key == "coverage_count":
                cells.append(row["coverage_text"])
            elif "time" in key:
                cells.append(_format_time(row.get(key)))
            else:
                cells.append(_format_score(row.get(key)))
        body_lines.append(" & ".join(_latex_escape(str(cells[0])) if idx == 0 else str(cell) for idx, cell in enumerate(cells)) + r" \\")
    body = "\n".join(body_lines)
    return rf"""
\begin{{table*}}[p]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}l{'c' * len(GLOBAL_SUMMARY_COLUMNS)}@{{}}}}
\toprule
Model & {header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Global appendix summary by model. Score columns report medians over available datasets; time columns report median seconds per 1K rows.}}
\label{{tab:appendix_model_global_summary}}
\end{{table*}}
""".strip() + "\n"


def _render_full_results_tex(rows: list[dict[str, Any]], prefix: str, title: str, label: str) -> str:
    prefix_rows = [row for row in rows if row.get("dataset_prefix") == prefix]
    body_lines = []
    for row in prefix_rows:
        cells = [
            row["dataset_id"],
            row["model_label"],
            _format_score(row.get("overall_score")),
            _format_score(row.get("query_success_rate")),
            _format_score(row.get("subgroup_structure")),
            _format_score(row.get("conditional_dependency_structure")),
            _format_score(row.get("tail_rarity_structure")),
            _format_score(row.get("missingness_structure")),
            _format_score(row.get("cardinality_range_score")),
            _format_score(row.get("missing_introduction_score")),
        ]
        formatted = []
        for idx, cell in enumerate(cells):
            formatted.append(_latex_escape(str(cell)) if idx in {0, 1} else str(cell))
        body_lines.append(" & ".join(formatted) + r" \\")
    body = "\n".join(body_lines)
    return rf"""
\begin{{landscape}}
\begin{{center}}
\scriptsize
\setlength{{\tabcolsep}}{{3.3pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\begin{{longtable}}{{@{{}}llcccccccc@{{}}}}
\caption{{{title}}}
\label{{{label}}}\\
\toprule
Dataset & Model & Overall & QSR & Subg. & Cond. & Tail & Miss. & Card. & MissIntro \\
\midrule
\endfirsthead
\toprule
Dataset & Model & Overall & QSR & Subg. & Cond. & Tail & Miss. & Card. & MissIntro \\
\midrule
\endhead
{body}
\bottomrule
\end{{longtable}}
\end{{center}}
\end{{landscape}}
""".strip() + "\n"


def _render_generated_section(input_prefix: str) -> str:
    prefix = input_prefix.rstrip("/")
    return rf"""
\section{{Full Benchmark Results}}
\label{{app:full_benchmark_results}}
This appendix reports reproducible large tables for all benchmark datasets and all released models. We separate three kinds of information: coverage and audited runtime availability, global model-level medians, and full per-dataset result tables split by dataset family.

\input{{{prefix}/model_coverage_matrix_generated}}
\input{{{prefix}/model_global_summary_generated}}
\input{{{prefix}/full_results_c_generated}}
\input{{{prefix}/full_results_m_generated}}
\input{{{prefix}/full_results_n_generated}}
""".strip() + "\n"


def _render_standalone_tex() -> str:
    return r"""
\documentclass[11pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage{array}
\usepackage{hyperref}
\begin{document}
\input{standalone_appendix_full_results_generated}
\end{document}
""".strip() + "\n"


def _detect_latex_engine(explicit: str | None) -> list[str] | None:
    if explicit:
        return [explicit]
    for name in ["tectonic", "pdflatex", "xelatex", "latexmk"]:
        resolved = shutil.which(name)
        if resolved:
            return [resolved]
    return None


def _run_compile(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)


def _compile_tex(engine: list[str], tex_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_name = Path(engine[0]).stem.lower()
    if engine_name == "tectonic":
        command = engine + ["--keep-logs", "--reruns", "1", "--outdir", str(out_dir), str(tex_path.name)]
        _run_compile(command, cwd=tex_path.parent)
        pdf_path = out_dir / tex_path.with_suffix(".pdf").name
    elif engine_name == "latexmk":
        command = engine + ["-pdf", "-interaction=nonstopmode", f"-output-directory={out_dir}", str(tex_path.name)]
        _run_compile(command, cwd=tex_path.parent)
        pdf_path = out_dir / tex_path.with_suffix(".pdf").name
    else:
        command = engine + ["-interaction=nonstopmode", f"-output-directory={out_dir}", str(tex_path.name)]
        _run_compile(command, cwd=tex_path.parent)
        _run_compile(command, cwd=tex_path.parent)
        pdf_path = out_dir / tex_path.with_suffix(".pdf").name
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected PDF was not created: {pdf_path}")
    return pdf_path


def _mirror_latest(run_dir: Path) -> Path:
    task_name = run_dir.parent.parent.name
    latest_dir = OUTPUT_ROOT / task_name / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    return latest_dir


def _copy_pdf_if_exists(src: Path | None, dst: Path) -> None:
    if src is None or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_appendix_table_bundle(
    *,
    run_tag: str,
    analysis_run_dir: Path | None = None,
    validation_run_dir: Path | None = None,
    paper_dir: Path | None = None,
    compile_pdf: bool = True,
    latex_engine: str | None = None,
    runtime_audit_csv: Path | None = None,
    rebuild_runtime_audit: bool = False,
    task_name: str = "appendix_tables",
) -> dict[str, Any]:
    run_dir = make_task_run_dir(task_name, run_tag)
    sources = RunSources(
        analysis_run_dir=(analysis_run_dir or _resolve_latest_task_run_dir("analysis")).resolve(),
        validation_run_dir=(validation_run_dir or _resolve_latest_task_run_dir("validation")).resolve(),
        paper_dir=_resolve_paper_dir(paper_dir),
    )

    dataset_ids = _score_dataset_ids()
    analysis_rows = _load_analysis_asset_rows(sources.analysis_run_dir)
    validation_rows = _load_validation_rows(sources.validation_run_dir)
    family_rows = _load_family_rows(sources.analysis_run_dir)
    runtime_audit_path = _resolve_runtime_audit_source(runtime_audit_csv, rebuild_runtime_audit, run_dir)
    runtime_rows = _load_runtime_rows(runtime_audit_path)

    full_rows = _build_full_rows(analysis_rows, validation_rows, family_rows, runtime_rows, dataset_ids)
    global_summary_rows = _build_global_summary_rows(full_rows, dataset_ids)
    coverage_rows = _build_coverage_matrix_rows(full_rows, dataset_ids)
    missing_time_summary_rows, missing_time_detail_rows = _build_missing_time_summary_rows(runtime_rows, dataset_ids)

    tables_dir = run_dir / "tables"
    latex_dir = run_dir / "latex"
    pdf_dir = run_dir / "pdf"

    write_csv(tables_dir / "dataset_model_full_results.csv", full_rows)
    write_csv(tables_dir / "dataset_model_full_results_c.csv", [row for row in full_rows if row["dataset_prefix"] == "c"])
    write_csv(tables_dir / "dataset_model_full_results_m.csv", [row for row in full_rows if row["dataset_prefix"] == "m"])
    write_csv(tables_dir / "dataset_model_full_results_n.csv", [row for row in full_rows if row["dataset_prefix"] == "n"])
    write_csv(tables_dir / "model_global_summary.csv", global_summary_rows)
    write_csv(tables_dir / "model_coverage_matrix.csv", coverage_rows)
    write_csv(tables_dir / "model_missing_time_summary.csv", missing_time_summary_rows)
    write_csv(tables_dir / "model_missing_time_detail.csv", missing_time_detail_rows)

    _write_text(latex_dir / "model_coverage_matrix_generated.tex", _render_coverage_matrix_tex(coverage_rows))
    _write_text(latex_dir / "model_global_summary_generated.tex", _render_global_summary_tex(global_summary_rows))
    _write_text(
        latex_dir / "full_results_c_generated.tex",
        _render_full_results_tex(
            full_rows,
            prefix="c",
            title="Full per-dataset results for C-series datasets.",
            label="tab:appendix_full_results_c",
        ),
    )
    _write_text(
        latex_dir / "full_results_m_generated.tex",
        _render_full_results_tex(
            full_rows,
            prefix="m",
            title="Full per-dataset results for M-series datasets.",
            label="tab:appendix_full_results_m",
        ),
    )
    _write_text(
        latex_dir / "full_results_n_generated.tex",
        _render_full_results_tex(
            full_rows,
            prefix="n",
            title="Full per-dataset results for N-series datasets.",
            label="tab:appendix_full_results_n",
        ),
    )
    _write_text(
        latex_dir / "paper_appendix_full_results_generated.tex",
        _render_generated_section(f"../../Evaluation/{task_name}/latest/latex"),
    )
    _write_text(
        latex_dir / "standalone_appendix_full_results_generated.tex",
        _render_generated_section("."),
    )
    standalone_tex_path = latex_dir / "standalone_appendix_tables.tex"
    _write_text(standalone_tex_path, _render_standalone_tex())

    standalone_pdf_path: Path | None = None
    full_paper_pdf_path: Path | None = None
    compile_info: dict[str, Any] = {"compile_pdf": compile_pdf, "latex_engine": None}
    latest_dir = _mirror_latest(run_dir)
    if compile_pdf:
        engine = _detect_latex_engine(latex_engine)
        if engine is None:
            raise RuntimeError("No LaTeX engine found. Install tectonic/pdflatex or pass --latex-engine.")
        compile_info["latex_engine"] = engine[0]
        standalone_pdf_path = _compile_tex(engine, latest_dir / "latex" / "standalone_appendix_tables.tex", pdf_dir)
        paper_main = sources.paper_dir / "main.tex"
        full_paper_pdf_path = _compile_tex(engine, paper_main, sources.paper_dir / "out")
        _copy_pdf_if_exists(standalone_pdf_path, latest_dir / "pdf" / "appendix_tables_preview.pdf")
        _copy_pdf_if_exists(full_paper_pdf_path, pdf_dir / "tabquerybench_full_paper.pdf")
        _copy_pdf_if_exists(full_paper_pdf_path, latest_dir / "pdf" / "tabquerybench_full_paper.pdf")

    manifest = {
        "task": task_name,
        "run_tag": run_tag,
        "analysis_run_dir": str(sources.analysis_run_dir),
        "validation_run_dir": str(sources.validation_run_dir),
        "paper_dir": str(sources.paper_dir),
        "dataset_count": len(dataset_ids),
        "row_count_full_results": len(full_rows),
        "row_count_global_summary": len(global_summary_rows),
        "row_count_missing_time_summary": len(missing_time_summary_rows),
        "runtime_audit_csv": str(runtime_audit_path),
        "standalone_pdf": str(standalone_pdf_path) if standalone_pdf_path else None,
        "paper_pdf": str(full_paper_pdf_path) if full_paper_pdf_path else None,
        "latest_dir": str(latest_dir),
        "compile": compile_info,
        "runtime_audit_rebuilt": rebuild_runtime_audit or runtime_audit_csv is None and not (PROJECT_ROOT / "Evaluation" / "time_score_tradeoff" / "time_score_tradeoff_audit.csv").exists(),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(latest_dir / "manifest.json", manifest)
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "latest_dir": latest_dir,
        "standalone_pdf": standalone_pdf_path,
        "paper_pdf": full_paper_pdf_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reproducible appendix tables and PDFs.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag.")
    parser.add_argument("--analysis-run-dir", type=Path, default=None, help="Existing analysis run dir.")
    parser.add_argument("--validation-run-dir", type=Path, default=None, help="Existing validation run dir.")
    parser.add_argument("--paper-dir", type=Path, default=None, help="Paper directory containing main.tex.")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip LaTeX compilation.")
    parser.add_argument("--latex-engine", type=str, default=None, help="Explicit LaTeX engine executable.")
    parser.add_argument("--runtime-audit-csv", type=Path, default=None, help="Optional existing runtime audit CSV.")
    parser.add_argument("--rebuild-runtime-audit", action="store_true", help="Force rebuilding runtime audit from raw logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_appendix_table_bundle(
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
