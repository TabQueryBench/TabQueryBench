"""Build reusable figures from Evaluation/analysis and Evaluation/validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.eval.analysis.runner import run_sql_analysis
from src.eval.common import DEFAULT_SQL_SOURCE_VERSION, now_run_tag, write_csv, write_json
from src.eval.final_outputs import (
    build_image_report_tex,
    compile_tex,
    copy_files,
    render_pdf_to_png,
    task_version_final_dir,
    write_json as write_final_json,
    write_versioned_final_readme,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = PROJECT_ROOT / "Evaluation"
ANALYSIS_ROOT = EVALUATION_ROOT / "analysis"
VALIDATION_ROOT = EVALUATION_ROOT / "validation"
VIS_ROOT = EVALUATION_ROOT / "SQLvisualize"
TASK_NAME = "SQLvisualize"

ANALYSIS_FAMILIES: list[tuple[str, str]] = [
    ("subgroup_structure", "Subgroup"),
    ("conditional_dependency_structure", "Conditional"),
    ("tail_rarity_structure", "Tail/Rarity"),
    ("missingness_structure", "Missingness"),
]

VALIDATION_CHANNELS: list[tuple[str, str]] = [
    ("cardinality_range_score", "Cardinality"),
    ("missing_introduction_score", "Missing Intro"),
]

ANALYSIS_MODEL_METRICS: list[tuple[str, str]] = ANALYSIS_FAMILIES + [
    ("analysis_overall_score", "Overall"),
    ("analysis_query_success_rate", "Query Success"),
]

VALIDATION_MODEL_METRICS: list[tuple[str, str]] = VALIDATION_CHANNELS + [
    ("validation_overall_score", "Overall"),
]

COMBINED_MODEL_METRICS: list[tuple[str, str]] = ANALYSIS_FAMILIES + VALIDATION_CHANNELS

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

FAMILY_COLORS = {
    "subgroup_structure": "#4C78A8",
    "conditional_dependency_structure": "#2A9D8F",
    "tail_rarity_structure": "#F4A261",
    "missingness_structure": "#7B8C6B",
    "cardinality_range_score": "#5B8E7D",
    "missing_introduction_score": "#7C6EA6",
}


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _mean(values: list[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return round(sum(cleaned) / len(cleaned), 6)


def _natural_key(text: str) -> list[Any]:
    return [int(item) if item.isdigit() else item.lower() for item in re.split(r"(\d+)", text)]


def _normalize_model_id(model_id: str) -> str:
    key = str(model_id or "").strip().lower()
    if key == "rtf":
        return "realtabformer"
    return key


def _model_label(model_id: str) -> str:
    key = _normalize_model_id(model_id)
    return MODEL_LABELS.get(key, key or "unknown")


def _model_sort_key(model_id: str) -> tuple[int, Any]:
    return (0, _natural_key(_model_label(model_id)))


def _score_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "sqlviz_scores",
        ["#F8FBF6", "#DCECC2", "#A7D46F", "#63B058", "#2F7D32"],
    )
    cmap.set_bad("#ECEFF3")
    return cmap


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _analysis_summary_paths(run_dir: Path) -> tuple[Path, Path]:
    summary_dir = run_dir / "summaries"
    return (
        summary_dir / "analysis_query_scores__all_datasets.jsonl",
        summary_dir / "analysis_asset_scores__all_datasets.csv",
    )


def _validation_summary_paths(run_dir: Path) -> tuple[Path, Path]:
    summary_dir = run_dir / "summaries"
    return (
        summary_dir / "validation_summary__all_datasets.csv",
        summary_dir / "validation_details__all_datasets.jsonl",
    )


def _is_nonempty_analysis_run(run_dir: Path) -> bool:
    query_path, asset_path = _analysis_summary_paths(run_dir)
    return query_path.exists() and query_path.stat().st_size > 0 and _row_count(asset_path) > 0


def _is_nonempty_validation_run(run_dir: Path) -> bool:
    summary_path, _ = _validation_summary_paths(run_dir)
    return summary_path.exists() and _row_count(summary_path) > 0


def _resolve_latest_run_dir(task_root: Path, latest_file: Path, nonempty_check) -> Path | None:
    latest_payload = _read_json(latest_file, {}) or {}
    latest_run_dir = latest_payload.get("run_dir")
    if latest_run_dir:
        candidate = Path(str(latest_run_dir))
        if candidate.exists() and nonempty_check(candidate):
            return candidate

    runs_root = task_root / "runs"
    if not runs_root.exists():
        return None
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    for run_dir in sorted(candidates, key=lambda path: path.name, reverse=True):
        if nonempty_check(run_dir):
            return run_dir
    return None


def _ensure_analysis_run_dir(
    *,
    requested_run_dir: Path | None,
    allow_rebuild: bool,
    force_rebuild: bool,
    engines: tuple[str, ...],
    sql_source_version: str,
    latest_only: bool,
    max_workers: int,
) -> Path:
    if requested_run_dir is not None:
        candidate = requested_run_dir.expanduser().resolve()
        if not _is_nonempty_analysis_run(candidate):
            raise FileNotFoundError(f"Analysis run dir is missing or empty: {candidate}")
        return candidate

    if not force_rebuild:
        resolved = _resolve_latest_run_dir(
            ANALYSIS_ROOT,
            ANALYSIS_ROOT / "LATEST_RUN.json",
            _is_nonempty_analysis_run,
        )
        if resolved is not None:
            return resolved

    if not allow_rebuild:
        raise FileNotFoundError("No non-empty analysis run found under Evaluation/analysis.")

    rebuilt = run_sql_analysis(
        run_tag=now_run_tag(),
        datasets=None,
        latest_only=latest_only,
        engines=engines,
        sql_source_version=sql_source_version,
        include_all_sql_statements=True,
        max_sql_per_dataset=0,
        query_row_limit=0,
        max_workers=max_workers,
    )
    run_dir = Path(str(rebuilt["run_dir"])).resolve()
    if not _is_nonempty_analysis_run(run_dir):
        raise RuntimeError(f"Analysis rerun completed but produced no usable rows: {run_dir}")
    return run_dir


def _ensure_validation_run_dir(*, requested_run_dir: Path | None) -> Path:
    if requested_run_dir is not None:
        candidate = requested_run_dir.expanduser().resolve()
        if not _is_nonempty_validation_run(candidate):
            raise FileNotFoundError(f"Validation run dir is missing or empty: {candidate}")
        return candidate

    resolved = _resolve_latest_run_dir(
        VALIDATION_ROOT,
        VALIDATION_ROOT / "LATEST_RUN.json",
        _is_nonempty_validation_run,
    )
    if resolved is None:
        raise FileNotFoundError("No non-empty validation run found under Evaluation/validation.")
    return resolved


def _rollup_rows(rows: list[dict[str, Any]], group_field: str, metric_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field) or "")].append(row)

    out: list[dict[str, Any]] = []
    for group_value, items in grouped.items():
        record: dict[str, Any] = {group_field: group_value, "group_count": len(items)}
        for metric in metric_fields:
            record[metric] = _mean([_to_float(item.get(metric)) for item in items])
        out.append(record)
    return out


def _build_analysis_tables(query_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dataset_model_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    dataset_model_success: dict[tuple[str, str], list[float]] = defaultdict(list)
    dataset_model_family_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for row in query_rows:
        dataset_id = str(row.get("dataset_id") or "")
        model_id = _normalize_model_id(str(row.get("model_id") or ""))
        family_id = str(row.get("family_id") or "")
        score = _to_float(row.get("query_score"))
        success = 1.0 if str(row.get("synthetic_exec_ok") or "").lower() == "true" or row.get("synthetic_exec_ok") is True else 0.0
        if not dataset_id or not model_id or score is None:
            continue
        dataset_model_scores[(dataset_id, model_id)].append(score)
        dataset_model_success[(dataset_id, model_id)].append(success)
        if family_id:
            dataset_model_family_scores[(dataset_id, model_id, family_id)].append(score)

    dataset_model_rows: list[dict[str, Any]] = []
    for (dataset_id, model_id), scores in sorted(dataset_model_scores.items()):
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "analysis_overall_score": _mean(scores),
            "analysis_query_success_rate": _mean(dataset_model_success.get((dataset_id, model_id), [])),
        }
        for family_id, _ in ANALYSIS_FAMILIES:
            row[family_id] = _mean(dataset_model_family_scores.get((dataset_id, model_id, family_id), []))
        dataset_model_rows.append(row)

    dataset_model_rows.sort(key=lambda row: (_natural_key(str(row.get("dataset_id") or "")), _model_sort_key(str(row.get("model_id") or ""))))

    model_rows = _rollup_rows(
        dataset_model_rows,
        "model_id",
        [field for field, _ in ANALYSIS_MODEL_METRICS],
    )
    dataset_rows = _rollup_rows(
        dataset_model_rows,
        "dataset_id",
        [field for field, _ in ANALYSIS_MODEL_METRICS],
    )
    model_rows.sort(key=lambda row: _model_sort_key(str(row.get("model_id") or "")))
    dataset_rows.sort(key=lambda row: _natural_key(str(row.get("dataset_id") or "")))
    return {
        "dataset_model_rows": dataset_model_rows,
        "model_rows": model_rows,
        "dataset_rows": dataset_rows,
    }


def _build_validation_tables(summary_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dataset_model_channels: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in summary_rows:
        dataset_id = str(row.get("dataset_id") or "")
        model_id = _normalize_model_id(str(row.get("model_id") or ""))
        if not dataset_id or not model_id:
            continue
        for field, _ in VALIDATION_CHANNELS:
            value = _to_float(row.get(field))
            if value is not None:
                dataset_model_channels[(dataset_id, model_id)][field].append(value)

    dataset_model_rows: list[dict[str, Any]] = []
    for (dataset_id, model_id), metrics in sorted(dataset_model_channels.items()):
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "model_id": model_id,
        }
        overall_values: list[float | None] = []
        for field, _ in VALIDATION_CHANNELS:
            metric_value = _mean(metrics.get(field, []))
            row[field] = metric_value
            overall_values.append(metric_value)
        row["validation_overall_score"] = _mean(overall_values)
        dataset_model_rows.append(row)

    dataset_model_rows.sort(key=lambda row: (_natural_key(str(row.get("dataset_id") or "")), _model_sort_key(str(row.get("model_id") or ""))))

    model_rows = _rollup_rows(
        dataset_model_rows,
        "model_id",
        [field for field, _ in VALIDATION_MODEL_METRICS],
    )
    dataset_rows = _rollup_rows(
        dataset_model_rows,
        "dataset_id",
        [field for field, _ in VALIDATION_MODEL_METRICS],
    )
    model_rows.sort(key=lambda row: _model_sort_key(str(row.get("model_id") or "")))
    dataset_rows.sort(key=lambda row: _natural_key(str(row.get("dataset_id") or "")))
    return {
        "dataset_model_rows": dataset_model_rows,
        "model_rows": model_rows,
        "dataset_rows": dataset_rows,
    }


def _build_combined_model_rows(
    analysis_model_rows: list[dict[str, Any]],
    validation_model_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    analysis_by_model = {str(row.get("model_id") or ""): row for row in analysis_model_rows}
    validation_by_model = {str(row.get("model_id") or ""): row for row in validation_model_rows}
    model_ids = sorted(set(analysis_by_model) | set(validation_by_model), key=_model_sort_key)

    out: list[dict[str, Any]] = []
    for model_id in model_ids:
        merged = {"model_id": model_id}
        analysis_row = analysis_by_model.get(model_id, {})
        validation_row = validation_by_model.get(model_id, {})
        for field, _ in COMBINED_MODEL_METRICS:
            merged[field] = analysis_row.get(field)
            if field in validation_row:
                merged[field] = validation_row.get(field)
        out.append(merged)
    return out


def _build_combined_dataset_model_rows(
    analysis_dataset_model_rows: list[dict[str, Any]],
    validation_dataset_model_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    analysis_by_key = {
        (str(row.get("dataset_id") or ""), str(row.get("model_id") or "")): row for row in analysis_dataset_model_rows
    }
    validation_by_key = {
        (str(row.get("dataset_id") or ""), str(row.get("model_id") or "")): row for row in validation_dataset_model_rows
    }
    keys = sorted(set(analysis_by_key) & set(validation_by_key), key=lambda item: (_natural_key(item[0]), _model_sort_key(item[1])))

    out: list[dict[str, Any]] = []
    for key in keys:
        analysis_row = analysis_by_key[key]
        validation_row = validation_by_key[key]
        out.append(
            {
                "dataset_id": key[0],
                "model_id": key[1],
                "analysis_overall_score": analysis_row.get("analysis_overall_score"),
                "analysis_query_success_rate": analysis_row.get("analysis_query_success_rate"),
                "validation_overall_score": validation_row.get("validation_overall_score"),
            }
        )
    return out


def _matrix_from_rows(
    rows: list[dict[str, Any]],
    *,
    row_field: str,
    metric_specs: list[tuple[str, str]],
    row_sort_key,
    row_label_fn,
) -> tuple[np.ndarray, list[str], list[str]]:
    ordered_rows = sorted(rows, key=lambda row: row_sort_key(str(row.get(row_field) or "")))
    row_labels = [row_label_fn(str(row.get(row_field) or "")) for row in ordered_rows]
    col_labels = [label for _, label in metric_specs]
    matrix = np.full((len(ordered_rows), len(metric_specs)), np.nan, dtype=float)
    for ridx, row in enumerate(ordered_rows):
        for cidx, (field, _) in enumerate(metric_specs):
            value = _to_float(row.get(field))
            if value is not None:
                matrix[ridx, cidx] = value
    return matrix, row_labels, col_labels


def _plot_heatmap(
    *,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    out_path: Path,
    title: str,
    subtitle: str,
    annotate: bool,
    separators: list[float] | None = None,
) -> None:
    fig_w = max(9.0, 0.92 * max(1, len(col_labels)) + 4.8)
    fig_h = max(4.8, 0.34 * max(1, len(row_labels)) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")

    cmap = _score_cmap()
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10.5, color="#172338")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=(10 if len(row_labels) <= 20 else 8.2), color="#172338")
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if separators:
        for xpos in separators:
            ax.axvline(x=xpos, color="#20314A", linewidth=2.2)

    if annotate and matrix.shape[0] * matrix.shape[1] <= 220:
        for ridx in range(matrix.shape[0]):
            for cidx in range(matrix.shape[1]):
                value = matrix[ridx, cidx]
                text = "N/A" if np.isnan(value) else f"{float(value):.3f}"
                color = "#6D7688" if np.isnan(value) else "#10263B"
                ax.text(cidx, ridx, text, ha="center", va="center", fontsize=9.2, fontweight="bold", color=color)

    ax.set_title(title, fontsize=20, fontweight="bold", color="#162238", pad=16)
    fig.text(0.5, 0.94, subtitle, ha="center", fontsize=10, color="#5D6677")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)
    _save(fig, out_path)


def _plot_distribution(
    *,
    rows: list[dict[str, Any]],
    metric_specs: list[tuple[str, str]],
    out_path: Path,
    title: str,
    subtitle: str,
) -> None:
    series = []
    labels = []
    colors = []
    rng = np.random.default_rng(42)
    for field, label in metric_specs:
        values = [_to_float(row.get(field)) for row in rows]
        cleaned = [float(value) for value in values if value is not None]
        if not cleaned:
            continue
        series.append(cleaned)
        labels.append(label)
        colors.append(FAMILY_COLORS.get(field, "#4C78A8"))

    fig, ax = plt.subplots(figsize=(max(8.5, 1.2 * len(labels) + 3.0), 5.6), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.22)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    box = ax.boxplot(series, patch_artist=True, widths=0.56, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(color)
    for median in box["medians"]:
        median.set_color("#20314A")
        median.set_linewidth(2.0)

    for idx, values in enumerate(series, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            np.full(len(values), idx, dtype=float) + jitter,
            values,
            s=16,
            alpha=0.35,
            color=colors[idx - 1],
            edgecolors="none",
        )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=19, fontweight="bold", color="#162238", pad=12)
    fig.text(0.5, 0.93, subtitle, ha="center", fontsize=10, color="#5D6677")
    _save(fig, out_path)


def _plot_scatter(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.6), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    model_ids = sorted({_normalize_model_id(str(row.get("model_id") or "")) for row in rows}, key=_model_sort_key)
    palette = plt.cm.get_cmap("tab20", max(1, len(model_ids)))
    color_map = {model_id: palette(idx) for idx, model_id in enumerate(model_ids)}

    x_values: list[float] = []
    y_values: list[float] = []
    for model_id in model_ids:
        points = [row for row in rows if _normalize_model_id(str(row.get("model_id") or "")) == model_id]
        xs = [_to_float(row.get("analysis_overall_score")) for row in points]
        ys = [_to_float(row.get("validation_overall_score")) for row in points]
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if not pairs:
            continue
        x_values.extend(x for x, _ in pairs)
        y_values.extend(y for _, y in pairs)
        ax.scatter(
            [x for x, _ in pairs],
            [y for _, y in pairs],
            s=42,
            alpha=0.78,
            color=color_map[model_id],
            label=_model_label(model_id),
            edgecolors="white",
            linewidths=0.45,
        )

    if x_values and y_values:
        ax.axvline(sum(x_values) / len(x_values), linestyle="--", linewidth=1.1, color="#7A869A")
        ax.axhline(sum(y_values) / len(y_values), linestyle="--", linewidth=1.1, color="#7A869A")

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Analysis Overall Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Overall Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=20, fontweight="bold", color="#162238", pad=12)
    fig.text(0.5, 0.93, subtitle, ha="center", fontsize=10, color="#5D6677")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower left")
    _save(fig, out_path)


def _write_sqlvisualize_final_bundle(
    *,
    out_dir: Path,
    manifest: dict[str, Any],
    latex_engine: str | None,
) -> dict[str, Any]:
    sql_source_version = str(manifest.get("analysis_sql_source_version") or DEFAULT_SQL_SOURCE_VERSION)
    final_dir = task_version_final_dir(TASK_NAME, sql_source_version)
    final_dir.mkdir(parents=True, exist_ok=True)
    write_versioned_final_readme(
        task_name=TASK_NAME,
        title="SQLvisualize final outputs",
        summary="Versioned final bundles for paper-facing SQL/validation visualization artifacts.",
        notes=[
            "Each bundle contains the report source (`.tex`), compiled PDF, rendered PNG overview, and the key PNG/CSV files copied from the concrete run directory.",
        ],
    )

    figures_dir = final_dir / "figures"
    tables_dir = final_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    figure_files = sorted(out_dir.rglob("*.png"))
    table_files = sorted((out_dir / "tables").glob("*.csv"))
    copy_files(figures_dir, figure_files)
    copy_files(tables_dir, table_files)

    report_note_path = final_dir / "sqlvisualize_summary.md"
    report_tex_path = final_dir / "sqlvisualize_report.tex"
    report_png_path = final_dir / "sqlvisualize_report.png"
    report_manifest_path = final_dir / "sqlvisualize_final_manifest.json"

    note_lines = [
        "# SQLvisualize Final Bundle",
        "",
        f"- analysis_run_dir: `{manifest['analysis_run_dir']}`",
        f"- validation_run_dir: `{manifest['validation_run_dir']}`",
        f"- sql_source: `{manifest.get('analysis_sql_source_label') or ''}` (`{sql_source_version}`)",
        f"- analysis_query_row_count: `{manifest['analysis_query_row_count']}`",
        f"- validation_summary_row_count: `{manifest['validation_summary_row_count']}`",
        f"- figure_count: `{manifest['figure_count']}`",
        "",
    ]
    _write_text(report_note_path, "\n".join(note_lines))

    key_images = [
        {
            "heading": "Analysis Family Heatmap",
            "caption": "Model-level mean SQL analysis scores across analytics families.",
            "path": figures_dir / "01_analysis_model_family_heatmap.png",
        },
        {
            "heading": "Validation Channel Heatmap",
            "caption": "Model-level mean validation scores across integrity channels.",
            "path": figures_dir / "01_validation_model_channel_heatmap.png",
        },
        {
            "heading": "Combined Capability Heatmap",
            "caption": "Paper-facing merged view of analytics families and validation channels.",
            "path": figures_dir / "01_model_capability_heatmap.png",
        },
        {
            "heading": "Analysis vs Validation Scatter",
            "caption": "Dataset-model tradeoff view showing whether high SQL analysis scores align with strong validation scores.",
            "path": figures_dir / "02_analysis_validation_scatter.png",
        },
    ]
    report_tex = build_image_report_tex(
        title="SQLvisualize Final Report",
        subtitle="Key heatmaps and alignment plots built from the current analysis + validation outputs.",
        intro_lines=[
            f"analysis_run_dir={manifest['analysis_run_dir']}",
            f"validation_run_dir={manifest['validation_run_dir']}",
            f"sql_source={manifest.get('analysis_sql_source_label') or ''} ({sql_source_version})",
        ],
        image_entries=[entry for entry in key_images if entry["path"].exists()],
    )
    _write_text(report_tex_path, report_tex)
    report_pdf_path, report_log_path = compile_tex(report_tex_path, latex_engine=latex_engine)
    render_pdf_to_png(report_pdf_path, report_png_path, densest_page=True)

    write_final_json(final_dir / "sqlvisualize_run_manifest.json", manifest)
    final_manifest = {
        "task": TASK_NAME,
        "run_tag": manifest.get("run_tag"),
        "run_dir": str(out_dir.resolve()),
        "final_dir": str(final_dir.resolve()),
        "sql_source_version": sql_source_version,
        "sql_source_label": manifest.get("analysis_sql_source_label"),
        "report_note": str(report_note_path.resolve()),
        "report_tex": str(report_tex_path.resolve()),
        "report_pdf": str(report_pdf_path.resolve()),
        "report_png": str(report_png_path.resolve()),
        "report_compile_log": str(report_log_path.resolve()),
        "figure_dir": str(figures_dir.resolve()),
        "table_dir": str(tables_dir.resolve()),
    }
    write_final_json(report_manifest_path, final_manifest)
    return final_manifest


def run_sqlvisualize(
    *,
    run_tag: str | None = None,
    analysis_run_dir: Path | None = None,
    validation_run_dir: Path | None = None,
    allow_analysis_rebuild: bool = False,
    force_analysis_rebuild: bool = False,
    analysis_engines: tuple[str, ...] = ("cli-all",),
    analysis_sql_source_version: str = DEFAULT_SQL_SOURCE_VERSION,
    analysis_latest_only: bool = False,
    analysis_max_workers: int = 4,
    latex_engine: str | None = None,
) -> dict[str, Any]:
    resolved_analysis_dir = _ensure_analysis_run_dir(
        requested_run_dir=analysis_run_dir,
        allow_rebuild=allow_analysis_rebuild,
        force_rebuild=force_analysis_rebuild,
        engines=analysis_engines,
        sql_source_version=analysis_sql_source_version,
        latest_only=analysis_latest_only,
        max_workers=analysis_max_workers,
    )
    resolved_validation_dir = _ensure_validation_run_dir(requested_run_dir=validation_run_dir)

    analysis_query_path, _ = _analysis_summary_paths(resolved_analysis_dir)
    validation_summary_path, _ = _validation_summary_paths(resolved_validation_dir)

    analysis_query_rows = _read_jsonl_rows(analysis_query_path)
    validation_summary_rows = _read_csv_rows(validation_summary_path)
    if not analysis_query_rows:
        raise RuntimeError(f"No analysis query rows available in {resolved_analysis_dir}")
    if not validation_summary_rows:
        raise RuntimeError(f"No validation summary rows available in {resolved_validation_dir}")

    analysis_tables = _build_analysis_tables(analysis_query_rows)
    validation_tables = _build_validation_tables(validation_summary_rows)
    combined_model_rows = _build_combined_model_rows(
        analysis_tables["model_rows"],
        validation_tables["model_rows"],
    )
    combined_dataset_model_rows = _build_combined_dataset_model_rows(
        analysis_tables["dataset_model_rows"],
        validation_tables["dataset_model_rows"],
    )

    actual_run_tag = run_tag or now_run_tag()
    out_dir = VIS_ROOT / "runs" / actual_run_tag
    analysis_dir = out_dir / "analysis"
    validation_dir = out_dir / "validation"
    combined_dir = out_dir / "combined"
    tables_dir = out_dir / "tables"

    analysis_matrix, analysis_rows, analysis_cols = _matrix_from_rows(
        analysis_tables["model_rows"],
        row_field="model_id",
        metric_specs=ANALYSIS_MODEL_METRICS,
        row_sort_key=_model_sort_key,
        row_label_fn=_model_label,
    )
    _plot_heatmap(
        matrix=analysis_matrix,
        row_labels=analysis_rows,
        col_labels=analysis_cols,
        out_path=analysis_dir / "01_analysis_model_family_heatmap.png",
        title="Analysis Evaluation: Model-Family Heatmap",
        subtitle="Dataset-level mean across model-dataset pairs",
        annotate=True,
        separators=[3.5, 4.5],
    )

    analysis_dataset_matrix, analysis_dataset_rows, analysis_dataset_cols = _matrix_from_rows(
        analysis_tables["dataset_rows"],
        row_field="dataset_id",
        metric_specs=ANALYSIS_FAMILIES + [("analysis_overall_score", "Overall")],
        row_sort_key=lambda value: _natural_key(value),
        row_label_fn=lambda value: value.upper(),
    )
    _plot_heatmap(
        matrix=analysis_dataset_matrix,
        row_labels=analysis_dataset_rows,
        col_labels=analysis_dataset_cols,
        out_path=analysis_dir / "02_analysis_dataset_family_heatmap.png",
        title="Analysis Evaluation: Dataset Difficulty Heatmap",
        subtitle="Each row is a dataset average across available models",
        annotate=False,
        separators=[3.5],
    )

    _plot_distribution(
        rows=analysis_tables["dataset_model_rows"],
        metric_specs=ANALYSIS_FAMILIES + [("analysis_overall_score", "Overall")],
        out_path=analysis_dir / "03_analysis_family_distributions.png",
        title="Analysis Evaluation: Family Score Distributions",
        subtitle="Distribution across dataset-model pairs",
    )

    validation_matrix, validation_rows, validation_cols = _matrix_from_rows(
        validation_tables["model_rows"],
        row_field="model_id",
        metric_specs=VALIDATION_MODEL_METRICS,
        row_sort_key=_model_sort_key,
        row_label_fn=_model_label,
    )
    _plot_heatmap(
        matrix=validation_matrix,
        row_labels=validation_rows,
        col_labels=validation_cols,
        out_path=validation_dir / "01_validation_model_channel_heatmap.png",
        title="Validation Evaluation: Model-Channel Heatmap",
        subtitle="Dataset-level mean across model-dataset pairs",
        annotate=True,
        separators=[3.5],
    )

    validation_dataset_matrix, validation_dataset_rows, validation_dataset_cols = _matrix_from_rows(
        validation_tables["dataset_rows"],
        row_field="dataset_id",
        metric_specs=VALIDATION_CHANNELS + [("validation_overall_score", "Overall")],
        row_sort_key=lambda value: _natural_key(value),
        row_label_fn=lambda value: value.upper(),
    )
    _plot_heatmap(
        matrix=validation_dataset_matrix,
        row_labels=validation_dataset_rows,
        col_labels=validation_dataset_cols,
        out_path=validation_dir / "02_validation_dataset_channel_heatmap.png",
        title="Validation Evaluation: Dataset Difficulty Heatmap",
        subtitle="Each row is a dataset average across available models",
        annotate=False,
        separators=[3.5],
    )

    _plot_distribution(
        rows=validation_tables["dataset_model_rows"],
        metric_specs=VALIDATION_CHANNELS + [("validation_overall_score", "Overall")],
        out_path=validation_dir / "03_validation_channel_distributions.png",
        title="Validation Evaluation: Channel Score Distributions",
        subtitle="Distribution across dataset-model pairs",
    )

    combined_matrix, combined_rows, combined_cols = _matrix_from_rows(
        combined_model_rows,
        row_field="model_id",
        metric_specs=COMBINED_MODEL_METRICS,
        row_sort_key=_model_sort_key,
        row_label_fn=_model_label,
    )
    _plot_heatmap(
        matrix=combined_matrix,
        row_labels=combined_rows,
        col_labels=combined_cols,
        out_path=combined_dir / "01_model_capability_heatmap.png",
        title="Paper-Aligned Capability Heatmap",
        subtitle="Analytics families + validation channels aggregated at model level",
        annotate=True,
        separators=[3.5],
    )

    _plot_scatter(
        rows=combined_dataset_model_rows,
        out_path=combined_dir / "02_analysis_validation_scatter.png",
        title="Analysis vs Validation Across Dataset-Model Pairs",
        subtitle="Higher-right points preserve analytics and integrity better at the same time",
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "analysis_dataset_model_summary.csv", analysis_tables["dataset_model_rows"])
    write_csv(tables_dir / "analysis_model_summary.csv", analysis_tables["model_rows"])
    write_csv(tables_dir / "analysis_dataset_summary.csv", analysis_tables["dataset_rows"])
    write_csv(tables_dir / "validation_dataset_model_summary.csv", validation_tables["dataset_model_rows"])
    write_csv(tables_dir / "validation_model_summary.csv", validation_tables["model_rows"])
    write_csv(tables_dir / "validation_dataset_summary.csv", validation_tables["dataset_rows"])
    write_csv(tables_dir / "combined_model_summary.csv", combined_model_rows)
    write_csv(tables_dir / "combined_dataset_model_summary.csv", combined_dataset_model_rows)

    figure_paths = sorted(str(path.resolve()) for path in out_dir.rglob("*.png"))
    manifest = {
        "task": "sqlvisualize",
        "run_tag": actual_run_tag,
        "run_dir": str(out_dir.resolve()),
        "analysis_run_dir": str(resolved_analysis_dir.resolve()),
        "validation_run_dir": str(resolved_validation_dir.resolve()),
        "analysis_sql_source_version": (
            str(analysis_query_rows[0].get("sql_source_version") or "") if analysis_query_rows else ""
        ),
        "analysis_sql_source_label": (
            str(analysis_query_rows[0].get("sql_source_label") or "") if analysis_query_rows else ""
        ),
        "analysis_query_row_count": len(analysis_query_rows),
        "validation_summary_row_count": len(validation_summary_rows),
        "figure_count": len(figure_paths),
        "figures": figure_paths,
    }
    final_manifest = _write_sqlvisualize_final_bundle(out_dir=out_dir, manifest=manifest, latex_engine=latex_engine)
    manifest["final_outputs"] = final_manifest
    write_json(out_dir / "manifest.json", manifest)
    write_json(VIS_ROOT / "LATEST_RUN.json", {"run_tag": actual_run_tag, "run_dir": str(out_dir.resolve())})
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable SQL/validation figures from evaluation outputs.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag.")
    parser.add_argument("--analysis-run-dir", type=Path, default=None, help="Optional analysis run dir.")
    parser.add_argument("--validation-run-dir", type=Path, default=None, help="Optional validation run dir.")
    parser.add_argument("--no-rebuild-analysis", action="store_true", help="Do not rebuild analysis if missing.")
    parser.add_argument("--force-rebuild-analysis", action="store_true", help="Force rebuilding analysis.")
    parser.add_argument("--engines", nargs="*", default=["cli"], help="SQL engine filter when rebuilding analysis.")
    parser.add_argument("--sql-source-version", type=str, default=DEFAULT_SQL_SOURCE_VERSION, help="SQL source version.")
    parser.add_argument("--latest-only", action="store_true", help="Use latest-only synthetic assets when rebuilding analysis.")
    parser.add_argument("--max-workers", type=int, default=1, help="Worker count when rebuilding analysis.")
    parser.add_argument("--latex-engine", type=str, default=None, help="Optional LaTeX engine.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_sqlvisualize(
        run_tag=args.run_tag or now_run_tag(),
        analysis_run_dir=args.analysis_run_dir,
        validation_run_dir=args.validation_run_dir,
        allow_rebuild_analysis=not args.no_rebuild_analysis,
        force_rebuild_analysis=args.force_rebuild_analysis,
        engines=tuple(args.engines),
        sql_source_version=args.sql_source_version,
        latest_only=args.latest_only,
        max_workers=max(1, int(args.max_workers)),
        latex_engine=args.latex_engine,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
