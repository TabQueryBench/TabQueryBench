from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from PIL import Image

from src.eval.query_fivepart_breakdown.common_final import render_final_readme, sync_final_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "Evaluation" / "overview_regenerated"
DATA_DIR = OUTPUT_ROOT / "data"
FIGURES_DIR = OUTPUT_ROOT / "figures"
FINAL_DIR = OUTPUT_ROOT / "final"

BENCHMARK_OVERALL_FINAL = PROJECT_ROOT / "Evaluation" / "benchmark_overall_table" / "final"
DEFAULT_DATASET_LEVEL_CSV = BENCHMARK_OVERALL_FINAL / "benchmark_overall_table_real_dataset_level.csv"
DEFAULT_MODEL_SUMMARY_CSV = BENCHMARK_OVERALL_FINAL / "benchmark_overall_table_real_model_summary.csv"
DEFAULT_SOURCE_MANIFEST_CSV = BENCHMARK_OVERALL_FINAL / "benchmark_overall_table_real_sources.csv"
DEFAULT_MODEL_RADAR_MANIFEST = PROJECT_ROOT / "Evaluation" / "model_radar" / "final" / "manifest.json"
DEFAULT_OVERVIEW_BACKGROUND_PNG = PROJECT_ROOT / "Evaluation" / "overview" / "final" / "overview7.png"

DEFAULT_BAR_METRIC_KEY = "query_overall"
DEFAULT_BAR_METRIC_LABEL = "Query Overall"
SECONDARY_BAR_METRIC_KEY = "distance_overall"
SQL_PANEL_RATIO = 1423 / 324
DISTANCE_PANEL_RATIO = 943 / 389
RADAR_PANEL_RATIO = 456 / 395
SQL_PANEL_BORDER = "#5140C8"
DISTANCE_PANEL_BORDER = "#FF661F"
RADAR_PANEL_BORDER = "#5140C8"
COMBINED_PREVIEW_SIZE = (1448, 740)
COMBINED_SQL_BOX = (14, 13, 1434, 358)
COMBINED_DISTANCE_BOX = (17, 378, 979, 773)
COMBINED_RADAR_BOX = (985, 378, 1441, 773)
FULL_OVERVIEW_SIZE = (1448, 1086)
FULL_OVERVIEW_SQL_BOX = (13, 247, 1436, 584)
FULL_OVERVIEW_DISTANCE_BOX = (14, 590, 968, 1001)
FULL_OVERVIEW_RADAR_BOX = (975, 590, 1438, 1001)
RADAR_AXIS_ORDER = [
    ("distance_overall", "Distance"),
    ("subgroup_structure", "Subgroup"),
    ("conditional_dependency_structure", "Conditional"),
    ("tail_breakdown", "Tail"),
    ("missingness_structure", "Missingness"),
    ("cardinality_structure", "Cardinality"),
]

RADAR_PANEL_LABEL_RADIUS = {
    "Distance": 1.02,
    "Subgroup": 1.13,
    "Conditional": 1.13,
    "Tail": 1.07,
    "Missingness": 1.13,
    "Cardinality": 1.13,
}

MODEL_ORDER = [
    "realtabformer",
    "bayesnet",
    "tabpfgen",
    "arf",
    "ctgan",
    "tvae",
    "tabdiff",
    "tabbyflow",
    "tabsyn",
    "forestdiffusion",
    "tabddpm",
]

PANEL_MODEL_ORDER = [
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
    "real": "REAL",
    "arf": "ARF",
    "bayesnet": "BayesNet",
    "ctgan": "CTGAN",
    "forestdiffusion": "ForestDiffusion",
    "realtabformer": "RealTabFormer",
    "tabbyflow": "TabbyFlow",
    "tabddpm": "TabDDPM",
    "tabdiff": "TabDiff",
    "tabpfgen": "TabPFGen",
    "tabsyn": "TabSyn",
    "tvae": "TVAE",
}

MODEL_SHORT_LABELS = {
    "arf": "ARF",
    "bayesnet": "BayesNet",
    "ctgan": "CTGAN",
    "forestdiffusion": "ForestDiff",
    "realtabformer": "RTF",
    "tabbyflow": "TabbyFlow",
    "tabddpm": "TabDDPM",
    "tabdiff": "TabDiff",
    "tabpfgen": "TabPFGen",
    "tabsyn": "TabSyn",
    "tvae": "TVAE",
}

MODEL_COLORS = {
    "real": "#000000",
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

SQL_PANEL_AXES = [
    ("subgroup_structure_mean", "Subgroup"),
    ("conditional_dependency_structure_mean", "Conditional"),
    ("tail_breakdown_mean", "Tail"),
    ("missingness_structure_mean", "Missingness"),
    ("cardinality_structure_mean", "Cardinality"),
]

DISTANCE_PANEL_AXES = [
    ("jsd_distance_mean", "JSD"),
    ("ks_distance_mean", "KSD"),
    ("tvd_distance_mean", "TVD"),
    ("wasserstein_distance_mean", "Wasserstein"),
]


@dataclass(frozen=True)
class SourceVerdict:
    dataset_level_csv: Path
    source_manifest_csv: Path
    current_result_line: str
    current_result_label: str
    current_result_note: str
    radar_manifest_path: Path | None


def _ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, DATA_DIR, FIGURES_DIR, FINAL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _load_source_verdict(
    dataset_level_csv: Path,
    source_manifest_csv: Path,
    model_radar_manifest_path: Path,
) -> SourceVerdict:
    if not dataset_level_csv.exists():
        raise FileNotFoundError(f"Missing dataset-level benchmark summary: {dataset_level_csv}")
    if not source_manifest_csv.exists():
        raise FileNotFoundError(f"Missing benchmark source manifest: {source_manifest_csv}")

    if model_radar_manifest_path.exists():
        payload = json.loads(model_radar_manifest_path.read_text(encoding="utf-8"))
        line_version = str(payload.get("sql_source_version") or "v2")
        line_label = str(payload.get("sql_source_label") or f"{line_version}_current")
        note = "Aligned to the existing paper-facing model_radar manifest."
        radar_manifest = model_radar_manifest_path
    else:
        line_version = "v2"
        line_label = "v2_current"
        note = "Fell back to the existing paper-facing benchmark summary artifacts."
        radar_manifest = None

    return SourceVerdict(
        dataset_level_csv=dataset_level_csv,
        source_manifest_csv=source_manifest_csv,
        current_result_line=line_version,
        current_result_label=line_label,
        current_result_note=note,
        radar_manifest_path=radar_manifest,
    )


def _read_dataset_level_table(dataset_level_csv: Path) -> pd.DataFrame:
    table = pd.read_csv(dataset_level_csv)
    if "metric_key" not in table.columns or "metric_value" not in table.columns:
        raise ValueError(f"Unexpected dataset-level schema: {dataset_level_csv}")
    table["model_id"] = table["model_id"].astype(str).str.lower()
    table["metric_key"] = table["metric_key"].astype(str)
    table["metric_value"] = pd.to_numeric(table["metric_value"], errors="coerce")
    table = table[table["row_kind"] == "synthetic"].copy()
    table = table[table["model_id"].isin(MODEL_COLORS)].copy()
    return table


def _read_model_summary_table(model_summary_csv: Path) -> pd.DataFrame:
    table = pd.read_csv(model_summary_csv)
    table["model_id"] = table["model_id"].astype(str).str.lower()
    table = table[table["row_kind"] == "synthetic"].copy()
    table = table[table["model_id"].isin(MODEL_COLORS)].copy()
    return table


def _read_source_manifest(source_manifest_csv: Path) -> pd.DataFrame:
    return pd.read_csv(source_manifest_csv)


def _metric_title_lookup(source_manifest: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in source_manifest.to_dict("records"):
        lookup[str(row["metric_key"])] = str(row["metric_title"])
    return lookup


def _source_file_lookup(source_manifest: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in source_manifest.to_dict("records"):
        lookup[str(row["metric_key"])] = str(row["source_file"])
    return lookup


def _aggregate_metric_rows(metric_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metric_df.groupby(["model_id", "model_label"], as_index=False).agg(
            metric_mean=("metric_value", "mean"),
            metric_std=("metric_value", "std"),
            dataset_count=("metric_value", "count"),
        )
    )
    grouped["metric_std"] = grouped["metric_std"].fillna(0.0)
    grouped["metric_se"] = grouped["metric_std"] / np.sqrt(grouped["dataset_count"])
    grouped["metric_ci95_radius"] = 1.96 * grouped["metric_se"]
    grouped["metric_ci95_low"] = grouped["metric_mean"] - grouped["metric_ci95_radius"]
    grouped["metric_ci95_high"] = grouped["metric_mean"] + grouped["metric_ci95_radius"]
    grouped["model_color"] = grouped["model_id"].map(MODEL_COLORS)
    grouped["model_rank_desc"] = grouped["metric_mean"].rank(method="first", ascending=False).astype(int)
    grouped["frozen_model_order"] = grouped["model_id"].map(
        {model_id: index for index, model_id in enumerate(MODEL_ORDER, start=1)}
    )
    return grouped


def _build_bar_source_table(
    dataset_level: pd.DataFrame,
    source_manifest: pd.DataFrame,
    *,
    bar_metric_key: str,
    verdict: SourceVerdict,
) -> pd.DataFrame:
    metric_df = dataset_level[dataset_level["metric_key"] == bar_metric_key].copy()
    if metric_df.empty:
        raise ValueError(f"No rows found for metric_key={bar_metric_key!r}")

    grouped = _aggregate_metric_rows(metric_df)
    metric_titles = _metric_title_lookup(source_manifest)
    source_files = _source_file_lookup(source_manifest)
    grouped["chart_name"] = "overview_model_bar_chart"
    grouped["metric_key"] = bar_metric_key
    grouped["metric_title"] = metric_titles.get(bar_metric_key, DEFAULT_BAR_METRIC_LABEL)
    grouped["sort_rule"] = "descending metric_mean"
    grouped["error_bar_rule"] = "95% CI across dataset-level metric values"
    grouped["source_of_truth_dataset_level_csv"] = str(verdict.dataset_level_csv.resolve())
    grouped["source_of_truth_metric_manifest_csv"] = str(verdict.source_manifest_csv.resolve())
    grouped["source_of_truth_line_version"] = verdict.current_result_line
    grouped["source_of_truth_line_label"] = verdict.current_result_label
    grouped["primary_source_file"] = source_files.get(bar_metric_key, "derived from benchmark overall dataset-level table")
    grouped["derivation"] = (
        "dataset-level query_overall values already materialized in benchmark_overall_table_real_dataset_level.csv"
        if bar_metric_key == "query_overall"
        else (
            "dataset-level distance_overall values already materialized in benchmark_overall_table_real_dataset_level.csv"
            if bar_metric_key == "distance_overall"
            else "dataset-level metric values already materialized in benchmark_overall_table_real_dataset_level.csv"
        )
    )
    return grouped.sort_values(["metric_mean", "model_label"], ascending=[False, True]).reset_index(drop=True)


def _build_radar_source_table(
    dataset_level: pd.DataFrame,
    source_manifest: pd.DataFrame,
    *,
    verdict: SourceVerdict,
) -> pd.DataFrame:
    metric_titles = _metric_title_lookup(source_manifest)
    source_files = _source_file_lookup(source_manifest)
    rows: list[dict[str, Any]] = []
    for axis_order, (metric_key, axis_label) in enumerate(RADAR_AXIS_ORDER, start=1):
        metric_df = dataset_level[dataset_level["metric_key"] == metric_key].copy()
        if metric_df.empty:
            raise ValueError(f"Radar axis source missing: {metric_key}")
        grouped = _aggregate_metric_rows(metric_df)
        for row in grouped.to_dict("records"):
            rows.append(
                {
                    "chart_name": "overview_model_radar_chart",
                    "model_id": row["model_id"],
                    "model_label": row["model_label"],
                    "model_color": row["model_color"],
                    "axis_order": axis_order,
                    "axis_key": metric_key,
                    "axis_label": axis_label,
                    "axis_metric_title": metric_titles.get(metric_key, axis_label),
                    "axis_value_mean": row["metric_mean"],
                    "axis_value_std": row["metric_std"],
                    "axis_value_se": row["metric_se"],
                    "axis_value_ci95_radius": row["metric_ci95_radius"],
                    "axis_value_ci95_low": row["metric_ci95_low"],
                    "axis_value_ci95_high": row["metric_ci95_high"],
                    "dataset_count": row["dataset_count"],
                    "radial_min": 0.0,
                    "radial_max": 1.0,
                    "normalization_rule": "none; benchmark scores are already on a 0-1 scale",
                    "closed_polygon_order": "Distance -> Subgroup -> Conditional -> Tail -> Missingness -> Cardinality -> Distance",
                    "source_of_truth_dataset_level_csv": str(verdict.dataset_level_csv.resolve()),
                    "source_of_truth_metric_manifest_csv": str(verdict.source_manifest_csv.resolve()),
                    "source_of_truth_line_version": verdict.current_result_line,
                    "source_of_truth_line_label": verdict.current_result_label,
                    "primary_source_file": source_files.get(metric_key, ""),
                }
            )
    radar_table = pd.DataFrame(rows)
    radar_table["radar_mean_across_axes"] = radar_table.groupby("model_id")["axis_value_mean"].transform("mean")
    radar_table["radar_rank_desc"] = (
        radar_table[["model_id", "radar_mean_across_axes"]]
        .drop_duplicates()
        .sort_values(["radar_mean_across_axes", "model_id"], ascending=[False, True])
        .assign(radar_rank_desc=lambda frame: np.arange(1, len(frame) + 1))
        .set_index("model_id")["radar_rank_desc"]
        .reindex(radar_table["model_id"])
        .to_numpy()
    )
    return radar_table.sort_values(["radar_mean_across_axes", "axis_order"], ascending=[False, True]).reset_index(drop=True)


def _build_panel_source_table(
    model_summary: pd.DataFrame,
    *,
    model_summary_csv: Path,
    verdict: SourceVerdict,
    panel_name: str,
    axis_specs: list[tuple[str, str]],
    transform_rule: str,
    source_field_note: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    panel_order_lookup = {model_id: index for index, model_id in enumerate(PANEL_MODEL_ORDER, start=1)}
    for axis_order, (field_name, axis_label) in enumerate(axis_specs, start=1):
        if field_name not in model_summary.columns:
            raise ValueError(f"Missing panel source field: {field_name}")
        for row in model_summary.to_dict("records"):
            model_id = str(row["model_id"])
            if model_id not in panel_order_lookup:
                continue
            raw_value = pd.to_numeric(row.get(field_name), errors="coerce")
            if pd.isna(raw_value):
                continue
            value = float(raw_value)
            if transform_rule == "one_minus_distance":
                value = max(0.0, min(1.0, 1.0 - value))
            rows.append(
                {
                    "panel_name": panel_name,
                    "model_id": model_id,
                    "model_label": MODEL_LABELS.get(model_id, str(row.get("model_label", model_id))),
                    "model_color": MODEL_COLORS[model_id],
                    "panel_model_order": panel_order_lookup[model_id],
                    "axis_order": axis_order,
                    "axis_field": field_name,
                    "axis_label": axis_label,
                    "axis_value": value,
                    "source_value_raw": float(raw_value),
                    "value_transform": transform_rule,
                    "source_of_truth_model_summary_csv": str(model_summary_csv.resolve()),
                    "source_of_truth_line_version": verdict.current_result_line,
                    "source_of_truth_line_label": verdict.current_result_label,
                    "source_field_note": source_field_note,
                }
            )
    return pd.DataFrame(rows).sort_values(["axis_order", "panel_model_order"]).reset_index(drop=True)


def _build_query_vs_wasserstein_table(model_summary: pd.DataFrame) -> pd.DataFrame:
    table = model_summary[model_summary["row_kind"] == "synthetic"].copy()
    table = table[table["model_id"].isin(MODEL_COLORS)].copy()
    table["query_avg_100"] = pd.to_numeric(table["query_overall_mean"], errors="coerce") * 100.0
    table["wasserstein_100"] = (1.0 - pd.to_numeric(table["wasserstein_distance_mean"], errors="coerce")) * 100.0
    table["query_rank"] = table["query_avg_100"].rank(method="min", ascending=False).astype(int)
    table["wasserstein_rank"] = table["wasserstein_100"].rank(method="min", ascending=False).astype(int)
    table["model_color"] = table["model_id"].map(MODEL_COLORS)
    table["model_short_label"] = table["model_id"].map(MODEL_SHORT_LABELS).fillna(table["model_label"])
    return table[
        [
            "model_id",
            "model_label",
            "model_short_label",
            "model_color",
            "query_avg_100",
            "query_rank",
            "wasserstein_100",
            "wasserstein_rank",
        ]
    ].sort_values(["query_rank", "wasserstein_rank", "model_label"]).reset_index(drop=True)


def _build_query_vs_distance_avg_table(model_summary: pd.DataFrame) -> pd.DataFrame:
    table = model_summary[model_summary["row_kind"] == "synthetic"].copy()
    table = table[table["model_id"].isin(MODEL_COLORS)].copy()
    table["query_avg_100"] = pd.to_numeric(table["query_overall_mean"], errors="coerce") * 100.0
    table["distance_avg_100"] = pd.to_numeric(table["distance_overall_mean"], errors="coerce") * 100.0
    table["query_rank"] = table["query_avg_100"].rank(method="min", ascending=False).astype(int)
    table["distance_rank"] = table["distance_avg_100"].rank(method="min", ascending=False).astype(int)
    table["model_color"] = table["model_id"].map(MODEL_COLORS)
    table["model_short_label"] = table["model_id"].map(MODEL_SHORT_LABELS).fillna(table["model_label"])
    return table[
        [
            "model_id",
            "model_label",
            "model_short_label",
            "model_color",
            "query_avg_100",
            "query_rank",
            "distance_avg_100",
            "distance_rank",
        ]
    ].sort_values(["query_rank", "distance_rank", "model_label"]).reset_index(drop=True)


def _bar_chart_title(metric_key: str) -> str:
    if metric_key == "query_overall":
        return "Model Comparison on Query Overall"
    if metric_key == "distance_overall":
        return "Model Comparison on Distance Overall"
    return f"Model Comparison on {metric_key}"


def _plot_model_bar_chart(bar_table: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    plot_table = bar_table.sort_values(["metric_mean", "model_label"], ascending=[False, True]).reset_index(drop=True)
    metric_key = str(plot_table["metric_key"].iloc[0])
    fig, ax = plt.subplots(figsize=(10.4, 3.6), constrained_layout=True)
    x = np.arange(len(plot_table))
    ax.bar(
        x,
        plot_table["metric_mean"],
        color=plot_table["model_color"],
        edgecolor="#333333",
        linewidth=0.6,
        zorder=3,
    )
    ax.errorbar(
        x,
        plot_table["metric_mean"],
        yerr=plot_table["metric_ci95_radius"],
        fmt="none",
        ecolor="#222222",
        elinewidth=0.9,
        capsize=2.4,
        zorder=4,
    )
    ax.set_ylim(0.0, min(1.02, max(0.55, float(plot_table["metric_ci95_high"].max()) + 0.06)))
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_table["model_label"], rotation=30, ha="right")
    ax.set_title(_bar_chart_title(metric_key), fontsize=12, weight="bold")
    ax.grid(axis="y", color="#D9DEE6", linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for idx, value in enumerate(plot_table["metric_mean"]):
        ax.text(idx, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_model_radar_chart(radar_table: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    axis_labels = [label for _, label in RADAR_AXIS_ORDER]
    angles = np.linspace(0, 2 * math.pi, len(axis_labels), endpoint=False).tolist()
    angles += angles[:1]

    model_order = (
        radar_table[["model_id", "model_label", "model_color", "radar_mean_across_axes"]]
        .drop_duplicates()
        .sort_values(["radar_mean_across_axes", "model_label"], ascending=[False, True])
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.8), subplot_kw={"polar": True}, constrained_layout=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(color="#D9DEE6", linewidth=0.8)
    ax.spines["polar"].set_color("#AEB8C4")
    ax.set_title("All-Model Radar Across Six Formal Axes", fontsize=12, weight="bold", pad=18)

    for row in model_order.to_dict("records"):
        model_df = radar_table[radar_table["model_id"] == row["model_id"]].sort_values("axis_order")
        values = model_df["axis_value_mean"].tolist()
        values += values[:1]
        ax.plot(
            angles,
            values,
            color=row["model_color"],
            linewidth=1.8,
            label=row["model_label"],
        )
        ax.fill(angles, values, color=row["model_color"], alpha=0.05)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.4,
        columnspacing=1.2,
    )
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _start_panel_figure(width: float, height: float, border_color: str, show_frame: bool = True) -> tuple[plt.Figure, Any]:
    fig = plt.figure(figsize=(width, height), facecolor="none")
    panel_ax = fig.add_axes([0, 0, 1, 1])
    panel_ax.set_axis_off()
    if show_frame:
        panel = FancyBboxPatch(
            (0.012, 0.06),
            0.976,
            0.9,
            boxstyle="round,pad=0.008,rounding_size=0.03",
            linewidth=2.4,
            edgecolor=border_color,
            facecolor="white",
            transform=panel_ax.transAxes,
        )
        panel_ax.add_patch(panel)
    return fig, panel_ax


def _draw_panel_subplots(
    panel_table: pd.DataFrame,
    *,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path | None = None,
    panel_title: str | None,
    border_color: str,
    panel_ratio: float,
    subplot_count: int,
    subplot_left: float,
    subplot_width: float,
    subplot_gap: float,
    y_label: str | None = "Score (↑)",
    show_value_labels: bool = False,
    panel_title_fontsize: float | None = None,
    panel_note: str | None = None,
    panel_note_fontsize: float | None = None,
    subplot_title_fontsize: float = 10.5,
    ylabel_fontsize: float = 9.5,
    ytick_fontsize: float = 7.5,
    value_label_fontsize: float = 5.7,
    bar_width: float = 0.44,
    x_step: float = 1.0,
    x_margin: float = 0.55,
    ylabel_pad: float = 4.0,
    value_label_x_jitter: float = 0.045,
    value_label_y_stagger: float = 0.018,
    value_label_rotation: float = 0.0,
    value_label_scale: float = 1.0,
    value_label_decimals: int = 2,
) -> None:
    width = 12.4 if subplot_count >= 4 else 5.6
    height = width / panel_ratio
    fig, panel_ax = _start_panel_figure(width, height, border_color)
    if panel_title:
        panel_ax.text(
            0.5,
            0.92,
            panel_title,
            ha="center",
            va="center",
            fontsize=panel_title_fontsize if panel_title_fontsize is not None else (16 if subplot_count >= 4 else 14),
            fontweight="bold",
            color=border_color,
            transform=panel_ax.transAxes,
        )
    if panel_note:
        panel_ax.text(
            0.5,
            0.855,
            panel_note,
            ha="center",
            va="center",
            fontsize=panel_note_fontsize if panel_note_fontsize is not None else 8.0,
            color="#5A6170",
            fontstyle="italic",
            transform=panel_ax.transAxes,
        )

    for axis_index in range(1, subplot_count):
        separator_x = subplot_left + axis_index * subplot_width + (axis_index - 0.5) * subplot_gap
        panel_ax.plot(
            [separator_x, separator_x],
            [0.20, 0.78],
            color="#222222",
            linewidth=1.35,
            linestyle="-",
            alpha=0.72,
            transform=panel_ax.transAxes,
            zorder=1,
        )

    axis_pairs = list(panel_table[["axis_order", "axis_label"]].drop_duplicates().itertuples(index=False, name=None))
    for axis_order, axis_label in sorted(axis_pairs):
        axis_table = panel_table[panel_table["axis_order"] == axis_order].sort_values("panel_model_order")
        ax_left = subplot_left + (axis_order - 1) * (subplot_width + subplot_gap)
        ax = fig.add_axes([ax_left, 0.22, subplot_width, 0.50], facecolor="none")
        x = np.arange(len(axis_table), dtype=float) * x_step
        ax.bar(
            x,
            axis_table["axis_value"],
            color=axis_table["model_color"],
            edgecolor="#333333",
            linewidth=0.38,
            width=bar_width,
            zorder=3,
        )
        ax.set_xlim(float(x.min()) - x_margin, float(x.max()) + x_margin)
        ax.set_ylim(0.0, 1.08 if show_value_labels else 1.0)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_title(axis_label, fontsize=subplot_title_fontsize, fontweight="bold", color=border_color, pad=10)
        ax.grid(axis="y", color="#E2E5EC", linewidth=0.75, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9AA3B2")
        ax.spines["bottom"].set_color("#9AA3B2")
        ax.tick_params(axis="y", labelsize=ytick_fontsize, length=2)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        if axis_order == 1:
            if y_label:
                ax.set_ylabel(y_label, fontsize=ylabel_fontsize)
                ax.yaxis.labelpad = ylabel_pad
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
        if show_value_labels:
            for idx, value in enumerate(axis_table["axis_value"].tolist()):
                if value_label_rotation:
                    label_y = min(float(value) + 0.012, 1.05)
                    label_x = float(x[idx])
                else:
                    label_y = min(float(value) + 0.016 + (value_label_y_stagger if idx % 2 else 0.0), 1.05)
                    label_x = float(x[idx]) + (-value_label_x_jitter if idx % 2 == 0 else value_label_x_jitter)
                ax.text(
                    label_x,
                    label_y,
                    f"{float(value) * value_label_scale:.{value_label_decimals}f}",
                    ha="center",
                    va="bottom",
                    fontsize=value_label_fontsize,
                    color="#2E3947",
                    zorder=5,
                    rotation=value_label_rotation,
                    rotation_mode="anchor",
                )

    fig.savefig(output_png, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
    if output_svg is not None:
        fig.savefig(output_svg, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _plot_sql_family_panel(
    panel_table: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path | None = None,
) -> None:
    _draw_panel_subplots(
        panel_table,
        output_png=output_png,
        output_pdf=output_pdf,
        output_svg=output_svg,
        panel_title="Query-Grounded Structural Families",
        panel_note="SQL-side score ≠ (1 − distance)",
        border_color=SQL_PANEL_BORDER,
        panel_ratio=SQL_PANEL_RATIO,
        subplot_count=5,
        subplot_left=0.032,
        subplot_width=0.176,
        subplot_gap=0.012,
        show_value_labels=True,
        panel_title_fontsize=13.5,
        panel_note_fontsize=7.8,
        subplot_title_fontsize=9.4,
        ylabel_fontsize=8.8,
        ytick_fontsize=7.0,
        value_label_fontsize=6.4,
        x_margin=0.50,
        ylabel_pad=2.0,
        value_label_rotation=90.0,
        value_label_scale=100.0,
        value_label_decimals=0,
    )


def _plot_sql_family_panel_minimal(
    panel_table: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path | None = None,
) -> None:
    _draw_panel_subplots(
        panel_table,
        output_png=output_png,
        output_pdf=output_pdf,
        output_svg=output_svg,
        panel_title=None,
        panel_note=None,
        border_color=SQL_PANEL_BORDER,
        panel_ratio=SQL_PANEL_RATIO,
        subplot_count=5,
        subplot_left=0.032,
        subplot_width=0.176,
        subplot_gap=0.012,
        y_label=None,
        show_value_labels=True,
        subplot_title_fontsize=9.4,
        ylabel_fontsize=8.8,
        ytick_fontsize=7.0,
        value_label_fontsize=6.4,
        x_margin=0.50,
        ylabel_pad=2.0,
        value_label_rotation=0.0,
        value_label_scale=100.0,
        value_label_decimals=0,
    )


def _plot_distance_metric_panel(
    panel_table: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path | None = None,
) -> None:
    _draw_panel_subplots(
        panel_table,
        output_png=output_png,
        output_pdf=output_pdf,
        output_svg=output_svg,
        panel_title="Distance-Based Fidelity",
        panel_note="Fidelity score = 1 − distance ≠ SQL-side score",
        border_color=DISTANCE_PANEL_BORDER,
        panel_ratio=DISTANCE_PANEL_RATIO,
        subplot_count=4,
        subplot_left=0.037,
        subplot_width=0.228,
        subplot_gap=0.008,
        show_value_labels=True,
        panel_title_fontsize=20.5,
        panel_note_fontsize=11.7,
        subplot_title_fontsize=14.2,
        ylabel_fontsize=10.0,
        ytick_fontsize=10.5,
        value_label_fontsize=8.2,
        bar_width=0.62,
        x_step=1.32,
        x_margin=0.18,
        ylabel_pad=-5.5,
        value_label_x_jitter=0.06,
        value_label_y_stagger=0.02,
        value_label_rotation=90.0,
        value_label_scale=100.0,
        value_label_decimals=0,
    )


def _plot_distance_metric_panel_minimal(
    panel_table: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path | None = None,
) -> None:
    _draw_panel_subplots(
        panel_table,
        output_png=output_png,
        output_pdf=output_pdf,
        output_svg=output_svg,
        panel_title=None,
        panel_note=None,
        border_color=DISTANCE_PANEL_BORDER,
        panel_ratio=DISTANCE_PANEL_RATIO,
        subplot_count=4,
        subplot_left=0.037,
        subplot_width=0.228,
        subplot_gap=0.008,
        y_label=None,
        show_value_labels=True,
        subplot_title_fontsize=14.2,
        ylabel_fontsize=10.0,
        ytick_fontsize=10.5,
        value_label_fontsize=8.2,
        bar_width=0.62,
        x_step=1.32,
        x_margin=0.18,
        ylabel_pad=-5.5,
        value_label_x_jitter=0.06,
        value_label_y_stagger=0.02,
        value_label_rotation=0.0,
        value_label_scale=100.0,
        value_label_decimals=0,
    )


def _load_local_radar_panel_table() -> pd.DataFrame:
    from src.eval.model_radar.runner import run_model_radar

    manifest = run_model_radar()
    summary_csv = Path(str(manifest["png_path"])).with_name("model_radar_summary.csv")
    if not summary_csv.exists():
        summary_csv = PROJECT_ROOT / "Evaluation" / "model_radar" / "final" / "model_radar_summary.csv"
    radar = pd.read_csv(summary_csv)
    radar["model_id"] = radar["model_id"].astype(str).str.lower()
    radar = radar[radar["model_id"].isin(MODEL_COLORS)].copy()
    return radar


def _plot_radar_panel(output_png: Path, output_pdf: Path) -> None:
    _plot_radar_panel_variant(
        output_png=output_png,
        output_pdf=output_pdf,
        output_svg=None,
        show_title=True,
        show_axis_labels=True,
    )


def _plot_radar_panel_variant(
    *,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path | None,
    show_title: bool,
    show_axis_labels: bool,
    show_panel_frame: bool = True,
    axes_rect: tuple[float, float, float, float] = (0.04, 0.06, 0.92, 0.73),
) -> None:
    radar_summary = _load_local_radar_panel_table()
    width = 5.6
    height = width / RADAR_PANEL_RATIO
    fig, panel_ax = _start_panel_figure(width, height, RADAR_PANEL_BORDER, show_frame=show_panel_frame)
    if show_title:
        panel_ax.text(
            0.5,
            0.915,
            "Six-Axis Radar",
            ha="center",
            va="center",
            fontsize=15.2,
            fontweight="bold",
            color=RADAR_PANEL_BORDER,
            transform=panel_ax.transAxes,
        )

    axis_labels = [label for _, label in RADAR_AXIS_ORDER]
    angles = np.linspace(0, 2 * math.pi, len(axis_labels), endpoint=False).tolist()
    angles += angles[:1]
    ax = fig.add_axes(list(axes_rect), projection="polar", facecolor="none")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.xaxis.grid(True, color="#D9DEE6", linewidth=0.8, linestyle="-", alpha=0.95)
    ax.yaxis.grid(True, color="#D9DEE6", linewidth=0.9, linestyle="-", alpha=0.95)
    ax.spines["polar"].set_color("#AEB8C4")
    for idx, gridline in enumerate(ax.yaxis.get_gridlines()):
        if idx % 2 == 0:
            gridline.set_linestyle("-")
            gridline.set_linewidth(0.95)
            gridline.set_alpha(0.95)
        else:
            gridline.set_linestyle((0, (2.2, 2.2)))
            gridline.set_linewidth(0.9)
            gridline.set_alpha(0.95)

    for angle, label in zip(angles[:-1], axis_labels):
        if show_axis_labels:
            radius = RADAR_PANEL_LABEL_RADIUS.get(label, 1.28)
            ax.text(
                angle,
                radius,
                label,
                fontsize=9.5 if label == "Distance" else 10.1,
                color="#000000",
                ha="center",
                va="center",
                clip_on=False,
                zorder=18,
            )

    plotted = radar_summary.sort_values(["radar_score_mean", "model_label"], ascending=[False, True])
    for row in plotted.to_dict("records"):
        values = [
            float(row["distance_score"]),
            float(row["subgroup_score"]),
            float(row["conditional_score"]),
            float(row["tail_score"]),
            float(row["missingness_score"]),
            float(row["cardinality_score"]),
        ]
        values += values[:1]
        ax.plot(angles, values, color=row["model_color"], linewidth=1.35, alpha=0.96)

    radial_label_angles = [
        math.radians(30),
        math.radians(150),
        math.radians(270),
    ]
    for label_angle in radial_label_angles:
        for radius, label_text in zip([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"]):
            ax.text(
                label_angle,
                radius,
                label_text,
                fontsize=8.8,
                fontweight="bold",
                color="#2E3947",
                ha="center",
                va="center",
                zorder=20,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white", alpha=0.95)],
            )
    fig.savefig(output_png, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
    if output_svg is not None:
        fig.savefig(output_svg, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _resize_to_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    target_width = right - left
    target_height = bottom - top
    if "A" in image.getbands():
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is not None:
            image = image.crop(bbox)
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _build_combined_preview(
    *,
    sql_panel_png: Path,
    distance_panel_png: Path,
    radar_panel_png: Path,
    output_png: Path,
    output_pdf: Path,
) -> None:
    canvas = Image.new("RGBA", COMBINED_PREVIEW_SIZE, (255, 255, 255, 0))
    sql_img = _resize_to_box(Image.open(sql_panel_png).convert("RGBA"), COMBINED_SQL_BOX)
    distance_img = _resize_to_box(Image.open(distance_panel_png).convert("RGBA"), COMBINED_DISTANCE_BOX)
    radar_img = _resize_to_box(Image.open(radar_panel_png).convert("RGBA"), COMBINED_RADAR_BOX)
    canvas.alpha_composite(sql_img, (COMBINED_SQL_BOX[0], COMBINED_SQL_BOX[1]))
    canvas.alpha_composite(distance_img, (COMBINED_DISTANCE_BOX[0], COMBINED_DISTANCE_BOX[1]))
    canvas.alpha_composite(radar_img, (COMBINED_RADAR_BOX[0], COMBINED_RADAR_BOX[1]))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)
    canvas.convert("RGB").save(output_pdf, "PDF", resolution=300.0)


def _png_to_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_full_overview_preview_and_svg(
    *,
    background_png: Path,
    sql_panel_png: Path,
    distance_panel_png: Path,
    radar_panel_png: Path,
    output_preview_png: Path,
    output_svg: Path,
    output_layout_json: Path,
) -> None:
    bg = Image.open(background_png).convert("RGBA")
    if bg.size != FULL_OVERVIEW_SIZE:
        bg = bg.resize(FULL_OVERVIEW_SIZE, Image.Resampling.LANCZOS)

    sql_img = _resize_to_box(Image.open(sql_panel_png).convert("RGBA"), FULL_OVERVIEW_SQL_BOX)
    distance_img = _resize_to_box(Image.open(distance_panel_png).convert("RGBA"), FULL_OVERVIEW_DISTANCE_BOX)
    radar_img = _resize_to_box(Image.open(radar_panel_png).convert("RGBA"), FULL_OVERVIEW_RADAR_BOX)

    bg.alpha_composite(sql_img, (FULL_OVERVIEW_SQL_BOX[0], FULL_OVERVIEW_SQL_BOX[1]))
    bg.alpha_composite(distance_img, (FULL_OVERVIEW_DISTANCE_BOX[0], FULL_OVERVIEW_DISTANCE_BOX[1]))
    bg.alpha_composite(radar_img, (FULL_OVERVIEW_RADAR_BOX[0], FULL_OVERVIEW_RADAR_BOX[1]))
    output_preview_png.parent.mkdir(parents=True, exist_ok=True)
    bg.save(output_preview_png)

    bg_uri = _png_to_data_uri(background_png)
    sql_uri = _png_to_data_uri(sql_panel_png)
    distance_uri = _png_to_data_uri(distance_panel_png)
    radar_uri = _png_to_data_uri(radar_panel_png)

    def _image_tag(uri: str, box: tuple[int, int, int, int]) -> str:
        left, top, right, bottom = box
        width = right - left
        height = bottom - top
        return f'<image x="{left}" y="{top}" width="{width}" height="{height}" href="{uri}" preserveAspectRatio="none" />'

    svg = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'width="{FULL_OVERVIEW_SIZE[0]}" height="{FULL_OVERVIEW_SIZE[1]}" '
                f'viewBox="0 0 {FULL_OVERVIEW_SIZE[0]} {FULL_OVERVIEW_SIZE[1]}">'
            ),
            f'<rect x="0" y="0" width="{FULL_OVERVIEW_SIZE[0]}" height="{FULL_OVERVIEW_SIZE[1]}" fill="white" />',
            (
                f'<image x="0" y="0" width="{FULL_OVERVIEW_SIZE[0]}" height="{FULL_OVERVIEW_SIZE[1]}" '
                f'href="{bg_uri}" preserveAspectRatio="none" />'
            ),
            _image_tag(sql_uri, FULL_OVERVIEW_SQL_BOX),
            _image_tag(distance_uri, FULL_OVERVIEW_DISTANCE_BOX),
            _image_tag(radar_uri, FULL_OVERVIEW_RADAR_BOX),
            "</svg>",
        ]
    )
    output_svg.write_text(svg, encoding="utf-8")

    layout_payload = {
        "canvas_size": {"width": FULL_OVERVIEW_SIZE[0], "height": FULL_OVERVIEW_SIZE[1]},
        "background_png": str(background_png.resolve()),
        "sql_panel_png": str(sql_panel_png.resolve()),
        "distance_panel_png": str(distance_panel_png.resolve()),
        "radar_panel_png": str(radar_panel_png.resolve()),
        "sql_box": {"left": FULL_OVERVIEW_SQL_BOX[0], "top": FULL_OVERVIEW_SQL_BOX[1], "right": FULL_OVERVIEW_SQL_BOX[2], "bottom": FULL_OVERVIEW_SQL_BOX[3]},
        "distance_box": {"left": FULL_OVERVIEW_DISTANCE_BOX[0], "top": FULL_OVERVIEW_DISTANCE_BOX[1], "right": FULL_OVERVIEW_DISTANCE_BOX[2], "bottom": FULL_OVERVIEW_DISTANCE_BOX[3]},
        "radar_box": {"left": FULL_OVERVIEW_RADAR_BOX[0], "top": FULL_OVERVIEW_RADAR_BOX[1], "right": FULL_OVERVIEW_RADAR_BOX[2], "bottom": FULL_OVERVIEW_RADAR_BOX[3]},
        "note": "Self-contained SVG wrapper built from the current overview7 background plus regenerated overlay panels for Figma adjustment.",
    }
    output_layout_json.write_text(json.dumps(layout_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_svg_inner(svg_path: Path) -> tuple[str, str]:
    text = svg_path.read_text(encoding="utf-8")
    start = text.find("<svg")
    if start < 0:
        raise ValueError(f"Not a valid SVG file: {svg_path}")
    start_tag_end = text.find(">", start)
    end = text.rfind("</svg>")
    if start_tag_end < 0 or end < 0:
        raise ValueError(f"Malformed SVG root: {svg_path}")
    start_tag = text[start : start_tag_end + 1]
    inner = text[start_tag_end + 1 : end]
    viewbox = "0 0 100 100"
    for marker in ['viewBox="', "viewBox='"]:
        idx = start_tag.find(marker)
        if idx >= 0:
            vb_start = idx + len(marker)
            vb_end = start_tag.find(marker[-1], vb_start)
            viewbox = start_tag[vb_start:vb_end]
            break
    return viewbox, inner


def _svg_panel_tag(svg_path: Path, box: tuple[int, int, int, int], group_id: str) -> str:
    viewbox, inner = _read_svg_inner(svg_path)
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    return (
        f'<svg id="{group_id}" x="{left}" y="{top}" width="{width}" height="{height}" '
        f'viewBox="{viewbox}" preserveAspectRatio="none">{inner}</svg>'
    )


def _build_editable_overview_svg(
    *,
    sql_panel_svg: Path,
    distance_panel_svg: Path,
    radar_panel_svg: Path,
    output_svg: Path,
    include_figure_title: bool = True,
    include_top_workflow: bool = True,
    include_legend: bool = True,
) -> None:
    model_labels = [MODEL_LABELS[model_id] for model_id in PANEL_MODEL_ORDER]
    model_colors = [MODEL_COLORS[model_id] for model_id in PANEL_MODEL_ORDER]

    def rect(x: float, y: float, w: float, h: float, rx: float, stroke: str, fill: str = "#ffffff", sw: float = 2.5) -> str:
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'

    def text(x: float, y: float, value: str, size: float, fill: str = "#111111", weight: str = "400", anchor: str = "middle", italic: bool = False) -> str:
        style = 'font-family="Arial, DejaVu Sans, sans-serif"'
        font_style = ' font-style="italic"' if italic else ""
        return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}" {style}{font_style}>{value}</text>'

    pieces: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{FULL_OVERVIEW_SIZE[0]}" height="{FULL_OVERVIEW_SIZE[1]}" viewBox="0 0 {FULL_OVERVIEW_SIZE[0]} {FULL_OVERVIEW_SIZE[1]}">',
        f'<rect x="0" y="0" width="{FULL_OVERVIEW_SIZE[0]}" height="{FULL_OVERVIEW_SIZE[1]}" fill="#ffffff" />',
    ]
    if include_figure_title:
        pieces.append(text(724, 34, "TabQueryBench: Query-Grounded Evaluation of Synthetic Tabular Data", 31, weight="700"))

    # Top workflow row
    top_boxes = [
        (104, 50, 332, 166, "#182589", "1", "#182589", "Workload Sources", "Benchmark Queries", "SQL Repos", "Docs"),
        (524, 50, 328, 166, "#0d6b13", "2", "#0d6b13", "Template Library", "Reusable Templates", "Subgroup", "Conditional", "Tail"),
        (938, 50, 357, 166, "#ff2a10", "3", "#ff2a10", "Dataset Grounding", "Profile", "Select", "Generate SQL"),
    ]
    if include_top_workflow:
        for x, y, w, h, stroke, num, num_fill, title_str, *rest in top_boxes:
            pieces.append(rect(x, y, w, h, 18, stroke, sw=2.0))
            pieces.append(f'<circle cx="{x-2}" cy="{y+10}" r="18" fill="{num_fill}" />')
            pieces.append(text(x - 2, y + 17, num, 16, fill="#ffffff", weight="700"))
            pieces.append(text(x + w / 2, y + 27, title_str, 24, fill=stroke, weight="700"))
            if title_str == "Workload Sources":
                pieces.append(text(x + 46, y + 58, rest[0], 14, anchor="start"))
                pieces.append(text(x + 163, y + 58, rest[1], 14, anchor="start"))
                pieces.append(text(x + 264, y + 58, rest[2], 14, anchor="start"))
                pieces.extend(
                    [
                        f'<line x1="{x+117}" y1="{y+78}" x2="{x+117}" y2="{y+148}" stroke="#c6cbd7" stroke-dasharray="3,3" />',
                        f'<line x1="{x+226}" y1="{y+78}" x2="{x+226}" y2="{y+148}" stroke="#c6cbd7" stroke-dasharray="3,3" />',
                        f'<rect x="{x+28}" y="{y+88}" width="34" height="52" rx="4" fill="none" stroke="{stroke}" stroke-width="2"/>',
                        f'<line x1="{x+36}" y1="{y+102}" x2="{x+54}" y2="{y+102}" stroke="{stroke}" stroke-width="2"/>',
                        f'<line x1="{x+36}" y1="{y+114}" x2="{x+54}" y2="{y+114}" stroke="{stroke}" stroke-width="2"/>',
                        f'<line x1="{x+36}" y1="{y+126}" x2="{x+50}" y2="{y+126}" stroke="{stroke}" stroke-width="2"/>',
                        f'<circle cx="{x+78}" cy="{y+127}" r="10" fill="none" stroke="#5d64d7" stroke-width="2"/>',
                        f'<line x1="{x+84}" y1="{y+133}" x2="{x+94}" y2="{y+144}" stroke="#5d64d7" stroke-width="3"/>',
                        f'<ellipse cx="{x+170}" cy="{y+100}" rx="25" ry="10" fill="none" stroke="{stroke}" stroke-width="2"/>',
                        f'<ellipse cx="{x+170}" cy="{y+136}" rx="25" ry="10" fill="none" stroke="{stroke}" stroke-width="2"/>',
                        f'<line x1="{x+145}" y1="{y+100}" x2="{x+145}" y2="{y+136}" stroke="{stroke}" stroke-width="2"/>',
                        f'<line x1="{x+195}" y1="{y+100}" x2="{x+195}" y2="{y+136}" stroke="{stroke}" stroke-width="2"/>',
                        f'<path d="M {x+250} {y+97} q 15 -8 30 0 v 44 q -15 -8 -30 0 z" fill="none" stroke="{stroke}" stroke-width="2"/>',
                        f'<path d="M {x+310} {y+97} q -15 -8 -30 0 v 44 q 15 -8 30 0 z" fill="none" stroke="{stroke}" stroke-width="2"/>',
                    ]
                )
            elif title_str == "Template Library":
                pieces.append(text(x + w / 2, y + 58, rest[0], 16))
                pieces.append(f'<rect x="{x+18}" y="{y+85}" width="{w-36}" height="66" rx="16" fill="none" stroke="#888888" stroke-dasharray="5,4" stroke-width="2"/>')
                chip_specs = [
                    (x + 26, y + 102, 94, 34, "#d9f0c2", "#4e7c2f", rest[1]),
                    (x + 132, y + 102, 107, 34, "#d6e9ff", "#3f76c5", rest[2]),
                    (x + 250, y + 102, 73, 34, "#ffe3b5", "#f08b28", rest[3]),
                ]
                for cx, cy, cw, ch, fill, stroke, label in chip_specs:
                    pieces.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
                    pieces.append(text(cx + cw / 2, cy + 23, label, 14, fill="#1a1a1a"))
            else:
                pieces.append(text(x + 63, y + 58, rest[0], 14))
                pieces.append(text(x + 175, y + 58, rest[1], 14))
                pieces.append(text(x + 286, y + 58, rest[2], 14))
                pieces.extend(
                    [
                        f'<line x1="{x+110}" y1="{y+111}" x2="{x+134}" y2="{y+111}" stroke="#666666" stroke-width="2" marker-end="url(#arrowHead)"/>',
                        f'<line x1="{x+219}" y1="{y+111}" x2="{x+243}" y2="{y+111}" stroke="#666666" stroke-width="2" marker-end="url(#arrowHead)"/>',
                        f'<line x1="{x+48}" y1="{y+126}" x2="{x+78}" y2="{y+126}" stroke="{stroke}" stroke-width="4"/>',
                        f'<line x1="{x+55}" y1="{y+96}" x2="{x+55}" y2="{y+126}" stroke="{stroke}" stroke-width="4"/>',
                        f'<line x1="{x+66}" y1="{y+104}" x2="{x+66}" y2="{y+126}" stroke="{stroke}" stroke-width="4"/>',
                        f'<line x1="{x+77}" y1="{y+90}" x2="{x+77}" y2="{y+126}" stroke="{stroke}" stroke-width="4"/>',
                        f'<circle cx="{x+73}" cy="{y+124}" r="14" fill="none" stroke="{stroke}" stroke-width="2"/>',
                        f'<line x1="{x+82}" y1="{y+133}" x2="{x+94}" y2="{y+145}" stroke="{stroke}" stroke-width="3"/>',
                        f'<path d="M {x+147} {y+91} h 38 l -9 22 v 23 h -20 v -23 z" fill="none" stroke="#222222" stroke-width="2"/>',
                        f'<circle cx="{x+197}" cy="{y+95}" r="7" fill="{stroke}"/><circle cx="{x+197}" cy="{y+120}" r="7" fill="{stroke}"/><circle cx="{x+197}" cy="{y+145}" r="7" fill="{stroke}"/>',
                        f'<path d="M {x+280} {y+90} h 28 l 16 16 v 56 h -44 z" fill="none" stroke="{stroke}" stroke-width="2"/>',
                        f'<line x1="{x+288}" y1="{y+106}" x2="{x+314}" y2="{y+106}" stroke="{stroke}" stroke-width="2"/>',
                        f'<line x1="{x+288}" y1="{y+118}" x2="{x+314}" y2="{y+118}" stroke="{stroke}" stroke-width="2"/>',
                        text(x + 302, y + 142, "SQL", 20, fill=stroke, weight="700"),
                    ]
                )

        pieces.insert(
            2,
            '<defs><marker id="arrowHead" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 z" fill="#666666"/></marker></defs>',
        )
        pieces.extend(
            [
                '<line x1="438" y1="112" x2="500" y2="112" stroke="#666666" stroke-width="4" marker-end="url(#arrowHead)"/>',
                '<line x1="855" y1="112" x2="913" y2="112" stroke="#666666" stroke-width="4" marker-end="url(#arrowHead)"/>',
                '<path d="M 257 218 H 1118 Q 1130 218 1130 206 V 200" fill="none" stroke="#555555" stroke-width="2.2" stroke-dasharray="5,4"/>',
                '<path d="M 690 217 V 241" fill="none" stroke="#182589" stroke-width="3" marker-end="url(#arrowHead)"/>',
            ]
        )

    # Panels as nested editable SVG blocks
    pieces.append(_svg_panel_tag(sql_panel_svg, FULL_OVERVIEW_SQL_BOX, "sql-panel-vector"))
    pieces.append(_svg_panel_tag(distance_panel_svg, FULL_OVERVIEW_DISTANCE_BOX, "distance-panel-vector"))
    pieces.append(rect(FULL_OVERVIEW_RADAR_BOX[0], FULL_OVERVIEW_RADAR_BOX[1], FULL_OVERVIEW_RADAR_BOX[2] - FULL_OVERVIEW_RADAR_BOX[0], FULL_OVERVIEW_RADAR_BOX[3] - FULL_OVERVIEW_RADAR_BOX[1], 18, RADAR_PANEL_BORDER, sw=2.4))
    radar_inner_box = (
        FULL_OVERVIEW_RADAR_BOX[0] + 18,
        FULL_OVERVIEW_RADAR_BOX[1] + 18,
        FULL_OVERVIEW_RADAR_BOX[2] - 18,
        FULL_OVERVIEW_RADAR_BOX[3] - 18,
    )
    pieces.append(_svg_panel_tag(radar_panel_svg, radar_inner_box, "radar-panel-vector"))

    # Legend
    if include_legend:
        legend_x, legend_y, legend_w, legend_h = 17, 998, 1415, 66
        pieces.append(rect(legend_x, legend_y, legend_w, legend_h, 10, "#1b2c77", sw=1.8))
        start_x = 45
        y_mid = legend_y + 33
        step = 126
        for idx, (label, color) in enumerate(zip(model_labels, model_colors)):
            lx = start_x + idx * step
            pieces.append(f'<rect x="{lx}" y="{y_mid-10}" width="18" height="18" fill="{color}" stroke="#333333" stroke-width="0.6" />')
            pieces.append(text(lx + 32, y_mid + 4, label, 16, anchor="start", weight="600"))

    pieces.append("</svg>")
    output_svg.write_text("\n".join(pieces), encoding="utf-8")


def _build_rank_compare_svg(
    comparison_table: pd.DataFrame,
    *,
    bottom_score_col: str,
    bottom_rank_col: str,
    bottom_model_id_col: str,
    bottom_model_label_col: str,
    bottom_model_short_label_col: str,
    bottom_color_col: str,
    title: str,
    subtitle: str,
    bottom_title: str,
    bottom_chip: str,
    bottom_accent: str,
    output_svg: Path,
    output_csv: Path,
) -> None:
    query_top = comparison_table.sort_values(["query_rank", "query_model_label"]).head(5).copy()
    bottom_top = comparison_table.sort_values([bottom_rank_col, bottom_model_label_col]).head(5).copy()

    rows: list[dict[str, Any]] = []
    for rank in range(1, 6):
        q = query_top[query_top["query_rank"] == rank].iloc[0]
        b = bottom_top[bottom_top[bottom_rank_col] == rank].iloc[0]
        rows.append(
            {
                "rank": rank,
                "query_model_id": q["query_model_id"],
                "query_model_label": q["query_model_label"],
                "query_model_short_label": q["query_model_short_label"],
                "query_score_100": float(q["query_avg_100"]),
                "query_color": q["query_color"],
                "bottom_model_id": b[bottom_model_id_col],
                "bottom_model_label": b[bottom_model_label_col],
                "bottom_model_short_label": b[bottom_model_short_label_col],
                "bottom_score_100": float(b[bottom_score_col]),
                "bottom_color": b[bottom_color_col],
                "same_model_at_rank": bool(q["query_model_id"] == b[bottom_model_id_col]),
            }
        )
    out_df = pd.DataFrame(rows)
    output_csv.write_text(out_df.to_csv(index=False), encoding="utf-8")

    W, H = 1540, 560
    left_margin = 210
    top_row_y = 140
    bottom_row_y = 340
    card_w = 210
    card_h = 108
    gap = 36
    start_x = left_margin
    title_y = 42

    def esc(value: str) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def text(x: float, y: float, value: str, size: float, *, fill: str = "#111111", weight: str = "400", anchor: str = "middle", italic: bool = False) -> str:
        font_style = ' font-style="italic"' if italic else ""
        return (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" font-family="Arial, DejaVu Sans, sans-serif"{font_style}>'
            f"{esc(value)}</text>"
        )

    def round_rect(x: float, y: float, w: float, h: float, rx: float, *, fill: str = "#ffffff", stroke: str = "#333333", sw: float = 2.0) -> str:
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'

    def rank_badge(rank: int, cx: float, cy: float) -> str:
        if rank == 1:
            fill = "#f4c430"
            label = "1"
        elif rank == 2:
            fill = "#c0c6d4"
            label = "2"
        elif rank == 3:
            fill = "#d69456"
            label = "3"
        else:
            fill = "#ffffff"
            label = str(rank)
        badge = [
            f'<circle cx="{cx}" cy="{cy}" r="20" fill="{fill}" stroke="#2f3747" stroke-width="2" />',
            text(cx, cy + 6, label, 18, fill="#1d2432", weight="700"),
        ]
        if rank <= 3:
            badge.append(f'<path d="M {cx-10} {cy-25} L {cx-4} {cy-34} L {cx} {cy-27} L {cx+4} {cy-34} L {cx+10} {cy-25} Z" fill="{fill}" stroke="#2f3747" stroke-width="1.5" />')
        return "".join(badge)

    def query_icon(x: float, y: float) -> str:
        return "".join(
            [
                f'<rect x="{x}" y="{y}" width="26" height="34" rx="4" fill="none" stroke="#5140c8" stroke-width="2"/>',
                f'<line x1="{x+6}" y1="{y+10}" x2="{x+18}" y2="{y+10}" stroke="#5140c8" stroke-width="2"/>',
                f'<line x1="{x+6}" y1="{y+18}" x2="{x+18}" y2="{y+18}" stroke="#5140c8" stroke-width="2"/>',
                f'<circle cx="{x+30}" cy="{y+24}" r="8" fill="none" stroke="#5140c8" stroke-width="2"/>',
                f'<line x1="{x+36}" y1="{y+30}" x2="{x+44}" y2="{y+38}" stroke="#5140c8" stroke-width="3"/>',
            ]
        )

    def water_icon(x: float, y: float) -> str:
        return "".join(
            [
                f'<path d="M {x+14} {y} C {x+22} {y+12}, {x+28} {y+18}, {x+28} {y+28} C {x+28} {y+40}, {x+22} {y+48}, {x+14} {y+48} C {x+6} {y+48}, {x} {y+40}, {x} {y+28} C {x} {y+18}, {x+6} {y+12}, {x+14} {y} Z" fill="#ff661f" fill-opacity="0.15" stroke="#ff661f" stroke-width="2"/>',
                f'<path d="M {x+34} {y+38} q 10 -8 20 0 q 10 8 20 0" fill="none" stroke="#ff661f" stroke-width="2.5"/>',
            ]
        )

    def metric_chip(x: float, y: float, label: str, stroke: str, fill: str) -> str:
        return "".join(
            [
                f'<rect x="{x}" y="{y}" width="144" height="32" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>',
                text(x + 72, y + 22, label, 16, fill=stroke, weight="700"),
            ]
        )

    def model_card(x: float, y: float, rank: int, short_label: str, score: float, color: str, accent: str) -> str:
        return "".join(
            [
                round_rect(x, y, card_w, card_h, 18, fill="#ffffff", stroke=accent, sw=2.2),
                rank_badge(rank, x + 26, y + 28),
                f'<rect x="{x+56}" y="{y+18}" width="132" height="30" rx="15" fill="{color}" fill-opacity="0.16" stroke="{color}" stroke-width="1.5"/>',
                text(x + 122, y + 39, short_label, 17, fill="#1f2430", weight="700"),
                text(x + 105, y + 73, f"score {int(round(score))}", 14, fill="#6a7282", weight="600"),
                text(x + 105, y + 95, f"Rank {rank}", 13, fill=accent, weight="700"),
            ]
        )

    pieces = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff" />',
        f'<rect x="18" y="18" width="{W-36}" height="{H-36}" rx="26" fill="#fffdfb" stroke="#e9e5df" stroke-width="1.5" />',
        text(W / 2, title_y, title, 28, fill="#202532", weight="700"),
        text(W / 2, 70, subtitle, 16, fill="#697180", weight="500"),
        query_icon(48, 120),
        text(106, 145, "Query Average", 24, fill="#5140c8", weight="700", anchor="start"),
        metric_chip(44, 160, "Query Avg", "#5140c8", "#f3f0ff"),
        water_icon(42, 320),
        text(106, 350, bottom_title, 24, fill=bottom_accent, weight="700", anchor="start"),
        metric_chip(44, 366, bottom_chip, bottom_accent, "#fff2ea"),
    ]

    for row in rows:
        rank = int(row["rank"])
        x = start_x + (rank - 1) * (card_w + gap)
        pieces.append(
            model_card(
                x,
                top_row_y,
                rank,
                str(row["query_model_short_label"]),
                float(row["query_score_100"]),
                str(row["query_color"]),
                "#5140c8",
            )
        )
        pieces.append(
            model_card(
                x,
                bottom_row_y,
                rank,
                str(row["bottom_model_short_label"]),
                float(row["bottom_score_100"]),
                str(row["bottom_color"]),
                bottom_accent,
            )
        )
        symbol = "=" if bool(row["same_model_at_rank"]) else "&#8800;"
        symbol_fill = "#2d8a54" if bool(row["same_model_at_rank"]) else "#d94841"
        pieces.append(f'<circle cx="{x + card_w/2}" cy="{(top_row_y + bottom_row_y + card_h) / 2}" r="20" fill="{symbol_fill}" fill-opacity="0.12" stroke="{symbol_fill}" stroke-width="1.5" />')
        pieces.append(
            f'<text x="{x + card_w / 2}" y="{(top_row_y + bottom_row_y + card_h) / 2 + 7}" '
            f'text-anchor="middle" font-size="24" font-weight="700" fill="{symbol_fill}" '
            f'font-family="Arial, DejaVu Sans, sans-serif">{symbol}</text>'
        )
        pieces.append(f'<line x1="{x + card_w/2}" y1="{top_row_y + card_h + 10}" x2="{x + card_w/2}" y2="{bottom_row_y - 12}" stroke="{symbol_fill}" stroke-width="2.2" stroke-dasharray="6,6" stroke-opacity="0.55" />')

    pieces.extend(
        [
            text(1318, 92, "Same model at same rank", 15, fill="#5f6777", weight="700", anchor="start"),
            f'<circle cx="1288" cy="86" r="12" fill="#2d8a54" fill-opacity="0.12" stroke="#2d8a54" stroke-width="1.5" />',
            text(1288, 92, "=", 16, fill="#2d8a54", weight="700"),
            text(1318, 122, "Different model at same rank", 15, fill="#5f6777", weight="700", anchor="start"),
            f'<circle cx="1288" cy="116" r="12" fill="#d94841" fill-opacity="0.12" stroke="#d94841" stroke-width="1.5" />',
            f'<text x="1288" y="122" text-anchor="middle" font-size="16" font-weight="700" fill="#d94841" font-family="Arial, DejaVu Sans, sans-serif">&#8800;</text>',
        ]
    )

    pieces.append("</svg>")
    output_svg.write_text("\n".join(pieces), encoding="utf-8")


def _build_query_vs_wasserstein_svg(
    comparison_table: pd.DataFrame,
    *,
    output_svg: Path,
    output_csv: Path,
) -> None:
    table = pd.DataFrame(
        {
            "query_model_id": comparison_table["model_id"],
            "query_model_label": comparison_table["model_label"],
            "query_model_short_label": comparison_table["model_short_label"],
            "query_color": comparison_table["model_color"],
            "query_avg_100": comparison_table["query_avg_100"],
            "query_rank": comparison_table["query_rank"],
            "bottom_model_id": comparison_table["model_id"],
            "bottom_model_label": comparison_table["model_label"],
            "bottom_model_short_label": comparison_table["model_short_label"],
            "bottom_color": comparison_table["model_color"],
            "bottom_score_100": comparison_table["wasserstein_100"],
            "bottom_rank": comparison_table["wasserstein_rank"],
        }
    )
    _build_rank_compare_svg(
        table,
        bottom_score_col="bottom_score_100",
        bottom_rank_col="bottom_rank",
        bottom_model_id_col="bottom_model_id",
        bottom_model_label_col="bottom_model_label",
        bottom_model_short_label_col="bottom_model_short_label",
        bottom_color_col="bottom_color",
        title="Query Average vs Wasserstein Fidelity Ranking",
        subtitle="Top-5 side-by-side alignment shows the ranking order is not the same",
        bottom_title="Wasserstein Fidelity",
        bottom_chip="1 - Wasserstein",
        bottom_accent="#ff661f",
        output_svg=output_svg,
        output_csv=output_csv,
    )


def _build_query_vs_distance_avg_svg(
    comparison_table: pd.DataFrame,
    *,
    output_svg: Path,
    output_csv: Path,
) -> None:
    table = pd.DataFrame(
        {
            "query_model_id": comparison_table["model_id"],
            "query_model_label": comparison_table["model_label"],
            "query_model_short_label": comparison_table["model_short_label"],
            "query_color": comparison_table["model_color"],
            "query_avg_100": comparison_table["query_avg_100"],
            "query_rank": comparison_table["query_rank"],
            "bottom_model_id": comparison_table["model_id"],
            "bottom_model_label": comparison_table["model_label"],
            "bottom_model_short_label": comparison_table["model_short_label"],
            "bottom_color": comparison_table["model_color"],
            "bottom_score_100": comparison_table["distance_avg_100"],
            "bottom_rank": comparison_table["distance_rank"],
        }
    )
    _build_rank_compare_svg(
        table,
        bottom_score_col="bottom_score_100",
        bottom_rank_col="bottom_rank",
        bottom_model_id_col="bottom_model_id",
        bottom_model_label_col="bottom_model_label",
        bottom_model_short_label_col="bottom_model_short_label",
        bottom_color_col="bottom_color",
        title="Query Average vs Distance Average Ranking",
        subtitle="Top-5 side-by-side alignment compares the formal SQL score with the distance-average score",
        bottom_title="Distance Average",
        bottom_chip="Distance Avg",
        bottom_accent="#ff661f",
        output_svg=output_svg,
        output_csv=output_csv,
    )


def _write_summary_markdown(
    path: Path,
    *,
    verdict: SourceVerdict,
    bar_table: pd.DataFrame,
    distance_bar_table: pd.DataFrame,
    radar_table: pd.DataFrame,
    bar_metric_key: str,
    bar_metric_label: str,
    source_manifest: pd.DataFrame,
) -> None:
    source_lookup = _source_file_lookup(source_manifest)
    radar_axis_lines = [
        f"  - {label}: `{source_lookup.get(metric_key, 'see benchmark overall dataset-level source')}`"
        for metric_key, label in RADAR_AXIS_ORDER
    ]
    lines = [
        "# Overview Regenerated Summary",
        "",
        "## Paper-facing purpose",
        "",
        "- `overview_model_bar_chart.*`: compact ranking chart for model-vs-model comparison on the query-overall summary indicator.",
        "- `overview_model_bar_chart_distance_overall.*`: companion ranking chart for the distance-overall summary indicator.",
        "- `overview_model_radar_chart.*`: all-model six-axis profile chart for the overview panel.",
        "- `overview_sql_panel.*`: Figma-overlay SQL-side panel split into five structural family subplots.",
        "- `overview_distance_panel.*`: Figma-overlay distribution panel split into JSD / KSD / TVD / Wasserstein subplots.",
        "- `overview_radar_panel.*`: compact radar panel rendered from the existing local `model_radar` result line.",
        "- `overview_panels_combined_preview.*`: merged preview laid out at the current overview-composition ratio for font-size QA before Figma replacement.",
        "- `overview_full_figure.svg`: self-contained paper-overview SVG assembled from the current `overview7` background and the regenerated overlay panels.",
        "",
        "## Source-of-truth decision",
        "",
        f"- Current result line used: `{verdict.current_result_label}` (`{verdict.current_result_line}`).",
        f"- Decision note: {verdict.current_result_note}",
        f"- Consolidated benchmark dataset-level source: `{verdict.dataset_level_csv}`",
        f"- Consolidated metric manifest source: `{verdict.source_manifest_csv}`",
    ]
    if verdict.radar_manifest_path is not None:
        lines.append(f"- Existing paper-facing radar manifest checked: `{verdict.radar_manifest_path}`")
    lines.extend(
        [
            "",
            "## Bar chart spec",
            "",
            f"- Selected summary indicator: `{bar_metric_key}` ({bar_metric_label}).",
            "- Field used from the consolidated dataset-level table: `metric_value` after filtering `metric_key` to the selected summary indicator.",
            "- Aggregation: mean across dataset-level rows per model.",
            "- Error bar: `95% CI = 1.96 * std / sqrt(n_datasets)`.",
            "- Sorting: descending by the aggregated mean.",
            "- Included roster: 11 synthetic generators in the frozen paper-facing model set; `REAL` intentionally omitted to keep the comparison generator-vs-generator.",
            "",
            "## Overlay panel spec",
            "",
            "- `overview_sql_panel.*` uses the current `benchmark_overall_table_real_model_summary.csv` columns `subgroup_structure_mean`, `conditional_dependency_structure_mean`, `tail_breakdown_mean`, `missingness_structure_mean`, and `cardinality_structure_mean`.",
            "- `overview_distance_panel.*` uses the same model-summary file but converts `jsd_distance_mean`, `ks_distance_mean`, `tvd_distance_mean`, and `wasserstein_distance_mean` into higher-is-better scores via `1 - distance` to match the paper overview convention.",
            "- Panel model order is frozen to the existing overview legend order rather than value sorting, so these files can be dropped into Figma without reshuffling colors.",
            "- Panel bars keep two-decimal value labels above each bar; extra headroom is reserved so the labels remain readable after the three panels are merged back together.",
            "",
            "## Radar chart spec",
            "",
            "- Axes are fixed in this closed order: `Distance -> Subgroup -> Conditional -> Tail -> Missingness -> Cardinality -> Distance`.",
            "- Radial range is fixed to `0-1` for every axis.",
            "- Additional normalization: none. The plotted scores are already benchmark-scale scores on `[0, 1]`.",
            "- Aggregation: mean across dataset-level rows per model and per axis.",
            "- Legend order: descending by the arithmetic mean over the six plotted axes.",
            "- Raw upstream source files behind each axis:",
            *radar_axis_lines,
            "",
            "## Output recommendation",
            "",
            "- For direct paper embedding, prefer `overview_model_bar_chart.pdf`, `overview_model_bar_chart_distance_overall.pdf`, and `overview_model_radar_chart.pdf`.",
            "- Use the `.png` variants for quick slide/mockup replacement.",
            "- The corresponding `*_source.csv` files are the audit tables to keep with the figure assets.",
            "",
            "## Files generated here",
            "",
            "- `overview_model_bar_chart.png`",
            "- `overview_model_bar_chart.pdf`",
            "- `overview_model_bar_chart_source.csv`",
            "- `overview_model_bar_chart_distance_overall.png`",
            "- `overview_model_bar_chart_distance_overall.pdf`",
            "- `overview_model_bar_chart_distance_overall_source.csv`",
            "- `overview_model_radar_chart.png`",
            "- `overview_model_radar_chart.pdf`",
            "- `overview_model_radar_chart_source.csv`",
            "- `overview_sql_panel.png`",
            "- `overview_sql_panel.pdf`",
            "- `overview_sql_panel_source.csv`",
            "- `overview_distance_panel.png`",
            "- `overview_distance_panel.pdf`",
            "- `overview_distance_panel_source.csv`",
            "- `overview_radar_panel.png`",
            "- `overview_radar_panel.pdf`",
            "- `overview_panels_combined_preview.png`",
            "- `overview_panels_combined_preview.pdf`",
            "- `overview_full_figure_preview.png`",
            "- `overview_full_figure.svg`",
            "- `overview_full_figure_layout.json`",
            "- `overview_regenerated_summary.md`",
            "",
            "## Top models in this rebuild",
            "",
        ]
    )

    for row in bar_table.head(5).to_dict("records"):
        lines.append(f"- Bar: {row['model_label']} = {row['metric_mean']:.3f} +/- {row['metric_ci95_radius']:.3f}")
    for row in distance_bar_table.head(5).to_dict("records"):
        lines.append(f"- Distance overall bar: {row['model_label']} = {row['metric_mean']:.3f} +/- {row['metric_ci95_radius']:.3f}")
    for row in (
        radar_table[["model_id", "model_label", "radar_mean_across_axes"]]
        .drop_duplicates()
        .sort_values(["radar_mean_across_axes", "model_label"], ascending=[False, True])
        .head(5)
        .to_dict("records")
    ):
        lines.append(f"- Radar mean over six axes: {row['model_label']} = {row['radar_mean_across_axes']:.3f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_output_readme() -> str:
    return render_final_readme(
        title="overview_regenerated outputs",
        summary="Auditable Python-regenerated overview charts for the current paper line.",
        primary_files=[
            "overview_model_bar_chart.pdf",
            "overview_model_bar_chart.png",
            "overview_model_bar_chart_distance_overall.pdf",
            "overview_model_bar_chart_distance_overall.png",
            "overview_model_radar_chart.pdf",
            "overview_model_radar_chart.png",
            "overview_sql_panel.pdf",
            "overview_sql_panel.png",
            "overview_sql_panel_source.csv",
            "overview_distance_panel.pdf",
            "overview_distance_panel.png",
            "overview_distance_panel_source.csv",
            "overview_radar_panel.pdf",
            "overview_radar_panel.png",
            "overview_panels_combined_preview.pdf",
            "overview_panels_combined_preview.png",
            "overview_full_figure.svg",
            "overview_full_figure_preview.png",
        ],
        must_do_files=[
            "overview_model_bar_chart.pdf",
            "overview_model_bar_chart.png",
            "overview_model_bar_chart_source.csv",
            "overview_model_bar_chart_distance_overall.pdf",
            "overview_model_bar_chart_distance_overall.png",
            "overview_model_bar_chart_distance_overall_source.csv",
            "overview_model_radar_chart.pdf",
            "overview_model_radar_chart.png",
            "overview_model_radar_chart_source.csv",
            "overview_sql_panel.pdf",
            "overview_sql_panel.png",
            "overview_sql_panel_source.csv",
            "overview_distance_panel.pdf",
            "overview_distance_panel.png",
            "overview_distance_panel_source.csv",
            "overview_radar_panel.pdf",
            "overview_radar_panel.png",
            "overview_panels_combined_preview.pdf",
            "overview_panels_combined_preview.png",
            "overview_full_figure.svg",
            "overview_full_figure_preview.png",
            "overview_full_figure_layout.json",
            "overview_regenerated_summary.md",
        ],
        support_files=[
            "README.md",
        ],
        notes=[
            "",
            "All files in `final/` are copied from the canonical outputs under `data/` and `figures/`.",
        ],
    )


def run_overview_regenerated(
    *,
    dataset_level_csv: Path = DEFAULT_DATASET_LEVEL_CSV,
    model_summary_csv: Path = DEFAULT_MODEL_SUMMARY_CSV,
    source_manifest_csv: Path = DEFAULT_SOURCE_MANIFEST_CSV,
    model_radar_manifest_path: Path = DEFAULT_MODEL_RADAR_MANIFEST,
    overview_background_png: Path = DEFAULT_OVERVIEW_BACKGROUND_PNG,
    bar_metric_key: str = DEFAULT_BAR_METRIC_KEY,
) -> dict[str, Any]:
    _ensure_dirs()
    verdict = _load_source_verdict(dataset_level_csv, source_manifest_csv, model_radar_manifest_path)
    dataset_level = _read_dataset_level_table(verdict.dataset_level_csv)
    model_summary = _read_model_summary_table(model_summary_csv)
    source_manifest = _read_source_manifest(verdict.source_manifest_csv)
    metric_titles = _metric_title_lookup(source_manifest)
    bar_metric_label = metric_titles.get(bar_metric_key, DEFAULT_BAR_METRIC_LABEL)

    bar_table = _build_bar_source_table(
        dataset_level,
        source_manifest,
        bar_metric_key=bar_metric_key,
        verdict=verdict,
    )
    distance_bar_table = _build_bar_source_table(
        dataset_level,
        source_manifest,
        bar_metric_key=SECONDARY_BAR_METRIC_KEY,
        verdict=verdict,
    )
    radar_table = _build_radar_source_table(
        dataset_level,
        source_manifest,
        verdict=verdict,
    )
    query_vs_wasserstein_table = _build_query_vs_wasserstein_table(model_summary)
    query_vs_distance_avg_table = _build_query_vs_distance_avg_table(model_summary)
    sql_panel_table = _build_panel_source_table(
        model_summary,
        model_summary_csv=model_summary_csv,
        verdict=verdict,
        panel_name="overview_sql_panel",
        axis_specs=SQL_PANEL_AXES,
        transform_rule="none",
        source_field_note="Direct family score means from benchmark_overall_table_real_model_summary.csv",
    )
    distance_panel_table = _build_panel_source_table(
        model_summary,
        model_summary_csv=model_summary_csv,
        verdict=verdict,
        panel_name="overview_distance_panel",
        axis_specs=DISTANCE_PANEL_AXES,
        transform_rule="one_minus_distance",
        source_field_note="Panel score uses 1 - raw distance mean so higher is better and aligns with the paper overview convention",
    )

    bar_csv = DATA_DIR / "overview_model_bar_chart_source.csv"
    distance_bar_csv = DATA_DIR / "overview_model_bar_chart_distance_overall_source.csv"
    radar_csv = DATA_DIR / "overview_model_radar_chart_source.csv"
    sql_panel_csv = DATA_DIR / "overview_sql_panel_source.csv"
    distance_panel_csv = DATA_DIR / "overview_distance_panel_source.csv"
    query_vs_wasserstein_csv = DATA_DIR / "overview_query_vs_wasserstein_rank_source.csv"
    query_vs_distance_avg_csv = DATA_DIR / "overview_query_vs_distance_avg_rank_source.csv"
    bar_png = FIGURES_DIR / "overview_model_bar_chart.png"
    bar_pdf = FIGURES_DIR / "overview_model_bar_chart.pdf"
    distance_bar_png = FIGURES_DIR / "overview_model_bar_chart_distance_overall.png"
    distance_bar_pdf = FIGURES_DIR / "overview_model_bar_chart_distance_overall.pdf"
    radar_png = FIGURES_DIR / "overview_model_radar_chart.png"
    radar_pdf = FIGURES_DIR / "overview_model_radar_chart.pdf"
    sql_panel_png = FIGURES_DIR / "overview_sql_panel.png"
    sql_panel_pdf = FIGURES_DIR / "overview_sql_panel.pdf"
    sql_panel_svg = FIGURES_DIR / "overview_sql_panel.svg"
    distance_panel_png = FIGURES_DIR / "overview_distance_panel.png"
    distance_panel_pdf = FIGURES_DIR / "overview_distance_panel.pdf"
    distance_panel_svg = FIGURES_DIR / "overview_distance_panel.svg"
    radar_panel_png = FIGURES_DIR / "overview_radar_panel.png"
    radar_panel_pdf = FIGURES_DIR / "overview_radar_panel.pdf"
    radar_panel_svg = FIGURES_DIR / "overview_radar_panel.svg"
    sql_panel_minimal_png = FIGURES_DIR / "overview_sql_panel_minimal.png"
    sql_panel_minimal_pdf = FIGURES_DIR / "overview_sql_panel_minimal.pdf"
    sql_panel_minimal_svg = FIGURES_DIR / "overview_sql_panel_minimal.svg"
    distance_panel_minimal_png = FIGURES_DIR / "overview_distance_panel_minimal.png"
    distance_panel_minimal_pdf = FIGURES_DIR / "overview_distance_panel_minimal.pdf"
    distance_panel_minimal_svg = FIGURES_DIR / "overview_distance_panel_minimal.svg"
    radar_panel_minimal_png = FIGURES_DIR / "overview_radar_panel_minimal.png"
    radar_panel_minimal_pdf = FIGURES_DIR / "overview_radar_panel_minimal.pdf"
    radar_panel_minimal_svg = FIGURES_DIR / "overview_radar_panel_minimal.svg"
    radar_plot_only_svg = FIGURES_DIR / "overview_radar_plot_only.svg"
    radar_plot_only_png = FIGURES_DIR / "overview_radar_plot_only.png"
    radar_plot_only_pdf = FIGURES_DIR / "overview_radar_plot_only.pdf"
    combined_preview_png = FIGURES_DIR / "overview_panels_combined_preview.png"
    combined_preview_pdf = FIGURES_DIR / "overview_panels_combined_preview.pdf"
    full_overview_preview_png = FIGURES_DIR / "overview_full_figure_preview.png"
    full_overview_svg = FIGURES_DIR / "overview_full_figure.svg"
    full_overview_editable_svg = FIGURES_DIR / "overview_full_vector_editable.svg"
    full_overview_editable_minimal_svg = FIGURES_DIR / "overview_full_vector_editable_minimal.svg"
    query_vs_wasserstein_svg = FIGURES_DIR / "overview_query_vs_wasserstein_rank_compare.svg"
    query_vs_distance_avg_svg = FIGURES_DIR / "overview_query_vs_distance_avg_rank_compare.svg"
    full_overview_layout_json = DATA_DIR / "overview_full_figure_layout.json"
    radar_plot_only_minimal_svg = FIGURES_DIR / "overview_radar_plot_only_minimal.svg"
    radar_plot_only_minimal_png = FIGURES_DIR / "overview_radar_plot_only_minimal.png"
    radar_plot_only_minimal_pdf = FIGURES_DIR / "overview_radar_plot_only_minimal.pdf"
    summary_md = OUTPUT_ROOT / "overview_regenerated_summary.md"
    readme_path = OUTPUT_ROOT / "README.md"
    manifest_path = OUTPUT_ROOT / "manifest.json"

    bar_table.to_csv(bar_csv, index=False)
    distance_bar_table.to_csv(distance_bar_csv, index=False)
    radar_table.to_csv(radar_csv, index=False)
    sql_panel_table.to_csv(sql_panel_csv, index=False)
    distance_panel_table.to_csv(distance_panel_csv, index=False)
    query_vs_wasserstein_table.to_csv(query_vs_wasserstein_csv, index=False)
    query_vs_distance_avg_table.to_csv(query_vs_distance_avg_csv, index=False)
    _plot_model_bar_chart(bar_table, bar_png, bar_pdf)
    _plot_model_bar_chart(distance_bar_table, distance_bar_png, distance_bar_pdf)
    _plot_model_radar_chart(radar_table, radar_png, radar_pdf)
    _plot_sql_family_panel(sql_panel_table, sql_panel_png, sql_panel_pdf, sql_panel_svg)
    _plot_distance_metric_panel(distance_panel_table, distance_panel_png, distance_panel_pdf, distance_panel_svg)
    _plot_radar_panel_variant(
        output_png=radar_panel_png,
        output_pdf=radar_panel_pdf,
        output_svg=radar_panel_svg,
        show_title=True,
        show_axis_labels=True,
    )
    _plot_sql_family_panel_minimal(sql_panel_table, sql_panel_minimal_png, sql_panel_minimal_pdf, sql_panel_minimal_svg)
    _plot_distance_metric_panel_minimal(distance_panel_table, distance_panel_minimal_png, distance_panel_minimal_pdf, distance_panel_minimal_svg)
    _plot_radar_panel_variant(
        output_png=radar_panel_minimal_png,
        output_pdf=radar_panel_minimal_pdf,
        output_svg=radar_panel_minimal_svg,
        show_title=False,
        show_axis_labels=False,
    )
    _plot_radar_panel_variant(
        output_png=radar_plot_only_png,
        output_pdf=radar_plot_only_pdf,
        output_svg=radar_plot_only_svg,
        show_title=False,
        show_axis_labels=True,
        show_panel_frame=False,
        axes_rect=(0.10, 0.10, 0.80, 0.80),
    )
    _plot_radar_panel_variant(
        output_png=radar_plot_only_minimal_png,
        output_pdf=radar_plot_only_minimal_pdf,
        output_svg=radar_plot_only_minimal_svg,
        show_title=False,
        show_axis_labels=False,
        show_panel_frame=False,
        axes_rect=(0.10, 0.10, 0.80, 0.80),
    )
    _build_combined_preview(
        sql_panel_png=sql_panel_png,
        distance_panel_png=distance_panel_png,
        radar_panel_png=radar_panel_png,
        output_png=combined_preview_png,
        output_pdf=combined_preview_pdf,
    )
    _build_full_overview_preview_and_svg(
        background_png=overview_background_png,
        sql_panel_png=sql_panel_png,
        distance_panel_png=distance_panel_png,
        radar_panel_png=radar_panel_png,
        output_preview_png=full_overview_preview_png,
        output_svg=full_overview_svg,
        output_layout_json=full_overview_layout_json,
    )
    _build_editable_overview_svg(
        sql_panel_svg=sql_panel_svg,
        distance_panel_svg=distance_panel_svg,
        radar_panel_svg=radar_plot_only_svg,
        output_svg=full_overview_editable_svg,
    )
    _build_editable_overview_svg(
        sql_panel_svg=sql_panel_minimal_svg,
        distance_panel_svg=distance_panel_minimal_svg,
        radar_panel_svg=radar_plot_only_minimal_svg,
        output_svg=full_overview_editable_minimal_svg,
    )
    _build_query_vs_wasserstein_svg(
        query_vs_wasserstein_table,
        output_svg=query_vs_wasserstein_svg,
        output_csv=query_vs_wasserstein_csv,
    )
    _build_query_vs_distance_avg_svg(
        query_vs_distance_avg_table,
        output_svg=query_vs_distance_avg_svg,
        output_csv=query_vs_distance_avg_csv,
    )
    _write_summary_markdown(
        summary_md,
        verdict=verdict,
        bar_table=bar_table,
        distance_bar_table=distance_bar_table,
        radar_table=radar_table,
        bar_metric_key=bar_metric_key,
        bar_metric_label=bar_metric_label,
        source_manifest=source_manifest,
    )
    readme_path.write_text(_build_output_readme(), encoding="utf-8")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_result_line": verdict.current_result_line,
        "current_result_label": verdict.current_result_label,
        "bar_metric_key": bar_metric_key,
        "bar_metric_label": bar_metric_label,
        "distance_bar_metric_key": SECONDARY_BAR_METRIC_KEY,
        "dataset_level_csv": str(verdict.dataset_level_csv.resolve()),
        "model_summary_csv": str(model_summary_csv.resolve()),
        "source_manifest_csv": str(verdict.source_manifest_csv.resolve()),
        "bar_source_csv": str(bar_csv.resolve()),
        "distance_bar_source_csv": str(distance_bar_csv.resolve()),
        "radar_source_csv": str(radar_csv.resolve()),
        "sql_panel_source_csv": str(sql_panel_csv.resolve()),
        "distance_panel_source_csv": str(distance_panel_csv.resolve()),
        "query_vs_wasserstein_source_csv": str(query_vs_wasserstein_csv.resolve()),
        "query_vs_distance_avg_source_csv": str(query_vs_distance_avg_csv.resolve()),
        "bar_png": str(bar_png.resolve()),
        "bar_pdf": str(bar_pdf.resolve()),
        "distance_bar_png": str(distance_bar_png.resolve()),
        "distance_bar_pdf": str(distance_bar_pdf.resolve()),
        "radar_png": str(radar_png.resolve()),
        "radar_pdf": str(radar_pdf.resolve()),
        "sql_panel_png": str(sql_panel_png.resolve()),
        "sql_panel_pdf": str(sql_panel_pdf.resolve()),
        "sql_panel_svg": str(sql_panel_svg.resolve()),
        "distance_panel_png": str(distance_panel_png.resolve()),
        "distance_panel_pdf": str(distance_panel_pdf.resolve()),
        "distance_panel_svg": str(distance_panel_svg.resolve()),
        "radar_panel_png": str(radar_panel_png.resolve()),
        "radar_panel_pdf": str(radar_panel_pdf.resolve()),
        "radar_panel_svg": str(radar_panel_svg.resolve()),
        "sql_panel_minimal_png": str(sql_panel_minimal_png.resolve()),
        "sql_panel_minimal_pdf": str(sql_panel_minimal_pdf.resolve()),
        "sql_panel_minimal_svg": str(sql_panel_minimal_svg.resolve()),
        "distance_panel_minimal_png": str(distance_panel_minimal_png.resolve()),
        "distance_panel_minimal_pdf": str(distance_panel_minimal_pdf.resolve()),
        "distance_panel_minimal_svg": str(distance_panel_minimal_svg.resolve()),
        "radar_panel_minimal_png": str(radar_panel_minimal_png.resolve()),
        "radar_panel_minimal_pdf": str(radar_panel_minimal_pdf.resolve()),
        "radar_panel_minimal_svg": str(radar_panel_minimal_svg.resolve()),
        "radar_plot_only_png": str(radar_plot_only_png.resolve()),
        "radar_plot_only_pdf": str(radar_plot_only_pdf.resolve()),
        "radar_plot_only_svg": str(radar_plot_only_svg.resolve()),
        "radar_plot_only_minimal_png": str(radar_plot_only_minimal_png.resolve()),
        "radar_plot_only_minimal_pdf": str(radar_plot_only_minimal_pdf.resolve()),
        "radar_plot_only_minimal_svg": str(radar_plot_only_minimal_svg.resolve()),
        "combined_preview_png": str(combined_preview_png.resolve()),
        "combined_preview_pdf": str(combined_preview_pdf.resolve()),
        "full_overview_preview_png": str(full_overview_preview_png.resolve()),
        "full_overview_svg": str(full_overview_svg.resolve()),
        "full_overview_editable_svg": str(full_overview_editable_svg.resolve()),
        "full_overview_editable_minimal_svg": str(full_overview_editable_minimal_svg.resolve()),
        "query_vs_wasserstein_svg": str(query_vs_wasserstein_svg.resolve()),
        "query_vs_distance_avg_svg": str(query_vs_distance_avg_svg.resolve()),
        "full_overview_layout_json": str(full_overview_layout_json.resolve()),
        "summary_md": str(summary_md.resolve()),
        "radar_axis_order": [label for _, label in RADAR_AXIS_ORDER],
        "radial_range": [0.0, 1.0],
        "normalization_rule": "none",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    sync_final_outputs(
        FINAL_DIR,
        [
            bar_csv,
            distance_bar_csv,
            radar_csv,
            sql_panel_csv,
            distance_panel_csv,
            query_vs_wasserstein_csv,
            query_vs_distance_avg_csv,
            bar_png,
            bar_pdf,
            distance_bar_png,
            distance_bar_pdf,
            radar_png,
            radar_pdf,
            sql_panel_png,
            sql_panel_pdf,
            sql_panel_svg,
            distance_panel_png,
            distance_panel_pdf,
            distance_panel_svg,
            radar_panel_png,
            radar_panel_pdf,
            radar_panel_svg,
            sql_panel_minimal_png,
            sql_panel_minimal_pdf,
            sql_panel_minimal_svg,
            distance_panel_minimal_png,
            distance_panel_minimal_pdf,
            distance_panel_minimal_svg,
            radar_panel_minimal_png,
            radar_panel_minimal_pdf,
            radar_panel_minimal_svg,
            radar_plot_only_png,
            radar_plot_only_pdf,
            radar_plot_only_svg,
            radar_plot_only_minimal_png,
            radar_plot_only_minimal_pdf,
            radar_plot_only_minimal_svg,
            combined_preview_png,
            combined_preview_pdf,
            full_overview_preview_png,
            full_overview_svg,
            full_overview_editable_svg,
            full_overview_editable_minimal_svg,
            query_vs_wasserstein_svg,
            query_vs_distance_avg_svg,
            full_overview_layout_json,
            summary_md,
            readme_path,
            manifest_path,
        ],
        must_do_aliases={
            "overview_model_bar_chart_source.csv": bar_csv,
            "overview_model_bar_chart_distance_overall_source.csv": distance_bar_csv,
            "overview_model_radar_chart_source.csv": radar_csv,
            "overview_sql_panel_source.csv": sql_panel_csv,
            "overview_distance_panel_source.csv": distance_panel_csv,
            "overview_model_bar_chart.png": bar_png,
            "overview_model_bar_chart.pdf": bar_pdf,
            "overview_model_bar_chart_distance_overall.png": distance_bar_png,
            "overview_model_bar_chart_distance_overall.pdf": distance_bar_pdf,
            "overview_model_radar_chart.png": radar_png,
            "overview_model_radar_chart.pdf": radar_pdf,
            "overview_sql_panel.png": sql_panel_png,
            "overview_sql_panel.pdf": sql_panel_pdf,
            "overview_sql_panel.svg": sql_panel_svg,
            "overview_distance_panel.png": distance_panel_png,
            "overview_distance_panel.pdf": distance_panel_pdf,
            "overview_distance_panel.svg": distance_panel_svg,
            "overview_radar_panel.png": radar_panel_png,
            "overview_radar_panel.pdf": radar_panel_pdf,
            "overview_radar_panel.svg": radar_panel_svg,
            "overview_sql_panel_minimal.png": sql_panel_minimal_png,
            "overview_sql_panel_minimal.pdf": sql_panel_minimal_pdf,
            "overview_sql_panel_minimal.svg": sql_panel_minimal_svg,
            "overview_distance_panel_minimal.png": distance_panel_minimal_png,
            "overview_distance_panel_minimal.pdf": distance_panel_minimal_pdf,
            "overview_distance_panel_minimal.svg": distance_panel_minimal_svg,
            "overview_radar_panel_minimal.png": radar_panel_minimal_png,
            "overview_radar_panel_minimal.pdf": radar_panel_minimal_pdf,
            "overview_radar_panel_minimal.svg": radar_panel_minimal_svg,
            "overview_radar_plot_only.png": radar_plot_only_png,
            "overview_radar_plot_only.pdf": radar_plot_only_pdf,
            "overview_radar_plot_only.svg": radar_plot_only_svg,
            "overview_radar_plot_only_minimal.png": radar_plot_only_minimal_png,
            "overview_radar_plot_only_minimal.pdf": radar_plot_only_minimal_pdf,
            "overview_radar_plot_only_minimal.svg": radar_plot_only_minimal_svg,
            "overview_panels_combined_preview.png": combined_preview_png,
            "overview_panels_combined_preview.pdf": combined_preview_pdf,
            "overview_full_figure_preview.png": full_overview_preview_png,
            "overview_full_figure.svg": full_overview_svg,
            "overview_full_vector_editable.svg": full_overview_editable_svg,
            "overview_full_vector_editable_minimal.svg": full_overview_editable_minimal_svg,
            "overview_query_vs_wasserstein_rank_compare.svg": query_vs_wasserstein_svg,
            "overview_query_vs_wasserstein_rank_source.csv": query_vs_wasserstein_csv,
            "overview_query_vs_distance_avg_rank_compare.svg": query_vs_distance_avg_svg,
            "overview_query_vs_distance_avg_rank_source.csv": query_vs_distance_avg_csv,
            "overview_full_figure_layout.json": full_overview_layout_json,
            "overview_regenerated_summary.md": summary_md,
        },
        version_tag=verdict.current_result_line,
        copy_plain_files=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate auditable overview bar/radar charts.")
    parser.add_argument(
        "--dataset-level-csv",
        type=Path,
        default=DEFAULT_DATASET_LEVEL_CSV,
        help="Consolidated benchmark dataset-level CSV to aggregate from.",
    )
    parser.add_argument(
        "--source-manifest-csv",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST_CSV,
        help="Metric-to-source manifest CSV paired with the dataset-level CSV.",
    )
    parser.add_argument(
        "--model-summary-csv",
        type=Path,
        default=DEFAULT_MODEL_SUMMARY_CSV,
        help="Consolidated benchmark model-summary CSV used by the split overview panels.",
    )
    parser.add_argument(
        "--model-radar-manifest",
        type=Path,
        default=DEFAULT_MODEL_RADAR_MANIFEST,
        help="Optional manifest used to infer the current paper result line.",
    )
    parser.add_argument(
        "--bar-metric-key",
        default=DEFAULT_BAR_METRIC_KEY,
        help="Metric key for the ranking bar chart. Default: query_overall.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_overview_regenerated(
        dataset_level_csv=args.dataset_level_csv,
        model_summary_csv=args.model_summary_csv,
        source_manifest_csv=args.source_manifest_csv,
        model_radar_manifest_path=args.model_radar_manifest,
        bar_metric_key=args.bar_metric_key,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
