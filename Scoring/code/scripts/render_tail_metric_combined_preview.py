#!/usr/bin/env python3
"""Render a one-figure grouped preview that combines the three tail submetrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_tail_metric_model_grid import (
    METRICS,
    _configure_style,
    _load_model_threshold,
    _metric_pivot,
    _model_order,
    _save,
    _style_axis,
)
from scripts.render_tail_stress_main_figure import DECOMP_COLORS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _measure_text_width_axes(ax, renderer, text: str, fontsize: float) -> float:
    probe = ax.text(0.0, 0.0, text, transform=ax.transAxes, fontsize=fontsize, alpha=0.0)
    bbox = probe.get_window_extent(renderer=renderer)
    probe.remove()
    inv = ax.transAxes.inverted()
    x0 = inv.transform((bbox.x0, bbox.y0))[0]
    x1 = inv.transform((bbox.x1, bbox.y0))[0]
    return x1 - x0


def _draw_patch_icon(ax, x: float, y: float, color: str, *, alpha: float) -> None:
    ax.add_patch(
        Rectangle(
            (x - 0.010, y - 0.013),
            0.020,
            0.026,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            clip_on=False,
            zorder=5,
        )
    )


def _draw_square_icon(ax, x: float, y: float, color: str) -> None:
    ax.plot(
        [x],
        [y],
        marker="s",
        markersize=7.5,
        linestyle="none",
        color=color,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )


def _draw_line_icon(ax, x: float, y: float, color: str) -> None:
    ax.plot(
        [x - 0.013, x + 0.013],
        [y, y],
        color=color,
        linewidth=2.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )


def _draw_manual_legend_row(ax, y: float, items: list[dict[str, str]], icon_centers: list[float], text_starts: list[float], fontsize: float) -> None:
    for item, icon_x, text_x in zip(items, icon_centers, text_starts):
        kind = item["kind"]
        if kind == "patch":
            _draw_patch_icon(ax, icon_x, y, item["color"], alpha=float(item["alpha"]))
        elif kind == "square":
            _draw_square_icon(ax, icon_x, y, item["color"])
        else:
            _draw_line_icon(ax, icon_x, y, item["color"])

        ax.text(
            text_x,
            y,
            item["label"],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fontsize,
        )


def render_combined_preview(tables_dir: Path, output: Path) -> Path:
    df = _load_model_threshold(tables_dir)
    model_order = _model_order(df)
    metric_payloads: list[dict[str, object]] = []

    for metric, ylabel in METRICS:
        pivot = _metric_pivot(df, metric, model_order)
        values = pivot.to_numpy(dtype=float)
        metric_payloads.append(
            {
                "metric": metric,
                "label": ylabel.replace(" score", ""),
                "pivot": pivot,
                "mean": np.nanmean(values, axis=1),
                "q25": np.nanquantile(values, 0.25, axis=1),
                "q75": np.nanquantile(values, 0.75, axis=1),
                "vmin": np.nanmin(values, axis=1),
                "vmax": np.nanmax(values, axis=1),
            }
        )

    x = np.arange(len(model_order), dtype=float)
    offsets = np.array([-0.24, 0.0, 0.24], dtype=float)
    box_width = 0.18

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "legend.title_fontsize": 15,
        }
    )
    fig, ax = plt.subplots(figsize=(17.6, 6.9), constrained_layout=False)

    for idx, payload in enumerate(metric_payloads):
        metric = str(payload["metric"])
        color = DECOMP_COLORS[f"{metric}_mean"]
        x_pos = x + offsets[idx]
        q25 = np.asarray(payload["q25"], dtype=float)
        q75 = np.asarray(payload["q75"], dtype=float)
        means = np.asarray(payload["mean"], dtype=float)
        vmin = np.asarray(payload["vmin"], dtype=float)
        vmax = np.asarray(payload["vmax"], dtype=float)

        for j, xpos in enumerate(x_pos):
            if np.isnan(means[j]):
                continue
            rect = Rectangle(
                (float(xpos) - box_width / 2.0, float(q25[j])),
                box_width,
                max(1e-6, float(q75[j] - q25[j])),
                facecolor=color,
                edgecolor="none",
                alpha=0.28,
                zorder=1,
            )
            ax.add_patch(rect)
            ax.vlines(xpos, float(vmin[j]), float(vmax[j]), color=color, linewidth=2.0, alpha=0.9, zorder=2)
            ax.hlines([float(vmin[j]), float(vmax[j])], xpos - 0.045, xpos + 0.045, color=color, linewidth=2.0, zorder=2)
            ax.scatter([xpos], [float(means[j])], marker="s", s=42, color=color, zorder=3)

    for boundary in np.arange(len(model_order) - 1, dtype=float) + 0.5:
        ax.axvline(boundary, color="#EEEEEE", linewidth=0.8, zorder=0)

    ax.set_xticks(x, model_order, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0.0, 1.0)
    _style_axis(ax)

    metric_items = [
        {
            "kind": "patch",
            "label": label.replace(" score", ""),
            "color": DECOMP_COLORS[f"{metric}_mean"],
            "alpha": "0.5",
        }
        for metric, label in METRICS
    ]
    summary_items = [
        {"kind": "square", "label": "Mean over thresholds", "color": "#555555", "alpha": "1.0"},
        {"kind": "line", "label": "Min-max over thresholds", "color": "#555555", "alpha": "1.0"},
        {"kind": "patch", "label": "IQR over thresholds", "color": "#BBBBBB", "alpha": "0.35"},
    ]

    row_title_x = 0.30
    row2_title_x = 0.245
    row1_y = 0.955
    row2_y = 0.875

    title1 = ax.text(
        row_title_x,
        row1_y,
        "Tail subitems",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=15,
    )
    title2 = ax.text(
        row2_title_x,
        row2_y,
        "Within each subitem",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=15,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    title1_right = inv.transform(title1.get_window_extent(renderer=renderer).corners()[2])[0]
    title2_right = inv.transform(title2.get_window_extent(renderer=renderer).corners()[2])[0]
    title_right = max(title1_right, title2_right)
    legend_fontsize = 14
    text_gap = 0.020
    col_gap = 0.030
    icon_centers: list[float] = []
    text_starts: list[float] = []
    next_icon_x = title_right + 0.020

    for metric_item, summary_item in zip(metric_items, summary_items):
        icon_centers.append(next_icon_x)
        text_x = next_icon_x + text_gap
        text_starts.append(text_x)
        column_text_width = max(
            _measure_text_width_axes(ax, renderer, str(metric_item["label"]), legend_fontsize),
            _measure_text_width_axes(ax, renderer, str(summary_item["label"]), legend_fontsize),
        )
        next_icon_x = text_x + column_text_width + col_gap

    _draw_manual_legend_row(ax, row1_y, metric_items, icon_centers, text_starts, legend_fontsize)
    _draw_manual_legend_row(ax, row2_y, summary_items, icon_centers, text_starts, legend_fontsize)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    _save(fig, output)
    return output


def main() -> int:
    args = _build_parser().parse_args()
    _configure_style()
    out_path = render_combined_preview(args.tables_dir, args.output)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
