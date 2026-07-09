#!/usr/bin/env python3
"""Render local preview figures with one bar-like slot per model and a threshold line on top."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_tail_stress_main_figure import MODEL_COLORS, THRESHOLD_ORDER


METRICS = [
    ("tail_set_consistency", "Tail set consistency score"),
    ("tail_mass_similarity", "Tail mass similarity score"),
    ("tail_concentration_consistency", "Tail concentration consistency score"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _blend_with_white(color: str, strength: float) -> tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    return tuple((base * (1.0 - strength) + white * strength).tolist())


def _load_model_threshold(tables_dir: Path) -> pd.DataFrame:
    path = tables_dir / "model_threshold_summary.csv"
    df = pd.read_csv(path)
    df = df[df["model_label"].isin(MODEL_COLORS)].copy()
    df["threshold_label"] = pd.Categorical(df["threshold_label"], categories=THRESHOLD_ORDER, ordered=True)
    df = df.sort_values(["model_label", "threshold_label"]).reset_index(drop=True)
    return df


def _model_order(df: pd.DataFrame) -> list[str]:
    present = set(df["model_label"].dropna().unique().tolist())
    return [label for label in MODEL_COLORS if label in present]


def _metric_pivot(df: pd.DataFrame, metric: str, model_order: list[str]) -> pd.DataFrame:
    return (
        df.pivot_table(index="model_label", columns="threshold_label", values=metric, aggfunc="mean")
        .reindex(index=model_order, columns=THRESHOLD_ORDER)
        .astype(float)
    )


def _add_gradient_line(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, base_color: str) -> None:
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    strengths = np.linspace(0.0, 0.75, max(1, len(segments)))
    colors = [_blend_with_white(base_color, float(s)) for s in strengths]
    lc = LineCollection(segments, colors=colors, linewidths=2.2, zorder=4, capstyle="round")
    ax.add_collection(lc)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        face = _blend_with_white(base_color, float(np.linspace(0.0, 0.82, len(xs))[idx]))
        ax.scatter([x], [y], s=20, color=face, edgecolor="white", linewidth=0.45, zorder=5)


def _threshold_legend_handles() -> list[Line2D]:
    strength_map = {"10%": 0.0, "1%": 0.46, "0.001%": 0.82}
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_blend_with_white("#666666", strength),
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6,
            label=label,
        )
        for label, strength in strength_map.items()
    ]


def render_barline_preview(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    model_order = _model_order(df)
    shade_strengths = np.linspace(0.18, 0.82, len(THRESHOLD_ORDER))
    x_centers = np.arange(len(model_order), dtype=float)
    outputs: list[Path] = []

    for metric, ylabel in METRICS:
        pivot = _metric_pivot(df, metric, model_order)
        values = pivot.to_numpy(dtype=float)
        means = np.nanmean(values, axis=1)
        fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)

        for idx, model in enumerate(model_order):
            base = MODEL_COLORS[model]
            mean_val = float(means[idx])
            xs = np.linspace(x_centers[idx] - 0.29, x_centers[idx] + 0.29, len(THRESHOLD_ORDER))
            ys = pivot.loc[model].to_numpy(dtype=float)

            # Bar-like body uses the mean score as the filled height.
            ax.bar(
                x_centers[idx],
                mean_val,
                width=0.64,
                color=_blend_with_white(base, 0.72),
                edgecolor=_blend_with_white(base, 0.25),
                linewidth=1.0,
                zorder=1,
            )
            # Show mean as a light reference line inside the bar body.
            ax.hlines(
                mean_val,
                x_centers[idx] - 0.28,
                x_centers[idx] + 0.28,
                colors=_blend_with_white(base, 0.15),
                linewidth=1.4,
                zorder=2,
            )
            _add_gradient_line(ax, xs, ys, base)

        ax.set_xticks(x_centers, model_order, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_ylim(0.0, min(1.0, float(np.nanmax(values)) + 0.08))
        ax.set_title(f"{ylabel}: one model slot with threshold trajectory on top", loc="left", pad=6)
        _style_axis(ax)

        legend_items = [
            plt.Line2D([0], [0], color="#555555", linewidth=2.2, label="Threshold trajectory"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#DDDDDD", edgecolor="#BBBBBB", label="Mean score bar"),
        ]
        left_legend = ax.legend(handles=legend_items, loc="upper left", frameon=False, ncols=2)
        ax.add_artist(left_legend)
        right_legend = ax.legend(
            handles=_threshold_legend_handles(),
            title="Threshold shading",
            loc="upper right",
            frameon=False,
            ncols=3,
            handletextpad=0.4,
            columnspacing=0.8,
            borderaxespad=0.2,
        )
        right_legend.get_title().set_fontsize(8)

        out = output_dir / f"{metric}__barline_preview.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, facecolor="white")
        fig.savefig(out.with_suffix(".pdf"), dpi=300, facecolor="white")
        plt.close(fig)
        outputs.append(out)
    return outputs


def main() -> int:
    args = _build_parser().parse_args()
    _configure_style()
    df = _load_model_threshold(args.tables_dir)
    outputs = render_barline_preview(df, args.output_dir)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
