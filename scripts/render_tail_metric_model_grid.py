#!/usr/bin/env python3
"""Render six model-by-threshold metric figures for tail submetrics."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
            "axes.titlesize": 10.0,
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


def _blend_with_white(color: str, strength: float) -> tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    mixed = base * (1.0 - strength) + white * strength
    return tuple(float(v) for v in mixed)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    pivot = (
        df.pivot_table(index="model_label", columns="threshold_label", values=metric, aggfunc="mean")
        .reindex(index=model_order, columns=THRESHOLD_ORDER)
        .astype(float)
    )
    return pivot


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), dpi=300, facecolor="white")
    plt.close(fig)


def _threshold_legend_handles() -> list[Patch]:
    samples = [("10%", 0.00), ("2%", 0.30), ("0.5%", 0.55), ("0.001%", 0.80)]
    return [
        Patch(facecolor=_blend_with_white("#666666", strength), edgecolor="none", label=label)
        for label, strength in samples
    ]


def render_errorbar_style(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    model_order = _model_order(df)
    x = np.arange(len(model_order))
    outputs: list[Path] = []

    for metric, ylabel in METRICS:
        pivot = _metric_pivot(df, metric, model_order)
        values = pivot.to_numpy(dtype=float)
        means = np.nanmean(values, axis=1)
        lows = means - np.nanmin(values, axis=1)
        highs = np.nanmax(values, axis=1) - means
        q25 = np.nanquantile(values, 0.25, axis=1)
        q75 = np.nanquantile(values, 0.75, axis=1)

        fig, ax = plt.subplots(figsize=(8.3, 4.6), constrained_layout=True)
        for idx, model in enumerate(model_order):
            base = MODEL_COLORS[model]
            # IQR band
            ax.add_patch(
                plt.Rectangle(
                    (x[idx] - 0.26, q25[idx]),
                    0.52,
                    max(1e-6, q75[idx] - q25[idx]),
                    facecolor=_blend_with_white(base, 0.55),
                    edgecolor="none",
                    alpha=0.85,
                    zorder=1,
                )
            )
            ax.errorbar(
                x[idx],
                means[idx],
                yerr=np.array([[lows[idx]], [highs[idx]]]),
                fmt="s",
                color=base,
                markersize=4.8,
                elinewidth=1.7,
                capsize=4,
                capthick=1.2,
                zorder=3,
            )

        ax.set_xticks(x, model_order, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_ylim(0.0, min(1.0, np.nanmax(values) + 0.08))
        ax.set_title(f"{ylabel}: threshold sweep summarized as mean + range", loc="left", pad=6)
        _style_axis(ax)

        legend_items = [
            Line2D([0], [0], marker="s", color="#555555", linestyle="none", markersize=5, label="Mean over thresholds"),
            Line2D([0], [0], color="#555555", linewidth=1.7, label="Min-max over thresholds"),
            Patch(facecolor="#D9D9D9", edgecolor="none", label="IQR over thresholds"),
        ]
        ax.legend(handles=legend_items, loc="upper left", frameon=False, ncols=3)
        ax.text(
            0.995,
            0.02,
            "Each model summarizes the full 10-threshold sweep",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="#666666",
        )

        out = output_dir / f"{metric}__errorbar_summary.png"
        _save(fig, out)
        outputs.append(out)

    return outputs


def render_layered_bar_style(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    model_order = _model_order(df)
    x = np.arange(len(model_order))
    shade_strengths = np.linspace(0.00, 0.82, len(THRESHOLD_ORDER))
    outputs: list[Path] = []

    for metric, ylabel in METRICS:
        pivot = _metric_pivot(df, metric, model_order)
        fig, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)

        for idx, model in enumerate(model_order):
            base = MODEL_COLORS[model]
            column = pivot.loc[model].to_list()
            # draw from light to dark so broader/taller bars remain visible
            for threshold_idx in range(len(THRESHOLD_ORDER) - 1, -1, -1):
                value = column[threshold_idx]
                if pd.isna(value):
                    continue
                strength = float(shade_strengths[threshold_idx])
                ax.bar(
                    x[idx],
                    float(value),
                    width=0.72,
                    color=_blend_with_white(base, strength),
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=2 + threshold_idx * 0.01,
                )

        ax.set_xticks(x, model_order, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_ylim(0.0, min(1.0, float(np.nanmax(pivot.to_numpy(dtype=float))) + 0.08))
        ax.set_title(f"{ylabel}: layered threshold bars", loc="left", pad=6)
        _style_axis(ax)
        legend = ax.legend(
            handles=_threshold_legend_handles(),
            title="Threshold shading",
            loc="upper left",
            frameon=False,
            ncols=4,
        )
        legend.get_title().set_fontsize(8)
        ax.text(
            0.995,
            0.02,
            "Darker bars = broader tail threshold; lighter bars = rarer tail threshold",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="#666666",
        )

        out = output_dir / f"{metric}__layered_bars.png"
        _save(fig, out)
        outputs.append(out)

    return outputs


def main() -> int:
    args = _build_parser().parse_args()
    _configure_style()
    df = _load_model_threshold(args.tables_dir)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    outputs.extend(render_errorbar_style(df, output_dir))
    outputs.extend(render_layered_bar_style(df, output_dir))
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
