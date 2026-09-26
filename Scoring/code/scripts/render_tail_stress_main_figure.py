#!/usr/bin/env python3
"""Render a paper-ready multi-panel tail stress figure from summary CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors


THRESHOLD_ORDER = ["10%", "8%", "6%", "4%", "2%", "1%", "0.5%", "0.1%", "0.01%", "0.001%"]

# Frozen paper color convention from README.md.
MODEL_COLORS = {
    "RealTabFormer": "#332288",
    "TVAE": "#4477AA",
    "ForestDiffusion": "#228833",
    "TabDDPM": "#EE7733",
    "TabSyn": "#66CCEE",
    "TabDiff": "#AA3377",
    "CTGAN": "#EE6677",
    "ARF": "#777777",
    "BayesNet": "#CCBB44",
    "TabPFGen": "#009988",
    "TabbyFlow": "#882255",
}

TAIL_COLOR = "#E4572E"
HEAD_COLOR = "#4C78A8"
DECOMP_COLORS = {
    "tail_set_consistency_mean": "#D1495B",
    "tail_mass_similarity_mean": "#2A9D8F",
    "tail_concentration_consistency_mean": "#6D597A",
}
DECOMP_LABELS = {
    "tail_set_consistency_mean": "Tail set consistency",
    "tail_mass_similarity_mean": "Tail mass similarity",
    "tail_concentration_consistency_mean": "Tail concentration consistency",
}
MODEL_LABEL_OFFSETS = {
    "ARF": (6, 8),
    "BayesNet": (6, -12),
    "CTGAN": (6, -10),
    "ForestDiffusion": (6, -10),
    "RealTabFormer": (6, 8),
    "TVAE": (6, 8),
    "TabDDPM": (6, -10),
    "TabDiff": (6, 8),
    "TabPFGen": (6, 8),
    "TabSyn": (6, 6),
    "TabbyFlow": (6, -10),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        required=True,
        help="Directory containing tail-threshold summary CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the PNG/PDF figure will be written.",
    )
    return parser


def _read_required_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    file_map = {
        "global": "global_threshold_summary.csv",
        "model_fragility": "model_fragility_summary.csv",
        "model_threshold": "model_threshold_summary.csv",
        "prefix_threshold": "prefix_threshold_summary.csv",
        "dataset_threshold": "dataset_threshold_summary.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, name in file_map.items():
        path = tables_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required input CSV: {path}")
        tables[key] = pd.read_csv(path)
    return tables


def _ordered(df: pd.DataFrame, label_col: str = "threshold_label") -> pd.DataFrame:
    ordered = df.copy()
    ordered[label_col] = pd.Categorical(ordered[label_col], categories=THRESHOLD_ORDER, ordered=True)
    ordered = ordered.sort_values(label_col).reset_index(drop=True)
    return ordered


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10.5,
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
    mixed = base * (1.0 - strength) + white * strength
    return tuple(float(v) for v in mixed)


def _shade_ultra_tail(ax: plt.Axes) -> None:
    ax.axvspan(6.5, 9.5, color="#EFEFEF", alpha=1.0, zorder=0)
    ax.axvline(6.5, color="#999999", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        7.95,
        0.985,
        "Low-support\nultra-tail",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )


def _render_panel_a(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    x = list(range(len(global_df)))
    tail = global_df["tail_overall_mean"].tolist()
    head = global_df["head_proxy_mean"].tolist()

    _shade_ultra_tail(ax)
    ax.plot(x, tail, color=TAIL_COLOR, marker="o", linewidth=2.0, markersize=4.4, label="Tail score", zorder=3)
    ax.plot(x, head, color=HEAD_COLOR, marker="o", linewidth=2.0, markersize=4.4, label="Head proxy score", zorder=3)

    ax.set_xticks(x, global_df["threshold_label"].tolist(), rotation=30, ha="right")
    ax.set_ylim(0.20, 0.60)
    ax.set_ylabel("Score")
    ax.set_xlabel("Tail threshold")
    ax.set_title("A. Tail degrades while head remains stable", loc="left", pad=6)
    _style_axis(ax)

    ax.text(
        0.03,
        0.53,
        f"Tail: {tail[0]:.3f} -> {tail[6]:.3f}",
        transform=ax.transAxes,
        color=TAIL_COLOR,
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
    )
    ax.text(
        0.03,
        0.45,
        f"Head: {head[0]:.3f} -> {head[-1]:.3f}",
        transform=ax.transAxes,
        color=HEAD_COLOR,
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
    )
    ax.legend(loc="lower left", frameon=False, ncols=1, bbox_to_anchor=(0.005, 0.005))


def _render_panel_b(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    x = list(range(len(global_df)))
    baseline = global_df.iloc[0]

    for metric in (
        "tail_set_consistency_mean",
        "tail_mass_similarity_mean",
        "tail_concentration_consistency_mean",
    ):
        base_value = float(baseline[metric])
        values = [float(v) / base_value if base_value else 0.0 for v in global_df[metric].tolist()]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            markersize=4.2,
            color=DECOMP_COLORS[metric],
            label=DECOMP_LABELS[metric],
        )

    _shade_ultra_tail(ax)
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, global_df["threshold_label"].tolist(), rotation=30, ha="right")
    ax.set_ylim(0.60, 1.12)
    ax.set_ylabel("Relative score vs. 10% threshold")
    ax.set_xlabel("Tail threshold")
    ax.set_title("B. Tail set consistency and tail mass similarity break first", loc="left", pad=6)
    _style_axis(ax)
    ax.legend(loc="lower left", frameon=False)


def _render_panel_c(ax: plt.Axes, model_threshold_df: pd.DataFrame) -> None:
    ordered = _ordered(model_threshold_df)
    ordered = ordered[ordered["model_label"].isin(MODEL_COLORS)].copy()
    model_labels = [label for label in MODEL_COLORS if label in set(ordered["model_label"].dropna().unique().tolist())]
    shade_strengths = np.linspace(0.0, 0.78, len(THRESHOLD_ORDER))

    x_all: list[float] = []
    y_all: list[float] = []

    for label in model_labels:
        subset = ordered[ordered["model_label"] == label].copy()
        subset = subset.dropna(subset=["tail_set_consistency", "tail_mass_similarity"])
        if subset.empty:
            continue
        x_vals = subset["tail_set_consistency"].astype(float).tolist()
        y_vals = subset["tail_mass_similarity"].astype(float).tolist()
        x_all.extend(x_vals)
        y_all.extend(y_vals)

        base_color = MODEL_COLORS[label]
        ax.plot(x_vals, y_vals, color=base_color, linewidth=1.0, alpha=0.38, zorder=1)

        for idx, (_, row) in enumerate(subset.iterrows()):
            color = _blend_with_white(base_color, float(shade_strengths[idx]))
            size = 36 if idx == 0 else 28
            ax.scatter(
                [float(row["tail_set_consistency"])],
                [float(row["tail_mass_similarity"])],
                s=size,
                color=color,
                edgecolor="white",
                linewidth=0.55,
                zorder=2 + idx * 0.01,
            )

        first = subset.iloc[0]
        dx, dy = MODEL_LABEL_OFFSETS.get(label, (6, 6))
        ax.annotate(
            label,
            (float(first["tail_set_consistency"]), float(first["tail_mass_similarity"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.3,
            ha="left",
            va="center",
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.2},
            zorder=4,
        )

    if not x_all or not y_all:
        ax.text(0.5, 0.5, "No paper-roster models available", ha="center", va="center", fontsize=9, color="#666666")
        ax.set_axis_off()
        return

    ax.set_xlim(max(0.0, min(x_all) - 0.02), min(1.0, max(x_all) + 0.06))
    ax.set_ylim(max(0.0, min(y_all) - 0.03), min(1.0, max(y_all) + 0.06))
    ax.set_xlabel("Tail set consistency score")
    ax.set_ylabel("Tail mass similarity score")
    ax.set_title("C. Tail set consistency and tail mass similarity erode together", loc="left", pad=6)
    _style_axis(ax)

    threshold_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_blend_with_white("#666666", strength),
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5,
            label=label,
        )
        for strength, label in [
            (shade_strengths[0], "10%"),
            (shade_strengths[5], "1%"),
            (shade_strengths[-1], "0.001%"),
        ]
    ]
    legend = ax.legend(
        handles=threshold_handles,
        title="Threshold shading",
        loc="lower right",
        frameon=False,
        ncols=3,
        bbox_to_anchor=(1.0, 1.02),
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    legend.get_title().set_fontsize(7.8)
    ax.add_artist(legend)
    ax.text(
        0.995,
        0.03,
        "Same hue = same model\nlighter points = rarer threshold",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#666666",
    )


def render_main_figure(tables_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    tables = _read_required_tables(tables_dir)
    global_df = _ordered(tables["global"])

    _configure_style()
    fig = plt.figure(figsize=(7.1, 3.55), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)
    mosaic = fig.subplot_mosaic(
        [["T", "T"], ["A", "B"]],
        height_ratios=[0.18, 1.0],
    )

    mosaic["T"].axis("off")
    mosaic["T"].text(
        0.0,
        0.72,
        "Tail stress testing reveals rare-event fragility",
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
    )

    _render_panel_a(mosaic["A"], global_df)
    _render_panel_b(mosaic["B"], global_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "tail_stress_main_figure.png"
    pdf_path = output_dir / "tail_stress_main_figure.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    png_path, pdf_path = render_main_figure(args.tables_dir, args.output_dir)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
