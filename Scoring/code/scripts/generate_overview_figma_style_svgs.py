from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "DejaVu Serif"

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path("/Users/jialinzhang/Documents/HKUNAISS/SyntheticNips/SQLagent")
DATA_DIR = ROOT / "Evaluation" / "overview_regenerated" / "data"
OUT_DIR = ROOT / "Evaluation" / "overview_regenerated" / "figures"

DISTANCE_SOURCE = DATA_DIR / "overview_distance_panel_source.csv"
QUERY_SOURCE = DATA_DIR / "overview_sql_panel_source.csv"

DISTANCE_TITLE = "Distance-Based Fidelity"
QUERY_TITLE = "Workload-Grounded Query Fidelity"

DISTANCE_COLOR = "#ff5a1f"
QUERY_COLOR = "#4b4be0"
GRID_COLOR = "#ebebeb"
FRAME_COLOR = "#d8d8d8"
SEPARATOR_COLOR = "#7f7f7f"
TEXT_COLOR = "#444444"


def _panel_groups(source_csv: Path) -> tuple[list[str], list[str], dict[str, list[dict[str, object]]]]:
    df = pd.read_csv(source_csv)
    axis_order = (
        df[["axis_order", "axis_label"]]
        .drop_duplicates()
        .sort_values("axis_order")
    )
    model_order = (
        df[["panel_model_order", "model_id"]]
        .drop_duplicates()
        .sort_values("panel_model_order")
    )
    axes = axis_order["axis_label"].tolist()
    models = model_order["model_id"].tolist()
    grouped: dict[str, list[dict[str, object]]] = {}
    for axis in axes:
        subset = df[df["axis_label"] == axis].sort_values("panel_model_order")
        grouped[axis] = subset.to_dict("records")
    return axes, models, grouped


def _draw_panel(
    source_csv: Path,
    title: str,
    title_color: str,
    out_svg: Path,
    *,
    width: float,
    height: float,
) -> None:
    axes, models, grouped = _panel_groups(source_csv)
    n_panels = len(axes)

    fig, axs = plt.subplots(
        1,
        n_panels,
        figsize=(width, height),
        sharey=True,
        gridspec_kw={"wspace": 0.08, "left": 0.055, "right": 0.985, "top": 0.77, "bottom": 0.20},
    )
    if n_panels == 1:
        axs = [axs]

    fig.patch.set_facecolor("white")
    # Outer editable frame
    fig.add_artist(
        Rectangle(
            (0.006, 0.01),
            0.988,
            0.98,
            transform=fig.transFigure,
            fill=False,
            linewidth=1.0,
            edgecolor=FRAME_COLOR,
        )
    )
    fig.text(
        0.035,
        0.90,
        title,
        ha="left",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=title_color,
    )

    for idx, (ax, axis_label) in enumerate(zip(axs, axes, strict=True)):
        rows = grouped[axis_label]
        x = list(range(len(rows)))
        values = [float(r["axis_value"]) * 100.0 for r in rows]
        colors = [str(r["model_color"]) for r in rows]
        labels = [str(r["model_label"]) for r in rows]

        ax.bar(x, values, color=colors, width=0.52, edgecolor="none")
        ax.set_ylim(0, 120)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_xticks([])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="y", labelsize=9, colors=TEXT_COLOR, length=0)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)

        ax.text(
            0.5,
            0.98,
            axis_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            color=title_color,
        )

        for xi, val in zip(x, values, strict=True):
            ax.text(
                xi,
                val + 2.3,
                f"{int(round(val))}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TEXT_COLOR,
            )

        # Remove subplot box; keep only separators between groups.
        for spine in ax.spines.values():
            spine.set_visible(False)

        if idx < n_panels - 1:
            bbox = ax.get_position()
            xline = bbox.x1 + 0.006
            fig.add_artist(
                Line2D(
                    [xline, xline],
                    [bbox.y0 - 0.02, bbox.y1 + 0.01],
                    transform=fig.transFigure,
                    color=SEPARATOR_COLOR,
                    linewidth=1.0,
                )
            )

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", transparent=False)
    plt.close(fig)


def main() -> None:
    _draw_panel(
        DISTANCE_SOURCE,
        DISTANCE_TITLE,
        DISTANCE_COLOR,
        OUT_DIR / "overview_distance_panel_figma_style.svg",
        width=12.0,
        height=3.4,
    )
    _draw_panel(
        QUERY_SOURCE,
        QUERY_TITLE,
        QUERY_COLOR,
        OUT_DIR / "overview_query_panel_figma_style.svg",
        width=15.0,
        height=3.5,
    )


if __name__ == "__main__":
    main()
