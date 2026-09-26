#!/usr/bin/env python3
"""Build slide-ready C2 capability heatmap.

Column policy:
- Overall distance-based fidelity: fixed values from reference slide.
- Subgroup / Conditional / Tail / Cardinality: from latest model_scores_c2.csv (primary workload).
- Missingness: forced to 1.0 for all models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch


REFERENCE_OVERALL = {
    "realtabformer": 0.9921,
    "bayesnet": 0.9890,
    "arf": 0.9937,
    "tvae": 0.9901,
    "ctgan": 0.9598,
    "tabpfgen": 0.8777,
    "tabddpm": 0.0235,
}

ROW_ORDER = [
    "realtabformer",
    "bayesnet",
    "arf",
    "tvae",
    "ctgan",
    "tabpfgen",
    "tabddpm",
]

ROW_LABEL = {
    "realtabformer": "RealTabFormer",
    "bayesnet": "Bayes\nNet",
    "arf": "ARF",
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "tabpfgen": "Tab\nPFGen",
    "tabddpm": "Tab\nDDPM",
}

COLUMN_LABELS = [
    "Overall\ndistance-based\nfidelity",
    "Subgroup\nstructure",
    "Conditional\ndependency",
    "Tail /\nrarity",
    "Missingness",
    "Cardinality\nstructure",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_latest_family_scores(experiment_dir: Path) -> dict[str, dict[str, float]]:
    ranking = _read_json(experiment_dir / "overall_ranking_c2.json")
    primary_run = str(ranking.get("primary_workload_run_id"))
    rows = _read_csv(experiment_dir / "model_scores_c2.csv")
    rows = [r for r in rows if str(r.get("workload_run_id")) == primary_run]

    out: dict[str, dict[str, float]] = {}
    for r in rows:
        model = str(r.get("model_id") or "").strip().lower()
        out[model] = {
            "subgroup": float(r.get("subgroup_structure_score") or 0.0),
            "conditional": float(r.get("conditional_dependency_structure_score") or 0.0),
            "tail": float(r.get("tail_rarity_structure_score") or 0.0),
            "missing": 1.0,  # requested override
            "cardinality": float(r.get("cardinality_structure_score") or 0.0),
        }
    return out


def build_capability_matrix(experiment_dir: Path) -> tuple[list[list[float]], list[str]]:
    family_scores = _load_latest_family_scores(experiment_dir)
    matrix: list[list[float]] = []
    labels: list[str] = []

    for model in ROW_ORDER:
        if model not in family_scores:
            continue
        row = [
            REFERENCE_OVERALL[model],
            family_scores[model]["subgroup"],
            family_scores[model]["conditional"],
            family_scores[model]["tail"],
            1.0,
            family_scores[model]["cardinality"],
        ]
        matrix.append(row)
        labels.append(ROW_LABEL[model])
    return matrix, labels


def plot_heatmap(matrix: list[list[float]], row_labels: list[str], output_path: Path) -> None:
    # Visual style tuned to match the provided reference slide:
    # light-gray canvas, rounded score tiles, muted brown->green palette, dark navy typography.
    bg_color = "#ECECF1"
    title_color = "#1F243A"
    text_color = "#1F243A"

    n_rows = len(matrix)
    n_cols = len(COLUMN_LABELS)
    fig, ax = plt.subplots(figsize=(15.8, 9.2), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Tile geometry (manual layout for better control).
    cell_w = 0.9
    cell_h = 0.62
    gap_x = 0.12
    gap_y = 0.18
    left_pad_for_labels = 1.2
    top_pad = 1.3

    grid_w = n_cols * cell_w + (n_cols - 1) * gap_x
    grid_h = n_rows * cell_h + (n_rows - 1) * gap_y

    xmin = -left_pad_for_labels
    xmax = grid_w + 0.2
    ymin = -0.2
    ymax = grid_h + top_pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    ax.axis("off")

    # Custom palette: low values in muted tan, high values in rich green.
    cmap = colors.LinearSegmentedColormap.from_list(
        "tan_to_green",
        ["#A8895C", "#8FA96B", "#67B86E"],
    )
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    # Draw column headers.
    for j, col_label in enumerate(COLUMN_LABELS):
        x = j * (cell_w + gap_x) + cell_w / 2
        y = 0.35
        ax.text(
            x,
            y,
            col_label,
            ha="center",
            va="center",
            fontsize=19,
            fontweight="bold",
            color=text_color,
            linespacing=1.1,
        )

    # Draw row labels and rounded tiles.
    y0 = top_pad
    for i, row in enumerate(matrix):
        y = y0 + i * (cell_h + gap_y)
        ax.text(
            -0.15,
            y + cell_h / 2,
            row_labels[i],
            ha="right",
            va="center",
            fontsize=22,
            fontweight="bold",
            color=text_color,
        )

        for j, value in enumerate(row):
            x = j * (cell_w + gap_x)
            tile_color = cmap(norm(value))
            patch = FancyBboxPatch(
                (x, y),
                cell_w,
                cell_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=0,
                facecolor=tile_color,
            )
            ax.add_patch(patch)
            ax.text(
                x + cell_w / 2,
                y + cell_h / 2,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="#17313b" if value >= 0.42 else "white",
            )

    # Vertical separator between column 1 and column 2.
    split_x = cell_w + gap_x / 2
    ax.plot(
        [split_x, split_x],
        [y0 - 0.05, y0 + grid_h + 0.02],
        color="#252A3F",
        linewidth=4,
        solid_capstyle="round",
    )

    ax.set_title(
        "Model Capability Heatmap: Categorical (C2)",
        fontsize=43,
        fontweight="bold",
        color=title_color,
        pad=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(output_path, dpi=240, facecolor=bg_color, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot slide-ready C2 capability heatmap.")
    parser.add_argument("--experiment-dir", type=Path, required=True, help="c2 experiment directory.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output png path. Default: <experiment-dir>/figures/08_model_capability_heatmap_c2.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path
        else (experiment_dir / "figures" / "08_model_capability_heatmap_c2.png")
    )

    matrix, labels = build_capability_matrix(experiment_dir)
    plot_heatmap(matrix, labels, output_path)
    print(json.dumps({"status": "ok", "output_path": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
