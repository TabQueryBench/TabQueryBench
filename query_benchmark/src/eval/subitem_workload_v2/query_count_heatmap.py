from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAMILY_ORDER = ["subgroup", "conditional", "tail", "missingness", "cardinality"]


def natural_dataset_key(dataset_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", dataset_id)
    if not match:
        return (dataset_id.lower(), -1)
    return (match.group(1).lower(), int(match.group(2)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a dataset-by-family query count heatmap for subitem workload v2."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_matrix(input_csv: Path) -> tuple[list[str], np.ndarray]:
    rows = list(csv.DictReader(input_csv.open("r", encoding="utf-8", newline="")))
    datasets = sorted({row["dataset_id"] for row in rows}, key=natural_dataset_key)
    dataset_index = {dataset_id: idx for idx, dataset_id in enumerate(datasets)}
    family_index = {family: idx for idx, family in enumerate(FAMILY_ORDER)}
    matrix = np.zeros((len(datasets), len(FAMILY_ORDER)), dtype=int)

    for row in rows:
        dataset_id = row["dataset_id"]
        family = row["family_label"]
        if family not in family_index:
            continue
        matrix[dataset_index[dataset_id], family_index[family]] = int(row["query_count"])
    return datasets, matrix


def render_heatmap(datasets: list[str], matrix: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_height = max(10.5, 0.26 * len(datasets) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")

    ax.set_xticks(np.arange(len(FAMILY_ORDER)))
    ax.set_xticklabels(
        ["Subgroup", "Conditional", "Tail", "Missingness", "Cardinality"],
        fontsize=11,
    )
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=8)
    ax.set_xlabel("Query family", fontsize=12)
    ax.set_ylabel("Dataset", fontsize=12)
    ax.set_title("Accepted query counts by dataset and family (v2)", fontsize=13, pad=12)

    max_value = int(matrix.max()) if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            text_color = "white" if max_value and value >= 0.58 * max_value else "black"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.024, pad=0.02)
    colorbar.ax.set_ylabel("Accepted query count", rotation=90, fontsize=10)

    ax.set_xticks(np.arange(-0.5, len(FAMILY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    png_path = output_dir / "dataset_family_query_count_heatmap.png"
    pdf_path = output_dir / "dataset_family_query_count_heatmap.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    max_dataset_idx = int(np.argmax(matrix.sum(axis=1))) if matrix.size else 0
    summary_path = output_dir / "dataset_family_query_count_heatmap.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Dataset-Family Query Count Heatmap (v2)",
                "",
                f"- Source CSV: `{output_dir / 'dataset_family_query_token_summary.csv'}`",
                f"- Datasets covered: `{len(datasets)}`",
                f"- Families shown: `{len(FAMILY_ORDER)}`",
                f"- Highest total-query dataset: `{datasets[max_dataset_idx]}` with `{int(matrix.sum(axis=1)[max_dataset_idx])}` accepted queries",
                "- Each heatmap cell shows the accepted query count for one dataset-family pair.",
                "- Colors encode relative query density; numeric labels give the exact counts.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    datasets, matrix = load_matrix(args.input_csv.resolve())
    render_heatmap(datasets, matrix, args.output_dir.resolve())


if __name__ == "__main__":
    main()
