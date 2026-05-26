#!/usr/bin/env python3
"""Preview a clustered-by-subitem subgroup bar layout.

This is a temporary review artifact only. It does not overwrite the official
`subgroup_family_subitem_bars_appendix` bundle.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EVAL_ROOT = PROJECT_ROOT / "Evaluation" / "query_fivepart_breakdown" / "subgroup_breakdown"
INPUT_CSV = EVAL_ROOT / "data" / "model_subitem_heatmap.csv"
OUTPUT_ROOT = EVAL_ROOT / "review_clustered_subitem_bars"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"

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
MODEL_COLORS = {
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


def _ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, DATA_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _write_include_tex(path: Path, title: str, pdf_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                r"\documentclass[border=4pt]{standalone}",
                r"\usepackage{graphicx}",
                r"\begin{document}",
                rf"\textbf{{{title}}}\\[0.5em]",
                rf"\includegraphics[width=\textwidth]{{{pdf_name}}}",
                r"\end{document}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_preview() -> dict[str, Path]:
    _ensure_dirs()
    heatmap_df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    plot_df = heatmap_df.loc[heatmap_df["subitem_id"] != "family_mean"].copy().reset_index(drop=True)
    if plot_df.empty:
        raise RuntimeError("No subgroup subitem rows available for preview.")

    cluster_gap = 1.35
    intra_gap = 0.10
    bar_width = 0.90

    preview_rows: list[dict[str, object]] = []
    x_positions: list[float] = []
    x_labels: list[str] = []
    x_values: list[float] = []
    x_colors: list[str] = []
    cluster_centers: list[tuple[float, str]] = []
    separators: list[float] = []

    cursor = 0.0
    for cluster_index, row in enumerate(plot_df.itertuples(index=False)):
        cluster_start = cursor
        for model_id in MODEL_ORDER:
            score = getattr(row, model_id, None)
            if score is None or pd.isna(score):
                continue
            x_positions.append(cursor)
            x_labels.append(MODEL_LABELS.get(model_id, model_id))
            x_values.append(float(score))
            x_colors.append(MODEL_COLORS.get(model_id, "#777777"))
            preview_rows.append(
                {
                    "cluster_index": cluster_index,
                    "subitem_id": row.subitem_id,
                    "subitem_label": row.subitem_label,
                    "model_id": model_id,
                    "model_label": MODEL_LABELS.get(model_id, model_id),
                    "x": round(cursor, 6),
                    "score": round(float(score), 6),
                }
            )
            cursor += 1.0 + intra_gap
        cluster_end = cursor - (1.0 + intra_gap)
        cluster_centers.append(((cluster_start + cluster_end) / 2.0, str(row.subitem_label)))
        if cluster_index < len(plot_df) - 1:
            separators.append(cursor - intra_gap / 2.0 + cluster_gap / 2.0)
            cursor += cluster_gap

    preview_df = pd.DataFrame(preview_rows)
    preview_csv = DATA_DIR / "subgroup_clustered_preview_positions.csv"
    preview_df.to_csv(preview_csv, index=False, encoding="utf-8-sig")

    pdf_path = FIG_DIR / "subgroup_clustered_subitem_bars_preview.pdf"
    png_path = FIG_DIR / "subgroup_clustered_subitem_bars_preview.png"
    svg_path = FIG_DIR / "subgroup_clustered_subitem_bars_preview.svg"
    tex_path = FIG_DIR / "subgroup_clustered_subitem_bars_preview.tex"

    fig, ax = plt.subplots(figsize=(15.8, 7.8))
    ax.bar(x_positions, x_values, width=bar_width, color=x_colors, edgecolor=x_colors, linewidth=0.8)
    for sep in separators:
        ax.axvline(sep, color="#555555", linestyle="--", linewidth=1.05, alpha=0.7)
    for center_x, label in cluster_centers:
        ax.text(center_x, 1.035, label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Subgroup breakdown preview: cluster by subitem")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, ha="center", va="top", fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    _write_include_tex(tex_path, "Subgroup clustered-by-subitem preview", pdf_path.name)
    return {
        "preview_csv": preview_csv,
        "pdf": pdf_path,
        "png": png_path,
        "svg": svg_path,
        "tex": tex_path,
    }


if __name__ == "__main__":
    outputs = build_preview()
    for key, value in outputs.items():
        print(f"{key}: {value}")
