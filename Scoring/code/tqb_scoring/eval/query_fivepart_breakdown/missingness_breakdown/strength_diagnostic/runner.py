#!/usr/bin/env python3
"""Paper-facing auxiliary diagnostic for missingness relation-strength fidelity."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_scoring.eval.query_fivepart_breakdown.common_final import render_final_readme, sync_final_outputs
from tqb_scoring.eval.query_fivepart_breakdown.common_heatmap_palette import get_heatmap_cmap

MISSINGNESS_ROOT = PROJECT_ROOT.parent / "results" / "query_fivepart_breakdown" / "missingness_breakdown"
INPUT_DATA_DIR = MISSINGNESS_ROOT / "data"
OUTPUT_ROOT = MISSINGNESS_ROOT / "strength_diagnostic"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"
FINAL_DIR = OUTPUT_ROOT / "final"

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
PROFILE_COLOR = "#E76F51"
STRENGTH_COLOR = "#2A9D8F"


def _ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, DATA_DIR, FIG_DIR, FINAL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _dataset_sort_key(dataset_id: str) -> tuple[int, int, str]:
    text = str(dataset_id or "").strip()
    if len(text) < 2 or not text[1:].isdigit():
        return (99, 10**9, text)
    prefix_order = {"c": 0, "m": 1, "n": 2}.get(text[0].lower(), 50)
    return (prefix_order, int(text[1:]), text)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


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


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_model_df = pd.read_csv(INPUT_DATA_DIR / "dataset_model_scores.csv", encoding="utf-8-sig")
    model_summary_df = pd.read_csv(INPUT_DATA_DIR / "model_summary.csv", encoding="utf-8-sig")
    for column in [
        "co_missingness_pattern_consistency",
        "co_missing_strength_score",
        "co_missing_composite_score",
        "profile_minus_strength",
    ]:
        if column in dataset_model_df.columns:
            dataset_model_df[column] = pd.to_numeric(dataset_model_df[column], errors="coerce")
    return dataset_model_df, model_summary_df


def _build_dataset_model_strength_df(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    df = dataset_model_df.copy()
    df = df.loc[df["co_missing_strength_score"].notna()].copy()
    df["strength_minus_profile"] = df["co_missing_strength_score"] - df["co_missingness_pattern_consistency"]
    df["dataset_sort"] = df["dataset_id"].map(_dataset_sort_key)
    df["model_order"] = df["model_id"].map({model_id: idx for idx, model_id in enumerate(MODEL_ORDER)})
    df = df.sort_values(["dataset_sort", "model_order"]).drop(columns=["dataset_sort", "model_order"]).reset_index(drop=True)
    return df[
        [
            "dataset_id",
            "dataset_prefix",
            "model_id",
            "model_label",
            "co_missingness_pattern_consistency",
            "co_missing_strength_score",
            "co_missing_composite_score",
            "profile_minus_strength",
            "strength_minus_profile",
            "asset_count",
            "applicable_asset_count",
        ]
    ]


def _build_model_strength_summary(dataset_model_strength_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        subset = dataset_model_strength_df.loc[dataset_model_strength_df["model_id"] == model_id].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "dataset_count": int(subset["dataset_id"].nunique()),
                "panel_count": int(subset.shape[0]),
                "profile_score_mean": round(float(subset["co_missingness_pattern_consistency"].mean()), 6),
                "strength_score_mean": round(float(subset["co_missing_strength_score"].mean()), 6),
                "composite_score_mean": round(float(subset["co_missing_composite_score"].mean()), 6),
                "strength_minus_profile_mean": round(float(subset["strength_minus_profile"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def _plot_model_dumbbell(summary_df: pd.DataFrame, pdf_path: Path, png_path: Path, svg_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 6.8))
    y_positions = list(range(len(summary_df)))
    for idx, row in enumerate(summary_df.itertuples()):
        model_id = str(row.model_id)
        color = MODEL_COLORS.get(model_id, "#777777")
        profile = float(row.profile_score_mean)
        strength = float(row.strength_score_mean)
        ax.plot([profile, strength], [idx, idx], color=color, linewidth=2.2, alpha=0.95)
        ax.scatter(profile, idx, s=70, facecolors="white", edgecolors=color, linewidth=1.8, zorder=3)
        ax.scatter(strength, idx, s=70, facecolors=color, edgecolors=color, marker="s", linewidth=1.0, zorder=4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary_df["model_label"])
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Mean score over applicable dataset-model panels")
    ax.set_title("Missingness auxiliary insight: profile vs strength")
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.01, 1.01, "Hollow circle = canonical profile-only score; filled square = strength-only score", transform=ax.transAxes, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _plot_gap_bars(summary_df: pd.DataFrame, pdf_path: Path, png_path: Path, svg_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    colors = [MODEL_COLORS.get(str(model_id), "#777777") for model_id in summary_df["model_id"]]
    values = pd.to_numeric(summary_df["strength_minus_profile_mean"], errors="coerce").fillna(0.0)
    ax.bar(range(len(summary_df)), values, color=colors, edgecolor=colors)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(range(len(summary_df)))
    ax.set_xticklabels(summary_df["model_label"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Strength minus profile")
    ax.set_title("How much relation-strength fidelity differs from canonical profile fidelity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _plot_strength_heatmap(dataset_model_strength_df: pd.DataFrame, pdf_path: Path, png_path: Path, svg_path: Path) -> None:
    pivot = (
        dataset_model_strength_df.pivot_table(
            index="dataset_id",
            columns="model_id",
            values="co_missing_strength_score",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    ordered_models = [model_id for model_id in MODEL_ORDER if model_id in pivot.columns]
    pivot["dataset_sort"] = pivot["dataset_id"].map(_dataset_sort_key)
    pivot = pivot.sort_values(["dataset_sort", "dataset_id"]).drop(columns=["dataset_sort"]).reset_index(drop=True)
    matrix = pivot[ordered_models].to_numpy(dtype=float)
    fig_height = max(5.8, 0.35 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(9.8, fig_height))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap=get_heatmap_cmap())
    ax.set_xticks(range(len(ordered_models)))
    ax.set_xticklabels([_model_label(model_id) for model_id in ordered_models], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot["dataset_id"], fontsize=7)
    ax.set_title("Dataset-model heatmap of missingness relation-strength fidelity")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _draw_distribution_summary(
    ax: plt.Axes,
    center: float,
    values: list[float],
    color: str,
    offset: float,
    box_width: float = 0.18,
) -> None:
    cleaned = [float(value) for value in values if pd.notna(value)]
    if not cleaned:
        return
    arr = np.asarray(cleaned, dtype=float)
    mean = float(arr.mean())
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    ymin = float(arr.min())
    ymax = float(arr.max())
    xpos = center + offset
    ax.vlines(xpos, ymin, ymax, color=color, linewidth=1.1, alpha=0.95, zorder=2)
    ax.hlines([ymin, ymax], xpos - box_width * 0.26, xpos + box_width * 0.26, color=color, linewidth=1.1, alpha=0.95, zorder=2)
    ax.add_patch(
        Rectangle(
            (xpos - box_width / 2, q1),
            box_width,
            max(0.0, q3 - q1),
            facecolor=color,
            edgecolor="none",
            alpha=0.24,
            zorder=1,
        )
    )
    ax.scatter([xpos], [mean], s=28, marker="s", facecolor=color, edgecolor=color, linewidth=0.8, zorder=3)


def _plot_profile_strength_distribution(dataset_model_strength_df: pd.DataFrame, pdf_path: Path, png_path: Path, svg_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    x_positions = list(range(len(MODEL_ORDER)))
    profile_offset = -0.17
    strength_offset = 0.17

    for idx, model_id in enumerate(MODEL_ORDER):
        subset = dataset_model_strength_df.loc[dataset_model_strength_df["model_id"] == model_id].copy()
        if subset.empty:
            continue
        _draw_distribution_summary(
            ax,
            center=float(idx),
            values=subset["co_missingness_pattern_consistency"].tolist(),
            color=PROFILE_COLOR,
            offset=profile_offset,
        )
        _draw_distribution_summary(
            ax,
            center=float(idx),
            values=subset["co_missing_strength_score"].tolist(),
            color=STRENGTH_COLOR,
            offset=strength_offset,
        )

    ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_model_label(model_id) for model_id in MODEL_ORDER], rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_title("Co-missingness: profile vs strength across dataset-model panels")
    ax.grid(axis="y", alpha=0.28, linestyle=":")
    ax.grid(axis="x", alpha=0.12)

    legend_handles = [
        Patch(facecolor=PROFILE_COLOR, edgecolor="none", alpha=0.6, label="Profile-only co-missing"),
        Patch(facecolor=STRENGTH_COLOR, edgecolor="none", alpha=0.6, label="Strength-only co-missing"),
        Line2D([0], [0], marker="s", color="#444444", markerfacecolor="#444444", markersize=5, linewidth=0, label="Mean over datasets"),
        Line2D([0], [0], color="#444444", linewidth=1.1, label="Min-max over datasets"),
        Patch(facecolor="#999999", edgecolor="none", alpha=0.24, label="IQR over datasets"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=8.5, handletextpad=0.5, columnspacing=1.4)

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def run_strength_diagnostic() -> dict[str, Any]:
    _ensure_dirs()
    dataset_model_df, _ = _load_inputs()
    dataset_model_strength_df = _build_dataset_model_strength_df(dataset_model_df)
    model_strength_summary_df = _build_model_strength_summary(dataset_model_strength_df)

    _write_csv(dataset_model_strength_df, DATA_DIR / "dataset_model_strength_scores.csv")
    _write_csv(model_strength_summary_df, DATA_DIR / "model_strength_summary.csv")

    main_pdf = FIG_DIR / "missing_strength_profile_vs_strength_model_dumbbell_main.pdf"
    main_png = FIG_DIR / "missing_strength_profile_vs_strength_model_dumbbell_main.png"
    main_svg = FIG_DIR / "missing_strength_profile_vs_strength_model_dumbbell_main.svg"
    main_tex = FIG_DIR / "missing_strength_profile_vs_strength_model_dumbbell_main.tex"
    gap_pdf = FIG_DIR / "missing_strength_gap_bars_appendix.pdf"
    gap_png = FIG_DIR / "missing_strength_gap_bars_appendix.png"
    gap_svg = FIG_DIR / "missing_strength_gap_bars_appendix.svg"
    gap_tex = FIG_DIR / "missing_strength_gap_bars_appendix.tex"
    heat_pdf = FIG_DIR / "missing_strength_dataset_model_heatmap_appendix.pdf"
    heat_png = FIG_DIR / "missing_strength_dataset_model_heatmap_appendix.png"
    heat_svg = FIG_DIR / "missing_strength_dataset_model_heatmap_appendix.svg"
    heat_tex = FIG_DIR / "missing_strength_dataset_model_heatmap_appendix.tex"
    dist_pdf = FIG_DIR / "missing_strength_profile_vs_strength_distribution_main.pdf"
    dist_png = FIG_DIR / "missing_strength_profile_vs_strength_distribution_main.png"
    dist_svg = FIG_DIR / "missing_strength_profile_vs_strength_distribution_main.svg"
    dist_tex = FIG_DIR / "missing_strength_profile_vs_strength_distribution_main.tex"

    _plot_model_dumbbell(model_strength_summary_df, main_pdf, main_png, main_svg)
    _plot_gap_bars(model_strength_summary_df, gap_pdf, gap_png, gap_svg)
    _plot_strength_heatmap(dataset_model_strength_df, heat_pdf, heat_png, heat_svg)
    _plot_profile_strength_distribution(dataset_model_strength_df, dist_pdf, dist_png, dist_svg)
    _write_include_tex(main_tex, "Missingness auxiliary insight: profile vs strength", main_pdf.name)
    _write_include_tex(gap_tex, "Strength minus profile", gap_pdf.name)
    _write_include_tex(heat_tex, "Dataset-model strength heatmap", heat_pdf.name)
    _write_include_tex(dist_tex, "Co-missing profile vs strength distribution", dist_pdf.name)

    report_lines = [
        "# Missingness Strength Diagnostic",
        "",
        "- Canonical missingness family score now keeps `co_missingness_pattern_consistency` as profile-only.",
        "- This auxiliary diagnostic isolates `co_missing_strength_score` so we can study whether models preserve the strength of structured missingness relations even when detailed profiles differ.",
        f"- Applicable dataset-model panels: `{dataset_model_strength_df.shape[0]}`",
        f"- Applicable datasets: `{dataset_model_strength_df['dataset_id'].nunique() if not dataset_model_strength_df.empty else 0}`",
        "",
    ]
    (OUTPUT_ROOT / "analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (OUTPUT_ROOT / "paper_caption.txt").write_text(
        "Auxiliary missingness diagnostic comparing canonical profile-only co-missing fidelity against relation-strength fidelity. "
        "The hollow-circle endpoint shows how well each model preserves detailed conditional missingness profiles, while the filled-square endpoint shows whether the overall dependence strength of missingness on related variables is retained.",
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "paper_paragraphs.md").write_text(
        "\n".join(
            [
                "Structured missingness can fail in two different ways: a model may distort the detailed profile of conditional missingness rates, or it may alter how strongly missingness depends on related variables.",
                "",
                "To separate these effects, we keep profile-only fidelity as the canonical co-missing subitem and report relation-strength fidelity as an auxiliary diagnostic. Differences between the two indicate whether a model preserves broad dependence amplitude more easily than fine-grained missingness profiles.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    final_files = [
        DATA_DIR / "dataset_model_strength_scores.csv",
        DATA_DIR / "model_strength_summary.csv",
        OUTPUT_ROOT / "analysis_report.md",
        OUTPUT_ROOT / "paper_caption.txt",
        OUTPUT_ROOT / "paper_paragraphs.md",
        main_pdf,
        main_png,
        main_svg,
        main_tex,
        dist_pdf,
        dist_png,
        dist_svg,
        dist_tex,
        gap_pdf,
        gap_png,
        gap_svg,
        gap_tex,
        heat_pdf,
        heat_png,
        heat_svg,
        heat_tex,
    ]
    must_do = {
        main_pdf.name: main_pdf,
        main_png.name: main_png,
        main_svg.name: main_svg,
        main_tex.name: main_tex,
        dist_pdf.name: dist_pdf,
        dist_png.name: dist_png,
        dist_svg.name: dist_svg,
        dist_tex.name: dist_tex,
        gap_pdf.name: gap_pdf,
        gap_png.name: gap_png,
        gap_svg.name: gap_svg,
        gap_tex.name: gap_tex,
        heat_pdf.name: heat_pdf,
        heat_png.name: heat_png,
        heat_svg.name: heat_svg,
        heat_tex.name: heat_tex,
    }
    sync_final_outputs(FINAL_DIR, final_files, must_do)
    (FINAL_DIR / "README.md").write_text(
        render_final_readme(
            title="Missingness Strength Diagnostic",
            summary="Auxiliary paper-facing bundle isolating relation-strength fidelity from the canonical profile-only co-missing score.",
            primary_files=[item.name for item in final_files if item.name.endswith((".png", ".pdf", ".tex"))][:12],
            must_do_files=list(must_do.keys()),
            support_files=[
                "dataset_model_strength_scores.csv",
                "model_strength_summary.csv",
                "analysis_report.md",
                "paper_caption.txt",
                "paper_paragraphs.md",
            ],
        ),
        encoding="utf-8",
    )
    return {
        "dataset_model_strength_scores": DATA_DIR / "dataset_model_strength_scores.csv",
        "model_strength_summary": DATA_DIR / "model_strength_summary.csv",
        "main_figure_png": main_png,
        "heatmap_png": heat_png,
    }


if __name__ == "__main__":
    outputs = run_strength_diagnostic()
    for key, value in outputs.items():
        print(f"{key}: {value}")
