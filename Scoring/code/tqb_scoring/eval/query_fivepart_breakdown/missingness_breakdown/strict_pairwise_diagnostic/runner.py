#!/usr/bin/env python3
"""Paper-facing auxiliary diagnostic for strict missing-only pairwise co-missingness."""

from __future__ import annotations

from collections import defaultdict
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
from tqb_scoring.eval.query_fivepart_breakdown.missingness_breakdown.review_strict_pairwise import _strict_pairwise_score_for_asset

MISSINGNESS_ROOT = PROJECT_ROOT.parent / "results" / "query_fivepart_breakdown" / "missingness_breakdown"
INPUT_DATA_DIR = MISSINGNESS_ROOT / "data"
OUTPUT_ROOT = MISSINGNESS_ROOT / "strict_pairwise_diagnostic"
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
EXCLUDED_MODELS = {"cdtd", "codi", "goggle"}
MODEL_ALIASES = {"rtf": "realtabformer"}
SERVER_PRIORITY = {"rtx_5090": 2, "rtx_pro_6000": 1}
ROOT_PRIORITY = {"SynOutput-5090": 2, "SynOutput": 1}
BROAD_PROFILE_COLOR = "#E76F51"
STRICT_PAIRWISE_COLOR = "#6D597A"


def _ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, DATA_DIR, FIG_DIR, FINAL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _normalize_model(model_id: Any) -> str:
    key = str(model_id or "").strip().lower()
    return MODEL_ALIASES.get(key, key)


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


def _asset_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    server = str(row.get("server_type") or "").strip().lower()
    root_name = str(row.get("root_name") or "").strip()
    run_id = str(row.get("run_id") or "").strip()
    asset_key = str(row.get("asset_key") or "").strip()
    return (
        SERVER_PRIORITY.get(server, 0),
        ROOT_PRIORITY.get(root_name, 0),
        run_id,
        asset_key,
    )


def _load_primary_asset_rows() -> pd.DataFrame:
    asset_df = pd.read_csv(INPUT_DATA_DIR / "direct_asset_scores.csv", encoding="utf-8-sig")
    asset_df["model_id"] = asset_df["model_id"].map(_normalize_model)
    asset_df = asset_df.loc[
        (asset_df["status"] == "ok")
        & (~asset_df["model_id"].isin(EXCLUDED_MODELS))
        & asset_df["model_id"].isin(MODEL_ORDER)
    ].copy()

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in asset_df.to_dict(orient="records"):
        grouped[(str(row["dataset_id"]), str(row["model_id"]))].append(row)

    chosen_rows: list[dict[str, Any]] = []
    for key, items in grouped.items():
        ranked = sorted(items, key=_asset_sort_key, reverse=True)
        chosen_rows.append(ranked[0])
    return pd.DataFrame(chosen_rows)


def _build_strict_panel_df(asset_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in asset_df.itertuples(index=False):
        strict = _strict_pairwise_score_for_asset(str(row.dataset_id), Path(str(row.synthetic_csv_path)))
        rows.append(
            {
                "dataset_id": str(row.dataset_id),
                "dataset_prefix": str(row.dataset_id)[0].lower(),
                "model_id": str(row.model_id),
                "model_label": _model_label(str(row.model_id)),
                "current_broad_profile_score": float(row.co_missingness_pattern_consistency),
                "current_strength_score": float(getattr(row, "co_missing_strength_score", float("nan"))),
                "strict_status": strict["strict_status"],
                "strict_pairwise_score": strict["strict_pairwise_score"],
                "strict_pair_count": int(strict["strict_pair_count"]),
                "active_missing_target_count": int(strict["active_missing_target_count"]),
            }
        )
    df = pd.DataFrame(rows)
    df["delta_strict_minus_broad"] = pd.to_numeric(df["strict_pairwise_score"], errors="coerce") - pd.to_numeric(df["current_broad_profile_score"], errors="coerce")
    df["dataset_sort"] = df["dataset_id"].map(_dataset_sort_key)
    df["model_order"] = df["model_id"].map({model_id: idx for idx, model_id in enumerate(MODEL_ORDER)})
    df = df.sort_values(["dataset_sort", "model_order"]).drop(columns=["dataset_sort", "model_order"]).reset_index(drop=True)
    return df


def _build_model_summary(panel_df: pd.DataFrame) -> pd.DataFrame:
    overlap_df = panel_df.loc[panel_df["strict_pairwise_score"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        subset = overlap_df.loc[overlap_df["model_id"] == model_id].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "dataset_count_overlap": int(subset["dataset_id"].nunique()),
                "panel_count_overlap": int(subset.shape[0]),
                "broad_profile_score_mean": round(float(subset["current_broad_profile_score"].mean()), 6),
                "strict_pairwise_score_mean": round(float(subset["strict_pairwise_score"].mean()), 6),
                "delta_strict_minus_broad_mean": round(float(subset["delta_strict_minus_broad"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def _build_coverage_summary(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, group in panel_df.groupby("dataset_id", sort=False):
        rows.append(
            {
                "dataset_id": dataset_id,
                "model_panel_count": int(group.shape[0]),
                "strict_applicable_panel_count": int(group["strict_pairwise_score"].notna().sum()),
                "active_missing_target_count": int(pd.to_numeric(group["active_missing_target_count"], errors="coerce").fillna(0).max()),
                "strict_pair_count": int(pd.to_numeric(group["strict_pair_count"], errors="coerce").fillna(0).max()),
            }
        )
    coverage_df = pd.DataFrame(rows)
    if coverage_df.empty:
        return coverage_df
    coverage_df["dataset_sort"] = coverage_df["dataset_id"].map(_dataset_sort_key)
    return coverage_df.sort_values(["dataset_sort", "dataset_id"]).drop(columns=["dataset_sort"]).reset_index(drop=True)


def _plot_model_dumbbell(summary_df: pd.DataFrame, pdf_path: Path, png_path: Path, svg_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    y_positions = list(range(len(summary_df)))
    for idx, row in enumerate(summary_df.itertuples()):
        model_id = str(row.model_id)
        color = MODEL_COLORS.get(model_id, "#777777")
        broad = float(row.broad_profile_score_mean)
        strict = float(row.strict_pairwise_score_mean)
        ax.plot([broad, strict], [idx, idx], color=color, linewidth=2.2, alpha=0.95)
        ax.scatter(broad, idx, s=70, facecolors="white", edgecolors=color, linewidth=1.8, zorder=3)
        ax.scatter(strict, idx, s=70, facecolors=color, edgecolors=color, marker="D", linewidth=1.0, zorder=4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary_df["model_label"])
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Mean score over strict-overlap panels")
    ax.set_title("Broad structured missingness vs strict missing-only pairwise co-missingness")
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.01, 1.01, "Hollow circle = broad profile-only score; filled diamond = strict missing-only pairwise score", transform=ax.transAxes, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _plot_panel_scatter(panel_df: pd.DataFrame, pdf_path: Path, png_path: Path, svg_path: Path) -> None:
    overlap_df = panel_df.loc[panel_df["strict_pairwise_score"].notna()].copy()
    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    ax.scatter(
        overlap_df["current_broad_profile_score"],
        overlap_df["strict_pairwise_score"],
        s=34,
        color="#4C78A8",
        alpha=0.78,
        edgecolors="none",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.1)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Broad profile-only co-missing score")
    ax.set_ylabel("Strict missing-only pairwise score")
    ax.set_title(
        "Dataset-model overlap panels\n"
        f"n={overlap_df.shape[0]}, datasets={overlap_df['dataset_id'].nunique() if not overlap_df.empty else 0}"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_bars(coverage_df: pd.DataFrame, pdf_path: Path, png_path: Path, svg_path: Path) -> None:
    x = range(len(coverage_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    ax.bar([item - width / 2 for item in x], coverage_df["model_panel_count"], width=width, color="#BDBDBD", edgecolor="#777777", label="All available model panels")
    ax.bar([item + width / 2 for item in x], coverage_df["strict_applicable_panel_count"], width=width, color="#4C78A8", edgecolor="#2C5A88", label="Strict-pairwise applicable panels")
    ax.set_xticks(list(x))
    ax.set_xticklabels(coverage_df["dataset_id"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Panel count")
    ax.set_title("Coverage shrinkage when only missing-only pairs are retained")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
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


def _plot_broad_vs_strict_distribution(panel_df: pd.DataFrame, pdf_path: Path, png_path: Path, svg_path: Path) -> None:
    overlap_df = panel_df.loc[panel_df["strict_pairwise_score"].notna()].copy()
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    x_positions = list(range(len(MODEL_ORDER)))
    broad_offset = -0.17
    strict_offset = 0.17

    for idx, model_id in enumerate(MODEL_ORDER):
        subset = overlap_df.loc[overlap_df["model_id"] == model_id].copy()
        if subset.empty:
            continue
        _draw_distribution_summary(
            ax,
            center=float(idx),
            values=subset["current_broad_profile_score"].tolist(),
            color=BROAD_PROFILE_COLOR,
            offset=broad_offset,
        )
        _draw_distribution_summary(
            ax,
            center=float(idx),
            values=subset["strict_pairwise_score"].tolist(),
            color=STRICT_PAIRWISE_COLOR,
            offset=strict_offset,
        )

    ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_model_label(model_id) for model_id in MODEL_ORDER], rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_title("Profile-only co-missing: broad vs missing-only pairwise")
    ax.grid(axis="y", alpha=0.28, linestyle=":")
    ax.grid(axis="x", alpha=0.12)

    legend_handles = [
        Patch(facecolor=BROAD_PROFILE_COLOR, edgecolor="none", alpha=0.6, label="Broad co-missing profile"),
        Patch(facecolor=STRICT_PAIRWISE_COLOR, edgecolor="none", alpha=0.6, label="Missing-only pairwise profile"),
        Line2D([0], [0], marker="s", color="#444444", markerfacecolor="#444444", markersize=5, linewidth=0, label="Mean over overlap datasets"),
        Line2D([0], [0], color="#444444", linewidth=1.1, label="Min-max over overlap datasets"),
        Patch(facecolor="#999999", edgecolor="none", alpha=0.24, label="IQR over overlap datasets"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=8.5, handletextpad=0.5, columnspacing=1.4)

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def run_strict_pairwise_diagnostic() -> dict[str, Any]:
    _ensure_dirs()
    asset_df = _load_primary_asset_rows()
    panel_df = _build_strict_panel_df(asset_df)
    model_df = _build_model_summary(panel_df)
    coverage_df = _build_coverage_summary(panel_df)

    _write_csv(panel_df, DATA_DIR / "strict_pairwise_panel_scores.csv")
    _write_csv(model_df, DATA_DIR / "strict_pairwise_model_summary.csv")
    _write_csv(coverage_df, DATA_DIR / "strict_pairwise_coverage_summary.csv")

    main_pdf = FIG_DIR / "strict_pairwise_vs_broad_model_dumbbell_main.pdf"
    main_png = FIG_DIR / "strict_pairwise_vs_broad_model_dumbbell_main.png"
    main_svg = FIG_DIR / "strict_pairwise_vs_broad_model_dumbbell_main.svg"
    main_tex = FIG_DIR / "strict_pairwise_vs_broad_model_dumbbell_main.tex"
    scatter_pdf = FIG_DIR / "strict_pairwise_panel_scatter_appendix.pdf"
    scatter_png = FIG_DIR / "strict_pairwise_panel_scatter_appendix.png"
    scatter_svg = FIG_DIR / "strict_pairwise_panel_scatter_appendix.svg"
    scatter_tex = FIG_DIR / "strict_pairwise_panel_scatter_appendix.tex"
    coverage_pdf = FIG_DIR / "strict_pairwise_coverage_bars_appendix.pdf"
    coverage_png = FIG_DIR / "strict_pairwise_coverage_bars_appendix.png"
    coverage_svg = FIG_DIR / "strict_pairwise_coverage_bars_appendix.svg"
    coverage_tex = FIG_DIR / "strict_pairwise_coverage_bars_appendix.tex"
    dist_pdf = FIG_DIR / "strict_pairwise_profile_distribution_main.pdf"
    dist_png = FIG_DIR / "strict_pairwise_profile_distribution_main.png"
    dist_svg = FIG_DIR / "strict_pairwise_profile_distribution_main.svg"
    dist_tex = FIG_DIR / "strict_pairwise_profile_distribution_main.tex"

    _plot_model_dumbbell(model_df, main_pdf, main_png, main_svg)
    _plot_panel_scatter(panel_df, scatter_pdf, scatter_png, scatter_svg)
    _plot_coverage_bars(coverage_df, coverage_pdf, coverage_png, coverage_svg)
    _plot_broad_vs_strict_distribution(panel_df, dist_pdf, dist_png, dist_svg)
    _write_include_tex(main_tex, "Strict pairwise vs broad co-missingness", main_pdf.name)
    _write_include_tex(scatter_tex, "Strict pairwise overlap scatter", scatter_pdf.name)
    _write_include_tex(coverage_tex, "Strict pairwise coverage shrinkage", coverage_pdf.name)
    _write_include_tex(dist_tex, "Profile-only broad vs strict pairwise distribution", dist_pdf.name)

    status_counts = panel_df["strict_status"].value_counts(dropna=False).to_dict()
    report_lines = [
        "# Strict Pairwise Co-Missing Diagnostic",
        "",
        "- Canonical missingness family score keeps the broader profile-only structured-missingness view.",
        "- This auxiliary diagnostic restricts the second axis to missing-only column pairs.",
        f"- Primary asset panels reviewed: `{panel_df.shape[0]}`",
        f"- Strict-overlap panels: `{int(panel_df['strict_pairwise_score'].notna().sum())}`",
        f"- Strict-overlap datasets: `{int(panel_df.loc[panel_df['strict_pairwise_score'].notna(), 'dataset_id'].nunique())}`",
        "",
        "## Status counts",
        "",
    ]
    report_lines.extend([f"- `{key}`: `{value}`" for key, value in status_counts.items()])
    (OUTPUT_ROOT / "analysis_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "paper_caption.txt").write_text(
        "Auxiliary missingness diagnostic comparing the broad profile-only co-missing score against a stricter version that only retains pairs of columns that both exhibit meaningful native missingness. "
        "Coverage bars show that the stricter definition is more selective because datasets with only one active missing target column become inapplicable.",
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "paper_paragraphs.md").write_text(
        "\n".join(
            [
                "The canonical missingness family intentionally keeps a broad structured-missingness view, where the missingness of one target column can depend on the states of any other usable column.",
                "",
                "As a sensitivity analysis, we also evaluate a strict pairwise variant that only retains pairs of columns that both carry meaningful native missingness. Differences between the two reveal whether a model preserves general conditional missingness structure more easily than direct co-missing behavior among missing columns themselves.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    final_files = [
        DATA_DIR / "strict_pairwise_panel_scores.csv",
        DATA_DIR / "strict_pairwise_model_summary.csv",
        DATA_DIR / "strict_pairwise_coverage_summary.csv",
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
        scatter_pdf,
        scatter_png,
        scatter_svg,
        scatter_tex,
        coverage_pdf,
        coverage_png,
        coverage_svg,
        coverage_tex,
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
        scatter_pdf.name: scatter_pdf,
        scatter_png.name: scatter_png,
        scatter_svg.name: scatter_svg,
        scatter_tex.name: scatter_tex,
        coverage_pdf.name: coverage_pdf,
        coverage_png.name: coverage_png,
        coverage_svg.name: coverage_svg,
        coverage_tex.name: coverage_tex,
    }
    sync_final_outputs(FINAL_DIR, final_files, must_do)
    (FINAL_DIR / "README.md").write_text(
        render_final_readme(
            title="Strict Pairwise Co-Missing Diagnostic",
            summary="Auxiliary paper-facing bundle that restricts co-missingness to missing-only column pairs and reports the resulting coverage shrinkage.",
            primary_files=[item.name for item in final_files if item.name.endswith((".png", ".pdf", ".tex"))][:12],
            must_do_files=list(must_do.keys()),
            support_files=[
                "strict_pairwise_panel_scores.csv",
                "strict_pairwise_model_summary.csv",
                "strict_pairwise_coverage_summary.csv",
                "analysis_report.md",
                "paper_caption.txt",
                "paper_paragraphs.md",
            ],
        ),
        encoding="utf-8",
    )
    return {
        "strict_pairwise_panel_scores": DATA_DIR / "strict_pairwise_panel_scores.csv",
        "strict_pairwise_model_summary": DATA_DIR / "strict_pairwise_model_summary.csv",
        "main_figure_png": main_png,
        "coverage_png": coverage_png,
    }


if __name__ == "__main__":
    outputs = run_strict_pairwise_diagnostic()
    for key, value in outputs.items():
        print(f"{key}: {value}")
