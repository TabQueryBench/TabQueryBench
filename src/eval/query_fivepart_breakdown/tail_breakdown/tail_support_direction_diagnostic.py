#!/usr/bin/env python3
"""Diagnose whether tail-set failures come more from extra or missing tail states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]

from src.eval.query_fivepart_breakdown.common_final import render_final_readme, sync_final_outputs


EVALUATION_ROOT = PROJECT_ROOT / "Evaluation"
TAIL_BREAKDOWN_ROOT = EVALUATION_ROOT / "query_fivepart_breakdown" / "tail_breakdown"
SOURCE_CSV = TAIL_BREAKDOWN_ROOT / "data" / "tail_threshold_asset_rows_enriched.csv"

OUTPUT_ROOT = TAIL_BREAKDOWN_ROOT / "support_diagnostics"
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


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, DATA_DIR, FIG_DIR, FINAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _lighten(hex_color: str, amount: float = 0.62) -> str:
    rgb = np.array(mcolors.to_rgb(hex_color))
    mixed = rgb + (1.0 - rgb) * amount
    return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))


def _load_case_rows() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig")
    df["model_id"] = df["model_id"].astype(str).str.strip().str.lower()
    df = df.loc[~df["model_id"].isin(EXCLUDED_MODELS)].copy()
    numeric_cols = [
        "threshold_pct",
        "tail_real_key_count",
        "tail_syn_key_count",
        "tail_union_key_count",
        "tail_set_consistency",
        "tail_mass_similarity",
        "tail_concentration_consistency",
        "tail_coverage_score",
        "tail_breakdown_score",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["intersection_count"] = (
        df["tail_real_key_count"] + df["tail_syn_key_count"] - df["tail_union_key_count"]
    ).clip(lower=0.0)
    df["missing_count"] = (df["tail_union_key_count"] - df["tail_syn_key_count"]).clip(lower=0.0)
    df["extra_count"] = (df["tail_union_key_count"] - df["tail_real_key_count"]).clip(lower=0.0)
    real_denom = df["tail_real_key_count"].replace(0, pd.NA)
    df["missing_rate_vs_real"] = df["missing_count"] / real_denom
    df["extra_rate_vs_real"] = df["extra_count"] / real_denom
    df["syn_to_real_ratio"] = df["tail_syn_key_count"] / real_denom
    df["dominance"] = "balanced"
    df.loc[df["extra_count"] > df["missing_count"], "dominance"] = "extra"
    df.loc[df["missing_count"] > df["extra_count"], "dominance"] = "missing"
    return df


def _build_dataset_model_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    threshold_df = (
        case_df.groupby(
            ["dataset_id", "dataset_prefix", "model_id", "model_label", "threshold_label", "threshold_pct"],
            as_index=False,
            observed=True,
        )
        .agg(
            asset_count=("asset_key", "nunique"),
            real_tail_key_count=("tail_real_key_count", "mean"),
            syn_tail_key_count=("tail_syn_key_count", "mean"),
            missing_count=("missing_count", "mean"),
            extra_count=("extra_count", "mean"),
            missing_rate_vs_real=("missing_rate_vs_real", "mean"),
            extra_rate_vs_real=("extra_rate_vs_real", "mean"),
            syn_to_real_ratio=("syn_to_real_ratio", "mean"),
            tail_set_consistency=("tail_set_consistency", "mean"),
            tail_mass_similarity=("tail_mass_similarity", "mean"),
            tail_concentration_consistency=("tail_concentration_consistency", "mean"),
            tail_breakdown_score=("tail_breakdown_score", "mean"),
        )
        .reset_index(drop=True)
    )
    threshold_df["dominance"] = "balanced"
    threshold_df.loc[threshold_df["extra_count"] > threshold_df["missing_count"], "dominance"] = "extra"
    threshold_df.loc[threshold_df["missing_count"] > threshold_df["extra_count"], "dominance"] = "missing"

    grouped = (
        threshold_df.groupby(["dataset_id", "dataset_prefix", "model_id"], as_index=False)
        .agg(
            model_label=("model_label", "first"),
            threshold_case_count=("threshold_label", "nunique"),
            asset_count=("asset_count", "max"),
            real_tail_key_count_mean=("real_tail_key_count", "mean"),
            real_tail_key_count_median=("real_tail_key_count", "median"),
            syn_tail_key_count_mean=("syn_tail_key_count", "mean"),
            syn_tail_key_count_median=("syn_tail_key_count", "median"),
            missing_count_mean=("missing_count", "mean"),
            missing_count_median=("missing_count", "median"),
            extra_count_mean=("extra_count", "mean"),
            extra_count_median=("extra_count", "median"),
            missing_rate_vs_real_mean=("missing_rate_vs_real", "mean"),
            missing_rate_vs_real_median=("missing_rate_vs_real", "median"),
            extra_rate_vs_real_mean=("extra_rate_vs_real", "mean"),
            extra_rate_vs_real_median=("extra_rate_vs_real", "median"),
            syn_to_real_ratio_mean=("syn_to_real_ratio", "mean"),
            syn_to_real_ratio_median=("syn_to_real_ratio", "median"),
            tail_set_consistency_mean=("tail_set_consistency", "mean"),
            tail_mass_similarity_mean=("tail_mass_similarity", "mean"),
            tail_concentration_consistency_mean=("tail_concentration_consistency", "mean"),
            tail_breakdown_score_mean=("tail_breakdown_score", "mean"),
        )
        .reset_index(drop=True)
    )

    dominance_counts = (
        threshold_df.groupby(["dataset_id", "model_id", "dominance"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["extra", "balanced", "missing"]:
        if column not in dominance_counts.columns:
            dominance_counts[column] = 0
    merged = grouped.merge(dominance_counts, on=["dataset_id", "model_id"], how="left")
    total = merged["threshold_case_count"].replace(0, pd.NA)
    merged["extra_dominant_share"] = merged["extra"] / total
    merged["balanced_share"] = merged["balanced"] / total
    merged["missing_dominant_share"] = merged["missing"] / total
    merged["net_extra_minus_missing_share"] = merged["extra_dominant_share"] - merged["missing_dominant_share"]

    def _dominant_label(row: pd.Series) -> str:
        scores = {
            "extra": float(row.get("extra_dominant_share", 0.0) or 0.0),
            "balanced": float(row.get("balanced_share", 0.0) or 0.0),
            "missing": float(row.get("missing_dominant_share", 0.0) or 0.0),
        }
        best = max(scores.values())
        winners = [key for key, value in scores.items() if abs(value - best) <= 1e-12]
        if len(winners) != 1:
            return "balanced"
        return winners[0]

    merged["dominant_error_type"] = merged.apply(_dominant_label, axis=1)
    merged["model_order"] = merged["model_id"].map({model_id: idx for idx, model_id in enumerate(MODEL_ORDER)})
    return merged.sort_values(["dataset_id", "model_order", "model_id"]).drop(columns=["model_order"]).reset_index(drop=True)


def _build_model_summary(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        subset = dataset_model_df.loc[dataset_model_df["model_id"] == model_id].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "dataset_count": int(subset["dataset_id"].nunique()),
                "dataset_model_panel_count": int(subset.shape[0]),
                "real_tail_key_count_mean": round(float(subset["real_tail_key_count_mean"].mean()), 6),
                "real_tail_key_count_median": round(float(subset["real_tail_key_count_median"].median()), 6),
                "syn_tail_key_count_mean": round(float(subset["syn_tail_key_count_mean"].mean()), 6),
                "syn_tail_key_count_median": round(float(subset["syn_tail_key_count_median"].median()), 6),
                "missing_count_mean": round(float(subset["missing_count_mean"].mean()), 6),
                "missing_count_median": round(float(subset["missing_count_median"].median()), 6),
                "extra_count_mean": round(float(subset["extra_count_mean"].mean()), 6),
                "extra_count_median": round(float(subset["extra_count_median"].median()), 6),
                "missing_rate_vs_real_mean": round(float(subset["missing_rate_vs_real_mean"].mean()), 6),
                "extra_rate_vs_real_mean": round(float(subset["extra_rate_vs_real_mean"].mean()), 6),
                "syn_to_real_ratio_mean": round(float(subset["syn_to_real_ratio_mean"].mean()), 6),
                "syn_to_real_ratio_median": round(float(subset["syn_to_real_ratio_median"].median()), 6),
                "extra_dominant_share": round(float(subset["extra_dominant_share"].mean()), 6),
                "balanced_share": round(float(subset["balanced_share"].mean()), 6),
                "missing_dominant_share": round(float(subset["missing_dominant_share"].mean()), 6),
                "net_extra_minus_missing_share": round(float(subset["net_extra_minus_missing_share"].mean()), 6),
                "tail_set_consistency_mean": round(float(subset["tail_set_consistency_mean"].mean()), 6),
                "tail_breakdown_score_mean": round(float(subset["tail_breakdown_score_mean"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def _build_paper_table(model_summary_df: pd.DataFrame) -> pd.DataFrame:
    table = model_summary_df[
        [
            "model_label",
            "dataset_count",
            "real_tail_key_count_median",
            "syn_tail_key_count_median",
            "extra_dominant_share",
            "balanced_share",
            "missing_dominant_share",
            "net_extra_minus_missing_share",
        ]
    ].copy()
    for column in ["extra_dominant_share", "balanced_share", "missing_dominant_share", "net_extra_minus_missing_share"]:
        table[column] = (table[column] * 100.0).round(2)
    return table.rename(
        columns={
            "model_label": "model",
            "dataset_count": "datasets",
            "real_tail_key_count_median": "median_real_tail_states",
            "syn_tail_key_count_median": "median_synthetic_tail_states",
            "extra_dominant_share": "extra_dominant_pct",
            "balanced_share": "balanced_pct",
            "missing_dominant_share": "missing_dominant_pct",
            "net_extra_minus_missing_share": "net_extra_minus_missing_pct",
        }
    )


def _plot_direction_bars(model_summary_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    plot_df = model_summary_df.copy()
    x = np.arange(plot_df.shape[0], dtype=float)
    width = 0.23
    fig, ax = plt.subplots(figsize=(14.6, 7.8))
    extra_positions = x - width
    balanced_positions = x
    missing_positions = x + width

    extra_values = plot_df["extra_dominant_share"].to_numpy(dtype=float)
    balanced_values = plot_df["balanced_share"].to_numpy(dtype=float)
    missing_values = plot_df["missing_dominant_share"].to_numpy(dtype=float)
    model_ids = plot_df["model_id"].tolist()

    for idx, model_id in enumerate(model_ids):
        color = MODEL_COLORS[model_id]
        ax.bar(
            extra_positions[idx],
            extra_values[idx],
            width=width * 0.94,
            color=color,
            edgecolor=color,
            linewidth=1.0,
            label="Extra-dominant panels" if idx == 0 else None,
        )
        ax.bar(
            balanced_positions[idx],
            balanced_values[idx],
            width=width * 0.94,
            color=_lighten(color),
            edgecolor=color,
            linewidth=1.0,
            hatch="..",
            label="Balanced panels" if idx == 0 else None,
        )
        ax.bar(
            missing_positions[idx],
            missing_values[idx],
            width=width * 0.94,
            color="white",
            edgecolor=color,
            linewidth=1.2,
            hatch="///",
            label="Missing-dominant panels" if idx == 0 else None,
        )

    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Share of dataset-model panels")
    ax.set_xlabel("Models")
    ax.set_title("Tail-set direction diagnostic: extra vs missing rare states")
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(model_id) for model_id in model_ids], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        edgecolor="#D9D9D9",
    )
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _write_direction_bars_tex(model_summary_df: pd.DataFrame, path: Path) -> None:
    lines: list[str] = [
        r"\documentclass[tikz,border=4pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\usepackage{xcolor}",
        r"\usetikzlibrary{patterns,patterns.meta}",
        r"\pgfplotsset{compat=1.18}",
        "",
    ]
    for model_id in model_summary_df["model_id"].tolist():
        color = MODEL_COLORS[model_id].replace("#", "")
        lines.append(rf"\definecolor{{model{model_id}}}{{HTML}}{{{color}}}")
    lines.extend(
        [
            r"\begin{document}",
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"width=15.6cm,",
            r"height=8.8cm,",
            r"ymin=0.0, ymax=1.02,",
            r"ymajorgrids,",
            r"grid style={draw=gray!20},",
            r"major grid style={draw=gray!28},",
            r"axis line style={draw=black!70},",
            r"tick style={draw=black!70},",
            r"xlabel={Models},",
            r"ylabel={Share of dataset-model panels},",
            r"title={Tail-set direction diagnostic: extra vs missing rare states},",
            rf"xtick={{{','.join(str(idx) for idx in range(1, len(model_summary_df) + 1))}}},",
            rf"xticklabels={{{','.join(_model_label(model_id) for model_id in model_summary_df['model_id'].tolist())}}},",
            r"x tick label style={rotate=45, anchor=east, font=\scriptsize},",
            r"legend style={draw=black!18, fill=white, fill opacity=0.94, text opacity=1, font=\scriptsize, rounded corners=2pt, at={(0.98,0.98)}, anchor=north east},",
            r"legend cell align={left},",
            r"]",
        ]
    )

    for series_name, bar_shift, style_key in [
        ("extra_dominant_share", -7.0, "extra"),
        ("balanced_share", 0.0, "balanced"),
        ("missing_dominant_share", 7.0, "missing"),
    ]:
        for idx, row in enumerate(model_summary_df.itertuples(index=False), start=1):
            value = float(getattr(row, series_name))
            model_id = str(row.model_id)
            if style_key == "extra":
                plot_style = rf"ybar, bar width=5.2pt, draw=model{model_id}, fill=model{model_id}, bar shift={bar_shift:.1f}pt"
            elif style_key == "balanced":
                plot_style = (
                    rf"ybar, bar width=5.2pt, draw=model{model_id}, fill=model{model_id}!30!white, "
                    rf"pattern=dots, pattern color=model{model_id}, bar shift={bar_shift:.1f}pt"
                )
            else:
                plot_style = (
                    rf"ybar, bar width=5.2pt, draw=model{model_id}, fill=white, "
                    rf"pattern=north east lines, pattern color=model{model_id}, bar shift={bar_shift:.1f}pt"
                )
            lines.append(rf"\addplot+[{plot_style}, forget plot] coordinates {{({idx},{value:.6f})}};")

    lines.extend(
        [
            r"\addlegendimage{area legend, draw=black!80, fill=black!65}",
            r"\addlegendentry{Extra-dominant panels}",
            r"\addlegendimage{area legend, draw=black!80, fill=black!20, pattern=dots, pattern color=black!80}",
            r"\addlegendentry{Balanced panels}",
            r"\addlegendimage{area legend, draw=black!80, fill=white, pattern=north east lines, pattern color=black!80}",
            r"\addlegendentry{Missing-dominant panels}",
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_tail_support_direction_diagnostic() -> dict[str, Any]:
    _ensure_dirs()
    case_df = _load_case_rows()
    dataset_model_df = _build_dataset_model_summary(case_df)
    model_summary_df = _build_model_summary(dataset_model_df)
    paper_table_df = _build_paper_table(model_summary_df)

    case_path = DATA_DIR / "tail_support_direction_case_rows.csv"
    dataset_model_path = DATA_DIR / "tail_support_direction_dataset_model_summary.csv"
    model_summary_path = DATA_DIR / "tail_support_direction_model_summary.csv"
    paper_table_path = DATA_DIR / "tail_support_direction_paper_table.csv"
    case_df.to_csv(case_path, index=False, encoding="utf-8")
    dataset_model_df.to_csv(dataset_model_path, index=False, encoding="utf-8")
    model_summary_df.to_csv(model_summary_path, index=False, encoding="utf-8")
    paper_table_df.to_csv(paper_table_path, index=False, encoding="utf-8")

    figure_tex = FIG_DIR / "tail_support_direction_model_bars_appendix.tex"
    figure_pdf = FIG_DIR / "tail_support_direction_model_bars_appendix.pdf"
    figure_png = FIG_DIR / "tail_support_direction_model_bars_appendix.png"
    _write_direction_bars_tex(model_summary_df, figure_tex)
    _plot_direction_bars(model_summary_df, figure_pdf, figure_png)

    note_lines = [
        "# Tail Support Direction Diagnostic",
        "",
        "This diagnostic asks whether low `tail_set_consistency` is mainly caused by inventing extra rare states or by dropping real rare states.",
        "",
        "## Recommended paper usage",
        "",
        "- Main text: use `tail_support_direction_model_bars_appendix` to show whether each model is more often extra-dominant or missing-dominant.",
        "- Main text table: use `tail_support_direction_paper_table.csv`, which keeps one row per model and includes median real/synthetic tail-state counts plus direction shares.",
        "- Appendix table: use `tail_support_direction_dataset_model_summary.csv`, which keeps one row per dataset-model panel and is small enough for a longtable appendix but too large for the main text.",
        "",
        "## Column notes",
        "",
        "- `real_tail_key_count_*` counts the number of real tail support states under the thresholded tail-support definition.",
        "- `syn_tail_key_count_*` counts the synthetic tail support states under the same threshold.",
        "- `extra_dominant_share` is the share of dataset-model panels where synthetic data introduces more tail states than it misses.",
        "- `missing_dominant_share` is the share of dataset-model panels where synthetic data misses more real tail states than it invents.",
        "",
    ]
    note_path = OUTPUT_ROOT / "analysis_note.md"
    note_path.write_text("\n".join(note_lines), encoding="utf-8")

    final_files = [
        figure_tex,
        figure_pdf,
        figure_png,
        model_summary_path,
        paper_table_path,
        dataset_model_path,
        note_path,
    ]
    must_do_aliases: dict[str, Path] = {}
    sync_final_outputs(FINAL_DIR, final_files, must_do_aliases)
    readme = render_final_readme(
        title="Tail Support Direction Diagnostic Final",
        summary="This directory contains a support diagnostic for interpreting low tail-set consistency: whether models more often invent extra rare states or drop real rare states.",
        primary_files=[
            "tail_support_direction_model_bars_appendix.tex",
            "tail_support_direction_model_bars_appendix.pdf",
            "tail_support_direction_model_bars_appendix.png",
            "tail_support_direction_paper_table.csv",
            "tail_support_direction_model_summary.csv",
        ],
        must_do_files=[],
        support_files=[
            "tail_support_direction_dataset_model_summary.csv",
            "analysis_note.md",
        ],
        notes=[
            "The paper-facing compact table is `tail_support_direction_paper_table.csv`.",
            "The appendix-scale table is `tail_support_direction_dataset_model_summary.csv`.",
        ],
    )
    (FINAL_DIR / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "task": "tail_support_direction_diagnostic",
        "case_row_count": int(case_df.shape[0]),
        "dataset_model_panel_count": int(dataset_model_df.shape[0]),
        "model_count": int(model_summary_df.shape[0]),
        "figure_png": str(figure_png.resolve()),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = run_tail_support_direction_diagnostic()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
