#!/usr/bin/env python3
"""Render merged paper-ready figures for tail-threshold v2 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THRESHOLD_ORDER = ["10%", "8%", "6%", "4%", "3%", "2%", "1%", "0.5%", "0.1%"]
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
OVERALL_COLOR = "#2F4B7C"
COVERAGE_COLOR = "#D1495B"
SIZE_COLOR = "#2A9D8F"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-asset-csv", type=Path, required=True)
    parser.add_argument("--override-asset-csv", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run-tag", type=str, required=True)
    parser.add_argument("--source-run-dir", type=str, required=True)
    parser.add_argument("--notes", type=str, default="")
    return parser


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["threshold_label"] = pd.Categorical(out["threshold_label"], categories=THRESHOLD_ORDER, ordered=True)
    return out.sort_values(["threshold_label"]).reset_index(drop=True)


def _load_and_merge(base_csv: Path, override_csvs: list[Path]) -> pd.DataFrame:
    merged = pd.read_csv(base_csv)
    for override_csv in override_csvs:
        patch = pd.read_csv(override_csv)
        if patch.empty:
            continue
        patch_datasets = sorted(set(patch["dataset_id"].astype(str)))
        merged = merged[~merged["dataset_id"].astype(str).isin(patch_datasets)].copy()
        merged = pd.concat([merged, patch], ignore_index=True)
    merged["threshold_pct"] = merged["threshold_pct"].astype(float)
    merged["tail_overall_score"] = merged["tail_overall_score"].astype(float)
    merged["tail_coverage_score"] = merged["tail_coverage_score"].astype(float)
    merged["tail_size_score"] = merged["tail_size_score"].astype(float)
    merged["model_label"] = merged["model_label"].fillna(merged["model_id"])
    merged = merged[merged["model_label"].isin(MODEL_COLORS)].copy()
    return merged


def _global_summary(asset_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold_label, block in asset_df.groupby("threshold_label", sort=False):
        rows.append(
            {
                "threshold_label": threshold_label,
                "threshold_pct": float(block["threshold_pct"].iloc[0]),
                "asset_count": int(len(block)),
                "tail_overall_mean": float(block["tail_overall_score"].mean()),
                "tail_coverage_mean": float(block["tail_coverage_score"].mean()),
                "tail_size_mean": float(block["tail_size_score"].mean()),
                "categorical_coverage_mean": float(block["categorical_coverage_score"].dropna().mean())
                if block["categorical_coverage_score"].notna().any()
                else np.nan,
                "categorical_size_mean": float(block["categorical_size_score"].dropna().mean())
                if block["categorical_size_score"].notna().any()
                else np.nan,
                "numerical_coverage_mean": float(block["numerical_coverage_score"].dropna().mean())
                if block["numerical_coverage_score"].notna().any()
                else np.nan,
                "numerical_size_mean": float(block["numerical_size_score"].dropna().mean())
                if block["numerical_size_score"].notna().any()
                else np.nan,
            }
        )
    return _ordered(pd.DataFrame(rows))


def _model_summary(asset_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = asset_df.groupby(["model_id", "model_label", "threshold_label"], sort=False)
    for (model_id, model_label, threshold_label), block in grouped:
        rows.append(
            {
                "model_id": model_id,
                "model_label": model_label,
                "threshold_label": threshold_label,
                "threshold_pct": float(block["threshold_pct"].iloc[0]),
                "asset_count": int(len(block)),
                "tail_overall_score": float(block["tail_overall_score"].mean()),
                "tail_coverage_score": float(block["tail_coverage_score"].mean()),
                "tail_size_score": float(block["tail_size_score"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["threshold_label"] = pd.Categorical(out["threshold_label"], categories=THRESHOLD_ORDER, ordered=True)
    return out.sort_values(["model_label", "threshold_label"]).reset_index(drop=True)


def _dataset_summary(asset_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = asset_df.groupby(["dataset_id", "dataset_prefix", "threshold_label"], sort=False)
    for (dataset_id, dataset_prefix, threshold_label), block in grouped:
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": dataset_prefix,
                "threshold_label": threshold_label,
                "threshold_pct": float(block["threshold_pct"].iloc[0]),
                "asset_count": int(len(block)),
                "tail_overall_score": float(block["tail_overall_score"].mean()),
                "tail_coverage_score": float(block["tail_coverage_score"].mean()),
                "tail_size_score": float(block["tail_size_score"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["threshold_label"] = pd.Categorical(out["threshold_label"], categories=THRESHOLD_ORDER, ordered=True)
    return out.sort_values(["dataset_id", "threshold_label"]).reset_index(drop=True)


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


def _panel_a(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    x = np.arange(len(global_df))
    ax.plot(x, global_df["tail_overall_mean"], color=OVERALL_COLOR, marker="o", linewidth=2.1, label="Overall")
    ax.plot(x, global_df["tail_coverage_mean"], color=COVERAGE_COLOR, marker="o", linewidth=2.0, label="Coverage")
    ax.plot(x, global_df["tail_size_mean"], color=SIZE_COLOR, marker="o", linewidth=2.0, label="Size")
    ax.set_xticks(x, global_df["threshold_label"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlabel("Tail threshold")
    ax.set_title("A. Tail score falls as the threshold tightens", loc="left", pad=6)
    _style_axis(ax)
    ax.legend(loc="lower left", frameon=False)


def _panel_b(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    x = np.arange(len(global_df))
    baseline = global_df.iloc[0]
    for col, color, label in [
        ("tail_overall_mean", OVERALL_COLOR, "Overall"),
        ("tail_coverage_mean", COVERAGE_COLOR, "Coverage"),
        ("tail_size_mean", SIZE_COLOR, "Size"),
    ]:
        base = float(baseline[col]) if float(baseline[col]) else 1.0
        vals = [float(v) / base for v in global_df[col]]
        ax.plot(x, vals, color=color, marker="o", linewidth=2.0, label=label)
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, global_df["threshold_label"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Relative score vs. 10%")
    ax.set_xlabel("Tail threshold")
    ax.set_title("B. Coverage breaks faster than size", loc="left", pad=6)
    _style_axis(ax)
    ax.legend(loc="lower left", frameon=False)


def _panel_c(ax: plt.Axes, model_df: pd.DataFrame) -> None:
    focus = model_df[model_df["threshold_label"] == "3%"].copy()
    focus = focus.sort_values("model_label")
    for _, row in focus.iterrows():
        label = str(row["model_label"])
        color = MODEL_COLORS.get(label, "#666666")
        x = float(row["tail_coverage_score"])
        y = float(row["tail_size_score"])
        ax.scatter([x], [y], color=color, s=42, edgecolor="white", linewidth=0.5, zorder=3)
        ax.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7, color=color)
    low = min(float(focus["tail_coverage_score"].min()), float(focus["tail_size_score"].min())) - 0.02
    high = max(float(focus["tail_coverage_score"].max()), float(focus["tail_size_score"].max())) + 0.02
    ax.plot([low, high], [low, high], linestyle="--", color="#999999", linewidth=1.0, zorder=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Coverage score at 3%")
    ax.set_ylabel("Size score at 3%")
    ax.set_title("C. Most models retain more tail size than exact tail coverage", loc="left", pad=6)
    _style_axis(ax)


def _render_f1_absolute_summary(global_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    _configure_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    _panel_a(ax, global_df)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _render_f2_relative_summary(global_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    _configure_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    _panel_b(ax, global_df)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _render_main_figure(global_df: pd.DataFrame, model_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    _configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), constrained_layout=True)
    _panel_a(axes[0], global_df)
    _panel_b(axes[1], global_df)
    _panel_c(axes[2], model_df)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _render_submetrics_figure(model_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    _configure_style()
    model_order = [m for m in MODEL_COLORS if m in set(model_df["model_label"].tolist())]
    fig, ax = plt.subplots(figsize=(11.5, 4.3), constrained_layout=True)
    x = np.arange(len(model_order))
    width = 0.18
    for idx, (metric, color, label, offset) in enumerate(
        [
            ("tail_coverage_score", COVERAGE_COLOR, "Coverage", -width * 0.6),
            ("tail_size_score", SIZE_COLOR, "Size", width * 0.6),
        ]
    ):
        means, lows, highs = [], [], []
        for model_label in model_order:
            subset = model_df[model_df["model_label"] == model_label].copy()
            vals = subset[metric].astype(float).tolist()
            means.append(float(np.mean(vals)))
            lows.append(float(np.min(vals)))
            highs.append(float(np.max(vals)))
        xs = x + offset
        ax.bar(xs, means, width=width, color=color, alpha=0.65, label=label)
        yerr = np.vstack([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
        ax.errorbar(xs, means, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.15, capsize=3)
    ax.set_xticks(x, model_order, rotation=30, ha="right")
    ax.set_ylabel("Mean score across thresholds")
    ax.set_xlabel("Model")
    ax.set_title("Coverage and size summaries across tail thresholds", loc="left", pad=6)
    _style_axis(ax)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _render_model_overall_absolute(model_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    _configure_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    for model_label, block in model_df.groupby("model_label", sort=False):
        ordered = _ordered(block)
        x = np.arange(len(ordered))
        ax.plot(
            x,
            ordered["tail_overall_score"].astype(float).to_numpy(),
            marker="o",
            linewidth=1.8,
            label=str(model_label),
        )
    ax.set_xticks(x, THRESHOLD_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Tail overall score")
    ax.set_xlabel("Tail threshold")
    ax.set_title("All-model tail overall curves under threshold tightening", loc="left", pad=6)
    _style_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, ncol=1)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _render_model_overall_relative(model_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    _configure_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    for model_label, block in model_df.groupby("model_label", sort=False):
        ordered = _ordered(block)
        x = np.arange(len(ordered))
        vals = ordered["tail_overall_score"].astype(float).to_numpy()
        baseline = vals[0] if len(vals) and vals[0] else 1.0
        rel = vals / baseline
        ax.plot(x, rel, marker="o", linewidth=1.8, label=str(model_label))
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, THRESHOLD_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Relative to 10% baseline")
    ax.set_xlabel("Tail threshold")
    ax.set_title("All-model relative tail degradation curves", loc="left", pad=6)
    _style_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, ncol=1)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_tex_assets(output_dir: Path) -> None:
    _write_text(
        output_dir / "tail_stress_main_figure_embedded.tex",
        "\\centering\\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/tail_stress_main_figure.png}\n",
    )
    _write_text(
        output_dir / "tail_stress_main_figure.tex",
        "% Tail threshold v2 main figure.\n"
        "\\begin{figure}[t]\n"
        "    \\centering\n"
        "    \\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/tail_stress_main_figure.png}\n"
        "    \\caption{Tail-threshold v2 summary under threshold tightening. The updated metric retains categorical tail logic, replaces numerical tails with quantile-range coverage and real-cutoff size checks, and removes the former concentration component. As thresholds tighten, overall tail fidelity drops; coverage declines faster than size, showing that retaining the exact tail region is harder than preserving a coarse amount of tail mass.}\n"
        "    \\label{fig:tail-stress-main-v2}\n"
        "\\end{figure}\n",
    )
    _write_text(
        output_dir / "tail_submetrics_combined_figure_embedded.tex",
        "\\centering\\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/tail_submetrics_combined_figure.png}\n",
    )
    _write_text(
        output_dir / "tail_submetrics_combined_figure.tex",
        "% Tail threshold v2 combined submetrics figure.\n"
        "\\begin{figure}[t]\n"
        "    \\centering\n"
        "    \\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/tail_submetrics_combined_figure.png}\n"
        "    \\caption{Coverage and size summaries for the updated tail metric. Bars report each model's mean over thresholds from 10\\% to 0.1\\%, with whiskers showing the observed range across thresholds. The persistent gap between coverage and size indicates that many generators preserve approximate tail mass more readily than exact tail support identity.}\n"
        "    \\label{fig:tail-submetrics-combined-v2}\n"
        "\\end{figure}\n",
    )
    _write_text(
        output_dir / "f3_tail_overall_absolute_embedded.tex",
        "\\centering\\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/f3_tail_overall_absolute.png}\n",
    )
    _write_text(
        output_dir / "f3_tail_overall_absolute.tex",
        "% Tail threshold v2 all-model absolute overall curves.\n"
        "\\begin{figure}[t]\n"
        "    \\centering\n"
        "    \\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/f3_tail_overall_absolute.png}\n"
        "    \\caption{Tail overall score for all paper generators across threshold tightening, shown in absolute terms.}\n"
        "    \\label{fig:tail-overall-absolute-all-models-v2}\n"
        "\\end{figure}\n",
    )
    _write_text(
        output_dir / "f4_tail_overall_relative_embedded.tex",
        "\\centering\\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/f4_tail_overall_relative.png}\n",
    )
    _write_text(
        output_dir / "f4_tail_overall_relative.tex",
        "% Tail threshold v2 all-model relative overall curves.\n"
        "\\begin{figure}[t]\n"
        "    \\centering\n"
        "    \\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/f4_tail_overall_relative.png}\n"
        "    \\caption{Tail overall score for all paper generators relative to each model's 10\\% baseline under threshold tightening.}\n"
        "    \\label{fig:tail-overall-relative-all-models-v2}\n"
        "\\end{figure}\n",
    )
    _write_text(
        output_dir / "f1_tail_summary_absolute_embedded.tex",
        "\\centering\\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/f1_tail_summary_absolute.png}\n",
    )
    _write_text(
        output_dir / "f1_tail_summary_absolute.tex",
        "% Tail threshold v2 summary absolute curves.\n"
        "\\begin{figure}[t]\n"
        "    \\centering\n"
        "    \\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/f1_tail_summary_absolute.png}\n"
        "    \\caption{Absolute tail summary curves for the updated tail metric.}\n"
        "    \\label{fig:tail-summary-absolute-v2}\n"
        "\\end{figure}\n",
    )
    _write_text(
        output_dir / "f2_tail_summary_relative_embedded.tex",
        "\\centering\\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/f2_tail_summary_relative.png}\n",
    )
    _write_text(
        output_dir / "f2_tail_summary_relative.tex",
        "% Tail threshold v2 summary relative curves.\n"
        "\\begin{figure}[t]\n"
        "    \\centering\n"
        "    \\includegraphics[width=\\linewidth]{Evaluation/tail_threshold_v2/final/f2_tail_summary_relative.png}\n"
        "    \\caption{Relative tail summary curves for the updated tail metric.}\n"
        "    \\label{fig:tail-summary-relative-v2}\n"
        "\\end{figure}\n",
    )


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_asset = _load_and_merge(args.base_asset_csv, args.override_asset_csv)
    global_df = _global_summary(merged_asset)
    model_df = _model_summary(merged_asset)
    dataset_df = _dataset_summary(merged_asset)

    data_dir = output_dir / "data"
    summaries_dir = output_dir / "summaries"
    data_dir.mkdir(exist_ok=True)
    summaries_dir.mkdir(exist_ok=True)

    merged_asset.to_csv(data_dir / "tail_threshold_v2_asset_scores_merged.csv", index=False)
    global_df.to_csv(summaries_dir / "tail_threshold_v2_global_summary_merged.csv", index=False)
    model_df.to_csv(summaries_dir / "tail_threshold_v2_model_summary_merged.csv", index=False)
    dataset_df.to_csv(summaries_dir / "tail_threshold_v2_dataset_summary_merged.csv", index=False)

    _render_main_figure(global_df, model_df, output_dir / "tail_stress_main_figure.png", output_dir / "tail_stress_main_figure.pdf")
    _render_f1_absolute_summary(
        global_df,
        output_dir / "f1_tail_summary_absolute.png",
        output_dir / "f1_tail_summary_absolute.pdf",
    )
    _render_f2_relative_summary(
        global_df,
        output_dir / "f2_tail_summary_relative.png",
        output_dir / "f2_tail_summary_relative.pdf",
    )
    _render_submetrics_figure(
        model_df,
        output_dir / "tail_submetrics_combined_figure.png",
        output_dir / "tail_submetrics_combined_figure.pdf",
    )
    _render_model_overall_absolute(
        model_df,
        output_dir / "f3_tail_overall_absolute.png",
        output_dir / "f3_tail_overall_absolute.pdf",
    )
    _render_model_overall_relative(
        model_df,
        output_dir / "f4_tail_overall_relative.png",
        output_dir / "f4_tail_overall_relative.pdf",
    )
    _write_tex_assets(output_dir)

    manifest = {
        "task": "tail_threshold_v2_figures",
        "title": "Tail/Range stress testing under threshold tightening (v2)",
        "source_run_tag": args.source_run_tag,
        "source_run_dir": args.source_run_dir,
        "override_asset_csvs": [str(p) for p in args.override_asset_csv],
        "dataset_count": int(merged_asset["dataset_id"].nunique()),
        "asset_count": int(len(merged_asset)),
        "threshold_labels": THRESHOLD_ORDER,
        "notes": args.notes,
        "outputs": {
            "merged_asset_scores_csv": str((data_dir / "tail_threshold_v2_asset_scores_merged.csv").resolve()),
            "merged_global_summary_csv": str((summaries_dir / "tail_threshold_v2_global_summary_merged.csv").resolve()),
            "merged_model_summary_csv": str((summaries_dir / "tail_threshold_v2_model_summary_merged.csv").resolve()),
            "merged_dataset_summary_csv": str((summaries_dir / "tail_threshold_v2_dataset_summary_merged.csv").resolve()),
            "tail_stress_main_figure_png": str((output_dir / "tail_stress_main_figure.png").resolve()),
            "tail_stress_main_figure_pdf": str((output_dir / "tail_stress_main_figure.pdf").resolve()),
            "f1_tail_summary_absolute_png": str((output_dir / "f1_tail_summary_absolute.png").resolve()),
            "f1_tail_summary_absolute_pdf": str((output_dir / "f1_tail_summary_absolute.pdf").resolve()),
            "f2_tail_summary_relative_png": str((output_dir / "f2_tail_summary_relative.png").resolve()),
            "f2_tail_summary_relative_pdf": str((output_dir / "f2_tail_summary_relative.pdf").resolve()),
            "tail_submetrics_combined_figure_png": str((output_dir / "tail_submetrics_combined_figure.png").resolve()),
            "tail_submetrics_combined_figure_pdf": str((output_dir / "tail_submetrics_combined_figure.pdf").resolve()),
            "f3_tail_overall_absolute_png": str((output_dir / "f3_tail_overall_absolute.png").resolve()),
            "f3_tail_overall_absolute_pdf": str((output_dir / "f3_tail_overall_absolute.pdf").resolve()),
            "f4_tail_overall_relative_png": str((output_dir / "f4_tail_overall_relative.png").resolve()),
            "f4_tail_overall_relative_pdf": str((output_dir / "f4_tail_overall_relative.pdf").resolve()),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
