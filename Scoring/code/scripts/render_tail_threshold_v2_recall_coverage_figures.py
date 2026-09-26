#!/usr/bin/env python3
"""Recompute Figure 6/7 style tail-threshold v2 figures under real-recall coverage."""

from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--base-column-csv", type=Path, required=True)
    parser.add_argument("--override-column-csv", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="recall_coverage")
    return parser


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
    merged["model_label"] = merged["model_label"].fillna(merged["model_id"])
    merged = merged[merged["model_label"].isin(MODEL_COLORS)].copy()
    return merged.reset_index(drop=True)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["threshold_label"] = pd.Categorical(out["threshold_label"], categories=THRESHOLD_ORDER, ordered=True)
    return out.sort_values(["threshold_label"]).reset_index(drop=True)


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


def _render_f2_relative_summary(global_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
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
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _render_model_overall_relative(model_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
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


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _categorical_recall_coverage(row: pd.Series) -> float:
    real_count = int(row.get("real_tail_token_count", 0) or 0)
    syn_count = int(row.get("syn_tail_token_count", 0) or 0)
    if real_count <= 0:
        return 1.0
    jaccard = float(row.get("coverage_score", row.get("coverage", 0.0)) or 0.0)
    if jaccard <= 0.0 or syn_count <= 0:
        return 0.0
    inter = jaccard * (real_count + syn_count) / (1.0 + jaccard)
    inter = int(round(inter))
    inter = max(0, min(inter, real_count, syn_count))
    return _clip01(inter / real_count)


def _real_interval_recall(a0: float, a1: float, b0: float, b1: float) -> float:
    real_left = min(float(a0), float(a1))
    real_right = max(float(a0), float(a1))
    syn_left = min(float(b0), float(b1))
    syn_right = max(float(b0), float(b1))
    overlap = max(0.0, min(real_right, syn_right) - max(real_left, syn_left))
    real_len = max(0.0, real_right - real_left)
    if real_len <= 1e-12:
        point = real_left
        return 1.0 if (syn_left - 1e-12) <= point <= (syn_right + 1e-12) else 0.0
    return _clip01(overlap / real_len)


def _numerical_recall_coverage(row: pd.Series) -> float:
    low = _real_interval_recall(
        row["real_min"],
        row["real_low_cutoff"],
        row["syn_min"],
        row["syn_low_cutoff"],
    )
    high = _real_interval_recall(
        row["real_high_cutoff"],
        row["real_max"],
        row["syn_high_cutoff"],
        row["syn_max"],
    )
    return _clip01(0.5 * (low + high))


def _recompute_column_coverage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_cov = []
    for _, row in out.iterrows():
        mode = str(row.get("column_mode", "")).strip().lower()
        if mode == "categorical":
            new_cov.append(_categorical_recall_coverage(row))
        elif mode == "numerical":
            new_cov.append(_numerical_recall_coverage(row))
        else:
            new_cov.append(float("nan"))
    out["coverage_score_recall"] = new_cov
    return out


def _mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def _asset_summary(column_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = [
        "asset_key",
        "asset_dir",
        "dataset_id",
        "dataset_prefix",
        "model_id",
        "model_label",
        "threshold_label",
        "threshold_pct",
    ]
    for keys, block in column_df.groupby(group_cols, sort=False):
        key_map = dict(zip(group_cols, keys))
        tail_coverage = _mean(block["coverage_score_recall"])
        tail_size = _mean(block["size_score"])
        rows.append(
            {
                **key_map,
                "column_count": int(len(block)),
                "tail_coverage_score": tail_coverage,
                "tail_size_score": tail_size,
                "tail_overall_score": float(np.nanmean([tail_coverage, tail_size])),
                "categorical_coverage_score": _mean(
                    block.loc[block["column_mode"] == "categorical", "coverage_score_recall"]
                ),
                "categorical_size_score": _mean(block.loc[block["column_mode"] == "categorical", "size_score"]),
                "numerical_coverage_score": _mean(
                    block.loc[block["column_mode"] == "numerical", "coverage_score_recall"]
                ),
                "numerical_size_score": _mean(block.loc[block["column_mode"] == "numerical", "size_score"]),
            }
        )
    out = pd.DataFrame(rows)
    out["threshold_label"] = pd.Categorical(out["threshold_label"], categories=THRESHOLD_ORDER, ordered=True)
    return out.sort_values(["dataset_id", "model_label", "threshold_label"]).reset_index(drop=True)


def _global_summary(asset_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold_label, block in asset_df.groupby("threshold_label", sort=False):
        rows.append(
            {
                "threshold_label": threshold_label,
                "threshold_pct": float(block["threshold_pct"].iloc[0]),
                "asset_count": int(len(block)),
                "tail_overall_mean": float(block["tail_overall_score"].mean()),
                "tail_coverage_mean": float(block["tail_coverage_score"].mean()),
                "tail_size_mean": float(block["tail_size_score"].mean()),
                "categorical_coverage_mean": _mean(block["categorical_coverage_score"]),
                "categorical_size_mean": _mean(block["categorical_size_score"]),
                "numerical_coverage_mean": _mean(block["numerical_coverage_score"]),
                "numerical_size_mean": _mean(block["numerical_size_score"]),
            }
        )
    return _ordered(pd.DataFrame(rows))


def _model_summary(asset_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
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


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_columns = _load_and_merge(args.base_column_csv, args.override_column_csv)
    recomputed_columns = _recompute_column_coverage(merged_columns)
    asset_df = _asset_summary(recomputed_columns)
    global_df = _global_summary(asset_df)
    model_df = _model_summary(asset_df)

    data_dir = output_dir / "data"
    summaries_dir = output_dir / "summaries"
    data_dir.mkdir(exist_ok=True)
    summaries_dir.mkdir(exist_ok=True)

    label = args.label
    recomputed_columns.to_csv(data_dir / f"tail_threshold_v2_column_scores_{label}.csv", index=False)
    asset_df.to_csv(data_dir / f"tail_threshold_v2_asset_scores_{label}.csv", index=False)
    global_df.to_csv(summaries_dir / f"tail_threshold_v2_global_summary_{label}.csv", index=False)
    model_df.to_csv(summaries_dir / f"tail_threshold_v2_model_summary_{label}.csv", index=False)

    _configure_style()
    _render_f2_relative_summary(
        global_df,
        output_dir / f"f2_tail_summary_relative_{label}.png",
        output_dir / f"f2_tail_summary_relative_{label}.pdf",
    )
    _render_model_overall_relative(
        model_df,
        output_dir / f"f4_tail_overall_relative_{label}.png",
        output_dir / f"f4_tail_overall_relative_{label}.pdf",
    )

    manifest = {
        "task": "tail_threshold_v2_recall_coverage_figures",
        "label": label,
        "base_column_csv": str(args.base_column_csv),
        "override_column_csvs": [str(p) for p in args.override_column_csv],
        "dataset_count": int(asset_df["dataset_id"].nunique()),
        "asset_count": int(len(asset_df)),
        "threshold_labels": THRESHOLD_ORDER,
        "outputs": {
            "column_scores_csv": str((data_dir / f"tail_threshold_v2_column_scores_{label}.csv").resolve()),
            "asset_scores_csv": str((data_dir / f"tail_threshold_v2_asset_scores_{label}.csv").resolve()),
            "global_summary_csv": str((summaries_dir / f"tail_threshold_v2_global_summary_{label}.csv").resolve()),
            "model_summary_csv": str((summaries_dir / f"tail_threshold_v2_model_summary_{label}.csv").resolve()),
            "f2_png": str((output_dir / f"f2_tail_summary_relative_{label}.png").resolve()),
            "f2_pdf": str((output_dir / f"f2_tail_summary_relative_{label}.pdf").resolve()),
            "f4_png": str((output_dir / f"f4_tail_overall_relative_{label}.png").resolve()),
            "f4_pdf": str((output_dir / f"f4_tail_overall_relative_{label}.pdf").resolve()),
        },
    }
    (output_dir / f"manifest_{label}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
