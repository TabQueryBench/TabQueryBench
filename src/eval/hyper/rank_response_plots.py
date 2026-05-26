from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.hyper.model_metric_cards import DATASET_ORDER, MODEL_DISPLAY, MODEL_ORDER


@dataclass
class RankResponseArtifacts:
    merged_csv: Path
    selected_csv: Path
    selection_summary_csv: Path
    query_ratio_png: Path
    query_delta_png: Path
    distance_ratio_png: Path
    distance_delta_png: Path
    overall_ratio_png: Path
    overall_delta_png: Path
    query_heatmap_png: Path
    distance_heatmap_png: Path
    gap_heatmap_png: Path


def _select_five_indices(n: int) -> list[int]:
    if n <= 5:
        return list(range(n))
    raw = np.linspace(0, n - 1, 5)
    indices: list[int] = []
    for value in raw:
        idx = int(round(float(value)))
        idx = min(max(idx, 0), n - 1)
        if idx not in indices:
            indices.append(idx)
    remaining = [idx for idx in range(n) if idx not in indices]
    while len(indices) < 5 and remaining:
        indices.append(remaining.pop(0))
    return sorted(indices[:5])


def _dataset_sort_series(series: pd.Series) -> pd.Series:
    mapping = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    return series.map(mapping).fillna(999).astype(int)


def select_canonical_five(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["dataset_sort"] = _dataset_sort_series(working["dataset_id"])
    working = working.sort_values(["dataset_sort", "model_id", "train_sec", "generate_sec", "run_id"]).reset_index(drop=True)

    selected_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for (dataset_id, model_id), group in working.groupby(["dataset_id", "model_id"], sort=False):
        group = group.sort_values(["train_sec", "generate_sec", "run_id"]).reset_index(drop=True)
        indices = _select_five_indices(len(group))
        chosen = group.iloc[indices].copy().reset_index(drop=True)
        chosen["selected_rank"] = np.arange(1, len(chosen) + 1)
        chosen["model_display"] = chosen["model_id"].map(MODEL_DISPLAY).fillna(chosen["model_id"])
        selected_parts.append(chosen)

        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "model_display": MODEL_DISPLAY.get(model_id, model_id),
                "available_runs": int(len(group)),
                "selected_runs": int(len(chosen)),
                "selected_run_ids": " | ".join(chosen["run_id"].astype(str).tolist()),
                "selected_train_sec": " | ".join(f"{float(v):.3f}" for v in chosen["train_sec"].tolist()),
            }
        )

    selected = pd.concat(selected_parts, ignore_index=True)
    baseline = (
        selected[selected["selected_rank"] == 1][["dataset_id", "model_id", "sql_overall_score", "distance_overall_score"]]
        .rename(
            columns={
                "sql_overall_score": "baseline_query_score",
                "distance_overall_score": "baseline_distance_score",
            }
        )
        .reset_index(drop=True)
    )
    selected = selected.merge(baseline, on=["dataset_id", "model_id"], how="left")
    query_den = selected["baseline_query_score"].astype(float)
    dist_den = selected["baseline_distance_score"].astype(float)
    selected["query_ratio"] = np.where(query_den.abs() > 1e-12, selected["sql_overall_score"] / query_den, np.nan)
    selected["query_delta"] = selected["sql_overall_score"] - selected["baseline_query_score"]
    selected["distance_ratio"] = np.where(dist_den.abs() > 1e-12, selected["distance_overall_score"] / dist_den, np.nan)
    selected["distance_delta"] = selected["distance_overall_score"] - selected["baseline_distance_score"]

    for column_name in ["query_ratio", "distance_ratio"]:
        selected[column_name] = pd.to_numeric(selected[column_name], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return selected, pd.DataFrame(summary_rows)


def _style_axes(ax: plt.Axes, title: str, ylabel: str, center_zero: bool = False, baseline_one: bool = False) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Hyperparameter rank (1 = fastest, 5 = slowest)", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(False, axis="x")
    if center_zero:
        ax.axhline(0.0, color="#555555", linewidth=1.0, alpha=0.9)
    if baseline_one:
        ax.axhline(1.0, color="#555555", linewidth=1.0, alpha=0.9)


def _plot_model_lines(summary_df: pd.DataFrame, value_col: str, title: str, ylabel: str, output_png: Path, output_pdf: Path, *, center_zero: bool = False, baseline_one: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    cmap = plt.get_cmap("tab20")
    for idx, model_id in enumerate(MODEL_ORDER):
        model_df = summary_df[summary_df["model_id"] == model_id].sort_values("selected_rank")
        if model_df.empty:
            continue
        ax.plot(
            model_df["selected_rank"],
            model_df[value_col],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=cmap(idx),
            label=MODEL_DISPLAY.get(model_id, model_id),
        )
    _style_axes(ax, title, ylabel, center_zero=center_zero, baseline_one=baseline_one)
    ax.legend(ncol=3, fontsize=9, frameon=False, loc="best")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_overall_lines(overall_df: pd.DataFrame, ratio_col_query: str, ratio_col_distance: str, title: str, ylabel: str, output_png: Path, output_pdf: Path, *, center_zero: bool = False, baseline_one: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(overall_df["selected_rank"], overall_df[ratio_col_query], marker="o", linewidth=2.4, markersize=6, color="#1f77b4", label="Query")
    ax.plot(overall_df["selected_rank"], overall_df[ratio_col_distance], marker="s", linewidth=2.4, markersize=6, color="#d62728", label="Distance")
    _style_axes(ax, title, ylabel, center_zero=center_zero, baseline_one=baseline_one)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_heatmap(
    summary_df: pd.DataFrame,
    value_col: str,
    title: str,
    colorbar_label: str,
    output_png: Path,
    output_pdf: Path,
    *,
    fixed_vmax: float | None = None,
    mark_clipped_black: bool = False,
) -> None:
    wide = (
        summary_df.pivot(index="model_display", columns="selected_rank", values=value_col)
        .reindex([MODEL_DISPLAY.get(model_id, model_id) for model_id in MODEL_ORDER])
    )
    data = wide.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
    if fixed_vmax is not None:
        vmax = float(fixed_vmax)
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    im = ax.imshow(data, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Hyperparameter rank (1 = fastest, 5 = slowest)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Model", fontsize=11, fontweight="bold")
    ax.set_xticks(range(wide.shape[1]))
    ax.set_xticklabels([str(col) for col in wide.columns.tolist()], fontsize=10)
    ax.set_yticks(range(wide.shape[0]))
    ax.set_yticklabels(wide.index.tolist(), fontsize=10)
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            is_clipped = abs(val) > vmax
            if mark_clipped_black and is_clipped:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor="black", edgecolor="white", linewidth=0.8)
                ax.add_patch(rect)
                text_color = "white"
            else:
                text_color = "white" if abs(val) > vmax * 0.45 else "#222"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8.5, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=10, fontweight="bold")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_rank_response_plots(merged_csv: Path, output_root: Path) -> dict[str, Any]:
    df = pd.read_csv(merged_csv)
    selected, selection_summary = select_canonical_five(df)

    output_root.mkdir(parents=True, exist_ok=True)
    selected_csv = output_root / "hyper_rank_response_selected_runs.csv"
    selection_summary_csv = output_root / "hyper_rank_response_selection_summary.csv"
    selected.to_csv(selected_csv, index=False)
    selection_summary.to_csv(selection_summary_csv, index=False)

    model_rank_summary = (
        selected.groupby(["model_id", "model_display", "selected_rank"], as_index=False)[
            ["query_ratio", "query_delta", "distance_ratio", "distance_delta"]
        ]
        .mean()
    )
    overall_rank_summary = (
        selected.groupby("selected_rank", as_index=False)[["query_ratio", "query_delta", "distance_ratio", "distance_delta"]].mean()
    )

    query_ratio_png = output_root / "hyper_query_ratio_by_model.png"
    query_ratio_pdf = output_root / "hyper_query_ratio_by_model.pdf"
    query_delta_png = output_root / "hyper_query_delta_by_model.png"
    query_delta_pdf = output_root / "hyper_query_delta_by_model.pdf"
    distance_ratio_png = output_root / "hyper_distance_ratio_by_model.png"
    distance_ratio_pdf = output_root / "hyper_distance_ratio_by_model.pdf"
    distance_delta_png = output_root / "hyper_distance_delta_by_model.png"
    distance_delta_pdf = output_root / "hyper_distance_delta_by_model.pdf"
    overall_ratio_png = output_root / "hyper_overall_ratio_query_vs_distance.png"
    overall_ratio_pdf = output_root / "hyper_overall_ratio_query_vs_distance.pdf"
    overall_delta_png = output_root / "hyper_overall_delta_query_vs_distance.png"
    overall_delta_pdf = output_root / "hyper_overall_delta_query_vs_distance.pdf"
    query_heatmap_png = output_root / "hyper_query_delta_heatmap.png"
    query_heatmap_pdf = output_root / "hyper_query_delta_heatmap.pdf"
    distance_heatmap_png = output_root / "hyper_distance_delta_heatmap.png"
    distance_heatmap_pdf = output_root / "hyper_distance_delta_heatmap.pdf"
    gap_heatmap_png = output_root / "hyper_query_distance_gap_heatmap.png"
    gap_heatmap_pdf = output_root / "hyper_query_distance_gap_heatmap.pdf"
    shared_heatmap_vmax = float(model_rank_summary["distance_delta"].abs().max())
    shared_heatmap_vmax = max(shared_heatmap_vmax, 1e-6)

    _plot_model_lines(
        model_rank_summary,
        "query_ratio",
        "Query score ratio vs. fastest hyperparameter setting",
        "Query score / rank-1 query score",
        query_ratio_png,
        query_ratio_pdf,
        baseline_one=True,
    )
    _plot_model_lines(
        model_rank_summary,
        "query_delta",
        "Query score delta vs. fastest hyperparameter setting",
        "Query score - rank-1 query score",
        query_delta_png,
        query_delta_pdf,
        center_zero=True,
    )
    _plot_model_lines(
        model_rank_summary,
        "distance_ratio",
        "Distance score ratio vs. fastest hyperparameter setting",
        "Distance score / rank-1 distance score",
        distance_ratio_png,
        distance_ratio_pdf,
        baseline_one=True,
    )
    _plot_model_lines(
        model_rank_summary,
        "distance_delta",
        "Distance score delta vs. fastest hyperparameter setting",
        "Distance score - rank-1 distance score",
        distance_delta_png,
        distance_delta_pdf,
        center_zero=True,
    )
    _plot_overall_lines(
        overall_rank_summary,
        "query_ratio",
        "distance_ratio",
        "Overall average ratio: Query vs. Distance",
        "Score ratio relative to rank 1",
        overall_ratio_png,
        overall_ratio_pdf,
        baseline_one=True,
    )
    _plot_overall_lines(
        overall_rank_summary,
        "query_delta",
        "distance_delta",
        "Overall average delta: Query vs. Distance",
        "Score delta relative to rank 1",
        overall_delta_png,
        overall_delta_pdf,
        center_zero=True,
    )
    _plot_metric_heatmap(
        model_rank_summary,
        "query_delta",
        "Query delta by model and hyperparameter rank",
        "Query delta relative to rank 1",
        query_heatmap_png,
        query_heatmap_pdf,
        fixed_vmax=shared_heatmap_vmax,
        mark_clipped_black=True,
    )
    _plot_metric_heatmap(
        model_rank_summary,
        "distance_delta",
        "Distance delta by model and hyperparameter rank",
        "Distance delta relative to rank 1",
        distance_heatmap_png,
        distance_heatmap_pdf,
        fixed_vmax=shared_heatmap_vmax,
    )
    _plot_metric_heatmap(
        model_rank_summary.assign(gap=lambda d: d["query_delta"] - d["distance_delta"]),
        "gap",
        "Query-Distance delta gap by model and hyperparameter rank",
        "Query delta - Distance delta",
        gap_heatmap_png,
        gap_heatmap_pdf,
        fixed_vmax=shared_heatmap_vmax,
        mark_clipped_black=True,
    )

    summary_md = output_root / "hyper_rank_response_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Hyper rank-response plots",
                "",
                f"- Source merged table: `{merged_csv}`",
                f"- Selected five-run table: `{selected_csv}`",
                f"- Selection summary: `{selection_summary_csv}`",
                "",
                "Generated figures:",
                f"- `{query_ratio_png.name}`",
                f"- `{query_delta_png.name}`",
                f"- `{distance_ratio_png.name}`",
                f"- `{distance_delta_png.name}`",
                f"- `{overall_ratio_png.name}`",
                f"- `{overall_delta_png.name}`",
                f"- `{query_heatmap_png.name}`",
                f"- `{distance_heatmap_png.name}`",
                f"- `{gap_heatmap_png.name}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "selected_csv": str(selected_csv),
        "selection_summary_csv": str(selection_summary_csv),
        "query_ratio_png": str(query_ratio_png),
        "query_delta_png": str(query_delta_png),
        "distance_ratio_png": str(distance_ratio_png),
        "distance_delta_png": str(distance_delta_png),
        "overall_ratio_png": str(overall_ratio_png),
        "overall_delta_png": str(overall_delta_png),
        "query_heatmap_png": str(query_heatmap_png),
        "distance_heatmap_png": str(distance_heatmap_png),
        "gap_heatmap_png": str(gap_heatmap_png),
        "summary_md": str(summary_md),
    }
