#!/usr/bin/env python3
"""Legacy preview wrapper for a concentration-only tail stress figure."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_tail_stress_main_figure import (
    DECOMP_COLORS,
    DECOMP_LABELS,
    THRESHOLD_ORDER,
    _configure_style,
    _ordered,
    _render_panel_a,
    _render_panel_c,
    _shade_ultra_tail,
    _style_axis,
)
from tqb_scoring.eval.tail_threshold.runner import (
    _build_key_counter,
    _build_transformers,
    _dataset_prefix,
    _is_id_like,
    _load_target_column,
    _mean,
    _read_csv_rows,
    _select_bottom_band,
    _sorted_support_items,
    _threshold_specs,
    resolve_real_split_path,
)


CONCENTRATION_PREVIEW_COLOR = "#2F6690"
CONCENTRATION_PREVIEW_LABEL = "Tail concentration (preview)"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--asset-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-rows-per-table", type=int, default=50000)
    return parser


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _read_csv_rows_limited(path: Path, max_rows: int | None) -> tuple[list[str], list[dict[str, str]]]:
    if not max_rows or max_rows <= 0:
        return _read_csv_rows(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for idx, row in enumerate(reader):
            rows.append(dict(row))
            if idx + 1 >= max_rows:
                break
        columns = [str(col) for col in (reader.fieldnames or [])]
    return columns, rows


def _column_tail_rates(
    counts: dict[str, int],
    tail_real_keys: set[str],
    feature_columns: list[str],
    total_per_column: int,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    if total_per_column <= 0:
        return {column: 0.0 for column in feature_columns}
    for column in feature_columns:
        col_prefix = f"{column}::"
        tail_mass = sum(value for key, value in counts.items() if key in tail_real_keys and key.startswith(col_prefix))
        rates[column] = float(tail_mass) / float(total_per_column)
    return rates


def _tail_concentration_preview(
    real_counts: dict[str, int],
    syn_counts: dict[str, int],
    tail_real_keys: set[str],
    feature_columns: list[str],
    n_real: int,
    n_syn: int,
) -> float:
    if not feature_columns:
        return 1.0
    real_rates = _column_tail_rates(real_counts, tail_real_keys, feature_columns, total_per_column=n_real)
    syn_rates = _column_tail_rates(syn_counts, tail_real_keys, feature_columns, total_per_column=n_syn)
    subgroup_scores: list[float] = []
    for column in feature_columns:
        t_real = float(real_rates.get(column, 0.0))
        t_syn = float(syn_rates.get(column, 0.0))
        if t_real <= 1e-12:
            score = 1.0 if t_syn <= 1e-12 else 0.0
        else:
            score = 1.0 - abs(t_syn - t_real) / max(t_real, 1e-12)
        subgroup_scores.append(_clip01(score))
    return float(sum(subgroup_scores) / len(subgroup_scores)) if subgroup_scores else 1.0


def _compute_dataset_proxy_rows(
    dataset_id: str,
    asset_rows: list[dict[str, Any]],
    threshold_pcts: list[float],
    max_rows_per_table: int,
) -> list[dict[str, Any]]:
    real_csv = resolve_real_split_path(dataset_id, split="train")
    columns, rows_real = _read_csv_rows_limited(real_csv, max_rows_per_table)
    target_column = _load_target_column(dataset_id, columns)
    feature_columns = [column for column in columns if column != target_column and not _is_id_like(column)]
    if not feature_columns or not rows_real:
        return []

    transformers = _build_transformers(rows_real, feature_columns, numeric_bins=10)
    real_counts = _build_key_counter(rows_real, feature_columns, transformers)
    real_tail_items = _sorted_support_items(real_counts, reverse=False)
    threshold_specs = _threshold_specs(threshold_pcts)
    real_tail_map = {spec.label: _select_bottom_band(real_tail_items, spec.ratio)[0] for spec in threshold_specs}
    n_real = len(rows_real)

    results: list[dict[str, Any]] = []
    deduped_assets = {}
    for row in asset_rows:
        deduped_assets[str(row["asset_key"])] = row

    for asset in deduped_assets.values():
        _, rows_syn = _read_csv_rows_limited(Path(str(asset["synthetic_csv_path"])), max_rows_per_table)
        syn_counts = _build_key_counter(rows_syn, feature_columns, transformers)
        n_syn = len(rows_syn)
        for spec in threshold_specs:
            score = _tail_concentration_preview(
                real_counts=real_counts,
                syn_counts=syn_counts,
                tail_real_keys=real_tail_map[spec.label],
                feature_columns=feature_columns,
                n_real=n_real,
                n_syn=n_syn,
            )
            results.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_prefix": _dataset_prefix(dataset_id),
                    "asset_key": asset["asset_key"],
                    "model_id": asset["model_id"],
                    "model_label": asset["model_label"],
                    "threshold_label": spec.label,
                    "threshold_pct": spec.pct,
                    "tail_concentration_consistency_preview": round(score, 6),
                }
            )
    return results


def _compute_proxy_summary(asset_summary_csv: Path, max_workers: int, max_rows_per_table: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets_df = pd.read_csv(asset_summary_csv)
    threshold_pcts = [float(text.rstrip("%")) for text in THRESHOLD_ORDER]
    dataset_to_rows: dict[str, list[dict[str, Any]]] = {}
    for row in assets_df.to_dict("records"):
        dataset_to_rows.setdefault(str(row["dataset_id"]), []).append(row)

    results: list[dict[str, Any]] = []
    if max_workers <= 1:
        for dataset_id, rows in dataset_to_rows.items():
            results.extend(_compute_dataset_proxy_rows(dataset_id, rows, threshold_pcts, max_rows_per_table))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_compute_dataset_proxy_rows, dataset_id, rows, threshold_pcts, max_rows_per_table): dataset_id
                for dataset_id, rows in dataset_to_rows.items()
            }
            for future in as_completed(futures):
                results.extend(future.result())

    proxy_df = pd.DataFrame(results)
    proxy_df["threshold_label"] = pd.Categorical(proxy_df["threshold_label"], categories=THRESHOLD_ORDER, ordered=True)
    global_proxy = (
        proxy_df.groupby(["threshold_label"], observed=True)["tail_concentration_consistency_preview"]
        .mean()
        .reset_index()
        .rename(columns={"tail_concentration_consistency_preview": "tail_concentration_consistency_preview_mean"})
    )
    return proxy_df, global_proxy


def _render_panel_b_with_proxy(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    x = list(range(len(global_df)))
    baseline = global_df.iloc[0]

    primary_metric_order = [
        "tail_set_consistency_mean",
        "tail_mass_similarity_mean",
        "tail_concentration_consistency_mean",
    ]
    subgroup_metric = "tail_concentration_consistency_preview_mean"

    for metric in primary_metric_order:
        base_value = float(baseline[metric])
        values = [float(v) / base_value if base_value else 0.0 for v in global_df[metric].tolist()]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            color=DECOMP_COLORS[metric],
            label=DECOMP_LABELS[metric],
        )

    _shade_ultra_tail(ax)
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, global_df["threshold_label"].tolist(), rotation=30, ha="right")
    ax.set_ylim(0.58, 1.12)
    ax.set_ylabel("Relative score vs. 10% threshold")
    ax.set_xlabel("Tail threshold")
    ax.set_title("B. Identity, mass, and subgroup-tail allocation break first", loc="left", pad=6)
    _style_axis(ax)
    primary_legend = ax.legend(loc="lower left", frameon=False, fontsize=7.3)
    ax.text(
        0.61,
        0.88,
        "Purple stays stable because it\nonly checks the tail's internal shape",
        transform=ax.transAxes,
        color=DECOMP_COLORS["tail_concentration_consistency_mean"],
        fontsize=7.4,
        ha="left",
        va="top",
    )
    ax.add_artist(primary_legend)

    ax2 = ax.twinx()
    base_value = float(baseline[subgroup_metric])
    subgroup_values = [float(v) / base_value if base_value else 0.0 for v in global_df[subgroup_metric].tolist()]
    ax2.plot(
        x,
        subgroup_values,
        marker="s",
        linewidth=1.8,
        markersize=4.0,
        linestyle="--",
        color=SUBGROUP_PROXY_COLOR,
        label=SUBGROUP_PROXY_LABEL,
    )
    ax2.set_ylabel("Proxy relative score", color=SUBGROUP_PROXY_COLOR)
    ax2.tick_params(axis="y", colors=SUBGROUP_PROXY_COLOR, labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.grid(False)
    ymin = min(subgroup_values)
    ymax = max(subgroup_values)
    pad = max(0.03, 0.08 * (ymax - ymin))
    ax2.set_ylim(ymin - pad, ymax + pad)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=7.2)
    ax2.text(
        0.985,
        0.08,
        "Right axis only",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=SUBGROUP_PROXY_COLOR,
    )


def render_preview(
    tables_dir: Path,
    asset_summary_csv: Path,
    output_dir: Path,
    max_workers: int,
    max_rows_per_table: int,
) -> tuple[Path, Path]:
    global_df = _ordered(pd.read_csv(tables_dir / "global_threshold_summary.csv"))
    model_threshold_df = pd.read_csv(tables_dir / "model_threshold_summary.csv")
    proxy_rows, global_proxy = _compute_proxy_summary(
        asset_summary_csv,
        max_workers=max_workers,
        max_rows_per_table=max_rows_per_table,
    )
    merged = global_df.merge(global_proxy, on="threshold_label", how="left")

    _configure_style()
    fig = plt.figure(figsize=(7.4, 6.7), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)
    mosaic = fig.subplot_mosaic([["T", "T"], ["A", "B"], ["C", "C"]], height_ratios=[0.18, 1.0, 1.02])
    mosaic["T"].axis("off")
    mosaic["T"].text(
        0.0,
        0.72,
        "Tail stress testing with subgroup-tail allocation preview",
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
    )
    mosaic["T"].text(
        0.995,
        0.20,
        f"Preview only: subgroup metric is a column-subgroup proxy | sampled first {max_rows_per_table:,} rows/table",
        fontsize=7.4,
        color="#666666",
        ha="right",
        va="center",
    )

    _render_panel_a(mosaic["A"], global_df)
    _render_panel_b_with_proxy(mosaic["B"], merged)
    _render_panel_c(mosaic["C"], model_threshold_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_rows.to_csv(output_dir / "tail_concentration_preview_asset_rows.csv", index=False)
    merged.to_csv(output_dir / "global_threshold_with_concentration_preview.csv", index=False)
    png_path = output_dir / "tail_stress_main_figure_with_concentration_preview.png"
    pdf_path = output_dir / "tail_stress_main_figure_with_concentration_preview.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = _build_parser().parse_args()
    png_path, pdf_path = render_preview(
        tables_dir=args.tables_dir,
        asset_summary_csv=args.asset_summary_csv,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        max_rows_per_table=args.max_rows_per_table,
    )
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
