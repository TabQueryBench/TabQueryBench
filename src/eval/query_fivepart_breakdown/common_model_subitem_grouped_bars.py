from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _escape_tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = str(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _tex_preamble() -> str:
    return "\n".join(
        [
            r"\documentclass[tikz,border=4pt]{standalone}",
            r"\usepackage{pgfplots}",
            r"\usepackage{xcolor}",
            r"\pgfplotsset{compat=1.18}",
            "",
        ]
    )


def _filtered_subitem_rows(heatmap_df: pd.DataFrame, *, include_summary_row: bool) -> pd.DataFrame:
    if include_summary_row:
        return heatmap_df.copy().reset_index(drop=True)
    out = heatmap_df.loc[heatmap_df["subitem_id"].astype(str).str.lower() != "family_mean"].copy()
    return out.reset_index(drop=True)


def _cluster_layout(
    heatmap_df: pd.DataFrame,
    *,
    model_order: Sequence[str],
    model_label_map: Mapping[str, str],
    include_real_baseline: bool,
    real_label: str,
    real_value: float,
    include_summary_row: bool,
    intra_gap: float,
    cluster_gap: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[float]]:
    plot_df = _filtered_subitem_rows(heatmap_df, include_summary_row=include_summary_row)
    displayed_models = ([("__real__", real_label)] if include_real_baseline else []) + [
        (model_id, model_label_map.get(model_id, model_id)) for model_id in model_order
    ]

    bars: list[dict[str, object]] = []
    clusters: list[dict[str, object]] = []
    separators: list[float] = []
    cursor = 0.0
    for cluster_index, row in enumerate(plot_df.itertuples(index=False)):
        cluster_start = cursor
        for model_key, model_label in displayed_models:
            if model_key == "__real__":
                value = float(real_value)
            else:
                raw = getattr(row, model_key, None)
                value = None if raw is None or pd.isna(raw) else float(raw)
            if value is None:
                cursor += 1.0 + intra_gap
                continue
            bars.append(
                {
                    "cluster_index": cluster_index,
                    "subitem_id": str(row.subitem_id),
                    "subitem_label": str(row.subitem_label),
                    "model_id": str(model_key),
                    "model_label": str(model_label),
                    "x": float(cursor),
                    "score": float(value),
                }
            )
            cursor += 1.0 + intra_gap
        cluster_end = cursor - (1.0 + intra_gap)
        clusters.append(
            {
                "cluster_index": cluster_index,
                "subitem_id": str(row.subitem_id),
                "subitem_label": str(row.subitem_label),
                "start_x": float(cluster_start),
                "end_x": float(cluster_end),
                "center_x": float((cluster_start + cluster_end) / 2.0),
            }
        )
        if cluster_index < len(plot_df) - 1:
            separators.append(float(cursor - intra_gap / 2.0 + cluster_gap / 2.0))
            cursor += cluster_gap
    return bars, clusters, separators


def write_model_subitem_grouped_bar_tex(
    heatmap_df: pd.DataFrame,
    *,
    model_order: Sequence[str],
    model_label_map: Mapping[str, str],
    model_color_map: Mapping[str, str],
    title: str,
    y_label: str,
    path: Path,
    include_real_baseline: bool = True,
    real_label: str = "REAL",
    real_value: float = 1.0,
    include_summary_row: bool = False,
    intra_gap: float = 0.10,
    cluster_gap: float = 1.35,
) -> None:
    if heatmap_df.empty:
        path.write_text("", encoding="utf-8")
        return

    bars, clusters, separators = _cluster_layout(
        heatmap_df,
        model_order=model_order,
        model_label_map=model_label_map,
        include_real_baseline=include_real_baseline,
        real_label=real_label,
        real_value=real_value,
        include_summary_row=include_summary_row,
        intra_gap=intra_gap,
        cluster_gap=cluster_gap,
    )
    if not bars:
        path.write_text("", encoding="utf-8")
        return

    label_lookup = {}
    for item in bars:
        label_lookup[float(item["x"])] = str(item["model_label"])
    x_positions = [float(item["x"]) for item in bars]
    x_labels = [label_lookup[pos] for pos in x_positions]
    max_x = max(x_positions) + 1.0

    color_defs = []
    for model_id in ["__real__", *model_order]:
        color = "#000000" if model_id == "__real__" else str(model_color_map.get(model_id, "#777777"))
        color_defs.append(rf"\definecolor{{bar{model_id.replace('_', '').replace('-', '')}}}{{HTML}}{{{color.replace('#', '')}}}")

    lines = [
        _tex_preamble(),
        *color_defs,
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        rf"width={max(13.5, 0.32 * len(x_positions) + 3.6):.2f}cm,",
        r"height=8.8cm,",
        r"ymin=0.0, ymax=1.08,",
        rf"ylabel={{{_escape_tex(y_label)}}},",
        rf"title={{{_escape_tex(title)}}},",
        r"ymajorgrids,",
        r"grid style={draw=gray!22},",
        r"major grid style={draw=gray!30},",
        r"axis line style={draw=black!70},",
        r"tick style={draw=black!70},",
        rf"xtick={{{','.join(f'{item:.4f}' for item in x_positions)}}},",
        rf"xticklabels={{{','.join(_escape_tex(item) for item in x_labels)}}},",
        r"x tick label style={rotate=90, anchor=east, font=\scriptsize},",
        r"enlarge x limits=0.01,",
        r"clip=false,",
        r"]",
    ]
    for item in bars:
        model_id = str(item["model_id"])
        color_name = f"bar{model_id.replace('_', '').replace('-', '')}"
        lines.append(
            rf"\addplot+[ybar, bar width=5.8pt, draw={color_name}, fill={color_name}] coordinates {{({float(item['x']):.4f},{float(item['score']):.6f})}};"
        )
    for sep in separators:
        lines.append(rf"\draw[dashed, gray!70, line width=0.6pt] (axis cs:{sep:.4f},0) -- (axis cs:{sep:.4f},1.08);")
    for cluster in clusters:
        lines.append(
            rf"\node[anchor=south, font=\bfseries\small] at (axis cs:{float(cluster['center_x']):.4f},1.035) {{{_escape_tex(str(cluster['subitem_label']))}}};"
        )
    lines.extend([r"\end{axis}", r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_model_subitem_grouped_bar_preview(
    heatmap_df: pd.DataFrame,
    *,
    model_order: Sequence[str],
    model_label_map: Mapping[str, str],
    model_color_map: Mapping[str, str],
    title: str,
    y_label: str,
    pdf_path: Path,
    png_path: Path,
    include_real_baseline: bool = True,
    real_label: str = "REAL",
    real_value: float = 1.0,
    include_summary_row: bool = False,
    intra_gap: float = 0.10,
    cluster_gap: float = 1.35,
) -> None:
    if heatmap_df.empty:
        return

    bars, clusters, separators = _cluster_layout(
        heatmap_df,
        model_order=model_order,
        model_label_map=model_label_map,
        include_real_baseline=include_real_baseline,
        real_label=real_label,
        real_value=real_value,
        include_summary_row=include_summary_row,
        intra_gap=intra_gap,
        cluster_gap=cluster_gap,
    )
    if not bars:
        return

    x_positions = [float(item["x"]) for item in bars]
    values = [float(item["score"]) for item in bars]
    x_labels = [str(item["model_label"]) for item in bars]
    colors = ["#000000" if str(item["model_id"]) == "__real__" else str(model_color_map.get(str(item["model_id"]), "#777777")) for item in bars]

    fig_width = max(14.0, 0.33 * len(x_positions) + 3.8)
    fig, ax = plt.subplots(figsize=(fig_width, 7.9))
    ax.bar(x_positions, values, width=0.90, color=colors, edgecolor=colors, linewidth=0.8)
    for sep in separators:
        ax.axvline(sep, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75)
    for cluster in clusters:
        ax.text(
            float(cluster["center_x"]),
            1.035,
            str(cluster["subitem_label"]),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, ha="center", va="top", fontsize=8)
    ax.grid(axis="y", alpha=0.24)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
