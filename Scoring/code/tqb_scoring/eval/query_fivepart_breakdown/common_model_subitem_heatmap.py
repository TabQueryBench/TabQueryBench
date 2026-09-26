from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tqb_scoring.eval.query_fivepart_breakdown.common_heatmap_palette import (
    format_heatmap_latex_cell,
    get_heatmap_cmap,
)


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
            r"\documentclass{standalone}",
            r"\usepackage[table]{xcolor}",
            r"\usepackage{xcolor}",
            r"\usepackage{booktabs}",
            "",
        ]
    )


def build_model_subitem_heatmap_df(
    summary_df: pd.DataFrame,
    *,
    model_id_col: str,
    model_order: Sequence[str],
    subitem_specs: Sequence[tuple[str, str, str]],
    summary_row_spec: tuple[str, str, str] | None = None,
) -> pd.DataFrame:
    columns = ["subitem_id", "subitem_label", *model_order]
    if summary_df.empty:
        return pd.DataFrame(columns=columns)
    indexed = summary_df.set_index(model_id_col, drop=False)
    rows: list[dict[str, float | str | None]] = []
    for subitem_id, subitem_label, mean_col in subitem_specs:
        payload: dict[str, float | str | None] = {
            "subitem_id": subitem_id,
            "subitem_label": subitem_label,
        }
        for model_id in model_order:
            value = None
            if model_id in indexed.index and mean_col in indexed.columns:
                raw = indexed.loc[model_id, mean_col]
                if isinstance(raw, pd.Series):
                    raw = raw.iloc[0]
                value = float(raw) if pd.notna(raw) else None
            payload[model_id] = value
        rows.append(payload)
    if summary_row_spec is not None:
        subitem_id, subitem_label, mean_col = summary_row_spec
        payload = {
            "subitem_id": subitem_id,
            "subitem_label": subitem_label,
        }
        for model_id in model_order:
            value = None
            if model_id in indexed.index and mean_col in indexed.columns:
                raw = indexed.loc[model_id, mean_col]
                if isinstance(raw, pd.Series):
                    raw = raw.iloc[0]
                value = float(raw) if pd.notna(raw) else None
            payload[model_id] = value
        rows.append(payload)
    return pd.DataFrame(rows, columns=columns)


def write_model_subitem_heatmap_tex(
    heatmap_df: pd.DataFrame,
    *,
    model_order: Sequence[str],
    model_label_map: Mapping[str, str],
    title: str,
    colorbar_title: str,
    path: Path,
) -> None:
    if heatmap_df.empty:
        path.write_text("", encoding="utf-8")
        return

    n_cols = len(model_order)
    lines = [
        _tex_preamble(),
        r"\begin{document}",
        r"\scriptsize",
        rf"\textbf{{{_escape_tex(title)}}}\\[0.4em]",
        rf"\emph{{{_escape_tex(colorbar_title)}, 0--1; missing cells stay white.}}\\[0.5em]",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{l{'c' * n_cols}}}",
        r"\toprule",
        "Subitem & " + " & ".join(_escape_tex(model_label_map.get(model_id, model_id)) for model_id in model_order) + r" \\",
        r"\midrule",
    ]
    for row_index, row in enumerate(heatmap_df.itertuples(index=False), start=1):
        _ = row_index
        cells = [_escape_tex(getattr(row, "subitem_label"))]
        for model_id in model_order:
            value = getattr(row, model_id)
            cells.append(format_heatmap_latex_cell(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_model_subitem_heatmap_preview(
    heatmap_df: pd.DataFrame,
    *,
    model_order: Sequence[str],
    model_label_map: Mapping[str, str],
    title: str,
    pdf_path: Path,
    png_path: Path,
) -> None:
    if heatmap_df.empty:
        return
    plot_df = heatmap_df.copy()
    value_matrix = plot_df[model_order].to_numpy(dtype=float)
    fig_height = max(3.8, 0.72 * len(plot_df) + 1.9)
    fig, ax = plt.subplots(figsize=(14.0, fig_height))
    image = ax.imshow(value_matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap=get_heatmap_cmap())
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels([model_label_map.get(model_id, model_id) for model_id in model_order], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["subitem_label"], fontsize=10)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean score")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
