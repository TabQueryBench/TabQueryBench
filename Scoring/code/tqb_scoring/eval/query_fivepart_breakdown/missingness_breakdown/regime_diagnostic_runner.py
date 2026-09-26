#!/usr/bin/env python3
"""Build missingness regime diagnostic grouped-bar figures.

This diagnostic compares categorical, mixed, and numerical regimes for:

1. missingness_structure_score
2. co_missingness_pattern_consistency
3. marginal_missing_rate_consistency

It writes paper-friendly PNG/PDF previews plus concise text insights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = PROJECT_ROOT.parent / "results" / "query_fivepart_breakdown" / "missingness_breakdown" / "regime_diagnostic"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"
FINAL_DIR = OUTPUT_ROOT / "final"
SOURCE_CSV = (
    PROJECT_ROOT.parent / "results"
    / "query_fivepart_breakdown"
    / "missingness_breakdown"
    / "final"
    / "prefix_summary__v2.csv"
)

MODEL_COLORS = {
    "REAL": "#000000",
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

MODEL_ORDER = [
    "ARF",
    "BayesNet",
    "CTGAN",
    "ForestDiffusion",
    "RealTabFormer",
    "TabbyFlow",
    "TabDDPM",
    "TabDiff",
    "TabPFGen",
    "TabSyn",
    "TVAE",
]

PREFIX_ORDER = ["c", "m", "n"]
PREFIX_LABELS = {"c": "Categorical", "m": "Mixed", "n": "Numerical"}
PREFIX_COLORS = {"c": "#E6C229", "m": "#3FA7D6", "n": "#D1495B"}

METRIC_SPECS = [
    (
        "missingness_structure_score",
        "missingness_regime_grouped_bars_main",
        "Missingness family score across regimes",
    ),
    (
        "co_missingness_pattern_consistency",
        "missingness_regime_grouped_bars_profile_appendix",
        "Co-missingness profile score across regimes",
    ),
    (
        "marginal_missing_rate_consistency",
        "missingness_regime_grouped_bars_marginal_appendix",
        "Marginal missing-rate score across regimes",
    ),
]


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, DATA_DIR, FIG_DIR, FINAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _load_prefix_summary() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV)
    df = df.rename(columns={"model_label": "Model", "dataset_prefix": "Prefix"})
    df["Model"] = df["Model"].astype(str)
    df["Prefix"] = df["Prefix"].astype(str)
    return df


def _pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = (
        df.pivot(index="Model", columns="Prefix", values=metric)
        .reindex(index=MODEL_ORDER, columns=PREFIX_ORDER)
        .reset_index()
    )
    return pivot


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def _plot_grouped_bars(metric_df: pd.DataFrame, title: str, pdf_path: Path, png_path: Path) -> None:
    models = metric_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.23
    fig, ax = plt.subplots(figsize=(13.6, 5.8))

    for idx, prefix in enumerate(PREFIX_ORDER):
        values = pd.to_numeric(metric_df[prefix], errors="coerce").to_numpy(dtype=float)
        offset = (idx - 1) * width
        mask = ~np.isnan(values)
        ax.bar(
            x[mask] + offset,
            values[mask],
            width=width,
            color=PREFIX_COLORS[prefix],
            label=PREFIX_LABELS[prefix],
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_title(title, fontsize=15)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=90, ha="center", va="top")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _tex_escape(text: str) -> str:
    escaped = str(text)
    for src, dst in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        escaped = escaped.replace(src, dst)
    return escaped


def _write_grouped_bars_tex(metric_df: pd.DataFrame, title: str, tex_path: Path) -> None:
    models = metric_df["Model"].tolist()
    symbolic = ",".join(_tex_escape(model) for model in models)
    bar_width = "8pt"
    lines = [
        r"\documentclass[tikz,border=4pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\usepackage{xcolor}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        f"title={{{_tex_escape(title)}}},",
        r"width=15.6cm,",
        r"height=7.0cm,",
        r"ymin=0, ymax=1.02,",
        r"ylabel={Score},",
        r"xlabel={Model},",
        r"ymajorgrids=true,",
        r"grid style={gray!25},",
        r"legend style={draw=none, fill=none, at={(0.98,0.98)}, anchor=north east},",
        r"legend columns=3,",
        f"symbolic x coords={{{symbolic}}},",
        r"xtick=data,",
        r"x tick label style={rotate=90, anchor=east, font=\small},",
        r"enlarge x limits=0.05,",
        f"bar width={bar_width},",
        r"]",
    ]

    for idx, prefix in enumerate(PREFIX_ORDER):
        color = PREFIX_COLORS[prefix]
        label = PREFIX_LABELS[prefix]
        shift = (-1 + idx) * 10
        coords: list[str] = []
        for row in metric_df.itertuples(index=False):
            value = getattr(row, prefix)
            if pd.isna(value):
                continue
            coords.append(f"({_tex_escape(row.Model)},{float(value):.6f})")
        lines.extend(
            [
                rf"\addplot+[ybar, bar shift={shift}pt, draw=white, fill={color}] coordinates {{",
                " ".join(coords),
                r"};",
                rf"\addlegendentry{{{_tex_escape(label)}}}",
            ]
        )

    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def _build_story_txt(main_df: pd.DataFrame) -> str:
    numeric = pd.to_numeric(main_df["n"], errors="coerce")
    categorical = pd.to_numeric(main_df["c"], errors="coerce")
    mixed = pd.to_numeric(main_df["m"], errors="coerce")
    work = main_df.copy()
    work["drop_c_to_n"] = categorical - numeric
    work["drop_m_to_n"] = mixed - numeric
    work["range_max_min"] = pd.concat([categorical, mixed, numeric], axis=1).max(axis=1) - pd.concat([categorical, mixed, numeric], axis=1).min(axis=1)

    valid_n = work.loc[work["n"].notna()].copy()
    valid_n = valid_n.sort_values("n", ascending=False)
    largest_drop = valid_n.sort_values("drop_m_to_n", ascending=False)

    lines = [
        "Insight 1: Numerical missingness is the hardest regime for most models.",
        (
            "Across the regime summary, most models follow a categorical -> mixed -> numerical decline or at least end lower on numerical than on the other two regimes. "
            f"The numerical column is especially weak for ARF ({_format_value(valid_n.loc[valid_n['Model']=='ARF', 'n'].iloc[0])}), "
            f"BayesNet ({_format_value(valid_n.loc[valid_n['Model']=='BayesNet', 'n'].iloc[0])}), "
            f"CTGAN ({_format_value(valid_n.loc[valid_n['Model']=='CTGAN', 'n'].iloc[0])}), "
            f"TabSyn ({_format_value(valid_n.loc[valid_n['Model']=='TabSyn', 'n'].iloc[0])}), and TabbyFlow ({_format_value(valid_n.loc[valid_n['Model']=='TabbyFlow', 'n'].iloc[0])})."
        ),
        "",
        "Insight 2: Some models collapse sharply on numerical missingness, but RealTabFormer remains stable.",
        (
            f"RealTabFormer stays high in all three regimes with c/m/n = "
            f"{_format_value(valid_n.loc[valid_n['Model']=='RealTabFormer', 'c'].iloc[0])}/"
            f"{_format_value(valid_n.loc[valid_n['Model']=='RealTabFormer', 'm'].iloc[0])}/"
            f"{_format_value(valid_n.loc[valid_n['Model']=='RealTabFormer', 'n'].iloc[0])}. "
            f"By contrast, the largest mixed-to-numerical drops come from "
            f"{largest_drop.iloc[0]['Model']} ({_format_value(largest_drop.iloc[0]['m'])} -> {_format_value(largest_drop.iloc[0]['n'])}), "
            f"{largest_drop.iloc[1]['Model']} ({_format_value(largest_drop.iloc[1]['m'])} -> {_format_value(largest_drop.iloc[1]['n'])}), and "
            f"{largest_drop.iloc[2]['Model']} ({_format_value(largest_drop.iloc[2]['m'])} -> {_format_value(largest_drop.iloc[2]['n'])}). "
            "This suggests that preserving numerical missingness structure is not just uniformly harder; for some models it is a clear regime-specific failure mode."
        ),
    ]
    return "\n".join(lines) + "\n"


def _build_manifest(metric_tables: dict[str, str]) -> dict[str, Any]:
    return {
        "task": "missingness_regime_diagnostic",
        "source_csv": str(SOURCE_CSV),
        "output_root": str(OUTPUT_ROOT),
        "metric_tables": metric_tables,
        "model_order": MODEL_ORDER,
        "prefix_order": PREFIX_ORDER,
    }


def run() -> dict[str, Any]:
    _ensure_dirs()
    df = _load_prefix_summary()
    metric_tables: dict[str, str] = {}

    main_metric_df: pd.DataFrame | None = None
    for metric, stem, title in METRIC_SPECS:
        metric_df = _pivot_metric(df, metric)
        _write_csv(metric_df, DATA_DIR / f"{stem}.csv")
        _write_grouped_bars_tex(metric_df, title, FIG_DIR / f"{stem}.tex")
        _plot_grouped_bars(metric_df, title, FIG_DIR / f"{stem}.pdf", FIG_DIR / f"{stem}.png")
        metric_tables[metric] = f"data/{stem}.csv"
        if metric == "missingness_structure_score":
            main_metric_df = metric_df

    if main_metric_df is None:
        raise RuntimeError("Main metric table was not built.")

    story_txt = _build_story_txt(main_metric_df)
    (FINAL_DIR / "missingness_regime_insights.txt").write_text(story_txt, encoding="utf-8")
    (FINAL_DIR / "README.txt").write_text(
        "\n".join(
            [
                "Missingness regime diagnostic bundle",
                "",
                "Main figure:",
                "- figures/missingness_regime_grouped_bars_main.png",
                "- figures/missingness_regime_grouped_bars_main.tex",
                "",
                "Appendix figures:",
                "- figures/missingness_regime_grouped_bars_profile_appendix.png",
                "- figures/missingness_regime_grouped_bars_profile_appendix.tex",
                "- figures/missingness_regime_grouped_bars_marginal_appendix.png",
                "- figures/missingness_regime_grouped_bars_marginal_appendix.tex",
                "",
                "Text insight:",
                "- final/missingness_regime_insights.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = _build_manifest(metric_tables)
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
