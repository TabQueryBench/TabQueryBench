from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "DejaVu Serif"

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path("/Users/jialinzhang/Documents/HKUNAISS/SyntheticNips/SQLagent")
OVERVIEW_DATA_DIR = ROOT / "Evaluation" / "overview_regenerated" / "data"
OVERVIEW_FIG_DIR = ROOT / "Evaluation" / "overview_regenerated" / "figures"
TAX_MODEL_SUMMARY = (
    ROOT
    / "Evaluation"
    / "paper_final_registry"
    / "query_taxonomy_authoritative"
    / "query_taxonomy_model_scores_authoritative.csv"
)
CLASSICAL_MODEL_SUMMARY = (
    ROOT
    / "Evaluation"
    / "benchmark_overall_table"
    / "final"
    / "benchmark_overall_table_real_model_summary.csv"
)

QUERY_PANEL_SOURCE = OVERVIEW_DATA_DIR / "overview_sql_panel_source_taxonomy.csv"
QUERY_PANEL_SVG = OVERVIEW_FIG_DIR / "overview_query_panel_figma_style_taxonomy.svg"
RANK_SOURCE = OVERVIEW_DATA_DIR / "overview_query_vs_distance_avg_rank_source_taxonomy.csv"
RANK_SVG = OVERVIEW_FIG_DIR / "overview_query_vs_distance_avg_rank_compare_taxonomy.svg"
BAR_SOURCE = OVERVIEW_DATA_DIR / "overview_model_bar_chart_source_taxonomy.csv"
BAR_SVG = OVERVIEW_FIG_DIR / "overview_model_bar_chart_taxonomy.svg"

QUERY_TITLE = "Workload-Grounded Query Fidelity"
QUERY_COLOR = "#4b4be0"
GRID_COLOR = "#ebebeb"
FRAME_COLOR = "#d8d8d8"
SEPARATOR_COLOR = "#7f7f7f"
TEXT_COLOR = "#444444"

MODEL_ORDER = [
    "realtabformer",
    "bayesnet",
    "tabpfgen",
    "arf",
    "ctgan",
    "tvae",
    "tabdiff",
    "tabbyflow",
    "tabsyn",
    "forestdiffusion",
    "tabddpm",
]

PANEL_MODEL_ORDER = [
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

MODEL_SHORT_LABELS = {
    "arf": "ARF",
    "bayesnet": "BayesNet",
    "ctgan": "CTGAN",
    "forestdiffusion": "ForestDiff",
    "realtabformer": "RTF",
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

FAMILY_AXES = [
    ("subgroup_family_taxonomy", "Subgroup"),
    ("conditional_family_taxonomy", "Conditional"),
    ("tail_rarity_family_taxonomy", "Tail / Rarity"),
    ("missingness_family_taxonomy", "Missingness"),
    ("cardinality_range_family_taxonomy", "Cardinality / Range"),
]


def _taxonomy_model_table() -> pd.DataFrame:
    df = pd.read_csv(TAX_MODEL_SUMMARY)
    df["model_id"] = df["model_id"].astype(str).str.lower()
    return df[(df["row_kind"] == "synthetic") & (df["model_id"].isin(MODEL_COLORS))].copy()


def _classical_model_table() -> pd.DataFrame:
    df = pd.read_csv(CLASSICAL_MODEL_SUMMARY)
    df["model_id"] = df["model_id"].astype(str).str.lower()
    return df[(df["row_kind"] == "synthetic") & (df["model_id"].isin(MODEL_COLORS))].copy()


def _build_query_panel_source() -> pd.DataFrame:
    tax = _taxonomy_model_table()
    panel_order_lookup = {model_id: i for i, model_id in enumerate(PANEL_MODEL_ORDER, start=1)}
    rows: list[dict[str, object]] = []
    for axis_order, (metric_key, axis_label) in enumerate(FAMILY_AXES, start=1):
        subset = tax[tax["metric_key"] == metric_key].copy()
        for _, row in subset.iterrows():
            model_id = str(row["model_id"])
            if model_id not in panel_order_lookup:
                continue
            rows.append(
                {
                    "panel_name": "overview_sql_panel_taxonomy",
                    "model_id": model_id,
                    "model_label": MODEL_LABELS[model_id],
                    "model_color": MODEL_COLORS[model_id],
                    "panel_model_order": panel_order_lookup[model_id],
                    "axis_order": axis_order,
                    "axis_field": metric_key,
                    "axis_label": axis_label,
                    "axis_value": float(row["mean_value"]),
                    "source_value_raw": float(row["mean_value"]),
                    "value_transform": "none",
                    "source_of_truth_model_summary_csv": str(TAX_MODEL_SUMMARY.resolve()),
                    "source_of_truth_line_version": "taxonomy",
                    "source_of_truth_line_label": "query_taxonomy_authoritative",
                    "source_field_note": "Direct family score means from query_taxonomy_model_scores_authoritative.csv",
                }
            )
    out = pd.DataFrame(rows).sort_values(["axis_order", "panel_model_order"]).reset_index(drop=True)
    QUERY_PANEL_SOURCE.write_text(out.to_csv(index=False), encoding="utf-8")
    return out


def _panel_groups(df: pd.DataFrame) -> tuple[list[str], dict[str, list[dict[str, object]]]]:
    axis_order = df[["axis_order", "axis_label"]].drop_duplicates().sort_values("axis_order")
    axes = axis_order["axis_label"].tolist()
    grouped: dict[str, list[dict[str, object]]] = {}
    for axis in axes:
        grouped[axis] = df[df["axis_label"] == axis].sort_values("panel_model_order").to_dict("records")
    return axes, grouped


def _draw_query_panel(df: pd.DataFrame) -> None:
    axes, grouped = _panel_groups(df)
    fig, axs = plt.subplots(
        1,
        len(axes),
        figsize=(15.2, 3.6),
        sharey=True,
        gridspec_kw={"wspace": 0.08, "left": 0.055, "right": 0.985, "top": 0.77, "bottom": 0.20},
    )
    fig.patch.set_facecolor("white")
    fig.add_artist(
        Rectangle(
            (0.006, 0.01),
            0.988,
            0.98,
            transform=fig.transFigure,
            fill=False,
            linewidth=1.0,
            edgecolor=FRAME_COLOR,
        )
    )
    fig.text(0.035, 0.90, QUERY_TITLE, ha="left", va="center", fontsize=19, fontweight="bold", color=QUERY_COLOR)

    for idx, (ax, axis_label) in enumerate(zip(axs, axes, strict=True)):
        rows = grouped[axis_label]
        x = list(range(len(rows)))
        values = [float(r["axis_value"]) * 100.0 for r in rows]
        colors = [str(r["model_color"]) for r in rows]
        ax.bar(x, values, color=colors, width=0.52, edgecolor="none")
        ax.set_ylim(0, 120)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_xticks([])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="y", labelsize=9, colors=TEXT_COLOR, length=0)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        title_size = 13.2 if len(axis_label) > 12 else 15
        ax.text(
            0.5,
            0.98,
            axis_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=title_size,
            color=QUERY_COLOR,
        )
        for xi, val in zip(x, values, strict=True):
            ax.text(xi, val + 2.3, f"{int(round(val))}", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if idx < len(axes) - 1:
            bbox = ax.get_position()
            xline = bbox.x1 + 0.006
            fig.add_artist(
                Line2D(
                    [xline, xline],
                    [bbox.y0 - 0.02, bbox.y1 + 0.01],
                    transform=fig.transFigure,
                    color=SEPARATOR_COLOR,
                    linewidth=1.0,
                )
            )

    QUERY_PANEL_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(QUERY_PANEL_SVG, format="svg", transparent=False)
    plt.close(fig)


def _build_rank_source() -> pd.DataFrame:
    tax = _taxonomy_model_table()
    classical = _classical_model_table()
    query = (
        tax[tax["metric_key"] == "query_overall_taxonomy10"][["model_id", "mean_value"]]
        .rename(columns={"mean_value": "query_score"})
    )
    dist = classical[["model_id", "distance_overall_mean"]].rename(columns={"distance_overall_mean": "distance_score"})
    merged = query.merge(dist, on="model_id", how="inner")
    merged["query_rank"] = merged["query_score"].rank(method="min", ascending=False).astype(int)
    merged["distance_rank"] = merged["distance_score"].rank(method="min", ascending=False).astype(int)
    merged["model_label"] = merged["model_id"].map(MODEL_LABELS)
    merged["model_short_label"] = merged["model_id"].map(MODEL_SHORT_LABELS)
    merged["model_color"] = merged["model_id"].map(MODEL_COLORS)
    top_query = merged.sort_values(["query_rank", "model_label"]).head(5)
    top_distance = merged.sort_values(["distance_rank", "model_label"]).head(5)
    rows = []
    for rank in range(1, 6):
        q = top_query[top_query["query_rank"] == rank].iloc[0]
        d = top_distance[top_distance["distance_rank"] == rank].iloc[0]
        rows.append(
            {
                "rank": rank,
                "query_model_id": q["model_id"],
                "query_model_label": q["model_label"],
                "query_model_short_label": q["model_short_label"],
                "query_score_100": float(q["query_score"]) * 100.0,
                "query_color": q["model_color"],
                "bottom_model_id": d["model_id"],
                "bottom_model_label": d["model_label"],
                "bottom_model_short_label": d["model_short_label"],
                "bottom_score_100": float(d["distance_score"]) * 100.0,
                "bottom_color": d["model_color"],
                "same_model_at_rank": bool(q["model_id"] == d["model_id"]),
            }
        )
    out = pd.DataFrame(rows)
    RANK_SOURCE.write_text(out.to_csv(index=False), encoding="utf-8")
    return out


def _svg_text(x: float, y: float, value: str, size: float, *, fill: str = "#111111", weight: str = "400", anchor: str = "middle") -> str:
    esc = (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" font-family="Arial, DejaVu Sans, sans-serif">{esc}</text>'
    )


def _round_rect(x: float, y: float, w: float, h: float, rx: float, *, fill: str = "#ffffff", stroke: str = "#333333", sw: float = 2.0) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'


def _rank_badge(rank: int, cx: float, cy: float) -> str:
    if rank == 1:
        fill = "#f4c430"
    elif rank == 2:
        fill = "#c0c6d4"
    elif rank == 3:
        fill = "#d69456"
    else:
        fill = "#ffffff"
    pieces = [
        f'<circle cx="{cx}" cy="{cy}" r="20" fill="{fill}" stroke="#2f3747" stroke-width="2" />',
        _svg_text(cx, cy + 6, str(rank), 18, fill="#1d2432", weight="700"),
    ]
    if rank <= 3:
        pieces.append(
            f'<path d="M {cx-10} {cy-25} L {cx-4} {cy-34} L {cx} {cy-27} L {cx+4} {cy-34} L {cx+10} {cy-25} Z" fill="{fill}" stroke="#2f3747" stroke-width="1.5" />'
        )
    return "".join(pieces)


def _metric_chip(x: float, y: float, label: str, stroke: str, fill: str) -> str:
    return "".join(
        [
            f'<rect x="{x}" y="{y}" width="170" height="32" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>',
            _svg_text(x + 85, y + 22, label, 16, fill=stroke, weight="700"),
        ]
    )


def _query_icon(x: float, y: float) -> str:
    return "".join(
        [
            f'<rect x="{x}" y="{y}" width="26" height="34" rx="4" fill="none" stroke="#5140c8" stroke-width="2"/>',
            f'<line x1="{x+6}" y1="{y+10}" x2="{x+18}" y2="{y+10}" stroke="#5140c8" stroke-width="2"/>',
            f'<line x1="{x+6}" y1="{y+18}" x2="{x+18}" y2="{y+18}" stroke="#5140c8" stroke-width="2"/>',
            f'<circle cx="{x+30}" cy="{y+24}" r="8" fill="none" stroke="#5140c8" stroke-width="2"/>',
            f'<line x1="{x+36}" y1="{y+30}" x2="{x+44}" y2="{y+38}" stroke="#5140c8" stroke-width="3"/>',
        ]
    )


def _distance_icon(x: float, y: float) -> str:
    return "".join(
        [
            f'<path d="M {x+14} {y} C {x+22} {y+12}, {x+28} {y+18}, {x+28} {y+28} C {x+28} {y+40}, {x+22} {y+48}, {x+14} {y+48} C {x+6} {y+48}, {x} {y+40}, {x} {y+28} C {x} {y+18}, {x+6} {y+12}, {x+14} {y} Z" fill="#ff661f" fill-opacity="0.15" stroke="#ff661f" stroke-width="2"/>',
            f'<path d="M {x+34} {y+38} q 10 -8 20 0 q 10 8 20 0" fill="none" stroke="#ff661f" stroke-width="2.5"/>',
        ]
    )


def _model_card(x: float, y: float, rank: int, short_label: str, score: float, color: str, accent: str) -> str:
    return "".join(
        [
            _round_rect(x, y, 210, 108, 18, fill="#ffffff", stroke=accent, sw=2.2),
            _rank_badge(rank, x + 26, y + 28),
            f'<rect x="{x+56}" y="{y+18}" width="132" height="30" rx="15" fill="{color}" fill-opacity="0.16" stroke="{color}" stroke-width="1.5"/>',
            _svg_text(x + 122, y + 39, short_label, 17, fill="#1f2430", weight="700"),
            _svg_text(x + 105, y + 73, f"score {int(round(score))}", 14, fill="#6a7282", weight="600"),
            _svg_text(x + 105, y + 95, f"Rank {rank}", 13, fill=accent, weight="700"),
        ]
    )


def _draw_rank_compare(df: pd.DataFrame) -> None:
    W, H = 1540, 560
    DISPLAY_W, DISPLAY_H = 770, 280
    left_margin = 210
    top_row_y = 140
    bottom_row_y = 340
    card_w = 210
    card_h = 108
    gap = 36
    start_x = left_margin
    title_y = 42
    pieces = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{DISPLAY_W}" height="{DISPLAY_H}" viewBox="0 0 {W} {H}">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff" />',
        f'<rect x="18" y="18" width="{W-36}" height="{H-36}" rx="26" fill="#fffdfb" stroke="#e9e5df" stroke-width="1.5" />',
        _svg_text(W / 2, title_y, "Query Overall (taxonomy-10) vs Distance Overall Ranking", 28, fill="#202532", weight="700"),
        _svg_text(W / 2, 70, "Top-5 side-by-side alignment compares the taxonomy-side overall ranking with the classical distance-overall ranking", 16, fill="#697180", weight="500"),
        _query_icon(48, 120),
        _svg_text(106, 145, "Query Overall", 24, fill="#5140c8", weight="700", anchor="start"),
        _metric_chip(44, 160, "taxonomy-10 overall", "#5140c8", "#f3f0ff"),
        _distance_icon(42, 320),
        _svg_text(106, 350, "Distance Overall", 24, fill="#ff661f", weight="700", anchor="start"),
        _metric_chip(44, 366, "classical overall", "#ff661f", "#fff2ea"),
    ]
    for row in df.to_dict("records"):
        rank = int(row["rank"])
        x = start_x + (rank - 1) * (card_w + gap)
        pieces.append(_model_card(x, top_row_y, rank, str(row["query_model_short_label"]), float(row["query_score_100"]), str(row["query_color"]), "#5140c8"))
        pieces.append(_model_card(x, bottom_row_y, rank, str(row["bottom_model_short_label"]), float(row["bottom_score_100"]), str(row["bottom_color"]), "#ff661f"))
        same = bool(row["same_model_at_rank"])
        symbol = "=" if same else "&#8800;"
        symbol_fill = "#2d8a54" if same else "#d94841"
        cx = x + card_w / 2
        cy = (top_row_y + bottom_row_y + card_h) / 2
        pieces.append(f'<circle cx="{cx}" cy="{cy}" r="20" fill="{symbol_fill}" fill-opacity="0.12" stroke="{symbol_fill}" stroke-width="1.5" />')
        pieces.append(f'<text x="{cx}" y="{cy + 7}" text-anchor="middle" font-size="24" font-weight="700" fill="{symbol_fill}" font-family="Arial, DejaVu Sans, sans-serif">{symbol}</text>')
        pieces.append(f'<line x1="{cx}" y1="{top_row_y + card_h + 10}" x2="{cx}" y2="{bottom_row_y - 12}" stroke="{symbol_fill}" stroke-width="2.2" stroke-dasharray="6,6" stroke-opacity="0.55" />')
    pieces.extend(
        [
            _svg_text(1318, 92, "Same model at same rank", 15, fill="#5f6777", weight="700", anchor="start"),
            f'<circle cx="1288" cy="86" r="12" fill="#2d8a54" fill-opacity="0.12" stroke="#2d8a54" stroke-width="1.5" />',
            _svg_text(1288, 92, "=", 16, fill="#2d8a54", weight="700"),
            _svg_text(1318, 122, "Different model at same rank", 15, fill="#5f6777", weight="700", anchor="start"),
            f'<circle cx="1288" cy="116" r="12" fill="#d94841" fill-opacity="0.12" stroke="#d94841" stroke-width="1.5" />',
            f'<text x="1288" y="122" text-anchor="middle" font-size="16" font-weight="700" fill="#d94841" font-family="Arial, DejaVu Sans, sans-serif">&#8800;</text>',
        ]
    )
    pieces.append("</svg>")
    RANK_SVG.write_text("\n".join(pieces), encoding="utf-8")


def _draw_query_overall_bar() -> None:
    tax = _taxonomy_model_table()
    df = (
        tax[tax["metric_key"] == "query_overall_taxonomy10"][["model_id", "mean_value", "std_value", "dataset_count", "rank_dense"]]
        .copy()
    )
    df["model_label"] = df["model_id"].map(MODEL_LABELS)
    df["model_color"] = df["model_id"].map(MODEL_COLORS)
    df = df.sort_values(["mean_value", "model_label"], ascending=[False, True]).reset_index(drop=True)
    BAR_SOURCE.write_text(df.to_csv(index=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10.8, 3.8), constrained_layout=True)
    x = list(range(len(df)))
    ax.bar(x, df["mean_value"], color=df["model_color"], edgecolor="#333333", linewidth=0.6, zorder=3)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_label"], rotation=30, ha="right")
    ax.set_title("Model Comparison on Query Overall (taxonomy-10)", fontsize=12, weight="bold")
    ax.grid(axis="y", color="#D9DEE6", linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for idx, value in enumerate(df["mean_value"]):
        ax.text(idx, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(BAR_SVG, format="svg", transparent=False)
    plt.close(fig)


def main() -> None:
    OVERVIEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OVERVIEW_FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel_df = _build_query_panel_source()
    _draw_query_panel(panel_df)
    rank_df = _build_rank_source()
    _draw_rank_compare(rank_df)
    _draw_query_overall_bar()
    print("Wrote:")
    print(QUERY_PANEL_SOURCE)
    print(QUERY_PANEL_SVG)
    print(RANK_SOURCE)
    print(RANK_SVG)
    print(BAR_SOURCE)
    print(BAR_SVG)


if __name__ == "__main__":
    main()
