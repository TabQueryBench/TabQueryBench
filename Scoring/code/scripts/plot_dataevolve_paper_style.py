from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = (
    ROOT
    / "Evaluation"
    / "analysis"
    / "runs"
    / "dataevolve_v2_trainonly_query_replay_20260707"
    / "comparison_vs_11_models"
    / "paper_style_with_dataevolve"
)

DE_V2 = (
    ROOT
    / "Evaluation"
    / "analysis"
    / "runs"
    / "dataevolve_v2_trainonly_query_replay_20260707"
    / "summaries"
)
DE_V5 = ROOT / "Evaluation" / "analysis" / "runs" / "dataevolve_v5_full_20260707" / "summaries"

Q5 = ROOT / "Evaluation" / "query_fivepart_breakdown"
SUBGROUP = Q5 / "subgroup_breakdown" / "final"
CONDITIONAL = Q5 / "conditional_breakdown" / "final"
TAIL = Q5 / "tail_breakdown" / "final"
MISSING = Q5 / "missingness_breakdown" / "final"
CARD = Q5 / "cardinality" / "final"
COMPARISON = (
    ROOT
    / "Evaluation"
    / "analysis"
    / "runs"
    / "dataevolve_v2_trainonly_query_replay_20260707"
    / "comparison_vs_11_models"
)

PAPER_MODEL_ORDER = [
    "realtabformer",
    "tvae",
    "forestdiffusion",
    "tabddpm",
    "tabsyn",
    "tabdiff",
    "ctgan",
    "arf",
    "bayesnet",
    "tabpfgen",
    "tabbyflow",
]

LABELS = {
    "arf": "ARF",
    "bayesnet": "Bayes",
    "ctgan": "CTGAN",
    "forestdiffusion": "F-Diff",
    "realtabformer": "RTF",
    "tabbyflow": "T-Flow",
    "tabddpm": "T-DDPM",
    "tabdiff": "T-Diff",
    "tabpfgen": "TPF",
    "tabsyn": "T-Syn",
    "tvae": "TVAE",
    "dataevolve": "DataEvolve",
}

PREFIX_LABELS = {"c": "Categorical", "m": "Mixed", "n": "Numerical"}
FAMILY_LABELS = {
    "subgroup_structure": "Subgroup",
    "conditional_dependency_structure": "Conditional",
    "tail_rarity_structure": "Tail",
    "missingness_structure": "Missing",
    "cardinality_structure": "Cardinality",
}

COLORS = {
    "blue": "#4C78A8",
    "green": "#2A9D8F",
    "yellow": "#E6C229",
    "red": "#D1495B",
    "purple": "#6D597A",
    "gray": "#B9B9B9",
    "dark": "#2F2F2F",
    "light": "#F5F5F2",
}


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _de_query(version: str) -> pd.DataFrame:
    base = DE_V2 if version == "v2" else DE_V5
    return pd.read_json(base / "analysis_query_scores__all_datasets.jsonl", lines=True)


def _de_family(version: str) -> pd.DataFrame:
    base = DE_V2 if version == "v2" else DE_V5
    return _read_csv(base / "analysis_family_mean_scores__all_datasets.csv")


def _de_subitem(version: str) -> pd.DataFrame:
    base = DE_V2 if version == "v2" else DE_V5
    return _read_csv(base / "analysis_subitem_scores__all_datasets.csv")


def _mean_de_subitem(version: str, family_id: str, subitem_id: str) -> float:
    df = _de_subitem(version)
    mask = (df["family_id"] == family_id) & (df["subitem_id"] == subitem_id)
    return float(df.loc[mask, "subitem_score"].dropna().mean())


def _mean_de_family(version: str, family_id: str) -> float:
    df = _de_family(version)
    mask = df["family_id"] == family_id
    return float(df.loc[mask, "family_score"].dropna().mean())


def _mean_de_query_family(version: str, family_id: str) -> float:
    df = _de_query(version)
    mask = df["family_id"] == family_id
    return float(df.loc[mask, "query_score"].dropna().mean())


def _mean_de_query_by_prefix(version: str, family_id: str) -> pd.Series:
    df = _de_query(version)
    subset = df[df["family_id"] == family_id].copy()
    subset["prefix"] = subset["dataset_id"].str[0]
    return subset.groupby("prefix")["query_score"].mean()


def _box_with_points(ax, values, positions, width=0.46, color="#B9B9B9") -> None:
    bp = ax.boxplot(
        values,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["dark"], "linewidth": 1.2},
        boxprops={"linewidth": 0.8, "color": COLORS["dark"]},
        whiskerprops={"linewidth": 0.8, "color": COLORS["dark"]},
        capprops={"linewidth": 0.8, "color": COLORS["dark"]},
    )
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.52)


def _finish_axis(ax, title: str, ylim=(0, 1.02)) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_compact_overview() -> None:
    subgroup_prefix = _read_csv(SUBGROUP / "prefix_summary__v2.csv")
    conditional_prefix = _read_csv(CONDITIONAL / "prefix_summary__v2.csv")
    card_prefix = _read_csv(CARD / "prefix_plot_data__v2.csv")

    de_v2_subgroup = _mean_de_query_by_prefix("v2", "subgroup_structure")
    de_v2_cond = _mean_de_query_by_prefix("v2", "conditional_dependency_structure")
    de_v2_card = _mean_de_query_by_prefix("v2", "cardinality_structure")
    de_v5_subgroup = _mean_de_query_by_prefix("v5", "subgroup_structure")
    de_v5_cond = _mean_de_query_by_prefix("v5", "conditional_dependency_structure")
    de_v5_card = _mean_de_query_by_prefix("v5", "cardinality_structure")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    fig.suptitle(
        "Paper-style query-fidelity diagnostics with DataEvolve overlay",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    # A. Subgroup internal-vs-size by C/M/N. DataEvolve has internal only.
    ax = axes[0, 0]
    positions, labels, boxes, colors = [], [], [], []
    x = 1
    for prefix in ["c", "m", "n"]:
        frame = subgroup_prefix[subgroup_prefix["dataset_prefix"] == prefix]
        positions.extend([x, x + 0.55])
        labels.extend([f"{prefix.upper()}\nInternal", f"{prefix.upper()}\nSize"])
        boxes.extend(
            [
                frame["internal_profile_stability"].dropna().to_numpy(),
                frame["subgroup_size_stability"].dropna().to_numpy(),
            ]
        )
        colors.extend([COLORS["blue"], COLORS["green"]])
        if prefix in de_v2_subgroup.index:
            ax.scatter(
                [x],
                [de_v2_subgroup[prefix]],
                s=72,
                marker="D",
                color=COLORS["red"],
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
                label="DataEvolve V2" if prefix == "c" else None,
            )
        if prefix in de_v5_subgroup.index:
            ax.scatter(
                [x + 0.16],
                [de_v5_subgroup[prefix]],
                s=72,
                marker="D",
                facecolor="white",
                edgecolor=COLORS["red"],
                linewidth=1.6,
                zorder=5,
                label="DataEvolve V5" if prefix == "c" else None,
            )
        x += 1.65
    bp = ax.boxplot(boxes, positions=positions, widths=0.38, patch_artist=True, showfliers=False)
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.45)
        box.set_edgecolor(COLORS["dark"])
    for part in ["whiskers", "caps", "medians"]:
        for artist in bp[part]:
            artist.set_color(COLORS["dark"])
            artist.set_linewidth(0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.text(
        0.01,
        0.03,
        "DataEvolve replay exposes internal-profile subgroup queries only.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["dark"],
    )
    _finish_axis(ax, "(a) Subgroup: paper distribution plus DataEvolve internal overlay")
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    # B. Conditional subitems.
    ax = axes[0, 1]
    cond_model = _read_csv(CONDITIONAL / "model_summary__v2.csv")
    cond_items = [
        ("dependency_strength_similarity__mean", "Strength"),
        ("direction_consistency__mean", "Direction"),
        ("slice_level_consistency__mean", "Slice"),
    ]
    values = [cond_model[col].dropna().to_numpy() for col, _ in cond_items]
    _box_with_points(ax, values, [1, 2, 3], color=COLORS["blue"])
    de_v2 = [
        _mean_de_subitem("v2", "conditional_dependency_structure", "dependency_strength_similarity"),
        _mean_de_subitem("v2", "conditional_dependency_structure", "direction_consistency"),
        _mean_de_subitem("v2", "conditional_dependency_structure", "slice_level_consistency"),
    ]
    de_v5 = [
        _mean_de_subitem("v5", "conditional_dependency_structure", "dependency_strength_similarity"),
        _mean_de_subitem("v5", "conditional_dependency_structure", "direction_consistency"),
        _mean_de_subitem("v5", "conditional_dependency_structure", "slice_level_consistency"),
    ]
    ax.plot([1, 2, 3], de_v2, color=COLORS["red"], marker="D", linewidth=1.8, label="DataEvolve V2")
    ax.plot(
        [1.08, 2.08, 3.08],
        de_v5,
        color=COLORS["red"],
        marker="D",
        markerfacecolor="white",
        linewidth=1.4,
        linestyle="--",
        label="DataEvolve V5",
    )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([label for _, label in cond_items])
    _finish_axis(ax, "(b) Conditional: direction is the easiest DataEvolve branch")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # C. Cardinality by C/M/N.
    ax = axes[1, 0]
    values = [card_prefix[p].dropna().to_numpy() for p in ["c", "m", "n"]]
    _box_with_points(ax, values, [1, 2, 3], color=COLORS["purple"])
    ax.plot(
        [1, 2, 3],
        [de_v2_card.get(p, np.nan) for p in ["c", "m", "n"]],
        color=COLORS["red"],
        marker="D",
        linewidth=1.8,
        label="DataEvolve V2",
    )
    ax.plot(
        [1.08, 2.08, 3.08],
        [de_v5_card.get(p, np.nan) for p in ["c", "m", "n"]],
        color=COLORS["red"],
        marker="D",
        markerfacecolor="white",
        linestyle="--",
        linewidth=1.4,
        label="DataEvolve V5",
    )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Categorical", "Mixed", "Numerical"])
    _finish_axis(ax, "(c) Cardinality: DataEvolve sits below most paper-model medians")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    # D. Family-level profile.
    ax = axes[1, 1]
    subgroup_model = _read_csv(SUBGROUP / "model_summary__v2.csv")
    cond_model = _read_csv(CONDITIONAL / "model_summary__v2.csv")
    tail_model = _read_csv(TAIL / "model_summary.csv")
    miss_model = _read_csv(MISSING / "model_summary__v2.csv")
    card_model = _read_csv(CARD / "summary_by_model__v2.csv")
    family_boxes = [
        subgroup_model["subgroup_structure_score__mean"].dropna().to_numpy(),
        cond_model["conditional_dependency_structure_score__mean"].dropna().to_numpy(),
        tail_model["tail_breakdown_score__mean"].dropna().to_numpy(),
        miss_model["missingness_structure_score__mean"].dropna().to_numpy(),
        card_model.loc[card_model["model"].isin(PAPER_MODEL_ORDER), "overall_score_mean"].dropna().to_numpy(),
    ]
    _box_with_points(ax, family_boxes, [1, 2, 3, 4, 5], color=COLORS["gray"])
    de_family_v2 = [
        _mean_de_family("v2", "subgroup_structure"),
        _mean_de_family("v2", "conditional_dependency_structure"),
        _mean_de_family("v2", "tail_rarity_structure"),
        _mean_de_family("v2", "missingness_structure"),
        _mean_de_query_family("v2", "cardinality_structure"),
    ]
    de_family_v5 = [
        _mean_de_family("v5", "subgroup_structure"),
        _mean_de_family("v5", "conditional_dependency_structure"),
        _mean_de_family("v5", "tail_rarity_structure"),
        _mean_de_family("v5", "missingness_structure"),
        _mean_de_query_family("v5", "cardinality_structure"),
    ]
    ax.plot([1, 2, 3, 4, 5], de_family_v2, color=COLORS["red"], marker="D", linewidth=1.8, label="DataEvolve V2")
    ax.plot(
        [1.08, 2.08, 3.08, 4.08, 5.08],
        de_family_v5,
        color=COLORS["red"],
        marker="D",
        markerfacecolor="white",
        linestyle="--",
        linewidth=1.4,
        label="DataEvolve V5",
    )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Subgroup", "Conditional", "Tail", "Missing", "Cardinality"], rotation=15, ha="right")
    _finish_axis(ax, "(d) Family profile: DataEvolve follows the same weak axes")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save(fig, "paper_compact_overview_with_dataevolve.png")


def build_tail_missing() -> None:
    tail_model = _read_csv(TAIL / "model_summary.csv")
    miss_prefix = _read_csv(MISSING / "prefix_summary__v2.csv")
    miss_model = _read_csv(MISSING / "model_summary__v2.csv")
    de_v2_q = _de_query("v2")
    de_v5_q = _de_query("v5")

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.4))
    fig.suptitle("Tail and missingness diagnostics with DataEvolve", fontsize=15, fontweight="bold", y=0.99)

    # A. Formal tail submetrics by model, with DataEvolve appended.
    ax = axes[0, 0]
    order = [m for m in PAPER_MODEL_ORDER if m in set(tail_model["model_id"])]
    rows = tail_model.set_index("model_id").loc[order]
    x = np.arange(len(order) + 1)
    width = 0.22
    metrics = [
        ("tail_set_consistency__mean", "Tail set", COLORS["red"]),
        ("tail_mass_similarity__mean", "Tail mass", COLORS["green"]),
        ("tail_concentration_consistency__mean", "Tail conc.", COLORS["purple"]),
    ]
    for idx, (col, label, color) in enumerate(metrics):
        vals = rows[col].to_numpy(dtype=float).tolist()
        if label == "Tail mass":
            vals.append(np.nan)
        elif label == "Tail set":
            vals.append(_mean_de_subitem("v2", "tail_rarity_structure", "tail_set_consistency"))
        else:
            vals.append(_mean_de_subitem("v2", "tail_rarity_structure", "tail_concentration_consistency"))
        ax.bar(x + (idx - 1) * width, vals, width=width, color=color, alpha=0.78, label=label)
    ax.scatter(
        [len(order)],
        [_mean_de_subitem("v5", "tail_rarity_structure", "tail_set_consistency")],
        marker="D",
        s=70,
        facecolor="white",
        edgecolor=COLORS["red"],
        linewidth=1.5,
        zorder=5,
    )
    ax.scatter(
        [len(order) + width],
        [_mean_de_subitem("v5", "tail_rarity_structure", "tail_concentration_consistency")],
        marker="D",
        s=70,
        facecolor="white",
        edgecolor=COLORS["purple"],
        linewidth=1.5,
        zorder=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in order] + ["DataEvolve"], rotation=90)
    ax.text(
        0.97,
        0.05,
        "DataEvolve tail-mass branch not exported in replay.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
    )
    _finish_axis(ax, "(a) Formal tail submetrics")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")

    # B. Missingness by regime, paper panel-d style plus DataEvolve.
    ax = axes[0, 1]
    order = [m for m in PAPER_MODEL_ORDER if m in set(miss_prefix["model_id"])]
    xpos = np.arange(len(order) + 1)
    regime_colors = {"c": COLORS["yellow"], "m": "#3FA7D6", "n": COLORS["red"]}
    for idx, prefix in enumerate(["c", "m", "n"]):
        vals = []
        for model in order:
            frame = miss_prefix[(miss_prefix["model_id"] == model) & (miss_prefix["dataset_prefix"] == prefix)]
            vals.append(float(frame["missingness_structure_score"].iloc[0]) if len(frame) else np.nan)
        de_frame = de_v2_q[(de_v2_q["family_id"] == "missingness_structure")].copy()
        de_frame["prefix"] = de_frame["dataset_id"].str[0]
        de_val = de_frame.loc[de_frame["prefix"] == prefix, "query_score"].mean()
        vals.append(de_val)
        ax.bar(xpos + (idx - 1) * 0.22, vals, width=0.22, color=regime_colors[prefix], label=PREFIX_LABELS[prefix])
    ax.set_xticks(xpos)
    ax.set_xticklabels([LABELS[m] for m in order] + ["DataEvolve"], rotation=90)
    _finish_axis(ax, "(b) Missingness by regime")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower left")

    # C. Marginal-vs-co-missing split.
    ax = axes[1, 0]
    rows = miss_model[miss_model["model_id"].isin(PAPER_MODEL_ORDER)].set_index("model_id").loc[order]
    x = np.arange(len(order) + 1)
    marginal = rows["marginal_missing_rate_consistency__mean"].to_numpy(dtype=float).tolist()
    comissing = rows["co_missingness_pattern_consistency__mean"].to_numpy(dtype=float).tolist()
    marginal.append(_mean_de_subitem("v2", "missingness_structure", "marginal_missing_rate_consistency"))
    comissing.append(_mean_de_subitem("v2", "missingness_structure", "co_missingness_pattern_consistency"))
    ax.bar(x - 0.15, marginal, width=0.3, color=COLORS["blue"], alpha=0.78, label="Marginal rate")
    ax.bar(x + 0.15, comissing, width=0.3, color=COLORS["red"], alpha=0.78, label="Co-missing profile")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in order] + ["DataEvolve"], rotation=90)
    _finish_axis(ax, "(c) Missingness split: DataEvolve preserves rates but loses profile")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    # D. DataEvolve V2/V5 family profile, emphasizing the paper findings.
    ax = axes[1, 1]
    families = [
        "subgroup_structure",
        "conditional_dependency_structure",
        "tail_rarity_structure",
        "missingness_structure",
        "cardinality_structure",
    ]
    v2 = [
        _mean_de_family("v2", f) if f != "cardinality_structure" else _mean_de_query_family("v2", f)
        for f in families
    ]
    v5 = [
        _mean_de_family("v5", f) if f != "cardinality_structure" else _mean_de_query_family("v5", f)
        for f in families
    ]
    pos = np.arange(len(families))
    ax.bar(pos - 0.18, v2, width=0.36, color=COLORS["red"], alpha=0.78, label="DataEvolve V2")
    ax.bar(pos + 0.18, v5, width=0.36, color=COLORS["dark"], alpha=0.72, label="DataEvolve V5")
    ax.set_xticks(pos)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in families], rotation=15, ha="right")
    _finish_axis(ax, "(d) DataEvolve-only family profile")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save(fig, "paper_tail_missing_with_dataevolve.png")


def build_ranking() -> None:
    all_available = _read_csv(COMPARISON / "v2_model_comparison_all_available.csv")
    common = _read_csv(COMPARISON / "v2_model_comparison_common_8_datasets.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    for ax, df, title in [
        (axes[0], all_available, "All available datasets"),
        (axes[1], common, "Common 8 datasets"),
    ]:
        df = df.sort_values("mean_overall", ascending=True)
        colors = [COLORS["red"] if m == "dataevolve" else COLORS["blue"] for m in df["model_id"]]
        y = np.arange(len(df))
        ax.barh(y, df["mean_overall"], color=colors, alpha=0.82)
        ax.set_yticks(y)
        ax.set_yticklabels(df["model_label"])
        for yy, val in zip(y, df["mean_overall"]):
            ax.text(val + 0.01, yy, f"{val:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 0.9)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.7, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("V2 query fidelity ranking after adding DataEvolve", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "dataevolve_v2_12model_ranking.png")


def build_summary_table() -> None:
    rows = []
    for version in ["v2", "v5"]:
        rows.append(
            {
                "version": version,
                "subgroup": _mean_de_family(version, "subgroup_structure"),
                "conditional": _mean_de_family(version, "conditional_dependency_structure"),
                "tail": _mean_de_family(version, "tail_rarity_structure"),
                "missingness": _mean_de_family(version, "missingness_structure"),
                "cardinality_query": _mean_de_query_family(version, "cardinality_structure"),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "dataevolve_family_profile_v2_v5.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_compact_overview()
    build_tail_missing()
    build_ranking()
    build_summary_table()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
