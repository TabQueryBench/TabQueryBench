#!/usr/bin/env python3
"""Temporary audit: compare current broad co-missingness vs strict pairwise co-missingness.

This script does not modify the official missingness breakdown outputs.
It writes a standalone review bundle under:

Evaluation/query_fivepart_breakdown/missingness_breakdown/review_strict_pairwise
"""

from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.common import normalize_missing, resolve_real_split_path
from tests.comissing_condition_eval import _clip01, _relation_strength, build_dataset_context


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Evaluation"
    / "query_fivepart_breakdown"
    / "missingness_breakdown"
    / "review_strict_pairwise"
)
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"
NOTES_DIR = OUTPUT_ROOT / "notes"

CURRENT_ASSET_CSV = (
    PROJECT_ROOT
    / "Evaluation"
    / "query_fivepart_breakdown"
    / "missingness_breakdown"
    / "data"
    / "direct_asset_scores.csv"
)

EXCLUDED_MODELS = {"cdtd", "codi", "goggle"}
MODEL_ALIASES = {"rtf": "realtabformer"}
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


def _ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, DATA_DIR, FIG_DIR, NOTES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _normalize_model(model_id: Any) -> str:
    key = str(model_id or "").strip().lower()
    return MODEL_ALIASES.get(key, key)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _binary_missing_indicator(series: pd.Series) -> np.ndarray:
    return series.map(normalize_missing).to_numpy(dtype=float)


def _load_syn_target_df(synthetic_csv_path: Path, target_columns: list[str]) -> pd.DataFrame:
    try:
        syn_df = pd.read_csv(
            synthetic_csv_path,
            dtype=str,
            keep_default_na=False,
            usecols=lambda name: str(name) in set(target_columns),
        )
    except ValueError:
        syn_df = pd.read_csv(synthetic_csv_path, dtype=str, keep_default_na=False)
        syn_df = syn_df[[col for col in target_columns if col in syn_df.columns]]
    for column in target_columns:
        if column not in syn_df.columns:
            syn_df[column] = ""
    return syn_df[target_columns]


def _state_codes_from_indicator(indicator: np.ndarray) -> np.ndarray:
    # 0 = non-missing, 1 = missing
    return indicator.astype(np.int16, copy=False)


def _conditional_rate_stats(missing_indicator: np.ndarray, related_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support_counts = np.bincount(related_codes, minlength=2)
    missing_sums = np.bincount(related_codes, weights=missing_indicator, minlength=2)
    rates = np.zeros(2, dtype=float)
    nonzero = support_counts > 0
    rates[nonzero] = missing_sums[nonzero] / support_counts[nonzero]
    return support_counts, rates


def _strict_pair_score(
    real_target: np.ndarray,
    real_related: np.ndarray,
    syn_target: np.ndarray,
    syn_related: np.ndarray,
) -> tuple[float, float, float]:
    real_global_missing_rate = float(np.mean(real_target))
    real_related_codes = _state_codes_from_indicator(real_related)
    syn_related_codes = _state_codes_from_indicator(syn_related)

    real_support_counts, real_conditional_rates = _conditional_rate_stats(real_target, real_related_codes)
    supported_state_indices = tuple(int(idx) for idx in np.where(real_support_counts > 0)[0].tolist())
    real_state_probabilities = real_support_counts.astype(float) / max(1, len(real_target))
    real_strength = _relation_strength(real_global_missing_rate, real_state_probabilities, real_conditional_rates)

    syn_support_counts, syn_conditional_rates = _conditional_rate_stats(syn_target, syn_related_codes)
    syn_rates_fallback = syn_conditional_rates.copy()
    zero_support = syn_support_counts <= 0
    syn_rates_fallback[zero_support] = float(np.mean(syn_target))

    profile_distance = 0.0
    for idx in supported_state_indices:
        profile_distance += float(real_state_probabilities[idx]) * abs(
            float(real_conditional_rates[idx]) - float(syn_rates_fallback[idx])
        )
    profile_score = _clip01(1.0 - profile_distance)

    denom = max(real_global_missing_rate * (1.0 - real_global_missing_rate), 1e-12)
    syn_weighted_var = 0.0
    for idx in supported_state_indices:
        syn_weighted_var += float(real_state_probabilities[idx]) * (
            (float(syn_rates_fallback[idx]) - real_global_missing_rate) ** 2
        )
    syn_strength = _clip01(syn_weighted_var / denom)
    strength_score = _clip01(1.0 - abs(real_strength - syn_strength))

    edge_score = _clip01((0.7 * profile_score) + (0.3 * strength_score))
    return edge_score, profile_score, strength_score


def _ordered_edge_score_from_counts(
    target_idx: int,
    related_idx: int,
    row_count: int,
    real_missing_counts: np.ndarray,
    syn_missing_counts: np.ndarray,
    real_joint_missing_counts: np.ndarray,
    syn_joint_missing_counts: np.ndarray,
) -> float:
    real_target_rate = float(real_missing_counts[target_idx] / row_count)
    syn_target_rate = float(syn_missing_counts[target_idx] / row_count)

    real_related_missing = float(real_missing_counts[related_idx])
    real_related_nonmissing = float(row_count - real_related_missing)
    real_state_probs = np.array(
        [real_related_nonmissing / row_count, real_related_missing / row_count],
        dtype=float,
    )

    real_joint = float(real_joint_missing_counts[target_idx, related_idx])
    real_cond_nonmissing = (float(real_missing_counts[target_idx]) - real_joint) / max(real_related_nonmissing, 1.0)
    real_cond_missing = real_joint / max(real_related_missing, 1.0)
    real_conditional_rates = np.array([real_cond_nonmissing, real_cond_missing], dtype=float)
    real_strength = _relation_strength(real_target_rate, real_state_probs, real_conditional_rates)

    syn_related_missing = float(syn_missing_counts[related_idx])
    syn_related_nonmissing = float(row_count - syn_related_missing)
    syn_joint = float(syn_joint_missing_counts[target_idx, related_idx])

    syn_cond_nonmissing = (float(syn_missing_counts[target_idx]) - syn_joint) / max(syn_related_nonmissing, 1.0)
    syn_cond_missing = syn_joint / max(syn_related_missing, 1.0)
    syn_conditional_rates = np.array([syn_cond_nonmissing, syn_cond_missing], dtype=float)
    if syn_related_nonmissing <= 0:
        syn_conditional_rates[0] = syn_target_rate
    if syn_related_missing <= 0:
        syn_conditional_rates[1] = syn_target_rate

    profile_distance = float(
        real_state_probs[0] * abs(real_conditional_rates[0] - syn_conditional_rates[0])
        + real_state_probs[1] * abs(real_conditional_rates[1] - syn_conditional_rates[1])
    )
    profile_score = _clip01(1.0 - profile_distance)

    denom = max(syn_target_rate * (1.0 - syn_target_rate), 1e-12)
    syn_weighted_var = float(
        real_state_probs[0] * ((syn_conditional_rates[0] - syn_target_rate) ** 2)
        + real_state_probs[1] * ((syn_conditional_rates[1] - syn_target_rate) ** 2)
    )
    syn_strength = _clip01(syn_weighted_var / denom)
    strength_score = _clip01(1.0 - abs(real_strength - syn_strength))
    return _clip01((0.7 * profile_score) + (0.3 * strength_score))


@lru_cache(maxsize=None)
def _get_dataset_context(dataset_id: str):
    return build_dataset_context(dataset_id)


@lru_cache(maxsize=None)
def _get_real_df(dataset_id: str) -> pd.DataFrame:
    return pd.read_csv(resolve_real_split_path(dataset_id, split="train"), dtype=str, keep_default_na=False)


def _strict_pairwise_score_for_asset(dataset_id: str, synthetic_csv_path: Path) -> dict[str, Any]:
    try:
        context = _get_dataset_context(dataset_id)
    except FileNotFoundError:
        return {
            "strict_status": "real_train_csv_missing_locally",
            "strict_pairwise_score": None,
            "strict_pair_count": 0,
            "active_missing_target_count": 0,
        }
    missing_targets = [target.column for target in context.missing_targets]

    if len(missing_targets) < 2:
        return {
            "strict_status": "not_applicable_fewer_than_2_missing_targets",
            "strict_pairwise_score": None,
            "strict_pair_count": 0,
            "active_missing_target_count": len(missing_targets),
        }

    real_df = _get_real_df(dataset_id)
    syn_df = _load_syn_target_df(synthetic_csv_path, missing_targets)

    row_count = len(real_df)
    real_matrix = np.column_stack([_binary_missing_indicator(real_df[col]) for col in missing_targets]).astype(np.float32)
    syn_matrix = np.column_stack([_binary_missing_indicator(syn_df[col]) for col in missing_targets]).astype(np.float32)
    real_missing_counts = real_matrix.sum(axis=0)
    syn_missing_counts = syn_matrix.sum(axis=0)
    real_joint_missing_counts = real_matrix.T @ real_matrix
    syn_joint_missing_counts = syn_matrix.T @ syn_matrix

    pair_scores: list[float] = []
    for left_idx in range(len(missing_targets)):
        for right_idx in range(left_idx + 1, len(missing_targets)):
            left_given_right = _ordered_edge_score_from_counts(
                left_idx,
                right_idx,
                row_count,
                real_missing_counts,
                syn_missing_counts,
                real_joint_missing_counts,
                syn_joint_missing_counts,
            )
            right_given_left = _ordered_edge_score_from_counts(
                right_idx,
                left_idx,
                row_count,
                real_missing_counts,
                syn_missing_counts,
                real_joint_missing_counts,
                syn_joint_missing_counts,
            )
            pair_scores.append(float(mean([left_given_right, right_given_left])))

    return {
        "strict_status": "ok",
        "strict_pairwise_score": round(float(mean(pair_scores)), 6),
        "strict_pair_count": len(pair_scores),
        "active_missing_target_count": len(missing_targets),
        "pair_rows": [],
    }


def _review_one_asset(row_dict: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = _strict_pairwise_score_for_asset(str(row_dict["dataset_id"]), Path(str(row_dict["synthetic_csv_path"])))
    payload = {
        "dataset_id": str(row_dict["dataset_id"]),
        "model_id": str(row_dict["model_id"]),
        "model_label": _model_label(str(row_dict["model_id"])),
        "current_broad_comissing_score": float(row_dict["co_missingness_pattern_consistency"]),
        "current_marginal_score": float(row_dict["marginal_missing_rate_consistency"]),
        "current_family_score": float(row_dict["missingness_structure_score"]),
        "strict_status": result["strict_status"],
        "strict_pairwise_comissing_score": result["strict_pairwise_score"],
        "strict_pair_count": int(result["strict_pair_count"]),
        "active_missing_target_count": int(result["active_missing_target_count"]),
    }
    if result["strict_pairwise_score"] is not None:
        payload["delta_strict_minus_current"] = round(
            float(result["strict_pairwise_score"]) - float(row_dict["co_missingness_pattern_consistency"]), 6
        )
    else:
        payload["delta_strict_minus_current"] = None

    pair_rows = [
        {
            "dataset_id": str(row_dict["dataset_id"]),
            "model_id": str(row_dict["model_id"]),
            "model_label": _model_label(str(row_dict["model_id"])),
            **pair_row,
        }
        for pair_row in result.get("pair_rows", [])
    ]
    return payload, pair_rows


def _build_review_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    asset_df = pd.read_csv(CURRENT_ASSET_CSV, encoding="utf-8-sig")
    asset_df["model_id"] = asset_df["model_id"].map(_normalize_model)
    asset_df = asset_df.loc[
        (asset_df["status"] == "ok")
        & (~asset_df["model_id"].isin(EXCLUDED_MODELS))
        & asset_df["model_id"].isin(MODEL_ORDER)
    ].copy()

    review_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    row_dicts = asset_df.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_review_one_asset, row_dict) for row_dict in row_dicts]
        for future in as_completed(futures):
            payload, pair_payloads = future.result()
            review_rows.append(payload)
            pair_rows.extend(pair_payloads)

    review_df = pd.DataFrame(review_rows).sort_values(["dataset_id", "model_label"]).reset_index(drop=True)
    pair_df = pd.DataFrame(pair_rows)

    overlap_df = review_df.loc[review_df["strict_pairwise_comissing_score"].notna()].copy()
    grouped = []
    for model_id, group in overlap_df.groupby("model_id", sort=False):
        grouped.append(
            {
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "dataset_count_overlap": int(group["dataset_id"].nunique()),
                "panel_count_overlap": int(group.shape[0]),
                "current_broad_comissing_score__mean": round(float(group["current_broad_comissing_score"].mean()), 6),
                "strict_pairwise_comissing_score__mean": round(float(group["strict_pairwise_comissing_score"].mean()), 6),
                "delta_strict_minus_current__mean": round(float(group["delta_strict_minus_current"].mean()), 6),
            }
        )
    model_df = pd.DataFrame(grouped)
    if not model_df.empty:
        model_df["model_order"] = model_df["model_id"].map({m: i for i, m in enumerate(MODEL_ORDER)})
        model_df = model_df.sort_values("model_order").drop(columns=["model_order"]).reset_index(drop=True)

    coverage_rows = []
    for dataset_id, group in review_df.groupby("dataset_id", sort=False):
        coverage_rows.append(
            {
                "dataset_id": dataset_id,
                "model_panel_count": int(group.shape[0]),
                "strict_applicable_panel_count": int(group["strict_pairwise_comissing_score"].notna().sum()),
                "active_missing_target_count": int(group["active_missing_target_count"].max()),
                "strict_pair_count": int(pd.to_numeric(group["strict_pair_count"], errors="coerce").fillna(0).max()),
            }
        )
    coverage_df = pd.DataFrame(coverage_rows).sort_values("dataset_id").reset_index(drop=True)
    return review_df, pair_df, model_df, coverage_df


def _plot_review_figure(review_df: pd.DataFrame, model_df: pd.DataFrame, out_path: Path) -> None:
    overlap_df = review_df.loc[review_df["strict_pairwise_comissing_score"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), constrained_layout=True)
    ax0, ax1 = axes

    if not model_df.empty:
        y_positions = np.arange(len(model_df))
        for idx, row in enumerate(model_df.itertuples()):
            current = float(row.current_broad_comissing_score__mean)
            strict = float(row.strict_pairwise_comissing_score__mean)
            color = MODEL_COLORS.get(str(row.model_id), "#777777")
            ax0.plot([current, strict], [idx, idx], color=color, linewidth=2.0, alpha=0.9)
            ax0.scatter(current, idx, s=60, color="white", edgecolor=color, linewidth=1.8, zorder=3)
            ax0.scatter(strict, idx, s=60, color=color, edgecolor=color, linewidth=1.2, zorder=4)
        ax0.set_yticks(y_positions)
        ax0.set_yticklabels(list(model_df["model_label"]))
        ax0.set_xlim(0.0, 1.02)
        ax0.set_xlabel("Mean co-missingness score on overlap panels")
        ax0.set_title("Model-level comparison\nhollow = current broad, solid = strict pairwise")
        ax0.grid(axis="x", alpha=0.25, linewidth=0.8)
    else:
        ax0.text(0.5, 0.5, "No overlap panels available", ha="center", va="center", transform=ax0.transAxes)
        ax0.set_axis_off()

    if not overlap_df.empty:
        ax1.scatter(
            overlap_df["current_broad_comissing_score"],
            overlap_df["strict_pairwise_comissing_score"],
            s=34,
            color="#4C78A8",
            alpha=0.75,
            edgecolors="none",
        )
        ax1.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.2)
        ax1.set_xlim(0.0, 1.02)
        ax1.set_ylim(0.0, 1.02)
        ax1.set_xlabel("Current broad co-missingness score")
        ax1.set_ylabel("Strict pairwise co-missingness score")
        ax1.set_title(
            "Dataset-model panels\n"
            f"overlap n={overlap_df.shape[0]}, datasets={overlap_df['dataset_id'].nunique()}"
        )
        ax1.grid(alpha=0.25, linewidth=0.8)
    else:
        ax1.text(0.5, 0.5, "No strict-pairwise-applicable panels", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()

    fig.suptitle(
        "Review audit: broad structured missingness vs strict pairwise co-missingness",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_review() -> dict[str, Any]:
    _ensure_dirs()
    review_df, pair_df, model_df, coverage_df = _build_review_outputs()

    _write_csv(review_df, DATA_DIR / "current_vs_strict_pairwise_asset_review.csv")
    _write_csv(pair_df, DATA_DIR / "strict_pairwise_pair_scores.csv")
    _write_csv(model_df, DATA_DIR / "current_vs_strict_pairwise_model_summary.csv")
    _write_csv(coverage_df, DATA_DIR / "strict_pairwise_dataset_coverage.csv")

    figure_path = FIG_DIR / "current_vs_strict_pairwise_review.png"
    _plot_review_figure(review_df, model_df, figure_path)

    overlap_df = review_df.loc[review_df["strict_pairwise_comissing_score"].notna()].copy()
    note_lines = [
        "# Strict Pairwise Review",
        "",
        "This is a temporary audit only. It does not change the official missingness bundle.",
        "",
        "## Audit definition",
        "",
        "- Current broad score: official `co_missingness_pattern_consistency` from the direct evaluator.",
        "- Strict pairwise score: only use unordered pairs of active missing-target columns.",
        "- For each pair `(A, B)`, score `A | B_missing_indicator` and `B | A_missing_indicator` with the same 0.7 profile + 0.3 strength formula, then average the two directions.",
        "- Final strict pairwise score = mean over all unordered missing-target pairs.",
        "",
        "## Coverage",
        "",
        f"- Asset/panel rows reviewed: `{review_df.shape[0]}`",
        f"- Overlap rows with strict pairwise defined: `{overlap_df.shape[0]}`",
        f"- Datasets with strict pairwise defined: `{overlap_df['dataset_id'].nunique() if not overlap_df.empty else 0}`",
        f"- Models with strict pairwise defined: `{overlap_df['model_id'].nunique() if not overlap_df.empty else 0}`",
        "",
        "## Main caveat",
        "",
        "- Strict pairwise is undefined when a dataset has fewer than 2 active missing-target columns.",
        "- So this review is a support-reduced audit, not a drop-in replacement for the official broad score.",
    ]
    (NOTES_DIR / "review.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    return {
        "review_csv": DATA_DIR / "current_vs_strict_pairwise_asset_review.csv",
        "pair_csv": DATA_DIR / "strict_pairwise_pair_scores.csv",
        "model_csv": DATA_DIR / "current_vs_strict_pairwise_model_summary.csv",
        "coverage_csv": DATA_DIR / "strict_pairwise_dataset_coverage.csv",
        "figure_png": figure_path,
        "figure_pdf": figure_path.with_suffix(".pdf"),
        "note_md": NOTES_DIR / "review.md",
        "panel_count": int(review_df.shape[0]),
        "overlap_panel_count": int(overlap_df.shape[0]),
    }


if __name__ == "__main__":
    outputs = run_review()
    for key, value in outputs.items():
        print(f"{key}: {value}")
