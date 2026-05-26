#!/usr/bin/env python3
"""Experimental missingness diagnostic using missing-target pairs and centered profiles."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import os
from os import cpu_count
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.common import normalize_missing, write_csv
from tests.comissing_condition_eval import _clip01, _load_real_df, build_dataset_context

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Evaluation"
    / "query_fivepart_breakdown"
    / "missingness_breakdown"
    / "pairwise_centered_diagnostic"
)
DATA_DIR = OUTPUT_ROOT / "data"
FINAL_DIR = OUTPUT_ROOT / "final"
CURRENT_ASSET_CSV = (
    PROJECT_ROOT
    / "Evaluation"
    / "query_fivepart_breakdown"
    / "missingness_breakdown"
    / "data"
    / "direct_asset_scores.csv"
)
EMIT_PAIR_ROWS = os.environ.get("PAIRWISE_CENTERED_EMIT_PAIR_ROWS", "").strip().lower() in {"1", "true", "yes"}
MODEL_ALIASES = {"rtf": "realtabformer"}
PREFERRED_MODEL_ORDER = [
    "arf",
    "bayesnet",
    "cdtd",
    "codi",
    "ctgan",
    "forestdiffusion",
    "goggle",
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
    "cdtd": "CDTD",
    "codi": "CoDi",
    "ctgan": "CTGAN",
    "forestdiffusion": "ForestDiffusion",
    "goggle": "GOGGLE",
    "realtabformer": "RealTabFormer",
    "tabbyflow": "TabbyFlow",
    "tabddpm": "TabDDPM",
    "tabdiff": "TabDiff",
    "tabpfgen": "TabPFGen",
    "tabsyn": "TabSyn",
    "tvae": "TVAE",
}


def _ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, DATA_DIR, FINAL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _normalize_model(model_id: Any) -> str:
    key = str(model_id or "").strip().lower()
    return MODEL_ALIASES.get(key, key)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _model_sort_key(model_id: str) -> tuple[int, str]:
    if model_id in PREFERRED_MODEL_ORDER:
        return (PREFERRED_MODEL_ORDER.index(model_id), model_id)
    return (len(PREFERRED_MODEL_ORDER), model_id)


def _dataset_prefix(dataset_id: str) -> str:
    text = str(dataset_id or "").strip().lower()
    if not text:
        return "?"
    return text[0]


def _binary_missing_indicator(series: pd.Series) -> np.ndarray:
    return series.map(normalize_missing).to_numpy(dtype=float)


def _ordered_centered_profile_score_from_counts(
    *,
    target_idx: int,
    related_idx: int,
    real_row_count: int,
    syn_row_count: int,
    real_missing_counts: np.ndarray,
    syn_missing_counts: np.ndarray,
    real_joint_missing_counts: np.ndarray,
    syn_joint_missing_counts: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    real_target_rate = float(real_missing_counts[target_idx] / max(real_row_count, 1))
    syn_target_rate = float(syn_missing_counts[target_idx] / max(syn_row_count, 1))

    real_related_missing = float(real_missing_counts[related_idx])
    real_related_nonmissing = float(real_row_count - real_related_missing)
    real_state_probs = np.array(
        [real_related_nonmissing / max(real_row_count, 1), real_related_missing / max(real_row_count, 1)],
        dtype=float,
    )

    real_joint = float(real_joint_missing_counts[target_idx, related_idx])
    real_cond_nonmissing = (float(real_missing_counts[target_idx]) - real_joint) / max(real_related_nonmissing, 1.0)
    real_cond_missing = real_joint / max(real_related_missing, 1.0)
    real_conditional_rates = np.array([real_cond_nonmissing, real_cond_missing], dtype=float)

    syn_related_missing = float(syn_missing_counts[related_idx])
    syn_related_nonmissing = float(syn_row_count - syn_related_missing)
    syn_joint = float(syn_joint_missing_counts[target_idx, related_idx])
    syn_cond_nonmissing = (float(syn_missing_counts[target_idx]) - syn_joint) / max(syn_related_nonmissing, 1.0)
    syn_cond_missing = syn_joint / max(syn_related_missing, 1.0)
    syn_conditional_rates = np.array([syn_cond_nonmissing, syn_cond_missing], dtype=float)
    if syn_related_nonmissing <= 0:
        syn_conditional_rates[0] = syn_target_rate
    if syn_related_missing <= 0:
        syn_conditional_rates[1] = syn_target_rate

    delta_real = real_conditional_rates - real_target_rate
    delta_syn = syn_conditional_rates - syn_target_rate
    centered_distance = float(
        real_state_probs[0] * abs(float(delta_real[0]) - float(delta_syn[0]))
        + real_state_probs[1] * abs(float(delta_real[1]) - float(delta_syn[1]))
    )
    centered_profile_score = _clip01(1.0 - (0.5 * centered_distance))

    pair_row = {
        "target_missing_index": int(target_idx),
        "related_missing_index": int(related_idx),
        "real_target_missing_rate": round(real_target_rate, 6),
        "synthetic_target_missing_rate": round(syn_target_rate, 6),
        "real_conditional_nonmissing": round(float(real_conditional_rates[0]), 6),
        "real_conditional_missing": round(float(real_conditional_rates[1]), 6),
        "synthetic_conditional_nonmissing": round(float(syn_conditional_rates[0]), 6),
        "synthetic_conditional_missing": round(float(syn_conditional_rates[1]), 6),
        "real_delta_nonmissing": round(float(delta_real[0]), 6),
        "real_delta_missing": round(float(delta_real[1]), 6),
        "synthetic_delta_nonmissing": round(float(delta_syn[0]), 6),
        "synthetic_delta_missing": round(float(delta_syn[1]), 6),
        "centered_profile_distance": round(centered_distance, 6),
        "pairwise_centered_ordered_score": round(float(centered_profile_score), 6),
    }
    return centered_profile_score, pair_row


@lru_cache(maxsize=None)
def _get_dataset_context(dataset_id: str):
    return build_dataset_context(dataset_id)


@lru_cache(maxsize=None)
def _get_real_target_df(dataset_id: str, missing_targets_key: tuple[str, ...]) -> pd.DataFrame:
    real_df = _load_real_df(dataset_id)
    return real_df[list(missing_targets_key)].copy()


def _load_syn_target_df(synthetic_csv_path: Path, target_columns: list[str]) -> pd.DataFrame:
    target_set = set(target_columns)
    try:
        syn_df = pd.read_csv(
            synthetic_csv_path,
            dtype=str,
            keep_default_na=False,
            usecols=lambda name: str(name) in target_set,
        )
    except ValueError:
        syn_df = pd.read_csv(synthetic_csv_path, dtype=str, keep_default_na=False)
        syn_df = syn_df[[column for column in target_columns if column in syn_df.columns]]
    for column in target_columns:
        if column not in syn_df.columns:
            syn_df[column] = ""
    return syn_df[target_columns].copy()


def _pairwise_centered_score_for_asset(dataset_id: str, synthetic_csv_path: Path) -> dict[str, Any]:
    context = _get_dataset_context(dataset_id)
    missing_targets = [target.column for target in context.missing_targets]
    if len(missing_targets) < 2:
        return {
            "pairwise_centered_status": "not_applicable_fewer_than_2_missing_targets",
            "pairwise_centered_comissing_score": None,
            "pairwise_centered_pair_count": 0,
            "pairwise_centered_ordered_edge_count": 0,
            "active_missing_target_count": len(missing_targets),
            "pair_rows": [],
        }

    real_df = _get_real_target_df(dataset_id, tuple(missing_targets))
    syn_df = _load_syn_target_df(synthetic_csv_path, missing_targets)

    real_row_count = len(real_df)
    syn_row_count = len(syn_df)
    real_matrix = np.column_stack([_binary_missing_indicator(real_df[col]) for col in missing_targets]).astype(np.float32)
    syn_matrix = np.column_stack([_binary_missing_indicator(syn_df[col]) for col in missing_targets]).astype(np.float32)
    real_missing_counts = real_matrix.sum(axis=0)
    syn_missing_counts = syn_matrix.sum(axis=0)
    real_joint_missing_counts = real_matrix.T @ real_matrix
    syn_joint_missing_counts = syn_matrix.T @ syn_matrix

    target_count = len(missing_targets)
    real_target_rates = real_missing_counts / max(real_row_count, 1)
    syn_target_rates = syn_missing_counts / max(syn_row_count, 1)

    real_related_missing = real_missing_counts[np.newaxis, :]
    real_related_nonmissing = (real_row_count - real_missing_counts)[np.newaxis, :]
    syn_related_missing = syn_missing_counts[np.newaxis, :]
    syn_related_nonmissing = (syn_row_count - syn_missing_counts)[np.newaxis, :]

    real_cond_missing = np.divide(
        real_joint_missing_counts,
        np.maximum(real_related_missing, 1.0),
        out=np.zeros_like(real_joint_missing_counts, dtype=float),
    )
    real_cond_nonmissing = np.divide(
        real_missing_counts[:, np.newaxis] - real_joint_missing_counts,
        np.maximum(real_related_nonmissing, 1.0),
        out=np.zeros_like(real_joint_missing_counts, dtype=float),
    )
    syn_cond_missing = np.divide(
        syn_joint_missing_counts,
        np.maximum(syn_related_missing, 1.0),
        out=np.zeros_like(syn_joint_missing_counts, dtype=float),
    )
    syn_cond_nonmissing = np.divide(
        syn_missing_counts[:, np.newaxis] - syn_joint_missing_counts,
        np.maximum(syn_related_nonmissing, 1.0),
        out=np.zeros_like(syn_joint_missing_counts, dtype=float),
    )

    syn_missing_zero_mask = (syn_related_missing <= 0)[0]
    syn_nonmissing_zero_mask = (syn_related_nonmissing <= 0)[0]
    if bool(np.any(syn_missing_zero_mask)):
        syn_cond_missing[:, syn_missing_zero_mask] = syn_target_rates[:, np.newaxis]
    if bool(np.any(syn_nonmissing_zero_mask)):
        syn_cond_nonmissing[:, syn_nonmissing_zero_mask] = syn_target_rates[:, np.newaxis]

    real_delta_missing = real_cond_missing - real_target_rates[:, np.newaxis]
    real_delta_nonmissing = real_cond_nonmissing - real_target_rates[:, np.newaxis]
    syn_delta_missing = syn_cond_missing - syn_target_rates[:, np.newaxis]
    syn_delta_nonmissing = syn_cond_nonmissing - syn_target_rates[:, np.newaxis]

    real_state_prob_missing = real_related_missing / max(real_row_count, 1)
    real_state_prob_nonmissing = real_related_nonmissing / max(real_row_count, 1)
    centered_distance_matrix = (
        real_state_prob_nonmissing * np.abs(real_delta_nonmissing - syn_delta_nonmissing)
        + real_state_prob_missing * np.abs(real_delta_missing - syn_delta_missing)
    )
    ordered_score_matrix = np.clip(1.0 - (0.5 * centered_distance_matrix), 0.0, 1.0)
    np.fill_diagonal(ordered_score_matrix, np.nan)

    pair_score_matrix = 0.5 * (ordered_score_matrix + ordered_score_matrix.T)
    upper_left, upper_right = np.triu_indices(target_count, k=1)
    pair_scores = pair_score_matrix[upper_left, upper_right]
    ordered_edge_count = int(pair_scores.size * 2)
    pair_rows: list[dict[str, Any]] = []
    if EMIT_PAIR_ROWS:
        for left_idx, right_idx in zip(upper_left.tolist(), upper_right.tolist(), strict=False):
            left_score, left_row = _ordered_centered_profile_score_from_counts(
                target_idx=left_idx,
                related_idx=right_idx,
                real_row_count=real_row_count,
                syn_row_count=syn_row_count,
                real_missing_counts=real_missing_counts,
                syn_missing_counts=syn_missing_counts,
                real_joint_missing_counts=real_joint_missing_counts,
                syn_joint_missing_counts=syn_joint_missing_counts,
            )
            right_score, right_row = _ordered_centered_profile_score_from_counts(
                target_idx=right_idx,
                related_idx=left_idx,
                real_row_count=real_row_count,
                syn_row_count=syn_row_count,
                real_missing_counts=real_missing_counts,
                syn_missing_counts=syn_missing_counts,
                real_joint_missing_counts=real_joint_missing_counts,
                syn_joint_missing_counts=syn_joint_missing_counts,
            )
            pair_id = f"{missing_targets[left_idx]}__{missing_targets[right_idx]}"
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "target_missing_column": missing_targets[left_idx],
                    "related_missing_column": missing_targets[right_idx],
                    "direction": f"{missing_targets[left_idx]}|{missing_targets[right_idx]}",
                    **left_row,
                }
            )
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "target_missing_column": missing_targets[right_idx],
                    "related_missing_column": missing_targets[left_idx],
                    "direction": f"{missing_targets[right_idx]}|{missing_targets[left_idx]}",
                    **right_row,
                }
            )

    if pair_scores.size == 0:
        return {
            "pairwise_centered_status": "not_applicable_no_pairs",
            "pairwise_centered_comissing_score": None,
            "pairwise_centered_pair_count": 0,
            "pairwise_centered_ordered_edge_count": 0,
            "active_missing_target_count": len(missing_targets),
            "pair_rows": [],
        }

    return {
        "pairwise_centered_status": "ok",
        "pairwise_centered_comissing_score": round(float(np.nanmean(pair_scores)), 6),
        "pairwise_centered_pair_count": int(pair_scores.size),
        "pairwise_centered_ordered_edge_count": ordered_edge_count,
        "active_missing_target_count": len(missing_targets),
        "pair_rows": pair_rows,
    }


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _evaluate_asset_row(source_row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_id = str(source_row.get("dataset_id") or "").strip()
    synthetic_csv_path = Path(str(source_row.get("synthetic_csv_path") or "").strip())
    model_id = _normalize_model(source_row.get("model_id"))
    if not synthetic_csv_path.exists():
        payload = {
            **source_row,
            "dataset_id": dataset_id,
            "dataset_prefix": _dataset_prefix(dataset_id),
            "model_id": model_id,
            "model_label": _model_label(model_id),
            "current_status": source_row.get("status"),
            "marginal_missing_rate_consistency": _maybe_float(source_row.get("marginal_missing_rate_consistency")),
            "current_broad_comissing_score": _maybe_float(source_row.get("co_missingness_pattern_consistency")),
            "current_missingness_structure_score": _maybe_float(source_row.get("missingness_structure_score")),
            "pairwise_centered_status": "synthetic_csv_missing",
            "pairwise_centered_comissing_score": None,
            "pairwise_centered_pair_count": 0,
            "pairwise_centered_ordered_edge_count": 0,
            "active_missing_target_count": None,
            "pairwise_centered_missingness_structure_score": None,
            "delta_pairwise_centered_minus_current_broad": None,
        }
        return payload, []

    pairwise_result = _pairwise_centered_score_for_asset(dataset_id, synthetic_csv_path)
    payload = {
        **source_row,
        "dataset_id": dataset_id,
        "dataset_prefix": _dataset_prefix(dataset_id),
        "model_id": model_id,
        "model_label": _model_label(model_id),
        "current_status": source_row.get("status"),
        "marginal_missing_rate_consistency": _maybe_float(source_row.get("marginal_missing_rate_consistency")),
        "current_broad_comissing_score": _maybe_float(source_row.get("co_missingness_pattern_consistency")),
        "current_missingness_structure_score": _maybe_float(source_row.get("missingness_structure_score")),
        "pairwise_centered_status": pairwise_result.get("pairwise_centered_status"),
        "pairwise_centered_comissing_score": pairwise_result.get("pairwise_centered_comissing_score"),
        "pairwise_centered_pair_count": pairwise_result.get("pairwise_centered_pair_count"),
        "pairwise_centered_ordered_edge_count": pairwise_result.get("pairwise_centered_ordered_edge_count"),
        "active_missing_target_count": pairwise_result.get("active_missing_target_count"),
    }
    marginal = payload.get("marginal_missing_rate_consistency")
    pairwise_score = payload.get("pairwise_centered_comissing_score")
    if marginal is not None and pairwise_score is not None:
        payload["pairwise_centered_missingness_structure_score"] = round(float(np.mean([float(marginal), float(pairwise_score)])), 6)
    else:
        payload["pairwise_centered_missingness_structure_score"] = None
    current_broad = payload.get("current_broad_comissing_score")
    if current_broad is not None and pairwise_score is not None:
        payload["delta_pairwise_centered_minus_current_broad"] = round(float(pairwise_score) - float(current_broad), 6)
    else:
        payload["delta_pairwise_centered_minus_current_broad"] = None

    pair_rows = []
    for row in pairwise_result.get("pair_rows", []):
        pair_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": _dataset_prefix(dataset_id),
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "synthetic_csv_path": str(synthetic_csv_path),
                **row,
            }
        )
    return payload, pair_rows


def _mean_or_none(values: list[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def _summarize_asset_rows(asset_rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in asset_rows:
        grouped[tuple(str(row.get(key) or "") for key in group_keys)].append(row)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        payload = {field: value for field, value in zip(group_keys, key)}
        payload["asset_count"] = len(items)
        payload["current_applicable_asset_count"] = sum(1 for item in items if item.get("current_status") == "ok")
        payload["pairwise_centered_applicable_asset_count"] = sum(1 for item in items if item.get("pairwise_centered_status") == "ok")
        for field in (
            "marginal_missing_rate_consistency",
            "current_broad_comissing_score",
            "current_missingness_structure_score",
            "pairwise_centered_comissing_score",
            "pairwise_centered_missingness_structure_score",
            "delta_pairwise_centered_minus_current_broad",
        ):
            payload[field] = _mean_or_none([item.get(field) for item in items])
            if payload[field] is not None:
                payload[field] = round(float(payload[field]), 6)
        payload["pairwise_centered_pair_count__max"] = int(
            max(float(item.get("pairwise_centered_pair_count") or 0.0) for item in items)
        )
        payload["active_missing_target_count__max"] = int(
            max(float(item.get("active_missing_target_count") or 0.0) for item in items)
        )
        rows.append(payload)
    return rows


def run_pairwise_centered_diagnostic(max_workers: int | None = None) -> dict[str, Path]:
    _ensure_dirs()
    asset_df_source = pd.read_csv(CURRENT_ASSET_CSV, encoding="utf-8-sig")
    source_rows = asset_df_source.to_dict(orient="records")

    asset_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    worker_count = max_workers if max_workers is not None else min(8, max(1, (cpu_count() or 4) - 1))

    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        futures = [executor.submit(_evaluate_asset_row, row) for row in source_rows]
        for index, future in enumerate(as_completed(futures), start=1):
            asset_row, asset_pair_rows = future.result()
            asset_rows.append(asset_row)
            pair_rows.extend(asset_pair_rows)
            print(
                f"[pairwise-centered] asset={index}/{len(futures)}"
                f" dataset={asset_row['dataset_id']}"
                f" model={asset_row['model_id']}"
                f" status={asset_row['pairwise_centered_status']}",
                flush=True,
            )

    asset_df = pd.DataFrame(asset_rows)
    pair_df = pd.DataFrame(pair_rows)
    if not asset_df.empty:
        asset_df["model_sort"] = asset_df["model_id"].map(lambda item: _model_sort_key(str(item)))
        asset_df = asset_df.sort_values(["dataset_id", "model_sort", "model_id"]).drop(columns=["model_sort"]).reset_index(drop=True)

    dataset_model_df = pd.DataFrame(_summarize_asset_rows(asset_rows, ("dataset_id", "dataset_prefix", "model_id", "model_label")))
    model_overall_df = pd.DataFrame(_summarize_asset_rows(asset_rows, ("model_id", "model_label")))
    dataset_overall_df = pd.DataFrame(_summarize_asset_rows(asset_rows, ("dataset_id", "dataset_prefix")))
    if not model_overall_df.empty:
        model_overall_df["model_sort"] = model_overall_df["model_id"].map(lambda item: _model_sort_key(str(item)))
        model_overall_df = model_overall_df.sort_values(["model_sort", "model_id"]).drop(columns=["model_sort"]).reset_index(drop=True)
    if not dataset_model_df.empty:
        dataset_model_df["model_sort"] = dataset_model_df["model_id"].map(lambda item: _model_sort_key(str(item)))
        dataset_model_df = dataset_model_df.sort_values(["dataset_id", "model_sort", "model_id"]).drop(columns=["model_sort"]).reset_index(drop=True)
    if not pair_df.empty:
        pair_df["model_sort"] = pair_df["model_id"].map(lambda item: _model_sort_key(str(item)))
        pair_df = pair_df.sort_values(["dataset_id", "model_sort", "model_id", "pair_id", "direction"]).drop(columns=["model_sort"]).reset_index(drop=True)

    asset_csv = DATA_DIR / "pairwise_centered_asset_scores.csv"
    pair_csv = DATA_DIR / "pairwise_centered_pair_scores.csv"
    dataset_model_csv = DATA_DIR / "pairwise_centered_model_dataset_summary.csv"
    model_overall_csv = DATA_DIR / "pairwise_centered_model_overall_summary.csv"
    dataset_overall_csv = DATA_DIR / "pairwise_centered_dataset_overall_summary.csv"

    write_csv(asset_csv, asset_df.to_dict(orient="records"))
    write_csv(pair_csv, pair_df.to_dict(orient="records"))
    write_csv(dataset_model_csv, dataset_model_df.to_dict(orient="records"))
    write_csv(model_overall_csv, model_overall_df.to_dict(orient="records"))
    write_csv(dataset_overall_csv, dataset_overall_df.to_dict(orient="records"))

    applicable_panels = int(
        asset_df["pairwise_centered_status"].eq("ok").sum()
    ) if not asset_df.empty else 0
    applicable_datasets = int(
        asset_df.loc[asset_df["pairwise_centered_status"].eq("ok"), "dataset_id"].nunique()
    ) if not asset_df.empty else 0
    readme_lines = [
        "# Pairwise-Centered Co-Missing Diagnostic",
        "",
        "- This is an experimental diagnostic and does not modify the official missingness bundle.",
        "- Canonical marginal score is reused unchanged from the direct missingness evaluator.",
        "- Experimental co-missing score restricts the second axis to pairs of active missing-target columns.",
        "- For each ordered pair `(Mi | Mj)`, we compare centered profiles:",
        "  - `delta_real(r) = P_real(Mi=1 | Mj=r) - P_real(Mi=1)`",
        "  - `delta_syn(r) = P_syn(Mi=1 | Mj=r) - P_syn(Mi=1)`",
        "  - `score = 1 - 0.5 * sum_r P_real(Mj=r) * |delta_real(r) - delta_syn(r)|`",
        "- Final experimental co-missing score = mean over unordered missing-target pairs after averaging both directions.",
        "",
        f"- Asset panels evaluated: `{asset_df.shape[0]}`",
        f"- Pairwise-applicable panels: `{applicable_panels}`",
        f"- Pairwise-applicable datasets: `{applicable_datasets}`",
        "",
        "## Files",
        "",
        "- `data/pairwise_centered_asset_scores.csv`",
        "- `data/pairwise_centered_pair_scores.csv`",
        "- `data/pairwise_centered_model_dataset_summary.csv`",
        "- `data/pairwise_centered_model_overall_summary.csv`",
        "- `data/pairwise_centered_dataset_overall_summary.csv`",
    ]
    (OUTPUT_ROOT / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    (FINAL_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    for src in (asset_csv, pair_csv, dataset_model_csv, model_overall_csv, dataset_overall_csv):
        (FINAL_DIR / src.name).write_text(src.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")

    return {
        "asset_scores": asset_csv,
        "pair_scores": pair_csv,
        "model_dataset_summary": dataset_model_csv,
        "model_overall_summary": model_overall_csv,
        "dataset_overall_summary": dataset_overall_csv,
    }


if __name__ == "__main__":
    outputs = run_pairwise_centered_diagnostic()
    for key, value in outputs.items():
        print(f"{key}: {value}")
