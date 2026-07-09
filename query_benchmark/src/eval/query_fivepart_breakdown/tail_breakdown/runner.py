#!/usr/bin/env python3
"""Decompose tail robustness with the canonical three-part tail lens."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.tail_threshold.runner import (
    _build_key_counter,
    _build_transformers,
    _infer_threshold_specs_from_rows,
    _is_id_like,
    _load_existing_dataset_outputs,
    _load_target_column,
    _read_csv_rows,
    _select_bottom_band,
    _sorted_support_items,
    _threshold_specs,
    resolve_real_split_path,
)
from src.eval.common import DEFAULT_SQL_SOURCE_VERSION, resolve_requested_sql_source_version, sql_source_label
from src.eval.query_fivepart_breakdown.common_final import render_final_readme, sync_final_outputs, versioned_name
from src.eval.query_fivepart_breakdown.common_heatmap_palette import (
    format_heatmap_latex_cell,
    get_heatmap_cmap,
)
from src.eval.query_fivepart_breakdown.common_model_subitem_grouped_bars import (
    plot_model_subitem_grouped_bar_preview,
    write_model_subitem_grouped_bar_tex,
)
from src.eval.query_fivepart_breakdown.common_model_subitem_heatmap import (
    build_model_subitem_heatmap_df,
    plot_model_subitem_heatmap_preview,
    write_model_subitem_heatmap_tex,
)


EVALUATION_ROOT = PROJECT_ROOT / "Evaluation"
TAIL_THRESHOLD_ROOT = EVALUATION_ROOT / "tail_threshold"
OUTPUT_ROOT = EVALUATION_ROOT / "query_fivepart_breakdown" / "tail_breakdown"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
FINAL_DIR = OUTPUT_ROOT / "final"
OUTPUT_VERSION_TAG = resolve_requested_sql_source_version("analysis", DEFAULT_SQL_SOURCE_VERSION)

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
MODEL_LABELS = {
    "arf": "ARF",
    "bayesnet": "BayesNet",
    "cdtd": "CDTD",
    "codi": "CoDi",
    "ctgan": "CTGAN",
    "forestdiffusion": "ForestDiffusion",
    "goggle": "GOGGLE",
    "realtabformer": "RealTabFormer",
    "rtf": "RealTabFormer",
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
EXCLUDED_MODELS = {"cdtd", "codi", "goggle"}
MODEL_ALIASES = {"rtf": "realtabformer"}
PREFIX_LABELS = {"c": "Categorical", "m": "Mixed", "n": "Numerical"}
SUBITEM_LABELS = {
    "tail_set_consistency": "Tail set consistency",
    "tail_mass_similarity": "Tail mass similarity",
    "tail_concentration_consistency": "Tail concentration consistency",
}

DEFAULT_MAX_WORKERS = 4
DEFAULT_PROXY_MAX_ROWS = 50_000


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, DATA_DIR, FIG_DIR, TABLE_DIR, FINAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _normalize_model(model_id: Any) -> str:
    key = str(model_id or "").strip().lower()
    return MODEL_ALIASES.get(key, key)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _model_sort_key(model_id: str) -> tuple[int, str]:
    label = _model_label(model_id)
    if label == "REAL":
        return (0, label)
    return (1, label.lower())


def _dataset_prefix(dataset_id: str) -> str:
    return str(dataset_id or "").strip().lower()[:1]


def _dataset_sort_key(dataset_id: str) -> tuple[int, int, str]:
    text = str(dataset_id).strip()
    prefix = text[:1].lower()
    number_part = text[1:]
    try:
        number = int(number_part)
    except Exception:
        number = 10**9
    return ({"c": 0, "m": 1, "n": 2}.get(prefix, 50), number, text)


def _threshold_order(threshold_specs: list[Any]) -> list[str]:
    return [str(spec.label) for spec in threshold_specs]


def _metric_stats(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = int(clean.shape[0])
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "se": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci95_radius": None,
        }
    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1)) if n > 1 else 0.0
    se_val = float(std_val / math.sqrt(n)) if n > 1 else 0.0
    ci_radius = 1.96 * se_val
    return {
        "n": n,
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "se": round(se_val, 6),
        "ci95_low": round(mean_val - ci_radius, 6),
        "ci95_high": round(mean_val + ci_radius, 6),
        "ci95_radius": round(ci_radius, 6),
    }


def _resolve_tail_threshold_full_run() -> Path:
    candidates = [path for path in (TAIL_THRESHOLD_ROOT / "runs").iterdir() if path.is_dir() and (path / "datasets").exists()]
    if not candidates:
        raise FileNotFoundError("No tail_threshold full run with dataset-level outputs was found.")
    ranked: list[tuple[int, int, str, Path]] = []
    for candidate in candidates:
        asset_rows, _, _ = _load_existing_dataset_outputs(candidate)
        ranked.append((1 if "full" in candidate.name.lower() else 0, len(asset_rows), candidate.name, candidate))
    ranked.sort(reverse=True)
    return ranked[0][3]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def _resolve_local_repo_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    candidate = Path(text)
    if candidate.exists():
        return str(candidate)
    normalized = text.replace("\\", "/")
    marker = "/SQLagent/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        local = PROJECT_ROOT / Path(relative)
        if local.exists():
            return str(local)
    if normalized.endswith("/SQLagent"):
        return str(PROJECT_ROOT)
    return text


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
            r"\usepgfplotslibrary{groupplots}",
            r"\usepackage{xcolor}",
            r"\pgfplotsset{compat=1.18}",
            "",
        ]
    )


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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _column_tail_rates(
    counts: dict[str, int],
    tail_real_keys: set[str],
    feature_columns: list[str],
    total_per_column: int,
) -> dict[str, float]:
    if total_per_column <= 0:
        return {column: 0.0 for column in feature_columns}
    rates: dict[str, float] = {}
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
    if not columns or not rows_real:
        return []
    target_column = _load_target_column(dataset_id, columns)
    feature_columns = [column for column in columns if column != target_column and not _is_id_like(column)]
    if not feature_columns:
        return []

    transformers = _build_transformers(rows_real, feature_columns, numeric_bins=10)
    real_counts = _build_key_counter(rows_real, feature_columns, transformers)
    real_tail_items = _sorted_support_items(real_counts, reverse=False)
    threshold_specs = _threshold_specs(threshold_pcts)
    real_tail_map = {spec.label: _select_bottom_band(real_tail_items, spec.ratio)[0] for spec in threshold_specs}
    n_real = len(rows_real)

    results: list[dict[str, Any]] = []
    deduped_assets = {str(row["asset_key"]): row for row in asset_rows}
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


def _compute_proxy_rows(
    asset_df: pd.DataFrame,
    threshold_specs: list[Any],
    max_workers: int,
    max_rows_per_table: int,
) -> pd.DataFrame:
    threshold_pcts = [float(spec.pct) for spec in threshold_specs]
    threshold_order = _threshold_order(threshold_specs)
    dataset_to_rows: dict[str, list[dict[str, Any]]] = {}
    for row in asset_df.to_dict("records"):
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
    if proxy_df.empty:
        return proxy_df
    proxy_df["threshold_label"] = pd.Categorical(proxy_df["threshold_label"], categories=threshold_order, ordered=True)
    proxy_df = proxy_df.sort_values(["dataset_id", "model_id", "threshold_label"]).reset_index(drop=True)
    return proxy_df


def _load_asset_frame(source_run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Any]]:
    asset_rows, _, manifest_rows = _load_existing_dataset_outputs(source_run_dir)
    if not asset_rows:
        raise RuntimeError(f"No tail_threshold asset rows found under {source_run_dir}")
    threshold_specs = _infer_threshold_specs_from_rows(asset_rows)
    threshold_order = _threshold_order(threshold_specs)
    asset_df = pd.DataFrame(asset_rows)
    for column in ["synthetic_csv_path", "asset_dir", "root_path"]:
        if column in asset_df.columns:
            asset_df[column] = asset_df[column].map(_resolve_local_repo_path)
    asset_df["model_id"] = asset_df["model_id"].map(_normalize_model)
    asset_df["model_label"] = asset_df["model_id"].map(_model_label)
    asset_df["dataset_prefix"] = asset_df["dataset_id"].map(_dataset_prefix)
    asset_df = asset_df[~asset_df["model_id"].isin(EXCLUDED_MODELS)].copy()
    numeric_cols = [
        "threshold_pct",
        "tail_set_consistency",
        "tail_mass_similarity",
        "tail_concentration_consistency",
        "tail_overall_score",
        "head_proxy_overall_score",
        "tail_head_gap",
    ]
    for column in numeric_cols:
        if column in asset_df.columns:
            asset_df[column] = pd.to_numeric(asset_df[column], errors="coerce")
    asset_df["threshold_label"] = pd.Categorical(asset_df["threshold_label"], categories=threshold_order, ordered=True)
    asset_df = asset_df.sort_values(["dataset_id", "model_id", "threshold_label"]).reset_index(drop=True)
    manifest_df = pd.DataFrame(manifest_rows)
    return asset_df, manifest_df, threshold_specs


def _build_dataset_model_threshold_scores(merged_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "tail_set_consistency",
        "tail_mass_similarity",
        "tail_concentration_consistency",
        "tail_concentration_consistency_preview",
        "tail_coverage_score",
        "tail_breakdown_score",
        "coverage_minus_concentration",
        "head_proxy_overall_score",
        "tail_head_gap",
    ]
    grouped = (
        merged_df.groupby(
            ["dataset_id", "dataset_prefix", "model_id", "model_label", "threshold_label", "threshold_pct"],
            as_index=False,
            observed=True,
        )
        .agg(
            asset_count=("asset_key", "nunique"),
            **{metric: (metric, "mean") for metric in metrics},
        )
        .sort_values(["dataset_id", "model_id", "threshold_pct"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return grouped


def _build_dataset_model_scores(
    dataset_model_threshold_df: pd.DataFrame,
    threshold_specs: list[Any],
) -> pd.DataFrame:
    base = (
        dataset_model_threshold_df.groupby(["dataset_id", "dataset_prefix", "model_id", "model_label"], as_index=False)
        .agg(
            threshold_count=("threshold_label", "nunique"),
            tail_set_consistency=("tail_set_consistency", "mean"),
            tail_mass_similarity=("tail_mass_similarity", "mean"),
            tail_concentration_consistency=("tail_concentration_consistency", "mean"),
            tail_concentration_consistency_preview=("tail_concentration_consistency_preview", "mean"),
            tail_coverage_score=("tail_coverage_score", "mean"),
            tail_breakdown_score=("tail_breakdown_score", "mean"),
            coverage_minus_concentration=("coverage_minus_concentration", "mean"),
            head_proxy_overall_score=("head_proxy_overall_score", "mean"),
            tail_head_gap=("tail_head_gap", "mean"),
        )
        .reset_index(drop=True)
    )

    pivot = dataset_model_threshold_df.pivot_table(
        index=["dataset_id", "dataset_prefix", "model_id", "model_label"],
        columns="threshold_label",
        values=["tail_breakdown_score", "tail_concentration_consistency", "tail_coverage_score"],
        aggfunc="mean",
        observed=True,
    )
    threshold_order = _threshold_order(threshold_specs)
    widest_label = threshold_order[0] if threshold_order else None
    rarest_label = threshold_order[-1] if threshold_order else None
    if not pivot.empty:
        for metric in ["tail_breakdown_score", "tail_concentration_consistency", "tail_coverage_score"]:
            if widest_label and rarest_label and (metric, widest_label) in pivot.columns and (metric, rarest_label) in pivot.columns:
                pivot[(f"{metric}_fragility_drop", "")] = pivot[(metric, widest_label)] - pivot[(metric, rarest_label)]
    pivot = pivot.reset_index()
    pivot.columns = [
        column if isinstance(column, str) else (column[0] if not column[1] else f"{column[0]}__{column[1]}")
        for column in pivot.columns
    ]
    keep_cols = [
        "dataset_id",
        "dataset_prefix",
        "model_id",
        "model_label",
        "tail_breakdown_score_fragility_drop",
        "tail_concentration_consistency_fragility_drop",
        "tail_coverage_score_fragility_drop",
    ]
    fragility = pivot[[col for col in keep_cols if col in pivot.columns]].copy()
    merged = base.merge(fragility, on=["dataset_id", "dataset_prefix", "model_id", "model_label"], how="left")
    merged = merged.sort_values(
        by=["dataset_prefix", "dataset_id", "model_id"],
        key=lambda s: s.map(_dataset_sort_key) if s.name == "dataset_id" else s,
    ).reset_index(drop=True)
    return merged


def _build_model_summary(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "tail_breakdown_score",
        "tail_coverage_score",
        "tail_set_consistency",
        "tail_mass_similarity",
        "tail_concentration_consistency",
        "coverage_minus_concentration",
        "tail_breakdown_score_fragility_drop",
        "tail_concentration_consistency_fragility_drop",
        "tail_coverage_score_fragility_drop",
    ]
    rows: list[dict[str, Any]] = []
    for model_id, group in dataset_model_df.groupby("model_id", sort=False):
        payload = {
            "model_id": model_id,
            "model_label": _model_label(model_id),
            "dataset_count": int(group["dataset_id"].nunique()),
            "dataset_prefixes": ",".join(sorted(group["dataset_prefix"].dropna().astype(str).unique())),
        }
        for metric in metrics:
            stats = _metric_stats(group[metric])
            payload[f"{metric}__mean"] = stats["mean"]
            payload[f"{metric}__std"] = stats["std"]
            payload[f"{metric}__se"] = stats["se"]
            payload[f"{metric}__ci95_low"] = stats["ci95_low"]
            payload[f"{metric}__ci95_high"] = stats["ci95_high"]
            payload[f"{metric}__ci95_radius"] = stats["ci95_radius"]
        rows.append(payload)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["model_sort"] = summary["model_id"].map(_model_sort_key)
    summary = summary.sort_values(["model_sort"]).drop(columns=["model_sort"])
    return summary.reset_index(drop=True)


def _build_prefix_summary(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_id, prefix), group in dataset_model_df.groupby(["model_id", "dataset_prefix"], sort=False):
        rows.append(
            {
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "dataset_prefix": prefix,
                "dataset_prefix_label": PREFIX_LABELS.get(prefix, prefix.upper()),
                "dataset_count": int(group["dataset_id"].nunique()),
                "tail_breakdown_score": round(float(group["tail_breakdown_score"].mean()), 6),
                "tail_coverage_score": round(float(group["tail_coverage_score"].mean()), 6),
                "tail_set_consistency": round(float(group["tail_set_consistency"].mean()), 6),
                "tail_mass_similarity": round(float(group["tail_mass_similarity"].mean()), 6),
                "tail_concentration_consistency": round(float(group["tail_concentration_consistency"].mean()), 6),
                "coverage_minus_concentration": round(float(group["coverage_minus_concentration"].mean()), 6),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["model_sort"] = out["model_id"].map(_model_sort_key)
    out = out.sort_values(["model_sort", "dataset_prefix"]).drop(columns=["model_sort"])
    return out.reset_index(drop=True)


def _build_dataset_summary(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, group in dataset_model_df.groupby("dataset_id", sort=False):
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": _dataset_prefix(dataset_id),
                "model_count": int(group["model_id"].nunique()),
                "tail_breakdown_score": round(float(group["tail_breakdown_score"].mean()), 6),
                "tail_coverage_score": round(float(group["tail_coverage_score"].mean()), 6),
                "tail_concentration_consistency": round(float(group["tail_concentration_consistency"].mean()), 6),
                "coverage_minus_concentration": round(float(group["coverage_minus_concentration"].mean()), 6),
                "tail_concentration_consistency_std_across_models": round(
                    float(group["tail_concentration_consistency"].std(ddof=1)) if len(group) > 1 else 0.0,
                    6,
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["sort_key"] = out["dataset_id"].map(_dataset_sort_key)
    out = out.sort_values(["tail_concentration_consistency", "sort_key"], ascending=[True, True]).drop(columns=["sort_key"])
    return out.reset_index(drop=True)


def _build_heatmap_data(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    heatmap = (
        dataset_model_df.pivot_table(index="dataset_id", columns="model_id", values="tail_concentration_consistency", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if heatmap.empty:
        return heatmap
    heatmap["sort_key"] = heatmap["dataset_id"].map(_dataset_sort_key)
    heatmap = heatmap.sort_values(["sort_key"]).drop(columns=["sort_key"])
    ordered_cols = [item for item in MODEL_ORDER if item in heatmap.columns]
    heatmap = heatmap[["dataset_id"] + ordered_cols]
    return heatmap.reset_index(drop=True)


def _build_prefix_plot_data(prefix_summary_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        prefix_summary_df.pivot_table(
            index=["model_id", "model_label"],
            columns="dataset_prefix",
            values="tail_concentration_consistency",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if pivot.empty:
        return pivot
    pivot["model_sort"] = pivot["model_id"].map(_model_sort_key)
    pivot = pivot.sort_values(["model_sort"]).drop(columns=["model_sort"])
    return pivot.reset_index(drop=True)


def _write_scatter_tex(
    model_summary_df: pd.DataFrame,
    *,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
    note_lines: list[str] | None = None,
) -> None:
    color_defs = [
        rf"\definecolor{{model{row.model_id}}}{{HTML}}{{{MODEL_COLORS[row.model_id].replace('#', '')}}}"
        for row in model_summary_df.itertuples()
        if row.model_id in MODEL_COLORS
    ]
    x_values = pd.to_numeric(model_summary_df[f"{x_metric}__mean"], errors="coerce").dropna()
    y_values = pd.to_numeric(model_summary_df[f"{y_metric}__mean"], errors="coerce").dropna()
    x_min = max(0.0, float(x_values.min()) - 0.08) if not x_values.empty else 0.0
    x_max = min(1.0, float(x_values.max()) + 0.08) if not x_values.empty else 1.0
    y_min = max(0.0, float(y_values.min()) - 0.08) if not y_values.empty else 0.0
    y_max = min(1.0, float(y_values.max()) + 0.08) if not y_values.empty else 1.0

    body = [_tex_preamble(), *color_defs, r"\begin{document}"]
    if note_lines:
        body.append(r"\begin{minipage}{13.2cm}")
        for line in note_lines:
            body.append(r"{\small " + _escape_tex(line) + r"\par}")
        body.append(r"\vspace{0.4em}")
    body.extend([r"\begin{tikzpicture}", r"\begin{axis}["])
    body.extend(
        [
            r"width=12.7cm,",
            r"height=9.3cm,",
            rf"xmin={x_min:.4f}, xmax={x_max:.4f},",
            rf"ymin={y_min:.4f}, ymax={y_max:.4f},",
            rf"xlabel={{{_escape_tex(x_label)}}},",
            rf"ylabel={{{_escape_tex(y_label)}}},",
            rf"title={{{_escape_tex(title)}}},",
            r"grid=both,",
            r"grid style={draw=gray!20},",
            r"major grid style={draw=gray!30},",
            r"axis line style={draw=black!70},",
            r"tick style={draw=black!70},",
            r"legend style={draw=none, fill=none, font=\scriptsize, at={(0.02,0.02)}, anchor=south west},",
            r"legend columns=3,",
            r"]",
        ]
    )
    body.append(r"\addplot[black!45, dashed, domain=0:1, samples=2] {x};")
    for row in model_summary_df.itertuples():
        color_name = f"model{row.model_id}"
        x = float(getattr(row, f"{x_metric}__mean"))
        y = float(getattr(row, f"{y_metric}__mean"))
        xerr = float(getattr(row, f"{x_metric}__ci95_radius") or 0.0)
        yerr = float(getattr(row, f"{y_metric}__ci95_radius") or 0.0)
        body.append(
            "\n".join(
                [
                    rf"\addplot+[only marks, mark=*, mark size=2.7pt, draw={color_name}, fill={color_name},",
                    r"error bars/.cd, x dir=both, x explicit, y dir=both, y explicit]",
                    rf"coordinates {{ ({x:.4f},{y:.4f}) +- ({xerr:.4f},{yerr:.4f}) }};",
                    rf"\addlegendentry{{{_escape_tex(row.model_label)}}}",
                ]
            )
        )
    body.extend([r"\end{axis}", r"\end{tikzpicture}"])
    if note_lines:
        body.append(r"\end{minipage}")
    body.extend([r"\end{document}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


def _write_prefix_bar_tex(prefix_plot_df: pd.DataFrame, path: Path) -> None:
    prefixes = ["c", "m", "n"]
    color_defs = [
        rf"\definecolor{{model{row.model_id}}}{{HTML}}{{{MODEL_COLORS[row.model_id].replace('#', '')}}}"
        for row in prefix_plot_df.itertuples()
        if row.model_id in MODEL_COLORS
    ]
    model_labels = [_escape_tex(label) for label in prefix_plot_df["model_label"].tolist()]
    symbolic = ",".join(model_labels)

    body = [_tex_preamble(), *color_defs, r"\begin{document}", r"\begin{tikzpicture}"]
    body.extend(
        [
            r"\begin{groupplot}[",
            r"group style={group size=3 by 1, horizontal sep=1.15cm},",
            r"width=5.0cm,",
            r"height=7.0cm,",
            r"ymin=0.0, ymax=1.0,",
            r"ymajorgrids,",
            r"grid style={draw=gray!20},",
            r"major grid style={draw=gray!30},",
            rf"symbolic x coords={{{symbolic}}},",
            r"xtick=data,",
            r"x tick label style={rotate=45, anchor=east, font=\scriptsize},",
            r"tick style={draw=black!70},",
            r"axis line style={draw=black!70},",
            r"]",
        ]
    )
    for prefix in prefixes:
        title = PREFIX_LABELS[prefix]
        ylabel = "Tail concentration consistency" if prefix == "c" else ""
        body.append(rf"\nextgroupplot[title={{{title}}}, ylabel={{{ylabel}}}]")
        for row in prefix_plot_df.itertuples():
            value = getattr(row, prefix, None)
            if value is None or pd.isna(value):
                continue
            label = _escape_tex(str(row.model_label))
            color_name = f"model{row.model_id}"
            body.append(
                rf"\addplot+[ybar, bar width=7.0pt, draw={color_name}, fill={color_name}] coordinates {{ ({label},{float(value):.4f}) }};"
            )
    body.extend([r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


def _write_heatmap_tex(heatmap_df: pd.DataFrame, path: Path) -> None:
    matrix = heatmap_df.copy()
    model_cols = [item for item in MODEL_ORDER if item in matrix.columns]
    if not model_cols:
        path.write_text("", encoding="utf-8")
        return
    display = matrix[["dataset_id"] + model_cols].copy().fillna("")
    lines = [
        r"\documentclass{standalone}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{booktabs}",
        r"\begin{document}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{l{'c' * len(model_cols)}}}",
        r"\toprule",
        "Dataset & " + " & ".join(_escape_tex(_model_label(col)) for col in model_cols) + r" \\",
        r"\midrule",
    ]
    for row in display.itertuples(index=False):
        dataset_id = getattr(row, "dataset_id")
        cells = [_escape_tex(str(dataset_id).upper())]
        for model_id in model_cols:
            value = getattr(row, model_id)
            if value == "":
                cells.append("")
                continue
            cells.append(format_heatmap_latex_cell(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_scatter_preview(
    model_summary_df: pd.DataFrame,
    *,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    pdf_path: Path,
    png_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.0, zorder=1)
    for row in model_summary_df.itertuples():
        x = float(getattr(row, f"{x_metric}__mean"))
        y = float(getattr(row, f"{y_metric}__mean"))
        xerr = float(getattr(row, f"{x_metric}__ci95_radius") or 0.0)
        yerr = float(getattr(row, f"{y_metric}__ci95_radius") or 0.0)
        color = MODEL_COLORS[row.model_id]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.15,
            capsize=3.0,
            markersize=6.7,
            zorder=3,
            label=row.model_label,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.24)
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), ncol=3, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefix_bar_preview(prefix_plot_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    prefixes = ["c", "m", "n"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.2), sharey=True)
    for ax, prefix in zip(axes, prefixes):
        values = pd.to_numeric(prefix_plot_df[prefix], errors="coerce")
        colors = [MODEL_COLORS.get(model_id, "#777777") for model_id in prefix_plot_df["model_id"]]
        ax.bar(range(len(prefix_plot_df)), values, color=colors)
        ax.set_title(PREFIX_LABELS[prefix])
        ax.set_xticks(range(len(prefix_plot_df)))
        ax.set_xticklabels(prefix_plot_df["model_label"], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.22)
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Tail concentration consistency")
    fig.suptitle("Tail concentration consistency by dataset family prefix")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_preview(heatmap_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    matrix = heatmap_df.copy()
    model_cols = [item for item in MODEL_ORDER if item in matrix.columns]
    ordered = matrix[["dataset_id"] + model_cols].copy()
    values = ordered[model_cols].to_numpy(dtype=float)
    fig_height = max(8.0, 0.20 * len(ordered) + 1.8)
    fig, ax = plt.subplots(figsize=(10.4, fig_height))
    image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap=get_heatmap_cmap())
    ax.set_xticks(range(len(model_cols)))
    ax.set_xticklabels([_model_label(item) for item in model_cols], rotation=45, ha="right")
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered["dataset_id"].str.upper().tolist(), fontsize=8)
    ax.set_title("Dataset-model tail concentration heatmap")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Tail concentration consistency")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _write_model_table_tex(model_summary_df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Tail breakdown & Tail coverage core & Tail concentration & Coverage-Concentration gap \\",
        r"\midrule",
    ]
    for row in model_summary_df.itertuples():
        lines.append(
            (
                f"{_escape_tex(row.model_label)} & "
                f"{float(getattr(row, 'tail_breakdown_score__mean')):.3f} & "
                f"{float(getattr(row, 'tail_coverage_score__mean')):.3f} & "
                f"{float(getattr(row, 'tail_concentration_consistency__mean')):.3f} & "
                f"{float(getattr(row, 'coverage_minus_concentration__mean')):.3f} \\\\"
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_report(
    source_run_dir: Path,
    dataset_model_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    prefix_summary_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
) -> None:
    top_concentration = model_summary_df.sort_values("tail_concentration_consistency__mean", ascending=False).iloc[0]
    top_coverage = model_summary_df.sort_values("tail_coverage_score__mean", ascending=False).iloc[0]
    top_overall = model_summary_df.sort_values("tail_breakdown_score__mean", ascending=False).iloc[0]
    most_concentration_heavy = model_summary_df.sort_values("coverage_minus_concentration__mean").iloc[0]
    most_coverage_heavy = model_summary_df.sort_values("coverage_minus_concentration__mean", ascending=False).iloc[0]
    hardest_dataset = dataset_summary_df.sort_values("tail_concentration_consistency").iloc[0]
    easiest_dataset = dataset_summary_df.sort_values("tail_concentration_consistency", ascending=False).iloc[0]

    lines = [
        "# Tail Breakdown Report",
        "",
        "## Scope",
        "",
        f"- Source tail-threshold run: `{source_run_dir.name}`",
        f"- Excluded models: `{', '.join(sorted(EXCLUDED_MODELS))}`",
        f"- Included models: `{model_summary_df.shape[0]}`",
        f"- Deduplicated dataset-model panels: `{dataset_model_df.shape[0]}`",
        f"- Threshold count per panel: `{dataset_model_df['threshold_count'].min()}-{dataset_model_df['threshold_count'].max()}`",
        "",
        "## Canonical tail views",
        "",
        "- Canonical tail-threshold components reused directly: `tail_set_consistency`, `tail_mass_similarity`, `tail_concentration_consistency`.",
        "- `tail_coverage_score = mean(tail_set_consistency, tail_mass_similarity)`",
        "- `tail_breakdown_score = mean(tail_set_consistency, tail_mass_similarity, tail_concentration_consistency)`",
        "- `coverage_minus_concentration = tail_coverage_score - tail_concentration_consistency`",
        "",
        "## Main findings",
        "",
        (
            f"1. `{top_concentration['model_label']}` is strongest on tail concentration "
            f"with mean tail concentration score `{top_concentration['tail_concentration_consistency__mean']:.3f}`."
        ),
        (
            f"2. `{top_coverage['model_label']}` is strongest on tail coverage core "
            f"(`tail_coverage_score`) with mean score `{top_coverage['tail_coverage_score__mean']:.3f}`, while "
            f"`{top_overall['model_label']}` leads the three-part tail breakdown overall "
            f"at `{top_overall['tail_breakdown_score__mean']:.3f}`."
        ),
        (
            f"3. `{most_coverage_heavy['model_label']}` is the most coverage-heavy model "
            f"(coverage minus concentration = `{most_coverage_heavy['coverage_minus_concentration__mean']:.3f}`), while "
            f"`{most_concentration_heavy['model_label']}` is the most concentration-leaning "
            f"(`{most_concentration_heavy['coverage_minus_concentration__mean']:.3f}`)."
        ),
        (
            f"4. Dataset difficulty remains uneven: `{hardest_dataset['dataset_id']}` is hardest on tail concentration "
            f"(`{hardest_dataset['tail_concentration_consistency']:.3f}` mean across models), while "
            f"`{easiest_dataset['dataset_id']}` is easiest (`{easiest_dataset['tail_concentration_consistency']:.3f}`)."
        ),
        "",
        "## Files to use first",
        "",
        "- `figures/tail_coverage_vs_concentration_scatter_main.pdf`",
        "- `figures/tail_coverage_vs_breakdown_bridge.pdf`",
        "- `figures/tail_prefix_bars_appendix.pdf`",
        "- `tables/tail_model_summary_generated.tex`",
        "- `data/model_summary.csv`",
        "",
        "## Prefix note",
        "",
        f"- Prefix coverage summary rows: `{prefix_summary_df.shape[0]}`",
        "- The `c / m / n` split is exported explicitly because tail concentration behavior differs by dataset family, not just by overall model average.",
        "",
    ]
    (OUTPUT_ROOT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def _build_readme(source_run_dir: Path) -> None:
    content = f"""# Tail Breakdown

This directory contains a canonical tail decomposition analysis built from the repository's `tail_threshold` dataset-level full run.

## Inputs

- Source run: `{source_run_dir.name}`
- Source root: `{source_run_dir.relative_to(PROJECT_ROOT)}`
- Full dataset-level tail outputs: `Evaluation/tail_threshold/runs/{source_run_dir.name}/datasets/`
- Color convention: `README.md`

## What this analysis exports

- threshold-level canonical tail decomposition
- deduplicated dataset-model tail concentration summaries
- model-level and prefix-level summaries
- paper-ready TikZ figures and LaTeX table snippets
- final copies under `Evaluation/query_fivepart_breakdown/tail_breakdown/final/`

## Re-run

```bash
python src/eval/query_fivepart_breakdown/tail_breakdown/runner.py
```

## Notes

- This breakdown now uses only the frozen three-part canonical tail contract.
"""
    (OUTPUT_ROOT / "README.md").write_text(content, encoding="utf-8")


def _try_compile_tex(tex_path: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["latexmk", "-pdf", tex_path.name],
            cwd=tex_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "latexmk not available"
    return proc.returncode == 0, proc.stdout[-1200:]


def _copy_final_artifacts(
    files: list[Path],
    must_do_aliases: dict[str, Path] | None = None,
    *,
    version_tag: str,
) -> None:
    sync_final_outputs(FINAL_DIR, files, must_do_aliases, version_tag=version_tag, copy_plain_files=False)


def run_tail_breakdown(
    *,
    source_run_dir: Path | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    proxy_max_rows: int = DEFAULT_PROXY_MAX_ROWS,
) -> dict[str, Any]:
    _ensure_dirs()
    resolved_source_run_dir = source_run_dir.expanduser().resolve() if source_run_dir is not None else _resolve_tail_threshold_full_run()
    asset_df, manifest_df, threshold_specs = _load_asset_frame(resolved_source_run_dir)
    merged_df = asset_df.copy()
    merged_df["tail_coverage_score"] = merged_df[["tail_set_consistency", "tail_mass_similarity"]].mean(axis=1, skipna=True)
    merged_df["tail_breakdown_score"] = merged_df[
        [
            "tail_set_consistency",
            "tail_mass_similarity",
            "tail_concentration_consistency",
        ]
    ].mean(axis=1, skipna=True)
    merged_df["coverage_minus_concentration"] = merged_df["tail_coverage_score"] - merged_df["tail_concentration_consistency"]
    proxy_df = pd.DataFrame()
    preview_source = (
        "subgroup_tail_concentration_consistency_proxy"
        if "subgroup_tail_concentration_consistency_proxy" in merged_df.columns
        else "tail_concentration_consistency"
    )
    merged_df["tail_concentration_consistency_preview"] = pd.to_numeric(
        merged_df[preview_source], errors="coerce"
    ).fillna(pd.to_numeric(merged_df["tail_concentration_consistency"], errors="coerce"))

    dataset_model_threshold_df = _build_dataset_model_threshold_scores(merged_df)
    dataset_model_df = _build_dataset_model_scores(dataset_model_threshold_df, threshold_specs)
    model_summary_df = _build_model_summary(dataset_model_df)
    prefix_summary_df = _build_prefix_summary(dataset_model_df)
    dataset_summary_df = _build_dataset_summary(dataset_model_df)
    heatmap_df = _build_heatmap_data(dataset_model_df)
    prefix_plot_df = _build_prefix_plot_data(prefix_summary_df)
    model_subitem_heatmap_df = build_model_subitem_heatmap_df(
        model_summary_df,
        model_id_col="model_id",
        model_order=MODEL_ORDER,
        subitem_specs=[
            (subitem_id, SUBITEM_LABELS[subitem_id], f"{subitem_id}__mean")
            for subitem_id in ["tail_set_consistency", "tail_mass_similarity", "tail_concentration_consistency"]
        ],
        summary_row_spec=("family_mean", "Family mean", "tail_breakdown_score__mean"),
    )

    _write_csv(manifest_df, DATA_DIR / "source_dataset_manifest.csv")
    _write_csv(merged_df, DATA_DIR / "tail_threshold_asset_rows_enriched.csv")
    _write_csv(dataset_model_threshold_df, DATA_DIR / "dataset_model_threshold_scores.csv")
    _write_csv(dataset_model_df, DATA_DIR / "dataset_model_scores.csv")
    _write_csv(model_summary_df, DATA_DIR / "model_summary.csv")
    _write_csv(prefix_summary_df, DATA_DIR / "prefix_summary.csv")
    _write_csv(dataset_summary_df, DATA_DIR / "dataset_summary.csv")
    _write_csv(heatmap_df, DATA_DIR / "dataset_model_heatmap.csv")
    _write_csv(prefix_plot_df, DATA_DIR / "prefix_plot_data.csv")
    _write_csv(model_subitem_heatmap_df, DATA_DIR / "model_subitem_heatmap.csv")

    tradeoff_tex = FIG_DIR / "tail_coverage_vs_concentration_scatter_main.tex"
    tradeoff_pdf = FIG_DIR / "tail_coverage_vs_concentration_scatter_main.pdf"
    tradeoff_png = FIG_DIR / "tail_coverage_vs_concentration_scatter_main.png"
    bridge_tex = FIG_DIR / "tail_coverage_vs_breakdown_bridge.tex"
    bridge_pdf = FIG_DIR / "tail_coverage_vs_breakdown_bridge.pdf"
    bridge_png = FIG_DIR / "tail_coverage_vs_breakdown_bridge.png"
    prefix_tex = FIG_DIR / "tail_prefix_bars_appendix.tex"
    prefix_pdf = FIG_DIR / "tail_prefix_bars_appendix.pdf"
    prefix_png = FIG_DIR / "tail_prefix_bars_appendix.png"
    heatmap_tex = FIG_DIR / "tail_dataset_model_heatmap_appendix.tex"
    heatmap_pdf = FIG_DIR / "tail_dataset_model_heatmap_appendix.pdf"
    heatmap_png = FIG_DIR / "tail_dataset_model_heatmap_appendix.png"
    model_subitem_heatmap_tex = FIG_DIR / "tail_model_subitem_heatmap_appendix.tex"
    model_subitem_heatmap_pdf = FIG_DIR / "tail_model_subitem_heatmap_appendix.pdf"
    model_subitem_heatmap_png = FIG_DIR / "tail_model_subitem_heatmap_appendix.png"
    grouped_bars_tex = FIG_DIR / "tail_family_subitem_bars_appendix.tex"
    grouped_bars_pdf = FIG_DIR / "tail_family_subitem_bars_appendix.pdf"
    grouped_bars_png = FIG_DIR / "tail_family_subitem_bars_appendix.png"

    _write_scatter_tex(
        model_summary_df,
        x_metric="tail_coverage_score",
        y_metric="tail_concentration_consistency",
        x_label="Tail coverage core score",
        y_label="Tail concentration consistency",
        title="Tail coverage core vs tail concentration",
        path=tradeoff_tex,
        note_lines=[
            "Main paper-facing view.",
            "X-axis is tail coverage core = mean(tail set consistency, tail mass similarity).",
            "Y-axis is tail concentration consistency.",
        ],
    )
    _write_scatter_tex(
        model_summary_df,
        x_metric="tail_coverage_score",
        y_metric="tail_breakdown_score",
        x_label="Tail coverage core score",
        y_label="Tail breakdown score",
        title="Tail coverage core vs tail breakdown score",
        path=bridge_tex,
        note_lines=[
            "Tail coverage core = mean(tail set consistency, tail mass similarity).",
            "Tail breakdown score = mean(tail set consistency, tail mass similarity, tail concentration consistency).",
        ],
    )
    _write_prefix_bar_tex(prefix_plot_df, prefix_tex)
    _write_heatmap_tex(heatmap_df, heatmap_tex)
    write_model_subitem_heatmap_tex(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        title="Tail model-subitem heatmap",
        colorbar_title="Mean score",
        path=model_subitem_heatmap_tex,
    )
    write_model_subitem_grouped_bar_tex(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        model_color_map=MODEL_COLORS,
        title="Tail family and subitem bars",
        y_label="Score",
        path=grouped_bars_tex,
    )

    _plot_scatter_preview(
        model_summary_df,
        x_metric="tail_coverage_score",
        y_metric="tail_concentration_consistency",
        x_label="Tail coverage core score",
        y_label="Tail concentration consistency",
        title="Tail coverage core vs tail concentration",
        pdf_path=tradeoff_pdf,
        png_path=tradeoff_png,
    )
    _plot_scatter_preview(
        model_summary_df,
        x_metric="tail_coverage_score",
        y_metric="tail_breakdown_score",
        x_label="Tail coverage core score",
        y_label="Tail breakdown score",
        title="Tail coverage core vs tail breakdown score",
        pdf_path=bridge_pdf,
        png_path=bridge_png,
    )
    _plot_prefix_bar_preview(prefix_plot_df, prefix_pdf, prefix_png)
    _plot_heatmap_preview(heatmap_df, heatmap_pdf, heatmap_png)
    plot_model_subitem_heatmap_preview(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        title="Tail model-subitem heatmap",
        pdf_path=model_subitem_heatmap_pdf,
        png_path=model_subitem_heatmap_png,
    )
    plot_model_subitem_grouped_bar_preview(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        model_color_map=MODEL_COLORS,
        title="Tail family and subitem bars",
        y_label="Score",
        pdf_path=grouped_bars_pdf,
        png_path=grouped_bars_png,
    )

    _write_model_table_tex(model_summary_df, TABLE_DIR / "tail_model_summary_generated.tex")
    _build_report(resolved_source_run_dir, dataset_model_df, model_summary_df, prefix_summary_df, dataset_summary_df)
    _build_readme(resolved_source_run_dir)

    compile_notes = {
        "tradeoff": _try_compile_tex(tradeoff_tex),
        "bridge": _try_compile_tex(bridge_tex),
        "prefix_bars": _try_compile_tex(prefix_tex),
        "heatmap": _try_compile_tex(heatmap_tex),
        "model_subitem_heatmap": _try_compile_tex(model_subitem_heatmap_tex),
        "family_subitem_bars": _try_compile_tex(grouped_bars_tex),
    }

    final_files = [
        tradeoff_tex,
        tradeoff_pdf,
        tradeoff_png,
        bridge_tex,
        bridge_pdf,
        bridge_png,
        prefix_tex,
        prefix_pdf,
        prefix_png,
        heatmap_tex,
        heatmap_pdf,
        heatmap_png,
        model_subitem_heatmap_tex,
        model_subitem_heatmap_pdf,
        model_subitem_heatmap_png,
        grouped_bars_tex,
        grouped_bars_pdf,
        grouped_bars_png,
        TABLE_DIR / "tail_model_summary_generated.tex",
        DATA_DIR / "model_summary.csv",
        DATA_DIR / "prefix_summary.csv",
        OUTPUT_ROOT / "analysis_report.md",
    ]
    must_do_aliases = {
        "tail_tradeoff_scatter_main.tex": tradeoff_tex,
        "tail_tradeoff_scatter_main.pdf": tradeoff_pdf,
        "tail_tradeoff_scatter_main.png": tradeoff_png,
        "tail_prefix_bars_appendix.tex": prefix_tex,
        "tail_prefix_bars_appendix.pdf": prefix_pdf,
        "tail_prefix_bars_appendix.png": prefix_png,
        "tail_dataset_model_heatmap_appendix.tex": heatmap_tex,
        "tail_dataset_model_heatmap_appendix.pdf": heatmap_pdf,
        "tail_dataset_model_heatmap_appendix.png": heatmap_png,
        "tail_model_subitem_heatmap_appendix.tex": model_subitem_heatmap_tex,
        "tail_model_subitem_heatmap_appendix.pdf": model_subitem_heatmap_pdf,
        "tail_model_subitem_heatmap_appendix.png": model_subitem_heatmap_png,
        "tail_family_subitem_bars_appendix.tex": grouped_bars_tex,
        "tail_family_subitem_bars_appendix.pdf": grouped_bars_pdf,
        "tail_family_subitem_bars_appendix.png": grouped_bars_png,
    }
    version_tag = OUTPUT_VERSION_TAG
    _copy_final_artifacts(final_files, must_do_aliases, version_tag=version_tag)

    final_readme = render_final_readme(
        title="Tail Breakdown Final",
        summary=f"This directory contains the paper-facing tail breakdown artifacts published under `{sql_source_label(version_tag)}` (`{version_tag}`), with the standardized must-do bundle mirrored into `final/must_do/` and `final/{version_tag}/must_do/`.",
        primary_files=[
            *[versioned_name(name, version_tag) for name in [
                "tail_tradeoff_scatter_main.tex",
                "tail_tradeoff_scatter_main.pdf",
                "tail_tradeoff_scatter_main.png",
                "tail_coverage_vs_breakdown_bridge.tex",
                "tail_coverage_vs_breakdown_bridge.pdf",
                "tail_coverage_vs_breakdown_bridge.png",
                "tail_prefix_bars_appendix.tex",
                "tail_prefix_bars_appendix.pdf",
                "tail_prefix_bars_appendix.png",
                "tail_dataset_model_heatmap_appendix.tex",
                "tail_dataset_model_heatmap_appendix.pdf",
                "tail_dataset_model_heatmap_appendix.png",
                "tail_model_subitem_heatmap_appendix.tex",
                "tail_model_subitem_heatmap_appendix.pdf",
                "tail_model_subitem_heatmap_appendix.png",
                "tail_family_subitem_bars_appendix.tex",
                "tail_family_subitem_bars_appendix.pdf",
                "tail_family_subitem_bars_appendix.png",
                "tail_model_summary_generated.tex",
                "model_summary.csv",
            ]],
        ],
        must_do_files=[versioned_name(name, version_tag) for name in must_do_aliases.keys()],
        support_files=[
            *[versioned_name(name, version_tag) for name in [
                "tail_coverage_vs_concentration_scatter_main.tex",
                "tail_coverage_vs_concentration_scatter_main.pdf",
                "tail_coverage_vs_concentration_scatter_main.png",
                "analysis_report.md",
                "prefix_summary.csv",
            ]],
        ],
        notes=[
            f"The active published version tag for this bundle is `{sql_source_label(version_tag)}` (`{version_tag}`).",
            "The `.tex` files are standalone TikZ sources. The `.pdf/.png` files are immediate previews for reading in the current environment.",
        ],
    )
    (FINAL_DIR / "README.md").write_text(final_readme, encoding="utf-8")
    manifest = {
        "task": "tail_breakdown",
        "sql_source_version": version_tag,
        "sql_source_label": sql_source_label(version_tag),
        "source_tail_threshold_run": resolved_source_run_dir.name,
        "excluded_models": sorted(EXCLUDED_MODELS),
        "included_models": model_summary_df["model_id"].tolist(),
        "dataset_panel_count": int(dataset_model_df.shape[0]),
        "threshold_panel_count": int(dataset_model_threshold_df.shape[0]),
        "proxy_asset_row_count": int(proxy_df.shape[0]),
        "compile_notes": {key: {"success": value[0], "note": value[1]} for key, value in compile_notes.items()},
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical tail breakdown bundle from a tail-threshold run.")
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional explicit Evaluation/tail_threshold run directory to use as the source.",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers for proxy computations.")
    parser.add_argument(
        "--proxy-max-rows",
        type=int,
        default=DEFAULT_PROXY_MAX_ROWS,
        help="Row cap used by the concentration proxy for very large tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_tail_breakdown(
        source_run_dir=args.source_run_dir,
        max_workers=max(1, args.max_workers),
        proxy_max_rows=max(1, args.proxy_max_rows),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
