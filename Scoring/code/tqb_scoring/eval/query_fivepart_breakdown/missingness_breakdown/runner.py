#!/usr/bin/env python3
"""Build a standardized missingness breakdown bundle.

This runner follows the same four-core-figure contract as the other
query five-part breakdown tasks:

1. `missingness_tradeoff_scatter_main`
2. `missingness_prefix_bars_appendix`
3. `missingness_dataset_model_heatmap_appendix`
4. `missingness_model_subitem_heatmap_appendix`

If the current unified analysis run still has no missingness query rows,
the runner emits explicit placeholder artifacts instead of silently
leaving the slot empty.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.analytics_contract import annotate_query_row_with_contract
from tqb_scoring.eval.query_fivepart_breakdown.common_final import render_final_readme, sync_final_outputs
from tqb_scoring.eval.common import (
    DEFAULT_SQL_SOURCE_VERSION,
    resolve_requested_sql_source_version,
    resolve_task_run_dir_for_sql_source,
    sql_source_label,
)
from tqb_scoring.eval.query_fivepart_breakdown.common_final import versioned_name
from tqb_scoring.eval.query_fivepart_breakdown.common_heatmap_palette import (
    format_heatmap_latex_cell,
    get_heatmap_cmap,
)
from tqb_scoring.eval.query_fivepart_breakdown.common_model_subitem_grouped_bars import (
    plot_model_subitem_grouped_bar_preview,
    write_model_subitem_grouped_bar_tex,
)
from tqb_scoring.eval.query_fivepart_breakdown.common_model_subitem_heatmap import (
    build_model_subitem_heatmap_df,
    plot_model_subitem_heatmap_preview,
    write_model_subitem_heatmap_tex,
)


EVALUATION_ROOT = PROJECT_ROOT.parent / "results"
ANALYSIS_ROOT = EVALUATION_ROOT / "analysis"
OUTPUT_ROOT = EVALUATION_ROOT / "query_fivepart_breakdown" / "missingness_breakdown"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"
FINAL_DIR = OUTPUT_ROOT / "final"
OUTPUT_VERSION_TAG = resolve_requested_sql_source_version("analysis", DEFAULT_SQL_SOURCE_VERSION)

TARGET_FAMILY = "missingness_structure"
REAL_MODEL_ID = "real"
SUBITEM_ORDER = [
    "marginal_missing_rate_consistency",
    "co_missingness_pattern_consistency",
]
EXTRA_INSIGHT_METRICS = [
    "co_missing_strength_score",
    "co_missing_composite_score",
]
SUBITEM_LABELS = {
    "marginal_missing_rate_consistency": "Marginal missing-rate consistency",
    "co_missingness_pattern_consistency": "Co-missingness pattern consistency",
}
MODEL_LABELS = {
    "real": "REAL",
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
    "real": "#000000",
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
EXCLUDED_MODELS = {"cdtd", "codi", "goggle"}
MODEL_ALIASES = {"rtf": "realtabformer"}
SERVER_PRIORITY = {"rtx_5090": 2, "rtx_pro_6000": 1}
ROOT_PRIORITY = {"SynOutput-5090": 2, "SynOutput": 1}


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, DATA_DIR, FIG_DIR, FINAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _normalize_model(model_id: Any) -> str:
    key = str(model_id or "").strip().lower()
    return MODEL_ALIASES.get(key, key)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _model_sort_key(model_id: str) -> tuple[int, str]:
    if str(model_id).strip().lower() == REAL_MODEL_ID:
        return (0, _model_label(model_id))
    return (1, _model_label(model_id).lower())


def _sorted_model_ids(model_ids: list[str] | set[str]) -> list[str]:
    return sorted({str(item) for item in model_ids}, key=_model_sort_key)


def _dataset_prefix(dataset_id: str) -> str:
    return str(dataset_id or "").strip().lower()[:1]


def _dataset_sort_key(dataset_id: str) -> tuple[int, int, str]:
    text = str(dataset_id or "").strip()
    if len(text) < 2 or not text[1:].isdigit():
        return (99, 10**9, text)
    prefix_order = {"c": 0, "m": 1, "n": 2}.get(text[0].lower(), 50)
    return (prefix_order, int(text[1:]), text)


def _asset_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    server = str(row.get("server_type") or "").strip().lower()
    root_name = str(row.get("root_name") or "").strip()
    run_id = str(row.get("run_id") or "").strip()
    asset_key = str(row.get("asset_key") or "").strip()
    return (
        SERVER_PRIORITY.get(server, 0),
        ROOT_PRIORITY.get(root_name, 0),
        run_id,
        asset_key,
    )


def _find_primary_analysis_run() -> Path:
    resolved = resolve_task_run_dir_for_sql_source("analysis", OUTPUT_VERSION_TAG)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(f"No analysis run found for sql_source_version={OUTPUT_VERSION_TAG!r}.")


def _run_direct_missingness_eval() -> dict[str, Path]:
    from tests.comissing_condition_eval import evaluate_all_synthetic_assets

    direct_root = DATA_DIR / "_direct_eval"
    expected = {
        "dataset_context": direct_root / "co_missing_dataset_context.csv",
        "asset_scores": direct_root / "co_missing_asset_scores.csv",
        "target_scores": direct_root / "co_missing_target_scores.csv",
        "model_dataset_summary": direct_root / "co_missing_model_dataset_summary.csv",
        "model_overall_summary": direct_root / "co_missing_model_overall_summary.csv",
    }
    if all(path.exists() for path in expected.values()):
        try:
            asset_df = pd.read_csv(expected["asset_scores"], encoding="utf-8-sig", nrows=5)
            model_dataset_df = pd.read_csv(expected["model_dataset_summary"], encoding="utf-8-sig", nrows=5)
            required_columns = {"co_missing_strength_score", "co_missing_composite_score"}
            if required_columns.issubset(set(asset_df.columns)) and required_columns.issubset(set(model_dataset_df.columns)):
                return expected
        except Exception:
            pass
    return evaluate_all_synthetic_assets(direct_root)


def _load_primary_assets(asset_csv: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    with asset_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset_id = str(row.get("dataset_id") or "").strip()
        model_id = _normalize_model(row.get("model_id"))
        if not dataset_id or not model_id or model_id in EXCLUDED_MODELS:
            continue
        row["model_id"] = model_id
        row["model_label"] = _model_label(model_id)
        grouped[(dataset_id, model_id)].append(row)

    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for key, items in grouped.items():
        ranked = sorted(items, key=_asset_sort_key, reverse=True)
        chosen[key] = ranked[0]
        for dropped in ranked[1:]:
            audit_rows.append(
                {
                    "dataset_id": key[0],
                    "model_id": key[1],
                    "kept_asset_key": ranked[0].get("asset_key"),
                    "dropped_asset_key": dropped.get("asset_key"),
                    "kept_run_id": ranked[0].get("run_id"),
                    "dropped_run_id": dropped.get("run_id"),
                }
            )
    return chosen, audit_rows


def _stream_missingness_query_rows(
    query_jsonl: Path,
    primary_assets: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    chosen_keys = {
        (dataset_id, model_id): str(row.get("asset_key") or "")
        for (dataset_id, model_id), row in primary_assets.items()
    }
    out: list[dict[str, Any]] = []
    with query_jsonl.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            dataset_id = str(row.get("dataset_id") or "").strip()
            model_id = _normalize_model(row.get("model_id"))
            if not dataset_id or not model_id or model_id in EXCLUDED_MODELS:
                continue
            if chosen_keys.get((dataset_id, model_id)) != str(row.get("asset_key") or ""):
                continue
            if str(row.get("family_id") or "").strip().lower() != TARGET_FAMILY:
                continue
            annotated = dict(row)
            if not annotated.get("canonical_subitem_id"):
                annotated = annotate_query_row_with_contract(annotated)
            subitem_id = str(annotated.get("canonical_subitem_id") or "").strip()
            if subitem_id not in SUBITEM_ORDER:
                continue
            try:
                score_value = float(annotated.get("query_score"))
            except Exception:
                continue
            out.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_prefix": _dataset_prefix(dataset_id),
                    "model_id": model_id,
                    "model_label": _model_label(model_id),
                    "asset_key": str(annotated.get("asset_key") or ""),
                    "subitem_id": subitem_id,
                    "subitem_label": SUBITEM_LABELS[subitem_id],
                    "query_id": str(annotated.get("query_id") or ""),
                    "query_score": score_value,
                    "template_id": str(annotated.get("template_id") or ""),
                    "template_name": str(annotated.get("template_name") or ""),
                    "question": str(annotated.get("question") or ""),
                    "sql_engine": str(annotated.get("sql_engine") or ""),
                }
            )
    return out


def _inject_real_rows(query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset_query: dict[tuple[str, str], dict[str, Any]] = {}
    for row in query_rows:
        key = (str(row["dataset_id"]), str(row["query_id"]))
        if key not in by_dataset_query:
            by_dataset_query[key] = row

    real_rows: list[dict[str, Any]] = []
    for (dataset_id, _query_id), row in sorted(by_dataset_query.items(), key=lambda item: (_dataset_sort_key(item[0][0]), item[0][1])):
        real_rows.append(
            {
                **row,
                "model_id": REAL_MODEL_ID,
                "model_label": _model_label(REAL_MODEL_ID),
                "asset_key": f"{dataset_id}__real_reference",
                "query_score": 1.0,
                "sql_engine": "real-reference",
            }
        )
    return query_rows + real_rows


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


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


def _build_dataset_model_scores(query_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subitems = (
        query_df.groupby(
            ["dataset_id", "dataset_prefix", "model_id", "model_label", "subitem_id", "subitem_label"],
            as_index=False,
        )
        .agg(query_count=("query_id", "count"), subitem_score=("query_score", "mean"))
        .reset_index(drop=True)
    )
    pivot = (
        subitems.pivot_table(
            index=["dataset_id", "dataset_prefix", "model_id", "model_label"],
            columns="subitem_id",
            values="subitem_score",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    counts = (
        subitems.pivot_table(
            index=["dataset_id", "dataset_prefix", "model_id", "model_label"],
            columns="subitem_id",
            values="query_count",
            aggfunc="sum",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide = pivot.merge(counts, on=["dataset_id", "dataset_prefix", "model_id", "model_label"], suffixes=("", "__query_count"))
    for metric in SUBITEM_ORDER:
        wide[metric] = pd.to_numeric(wide[metric], errors="coerce")
    wide["missingness_structure_score"] = wide[SUBITEM_ORDER].mean(axis=1, skipna=True)
    wide["marginal_minus_comissing"] = wide["marginal_missing_rate_consistency"] - wide["co_missingness_pattern_consistency"]
    wide["active_subitem_count"] = wide[SUBITEM_ORDER].notna().sum(axis=1)
    wide["dataset_sort"] = wide["dataset_id"].map(_dataset_sort_key)
    wide["model_sort"] = wide["model_id"].map(_model_sort_key)
    wide = wide.sort_values(["dataset_sort", "model_sort"]).drop(columns=["dataset_sort", "model_sort"]).reset_index(drop=True)
    return subitems, wide


def _build_dataset_model_scores_from_direct_summary(model_dataset_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_dataset_df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "dataset_id",
                    "dataset_prefix",
                    "model_id",
                    "model_label",
                    "subitem_id",
                    "subitem_label",
                    "query_count",
                    "subitem_score",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "dataset_id",
                    "dataset_prefix",
                    "model_id",
                    "model_label",
                    *SUBITEM_ORDER,
                    *EXTRA_INSIGHT_METRICS,
                    "missingness_structure_score",
                    "marginal_minus_comissing",
                    "profile_minus_strength",
                    "active_subitem_count",
                    "asset_count",
                    "applicable_asset_count",
                ]
            ),
        )

    df = model_dataset_df.copy()
    df["model_id"] = df["model_id"].map(_normalize_model)
    df = df.loc[~df["model_id"].isin(EXCLUDED_MODELS)].copy()
    df["model_label"] = df["model_id"].map(_model_label)
    df["dataset_prefix"] = df["dataset_id"].map(_dataset_prefix)
    for metric in SUBITEM_ORDER + EXTRA_INSIGHT_METRICS + ["missingness_structure_score"]:
        if metric not in df.columns:
            df[metric] = pd.NA
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["marginal_minus_comissing"] = df["marginal_missing_rate_consistency"] - df["co_missingness_pattern_consistency"]
    df["profile_minus_strength"] = df["co_missingness_pattern_consistency"] - df["co_missing_strength_score"]
    df["active_subitem_count"] = df[SUBITEM_ORDER].notna().sum(axis=1)
    df["dataset_sort"] = df["dataset_id"].map(_dataset_sort_key)
    df["model_sort"] = df["model_id"].map(_model_sort_key)
    wide = (
        df.sort_values(["dataset_sort", "model_sort"])[
            [
                "dataset_id",
                "dataset_prefix",
                "model_id",
                "model_label",
                *SUBITEM_ORDER,
                *EXTRA_INSIGHT_METRICS,
                "missingness_structure_score",
                "marginal_minus_comissing",
                "profile_minus_strength",
                "active_subitem_count",
                "asset_count",
                "applicable_asset_count",
            ]
        ]
        .reset_index(drop=True)
    )

    subitem_rows: list[dict[str, Any]] = []
    for row in wide.itertuples(index=False):
        applicable_count = int(getattr(row, "applicable_asset_count", 0) or 0)
        for subitem_id in SUBITEM_ORDER:
            subitem_rows.append(
                {
                    "dataset_id": row.dataset_id,
                    "dataset_prefix": row.dataset_prefix,
                    "model_id": row.model_id,
                    "model_label": row.model_label,
                    "subitem_id": subitem_id,
                    "subitem_label": SUBITEM_LABELS[subitem_id],
                    "query_count": applicable_count,
                    "subitem_score": getattr(row, subitem_id, None),
                }
            )
    return pd.DataFrame(subitem_rows), wide


def _apply_no_missing_full_score(
    subitem_df: pd.DataFrame,
    dataset_model_df: pd.DataFrame,
    dataset_context_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_model_df.empty or dataset_context_df.empty:
        return subitem_df, dataset_model_df

    missing_target_counts = pd.to_numeric(dataset_context_df.get("missing_target_count"), errors="coerce").fillna(0)
    no_missing_datasets = set(dataset_context_df.loc[missing_target_counts == 0, "dataset_id"].astype(str))
    if not no_missing_datasets:
        return subitem_df, dataset_model_df

    dataset_model_df = dataset_model_df.copy()
    mask = dataset_model_df["dataset_id"].astype(str).isin(no_missing_datasets)
    if not mask.any():
        return subitem_df, dataset_model_df

    for metric in SUBITEM_ORDER:
        dataset_model_df.loc[mask, metric] = 1.0
    dataset_model_df.loc[mask, "missingness_structure_score"] = 1.0
    dataset_model_df.loc[mask, "marginal_minus_comissing"] = 0.0
    dataset_model_df.loc[mask, "profile_minus_strength"] = 0.0
    dataset_model_df.loc[mask, "active_subitem_count"] = len(SUBITEM_ORDER)

    subitem_df = subitem_df.copy()
    if not subitem_df.empty:
        for metric in SUBITEM_ORDER:
            row_mask = subitem_df["dataset_id"].astype(str).isin(no_missing_datasets) & (subitem_df["subitem_id"] == metric)
            subitem_df.loc[row_mask, "subitem_score"] = 1.0
            subitem_df.loc[row_mask, "query_count"] = 0

    existing = set()
    if not subitem_df.empty:
        existing = set(map(tuple, subitem_df[["dataset_id", "model_id", "subitem_id"]].astype(str).to_records(index=False)))
    add_rows: list[dict[str, Any]] = []
    for row in dataset_model_df.loc[mask, ["dataset_id", "dataset_prefix", "model_id", "model_label"]].itertuples(index=False):
        for subitem_id in SUBITEM_ORDER:
            key = (str(row.dataset_id), str(row.model_id), str(subitem_id))
            if key not in existing:
                add_rows.append(
                    {
                        "dataset_id": row.dataset_id,
                        "dataset_prefix": row.dataset_prefix,
                        "model_id": row.model_id,
                        "model_label": row.model_label,
                        "subitem_id": subitem_id,
                        "subitem_label": SUBITEM_LABELS[subitem_id],
                        "query_count": 0,
                        "subitem_score": 1.0,
                    }
                )
    if add_rows:
        subitem_df = pd.concat([subitem_df, pd.DataFrame(add_rows)], ignore_index=True)

    dataset_model_df["dataset_sort"] = dataset_model_df["dataset_id"].map(_dataset_sort_key)
    dataset_model_df["model_sort"] = dataset_model_df["model_id"].map(_model_sort_key)
    dataset_model_df = dataset_model_df.sort_values(["dataset_sort", "model_sort"]).drop(columns=["dataset_sort", "model_sort"]).reset_index(drop=True)
    if not subitem_df.empty:
        subitem_df["dataset_sort"] = subitem_df["dataset_id"].map(_dataset_sort_key)
        subitem_df["model_sort"] = subitem_df["model_id"].map(_model_sort_key)
        subitem_df = subitem_df.sort_values(["dataset_sort", "model_sort", "subitem_id"]).drop(columns=["dataset_sort", "model_sort"]).reset_index(drop=True)
    return subitem_df, dataset_model_df


def _build_model_summary(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = SUBITEM_ORDER + EXTRA_INSIGHT_METRICS + ["missingness_structure_score", "marginal_minus_comissing", "profile_minus_strength"]
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
    return summary.sort_values(["model_sort", "model_id"]).drop(columns=["model_sort"]).reset_index(drop=True)


def _build_prefix_summary(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_id, prefix), group in dataset_model_df.groupby(["model_id", "dataset_prefix"], sort=False):
        rows.append(
            {
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "dataset_prefix": prefix,
                "dataset_count": int(group["dataset_id"].nunique()),
                "marginal_missing_rate_consistency": round(float(pd.to_numeric(group["marginal_missing_rate_consistency"], errors="coerce").dropna().mean()), 6),
                "co_missingness_pattern_consistency": round(float(pd.to_numeric(group["co_missingness_pattern_consistency"], errors="coerce").dropna().mean()), 6),
                "co_missing_strength_score": round(float(pd.to_numeric(group["co_missing_strength_score"], errors="coerce").dropna().mean()), 6),
                "missingness_structure_score": round(float(pd.to_numeric(group["missingness_structure_score"], errors="coerce").dropna().mean()), 6),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["model_sort"] = summary["model_id"].map(_model_sort_key)
    return summary.sort_values(["dataset_prefix", "model_sort"]).drop(columns=["model_sort"]).reset_index(drop=True)


def _build_dataset_summary(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, group in dataset_model_df.groupby("dataset_id", sort=False):
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": _dataset_prefix(dataset_id),
                "model_count": int(group["model_id"].nunique()),
                "missingness_structure_score": round(float(pd.to_numeric(group["missingness_structure_score"], errors="coerce").dropna().mean()), 6),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["dataset_sort"] = summary["dataset_id"].map(_dataset_sort_key)
    return summary.sort_values(["dataset_sort", "dataset_id"]).drop(columns=["dataset_sort"]).reset_index(drop=True)


def _build_heatmap_data(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    heatmap = (
        dataset_model_df.pivot_table(
            index="dataset_id",
            columns="model_id",
            values="missingness_structure_score",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if heatmap.empty:
        return heatmap
    ordered_models = [item for item in _sorted_model_ids(dataset_model_df["model_id"].tolist()) if item in heatmap.columns]
    heatmap["dataset_sort"] = heatmap["dataset_id"].map(_dataset_sort_key)
    heatmap = heatmap.sort_values(["dataset_sort", "dataset_id"]).drop(columns=["dataset_sort"]).reset_index(drop=True)
    return heatmap[["dataset_id"] + ordered_models]


def _build_prefix_plot_data(prefix_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in _sorted_model_ids(prefix_summary_df["model_id"].tolist()):
        payload: dict[str, Any] = {"model_id": model_id, "model_label": _model_label(model_id)}
        subset = prefix_summary_df.loc[prefix_summary_df["model_id"] == model_id]
        for prefix in ["c", "m", "n"]:
            match = subset.loc[subset["dataset_prefix"] == prefix, "missingness_structure_score"]
            payload[prefix] = float(match.iloc[0]) if not match.empty else None
        rows.append(payload)
    return pd.DataFrame(rows)


def _write_tradeoff_tex(model_summary_df: pd.DataFrame, path: Path) -> None:
    color_defs = [
        rf"\definecolor{{model{row.model_id}}}{{HTML}}{{{MODEL_COLORS[row.model_id].replace('#', '')}}}"
        for row in model_summary_df.itertuples()
        if row.model_id in MODEL_COLORS
    ]
    lines = [
        _tex_preamble(),
        *color_defs,
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"""\begin{axis}[
width=11.2cm,
height=8.4cm,
xmin=0, xmax=1.02,
ymin=0, ymax=1.02,
xlabel={Marginal missing-rate consistency},
ylabel={Co-missingness pattern consistency},
grid=major,
grid style={gray!20},
]""",
    ]
    for row in model_summary_df.itertuples():
        x = getattr(row, "marginal_missing_rate_consistency__mean")
        y = getattr(row, "co_missingness_pattern_consistency__mean")
        if x is None or y is None:
            continue
        color_name = f"model{row.model_id}"
        lines.append(
            rf"\addplot[only marks, mark=*, mark size=2.2pt, {color_name}] coordinates {{({float(x):.6f},{float(y):.6f})}};"
        )
        lines.append(
            rf"\node[anchor=west, font=\scriptsize, text={color_name}] at (axis cs:{float(x):.6f},{float(y):.6f}) {{{_escape_tex(row.model_label)}}};"
        )
    lines.extend([r"\end{axis}", r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_prefix_bar_tex(prefix_plot_df: pd.DataFrame, path: Path) -> None:
    model_labels = [_escape_tex(str(item)) for item in prefix_plot_df["model_label"].tolist()]
    xticklabels = ",".join(model_labels)
    color_defs = [
        rf"\definecolor{{model{row.model_id}}}{{HTML}}{{{MODEL_COLORS[row.model_id].replace('#', '')}}}"
        for row in prefix_plot_df.itertuples()
        if row.model_id in MODEL_COLORS
    ]
    lines = [
        _tex_preamble(),
        *color_defs,
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"""\begin{groupplot}[
group style={group size=3 by 1, horizontal sep=1.0cm},
width=0.31\textwidth,
height=0.46\textwidth,
ymin=0, ymax=1.02,
xtick={1,...,%d},
xticklabels={%s},
x tick label style={rotate=60, anchor=east, font=\scriptsize},
grid=major,
grid style={gray!20},
]"""
        % (len(model_labels), xticklabels),
    ]
    for prefix in ["c", "m", "n"]:
        lines.append(rf"\nextgroupplot[title={{{prefix.upper()} datasets}}, ylabel={{Mean score}}]")
        for idx, row in enumerate(prefix_plot_df.itertuples(), start=1):
            value = getattr(row, prefix, None)
            if value is None or pd.isna(value):
                continue
            color_name = f"model{row.model_id}" if row.model_id in MODEL_COLORS else "gray"
            lines.append(rf"\addplot[ybar, draw={color_name}, fill={color_name}] coordinates {{({idx},{float(value):.6f})}};")
    lines.extend([r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_heatmap_tex(heatmap_df: pd.DataFrame, path: Path) -> None:
    ordered = heatmap_df.copy()
    model_cols = [column for column in ordered.columns if column != "dataset_id"]
    display = ordered[["dataset_id"] + model_cols].copy().fillna("")
    lines = [
        r"\documentclass{standalone}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{xcolor}",
        r"\usepackage{booktabs}",
        r"\begin{document}",
        r"\scriptsize",
        r"\textbf{Missingness dataset-model heatmap}\\[0.4em]",
        r"\emph{Score, 0--1; missing cells stay white.}\\[0.5em]",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{l{'c' * len(model_cols)}}}",
        r"\toprule",
        "Dataset & " + " & ".join(_escape_tex(column) for column in model_cols) + r" \\",
        r"\midrule",
    ]
    for row in display.itertuples(index=False):
        cells = [_escape_tex(str(getattr(row, "dataset_id")))]
        for model in model_cols:
            cells.append(format_heatmap_latex_cell(getattr(row, model)))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_tradeoff_preview(model_summary_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for row in model_summary_df.itertuples():
        x = getattr(row, "marginal_missing_rate_consistency__mean")
        y = getattr(row, "co_missingness_pattern_consistency__mean")
        if x is None or y is None:
            continue
        color = MODEL_COLORS.get(str(row.model_id), "#777777")
        ax.scatter(float(x), float(y), s=56, color=color)
        ax.text(float(x) + 0.012, float(y) + 0.008, str(row.model_label), fontsize=8, color=color)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Marginal missing-rate consistency")
    ax.set_ylabel("Co-missingness pattern consistency")
    ax.set_title("Missingness trade-off scatter")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_prefix_bar_preview(prefix_plot_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.2), sharey=True)
    for ax, prefix in zip(axes, ["c", "m", "n"]):
        values = pd.to_numeric(prefix_plot_df[prefix], errors="coerce")
        colors = [MODEL_COLORS.get(model_id, "#777777") for model_id in prefix_plot_df["model_id"]]
        ax.bar(range(len(prefix_plot_df)), values, color=colors)
        ax.set_title(f"{prefix.upper()} datasets")
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(range(len(prefix_plot_df)))
        ax.set_xticklabels(prefix_plot_df["model_label"], rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean score")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_preview(heatmap_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    ordered = heatmap_df.copy()
    model_cols = [column for column in ordered.columns if column != "dataset_id"]
    matrix = ordered[model_cols].to_numpy(dtype=float)
    fig_height = max(7.2, 0.22 * len(ordered) + 1.8)
    fig, ax = plt.subplots(figsize=(10.4, fig_height))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap=get_heatmap_cmap())
    ax.set_xticks(range(len(model_cols)))
    ax.set_xticklabels(model_cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered["dataset_id"], fontsize=7)
    ax.set_title("Missingness dataset-model heatmap")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_placeholder_tex(path: Path, title: str, message: str) -> None:
    content = "\n".join(
        [
            _tex_preamble(),
            r"\begin{document}",
            r"\begin{tikzpicture}",
            r"\node[draw=gray!60, rounded corners=4pt, fill=gray!8, text width=11.5cm, align=center, inner sep=12pt] {",
            rf"\textbf{{{_escape_tex(title)}}}\\[0.7em]{_escape_tex(message)}",
            r"};",
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _plot_placeholder_preview(pdf_path: Path, png_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.38,
        message,
        ha="center",
        va="center",
        fontsize=11,
        wrap=True,
    )
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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


def _write_placeholder_bundle(run_dir: Path, audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    message = (
        "No `missingness_structure` query rows were found in the current unified analysis run. "
        "These standardized placeholders keep the five-part appendix structure stable until missingness queries are emitted."
    )
    placeholder_specs = [
        ("missingness_tradeoff_scatter_main", "Missingness Trade-Off Placeholder"),
        ("missingness_prefix_bars_appendix", "Missingness Prefix Placeholder"),
        ("missingness_dataset_model_heatmap_appendix", "Missingness Heatmap Placeholder"),
        ("missingness_model_subitem_heatmap_appendix", "Missingness Model-Subitem Placeholder"),
        ("missingness_family_subitem_bars_appendix", "Missingness Family/Subitem Bars Placeholder"),
    ]
    final_files: list[Path] = [DATA_DIR / "duplicate_asset_audit.csv", DATA_DIR / "missingness_query_rows.csv", OUTPUT_ROOT / "analysis_report.md"]
    must_do_aliases: dict[str, Path] = {}
    compile_notes: dict[str, tuple[bool, str]] = {}
    for stem, title in placeholder_specs:
        tex_path = FIG_DIR / f"{stem}.tex"
        pdf_path = FIG_DIR / f"{stem}.pdf"
        png_path = FIG_DIR / f"{stem}.png"
        _write_placeholder_tex(tex_path, title, message)
        _plot_placeholder_preview(pdf_path, png_path, title, message)
        compile_notes[stem] = _try_compile_tex(tex_path)
        final_files.extend([tex_path, pdf_path, png_path])
        must_do_aliases[f"{stem}.tex"] = tex_path
        must_do_aliases[f"{stem}.pdf"] = pdf_path
        must_do_aliases[f"{stem}.png"] = png_path

    _write_csv(pd.DataFrame(audit_rows), DATA_DIR / "duplicate_asset_audit.csv")
    _write_csv(pd.DataFrame(columns=["dataset_id", "model_id", "subitem_id", "query_id", "query_score"]), DATA_DIR / "missingness_query_rows.csv")
    (OUTPUT_ROOT / "analysis_report.md").write_text(
        "\n".join(
            [
                "# Missingness Breakdown",
                "",
                f"- Source analysis run: `{run_dir.name}`",
                "- Current status: no missingness query rows available in the unified analysis export.",
                "- Action taken: emitted standardized placeholder must-do figures so the five-part appendix structure stays complete.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    version_tag = OUTPUT_VERSION_TAG
    readme = render_final_readme(
        title="Missingness Breakdown Final",
        summary=f"This directory contains the standardized missingness slot for `{sql_source_label(version_tag)}` (`{version_tag}`). The current repository run has no missingness query rows yet, so `must_do/` contains explicit placeholders rather than silent gaps.",
        primary_files=[versioned_name(name, version_tag) for name in [
            "missingness_tradeoff_scatter_main.tex",
            "missingness_tradeoff_scatter_main.pdf",
            "missingness_tradeoff_scatter_main.png",
            "missingness_prefix_bars_appendix.tex",
            "missingness_prefix_bars_appendix.pdf",
            "missingness_prefix_bars_appendix.png",
            "missingness_dataset_model_heatmap_appendix.tex",
            "missingness_dataset_model_heatmap_appendix.pdf",
            "missingness_dataset_model_heatmap_appendix.png",
            "missingness_model_subitem_heatmap_appendix.tex",
            "missingness_model_subitem_heatmap_appendix.pdf",
            "missingness_model_subitem_heatmap_appendix.png",
            "missingness_family_subitem_bars_appendix.tex",
            "missingness_family_subitem_bars_appendix.pdf",
            "missingness_family_subitem_bars_appendix.png",
        ]],
        must_do_files=[versioned_name(name, version_tag) for name in must_do_aliases.keys()],
        support_files=[
            *[versioned_name(name, version_tag) for name in [
                "analysis_report.md",
                "missingness_query_rows.csv",
                "duplicate_asset_audit.csv",
            ]],
        ],
        notes=[
            "These placeholders are intentional and should be replaced automatically once missingness queries are present in the unified analysis run.",
        ],
    )
    sync_final_outputs(FINAL_DIR, final_files, must_do_aliases, version_tag=version_tag, copy_plain_files=False)
    (FINAL_DIR / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "task": "missingness_breakdown",
        "sql_source_version": version_tag,
        "sql_source_label": sql_source_label(version_tag),
        "source_analysis_run": run_dir.name,
        "query_row_count": 0,
        "status": "placeholder_no_query_rows",
        "compile_notes": {key: {"success": value[0], "note": value[1]} for key, value in compile_notes.items()},
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_missingness_breakdown(*, analysis_run_dir: Path | None = None) -> dict[str, Any]:
    _ensure_dirs()
    analysis_run_dir = analysis_run_dir.resolve() if analysis_run_dir is not None else _find_primary_analysis_run()
    audit_rows: list[dict[str, Any]] = []
    direct_outputs = _run_direct_missingness_eval()
    direct_dataset_context_df = pd.read_csv(direct_outputs["dataset_context"], encoding="utf-8-sig")
    direct_asset_scores_df = pd.read_csv(direct_outputs["asset_scores"], encoding="utf-8-sig")
    direct_target_scores_df = pd.read_csv(direct_outputs["target_scores"], encoding="utf-8-sig")
    direct_model_dataset_df = pd.read_csv(direct_outputs["model_dataset_summary"], encoding="utf-8-sig")
    direct_model_overall_df = pd.read_csv(direct_outputs["model_overall_summary"], encoding="utf-8-sig")

    subitem_df, dataset_model_df = _build_dataset_model_scores_from_direct_summary(direct_model_dataset_df)
    subitem_df, dataset_model_df = _apply_no_missing_full_score(
        subitem_df,
        dataset_model_df,
        direct_dataset_context_df,
    )
    if dataset_model_df.empty:
        return _write_placeholder_bundle(analysis_run_dir, audit_rows)

    model_summary_df = _build_model_summary(dataset_model_df)
    prefix_summary_df = _build_prefix_summary(dataset_model_df)
    dataset_summary_df = _build_dataset_summary(dataset_model_df)
    heatmap_df = _build_heatmap_data(dataset_model_df)
    prefix_plot_df = _build_prefix_plot_data(prefix_summary_df)
    model_subitem_heatmap_df = build_model_subitem_heatmap_df(
        model_summary_df.loc[model_summary_df["model_id"].isin(MODEL_ORDER)].copy(),
        model_id_col="model_id",
        model_order=MODEL_ORDER,
        subitem_specs=[
            (subitem_id, SUBITEM_LABELS[subitem_id], f"{subitem_id}__mean")
            for subitem_id in SUBITEM_ORDER
        ],
        summary_row_spec=("family_mean", "Family mean", "missingness_structure_score__mean"),
    )

    _write_csv(pd.DataFrame(audit_rows), DATA_DIR / "duplicate_asset_audit.csv")
    _write_csv(
        dataset_model_df[
            [
                "dataset_id",
                "dataset_prefix",
                "model_id",
                "model_label",
                "marginal_missing_rate_consistency",
                "co_missingness_pattern_consistency",
                "co_missing_strength_score",
                "co_missing_composite_score",
                "missingness_structure_score",
                "profile_minus_strength",
                "asset_count",
                "applicable_asset_count",
            ]
        ],
        DATA_DIR / "missingness_query_rows.csv",
    )
    _write_csv(subitem_df, DATA_DIR / "dataset_model_subitems.csv")
    _write_csv(dataset_model_df, DATA_DIR / "dataset_model_scores.csv")
    _write_csv(model_summary_df, DATA_DIR / "model_summary.csv")
    _write_csv(prefix_summary_df, DATA_DIR / "prefix_summary.csv")
    _write_csv(dataset_summary_df, DATA_DIR / "dataset_summary.csv")
    _write_csv(heatmap_df, DATA_DIR / "dataset_model_heatmap.csv")
    _write_csv(prefix_plot_df, DATA_DIR / "prefix_plot_data.csv")
    _write_csv(model_subitem_heatmap_df, DATA_DIR / "model_subitem_heatmap.csv")
    _write_csv(direct_dataset_context_df, DATA_DIR / "direct_dataset_context.csv")
    _write_csv(direct_asset_scores_df, DATA_DIR / "direct_asset_scores.csv")
    _write_csv(direct_target_scores_df, DATA_DIR / "direct_target_scores.csv")
    _write_csv(direct_model_dataset_df, DATA_DIR / "direct_model_dataset_summary.csv")
    _write_csv(direct_model_overall_df, DATA_DIR / "direct_model_overall_summary.csv")

    tradeoff_tex = FIG_DIR / "missingness_tradeoff_scatter_main.tex"
    tradeoff_pdf = FIG_DIR / "missingness_tradeoff_scatter_main.pdf"
    tradeoff_png = FIG_DIR / "missingness_tradeoff_scatter_main.png"
    prefix_tex = FIG_DIR / "missingness_prefix_bars_appendix.tex"
    prefix_pdf = FIG_DIR / "missingness_prefix_bars_appendix.pdf"
    prefix_png = FIG_DIR / "missingness_prefix_bars_appendix.png"
    heatmap_tex = FIG_DIR / "missingness_dataset_model_heatmap_appendix.tex"
    heatmap_pdf = FIG_DIR / "missingness_dataset_model_heatmap_appendix.pdf"
    heatmap_png = FIG_DIR / "missingness_dataset_model_heatmap_appendix.png"
    model_subitem_heatmap_tex = FIG_DIR / "missingness_model_subitem_heatmap_appendix.tex"
    model_subitem_heatmap_pdf = FIG_DIR / "missingness_model_subitem_heatmap_appendix.pdf"
    model_subitem_heatmap_png = FIG_DIR / "missingness_model_subitem_heatmap_appendix.png"
    grouped_bars_tex = FIG_DIR / "missingness_family_subitem_bars_appendix.tex"
    grouped_bars_pdf = FIG_DIR / "missingness_family_subitem_bars_appendix.pdf"
    grouped_bars_png = FIG_DIR / "missingness_family_subitem_bars_appendix.png"

    _write_tradeoff_tex(model_summary_df, tradeoff_tex)
    _write_prefix_bar_tex(prefix_plot_df, prefix_tex)
    _write_heatmap_tex(heatmap_df, heatmap_tex)
    write_model_subitem_heatmap_tex(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        title="Missingness model-subitem heatmap",
        colorbar_title="Mean score",
        path=model_subitem_heatmap_tex,
    )
    write_model_subitem_grouped_bar_tex(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        model_color_map=MODEL_COLORS,
        title="Missingness family and subitem bars",
        y_label="Score",
        path=grouped_bars_tex,
    )
    _plot_tradeoff_preview(model_summary_df, tradeoff_pdf, tradeoff_png)
    _plot_prefix_bar_preview(prefix_plot_df, prefix_pdf, prefix_png)
    _plot_heatmap_preview(heatmap_df, heatmap_pdf, heatmap_png)
    plot_model_subitem_heatmap_preview(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        title="Missingness model-subitem heatmap",
        pdf_path=model_subitem_heatmap_pdf,
        png_path=model_subitem_heatmap_png,
    )
    plot_model_subitem_grouped_bar_preview(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        model_color_map=MODEL_COLORS,
        title="Missingness family and subitem bars",
        y_label="Score",
        pdf_path=grouped_bars_pdf,
        png_path=grouped_bars_png,
    )

    compile_notes = {
        "tradeoff": _try_compile_tex(tradeoff_tex),
        "prefix": _try_compile_tex(prefix_tex),
        "heatmap": _try_compile_tex(heatmap_tex),
        "model_subitem_heatmap": _try_compile_tex(model_subitem_heatmap_tex),
        "family_subitem_bars": _try_compile_tex(grouped_bars_tex),
    }
    (OUTPUT_ROOT / "analysis_report.md").write_text(
        "\n".join(
            [
                "# Missingness Breakdown",
                "",
                "- Source mode: `direct_missingness_evaluator`",
                f"- Reference analysis run: `{analysis_run_dir.name}`",
                f"- Included models: `{model_summary_df.shape[0]}`",
                f"- Deduplicated dataset-model panels: `{dataset_model_df.shape[0]}`",
                f"- Direct model-dataset rows used: `{dataset_model_df.shape[0]}`",
                f"- Direct target rows used: `{direct_target_scores_df.shape[0]}`",
                "",
                "## Canonical decomposition",
                "",
                "- `missingness_structure = 0.5 * marginal_missing_rate_consistency + 0.5 * co_missingness_pattern_consistency`",
                "- Canonical `co_missingness_pattern_consistency` now uses profile-only edge averaging.",
                "- `co_missing_strength_score` is exported separately as an auxiliary diagnostic and is not folded into the two-subitem family score.",
                "- `co_missing_composite_score` preserves the previous 0.7-profile / 0.3-strength blend for sensitivity analysis only.",
                "- This bundle bypasses SQL query analysis and uses the canonical direct co-missing evaluator over real-train vs synthetic CSV pairs.",
                "- The standardized appendix bundle now mirrors tradeoff, prefix, and heatmap views just like the other query families.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    final_files = [
        tradeoff_tex,
        tradeoff_pdf,
        tradeoff_png,
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
        DATA_DIR / "model_summary.csv",
        DATA_DIR / "prefix_summary.csv",
        OUTPUT_ROOT / "analysis_report.md",
    ]
    must_do_aliases = {
        f"{stem}.{suffix}": FIG_DIR / f"{stem}.{suffix}"
        for stem in [
            "missingness_tradeoff_scatter_main",
            "missingness_prefix_bars_appendix",
            "missingness_dataset_model_heatmap_appendix",
            "missingness_model_subitem_heatmap_appendix",
            "missingness_family_subitem_bars_appendix",
        ]
        for suffix in ["tex", "pdf", "png"]
    }
    version_tag = OUTPUT_VERSION_TAG
    sync_final_outputs(FINAL_DIR, final_files, must_do_aliases, version_tag=version_tag, copy_plain_files=False)
    final_readme = render_final_readme(
        title="Missingness Breakdown Final",
        summary=f"This directory contains the paper-facing missingness breakdown artifacts published under `{sql_source_label(version_tag)}` (`{version_tag}`), with the standardized must-do bundle mirrored into `final/must_do/` and `final/{version_tag}/must_do/`.",
        primary_files=[versioned_name(name, version_tag) for name in [
            "missingness_tradeoff_scatter_main.tex",
            "missingness_tradeoff_scatter_main.pdf",
            "missingness_tradeoff_scatter_main.png",
            "missingness_prefix_bars_appendix.tex",
            "missingness_prefix_bars_appendix.pdf",
            "missingness_prefix_bars_appendix.png",
            "missingness_dataset_model_heatmap_appendix.tex",
            "missingness_dataset_model_heatmap_appendix.pdf",
            "missingness_dataset_model_heatmap_appendix.png",
            "missingness_model_subitem_heatmap_appendix.tex",
            "missingness_model_subitem_heatmap_appendix.pdf",
            "missingness_model_subitem_heatmap_appendix.png",
            "missingness_family_subitem_bars_appendix.tex",
            "missingness_family_subitem_bars_appendix.pdf",
            "missingness_family_subitem_bars_appendix.png",
            "model_summary.csv",
            "prefix_summary.csv",
        ]],
        must_do_files=[versioned_name(name, version_tag) for name in must_do_aliases.keys()],
        support_files=[versioned_name("analysis_report.md", version_tag)],
        notes=[
            f"The active published version tag for this bundle is `{sql_source_label(version_tag)}` (`{version_tag}`).",
            "The `.tex` files are standalone LaTeX sources. The `.pdf/.png` files are immediate previews for reading in the current environment.",
        ],
    )
    (FINAL_DIR / "README.md").write_text(final_readme, encoding="utf-8")

    manifest = {
        "task": "missingness_breakdown",
        "sql_source_version": version_tag,
        "sql_source_label": sql_source_label(version_tag),
        "source_mode": "direct_missingness_evaluator",
        "reference_analysis_run": analysis_run_dir.name,
        "included_models": model_summary_df["model_id"].tolist(),
        "dataset_panel_count": int(dataset_model_df.shape[0]),
        "query_row_count": int(dataset_model_df.shape[0]),
        "status": "implemented_direct",
        "compile_notes": {key: {"success": value[0], "note": value[1]} for key, value in compile_notes.items()},
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the missingness breakdown bundle.")
    parser.add_argument("--analysis-run-dir", type=Path, default=None, help="Optional analysis run dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_missingness_breakdown(analysis_run_dir=args.analysis_run_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
