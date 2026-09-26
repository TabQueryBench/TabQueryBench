#!/usr/bin/env python3
"""Decompose conditional_dependency_structure with a subgroup-focused lens."""

from __future__ import annotations

import csv
import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.analytics_contract import annotate_query_row_with_contract
from tqb_scoring.eval.common import (
    DEFAULT_SQL_SOURCE_VERSION,
    resolve_requested_sql_source_version,
    resolve_task_run_dir_for_sql_source,
    sql_source_label,
)
from tqb_scoring.eval.query_fivepart_breakdown.common_final import render_final_readme, sync_final_outputs, versioned_name
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
OUTPUT_ROOT = EVALUATION_ROOT / "query_fivepart_breakdown" / "conditional_breakdown"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
FINAL_DIR = OUTPUT_ROOT / "final"
TARGET_SQL_SOURCE_VERSION = resolve_requested_sql_source_version("analysis", DEFAULT_SQL_SOURCE_VERSION)

TARGET_FAMILY = "conditional_dependency_structure"
SUBITEM_ORDER = [
    "dependency_strength_similarity",
    "direction_consistency",
    "slice_level_consistency",
]
SUBITEM_LABELS = {
    "dependency_strength_similarity": "Dependency strength similarity",
    "direction_consistency": "Direction consistency",
    "slice_level_consistency": "Slice-level consistency",
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
REAL_COLOR = "#000000"
EXCLUDED_MODELS = {"cdtd", "codi", "goggle"}
MODEL_ALIASES = {"rtf": "realtabformer"}
ALLOWED_MODELS = set(MODEL_COLORS.keys())
SERVER_PRIORITY = {"rtx_5090": 2, "rtx_pro_6000": 1}
ROOT_PRIORITY = {"SynOutput-5090": 2, "SynOutput": 1}
TIMESTAMP_RE = re.compile(r"(20\d{6}_\d{6})")
PREFIX_LABELS = {"c": "Categorical", "m": "Mixed", "n": "Numerical"}
SUBITEM_BAR_COLORS = {
    "dependency_strength_similarity": "#4C78A8",
    "direction_consistency": "#F58518",
    "slice_level_consistency": "#54A24B",
}


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, DATA_DIR, FIG_DIR, TABLE_DIR, FINAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _normalize_model(model_id: Any) -> str:
    key = str(model_id or "").strip().lower()
    return MODEL_ALIASES.get(key, key)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _is_allowed_model(model_id: str) -> bool:
    return model_id in ALLOWED_MODELS and model_id not in EXCLUDED_MODELS


def _model_sort_rank(model_id: str) -> int:
    return MODEL_ORDER.index(model_id) if model_id in MODEL_ORDER else 999


def _dataset_prefix(dataset_id: str) -> str:
    return str(dataset_id or "").strip().lower()[:1]


def _dataset_sort_key(dataset_id: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", str(dataset_id).strip())
    if not match:
        return (99, 10**9, str(dataset_id))
    prefix, number = match.groups()
    return ({"c": 0, "m": 1, "n": 2}.get(prefix.lower(), 50), int(number), str(dataset_id))


def _parse_timestamp_key(*texts: Any) -> str:
    matches: list[str] = []
    for text in texts:
        if text is None:
            continue
        matches.extend(TIMESTAMP_RE.findall(str(text)))
    return max(matches) if matches else ""


def _asset_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _parse_timestamp_key(row.get("run_id"), row.get("synthetic_csv_path"), row.get("timestamp_utc")),
        str(row.get("timestamp_utc") or ""),
        SERVER_PRIORITY.get(str(row.get("server_type") or ""), 0),
        ROOT_PRIORITY.get(str(row.get("root_name") or ""), 0),
        float(row.get("overall_score") or 0.0),
        str(row.get("synthetic_csv_path") or ""),
    )


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


def _resolve_existing_run_dir(task_root: Path, *, sql_source_version: str = TARGET_SQL_SOURCE_VERSION) -> Path:
    task_name = task_root.name
    resolved = resolve_task_run_dir_for_sql_source(task_name, sql_source_version)
    if resolved is not None:
        return resolved
    runs_root = task_root / "runs"
    raise FileNotFoundError(f"No run directories found under {runs_root} for sql_source_version={sql_source_version!r}")


def _load_primary_assets(asset_csv: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    with asset_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset_id = str(row.get("dataset_id") or "").strip()
        model_id = _normalize_model(row.get("model_id"))
        if not dataset_id or not model_id or not _is_allowed_model(model_id):
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


def _stream_conditional_query_rows(
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
            if not dataset_id or not model_id or not _is_allowed_model(model_id):
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
            score = annotated.get("query_score")
            try:
                score_value = float(score)
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
                    "subitem_inference_source": str(annotated.get("subitem_inference_source") or ""),
                    "subitem_inference_note": str(annotated.get("subitem_inference_note") or ""),
                    "sql_engine": str(annotated.get("sql_engine") or ""),
                }
            )
    return out


def _build_dataset_model_scores(query_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subitems = (
        query_df.groupby(
            ["dataset_id", "dataset_prefix", "model_id", "model_label", "subitem_id", "subitem_label"],
            as_index=False,
        )
        .agg(query_count=("query_id", "count"), subitem_score=("query_score", "mean"))
        .sort_values(["dataset_id", "model_id", "subitem_id"])
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
    wide["conditional_dependency_structure_score"] = wide[SUBITEM_ORDER].mean(axis=1, skipna=True)
    wide["conditional_subgroup_score"] = wide[["direction_consistency", "slice_level_consistency"]].mean(axis=1, skipna=True)
    wide["direction_minus_slice"] = wide["direction_consistency"] - wide["slice_level_consistency"]
    wide["strength_minus_subgroup"] = wide["dependency_strength_similarity"] - wide["conditional_subgroup_score"]
    wide["active_subitem_count"] = wide[SUBITEM_ORDER].notna().sum(axis=1)
    wide["model_sort"] = wide["model_id"].map(_model_sort_rank)
    wide = (
        wide.sort_values(
            by=["dataset_prefix", "dataset_id", "model_sort", "model_id"],
            key=lambda s: s.map(_dataset_sort_key) if s.name == "dataset_id" else s,
        )
        .drop(columns=["model_sort"])
        .reset_index(drop=True)
    )
    return subitems, wide


def _build_model_summary(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = SUBITEM_ORDER + [
        "conditional_dependency_structure_score",
        "conditional_subgroup_score",
        "direction_minus_slice",
        "strength_minus_subgroup",
    ]
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
    summary["model_sort"] = summary["model_id"].map(_model_sort_rank)
    summary = summary.sort_values(["model_sort", "model_id"]).drop(columns=["model_sort"])
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
                "dependency_strength_similarity": round(float(group["dependency_strength_similarity"].mean()), 6),
                "direction_consistency": round(float(group["direction_consistency"].mean()), 6),
                "slice_level_consistency": round(float(group["slice_level_consistency"].mean()), 6),
                "conditional_dependency_structure_score": round(float(group["conditional_dependency_structure_score"].mean()), 6),
                "conditional_subgroup_score": round(float(group["conditional_subgroup_score"].mean()), 6),
                "direction_minus_slice": round(float(group["direction_minus_slice"].mean()), 6),
                "strength_minus_subgroup": round(float(group["strength_minus_subgroup"].mean()), 6),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["model_sort"] = out["model_id"].map(_model_sort_rank)
    out = out.sort_values(["model_sort", "dataset_prefix"]).drop(columns=["model_sort"])
    return out.reset_index(drop=True)


def _build_prefix_model_summaries(dataset_model_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    metrics = SUBITEM_ORDER + [
        "conditional_dependency_structure_score",
        "conditional_subgroup_score",
        "direction_minus_slice",
        "strength_minus_subgroup",
    ]
    for prefix in ["c", "m", "n"]:
        prefix_df = dataset_model_df[dataset_model_df["dataset_prefix"] == prefix].copy()
        if prefix_df.empty:
            out[prefix] = pd.DataFrame()
            continue
        rows: list[dict[str, Any]] = []
        for model_id, group in prefix_df.groupby("model_id", sort=False):
            payload = {
                "dataset_prefix": prefix,
                "dataset_prefix_label": PREFIX_LABELS.get(prefix, prefix.upper()),
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "dataset_count": int(group["dataset_id"].nunique()),
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
        if not summary.empty:
            summary["model_sort"] = summary["model_id"].map(_model_sort_rank)
            summary = summary.sort_values(["model_sort", "model_id"]).drop(columns=["model_sort"]).reset_index(drop=True)
        out[prefix] = summary
    return out


def _build_dataset_summary(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, group in dataset_model_df.groupby("dataset_id", sort=False):
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": _dataset_prefix(dataset_id),
                "model_count": int(group["model_id"].nunique()),
                "dependency_strength_similarity": round(float(group["dependency_strength_similarity"].mean()), 6),
                "direction_consistency": round(float(group["direction_consistency"].mean()), 6),
                "slice_level_consistency": round(float(group["slice_level_consistency"].mean()), 6),
                "conditional_dependency_structure_score": round(float(group["conditional_dependency_structure_score"].mean()), 6),
                "conditional_subgroup_score": round(float(group["conditional_subgroup_score"].mean()), 6),
                "direction_minus_slice": round(float(group["direction_minus_slice"].mean()), 6),
                "strength_minus_subgroup": round(float(group["strength_minus_subgroup"].mean()), 6),
                "subgroup_score_std_across_models": round(float(group["conditional_subgroup_score"].std(ddof=1)) if len(group) > 1 else 0.0, 6),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["sort_key"] = out["dataset_id"].map(_dataset_sort_key)
    out = out.sort_values(["conditional_subgroup_score", "sort_key"], ascending=[True, True]).drop(columns=["sort_key"])
    return out.reset_index(drop=True)


def _build_heatmap_data(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    heatmap = (
        dataset_model_df.pivot_table(index="dataset_id", columns="model_id", values="conditional_subgroup_score", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if heatmap.empty:
        return heatmap
    heatmap["sort_key"] = heatmap["dataset_id"].map(_dataset_sort_key)
    heatmap = heatmap.sort_values(["sort_key"]).drop(columns=["sort_key"])
    return heatmap.reset_index(drop=True)


def _build_prefix_plot_data(prefix_summary_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        prefix_summary_df.pivot_table(index=["model_id", "model_label"], columns="dataset_prefix", values="conditional_subgroup_score", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if pivot.empty:
        return pivot
    pivot["model_sort"] = pivot["model_id"].map(_model_sort_rank)
    pivot = pivot.sort_values(["model_sort"]).drop(columns=["model_sort"])
    pivot = pivot.reset_index(drop=True)
    real_row = {
        "model_id": "REAL",
        "model_label": "REAL",
        "c": 1.0,
        "m": 1.0,
        "n": 1.0,
    }
    pivot = pivot.loc[pivot["model_id"] != "REAL"].reset_index(drop=True)
    return pd.concat([pd.DataFrame([real_row]), pivot], ignore_index=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


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
            r"\usetikzlibrary{patterns}",
            r"\usepackage{xcolor}",
            r"\pgfplotsset{compat=1.18}",
            "",
        ]
    )


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
            r"width=12.8cm,",
            r"height=9.4cm,",
            rf"xmin={x_min:.2f}, xmax={x_max:.2f},",
            rf"ymin={y_min:.2f}, ymax={y_max:.2f},",
            rf"xlabel={{{_escape_tex(x_label)}}},",
            rf"ylabel={{{_escape_tex(y_label)}}},",
            rf"title={{{_escape_tex(title)}}},",
            r"grid=both,",
            r"grid style={draw=gray!20},",
            r"major grid style={draw=gray!30},",
            r"axis line style={draw=black!70},",
            r"tick style={draw=black!70},",
            r"clip=false,",
            r"legend style={draw=none, fill=none, font=\scriptsize, at={(0.02,0.98)}, anchor=north west},",
            r"legend columns=2,",
            r"]",
        ]
    )
    for row in model_summary_df.itertuples():
        color_name = f"model{row.model_id}"
        x = float(getattr(row, f"{x_metric}__mean"))
        y = float(getattr(row, f"{y_metric}__mean"))
        xerr = float(getattr(row, f"{x_metric}__ci95_radius") or 0.0)
        yerr = float(getattr(row, f"{y_metric}__ci95_radius") or 0.0)
        body.append(
            "\n".join(
                [
                    rf"\addplot+[only marks, mark=*, mark size=2.8pt, draw={color_name}, fill={color_name},",
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


def _write_dumbbell_tex(model_summary_df: pd.DataFrame, path: Path) -> None:
    ordered = model_summary_df.copy()
    ordered["sort_rank"] = range(len(ordered), 0, -1)
    color_defs = [
        rf"\definecolor{{model{row.model_id}}}{{HTML}}{{{MODEL_COLORS[row.model_id].replace('#', '')}}}"
        for row in ordered.itertuples()
        if row.model_id in MODEL_COLORS
    ]
    body = [_tex_preamble(), *color_defs, r"\begin{document}", r"\begin{tikzpicture}", r"\begin{axis}["]
    y_labels = ",".join(_escape_tex(item) for item in ordered["model_label"].tolist())
    y_ticks = ",".join(str(v) for v in ordered["sort_rank"].tolist())
    body.extend(
        [
            r"width=14.0cm,",
            r"height=10.2cm,",
            r"xmin=0.10, xmax=0.95,",
            rf"ytick={{{y_ticks}}},",
            rf"yticklabels={{{y_labels}}},",
            r"y dir=reverse,",
            r"xlabel={Score},",
            r"grid=both,",
            r"grid style={draw=gray!18},",
            r"tick style={draw=black!70},",
            r"legend style={draw=none, fill=none, font=\scriptsize, at={(0.03,0.03)}, anchor=south west},",
            r"]",
        ]
    )
    for row in ordered.itertuples():
        color_name = f"model{row.model_id}"
        y = int(row.sort_rank)
        strength = float(getattr(row, "dependency_strength_similarity__mean"))
        direction = float(getattr(row, "direction_consistency__mean"))
        slice_score = float(getattr(row, "slice_level_consistency__mean"))
        subgroup = float(getattr(row, "conditional_subgroup_score__mean"))
        body.append(
            "\n".join(
                [
                    rf"\draw[line width=0.8pt, color={color_name}!65] (axis cs:{direction:.4f},{y}) -- (axis cs:{slice_score:.4f},{y});",
                    rf"\addplot+[only marks, mark=square*, mark size=2.5pt, draw={color_name}, fill={color_name}] coordinates {{ ({strength:.4f},{y}) }};",
                    rf"\addplot+[only marks, mark=*, mark size=2.5pt, draw={color_name}, fill={color_name}] coordinates {{ ({direction:.4f},{y}) }};",
                    rf"\addplot+[only marks, mark=triangle*, mark size=2.8pt, draw={color_name}, fill={color_name}] coordinates {{ ({slice_score:.4f},{y}) }};",
                    rf"\addplot+[only marks, mark=diamond*, mark size=2.6pt, draw=black, fill=black] coordinates {{ ({subgroup:.4f},{y}) }};",
                ]
            )
        )
    body.extend(
        [
            r"\addlegendimage{only marks, mark=square*, mark size=2.5pt, draw=black, fill=black}",
            r"\addlegendentry{Dependency strength}",
            r"\addlegendimage{only marks, mark=*, mark size=2.5pt, draw=black, fill=black}",
            r"\addlegendentry{Direction consistency}",
            r"\addlegendimage{only marks, mark=triangle*, mark size=2.8pt, draw=black, fill=black}",
            r"\addlegendentry{Slice-level consistency}",
            r"\addlegendimage{only marks, mark=diamond*, mark size=2.6pt, draw=black, fill=black}",
            r"\addlegendentry{Derived subgroup score}",
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )
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


def _write_prefix_bar_tex(prefix_plot_df: pd.DataFrame, path: Path) -> None:
    color_defs = [
        rf"\definecolor{{model{str(row.model_id).lower()}}}{{HTML}}{{{(REAL_COLOR if str(row.model_id) == 'REAL' else MODEL_COLORS[row.model_id]).replace('#', '')}}}"
        for row in prefix_plot_df.itertuples()
        if str(row.model_id) == "REAL" or row.model_id in MODEL_COLORS
    ]
    body = [_tex_preamble(), *color_defs, r"\begin{document}", r"\begin{tikzpicture}", r"\begin{groupplot}["]
    body.extend(
        [
            r"group style={group size=3 by 1, horizontal sep=1.1cm},",
            r"width=4.2cm,",
            r"height=7.0cm,",
            r"ybar,",
            r"bar width=7pt,",
            r"ymin=0, ymax=1.02,",
            r"xtick=data,",
            r"x tick label style={rotate=45, anchor=east, font=\tiny},",
            r"tick label style={font=\scriptsize},",
            r"label style={font=\scriptsize},",
            r"grid=both,",
            r"grid style={draw=gray!20},",
            r"]",
        ]
    )
    tick_positions = ",".join(str(idx) for idx in range(1, len(prefix_plot_df) + 1))
    tick_labels = ",".join(_escape_tex(item) for item in prefix_plot_df["model_label"].tolist())
    for prefix in ["c", "m", "n"]:
        body.append(
            rf"\nextgroupplot[title={{{_escape_tex(PREFIX_LABELS[prefix])}}}, ylabel={{Derived subgroup score}}, xtick={{{tick_positions}}}, xticklabels={{{tick_labels}}}]"
        )
        for idx, row in enumerate(prefix_plot_df.itertuples(), start=1):
            if not hasattr(row, prefix):
                continue
            value = getattr(row, prefix)
            if pd.isna(value):
                continue
            body.append(
                rf"\addplot+[draw=model{str(row.model_id).lower()}, fill=model{str(row.model_id).lower()}] coordinates {{ ({idx},{float(value):.4f}) }};"
            )
    body.extend([r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


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
    fig, ax = plt.subplots(figsize=(8.7, 6.7))
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
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dumbbell_preview(model_summary_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    ordered = model_summary_df.copy().reset_index(drop=True)
    y = list(range(len(ordered)))
    fig, ax = plt.subplots(figsize=(9.4, max(6.2, 0.52 * len(ordered) + 1.7)))
    for idx, row in enumerate(ordered.itertuples()):
        color = MODEL_COLORS[row.model_id]
        strength = float(getattr(row, "dependency_strength_similarity__mean"))
        direction = float(getattr(row, "direction_consistency__mean"))
        slice_score = float(getattr(row, "slice_level_consistency__mean"))
        subgroup = float(getattr(row, "conditional_subgroup_score__mean"))
        ax.plot([direction, slice_score], [idx, idx], color=color, linewidth=2.0, alpha=0.75, zorder=1)
        ax.scatter(strength, idx, marker="s", s=54, color=color, zorder=3)
        ax.scatter(direction, idx, marker="o", s=58, color=color, zorder=3)
        ax.scatter(slice_score, idx, marker="^", s=64, color=color, zorder=3)
        ax.scatter(subgroup, idx, marker="D", s=44, color="black", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(0.10, 0.95)
    ax.set_xlabel("Score")
    ax.set_title("Conditional decomposition by model")
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.scatter([], [], marker="s", s=54, color="black", label="Dependency strength")
    ax.scatter([], [], marker="o", s=58, color="black", label="Direction consistency")
    ax.scatter([], [], marker="^", s=64, color="black", label="Slice-level consistency")
    ax.scatter([], [], marker="D", s=44, color="black", label="Derived subgroup score")
    ax.legend(frameon=False, loc="lower right")
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
    ax.set_title("Dataset-model conditional subgroup score heatmap")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Derived subgroup score")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_prefix_bar_preview(prefix_plot_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    prefixes = ["c", "m", "n"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.2), sharey=True)
    for ax, prefix in zip(axes, prefixes):
        values = pd.to_numeric(prefix_plot_df[prefix], errors="coerce")
        colors = [REAL_COLOR if str(model_id) == "REAL" else MODEL_COLORS.get(model_id, "#777777") for model_id in prefix_plot_df["model_id"]]
        ax.bar(range(len(prefix_plot_df)), values, color=colors)
        ax.set_title(PREFIX_LABELS[prefix])
        ax.set_xticks(range(len(prefix_plot_df)))
        ax.set_xticklabels(prefix_plot_df["model_label"], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.22)
        ax.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Derived subgroup score")
    fig.suptitle("Conditional subgroup score by dataset family prefix")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _prefix_scatter_note(prefix: str) -> list[str]:
    return [
        f"{PREFIX_LABELS.get(prefix, prefix.upper())} datasets only.",
        "Axes isolate the two subgroup-sensitive conditional branches; both directions include 95% CI error bars.",
    ]


def _prefix_bar_note(prefix: str) -> list[str]:
    return [
        f"{PREFIX_LABELS.get(prefix, prefix.upper())} datasets only.",
        "Each model is decomposed into the three canonical conditional subitems.",
    ]


def _build_real_augmented_prefix_summary(summary_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    summary_df = summary_df.copy()
    if summary_df.empty:
        return summary_df
    if "dataset_prefix" in summary_df.columns:
        summary_df = summary_df[summary_df["dataset_prefix"] == prefix].copy()
    if summary_df.empty:
        return summary_df
    if "model_sort" not in summary_df.columns:
        summary_df["model_sort"] = summary_df["model_id"].map(_model_sort_rank)
    summary_df = summary_df.sort_values(["model_sort", "model_id"]).drop(columns=["model_sort"]).reset_index(drop=True)

    for subitem_id in SUBITEM_ORDER:
        ci_col = f"{subitem_id}__ci95_radius"
        if ci_col not in summary_df.columns:
            summary_df[ci_col] = 0.0

    real_row: dict[str, Any] = {
        "model_id": "REAL",
        "model_label": "REAL",
        "dataset_prefix": prefix,
        "dataset_prefix_label": PREFIX_LABELS.get(prefix, prefix.upper()),
        "dataset_count": int(summary_df["dataset_count"].max()),
        "dependency_strength_similarity": 1.0,
        "direction_consistency": 1.0,
        "slice_level_consistency": 1.0,
        "conditional_dependency_structure_score": 1.0,
        "conditional_subgroup_score": 1.0,
        "direction_minus_slice": 0.0,
        "strength_minus_subgroup": 0.0,
        "dependency_strength_similarity__ci95_radius": 0.0,
        "direction_consistency__ci95_radius": 0.0,
        "slice_level_consistency__ci95_radius": 0.0,
    }
    real_df = pd.DataFrame([real_row])
    return pd.concat([real_df, summary_df], ignore_index=True)


def _write_prefix_subitem_bar_tex(summary_df: pd.DataFrame, prefix: str, path: Path) -> None:
    summary_df = _build_real_augmented_prefix_summary(summary_df, prefix)
    if summary_df.empty:
        path.write_text("", encoding="utf-8")
        return

    tick_positions = ",".join(str(idx + 1) for idx in range(len(summary_df)))
    tick_labels = ",".join(_escape_tex(item) for item in summary_df["model_label"].tolist())
    body = [_tex_preamble()]
    for row in summary_df.itertuples():
        color = REAL_COLOR if str(row.model_id) == "REAL" else MODEL_COLORS[str(row.model_id)]
        body.append(rf"\definecolor{{model{str(row.model_id).lower()}}}{{HTML}}{{{color.replace('#', '')}}}")
    body.extend([r"\begin{document}", r"\begin{minipage}{14.5cm}"])
    for line in _prefix_bar_note(prefix):
        body.append(r"{\small " + _escape_tex(line) + r"\par}")
    body.extend([r"\vspace{0.4em}", r"\begin{tikzpicture}", r"\begin{axis}["])
    body.extend(
        [
            r"width=14.0cm,",
            r"height=8.8cm,",
            r"ybar,",
            r"bar width=5.5pt,",
            r"ymin=0, ymax=1,",
            rf"xtick={{{tick_positions}}},",
            rf"xticklabels={{{tick_labels}}},",
            r"x tick label style={rotate=45, anchor=east, font=\scriptsize},",
            r"tick label style={font=\scriptsize},",
            r"label style={font=\scriptsize},",
            r"ylabel={Score},",
            rf"title={{{_escape_tex(f'Conditional subitem breakdown within {PREFIX_LABELS.get(prefix, prefix.upper())} datasets')}}},",
            r"legend style={draw=none, fill=none, font=\scriptsize, at={(0.98,0.98)}, anchor=north east},",
            r"legend columns=1,",
            r"grid=both,",
            r"grid style={draw=gray!20},",
            r"major grid style={draw=gray!30},",
            r"]",
        ]
    )
    bar_width = 0.22
    offsets = {
        "dependency_strength_similarity": -bar_width,
        "direction_consistency": 0.0,
        "slice_level_consistency": bar_width,
    }
    for subitem_id in SUBITEM_ORDER:
        for idx, row in summary_df.iterrows():
            if pd.isna(row[subitem_id]):
                continue
            model_key = str(row["model_id"]).lower()
            x = idx + 1 + offsets[subitem_id]
            y = float(row[subitem_id])
            yerr = float(row.get(f"{subitem_id}__ci95_radius", 0.0) or 0.0)
            if subitem_id == "dependency_strength_similarity":
                style = rf"draw=model{model_key}, fill=model{model_key}"
            elif subitem_id == "direction_consistency":
                style = rf"draw=model{model_key}, fill=white, pattern=north east lines, pattern color=model{model_key}"
            else:
                style = rf"draw=model{model_key}, fill=white, pattern=crosshatch, pattern color=model{model_key}"
            body.append(
                rf"\addplot+[bar width={bar_width:.2f}, {style}, error bars/.cd, y dir=both, y explicit] coordinates {{ ({x:.3f},{y:.4f}) +- (0,{yerr:.4f}) }};"
            )
    body.extend(
        [
            r"\addlegendimage{area legend, draw=black, fill=black}",
            rf"\addlegendentry{{{_escape_tex(SUBITEM_LABELS['dependency_strength_similarity'])}}}",
            r"\addlegendimage{area legend, draw=black, fill=white, pattern=north east lines, pattern color=black}",
            rf"\addlegendentry{{{_escape_tex(SUBITEM_LABELS['direction_consistency'])}}}",
            r"\addlegendimage{area legend, draw=black, fill=white, pattern=crosshatch, pattern color=black}",
            rf"\addlegendentry{{{_escape_tex(SUBITEM_LABELS['slice_level_consistency'])}}}",
            r"\node[anchor=west, font=\scriptsize] at (axis cs:0.32,0.995) {$\uparrow$ better};",
        ]
    )
    body.extend([r"\end{axis}", r"\end{tikzpicture}", r"\end{minipage}", r"\end{document}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


def _plot_prefix_subitem_bar_preview(summary_df: pd.DataFrame, prefix: str, pdf_path: Path, png_path: Path) -> None:
    summary_df = _build_real_augmented_prefix_summary(summary_df, prefix)
    if summary_df.empty:
        return

    x = list(range(len(summary_df)))
    width = 0.24
    offsets = [-width, 0.0, width]
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    for offset, subitem_id in zip(offsets, SUBITEM_ORDER):
        values = pd.to_numeric(summary_df[subitem_id], errors="coerce")
        errors = pd.to_numeric(summary_df.get(f"{subitem_id}__ci95_radius", 0.0), errors="coerce").fillna(0.0)
        edgecolors = [REAL_COLOR if str(model_id) == "REAL" else MODEL_COLORS[str(model_id)] for model_id in summary_df["model_id"]]
        if subitem_id == "dependency_strength_similarity":
            facecolors = edgecolors
            hatch = None
        elif subitem_id == "direction_consistency":
            facecolors = ["white"] * len(summary_df)
            hatch = "///"
        else:
            facecolors = ["white"] * len(summary_df)
            hatch = "xx"
        ax.bar(
            [item + offset for item in x],
            values,
            width=width,
            color=facecolors,
            edgecolor=edgecolors,
            linewidth=1.8,
            hatch=hatch,
            label=SUBITEM_LABELS[subitem_id],
            yerr=errors,
            error_kw={"elinewidth": 1.4, "ecolor": "black", "capsize": 4, "capthick": 1.2},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model_label"], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Conditional subitem breakdown within {PREFIX_LABELS.get(prefix, prefix.upper())} datasets")
    ax.grid(axis="y", linestyle="--", alpha=0.24)
    ax.text(0.02, 0.98, "↑ better", transform=ax.transAxes, ha="left", va="top", fontsize=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_model_table_tex(model_summary_df: pd.DataFrame, path: Path) -> None:
    ordered = model_summary_df.copy()
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Conditional & Derived subgroup & Strength & Direction & Slice \\",
        r"\midrule",
    ]
    for row in ordered.itertuples():
        lines.append(
            (
                f"{_escape_tex(row.model_label)} & "
                f"{float(getattr(row, 'conditional_dependency_structure_score__mean')):.3f} & "
                f"{float(getattr(row, 'conditional_subgroup_score__mean')):.3f} & "
                f"{float(getattr(row, 'dependency_strength_similarity__mean')):.3f} & "
                f"{float(getattr(row, 'direction_consistency__mean')):.3f} & "
                f"{float(getattr(row, 'slice_level_consistency__mean')):.3f} \\\\"
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_report(
    run_dir: Path,
    query_df: pd.DataFrame,
    dataset_model_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    prefix_summary_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
) -> None:
    top_subgroup = model_summary_df.sort_values("conditional_subgroup_score__mean", ascending=False).iloc[0]
    top_canonical = model_summary_df.sort_values("conditional_dependency_structure_score__mean", ascending=False).iloc[0]
    most_direction_heavy = model_summary_df.sort_values("direction_minus_slice__mean", ascending=False).iloc[0]
    most_slice_heavy = model_summary_df.sort_values("direction_minus_slice__mean", ascending=True).iloc[0]
    most_strength_optimistic = model_summary_df.sort_values("strength_minus_subgroup__mean", ascending=False).iloc[0]
    hardest_dataset = dataset_summary_df.sort_values("conditional_subgroup_score").iloc[0]
    easiest_dataset = dataset_summary_df.sort_values("conditional_subgroup_score", ascending=False).iloc[0]

    lines = [
        "# Conditional Breakdown Report",
        "",
        "## Scope",
        "",
        f"- Source analysis run: `{run_dir.name}`",
        f"- Family analyzed: `{TARGET_FAMILY}`",
        f"- Excluded models: `{', '.join(sorted(EXCLUDED_MODELS))}`",
        f"- Included models: `{model_summary_df.shape[0]}` from the frozen README roster",
        f"- Deduplicated dataset-model panels: `{dataset_model_df.shape[0]}`",
        f"- Conditional query rows used: `{query_df.shape[0]}`",
        "",
        "## Canonical and derived views",
        "",
        "- Canonical score: `mean(dependency_strength_similarity, direction_consistency, slice_level_consistency)`",
        "- Derived subgroup score: `mean(direction_consistency, slice_level_consistency)`",
        "- The derived score is not a replacement for the frozen contract; it isolates the two subgroup-sensitive conditional branches for paper analysis.",
        "",
        "## Main findings",
        "",
        (
            f"1. `{top_subgroup['model_label']}` is the strongest model on the subgroup-facing conditional view "
            f"with mean derived subgroup score `{top_subgroup['conditional_subgroup_score__mean']:.3f}`."
        ),
        (
            f"2. Canonically, `{top_canonical['model_label']}` leads the full conditional family "
            f"with mean conditional score `{top_canonical['conditional_dependency_structure_score__mean']:.3f}`."
        ),
        (
            f"3. `{most_direction_heavy['model_label']}` is the most direction-heavy model "
            f"(direction minus slice = `{most_direction_heavy['direction_minus_slice__mean']:.3f}`), while "
            f"`{most_slice_heavy['model_label']}` is the most slice-heavy "
            f"(`{most_slice_heavy['direction_minus_slice__mean']:.3f}`)."
        ),
        (
            f"4. `{most_strength_optimistic['model_label']}` shows the largest strength-to-subgroup drop risk: "
            f"its dependency-strength mean exceeds its subgroup-facing mean by "
            f"`{most_strength_optimistic['strength_minus_subgroup__mean']:.3f}`."
        ),
        (
            f"5. Dataset difficulty is uneven: `{hardest_dataset['dataset_id']}` is hardest on the subgroup-facing conditional view "
            f"(`{hardest_dataset['conditional_subgroup_score']:.3f}` mean across models), while "
            f"`{easiest_dataset['dataset_id']}` is easiest (`{easiest_dataset['conditional_subgroup_score']:.3f}`)."
        ),
        "",
        "## Files to use first",
        "",
        "- `figures/conditional_subgroup_tradeoff_scatter_main.pdf`",
        "- `figures/conditional_strength_vs_subgroup_bridge.pdf`",
        "- `figures/conditional_branch_dumbbell_main.pdf`",
        "- `tables/conditional_model_summary_generated.tex`",
        "- `data/model_summary.csv`",
        "",
        "## README compliance note",
        "",
        "- All plotted models are restricted to the frozen README roster with fixed colors.",
        "- Scatter plots now use legends instead of point-side model labels, matching the README figure annotation rule.",
        "- Model order is fixed globally instead of being re-sorted by score.",
        "",
        "## Prefix note",
        "",
        f"- Prefix coverage summary rows: `{prefix_summary_df.shape[0]}`",
        "- Prefix-level figures are exported for `c / m / n` slice checks, but the paper-facing core keeps the full deduplicated panel.",
        "",
    ]
    (OUTPUT_ROOT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def _build_readme(run_dir: Path) -> None:
    content = f"""# Conditional Breakdown

This directory contains a conditional-focused decomposition analysis built from the repository's unified `analysis` outputs.

## Inputs

- Source run: `{run_dir.name}`
- Query-level source: `{(run_dir / 'summaries' / 'analysis_query_scores__all_datasets.jsonl').relative_to(PROJECT_ROOT)}`
- Asset-level source: `{(run_dir / 'summaries' / 'analysis_asset_scores__all_datasets.csv').relative_to(PROJECT_ROOT)}`
- Canonical contract: `doc/analytics_family_subitem_contract_v1.md`

## What this analysis exports

- deduplicated dataset-model conditional scores
- canonical three-branch conditional summaries
- a subgroup-facing derived conditional view for `direction + slice`
- paper-ready TikZ figures and LaTeX table snippets
- final copies under `Evaluation/query_fivepart_breakdown/conditional_breakdown/final/`

## Re-run

```bash
python tqb_scoring/eval/query_fivepart_breakdown/conditional_breakdown/runner.py
```

## TeX compilation

The runner writes standalone `.tex` files and tries `latexmk -pdf` when available.  
If no local TeX compiler exists, it still exports matching preview `.pdf/.png` files for immediate inspection.
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


def run_conditional_breakdown(
    *,
    analysis_run_dir: Path | None = None,
    publish_final: bool = True,
) -> dict[str, Any]:
    _ensure_dirs()
    run_dir = analysis_run_dir.resolve() if analysis_run_dir is not None else _resolve_existing_run_dir(ANALYSIS_ROOT, sql_source_version=TARGET_SQL_SOURCE_VERSION)
    summary_dir = run_dir / "summaries"
    asset_csv = summary_dir / "analysis_asset_scores__all_datasets.csv"
    query_jsonl = summary_dir / "analysis_query_scores__all_datasets.jsonl"
    if not asset_csv.exists() or not query_jsonl.exists():
        raise FileNotFoundError("Primary analysis run is missing required summary files.")

    primary_assets, duplicate_audit_rows = _load_primary_assets(asset_csv)
    query_rows = _stream_conditional_query_rows(query_jsonl, primary_assets)
    if not query_rows:
        raise RuntimeError("No conditional query rows were recovered from the selected analysis run.")

    query_df = pd.DataFrame(query_rows)
    subitem_df, dataset_model_df = _build_dataset_model_scores(query_df)
    model_summary_df = _build_model_summary(dataset_model_df)
    prefix_summary_df = _build_prefix_summary(dataset_model_df)
    prefix_model_summaries = _build_prefix_model_summaries(dataset_model_df)
    dataset_summary_df = _build_dataset_summary(dataset_model_df)
    heatmap_df = _build_heatmap_data(dataset_model_df)
    prefix_plot_df = _build_prefix_plot_data(prefix_summary_df)
    model_subitem_heatmap_df = build_model_subitem_heatmap_df(
        model_summary_df,
        model_id_col="model_id",
        model_order=MODEL_ORDER,
        subitem_specs=[
            (subitem_id, SUBITEM_LABELS[subitem_id], f"{subitem_id}__mean")
            for subitem_id in SUBITEM_ORDER
        ],
        summary_row_spec=("family_mean", "Family mean", "conditional_dependency_structure_score__mean"),
    )

    _write_csv(pd.DataFrame(duplicate_audit_rows), DATA_DIR / "duplicate_asset_audit.csv")
    _write_csv(query_df, DATA_DIR / "conditional_query_rows.csv")
    _write_csv(subitem_df, DATA_DIR / "dataset_model_subitems.csv")
    _write_csv(dataset_model_df, DATA_DIR / "dataset_model_scores.csv")
    _write_csv(model_summary_df, DATA_DIR / "model_summary.csv")
    _write_csv(prefix_summary_df, DATA_DIR / "prefix_summary.csv")
    for prefix, summary_df in prefix_model_summaries.items():
        _write_csv(summary_df, DATA_DIR / f"model_summary__{prefix}.csv")
    _write_csv(dataset_summary_df, DATA_DIR / "dataset_summary.csv")
    _write_csv(heatmap_df, DATA_DIR / "dataset_model_heatmap.csv")
    _write_csv(prefix_plot_df, DATA_DIR / "prefix_plot_data.csv")
    _write_csv(model_subitem_heatmap_df, DATA_DIR / "model_subitem_heatmap.csv")

    subgroup_tradeoff_tex = FIG_DIR / "conditional_subgroup_tradeoff_scatter_main.tex"
    subgroup_tradeoff_pdf = FIG_DIR / "conditional_subgroup_tradeoff_scatter_main.pdf"
    subgroup_tradeoff_png = FIG_DIR / "conditional_subgroup_tradeoff_scatter_main.png"
    strength_bridge_tex = FIG_DIR / "conditional_strength_vs_subgroup_bridge.tex"
    strength_bridge_pdf = FIG_DIR / "conditional_strength_vs_subgroup_bridge.pdf"
    strength_bridge_png = FIG_DIR / "conditional_strength_vs_subgroup_bridge.png"
    dumbbell_tex = FIG_DIR / "conditional_branch_dumbbell_main.tex"
    dumbbell_pdf = FIG_DIR / "conditional_branch_dumbbell_main.pdf"
    dumbbell_png = FIG_DIR / "conditional_branch_dumbbell_main.png"
    heatmap_tex = FIG_DIR / "conditional_dataset_model_heatmap_appendix.tex"
    heatmap_pdf = FIG_DIR / "conditional_dataset_model_heatmap_appendix.pdf"
    heatmap_png = FIG_DIR / "conditional_dataset_model_heatmap_appendix.png"
    prefix_tex = FIG_DIR / "conditional_prefix_bars_appendix.tex"
    prefix_pdf = FIG_DIR / "conditional_prefix_bars_appendix.pdf"
    prefix_png = FIG_DIR / "conditional_prefix_bars_appendix.png"
    model_subitem_heatmap_tex = FIG_DIR / "conditional_model_subitem_heatmap_appendix.tex"
    model_subitem_heatmap_pdf = FIG_DIR / "conditional_model_subitem_heatmap_appendix.pdf"
    model_subitem_heatmap_png = FIG_DIR / "conditional_model_subitem_heatmap_appendix.png"
    grouped_bars_tex = FIG_DIR / "conditional_family_subitem_bars_appendix.tex"
    grouped_bars_pdf = FIG_DIR / "conditional_family_subitem_bars_appendix.pdf"
    grouped_bars_png = FIG_DIR / "conditional_family_subitem_bars_appendix.png"
    prefix_detail_paths: list[Path] = []

    _write_scatter_tex(
        model_summary_df,
        x_metric="slice_level_consistency",
        y_metric="direction_consistency",
        x_label="Slice-level consistency",
        y_label="Direction consistency",
        title="Conditional subgroup trade-off across canonical branches",
        path=subgroup_tradeoff_tex,
        note_lines=[
            "Main paper-facing view.",
            "Axes isolate the two subgroup-sensitive conditional branches; both directions include 95% CI error bars.",
        ],
    )
    _write_scatter_tex(
        model_summary_df,
        x_metric="dependency_strength_similarity",
        y_metric="conditional_subgroup_score",
        x_label="Dependency strength similarity",
        y_label="Derived subgroup score",
        title="How much conditional strength survives into subgroup slices",
        path=strength_bridge_tex,
        note_lines=[
            "Derived subgroup score = mean(direction consistency, slice-level consistency).",
            "This figure is analytical only and does not replace the frozen canonical conditional score.",
        ],
    )
    _write_dumbbell_tex(model_summary_df, dumbbell_tex)
    _write_heatmap_tex(heatmap_df, heatmap_tex)
    _write_prefix_bar_tex(prefix_plot_df, prefix_tex)
    write_model_subitem_heatmap_tex(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        title="Conditional model-subitem heatmap",
        colorbar_title="Mean score",
        path=model_subitem_heatmap_tex,
    )
    write_model_subitem_grouped_bar_tex(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        model_color_map=MODEL_COLORS,
        title="Conditional family and subitem bars",
        y_label="Score",
        path=grouped_bars_tex,
    )
    for prefix in ["c", "m", "n"]:
        summary_df = prefix_model_summaries.get(prefix)
        if summary_df is None or summary_df.empty:
            continue
        if prefix == "c":
            prefix_bar_tex = FIG_DIR / "conditional_subgroup_tradeoff_scatter__c.tex"
            prefix_bar_pdf = FIG_DIR / "conditional_subgroup_tradeoff_scatter__c.pdf"
            prefix_bar_png = FIG_DIR / "conditional_subgroup_tradeoff_scatter__c.png"
            _write_prefix_subitem_bar_tex(summary_df, prefix, prefix_bar_tex)
            _plot_prefix_subitem_bar_preview(summary_df, prefix, prefix_bar_pdf, prefix_bar_png)
            prefix_detail_paths.extend([prefix_bar_tex, prefix_bar_pdf, prefix_bar_png])
        else:
            prefix_tradeoff_tex = FIG_DIR / f"conditional_subgroup_tradeoff_scatter__{prefix}.tex"
            prefix_tradeoff_pdf = FIG_DIR / f"conditional_subgroup_tradeoff_scatter__{prefix}.pdf"
            prefix_tradeoff_png = FIG_DIR / f"conditional_subgroup_tradeoff_scatter__{prefix}.png"
            _write_scatter_tex(
                summary_df,
                x_metric="slice_level_consistency",
                y_metric="direction_consistency",
                x_label="Slice-level consistency",
                y_label="Direction consistency",
                title=f"Conditional subgroup trade-off within {PREFIX_LABELS.get(prefix, prefix.upper())} datasets",
                path=prefix_tradeoff_tex,
                note_lines=_prefix_scatter_note(prefix),
            )
            _plot_scatter_preview(
                summary_df,
                x_metric="slice_level_consistency",
                y_metric="direction_consistency",
                x_label="Slice-level consistency",
                y_label="Direction consistency",
                title=f"Conditional subgroup trade-off within {PREFIX_LABELS.get(prefix, prefix.upper())} datasets",
                pdf_path=prefix_tradeoff_pdf,
                png_path=prefix_tradeoff_png,
            )
            prefix_detail_paths.extend([prefix_tradeoff_tex, prefix_tradeoff_pdf, prefix_tradeoff_png])

    _plot_scatter_preview(
        model_summary_df,
        x_metric="slice_level_consistency",
        y_metric="direction_consistency",
        x_label="Slice-level consistency",
        y_label="Direction consistency",
        title="Conditional subgroup trade-off across canonical branches",
        pdf_path=subgroup_tradeoff_pdf,
        png_path=subgroup_tradeoff_png,
    )
    _plot_scatter_preview(
        model_summary_df,
        x_metric="dependency_strength_similarity",
        y_metric="conditional_subgroup_score",
        x_label="Dependency strength similarity",
        y_label="Derived subgroup score",
        title="How much conditional strength survives into subgroup slices",
        pdf_path=strength_bridge_pdf,
        png_path=strength_bridge_png,
    )
    _plot_dumbbell_preview(model_summary_df, dumbbell_pdf, dumbbell_png)
    _plot_heatmap_preview(heatmap_df, heatmap_pdf, heatmap_png)
    _plot_prefix_bar_preview(prefix_plot_df, prefix_pdf, prefix_png)
    plot_model_subitem_heatmap_preview(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        title="Conditional model-subitem heatmap",
        pdf_path=model_subitem_heatmap_pdf,
        png_path=model_subitem_heatmap_png,
    )
    plot_model_subitem_grouped_bar_preview(
        model_subitem_heatmap_df,
        model_order=MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        model_color_map=MODEL_COLORS,
        title="Conditional family and subitem bars",
        y_label="Score",
        pdf_path=grouped_bars_pdf,
        png_path=grouped_bars_png,
    )

    _write_model_table_tex(model_summary_df, TABLE_DIR / "conditional_model_summary_generated.tex")
    _build_report(run_dir, query_df, dataset_model_df, model_summary_df, prefix_summary_df, dataset_summary_df)
    _build_readme(run_dir)

    compile_notes = {
        "subgroup_tradeoff": _try_compile_tex(subgroup_tradeoff_tex),
        "strength_bridge": _try_compile_tex(strength_bridge_tex),
        "dumbbell": _try_compile_tex(dumbbell_tex),
        "heatmap": _try_compile_tex(heatmap_tex),
        "prefix_bars": _try_compile_tex(prefix_tex),
        "model_subitem_heatmap": _try_compile_tex(model_subitem_heatmap_tex),
        "family_subitem_bars": _try_compile_tex(grouped_bars_tex),
    }
    prefix_bar_tex = FIG_DIR / "conditional_subgroup_tradeoff_scatter__c.tex"
    if prefix_bar_tex.exists():
        compile_notes["prefix_bar_c"] = _try_compile_tex(prefix_bar_tex)
    for prefix in ["m", "n"]:
        prefix_tradeoff_tex = FIG_DIR / f"conditional_subgroup_tradeoff_scatter__{prefix}.tex"
        if prefix_tradeoff_tex.exists():
            compile_notes[f"prefix_tradeoff_{prefix}"] = _try_compile_tex(prefix_tradeoff_tex)

    final_files = [
        subgroup_tradeoff_tex,
        subgroup_tradeoff_pdf,
        subgroup_tradeoff_png,
        strength_bridge_tex,
        strength_bridge_pdf,
        strength_bridge_png,
        dumbbell_tex,
        dumbbell_pdf,
        dumbbell_png,
        heatmap_tex,
        heatmap_pdf,
        heatmap_png,
        prefix_tex,
        prefix_pdf,
        prefix_png,
        model_subitem_heatmap_tex,
        model_subitem_heatmap_pdf,
        model_subitem_heatmap_png,
        grouped_bars_tex,
        grouped_bars_pdf,
        grouped_bars_png,
        TABLE_DIR / "conditional_model_summary_generated.tex",
        DATA_DIR / "model_summary.csv",
        DATA_DIR / "prefix_summary.csv",
        OUTPUT_ROOT / "analysis_report.md",
    ]
    final_files.extend(prefix_detail_paths)
    for prefix in ["c", "m", "n"]:
        csv_path = DATA_DIR / f"model_summary__{prefix}.csv"
        if csv_path.exists():
            final_files.append(csv_path)
    must_do_aliases = {
        "conditional_tradeoff_scatter_main.tex": subgroup_tradeoff_tex,
        "conditional_tradeoff_scatter_main.pdf": subgroup_tradeoff_pdf,
        "conditional_tradeoff_scatter_main.png": subgroup_tradeoff_png,
        "conditional_prefix_bars_appendix.tex": prefix_tex,
        "conditional_prefix_bars_appendix.pdf": prefix_pdf,
        "conditional_prefix_bars_appendix.png": prefix_png,
        "conditional_dataset_model_heatmap_appendix.tex": heatmap_tex,
        "conditional_dataset_model_heatmap_appendix.pdf": heatmap_pdf,
        "conditional_dataset_model_heatmap_appendix.png": heatmap_png,
        "conditional_model_subitem_heatmap_appendix.tex": model_subitem_heatmap_tex,
        "conditional_model_subitem_heatmap_appendix.pdf": model_subitem_heatmap_pdf,
        "conditional_model_subitem_heatmap_appendix.png": model_subitem_heatmap_png,
        "conditional_family_subitem_bars_appendix.tex": grouped_bars_tex,
        "conditional_family_subitem_bars_appendix.pdf": grouped_bars_pdf,
        "conditional_family_subitem_bars_appendix.png": grouped_bars_png,
    }
    version_tag = TARGET_SQL_SOURCE_VERSION
    if publish_final:
        _copy_final_artifacts(final_files, must_do_aliases, version_tag=version_tag)

        final_readme = render_final_readme(
            title="Conditional Breakdown Final",
            summary=f"This directory contains the paper-facing conditional breakdown artifacts for `{sql_source_label(version_tag)}` (`{version_tag}`), with the standardized must-do bundle mirrored into `final/must_do/` and `final/{version_tag}/must_do/`.",
            primary_files=[
                *[versioned_name(name, version_tag) for name in [
                    "conditional_tradeoff_scatter_main.tex",
                    "conditional_tradeoff_scatter_main.pdf",
                    "conditional_tradeoff_scatter_main.png",
                    "conditional_strength_vs_subgroup_bridge.tex",
                    "conditional_strength_vs_subgroup_bridge.pdf",
                    "conditional_strength_vs_subgroup_bridge.png",
                    "conditional_branch_dumbbell_main.tex",
                    "conditional_branch_dumbbell_main.pdf",
                    "conditional_branch_dumbbell_main.png",
                    "conditional_prefix_bars_appendix.tex",
                    "conditional_prefix_bars_appendix.pdf",
                    "conditional_prefix_bars_appendix.png",
                    "conditional_dataset_model_heatmap_appendix.tex",
                    "conditional_dataset_model_heatmap_appendix.pdf",
                    "conditional_dataset_model_heatmap_appendix.png",
                    "conditional_model_subitem_heatmap_appendix.tex",
                    "conditional_model_subitem_heatmap_appendix.pdf",
                    "conditional_model_subitem_heatmap_appendix.png",
                    "conditional_family_subitem_bars_appendix.tex",
                    "conditional_family_subitem_bars_appendix.pdf",
                    "conditional_family_subitem_bars_appendix.png",
                    "conditional_model_summary_generated.tex",
                    "model_summary.csv",
                ]],
            ],
            must_do_files=[versioned_name(name, version_tag) for name in must_do_aliases.keys()],
            support_files=[
                *[versioned_name(name, version_tag) for name in [
                    "conditional_subgroup_tradeoff_scatter_main.tex",
                    "conditional_subgroup_tradeoff_scatter_main.pdf",
                    "conditional_subgroup_tradeoff_scatter_main.png",
                    "analysis_report.md",
                    "prefix_summary.csv",
                ]],
            ],
            notes=[
                f"The active SQL source for this final bundle is `{sql_source_label(version_tag)}` (`{version_tag}`).",
                "The `.tex` files are standalone LaTeX sources. The `.pdf/.png` files are immediate previews for reading in the current environment.",
            ],
        )
        (FINAL_DIR / "README.md").write_text(final_readme, encoding="utf-8")

    manifest = {
        "task": "conditional_breakdown",
        "sql_source_version": version_tag,
        "sql_source_label": sql_source_label(version_tag),
        "source_analysis_run": run_dir.name,
        "excluded_models": sorted(EXCLUDED_MODELS),
        "included_models": [model_id for model_id in MODEL_ORDER if model_id in set(model_summary_df["model_id"].tolist())],
        "dataset_panel_count": int(dataset_model_df.shape[0]),
        "query_row_count": int(query_df.shape[0]),
        "compile_notes": {key: {"success": value[0], "note": value[1]} for key, value in compile_notes.items()},
        "publish_final": bool(publish_final),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conditional breakdown artifacts from analysis outputs.")
    parser.add_argument("--analysis-run-dir", type=Path, default=None, help="Optional analysis run dir.")
    parser.add_argument("--skip-final-publish", action="store_true", help="Skip writing shared final outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_conditional_breakdown(
        analysis_run_dir=args.analysis_run_dir,
        publish_final=not args.skip_final_publish,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
