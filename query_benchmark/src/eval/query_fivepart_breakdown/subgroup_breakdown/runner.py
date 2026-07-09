#!/usr/bin/env python3
"""Decompose subgroup_structure into its canonical subitems and export paper-facing artifacts."""

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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.analytics_contract import annotate_query_row_with_contract
from src.eval.common import (
    DEFAULT_SQL_SOURCE_VERSION,
    resolve_requested_sql_source_version,
    resolve_task_run_dir_for_sql_source,
    sql_source_label,
)
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
ANALYSIS_ROOT = EVALUATION_ROOT / "analysis"
OUTPUT_ROOT = EVALUATION_ROOT / "query_fivepart_breakdown" / "subgroup_breakdown"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
FINAL_DIR = OUTPUT_ROOT / "final"
TARGET_SQL_SOURCE_VERSION = resolve_requested_sql_source_version("analysis", DEFAULT_SQL_SOURCE_VERSION)

TARGET_FAMILY = "subgroup_structure"
REAL_MODEL_ID = "real"
SUBITEM_ORDER = [
    "internal_profile_stability",
    "subgroup_size_stability",
]
SUBITEM_LABELS = {
    "internal_profile_stability": "Internal profile stability",
    "subgroup_size_stability": "Subgroup size stability",
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
PREFIX_COLORS = {"c": "#577590", "m": "#43AA8B", "n": "#F3722C"}
EXCLUDED_MODELS = {"cdtd", "codi", "goggle"}
MUST_DO_MODEL_ORDER = [
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
MODEL_ALIASES = {"rtf": "realtabformer"}
SERVER_PRIORITY = {"rtx_5090": 2, "rtx_pro_6000": 1}
ROOT_PRIORITY = {"SynOutput-5090": 2, "SynOutput": 1}
TIMESTAMP_RE = re.compile(r"(20\d{6}_\d{6})")
PROBE_SUBITEM_RE = re.compile(r"\bprobe\s+([a-z_]+)\b", re.IGNORECASE)


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
    if str(model_id).strip().lower() == REAL_MODEL_ID:
        return (0, label)
    return (1, label.lower())


def _sorted_model_ids(model_ids: list[str] | set[str]) -> list[str]:
    return sorted({str(item) for item in model_ids}, key=_model_sort_key)


def _dataset_prefix(dataset_id: str) -> str:
    return str(dataset_id or "").strip().lower()[:1]


def _dataset_sort_key(dataset_id: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", str(dataset_id).strip())
    if not match:
        return (99, 10**9, str(dataset_id))
    prefix, number = match.groups()
    return ({"c": 0, "m": 1, "n": 2}.get(prefix.lower(), 50), int(number), str(dataset_id))


def _resolve_subitem_id(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "")
    match = PROBE_SUBITEM_RE.search(question)
    if match:
        probed = match.group(1).strip()
        if probed in SUBITEM_ORDER:
            return probed

    canonical = str(row.get("canonical_subitem_id") or "").strip()
    if canonical in SUBITEM_ORDER:
        return canonical

    annotated = annotate_query_row_with_contract(dict(row))
    canonical = str(annotated.get("canonical_subitem_id") or "").strip()
    if canonical in SUBITEM_ORDER:
        return canonical
    return ""


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


def _find_primary_analysis_run(sql_source_version: str = TARGET_SQL_SOURCE_VERSION) -> Path:
    resolved = resolve_task_run_dir_for_sql_source("analysis", sql_source_version)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(f"No analysis run found for sql_source_version={sql_source_version!r}.")


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


def _stream_subgroup_query_rows(
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
            subitem_id = _resolve_subitem_id(annotated)
            if subitem_id not in SUBITEM_ORDER:
                continue
            if not annotated.get("canonical_subitem_id"):
                annotated = annotate_query_row_with_contract(annotated)
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
                    "intended_facet_id": str(annotated.get("intended_facet_id") or ""),
                    "variant_semantic_role": str(annotated.get("variant_semantic_role") or ""),
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


def _build_dataset_model_scores(query_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subitems = (
        query_df.groupby(
            ["dataset_id", "dataset_prefix", "model_id", "model_label", "subitem_id", "subitem_label"],
            as_index=False,
        )
        .agg(query_count=("query_id", "count"), subitem_score=("query_score", "mean"))
        .reset_index(drop=True)
    )
    subitems["dataset_sort"] = subitems["dataset_id"].map(_dataset_sort_key)
    subitems["model_sort"] = subitems["model_id"].map(_model_sort_key)
    subitems = subitems.sort_values(["dataset_sort", "model_sort", "subitem_id"]).drop(columns=["dataset_sort", "model_sort"]).reset_index(drop=True)
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
    wide["internal_profile_stability"] = pd.to_numeric(wide["internal_profile_stability"], errors="coerce")
    wide["subgroup_size_stability"] = pd.to_numeric(wide["subgroup_size_stability"], errors="coerce")
    wide["subgroup_structure_score"] = wide[SUBITEM_ORDER].mean(axis=1, skipna=True)
    wide["profile_minus_size"] = wide["internal_profile_stability"] - wide["subgroup_size_stability"]
    wide["active_subitem_count"] = wide[SUBITEM_ORDER].notna().sum(axis=1)
    wide["dataset_sort"] = wide["dataset_id"].map(_dataset_sort_key)
    wide["model_sort"] = wide["model_id"].map(_model_sort_key)
    wide = wide.sort_values(["dataset_sort", "model_sort"]).drop(columns=["dataset_sort", "model_sort"]).reset_index(drop=True)
    return subitems, wide


def _build_model_summary(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, group in dataset_model_df.groupby("model_id", sort=False):
        payload = {
            "model_id": model_id,
            "model_label": _model_label(model_id),
            "dataset_count": int(group["dataset_id"].nunique()),
            "dataset_prefixes": ",".join(sorted(group["dataset_prefix"].dropna().astype(str).unique())),
        }
        for metric in SUBITEM_ORDER + ["subgroup_structure_score", "profile_minus_size"]:
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
                "dataset_count": int(group["dataset_id"].nunique()),
                "internal_profile_stability": round(float(group["internal_profile_stability"].mean()), 6),
                "subgroup_size_stability": round(float(group["subgroup_size_stability"].mean()), 6),
                "subgroup_structure_score": round(float(group["subgroup_structure_score"].mean()), 6),
                "profile_minus_size": round(float(group["profile_minus_size"].mean()), 6),
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
                "internal_profile_stability": round(float(group["internal_profile_stability"].mean()), 6),
                "subgroup_size_stability": round(float(group["subgroup_size_stability"].mean()), 6),
                "subgroup_structure_score": round(float(group["subgroup_structure_score"].mean()), 6),
                "profile_minus_size": round(float(group["profile_minus_size"].mean()), 6),
                "score_std_across_models": round(float(group["subgroup_structure_score"].std(ddof=1)) if len(group) > 1 else 0.0, 6),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["sort_key"] = out["dataset_id"].map(_dataset_sort_key)
    out = out.sort_values(["sort_key"]).drop(columns=["sort_key"])
    return out.reset_index(drop=True)


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
            r"\usepackage{xcolor}",
            r"\usetikzlibrary{patterns}",
            r"\pgfplotsset{compat=1.18}",
            "",
        ]
    )


def _write_tradeoff_tex(model_summary_df: pd.DataFrame, path: Path) -> None:
    model_labels = [_escape_tex(str(item)) for item in model_summary_df["model_label"].tolist()]
    color_defs = [
        rf"\definecolor{{model{row.model_id}}}{{HTML}}{{{MODEL_COLORS[row.model_id].replace('#', '')}}}"
        for row in model_summary_df.itertuples()
        if row.model_id in MODEL_COLORS
    ]
    body = [_tex_preamble(), *color_defs, r"\begin{document}", r"\begin{tikzpicture}", r"\begin{axis}["]
    body.extend(
        [
            r"width=14.6cm,",
            r"height=8.4cm,",
            r"ymin=0.0, ymax=1.03,",
            r"xlabel={Model},",
            r"ylabel={Score},",
            r"ymajorgrids,",
            r"grid style={draw=gray!20},",
            r"major grid style={draw=gray!30},",
            r"axis line style={draw=black!70},",
            r"tick style={draw=black!70},",
            rf"symbolic x coords={{{','.join(model_labels)}}},",
            r"xtick=data,",
            r"xticklabel style={rotate=45, anchor=east, font=\scriptsize},",
            r"legend style={draw=none, fill=none, font=\scriptsize, at={(0.98,0.98)}, anchor=north east},",
            r"]",
        ]
    )
    for row in model_summary_df.itertuples():
        color_name = f"model{row.model_id}"
        label = _escape_tex(row.model_label)
        profile_score = float(getattr(row, "internal_profile_stability__mean"))
        size_score = float(getattr(row, "subgroup_size_stability__mean"))
        profile_err = float(getattr(row, "internal_profile_stability__ci95_radius") or 0.0)
        size_err = float(getattr(row, "subgroup_size_stability__ci95_radius") or 0.0)
        body.append(
            "\n".join(
                [
                    rf"\addplot+[ybar, bar width=6.5pt, bar shift=-3.8pt, draw={color_name}, fill={color_name},",
                    r"error bars/.cd, y dir=both, y explicit, error bar style={line width=0.8pt}]",
                    rf"coordinates {{ ({label},{profile_score:.4f}) +- (0,{profile_err:.4f}) }};",
                    rf"\addplot+[ybar, bar width=6.5pt, bar shift=3.8pt, draw={color_name}, fill=white, pattern=north east lines, pattern color={color_name},",
                    r"error bars/.cd, y dir=both, y explicit, error bar style={line width=0.8pt}]",
                    rf"coordinates {{ ({label},{size_score:.4f}) +- (0,{size_err:.4f}) }};",
                ]
            )
        )
    body.extend(
        [
            r"\addlegendimage{area legend, draw=black, fill=black}",
            r"\addlegendentry{Internal profile stability}",
            r"\addlegendimage{area legend, draw=black, fill=white, pattern=north east lines, pattern color=black}",
            r"\addlegendentry{Subgroup size stability}",
            r"\node[anchor=west, font=\scriptsize] at (rel axis cs:0.02,0.90) {$\uparrow$ better};",
        ]
    )
    body.extend([r"\end{axis}", r"\end{tikzpicture}", r"\end{document}", ""])
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
            r"width=13.2cm,",
            r"height=9.8cm,",
            r"xmin=0.15, xmax=1.03,",
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
        size_score = float(getattr(row, "subgroup_size_stability__mean"))
        profile_score = float(getattr(row, "internal_profile_stability__mean"))
        overall = float(getattr(row, "subgroup_structure_score__mean"))
        body.append(
            "\n".join(
                [
                    rf"\draw[line width=0.9pt, color={color_name}!70] (axis cs:{size_score:.4f},{y}) -- (axis cs:{profile_score:.4f},{y});",
                    rf"\addplot+[only marks, mark=square*, mark size=2.4pt, draw={color_name}, fill={color_name}] coordinates {{ ({size_score:.4f},{y}) }};",
                    rf"\addplot+[only marks, mark=*, mark size=2.6pt, draw={color_name}, fill={color_name}] coordinates {{ ({profile_score:.4f},{y}) }};",
                    rf"\addplot+[only marks, mark=diamond*, mark size=2.6pt, draw=black, fill=black] coordinates {{ ({overall:.4f},{y}) }};",
                ]
            )
        )
    body.extend(
        [
            r"\addlegendimage{only marks, mark=square*, mark size=2.4pt, draw=black, fill=black}",
            r"\addlegendentry{Subgroup size stability}",
            r"\addlegendimage{only marks, mark=*, mark size=2.6pt, draw=black, fill=black}",
            r"\addlegendentry{Internal profile stability}",
            r"\addlegendimage{only marks, mark=diamond*, mark size=2.6pt, draw=black, fill=black}",
            r"\addlegendentry{Canonical subgroup score}",
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )
    path.write_text("\n".join(body), encoding="utf-8")


def _write_prefix_tradeoff_tex(prefix_summary_df: pd.DataFrame, path: Path) -> None:
    prefixes = [item for item in ["c", "m", "n"] if item in set(prefix_summary_df["dataset_prefix"].astype(str))]
    color_defs = [
        rf"\definecolor{{model{row.model_id}}}{{HTML}}{{{MODEL_COLORS[row.model_id].replace('#', '')}}}"
        for row in prefix_summary_df.drop_duplicates(subset=["model_id"]).itertuples()
        if row.model_id in MODEL_COLORS
    ]
    body = [
        _tex_preamble(),
        r"\usepgfplotslibrary{groupplots}",
        *color_defs,
        r"\begin{document}",
        r"\begin{tikzpicture}",
        rf"\begin{{groupplot}}[group style={{group size={max(1, len(prefixes))} by 1, horizontal sep=1.6cm}},",
        r"width=5.6cm,",
        r"height=5.1cm,",
        r"xmin=0.15, xmax=1.03,",
        r"ymin=0.15, ymax=1.03,",
        r"grid=both,",
        r"grid style={draw=gray!18},",
        r"major grid style={draw=gray!28},",
        r"axis line style={draw=black!70},",
        r"tick style={draw=black!70},",
        r"legend style={draw=none, fill=none, font=\scriptsize, at={(0.5,-0.24)}, anchor=north},",
        r"xlabel={Subgroup size stability},",
        r"ylabel={Internal profile stability}]",
    ]
    for idx, prefix in enumerate(prefixes):
        body.append(rf"\nextgroupplot[title={{{prefix.upper()} prefix}}]")
        body.append(r"\addplot[black!45, dashed, domain=0.15:1.03, samples=2] {x};")
        subset = prefix_summary_df.loc[prefix_summary_df["dataset_prefix"] == prefix].copy()
        subset["model_sort"] = subset["model_id"].map(_model_sort_key)
        subset = subset.sort_values("model_sort").drop(columns=["model_sort"])
        for row in subset.itertuples():
            color_name = f"model{row.model_id}"
            body.append(
                rf"\addplot+[only marks, mark=*, mark size=2.5pt, draw={color_name}, fill={color_name}] coordinates {{ ({float(row.subgroup_size_stability):.4f},{float(row.internal_profile_stability):.4f}) }};"
            )
            if idx == 0:
                body.append(rf"\addlegendentry{{{_escape_tex(row.model_label)}}}")
        body.append(r"\node[anchor=west, font=\scriptsize] at (rel axis cs:0.02,0.98) {$\uparrow$ better \ \ $\rightarrow$ better};")
    body.extend([r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


def _write_prefix_dumbbell_tex(prefix_summary_df: pd.DataFrame, path: Path) -> None:
    prefixes = [item for item in ["c", "m", "n"] if item in set(prefix_summary_df["dataset_prefix"].astype(str))]
    color_defs = [
        rf"\definecolor{{model{row.model_id}}}{{HTML}}{{{MODEL_COLORS[row.model_id].replace('#', '')}}}"
        for row in prefix_summary_df.drop_duplicates(subset=["model_id"]).itertuples()
        if row.model_id in MODEL_COLORS
    ]
    body = [
        _tex_preamble(),
        r"\usepgfplotslibrary{groupplots}",
        *color_defs,
        r"\begin{document}",
        r"\begin{tikzpicture}",
        rf"\begin{{groupplot}}[group style={{group size={max(1, len(prefixes))} by 1, horizontal sep=1.5cm}},",
        r"width=5.6cm,",
        r"height=7.6cm,",
        r"xmin=0.15, xmax=1.03,",
        r"grid=both,",
        r"grid style={draw=gray!18},",
        r"tick style={draw=black!70},",
        r"xlabel={Score},",
        r"legend style={draw=none, fill=none, font=\scriptsize, at={(0.5,-0.2)}, anchor=north}]",
    ]
    for idx, prefix in enumerate(prefixes):
        subset = prefix_summary_df.loc[prefix_summary_df["dataset_prefix"] == prefix].copy()
        subset["model_sort"] = subset["model_id"].map(_model_sort_key)
        subset = subset.sort_values("model_sort").drop(columns=["model_sort"]).reset_index(drop=True)
        subset["sort_rank"] = range(len(subset), 0, -1)
        yticks = ",".join(str(v) for v in subset["sort_rank"].tolist())
        ylabels = ",".join(_escape_tex(item) for item in subset["model_label"].tolist())
        body.append(rf"\nextgroupplot[title={{{prefix.upper()} prefix}}, y dir=reverse, ytick={{{yticks}}}, yticklabels={{{ylabels}}}]")
        for row in subset.itertuples():
            color_name = f"model{row.model_id}"
            y = int(row.sort_rank)
            body.extend(
                [
                    rf"\draw[line width=0.9pt, color={color_name}!70] (axis cs:{float(row.subgroup_size_stability):.4f},{y}) -- (axis cs:{float(row.internal_profile_stability):.4f},{y});",
                    rf"\addplot+[only marks, mark=square*, mark size=2.3pt, draw={color_name}, fill={color_name}] coordinates {{ ({float(row.subgroup_size_stability):.4f},{y}) }};",
                    rf"\addplot+[only marks, mark=*, mark size=2.5pt, draw={color_name}, fill={color_name}] coordinates {{ ({float(row.internal_profile_stability):.4f},{y}) }};",
                    rf"\addplot+[only marks, mark=diamond*, mark size=2.5pt, draw=black, fill=black] coordinates {{ ({float(row.subgroup_structure_score):.4f},{y}) }};",
                ]
            )
        if idx == 0:
            body.extend(
                [
                    r"\addlegendimage{only marks, mark=square*, mark size=2.3pt, draw=black, fill=black}",
                    r"\addlegendentry{Subgroup size stability}",
                    r"\addlegendimage{only marks, mark=*, mark size=2.5pt, draw=black, fill=black}",
                    r"\addlegendentry{Internal profile stability}",
                    r"\addlegendimage{only marks, mark=diamond*, mark size=2.5pt, draw=black, fill=black}",
                    r"\addlegendentry{Canonical subgroup score}",
                ]
            )
    body.extend([r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


def _write_heatmap_tex(heatmap_df: pd.DataFrame, path: Path) -> None:
    matrix = heatmap_df.copy()
    model_cols = [item for item in _sorted_model_ids(set(matrix.columns) - {"dataset_id"}) if item in matrix.columns]
    if not model_cols:
        path.write_text("", encoding="utf-8")
        return
    display = matrix[["dataset_id"] + model_cols].copy()
    display = display.fillna("")
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
        cells = [_escape_tex(dataset_id.upper())]
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
        rf"\definecolor{{model{row.model_id}}}{{HTML}}{{{MODEL_COLORS[row.model_id].replace('#', '')}}}"
        for row in prefix_plot_df.itertuples()
        if row.model_id in MODEL_COLORS
    ]
    x_labels = ",".join(_escape_tex(str(item)) for item in prefix_plot_df["model_label"].tolist())
    body = [
        _tex_preamble(),
        r"\usepgfplotslibrary{groupplots}",
        *color_defs,
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"group style={group size=3 by 1, horizontal sep=1.15cm},",
        r"width=5.0cm,",
        r"height=7.0cm,",
        r"ymin=0.0, ymax=1.0,",
        r"ymajorgrids,",
        r"grid style={draw=gray!20},",
        r"major grid style={draw=gray!30},",
        rf"symbolic x coords={{{x_labels}}},",
        r"xtick=data,",
        r"x tick label style={rotate=45, anchor=east, font=\scriptsize},",
        r"tick style={draw=black!70},",
        r"axis line style={draw=black!70},",
        r"]",
    ]
    for idx, prefix in enumerate(["c", "m", "n"]):
        if prefix not in prefix_plot_df.columns:
            continue
        ylabel = "Subgroup structure score" if idx == 0 else ""
        title = {"c": "Categorical", "m": "Mixed", "n": "Numerical"}[prefix]
        body.append(rf"\nextgroupplot[title={{{title}}}, ylabel={{{ylabel}}}]")
        for row in prefix_plot_df.itertuples():
            value = getattr(row, prefix)
            if pd.isna(value):
                continue
            color_name = f"model{row.model_id}"
            body.append(
                rf"\addplot+[ybar, bar width=7.0pt, draw={color_name}, fill={color_name}] "
                rf"coordinates {{ ({_escape_tex(row.model_label)},{float(value):.4f}) }};"
            )
    body.extend([r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


def _plot_tradeoff_preview(model_summary_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    ordered = model_summary_df.copy().reset_index(drop=True)
    x = list(range(len(ordered)))
    width = 0.36
    colors = [MODEL_COLORS[row.model_id] for row in ordered.itertuples()]
    profile_scores = [float(getattr(row, "internal_profile_stability__mean")) for row in ordered.itertuples()]
    size_scores = [float(getattr(row, "subgroup_size_stability__mean")) for row in ordered.itertuples()]
    profile_err = [float(getattr(row, "internal_profile_stability__ci95_radius") or 0.0) for row in ordered.itertuples()]
    size_err = [float(getattr(row, "subgroup_size_stability__ci95_radius") or 0.0) for row in ordered.itertuples()]

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    ax.bar(
        [item - width / 2 for item in x],
        profile_scores,
        width=width,
        color=colors,
        edgecolor=colors,
        yerr=profile_err,
        capsize=3.0,
        label="Internal profile stability",
    )
    ax.bar(
        [item + width / 2 for item in x],
        size_scores,
        width=width,
        color="white",
        edgecolor=colors,
        linewidth=1.1,
        hatch="////",
        yerr=size_err,
        capsize=3.0,
        label="Subgroup size stability",
    )
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Subgroup canonical branch scores by model")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["model_label"].tolist(), rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.24)
    ax.text(0.02, 0.98, "↑ better", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dumbbell_preview(model_summary_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    ordered = model_summary_df.copy().reset_index(drop=True)
    y = list(range(len(ordered)))
    fig, ax = plt.subplots(figsize=(9.0, max(5.8, 0.48 * len(ordered) + 1.5)))
    for idx, row in enumerate(ordered.itertuples()):
        color = MODEL_COLORS[row.model_id]
        size_score = float(getattr(row, "subgroup_size_stability__mean"))
        profile_score = float(getattr(row, "internal_profile_stability__mean"))
        overall = float(getattr(row, "subgroup_structure_score__mean"))
        ax.plot([size_score, profile_score], [idx, idx], color=color, linewidth=2.1, alpha=0.75, zorder=1)
        ax.scatter(size_score, idx, marker="s", s=52, color=color, zorder=3)
        ax.scatter(profile_score, idx, marker="o", s=56, color=color, zorder=3)
        ax.scatter(overall, idx, marker="D", s=42, color="black", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(0.15, 1.03)
    ax.set_xlabel("Score")
    ax.set_title("Subgroup decomposition by model")
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.scatter([], [], marker="s", s=52, color="black", label="Subgroup size stability")
    ax.scatter([], [], marker="o", s=56, color="black", label="Internal profile stability")
    ax.scatter([], [], marker="D", s=42, color="black", label="Canonical subgroup score")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefix_tradeoff_preview(prefix_summary_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    prefixes = [item for item in ["c", "m", "n"] if item in set(prefix_summary_df["dataset_prefix"].astype(str))]
    fig, axes = plt.subplots(1, len(prefixes), figsize=(5.7 * max(1, len(prefixes)), 5.2), squeeze=False)
    legend_handles = None
    for ax, prefix in zip(axes[0], prefixes):
        ax.plot([0.15, 1.03], [0.15, 1.03], linestyle="--", color="#888888", linewidth=1.0, zorder=1)
        subset = prefix_summary_df.loc[prefix_summary_df["dataset_prefix"] == prefix].copy()
        subset["model_sort"] = subset["model_id"].map(_model_sort_key)
        subset = subset.sort_values("model_sort").drop(columns=["model_sort"])
        handles = []
        for row in subset.itertuples():
            color = MODEL_COLORS[row.model_id]
            ax.scatter(float(row.subgroup_size_stability), float(row.internal_profile_stability), s=55, color=color, zorder=3)
            handles.append(
                plt.Line2D([0], [0], marker="o", linestyle="", markersize=6.8, markerfacecolor=color, markeredgecolor=color, label=row.model_label)
            )
        if legend_handles is None:
            legend_handles = handles
        ax.set_xlim(0.15, 1.03)
        ax.set_ylim(0.15, 1.03)
        ax.set_title(f"{prefix.upper()} prefix")
        ax.set_xlabel("Subgroup size stability")
        ax.grid(True, linestyle="--", alpha=0.22)
        ax.text(0.02, 0.98, "↑ better   → better", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    axes[0][0].set_ylabel("Internal profile stability")
    if legend_handles:
        fig.legend(handles=legend_handles, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Subgroup trade-off by dataset prefix")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefix_dumbbell_preview(prefix_summary_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    prefixes = [item for item in ["c", "m", "n"] if item in set(prefix_summary_df["dataset_prefix"].astype(str))]
    fig, axes = plt.subplots(1, len(prefixes), figsize=(5.8 * max(1, len(prefixes)), 7.6), squeeze=False)
    for ax, prefix in zip(axes[0], prefixes):
        subset = prefix_summary_df.loc[prefix_summary_df["dataset_prefix"] == prefix].copy()
        subset["model_sort"] = subset["model_id"].map(_model_sort_key)
        subset = subset.sort_values("model_sort").drop(columns=["model_sort"]).reset_index(drop=True)
        y = list(range(len(subset)))
        for idx, row in enumerate(subset.itertuples()):
            color = MODEL_COLORS[row.model_id]
            ax.plot(
                [float(row.subgroup_size_stability), float(row.internal_profile_stability)],
                [idx, idx],
                color=color,
                linewidth=2.0,
                alpha=0.75,
                zorder=1,
            )
            ax.scatter(float(row.subgroup_size_stability), idx, marker="s", s=48, color=color, zorder=3)
            ax.scatter(float(row.internal_profile_stability), idx, marker="o", s=52, color=color, zorder=3)
            ax.scatter(float(row.subgroup_structure_score), idx, marker="D", s=38, color="black", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(subset["model_label"].tolist())
        ax.invert_yaxis()
        ax.set_xlim(0.15, 1.03)
        ax.set_title(f"{prefix.upper()} prefix")
        ax.set_xlabel("Score")
        ax.grid(axis="x", linestyle="--", alpha=0.22)
    axes[0][0].scatter([], [], marker="s", s=48, color="black", label="Subgroup size stability")
    axes[0][0].scatter([], [], marker="o", s=52, color="black", label="Internal profile stability")
    axes[0][0].scatter([], [], marker="D", s=38, color="black", label="Canonical subgroup score")
    fig.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Subgroup decomposition by dataset prefix")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_preview(heatmap_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    matrix = heatmap_df.copy()
    model_cols = [item for item in _sorted_model_ids(set(matrix.columns) - {"dataset_id"}) if item in matrix.columns]
    ordered = matrix[["dataset_id"] + model_cols].copy()
    values = ordered[model_cols].to_numpy(dtype=float)
    fig_height = max(8.0, 0.20 * len(ordered) + 1.8)
    fig, ax = plt.subplots(figsize=(10.4, fig_height))
    image = ax.imshow(values, aspect="auto", vmin=0.15, vmax=1.0, cmap=get_heatmap_cmap())
    ax.set_xticks(range(len(model_cols)))
    ax.set_xticklabels([_model_label(item) for item in model_cols], rotation=45, ha="right")
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered["dataset_id"].str.upper().tolist(), fontsize=8)
    ax.set_title("Dataset-model subgroup score heatmap")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Subgroup structure score")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_prefix_bar_preview(prefix_plot_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    prefixes = ["c", "m", "n"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.2), sharey=True)
    for ax, prefix in zip(axes, prefixes):
        if prefix not in prefix_plot_df.columns:
            continue
        values = pd.to_numeric(prefix_plot_df[prefix], errors="coerce")
        colors = [MODEL_COLORS.get(model_id, "#777777") for model_id in prefix_plot_df["model_id"]]
        ax.bar(range(len(prefix_plot_df)), values, color=colors)
        ax.set_title({"c": "Categorical", "m": "Mixed", "n": "Numerical"}[prefix])
        ax.set_xticks(range(len(prefix_plot_df)))
        ax.set_xticklabels(prefix_plot_df["model_label"], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.22)
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Subgroup structure score")
    fig.suptitle("Subgroup score by dataset family prefix")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_model_table_tex(model_summary_df: pd.DataFrame, path: Path) -> None:
    ordered = model_summary_df.copy()
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Subgroup score & Internal profile & Subgroup size & Datasets \\",
        r"\midrule",
    ]
    for row in ordered.itertuples():
        lines.append(
            (
                f"{_escape_tex(row.model_label)} & "
                f"{float(getattr(row, 'subgroup_structure_score__mean')):.3f} & "
                f"{float(getattr(row, 'internal_profile_stability__mean')):.3f} & "
                f"{float(getattr(row, 'subgroup_size_stability__mean')):.3f} & "
                f"{int(getattr(row, 'dataset_count'))} \\\\"
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_heatmap_data(dataset_model_df: pd.DataFrame) -> pd.DataFrame:
    heatmap = (
        dataset_model_df.pivot_table(index="dataset_id", columns="model_id", values="subgroup_structure_score", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if heatmap.empty:
        return heatmap
    heatmap["sort_key"] = heatmap["dataset_id"].map(_dataset_sort_key)
    heatmap = heatmap.sort_values(["sort_key"]).drop(columns=["sort_key"])
    return heatmap.reset_index(drop=True)


def _build_prefix_plot_data(prefix_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in _sorted_model_ids(prefix_summary_df["model_id"].tolist()):
        payload: dict[str, Any] = {
            "model_id": model_id,
            "model_label": _model_label(model_id),
        }
        subset = prefix_summary_df.loc[prefix_summary_df["model_id"] == model_id]
        for prefix in ["c", "m", "n"]:
            match = subset.loc[subset["dataset_prefix"] == prefix, "subgroup_structure_score"]
            payload[prefix] = float(match.iloc[0]) if not match.empty else None
        rows.append(payload)
    return pd.DataFrame(rows)


def _build_report(
    run_dir: Path,
    query_df: pd.DataFrame,
    dataset_model_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    prefix_summary_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
) -> None:
    top_model = model_summary_df.iloc[0]
    synthetic_only = model_summary_df.loc[model_summary_df["model_id"] != REAL_MODEL_ID].reset_index(drop=True)
    best_synthetic = (
        synthetic_only.sort_values("subgroup_structure_score__mean", ascending=False).iloc[0]
        if not synthetic_only.empty
        else top_model
    )
    comparison_df = synthetic_only if not synthetic_only.empty else model_summary_df
    most_balanced = comparison_df.iloc[(comparison_df["profile_minus_size__mean"].abs()).argmin()]
    most_profile_heavy = comparison_df.sort_values("profile_minus_size__mean", ascending=False).iloc[0]
    most_size_heavy = comparison_df.sort_values("profile_minus_size__mean", ascending=True).iloc[0]
    hardest_dataset = dataset_summary_df.sort_values("subgroup_structure_score").iloc[0]
    easiest_dataset = dataset_summary_df.sort_values("subgroup_structure_score", ascending=False).iloc[0]

    lines = [
        "# Subgroup Breakdown Report",
        "",
        "## Scope",
        "",
        f"- Source analysis run: `{run_dir.name}`",
        f"- Family analyzed: `{TARGET_FAMILY}`",
        f"- Excluded models: `{', '.join(sorted(EXCLUDED_MODELS))}`",
        f"- Included models: `{model_summary_df.shape[0]}`",
        f"- Deduplicated dataset-model panels: `{dataset_model_df.shape[0]}`",
        f"- Subgroup query rows used: `{query_df.shape[0]}`",
        "",
        "## Canonical decomposition",
        "",
        "- `subgroup_structure = 0.5 * internal_profile_stability + 0.5 * subgroup_size_stability`",
        "- `internal_profile_stability` captures subgroup-internal feature/distribution behavior.",
        "- `subgroup_size_stability` captures whether subgroup support/size structure is preserved.",
        "",
        "## Main findings",
        "",
        (
            f"1. `REAL` is the expected perfect upper bound with subgroup score `1.000`. "
            f"Among synthetic generators, `{best_synthetic['model_label']}` is strongest with mean subgroup score "
            f"`{best_synthetic['subgroup_structure_score__mean']:.3f}` across `{int(best_synthetic['dataset_count'])}` datasets."
        ),
        (
            f"2. `{most_profile_heavy['model_label']}` leans most toward internal-profile preservation "
            f"(profile minus size = `{most_profile_heavy['profile_minus_size__mean']:.3f}`), while "
            f"`{most_size_heavy['model_label']}` is the clearest size-heavy model "
            f"(profile minus size = `{most_size_heavy['profile_minus_size__mean']:.3f}`)."
        ),
        (
            f"3. `{most_balanced['model_label']}` is the most balanced model between the two subgroup branches "
            f"with mean absolute branch gap `{abs(float(most_balanced['profile_minus_size__mean'])):.3f}`."
        ),
        (
            f"4. Dataset difficulty is uneven: `{hardest_dataset['dataset_id']}` is the hardest dataset on subgroup score "
            f"(`{hardest_dataset['subgroup_structure_score']:.3f}` mean across models), while "
            f"`{easiest_dataset['dataset_id']}` is the easiest (`{easiest_dataset['subgroup_structure_score']:.3f}`)."
        ),
        "",
        "## Files to use first",
        "",
        "- `figures/subgroup_tradeoff_scatter_main.pdf`",
        "- `figures/subgroup_branch_dumbbell_main.pdf`",
        "- `figures/subgroup_prefix_bars_appendix.pdf`",
        "- `tables/subgroup_model_summary_generated.tex`",
        "- `data/model_summary.csv`",
        "",
        "## Prefix note",
        "",
        f"- Prefix coverage summary rows: `{prefix_summary_df.shape[0]}`",
        "- The `c / m / n` split is exported explicitly because subgroup behavior differs by dataset family, not just by overall model average.",
        "",
    ]
    (OUTPUT_ROOT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def _build_readme(run_dir: Path) -> None:
    content = f"""# Subgroup Breakdown

This directory contains a subgroup-focused decomposition analysis built from the repository's unified `analysis` outputs.

## Inputs

- Source run: `{run_dir.name}`
- Query-level source: `{(run_dir / 'summaries' / 'analysis_query_scores__all_datasets.jsonl').relative_to(PROJECT_ROOT)}`
- Asset-level source: `{(run_dir / 'summaries' / 'analysis_asset_scores__all_datasets.csv').relative_to(PROJECT_ROOT)}`
- Canonical contract: `doc/analytics_family_subitem_contract_v1.md`

## What this analysis exports

- deduplicated dataset-model subgroup scores
- model-level subgroup summaries
- canonical subgroup decomposition figures
- paper-ready LaTeX table snippets
- final copies under `Evaluation/query_fivepart_breakdown/subgroup_breakdown/final/`

## Re-run

```bash
python src/eval/query_fivepart_breakdown/subgroup_breakdown/runner.py
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


def run_subgroup_breakdown(
    *,
    analysis_run_dir: Path | None = None,
    publish_final: bool = True,
) -> dict[str, Any]:
    _ensure_dirs()
    run_dir = analysis_run_dir.resolve() if analysis_run_dir is not None else _find_primary_analysis_run(TARGET_SQL_SOURCE_VERSION)
    summary_dir = run_dir / "summaries"
    asset_csv = summary_dir / "analysis_asset_scores__all_datasets.csv"
    query_jsonl = summary_dir / "analysis_query_scores__all_datasets.jsonl"
    if not asset_csv.exists() or not query_jsonl.exists():
        raise FileNotFoundError("Primary analysis run is missing required summary files.")

    primary_assets, duplicate_audit_rows = _load_primary_assets(asset_csv)
    query_rows = _stream_subgroup_query_rows(query_jsonl, primary_assets)
    if not query_rows:
        raise RuntimeError("No subgroup query rows were recovered from the selected analysis run.")
    query_rows = _inject_real_rows(query_rows)

    query_df = pd.DataFrame(query_rows)
    subitem_df, dataset_model_df = _build_dataset_model_scores(query_df)
    model_summary_df = _build_model_summary(dataset_model_df)
    prefix_summary_df = _build_prefix_summary(dataset_model_df)
    dataset_summary_df = _build_dataset_summary(dataset_model_df)
    heatmap_df = _build_heatmap_data(dataset_model_df)
    prefix_plot_df = _build_prefix_plot_data(prefix_summary_df)
    model_subitem_heatmap_df = build_model_subitem_heatmap_df(
        model_summary_df.loc[model_summary_df["model_id"].isin(MUST_DO_MODEL_ORDER)].copy(),
        model_id_col="model_id",
        model_order=MUST_DO_MODEL_ORDER,
        subitem_specs=[
            (subitem_id, SUBITEM_LABELS[subitem_id], f"{subitem_id}__mean")
            for subitem_id in SUBITEM_ORDER
        ],
        summary_row_spec=("family_mean", "Family mean", "subgroup_structure_score__mean"),
    )

    _write_csv(pd.DataFrame(duplicate_audit_rows), DATA_DIR / "duplicate_asset_audit.csv")
    _write_csv(query_df, DATA_DIR / "subgroup_query_rows.csv")
    _write_csv(subitem_df, DATA_DIR / "dataset_model_subitems.csv")
    _write_csv(dataset_model_df, DATA_DIR / "dataset_model_scores.csv")
    _write_csv(model_summary_df, DATA_DIR / "model_summary.csv")
    _write_csv(prefix_summary_df, DATA_DIR / "prefix_summary.csv")
    _write_csv(dataset_summary_df, DATA_DIR / "dataset_summary.csv")
    _write_csv(heatmap_df, DATA_DIR / "dataset_model_heatmap.csv")
    _write_csv(prefix_plot_df, DATA_DIR / "prefix_plot_data.csv")
    _write_csv(model_subitem_heatmap_df, DATA_DIR / "model_subitem_heatmap.csv")

    tradeoff_tex = FIG_DIR / "subgroup_tradeoff_scatter_main.tex"
    tradeoff_pdf = FIG_DIR / "subgroup_tradeoff_scatter_main.pdf"
    tradeoff_png = FIG_DIR / "subgroup_tradeoff_scatter_main.png"
    dumbbell_tex = FIG_DIR / "subgroup_branch_dumbbell_main.tex"
    dumbbell_pdf = FIG_DIR / "subgroup_branch_dumbbell_main.pdf"
    dumbbell_png = FIG_DIR / "subgroup_branch_dumbbell_main.png"
    prefix_bars_tex = FIG_DIR / "subgroup_prefix_bars_appendix.tex"
    prefix_bars_pdf = FIG_DIR / "subgroup_prefix_bars_appendix.pdf"
    prefix_bars_png = FIG_DIR / "subgroup_prefix_bars_appendix.png"
    heatmap_tex = FIG_DIR / "subgroup_dataset_model_heatmap_appendix.tex"
    heatmap_pdf = FIG_DIR / "subgroup_dataset_model_heatmap_appendix.pdf"
    heatmap_png = FIG_DIR / "subgroup_dataset_model_heatmap_appendix.png"
    model_subitem_heatmap_tex = FIG_DIR / "subgroup_model_subitem_heatmap_appendix.tex"
    model_subitem_heatmap_pdf = FIG_DIR / "subgroup_model_subitem_heatmap_appendix.pdf"
    model_subitem_heatmap_png = FIG_DIR / "subgroup_model_subitem_heatmap_appendix.png"
    grouped_bars_tex = FIG_DIR / "subgroup_family_subitem_bars_appendix.tex"
    grouped_bars_pdf = FIG_DIR / "subgroup_family_subitem_bars_appendix.pdf"
    grouped_bars_png = FIG_DIR / "subgroup_family_subitem_bars_appendix.png"

    _write_tradeoff_tex(model_summary_df, tradeoff_tex)
    _write_dumbbell_tex(model_summary_df, dumbbell_tex)
    _write_prefix_bar_tex(prefix_plot_df, prefix_bars_tex)
    _write_heatmap_tex(heatmap_df, heatmap_tex)
    write_model_subitem_heatmap_tex(
        model_subitem_heatmap_df,
        model_order=MUST_DO_MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        title="Subgroup model-subitem heatmap",
        colorbar_title="Mean score",
        path=model_subitem_heatmap_tex,
    )
    write_model_subitem_grouped_bar_tex(
        model_subitem_heatmap_df,
        model_order=MUST_DO_MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        model_color_map=MODEL_COLORS,
        title="Subgroup family and subitem bars",
        y_label="Score",
        path=grouped_bars_tex,
    )

    _plot_tradeoff_preview(model_summary_df, tradeoff_pdf, tradeoff_png)
    _plot_dumbbell_preview(model_summary_df, dumbbell_pdf, dumbbell_png)
    _plot_prefix_bar_preview(prefix_plot_df, prefix_bars_pdf, prefix_bars_png)
    _plot_heatmap_preview(heatmap_df, heatmap_pdf, heatmap_png)
    plot_model_subitem_heatmap_preview(
        model_subitem_heatmap_df,
        model_order=MUST_DO_MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        title="Subgroup model-subitem heatmap",
        pdf_path=model_subitem_heatmap_pdf,
        png_path=model_subitem_heatmap_png,
    )
    plot_model_subitem_grouped_bar_preview(
        model_subitem_heatmap_df,
        model_order=MUST_DO_MODEL_ORDER,
        model_label_map=MODEL_LABELS,
        model_color_map=MODEL_COLORS,
        title="Subgroup family and subitem bars",
        y_label="Score",
        pdf_path=grouped_bars_pdf,
        png_path=grouped_bars_png,
    )

    _write_model_table_tex(model_summary_df, TABLE_DIR / "subgroup_model_summary_generated.tex")
    _build_report(run_dir, query_df, dataset_model_df, model_summary_df, prefix_summary_df, dataset_summary_df)
    _build_readme(run_dir)

    compile_notes = {
        "tradeoff": _try_compile_tex(tradeoff_tex),
        "dumbbell": _try_compile_tex(dumbbell_tex),
        "prefix_bars": _try_compile_tex(prefix_bars_tex),
        "heatmap": _try_compile_tex(heatmap_tex),
        "model_subitem_heatmap": _try_compile_tex(model_subitem_heatmap_tex),
        "family_subitem_bars": _try_compile_tex(grouped_bars_tex),
    }

    final_files = [
        tradeoff_tex,
        tradeoff_pdf,
        tradeoff_png,
        dumbbell_tex,
        dumbbell_pdf,
        dumbbell_png,
        prefix_bars_tex,
        prefix_bars_pdf,
        prefix_bars_png,
        heatmap_tex,
        heatmap_pdf,
        heatmap_png,
        model_subitem_heatmap_tex,
        model_subitem_heatmap_pdf,
        model_subitem_heatmap_png,
        grouped_bars_tex,
        grouped_bars_pdf,
        grouped_bars_png,
        TABLE_DIR / "subgroup_model_summary_generated.tex",
        DATA_DIR / "model_summary.csv",
        DATA_DIR / "prefix_summary.csv",
        OUTPUT_ROOT / "analysis_report.md",
    ]
    must_do_aliases = {
        path.name: path
        for path in [
            tradeoff_tex,
            tradeoff_pdf,
            tradeoff_png,
            prefix_bars_tex,
            prefix_bars_pdf,
            prefix_bars_png,
            heatmap_tex,
            heatmap_pdf,
            heatmap_png,
            model_subitem_heatmap_tex,
            model_subitem_heatmap_pdf,
            model_subitem_heatmap_png,
            grouped_bars_tex,
            grouped_bars_pdf,
            grouped_bars_png,
        ]
    }
    version_tag = TARGET_SQL_SOURCE_VERSION
    if publish_final:
        _copy_final_artifacts(final_files, must_do_aliases, version_tag=version_tag)

        final_readme = render_final_readme(
            title="Subgroup Breakdown Final",
            summary=f"This directory contains the paper-facing subgroup breakdown artifacts for `{sql_source_label(version_tag)}` (`{version_tag}`), with the standardized must-do bundle mirrored into `final/must_do/` and `final/{version_tag}/must_do/`.",
            primary_files=[
                *[versioned_name(name, version_tag) for name in [
                    "subgroup_tradeoff_scatter_main.tex",
                    "subgroup_tradeoff_scatter_main.pdf",
                    "subgroup_tradeoff_scatter_main.png",
                    "subgroup_branch_dumbbell_main.tex",
                    "subgroup_branch_dumbbell_main.pdf",
                    "subgroup_branch_dumbbell_main.png",
                    "subgroup_prefix_bars_appendix.tex",
                    "subgroup_prefix_bars_appendix.pdf",
                    "subgroup_prefix_bars_appendix.png",
                    "subgroup_dataset_model_heatmap_appendix.tex",
                    "subgroup_dataset_model_heatmap_appendix.pdf",
                    "subgroup_dataset_model_heatmap_appendix.png",
                    "subgroup_model_subitem_heatmap_appendix.tex",
                    "subgroup_model_subitem_heatmap_appendix.pdf",
                    "subgroup_model_subitem_heatmap_appendix.png",
                    "subgroup_family_subitem_bars_appendix.tex",
                    "subgroup_family_subitem_bars_appendix.pdf",
                    "subgroup_family_subitem_bars_appendix.png",
                    "subgroup_model_summary_generated.tex",
                    "model_summary.csv",
                    "prefix_summary.csv",
                ]],
            ],
            must_do_files=[versioned_name(name, version_tag) for name in must_do_aliases.keys()],
            notes=[
                f"The active SQL source for this final bundle is `{sql_source_label(version_tag)}` (`{version_tag}`).",
                "The `.tex` files are standalone LaTeX sources. The `.pdf/.png` files are immediate previews for reading in the current environment.",
            ],
        )
        (FINAL_DIR / "README.md").write_text(final_readme, encoding="utf-8")

    manifest = {
        "task": "subgroup_breakdown",
        "sql_source_version": version_tag,
        "sql_source_label": sql_source_label(version_tag),
        "source_analysis_run": run_dir.name,
        "excluded_models": sorted(EXCLUDED_MODELS),
        "included_models": model_summary_df["model_id"].tolist(),
        "dataset_panel_count": int(dataset_model_df.shape[0]),
        "query_row_count": int(query_df.shape[0]),
        "compile_notes": {key: {"success": value[0], "note": value[1]} for key, value in compile_notes.items()},
        "publish_final": bool(publish_final),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subgroup breakdown artifacts from analysis outputs.")
    parser.add_argument("--analysis-run-dir", type=Path, default=None, help="Optional analysis run dir.")
    parser.add_argument("--skip-final-publish", action="store_true", help="Skip writing shared final outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_subgroup_breakdown(
        analysis_run_dir=args.analysis_run_dir,
        publish_final=not args.skip_final_publish,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
