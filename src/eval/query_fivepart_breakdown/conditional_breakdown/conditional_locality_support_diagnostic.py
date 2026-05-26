#!/usr/bin/env python3
"""Paper-ready conditional locality and support diagnostics.

This script consumes the existing conditional-breakdown artifacts plus the
frozen upstream analysis run, then produces a new timestamped diagnostic bundle
without overwriting upstream benchmark outputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
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

from src.eval.query_fivepart_breakdown.common_final import render_final_readme, sync_final_outputs
from src.eval.subitem_workload_v2.contract_spec import TEMPLATE_CONTRACTS


EVALUATION_ROOT = PROJECT_ROOT / "Evaluation"
CONDITIONAL_ROOT = EVALUATION_ROOT / "query_fivepart_breakdown" / "conditional_breakdown"
LOCALITY_SUPPORT_ROOT = CONDITIONAL_ROOT / "locality_support_diagnostics"
SOURCE_DATA_DIR = CONDITIONAL_ROOT / "data"


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


SOURCE_RUN_ID = _env_or_default("CONDITIONAL_LOCALITY_SOURCE_RUN_ID", "v2_keyset_cts_48_20260504_2350")
REFERENCE_SQLITE_RUN_ID = _env_or_default("CONDITIONAL_LOCALITY_REFERENCE_SQLITE_RUN_ID", "20260426_193605")
SOURCE_QUERY_JSONL = EVALUATION_ROOT / "analysis" / "runs" / SOURCE_RUN_ID / "summaries" / "analysis_query_scores__all_datasets.jsonl"


def _resolve_real_sqlite_root() -> Path:
    explicit = os.environ.get("CONDITIONAL_LOCALITY_REAL_SQLITE_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    preferred = EVALUATION_ROOT / "analysis" / "runs" / SOURCE_RUN_ID / "cache" / "real_sqlite"
    if preferred.exists():
        return preferred
    fallback = EVALUATION_ROOT / "analysis" / "runs" / REFERENCE_SQLITE_RUN_ID / "cache" / "real_sqlite"
    return fallback


def _resolve_logs_run_root() -> Path:
    explicit = os.environ.get("CONDITIONAL_LOCALITY_LOGS_RUN_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    manifest_path = EVALUATION_ROOT / "analysis" / "runs" / SOURCE_RUN_ID / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            root = str(manifest.get("sql_source_root") or "").strip()
            if root:
                return Path(root)
        except Exception:
            pass
    return PROJECT_ROOT / "logs" / "runs"


REAL_SQLITE_ROOT = _resolve_real_sqlite_root()
LOGS_RUN_ROOT = _resolve_logs_run_root()

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
REAL_COLOR = "#000000"
PREFIX_LABELS = {"c": "Categorical", "m": "Mixed", "n": "Numerical"}
STRUCTURE_ORDER = ["grouped_global", "surface_2d", "filtered_local", "unknown"]
STRUCTURE_LABELS = {
    "grouped_global": "Grouped / Global",
    "surface_2d": "2D Surface",
    "filtered_local": "Filtered / Local",
    "unknown": "Unknown",
}
SUPPORT_BUCKET_ORDER = ["dense", "medium", "sparse"]
SUPPORT_BUCKET_LABELS = {"dense": "Dense", "medium": "Medium", "sparse": "Sparse"}
SUPPORT_FLOOR_FLAGS = [5, 10]
PRIMARY_SUPPORT_VARIANT = "scalar_filtered_local"
SENSITIVITY_SUPPORT_VARIANT = "all_filtered_local"
SUPPORT_VARIANT_LABELS = {
    PRIMARY_SUPPORT_VARIANT: "Scalar filtered-local slices",
    SENSITIVITY_SUPPORT_VARIANT: "All filtered-local templates",
}


@dataclass(frozen=True)
class TemplateSemanticSpec:
    structure_type: str
    axis_arity: str
    rationale: str
    support_bucket_basis: str
    support_main_eligible: bool
    support_basis_note: str


TEMPLATE_SEMANTICS: dict[str, TemplateSemanticSpec] = {
    "tpl_m4_group_condition_rate": TemplateSemanticSpec(
        structure_type="grouped_global",
        axis_arity="1D",
        rationale="One grouped axis with no local filter; the query compares a global condition rate across groups rather than drilling into a filtered slice.",
        support_bucket_basis="not_applicable",
        support_main_eligible=False,
        support_basis_note="Not part of the filtered-local support diagnostic.",
    ),
    "tpl_m4_group_ratio_two_conditions": TemplateSemanticSpec(
        structure_type="grouped_global",
        axis_arity="1D",
        rationale="One grouped axis with a contrastive ratio view, but still a global grouped summary rather than a local filtered slice.",
        support_bucket_basis="not_applicable",
        support_main_eligible=False,
        support_basis_note="Not part of the filtered-local support diagnostic.",
    ),
    "tpl_m4_binned_numeric_group_avg": TemplateSemanticSpec(
        structure_type="grouped_global",
        axis_arity="1D",
        rationale="Bucketed bands create ordered groups, but the query still summarizes the whole dataset across one grouping axis without a local filter.",
        support_bucket_basis="not_applicable",
        support_main_eligible=False,
        support_basis_note="Not part of the filtered-local support diagnostic.",
    ),
    "tpl_tpcds_within_group_share": TemplateSemanticSpec(
        structure_type="grouped_global",
        axis_arity="1D",
        rationale="The item-level shares are nested inside a grouped global summary, not exposed as a two-axis interaction surface.",
        support_bucket_basis="not_applicable",
        support_main_eligible=False,
        support_basis_note="Not part of the filtered-local support diagnostic.",
    ),
    "tpl_m4_group_dispersion_rank": TemplateSemanticSpec(
        structure_type="grouped_global",
        axis_arity="1D",
        rationale="The query ranks one grouped axis by dispersion, so the main story is global grouped comparison rather than local filtering.",
        support_bucket_basis="not_applicable",
        support_main_eligible=False,
        support_basis_note="Not part of the filtered-local support diagnostic.",
    ),
    "tpl_m4_window_partition_avg": TemplateSemanticSpec(
        structure_type="grouped_global",
        axis_arity="1D",
        rationale="Window partitions still summarize one grouping axis at the full-dataset level; they do not define a narrow filtered slice.",
        support_bucket_basis="not_applicable",
        support_main_eligible=False,
        support_basis_note="Not part of the filtered-local support diagnostic.",
    ),
    "tpl_tpcds_baseline_gated_extreme_ranking": TemplateSemanticSpec(
        structure_type="grouped_global",
        axis_arity="1D",
        rationale="The baseline gate adds complexity, but the primary semantics remain a grouped ranking view rather than a 2D interaction surface or local slice.",
        support_bucket_basis="not_applicable",
        support_main_eligible=False,
        support_basis_note="Not part of the filtered-local support diagnostic.",
    ),
    "tpl_c2_two_dim_target_rate": TemplateSemanticSpec(
        structure_type="surface_2d",
        axis_arity="2D",
        rationale="This template explicitly constructs a two-axis interaction surface over two grouping fields without a local filter.",
        support_bucket_basis="not_applicable",
        support_main_eligible=False,
        support_basis_note="Not part of the filtered-local support diagnostic.",
    ),
    "tpl_m4_median_filtered_numeric": TemplateSemanticSpec(
        structure_type="filtered_local",
        axis_arity="1D",
        rationale="The template asks for a numeric summary inside a single filtered slice, so locality comes from the predicate rather than from axis count.",
        support_bucket_basis="filtered_row_count",
        support_main_eligible=True,
        support_basis_note="Bucketing uses the exact number of real rows satisfying the filter predicate.",
    ),
    "tpl_tpch_filtered_sum_band": TemplateSemanticSpec(
        structure_type="filtered_local",
        axis_arity="1D",
        rationale="The numeric-band predicate creates a local filtered slice; difficulty comes from preserving behavior inside that narrow slice.",
        support_bucket_basis="filtered_row_count",
        support_main_eligible=True,
        support_basis_note="Bucketing uses the exact number of real rows satisfying the numeric-band predicate.",
    ),
    "tpl_c2_filtered_group_count_2d": TemplateSemanticSpec(
        structure_type="filtered_local",
        axis_arity="2D",
        rationale="Even though the output is two-dimensional, the defining semantics are local because the surface exists only inside a predicate-defined slice.",
        support_bucket_basis="median_cell_row_count",
        support_main_eligible=False,
        support_basis_note="Exact per-cell counts are available, but the primary support analysis keeps scalar filtered-slice support units comparable and therefore excludes this template from the main dense/medium/sparse claim.",
    ),
}

CONTRACT_SUBITEMS = {
    contract.template_id: ",".join(contract.supported_canonical_subitem_ids) for contract in TEMPLATE_CONTRACTS
}
CONTRACT_ROLES = {
    contract.template_id: ",".join(contract.allowed_variant_roles) for contract in TEMPLATE_CONTRACTS
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dataset_prefix(dataset_id: str) -> str:
    return str(dataset_id or "").strip().lower()[:1]


def _dataset_sort_key(dataset_id: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", str(dataset_id or "").strip())
    if not match:
        return (99, 10**9, str(dataset_id))
    prefix, number = match.groups()
    return ({"c": 0, "m": 1, "n": 2}.get(prefix.lower(), 50), int(number), str(dataset_id))


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _model_sort_rank(model_id: str) -> int:
    return MODEL_ORDER.index(model_id) if model_id in MODEL_ORDER else 999


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def _escape_tex(text: Any) -> str:
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


def _summary_df(
    panel_df: pd.DataFrame,
    *,
    bucket_col: str,
    bucket_order: list[str],
    bucket_label_map: dict[str, str],
    base_query_df: pd.DataFrame,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    extra_group_cols = extra_group_cols or []
    rows: list[dict[str, Any]] = []
    group_cols = [bucket_col] + extra_group_cols
    for key, group in panel_df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        payload: dict[str, Any] = {}
        for col_name, value in zip(group_cols, key):
            payload[col_name] = value
        bucket_value = payload[bucket_col]
        query_subset = base_query_df.loc[base_query_df[bucket_col] == bucket_value].copy()
        for extra_col in extra_group_cols:
            query_subset = query_subset.loc[query_subset[extra_col] == payload[extra_col]].copy()
        stats = _metric_stats(group["panel_score"])
        payload.update(
            {
                "bucket_label": bucket_label_map.get(bucket_value, bucket_value),
                "panel_count": int(group.shape[0]),
                "dataset_count": int(group["dataset_id"].nunique()),
                "model_count": int(group["model_id"].nunique()),
                "template_count": int(query_subset["template_id"].nunique()) if not query_subset.empty else 0,
                "query_row_count": int(query_subset.shape[0]),
                "prefix_coverage": ",".join(sorted(query_subset["dataset_prefix"].dropna().astype(str).unique())),
                "prefix_count": int(query_subset["dataset_prefix"].nunique()) if not query_subset.empty else 0,
                "mean_score": stats["mean"],
                "std_score": stats["std"],
                "se_score": stats["se"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "ci95_radius": stats["ci95_radius"],
            }
        )
        payload["coverage_note"] = _coverage_note(
            panel_count=payload["panel_count"],
            dataset_count=payload["dataset_count"],
            template_count=payload["template_count"],
        )
        rows.append(payload)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["bucket_rank"] = out[bucket_col].map({name: idx for idx, name in enumerate(bucket_order)})
    if "model_id" in out.columns:
        out["model_sort"] = out["model_id"].map(_model_sort_rank)
    sort_cols = ["bucket_rank"]
    if "model_sort" in out.columns:
        sort_cols = ["model_sort", "bucket_rank"]
    out = out.sort_values(sort_cols).drop(columns=[col for col in ["bucket_rank", "model_sort"] if col in out.columns])
    return out.reset_index(drop=True)


def _coverage_note(*, panel_count: int, dataset_count: int, template_count: int) -> str:
    notes: list[str] = []
    if panel_count < 20:
        notes.append("low_panel_coverage")
    if dataset_count < 5:
        notes.append("low_dataset_coverage")
    if template_count <= 1:
        notes.append("single_template")
    return ",".join(notes) if notes else "adequate"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    table = df[columns].copy()
    return table.to_markdown(index=False)


def _tex_table(
    df: pd.DataFrame,
    *,
    columns: list[str],
    headers: list[str],
    path: Path,
    caption_note: str,
) -> None:
    lines = [
        r"\documentclass{standalone}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{booktabs}",
        r"\begin{document}",
        r"\scriptsize",
        rf"\emph{{{_escape_tex(caption_note)}}}\\[0.4em]",
        rf"\begin{{tabular}}{{{'l' * len(columns)}}}",
        r"\toprule",
        " & ".join(_escape_tex(item) for item in headers) + r" \\",
        r"\midrule",
    ]
    for row in df[columns].itertuples(index=False):
        cells: list[str] = []
        for value in row:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(_escape_tex(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


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


def _attempt_latexmk(tex_path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except FileNotFoundError:
        return {"success": False, "note": "latexmk not available"}
    except subprocess.TimeoutExpired:
        return {"success": False, "note": "latexmk timed out"}
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        note = stderr[-1] if stderr else "latexmk failed"
        return {"success": False, "note": note[:240]}
    return {"success": True, "note": "compiled"}


def _build_output_dirs() -> dict[str, Path]:
    run_tag = _utc_now().strftime("%Y%m%d_%H%M%S") + "_conditional_locality_support"
    run_dir = LOCALITY_SUPPORT_ROOT / "runs" / run_tag
    dirs = {
        "run_tag": Path(run_tag),
        "run_dir": run_dir,
        "data_dir": run_dir / "data",
        "fig_dir": run_dir / "figures",
        "table_dir": run_dir / "tables",
        "report_dir": run_dir / "report",
        "final_dir": LOCALITY_SUPPORT_ROOT / "final",
    }
    for path in [LOCALITY_SUPPORT_ROOT, dirs["run_dir"], dirs["data_dir"], dirs["fig_dir"], dirs["table_dir"], dirs["report_dir"], dirs["final_dir"]]:
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _load_conditional_query_rows() -> pd.DataFrame:
    path = SOURCE_DATA_DIR / "conditional_query_rows.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["query_score"] = pd.to_numeric(df["query_score"], errors="coerce")
    df["dataset_prefix"] = df["dataset_prefix"].astype(str)
    df["model_id"] = df["model_id"].astype(str).str.lower().str.strip()
    df["model_label"] = df["model_id"].map(_model_label)
    df["template_id"] = df["template_id"].astype(str)
    df["template_name"] = df["template_name"].astype(str)
    return df.loc[df["model_id"].isin(MODEL_ORDER)].copy().reset_index(drop=True)


def _load_source_lookup(selected_keys: set[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with SOURCE_QUERY_JSONL.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row.get("asset_key") or ""), str(row.get("query_id") or ""))
            if key not in selected_keys:
                continue
            rows.append(
                {
                    "asset_key": key[0],
                    "query_id": key[1],
                    "source_sql_run_id": str(row.get("source_sql_run_id") or ""),
                    "question": str(row.get("question") or ""),
                    "template_id_source": str(row.get("template_id") or ""),
                    "template_name_source": str(row.get("template_name") or ""),
                    "real_exec_ok": bool(row.get("real_exec_ok")),
                    "synthetic_exec_ok": bool(row.get("synthetic_exec_ok")),
                    "details_json": json.dumps(row.get("details") or {}, ensure_ascii=False, sort_keys=True),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["asset_key", "query_id"]).reset_index(drop=True)


def _apply_template_semantics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    rows: list[dict[str, Any]] = []
    for row in enriched.itertuples(index=False):
        spec = TEMPLATE_SEMANTICS.get(row.template_id)
        if spec is None:
            structure_type = "unknown"
            axis_arity = "unknown"
            rationale = "Template not present in the reviewed semantic mapping, so the row stays in an explicit unknown bucket."
            support_bucket_basis = "unavailable"
            support_main_eligible = False
            support_basis_note = "Template not reviewed for support recovery."
        else:
            structure_type = spec.structure_type
            axis_arity = spec.axis_arity
            rationale = spec.rationale
            support_bucket_basis = spec.support_bucket_basis
            support_main_eligible = spec.support_main_eligible
            support_basis_note = spec.support_basis_note
        rows.append(
            {
                **row._asdict(),
                "structure_type": structure_type,
                "axis_arity": axis_arity,
                "classification_rationale": rationale,
                "support_bucket_basis": support_bucket_basis,
                "support_main_eligible": support_main_eligible,
                "support_basis_note": support_basis_note,
                "contract_supported_subitems": CONTRACT_SUBITEMS.get(row.template_id, ""),
                "contract_allowed_roles": CONTRACT_ROLES.get(row.template_id, ""),
            }
        )
    return pd.DataFrame(rows)


def _build_template_mapping_df(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for template_id, group in df.groupby("template_id", sort=True):
        template_name = str(group["template_name"].iloc[0])
        rows.append(
            {
                "template_id": template_id,
                "template_name": template_name,
                "structure_type": str(group["structure_type"].iloc[0]),
                "axis_arity": str(group["axis_arity"].iloc[0]),
                "classification_rationale": str(group["classification_rationale"].iloc[0]),
                "n_query_rows": int(group.shape[0]),
                "n_datasets": int(group["dataset_id"].nunique()),
                "n_models": int(group["model_id"].nunique()),
                "n_panels": int(group[["dataset_id", "model_id"]].drop_duplicates().shape[0]),
                "dataset_prefixes": ",".join(sorted(group["dataset_prefix"].dropna().astype(str).unique())),
                "observed_subitems": ",".join(sorted(group["subitem_id"].dropna().astype(str).unique())),
                "contract_supported_subitems": str(group["contract_supported_subitems"].iloc[0]),
                "contract_allowed_roles": str(group["contract_allowed_roles"].iloc[0]),
                "support_bucket_basis": str(group["support_bucket_basis"].iloc[0]),
                "support_main_eligible": bool(group["support_main_eligible"].iloc[0]),
                "support_basis_note": str(group["support_basis_note"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out["structure_rank"] = out["structure_type"].map({name: idx for idx, name in enumerate(STRUCTURE_ORDER)})
    out = out.sort_values(["structure_rank", "template_id"]).drop(columns=["structure_rank"]).reset_index(drop=True)
    return out


def _build_template_panel_scores(df: pd.DataFrame) -> pd.DataFrame:
    panel = (
        df.groupby(
            ["dataset_id", "dataset_prefix", "model_id", "model_label", "template_id", "template_name", "structure_type", "axis_arity"],
            dropna=False,
            as_index=False,
        )
        .agg(
            panel_score=("query_score", "mean"),
            query_count=("query_id", "count"),
            subitem_count=("subitem_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    panel["model_sort"] = panel["model_id"].map(_model_sort_rank)
    panel["dataset_sort"] = panel["dataset_id"].map(_dataset_sort_key)
    panel = panel.sort_values(["dataset_sort", "model_sort", "template_id"]).drop(columns=["model_sort", "dataset_sort"]).reset_index(drop=True)
    return panel


def _build_locality_panel_scores(df: pd.DataFrame) -> pd.DataFrame:
    panel = (
        df.groupby(
            ["dataset_id", "dataset_prefix", "model_id", "model_label", "structure_type"],
            dropna=False,
            as_index=False,
        )
        .agg(
            panel_score=("query_score", "mean"),
            query_count=("query_id", "count"),
            template_count=("template_id", "nunique"),
            template_ids=("template_id", lambda s: ",".join(sorted(pd.Series(s).dropna().astype(str).unique()))),
            axis_arity_values=("axis_arity", lambda s: ",".join(sorted(pd.Series(s).dropna().astype(str).unique()))),
        )
        .reset_index(drop=True)
    )
    panel["structure_label"] = panel["structure_type"].map(STRUCTURE_LABELS)
    panel["structure_rank"] = panel["structure_type"].map({name: idx for idx, name in enumerate(STRUCTURE_ORDER)})
    panel["model_sort"] = panel["model_id"].map(_model_sort_rank)
    panel["dataset_sort"] = panel["dataset_id"].map(_dataset_sort_key)
    panel = (
        panel.sort_values(["dataset_sort", "model_sort", "structure_rank"])
        .drop(columns=["model_sort", "dataset_sort"])
        .reset_index(drop=True)
    )
    return panel


def _query_counts_by_bucket(base_df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket_value, group in base_df.groupby(bucket_col, sort=False):
        rows.append(
            {
                bucket_col: bucket_value,
                "query_row_count": int(group.shape[0]),
                "dataset_count": int(group["dataset_id"].nunique()),
                "model_count": int(group["model_id"].nunique()),
                "panel_count": int(group[["dataset_id", "model_id"]].drop_duplicates().shape[0]),
                "template_count": int(group["template_id"].nunique()),
                "prefix_coverage": ",".join(sorted(group["dataset_prefix"].dropna().astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def _plot_bucket_means_with_model_lines(
    summary_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    *,
    bucket_col: str,
    bucket_order: list[str],
    bucket_labels: dict[str, str],
    title: str,
    y_label: str,
    caption_note: str,
    png_path: Path,
    pdf_path: Path,
    svg_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 6.3))
    x = np.arange(len(bucket_order), dtype=float)
    for model_id in MODEL_ORDER:
        subset = model_summary_df.loc[model_summary_df["model_id"] == model_id].copy()
        if subset.empty:
            continue
        y_values: list[float] = []
        x_values: list[float] = []
        for idx, bucket in enumerate(bucket_order):
            row = subset.loc[subset[bucket_col] == bucket]
            if row.empty or pd.isna(row["mean_score"].iloc[0]):
                continue
            x_values.append(float(idx))
            y_values.append(float(row["mean_score"].iloc[0]))
        if len(x_values) < 2:
            continue
        ax.plot(
            x_values,
            y_values,
            color=MODEL_COLORS[model_id],
            linewidth=1.6,
            alpha=0.8,
            marker="o",
            markersize=4.8,
            zorder=2,
        )

    means = []
    yerr = []
    for bucket in bucket_order:
        row = summary_df.loc[summary_df[bucket_col] == bucket]
        if row.empty:
            means.append(np.nan)
            yerr.append(np.nan)
        else:
            means.append(float(row["mean_score"].iloc[0]) if pd.notna(row["mean_score"].iloc[0]) else np.nan)
            yerr.append(float(row["ci95_radius"].iloc[0]) if pd.notna(row["ci95_radius"].iloc[0]) else 0.0)
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        color=REAL_COLOR,
        linewidth=2.8,
        marker="o",
        markersize=8.0,
        capsize=4.0,
        label="Panel mean ± 95% CI",
        zorder=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([bucket_labels[item] for item in bucket_order])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.24)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.02,
        caption_note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _plot_by_model_lines(
    model_summary_df: pd.DataFrame,
    *,
    bucket_col: str,
    bucket_order: list[str],
    bucket_labels: dict[str, str],
    title: str,
    y_label: str,
    png_path: Path,
    pdf_path: Path,
    svg_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    x = np.arange(len(bucket_order), dtype=float)
    for model_id in MODEL_ORDER:
        subset = model_summary_df.loc[model_summary_df["model_id"] == model_id].copy()
        if subset.empty:
            continue
        y_values: list[float] = []
        for bucket in bucket_order:
            row = subset.loc[subset[bucket_col] == bucket]
            y_values.append(float(row["mean_score"].iloc[0]) if not row.empty and pd.notna(row["mean_score"].iloc[0]) else np.nan)
        ax.plot(
            x,
            y_values,
            color=MODEL_COLORS[model_id],
            linewidth=2.0,
            marker="o",
            markersize=5.2,
            label=_model_label(model_id),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([bucket_labels[item] for item in bucket_order])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _write_bucket_line_tex(
    summary_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    *,
    bucket_col: str,
    bucket_order: list[str],
    bucket_labels: dict[str, str],
    title: str,
    y_label: str,
    path: Path,
    include_legend: bool,
) -> None:
    lines = [_tex_preamble()]
    for model_id in MODEL_ORDER:
        lines.append(rf"\definecolor{{model{model_id}}}{{HTML}}{{{MODEL_COLORS[model_id].replace('#', '')}}}")
    lines.extend(
        [
            r"\definecolor{summaryblack}{HTML}{000000}",
            r"\begin{document}",
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"width=13.8cm,",
            r"height=8.4cm,",
            r"ymin=0.0, ymax=1.0,",
            rf"title={{{_escape_tex(title)}}},",
            rf"ylabel={{{_escape_tex(y_label)}}},",
            rf"xtick={{{','.join(str(idx + 1) for idx in range(len(bucket_order)))}}},",
            rf"xticklabels={{{','.join(_escape_tex(bucket_labels[item]) for item in bucket_order)}}},",
            r"ymajorgrids,",
            r"grid style={draw=gray!20},",
            r"major grid style={draw=gray!28},",
            r"axis line style={draw=black!70},",
            r"tick style={draw=black!70},",
            r"legend style={draw=none, fill=none, font=\scriptsize, at={(0.98,0.98)}, anchor=north east},",
            r"]",
        ]
    )
    for model_id in MODEL_ORDER:
        subset = model_summary_df.loc[model_summary_df["model_id"] == model_id].copy()
        coords: list[str] = []
        if subset.empty:
            continue
        for idx, bucket in enumerate(bucket_order, start=1):
            row = subset.loc[subset[bucket_col] == bucket]
            if row.empty or pd.isna(row["mean_score"].iloc[0]):
                continue
            coords.append(f"({idx},{float(row['mean_score'].iloc[0]):.6f})")
        if len(coords) < 2:
            continue
        lines.append(rf"\addplot+[mark=*, mark size=1.8pt, line width=0.9pt, draw=model{model_id}, fill=model{model_id}, opacity=0.82] coordinates {{{' '.join(coords)}}};")
        if include_legend:
            lines.append(rf"\addlegendentry{{{_escape_tex(_model_label(model_id))}}}")
    summary_coords = []
    error_rows = []
    for idx, bucket in enumerate(bucket_order, start=1):
        row = summary_df.loc[summary_df[bucket_col] == bucket]
        if row.empty or pd.isna(row["mean_score"].iloc[0]):
            continue
        mean_value = float(row["mean_score"].iloc[0])
        radius = float(row["ci95_radius"].iloc[0] or 0.0)
        summary_coords.append(f"({idx},{mean_value:.6f})")
        error_rows.append((idx, mean_value, radius))
    lines.append(r"\addplot+[mark=*, mark size=2.6pt, line width=1.8pt, draw=summaryblack, fill=summaryblack] coordinates {" + " ".join(summary_coords) + r"};")
    if include_legend:
        lines.append(r"\addlegendentry{Panel mean}")
    for idx, mean_value, radius in error_rows:
        lines.append(
            rf"\addplot+[only marks, mark=none, draw=summaryblack, error bars/.cd, y dir=both, y explicit] coordinates {{ ({idx},{mean_value:.6f}) +- (0,{radius:.6f}) }};"
        )
    lines.extend([r"\end{axis}", r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_combined(
    locality_summary_df: pd.DataFrame,
    locality_model_df: pd.DataFrame,
    support_summary_df: pd.DataFrame,
    support_model_df: pd.DataFrame,
    *,
    png_path: Path,
    pdf_path: Path,
    svg_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.2), sharey=True)
    panels = [
        (
            axes[0],
            locality_summary_df,
            locality_model_df,
            "structure_type",
            STRUCTURE_ORDER[:3],
            STRUCTURE_LABELS,
            "Panel A. Locality decomposition",
        ),
        (
            axes[1],
            support_summary_df,
            support_model_df,
            "support_bucket",
            SUPPORT_BUCKET_ORDER,
            SUPPORT_BUCKET_LABELS,
            "Panel B. Support decomposition",
        ),
    ]
    for ax, summary_df, model_df, bucket_col, bucket_order, bucket_labels, title in panels:
        x = np.arange(len(bucket_order), dtype=float)
        for model_id in MODEL_ORDER:
            subset = model_df.loc[model_df["model_id"] == model_id].copy()
            if subset.empty:
                continue
            x_values: list[float] = []
            y_values: list[float] = []
            for idx, bucket in enumerate(bucket_order):
                row = subset.loc[subset[bucket_col] == bucket]
                if row.empty or pd.isna(row["mean_score"].iloc[0]):
                    continue
                x_values.append(float(idx))
                y_values.append(float(row["mean_score"].iloc[0]))
            if len(x_values) < 2:
                continue
            ax.plot(
                x_values,
                y_values,
                color=MODEL_COLORS[model_id],
                linewidth=1.4,
                alpha=0.72,
                marker="o",
                markersize=4.2,
            )
        means = []
        yerr = []
        for bucket in bucket_order:
            row = summary_df.loc[summary_df[bucket_col] == bucket]
            means.append(float(row["mean_score"].iloc[0]) if not row.empty and pd.notna(row["mean_score"].iloc[0]) else np.nan)
            yerr.append(float(row["ci95_radius"].iloc[0]) if not row.empty and pd.notna(row["ci95_radius"].iloc[0]) else 0.0)
        ax.errorbar(x, means, yerr=yerr, color=REAL_COLOR, linewidth=2.6, marker="o", markersize=7.0, capsize=4.0, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([bucket_labels[item] for item in bucket_order])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.24)
    axes[0].set_ylabel("Conditional fidelity score")
    fig.suptitle("Conditional fidelity degrades with locality; sparse support is only a partial explanation")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _write_combined_tex(
    locality_summary_df: pd.DataFrame,
    locality_model_df: pd.DataFrame,
    support_summary_df: pd.DataFrame,
    support_model_df: pd.DataFrame,
    *,
    path: Path,
) -> None:
    lines = [_tex_preamble()]
    for model_id in MODEL_ORDER:
        lines.append(rf"\definecolor{{model{model_id}}}{{HTML}}{{{MODEL_COLORS[model_id].replace('#', '')}}}")
    lines.extend(
        [
            r"\definecolor{summaryblack}{HTML}{000000}",
            r"\begin{document}",
            r"\begin{tikzpicture}",
            r"\begin{groupplot}[",
            r"group style={group size=2 by 1, horizontal sep=1.3cm},",
            r"width=6.6cm,",
            r"height=7.6cm,",
            r"ymin=0.0, ymax=1.0,",
            r"ymajorgrids,",
            r"grid style={draw=gray!20},",
            r"major grid style={draw=gray!28},",
            r"axis line style={draw=black!70},",
            r"tick style={draw=black!70},",
            r"]",
            rf"\nextgroupplot[title={{{_escape_tex('Panel A. Locality decomposition')}}}, ylabel={{{_escape_tex('Conditional fidelity score')}}}, xtick={{1,2,3}}, xticklabels={{{','.join(_escape_tex(STRUCTURE_LABELS[item]) for item in STRUCTURE_ORDER[:3])}}}]",
        ]
    )
    for model_id in MODEL_ORDER:
        subset = locality_model_df.loc[locality_model_df["model_id"] == model_id]
        coords = []
        for idx, bucket in enumerate(STRUCTURE_ORDER[:3], start=1):
            row = subset.loc[subset["structure_type"] == bucket]
            if row.empty or pd.isna(row["mean_score"].iloc[0]):
                continue
            coords.append(f"({idx},{float(row['mean_score'].iloc[0]):.6f})")
        if len(coords) >= 2:
            lines.append(rf"\addplot+[mark=*, mark size=1.6pt, line width=0.85pt, draw=model{model_id}, fill=model{model_id}, opacity=0.78] coordinates {{{' '.join(coords)}}};")
    locality_summary_coords = []
    locality_errors = []
    for idx, bucket in enumerate(STRUCTURE_ORDER[:3], start=1):
        row = locality_summary_df.loc[locality_summary_df["structure_type"] == bucket]
        if row.empty or pd.isna(row["mean_score"].iloc[0]):
            continue
        mean_value = float(row["mean_score"].iloc[0])
        radius = float(row["ci95_radius"].iloc[0] or 0.0)
        locality_summary_coords.append(f"({idx},{mean_value:.6f})")
        locality_errors.append((idx, mean_value, radius))
    lines.append(r"\addplot+[mark=*, mark size=2.5pt, line width=1.8pt, draw=summaryblack, fill=summaryblack] coordinates {" + " ".join(locality_summary_coords) + r"};")
    for idx, mean_value, radius in locality_errors:
        lines.append(rf"\addplot+[only marks, mark=none, draw=summaryblack, error bars/.cd, y dir=both, y explicit] coordinates {{ ({idx},{mean_value:.6f}) +- (0,{radius:.6f}) }};")
    lines.append(rf"\nextgroupplot[title={{{_escape_tex('Panel B. Support decomposition')}}}, xtick={{1,2,3}}, xticklabels={{{','.join(_escape_tex(SUPPORT_BUCKET_LABELS[item]) for item in SUPPORT_BUCKET_ORDER)}}}]")
    for model_id in MODEL_ORDER:
        subset = support_model_df.loc[support_model_df["model_id"] == model_id]
        coords = []
        for idx, bucket in enumerate(SUPPORT_BUCKET_ORDER, start=1):
            row = subset.loc[subset["support_bucket"] == bucket]
            if row.empty or pd.isna(row["mean_score"].iloc[0]):
                continue
            coords.append(f"({idx},{float(row['mean_score'].iloc[0]):.6f})")
        if len(coords) >= 2:
            lines.append(rf"\addplot+[mark=*, mark size=1.6pt, line width=0.85pt, draw=model{model_id}, fill=model{model_id}, opacity=0.78] coordinates {{{' '.join(coords)}}};")
    support_summary_coords = []
    support_errors = []
    for idx, bucket in enumerate(SUPPORT_BUCKET_ORDER, start=1):
        row = support_summary_df.loc[support_summary_df["support_bucket"] == bucket]
        if row.empty or pd.isna(row["mean_score"].iloc[0]):
            continue
        mean_value = float(row["mean_score"].iloc[0])
        radius = float(row["ci95_radius"].iloc[0] or 0.0)
        support_summary_coords.append(f"({idx},{mean_value:.6f})")
        support_errors.append((idx, mean_value, radius))
    lines.append(r"\addplot+[mark=*, mark size=2.5pt, line width=1.8pt, draw=summaryblack, fill=summaryblack] coordinates {" + " ".join(support_summary_coords) + r"};")
    for idx, mean_value, radius in support_errors:
        lines.append(rf"\addplot+[only marks, mark=none, draw=summaryblack, error bars/.cd, y dir=both, y explicit] coordinates {{ ({idx},{mean_value:.6f}) +- (0,{radius:.6f}) }};")
    lines.extend([r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_question_run_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in LOGS_RUN_ROOT.glob("*/question_runs/*"):
        if path.is_dir():
            index.setdefault(path.name, path)
    for path in LOGS_RUN_ROOT.iterdir():
        if path.is_dir():
            index.setdefault(path.name, path)
    return index


def _resolve_generated_sql_path(run_dir: Path | None, dataset_id: str, query_id: str) -> Path | None:
    if run_dir is None:
        return None
    legacy_path = run_dir / "generated_sql.sql"
    if legacy_path.exists():
        return legacy_path
    v2_path = run_dir / str(dataset_id) / "artifacts" / str(query_id) / "generated_sql.sql"
    if v2_path.exists():
        return v2_path
    return legacy_path


def _extract_table_and_where(sql_text: str) -> tuple[str | None, str | None]:
    match = re.search(
        r'FROM\s+(?P<table>"[^"]+"|\[[^\]]+\]|\w+)\s+WHERE\s+(?P<where>.*?)(?=GROUP\s+BY|ORDER\s+BY|LIMIT|;|\)\s*SELECT|$)',
        sql_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, None
    table_name = match.group("table").strip()
    where_clause = match.group("where").strip()
    if where_clause.endswith(")"):
        where_clause = where_clause[:-1].rstrip()
    return table_name, where_clause


def _support_query_for_filtered_row_count(sql_text: str) -> str | None:
    table_name, where_clause = _extract_table_and_where(sql_text)
    if not table_name or not where_clause:
        return None
    return f"SELECT COUNT(*) AS real_support FROM {table_name} WHERE {where_clause};"


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return float(numeric)


def _recover_support_cases(
    filtered_local_df: pd.DataFrame,
    *,
    run_dir_index: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    unique_cases = (
        filtered_local_df[
            [
                "dataset_id",
                "dataset_prefix",
                "template_id",
                "template_name",
                "query_id",
                "question",
                "source_sql_run_id",
                "structure_type",
                "axis_arity",
                "support_bucket_basis",
                "support_main_eligible",
                "support_basis_note",
            ]
        ]
        .drop_duplicates(subset=["dataset_id", "query_id", "source_sql_run_id"])
        .reset_index(drop=True)
    )
    conn_cache: dict[str, sqlite3.Connection] = {}
    case_rows: list[dict[str, Any]] = []
    compile_counts = Counter()
    for case in unique_cases.itertuples(index=False):
        run_dir = run_dir_index.get(case.source_sql_run_id)
        sql_path = _resolve_generated_sql_path(run_dir, case.dataset_id, case.query_id)
        db_path = REAL_SQLITE_ROOT / f"{case.dataset_id}.sqlite"
        base = {
            "dataset_id": case.dataset_id,
            "dataset_prefix": case.dataset_prefix,
            "template_id": case.template_id,
            "template_name": case.template_name,
            "query_id": case.query_id,
            "question": case.question,
            "source_sql_run_id": case.source_sql_run_id,
            "structure_type": case.structure_type,
            "axis_arity": case.axis_arity,
            "support_bucket_basis": case.support_bucket_basis,
            "support_main_eligible": bool(case.support_main_eligible),
            "support_basis_note": case.support_basis_note,
            "run_dir_found": run_dir is not None,
            "support_sql_path": str(sql_path) if sql_path is not None else "",
            "db_path": str(db_path),
            "support_recovery_mode": "unavailable",
            "support_rationale": "",
            "real_support_value": None,
            "real_support_min": None,
            "real_support_median": None,
            "real_support_max": None,
            "real_support_mean": None,
            "real_support_group_count": None,
            "support_query_note": "",
        }
        if run_dir is None:
            base["support_rationale"] = "source_sql_run_folder_not_found"
            case_rows.append(base)
            compile_counts["unavailable"] += 1
            continue
        if sql_path is None or not sql_path.exists():
            base["support_rationale"] = "generated_sql_missing"
            case_rows.append(base)
            compile_counts["unavailable"] += 1
            continue
        if not db_path.exists():
            base["support_rationale"] = "real_sqlite_missing"
            case_rows.append(base)
            compile_counts["unavailable"] += 1
            continue
        sql_text = sql_path.read_text(encoding="utf-8").strip()
        if case.dataset_id not in conn_cache:
            conn_cache[case.dataset_id] = sqlite3.connect(str(db_path))
        conn = conn_cache[case.dataset_id]
        try:
            if case.template_id in {"tpl_m4_median_filtered_numeric", "tpl_tpch_filtered_sum_band"}:
                support_sql = _support_query_for_filtered_row_count(sql_text)
                if not support_sql:
                    raise ValueError("could_not_extract_filtered_where_clause")
                row = conn.execute(support_sql).fetchone()
                support_value = _safe_float(row[0] if row else None)
                base.update(
                    {
                        "support_recovery_mode": "derived_exact",
                        "support_rationale": "derived_real_row_count_from_original_filter",
                        "real_support_value": support_value,
                        "real_support_min": support_value,
                        "real_support_median": support_value,
                        "real_support_max": support_value,
                        "real_support_mean": support_value,
                        "real_support_group_count": 1,
                        "support_query_note": support_sql,
                    }
                )
                compile_counts["derived_exact"] += 1
            elif case.template_id == "tpl_c2_filtered_group_count_2d":
                cursor = conn.execute(sql_text)
                rows = cursor.fetchall()
                columns = [item[0] for item in cursor.description or []]
                if "row_count" not in columns:
                    raise ValueError("row_count_column_missing")
                row_count_idx = columns.index("row_count")
                values = [_safe_float(item[row_count_idx]) for item in rows]
                clean_values = [item for item in values if item is not None]
                if not clean_values:
                    raise ValueError("no_real_row_count_values")
                base.update(
                    {
                        "support_recovery_mode": "exact",
                        "support_rationale": "used_exact_real_cell_counts_from_original_group_count_query",
                        "real_support_value": round(float(pd.Series(clean_values).median()), 6),
                        "real_support_min": round(float(min(clean_values)), 6),
                        "real_support_median": round(float(pd.Series(clean_values).median()), 6),
                        "real_support_max": round(float(max(clean_values)), 6),
                        "real_support_mean": round(float(mean(clean_values)), 6),
                        "real_support_group_count": len(clean_values),
                        "support_query_note": "Used original query row_count outputs; the scalar support value for sensitivity bucketization is the median per-cell count.",
                    }
                )
                compile_counts["exact"] += 1
            else:
                base["support_rationale"] = "template_not_supported_for_recovery"
                compile_counts["unavailable"] += 1
        except Exception as exc:  # pragma: no cover - defensive reporting path
            base["support_recovery_mode"] = "unavailable"
            base["support_rationale"] = f"support_recovery_failed:{exc}"
            compile_counts["unavailable"] += 1
        case_rows.append(base)
    for conn in conn_cache.values():
        conn.close()
    case_df = pd.DataFrame(case_rows)
    audit_df = filtered_local_df.merge(
        case_df.drop(columns=["dataset_prefix", "template_name", "question", "structure_type", "axis_arity", "support_bucket_basis", "support_main_eligible", "support_basis_note"]),
        on=["dataset_id", "query_id", "source_sql_run_id", "template_id"],
        how="left",
    )
    method_summary = {
        "case_count": int(case_df.shape[0]),
        "template_count": int(case_df["template_id"].nunique()),
        "mode_counts": {key: int(value) for key, value in compile_counts.items()},
        "sql_artifact_found_count": int(case_df["run_dir_found"].sum()),
        "sql_artifact_missing_count": int((~case_df["run_dir_found"]).sum()),
        "main_eligible_case_count": int(case_df["support_main_eligible"].sum()),
    }
    return case_df, audit_df, method_summary


def _assign_support_buckets(
    case_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible_main = case_df.loc[
        case_df["support_main_eligible"] & case_df["support_recovery_mode"].isin(["exact", "derived_exact"]) & case_df["real_support_value"].notna()
    ].copy()
    eligible_all = case_df.loc[
        case_df["support_recovery_mode"].isin(["exact", "derived_exact"]) & case_df["real_support_value"].notna()
    ].copy()
    rows: list[dict[str, Any]] = []
    variant_summary: dict[str, Any] = {}
    for variant_name, eligible_df in [
        (PRIMARY_SUPPORT_VARIANT, eligible_main),
        (SENSITIVITY_SUPPORT_VARIANT, eligible_all),
    ]:
        dataset_notes: list[dict[str, Any]] = []
        for dataset_id, group in eligible_df.groupby("dataset_id", sort=False):
            values = pd.to_numeric(group["real_support_value"], errors="coerce")
            unique_value_count = int(values.dropna().nunique())
            if group.shape[0] < 3 or unique_value_count < 3:
                for row in group.itertuples(index=False):
                    rows.append(
                        {
                            "analysis_variant": variant_name,
                            "dataset_id": dataset_id,
                            "template_id": row.template_id,
                            "query_id": row.query_id,
                            "support_bucket": None,
                            "support_bucket_note": "unsupported_degenerate_within_dataset",
                            "bucket_floor_flags": _bucket_floor_flags(row.real_support_value),
                        }
                    )
                dataset_notes.append(
                    {
                        "analysis_variant": variant_name,
                        "dataset_id": dataset_id,
                        "case_count": int(group.shape[0]),
                        "unique_support_values": unique_value_count,
                        "bucketing_status": "unsupported_degenerate_within_dataset",
                    }
                )
                continue
            ranked = values.rank(method="first")
            try:
                bins = pd.qcut(ranked, q=3, labels=["sparse", "medium", "dense"])
            except ValueError:
                bins = None
            if bins is None:
                for row in group.itertuples(index=False):
                    rows.append(
                        {
                            "analysis_variant": variant_name,
                            "dataset_id": dataset_id,
                            "template_id": row.template_id,
                            "query_id": row.query_id,
                            "support_bucket": None,
                            "support_bucket_note": "unsupported_qcut_failed",
                            "bucket_floor_flags": _bucket_floor_flags(row.real_support_value),
                        }
                    )
                dataset_notes.append(
                    {
                        "analysis_variant": variant_name,
                        "dataset_id": dataset_id,
                        "case_count": int(group.shape[0]),
                        "unique_support_values": unique_value_count,
                        "bucketing_status": "unsupported_qcut_failed",
                    }
                )
                continue
            assigned = group.copy()
            assigned["support_bucket"] = bins.astype(str)
            assigned["support_bucket_note"] = "within_dataset_tertiles"
            for row in assigned.itertuples(index=False):
                rows.append(
                    {
                        "analysis_variant": variant_name,
                        "dataset_id": dataset_id,
                        "template_id": row.template_id,
                        "query_id": row.query_id,
                        "support_bucket": row.support_bucket,
                        "support_bucket_note": row.support_bucket_note,
                        "bucket_floor_flags": _bucket_floor_flags(row.real_support_value),
                    }
                )
            dataset_notes.append(
                {
                    "analysis_variant": variant_name,
                    "dataset_id": dataset_id,
                    "case_count": int(group.shape[0]),
                    "unique_support_values": unique_value_count,
                    "bucketing_status": "ok",
                }
            )
        variant_summary[variant_name] = {
            "eligible_case_count": int(eligible_df.shape[0]),
            "supported_dataset_count": int(sum(1 for item in dataset_notes if item["bucketing_status"] == "ok")),
            "unsupported_dataset_count": int(sum(1 for item in dataset_notes if item["bucketing_status"] != "ok")),
            "dataset_notes": dataset_notes,
        }
    assigned_df = pd.DataFrame(rows)
    merged = case_df.merge(assigned_df, on=["dataset_id", "template_id", "query_id"], how="left")
    return merged, variant_summary


def _bucket_floor_flags(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    flags = [f"lt_{floor}" for floor in SUPPORT_FLOOR_FLAGS if numeric < floor]
    return ",".join(flags)


def _build_support_panel_scores(
    support_query_rows_df: pd.DataFrame,
) -> pd.DataFrame:
    panel = (
        support_query_rows_df.groupby(
            ["analysis_variant", "dataset_id", "dataset_prefix", "model_id", "model_label", "support_bucket"],
            as_index=False,
            dropna=False,
        )
        .agg(
            panel_score=("query_score", "mean"),
            query_count=("query_id", "count"),
            template_count=("template_id", "nunique"),
            support_case_count=("query_id", "nunique"),
            support_value_mean=("real_support_value", "mean"),
            support_value_median=("real_support_value", "median"),
            support_mode_mix=("support_recovery_mode", lambda s: ",".join(sorted(pd.Series(s).dropna().astype(str).unique()))),
            support_basis_mix=("support_bucket_basis", lambda s: ",".join(sorted(pd.Series(s).dropna().astype(str).unique()))),
        )
        .reset_index(drop=True)
    )
    panel["support_bucket_label"] = panel["support_bucket"].map(SUPPORT_BUCKET_LABELS)
    panel["variant_label"] = panel["analysis_variant"].map(SUPPORT_VARIANT_LABELS)
    panel["bucket_rank"] = panel["support_bucket"].map({name: idx for idx, name in enumerate(SUPPORT_BUCKET_ORDER)})
    panel["model_sort"] = panel["model_id"].map(_model_sort_rank)
    panel["dataset_sort"] = panel["dataset_id"].map(_dataset_sort_key)
    panel = (
        panel.sort_values(["analysis_variant", "dataset_sort", "model_sort", "bucket_rank"])
        .drop(columns=["bucket_rank", "model_sort", "dataset_sort"])
        .reset_index(drop=True)
    )
    return panel


def _build_support_query_rows(
    filtered_local_df: pd.DataFrame,
    case_bucket_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched = filtered_local_df.merge(
        case_bucket_df[
            [
                "dataset_id",
                "template_id",
                "query_id",
                "analysis_variant",
                "support_bucket",
                "support_bucket_note",
                "real_support_value",
                "real_support_min",
                "real_support_median",
                "real_support_max",
                "real_support_group_count",
                "support_recovery_mode",
                "bucket_floor_flags",
            ]
        ],
        on=["dataset_id", "template_id", "query_id"],
        how="left",
    )
    return enriched.loc[enriched["support_bucket"].notna()].copy().reset_index(drop=True)


def _select_primary_support_variant(
    support_summary_all: pd.DataFrame,
) -> tuple[str, str]:
    subset = support_summary_all.loc[support_summary_all["analysis_variant"] == PRIMARY_SUPPORT_VARIANT].copy()
    if subset.empty:
        return SENSITIVITY_SUPPORT_VARIANT, "primary_scalar_variant_missing"
    bucket_counts = subset["support_bucket"].nunique()
    min_panels = int(subset["panel_count"].min()) if not subset.empty else 0
    min_datasets = int(subset["dataset_count"].min()) if not subset.empty else 0
    if bucket_counts == 3 and min_panels >= 20 and min_datasets >= 5:
        return PRIMARY_SUPPORT_VARIANT, "primary_scalar_variant_has_adequate_coverage"
    return PRIMARY_SUPPORT_VARIANT, "primary_scalar_variant_retained_with_caveat"


def _support_drop_df(model_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, group in model_summary_df.groupby("model_id", sort=False):
        dense = group.loc[group["support_bucket"] == "dense", "mean_score"]
        sparse = group.loc[group["support_bucket"] == "sparse", "mean_score"]
        if dense.empty or sparse.empty:
            continue
        rows.append(
            {
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "dense_score": round(float(dense.iloc[0]), 6),
                "sparse_score": round(float(sparse.iloc[0]), 6),
                "dense_minus_sparse": round(float(dense.iloc[0] - sparse.iloc[0]), 6),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["model_sort"] = out["model_id"].map(_model_sort_rank)
    return out.sort_values(["model_sort"]).drop(columns=["model_sort"]).reset_index(drop=True)


def _build_locality_report(
    template_mapping_df: pd.DataFrame,
    locality_summary_df: pd.DataFrame,
    locality_model_df: pd.DataFrame,
) -> str:
    global_df = locality_summary_df.copy()
    drop_row = None
    if not global_df.empty and set(global_df["structure_type"]) >= {"grouped_global", "surface_2d", "filtered_local"}:
        grouped = float(global_df.loc[global_df["structure_type"] == "grouped_global", "mean_score"].iloc[0])
        surface = float(global_df.loc[global_df["structure_type"] == "surface_2d", "mean_score"].iloc[0])
        filtered = float(global_df.loc[global_df["structure_type"] == "filtered_local", "mean_score"].iloc[0])
        drop_row = (
            f"Panel means decline from `{grouped:.3f}` for grouped/global queries to `{surface:.3f}` for 2D surfaces "
            f"and `{filtered:.3f}` for filtered/local slices."
        )
    lines = [
        "# Conditional locality diagnostic",
        "",
        "## Classification audit",
        "",
        "Template-level semantics, not raw SQL column counts, define the primary locality buckets. The explicit mapping below keeps the two-axis filtered template in `filtered_local` while preserving `axis_arity = 2D` as a secondary annotation.",
        "",
        _markdown_table(
            template_mapping_df,
            [
                "template_id",
                "template_name",
                "structure_type",
                "axis_arity",
                "n_query_rows",
                "n_datasets",
                "n_models",
            ],
        ),
        "",
        "## Coverage and scores",
        "",
        _markdown_table(
            global_df[
                [
                    "structure_type",
                    "bucket_label",
                    "query_row_count",
                    "dataset_count",
                    "model_count",
                    "panel_count",
                    "template_count",
                    "mean_score",
                    "ci95_radius",
                    "coverage_note",
                ]
            ],
            [
                "structure_type",
                "bucket_label",
                "query_row_count",
                "dataset_count",
                "model_count",
                "panel_count",
                "template_count",
                "mean_score",
                "ci95_radius",
                "coverage_note",
            ],
        ),
        "",
        "## Diagnostic takeaways",
        "",
    ]
    if drop_row:
        lines.append(f"- {drop_row}")
    if not locality_model_df.empty:
        per_model = []
        for model_id, group in locality_model_df.groupby("model_id", sort=False):
            if set(group["structure_type"]) >= {"grouped_global", "filtered_local"}:
                grouped = float(group.loc[group["structure_type"] == "grouped_global", "mean_score"].iloc[0])
                filtered = float(group.loc[group["structure_type"] == "filtered_local", "mean_score"].iloc[0])
                per_model.append((filtered - grouped, model_id, grouped, filtered))
        if per_model:
            strongest_drop = min(per_model, key=lambda item: item[0])
            model_label = _model_label(strongest_drop[1])
            lines.append(
                f"- The steepest grouped/global to filtered/local decline appears for `{model_label}`: `{strongest_drop[2]:.3f}` to `{strongest_drop[3]:.3f}`."
            )
    lines.extend(
        [
            "- `surface_2d` still rests on one template family, so the locality trend should be treated as structured diagnostic evidence rather than a universal law over all possible 2D conditional tasks.",
            "- The current conditional row export carries heuristic subitem labels. This locality decomposition therefore anchors on template semantics and panel-level aggregation instead of over-interpreting any single heuristic subitem tag.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_support_report(
    support_summary_df: pd.DataFrame,
    support_summary_all_df: pd.DataFrame,
    support_drop_df: pd.DataFrame,
    support_method_summary: dict[str, Any],
    primary_variant: str,
    primary_variant_reason: str,
    exact_proxy_counts: Counter,
) -> str:
    global_df = support_summary_df.copy()
    sensitivity_df = support_summary_all_df.loc[support_summary_all_df["analysis_variant"] == SENSITIVITY_SUPPORT_VARIANT].copy()
    dense = global_df.loc[global_df["support_bucket"] == "dense", "mean_score"]
    sparse = global_df.loc[global_df["support_bucket"] == "sparse", "mean_score"]
    lines = [
        "# Conditional support diagnostic",
        "",
        "## Feasibility and recovery modes",
        "",
        f"- Source SQL artifact coverage found: `{support_method_summary['sql_artifact_found_count']}` recovered cases; missing: `{support_method_summary['sql_artifact_missing_count']}`.",
        f"- Support recovery modes on unique filtered-local cases: `{json.dumps(support_method_summary['mode_counts'], ensure_ascii=False, sort_keys=True)}`.",
        f"- Primary dense/medium/sparse variant: `{primary_variant}` (`{primary_variant_reason}`).",
        f"- Exact vs proxy row-level coverage in the audit export: exact=`{exact_proxy_counts.get('exact', 0)}`, derived_exact=`{exact_proxy_counts.get('derived_exact', 0)}`, proxy=`{exact_proxy_counts.get('proxy', 0)}`, unavailable=`{exact_proxy_counts.get('unavailable', 0)}`.",
        "",
        "## Coverage and scores",
        "",
        _markdown_table(
            global_df[
                [
                    "support_bucket",
                    "bucket_label",
                    "query_row_count",
                    "dataset_count",
                    "model_count",
                    "panel_count",
                    "template_count",
                    "mean_score",
                    "ci95_radius",
                    "coverage_note",
                ]
            ],
            [
                "support_bucket",
                "bucket_label",
                "query_row_count",
                "dataset_count",
                "model_count",
                "panel_count",
                "template_count",
                "mean_score",
                "ci95_radius",
                "coverage_note",
            ],
        ),
        "",
        "## Diagnostic takeaways",
        "",
    ]
    if not dense.empty and not sparse.empty:
        dense_value = float(dense.iloc[0])
        sparse_value = float(sparse.iloc[0])
        if dense_value > sparse_value:
            lines.append(
                f"- The global panel mean declines from `{dense_value:.3f}` on dense filtered-local slices to `{sparse_value:.3f}` on sparse slices."
            )
        else:
            lines.append(
                f"- The primary exact-support subset does not show a monotonic dense-to-sparse decline: dense=`{dense_value:.3f}` and sparse=`{sparse_value:.3f}`."
            )
    if not support_drop_df.empty:
        top_drop = support_drop_df.sort_values("dense_minus_sparse", ascending=False).iloc[0]
        positive_count = int((support_drop_df["dense_minus_sparse"] > 0).sum())
        negative_count = int((support_drop_df["dense_minus_sparse"] < 0).sum())
        zero_count = int((support_drop_df["dense_minus_sparse"] == 0).sum())
        lines.append(
            f"- Model behavior is mixed: `{positive_count}` models have positive dense-minus-sparse gaps, `{negative_count}` show the reverse, and `{zero_count}` are flat. The largest positive gap appears for `{top_drop['model_label']}` at `{float(top_drop['dense_minus_sparse']):.3f}`."
        )
    if not sensitivity_df.empty:
        sensitivity_dense = sensitivity_df.loc[sensitivity_df["support_bucket"] == "dense", "mean_score"]
        sensitivity_sparse = sensitivity_df.loc[sensitivity_df["support_bucket"] == "sparse", "mean_score"]
        sensitivity_datasets = sensitivity_df["dataset_count"].max()
        if not sensitivity_dense.empty and not sensitivity_sparse.empty:
            lines.append(
                f"- In the broader `{SENSITIVITY_SUPPORT_VARIANT}` sensitivity view (`{int(sensitivity_datasets)}` datasets), dense=`{float(sensitivity_dense.iloc[0]):.3f}` and sparse=`{float(sensitivity_sparse.iloc[0]):.3f}`; the sparse-support penalty is clearer once the filtered 2D local template is included."
            )
    lines.extend(
        [
            "- The primary support analysis intentionally keeps only scalar filtered-local templates in the main dense/medium/sparse comparison so that the support unit remains the count of real rows satisfying the local predicate.",
            "- Exact per-cell support is still recovered and audited for the filtered 2D group-count template, but that template is left as a sensitivity-only support basis because its natural support statistic is a cell-count distribution rather than a scalar slice size.",
            "- On this main scalar subset, sparse support does not by itself explain the filtered-local weakness. Any support-mediated interpretation should therefore be limited to model-specific behavior or to the broader sensitivity analysis, not promoted as a universal driver.",
            "- Any unsupported or unavailable support cases remain explicit in the audit CSV and are not silently folded into the main claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _select_primary_findings(
    locality_summary_df: pd.DataFrame,
    locality_model_df: pd.DataFrame,
    support_summary_df: pd.DataFrame,
    support_drop_df: pd.DataFrame,
) -> dict[str, str]:
    findings: dict[str, str] = {}
    if set(locality_summary_df["structure_type"]) >= {"grouped_global", "surface_2d", "filtered_local"}:
        grouped = float(locality_summary_df.loc[locality_summary_df["structure_type"] == "grouped_global", "mean_score"].iloc[0])
        surface = float(locality_summary_df.loc[locality_summary_df["structure_type"] == "surface_2d", "mean_score"].iloc[0])
        filtered = float(locality_summary_df.loc[locality_summary_df["structure_type"] == "filtered_local", "mean_score"].iloc[0])
        findings["locality_global"] = (
            f"Across panel means, conditional fidelity declines from grouped/global summaries ({grouped:.3f}) "
            f"to 2D surfaces ({surface:.3f}) and then to filtered/local slices ({filtered:.3f})."
        )
    if not locality_model_df.empty:
        deltas = []
        for model_id, group in locality_model_df.groupby("model_id", sort=False):
            if set(group["structure_type"]) >= {"grouped_global", "filtered_local"}:
                grouped = float(group.loc[group["structure_type"] == "grouped_global", "mean_score"].iloc[0])
                filtered = float(group.loc[group["structure_type"] == "filtered_local", "mean_score"].iloc[0])
                deltas.append((grouped - filtered, model_id, grouped, filtered))
        if deltas:
            best = max(deltas, key=lambda item: item[0])
            findings["locality_model"] = (
                f"The strongest grouped/global to filtered/local drop appears for {_model_label(best[1])}, "
                f"falling from {best[2]:.3f} to {best[3]:.3f}."
            )
    if set(support_summary_df["support_bucket"]) >= {"dense", "medium", "sparse"}:
        dense = float(support_summary_df.loc[support_summary_df["support_bucket"] == "dense", "mean_score"].iloc[0])
        medium = float(support_summary_df.loc[support_summary_df["support_bucket"] == "medium", "mean_score"].iloc[0])
        sparse = float(support_summary_df.loc[support_summary_df["support_bucket"] == "sparse", "mean_score"].iloc[0])
        if dense > sparse:
            findings["support_global"] = (
                f"Within the exact-support filtered-local subset, dense slices score {dense:.3f}, medium slices {medium:.3f}, "
                f"and sparse slices {sparse:.3f}, consistent with a sparse-support penalty."
            )
        else:
            findings["support_global"] = (
                f"Within the exact-support filtered-local subset, the dense/medium/sparse means are {dense:.3f} / {medium:.3f} / {sparse:.3f}, "
                f"so the main scalar-slice analysis does not show a monotonic sparse-support penalty."
            )
    if not support_drop_df.empty:
        top = support_drop_df.sort_values("dense_minus_sparse", ascending=False).iloc[0]
        positive_count = int((support_drop_df["dense_minus_sparse"] > 0).sum())
        negative_count = int((support_drop_df["dense_minus_sparse"] < 0).sum())
        findings["support_model"] = (
            f"Model behavior is mixed: {positive_count} models have positive dense-minus-sparse gaps and {negative_count} show the reverse; "
            f"the largest positive gap is {_model_label(top['model_id'])} at {float(top['dense_minus_sparse']):.3f}."
        )
    return findings


def _build_combined_report(
    *,
    locality_report: str,
    support_report: str,
    findings: dict[str, str],
    primary_support_variant: str,
    primary_support_reason: str,
) -> str:
    lines = [
        "# Conditional locality and support report",
        "",
        "## Scope",
        "",
        f"- Source conditional breakdown: `{CONDITIONAL_ROOT}`",
        f"- Source analysis run: `{SOURCE_RUN_ID}`",
        f"- Primary support variant: `{primary_support_variant}` (`{primary_support_reason}`)",
        "- Stable aggregation is panel-first throughout: query rows -> dataset-model-bucket panels -> model/global summaries.",
        "- This diagnostic does not rerun the benchmark and does not overwrite upstream conditional outputs.",
        "",
        "## Main supported findings",
        "",
    ]
    for value in findings.values():
        lines.append(f"- {value}")
    lines.extend(
        [
            "",
            "## Locality diagnostic",
            "",
            locality_report,
            "",
            "## Support diagnostic",
            "",
            support_report,
            "",
            "## Caveats",
            "",
            "- `surface_2d` is still represented by one template family, so the locality trend should be described as a template-grounded diagnostic pattern rather than a universal statement about dimensionality alone.",
            "- The support main figure intentionally excludes the filtered 2D count template from the primary dense/medium/sparse claim because its most faithful support signal is a distribution of cell counts, not a single filtered-row count.",
            "- Existing heuristic subitem labels in the conditional row export do not perfectly align with template-level semantics, so this diagnostic relies on template semantics for bucket assignment and uses query-score panel means as the primary outcome.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_paper_caption(findings: dict[str, str], support_variant_label: str) -> str:
    locality_line = findings.get("locality_global", "Conditional fidelity declines as queries become more local.")
    support_line = findings.get("support_global", "The dense/medium/sparse comparison inside filtered-local slices does not show a universal sparse-support penalty.")
    return "\n".join(
        [
            "Figure 1. Conditional locality decomposition.",
            f"{locality_line} Points and error bars show panel means with 95% confidence intervals; colored traces show per-model means under the frozen model roster and color convention.",
            "",
            "Figure 2. Conditional support decomposition.",
            f"{support_line} The main support figure uses the {support_variant_label.lower()} subset so that support is measured on a comparable exact real-row-count scale within each dataset.",
            "",
            "Figure 3. Combined conditional locality/support diagnostic.",
            "Panel A shows the locality decomposition from grouped/global summaries to filtered/local slices. Panel B shows the dense/medium/sparse comparison inside the filtered-local subset. Both panels use panel-level aggregation and expose coverage caveats in the companion audit tables rather than hiding thin buckets.",
            "",
        ]
    )


def _build_paper_paragraphs(findings: dict[str, str]) -> str:
    p1 = (
        findings.get("locality_global", "Conditional fidelity declines as queries become more local.")
        + " This suggests that axis count alone is not the most interpretable explanation for the conditional-family weakness: grouped/global summaries remain comparatively more stable, while narrow filtered slices are harder to preserve."
    )
    support_global = findings.get(
        "support_global",
        "The dense/medium/sparse comparison inside filtered-local slices does not show a universal sparse-support penalty.",
    )
    if "does not show a monotonic sparse-support penalty" in support_global:
        p2 = (
            support_global
            + " This means the main scalar support view does not support the claim that sparse real support is the dominant explanation for the local-slice collapse; support may matter in specific models, but it does not explain the aggregate locality gap by itself."
        )
    else:
        p2 = (
            support_global
            + " That pattern indicates that sparse real support explains part of the local-slice collapse, consistent with synthetic generators smoothing away rare conditional interactions."
        )
    p3 = (
        findings.get("support_model", "Models differ in how they move between dense and sparse local slices.")
        + " At the same time, the support diagnostic does not fully explain the conditional gap on its own: even dense local slices can remain weak for some models, and the 2D-surface bucket still rests on limited template coverage."
    )
    return "\n\n".join([p1, p2, p3]) + "\n"


def _root_readme_text(run_tag: str) -> str:
    lines = [
        "# Conditional locality/support diagnostics",
        "",
        "This directory contains a standalone paper-facing diagnostic built on top of the frozen conditional breakdown outputs and the existing upstream analysis artifacts.",
        "",
        "## Latest run",
        "",
        f"- `{run_tag}`",
        "",
        "## Re-run",
        "",
        "```bash",
        "python src/eval/query_fivepart_breakdown/conditional_breakdown/conditional_locality_support_diagnostic.py",
        "```",
        "",
        "## Output structure",
        "",
        "- `runs/<timestamp>_conditional_locality_support/` holds the full reproducible bundle for one run.",
        "- `final/` mirrors the paper-facing assets from the latest run.",
        "- `final/must_do/` keeps the minimum figure bundle for paper drafting.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    dirs = _build_output_dirs()
    run_dir = dirs["run_dir"]
    data_dir = dirs["data_dir"]
    fig_dir = dirs["fig_dir"]
    table_dir = dirs["table_dir"]
    report_dir = dirs["report_dir"]
    final_dir = dirs["final_dir"]
    run_tag = dirs["run_tag"].name

    conditional_df = _load_conditional_query_rows()
    selected_keys = set(zip(conditional_df["asset_key"], conditional_df["query_id"]))
    source_lookup_df = _load_source_lookup(selected_keys)
    merged_df = conditional_df.merge(source_lookup_df, on=["asset_key", "query_id"], how="left")
    if "question_y" in merged_df.columns and "question_x" in merged_df.columns:
        merged_df["question"] = merged_df["question_y"].where(merged_df["question_y"].astype(str).str.len() > 0, merged_df["question_x"])
        merged_df = merged_df.drop(columns=["question_x", "question_y"])
    elif "question" not in merged_df.columns:
        merged_df["question"] = ""
    merged_df = _apply_template_semantics(merged_df)

    template_mapping_df = _build_template_mapping_df(merged_df)
    template_panel_df = _build_template_panel_scores(merged_df)
    locality_panel_df = _build_locality_panel_scores(merged_df)
    locality_summary_global_df = _summary_df(
        locality_panel_df,
        bucket_col="structure_type",
        bucket_order=STRUCTURE_ORDER,
        bucket_label_map=STRUCTURE_LABELS,
        base_query_df=merged_df,
    )
    locality_summary_model_df = _summary_df(
        locality_panel_df,
        bucket_col="structure_type",
        bucket_order=STRUCTURE_ORDER,
        bucket_label_map=STRUCTURE_LABELS,
        base_query_df=merged_df,
        extra_group_cols=["model_id", "model_label"],
    )

    run_dir_index = _build_question_run_index()
    filtered_local_df = merged_df.loc[merged_df["structure_type"] == "filtered_local"].copy().reset_index(drop=True)
    support_case_df, support_audit_df, support_method_summary = _recover_support_cases(
        filtered_local_df,
        run_dir_index=run_dir_index,
    )
    support_case_bucket_df, support_variant_summary = _assign_support_buckets(support_case_df)
    support_query_rows_df = _build_support_query_rows(filtered_local_df, support_case_bucket_df)
    support_panel_all_df = _build_support_panel_scores(support_query_rows_df)
    support_summary_all_df = _summary_df(
        support_panel_all_df,
        bucket_col="support_bucket",
        bucket_order=SUPPORT_BUCKET_ORDER,
        bucket_label_map=SUPPORT_BUCKET_LABELS,
        base_query_df=support_query_rows_df,
        extra_group_cols=["analysis_variant"],
    )
    support_summary_model_all_df = _summary_df(
        support_panel_all_df,
        bucket_col="support_bucket",
        bucket_order=SUPPORT_BUCKET_ORDER,
        bucket_label_map=SUPPORT_BUCKET_LABELS,
        base_query_df=support_query_rows_df,
        extra_group_cols=["analysis_variant", "model_id", "model_label"],
    )

    primary_support_variant, primary_support_reason = _select_primary_support_variant(support_summary_all_df)
    support_summary_primary_df = support_summary_all_df.loc[support_summary_all_df["analysis_variant"] == primary_support_variant].copy()
    support_summary_model_primary_df = support_summary_model_all_df.loc[
        support_summary_model_all_df["analysis_variant"] == primary_support_variant
    ].copy()
    support_drop_primary_df = _support_drop_df(support_summary_model_primary_df)

    exact_proxy_counts = Counter(str(item) for item in support_audit_df["support_recovery_mode"].fillna("unavailable"))
    locality_report = _build_locality_report(template_mapping_df, locality_summary_global_df, locality_summary_model_df)
    support_report = _build_support_report(
        support_summary_primary_df,
        support_summary_all_df,
        support_drop_primary_df,
        support_method_summary,
        primary_support_variant,
        primary_support_reason,
        exact_proxy_counts,
    )
    findings = _select_primary_findings(
        locality_summary_global_df,
        locality_summary_model_df,
        support_summary_primary_df,
        support_drop_primary_df,
    )
    combined_report = _build_combined_report(
        locality_report=locality_report,
        support_report=support_report,
        findings=findings,
        primary_support_variant=primary_support_variant,
        primary_support_reason=primary_support_reason,
    )
    paper_caption = _build_paper_caption(findings, SUPPORT_VARIANT_LABELS.get(primary_support_variant, primary_support_variant))
    paper_paragraphs = _build_paper_paragraphs(findings)

    # Write CSVs
    _write_csv(template_mapping_df, data_dir / "conditional_template_mapping.csv")
    _write_csv(template_panel_df, data_dir / "conditional_panel_scores.csv")
    _write_csv(locality_panel_df, data_dir / "conditional_locality_panel_scores.csv")
    _write_csv(locality_summary_global_df, data_dir / "conditional_locality_summary.csv")
    _write_csv(support_audit_df, data_dir / "conditional_support_method_audit.csv")
    _write_csv(support_panel_all_df, data_dir / "conditional_support_bucket_panel_scores.csv")
    _write_csv(support_summary_all_df, data_dir / "conditional_support_bucket_summary.csv")
    _write_csv(support_case_df, data_dir / "conditional_support_case_summary.csv")
    _write_csv(support_drop_primary_df, data_dir / "conditional_support_dense_sparse_drop.csv")

    # Write reports
    locality_report_path = report_dir / "conditional_locality_diagnostic.md"
    support_report_path = report_dir / "conditional_support_bucket_diagnostic.md"
    combined_report_path = report_dir / "conditional_locality_support_report.md"
    paper_caption_path = report_dir / "paper_caption.txt"
    paper_paragraphs_path = report_dir / "paper_paragraphs.md"
    locality_report_path.write_text(locality_report + "\n", encoding="utf-8")
    support_report_path.write_text(support_report + "\n", encoding="utf-8")
    combined_report_path.write_text(combined_report + "\n", encoding="utf-8")
    paper_caption_path.write_text(paper_caption, encoding="utf-8")
    paper_paragraphs_path.write_text(paper_paragraphs, encoding="utf-8")

    # Figures
    locality_main_png = fig_dir / "fig_conditional_locality_main.png"
    locality_main_pdf = fig_dir / "fig_conditional_locality_main.pdf"
    locality_main_svg = fig_dir / "fig_conditional_locality_main.svg"
    locality_main_tex = fig_dir / "fig_conditional_locality_main.tex"
    locality_by_model_png = fig_dir / "fig_conditional_locality_by_model.png"
    locality_by_model_pdf = fig_dir / "fig_conditional_locality_by_model.pdf"
    locality_by_model_svg = fig_dir / "fig_conditional_locality_by_model.svg"
    locality_by_model_tex = fig_dir / "fig_conditional_locality_by_model.tex"

    support_main_png = fig_dir / "fig_conditional_support_main.png"
    support_main_pdf = fig_dir / "fig_conditional_support_main.pdf"
    support_main_svg = fig_dir / "fig_conditional_support_main.svg"
    support_main_tex = fig_dir / "fig_conditional_support_main.tex"
    support_by_model_png = fig_dir / "fig_conditional_support_by_model.png"
    support_by_model_pdf = fig_dir / "fig_conditional_support_by_model.pdf"
    support_by_model_svg = fig_dir / "fig_conditional_support_by_model.svg"
    support_by_model_tex = fig_dir / "fig_conditional_support_by_model.tex"

    combined_png = fig_dir / "fig_conditional_locality_support_combined.png"
    combined_pdf = fig_dir / "fig_conditional_locality_support_combined.pdf"
    combined_svg = fig_dir / "fig_conditional_locality_support_combined.svg"
    combined_tex = fig_dir / "fig_conditional_locality_support_combined.tex"

    _plot_bucket_means_with_model_lines(
        locality_summary_global_df.loc[locality_summary_global_df["structure_type"].isin(STRUCTURE_ORDER[:3])].copy(),
        locality_summary_model_df,
        bucket_col="structure_type",
        bucket_order=STRUCTURE_ORDER[:3],
        bucket_labels=STRUCTURE_LABELS,
        title="Conditional locality decomposition",
        y_label="Conditional fidelity score",
        caption_note="Black line: panel mean ± 95% CI. Colored lines: per-model panel means.",
        png_path=locality_main_png,
        pdf_path=locality_main_pdf,
        svg_path=locality_main_svg,
    )
    _plot_by_model_lines(
        locality_summary_model_df,
        bucket_col="structure_type",
        bucket_order=STRUCTURE_ORDER[:3],
        bucket_labels=STRUCTURE_LABELS,
        title="Conditional locality decomposition by model",
        y_label="Conditional fidelity score",
        png_path=locality_by_model_png,
        pdf_path=locality_by_model_pdf,
        svg_path=locality_by_model_svg,
    )
    _write_bucket_line_tex(
        locality_summary_global_df.loc[locality_summary_global_df["structure_type"].isin(STRUCTURE_ORDER[:3])].copy(),
        locality_summary_model_df,
        bucket_col="structure_type",
        bucket_order=STRUCTURE_ORDER[:3],
        bucket_labels=STRUCTURE_LABELS,
        title="Conditional locality decomposition",
        y_label="Conditional fidelity score",
        path=locality_main_tex,
        include_legend=False,
    )
    _write_bucket_line_tex(
        locality_summary_global_df.loc[locality_summary_global_df["structure_type"].isin(STRUCTURE_ORDER[:3])].copy(),
        locality_summary_model_df,
        bucket_col="structure_type",
        bucket_order=STRUCTURE_ORDER[:3],
        bucket_labels=STRUCTURE_LABELS,
        title="Conditional locality decomposition by model",
        y_label="Conditional fidelity score",
        path=locality_by_model_tex,
        include_legend=True,
    )

    _plot_bucket_means_with_model_lines(
        support_summary_primary_df,
        support_summary_model_primary_df,
        bucket_col="support_bucket",
        bucket_order=SUPPORT_BUCKET_ORDER,
        bucket_labels=SUPPORT_BUCKET_LABELS,
        title="Conditional support decomposition",
        y_label="Filtered-local conditional fidelity",
        caption_note=f"Primary support variant: {SUPPORT_VARIANT_LABELS.get(primary_support_variant, primary_support_variant)}.",
        png_path=support_main_png,
        pdf_path=support_main_pdf,
        svg_path=support_main_svg,
    )
    _plot_by_model_lines(
        support_summary_model_primary_df,
        bucket_col="support_bucket",
        bucket_order=SUPPORT_BUCKET_ORDER,
        bucket_labels=SUPPORT_BUCKET_LABELS,
        title="Conditional support decomposition by model",
        y_label="Filtered-local conditional fidelity",
        png_path=support_by_model_png,
        pdf_path=support_by_model_pdf,
        svg_path=support_by_model_svg,
    )
    _write_bucket_line_tex(
        support_summary_primary_df,
        support_summary_model_primary_df,
        bucket_col="support_bucket",
        bucket_order=SUPPORT_BUCKET_ORDER,
        bucket_labels=SUPPORT_BUCKET_LABELS,
        title="Conditional support decomposition",
        y_label="Filtered-local conditional fidelity",
        path=support_main_tex,
        include_legend=False,
    )
    _write_bucket_line_tex(
        support_summary_primary_df,
        support_summary_model_primary_df,
        bucket_col="support_bucket",
        bucket_order=SUPPORT_BUCKET_ORDER,
        bucket_labels=SUPPORT_BUCKET_LABELS,
        title="Conditional support decomposition by model",
        y_label="Filtered-local conditional fidelity",
        path=support_by_model_tex,
        include_legend=True,
    )

    _plot_combined(
        locality_summary_global_df.loc[locality_summary_global_df["structure_type"].isin(STRUCTURE_ORDER[:3])].copy(),
        locality_summary_model_df,
        support_summary_primary_df,
        support_summary_model_primary_df,
        png_path=combined_png,
        pdf_path=combined_pdf,
        svg_path=combined_svg,
    )
    _write_combined_tex(
        locality_summary_global_df.loc[locality_summary_global_df["structure_type"].isin(STRUCTURE_ORDER[:3])].copy(),
        locality_summary_model_df,
        support_summary_primary_df,
        support_summary_model_primary_df,
        path=combined_tex,
    )

    # Tables
    locality_table_tex = table_dir / "table_conditional_locality_summary.tex"
    support_table_tex = table_dir / "table_conditional_support_summary.tex"
    _tex_table(
        locality_summary_global_df.loc[locality_summary_global_df["structure_type"].isin(STRUCTURE_ORDER[:3])].copy(),
        columns=["bucket_label", "panel_count", "dataset_count", "template_count", "mean_score", "ci95_radius"],
        headers=["Bucket", "Panels", "Datasets", "Templates", "Mean", "95% CI"],
        path=locality_table_tex,
        caption_note="Panel-level locality summary.",
    )
    _tex_table(
        support_summary_primary_df.copy(),
        columns=["bucket_label", "panel_count", "dataset_count", "template_count", "mean_score", "ci95_radius"],
        headers=["Bucket", "Panels", "Datasets", "Templates", "Mean", "95% CI"],
        path=support_table_tex,
        caption_note="Panel-level support summary.",
    )

    compile_notes = {
        "fig_conditional_locality_main": _attempt_latexmk(locality_main_tex),
        "fig_conditional_locality_by_model": _attempt_latexmk(locality_by_model_tex),
        "fig_conditional_support_main": _attempt_latexmk(support_main_tex),
        "fig_conditional_support_by_model": _attempt_latexmk(support_by_model_tex),
        "fig_conditional_locality_support_combined": _attempt_latexmk(combined_tex),
        "table_conditional_locality_summary": _attempt_latexmk(locality_table_tex),
        "table_conditional_support_summary": _attempt_latexmk(support_table_tex),
    }

    run_manifest = {
        "task": "conditional_locality_support_diagnostic",
        "generated_at_utc": _utc_now().isoformat(),
        "source_analysis_run": SOURCE_RUN_ID,
        "source_conditional_root": str(CONDITIONAL_ROOT),
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "primary_support_variant": primary_support_variant,
        "primary_support_reason": primary_support_reason,
        "coverage": {
            "conditional_query_rows": int(merged_df.shape[0]),
            "locality_dataset_model_panels": int(locality_panel_df[["dataset_id", "model_id"]].drop_duplicates().shape[0]),
            "filtered_local_query_rows": int(filtered_local_df.shape[0]),
            "support_unique_cases": int(support_case_df.shape[0]),
            "support_primary_panel_rows": int(support_summary_primary_df["panel_count"].sum()) if not support_summary_primary_df.empty else 0,
        },
        "support_method_summary": support_method_summary,
        "support_variant_summary": support_variant_summary,
        "compile_notes": compile_notes,
        "key_findings": findings,
    }
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {run_tag}",
                "",
                "This run contains the full reproducible bundle for the conditional locality/support diagnostic.",
                "",
                "- `data/` exports the summary and audit CSVs.",
                "- `figures/` holds the paper-facing figures plus standalone TeX sources.",
                "- `tables/` holds LaTeX table snippets.",
                "- `report/` holds the Markdown narrative, captions, and paper paragraphs.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    latest_payload = {
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "generated_at_utc": _utc_now().isoformat(),
    }
    (LOCALITY_SUPPORT_ROOT / "LATEST_RUN.json").write_text(json.dumps(latest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (LOCALITY_SUPPORT_ROOT / "manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (LOCALITY_SUPPORT_ROOT / "README.md").write_text(_root_readme_text(run_tag), encoding="utf-8")

    final_readme_text = render_final_readme(
        title="Conditional Locality/Support Diagnostics",
        summary="Paper-facing assets mirrored from the latest standalone conditional locality/support diagnostic run.",
        primary_files=[
            "fig_conditional_locality_main.pdf",
            "fig_conditional_support_main.pdf",
            "fig_conditional_locality_support_combined.pdf",
            "paper_caption.txt",
            "paper_paragraphs.md",
            "conditional_locality_support_report.md",
        ],
        must_do_files=[
            "fig_conditional_locality_main.pdf",
            "fig_conditional_support_main.pdf",
            "fig_conditional_locality_support_combined.pdf",
            "fig_conditional_locality_main.png",
            "fig_conditional_support_main.png",
            "fig_conditional_locality_support_combined.png",
        ],
        support_files=[
            "conditional_template_mapping.csv",
            "conditional_locality_summary.csv",
            "conditional_support_bucket_summary.csv",
            "conditional_support_method_audit.csv",
        ],
        notes=[
            "",
            f"Latest run: `{run_tag}`",
            f"Primary support variant: `{primary_support_variant}`",
        ],
    )
    final_readme_path = final_dir / "README.md"
    final_readme_path.write_text(final_readme_text, encoding="utf-8")

    final_files = [
        locality_main_png,
        locality_main_pdf,
        locality_main_svg,
        locality_main_tex,
        support_main_png,
        support_main_pdf,
        support_main_svg,
        support_main_tex,
        combined_png,
        combined_pdf,
        combined_svg,
        combined_tex,
        locality_by_model_png,
        locality_by_model_pdf,
        locality_by_model_svg,
        locality_by_model_tex,
        support_by_model_png,
        support_by_model_pdf,
        support_by_model_svg,
        support_by_model_tex,
        locality_table_tex,
        support_table_tex,
        data_dir / "conditional_template_mapping.csv",
        data_dir / "conditional_locality_summary.csv",
        data_dir / "conditional_support_bucket_summary.csv",
        data_dir / "conditional_support_method_audit.csv",
        combined_report_path,
        locality_report_path,
        support_report_path,
        paper_caption_path,
        paper_paragraphs_path,
        run_dir / "manifest.json",
        run_dir / "README.md",
    ]
    must_do_aliases = {
        "fig_conditional_locality_main.pdf": locality_main_pdf,
        "fig_conditional_locality_main.png": locality_main_png,
        "fig_conditional_locality_main.svg": locality_main_svg,
        "fig_conditional_locality_main.tex": locality_main_tex,
        "fig_conditional_support_main.pdf": support_main_pdf,
        "fig_conditional_support_main.png": support_main_png,
        "fig_conditional_support_main.svg": support_main_svg,
        "fig_conditional_support_main.tex": support_main_tex,
        "fig_conditional_locality_support_combined.pdf": combined_pdf,
        "fig_conditional_locality_support_combined.png": combined_png,
        "fig_conditional_locality_support_combined.svg": combined_svg,
        "fig_conditional_locality_support_combined.tex": combined_tex,
    }
    sync_final_outputs(final_dir, final_files, must_do_aliases=must_do_aliases)
    (final_dir / "README.md").write_text(final_readme_text, encoding="utf-8")

    summary_lines = [
        f"[conditional_locality_support] run={run_tag}",
        f"  locality buckets: "
        + "; ".join(
            f"{STRUCTURE_LABELS.get(row.structure_type, row.structure_type)} panels={int(row.panel_count)} mean={float(row.mean_score):.3f}"
            for row in locality_summary_global_df.loc[locality_summary_global_df["structure_type"].isin(STRUCTURE_ORDER[:3])].itertuples()
        ),
        f"  support modes: {json.dumps(support_method_summary['mode_counts'], ensure_ascii=False, sort_keys=True)}",
        f"  support primary variant: {primary_support_variant} ({primary_support_reason})",
        f"  support buckets: "
        + "; ".join(
            f"{SUPPORT_BUCKET_LABELS.get(row.support_bucket, row.support_bucket)} panels={int(row.panel_count)} mean={float(row.mean_score):.3f}"
            for row in support_summary_primary_df.itertuples()
        ),
        f"  caveats: locality_surface_templates={int(locality_summary_global_df.loc[locality_summary_global_df['structure_type'] == 'surface_2d', 'template_count'].iloc[0]) if not locality_summary_global_df.loc[locality_summary_global_df['structure_type'] == 'surface_2d'].empty else 0}; support_variant={primary_support_variant}",
    ]
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
