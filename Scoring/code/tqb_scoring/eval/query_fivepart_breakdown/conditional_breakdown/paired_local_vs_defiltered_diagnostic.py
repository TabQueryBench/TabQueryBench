#!/usr/bin/env python3
"""Paired diagnostic for filtered-local conditional queries vs defiltered counterparts.

This diagnostic keeps the original grounded local conditional query fixed and
constructs an unfiltered counterpart by removing the predicate/WHERE clause
from the generated SQL. The goal is a paper-facing comparison where the broad
and local variants differ only by the presence of the filter.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
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

from tqb_query.benchmark.sql_exec import execute_sql
from tqb_scoring.eval.analysis.runner import _build_real_sqlite, _build_synthetic_sqlite
from tqb_scoring.eval.common import ROOT_CONFIGS, load_sql_result_role_annotations, now_run_tag, write_csv, write_json
from tqb_scoring.eval.query_fivepart_breakdown.common_final import render_final_readme, sync_final_outputs
from tqb_scoring.evaluation.real_panel_experiment import _compare_execution_results


EVALUATION_ROOT = PROJECT_ROOT.parent / "results"
CONDITIONAL_ROOT = EVALUATION_ROOT / "query_fivepart_breakdown" / "conditional_breakdown"
OUTPUT_ROOT = CONDITIONAL_ROOT / "paired_local_vs_defiltered_diagnostic"
PAPER_ROOT = PROJECT_ROOT / "Paper" / "69b27219c555c38a69bb2156"
PAPER_TABLE_PATH = PAPER_ROOT / "figures" / "conditional_breakdown" / "conditional_broad_local_template_table_embedded.tex"


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
PREFIX_LABELS = {"c": "Categorical", "m": "Mixed", "n": "Numerical"}
LOCAL_TEMPLATE_ORDER = [
    "tpl_c2_filtered_group_count_2d",
    "tpl_m4_median_filtered_numeric",
    "tpl_tpch_filtered_sum_band",
    "tpl_conditional_group_quantiles",
    "tpl_rtabench_time_bucket_filtered_count",
]
LOCAL_TEMPLATE_LABELS = {
    "tpl_c2_filtered_group_count_2d": "L1 Filtered 2D Group Count",
    "tpl_m4_median_filtered_numeric": "L2 Filtered Median Slice",
    "tpl_tpch_filtered_sum_band": "L3 Filtered Numeric-Band Sum",
    "tpl_conditional_group_quantiles": "L4 Filtered Group Quantile",
    "tpl_rtabench_time_bucket_filtered_count": "L5 Filtered Time-Bucket Count",
}
LOCAL_VARIANT_LABEL = "Filtered local"
BROAD_VARIANT_LABEL = "Defiltered counterpart"
SOURCE_RUN_ID = os.environ.get("PAIRED_LOCAL_SOURCE_RUN_ID", "v2_keyset_cts_48_20260504_2350").strip() or "v2_keyset_cts_48_20260504_2350"
SOURCE_QUERY_JSONL = EVALUATION_ROOT / "analysis" / "runs" / SOURCE_RUN_ID / "summaries" / "analysis_query_scores__all_datasets.jsonl"
LOGS_RUN_ROOT = Path(os.environ.get("PAIRED_LOCAL_LOGS_ROOT", str(PROJECT_ROOT.parent.parent / "Query" / "code" / "logs" / "subitem_workload_v2" / "runs"))).expanduser()
CACHE_ROOT = OUTPUT_ROOT / "cache"
ROW_LIMIT = 0


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


def _model_sort_rank(model_id: str) -> int:
    return MODEL_ORDER.index(model_id) if model_id in MODEL_ORDER else 999


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return float(numeric)


def _quantile(series: pd.Series, q: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return round(float(clean.quantile(q)), 6)


def _mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    return round(float(mean(clean)), 6)


def _markdown_table(df: pd.DataFrame) -> str:
    safe = df.copy().replace({np.nan: ""})
    columns = [str(col) for col in safe.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in safe.iterrows():
        values = [str(row[col]) if row[col] != "" else "" for col in safe.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


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
    return None


def _normalize_local_path(win_path: str) -> Path | None:
    text = str(win_path or "").strip()
    if not text:
        return None
    direct = Path(text)
    if direct.exists():
        return direct
    normalized = Path(text.replace("\\", "/"))
    if normalized.exists():
        return normalized
    return None


def _resolve_synthetic_csv_path(row: dict[str, Any]) -> Path | None:
    direct = _normalize_local_path(str(row.get("synthetic_csv_path") or row.get("synthetic_source_path") or ""))
    if direct is not None:
        return direct
    root_name = str(row.get("root_name") or row.get("synthetic_source_root_name") or "").strip()
    root_cfg = ROOT_CONFIGS.get(root_name)
    if not root_cfg:
        return None
    file_name = Path(str(row.get("synthetic_csv_path") or row.get("synthetic_source_path") or "").replace("\\", "/")).name
    dataset_id = str(row.get("dataset_id") or "")
    model_id = str(row.get("model_id") or "")
    candidate = Path(root_cfg["path"]) / dataset_id / model_id / "synthetic_data" / file_name
    if candidate.exists():
        return candidate
    asset_dir = Path(root_cfg["path"]) / dataset_id / model_id
    for found in asset_dir.rglob(file_name):
        return found
    return None


def _remove_where_clause(sql_text: str) -> str | None:
    pattern = re.compile(
        r"(?P<prefix>\bFROM\s+(?:\"[^\"]+\"|\[[^\]]+\]|\w+))\s+WHERE\s+(?P<where>.*?)(?=(\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|;|\)\s*SELECT|$))",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(sql_text)
    if not match:
        return None
    return sql_text[: match.start()] + match.group("prefix") + sql_text[match.end() :]


def _build_defiltered_sql(template_id: str, sql_text: str) -> str | None:
    if template_id not in set(LOCAL_TEMPLATE_ORDER):
        return None
    return _remove_where_clause(sql_text)


def _load_local_source_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with SOURCE_QUERY_JSONL.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("family_id") or "") != "conditional_dependency_structure":
                continue
            model_id = str(row.get("model_id") or "").strip().lower()
            if model_id not in MODEL_ORDER:
                continue
            template_id = str(row.get("template_id") or "")
            if template_id not in set(LOCAL_TEMPLATE_ORDER):
                continue
            details = row.get("details") or {}
            local_score = _safe_float(details.get("key_set_score"))
            if local_score is None:
                local_score = _safe_float(row.get("query_score"))
            rows.append(
                {
                    "dataset_id": str(row.get("dataset_id") or ""),
                    "dataset_prefix": _dataset_prefix(str(row.get("dataset_id") or "")),
                    "model_id": model_id,
                    "model_label": _model_label(model_id),
                    "asset_key": str(row.get("asset_key") or ""),
                    "query_id": str(row.get("query_id") or ""),
                    "template_id": template_id,
                    "template_label": LOCAL_TEMPLATE_LABELS.get(template_id, template_id),
                    "template_name": str(row.get("template_name") or ""),
                    "question": str(row.get("question") or ""),
                    "source_sql_run_id": str(row.get("source_sql_run_id") or ""),
                    "local_score": local_score,
                    "local_query_score_raw": _safe_float(row.get("query_score")),
                    "local_key_set_score": _safe_float(details.get("key_set_score")),
                    "local_details_json": json.dumps(details, ensure_ascii=False, sort_keys=True),
                    "synthetic_csv_path": str(row.get("synthetic_csv_path") or row.get("synthetic_source_path") or ""),
                    "root_name": str(row.get("root_name") or row.get("synthetic_source_root_name") or ""),
                    "run_id": str(row.get("run_id") or row.get("synthetic_source_run_id") or ""),
                    "server_type": str(row.get("server_type") or ""),
                    "sql_source_version": str(row.get("sql_source_version") or "v2"),
                }
            )
    out = pd.DataFrame(rows)
    out["model_sort"] = out["model_id"].map(_model_sort_rank)
    out["dataset_sort"] = out["dataset_id"].map(_dataset_sort_key)
    out = out.sort_values(["dataset_sort", "model_sort", "template_id", "query_id"]).drop(columns=["model_sort", "dataset_sort"]).reset_index(drop=True)
    return out


def _score_defiltered_counterparts(source_df: pd.DataFrame) -> pd.DataFrame:
    run_dir_index = _build_question_run_index()
    annotation_cache: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    real_sqlite_cache: dict[str, tuple[Path, str]] = {}
    synthetic_sqlite_cache: dict[str, Path] = {}
    real_exec_cache: dict[tuple[str, str], Any] = {}
    local_sql_cache: dict[tuple[str, str, str], tuple[str, str | None, str]] = {}

    rows: list[dict[str, Any]] = []
    total_rows = int(source_df.shape[0])

    for idx, row in enumerate(source_df.to_dict(orient="records"), start=1):
        dataset_id = row["dataset_id"]
        sql_source_version = row["sql_source_version"] or "v2"
        if dataset_id not in annotation_cache:
            annotation_cache[dataset_id] = load_sql_result_role_annotations(dataset_id, sql_source_version=sql_source_version)
        annotation = annotation_cache[dataset_id].get((sql_source_version, row["query_id"]))

        sql_cache_key = (row["source_sql_run_id"], dataset_id, row["query_id"])
        if sql_cache_key not in local_sql_cache:
            run_dir = run_dir_index.get(row["source_sql_run_id"])
            sql_path = _resolve_generated_sql_path(run_dir, dataset_id, row["query_id"])
            local_sql = sql_path.read_text(encoding="utf-8").strip() if sql_path and sql_path.exists() else ""
            broad_sql = _build_defiltered_sql(row["template_id"], local_sql) if local_sql else None
            local_sql_cache[sql_cache_key] = (local_sql, broad_sql, str(sql_path) if sql_path else "")
        local_sql, broad_sql, sql_path_text = local_sql_cache[sql_cache_key]

        base = {
            **row,
            "source_sql_path": sql_path_text,
            "local_sql": local_sql,
            "broad_sql": broad_sql or "",
            "broad_score": None,
            "broad_key_set_score": None,
            "broad_profile_score": None,
            "broad_row_count_score": None,
            "broad_column_score": None,
            "delta_broad_minus_local": None,
            "broad_exec_ok": False,
            "synthetic_exec_ok": False,
            "scoring_note": "",
        }

        if not broad_sql:
            base["scoring_note"] = "could_not_build_defiltered_sql"
            rows.append(base)
            continue

        synthetic_csv = _resolve_synthetic_csv_path(row)
        if synthetic_csv is None or not synthetic_csv.exists():
            base["scoring_note"] = "synthetic_csv_missing"
            rows.append(base)
            continue

        if dataset_id not in real_sqlite_cache:
            real_sqlite_cache[dataset_id] = _build_real_sqlite(dataset_id, CACHE_ROOT)
        real_sqlite_path, table_name = real_sqlite_cache[dataset_id]

        asset_key = row["asset_key"]
        if asset_key not in synthetic_sqlite_cache:
            dummy_asset = type("DummyAsset", (), {})()
            dummy_asset.dataset_id = dataset_id
            dummy_asset.asset_key = asset_key
            dummy_asset.synthetic_csv_path = str(synthetic_csv)
            synthetic_sqlite_cache[asset_key] = _build_synthetic_sqlite(dummy_asset, CACHE_ROOT, table_name)
        synthetic_sqlite_path = synthetic_sqlite_cache[asset_key]

        real_exec_key = (dataset_id, broad_sql)
        if real_exec_key not in real_exec_cache:
            real_exec_cache[real_exec_key] = execute_sql(real_sqlite_path, broad_sql, row_limit=ROW_LIMIT)
        real_exec = real_exec_cache[real_exec_key]
        syn_exec = execute_sql(synthetic_sqlite_path, broad_sql, row_limit=ROW_LIMIT)
        _, detail = _compare_execution_results(real_exec, syn_exec, result_role_annotation=annotation)

        broad_score = _safe_float(detail.get("key_set_score"))
        base.update(
            {
                "broad_score": broad_score,
                "broad_key_set_score": broad_score,
                "broad_profile_score": _safe_float(detail.get("profile_score")),
                "broad_row_count_score": _safe_float(detail.get("row_count_score")),
                "broad_column_score": _safe_float(detail.get("column_score")),
                "delta_broad_minus_local": (round(float(broad_score - row["local_score"]), 6) if broad_score is not None and row["local_score"] is not None else None),
                "broad_exec_ok": bool(getattr(real_exec, "ok", False)) and bool(getattr(syn_exec, "ok", False)),
                "synthetic_exec_ok": bool(getattr(syn_exec, "ok", False)),
                "scoring_note": "key_set_score_only_for_paired_comparability",
                "broad_details_json": json.dumps(detail, ensure_ascii=False, sort_keys=True),
            }
        )
        rows.append(base)
        if idx == 1 or idx % 100 == 0 or idx == total_rows:
            print(
                f"[paired-local] progress={idx}/{total_rows} dataset={dataset_id} model={row['model_id']} "
                f"template={row['template_id']} cached_real={len(real_exec_cache)} cached_assets={len(synthetic_sqlite_cache)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def _build_panel_df(df: pd.DataFrame) -> pd.DataFrame:
    panel = (
        df.groupby(
            ["dataset_id", "dataset_prefix", "model_id", "model_label", "template_id", "template_label"],
            dropna=False,
            as_index=False,
        )
        .agg(
            local_score=("local_score", "mean"),
            broad_score=("broad_score", "mean"),
            query_count=("query_id", "count"),
        )
        .reset_index(drop=True)
    )
    panel["delta_broad_minus_local"] = pd.to_numeric(panel["broad_score"], errors="coerce") - pd.to_numeric(panel["local_score"], errors="coerce")
    panel["model_sort"] = panel["model_id"].map(_model_sort_rank)
    panel["dataset_sort"] = panel["dataset_id"].map(_dataset_sort_key)
    panel = panel.sort_values(["dataset_sort", "model_sort", "template_id"]).drop(columns=["model_sort", "dataset_sort"]).reset_index(drop=True)
    return panel


def _build_variant_table_rows(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    for template_id in LOCAL_TEMPLATE_ORDER:
        subset = panel_df.loc[panel_df["template_id"] == template_id].copy()
        if subset.empty:
            continue
        for variant, score_col in [(BROAD_VARIANT_LABEL, "broad_score"), (LOCAL_VARIANT_LABEL, "local_score")]:
            series = pd.to_numeric(subset[score_col], errors="coerce").dropna()
            rows.append(
                {
                    "bucket": variant,
                    "template_id": template_id,
                    "template_label": LOCAL_TEMPLATE_LABELS.get(template_id, template_id),
                    "dataset_count": int(subset["dataset_id"].nunique()),
                    "query_count": int(subset["query_count"].sum()),
                    "mean_score": round(float(series.mean()), 6) if not series.empty else None,
                    "median_score": round(float(series.median()), 6) if not series.empty else None,
                    "q1_score": _quantile(series, 0.25),
                    "q3_score": _quantile(series, 0.75),
                }
            )
        template_rows.append(
            {
                "template_id": template_id,
                "template_label": LOCAL_TEMPLATE_LABELS.get(template_id, template_id),
                "dataset_count": int(subset["dataset_id"].nunique()),
                "query_count": int(subset["query_count"].sum()),
                "broad_mean": _mean_or_none(subset["broad_score"].tolist()),
                "local_mean": _mean_or_none(subset["local_score"].tolist()),
                "delta_mean": _mean_or_none(subset["delta_broad_minus_local"].tolist()),
            }
        )
    table_df = pd.DataFrame(rows)
    overall_rows = []
    for variant, score_col in [(BROAD_VARIANT_LABEL, "broad_score"), (LOCAL_VARIANT_LABEL, "local_score")]:
        series = pd.to_numeric(panel_df[score_col], errors="coerce").dropna()
        overall_rows.append(
            {
                "bucket": variant,
                "template_id": "overall",
                "template_label": "overall",
                "dataset_count": int(panel_df["dataset_id"].nunique()),
                "query_count": int(panel_df["query_count"].sum()),
                "mean_score": round(float(series.mean()), 6) if not series.empty else None,
                "median_score": round(float(series.median()), 6) if not series.empty else None,
                "q1_score": _quantile(series, 0.25),
                "q3_score": _quantile(series, 0.75),
            }
        )
    full_table_df = pd.concat([pd.DataFrame(overall_rows), table_df], ignore_index=True)
    return full_table_df, pd.DataFrame(template_rows)


def _build_model_summary(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, subset in panel_df.groupby("model_id", sort=False):
        rows.append(
            {
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "dataset_count": int(subset["dataset_id"].nunique()),
                "template_count": int(subset["template_id"].nunique()),
                "query_count": int(subset["query_count"].sum()),
                "broad_mean": _mean_or_none(subset["broad_score"].tolist()),
                "local_mean": _mean_or_none(subset["local_score"].tolist()),
                "delta_mean": _mean_or_none(subset["delta_broad_minus_local"].tolist()),
            }
        )
    out = pd.DataFrame(rows)
    out["model_sort"] = out["model_id"].map(_model_sort_rank)
    out = out.sort_values(["model_sort"]).drop(columns=["model_sort"]).reset_index(drop=True)
    return out


def _build_dataset_summary(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, subset in panel_df.groupby("dataset_id", sort=False):
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": _dataset_prefix(dataset_id),
                "model_count": int(subset["model_id"].nunique()),
                "template_count": int(subset["template_id"].nunique()),
                "query_count": int(subset["query_count"].sum()),
                "broad_mean": _mean_or_none(subset["broad_score"].tolist()),
                "local_mean": _mean_or_none(subset["local_score"].tolist()),
                "delta_mean": _mean_or_none(subset["delta_broad_minus_local"].tolist()),
            }
        )
    out = pd.DataFrame(rows)
    out["dataset_sort"] = out["dataset_id"].map(_dataset_sort_key)
    out = out.sort_values(["dataset_sort"]).drop(columns=["dataset_sort"]).reset_index(drop=True)
    return out


def _plot_template_bars(template_summary_df: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    order = [tid for tid in LOCAL_TEMPLATE_ORDER if tid in set(template_summary_df["template_id"])]
    x = np.arange(len(order))
    broad = [float(template_summary_df.loc[template_summary_df["template_id"] == tid, "broad_mean"].iloc[0]) for tid in order]
    local = [float(template_summary_df.loc[template_summary_df["template_id"] == tid, "local_mean"].iloc[0]) for tid in order]
    labels = [LOCAL_TEMPLATE_LABELS[tid].replace("Filtered ", "").replace(" Numeric-", " Num-") for tid in order]
    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    width = 0.34
    ax.bar(x - width / 2, broad, width=width, color="#4C78A8", label=BROAD_VARIANT_LABEL)
    ax.bar(x + width / 2, local, width=width, color="#F58518", label=LOCAL_VARIANT_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean paired score")
    ax.set_title("Conditional local queries are weaker than their defiltered counterparts")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_model_dumbbell(model_summary_df: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    ordered = model_summary_df.copy().reset_index(drop=True)
    y = np.arange(len(ordered))
    for idx, row in ordered.iterrows():
        model_id = str(row["model_id"])
        broad = float(row["broad_mean"])
        local = float(row["local_mean"])
        color = MODEL_COLORS.get(model_id, "#555555")
        ax.plot([broad, local], [idx, idx], color=color, linewidth=2.1, alpha=0.85)
        ax.scatter(broad, idx, marker="o", s=55, color="#4C78A8", zorder=3)
        ax.scatter(local, idx, marker="o", s=55, color="#F58518", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model_label"].tolist())
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean paired score")
    ax.set_title("Model-level paired gap: defiltered broad vs filtered local")
    ax.grid(axis="x", linestyle="--", alpha=0.24)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_panel_scatter(panel_df: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.5))
    for template_id, subset in panel_df.groupby("template_id", sort=False):
        ax.scatter(
            subset["broad_score"],
            subset["local_score"],
            s=26,
            alpha=0.72,
            label=LOCAL_TEMPLATE_LABELS.get(template_id, template_id),
        )
    ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Defiltered counterpart score")
    ax.set_ylabel("Filtered local score")
    ax.set_title("Paired local-vs-defiltered panel scores")
    ax.grid(linestyle="--", alpha=0.18)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_embedded_tex(table_df: pd.DataFrame, path: Path) -> None:
    def fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "--"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.3f}"

    overall_broad = table_df.loc[(table_df["bucket"] == BROAD_VARIANT_LABEL) & (table_df["template_id"] == "overall")].iloc[0]
    overall_local = table_df.loc[(table_df["bucket"] == LOCAL_VARIANT_LABEL) & (table_df["template_id"] == "overall")].iloc[0]

    template_lines = []
    label_map = {
        "tpl_c2_filtered_group_count_2d": "P1",
        "tpl_m4_median_filtered_numeric": "P2",
        "tpl_tpch_filtered_sum_band": "P3",
        "tpl_conditional_group_quantiles": "P4",
        "tpl_rtabench_time_bucket_filtered_count": "P5",
    }
    for template_id in LOCAL_TEMPLATE_ORDER:
        broad_row = table_df.loc[(table_df["bucket"] == BROAD_VARIANT_LABEL) & (table_df["template_id"] == template_id)]
        local_row = table_df.loc[(table_df["bucket"] == LOCAL_VARIANT_LABEL) & (table_df["template_id"] == template_id)]
        if broad_row.empty or local_row.empty:
            continue
        broad_row = broad_row.iloc[0]
        local_row = local_row.iloc[0]
        template_lines.extend(
            [
                f"& {label_map[template_id]} broad & \\textbf{{{int(broad_row['dataset_count'])}}} & \\textbf{{{int(broad_row['query_count'])}}} & \\textbf{{{fmt(broad_row['mean_score'])}}} & \\textbf{{{fmt(broad_row['median_score'])}}} & \\textbf{{{fmt(broad_row['q1_score'])}}} & \\textbf{{{fmt(broad_row['q3_score'])}}} \\\\",
                f"& {label_map[template_id]} local & \\textbf{{{int(local_row['dataset_count'])}}} & \\textbf{{{int(local_row['query_count'])}}} & \\textbf{{{fmt(local_row['mean_score'])}}} & \\textbf{{{fmt(local_row['median_score'])}}} & \\textbf{{{fmt(local_row['q1_score'])}}} & \\textbf{{{fmt(local_row['q3_score'])}}} \\\\",
            ]
        )

    text = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            r"\begin{tabular}{llcccccc}",
            r"\toprule",
            r"Bucket & Template & Dsets & Queries & Mean & Median & Q1 & Q3 \\",
            r"\midrule",
            rf"\textbf{{Defiltered}} & \textbf{{overall}} & \textbf{{{int(overall_broad['dataset_count'])}}} & \textbf{{{int(overall_broad['query_count'])}}} & \textbf{{{fmt(overall_broad['mean_score'])}}} & \textbf{{{fmt(overall_broad['median_score'])}}} & \textbf{{{fmt(overall_broad['q1_score'])}}} & \textbf{{{fmt(overall_broad['q3_score'])}}} \\",
            *template_lines,
            r"\midrule",
            rf"\textbf{{Local}} & \textbf{{overall}} & \textbf{{{int(overall_local['dataset_count'])}}} & \textbf{{{int(overall_local['query_count'])}}} & \textbf{{{fmt(overall_local['mean_score'])}}} & \textbf{{{fmt(overall_local['median_score'])}}} & \textbf{{{fmt(overall_local['q1_score'])}}} & \textbf{{{fmt(overall_local['q3_score'])}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\vspace{0.28em}",
            r"\begin{minipage}{0.97\columnwidth}",
            r"\raggedright",
            r"\footnotesize",
            r"\textbf{Template notes.}",
            r"\textbf{P1}: Filtered Two-Dimensional Group Count VS the same query with the predicate removed.",
            r"\textbf{P2}: Filtered Median Numeric Slice VS the same query with the predicate removed.",
            r"\textbf{P3}: Filtered Sum in Numeric Band VS the same query with the numeric-band filter removed.",
            r"\textbf{P4}: Filtered Group Quantile VS the same query with the group-level filter removed.",
            r"\textbf{P5}: Filtered Time-Bucket Count VS the same query with the class filter removed.",
            r"\end{minipage}",
            r"\caption{\textbf{Defiltered counterparts VS filtered local slices.}}",
            r"\label{tab:conditional-broad-local}",
            r"\end{table}",
            "",
        ]
    )
    _write_text(path, text)


def _build_report_text(
    *,
    source_rows_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    template_summary_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
) -> str:
    overall_broad = _mean_or_none(panel_df["broad_score"].tolist())
    overall_local = _mean_or_none(panel_df["local_score"].tolist())
    overall_delta = _mean_or_none(panel_df["delta_broad_minus_local"].tolist())
    best_gap = model_summary_df.sort_values("delta_mean", ascending=False).head(3)
    lines = [
        "# Paired local-vs-defiltered conditional diagnostic",
        "",
        f"- Source analysis run: `{SOURCE_RUN_ID}`",
        f"- Included local templates: `{', '.join(LOCAL_TEMPLATE_ORDER)}`",
        f"- Query rows scored: `{int(source_rows_df.shape[0])}`",
        f"- Panel count: `{int(panel_df.shape[0])}`",
        f"- Overall defiltered mean: `{overall_broad:.3f}`" if overall_broad is not None else "- Overall defiltered mean: `NA`",
        f"- Overall local mean: `{overall_local:.3f}`" if overall_local is not None else "- Overall local mean: `NA`",
        f"- Mean broad-minus-local gap: `{overall_delta:.3f}`" if overall_delta is not None else "- Mean broad-minus-local gap: `NA`",
        "",
        "## Template summary",
        "",
        _markdown_table(template_summary_df),
        "",
        "## Largest model-level paired gaps",
        "",
        _markdown_table(best_gap),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    run_tag = now_run_tag() + "_paired_local_vs_defiltered"
    run_dir = OUTPUT_ROOT / "runs" / run_tag
    data_dir = run_dir / "data"
    fig_dir = run_dir / "figures"
    report_dir = run_dir / "report"
    final_dir = OUTPUT_ROOT / "final"
    for path in [OUTPUT_ROOT, run_dir, data_dir, fig_dir, report_dir, final_dir, CACHE_ROOT]:
        path.mkdir(parents=True, exist_ok=True)

    source_rows_df = _load_local_source_rows()
    paired_df = _score_defiltered_counterparts(source_rows_df)
    paired_df = paired_df.loc[pd.notna(paired_df["broad_score"])].copy().reset_index(drop=True)
    panel_df = _build_panel_df(paired_df)
    table_df, template_summary_df = _build_variant_table_rows(panel_df)
    model_summary_df = _build_model_summary(panel_df)
    dataset_summary_df = _build_dataset_summary(panel_df)

    write_csv(data_dir / "paired_query_rows.csv", paired_df.to_dict(orient="records"))
    write_csv(data_dir / "paired_panel_scores.csv", panel_df.to_dict(orient="records"))
    write_csv(data_dir / "paired_variant_table_summary.csv", table_df.to_dict(orient="records"))
    write_csv(data_dir / "paired_template_summary.csv", template_summary_df.to_dict(orient="records"))
    write_csv(data_dir / "paired_model_summary.csv", model_summary_df.to_dict(orient="records"))
    write_csv(data_dir / "paired_dataset_summary.csv", dataset_summary_df.to_dict(orient="records"))

    template_bar_png = fig_dir / "fig_local_vs_defiltered_by_template.png"
    template_bar_pdf = fig_dir / "fig_local_vs_defiltered_by_template.pdf"
    model_dumbbell_png = fig_dir / "fig_local_vs_defiltered_by_model.png"
    model_dumbbell_pdf = fig_dir / "fig_local_vs_defiltered_by_model.pdf"
    scatter_png = fig_dir / "fig_local_vs_defiltered_scatter.png"
    scatter_pdf = fig_dir / "fig_local_vs_defiltered_scatter.pdf"
    _plot_template_bars(template_summary_df, template_bar_png, template_bar_pdf)
    _plot_model_dumbbell(model_summary_df, model_dumbbell_png, model_dumbbell_pdf)
    _plot_panel_scatter(panel_df, scatter_png, scatter_pdf)

    report_text = _build_report_text(
        source_rows_df=source_rows_df,
        panel_df=panel_df,
        template_summary_df=template_summary_df,
        model_summary_df=model_summary_df,
    )
    _write_text(report_dir / "paired_local_vs_defiltered_report.md", report_text)
    _write_embedded_tex(table_df, PAPER_TABLE_PATH)

    primary_files = [
        data_dir / "paired_variant_table_summary.csv",
        data_dir / "paired_model_summary.csv",
        template_bar_pdf,
        model_dumbbell_pdf,
        scatter_pdf,
        report_dir / "paired_local_vs_defiltered_report.md",
    ]
    must_do_aliases = {
        "fig_local_vs_defiltered_by_template.pdf": template_bar_pdf,
        "fig_local_vs_defiltered_by_model.pdf": model_dumbbell_pdf,
        "fig_local_vs_defiltered_scatter.pdf": scatter_pdf,
        "paired_variant_table_summary.csv": data_dir / "paired_variant_table_summary.csv",
        "paired_model_summary.csv": data_dir / "paired_model_summary.csv",
    }
    sync_final_outputs(final_dir, primary_files, must_do_aliases=must_do_aliases, version_tag="v1")

    readme_text = render_final_readme(
        title="conditional paired local-vs-defiltered diagnostic",
        summary="Paper-facing paired diagnostic that compares each filtered-local conditional query against the same SQL with its filter removed.",
        primary_files=[
            "fig_local_vs_defiltered_by_template.pdf",
            "fig_local_vs_defiltered_by_model.pdf",
            "fig_local_vs_defiltered_scatter.pdf",
            "paired_variant_table_summary.csv",
            "paired_model_summary.csv",
            "paired_local_vs_defiltered_report.md",
        ],
        must_do_files=[
            "fig_local_vs_defiltered_by_template.pdf",
            "fig_local_vs_defiltered_by_model.pdf",
            "fig_local_vs_defiltered_scatter.pdf",
        ],
        support_files=[
            "paired_dataset_summary.csv",
            "paired_query_rows.csv",
            "paired_panel_scores.csv",
            "paired_template_summary.csv",
        ],
        notes=[
            "",
            f"Latest run: `{run_tag}`",
            f"Source analysis run: `{SOURCE_RUN_ID}`",
        ],
    )
    _write_text(final_dir / "README.md", readme_text)

    manifest = {
        "task": "paired_local_vs_defiltered_diagnostic",
        "run_tag": run_tag,
        "run_dir": str(run_dir.resolve()),
        "final_dir": str(final_dir.resolve()),
        "source_analysis_run_id": SOURCE_RUN_ID,
        "query_row_count": int(source_rows_df.shape[0]),
        "scored_query_row_count": int(paired_df.shape[0]),
        "panel_count": int(panel_df.shape[0]),
        "template_count": int(template_summary_df.shape[0]),
        "paper_table_path": str(PAPER_TABLE_PATH.resolve()),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(final_dir / "manifest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
