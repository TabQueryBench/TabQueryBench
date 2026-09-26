#!/usr/bin/env python3
"""Analyze validation-layer cardinality/range dynamics from real repository outputs.

This script intentionally operates only on repository artifacts and never invents
metric values. It:

1. inventories relevant files,
2. loads the best unified validation run,
3. normalizes model aliases and deduplicates dataset-model assets,
4. builds column-level analysis units for discrete and continuous profiles,
5. aggregates dynamic-bucket, model-level, and dataset-level summaries,
6. emits CSV, LaTeX figure/table snippets, and markdown analysis notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_scoring.eval.common import DEFAULT_SQL_SOURCE_VERSION, resolve_requested_sql_source_version, sql_source_label
from tqb_scoring.eval.query_fivepart_breakdown.common_final import render_final_readme, sync_final_outputs, versioned_name
from tqb_scoring.eval.query_fivepart_breakdown.common_heatmap_palette import (
    get_heatmap_cmap,
    heatmap_hex as _shared_heatmap_hex,
    text_hex_for_fill as _shared_text_hex_for_fill,
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

CARDINALITY_ROOT = PROJECT_ROOT.parent / "results" / "query_fivepart_breakdown" / "cardinality"
DATA_DIR = CARDINALITY_ROOT / "data"
FIG_DIR = CARDINALITY_ROOT / "figures"
TABLE_DIR = CARDINALITY_ROOT / "tables"
FINAL_DIR = CARDINALITY_ROOT / "final"
OUTPUT_VERSION_TAG = resolve_requested_sql_source_version("analysis", DEFAULT_SQL_SOURCE_VERSION)

MODEL_ORDER = [
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
README_FIGURE_MODELS = [
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
FIGURE_MODEL_ORDER = [model for model in MODEL_ORDER if model in README_FIGURE_MODELS]
HEATMAP_EXCLUDED_MODELS = set(MODEL_ORDER) - set(README_FIGURE_MODELS)
HEATMAP_EXCLUDED_DATASETS = {"c21", "n13"}
MODEL_ALIASES = {
    "rtf": "realtabformer",
}
SERVER_PRIORITY = {
    "rtx_5090": 2,
    "rtx_pro_6000": 1,
}
ROOT_PRIORITY = {
    "SynOutput-5090": 2,
    "SynOutput": 1,
}
DISCRETE_BUCKET_SPECS = [
    ("1-10", 1, 10),
    ("11-30", 11, 30),
    ("31-100", 31, 100),
    ("101-300", 101, 300),
    ("301-1000", 301, 1000),
    (">1000", 1001, math.inf),
]
DISCRETE_BUCKET_ORDER = [item[0] for item in DISCRETE_BUCKET_SPECS]
TIMESTAMP_RE = re.compile(r"(20\d{6}_\d{6})")


@dataclass
class ValidationRunInfo:
    run_id: str
    run_dir: Path
    manifest_path: Path
    details_jsonl: Path
    summary_csv: Path
    dataset_count: int
    asset_count: int


def _ensure_dirs() -> None:
    for path in [CARDINALITY_ROOT, DATA_DIR, FIG_DIR, TABLE_DIR, FINAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_validation_runs() -> list[ValidationRunInfo]:
    runs: list[ValidationRunInfo] = []
    validation_root = PROJECT_ROOT.parent / "results" / "validation" / "runs"
    for manifest_path in sorted(validation_root.glob("*/manifest.json")):
        run_dir = manifest_path.parent
        summary_dir = run_dir / "summaries"
        details_jsonl = summary_dir / "validation_details__all_datasets.jsonl"
        summary_csv = summary_dir / "validation_summary__all_datasets.csv"
        if not details_jsonl.exists() or not summary_csv.exists():
            continue
        manifest = _read_json(manifest_path)
        runs.append(
            ValidationRunInfo(
                run_id=run_dir.name,
                run_dir=run_dir,
                manifest_path=manifest_path,
                details_jsonl=details_jsonl,
                summary_csv=summary_csv,
                dataset_count=int(manifest.get("dataset_count") or 0),
                asset_count=int(manifest.get("asset_count") or 0),
            )
        )
    return runs


def _pick_primary_run(runs: list[ValidationRunInfo]) -> ValidationRunInfo:
    if not runs:
        raise FileNotFoundError("No unified validation runs found under Evaluation/validation/runs.")
    return sorted(runs, key=lambda item: (item.dataset_count, item.asset_count, item.run_id), reverse=True)[0]


def _resolve_primary_run(runs: list[ValidationRunInfo], source_run_dir: Path | None) -> ValidationRunInfo:
    if source_run_dir is None:
        return _pick_primary_run(runs)
    resolved = source_run_dir.expanduser().resolve()
    for run in runs:
        if run.run_dir.resolve() == resolved:
            return run
    raise FileNotFoundError(f"Requested validation run was not found or is incomplete: {resolved}")


def _pick_paper_dir() -> Path:
    candidates = sorted((PROJECT_ROOT / "Paper").glob("*/main.tex"))
    if not candidates:
        raise FileNotFoundError("Could not find Paper/*/main.tex.")
    return candidates[0].parent


def _normalize_model(model_id: str) -> str:
    key = str(model_id or "").strip().lower()
    return MODEL_ALIASES.get(key, key)


def _parse_timestamp_key(*texts: Any) -> str:
    matches: list[str] = []
    for text in texts:
        if text is None:
            continue
        matches.extend(TIMESTAMP_RE.findall(str(text)))
    return max(matches) if matches else ""


def _dataset_sort_key(dataset: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", str(dataset).strip())
    if not match:
        return (99, 10**9, str(dataset))
    prefix, number = match.groups()
    prefix_order = {"c": 0, "m": 1, "n": 2}.get(prefix.lower(), 50)
    return (prefix_order, int(number), str(dataset))


def _dataset_prefix(dataset: str) -> str:
    return str(dataset or "").strip().lower()[:1]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    num = _coerce_float(value)
    if num is None:
        return None
    return int(round(num))


def _choose_best_duplicate(items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        asset = item.get("asset", {})
        return (
            _parse_timestamp_key(
                asset.get("run_id"),
                asset.get("synthetic_csv_path"),
                asset.get("timestamp_utc"),
            ),
            asset.get("timestamp_utc") or "",
            SERVER_PRIORITY.get(str(asset.get("server_type") or ""), 0),
            ROOT_PRIORITY.get(str(asset.get("root_name") or ""), 0),
            str(asset.get("synthetic_csv_path") or ""),
        )

    ranked = sorted(items, key=sort_key, reverse=True)
    return ranked[0], ranked[1:]


def _load_and_deduplicate_validation_details(details_jsonl: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = _read_jsonl(details_jsonl)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        asset = row.get("asset", {})
        dataset = str(asset.get("dataset_id") or "").strip()
        model = _normalize_model(str(asset.get("model_id") or "").strip())
        if not dataset or not model:
            continue
        row["_normalized_model"] = model
        grouped[(dataset, model)].append(row)

    chosen_rows: list[dict[str, Any]] = []
    duplicate_audit: list[dict[str, Any]] = []
    for (dataset, model), items in sorted(grouped.items()):
        chosen, dropped = _choose_best_duplicate(items)
        chosen_rows.append(chosen)
        for extra in dropped:
            duplicate_audit.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "kept_run_id": chosen.get("asset", {}).get("run_id"),
                    "kept_server_type": chosen.get("asset", {}).get("server_type"),
                    "kept_root_name": chosen.get("asset", {}).get("root_name"),
                    "dropped_run_id": extra.get("asset", {}).get("run_id"),
                    "dropped_server_type": extra.get("asset", {}).get("server_type"),
                    "dropped_root_name": extra.get("asset", {}).get("root_name"),
                }
            )
    return chosen_rows, duplicate_audit


def _assign_discrete_bucket(real_distinct: int | None) -> tuple[str | None, int | None]:
    if real_distinct is None:
        return None, None
    for idx, (label, lower, upper) in enumerate(DISCRETE_BUCKET_SPECS, start=1):
        if lower <= real_distinct <= upper:
            return label, idx
    return None, None


def _make_quantile_bucket_map(
    values: pd.Series,
    *,
    max_buckets: int = 5,
) -> tuple[pd.Series, list[str], dict[str, int]]:
    clean = values.astype(float).replace([math.inf, -math.inf], pd.NA).dropna()
    if clean.empty:
        return pd.Series(index=values.index, dtype="object"), [], {}
    transformed = clean.map(lambda x: math.log10(max(x, 1e-12)))
    unique_count = transformed.nunique()
    bucket_count = int(min(max_buckets, max(1, unique_count)))
    labels = [f"Q{i}" for i in range(1, bucket_count + 1)]

    # Rank-based splitting is more robust than qcut when many real widths tie at
    # the same value and qcut collapses adjacent bin edges.
    ordered_index = list(transformed.sort_values(kind="mergesort").index)
    assignments: dict[Any, str] = {}
    n = len(ordered_index)
    for pos, idx in enumerate(ordered_index):
        bucket_idx = min(bucket_count - 1, int((pos * bucket_count) / max(1, n)))
        assignments[idx] = labels[bucket_idx]
    bucketed = pd.Series(assignments)
    full = pd.Series(index=values.index, dtype="object")
    full.loc[clean.index] = bucketed
    order = {label: idx for idx, label in enumerate(labels, start=1)}
    return full, labels, order


def _dynamic_group(
    channel: str,
    bucket_label: str | None,
    bucket_rank: int | None,
    *,
    continuous_labels: list[str],
) -> str | None:
    if bucket_label is None or bucket_rank is None:
        return None
    if channel == "discrete":
        if bucket_rank <= 2:
            return "low"
        if bucket_rank >= max(5, len(DISCRETE_BUCKET_ORDER) - 1):
            return "high"
        return "mid"
    if not continuous_labels:
        return None
    max_rank = len(continuous_labels)
    low_cut = min(2, max_rank)
    high_cut = max(1, max_rank - 1)
    if bucket_rank <= low_cut:
        return "low"
    if bucket_rank >= high_cut:
        return "high"
    return "mid"


def _infer_failure_reason(row: dict[str, Any]) -> str:
    channel = row.get("channel")
    if channel == "discrete":
        real_distinct = _coerce_int(row.get("real_distinct_count")) or 0
        syn_distinct = _coerce_int(row.get("synthetic_distinct_count")) or 0
        missing_value_count = _coerce_int(row.get("missing_value_count")) or 0
        if real_distinct and missing_value_count >= real_distinct:
            return "full real-support collapse"
        if real_distinct and missing_value_count / max(1, real_distinct) >= 0.5:
            return "major missing-support loss"
        if syn_distinct > real_distinct and missing_value_count > 0:
            return "invented values while missing real support"
        if syn_distinct > real_distinct:
            return "support inflation beyond real distinct count"
        return "partial support retention failure"

    real_min = _coerce_float(row.get("real_min"))
    real_max = _coerce_float(row.get("real_max"))
    syn_min = _coerce_float(row.get("synthetic_min"))
    syn_max = _coerce_float(row.get("synthetic_max"))
    if None in {real_min, real_max, syn_min, syn_max}:
        return "invalid synthetic range envelope"
    narrower = syn_min > real_min and syn_max < real_max
    wider = syn_min < real_min and syn_max > real_max
    if narrower:
        return "range collapse on both sides"
    if wider:
        return "range over-expansion on both sides"
    if syn_min > real_min:
        return "left-tail under-coverage"
    if syn_max < real_max:
        return "right-tail under-coverage"
    if syn_min < real_min:
        return "left-tail over-expansion"
    if syn_max > real_max:
        return "right-tail over-expansion"
    return "range-envelope mismatch"


def _build_cleaned_results(
    rows: list[dict[str, Any]],
    *,
    source_file: Path,
) -> pd.DataFrame:
    unit_rows: list[dict[str, Any]] = []
    for row in rows:
        asset = row.get("asset", {})
        report = row.get("report", {})
        dataset = str(asset.get("dataset_id") or "").strip()
        model = str(row.get("_normalized_model") or "").strip()
        card_range = (((report.get("validation_channels") or {}).get("cardinality_range")) or {})
        details = card_range.get("details") or {}
        discrete = details.get("discrete_profile") or {}
        continuous = details.get("continuous_profile") or {}
        discrete_channel_score = _coerce_float(card_range.get("discrete_profile_score"))
        continuous_channel_score = _coerce_float(card_range.get("continuous_profile_score"))
        channel_score = _coerce_float(card_range.get("score"))
        source_asset_key = str(asset.get("asset_key") or "")
        source_path = str(source_file)

        for item in discrete.get("per_column") or []:
            real_distinct = _coerce_int(item.get("real_distinct"))
            syn_distinct = _coerce_int(item.get("syn_distinct"))
            missing_value_count = _coerce_int(item.get("missing_value_count")) or 0
            if real_distinct is None or real_distinct <= 0:
                continue
            support_retention = max(0.0, min(1.0, 1.0 - (missing_value_count / max(1, real_distinct))))
            unit_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "channel": "discrete",
                    "column": str(item.get("column") or ""),
                    "score": round(support_retention, 6),
                    "score_kind": "derived_support_retention_ratio",
                    "official_channel_score": discrete_channel_score,
                    "official_cardinality_range_score": channel_score,
                    "real_dynamic_value": float(real_distinct),
                    "synthetic_dynamic_value": float(syn_distinct) if syn_distinct is not None else None,
                    "real_distinct_count": real_distinct,
                    "synthetic_distinct_count": syn_distinct,
                    "missing_value_count": missing_value_count,
                    "risk_flag": bool(item.get("risk_flag")),
                    "real_min": None,
                    "real_max": None,
                    "synthetic_min": None,
                    "synthetic_max": None,
                    "source_file": source_path,
                    "source_asset_key": source_asset_key,
                }
            )

        for item in continuous.get("per_column") or []:
            score = _coerce_float(item.get("score"))
            real_min = _coerce_float(item.get("real_min"))
            real_max = _coerce_float(item.get("real_max"))
            syn_min = _coerce_float(item.get("syn_min"))
            syn_max = _coerce_float(item.get("syn_max"))
            if score is None:
                continue
            real_width = (real_max - real_min) if None not in {real_min, real_max} else None
            syn_width = (syn_max - syn_min) if None not in {syn_min, syn_max} else None
            unit_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "channel": "continuous",
                    "column": str(item.get("column") or ""),
                    "score": round(score, 6),
                    "score_kind": "official_per_column_range_score",
                    "official_channel_score": continuous_channel_score,
                    "official_cardinality_range_score": channel_score,
                    "real_dynamic_value": real_width,
                    "synthetic_dynamic_value": syn_width,
                    "real_distinct_count": None,
                    "synthetic_distinct_count": None,
                    "missing_value_count": None,
                    "risk_flag": None,
                    "real_min": real_min,
                    "real_max": real_max,
                    "synthetic_min": syn_min,
                    "synthetic_max": syn_max,
                    "source_file": source_path,
                    "source_asset_key": source_asset_key,
                }
            )

    df = pd.DataFrame(unit_rows)
    if df.empty:
        return df

    discrete_mask = df["channel"] == "discrete"
    continuous_mask = df["channel"] == "continuous"

    discrete_buckets = df.loc[discrete_mask, "real_dynamic_value"].map(lambda v: _assign_discrete_bucket(_coerce_int(v))[0])
    discrete_bucket_ranks = df.loc[discrete_mask, "real_dynamic_value"].map(lambda v: _assign_discrete_bucket(_coerce_int(v))[1])
    df.loc[discrete_mask, "dynamic_bucket"] = discrete_buckets
    df.loc[discrete_mask, "bucket_rank"] = discrete_bucket_ranks

    continuous_bucket_series, continuous_labels, continuous_bucket_order = _make_quantile_bucket_map(
        df.loc[continuous_mask, "real_dynamic_value"].astype(float)
    )
    df.loc[continuous_mask, "dynamic_bucket"] = continuous_bucket_series
    df.loc[continuous_mask, "bucket_rank"] = df.loc[continuous_mask, "dynamic_bucket"].map(continuous_bucket_order)

    df["dynamic_regime"] = [
        _dynamic_group(
            channel=str(channel),
            bucket_label=str(bucket) if pd.notna(bucket) else None,
            bucket_rank=int(rank) if pd.notna(rank) else None,
            continuous_labels=continuous_labels,
        )
        for channel, bucket, rank in zip(df["channel"], df["dynamic_bucket"], df["bucket_rank"])
    ]

    discrete_rank_max = max(len(DISCRETE_BUCKET_ORDER), 1)
    continuous_rank_max = max(len(continuous_labels), 1)
    dynamic_intensity: list[float | None] = []
    for channel, rank in zip(df["channel"], df["bucket_rank"]):
        if pd.isna(rank):
            dynamic_intensity.append(None)
            continue
        denom = discrete_rank_max if channel == "discrete" else continuous_rank_max
        dynamic_intensity.append(float(rank) / float(max(1, denom)))
    df["dynamic_intensity"] = dynamic_intensity

    ordered_cols = [
        "dataset",
        "model",
        "channel",
        "column",
        "score",
        "score_kind",
        "official_channel_score",
        "official_cardinality_range_score",
        "real_dynamic_value",
        "synthetic_dynamic_value",
        "real_distinct_count",
        "synthetic_distinct_count",
        "missing_value_count",
        "risk_flag",
        "real_min",
        "real_max",
        "synthetic_min",
        "synthetic_max",
        "dynamic_bucket",
        "bucket_rank",
        "dynamic_regime",
        "dynamic_intensity",
        "source_file",
        "source_asset_key",
    ]
    sort_df = df[ordered_cols].copy()
    sort_df["dataset_sort_key"] = sort_df["dataset"].map(_dataset_sort_key)
    return (
        sort_df.sort_values(["channel", "dataset_sort_key", "model", "column"])
        .drop(columns=["dataset_sort_key"])
        .reset_index(drop=True)
    )


def _summary_stats(values: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(clean.shape[0])
    if n == 0:
        return {
            "n_units": 0,
            "mean_score": None,
            "std_score": None,
            "se_score": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1)) if n > 1 else 0.0
    se_val = float(std_val / math.sqrt(n)) if n > 1 else 0.0
    ci = 1.96 * se_val
    return {
        "n_units": n,
        "mean_score": round(mean_val, 6),
        "std_score": round(std_val, 6),
        "se_score": round(se_val, 6),
        "ci95_low": round(mean_val - ci, 6),
        "ci95_high": round(mean_val + ci, 6),
    }


def _build_summary_by_dynamic_bucket(clean_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = clean_df.groupby(["model", "channel", "dynamic_bucket"], dropna=False)
    for (model, channel, bucket), group in grouped:
        if pd.isna(bucket):
            continue
        stats = _summary_stats(group["score"])
        rows.append(
            {
                "model": model,
                "channel": channel,
                "dynamic_bucket": bucket,
                **stats,
                "bucket_rank": int(group["bucket_rank"].iloc[0]) if pd.notna(group["bucket_rank"].iloc[0]) else None,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["channel", "bucket_rank", "model"]).reset_index(drop=True)


def _build_model_drop_table(clean_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, channel), group in clean_df.groupby(["model", "channel"]):
        low = pd.to_numeric(group.loc[group["dynamic_regime"] == "low", "score"], errors="coerce").dropna()
        high = pd.to_numeric(group.loc[group["dynamic_regime"] == "high", "score"], errors="coerce").dropna()
        low_mean = float(low.mean()) if not low.empty else None
        high_mean = float(high.mean()) if not high.empty else None
        absolute_drop = (low_mean - high_mean) if (low_mean is not None and high_mean is not None) else None
        relative_drop = (
            (absolute_drop / low_mean) if (absolute_drop is not None and low_mean not in {None, 0.0}) else None
        )
        rows.append(
            {
                "model": model,
                "channel": channel,
                "low_dynamic_mean": round(low_mean, 6) if low_mean is not None else None,
                "high_dynamic_mean": round(high_mean, 6) if high_mean is not None else None,
                "absolute_drop": round(absolute_drop, 6) if absolute_drop is not None else None,
                "relative_drop": round(relative_drop, 6) if relative_drop is not None else None,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rank_by_robustness"] = (
        df.sort_values(["channel", "absolute_drop", "high_dynamic_mean"], ascending=[True, True, False])
        .groupby("channel")
        .cumcount()
        .add(1)
    )
    return df.sort_values(["channel", "rank_by_robustness", "model"]).reset_index(drop=True)


def _build_summary_by_model(clean_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in clean_df.groupby("model"):
        discrete_scores = pd.to_numeric(group.loc[group["channel"] == "discrete", "score"], errors="coerce").dropna()
        continuous_scores = pd.to_numeric(group.loc[group["channel"] == "continuous", "score"], errors="coerce").dropna()
        all_scores = pd.Series(
            [val for val in [float(discrete_scores.mean()) if not discrete_scores.empty else None, float(continuous_scores.mean()) if not continuous_scores.empty else None] if val is not None]
        )

        low_discrete = pd.to_numeric(
            group.loc[(group["channel"] == "discrete") & (group["dynamic_regime"] == "low"), "score"],
            errors="coerce",
        ).dropna()
        low_continuous = pd.to_numeric(
            group.loc[(group["channel"] == "continuous") & (group["dynamic_regime"] == "low"), "score"],
            errors="coerce",
        ).dropna()
        high_discrete = pd.to_numeric(
            group.loc[(group["channel"] == "discrete") & (group["dynamic_regime"] == "high"), "score"],
            errors="coerce",
        ).dropna()
        high_continuous = pd.to_numeric(
            group.loc[(group["channel"] == "continuous") & (group["dynamic_regime"] == "high"), "score"],
            errors="coerce",
        ).dropna()

        low_parts = [float(series.mean()) for series in [low_discrete, low_continuous] if not series.empty]
        high_parts = [float(series.mean()) for series in [high_discrete, high_continuous] if not series.empty]
        low_mean = float(mean(low_parts)) if low_parts else None
        high_mean = float(mean(high_parts)) if high_parts else None
        gap = (high_mean - low_mean) if (low_mean is not None and high_mean is not None) else None
        relative_drop = (
            (low_mean - high_mean) / low_mean if (low_mean is not None and high_mean is not None and low_mean != 0) else None
        )
        rows.append(
            {
                "model": model,
                "n_datasets": int(group["dataset"].nunique()),
                "n_units": int(group.shape[0]),
                "discrete_score_mean": round(float(discrete_scores.mean()), 6) if not discrete_scores.empty else None,
                "continuous_score_mean": round(float(continuous_scores.mean()), 6) if not continuous_scores.empty else None,
                "overall_score_mean": round(float(all_scores.mean()), 6) if not all_scores.empty else None,
                "low_dynamic_score_mean": round(low_mean, 6) if low_mean is not None else None,
                "high_dynamic_score_mean": round(high_mean, 6) if high_mean is not None else None,
                "high_minus_low_gap": round(gap, 6) if gap is not None else None,
                "relative_drop": round(relative_drop, 6) if relative_drop is not None else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["model_order"] = df["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER, start=1)})
    df["rank_overall"] = df["overall_score_mean"].rank(ascending=False, method="min")
    df["rank_high_dynamic"] = df["high_dynamic_score_mean"].rank(ascending=False, method="min")
    df["rank_drop_robustness"] = df["relative_drop"].rank(ascending=True, method="min")
    rank_cols = ["rank_overall", "rank_high_dynamic", "rank_drop_robustness"]
    for col in rank_cols:
        df[col] = df[col].astype("Int64")
    return df.sort_values(["model_order", "model"]).drop(columns=["model_order"]).reset_index(drop=True)


def _build_summary_by_dataset(clean_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in clean_df.groupby("dataset"):
        discrete_scores = pd.to_numeric(group.loc[group["channel"] == "discrete", "score"], errors="coerce").dropna()
        continuous_scores = pd.to_numeric(group.loc[group["channel"] == "continuous", "score"], errors="coerce").dropna()
        all_scores = pd.to_numeric(group["score"], errors="coerce").dropna()
        discrete_mean = float(discrete_scores.mean()) if not discrete_scores.empty else None
        continuous_mean = float(continuous_scores.mean()) if not continuous_scores.empty else None
        if discrete_mean is None:
            hardest_channel = "continuous"
        elif continuous_mean is None:
            hardest_channel = "discrete"
        else:
            hardest_channel = "discrete" if discrete_mean < continuous_mean else "continuous"
        rows.append(
            {
                "dataset": dataset,
                "n_models": int(group["model"].nunique()),
                "n_units": int(group.shape[0]),
                "dataset_dynamic_intensity": round(float(pd.to_numeric(group["dynamic_intensity"], errors="coerce").dropna().mean()), 6),
                "mean_score": round(float(all_scores.mean()), 6) if not all_scores.empty else None,
                "discrete_mean_score": round(discrete_mean, 6) if discrete_mean is not None else None,
                "continuous_mean_score": round(continuous_mean, 6) if continuous_mean is not None else None,
                "high_dynamic_share": round(float((group["dynamic_regime"] == "high").mean()), 6),
                "hardest_channel": hardest_channel,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rank_by_difficulty"] = df["mean_score"].rank(ascending=True, method="min").astype("Int64")
    df["dataset_sort_key"] = df["dataset"].map(_dataset_sort_key)
    return df.sort_values(["dataset_sort_key", "dataset"]).drop(columns=["dataset_sort_key"]).reset_index(drop=True)


def _build_top_failure_cases(clean_df: pd.DataFrame, n_rows: int = 40) -> pd.DataFrame:
    high_df = clean_df.loc[clean_df["dynamic_regime"] == "high"].copy()
    if high_df.empty:
        return pd.DataFrame(columns=["dataset", "model", "channel", "column", "dynamic_bucket", "real_dynamic_value", "score", "short_reason", "source_file"])
    high_df["short_reason"] = [ _infer_failure_reason(row) for row in high_df.to_dict("records") ]
    high_df = high_df.sort_values(["score", "dynamic_intensity", "dataset", "model"], ascending=[True, False, True, True])
    cols = [
        "dataset",
        "model",
        "channel",
        "column",
        "dynamic_bucket",
        "real_dynamic_value",
        "score",
        "short_reason",
        "source_file",
    ]
    return high_df[cols].head(n_rows).reset_index(drop=True)


def _build_heatmap_data(clean_df: pd.DataFrame, dataset_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        clean_df.groupby(["dataset", "model", "channel"], as_index=False)
        .agg(score=("score", "mean"))
    )
    merged = grouped.merge(
        dataset_summary[["dataset", "dataset_dynamic_intensity", "rank_by_difficulty"]],
        on="dataset",
        how="left",
    )
    merged = merged.rename(columns={"rank_by_difficulty": "dataset_dynamic_rank"})
    merged["score"] = merged["score"].round(6)
    merged["dataset_sort_key"] = merged["dataset"].map(_dataset_sort_key)
    return (
        merged.sort_values(["channel", "dataset_sort_key", "model"])
        .drop(columns=["dataset_sort_key"])
        .reset_index(drop=True)
    )


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _maybe_float_str(value: Any, digits: int = 3) -> str:
    val = _coerce_float(value)
    if val is None:
        return "--"
    return f"{val:.{digits}f}"


def _latex_escape(text: Any) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


MODEL_COLOR_HEX = {
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

MODEL_MARKER_MAP = {
    "arf": "*",
    "bayesnet": "square*",
    "codi": "triangle*",
    "ctgan": "diamond*",
    "forestdiffusion": "otimes*",
    "realtabformer": "pentagon*",
    "tabbyflow": "asterisk",
    "tabddpm": "Mercedes star",
    "tabdiff": "x",
    "tabpfgen": "oplus*",
    "tabsyn": "triangle",
    "tvae": "diamond",
}
REGIME_PREFIX_ORDER = ["c", "m", "n"]
REGIME_DISPLAY_LABEL = {"c": "Categorical", "m": "Mixed", "n": "Numerical"}
TERCILE_LABEL_ORDER = ["low", "mid", "high"]
TERCILE_DISPLAY_LABEL = {"low": "Low", "mid": "Mid", "high": "High"}


def _hex_to_pgf_color(hex_color: str) -> str:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return "gray!50"
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return f"{{rgb,255:red,{r}; green,{g}; blue,{b}}}"


def _model_color_name(model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "", str(model))
    return f"modelcolor{safe}"


def _render_model_color_definitions(models: list[str] | None = None) -> str:
    selected = models or FIGURE_MODEL_ORDER
    lines: list[str] = []
    for model in selected:
        hex_color = MODEL_COLOR_HEX.get(model)
        if not hex_color:
            continue
        lines.append(rf"\definecolor{{{_model_color_name(model)}}}{{HTML}}{{{hex_color.lstrip('#')}}}")
    return "\n".join(lines)


def _model_style(model: str, *, width: str = "semithick") -> tuple[str, str, str]:
    color = _model_color_name(model) if model in MODEL_COLOR_HEX else "gray!50"
    marker = MODEL_MARKER_MAP.get(model, "*")
    return color, marker, width


def _render_shared_model_legend() -> str:
    row1 = FIGURE_MODEL_ORDER[:6]
    row2 = FIGURE_MODEL_ORDER[6:12]

    def cell(model: str) -> str:
        if not model:
            return ""
        color, _, _ = _model_style(model)
        return rf"\textcolor{{{color}}}{{\rule[0.5ex]{{1.2em}}{{1.2pt}}}}\ \textcolor{{{color}}}{{{model}}}"

    row2_padded = row2 + [""] * (6 - len(row2))

    return "\n".join(
        [
            r"\begin{adjustbox}{max width=\textwidth}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{>{\raggedright\arraybackslash}m{0.158\textwidth}>{\raggedright\arraybackslash}m{0.158\textwidth}>{\raggedright\arraybackslash}m{0.158\textwidth}>{\raggedright\arraybackslash}m{0.158\textwidth}>{\raggedright\arraybackslash}m{0.158\textwidth}>{\raggedright\arraybackslash}m{0.158\textwidth}}",
            " & ".join(cell(model) for model in row1) + r" \\",
            " & ".join(cell(model) for model in row2_padded) + r" \\",
            r"\end{tabular}",
            r"\end{adjustbox}",
        ]
    )


def _build_plot_csvs(
    summary_by_bucket: pd.DataFrame,
    model_summary: pd.DataFrame,
    model_drop: pd.DataFrame,
    heatmap_data: pd.DataFrame,
    dataset_summary: pd.DataFrame,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    combined_curve_frames: list[pd.DataFrame] = []

    for channel in ["discrete", "continuous"]:
        subset = summary_by_bucket.loc[summary_by_bucket["channel"] == channel].copy()
        if subset.empty:
            continue
        bucket_order = (
            DISCRETE_BUCKET_ORDER
            if channel == "discrete"
            else sorted(subset["dynamic_bucket"].dropna().unique(), key=lambda b: int(str(b).replace("Q", "")))
        )
        pivot = (
            subset.pivot(index="dynamic_bucket", columns="model", values="mean_score")
            .reindex(bucket_order)
            .reset_index()
        )
        pivot.insert(0, "bucket_idx", range(1, len(pivot) + 1))
        curve_path = DATA_DIR / f"figure1_{channel}_curves.csv"
        pivot.to_csv(curve_path, index=False)
        meta[f"{channel}_curve_path"] = curve_path
        meta[f"{channel}_bucket_order"] = bucket_order

        combined = pivot.copy()
        combined.insert(0, "channel", channel)
        if channel == "discrete":
            combined.insert(1, "plot_x", range(1, len(combined) + 1))
        else:
            combined.insert(1, "plot_x", range(8, 8 + len(combined)))
        combined_curve_frames.append(combined)

    if combined_curve_frames:
        combined_curves = pd.concat(combined_curve_frames, ignore_index=True)
        combined_curve_path = DATA_DIR / "figure1_combined_curves.csv"
        combined_curves.to_csv(combined_curve_path, index=False)
        meta["combined_curve_path"] = combined_curve_path
        meta["combined_xticks"] = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
        meta["combined_xticklabels"] = [
            "1-10",
            "11-30",
            "31-100",
            "101-300",
            "301-1000",
            ">1000",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5",
        ]

    for channel in ["discrete", "continuous"]:
        subset = model_drop.loc[model_drop["channel"] == channel].copy()
        if subset.empty:
            continue
        subset = subset.set_index("model").reindex([m for m in MODEL_ORDER if m in set(subset["model"])]).reset_index()
        slope_rows = pd.DataFrame({"x": [1, 2], "regime": ["Low dynamic", "High dynamic"]})
        for _, row in subset.iterrows():
            slope_rows[row["model"]] = [row["low_dynamic_mean"], row["high_dynamic_mean"]]
        slope_path = DATA_DIR / f"figure2_{channel}_slopes.csv"
        slope_rows.to_csv(slope_path, index=False)
        meta[f"{channel}_slope_path"] = slope_path

    scatter = model_summary.loc[
        model_summary["model"].isin(FIGURE_MODEL_ORDER),
        ["model", "overall_score_mean", "high_dynamic_score_mean", "relative_drop"],
    ].copy()
    scatter = scatter.sort_values("model", key=lambda s: s.map({m: i for i, m in enumerate(FIGURE_MODEL_ORDER)}))
    scatter_path = DATA_DIR / "figure4_scatter.csv"
    scatter.to_csv(scatter_path, index=False)
    meta["scatter_path"] = scatter_path

    compact_targets: dict[str, pd.DataFrame] = {}
    for channel in ["discrete", "continuous"]:
        subset = heatmap_data.loc[heatmap_data["channel"] == channel].copy()
        if subset.empty:
            continue
        subset = subset.loc[
            (~subset["model"].isin(HEATMAP_EXCLUDED_MODELS)) & (~subset["dataset"].isin(HEATMAP_EXCLUDED_DATASETS))
        ].copy()
        dataset_ranks = dataset_summary.loc[
            ~dataset_summary["dataset"].isin(HEATMAP_EXCLUDED_DATASETS),
            ["dataset", "dataset_dynamic_intensity"],
        ].sort_values("dataset_dynamic_intensity", ascending=False)
        hardest = list(dataset_ranks.head(15)["dataset"])
        easiest = list(dataset_ranks.tail(15)["dataset"])
        keep_order = hardest + [d for d in easiest if d not in hardest]
        keep_order = sorted(keep_order, key=_dataset_sort_key)
        compact = subset.loc[subset["dataset"].isin(keep_order)].copy()
        compact["dataset"] = pd.Categorical(compact["dataset"], categories=keep_order, ordered=True)
        compact["model"] = pd.Categorical(
            compact["model"],
            categories=[m for m in MODEL_ORDER if m not in HEATMAP_EXCLUDED_MODELS],
            ordered=True,
        )
        compact = compact.sort_values(["dataset", "model"])
        wide = compact.pivot(index="dataset", columns="model", values="score").reset_index()
        wide.insert(0, "display_group", ["hardest" if idx < len(hardest) else "easiest" for idx in range(len(wide))])
        path = DATA_DIR / f"figure3_{channel}_heatmap_compact.csv"
        wide.to_csv(path, index=False)
        compact_targets[channel] = wide
        meta[f"{channel}_heatmap_compact_path"] = path
    return meta


def _build_standard_tradeoff_data(model_summary: pd.DataFrame) -> pd.DataFrame:
    tradeoff = model_summary.loc[
        model_summary["model"].isin(FIGURE_MODEL_ORDER),
        ["model", "discrete_score_mean", "high_dynamic_score_mean", "overall_score_mean", "relative_drop"],
    ].copy()
    tradeoff = tradeoff.dropna(subset=["discrete_score_mean", "high_dynamic_score_mean"]).reset_index(drop=True)
    tradeoff["model_order"] = tradeoff["model"].map({name: idx for idx, name in enumerate(FIGURE_MODEL_ORDER, start=1)})
    tradeoff = tradeoff.sort_values(["model_order", "model"]).drop(columns=["model_order"]).reset_index(drop=True)
    return tradeoff


def _build_standard_prefix_plot_data(clean_df: pd.DataFrame) -> pd.DataFrame:
    working = clean_df.loc[clean_df["model"].isin(FIGURE_MODEL_ORDER)].copy()
    working["dataset_prefix"] = working["dataset"].map(_dataset_prefix)
    grouped = (
        working.groupby(["model", "dataset_prefix"], as_index=False)
        .agg(score=("score", "mean"))
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for model in [name for name in FIGURE_MODEL_ORDER if name in set(grouped["model"])]:
        payload: dict[str, Any] = {"model": model}
        subset = grouped.loc[grouped["model"] == model]
        for prefix in ["c", "m", "n"]:
            series = pd.to_numeric(subset.loc[subset["dataset_prefix"] == prefix, "score"], errors="coerce").dropna()
            payload[prefix] = round(float(series.iloc[0]), 6) if not series.empty else None
        rows.append(payload)
    return pd.DataFrame(rows)


def _split_upper_heavy_terciles(items: list[str]) -> list[list[str]]:
    count = len(items)
    base = count // 3
    remainder = count % 3
    sizes = [base, base, base]
    for idx in range(remainder):
        sizes[2 - idx] += 1
    groups: list[list[str]] = []
    start = 0
    for size in sizes:
        groups.append(items[start : start + size])
        start += size
    return groups


def _build_regime_tercile_dataset_groups(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prefix in REGIME_PREFIX_ORDER:
        subset = dataset_summary.loc[dataset_summary["dataset"].map(_dataset_prefix) == prefix].copy()
        if subset.empty:
            continue
        subset["dataset_sort_key"] = subset["dataset"].map(_dataset_sort_key)
        subset = subset.sort_values(
            ["dataset_dynamic_intensity", "dataset_sort_key", "dataset"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        grouped_datasets = _split_upper_heavy_terciles(subset["dataset"].tolist())
        for tercile_index, (tercile_key, dataset_ids) in enumerate(
            zip(TERCILE_LABEL_ORDER, grouped_datasets), start=1
        ):
            if not dataset_ids:
                continue
            group_key = f"{prefix.upper()}-{tercile_key}"
            for dataset_order_in_group, dataset in enumerate(dataset_ids, start=1):
                row = subset.loc[subset["dataset"] == dataset].iloc[0]
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_prefix": prefix,
                        "dataset_regime": REGIME_DISPLAY_LABEL[prefix],
                        "tercile_key": tercile_key,
                        "tercile_index": tercile_index,
                        "tercile_display": TERCILE_DISPLAY_LABEL[tercile_key],
                        "group_key": group_key,
                        "group_order": (REGIME_PREFIX_ORDER.index(prefix) * 3) + tercile_index,
                        "dataset_order_in_group": dataset_order_in_group,
                        "group_dataset_count": len(dataset_ids),
                        "dataset_dynamic_intensity": row["dataset_dynamic_intensity"],
                        "dataset_mean_score": row["mean_score"],
                    }
                )
    group_df = pd.DataFrame(rows)
    if group_df.empty:
        return group_df
    return group_df.sort_values(["group_order", "dataset_order_in_group", "dataset"]).reset_index(drop=True)


def _build_regime_tercile_model_matrix(clean_df: pd.DataFrame, group_df: pd.DataFrame) -> pd.DataFrame:
    dataset_model = (
        clean_df.groupby(["dataset", "model"], as_index=False)
        .agg(score=("score", "mean"))
        .reset_index(drop=True)
    )
    merged = dataset_model.merge(
        group_df[
            [
                "dataset",
                "dataset_prefix",
                "dataset_regime",
                "tercile_key",
                "tercile_display",
                "group_key",
                "group_order",
                "group_dataset_count",
            ]
        ],
        on="dataset",
        how="left",
    )
    merged = merged.dropna(subset=["group_key"]).copy()
    grouped = (
        merged.groupby(
            [
                "model",
                "dataset_prefix",
                "dataset_regime",
                "tercile_key",
                "tercile_display",
                "group_key",
                "group_order",
                "group_dataset_count",
            ],
            as_index=False,
        )
        .agg(mean_score=("score", "mean"), dataset_count=("dataset", "nunique"))
        .reset_index(drop=True)
    )
    if grouped.empty:
        return grouped
    column_order = [f"{prefix.upper()}-{tercile}" for prefix in REGIME_PREFIX_ORDER for tercile in TERCILE_LABEL_ORDER]
    rename_map = {
        f"{prefix.upper()}-{tercile}": f"{prefix.upper()}-{tercile}"
        for prefix in REGIME_PREFIX_ORDER
        for tercile in TERCILE_LABEL_ORDER
    }
    wide = grouped.pivot(index="model", columns="group_key", values="mean_score").reindex(
        [model for model in MODEL_ORDER if model in set(grouped["model"])]
    )
    wide = wide.reindex(columns=column_order).rename(columns=rename_map).reset_index()
    return wide.round(6)


def _build_cardinality_model_subitem_heatmap_df(model_summary: pd.DataFrame) -> pd.DataFrame:
    return build_model_subitem_heatmap_df(
        model_summary.loc[model_summary["model"].isin(FIGURE_MODEL_ORDER)].copy(),
        model_id_col="model",
        model_order=FIGURE_MODEL_ORDER,
        subitem_specs=[
            ("support_rank_profile_consistency", "Support-rank profile consistency", "discrete_score_mean"),
            ("high_cardinality_response_stability", "High-cardinality response stability", "high_dynamic_score_mean"),
        ],
        summary_row_spec=("family_mean", "Family mean", "overall_score_mean"),
    )


def _write_regime_tercile_table_tex(matrix_df: pd.DataFrame, group_df: pd.DataFrame, path: Path) -> None:
    group_counts = (
        group_df.groupby(["dataset_prefix", "tercile_key"], as_index=False)["dataset"]
        .nunique()
        .rename(columns={"dataset": "dataset_count"})
    )
    count_lookup = {
        (str(row.dataset_prefix), str(row.tercile_key)): int(row.dataset_count)
        for row in group_counts.itertuples()
    }
    ordered_cols = [
        "C-low",
        "C-mid",
        "C-high",
        "M-low",
        "M-mid",
        "M-high",
        "N-low",
        "N-mid",
        "N-high",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Categorical} & \multicolumn{3}{c}{Mixed} & \multicolumn{3}{c}{Numerical} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        (
            "Model & "
            f"Low ({count_lookup.get(('c', 'low'), 0)}) & Mid ({count_lookup.get(('c', 'mid'), 0)}) & High ({count_lookup.get(('c', 'high'), 0)}) & "
            f"Low ({count_lookup.get(('m', 'low'), 0)}) & Mid ({count_lookup.get(('m', 'mid'), 0)}) & High ({count_lookup.get(('m', 'high'), 0)}) & "
            f"Low ({count_lookup.get(('n', 'low'), 0)}) & Mid ({count_lookup.get(('n', 'mid'), 0)}) & High ({count_lookup.get(('n', 'high'), 0)}) \\\\"
        ),
        r"\midrule",
    ]
    for _, row in matrix_df.iterrows():
        values = " & ".join(_maybe_float_str(row.get(col), 3) for col in ordered_cols)
        lines.append(rf"{_latex_escape(row['model'])} & {values} \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Regime-split tercile matrix for cardinality/range. Within each regime, datasets are sorted by increasing `dataset_dynamic_intensity` and then partitioned into three contiguous groups, with any remainder assigned to the harder groups. Cell values first average within each dataset and then average equally across datasets in the tercile.}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_standard_tradeoff_tex(tradeoff_df: pd.DataFrame, path: Path) -> None:
    models = [str(item) for item in tradeoff_df["model"].tolist()]
    lines = [
        _render_model_color_definitions(models),
        r"\begin{figure}[p]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"""\begin{axis}[
width=0.74\textwidth,
height=0.56\textwidth,
xmin=0, xmax=1.02,
ymin=0, ymax=1.02,
xlabel={Support-rank profile proxy},
ylabel={High-cardinality response stability proxy},
grid=major,
grid style={gray!20},
]""",
    ]
    for row in tradeoff_df.itertuples():
        color, marker, width = _model_style(str(row.model), width="thick")
        lines.append(
            rf"\addplot[{width}, only marks, {color}, mark={marker}, mark size=2.3pt] coordinates {{({float(row.discrete_score_mean):.6f},{float(row.high_dynamic_score_mean):.6f})}};"
        )
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\vspace{0.4em}",
            _render_shared_model_legend(),
            r"\caption{Standardized cardinality trade-off view. The x-axis uses the discrete support-retention branch as the support-rank proxy, while the y-axis uses the high-dynamic regime mean as the high-cardinality stability proxy.}",
            r"\end{figure}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_standard_prefix_tex(prefix_plot_df: pd.DataFrame, path: Path) -> None:
    model_labels = [str(item) for item in prefix_plot_df["model"].tolist()]
    xticklabels = ",".join(model_labels)
    lines = [
        _render_model_color_definitions(model_labels),
        r"\begin{figure}[p]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"""\begin{groupplot}[
group style={group size=3 by 1, horizontal sep=1.1cm},
width=0.29\textwidth,
height=0.48\textwidth,
ymin=0, ymax=1.02,
xtick={1,...,%d},
xticklabels={%s},
x tick label style={rotate=60, anchor=east, font=\scriptsize},
grid=major,
grid style={gray!20},
]
"""
        % (len(model_labels), xticklabels),
    ]
    for prefix in ["c", "m", "n"]:
        lines.append(rf"\nextgroupplot[title={{{prefix.upper()} datasets}}, ylabel={{Mean cardinality score}}]")
        for idx, row in enumerate(prefix_plot_df.itertuples(), start=1):
            value = getattr(row, prefix, None)
            if value is None or pd.isna(value):
                continue
            color, _marker, _width = _model_style(str(row.model))
            lines.append(rf"\addplot[ybar, draw={color}, fill={color}] coordinates {{({idx},{float(value):.6f})}};")
    lines.extend(
        [
            r"\end{groupplot}",
            r"\end{tikzpicture}",
            r"\caption{Standardized prefix bars for cardinality. Each panel averages the merged cardinality score across datasets whose identifiers begin with `c`, `m`, or `n`.}",
            r"\end{figure}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_standard_tradeoff_preview(tradeoff_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for row in tradeoff_df.itertuples():
        color_hex = MODEL_COLOR_HEX.get(str(row.model), "#777777")
        ax.scatter(float(row.discrete_score_mean), float(row.high_dynamic_score_mean), s=58, color=color_hex)
        ax.text(
            float(row.discrete_score_mean) + 0.012,
            float(row.high_dynamic_score_mean) + 0.008,
            str(row.model),
            fontsize=8,
            color=color_hex,
        )
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Support-rank profile proxy")
    ax.set_ylabel("High-cardinality response stability proxy")
    ax.set_title("Cardinality trade-off scatter")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_standard_prefix_preview(prefix_plot_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.2), sharey=True)
    for ax, prefix in zip(axes, ["c", "m", "n"]):
        values = pd.to_numeric(prefix_plot_df[prefix], errors="coerce")
        colors = [MODEL_COLOR_HEX.get(str(model), "#777777") for model in prefix_plot_df["model"]]
        ax.bar(range(len(prefix_plot_df)), values, color=colors)
        ax.set_title(f"{prefix.upper()} datasets")
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(range(len(prefix_plot_df)))
        ax.set_xticklabels(prefix_plot_df["model"], rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean cardinality score")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_regime_tercile_heatmap_preview(matrix_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    import matplotlib.pyplot as plt

    if matrix_df.empty:
        return
    ordered_cols = [
        "C-low",
        "C-mid",
        "C-high",
        "M-low",
        "M-mid",
        "M-high",
        "N-low",
        "N-mid",
        "N-high",
    ]
    plot_df = matrix_df.set_index("model").reindex(columns=ordered_cols)
    matrix = plot_df.to_numpy(dtype=float)
    cmap = get_heatmap_cmap().copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(len(ordered_cols)))
    ax.set_xticklabels(ordered_cols, rotation=35, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=9)
    for label in ax.get_yticklabels():
        model = label.get_text()
        label.set_color(MODEL_COLOR_HEX.get(model, "#333333"))

    for x in [2.5, 5.5]:
        ax.axvline(x=x, color="#666666", linewidth=1.2)
    ax.text(1.0, -1.15, "Categorical", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(4.0, -1.15, "Mixed", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(7.0, -1.15, "Numerical", ha="center", va="center", fontsize=11, fontweight="bold")

    for row_idx, model in enumerate(plot_df.index):
        for col_idx, column in enumerate(ordered_cols):
            value = plot_df.loc[model, column]
            if pd.isna(value):
                ax.text(col_idx, row_idx, "--", ha="center", va="center", fontsize=7, color="#444444")
                continue
            fill_hex = _heatmap_hex(float(value))
            text_color = "#" + _text_hex_for_fill(fill_hex).lstrip("#")
            ax.text(col_idx, row_idx, f"{float(value):.2f}", ha="center", va="center", fontsize=7.5, color=text_color)

    ax.set_title("Regime-tercile cardinality matrix")
    ax.set_xlabel("Within-regime difficulty terciles (low to high)")
    ax.set_ylabel("Models")
    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Mean score")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset_model_heatmap_preview(
    dataset_summary: pd.DataFrame,
    heatmap_data: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14.0, 13.2), constrained_layout=True)
    image = None
    panel_specs = [("discrete", "Discrete channel"), ("continuous", "Continuous channel")]
    for ax, (channel, title) in zip(axes, panel_specs):
        pivot, keep_order, model_cols = _build_heatmap_subset(channel, dataset_summary, heatmap_data)
        if pivot.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No eligible rows for {channel}.", ha="center", va="center")
            continue
        matrix = pivot.to_numpy(dtype=float)
        image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap=get_heatmap_cmap())
        ax.set_xticks(range(len(model_cols)))
        ax.set_xticklabels(model_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(keep_order)))
        ax.set_yticklabels(keep_order, fontsize=7)
        ax.set_xlabel("Models")
        ax.set_ylabel("Datasets")
        ax.set_title(title)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.022, pad=0.015)
        cbar.set_label("score")
    fig.suptitle("Cardinality dataset-model heatmap", fontsize=14)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _build_figure_tex(
    model_summary: pd.DataFrame,
    model_drop: pd.DataFrame,
    plot_meta: dict[str, Any],
    dataset_summary: pd.DataFrame,
    heatmap_data: pd.DataFrame,
) -> None:
    figure1_path = FIG_DIR / "figure1_dynamic_response_curves.tex"
    figure1_path.write_text(_render_figure1(plot_meta), encoding="utf-8")

    figure2_path = FIG_DIR / "figure2_low_high_slope_plot.tex"
    figure2_path.write_text(_render_figure2(plot_meta, model_drop), encoding="utf-8")

    figure3_path = FIG_DIR / "figure3_dataset_model_heatmap.tex"
    figure3_path.write_text(_render_figure3(dataset_summary, heatmap_data), encoding="utf-8")

    figure4_path = FIG_DIR / "figure4_overall_vs_high_dynamic.tex"
    figure4_path.write_text(_render_figure4(model_summary), encoding="utf-8")

    wrapper = r"""\PassOptionsToPackage{table}{xcolor}
\documentclass[11pt]{article}
\usepackage[margin=0.6in]{geometry}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{subcaption}
\usepackage{adjustbox}
\usepackage{array}
\usepackage{makecell}
\pgfplotsset{compat=1.18}
\begin{document}
\input{figure1_dynamic_response_curves.tex}
\clearpage
\input{figure2_low_high_slope_plot.tex}
\clearpage
\input{figure3_dataset_model_heatmap.tex}
\clearpage
\input{figure4_overall_vs_high_dynamic.tex}
\end{document}
"""
    (FIG_DIR / "cardinality_range_dynamic_results.tex").write_text(wrapper, encoding="utf-8")


def _write_bundle_wrapper(path: Path, body: str) -> None:
    wrapper = "\n".join(
        [
            r"\PassOptionsToPackage{table}{xcolor}",
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=0.6in]{geometry}",
            r"\usepackage{tikz}",
            r"\usepackage{pgfplots}",
            r"\usepackage{pgfplotstable}",
            r"\usepackage{booktabs}",
            r"\usepackage{xcolor}",
            r"\usepackage{colortbl}",
            r"\usepackage{subcaption}",
            r"\usepackage{adjustbox}",
            r"\usepackage{array}",
            r"\usepackage{makecell}",
            r"\pgfplotsset{compat=1.18}",
            r"\usepgfplotslibrary{groupplots}",
            r"\begin{document}",
            body,
            r"\end{document}",
            "",
        ]
    )
    path.write_text(wrapper, encoding="utf-8")


def _sync_export_bundles(paper_dir: Path) -> None:
    paper_fig_dir = paper_dir.parent / "figures"
    final_data_dir = FINAL_DIR / "data"
    preview_dir = FINAL_DIR / "preview"
    for path in [paper_fig_dir, final_data_dir, preview_dir]:
        path.mkdir(parents=True, exist_ok=True)

    figure1_source = (FIG_DIR / "figure1_dynamic_response_curves.tex").read_text(encoding="utf-8")
    figure3_source = (FIG_DIR / "figure3_dataset_model_heatmap.tex").read_text(encoding="utf-8")

    final_figure1_source = figure1_source.replace("../data/", "data/")
    paper_figure1_source = figure1_source.replace("../data/", "../../Evaluation/query_fivepart_breakdown/cardinality/data/")

    (FINAL_DIR / "cardinality_dynamic_response_curves_source.tex").write_text(
        final_figure1_source, encoding="utf-8"
    )
    (FINAL_DIR / "cardinality_dataset_model_heatmap_source.tex").write_text(
        figure3_source, encoding="utf-8"
    )
    (paper_fig_dir / "cardinality_dynamic_response_curves_source.tex").write_text(
        paper_figure1_source, encoding="utf-8"
    )
    (paper_fig_dir / "cardinality_dataset_model_heatmap_source.tex").write_text(
        figure3_source, encoding="utf-8"
    )

    _write_bundle_wrapper(
        FINAL_DIR / "cardinality_dynamic_response_curves_main.tex",
        r"\input{cardinality_dynamic_response_curves_source.tex}",
    )
    _write_bundle_wrapper(
        FINAL_DIR / "cardinality_dataset_model_heatmap_main.tex",
        r"\input{cardinality_dataset_model_heatmap_source.tex}",
    )
    _write_bundle_wrapper(
        FINAL_DIR / "cardinality_main_figures.tex",
        "\n".join(
            [
                r"\input{cardinality_dynamic_response_curves_source.tex}",
                r"\clearpage",
                r"\input{cardinality_dataset_model_heatmap_source.tex}",
            ]
        ),
    )

    _write_bundle_wrapper(
        paper_fig_dir / "cardinality_dynamic_response_curves_main.tex",
        r"\input{cardinality_dynamic_response_curves_source.tex}",
    )
    _write_bundle_wrapper(
        paper_fig_dir / "cardinality_dataset_model_heatmap_main.tex",
        r"\input{cardinality_dataset_model_heatmap_source.tex}",
    )

    for csv_name in ["figure1_discrete_curves.csv", "figure1_continuous_curves.csv", "figure1_combined_curves.csv"]:
        shutil.copy2(DATA_DIR / csv_name, final_data_dir / csv_name)

    for pdf_name in [
        "cardinality_dynamic_response_curves_main.pdf",
        "cardinality_dataset_model_heatmap_main.pdf",
        "cardinality_main_figures.pdf",
    ]:
        src = FINAL_DIR / pdf_name
        if src.exists():
            shutil.copy2(src, paper_fig_dir / pdf_name)


def _render_figure1(plot_meta: dict[str, Any]) -> str:
    csv_path = Path(plot_meta["combined_curve_path"]).name
    curve_columns = set(pd.read_csv(DATA_DIR / csv_path).columns)
    xticks = plot_meta["combined_xticks"]
    xticklabels = plot_meta["combined_xticklabels"]

    lines = [
        _render_model_color_definitions(),
        r"\begin{figure}[p]",
        r"\centering",
        r"\begin{tikzpicture}",
        rf"""\begin{{axis}}[
width=0.9\textwidth,
height=0.42\textwidth,
scale only axis,
clip=false,
ymin=0, ymax=1.02,
xmin=0.5, xmax=12.5,
xtick={{{','.join(str(item) for item in xticks)}}},
xticklabels={{{','.join(xticklabels)}}},
x tick label style={{rotate=32, anchor=east, font=\scriptsize}},
ylabel={{Score}},
xlabel={{Dynamic-response buckets}},
grid=major,
grid style={{gray!20}},
]
""",
        r"\path[fill=orange!12, draw=none] (axis cs:4.5,0) rectangle (axis cs:6.5,1.02);",
        r"\path[fill=orange!12, draw=none] (axis cs:10.5,0) rectangle (axis cs:12.5,1.02);",
        r"\draw[black!45, dashed, line width=0.8pt] (axis cs:7,0) -- (axis cs:7,1.02);",
        r"\node[anchor=south, font=\small] at (axis cs:3.5,1.03) {Categorical support buckets};",
        r"\node[anchor=south, font=\small] at (axis cs:10,1.03) {Continuous range buckets};",
    ]
    for model in FIGURE_MODEL_ORDER:
        if model not in curve_columns:
            continue
        color, marker, width = _model_style(model)
        lines.append(
            rf"\addplot[{width}, {color}, mark={marker}, mark size=1.6pt] table[x=plot_x,y={model},col sep=comma] {{{'../data/' + csv_path}}};"
        )
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\vspace{0.4em}",
            _render_shared_model_legend(),
            r"\caption{Unified dynamic response curves for validation-layer cardinality/range. The left half traces categorical support-retention buckets grouped by real cardinality, while the right half traces continuous range-envelope buckets Q1--Q5 derived from log10(real range width). The dashed divider separates the two channels, and the shaded regions mark the highest-difficulty buckets within each channel.}",
            r"\end{figure}",
        ]
    )
    return "\n".join(lines)


def _plot_dynamic_response_curves_preview(plot_meta: dict[str, Any], pdf_path: Path, png_path: Path) -> None:
    import matplotlib.pyplot as plt

    combined_df = pd.read_csv(plot_meta["combined_curve_path"])
    x_positions = plot_meta["combined_xticks"]
    x_labels = plot_meta["combined_xticklabels"]
    mpl_marker_map = {
        "*": "*",
        "square*": "s",
        "triangle*": "^",
        "diamond*": "D",
        "otimes*": "P",
        "pentagon*": "p",
        "asterisk": "*",
        "Mercedes star": "X",
        "x": "x",
        "oplus*": "P",
        "triangle": "^",
        "diamond": "d",
    }

    fig, ax = plt.subplots(figsize=(15.5, 6.4))
    ax.axvspan(4.5, 6.5, color="#f4b183", alpha=0.18, lw=0)
    ax.axvspan(10.5, 12.5, color="#f4b183", alpha=0.18, lw=0)
    ax.axvline(7.0, color="#666666", linestyle="--", linewidth=1.0)

    for model in FIGURE_MODEL_ORDER:
        if model not in combined_df.columns:
            continue
        series = pd.to_numeric(combined_df[model], errors="coerce")
        color_hex = MODEL_COLOR_HEX.get(model, "#777777")
        ax.plot(
            combined_df["plot_x"],
            series,
            color=color_hex,
            marker=mpl_marker_map.get(MODEL_MARKER_MAP.get(model, "o"), "o"),
            linewidth=1.9,
            markersize=4.8,
            label=model,
        )

    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=32, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlabel("Dynamic-response buckets")
    ax.set_title("Unified dynamic response curves for cardinality")
    ax.grid(alpha=0.25)
    ax.text(3.5, 1.025, "Categorical support buckets", ha="center", va="bottom", fontsize=10)
    ax.text(10.0, 1.025, "Continuous range buckets", ha="center", va="bottom", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=6, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _render_figure2(plot_meta: dict[str, Any], model_drop: pd.DataFrame) -> str:
    def render_panel(channel: str, title: str) -> str:
        csv_name = Path(plot_meta[f"{channel}_slope_path"]).name
        slope_df = pd.read_csv(DATA_DIR / csv_name)
        channel_drop = model_drop.loc[
            (model_drop["channel"] == channel) & (model_drop["model"].isin(FIGURE_MODEL_ORDER))
        ].dropna(subset=["absolute_drop"]).copy()
        stable = channel_drop.sort_values(["absolute_drop", "high_dynamic_mean"], ascending=[True, False]).head(1)["model"].tolist()
        fragile = channel_drop.sort_values(["absolute_drop", "high_dynamic_mean"], ascending=[False, True]).head(1)["model"].tolist()
        lines = [
            r"\begin{subfigure}[t]{0.49\textwidth}",
            r"\centering",
            r"\begin{tikzpicture}",
            rf"""\begin{{axis}}[
width=\textwidth,
height=0.72\textwidth,
ymin=0, ymax=1.02,
xmin=0.7, xmax=2.3,
xtick={{1,2}},
xticklabels={{Low dynamic,High dynamic}},
ylabel={{Mean score}},
title={{{title}}},
grid=major,
grid style={{gray!20}},
]
""",
        ]
        for model in FIGURE_MODEL_ORDER:
            if model not in slope_df.columns:
                continue
            color, marker, width = _model_style(model)
            lines.append(
                rf"\addplot[{width}, {color}, mark={marker}, mark size=1.8pt] table[x=x,y={model},col sep=comma] {{{'../data/' + csv_name}}};"
            )
        if stable:
            stable_model = stable[0]
            stable_color, _, _ = _model_style(stable_model)
            lines.append(rf"\node[anchor=south west, font=\scriptsize, text={stable_color}] at (rel axis cs:0.03,0.95) {{Most stable: {stable_model}}};")
        if fragile:
            fragile_model = fragile[0]
            fragile_color, _, _ = _model_style(fragile_model)
            lines.append(rf"\node[anchor=north west, font=\scriptsize, text={fragile_color}] at (rel axis cs:0.03,0.85) {{Largest drop: {fragile_model}}};")
        lines.extend([r"\end{axis}", r"\end{tikzpicture}", r"\end{subfigure}"])
        return "\n".join(lines)

    return "\n".join(
        [
            _render_model_color_definitions(),
            r"\begin{figure}[p]",
            r"\centering",
            render_panel("discrete", "Panel A: Discrete low-to-high slope"),
            r"\hfill",
            render_panel("continuous", "Panel B: Continuous low-to-high slope"),
            r"\vspace{0.4em}",
            _render_shared_model_legend(),
            r"\par\vspace{0.45em}",
            r"\begin{minipage}{0.94\textwidth}\centering\small",
            r"Every displayed model keeps the same color as in Figure 1. Each line connects the model's low-dynamic mean score to its high-dynamic mean score.",
            r"\end{minipage}",
            r"\caption{Low-versus-high dynamic slope plots for all README color-convention models. Each line tracks one generator from low-difficulty buckets to high-difficulty buckets.}",
            r"\end{figure}",
        ]
    )


def _heatmap_hex(score: float | None) -> str:
    return _shared_heatmap_hex(score)


def _text_hex_for_fill(fill_hex: str) -> str:
    return _shared_text_hex_for_fill(fill_hex)


def _build_heatmap_subset(channel: str, dataset_summary: pd.DataFrame, heatmap_data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    subset = heatmap_data.loc[heatmap_data["channel"] == channel].copy()
    if subset.empty:
        return pd.DataFrame(), [], []
    subset = subset.loc[
        (~subset["model"].isin(HEATMAP_EXCLUDED_MODELS)) & (~subset["dataset"].isin(HEATMAP_EXCLUDED_DATASETS))
    ].copy()
    order_df = dataset_summary.loc[
        ~dataset_summary["dataset"].isin(HEATMAP_EXCLUDED_DATASETS)
    ].sort_values("dataset_dynamic_intensity", ascending=False)
    hardest = list(order_df.head(15)["dataset"])
    easiest = list(order_df.tail(15)["dataset"])
    keep_order = hardest + [d for d in easiest if d not in hardest]
    keep_order = sorted(keep_order, key=_dataset_sort_key)
    subset = subset.loc[subset["dataset"].isin(keep_order)].copy()
    pivot = subset.pivot(index="dataset", columns="model", values="score").reindex(keep_order)
    model_cols = [m for m in MODEL_ORDER if m not in HEATMAP_EXCLUDED_MODELS and m in pivot.columns]
    return pivot.loc[keep_order, model_cols], keep_order, model_cols


def _render_heatmap_colorbar(x_left: float, y_top: float, height: float, width: float = 0.36) -> list[str]:
    slices = 60
    lines = [rf"\node[anchor=south, font=\scriptsize] at ({x_left + width/2:.2f},{y_top + 0.28:.2f}) {{score}};"]
    for idx in range(slices):
        t0 = idx / slices
        t1 = (idx + 1) / slices
        y0 = y_top - (height * t0)
        y1 = y_top - (height * t1)
        fill_hex = _heatmap_hex(1.0 - t0)
        lines.append(
            rf"\fill[draw=none, fill={{rgb,255:red,{int(fill_hex[0:2], 16)}; green,{int(fill_hex[2:4], 16)}; blue,{int(fill_hex[4:6], 16)}}}] ({x_left:.2f},{y0:.2f}) rectangle ({x_left + width:.2f},{y1:.2f});"
        )
    lines.append(rf"\draw[black!25, line width=0.2pt] ({x_left:.2f},{y_top:.2f}) rectangle ({x_left + width:.2f},{y_top - height:.2f});")
    lines.append(rf"\node[anchor=west, font=\scriptsize] at ({x_left + width + 0.10:.2f},{y_top:.2f}) {{1.00}};")
    lines.append(rf"\node[anchor=west, font=\scriptsize] at ({x_left + width + 0.10:.2f},{y_top - height/2:.2f}) {{0.50}};")
    lines.append(rf"\node[anchor=west, font=\scriptsize] at ({x_left + width + 0.10:.2f},{y_top - height:.2f}) {{0.00}};")
    return lines


def _render_heatmap_panel(channel: str, dataset_summary: pd.DataFrame, heatmap_data: pd.DataFrame) -> str:
    pivot, keep_order, model_cols = _build_heatmap_subset(channel, dataset_summary, heatmap_data)
    if pivot.empty:
        return rf"\paragraph{{{channel.title()}}} No eligible rows."

    cell_w = 0.80
    cell_h = 0.28
    n_rows = len(keep_order)
    n_cols = len(model_cols)
    total_h = n_rows * cell_h
    total_w = n_cols * cell_w
    title = f"Panel {'A' if channel == 'discrete' else 'B'}: {channel.title()} channel"
    lines = [
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tikzpicture}[x=1cm,y=1cm]",
        rf"\node[anchor=west, font=\small\bfseries] at (-2.35,0.75) {{{title}}};",
    ]

    for row_idx, dataset in enumerate(keep_order):
        y_top = -(row_idx * cell_h)
        y_center = y_top - (cell_h / 2.0)
        lines.append(rf"\node[anchor=east, font=\fontsize{{4.0}}{{4.2}}\selectfont] at (-0.12,{y_center:.2f}) {{{_latex_escape(dataset)}}};")
        for col_idx, model in enumerate(model_cols):
            x_left = col_idx * cell_w
            score = pivot.loc[dataset, model]
            fill_hex = _heatmap_hex(score if pd.notna(score) else None)
            lines.append(
                rf"\fill[draw=none, fill={{rgb,255:red,{int(fill_hex[0:2], 16)}; green,{int(fill_hex[2:4], 16)}; blue,{int(fill_hex[4:6], 16)}}}] ({x_left:.2f},{y_top:.2f}) rectangle ({x_left + cell_w:.2f},{y_top - cell_h:.2f});"
            )

    for col_idx, model in enumerate(model_cols):
        x_center = (col_idx * cell_w) + (cell_w / 2.0)
        lines.append(
            rf"\node[anchor=south east, rotate=45, font=\fontsize{{4.0}}{{4.2}}\selectfont] at ({x_center:.2f},0.10) {{{_latex_escape(model)}}};"
        )

    lines.append(rf"\draw[black!20, line width=0.2pt] (0,0) rectangle ({total_w:.2f},{-total_h:.2f});")
    for row_idx in range(1, n_rows):
        y = -(row_idx * cell_h)
        lines.append(rf"\draw[white!35, line width=0.18pt] (0,{y:.2f}) -- ({total_w:.2f},{y:.2f});")
    for col_idx in range(1, n_cols):
        x = col_idx * cell_w
        lines.append(rf"\draw[white!35, line width=0.18pt] ({x:.2f},0) -- ({x:.2f},{-total_h:.2f});")

    lines.extend(_render_heatmap_colorbar(total_w + 0.65, 0.0, total_h))
    lines.extend([r"\end{tikzpicture}", r"\end{adjustbox}"])
    return "\n".join(lines)


def _render_figure3(dataset_summary: pd.DataFrame, heatmap_data: pd.DataFrame) -> str:
    return "\n".join(
        [
            r"\begin{figure}[p]",
            r"\centering",
            _render_heatmap_panel("discrete", dataset_summary, heatmap_data),
            r"\vspace{0.8em}",
            _render_heatmap_panel("continuous", dataset_summary, heatmap_data),
            r"\par\vspace{0.45em}",
            r"\begin{minipage}{0.94\textwidth}\centering\small",
            r"\texttt{c21} and \texttt{n13} are excluded from the display subset. Models \texttt{codi}, \texttt{goggle}, and \texttt{cdtd} are removed from the heatmap columns. Rows follow canonical dataset order (\texttt{c*}, then \texttt{m*}, then \texttt{n*}), and model columns stay alphabetical.",
            r"\end{minipage}",
            r"\caption{Compact dataset-by-model heatmaps for the filtered representative subset. Cell color alone encodes the mean score so the macro structure stays visually clean while preserving the same fixed palette across panels.}",
            r"\end{figure}",
        ]
    )


def _render_figure4(model_summary: pd.DataFrame) -> str:
    scatter_df = model_summary.loc[
        model_summary["model"].isin(FIGURE_MODEL_ORDER)
    ].dropna(subset=["overall_score_mean", "high_dynamic_score_mean"]).copy()
    scatter_df = scatter_df.sort_values("model", key=lambda s: s.map({m: i for i, m in enumerate(FIGURE_MODEL_ORDER)}))
    labels = scatter_df.sort_values("relative_drop", ascending=False).head(4)
    label_nodes = []
    for _, row in labels.iterrows():
        label_color, _, _ = _model_style(str(row["model"]))
        label_nodes.append(
            rf"\node[anchor=west, font=\scriptsize, text={label_color}] at (axis cs:{row['overall_score_mean']:.3f},{row['high_dynamic_score_mean']:.3f}) {{{row['model']}}};"
        )
    point_plots = []
    for _, row in scatter_df.iterrows():
        color, marker, _ = _model_style(str(row["model"]))
        point_plots.append(
            rf"\addplot[only marks, mark={marker}, mark size=2.4pt, {color}] coordinates {{({row['overall_score_mean']:.6f},{row['high_dynamic_score_mean']:.6f})}};"
        )
    return "\n".join(
        [
            _render_model_color_definitions(),
            r"\begin{figure}[p]",
            r"\centering",
            r"\begin{tikzpicture}",
            r"""\begin{axis}[
width=0.78\textwidth,
height=0.62\textwidth,
xmin=0, xmax=1.02,
ymin=0, ymax=1.02,
xlabel={Overall mean score},
ylabel={High-dynamic mean score},
grid=major,
grid style={gray!20},
]
\addplot[gray!60, dashed, domain=0:1] {x};
""",
            *point_plots,
            *label_nodes,
            r"\node[anchor=north west, align=left, font=\scriptsize] at (rel axis cs:0.03,0.97) {Points below the diagonal underperform\\their own overall averages in hard regimes.};",
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption{Overall average versus high-dynamic performance for the README color-convention models. The diagonal is the equality line. Models far below the line are examples where aggregate validation averages hide high-difficulty structural failures.}",
            r"\end{figure}",
        ]
    )


def _build_table_tex(model_summary: pd.DataFrame, model_drop: pd.DataFrame, failures: pd.DataFrame) -> None:
    top_overall = model_summary["overall_score_mean"].max()
    top_high = model_summary["high_dynamic_score_mean"].max()
    max_drop = model_summary["relative_drop"].max()

    table1_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Overall & Discrete & Continuous & High-dyn. & Rank (overall) & Rank (high) \\",
        r"\midrule",
    ]
    for _, row in model_summary.iterrows():
        overall = _maybe_float_str(row["overall_score_mean"], 3)
        discrete = _maybe_float_str(row["discrete_score_mean"], 3)
        continuous = _maybe_float_str(row["continuous_score_mean"], 3)
        high = _maybe_float_str(row["high_dynamic_score_mean"], 3)
        if _coerce_float(row["overall_score_mean"]) == _coerce_float(top_overall):
            overall = rf"\textbf{{{overall}}}"
        if _coerce_float(row["high_dynamic_score_mean"]) == _coerce_float(top_high):
            high = rf"\textbf{{{high}}}"
        table1_lines.append(
            rf"{_latex_escape(row['model'])} & {overall} & {discrete} & {continuous} & {high} & {row['rank_overall']} & {row['rank_high_dynamic']} \\"
        )
    table1_lines.extend([r"\bottomrule", r"\end{tabular}", r"\caption{Model-level summary for cardinality/range dynamic analysis.}", r"\end{table}"])
    (TABLE_DIR / "table1_model_summary.tex").write_text("\n".join(table1_lines), encoding="utf-8")

    table2_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Channel & Low dyn. & High dyn. & Abs. drop & Rel. drop & Robustness rank \\",
        r"\midrule",
    ]
    for _, row in model_drop.iterrows():
        rel_drop = _maybe_float_str(row["relative_drop"], 3)
        if _coerce_float(row["relative_drop"]) == _coerce_float(max_drop):
            rel_drop = rf"\textcolor{{red}}{{{rel_drop}}}"
        table2_lines.append(
            rf"{_latex_escape(row['model'])} & {_latex_escape(row['channel'])} & {_maybe_float_str(row['low_dynamic_mean'], 3)} & {_maybe_float_str(row['high_dynamic_mean'], 3)} & {_maybe_float_str(row['absolute_drop'], 3)} & {rel_drop} & {row['rank_by_robustness']} \\"
        )
    table2_lines.extend([r"\bottomrule", r"\end{tabular}", r"\caption{High-dynamic robustness from low to high difficulty.}", r"\end{table}"])
    (TABLE_DIR / "table2_high_dynamic_robustness.tex").write_text("\n".join(table2_lines), encoding="utf-8")

    table3_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Dataset & Model & Channel & Column & Bucket & Score & Reason \\",
        r"\midrule",
    ]
    for _, row in failures.head(12).iterrows():
        table3_lines.append(
            rf"{_latex_escape(row['dataset'])} & {_latex_escape(row['model'])} & {_latex_escape(row['channel'])} & {_latex_escape(row['column'])} & {_latex_escape(row['dynamic_bucket'])} & {_maybe_float_str(row['score'], 2)} & {_latex_escape(row['short_reason'])} \\"
        )
    table3_lines.extend([r"\bottomrule", r"\end{tabular}", r"\caption{Representative high-dynamic failure cases.}", r"\end{table}"])
    (TABLE_DIR / "table3_failure_cases.tex").write_text("\n".join(table3_lines), encoding="utf-8")

    wrapper = r"""\PassOptionsToPackage{table}{xcolor}
\documentclass[11pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{booktabs}
\usepackage{xcolor}
\begin{document}
\input{table1_model_summary.tex}
\input{table2_high_dynamic_robustness.tex}
\input{table3_failure_cases.tex}
\input{table4_regime_tercile_matrix.tex}
\end{document}
"""
    (TABLE_DIR / "latex_tables.tex").write_text(wrapper, encoding="utf-8")


def _coverage_summary(chosen_rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    datasets = {row.get("asset", {}).get("dataset_id") for row in chosen_rows}
    models = {row.get("_normalized_model") for row in chosen_rows}
    return len(datasets), len(models), len(chosen_rows)


def _findings(model_summary: pd.DataFrame, model_drop: pd.DataFrame, dataset_summary: pd.DataFrame) -> dict[str, Any]:
    overall_best = model_summary.sort_values("overall_score_mean", ascending=False).iloc[0]
    high_best = model_summary.dropna(subset=["high_dynamic_score_mean"]).sort_values("high_dynamic_score_mean", ascending=False).iloc[0]
    largest_drop = model_summary.dropna(subset=["relative_drop"]).sort_values("relative_drop", ascending=False).iloc[0]
    discrete_drop = model_drop.loc[model_drop["channel"] == "discrete"].dropna(subset=["absolute_drop"]).sort_values("absolute_drop", ascending=False).iloc[0]
    continuous_drop = model_drop.loc[model_drop["channel"] == "continuous"].dropna(subset=["absolute_drop"]).sort_values("absolute_drop", ascending=False).iloc[0]
    hardest_dataset = dataset_summary.sort_values("mean_score", ascending=True).iloc[0]
    easiest_dataset = dataset_summary.sort_values("mean_score", ascending=False).iloc[0]
    return {
        "overall_best_model": overall_best["model"],
        "overall_best_score": overall_best["overall_score_mean"],
        "high_best_model": high_best["model"],
        "high_best_score": high_best["high_dynamic_score_mean"],
        "largest_drop_model": largest_drop["model"],
        "largest_drop_value": largest_drop["relative_drop"],
        "discrete_largest_drop_model": discrete_drop["model"],
        "continuous_largest_drop_model": continuous_drop["model"],
        "hardest_dataset": hardest_dataset["dataset"],
        "easiest_dataset": easiest_dataset["dataset"],
    }


def _build_inventory(
    *,
    primary_run: ValidationRunInfo,
    paper_dir: Path,
    clean_df: pd.DataFrame,
    duplicate_audit: list[dict[str, Any]],
    chosen_rows: list[dict[str, Any]],
) -> None:
    datasets, models, chosen_assets = _coverage_summary(chosen_rows)
    paper_files = [
        paper_dir / "sections" / "introduction.tex",
        paper_dir / "sections" / "evaluation_methodology.tex",
        paper_dir / "sections" / "evaluation.tex",
        paper_dir / "sections" / "appendix_scoring_standard.tex",
    ]
    file_rows = [
        {
            "path": "README.md",
            "type": "markdown",
            "granularity": "project-level",
            "fields": "benchmark scope, evaluation layers, model roster, repo layout",
            "used": "yes",
            "reason": "Defines project framing and final scientific takeaway.",
        },
        *[
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "type": "tex",
                "granularity": "paper-section",
                "fields": "evaluation narrative, scoring interpretation, main-text phrasing",
                "used": "yes",
                "reason": "Aligns the analysis language with the paper draft.",
            }
            for path in paper_files
        ],
        {
            "path": "doc/synthetic_data_scoring_protocol_v0_4.md",
            "type": "markdown",
            "granularity": "metric-level",
            "fields": "step-by-step validation protocol, applicability rules",
            "used": "yes",
            "reason": "Cross-checks the implementation against the documented scoring standard.",
        },
        {
            "path": "tqb_scoring/evaluation/synthetic_validation_v4.py",
            "type": "python",
            "granularity": "metric-level",
            "fields": "exact score logic, per-column detail fields, channel composition",
            "used": "yes",
            "reason": "Primary source for the exact discrete and continuous formulas.",
        },
        {
            "path": str(primary_run.details_jsonl.relative_to(PROJECT_ROOT)),
            "type": "jsonl",
            "granularity": "dataset-model asset with column-level detail",
            "fields": "dataset_id, model_id, cardinality_range discrete/continuous details, per-column real/synthetic support and ranges",
            "used": "primary",
            "reason": "Main analysis source because it preserves the per-column structures needed for dynamic buckets.",
        },
        {
            "path": str(primary_run.summary_csv.relative_to(PROJECT_ROOT)),
            "type": "csv",
            "granularity": "dataset-model asset",
            "fields": "dataset_id, model_id, cardinality_range_score and other validation scores",
            "used": "secondary",
            "reason": "Used for schema cross-checking and run-level coverage confirmation.",
        },
        {
            "path": str(primary_run.manifest_path.relative_to(PROJECT_ROOT)),
            "type": "json",
            "granularity": "run-level",
            "fields": "dataset_count, asset_count",
            "used": "secondary",
            "reason": "Used to select the highest-coverage validation run.",
        },
        {
            "path": "logs/runs/ (legacy/v1 benchmark and grounded-SQL artifacts)",
            "type": "directory",
            "granularity": "grounded-SQL run artifacts",
            "fields": "query_results.jsonl, run manifests, traces",
            "used": "context only",
            "reason": "Confirms repository positioning as workload-grounded, but not used for cardinality/range metric values.",
        },
    ]
    unused_rows = [
        "Older validation runs under `Evaluation/validation/runs/20260426_100004`, `20260426_100103`, `20260426_100210`, and `20260426_105754` were excluded because they only cover 1--2 datasets.",
        "The run-level summary CSV is not rich enough for dynamic-bucket analysis because it lacks per-column support and range envelopes.",
        "Synthetic CSVs under `SynOutput/` and `SynOutput-5090/` were not read directly because the unified validation details already materialize the necessary real-vs-synthetic comparisons.",
    ]
    schema_issues = [
        "Model alias inconsistency: `rtf` appears alongside `realtabformer`; the analysis normalizes both to `realtabformer`.",
        "The full validation run contains duplicate `dataset × normalized_model` assets for 17 pairs, mostly across `SynOutput` vs `SynOutput-5090`; the analysis keeps the newest generation timestamp and records dropped duplicates.",
        "Coverage is incomplete for several generators even in the highest-coverage run, so the effective panel is not a complete `51 × 14` matrix.",
        "Windows-style paths are embedded in the unified evaluation outputs, but they are treated as metadata strings rather than local file dependencies.",
    ]

    lines = [
        "# Cardinality / Range Dynamic Inventory",
        "",
        "## 1. Related files found",
        "",
        "| Path | Type | Granularity | Main fields | Used? | Why |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in file_rows:
        lines.append(
            f"| `{row['path']}` | {row['type']} | {row['granularity']} | {row['fields']} | {row['used']} | {row['reason']} |"
        )

    lines.extend(
        [
            "",
            "## 2. Primary analysis source",
            "",
            f"- Selected run: `{primary_run.run_id}`",
            f"- Details file: `{primary_run.details_jsonl.relative_to(PROJECT_ROOT)}`",
            f"- Coverage before deduplication: `{primary_run.asset_count}` assets across `{primary_run.dataset_count}` datasets",
            f"- Coverage after deduplication and alias normalization: `{chosen_assets}` dataset-model assets across `{datasets}` datasets and `{models}` normalized models",
            f"- Column-level analysis units built: `{len(clean_df)}`",
            f"- Duplicate normalized dataset-model pairs resolved: `{len(duplicate_audit)}` dropped assets",
            "",
            "## 3. Field availability",
            "",
            "| Source | dataset id/name | model/generator | discrete/continuous channel | score | real distinct | synthetic distinct | real min/max | synthetic min/max | column name | overall validation score |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| validation_details__all_datasets.jsonl | yes | yes | yes | yes | yes (discrete per-column) | yes (discrete per-column) | yes (continuous per-column) | yes (continuous per-column) | yes | yes |",
            "| validation_summary__all_datasets.csv | yes | yes | no | channel-level only | no | no | no | no | no | yes |",
            "",
            "## 4. Field mappings used by the loader",
            "",
            "- `asset.model_id` -> normalized `model` via alias map `{rtf: realtabformer}`.",
            "- `report.validation_channels.cardinality_range.details.discrete_profile.per_column[].real_distinct` -> `real_distinct_count`.",
            "- `report.validation_channels.cardinality_range.details.discrete_profile.per_column[].syn_distinct` -> `synthetic_distinct_count`.",
            "- `report.validation_channels.cardinality_range.details.continuous_profile.per_column[].real_min/real_max` -> `real_min` / `real_max`.",
            "- `report.validation_channels.cardinality_range.details.continuous_profile.per_column[].syn_min/syn_max` -> `synthetic_min` / `synthetic_max`.",
            "- `report.validation_channels.cardinality_range.discrete_profile_score` / `continuous_profile_score` remain attached as official channel-level references.",
            "",
            "## 5. Files used only for cross-checking or context",
            "",
        ]
    )
    for item in unused_rows:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 6. Schema inconsistencies and missing fields",
            "",
        ]
    )
    for item in schema_issues:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 7. Coverage gaps that matter for interpretation",
            "",
            "- The normalized roster still reaches all 14 target generator names, but some generators have much smaller dataset coverage than the benchmark maximum.",
            "- `codi`, `forestdiffusion`, and `tabdiff` are especially sparse, so their high-dynamic conclusions should be read as conditional on smaller sample sizes.",
            "",
            "## 8. Final decision for the main analysis",
            "",
            "- Use `validation_details__all_datasets.jsonl` as the main source.",
            "- Use column-level units whenever possible.",
            "- Keep discrete and continuous channels separate until the final cross-channel summaries.",
            "- Treat the discrete unit score as a derived support-retention view built directly from official per-column support-loss counts; note this explicitly in the report.",
        ]
    )

    (CARDINALITY_ROOT / "inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_report(
    *,
    primary_run: ValidationRunInfo,
    clean_df: pd.DataFrame,
    chosen_rows: list[dict[str, Any]],
    model_summary: pd.DataFrame,
    summary_by_dataset: pd.DataFrame,
    model_drop: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    datasets, models, chosen_assets = _coverage_summary(chosen_rows)
    findings = _findings(model_summary, model_drop, summary_by_dataset)
    broad_coverage = model_summary.loc[model_summary["n_datasets"] >= 30].copy()
    broad_high_best = broad_coverage.sort_values("high_dynamic_score_mean", ascending=False).iloc[0]
    good_overall_hidden = model_summary.loc[
        (model_summary["overall_score_mean"] >= 0.8) & (model_summary["n_datasets"] >= 30)
    ].copy()
    good_overall_hidden["overall_minus_high"] = (
        good_overall_hidden["overall_score_mean"] - good_overall_hidden["high_dynamic_score_mean"]
    )
    hidden_examples = good_overall_hidden.sort_values("overall_minus_high", ascending=False).head(2)
    discrete_drop_mean = model_drop.loc[model_drop["channel"] == "discrete", "absolute_drop"].dropna().mean()
    continuous_drop_mean = model_drop.loc[model_drop["channel"] == "continuous", "absolute_drop"].dropna().mean()
    harder_channel = "discrete" if discrete_drop_mean > continuous_drop_mean else "continuous"
    high_dynamic_gap = (
        model_summary["overall_score_mean"] - model_summary["high_dynamic_score_mean"]
    ).dropna()
    hidden_failure_count = int((high_dynamic_gap > 0.05).sum())
    hardest_dataset = summary_by_dataset.sort_values("mean_score", ascending=True).iloc[0]
    widest_gap_dataset = (
        clean_df.groupby(["dataset", "model"], as_index=False)["score"].mean()
        .groupby("dataset")["score"]
        .agg(lambda s: float(s.max() - s.min()) if not s.empty else math.nan)
        .sort_values(ascending=False)
    )
    widest_gap_dataset_name = widest_gap_dataset.index[0]
    widest_gap_value = float(widest_gap_dataset.iloc[0])
    discrete_bucket_means = clean_df.loc[clean_df["channel"] == "discrete"].groupby("dynamic_bucket")["score"].mean()
    continuous_bucket_means = clean_df.loc[clean_df["channel"] == "continuous"].groupby("dynamic_bucket")["score"].mean()
    discrete_easy = float(discrete_bucket_means.get("1-10", math.nan))
    discrete_hard = float(discrete_bucket_means.get(">1000", math.nan))
    continuous_min = float(continuous_bucket_means.min())
    continuous_max = float(continuous_bucket_means.max())
    hidden_example_text = "; ".join(
        f"`{row.model}` ({row.overall_score_mean:.3f} -> {row.high_dynamic_score_mean:.3f})"
        for row in hidden_examples.itertuples()
    )

    lines = [
        "# Cardinality / Range Dynamic Analysis",
        "",
        "## 1. Data sources",
        "",
        f"- Primary evaluation run: `{primary_run.run_id}` from `Evaluation/validation/runs/`.",
        f"- Main metric file: `{primary_run.details_jsonl.relative_to(PROJECT_ROOT)}`.",
        f"- Supporting files: `README.md`, paper sections under `Paper/69b27219c555c38a69bb2156/sections/`, `doc/synthetic_data_scoring_protocol_v0_4.md`, and `tqb_scoring/evaluation/synthetic_validation_v4.py`.",
        f"- Coverage after normalization and deduplication: `{datasets}` datasets, `{models}` normalized models, `{chosen_assets}` dataset-model assets, and `{len(clean_df)}` column-level result units.",
        f"- Missing coverage remains non-trivial for several generators, especially `codi`, `forestdiffusion`, and `tabdiff`.",
        "",
        "## 2. Metric interpretation",
        "",
        "- In this repository, `cardinality/range` is one of the four Validation-layer channels, separate from analytics families and classical distance-based fidelity.",
        "- The discrete profile probes whether synthetic data preserves real observed support in non-continuous columns. In implementation terms, the official dataset-level score averages a column-risk indicator and a value-level missing-support indicator.",
        "- The continuous profile probes whether synthetic data preserves the real range envelope for continuous columns using per-column min/max deviation penalties.",
        "- For this analysis, discrete dynamic difficulty is tied to real distinct-count scale, while continuous dynamic difficulty is tied to real range width. This makes the probe difficulty-sensitive rather than treating all columns as equally hard.",
        "- The cleaned unit table keeps official channel scores as references, but the discrete column-level `score` used in bucket analysis is a derived support-retention ratio from the official per-column counts because the official implementation only exposes the two discrete indicators at dataset level.",
        "",
        "## 3. Dynamic bucket construction",
        "",
        "- Discrete channel: absolute log-style buckets over `real_distinct_count` = `1-10`, `11-30`, `31-100`, `101-300`, `301-1000`, `>1000`.",
        "- Continuous channel: quantile buckets over `log10(real_range_width)` because raw widths are not directly comparable across datasets with different units and scales.",
        "- Low/high dynamic groups: discrete uses the first two versus last two buckets; continuous uses the bottom two versus top two quantile buckets.",
        "- Analysis unit: column-level whenever per-column details exist. This holds for both discrete and continuous branches in the selected validation run.",
        "",
        "## 4. Main findings",
        "",
        f"1. The validation probe is difficulty-sensitive. Model separation widens in hard regimes: `{hidden_failure_count}` models lose more than `0.05` absolute score when moving from their own overall average to the high-dynamic subset.",
        f"2. The numerically highest high-dynamic score belongs to `{findings['high_best_model']}` (`{findings['high_best_score']:.3f}`), but that result comes from only `{int(model_summary.loc[model_summary['model'] == findings['high_best_model'], 'n_datasets'].iloc[0])}` datasets. Among broadly covered models (`n_datasets >= 30`), `{broad_high_best['model']}` is the strongest high-dynamic performer at `{broad_high_best['high_dynamic_score_mean']:.3f}`.",
        f"3. Overall averages do hide failure modes. The clearest broad-coverage examples are {hidden_example_text}. Both models look strong in aggregate, but their high-dynamic means are much lower once the analysis equal-weights the discrete and continuous hard regimes.",
        f"4. Discrete versus continuous difficulty is not symmetric. The discrete branch is much sharper: mean score falls from `{discrete_easy:.3f}` in the `1-10` bucket to `{discrete_hard:.3f}` in the `>1000` bucket. By contrast, continuous bucket means stay within `{continuous_min:.3f}` to `{continuous_max:.3f}`, so range-width difficulty is more model-specific than uniformly monotone.",
        f"5. Dataset-level difficulty matters, but not through one scalar alone. The hardest dataset by mean score is `{hardest_dataset['dataset']}` (`mean_score={hardest_dataset['mean_score']:.3f}`), and the largest across-model spread appears on `{widest_gap_dataset_name}` (`score range={widest_gap_value:.3f}`), which shows that some datasets amplify ranking differences much more than others even when their average dynamic intensity is not maximal.",
        f"6. The most fragile discrete and continuous models are not necessarily the same: discrete drop is largest for `{findings['discrete_largest_drop_model']}`, while continuous drop is largest for `{findings['continuous_largest_drop_model']}`.",
        "",
        "## 5. Suggested paper text",
        "",
        "Cardinality/range validation reveals a difficulty-sensitive structural failure mode that is largely hidden by aggregate fidelity summaries. When columns are easy, many generators appear similar. Once the benchmark moves into high-cardinality support or wide-range envelope regimes, however, model behavior diverges sharply. This is exactly the kind of integrity stress that matters for downstream analytics built on grouping, slicing, ranking, and thresholding over real schema columns.",
        "",
        "The discrete and continuous branches surface different weaknesses. In the current panel, high-cardinality support is the clearer global stressor: discrete support-retention falls sharply once real distinct counts leave the low-cardinality regime, and several generators that rank near the top overall collapse on the hardest discrete buckets. Continuous range-width behavior is still informative, but it is less uniformly monotone and instead separates models in a more generator-specific way.",
        "",
        "Most importantly, overall validation averages can mask high-dynamic failures. Several generators that look competitive in aggregate lose substantial score once evaluation is restricted to the structurally hardest columns. This supports the broader benchmark claim that synthetic-data evaluation cannot stop at generic similarity or pooled averages; it must test whether generators preserve the structures that downstream analytics depend on most.",
        "",
        "## 6. Suggested figure captions",
        "",
        "- Figure 1: Dynamic response curves for validation-layer cardinality/range. Scores are plotted against real-data difficulty buckets, showing that model gaps widen as support cardinality or range width increases.",
        "- Figure 2: Low-versus-high dynamic slope plot for the discrete and continuous branches. Steeper downward slopes indicate generators whose apparent average fidelity hides hard-regime brittleness.",
        "- Figure 3: Compact dataset-by-model heatmaps for a filtered representative subset. The figure keeps the hardest and easiest datasets from the panel, but displays them in canonical dataset order so the dataset lookup is stable across revisions.",
        "- Figure 4: Overall average versus high-dynamic performance. Points far below the equality line are models whose aggregate validation scores overstate robustness in the hardest structural regimes.",
        "",
        "## 7. Limitations",
        "",
        "- The selected run is the highest-coverage unified validation run in the repository, but it is still not a complete `51 × 14` panel; missing generator coverage affects confidence for sparse models.",
        "- The discrete unit score used for dynamic buckets is a derived support-retention ratio based on the official per-column support-loss counts, not a separately emitted protocol primitive.",
        "- Continuous dynamic buckets rely on quantiles of log-width because raw numerical ranges are not directly comparable across datasets with different measurement units.",
        "- Column-level analysis is the right unit for difficulty sensitivity, but it does not exactly match the official dataset-level discrete aggregation rule, which mixes equal-weighted column risk and globally weighted missing-support ratios.",
        "",
        "## 8. Connection to the project claim",
        "",
        "- The README argues that synthetic-data evaluation must move beyond asking whether synthetic rows merely resemble real rows under classical distances.",
        "- This analysis supports that claim in a concrete way: even inside the Validation layer, a seemingly simple structural probe already exposes failures that pooled averages can hide.",
        "- High-cardinality support and wide range envelopes are exactly the kinds of structural preconditions that downstream SQL analytics rely on. If those fail, the synthetic table can still look statistically plausible while being analytically unreliable.",
        "",
        "## 9. Representative failure cases",
        "",
    ]
    for _, row in failures.head(10).iterrows():
        lines.append(
            f"- `{row['dataset']}` / `{row['model']}` / `{row['channel']}` / `{row['column']}` / `{row['dynamic_bucket']}`: score `{row['score']:.3f}`; {row['short_reason']}."
        )

    (CARDINALITY_ROOT / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_readme(primary_run: ValidationRunInfo) -> None:
    content = f"""# Cardinality / Range Dynamic Analysis

This directory contains a difficulty-aware analysis of the Validation-layer `cardinality/range` channel using the repository's real unified evaluation outputs.

## What this analysis does

- loads the highest-coverage unified validation run,
- normalizes model aliases and resolves duplicate dataset-model assets,
- builds column-level discrete and continuous analysis units,
- constructs dynamic difficulty buckets,
- exports CSV summaries,
- generates LaTeX/PGFPlots figures and LaTeX tables,
- writes a paper-ready markdown report.

## Inputs

- Primary validation run: `{primary_run.run_id}`
- Unified details JSONL: `{primary_run.details_jsonl.relative_to(PROJECT_ROOT)}`
- Unified summary CSV: `{primary_run.summary_csv.relative_to(PROJECT_ROOT)}`
- Metric definitions: `tqb_scoring/evaluation/synthetic_validation_v4.py`
- Paper framing and metric narrative: `README.md`, `Paper/69b27219c555c38a69bb2156/sections/*`, `doc/synthetic_data_scoring_protocol_v0_4.md`

## Re-run

```bash
python tqb_scoring/eval/query_fivepart_breakdown/cardinality/runner.py
```

## Compile LaTeX figures

```bash
cd Evaluation/query_fivepart_breakdown/cardinality/figures
latexmk -pdf cardinality_range_dynamic_results.tex
```

## Outputs

- `inventory.md`
- `analysis_report.md`
- `data/cleaned_results.csv`
- `data/summary_by_model.csv`
- `data/summary_by_dynamic_bucket.csv`
- `data/summary_by_dataset.csv`
- `data/model_drop_table.csv`
- `data/top_failure_cases.csv`
- `data/heatmap_data.csv`
- `data/regime_tercile_dataset_groups.csv`
- `data/regime_tercile_model_matrix.csv`
- `figures/figure1_dynamic_response_curves.tex`
- `figures/figure2_low_high_slope_plot.tex`
- `figures/figure3_dataset_model_heatmap.tex`
- `figures/figure4_overall_vs_high_dynamic.tex`
- `figures/cardinality_regime_tercile_heatmap.pdf`
- `figures/cardinality_regime_tercile_heatmap.png`
- `tables/table1_model_summary.tex`
- `tables/table2_high_dynamic_robustness.tex`
- `tables/table3_failure_cases.tex`
- `tables/table4_regime_tercile_matrix.tex`

## Suggested placement

- Main text: Figure 1 and Figure 4.
- Appendix / backup slides: Figure 2 and Figure 3, plus the complete CSV tables.
"""
    (CARDINALITY_ROOT / "README.md").write_text(content, encoding="utf-8")


def _write_manifest(primary_run: ValidationRunInfo, clean_df: pd.DataFrame, chosen_rows: list[dict[str, Any]]) -> None:
    manifest = {
        "task": "cardinality_breakdown",
        "status": "completed",
        "published_output_version": OUTPUT_VERSION_TAG,
        "sql_source_version": OUTPUT_VERSION_TAG,
        "sql_source_label": sql_source_label(OUTPUT_VERSION_TAG),
        "source_validation_run_id": primary_run.run_id,
        "source_validation_dataset_count": primary_run.dataset_count,
        "source_validation_asset_count": primary_run.asset_count,
        "analysis_unit_count": int(len(clean_df)),
        "chosen_validation_row_count": int(len(chosen_rows)),
    }
    (CARDINALITY_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _try_compile_latex(tex_path: Path) -> tuple[bool, str]:
    import subprocess

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
    return proc.returncode == 0, proc.stdout[-2000:]


def _sync_standard_final_bundle() -> None:
    version_tag = OUTPUT_VERSION_TAG
    final_files = [
        FINAL_DIR / "cardinality_dynamic_response_curves_main.tex",
        FINAL_DIR / "cardinality_dynamic_response_curves_main.pdf",
        FINAL_DIR / "cardinality_dynamic_response_curves_main.png",
        FINAL_DIR / "cardinality_dynamic_response_curves_source.tex",
        FINAL_DIR / "cardinality_dataset_model_heatmap_main.tex",
        FINAL_DIR / "cardinality_dataset_model_heatmap_main.pdf",
        FINAL_DIR / "cardinality_dataset_model_heatmap_main.png",
        FINAL_DIR / "cardinality_dataset_model_heatmap_source.tex",
        FINAL_DIR / "cardinality_main_figures.tex",
        FINAL_DIR / "cardinality_main_figures.pdf",
        FINAL_DIR / "cardinality_main_figures.png",
        DATA_DIR / "summary_by_model.csv",
        DATA_DIR / "prefix_plot_data.csv",
        DATA_DIR / "regime_tercile_dataset_groups.csv",
        DATA_DIR / "regime_tercile_model_matrix.csv",
        DATA_DIR / "model_subitem_heatmap.csv",
        FIG_DIR / "cardinality_regime_tercile_heatmap.pdf",
        FIG_DIR / "cardinality_regime_tercile_heatmap.png",
        TABLE_DIR / "table4_regime_tercile_matrix.tex",
        FIG_DIR / "cardinality_family_subitem_bars_appendix.tex",
        FIG_DIR / "cardinality_family_subitem_bars_appendix.pdf",
        FIG_DIR / "cardinality_family_subitem_bars_appendix.png",
        CARDINALITY_ROOT / "analysis_report.md",
    ]
    must_do_aliases = {
        "cardinality_tradeoff_scatter_main.tex": FIG_DIR / "cardinality_tradeoff_scatter_main.tex",
        "cardinality_tradeoff_scatter_main.pdf": FIG_DIR / "cardinality_tradeoff_scatter_main.pdf",
        "cardinality_tradeoff_scatter_main.png": FIG_DIR / "cardinality_tradeoff_scatter_main.png",
        "cardinality_prefix_bars_appendix.tex": FIG_DIR / "cardinality_prefix_bars_appendix.tex",
        "cardinality_prefix_bars_appendix.pdf": FIG_DIR / "cardinality_prefix_bars_appendix.pdf",
        "cardinality_prefix_bars_appendix.png": FIG_DIR / "cardinality_prefix_bars_appendix.png",
        "cardinality_dataset_model_heatmap_appendix.tex": FINAL_DIR / "cardinality_dataset_model_heatmap_main.tex",
        "cardinality_dataset_model_heatmap_appendix.pdf": FINAL_DIR / "cardinality_dataset_model_heatmap_main.pdf",
        "cardinality_dataset_model_heatmap_appendix.png": FINAL_DIR / "cardinality_dataset_model_heatmap_main.png",
        "cardinality_model_subitem_heatmap_appendix.tex": FIG_DIR / "cardinality_model_subitem_heatmap_appendix.tex",
        "cardinality_model_subitem_heatmap_appendix.pdf": FIG_DIR / "cardinality_model_subitem_heatmap_appendix.pdf",
        "cardinality_model_subitem_heatmap_appendix.png": FIG_DIR / "cardinality_model_subitem_heatmap_appendix.png",
        "cardinality_family_subitem_bars_appendix.tex": FIG_DIR / "cardinality_family_subitem_bars_appendix.tex",
        "cardinality_family_subitem_bars_appendix.pdf": FIG_DIR / "cardinality_family_subitem_bars_appendix.pdf",
        "cardinality_family_subitem_bars_appendix.png": FIG_DIR / "cardinality_family_subitem_bars_appendix.png",
    }
    sync_final_outputs(
        FINAL_DIR,
        final_files,
        must_do_aliases,
        version_tag=version_tag,
        copy_plain_files=False,
    )

    readme = render_final_readme(
        title="Cardinality Final",
        summary=(
            "This directory contains the merged cardinality breakdown artifacts, "
            f"published for SQL source `{sql_source_label(version_tag)}` (`{version_tag}`), "
            "with the standardized five-figure must-do bundle mirrored into `final/must_do/`."
        ),
        primary_files=[
            versioned_name("cardinality_tradeoff_scatter_main.tex", version_tag),
            versioned_name("cardinality_tradeoff_scatter_main.pdf", version_tag),
            versioned_name("cardinality_tradeoff_scatter_main.png", version_tag),
            versioned_name("cardinality_prefix_bars_appendix.tex", version_tag),
            versioned_name("cardinality_prefix_bars_appendix.pdf", version_tag),
            versioned_name("cardinality_prefix_bars_appendix.png", version_tag),
            versioned_name("cardinality_dataset_model_heatmap_appendix.tex", version_tag),
            versioned_name("cardinality_dataset_model_heatmap_appendix.pdf", version_tag),
            versioned_name("cardinality_dataset_model_heatmap_appendix.png", version_tag),
            versioned_name("cardinality_model_subitem_heatmap_appendix.tex", version_tag),
            versioned_name("cardinality_model_subitem_heatmap_appendix.pdf", version_tag),
            versioned_name("cardinality_model_subitem_heatmap_appendix.png", version_tag),
            versioned_name("cardinality_family_subitem_bars_appendix.tex", version_tag),
            versioned_name("cardinality_family_subitem_bars_appendix.pdf", version_tag),
            versioned_name("cardinality_family_subitem_bars_appendix.png", version_tag),
            versioned_name("cardinality_dynamic_response_curves_main.tex", version_tag),
            versioned_name("cardinality_dynamic_response_curves_main.pdf", version_tag),
            versioned_name("cardinality_dynamic_response_curves_main.png", version_tag),
            versioned_name("cardinality_main_figures.tex", version_tag),
            versioned_name("cardinality_main_figures.pdf", version_tag),
            versioned_name("cardinality_main_figures.png", version_tag),
        ],
        must_do_files=[versioned_name(name, version_tag) for name in must_do_aliases.keys()],
        support_files=[
            versioned_name("cardinality_dynamic_response_curves_source.tex", version_tag),
            versioned_name("cardinality_dataset_model_heatmap_source.tex", version_tag),
            versioned_name("summary_by_model.csv", version_tag),
            versioned_name("prefix_plot_data.csv", version_tag),
            versioned_name("regime_tercile_dataset_groups.csv", version_tag),
            versioned_name("regime_tercile_model_matrix.csv", version_tag),
            versioned_name("model_subitem_heatmap.csv", version_tag),
            versioned_name("cardinality_regime_tercile_heatmap.pdf", version_tag),
            versioned_name("cardinality_regime_tercile_heatmap.png", version_tag),
            versioned_name("table4_regime_tercile_matrix.tex", version_tag),
            versioned_name("analysis_report.md", version_tag),
        ],
        notes=[
            "The wrappers point to local source files, so `final/` remains self-contained for Overleaf or appendix export.",
            f"Unsuffixed legacy outputs are preserved outside this published `{version_tag}` bundle.",
        ],
    )
    (FINAL_DIR / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze validation-layer cardinality/range dynamics from repository validation runs."
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional explicit Evaluation/validation run directory to use as the source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dirs()
    runs = _load_validation_runs()
    primary_run = _resolve_primary_run(runs, args.source_run_dir)
    paper_dir = _pick_paper_dir()
    chosen_rows, duplicate_audit = _load_and_deduplicate_validation_details(primary_run.details_jsonl)
    clean_df = _build_cleaned_results(chosen_rows, source_file=primary_run.details_jsonl)
    if clean_df.empty:
        raise RuntimeError("No cleaned analysis units could be built from the selected validation details file.")

    summary_by_bucket = _build_summary_by_dynamic_bucket(clean_df)
    model_drop = _build_model_drop_table(clean_df)
    model_summary = _build_summary_by_model(clean_df)
    dataset_summary = _build_summary_by_dataset(clean_df)
    failures = _build_top_failure_cases(clean_df)
    heatmap_data = _build_heatmap_data(clean_df, dataset_summary)

    _write_csv(clean_df, DATA_DIR / "cleaned_results.csv")
    _write_csv(model_summary, DATA_DIR / "summary_by_model.csv")
    _write_csv(summary_by_bucket.drop(columns=["bucket_rank"], errors="ignore"), DATA_DIR / "summary_by_dynamic_bucket.csv")
    _write_csv(dataset_summary, DATA_DIR / "summary_by_dataset.csv")
    _write_csv(model_drop, DATA_DIR / "model_drop_table.csv")
    _write_csv(failures, DATA_DIR / "top_failure_cases.csv")
    _write_csv(heatmap_data, DATA_DIR / "heatmap_data.csv")

    standard_tradeoff_df = _build_standard_tradeoff_data(model_summary)
    standard_prefix_df = _build_standard_prefix_plot_data(clean_df)
    regime_tercile_groups_df = _build_regime_tercile_dataset_groups(dataset_summary)
    regime_tercile_matrix_df = _build_regime_tercile_model_matrix(clean_df, regime_tercile_groups_df)
    model_subitem_heatmap_df = _build_cardinality_model_subitem_heatmap_df(model_summary)
    _write_csv(standard_tradeoff_df, DATA_DIR / "tradeoff_scatter_data.csv")
    _write_csv(standard_prefix_df, DATA_DIR / "prefix_plot_data.csv")
    _write_csv(regime_tercile_groups_df, DATA_DIR / "regime_tercile_dataset_groups.csv")
    _write_csv(regime_tercile_matrix_df, DATA_DIR / "regime_tercile_model_matrix.csv")
    _write_csv(model_subitem_heatmap_df, DATA_DIR / "model_subitem_heatmap.csv")

    plot_meta = _build_plot_csvs(summary_by_bucket, model_summary, model_drop, heatmap_data, dataset_summary)
    _build_figure_tex(model_summary, model_drop, plot_meta, dataset_summary, heatmap_data)
    _build_table_tex(model_summary, model_drop, failures)
    _write_regime_tercile_table_tex(
        regime_tercile_matrix_df,
        regime_tercile_groups_df,
        TABLE_DIR / "table4_regime_tercile_matrix.tex",
    )
    _write_standard_tradeoff_tex(standard_tradeoff_df, FIG_DIR / "cardinality_tradeoff_scatter_main.tex")
    _write_standard_prefix_tex(standard_prefix_df, FIG_DIR / "cardinality_prefix_bars_appendix.tex")
    write_model_subitem_heatmap_tex(
        model_subitem_heatmap_df,
        model_order=FIGURE_MODEL_ORDER,
        model_label_map={model: model for model in FIGURE_MODEL_ORDER},
        title="Cardinality model-subitem heatmap",
        colorbar_title="Mean score",
        path=FIG_DIR / "cardinality_model_subitem_heatmap_appendix.tex",
    )
    write_model_subitem_grouped_bar_tex(
        model_subitem_heatmap_df,
        model_order=FIGURE_MODEL_ORDER,
        model_label_map={model: model for model in FIGURE_MODEL_ORDER},
        model_color_map=MODEL_COLOR_HEX,
        title="Cardinality family and subitem bars",
        y_label="Score",
        path=FIG_DIR / "cardinality_family_subitem_bars_appendix.tex",
    )
    _plot_standard_tradeoff_preview(
        standard_tradeoff_df,
        FIG_DIR / "cardinality_tradeoff_scatter_main.pdf",
        FIG_DIR / "cardinality_tradeoff_scatter_main.png",
    )
    _plot_standard_prefix_preview(
        standard_prefix_df,
        FIG_DIR / "cardinality_prefix_bars_appendix.pdf",
        FIG_DIR / "cardinality_prefix_bars_appendix.png",
    )
    _plot_regime_tercile_heatmap_preview(
        regime_tercile_matrix_df,
        FIG_DIR / "cardinality_regime_tercile_heatmap.pdf",
        FIG_DIR / "cardinality_regime_tercile_heatmap.png",
    )
    plot_model_subitem_heatmap_preview(
        model_subitem_heatmap_df,
        model_order=FIGURE_MODEL_ORDER,
        model_label_map={model: model for model in FIGURE_MODEL_ORDER},
        title="Cardinality model-subitem heatmap",
        pdf_path=FIG_DIR / "cardinality_model_subitem_heatmap_appendix.pdf",
        png_path=FIG_DIR / "cardinality_model_subitem_heatmap_appendix.png",
    )
    plot_model_subitem_grouped_bar_preview(
        model_subitem_heatmap_df,
        model_order=FIGURE_MODEL_ORDER,
        model_label_map={model: model for model in FIGURE_MODEL_ORDER},
        model_color_map=MODEL_COLOR_HEX,
        title="Cardinality family and subitem bars",
        y_label="Score",
        pdf_path=FIG_DIR / "cardinality_family_subitem_bars_appendix.pdf",
        png_path=FIG_DIR / "cardinality_family_subitem_bars_appendix.png",
    )
    _plot_dynamic_response_curves_preview(
        plot_meta,
        FINAL_DIR / "cardinality_dynamic_response_curves_main.pdf",
        FINAL_DIR / "cardinality_dynamic_response_curves_main.png",
    )
    _sync_export_bundles(paper_dir)
    _plot_dataset_model_heatmap_preview(
        dataset_summary,
        heatmap_data,
        FINAL_DIR / "cardinality_dataset_model_heatmap_main.pdf",
        FINAL_DIR / "cardinality_dataset_model_heatmap_main.png",
    )
    _sync_standard_final_bundle()
    _build_inventory(
        primary_run=primary_run,
        paper_dir=paper_dir,
        clean_df=clean_df,
        duplicate_audit=duplicate_audit,
        chosen_rows=chosen_rows,
    )
    _build_report(
        primary_run=primary_run,
        clean_df=clean_df,
        chosen_rows=chosen_rows,
        model_summary=model_summary,
        summary_by_dataset=dataset_summary,
        model_drop=model_drop,
        failures=failures,
    )
    _build_readme(primary_run)
    _write_manifest(primary_run, clean_df, chosen_rows)

    compiled, note = _try_compile_latex(FIG_DIR / "cardinality_range_dynamic_results.tex")
    subitem_compiled, subitem_note = _try_compile_latex(FIG_DIR / "cardinality_model_subitem_heatmap_appendix.tex")
    if compiled:
        print("Compiled figures PDF successfully.")
    else:
        print(f"Figure PDF compilation skipped/failed: {note}")
    if not subitem_compiled:
        print(f"Model-subitem heatmap compilation skipped/failed: {subitem_note}")
    print(f"Analysis complete. Primary run: {primary_run.run_id}")


if __name__ == "__main__":
    main()
