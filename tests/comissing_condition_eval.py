from __future__ import annotations

import csv
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from src.eval.common import (
    SyntheticAsset,
    discover_synthetic_assets,
    list_dataset_ids,
    load_field_type_hints,
    normalize_missing,
    resolve_real_split_path,
    write_csv,
)

STATE_OTHER = "__OTHER__"
STATE_MISSING = "__Z_MISSING__"
CANONICAL_MARGINAL_AGGREGATION = "direct_mean_over_missing_targets"
CANONICAL_COMISSING_AGGREGATION = "direct_mean_over_edge_profiles"
COMPARISON_COMISSING_AGGREGATION = "weighted_by_real_relation_strength"
COMPOSITE_COMISSING_AGGREGATION = "direct_mean_over_edge_composites_0p7profile_0p3strength"
EPS = 1e-12
TOP_CATEGORIES = 8
NUMERIC_BINS = 5
MIN_MISSING_COUNT_ABS = 5
MIN_MISSING_RATE = 0.005


@dataclass(frozen=True)
class ColumnStateEncoder:
    column: str
    kind: str
    states: tuple[str, ...]
    top_categories: tuple[str, ...] = ()
    bin_edges: tuple[float, ...] = ()


@dataclass(frozen=True)
class EdgeDefinition:
    missing_target: str
    related_column: str
    encoder: ColumnStateEncoder
    real_missing_rate: float
    supported_state_indices: tuple[int, ...]
    real_state_probabilities: tuple[float, ...]
    real_conditional_missing_rates: tuple[float, ...]
    real_relation_strength: float


@dataclass(frozen=True)
class TargetDefinition:
    column: str
    missing_count: int
    missing_rate: float
    info_weight: float
    edges: tuple[EdgeDefinition, ...]


@dataclass(frozen=True)
class DatasetContext:
    dataset_id: str
    row_count: int
    columns: tuple[str, ...]
    column_kinds: dict[str, str]
    encoders: dict[str, ColumnStateEncoder]
    missing_targets: tuple[TargetDefinition, ...]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _binary_entropy(p: float) -> float:
    p = min(max(float(p), 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _load_real_df(dataset_id: str) -> pd.DataFrame:
    real_path = resolve_real_split_path(dataset_id, split="train")
    if not real_path.exists():
        raise FileNotFoundError(f"Train split missing for {dataset_id}: {real_path}")
    try:
        return pd.read_csv(real_path, dtype=str, keep_default_na=False)
    except pd.errors.ParserError:
        sample = real_path.read_text(encoding="utf-8", errors="replace")[:8192]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        return pd.read_csv(real_path, dtype=str, keep_default_na=False, sep=delimiter)


def _load_syn_df(synthetic_csv_path: Path, expected_columns: list[str]) -> pd.DataFrame:
    syn_df = pd.read_csv(synthetic_csv_path, dtype=str, keep_default_na=False)
    for column in expected_columns:
        if column not in syn_df.columns:
            syn_df[column] = ""
    syn_df = syn_df[expected_columns]
    return syn_df


def _infer_column_kind(series: pd.Series, hint: str) -> str:
    token = (hint or "").lower()
    if any(word in token for word in ["numeric", "integer", "float", "double", "decimal", "continuous"]):
        return "numeric"
    if any(word in token for word in ["categorical", "string", "text", "boolean", "ordinal"]):
        return "categorical"
    non_missing = series[~series.map(normalize_missing)]
    if non_missing.empty:
        return "categorical"
    parsed = pd.to_numeric(non_missing, errors="coerce")
    ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
    return "numeric" if ratio >= 0.95 else "categorical"


def _build_categorical_encoder(column: str, real_series: pd.Series) -> ColumnStateEncoder:
    non_missing = real_series[~real_series.map(normalize_missing)].astype(str)
    counts = non_missing.value_counts(dropna=False)
    top_categories = tuple(str(item) for item in counts.head(TOP_CATEGORIES).index.tolist())
    states = list(top_categories)
    if len(counts) > len(top_categories):
        states.append(STATE_OTHER)
    if bool(real_series.map(normalize_missing).any()):
        states.append(STATE_MISSING)
    return ColumnStateEncoder(
        column=column,
        kind="categorical",
        states=tuple(states),
        top_categories=top_categories,
    )


def _build_numeric_encoder(column: str, real_series: pd.Series) -> ColumnStateEncoder | None:
    parsed = pd.to_numeric(real_series[~real_series.map(normalize_missing)], errors="coerce").dropna()
    if len(parsed) < 8 or int(parsed.nunique()) < 4:
        return None
    quantiles = np.linspace(0.0, 1.0, NUMERIC_BINS + 1)
    edges = np.quantile(parsed.to_numpy(dtype=float), quantiles)
    edges = np.unique(edges.astype(float))
    if len(edges) < 3:
        return None
    inner_edges = tuple(float(value) for value in edges[1:-1].tolist())
    bin_count = len(inner_edges) + 1
    states = [f"bin_{idx}" for idx in range(bin_count)]
    if bool(real_series.map(normalize_missing).any()):
        states.append(STATE_MISSING)
    return ColumnStateEncoder(
        column=column,
        kind="numeric",
        states=tuple(states),
        bin_edges=inner_edges,
    )


def _build_encoder(column: str, real_series: pd.Series, hint: str) -> ColumnStateEncoder:
    inferred_kind = _infer_column_kind(real_series, hint)
    if inferred_kind == "numeric":
        numeric_encoder = _build_numeric_encoder(column, real_series)
        if numeric_encoder is not None:
            return numeric_encoder
    return _build_categorical_encoder(column, real_series)


def _encode_series(series: pd.Series, encoder: ColumnStateEncoder) -> pd.Series:
    normalized = series.fillna("").astype(str)
    if encoder.kind == "categorical":
        top = set(encoder.top_categories)

        def _map_value(value: str) -> str:
            if normalize_missing(value):
                return STATE_MISSING if STATE_MISSING in encoder.states else STATE_OTHER
            if value in top:
                return value
            return STATE_OTHER if STATE_OTHER in encoder.states else encoder.states[0]

        return normalized.map(_map_value)

    parsed = pd.to_numeric(normalized.where(~normalized.map(normalize_missing), np.nan), errors="coerce")
    bins = [-np.inf, *encoder.bin_edges, np.inf]
    labels = [state for state in encoder.states if state != STATE_MISSING]
    encoded = pd.cut(parsed, bins=bins, labels=labels, include_lowest=True).astype("object")
    if STATE_MISSING in encoder.states:
        encoded = encoded.where(~normalized.map(normalize_missing), STATE_MISSING)
    encoded = encoded.fillna(labels[0] if labels else STATE_MISSING)
    return encoded.astype(str)


def _encode_codes(series: pd.Series, encoder: ColumnStateEncoder) -> np.ndarray:
    encoded = _encode_series(series, encoder)
    return pd.Categorical(encoded, categories=list(encoder.states)).codes.astype(np.int16, copy=False)


def _state_support_counts(encoded_codes: np.ndarray, state_count: int) -> np.ndarray:
    valid = encoded_codes >= 0
    if not bool(np.any(valid)):
        return np.zeros(state_count, dtype=np.int64)
    return np.bincount(encoded_codes[valid], minlength=state_count)


def _conditional_rate_stats(missing_indicator: np.ndarray, encoded_codes: np.ndarray, state_count: int) -> tuple[np.ndarray, np.ndarray]:
    valid = encoded_codes >= 0
    if not bool(np.any(valid)):
        return np.zeros(state_count, dtype=np.int64), np.zeros(state_count, dtype=float)
    support_counts = np.bincount(encoded_codes[valid], minlength=state_count)
    missing_sums = np.bincount(encoded_codes[valid], weights=missing_indicator[valid], minlength=state_count)
    rates = np.zeros(state_count, dtype=float)
    nonzero = support_counts > 0
    rates[nonzero] = missing_sums[nonzero] / support_counts[nonzero]
    return support_counts, rates


def _relation_strength(global_missing_rate: float, state_probabilities: np.ndarray, conditional_rates: np.ndarray) -> float:
    denom = max(global_missing_rate * (1.0 - global_missing_rate), EPS)
    weighted_var = 0.0
    for weight, rate in zip(state_probabilities, conditional_rates):
        weighted_var += float(weight) * ((float(rate) - global_missing_rate) ** 2)
    return _clip01(weighted_var / denom)


def build_dataset_context(dataset_id: str) -> DatasetContext:
    real_df = _load_real_df(dataset_id)
    row_count = len(real_df)
    columns = [str(col) for col in real_df.columns]
    missing_counts = {
        col: int(real_df[col].map(normalize_missing).sum())
        for col in columns
    }
    target_defs: list[TargetDefinition] = []
    min_missing_count = max(MIN_MISSING_COUNT_ABS, int(math.ceil(row_count * MIN_MISSING_RATE)))

    active_target_columns = [
        col
        for col in columns
        if missing_counts[col] >= min_missing_count
        and 0 < missing_counts[col] < row_count
    ]
    if not active_target_columns:
        return DatasetContext(
            dataset_id=dataset_id,
            row_count=row_count,
            columns=tuple(columns),
            column_kinds={},
            encoders={},
            missing_targets=(),
        )

    hints = load_field_type_hints(dataset_id)
    column_kinds = {col: _infer_column_kind(real_df[col], hints.get(col, "")) for col in columns}
    encoders = {col: _build_encoder(col, real_df[col], hints.get(col, "")) for col in columns}
    real_encoded_cache = {
        col: _encode_codes(real_df[col], encoders[col])
        for col in columns
    }

    for target_col in active_target_columns:
        missing_indicator = real_df[target_col].map(normalize_missing).to_numpy(dtype=float)
        missing_count = missing_counts[target_col]
        missing_rate = float(missing_count / max(1, row_count))

        info_weight = _binary_entropy(missing_rate) * math.log1p(missing_count)
        edge_defs: list[EdgeDefinition] = []
        for related_col in columns:
            if related_col == target_col:
                continue
            encoder = encoders[related_col]
            encoded_real = real_encoded_cache[related_col]
            support_counts = _state_support_counts(encoded_real, len(encoder.states))
            supported_state_indices = tuple(int(idx) for idx in np.where(support_counts > 0)[0].tolist())
            if len(supported_state_indices) < 2:
                continue
            state_probabilities = support_counts.astype(float) / max(1, row_count)
            _, conditional_rates = _conditional_rate_stats(missing_indicator, encoded_real, len(encoder.states))
            strength = _relation_strength(missing_rate, state_probabilities, conditional_rates)
            edge_defs.append(
                EdgeDefinition(
                    missing_target=target_col,
                    related_column=related_col,
                    encoder=encoder,
                    real_missing_rate=missing_rate,
                    supported_state_indices=supported_state_indices,
                    real_state_probabilities=tuple(float(v) for v in state_probabilities.tolist()),
                    real_conditional_missing_rates=tuple(float(v) for v in conditional_rates.tolist()),
                    real_relation_strength=strength,
                )
            )

        if edge_defs:
            target_defs.append(
                TargetDefinition(
                    column=target_col,
                    missing_count=missing_count,
                    missing_rate=missing_rate,
                    info_weight=float(info_weight),
                    edges=tuple(edge_defs),
                )
            )

    return DatasetContext(
        dataset_id=dataset_id,
        row_count=row_count,
        columns=tuple(columns),
        column_kinds=column_kinds,
        encoders=encoders,
        missing_targets=tuple(target_defs),
    )


def _score_edge(
    target: TargetDefinition,
    edge: EdgeDefinition,
    missing_indicator: np.ndarray,
    encoded_syn: np.ndarray,
) -> tuple[float, float, float]:
    global_missing_rate = float(np.mean(missing_indicator))
    support_counts, synthetic_rates = _conditional_rate_stats(missing_indicator, encoded_syn, len(edge.encoder.states))

    profile_distance = 0.0
    synthetic_rates_fallback = synthetic_rates.copy()
    zero_support = support_counts <= 0
    synthetic_rates_fallback[zero_support] = global_missing_rate
    for idx in edge.supported_state_indices:
        real_weight = edge.real_state_probabilities[idx]
        syn_rate = synthetic_rates_fallback[idx]
        real_rate = edge.real_conditional_missing_rates[idx]
        profile_distance += float(real_weight) * abs(float(real_rate) - float(syn_rate))
    profile_score = _clip01(1.0 - profile_distance)

    denom = max(global_missing_rate * (1.0 - global_missing_rate), EPS)
    weighted_var = 0.0
    for idx in edge.supported_state_indices:
        weighted_var += float(edge.real_state_probabilities[idx]) * ((float(synthetic_rates_fallback[idx]) - global_missing_rate) ** 2)
    synthetic_strength = _clip01(weighted_var / denom)
    strength_score = _clip01(1.0 - abs(edge.real_relation_strength - synthetic_strength))

    edge_score = _clip01((0.7 * profile_score) + (0.3 * strength_score))
    return edge_score, profile_score, strength_score


def score_synthetic_df(context: DatasetContext, syn_df: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not context.missing_targets:
        return (
            {
                "status": "not_applicable_no_missing_targets",
                "marginal_aggregation_scheme": CANONICAL_MARGINAL_AGGREGATION,
                "canonical_aggregation_scheme": CANONICAL_COMISSING_AGGREGATION,
                "comparison_aggregation_scheme": COMPARISON_COMISSING_AGGREGATION,
                "marginal_missing_rate_consistency": None,
                "co_missingness_pattern_consistency": None,
                "missingness_structure_score": None,
                "comparison_missingness_structure_score": None,
                "canonical_score": None,
                "direct_mean_score": None,
                "weighted_score": None,
                "missing_target_count": 0,
                "edge_count": 0,
            },
            [],
        )

    target_rows: list[dict[str, Any]] = []
    marginal_target_scores: list[float] = []
    all_edge_scores: list[float] = []
    all_profile_scores: list[float] = []
    all_strength_scores: list[float] = []
    weighted_target_scores: list[tuple[float, float]] = []
    encoded_cache = {
        column: _encode_codes(syn_df[column], encoder)
        for column, encoder in context.encoders.items()
    }
    missing_indicator_cache = {
        target.column: syn_df[target.column].map(normalize_missing).to_numpy(dtype=float)
        for target in context.missing_targets
    }

    for target in context.missing_targets:
        missing_indicator = missing_indicator_cache[target.column]
        synthetic_missing_rate = float(np.mean(missing_indicator))
        marginal_target_score = _clip01(1.0 - abs(float(target.missing_rate) - synthetic_missing_rate))
        edge_scores: list[float] = []
        edge_weights: list[float] = []
        mean_profile_scores: list[float] = []
        mean_strength_scores: list[float] = []
        informative_edge_count = 0

        for edge in target.edges:
            edge_score, profile_score, strength_score = _score_edge(
                target,
                edge,
                missing_indicator,
                encoded_cache[edge.related_column],
            )
            edge_scores.append(edge_score)
            edge_weights.append(edge.real_relation_strength)
            mean_profile_scores.append(profile_score)
            mean_strength_scores.append(strength_score)
            all_edge_scores.append(edge_score)
            all_profile_scores.append(profile_score)
            all_strength_scores.append(strength_score)
            if edge.real_relation_strength > 0:
                informative_edge_count += 1

        if not edge_scores:
            continue

        marginal_target_scores.append(marginal_target_score)
        direct_target_score = float(mean(mean_profile_scores))
        strength_target_score = float(mean(mean_strength_scores))
        composite_target_score = float(mean(edge_scores))
        total_weight = float(sum(edge_weights))
        if total_weight > 0:
            weighted_target_score = float(sum(score * weight for score, weight in zip(edge_scores, edge_weights)) / total_weight)
        else:
            weighted_target_score = composite_target_score

        weighted_target_scores.append((weighted_target_score, target.info_weight))
        target_rows.append(
            {
                "missing_target": target.column,
                "missing_count_real": target.missing_count,
                "missing_rate_real": round(target.missing_rate, 6),
                "missing_rate_synthetic": round(synthetic_missing_rate, 6),
                "marginal_target_score": round(marginal_target_score, 6),
                "target_info_weight": round(target.info_weight, 6),
                "edge_count": len(edge_scores),
                "informative_edge_count": informative_edge_count,
                "co_missing_direct_target_score": round(direct_target_score, 6),
                "co_missing_profile_target_score": round(direct_target_score, 6),
                "co_missing_strength_target_score": round(strength_target_score, 6),
                "co_missing_composite_target_score": round(composite_target_score, 6),
                "co_missing_weighted_target_score": round(weighted_target_score, 6),
                "missingness_structure_target_score": round(float(mean([marginal_target_score, direct_target_score])), 6),
                "mean_profile_score": round(float(mean(mean_profile_scores)), 6),
                "mean_strength_score": round(float(mean(mean_strength_scores)), 6),
            }
        )

    if not all_profile_scores or not target_rows or not marginal_target_scores:
        return (
            {
                "status": "not_applicable_no_edges",
                "marginal_aggregation_scheme": CANONICAL_MARGINAL_AGGREGATION,
                "canonical_aggregation_scheme": CANONICAL_COMISSING_AGGREGATION,
                "composite_aggregation_scheme": COMPOSITE_COMISSING_AGGREGATION,
                "comparison_aggregation_scheme": COMPARISON_COMISSING_AGGREGATION,
                "marginal_missing_rate_consistency": None,
                "co_missingness_pattern_consistency": None,
                "co_missing_strength_score": None,
                "co_missing_composite_score": None,
                "missingness_structure_score": None,
                "comparison_missingness_structure_score": None,
                "canonical_score": None,
                "direct_mean_score": None,
                "weighted_score": None,
                "missing_target_count": len(context.missing_targets),
                "edge_count": 0,
            },
            target_rows,
        )

    marginal_missing_rate_consistency = float(mean(marginal_target_scores))
    direct_mean_score = float(mean(all_profile_scores))
    strength_score = float(mean(all_strength_scores))
    composite_score = float(mean(all_edge_scores))
    weight_sum = float(sum(weight for _, weight in weighted_target_scores))
    if weight_sum > 0:
        weighted_score = float(sum(score * weight for score, weight in weighted_target_scores) / weight_sum)
    else:
        weighted_score = float(mean(score for score, _ in weighted_target_scores))
    missingness_structure_score = float(mean([marginal_missing_rate_consistency, direct_mean_score]))
    comparison_missingness_structure_score = float(mean([marginal_missing_rate_consistency, weighted_score]))

    return (
        {
            "status": "ok",
            "marginal_aggregation_scheme": CANONICAL_MARGINAL_AGGREGATION,
            "canonical_aggregation_scheme": CANONICAL_COMISSING_AGGREGATION,
            "composite_aggregation_scheme": COMPOSITE_COMISSING_AGGREGATION,
            "comparison_aggregation_scheme": COMPARISON_COMISSING_AGGREGATION,
            "marginal_missing_rate_consistency": round(marginal_missing_rate_consistency, 6),
            "co_missingness_pattern_consistency": round(direct_mean_score, 6),
            "co_missing_strength_score": round(strength_score, 6),
            "co_missing_composite_score": round(composite_score, 6),
            "missingness_structure_score": round(missingness_structure_score, 6),
            "comparison_missingness_structure_score": round(comparison_missingness_structure_score, 6),
            "canonical_score": round(direct_mean_score, 6),
            "direct_mean_score": round(direct_mean_score, 6),
            "weighted_score": round(weighted_score, 6),
            "missing_target_count": len(target_rows),
            "edge_count": len(all_edge_scores),
            "score_gap_weighted_minus_direct": round(weighted_score - direct_mean_score, 6),
        },
        target_rows,
    )


def _dataset_context_rows(context: DatasetContext) -> dict[str, Any]:
    return {
        "dataset_id": context.dataset_id,
        "row_count": context.row_count,
        "column_count": len(context.columns),
        "missing_target_count": len(context.missing_targets),
        "edge_count": sum(len(target.edges) for target in context.missing_targets),
        "missing_targets": ",".join(target.column for target in context.missing_targets),
    }


def _evaluate_dataset_assets(dataset_id: str, dataset_assets: list[SyntheticAsset]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    context = build_dataset_context(dataset_id)
    context_row = _dataset_context_rows(context)
    asset_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for asset in dataset_assets:
        syn_df = _load_syn_df(Path(asset.synthetic_csv_path), list(context.columns))
        score_row, per_target_rows = score_synthetic_df(context, syn_df)
        asset_row = {
            **asset.to_dict(),
            "dataset_id": dataset_id,
            **score_row,
        }
        asset_rows.append(asset_row)
        for target_row in per_target_rows:
            target_rows.append(
                {
                    **asset.to_dict(),
                    "dataset_id": dataset_id,
                    "status": score_row.get("status"),
                    **target_row,
                }
            )

    return context_row, asset_rows, target_rows


def _mean_or_none(values: list[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(mean(cleaned))


def _summarize_asset_rows(asset_rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in asset_rows:
        grouped[tuple(str(row.get(key) or "") for key in group_keys)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        payload = {field: value for field, value in zip(group_keys, key)}
        payload["asset_count"] = len(rows)
        payload["applicable_asset_count"] = sum(1 for row in rows if row.get("status") == "ok")
        payload["marginal_aggregation_scheme"] = CANONICAL_MARGINAL_AGGREGATION
        payload["canonical_aggregation_scheme"] = CANONICAL_COMISSING_AGGREGATION
        payload["composite_aggregation_scheme"] = COMPOSITE_COMISSING_AGGREGATION
        payload["comparison_aggregation_scheme"] = COMPARISON_COMISSING_AGGREGATION
        payload["marginal_missing_rate_consistency"] = _mean_or_none(
            [row.get("marginal_missing_rate_consistency") for row in rows if row.get("status") == "ok"]
        )
        payload["co_missingness_pattern_consistency"] = _mean_or_none(
            [row.get("co_missingness_pattern_consistency") for row in rows if row.get("status") == "ok"]
        )
        payload["co_missing_strength_score"] = _mean_or_none(
            [row.get("co_missing_strength_score") for row in rows if row.get("status") == "ok"]
        )
        payload["co_missing_composite_score"] = _mean_or_none(
            [row.get("co_missing_composite_score") for row in rows if row.get("status") == "ok"]
        )
        payload["missingness_structure_score"] = _mean_or_none(
            [row.get("missingness_structure_score") for row in rows if row.get("status") == "ok"]
        )
        payload["comparison_missingness_structure_score"] = _mean_or_none(
            [row.get("comparison_missingness_structure_score") for row in rows if row.get("status") == "ok"]
        )
        payload["canonical_score"] = _mean_or_none([row.get("canonical_score") for row in rows if row.get("status") == "ok"])
        payload["direct_mean_score"] = _mean_or_none([row.get("direct_mean_score") for row in rows if row.get("status") == "ok"])
        payload["weighted_score"] = _mean_or_none([row.get("weighted_score") for row in rows if row.get("status") == "ok"])
        payload["score_gap_weighted_minus_direct"] = _mean_or_none(
            [row.get("score_gap_weighted_minus_direct") for row in rows if row.get("status") == "ok"]
        )
        for field in (
            "marginal_missing_rate_consistency",
            "co_missingness_pattern_consistency",
            "co_missing_strength_score",
            "co_missing_composite_score",
            "missingness_structure_score",
            "comparison_missingness_structure_score",
            "canonical_score",
            "direct_mean_score",
            "weighted_score",
            "score_gap_weighted_minus_direct",
        ):
            if payload[field] is not None:
                payload[field] = round(float(payload[field]), 6)
        summary_rows.append(payload)
    return summary_rows


def evaluate_all_synthetic_assets(output_dir: Path, max_workers: int | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_ids = list_dataset_ids()
    assets = discover_synthetic_assets(datasets=dataset_ids, latest_only=True)
    dataset_asset_map: dict[str, list[SyntheticAsset]] = defaultdict(list)
    for asset in assets:
        dataset_asset_map[asset.dataset_id].append(asset)

    dataset_context_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    worker_count = max_workers if max_workers is not None else min(8, max(1, (cpu_count() or 4) - 1))
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        for dataset_id in dataset_ids:
            futures[executor.submit(_evaluate_dataset_assets, dataset_id, dataset_asset_map.get(dataset_id, []))] = dataset_id
        for index, future in enumerate(as_completed(futures), start=1):
            dataset_id = futures[future]
            context_row, dataset_asset_rows, dataset_target_rows = future.result()
            dataset_context_rows.append(context_row)
            asset_rows.extend(dataset_asset_rows)
            target_rows.extend(dataset_target_rows)
            print(
                f"[co-missing] dataset={index}/{len(dataset_ids)}"
                f" id={dataset_id}"
                f" assets={len(dataset_asset_rows)}"
                f" missing_targets={context_row.get('missing_target_count')}",
                flush=True,
            )

    model_dataset_rows = _summarize_asset_rows(asset_rows, ("dataset_id", "model_id"))
    model_overall_rows = _summarize_asset_rows(asset_rows, ("model_id",))

    dataset_context_path = output_dir / "co_missing_dataset_context.csv"
    asset_scores_path = output_dir / "co_missing_asset_scores.csv"
    target_scores_path = output_dir / "co_missing_target_scores.csv"
    model_dataset_path = output_dir / "co_missing_model_dataset_summary.csv"
    model_overall_path = output_dir / "co_missing_model_overall_summary.csv"

    write_csv(dataset_context_path, dataset_context_rows)
    write_csv(asset_scores_path, asset_rows)
    write_csv(target_scores_path, target_rows)
    write_csv(model_dataset_path, model_dataset_rows)
    write_csv(model_overall_path, model_overall_rows)

    return {
        "dataset_context": dataset_context_path,
        "asset_scores": asset_scores_path,
        "target_scores": target_scores_path,
        "model_dataset_summary": model_dataset_path,
        "model_overall_summary": model_overall_path,
    }
