"""Tail-threshold v2 diagnostics with numeric-tail quantile ranges.

This implementation preserves the legacy categorical-tail logic while
introducing a true quantile-range view for numerical columns:

- categorical coverage: Jaccard over rare-support token sets
- categorical size: legacy mass-similarity on real tail states
- numerical coverage: interval-overlap consistency between real and synthetic
  low/high tail ranges
- numerical size: synthetic mass captured beyond real low/high cutoffs

The legacy concentration component is intentionally removed.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tqb_scoring.eval.common import (
    SyntheticAsset,
    TaskProgressTracker,
    discover_synthetic_assets,
    list_dataset_ids,
    make_task_run_dir,
    now_run_tag,
    resolve_real_split_path,
    sniff_csv_delimiter,
    write_csv,
    write_json,
)
from tqb_scoring.eval.tail_threshold.runner import MODEL_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = PROJECT_ROOT.parent / "results"
TAIL_THRESHOLD_ROOT = EVALUATION_ROOT / "tail_threshold_v2"

DEFAULT_THRESHOLD_PCTS = [10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1]
DEFAULT_NUMERIC_BINS = 10
DEFAULT_MAX_WORKERS = 4


@dataclass(frozen=True)
class ThresholdSpec:
    index: int
    pct: float
    ratio: float
    label: str


def _threshold_specs(percentages: list[float] | None = None) -> list[ThresholdSpec]:
    values = percentages or DEFAULT_THRESHOLD_PCTS
    out: list[ThresholdSpec] = []
    for idx, pct in enumerate(values):
        value = float(pct)
        out.append(ThresholdSpec(index=idx, pct=value, ratio=value / 100.0, label=f"{value:g}%"))
    return out


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null", "none"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 6)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "nan", "null", "none", "na", "n/a"}


def _safe_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _is_id_like(name: str) -> bool:
    text = str(name).strip().lower()
    return text in {"id", "row_id", "index"} or text.endswith("_id")


def _normalize_model_id(model_id: str) -> str:
    key = str(model_id or "").strip().lower()
    if key == "rtf":
        return "realtabformer"
    return key


def _model_label(model_id: str) -> str:
    key = _normalize_model_id(model_id)
    return MODEL_LABELS.get(key, key or "unknown")


def _dataset_prefix(dataset_id: str) -> str:
    return str(dataset_id or "").strip().lower()[:1]


def _asset_payload(asset: SyntheticAsset) -> dict[str, Any]:
    payload = asset.to_dict()
    raw_model_id = str(payload.get("model_id") or "")
    payload["model_id_raw"] = raw_model_id
    payload["model_id"] = _normalize_model_id(raw_model_id)
    payload["model_label"] = _model_label(payload["model_id"])
    return payload


def _sniff_delimiter(path: Path) -> str:
    return sniff_csv_delimiter(path)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    delimiter = _sniff_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
    # Some quoted comma CSVs with long text fields can be mis-sniffed as pipe.
    # If there are commas in the header line but no pipes, prefer comma parsing.
    if delimiter == "|" and "," in first_line and "|" not in first_line:
        delimiter = ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]
        columns = [str(col) for col in (reader.fieldnames or [])]
    return columns, rows


def _load_target_column(dataset_id: str, columns: list[str]) -> str:
    semantics_path = PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / dataset_id / "metadata" / "dataset_semantics.yaml"
    if semantics_path.exists():
        for raw in semantics_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("target_column:"):
                target = line.split(":", 1)[1].strip()
                if target in columns:
                    return target
    priors = ["class", "target", "label", "y", "outcome"]
    lower_map = {col.lower(): col for col in columns}
    for prior in priors:
        if prior in lower_map:
            return lower_map[prior]
    return columns[-1]


def _quantile_edges(values: list[float], bins: int) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(arr, quantiles).tolist()
    deduped: list[float] = []
    for value in edges:
        current = float(value)
        if not deduped or abs(current - deduped[-1]) > 1e-12:
            deduped.append(current)
    return deduped


def _build_transformers(
    rows_real: list[dict[str, str]],
    feature_columns: list[str],
    numeric_bins: int,
) -> dict[str, dict[str, Any]]:
    transformers: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        raw_values = [row.get(column) for row in rows_real]
        total = max(1, len(raw_values))
        numeric_values = [value for value in (_safe_float(item) for item in raw_values) if value is not None]
        numeric_ratio = len(numeric_values) / total
        unique_numeric = len({round(value, 8) for value in numeric_values})
        is_continuous_numeric = numeric_ratio >= 0.95 and unique_numeric >= 20
        if is_continuous_numeric:
            transformers[column] = {"mode": "numeric", "edges": _quantile_edges(numeric_values, bins=numeric_bins)}
        else:
            transformers[column] = {"mode": "categorical"}
    return transformers


def _tokenize_categorical(value: Any) -> str:
    if _is_missing(value):
        return "__MISSING__"
    return str(value).strip()


def _sorted_support_items(counter: Counter[str], *, reverse: bool) -> list[tuple[str, int]]:
    if reverse:
        return sorted(((key, int(value)) for key, value in counter.items() if int(value) > 0), key=lambda item: (-item[1], item[0]))
    return sorted(((key, int(value)) for key, value in counter.items() if int(value) > 0), key=lambda item: (item[1], item[0]))


def _select_bottom_band(items: list[tuple[str, int]], ratio: float) -> tuple[set[str], int]:
    if not items:
        return set(), 0
    keep_n = max(1, int(math.ceil(len(items) * max(0.0, float(ratio)))))
    selected = items[:keep_n]
    gate = int(selected[-1][1]) if selected else 0
    return {key for key, _ in selected}, gate


def _categorical_metrics(real_tokens: list[str], syn_tokens: list[str], ratio: float) -> dict[str, Any]:
    real_counts = Counter(real_tokens)
    syn_counts = Counter(syn_tokens)
    real_items = _sorted_support_items(real_counts, reverse=False)
    syn_items = _sorted_support_items(syn_counts, reverse=False)
    real_keys, real_gate = _select_bottom_band(real_items, ratio)
    syn_keys, syn_gate = _select_bottom_band(syn_items, ratio)
    union_keys = real_keys | syn_keys
    inter_keys = real_keys & syn_keys
    coverage = (len(inter_keys) / len(union_keys)) if union_keys else 1.0
    mass_real = (sum(real_counts.get(key, 0) for key in real_keys) / max(1, len(real_tokens))) if real_keys else 0.0
    mass_syn_on_real = (sum(syn_counts.get(key, 0) for key in real_keys) / max(1, len(syn_tokens))) if real_keys else 0.0
    if mass_real <= 1e-12:
        size = 1.0 if mass_syn_on_real <= 1e-12 else 0.0
    else:
        size = 1.0 - abs(mass_syn_on_real - mass_real) / mass_real
        size = max(0.0, min(1.0, size))
    return {
        "coverage": float(coverage),
        "size": float(size),
        "real_tail_token_count": len(real_keys),
        "syn_tail_token_count": len(syn_keys),
        "effective_gate_real": real_gate,
        "effective_gate_syn": syn_gate,
        "real_tail_mass": float(mass_real),
        "syn_tail_mass_on_real": float(mass_syn_on_real),
    }


def _interval_overlap_score(a0: float, a1: float, b0: float, b1: float) -> float:
    left = max(min(a0, a1), min(b0, b1))
    right = min(max(a0, a1), max(b0, b1))
    overlap = max(0.0, right - left)
    union_left = min(a0, a1, b0, b1)
    union_right = max(a0, a1, b0, b1)
    union = max(0.0, union_right - union_left)
    if union <= 1e-12:
        return 1.0 if abs(a0 - b0) <= 1e-12 and abs(a1 - b1) <= 1e-12 else 0.0
    return max(0.0, min(1.0, overlap / union))


def _numeric_metrics(real_values: list[float], syn_values: list[float], ratio: float) -> dict[str, Any]:
    real_arr = np.asarray(real_values, dtype=float)
    syn_arr = np.asarray(syn_values, dtype=float)
    q_real_low = float(np.quantile(real_arr, ratio))
    q_real_high = float(np.quantile(real_arr, 1.0 - ratio))
    q_syn_low = float(np.quantile(syn_arr, ratio))
    q_syn_high = float(np.quantile(syn_arr, 1.0 - ratio))
    real_min = float(np.min(real_arr))
    real_max = float(np.max(real_arr))
    syn_min = float(np.min(syn_arr))
    syn_max = float(np.max(syn_arr))

    coverage_low = _interval_overlap_score(real_min, q_real_low, syn_min, q_syn_low)
    coverage_high = _interval_overlap_score(q_real_high, real_max, q_syn_high, syn_max)
    coverage = 0.5 * (coverage_low + coverage_high)

    syn_low_mass = float(np.mean(syn_arr <= q_real_low))
    syn_high_mass = float(np.mean(syn_arr >= q_real_high))
    size_low = min(syn_low_mass / ratio, 1.0) if ratio > 0 else 1.0
    size_high = min(syn_high_mass / ratio, 1.0) if ratio > 0 else 1.0
    size = 0.5 * (size_low + size_high)

    return {
        "coverage": float(max(0.0, min(1.0, coverage))),
        "size": float(max(0.0, min(1.0, size))),
        "coverage_low": float(coverage_low),
        "coverage_high": float(coverage_high),
        "size_low": float(size_low),
        "size_high": float(size_high),
        "real_low_cutoff": q_real_low,
        "real_high_cutoff": q_real_high,
        "syn_low_cutoff": q_syn_low,
        "syn_high_cutoff": q_syn_high,
        "real_min": real_min,
        "real_max": real_max,
        "syn_min": syn_min,
        "syn_max": syn_max,
        "syn_low_mass_at_real_cutoff": syn_low_mass,
        "syn_high_mass_at_real_cutoff": syn_high_mass,
    }


def _run_dataset_threshold_sweep(
    dataset_id: str,
    dataset_assets: list[SyntheticAsset],
    threshold_specs: list[ThresholdSpec],
    numeric_bins: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    real_csv = resolve_real_split_path(dataset_id, split="train")
    if not real_csv.exists():
        return dataset_id, [], [], [], {"dataset_id": dataset_id, "status": "missing_real_csv", "asset_count": len(dataset_assets)}

    columns, rows_real = _read_csv_rows(real_csv)
    if not columns or not rows_real:
        return dataset_id, [], [], [], {"dataset_id": dataset_id, "status": "empty_real_csv", "asset_count": len(dataset_assets)}

    target_column = _load_target_column(dataset_id, columns)
    feature_columns = [column for column in columns if column != target_column and not _is_id_like(column)]
    if not feature_columns:
        return dataset_id, [], [], [], {"dataset_id": dataset_id, "status": "no_feature_columns", "asset_count": len(dataset_assets)}

    transformers = _build_transformers(rows_real, feature_columns, numeric_bins=numeric_bins)
    n_real = len(rows_real)

    real_column_cache: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        mode = str(transformers[column].get("mode") or "categorical")
        if mode == "numeric":
            values = [v for v in (_safe_float(row.get(column)) for row in rows_real) if v is not None]
            real_column_cache[column] = {"mode": "numeric", "values": values}
        else:
            real_column_cache[column] = {
                "mode": "categorical",
                "tokens": [_tokenize_categorical(row.get(column)) for row in rows_real],
            }

    asset_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    real_diagnostic_rows: list[dict[str, Any]] = []

    for asset in dataset_assets:
        asset_payload = _asset_payload(asset)
        _, rows_syn = _read_csv_rows(Path(asset.synthetic_csv_path))
        n_syn = len(rows_syn)

        syn_column_cache: dict[str, dict[str, Any]] = {}
        for column in feature_columns:
            mode = str(transformers[column].get("mode") or "categorical")
            if mode == "numeric":
                values = [v for v in (_safe_float(row.get(column)) for row in rows_syn) if v is not None]
                syn_column_cache[column] = {"mode": "numeric", "values": values}
            else:
                syn_column_cache[column] = {
                    "mode": "categorical",
                    "tokens": [_tokenize_categorical(row.get(column)) for row in rows_syn],
                }

        for spec in threshold_specs:
            coverage_values: list[float] = []
            size_values: list[float] = []
            cat_coverage_values: list[float] = []
            cat_size_values: list[float] = []
            num_coverage_values: list[float] = []
            num_size_values: list[float] = []
            active_cat = 0
            active_num = 0

            for column in feature_columns:
                real_meta = real_column_cache[column]
                syn_meta = syn_column_cache[column]
                mode = str(real_meta.get("mode") or "categorical")
                if mode == "numeric":
                    real_values = list(real_meta.get("values") or [])
                    syn_values = list(syn_meta.get("values") or [])
                    if len(real_values) < 2 or len(syn_values) < 2:
                        continue
                    metrics = _numeric_metrics(real_values, syn_values, spec.ratio)
                    active_num += 1
                    num_coverage_values.append(metrics["coverage"])
                    num_size_values.append(metrics["size"])
                else:
                    real_tokens = list(real_meta.get("tokens") or [])
                    syn_tokens = list(syn_meta.get("tokens") or [])
                    if not real_tokens or not syn_tokens:
                        continue
                    metrics = _categorical_metrics(real_tokens, syn_tokens, spec.ratio)
                    active_cat += 1
                    cat_coverage_values.append(metrics["coverage"])
                    cat_size_values.append(metrics["size"])

                coverage_values.append(metrics["coverage"])
                size_values.append(metrics["size"])
                column_rows.append(
                    {
                        **asset_payload,
                        "dataset_id": dataset_id,
                        "dataset_prefix": _dataset_prefix(dataset_id),
                        "threshold_label": spec.label,
                        "threshold_pct": spec.pct,
                        "tail_ratio": spec.ratio,
                        "column_name": column,
                        "column_mode": mode,
                        "coverage_score": round(float(metrics["coverage"]), 6),
                        "size_score": round(float(metrics["size"]), 6),
                        **{key: (round(float(value), 6) if isinstance(value, float) else value) for key, value in metrics.items()},
                    }
                )

            tail_coverage = _mean(coverage_values)
            tail_size = _mean(size_values)
            tail_overall = _mean([tail_coverage, tail_size])
            asset_rows.append(
                {
                    **asset_payload,
                    "dataset_id": dataset_id,
                    "dataset_prefix": _dataset_prefix(dataset_id),
                    "threshold_label": spec.label,
                    "threshold_pct": spec.pct,
                    "tail_ratio": spec.ratio,
                    "real_row_count": n_real,
                    "synthetic_row_count": n_syn,
                    "feature_column_count": len(feature_columns),
                    "active_categorical_columns": active_cat,
                    "active_numeric_columns": active_num,
                    "tail_coverage_score": tail_coverage,
                    "tail_size_score": tail_size,
                    "tail_overall_score": tail_overall,
                    "categorical_coverage_score": _mean(cat_coverage_values),
                    "categorical_size_score": _mean(cat_size_values),
                    "numerical_coverage_score": _mean(num_coverage_values),
                    "numerical_size_score": _mean(num_size_values),
                }
            )

    for spec in threshold_specs:
        items = [row for row in asset_rows if row.get("threshold_label") == spec.label]
        real_diagnostic_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": _dataset_prefix(dataset_id),
                "threshold_label": spec.label,
                "threshold_pct": spec.pct,
                "tail_ratio": spec.ratio,
                "real_row_count": n_real,
                "feature_column_count": len(feature_columns),
                "asset_count": len(items),
                "tail_overall_mean": _mean([_to_float(row.get("tail_overall_score")) for row in items]),
                "tail_coverage_mean": _mean([_to_float(row.get("tail_coverage_score")) for row in items]),
                "tail_size_mean": _mean([_to_float(row.get("tail_size_score")) for row in items]),
                "categorical_coverage_mean": _mean([_to_float(row.get("categorical_coverage_score")) for row in items]),
                "categorical_size_mean": _mean([_to_float(row.get("categorical_size_score")) for row in items]),
                "numerical_coverage_mean": _mean([_to_float(row.get("numerical_coverage_score")) for row in items]),
                "numerical_size_mean": _mean([_to_float(row.get("numerical_size_score")) for row in items]),
            }
        )

    manifest_row = {
        "dataset_id": dataset_id,
        "status": "ok",
        "asset_count": len(dataset_assets),
        "real_row_count": n_real,
        "feature_column_count": len(feature_columns),
    }
    return dataset_id, asset_rows, column_rows, real_diagnostic_rows, manifest_row


def _aggregate_group_mean(
    rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
    value_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_tuple, items in sorted(grouped.items()):
        payload = {group_key: key_tuple[idx] for idx, group_key in enumerate(group_keys)}
        for field in value_fields:
            payload[field] = _mean([_to_float(item.get(field)) for item in items])
        payload["asset_count"] = len(items)
        out.append(payload)
    return out


def _build_global_threshold_summary(asset_rows: list[dict[str, Any]], threshold_specs: list[ThresholdSpec]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in threshold_specs:
        items = [row for row in asset_rows if row.get("threshold_label") == spec.label]
        if not items:
            continue
        out.append(
            {
                "threshold_label": spec.label,
                "threshold_pct": spec.pct,
                "tail_ratio": spec.ratio,
                "tail_overall_mean": _mean([_to_float(row.get("tail_overall_score")) for row in items]),
                "tail_coverage_mean": _mean([_to_float(row.get("tail_coverage_score")) for row in items]),
                "tail_size_mean": _mean([_to_float(row.get("tail_size_score")) for row in items]),
                "categorical_coverage_mean": _mean([_to_float(row.get("categorical_coverage_score")) for row in items]),
                "categorical_size_mean": _mean([_to_float(row.get("categorical_size_score")) for row in items]),
                "numerical_coverage_mean": _mean([_to_float(row.get("numerical_coverage_score")) for row in items]),
                "numerical_size_mean": _mean([_to_float(row.get("numerical_size_score")) for row in items]),
                "asset_count": len(items),
            }
        )
    return out


def run_tail_threshold_experiment_v2(
    *,
    run_tag: str | None = None,
    datasets: list[str] | None = None,
    latest_only: bool = True,
    root_names: list[str] | None = None,
    threshold_percentages: list[float] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    numeric_bins: int = DEFAULT_NUMERIC_BINS,
) -> dict[str, Any]:
    dataset_ids = datasets or list_dataset_ids()
    threshold_specs = _threshold_specs(threshold_percentages)
    resolved_run_tag = run_tag or f"{now_run_tag()}_tail_threshold_v2"
    run_dir = make_task_run_dir("tail_threshold_v2", resolved_run_tag)
    data_dir = run_dir / "data"
    datasets_dir = run_dir / "datasets"
    summaries_dir = run_dir / "summaries"

    assets = discover_synthetic_assets(datasets=dataset_ids, latest_only=latest_only, root_names=root_names)
    by_dataset: dict[str, list[SyntheticAsset]] = defaultdict(list)
    for asset in assets:
        by_dataset[asset.dataset_id].append(asset)

    tracker = TaskProgressTracker(
        task_name="tail_threshold_v2",
        total_steps=len(dataset_ids),
        step_label="datasets",
        substep_label="assets",
        total_substeps=len(assets),
    )
    tracker.print_start(extra=f"run_tag={resolved_run_tag}")

    asset_rows_all: list[dict[str, Any]] = []
    column_rows_all: list[dict[str, Any]] = []
    real_diagnostic_rows_all: list[dict[str, Any]] = []
    dataset_manifest_rows: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        future_map = {
            pool.submit(
                _run_dataset_threshold_sweep,
                dataset_id,
                by_dataset.get(dataset_id, []),
                threshold_specs,
                numeric_bins,
            ): dataset_id
            for dataset_id in dataset_ids
        }
        for future in as_completed(future_map):
            dataset_id = future_map[future]
            asset_rows, column_rows, real_rows, manifest_row = [], [], [], {}
            try:
                _, asset_rows, column_rows, real_rows, manifest_row = future.result()
            except Exception as exc:
                manifest_row = {"dataset_id": dataset_id, "status": "error", "error": repr(exc), "asset_count": len(by_dataset.get(dataset_id, []))}

            asset_rows_all.extend(asset_rows)
            column_rows_all.extend(column_rows)
            real_diagnostic_rows_all.extend(real_rows)
            dataset_manifest_rows.append(manifest_row)

            dataset_dir = datasets_dir / dataset_id
            if asset_rows:
                write_csv(dataset_dir / f"tail_threshold_v2_asset_scores__{dataset_id}.csv", asset_rows)
            if column_rows:
                write_csv(dataset_dir / f"tail_threshold_v2_column_scores__{dataset_id}.csv", column_rows)
            if real_rows:
                write_csv(dataset_dir / f"tail_threshold_v2_dataset_summary__{dataset_id}.csv", real_rows)
            write_json(dataset_dir / "manifest.json", manifest_row)

            tracker.advance(
                step_name=dataset_id,
                substeps_done=int(manifest_row.get("asset_count") or 0),
                extra=f"status={manifest_row.get('status', 'ok')}",
            )

    asset_rows_all.sort(key=lambda row: (str(row.get("dataset_id") or ""), str(row.get("model_id") or ""), float(row.get("threshold_pct") or 0.0)))
    column_rows_all.sort(key=lambda row: (str(row.get("dataset_id") or ""), str(row.get("model_id") or ""), str(row.get("column_name") or ""), float(row.get("threshold_pct") or 0.0)))
    real_diagnostic_rows_all.sort(key=lambda row: (str(row.get("dataset_id") or ""), float(row.get("threshold_pct") or 0.0)))
    dataset_manifest_rows.sort(key=lambda row: str(row.get("dataset_id") or ""))

    global_summary_rows = _build_global_threshold_summary(asset_rows_all, threshold_specs)
    model_summary_rows = _aggregate_group_mean(
        asset_rows_all,
        group_keys=["model_id", "model_label", "threshold_label", "threshold_pct"],
        value_fields=[
            "tail_overall_score",
            "tail_coverage_score",
            "tail_size_score",
            "categorical_coverage_score",
            "categorical_size_score",
            "numerical_coverage_score",
            "numerical_size_score",
        ],
    )
    dataset_summary_rows = _aggregate_group_mean(
        asset_rows_all,
        group_keys=["dataset_id", "dataset_prefix", "threshold_label", "threshold_pct"],
        value_fields=[
            "tail_overall_score",
            "tail_coverage_score",
            "tail_size_score",
            "categorical_coverage_score",
            "categorical_size_score",
            "numerical_coverage_score",
            "numerical_size_score",
        ],
    )

    write_csv(data_dir / "tail_threshold_v2_asset_scores.csv", asset_rows_all)
    write_csv(data_dir / "tail_threshold_v2_column_scores.csv", column_rows_all)
    write_csv(data_dir / "tail_threshold_v2_dataset_diagnostics.csv", real_diagnostic_rows_all)
    write_csv(summaries_dir / "tail_threshold_v2_global_summary.csv", global_summary_rows)
    write_csv(summaries_dir / "tail_threshold_v2_model_summary.csv", model_summary_rows)
    write_csv(summaries_dir / "tail_threshold_v2_dataset_summary.csv", dataset_summary_rows)
    write_csv(run_dir / "dataset_manifest.csv", dataset_manifest_rows)

    manifest = {
        "task": "tail_threshold_v2",
        "run_tag": resolved_run_tag,
        "run_dir": str(run_dir.resolve()),
        "dataset_count": len(dataset_ids),
        "asset_count": len(assets),
        "latest_only": bool(latest_only),
        "root_names": list(root_names or []),
        "threshold_percentages": [spec.pct for spec in threshold_specs],
        "threshold_labels": [spec.label for spec in threshold_specs],
        "numeric_bins": int(numeric_bins),
        "max_workers": int(max_workers),
        "outputs": {
            "asset_scores_csv": str((data_dir / "tail_threshold_v2_asset_scores.csv").resolve()),
            "column_scores_csv": str((data_dir / "tail_threshold_v2_column_scores.csv").resolve()),
            "dataset_diagnostics_csv": str((data_dir / "tail_threshold_v2_dataset_diagnostics.csv").resolve()),
            "global_summary_csv": str((summaries_dir / "tail_threshold_v2_global_summary.csv").resolve()),
            "model_summary_csv": str((summaries_dir / "tail_threshold_v2_model_summary.csv").resolve()),
            "dataset_summary_csv": str((summaries_dir / "tail_threshold_v2_dataset_summary.csv").resolve()),
            "dataset_manifest_csv": str((run_dir / "dataset_manifest.csv").resolve()),
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(TAIL_THRESHOLD_ROOT / "final" / "manifest.json", manifest)
    return manifest
