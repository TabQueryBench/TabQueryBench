"""Run global tail-threshold sensitivity diagnostics and visualizations."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from tqb_scoring.eval.common import (
    SyntheticAsset,
    TaskProgressTracker,
    discover_synthetic_assets,
    list_dataset_ids,
    make_task_run_dir,
    now_run_tag,
    resolve_real_split_path,
    write_csv,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = PROJECT_ROOT.parent / "results"
TAIL_THRESHOLD_ROOT = EVALUATION_ROOT / "tail_threshold"

DEFAULT_THRESHOLD_PCTS = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.1, 0.01, 0.001]
DEFAULT_NUMERIC_BINS = 10
DEFAULT_MAX_WORKERS = 4
DEFAULT_REPRESENTATIVES_PER_PREFIX = 2

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

TAIL_COLOR = "#E76F51"
HEAD_COLOR = "#4C78A8"
SUBMETRIC_COLORS = {
    "tail_set_consistency": "#C8553D",
    "tail_mass_similarity": "#2A9D8F",
    "tail_concentration_consistency": "#6D597A",
    "tail_anchor_coverage": "#577590",
}
PREFIX_COLORS = {"c": "#577590", "m": "#43AA8B", "n": "#F3722C"}
REPRESENTATIVE_KIND_COLORS = {"fragility": "#E76F51", "hardness": "#6D597A"}


@dataclass(frozen=True)
class ThresholdSpec:
    index: int
    pct: float
    ratio: float
    label: str
    subgroup_keep_ratio: float


def _threshold_specs(percentages: list[float] | None = None) -> list[ThresholdSpec]:
    values = percentages or DEFAULT_THRESHOLD_PCTS
    specs: list[ThresholdSpec] = []
    for idx, pct in enumerate(values):
        pct_value = float(pct)
        ratio = pct_value / 100.0
        label = f"{pct_value:g}%"
        specs.append(
            ThresholdSpec(
                index=idx,
                pct=pct_value,
                ratio=ratio,
                label=label,
                subgroup_keep_ratio=max(0.0, 1.0 - ratio),
            )
        )
    return specs


def _threshold_label_token(label: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(label or "").strip()).strip("_").lower()
    return token or "threshold"


def _closest_threshold_label(
    threshold_specs: list[ThresholdSpec],
    target_pct: float,
    *,
    exclude: set[str] | None = None,
) -> str | None:
    blocked = exclude or set()
    candidates = [spec for spec in threshold_specs if spec.label not in blocked]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda spec: (abs(float(spec.pct) - float(target_pct)), -float(spec.pct), spec.index))
    return ranked[0].label


def _fragility_anchor_plan(threshold_specs: list[ThresholdSpec]) -> dict[str, Any]:
    labels = [spec.label for spec in threshold_specs]
    if not labels:
        return {
            "anchor_label": None,
            "primary_label": None,
            "secondary_label": None,
            "rarest_label": None,
            "comparison_labels": [],
        }

    anchor_label = labels[0]
    rarest_label = labels[-1]
    used = {anchor_label}
    primary_label = _closest_threshold_label(threshold_specs, 0.5, exclude=used)
    if primary_label:
        used.add(primary_label)
    secondary_label = _closest_threshold_label(threshold_specs, 0.1, exclude=used)
    if secondary_label:
        used.add(secondary_label)

    comparison_labels: list[str] = []
    for label in [primary_label, secondary_label, rarest_label]:
        if label and label != anchor_label and label not in comparison_labels:
            comparison_labels.append(label)

    return {
        "anchor_label": anchor_label,
        "primary_label": primary_label,
        "secondary_label": secondary_label,
        "rarest_label": rarest_label,
        "comparison_labels": comparison_labels,
    }


def _score_lookup(entries: dict[str, dict[str, Any]], label: str | None) -> float | None:
    if not label:
        return None
    return _to_float((entries.get(label) or {}).get("tail_overall_score"))


def _attach_legacy_fragility_fields(payload: dict[str, Any], entries: dict[str, dict[str, Any]]) -> None:
    legacy_score_fields = {
        "10%": "tail_10pct",
        "0.5%": "tail_0_5pct",
        "0.1%": "tail_0_1pct",
        "0.001%": "tail_0_001pct",
    }
    for label, field_name in legacy_score_fields.items():
        payload[field_name] = _score_lookup(entries, label)

    tail_10 = payload.get("tail_10pct")
    tail_05 = payload.get("tail_0_5pct")
    tail_01 = payload.get("tail_0_1pct")
    tail_0001 = payload.get("tail_0_001pct")
    payload["fragility_10_to_0_5"] = round(float(tail_10) - float(tail_05), 6) if tail_10 is not None and tail_05 is not None else None
    payload["fragility_10_to_0_1"] = round(float(tail_10) - float(tail_01), 6) if tail_10 is not None and tail_01 is not None else None
    payload["fragility_10_to_0_001"] = (
        round(float(tail_10) - float(tail_0001), 6) if tail_10 is not None and tail_0001 is not None else None
    )


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        columns = [str(col) for col in (reader.fieldnames or [])]
    return columns, rows


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
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return round(sum(cleaned) / len(cleaned), 6)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "na", "n/a"}


def _safe_float(value: Any) -> float | None:
    try:
        if _is_missing(value):
            return None
        return float(str(value).strip())
    except Exception:
        return None


def _is_id_like(name: str) -> bool:
    text = str(name).strip().lower()
    return text in {"id", "row_id", "index"} or text.endswith("_id")


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


def _bin_numeric(value: float, edges: list[float]) -> str:
    if not edges or len(edges) < 2:
        return "q1"
    for idx in range(len(edges) - 1):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == len(edges) - 2:
            if left <= value <= right:
                return f"q{idx + 1}"
        if left <= value < right:
            return f"q{idx + 1}"
    if value < edges[0]:
        return "below_q1"
    return f"above_q{len(edges) - 1}"


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
            transformers[column] = {"mode": "numeric_bin", "edges": _quantile_edges(numeric_values, bins=numeric_bins)}
        else:
            transformers[column] = {"mode": "categorical"}
    return transformers


def _tokenize(value: Any, rule: dict[str, Any]) -> str:
    if _is_missing(value):
        return "__MISSING__"
    mode = str(rule.get("mode") or "categorical")
    text = str(value).strip()
    if mode == "numeric_bin":
        numeric_value = _safe_float(value)
        if numeric_value is None:
            return "__MISSING__"
        return _bin_numeric(numeric_value, rule.get("edges") or [])
    return text


def _build_key_counter(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    transformers: dict[str, dict[str, Any]],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        for column in feature_columns:
            token = _tokenize(row.get(column), transformers[column])
            counter[f"{column}::{token}"] += 1
    return counter


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


def _select_top_band(items: list[tuple[str, int]], keep_ratio: float) -> tuple[set[str], int]:
    if not items:
        return set(), 0
    keep_n = max(1, int(math.ceil(len(items) * max(0.0, float(keep_ratio)))))
    selected = items[:keep_n]
    gate = int(selected[-1][1]) if selected else 0
    return {key for key, _ in selected}, gate


def _tv_similarity_over_keys(real_counts: Counter[str], syn_counts: Counter[str], keys: set[str]) -> float:
    if not keys:
        return 1.0
    real_total = sum(real_counts.get(key, 0) for key in keys)
    syn_total = sum(syn_counts.get(key, 0) for key in keys)
    if real_total <= 0 and syn_total <= 0:
        return 1.0
    if real_total <= 0 or syn_total <= 0:
        return 0.0
    tv = 0.0
    for key in keys:
        pr = real_counts.get(key, 0) / real_total
        ps = syn_counts.get(key, 0) / syn_total
        tv += abs(pr - ps)
    return max(0.0, min(1.0, 1.0 - 0.5 * tv))


def _band_metrics(
    *,
    real_counts: Counter[str],
    syn_counts: Counter[str],
    n_real: int,
    n_syn: int,
    real_keys: set[str],
    syn_keys: set[str],
    effective_gate_real: int,
    effective_gate_syn: int,
) -> dict[str, float]:
    union_keys = real_keys | syn_keys
    inter_keys = real_keys & syn_keys
    set_consistency = (len(inter_keys) / len(union_keys)) if union_keys else 1.0

    mass_real = (sum(real_counts.get(key, 0) for key in real_keys) / max(1, n_real)) if real_keys else 0.0
    mass_syn_on_real = (sum(syn_counts.get(key, 0) for key in real_keys) / max(1, n_syn)) if real_keys else 0.0
    if mass_real <= 1e-12:
        mass_similarity = 1.0 if mass_syn_on_real <= 1e-12 else 0.0
    else:
        mass_similarity = 1.0 - abs(mass_syn_on_real - mass_real) / mass_real
        mass_similarity = max(0.0, min(1.0, mass_similarity))

    concentration_consistency = _tv_similarity_over_keys(real_counts, syn_counts, union_keys)
    anchor_coverage = (sum(1 for key in real_keys if syn_counts.get(key, 0) > 0) / len(real_keys)) if real_keys else 1.0

    return {
        "set_consistency": float(set_consistency),
        "mass_similarity": float(mass_similarity),
        "concentration_consistency": float(concentration_consistency),
        "anchor_coverage": float(anchor_coverage),
        "real_key_count": float(len(real_keys)),
        "syn_key_count": float(len(syn_keys)),
        "union_key_count": float(len(union_keys)),
        "effective_gate_real": float(effective_gate_real),
        "effective_gate_syn": float(effective_gate_syn),
    }


def _normalize_model_id(model_id: str) -> str:
    key = str(model_id or "").strip().lower()
    if key == "rtf":
        return "realtabformer"
    return key


def _model_label(model_id: str) -> str:
    key = _normalize_model_id(model_id)
    return MODEL_LABELS.get(key, key or "unknown")


def _natural_key(text: str) -> list[Any]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


def _model_sort_key(model_id: str) -> tuple[int, Any]:
    return (0, _natural_key(_model_label(model_id)))


def _dataset_prefix(dataset_id: str) -> str:
    return str(dataset_id or "").strip().lower()[:1]


def _asset_payload(asset: SyntheticAsset) -> dict[str, Any]:
    payload = asset.to_dict()
    raw_model_id = str(payload.get("model_id") or "")
    payload["model_id_raw"] = raw_model_id
    payload["model_id"] = _normalize_model_id(raw_model_id)
    payload["model_label"] = _model_label(payload["model_id"])
    return payload


def _run_dataset_threshold_sweep(
    dataset_id: str,
    dataset_assets: list[SyntheticAsset],
    threshold_specs: list[ThresholdSpec],
    numeric_bins: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    real_csv = resolve_real_split_path(dataset_id, split="train")
    if not real_csv.exists():
        return dataset_id, [], [], {"dataset_id": dataset_id, "status": "missing_real_csv", "asset_count": len(dataset_assets)}

    columns, rows_real = _read_csv_rows(real_csv)
    if not columns or not rows_real:
        return dataset_id, [], [], {"dataset_id": dataset_id, "status": "empty_real_csv", "asset_count": len(dataset_assets)}

    target_column = _load_target_column(dataset_id, columns)
    feature_columns = [column for column in columns if column != target_column and not _is_id_like(column)]
    if not feature_columns:
        return dataset_id, [], [], {"dataset_id": dataset_id, "status": "no_feature_columns", "asset_count": len(dataset_assets)}

    transformers = _build_transformers(rows_real, feature_columns, numeric_bins=numeric_bins)
    real_counts = _build_key_counter(rows_real, feature_columns, transformers)
    real_tail_items = _sorted_support_items(real_counts, reverse=False)
    real_head_items = _sorted_support_items(real_counts, reverse=True)
    n_real = len(rows_real)

    real_band_map: dict[str, dict[str, Any]] = {}
    real_diagnostic_rows: list[dict[str, Any]] = []
    for spec in threshold_specs:
        tail_real_keys, tail_real_gate = _select_bottom_band(real_tail_items, spec.ratio)
        head_real_keys, head_real_gate = _select_top_band(real_head_items, spec.subgroup_keep_ratio)
        real_tail_mass = (sum(real_counts.get(key, 0) for key in tail_real_keys) / max(1, n_real)) if tail_real_keys else 0.0
        real_head_mass = (sum(real_counts.get(key, 0) for key in head_real_keys) / max(1, n_real)) if head_real_keys else 0.0
        real_band_map[spec.label] = {
            "tail_real_keys": tail_real_keys,
            "tail_real_gate": tail_real_gate,
            "head_real_keys": head_real_keys,
            "head_real_gate": head_real_gate,
        }
        real_diagnostic_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": _dataset_prefix(dataset_id),
                "threshold_label": spec.label,
                "threshold_pct": spec.pct,
                "tail_ratio": spec.ratio,
                "subgroup_keep_ratio": spec.subgroup_keep_ratio,
                "real_row_count": n_real,
                "real_total_key_count": len(real_tail_items),
                "real_tail_key_count": len(tail_real_keys),
                "real_head_key_count": len(head_real_keys),
                "real_tail_mass": round(real_tail_mass, 6),
                "real_head_mass": round(real_head_mass, 6),
                "tail_effective_gate_real": tail_real_gate,
                "head_effective_gate_real": head_real_gate,
            }
        )

    asset_rows: list[dict[str, Any]] = []
    for asset in dataset_assets:
        asset_payload = _asset_payload(asset)
        _, rows_syn = _read_csv_rows(Path(asset.synthetic_csv_path))
        syn_counts = _build_key_counter(rows_syn, feature_columns, transformers)
        syn_tail_items = _sorted_support_items(syn_counts, reverse=False)
        syn_head_items = _sorted_support_items(syn_counts, reverse=True)
        n_syn = len(rows_syn)

        for spec in threshold_specs:
            real_band = real_band_map[spec.label]
            tail_syn_keys, tail_syn_gate = _select_bottom_band(syn_tail_items, spec.ratio)
            head_syn_keys, head_syn_gate = _select_top_band(syn_head_items, spec.subgroup_keep_ratio)

            tail_metrics = _band_metrics(
                real_counts=real_counts,
                syn_counts=syn_counts,
                n_real=n_real,
                n_syn=n_syn,
                real_keys=real_band["tail_real_keys"],
                syn_keys=tail_syn_keys,
                effective_gate_real=int(real_band["tail_real_gate"]),
                effective_gate_syn=tail_syn_gate,
            )
            head_metrics = _band_metrics(
                real_counts=real_counts,
                syn_counts=syn_counts,
                n_real=n_real,
                n_syn=n_syn,
                real_keys=real_band["head_real_keys"],
                syn_keys=head_syn_keys,
                effective_gate_real=int(real_band["head_real_gate"]),
                effective_gate_syn=head_syn_gate,
            )

            tail_overall_score = _mean(
                [
                    tail_metrics["set_consistency"],
                    tail_metrics["mass_similarity"],
                    tail_metrics["concentration_consistency"],
                ]
            )
            head_overall_score = _mean(
                [
                    head_metrics["set_consistency"],
                    head_metrics["mass_similarity"],
                    head_metrics["concentration_consistency"],
                ]
            )

            asset_rows.append(
                {
                    **asset_payload,
                    "dataset_id": dataset_id,
                    "dataset_prefix": _dataset_prefix(dataset_id),
                    "threshold_label": spec.label,
                    "threshold_pct": spec.pct,
                    "tail_ratio": spec.ratio,
                    "subgroup_keep_ratio": spec.subgroup_keep_ratio,
                    "real_row_count": n_real,
                    "synthetic_row_count": n_syn,
                    "feature_column_count": len(feature_columns),
                    "tail_set_consistency": round(tail_metrics["set_consistency"], 6),
                    "tail_mass_similarity": round(tail_metrics["mass_similarity"], 6),
                    "tail_concentration_consistency": round(tail_metrics["concentration_consistency"], 6),
                    "tail_anchor_coverage": round(tail_metrics["anchor_coverage"], 6),
                    "tail_overall_score": tail_overall_score,
                    "tail_real_key_count": int(tail_metrics["real_key_count"]),
                    "tail_syn_key_count": int(tail_metrics["syn_key_count"]),
                    "tail_union_key_count": int(tail_metrics["union_key_count"]),
                    "tail_effective_gate_real": int(tail_metrics["effective_gate_real"]),
                    "tail_effective_gate_syn": int(tail_metrics["effective_gate_syn"]),
                    "head_set_consistency": round(head_metrics["set_consistency"], 6),
                    "head_mass_similarity": round(head_metrics["mass_similarity"], 6),
                    "head_concentration_consistency": round(head_metrics["concentration_consistency"], 6),
                    "head_anchor_coverage": round(head_metrics["anchor_coverage"], 6),
                    "head_proxy_overall_score": head_overall_score,
                    "head_real_key_count": int(head_metrics["real_key_count"]),
                    "head_syn_key_count": int(head_metrics["syn_key_count"]),
                    "head_union_key_count": int(head_metrics["union_key_count"]),
                    "head_effective_gate_real": int(head_metrics["effective_gate_real"]),
                    "head_effective_gate_syn": int(head_metrics["effective_gate_syn"]),
                    "tail_head_gap": round((head_overall_score or 0.0) - (tail_overall_score or 0.0), 6)
                    if head_overall_score is not None and tail_overall_score is not None
                    else None,
                }
            )

    manifest_row = {
        "dataset_id": dataset_id,
        "status": "ok",
        "asset_count": len(dataset_assets),
        "real_row_count": n_real,
        "feature_column_count": len(feature_columns),
        "real_total_key_count": len(real_tail_items),
    }
    return dataset_id, asset_rows, real_diagnostic_rows, manifest_row


def _score_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "tail_threshold_scores",
        ["#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"],
    )
    cmap.set_bad("#ECEFF3")
    return cmap


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _threshold_axis(ax: plt.Axes, specs: list[ThresholdSpec], *, xlabel: str = "Tail threshold (% of keys)") -> None:
    xs = [spec.pct for spec in specs]
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(xs)
    ax.set_xticklabels([spec.label for spec in specs], rotation=0)
    ax.set_xlabel(xlabel)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q))


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


def _build_global_threshold_summary(
    asset_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in threshold_specs:
        items = [row for row in asset_rows if row.get("threshold_label") == spec.label]
        if not items:
            continue
        tail_scores = [_to_float(row.get("tail_overall_score")) for row in items]
        head_scores = [_to_float(row.get("head_proxy_overall_score")) for row in items]
        tail_clean = [float(value) for value in tail_scores if value is not None]
        head_clean = [float(value) for value in head_scores if value is not None]
        out.append(
            {
                "threshold_label": spec.label,
                "threshold_pct": spec.pct,
                "tail_ratio": spec.ratio,
                "subgroup_keep_ratio": spec.subgroup_keep_ratio,
                "tail_overall_mean": _mean(tail_scores),
                "tail_overall_median": round(_quantile(tail_clean, 0.5), 6) if tail_clean else None,
                "tail_overall_p25": round(_quantile(tail_clean, 0.25), 6) if tail_clean else None,
                "tail_overall_p75": round(_quantile(tail_clean, 0.75), 6) if tail_clean else None,
                "head_proxy_mean": _mean(head_scores),
                "head_proxy_median": round(_quantile(head_clean, 0.5), 6) if head_clean else None,
                "tail_head_gap_mean": _mean([_to_float(row.get("tail_head_gap")) for row in items]),
                "tail_set_consistency_mean": _mean([_to_float(row.get("tail_set_consistency")) for row in items]),
                "tail_mass_similarity_mean": _mean([_to_float(row.get("tail_mass_similarity")) for row in items]),
                "tail_concentration_consistency_mean": _mean(
                    [_to_float(row.get("tail_concentration_consistency")) for row in items]
                ),
                "tail_anchor_coverage_mean": _mean([_to_float(row.get("tail_anchor_coverage")) for row in items]),
                "asset_count": len(items),
            }
        )
    return out


def _compute_model_fragility(
    model_summary_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in model_summary_rows:
        by_model[str(row.get("model_id") or "")][str(row.get("threshold_label") or "")] = row

    plan = _fragility_anchor_plan(threshold_specs)
    anchor_label = plan["anchor_label"]
    comparison_labels = list(plan["comparison_labels"])
    labels_to_capture = [label for label in [anchor_label, *comparison_labels] if label]
    out: list[dict[str, Any]] = []
    for model_id in sorted(by_model.keys(), key=_model_sort_key):
        entries = by_model[model_id]
        payload = {
            "model_id": model_id,
            "model_label": _model_label(model_id),
            "anchor_threshold_label": anchor_label,
            "primary_comparison_label": plan["primary_label"],
            "secondary_comparison_label": plan["secondary_label"],
            "rarest_threshold_label": plan["rarest_label"],
        }
        for label in labels_to_capture:
            payload[f"tail_at_{_threshold_label_token(label)}"] = _score_lookup(entries, label)

        anchor_score = _score_lookup(entries, anchor_label)
        payload["anchor_tail_score"] = anchor_score
        primary_score = _score_lookup(entries, plan["primary_label"])
        secondary_score = _score_lookup(entries, plan["secondary_label"])
        rarest_score = _score_lookup(entries, plan["rarest_label"])
        payload["primary_comparison_tail_score"] = primary_score
        payload["secondary_comparison_tail_score"] = secondary_score
        payload["rarest_tail_score"] = rarest_score
        payload["primary_fragility_drop"] = (
            round(float(anchor_score) - float(primary_score), 6)
            if anchor_score is not None and primary_score is not None
            else None
        )
        payload["secondary_fragility_drop"] = (
            round(float(anchor_score) - float(secondary_score), 6)
            if anchor_score is not None and secondary_score is not None
            else None
        )
        payload["anchor_to_rarest_fragility_drop"] = (
            round(float(anchor_score) - float(rarest_score), 6)
            if anchor_score is not None and rarest_score is not None
            else None
        )
        for label in comparison_labels:
            compare_score = _score_lookup(entries, label)
            payload[f"fragility_{_threshold_label_token(anchor_label)}_to_{_threshold_label_token(label)}"] = (
                round(float(anchor_score) - float(compare_score), 6)
                if anchor_score is not None and compare_score is not None
                else None
            )
        _attach_legacy_fragility_fields(payload, entries)
        out.append(payload)
    return out


def _compute_dataset_fragility(
    dataset_summary_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in dataset_summary_rows:
        by_dataset[str(row.get("dataset_id") or "")][str(row.get("threshold_label") or "")] = row

    plan = _fragility_anchor_plan(threshold_specs)
    anchor_label = plan["anchor_label"]
    comparison_labels = list(plan["comparison_labels"])
    out: list[dict[str, Any]] = []
    for dataset_id, entries in sorted(by_dataset.items()):
        anchor_score = _score_lookup(entries, anchor_label)
        if anchor_score is None:
            continue
        primary_score = _score_lookup(entries, plan["primary_label"])
        secondary_score = _score_lookup(entries, plan["secondary_label"])
        rarest_score = _score_lookup(entries, plan["rarest_label"])
        payload = {
            "dataset_id": dataset_id,
            "dataset_prefix": _dataset_prefix(dataset_id),
            "anchor_threshold_label": anchor_label,
            "primary_comparison_label": plan["primary_label"],
            "secondary_comparison_label": plan["secondary_label"],
            "rarest_threshold_label": plan["rarest_label"],
            "anchor_tail_score": anchor_score,
            "primary_comparison_tail_score": primary_score,
            "secondary_comparison_tail_score": secondary_score,
            "rarest_tail_score": rarest_score,
            "primary_fragility_drop": round(anchor_score - primary_score, 6) if primary_score is not None else None,
            "secondary_fragility_drop": round(anchor_score - secondary_score, 6) if secondary_score is not None else None,
            "anchor_to_rarest_fragility_drop": round(anchor_score - rarest_score, 6) if rarest_score is not None else None,
        }
        for label in [anchor_label, *comparison_labels]:
            if label:
                payload[f"tail_at_{_threshold_label_token(label)}"] = _score_lookup(entries, label)
        for label in comparison_labels:
            compare_score = _score_lookup(entries, label)
            payload[f"fragility_{_threshold_label_token(anchor_label)}_to_{_threshold_label_token(label)}"] = (
                round(anchor_score - compare_score, 6) if compare_score is not None else None
            )
        _attach_legacy_fragility_fields(payload, entries)
        out.append(
            payload
        )
    return out


def _select_representative_datasets(
    dataset_fragility_rows: list[dict[str, Any]],
    per_prefix: int,
) -> list[dict[str, Any]]:
    by_prefix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_fragility_rows:
        by_prefix[str(row.get("dataset_prefix") or "?")].append(row)

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for prefix in sorted(by_prefix.keys()):
        pool = by_prefix[prefix]
        fragility_candidates = sorted(
            [row for row in pool if row.get("primary_fragility_drop") is not None],
            key=lambda row: float(row["primary_fragility_drop"]),
            reverse=True,
        )
        hardness_candidates = sorted(
            [row for row in pool if row.get("anchor_tail_score") is not None],
            key=lambda row: float(row["anchor_tail_score"]),
        )

        picks: list[tuple[str, dict[str, Any]]] = []
        if fragility_candidates:
            picks.append(("fragility", fragility_candidates[0]))
        for candidate in hardness_candidates:
            if not picks or candidate["dataset_id"] != picks[0][1]["dataset_id"]:
                picks.append(("hardness", candidate))
                break

        extra_candidates = []
        for candidate in fragility_candidates:
            if candidate["dataset_id"] not in {row["dataset_id"] for _, row in picks}:
                extra_candidates.append(("fragility", candidate))
        for candidate in hardness_candidates:
            if candidate["dataset_id"] not in {row["dataset_id"] for _, row in picks}:
                extra_candidates.append(("hardness", candidate))

        picks = picks[:per_prefix]
        for kind, row in extra_candidates:
            if len(picks) >= per_prefix:
                break
            picks.append((kind, row))

        for kind, row in picks:
            dataset_id = str(row["dataset_id"])
            if dataset_id in used:
                continue
            used.add(dataset_id)
            selected.append(
                {
                    **row,
                    "selection_kind": kind,
                    "selection_reason": (
                        f"largest drop from {row.get('anchor_threshold_label')} to {row.get('primary_comparison_label')}"
                        if kind == "fragility"
                        else f"lowest tail score already at {row.get('anchor_threshold_label')}"
                    ),
                }
            )
    return selected


def _plot_global_tail_vs_head(
    summary_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
    out_path: Path,
) -> None:
    x = [spec.pct for spec in threshold_specs]
    tail = [float((next(row for row in summary_rows if row["threshold_label"] == spec.label))["tail_overall_mean"]) for spec in threshold_specs]
    head = [float((next(row for row in summary_rows if row["threshold_label"] == spec.label))["head_proxy_mean"]) for spec in threshold_specs]
    p25 = [float((next(row for row in summary_rows if row["threshold_label"] == spec.label))["tail_overall_p25"]) for spec in threshold_specs]
    p75 = [float((next(row for row in summary_rows if row["threshold_label"] == spec.label))["tail_overall_p75"]) for spec in threshold_specs]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.fill_between(x, p25, p75, color=TAIL_COLOR, alpha=0.14, label="Tail IQR")
    ax.plot(x, tail, marker="o", linewidth=2.6, color=TAIL_COLOR, label="Tail score")
    ax.plot(x, head, marker="o", linewidth=2.4, color=HEAD_COLOR, label="Head proxy score")
    _threshold_axis(ax, threshold_specs)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Global tail fragility: tail degrades faster than the head support band")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend()
    _save(fig, out_path)


def _plot_global_tail_submetrics(
    summary_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
    out_path: Path,
) -> None:
    metric_fields = [
        ("tail_set_consistency_mean", "Tail set consistency"),
        ("tail_mass_similarity_mean", "Tail mass similarity"),
        ("tail_concentration_consistency_mean", "Tail concentration consistency"),
        ("tail_anchor_coverage_mean", "Tail anchor coverage"),
    ]
    x = [spec.pct for spec in threshold_specs]
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for metric_field, label in metric_fields:
        y = [float((next(row for row in summary_rows if row["threshold_label"] == spec.label))[metric_field]) for spec in threshold_specs]
        ax.plot(x, y, marker="o", linewidth=2.3, label=label, color=SUBMETRIC_COLORS.get(metric_field.replace("_mean", ""), None))
    _threshold_axis(ax, threshold_specs)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Which tail behaviors break first as the threshold gets rarer?")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(loc="lower left")
    _save(fig, out_path)


def _plot_tail_distribution_boxplot(
    asset_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
    out_path: Path,
) -> None:
    labels = [spec.label for spec in threshold_specs]
    data = [
        [float(row["tail_overall_score"]) for row in asset_rows if row.get("threshold_label") == spec.label and row.get("tail_overall_score") is not None]
        for spec in threshold_specs
    ]
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    box = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.58)
    for patch in box["boxes"]:
        patch.set_facecolor("#F4A261")
        patch.set_alpha(0.55)
        patch.set_edgecolor("#9C4F2F")
    for median_line in box["medians"]:
        median_line.set_color("#7F0000")
        median_line.set_linewidth(1.8)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Tail overall score")
    ax.set_title("Asset-level tail score distribution across thresholds")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    _save(fig, out_path)


def _plot_threshold_key_diagnostics(
    diagnostic_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
    out_path: Path,
) -> None:
    x = [spec.pct for spec in threshold_specs]
    key_medians: list[float] = []
    key_means: list[float] = []
    frac_le_one: list[float] = []
    frac_le_two: list[float] = []
    tail_mass_medians: list[float] = []
    for spec in threshold_specs:
        rows = [row for row in diagnostic_rows if row.get("threshold_label") == spec.label]
        key_counts = [float(row["real_tail_key_count"]) for row in rows]
        tail_masses = [float(row["real_tail_mass"]) for row in rows]
        key_medians.append(float(np.median(key_counts)) if key_counts else 0.0)
        key_means.append(float(np.mean(key_counts)) if key_counts else 0.0)
        frac_le_one.append((sum(1 for value in key_counts if value <= 1.0) / len(key_counts)) if key_counts else 0.0)
        frac_le_two.append((sum(1 for value in key_counts if value <= 2.0) / len(key_counts)) if key_counts else 0.0)
        tail_mass_medians.append(float(np.median(tail_masses)) if tail_masses else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    axes[0].plot(x, key_medians, marker="o", linewidth=2.4, color="#264653", label="Median tail key count")
    axes[0].plot(x, key_means, marker="o", linewidth=2.0, color="#2A9D8F", label="Mean tail key count")
    axes[0].plot(x, tail_mass_medians, marker="o", linewidth=2.0, color="#E9C46A", label="Median real tail mass")
    _threshold_axis(axes[0], threshold_specs)
    axes[0].set_ylabel("Count / mass")
    axes[0].set_title("How much tail evidence is left?")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, frac_le_one, marker="o", linewidth=2.4, color="#C8553D", label="Datasets with <= 1 tail key")
    axes[1].plot(x, frac_le_two, marker="o", linewidth=2.2, color="#6D597A", label="Datasets with <= 2 tail keys")
    _threshold_axis(axes[1], threshold_specs)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Fraction of datasets")
    axes[1].set_title("When does the tail become statistically tiny?")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    axes[1].legend()

    _save(fig, out_path)


def _plot_model_heatmap(
    model_summary_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
    out_path: Path,
) -> None:
    model_ids = sorted({str(row.get("model_id") or "") for row in model_summary_rows}, key=_model_sort_key)
    lookup = {(str(row["model_id"]), str(row["threshold_label"])): _to_float(row.get("tail_overall_score")) for row in model_summary_rows}
    mat = np.full((len(model_ids), len(threshold_specs)), np.nan, dtype=float)
    for row_idx, model_id in enumerate(model_ids):
        for col_idx, spec in enumerate(threshold_specs):
            value = lookup.get((model_id, spec.label))
            if value is not None:
                mat[row_idx, col_idx] = float(value)

    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    im = ax.imshow(mat, aspect="auto", cmap=_score_cmap(), vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(threshold_specs)))
    ax.set_xticklabels([spec.label for spec in threshold_specs], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(model_ids)))
    ax.set_yticklabels([_model_label(model_id) for model_id in model_ids])
    ax.set_title("Model-by-threshold heatmap of tail fidelity")
    for row_idx in range(mat.shape[0]):
        for col_idx in range(mat.shape[1]):
            value = mat[row_idx, col_idx]
            if np.isnan(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value >= 0.52 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, out_path)


def _plot_model_fragility_bar(
    model_fragility_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    rows = [row for row in model_fragility_rows if row.get("primary_fragility_drop") is not None]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row["primary_fragility_drop"]), reverse=True)
    labels = [str(row["model_label"]) for row in rows]
    values = [float(row["primary_fragility_drop"]) for row in rows]
    colors = [TAIL_COLOR if value >= 0 else "#4C78A8" for value in values]
    anchor_label = str(rows[0].get("anchor_threshold_label") or "anchor")
    compare_label = str(rows[0].get("primary_comparison_label") or "comparison")

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    bars = ax.bar(np.arange(len(rows)), values, color=colors, alpha=0.82)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(f"Tail fragility: score({anchor_label}) - score({compare_label})")
    ax.set_title("Which models lose the most once the tail becomes rarer?")
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.24)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    _save(fig, out_path)


def _plot_dataset_heatmap(
    dataset_summary_rows: list[dict[str, Any]],
    dataset_fragility_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
    out_path: Path,
) -> None:
    ordered_datasets = [
        row["dataset_id"]
        for row in sorted(
            dataset_fragility_rows,
            key=lambda row: (
                -float(row["primary_fragility_drop"]) if row.get("primary_fragility_drop") is not None else 0.0,
                str(row["dataset_id"]),
            ),
        )
    ]
    lookup = {(str(row["dataset_id"]), str(row["threshold_label"])): _to_float(row.get("tail_overall_score")) for row in dataset_summary_rows}
    mat = np.full((len(ordered_datasets), len(threshold_specs)), np.nan, dtype=float)
    for row_idx, dataset_id in enumerate(ordered_datasets):
        for col_idx, spec in enumerate(threshold_specs):
            value = lookup.get((dataset_id, spec.label))
            if value is not None:
                mat[row_idx, col_idx] = float(value)

    fig_h = max(12.0, len(ordered_datasets) * 0.24)
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap=_score_cmap(), vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(threshold_specs)))
    ax.set_xticklabels([spec.label for spec in threshold_specs], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(ordered_datasets)))
    ax.set_yticklabels([dataset_id.upper() for dataset_id in ordered_datasets], fontsize=8)
    ax.set_title("Dataset-by-threshold heatmap ordered by tail fragility")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, out_path)


def _plot_prefix_lines(
    prefix_summary_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
    out_path: Path,
) -> None:
    x = [spec.pct for spec in threshold_specs]
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    for prefix in ["c", "m", "n"]:
        rows = [row for row in prefix_summary_rows if row.get("dataset_prefix") == prefix]
        if not rows:
            continue
        lookup = {str(row["threshold_label"]): row for row in rows}
        y = [float(lookup[spec.label]["tail_overall_score"]) for spec in threshold_specs if lookup.get(spec.label)]
        x_used = [spec.pct for spec in threshold_specs if lookup.get(spec.label)]
        ax.plot(x_used, y, marker="o", linewidth=2.4, label=prefix.upper(), color=PREFIX_COLORS[prefix])
    _threshold_axis(ax, threshold_specs)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Tail score")
    ax.set_title("Tail fragility differs by dataset family")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(title="Dataset prefix")
    _save(fig, out_path)


def _plot_representative_grid(
    representative_rows: list[dict[str, Any]],
    dataset_summary_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
    out_path: Path,
) -> None:
    if not representative_rows:
        return
    selected_ids = [str(row["dataset_id"]) for row in representative_rows]
    ncols = 2
    nrows = int(math.ceil(len(selected_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, max(4.6 * nrows, 5.0)))
    axes_list = np.atleast_1d(axes).reshape(-1)

    summary_lookup: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["dataset_id"]), str(row["threshold_label"])): row for row in dataset_summary_rows
    }
    diag_lookup: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["dataset_id"]), str(row["threshold_label"])): row for row in diagnostic_rows
    }
    primary_handles: list[Any] = []
    primary_labels: list[str] = []
    secondary_handles: list[Any] = []
    secondary_labels: list[str] = []

    for ax, rep in zip(axes_list, representative_rows):
        dataset_id = str(rep["dataset_id"])
        x = [spec.pct for spec in threshold_specs]
        tail = [float(summary_lookup[(dataset_id, spec.label)]["tail_overall_score"]) for spec in threshold_specs]
        head = [float(summary_lookup[(dataset_id, spec.label)]["head_proxy_overall_score"]) for spec in threshold_specs]
        key_count = [float(diag_lookup[(dataset_id, spec.label)]["real_tail_key_count"]) for spec in threshold_specs]
        ax.plot(x, tail, marker="o", linewidth=2.4, color=TAIL_COLOR, label="Tail")
        ax.plot(x, head, marker="o", linewidth=2.1, color=HEAD_COLOR, label="Head proxy")
        _threshold_axis(ax, threshold_specs)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", linestyle="--", alpha=0.22)
        ax.set_title(f"{dataset_id.upper()} | {rep['selection_kind']}: {rep['selection_reason']}")
        ax2 = ax.twinx()
        ax2.plot(x, key_count, marker="s", linewidth=1.6, color="#6D597A", alpha=0.8, label="Tail keys")
        ax2.set_ylabel("Real tail keys", color="#6D597A")
        ax2.tick_params(axis="y", labelcolor="#6D597A")
        if not primary_handles:
            primary_handles, primary_labels = ax.get_legend_handles_labels()
            secondary_handles, secondary_labels = ax2.get_legend_handles_labels()

    for ax in axes_list[len(representative_rows) :]:
        ax.axis("off")

    if primary_handles or secondary_handles:
        fig.legend(primary_handles + secondary_handles, primary_labels + secondary_labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Representative datasets where tail fidelity is especially fragile", y=1.02, fontsize=13)
    _save(fig, out_path)


def _plot_representative_model_lines(
    representative_rows: list[dict[str, Any]],
    model_summary_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
    out_dir: Path,
) -> list[str]:
    if not representative_rows:
        return []

    figures: list[str] = []
    model_lookup: dict[tuple[str, str, str], dict[str, Any]] = {
        (str(row["dataset_id"]), str(row["model_id"]), str(row["threshold_label"])): row for row in model_summary_rows
    }
    for rep in representative_rows:
        dataset_id = str(rep["dataset_id"])
        dataset_rows = [row for row in model_summary_rows if row.get("dataset_id") == dataset_id]
        model_ids = sorted({str(row["model_id"]) for row in dataset_rows}, key=_model_sort_key)
        x = [spec.pct for spec in threshold_specs]
        fig, ax = plt.subplots(figsize=(10.8, 6.0))
        for model_id in model_ids:
            y = []
            x_used = []
            for spec in threshold_specs:
                row = model_lookup.get((dataset_id, model_id, spec.label))
                if row is None or row.get("tail_overall_score") is None:
                    continue
                x_used.append(spec.pct)
                y.append(float(row["tail_overall_score"]))
            if not y:
                continue
            linewidth = 2.6 if model_id in {"realtabformer", "bayesnet", "ctgan", "tvae"} else 1.5
            alpha = 0.95 if linewidth > 2.0 else 0.7
            ax.plot(x_used, y, marker="o", linewidth=linewidth, alpha=alpha, label=_model_label(model_id))
        _threshold_axis(ax, threshold_specs)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Tail score")
        ax.set_title(f"{dataset_id.upper()} model lines across tail thresholds")
        ax.grid(axis="y", linestyle="--", alpha=0.24)
        ax.legend(ncol=2, fontsize=8)
        path = out_dir / f"{dataset_id}_model_lines.png"
        _save(fig, path)
        figures.append(str(path.resolve()))
    return figures


def _build_dataset_model_summary(asset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in asset_rows:
        grouped[(str(row["dataset_id"]), str(row["model_id"]), str(row["threshold_label"]))].append(row)

    out: list[dict[str, Any]] = []
    for (dataset_id, model_id, threshold_label), items in sorted(grouped.items()):
        base = items[0]
        out.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": base.get("dataset_prefix"),
                "model_id": model_id,
                "model_label": _model_label(model_id),
                "threshold_label": threshold_label,
                "threshold_pct": base.get("threshold_pct"),
                "tail_ratio": base.get("tail_ratio"),
                "tail_overall_score": _mean([_to_float(item.get("tail_overall_score")) for item in items]),
                "head_proxy_overall_score": _mean([_to_float(item.get("head_proxy_overall_score")) for item in items]),
                "tail_set_consistency": _mean([_to_float(item.get("tail_set_consistency")) for item in items]),
                "tail_mass_similarity": _mean([_to_float(item.get("tail_mass_similarity")) for item in items]),
                "tail_concentration_consistency": _mean(
                    [_to_float(item.get("tail_concentration_consistency")) for item in items]
                ),
                "asset_count": len(items),
            }
        )
    return out


def _load_existing_dataset_outputs(source_run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    asset_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    datasets_dir = source_run_dir / "datasets"
    if not datasets_dir.exists():
        return asset_rows, diagnostic_rows, manifest_rows

    for dataset_dir in sorted(path for path in datasets_dir.iterdir() if path.is_dir()):
        dataset_id = dataset_dir.name
        asset_files = sorted(dataset_dir.glob("tail_threshold_asset_scores__*.csv"))
        diagnostic_files = sorted(dataset_dir.glob("tail_threshold_real_diagnostics__*.csv"))
        dataset_asset_rows: list[dict[str, Any]] = []
        dataset_diagnostic_rows: list[dict[str, Any]] = []
        for path in asset_files:
            _, rows = _read_csv_rows(path)
            dataset_asset_rows.extend(rows)
        for path in diagnostic_files:
            _, rows = _read_csv_rows(path)
            dataset_diagnostic_rows.extend(rows)
        asset_rows.extend(dataset_asset_rows)
        diagnostic_rows.extend(dataset_diagnostic_rows)
        manifest_rows.append(
            {
                "dataset_id": dataset_id,
                "status": "ok" if dataset_asset_rows or dataset_diagnostic_rows else "empty",
                "asset_count": len({str(row.get("asset_key") or "") for row in dataset_asset_rows if row.get("asset_key")}),
                "row_count": len(dataset_asset_rows),
            }
        )

    return asset_rows, diagnostic_rows, manifest_rows


def _infer_threshold_specs_from_rows(
    asset_rows: list[dict[str, Any]],
    fallback_percentages: list[float] | None = None,
) -> list[ThresholdSpec]:
    pairs: list[tuple[float, str]] = []
    seen: set[tuple[float, str]] = set()
    for row in asset_rows:
        pct = _to_float(row.get("threshold_pct"))
        label = str(row.get("threshold_label") or "").strip()
        if pct is None or not label:
            continue
        key = (float(pct), label)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    if not pairs:
        return _threshold_specs(fallback_percentages)
    pairs = sorted(pairs, key=lambda item: float(item[0]), reverse=True)
    return [
        ThresholdSpec(
            index=idx,
            pct=float(pct),
            ratio=float(pct) / 100.0,
            label=label,
            subgroup_keep_ratio=max(0.0, 1.0 - (float(pct) / 100.0)),
        )
        for idx, (pct, label) in enumerate(pairs)
    ]


def _materialize_tail_threshold_outputs(
    *,
    run_dir: Path,
    asset_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    dataset_manifest_rows: list[dict[str, Any]],
    threshold_specs: list[ThresholdSpec],
    latest_only: bool,
    representatives_per_prefix: int,
    source_run_dir: Path | None = None,
    synthetic_root_filter: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    threshold_summary_rows = _build_global_threshold_summary(asset_rows, threshold_specs)
    model_summary_rows = _aggregate_group_mean(
        asset_rows,
        group_keys=["model_id", "model_label", "threshold_label", "threshold_pct", "tail_ratio"],
        value_fields=[
            "tail_overall_score",
            "head_proxy_overall_score",
            "tail_set_consistency",
            "tail_mass_similarity",
            "tail_concentration_consistency",
            "tail_anchor_coverage",
            "tail_head_gap",
        ],
    )
    dataset_summary_rows = _aggregate_group_mean(
        asset_rows,
        group_keys=["dataset_id", "dataset_prefix", "threshold_label", "threshold_pct", "tail_ratio"],
        value_fields=[
            "tail_overall_score",
            "head_proxy_overall_score",
            "tail_set_consistency",
            "tail_mass_similarity",
            "tail_concentration_consistency",
            "tail_anchor_coverage",
            "tail_head_gap",
        ],
    )
    prefix_summary_rows = _aggregate_group_mean(
        asset_rows,
        group_keys=["dataset_prefix", "threshold_label", "threshold_pct", "tail_ratio"],
        value_fields=[
            "tail_overall_score",
            "head_proxy_overall_score",
            "tail_set_consistency",
            "tail_mass_similarity",
            "tail_concentration_consistency",
            "tail_anchor_coverage",
            "tail_head_gap",
        ],
    )
    dataset_model_summary_rows = _build_dataset_model_summary(asset_rows)
    model_fragility_rows = _compute_model_fragility(model_summary_rows, threshold_specs)
    dataset_fragility_rows = _compute_dataset_fragility(dataset_summary_rows, threshold_specs)
    representative_rows = _select_representative_datasets(dataset_fragility_rows, representatives_per_prefix)

    summary_dir = run_dir / "summaries"
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    representatives_dir = figures_dir / "representatives"
    representatives_dir.mkdir(parents=True, exist_ok=True)

    write_csv(summary_dir / "tail_threshold_asset_scores__all_datasets.csv", asset_rows)
    write_csv(summary_dir / "tail_threshold_real_diagnostics__all_datasets.csv", diagnostic_rows)
    write_csv(summary_dir / "tail_threshold_dataset_manifest__all_datasets.csv", dataset_manifest_rows)

    write_csv(tables_dir / "global_threshold_summary.csv", threshold_summary_rows)
    write_csv(tables_dir / "model_threshold_summary.csv", model_summary_rows)
    write_csv(tables_dir / "dataset_threshold_summary.csv", dataset_summary_rows)
    write_csv(tables_dir / "prefix_threshold_summary.csv", prefix_summary_rows)
    write_csv(tables_dir / "dataset_model_threshold_summary.csv", dataset_model_summary_rows)
    write_csv(tables_dir / "model_fragility_summary.csv", model_fragility_rows)
    write_csv(tables_dir / "dataset_fragility_summary.csv", dataset_fragility_rows)
    write_csv(tables_dir / "representative_datasets.csv", representative_rows)

    figure_paths: list[str] = []

    global_tail_head = figures_dir / "01_global_tail_vs_head_proxy.png"
    _plot_global_tail_vs_head(threshold_summary_rows, threshold_specs, global_tail_head)
    figure_paths.append(str(global_tail_head.resolve()))

    global_tail_submetrics = figures_dir / "02_global_tail_submetrics.png"
    _plot_global_tail_submetrics(threshold_summary_rows, threshold_specs, global_tail_submetrics)
    figure_paths.append(str(global_tail_submetrics.resolve()))

    distribution_boxplot = figures_dir / "03_tail_score_distribution_boxplot.png"
    _plot_tail_distribution_boxplot(asset_rows, threshold_specs, distribution_boxplot)
    figure_paths.append(str(distribution_boxplot.resolve()))

    diagnostics_plot = figures_dir / "04_threshold_key_diagnostics.png"
    _plot_threshold_key_diagnostics(diagnostic_rows, threshold_specs, diagnostics_plot)
    figure_paths.append(str(diagnostics_plot.resolve()))

    model_heatmap = figures_dir / "05_model_threshold_heatmap.png"
    _plot_model_heatmap(model_summary_rows, threshold_specs, model_heatmap)
    figure_paths.append(str(model_heatmap.resolve()))

    model_fragility_bar = figures_dir / "06_model_fragility_bar.png"
    _plot_model_fragility_bar(model_fragility_rows, model_fragility_bar)
    figure_paths.append(str(model_fragility_bar.resolve()))

    dataset_heatmap = figures_dir / "07_dataset_threshold_heatmap.png"
    _plot_dataset_heatmap(dataset_summary_rows, dataset_fragility_rows, threshold_specs, dataset_heatmap)
    figure_paths.append(str(dataset_heatmap.resolve()))

    prefix_lines = figures_dir / "08_prefix_threshold_lines.png"
    _plot_prefix_lines(prefix_summary_rows, threshold_specs, prefix_lines)
    figure_paths.append(str(prefix_lines.resolve()))

    representative_grid = figures_dir / "09_representative_dataset_grid.png"
    _plot_representative_grid(representative_rows, dataset_summary_rows, diagnostic_rows, threshold_specs, representative_grid)
    if representative_grid.exists():
        figure_paths.append(str(representative_grid.resolve()))

    figure_paths.extend(_plot_representative_model_lines(representative_rows, dataset_model_summary_rows, threshold_specs, representatives_dir))

    manifest = {
        "task": "tail_threshold",
        "run_tag": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "dataset_count": len({str(row.get('dataset_id') or '') for row in asset_rows if row.get('dataset_id')}),
        "asset_count": len(asset_rows),
        "latest_only": latest_only,
        "synthetic_root_filter": [str(item) for item in (synthetic_root_filter or []) if str(item).strip()],
        "threshold_percentages": [spec.pct for spec in threshold_specs],
        "threshold_labels": [spec.label for spec in threshold_specs],
        "representative_dataset_count": len(representative_rows),
        "representative_datasets": representative_rows,
        "source_run_dir": str(source_run_dir.resolve()) if source_run_dir is not None else None,
        "figure_count": len(figure_paths),
        "figures": figure_paths,
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def build_tail_threshold_preview(
    *,
    source_run_dir: Path,
    run_tag: str | None = None,
    latest_only: bool = True,
    threshold_percentages: list[float] | None = None,
    representatives_per_prefix: int = DEFAULT_REPRESENTATIVES_PER_PREFIX,
) -> dict[str, Any]:
    source_dir = source_run_dir.expanduser().resolve()
    asset_rows, diagnostic_rows, dataset_manifest_rows = _load_existing_dataset_outputs(source_dir)
    if not asset_rows:
        raise FileNotFoundError(f"No dataset-level tail-threshold outputs found under: {source_dir}")
    threshold_specs = _infer_threshold_specs_from_rows(asset_rows, fallback_percentages=threshold_percentages)
    out_run_dir = make_task_run_dir("tail_threshold", run_tag or f"{source_dir.name}__preview")
    return _materialize_tail_threshold_outputs(
        run_dir=out_run_dir,
        asset_rows=asset_rows,
        diagnostic_rows=diagnostic_rows,
        dataset_manifest_rows=dataset_manifest_rows,
        threshold_specs=threshold_specs,
        latest_only=latest_only,
        representatives_per_prefix=representatives_per_prefix,
        source_run_dir=source_dir,
        synthetic_root_filter=None,
    )


def run_tail_threshold_experiment(
    *,
    run_tag: str | None = None,
    datasets: list[str] | None = None,
    latest_only: bool = True,
    root_names: tuple[str, ...] | list[str] | None = None,
    threshold_percentages: list[float] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    numeric_bins: int = DEFAULT_NUMERIC_BINS,
    representatives_per_prefix: int = DEFAULT_REPRESENTATIVES_PER_PREFIX,
) -> dict[str, Any]:
    dataset_ids = datasets or list_dataset_ids()
    threshold_specs = _threshold_specs(threshold_percentages)
    run_dir = make_task_run_dir("tail_threshold", run_tag or now_run_tag())
    normalized_root_names = tuple(str(item).strip() for item in (root_names or []) if str(item).strip())
    assets = discover_synthetic_assets(
        datasets=dataset_ids,
        latest_only=latest_only,
        root_names=normalized_root_names,
    )

    asset_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    dataset_manifest_rows: list[dict[str, Any]] = []

    dataset_asset_map = {dataset_id: [asset for asset in assets if asset.dataset_id == dataset_id] for dataset_id in dataset_ids}
    progress = TaskProgressTracker(
        task_name="tail_threshold",
        total_steps=len(dataset_ids),
        step_label="datasets",
        substep_label="assets",
        total_substeps=sum(len(dataset_asset_map.get(dataset_id, [])) for dataset_id in dataset_ids),
    )
    progress.print_start(
        extra=(
            f"run_dir={run_dir.resolve()} | thresholds={','.join(spec.label for spec in threshold_specs)} "
            f"| latest_only={latest_only}"
            f" | roots={','.join(normalized_root_names) if normalized_root_names else 'all'}"
        )
    )

    def _consume(
        dataset_id: str,
        dataset_asset_rows: list[dict[str, Any]],
        dataset_diagnostic_rows: list[dict[str, Any]],
        manifest_row: dict[str, Any],
    ) -> None:
        dataset_manifest_rows.append(manifest_row)
        progress.advance(
            step_name=dataset_id,
            substeps_done=int(manifest_row.get("asset_count") or 0),
            extra=f"status={manifest_row.get('status')}",
        )
        asset_rows.extend(dataset_asset_rows)
        diagnostic_rows.extend(dataset_diagnostic_rows)
        if dataset_asset_rows:
            write_csv(
                run_dir / "datasets" / dataset_id / f"tail_threshold_asset_scores__{dataset_id}.csv",
                dataset_asset_rows,
            )
        if dataset_diagnostic_rows:
            write_csv(
                run_dir / "datasets" / dataset_id / f"tail_threshold_real_diagnostics__{dataset_id}.csv",
                dataset_diagnostic_rows,
            )

    if max_workers > 1 and len(dataset_ids) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_dataset_threshold_sweep,
                    dataset_id,
                    dataset_asset_map.get(dataset_id, []),
                    threshold_specs,
                    numeric_bins,
                ): dataset_id
                for dataset_id in dataset_ids
            }
            for future in as_completed(futures):
                _consume(*future.result())
    else:
        for dataset_id in dataset_ids:
            _consume(
                *_run_dataset_threshold_sweep(
                    dataset_id,
                    dataset_asset_map.get(dataset_id, []),
                    threshold_specs,
                    numeric_bins,
                )
            )

    return _materialize_tail_threshold_outputs(
        run_dir=run_dir,
        asset_rows=asset_rows,
        diagnostic_rows=diagnostic_rows,
        dataset_manifest_rows=dataset_manifest_rows,
        threshold_specs=threshold_specs,
        latest_only=latest_only,
        representatives_per_prefix=representatives_per_prefix,
        source_run_dir=None,
        synthetic_root_filter=normalized_root_names,
    )
