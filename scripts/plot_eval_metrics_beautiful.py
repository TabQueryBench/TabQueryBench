#!/usr/bin/env python3
"""Generate cleaner evaluation visuals (no PPT writing).

Outputs:
- Per-dataset detailed heatmaps (major groups + sub-metrics)
- Cross-dataset major-group sub-metric panels
- Cross-dataset major summary heatmaps
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ORDER = [
    "realtabformer",
    "bayesnet",
    "arf",
    "tvae",
    "ctgan",
    "tabpfgen",
    "tabddpm",
]
MODEL_LABEL = {
    "realtabformer": "RealTabFormer",
    "bayesnet": "BayesNet",
    "arf": "ARF",
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "tabpfgen": "TabPFGen",
    "tabddpm": "TabDDPM",
}

METRIC_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Analytics": [
        ("subgroup_count_component_score", "Subgroup (Count)"),
        ("subgroup_ratio_component_score", "Subgroup (Ratio)"),
        ("subgroup_structure_score", "Subgroup"),
        ("conditional_dependency_structure_score", "Conditional"),
        ("tail_rarity_structure_score", "Tail/Rarity"),
        ("missingness_structure_score", "Missingness"),
    ],
    "Validation": [
        ("validation_cardinality_range_score", "Cardinality"),
        ("validation_missing_introduction_score", "Missing Intro"),
        ("validation_uniqueness_integrity_score", "Uniqueness"),
    ],
    "Reliability": [
        ("query_success_rate", "Query Success"),
    ],
}


@dataclass
class DatasetScore:
    dataset_id: str
    experiment_dir: Path
    primary_workload_run_id: str
    # model_id -> metric_key -> value
    scores: dict[str, dict[str, float]]
    # model_id -> subgroup sub-component details
    subgroup_components: dict[str, dict[str, float | int]]


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def _as_float_or_nan(v: Any) -> float:
    if v is None:
        return np.nan
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _cmap() -> LinearSegmentedColormap:
    # white -> olive -> green, closer to your sample style
    return LinearSegmentedColormap.from_list(
        "sqlagent_green",
        ["#F7F9F5", "#A8B17A", "#72BC6D", "#54B35F"],
    )


def _derive_subgroup_components(exp_dir: Path, primary_workload_run_id: str) -> dict[str, dict[str, float | int]]:
    """Derive subgroup count/ratio components from query-level scores."""
    query_path = exp_dir / "score_tables" / primary_workload_run_id / "query_scores.jsonl"
    if not query_path.exists():
        return {}

    count_tokens = ("count", "support", "total", "freq", "n")
    ratio_tokens = ("ratio", "rate", "proportion", "pct", "percent", "percentage", "share")
    per_model: dict[str, dict[str, Any]] = {}

    with query_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row.get("family_id") or "") != "subgroup_structure":
                continue
            model = str(row.get("model_id") or "").strip().lower()
            if not model:
                continue
            if model not in per_model:
                per_model[model] = {
                    "count_scores": [],
                    "ratio_scores": [],
                    "subgroup_query_count": 0,
                    "subgroup_ratio_query_count": 0,
                }
            details = row.get("details") or {}
            cols = [str(c).lower() for c in (details.get("real_columns") or [])]
            q_score = _as_float(row.get("query_score"), 0.0)
            per_model[model]["subgroup_query_count"] += 1

            has_count = any(any(tok in col for tok in count_tokens) for col in cols)
            has_ratio = any(any(tok in col for tok in ratio_tokens) for col in cols)

            if has_count:
                per_model[model]["count_scores"].append(q_score)
            if has_ratio:
                per_model[model]["ratio_scores"].append(q_score)
                per_model[model]["subgroup_ratio_query_count"] += 1

    out: dict[str, dict[str, float | int]] = {}
    for model, bucket in per_model.items():
        count_scores = bucket["count_scores"]
        ratio_scores = bucket["ratio_scores"]
        total_n = int(bucket["subgroup_query_count"])
        ratio_n = int(bucket["subgroup_ratio_query_count"])
        out[model] = {
            "subgroup_count_component_score": float(mean(count_scores)) if count_scores else 0.0,
            "subgroup_ratio_component_score": float(mean(ratio_scores)) if ratio_scores else 0.0,
            "subgroup_ratio_component_coverage": float(ratio_n / total_n) if total_n > 0 else 0.0,
            "subgroup_query_count": total_n,
            "subgroup_ratio_query_count": ratio_n,
        }
    return out


def _read_primary_scores(exp_dir: Path, dataset_id: str) -> DatasetScore:
    sw = json.loads((exp_dir / f"selected_workloads_{dataset_id}.json").read_text(encoding="utf-8"))
    primary = str(sw.get("primary_workload_run_id") or "")
    rows: list[dict[str, str]] = []
    with (exp_dir / f"model_scores_{dataset_id}.csv").open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if str(row.get("workload_run_id") or "") == primary:
                rows.append(row)

    scores: dict[str, dict[str, float]] = {}
    for row in rows:
        model = str(row.get("model_id") or "").strip().lower()
        if not model:
            continue
        rec = {}
        for k, v in row.items():
            if k in {"workload_run_id", "model_id", "n_repeats"}:
                continue
            rec[k] = _as_float(v, 0.0)
        scores[model] = rec

    subgroup_components = _derive_subgroup_components(exp_dir=exp_dir, primary_workload_run_id=primary)
    for model_id, comps in subgroup_components.items():
        if model_id not in scores:
            continue
        scores[model_id]["subgroup_count_component_score"] = _as_float(comps.get("subgroup_count_component_score"), 0.0)
        scores[model_id]["subgroup_ratio_component_score"] = _as_float(comps.get("subgroup_ratio_component_score"), 0.0)
        scores[model_id]["subgroup_ratio_component_coverage"] = _as_float(comps.get("subgroup_ratio_component_coverage"), 0.0)

    return DatasetScore(
        dataset_id=dataset_id,
        experiment_dir=exp_dir,
        primary_workload_run_id=primary,
        scores=scores,
        subgroup_components=subgroup_components,
    )


def _present_models(ds_list: list[DatasetScore]) -> list[str]:
    seen = set()
    for ds in ds_list:
        seen.update(ds.scores.keys())
    ordered = [m for m in MODEL_ORDER if m in seen]
    tail = sorted(seen - set(ordered))
    return ordered + tail


def _plot_matrix(
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    subtitle: str,
    out_path: Path,
    group_breaks: list[int] | None = None,
    group_labels: list[tuple[int, int, str]] | None = None,
) -> None:
    fig_w = max(8.5, 1.05 * len(col_labels) + 4.2)
    fig_h = max(5.4, 0.7 * len(row_labels) + 2.3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    cmap = _cmap().copy()
    cmap.set_bad(color="#ECEFF3")
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_facecolor("white")

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11, color="#172338", fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, color="#172338", rotation=0)
    ax.tick_params(axis="both", which="both", length=0)

    # cell borders
    ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#6D7688")
            else:
                vv = float(v)
                color = "#0f2a3a" if vv >= 0.5 else "#23344a"
                ax.text(j, i, f"{vv:.3f}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    # group separators
    if group_breaks:
        for gb in group_breaks:
            ax.axvline(gb - 0.5, color="#223145", linewidth=2.4)

    # group labels
    if group_labels:
        for start, end, label in group_labels:
            center = (start + end) / 2.0
            ax.text(
                center,
                -1.1,
                label,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="#1A2A45",
            )

    ax.set_title(title, fontsize=20, fontweight="bold", color="#162238", pad=18)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", fontsize=10.5, color="#5b677a")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_major_submetric_panel(
    ds_list: list[DatasetScore],
    models: list[str],
    major_name: str,
    submetrics: list[tuple[str, str]],
    out_path: Path,
) -> None:
    ds_ids = [x.dataset_id.upper() for x in ds_list]
    rows = len(submetrics)
    fig, axes = plt.subplots(rows, 1, figsize=(10.5, 2.4 * rows + 1.6), facecolor="white")
    if rows == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, submetrics):
        mat = []
        for m in models:
            row = []
            for ds in ds_list:
                row.append(_as_float(ds.scores.get(m, {}).get(key), 0.0))
            mat.append(row)
        arr = np.array(mat, dtype=float)
        im = ax.imshow(arr, cmap=_cmap(), vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=9, color="#1f2f47")
        ax.set_xticks(np.arange(len(ds_ids)))
        ax.set_xticklabels(ds_ids, fontsize=10, fontweight="bold", color="#1f2f47")
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color="#15233A", pad=7)
        ax.tick_params(axis="both", which="both", length=0)

        ax.set_xticks(np.arange(arr.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(arr.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isnan(v):
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#6D7688")
                else:
                    ax.text(j, i, f"{float(v):.3f}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#10273A")

    fig.suptitle(f"{major_name}: Sub-metric Breakdown (Cross-Dataset)", fontsize=18, fontweight="bold", color="#14223A")
    fig.text(0.5, 0.015, "Rows = models, Columns = datasets", ha="center", fontsize=10, color="#5A6778")
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.008)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.01, 0.03, 0.98, 0.96])
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_single_metric_cross_dataset(
    ds_list: list[DatasetScore],
    models: list[str],
    metric_key: str,
    metric_label: str,
    major_name: str,
    out_path: Path,
) -> None:
    ds_ids = [x.dataset_id.upper() for x in ds_list]
    mat = []
    for m in models:
        row = []
        for ds in ds_list:
            row.append(_as_float_or_nan(ds.scores.get(m, {}).get(metric_key)))
        mat.append(row)
    arr = np.array(mat, dtype=float)
    row_labels = [MODEL_LABEL.get(m, m) for m in models]
    _plot_matrix(
        arr,
        row_labels,
        ds_ids,
        title=f"{major_name} / {metric_label}",
        subtitle="Single sub-metric view across datasets",
        out_path=out_path,
        group_breaks=None,
        group_labels=None,
    )


def _plot_major_summary(ds_list: list[DatasetScore], models: list[str], out_path: Path) -> None:
    major_names = list(METRIC_GROUPS.keys())
    fig, axes = plt.subplots(1, len(ds_list), figsize=(4.8 * len(ds_list), 5.8), facecolor="white")
    if len(ds_list) == 1:
        axes = [axes]

    for ax, ds in zip(axes, ds_list):
        arr = []
        for m in models:
            rec = ds.scores.get(m, {})
            row = []
            for major in major_names:
                keys = [k for k, _ in METRIC_GROUPS[major]]
                vals = [_as_float_or_nan(rec.get(k)) for k in keys]
                valid = [x for x in vals if not np.isnan(x)]
                row.append(float(np.mean(valid)) if valid else np.nan)
            arr.append(row)
        arr_np = np.array(arr, dtype=float)

        im = ax.imshow(arr_np, cmap=_cmap(), vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(ds.dataset_id.upper(), fontsize=14, fontweight="bold", color="#15233A", pad=8)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=9, color="#1f2f47")
        ax.set_xticks(np.arange(len(major_names)))
        ax.set_xticklabels(major_names, fontsize=10, fontweight="bold", color="#1f2f47")
        ax.tick_params(axis="both", which="both", length=0)

        ax.set_xticks(np.arange(arr_np.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(arr_np.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(arr_np.shape[0]):
            for j in range(arr_np.shape[1]):
                v = arr_np[i, j]
                if np.isnan(v):
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#6D7688")
                else:
                    ax.text(j, i, f"{float(v):.3f}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#10273A")

    fig.suptitle("Major-Dimension Summary (mean of sub-metrics)", fontsize=18, fontweight="bold", color="#14223A")
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.01, 0.02, 0.98, 0.95])
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    ds_pairs = []
    for item in args.dataset_experiments:
        if "=" not in item:
            raise ValueError(f"Invalid item: {item}")
        ds, exp = item.split("=", 1)
        ds_pairs.append((ds.strip().lower(), Path(exp).expanduser().resolve()))

    ds_scores = [_read_primary_scores(exp, ds) for ds, exp in ds_pairs]
    models = _present_models(ds_scores)
    model_labels = [MODEL_LABEL.get(m, m) for m in models]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.output_dir or (PROJECT_ROOT / "logs" / "analysis" / f"eval_visuals_beautiful_{ts}")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []

    # 1) Per-dataset detailed matrix (major + sub-metrics)
    flat_metrics: list[tuple[str, str, str]] = []
    for major, items in METRIC_GROUPS.items():
        for key, label in items:
            flat_metrics.append((major, key, label))

    for ds in ds_scores:
        mat = []
        for m in models:
            rec = ds.scores.get(m, {})
            row = [_as_float_or_nan(rec.get(key)) for _, key, _ in flat_metrics]
            mat.append(row)
        arr = np.array(mat, dtype=float)
        col_labels = [x[2] for x in flat_metrics]

        group_labels = []
        group_breaks = []
        idx = 0
        for major, items in METRIC_GROUPS.items():
            start = idx
            end = idx + len(items) - 1
            group_labels.append((start, end, major))
            idx += len(items)
            if idx < len(flat_metrics):
                group_breaks.append(idx)

        out = out_dir / f"dataset_{ds.dataset_id}_major_submetrics.png"
        _plot_matrix(
            arr,
            model_labels,
            col_labels,
            title=f"Model Capability Heatmap: {ds.dataset_id.upper()}",
            subtitle=f"Primary workload: {ds.primary_workload_run_id} | Overall removed",
            out_path=out,
            group_breaks=group_breaks,
            group_labels=group_labels,
        )
        generated.append(str(out))

    # 2) Major -> sub-metric panels
    for major, items in METRIC_GROUPS.items():
        out = out_dir / f"major_{major.lower()}_submetrics_cross_dataset.png"
        _plot_major_submetric_panel(ds_scores, models, major, items, out)
        generated.append(str(out))
        for key, label in items:
            single_out = out_dir / f"single_{major.lower()}_{key}.png"
            _plot_single_metric_cross_dataset(ds_scores, models, key, label, major, single_out)
            generated.append(str(single_out))

    # 3) Major summary
    out_summary = out_dir / "major_summary_cross_dataset.png"
    _plot_major_summary(ds_scores, models, out_summary)
    generated.append(str(out_summary))

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "datasets": [x.dataset_id for x in ds_scores],
        "experiments": {x.dataset_id: str(x.experiment_dir) for x in ds_scores},
        "primary_workload_run_ids": {x.dataset_id: x.primary_workload_run_id for x in ds_scores},
        "models": models,
        "output_dir": str(out_dir),
        "generated_images": generated,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cleaner evaluation figures (no PPT write).")
    parser.add_argument(
        "--dataset-experiments",
        nargs="+",
        required=True,
        help="dataset_id=/abs/path/to/experiment_dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
