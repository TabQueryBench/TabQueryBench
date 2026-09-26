#!/usr/bin/env python3
"""Generate compact model capability heatmaps for multiple datasets.

Compact layout:
- Analytics: Subgroup, Conditional, Tail/Rarity, Missingness
- Validation: Cardinality, Missing Intro, Uniqueness
- Reliability: Query Success

Rules:
- Keep only one Subgroup column.
- If missingness is not applicable (blank/None), render as N/A.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
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

METRICS = [
    # analytics
    ("subgroup_structure_score", "Subgroup"),
    ("conditional_dependency_structure_score", "Conditional"),
    ("tail_rarity_structure_score", "Tail/Rarity"),
    ("missingness_structure_score", "Missingness"),
    # validation
    ("validation_cardinality_range_score", "Cardinality"),
    ("validation_missing_introduction_score", "Missing Intro"),
    ("validation_uniqueness_integrity_score", "Uniqueness"),
    # reliability
    ("query_success_rate", "Query Success"),
]


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
    return LinearSegmentedColormap.from_list(
        "green_tile",
        ["#F7F9F5", "#A8B17A", "#72BC6D", "#54B35F"],
    )


def _load_primary_scores(exp_dir: Path, dataset_id: str) -> tuple[str, dict[str, dict[str, float]]]:
    sw = json.loads((exp_dir / f"selected_workloads_{dataset_id}.json").read_text(encoding="utf-8"))
    primary = str(sw.get("primary_workload_run_id") or "")
    rows = []
    with (exp_dir / f"model_scores_{dataset_id}.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("workload_run_id") or "") == primary:
                rows.append(row)

    scores: dict[str, dict[str, float]] = {}
    for row in rows:
        model = str(row.get("model_id") or "").strip().lower()
        if not model:
            continue
        rec: dict[str, float] = {}
        for key, _ in METRICS:
            rec[key] = _as_float_or_nan(row.get(key))
        scores[model] = rec
    return primary, scores


def _plot_dataset(dataset_id: str, primary_run_id: str, scores: dict[str, dict[str, float]], out_path: Path) -> None:
    models_present = [m for m in MODEL_ORDER if m in scores]
    if not models_present:
        models_present = sorted(scores.keys())
    model_labels = [MODEL_LABEL.get(m, m) for m in models_present]

    mat = []
    for m in models_present:
        rec = scores.get(m, {})
        mat.append([rec.get(k, np.nan) for k, _ in METRICS])
    arr = np.array(mat, dtype=float)

    fig_w = max(11.0, 0.95 * len(METRICS) + 5.2)
    fig_h = max(5.9, 0.72 * len(models_present) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

    cmap = _cmap().copy()
    cmap.set_bad(color="#ECEFF3")
    im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_facecolor("white")

    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=13, color="#172338", fontweight="bold")
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([label for _, label in METRICS], fontsize=11, color="#172338")
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_xticks(np.arange(arr.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(arr.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Group separators
    ax.axvline(3.5, color="#20314A", linewidth=2.6)  # analytics | validation
    ax.axvline(6.5, color="#20314A", linewidth=2.6)  # validation | reliability

    # group labels
    ax.text(1.5, -1.15, "Analytics", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#1A2A45")
    ax.text(5.0, -1.15, "Validation", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#1A2A45")
    ax.text(7.0, -1.15, "Reliability", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#1A2A45")

    # values
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=10, fontweight="bold", color="#6D7688")
            else:
                ax.text(j, i, f"{float(v):.3f}", ha="center", va="center", fontsize=11, fontweight="bold", color="#10263B")

    title = f"Model Capability Heatmap: {dataset_id.upper()}"
    subtitle = f"Primary workload: {primary_run_id}  |  Subgroup kept as single column"
    ax.set_title(title, fontsize=28, fontweight="bold", color="#162238", pad=20)
    fig.text(0.5, 0.93, subtitle, ha="center", fontsize=11, color="#5D6677")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    pairs = []
    for item in args.dataset_experiments:
        if "=" not in item:
            raise ValueError(f"Invalid item: {item}")
        ds, exp = item.split("=", 1)
        pairs.append((ds.strip().lower(), Path(exp).expanduser().resolve()))

    out_dir = (args.output_dir or (PROJECT_ROOT.parent.parent / "Query" / "code" / "logs" / "analysis" / f"compact_heatmaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(out_dir),
        "datasets": [],
    }

    for ds, exp_dir in pairs:
        primary, scores = _load_primary_scores(exp_dir, ds)
        out_path = out_dir / f"dataset_{ds}_compact.png"
        _plot_dataset(ds, primary, scores, out_path)
        manifest["datasets"].append(
            {
                "dataset_id": ds,
                "experiment_dir": str(exp_dir),
                "primary_run_id": primary,
                "image": str(out_path),
            }
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot compact model capability heatmaps.")
    parser.add_argument("--dataset-experiments", nargs="+", required=True, help="dataset_id=/path/to/experiment")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

