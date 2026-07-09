#!/usr/bin/env python3
"""Plot family sub-item visuals (v0.4-style layout) from current experiment outputs.

Family sub-item layout:
- subgroup: 2
- conditional: 3
- tail: 3
- missingness: 2

Notes:
- This script derives sub-item values from query_scores details.
- It prefers richer v0.4.1 detail keys (`profile_score`, `key_set_score`) when present,
  and falls back to legacy (`set_score`, `row_count_score`, `column_score`) for compatibility.
- It is visualization-oriented and does not rewrite scoring logic.
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

FAMILY_SUBITEMS: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "subgroup_structure": [
        ("sg_internal_profile", "Internal Profile", ("profile_score", "key_set_score", "set_score")),
        ("sg_size_stability", "Size Stability", ("row_count_score",)),
    ],
    "conditional_dependency_structure": [
        ("cd_slice_consistency", "Slice Consistency", ("profile_score", "key_set_score", "set_score")),
        ("cd_support_stability", "Support Stability", ("row_count_score",)),
        ("cd_schema_match", "Schema Match", ("column_score",)),
    ],
    "tail_rarity_structure": [
        ("tr_set_consistency", "Tail Set", ("key_set_score", "profile_score", "set_score")),
        ("tr_mass_similarity", "Tail Mass", ("row_count_score",)),
        ("tr_concentration_consistency", "Tail Concentration", ("column_score",)),
    ],
    "missingness_structure": [
        ("ms_marginal_consistency", "Marginal Missing", ("profile_score", "set_score")),
        ("ms_comissing_consistency", "Co-missing", ("row_count_score",)),
    ],
}

FAMILY_LABEL = {
    "subgroup_structure": "Subgroup",
    "conditional_dependency_structure": "Conditional",
    "tail_rarity_structure": "Tail/Rarity",
    "missingness_structure": "Missingness",
}


@dataclass
class DatasetView:
    dataset_id: str
    experiment_dir: Path
    primary_run_id: str
    model_ids: list[str]
    # model -> subitem_key -> value
    subitem_scores: dict[str, dict[str, float]]
    # family query counts for diagnostics
    family_counts: dict[str, dict[str, int]]


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


def _cmap() -> LinearSegmentedColormap:
    # closer to user's preferred style
    return LinearSegmentedColormap.from_list(
        "green_tile",
        ["#f7f7f3", "#acb37f", "#76be72", "#53b260"],
    )


def _model_order_from_model_scores(exp_dir: Path, dataset_id: str, primary_run_id: str) -> list[str]:
    score_path = exp_dir / f"model_scores_{dataset_id}.csv"
    seen: list[str] = []
    if score_path.exists():
        with score_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if str(row.get("workload_run_id") or "") != primary_run_id:
                    continue
                m = str(row.get("model_id") or "").strip().lower()
                if m and m not in seen:
                    seen.append(m)
    ordered = [m for m in MODEL_ORDER if m in seen]
    tail = [m for m in seen if m not in ordered]
    return ordered + tail


def _load_dataset_view(dataset_id: str, exp_dir: Path) -> DatasetView:
    sw = json.loads((exp_dir / f"selected_workloads_{dataset_id}.json").read_text(encoding="utf-8"))
    primary = str(sw.get("primary_workload_run_id") or "")
    query_path = exp_dir / "score_tables" / primary / "query_scores.jsonl"
    if not query_path.exists():
        raise FileNotFoundError(f"query_scores.jsonl not found: {query_path}")

    model_ids = _model_order_from_model_scores(exp_dir, dataset_id, primary)

    # accum[model][subitem_key] = list[float]
    accum: dict[str, dict[str, list[float]]] = {}
    family_counts: dict[str, dict[str, int]] = {}

    with query_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fam = str(row.get("family_id") or "").strip()
            if fam not in FAMILY_SUBITEMS:
                continue
            model = str(row.get("model_id") or "").strip().lower()
            if not model:
                continue
            details = row.get("details") or {}

            if model not in accum:
                accum[model] = {}
            if model not in family_counts:
                family_counts[model] = {}
            family_counts[model][fam] = family_counts[model].get(fam, 0) + 1

            for subkey, _label, detail_keys in FAMILY_SUBITEMS[fam]:
                raw_val = None
                for dk in detail_keys:
                    if dk in details and details.get(dk) is not None:
                        raw_val = details.get(dk)
                        break
                val = _as_float(raw_val, np.nan)
                if subkey not in accum[model]:
                    accum[model][subkey] = []
                if not np.isnan(val):
                    accum[model][subkey].append(float(val))

    # aggregate means
    subitem_scores: dict[str, dict[str, float]] = {}
    all_subkeys = [k for fam in FAMILY_SUBITEMS.values() for k, _, _ in fam]
    for m in model_ids:
        subitem_scores[m] = {}
        for subkey in all_subkeys:
            vals = (accum.get(m, {}) or {}).get(subkey, [])
            subitem_scores[m][subkey] = float(mean(vals)) if vals else np.nan

    return DatasetView(
        dataset_id=dataset_id,
        experiment_dir=exp_dir,
        primary_run_id=primary,
        model_ids=model_ids,
        subitem_scores=subitem_scores,
        family_counts=family_counts,
    )


def _collect_global_model_order(views: list[DatasetView]) -> list[str]:
    seen = set()
    for v in views:
        seen.update(v.model_ids)
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
    family_group_breaks: list[int] | None = None,
    family_group_labels: list[tuple[int, int, str]] | None = None,
) -> None:
    fig_w = max(9.5, 1.0 * len(col_labels) + 4.4)
    fig_h = max(5.6, 0.72 * len(row_labels) + 2.4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    cmap = _cmap().copy()
    cmap.set_bad(color="#ECEFF3")
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_facecolor("white")

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11, color="#192740", fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, color="#192740", rotation=0)
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, fontweight="bold", color="#6D7688")
            else:
                ax.text(j, i, f"{float(v):.3f}", ha="center", va="center", fontsize=9.6, fontweight="bold", color="#10263B")

    if family_group_breaks:
        for gb in family_group_breaks:
            ax.axvline(gb - 0.5, color="#223145", linewidth=2.5)

    if family_group_labels:
        for start, end, label in family_group_labels:
            center = (start + end) / 2.0
            ax.text(center, -1.05, label, ha="center", va="bottom", fontsize=12, fontweight="bold", color="#15233A")

    ax.set_title(title, fontsize=20, fontweight="bold", color="#162238", pad=16)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", fontsize=10.3, color="#5B6678")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset_full_subitems(view: DatasetView, model_order: list[str], out_path: Path) -> None:
    flat_cols: list[tuple[str, str, str]] = []
    group_labels: list[tuple[int, int, str]] = []
    group_breaks: list[int] = []
    idx = 0
    for fam in ["subgroup_structure", "conditional_dependency_structure", "tail_rarity_structure", "missingness_structure"]:
        items = FAMILY_SUBITEMS[fam]
        start = idx
        for subkey, sublabel, _ in items:
            flat_cols.append((fam, subkey, sublabel))
            idx += 1
        end = idx - 1
        group_labels.append((start, end, FAMILY_LABEL[fam]))
        if idx < sum(len(v) for v in FAMILY_SUBITEMS.values()):
            group_breaks.append(idx)

    mat = []
    for m in model_order:
        row = []
        rec = view.subitem_scores.get(m, {})
        for _fam, subkey, _label in flat_cols:
            row.append(rec.get(subkey, np.nan))
        mat.append(row)
    arr = np.array(mat, dtype=float)

    row_labels = [MODEL_LABEL.get(m, m) for m in model_order]
    col_labels = [c[2] for c in flat_cols]
    subtitle = f"Primary workload: {view.primary_run_id} | Sub-items: 2/3/3/2"
    _plot_matrix(
        arr,
        row_labels,
        col_labels,
        title=f"{view.dataset_id.upper()} Family Sub-item Heatmap",
        subtitle=subtitle,
        out_path=out_path,
        family_group_breaks=group_breaks,
        family_group_labels=group_labels,
    )


def _plot_family_cross_dataset(
    views: list[DatasetView],
    family_id: str,
    model_order: list[str],
    out_path: Path,
) -> None:
    subitems = FAMILY_SUBITEMS[family_id]
    rows = len(views)
    fig, axes = plt.subplots(rows, 1, figsize=(10.6, 2.25 * rows + 1.6), facecolor="white")
    if rows == 1:
        axes = [axes]

    cmap = _cmap().copy()
    cmap.set_bad(color="#ECEFF3")

    for ax, view in zip(axes, views):
        mat = []
        for m in model_order:
            rec = view.subitem_scores.get(m, {})
            mat.append([rec.get(k, np.nan) for k, _lbl, _dk in subitems])
        arr = np.array(mat, dtype=float)
        im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

        ax.set_title(f"{view.dataset_id.upper()}  ({view.primary_run_id})", loc="left", fontsize=11.5, fontweight="bold", color="#13223A")
        ax.set_yticks(np.arange(len(model_order)))
        ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in model_order], fontsize=9, color="#1f2f47")
        ax.set_xticks(np.arange(len(subitems)))
        ax.set_xticklabels([lbl for _k, lbl, _dk in subitems], fontsize=10, color="#1f2f47", fontweight="bold")
        ax.tick_params(axis="both", which="both", length=0)

        ax.set_xticks(np.arange(arr.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(arr.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isnan(v):
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#6D7688")
                else:
                    ax.text(j, i, f"{float(v):.3f}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#10263B")

    fig.suptitle(f"{FAMILY_LABEL[family_id]}: Sub-item Breakdown Across Datasets", fontsize=17.5, fontweight="bold", color="#15233A")
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.01, 0.02, 0.98, 0.95])
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_subitem_single(
    views: list[DatasetView],
    family_id: str,
    subitem_key: str,
    subitem_label: str,
    model_order: list[str],
    out_path: Path,
) -> None:
    mat = []
    ds_labels = []
    for view in views:
        ds_labels.append(view.dataset_id.upper())
    for m in model_order:
        row = []
        for view in views:
            row.append(view.subitem_scores.get(m, {}).get(subitem_key, np.nan))
        mat.append(row)
    arr = np.array(mat, dtype=float)
    _plot_matrix(
        arr,
        [MODEL_LABEL.get(m, m) for m in model_order],
        ds_labels,
        title=f"{FAMILY_LABEL[family_id]} / {subitem_label}",
        subtitle="Single sub-item across datasets",
        out_path=out_path,
    )


def run(args: argparse.Namespace) -> Path:
    pairs = []
    for item in args.dataset_experiments:
        if "=" not in item:
            raise ValueError(f"Invalid dataset pair: {item}")
        ds, p = item.split("=", 1)
        pairs.append((ds.strip().lower(), Path(p).expanduser().resolve()))

    views = [_load_dataset_view(ds, exp_dir) for ds, exp_dir in pairs]
    model_order = _collect_global_model_order(views)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.output_dir or (PROJECT_ROOT / "logs" / "analysis" / f"family_subitems_v04_{ts}")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []

    # A) per-dataset full chart
    for view in views:
        out = out_dir / f"dataset_{view.dataset_id}_family_subitems.png"
        _plot_dataset_full_subitems(view, model_order, out)
        generated.append(str(out))

    # B) per-family panel across datasets
    for fam in ["subgroup_structure", "conditional_dependency_structure", "tail_rarity_structure", "missingness_structure"]:
        out = out_dir / f"family_{fam}_subitems_cross_dataset.png"
        _plot_family_cross_dataset(views, fam, model_order, out)
        generated.append(str(out))

    # C) each sub-item as a standalone chart
    for fam in ["subgroup_structure", "conditional_dependency_structure", "tail_rarity_structure", "missingness_structure"]:
        for subkey, sublabel, _detail in FAMILY_SUBITEMS[fam]:
            out = out_dir / f"single_{fam}_{subkey}.png"
            _plot_subitem_single(views, fam, subkey, sublabel, model_order, out)
            generated.append(str(out))

    # write summary table (csv)
    csv_path = out_dir / "subitem_scores_table.csv"
    all_subkeys = []
    for fam in ["subgroup_structure", "conditional_dependency_structure", "tail_rarity_structure", "missingness_structure"]:
        all_subkeys.extend([k for k, _lbl, _d in FAMILY_SUBITEMS[fam]])
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_id", "primary_run_id", "model_id"] + all_subkeys)
        for view in views:
            for m in model_order:
                rec = view.subitem_scores.get(m, {})
                row = [view.dataset_id, view.primary_run_id, m] + [rec.get(k, np.nan) for k in all_subkeys]
                writer.writerow(row)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "datasets": [v.dataset_id for v in views],
        "experiments": {v.dataset_id: str(v.experiment_dir) for v in views},
        "primary_run_ids": {v.dataset_id: v.primary_run_id for v in views},
        "model_order": model_order,
        "output_dir": str(out_dir),
        "generated_images": generated,
        "table_csv": str(csv_path),
        "definition_note": (
            "Sub-items are operationally derived from current query_scores details: "
            "profile_score / key_set_score / set_score / row_count_score / column_score, grouped by family."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v0.4 family sub-item visuals.")
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
