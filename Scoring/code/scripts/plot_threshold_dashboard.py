#!/usr/bin/env python3
"""Threshold dashboard with visualizations (ratio + score interval).

- Removes c3 by default.
- Default policy follows subgroup-vs-tail split: top 97% for subgroup.
- Legacy support-gate threshold sweeps are still available.
- Produces PNG charts + CSV tables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyze_subgroup_threshold_sensitivity import (
    PROJECT_ROOT,
    _build_group_stats,
    _collect_synthetic_csvs,
    _prepare_binning_maps,
    _read_csv_rows,
    _score_threshold_for_model,
    _tv_similarity,
)
from plot_six_dataset_subgroup_support import (
    _build_combos,
    _build_edges,
    _compute_support_counts,
    _load_field_registry,
    _select_subgroup_columns,
)


@dataclass
class ThresholdSpec:
    label: str
    mode: str
    abs_threshold: int | None = None
    pct_threshold: float | None = None
    cap: int | None = None

    def gate(self, n_real: int) -> int:
        if self.mode == "none":
            return 0
        if self.mode == "abs":
            return int(self.abs_threshold or 0)
        if self.mode == "pct":
            return int(math.ceil((self.pct_threshold or 0.0) * n_real))
        if self.mode == "max":
            return max(
                int(self.abs_threshold or 0),
                int(math.ceil((self.pct_threshold or 0.0) * n_real)),
            )
        if self.mode == "clamped_max":
            base = max(
                int(self.abs_threshold or 0),
                int(math.ceil((self.pct_threshold or 0.0) * n_real)),
            )
            return min(base, int(self.cap or base))
        raise ValueError(f"Unsupported mode: {self.mode}")


DEFAULT_DATASETS = ["c2", "c7", "m1", "m4", "m3", "n1", "n4"]
DEFAULT_SUBGROUP_KEEP_RATIO = 0.97


def _default_thresholds() -> list[ThresholdSpec]:
    return [
        ThresholdSpec("no_filter", "none"),
        ThresholdSpec("abs=5", "abs", abs_threshold=5),
        ThresholdSpec("abs=10", "abs", abs_threshold=10),
        ThresholdSpec("abs=20", "abs", abs_threshold=20),
        ThresholdSpec("abs=30", "abs", abs_threshold=30),
        ThresholdSpec("pct=0.1%", "pct", pct_threshold=0.001),
        ThresholdSpec("pct=0.2%", "pct", pct_threshold=0.002),
        ThresholdSpec("pct=0.5%", "pct", pct_threshold=0.005),
        ThresholdSpec("pct=1.0%", "pct", pct_threshold=0.01),
        ThresholdSpec("max(20,0.2%)", "max", abs_threshold=20, pct_threshold=0.002),
        ThresholdSpec("max(20,0.5%)", "max", abs_threshold=20, pct_threshold=0.005),
        ThresholdSpec("max(30,0.5%)", "max", abs_threshold=30, pct_threshold=0.005),
        ThresholdSpec("max(30,1.0%)", "max", abs_threshold=30, pct_threshold=0.01),
        ThresholdSpec(
            "min(max(20,0.2%),200)", "clamped_max", abs_threshold=20, pct_threshold=0.002, cap=200
        ),
        ThresholdSpec(
            "min(max(20,0.5%),200)", "clamped_max", abs_threshold=20, pct_threshold=0.005, cap=200
        ),
        ThresholdSpec(
            "min(max(15,0.2%),150)", "clamped_max", abs_threshold=15, pct_threshold=0.002, cap=150
        ),
        ThresholdSpec(
            "min(max(10,0.1%),120)", "clamped_max", abs_threshold=10, pct_threshold=0.001, cap=120
        ),
    ]


def _split_threshold_label(keep_ratio: float) -> str:
    top_pct = int(round(max(0.0, min(1.0, float(keep_ratio))) * 100))
    tail_pct = max(0, 100 - top_pct)
    return f"split{top_pct}_tail{tail_pct}"


def _infer_target_column(dataset_id: str, columns: list[str]) -> tuple[str, str]:
    sem_path = PROJECT_ROOT / "data" / dataset_id / "metadata" / "dataset_semantics.yaml"
    if sem_path.exists():
        for raw in sem_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("target_column:"):
                target = line.split(":", 1)[1].strip()
                if target in columns:
                    return target, "metadata"
    priors = ["class", "target", "label", "y", "outcome"]
    lower_map = {c.lower(): c for c in columns}
    for p in priors:
        if p in lower_map:
            return lower_map[p], "heuristic_name"
    return columns[-1], "fallback_last_column"


def _heatmap(df: pd.DataFrame, title: str, out: Path, vmin: float = 0.0, vmax: float = 1.0, cmap: str = "YlGnBu") -> None:
    if df.empty:
        return
    thresholds = df.index.tolist()
    datasets = df.columns.tolist()
    mat = df.to_numpy(dtype=float)

    fig_w = max(10, len(datasets) * 1.2)
    fig_h = max(8, len(thresholds) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([d.upper() for d in datasets], rotation=0)
    ax.set_yticks(np.arange(len(thresholds)))
    ax.set_yticklabels(thresholds)
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if (vmax - vmin) > 0 and val > (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_ratio_lines(ratio_long: pd.DataFrame, threshold_order: list[str], out: Path) -> None:
    if ratio_long.empty:
        return
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(threshold_order))

    for ds, g in ratio_long.groupby("dataset_id"):
        g2 = g.set_index("threshold_label").reindex(threshold_order)
        y = g2["eligible_ratio"].astype(float).to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, label=ds.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(threshold_order, rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Eligible ratio")
    ax.set_title("Eligible Ratio vs Threshold (7 datasets, c3 removed)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_score_band(score_long: pd.DataFrame, threshold_order: list[str], out: Path) -> None:
    if score_long.empty:
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(threshold_order))

    for ds, g in score_long.groupby("dataset_id"):
        g2 = g.set_index("threshold_label").reindex(threshold_order)
        y_min = g2["min_score"].astype(float).to_numpy()
        y_max = g2["max_score"].astype(float).to_numpy()
        y_mean = g2["mean_score"].astype(float).to_numpy()

        ax.fill_between(x, y_min, y_max, alpha=0.15)
        ax.plot(x, y_mean, marker="o", linewidth=2, label=f"{ds.upper()} mean")

    ax.set_xticks(x)
    ax.set_xticklabels(threshold_order, rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Score Interval Band (tabddpm excluded)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.selection_policy == "top_ratio":
        specs = [ThresholdSpec(_split_threshold_label(args.subgroup_keep_ratio), "none")]
    else:
        specs = _default_thresholds()
    threshold_order = [s.label for s in specs]

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_root = args.synthetic_root.expanduser().resolve()

    # -------- ratio (real only) --------
    ratio_rows: list[dict[str, Any]] = []
    ratio_availability: list[dict[str, Any]] = []

    for ds in datasets:
        real_csv = PROJECT_ROOT / "data" / ds / "raw" / f"{ds}-main.csv"
        if not real_csv.exists():
            ratio_availability.append({"dataset_id": ds, "status": "missing_real_csv"})
            continue

        cols, rows = _read_csv_rows(real_csv)
        n_real = len(rows)
        field_registry = _load_field_registry(ds)
        subgroup_cols = _select_subgroup_columns(field_registry, columns=cols, rows=rows, max_columns=args.max_columns)
        combos = _build_combos(subgroup_cols, max_combos=args.max_combos)
        edges_map = _build_edges(rows, subgroup_cols, bins=args.numeric_bins)
        supports = _compute_support_counts(rows, combos, edges_map)
        arr = np.asarray(supports, dtype=float)
        total_groups = len(supports)

        ratio_availability.append(
            {
                "dataset_id": ds,
                "status": "ok",
                "n_real": n_real,
                "subgroup_columns_used": "|".join(subgroup_cols),
                "combo_count": len(combos),
                "group_count": total_groups,
            }
        )

        for spec in specs:
            if args.selection_policy == "top_ratio":
                keep_ratio = max(0.0, min(1.0, float(args.subgroup_keep_ratio)))
                if total_groups <= 0:
                    eligible = 0
                    gate = 0
                else:
                    keep_n = max(1, int(math.ceil(total_groups * keep_ratio)))
                    ranked_desc = sorted((int(v) for v in supports if int(v) > 0), reverse=True)
                    eligible = min(keep_n, total_groups)
                    gate = int(ranked_desc[eligible - 1]) if ranked_desc else 0
            else:
                gate = spec.gate(n_real)
                eligible = int((arr >= gate).sum())
            ratio_rows.append(
                {
                    "dataset_id": ds,
                    "threshold_label": spec.label,
                    "gate": gate,
                    "eligible_groups": eligible,
                    "total_groups": total_groups,
                    "eligible_ratio": float(eligible / max(1, total_groups)),
                }
            )

    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df["threshold_label"] = pd.Categorical(ratio_df["threshold_label"], categories=threshold_order, ordered=True)
    ratio_df = ratio_df.sort_values(["dataset_id", "threshold_label"]).reset_index(drop=True)
    ratio_wide = ratio_df.pivot(index="threshold_label", columns="dataset_id", values="eligible_ratio")

    ratio_df.to_csv(out_dir / "threshold_ratio_long.csv", index=False)
    ratio_df.pivot(index="dataset_id", columns="threshold_label", values="eligible_ratio").reset_index().to_csv(
        out_dir / "threshold_ratio_wide.csv", index=False
    )
    pd.DataFrame(ratio_availability).to_csv(out_dir / "ratio_dataset_availability.csv", index=False)

    _heatmap(
        ratio_wide,
        title="Eligible Ratio Heatmap (c3 removed, includes no_filter + clamped rules)",
        out=out_dir / "eligible_ratio_heatmap.png",
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
    )
    _plot_ratio_lines(ratio_df, threshold_order, out_dir / "eligible_ratio_lines.png")

    # -------- score (synthetic, optional exclude tabddpm) --------
    score_rows: list[dict[str, Any]] = []
    score_availability: list[dict[str, Any]] = []

    for ds in datasets:
        real_csv = PROJECT_ROOT / "data" / ds / "raw" / f"{ds}-main.csv"
        if not real_csv.exists():
            score_availability.append({"dataset_id": ds, "status": "missing_real_csv"})
            continue

        cols_real, rows_real = _read_csv_rows(real_csv)
        if not cols_real or not rows_real:
            score_availability.append({"dataset_id": ds, "status": "empty_real"})
            continue

        syn_paths = _collect_synthetic_csvs(ds, synthetic_root, cols_real)
        if args.exclude_model:
            excludes = {x.strip().lower() for x in args.exclude_model.split(",") if x.strip()}
            syn_paths = [(m, p) for m, p in syn_paths if m.lower() not in excludes]

        if not syn_paths:
            score_availability.append({"dataset_id": ds, "status": "no_syn_after_exclusion", "usable_models": 0})
            continue

        target_col, target_source = _infer_target_column(ds, cols_real)
        field_registry = _load_field_registry(ds)
        subgroup_cols = _select_subgroup_columns(field_registry, columns=cols_real, rows=rows_real, max_columns=args.max_columns)
        combos = _build_combos(subgroup_cols, max_combos=args.max_combos)
        subgroup_flat = [c for combo in combos for c in combo]

        subgroup_edges, target_labels, target_edges = _prepare_binning_maps(
            rows_real=rows_real,
            subgroup_cols_flat=subgroup_flat,
            target_col=target_col,
            field_registry=field_registry,
            subgroup_bins=args.numeric_bins,
            target_bins=args.target_bins,
        )
        real_groups = _build_group_stats(
            rows=rows_real,
            combos=combos,
            target_col=target_col,
            subgroup_edges=subgroup_edges,
            target_edges=target_edges,
        )

        model_records: dict[str, list[dict[str, Any]]] = {}
        for model_id, syn_path in syn_paths:
            try:
                _, syn_rows = _read_csv_rows(syn_path)
            except Exception:
                continue

            syn_groups = _build_group_stats(
                rows=syn_rows,
                combos=combos,
                target_col=target_col,
                subgroup_edges=subgroup_edges,
                target_edges=target_edges,
            )

            recs = []
            for gk, (support_real, target_real) in real_groups.items():
                support_syn, target_syn = syn_groups.get(gk, (0, Counter()))
                similarity = _tv_similarity(target_real, target_syn, target_labels)
                retention = min(1.0, (support_syn / support_real)) if support_real > 0 else 0.0
                coverage = 1.0 if support_syn > 0 else 0.0
                recs.append(
                    {
                        "support_real": int(support_real),
                        "support_syn": int(support_syn),
                        "weight": math.sqrt(float(support_real)),
                        "similarity": float(similarity),
                        "retention": float(retention),
                        "coverage": float(coverage),
                    }
                )
            model_records[model_id] = recs

        if not model_records:
            score_availability.append({"dataset_id": ds, "status": "syn_read_failed", "usable_models": 0})
            continue

        score_availability.append(
            {
                "dataset_id": ds,
                "status": "ok",
                "usable_models": len(model_records),
                "target_column": target_col,
                "target_source": target_source,
            }
        )

        for spec in specs:
            model_scores = {}
            gate = spec.gate(len(rows_real))
            for model_id, recs in model_records.items():
                stat = _score_threshold_for_model(
                    records=recs,
                    abs_threshold=0 if spec.mode in {"none", "pct"} else int(spec.abs_threshold or 0),
                    pct_threshold=0.0 if spec.mode in {"none", "abs"} else float(spec.pct_threshold or 0.0),
                    n_real=len(rows_real),
                    selection_policy=args.selection_policy,
                    subgroup_keep_ratio=args.subgroup_keep_ratio,
                )
                # For clamped mode we override by recomputing with manual gate via filtered records
                if args.selection_policy != "top_ratio" and spec.mode == "clamped_max":
                    eligible = [r for r in recs if int(r["support_real"]) >= gate]
                    if eligible:
                        ws = [float(r["weight"]) for r in eligible]
                        sim = [float(r["similarity"]) for r in eligible]
                        ret = [float(r["retention"]) for r in eligible]
                        cov = [float(r["coverage"]) for r in eligible]
                        sw = sum(ws) if sum(ws) > 0 else 1.0
                        profile = sum(w * x for w, x in zip(ws, sim)) / sw
                        support = sum(w * x for w, x in zip(ws, ret)) / sw
                        coverage = sum(cov) / max(1, len(cov))
                        stat["subgroup_score"] = (profile + support + coverage) / 3.0
                    else:
                        stat["subgroup_score"] = 0.0
                model_scores[model_id] = float(stat["subgroup_score"])

            items = sorted(model_scores.items(), key=lambda kv: kv[1])
            worst_model, min_score = items[0]
            best_model, max_score = items[-1]
            score_rows.append(
                {
                    "dataset_id": ds,
                    "threshold_label": spec.label,
                    "gate": gate,
                    "usable_model_count": len(model_scores),
                    "min_score": float(min_score),
                    "max_score": float(max_score),
                    "mean_score": float(np.mean(list(model_scores.values()))),
                    "worst_model": worst_model,
                    "best_model": best_model,
                }
            )

    score_df = pd.DataFrame(score_rows)
    if not score_df.empty:
        score_df["threshold_label"] = pd.Categorical(score_df["threshold_label"], categories=threshold_order, ordered=True)
        score_df = score_df.sort_values(["dataset_id", "threshold_label"]).reset_index(drop=True)
        score_df.to_csv(out_dir / "threshold_score_range_long.csv", index=False)

        score_df2 = score_df.copy()
        score_df2["min_max"] = score_df2.apply(lambda r: f"{r['min_score']:.3f}~{r['max_score']:.3f}", axis=1)
        score_df2.pivot(index="dataset_id", columns="threshold_label", values="min_max").reset_index().to_csv(
            out_dir / "threshold_score_minmax_wide.csv", index=False
        )

        min_heat = score_df.pivot(index="threshold_label", columns="dataset_id", values="min_score")
        max_heat = score_df.pivot(index="threshold_label", columns="dataset_id", values="max_score")
        range_heat = score_df.assign(score_range=score_df["max_score"] - score_df["min_score"]).pivot(
            index="threshold_label", columns="dataset_id", values="score_range"
        )

        _heatmap(min_heat, "Min Score Heatmap (model panel)", out_dir / "score_min_heatmap.png", 0, 1, "YlOrRd")
        _heatmap(max_heat, "Max Score Heatmap (model panel)", out_dir / "score_max_heatmap.png", 0, 1, "YlGn")
        _heatmap(range_heat, "Score Range Heatmap (max-min)", out_dir / "score_range_heatmap.png", 0, 1, "PuBu")
        _plot_score_band(score_df, threshold_order, out_dir / "score_interval_band.png")

    pd.DataFrame(score_availability).to_csv(out_dir / "score_dataset_availability.csv", index=False)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "datasets": datasets,
        "threshold_order": threshold_order,
        "selection_policy": args.selection_policy,
        "subgroup_keep_ratio": args.subgroup_keep_ratio,
        "synthetic_root": str(synthetic_root),
        "exclude_model": args.exclude_model,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(out_dir)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build threshold dashboard plots.")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "SynData" / "synthetic_10ds_7models",
    )
    parser.add_argument("--exclude-model", type=str, default="tabddpm")
    parser.add_argument("--max-columns", type=int, default=6)
    parser.add_argument("--max-combos", type=int, default=18)
    parser.add_argument("--numeric-bins", type=int, default=4)
    parser.add_argument("--target-bins", type=int, default=5)
    parser.add_argument(
        "--selection-policy",
        type=str,
        choices=["top_ratio", "support_gate"],
        default="top_ratio",
        help="top_ratio keeps top subgroup supports (default split97_tail3); support_gate uses legacy threshold sweep.",
    )
    parser.add_argument(
        "--subgroup-keep-ratio",
        type=float,
        default=DEFAULT_SUBGROUP_KEEP_RATIO,
        help="When selection-policy=top_ratio, keep this ratio of subgroup supports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "analysis" / f"threshold_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
