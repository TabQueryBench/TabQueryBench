#!/usr/bin/env python3
"""Tail threshold sensitivity analysis (tail sub-metrics only).

This script evaluates how tail-support threshold choices affect:
1) tail set consistency
2) tail mass similarity
3) tail concentration consistency

Tail selection policies:
- support_gate (legacy): low-support keys with count <= gate
- bottom_ratio (default): tail is the bottom-ratio keys by support (e.g., bottom 3%)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBGROUP_KEEP_RATIO = 0.97


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


def _threshold_specs(exclude_no_filter: bool = True) -> list[ThresholdSpec]:
    specs = [
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
        ThresholdSpec("min(max(20,0.2%),200)", "clamped_max", abs_threshold=20, pct_threshold=0.002, cap=200),
        ThresholdSpec("min(max(20,0.5%),200)", "clamped_max", abs_threshold=20, pct_threshold=0.005, cap=200),
        ThresholdSpec("min(max(15,0.2%),150)", "clamped_max", abs_threshold=15, pct_threshold=0.002, cap=150),
        ThresholdSpec("min(max(10,0.1%),120)", "clamped_max", abs_threshold=10, pct_threshold=0.001, cap=120),
    ]
    if exclude_no_filter:
        specs = [s for s in specs if s.label != "no_filter"]
    return specs


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        headers = [str(h) for h in (reader.fieldnames or [])]
    return headers, rows


def _find_real_csv(dataset_id: str) -> Path | None:
    p1 = PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / dataset_id / "raw" / f"{dataset_id}-main.csv"
    if p1.exists():
        return p1
    p2 = PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / dataset_id / f"{dataset_id}-main.csv"
    if p2.exists():
        return p2
    return None


def _infer_model_id(path: Path, dataset_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(dataset_root.resolve())
        parts = rel.parts
        if parts:
            return parts[0].lower()
    except Exception:  # noqa: BLE001
        pass
    return path.parent.name.lower()


def _collect_synthetic_csvs(dataset_id: str, synthetic_root: Path, expected_columns: list[str]) -> list[tuple[str, Path]]:
    ds_root = synthetic_root / dataset_id
    if not ds_root.exists():
        return []
    expected = set(expected_columns)
    out: list[tuple[str, Path]] = []
    for p in sorted(ds_root.rglob("*.csv")):
        try:
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                r = csv.reader(f)
                header = next(r, [])
            if set(header) != expected:
                continue
            out.append((_infer_model_id(p, ds_root), p))
        except Exception:  # noqa: BLE001
            continue
    return out


def _load_target_column(dataset_id: str, columns: list[str]) -> str:
    sem_path = PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / dataset_id / "metadata" / "dataset_semantics.yaml"
    if sem_path.exists():
        for raw in sem_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("target_column:"):
                target = line.split(":", 1)[1].strip()
                if target in columns:
                    return target
    for p in ["class", "target", "label", "y", "outcome"]:
        for c in columns:
            if c.lower() == p:
                return c
    return columns[-1]


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in {"", "nan", "none", "null", "na", "n/a"}


def _is_id_like(name: str) -> bool:
    n = name.lower().strip()
    return n in {"id", "row_id", "index"} or n.endswith("_id")


def _safe_float(v: Any) -> float | None:
    try:
        if _is_missing(v):
            return None
        return float(str(v).strip())
    except Exception:
        return None


def _quantile_edges(values: list[float], bins: int) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(arr, qs).tolist()
    dedup: list[float] = []
    for x in edges:
        if not dedup or abs(float(x) - dedup[-1]) > 1e-12:
            dedup.append(float(x))
    return dedup


def _bin_numeric(x: float, edges: list[float]) -> str:
    if not edges or len(edges) < 2:
        return "q1"
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            if lo <= x <= hi:
                return f"q{i+1}"
        if lo <= x < hi:
            return f"q{i+1}"
    if x < edges[0]:
        return "below_q1"
    return f"above_q{len(edges)-1}"


def _build_transformers(rows_real: list[dict[str, str]], feature_cols: list[str], numeric_bins: int) -> dict[str, dict[str, Any]]:
    tx: dict[str, dict[str, Any]] = {}
    for col in feature_cols:
        vals = [r.get(col) for r in rows_real]
        n = max(1, len(vals))
        numeric_vals = [x for x in (_safe_float(v) for v in vals) if x is not None]
        num_ratio = len(numeric_vals) / n
        uniq_num = len(set(round(v, 8) for v in numeric_vals))
        is_numeric_cont = num_ratio >= 0.95 and uniq_num >= 20
        if is_numeric_cont:
            edges = _quantile_edges(numeric_vals, bins=numeric_bins)
            tx[col] = {"mode": "numeric_bin", "edges": edges}
        else:
            tx[col] = {"mode": "categorical"}
    return tx


def _tokenize(value: Any, rule: dict[str, Any]) -> str:
    if _is_missing(value):
        return "__MISSING__"
    mode = str(rule.get("mode") or "categorical")
    s = str(value).strip()
    if mode == "numeric_bin":
        fv = _safe_float(value)
        if fv is None:
            return "__MISSING__"
        return _bin_numeric(fv, rule.get("edges") or [])
    return s


def _build_key_counter(rows: list[dict[str, str]], feature_cols: list[str], transformers: dict[str, dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for row in rows:
        for col in feature_cols:
            token = _tokenize(row.get(col), transformers[col])
            key = f"{col}::{token}"
            c[key] += 1
    return c


def _tv_similarity_over_keys(real_counts: Counter, syn_counts: Counter, keys: set[str]) -> float:
    if not keys:
        return 1.0
    real_total = sum(real_counts.get(k, 0) for k in keys)
    syn_total = sum(syn_counts.get(k, 0) for k in keys)
    if real_total <= 0 and syn_total <= 0:
        return 1.0
    if real_total <= 0 or syn_total <= 0:
        return 0.0
    tv = 0.0
    for k in keys:
        pr = real_counts.get(k, 0) / real_total
        ps = syn_counts.get(k, 0) / syn_total
        tv += abs(pr - ps)
    tv *= 0.5
    return max(0.0, min(1.0, 1.0 - tv))


def _tail_metrics_for_model(
    *,
    real_counts: Counter,
    syn_counts: Counter,
    n_real: int,
    n_syn: int,
    gate: int,
    selection_policy: str,
    tail_ratio: float,
) -> dict[str, float]:
    if selection_policy == "bottom_ratio":
        ratio = max(0.0, min(1.0, float(tail_ratio)))
        real_items = sorted(((k, int(v)) for k, v in real_counts.items() if int(v) > 0), key=lambda kv: (kv[1], kv[0]))
        syn_items = sorted(((k, int(v)) for k, v in syn_counts.items() if int(v) > 0), key=lambda kv: (kv[1], kv[0]))
        k_real = max(1, int(math.ceil(len(real_items) * ratio))) if real_items else 0
        k_syn = max(1, int(math.ceil(len(syn_items) * ratio))) if syn_items else 0
        t_real = {k for k, _ in real_items[:k_real]}
        t_syn = {k for k, _ in syn_items[:k_syn]}
        effective_gate_real = int(real_items[k_real - 1][1]) if k_real > 0 else 0
        effective_gate_syn = int(syn_items[k_syn - 1][1]) if k_syn > 0 else 0
    else:
        # Legacy mode: tail defined as low-support keys by gate.
        t_real = {k for k, v in real_counts.items() if v <= gate and v > 0}
        t_syn = {k for k, v in syn_counts.items() if v <= gate and v > 0}
        effective_gate_real = int(gate)
        effective_gate_syn = int(gate)

    # A) tail set consistency
    union = t_real | t_syn
    inter = t_real & t_syn
    set_consistency = (len(inter) / len(union)) if union else 1.0

    # B) tail mass similarity (on real-tail anchors)
    mass_real = (sum(real_counts.get(k, 0) for k in t_real) / max(1, n_real)) if t_real else 0.0
    mass_syn_on_real = (sum(syn_counts.get(k, 0) for k in t_real) / max(1, n_syn)) if t_real else 0.0
    if mass_real <= 1e-12:
        mass_similarity = 1.0 if mass_syn_on_real <= 1e-12 else 0.0
    else:
        mass_similarity = 1.0 - abs(mass_syn_on_real - mass_real) / mass_real
        mass_similarity = max(0.0, min(1.0, mass_similarity))

    # C) tail concentration consistency (distribution over tail keys)
    conc_consistency = _tv_similarity_over_keys(real_counts, syn_counts, union)

    return {
        "tail_set_consistency": float(set_consistency),
        "tail_mass_similarity": float(mass_similarity),
        "tail_concentration_consistency": float(conc_consistency),
        "real_tail_key_count": float(len(t_real)),
        "syn_tail_key_count": float(len(t_syn)),
        "tail_key_union_count": float(len(union)),
        "effective_gate_real": float(effective_gate_real),
        "effective_gate_syn": float(effective_gate_syn),
    }


def _plot_eligible_ratio_lines(rows: list[dict[str, Any]], threshold_order: list[str], out: Path) -> None:
    by_ds: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_ds[str(r["dataset_id"])][str(r["threshold_label"])] = float(r["eligible_ratio"])
    x = np.arange(len(threshold_order))
    fig, ax = plt.subplots(figsize=(15, 6))
    for ds in sorted(by_ds.keys()):
        y = [by_ds[ds].get(t, np.nan) for t in threshold_order]
        ax.plot(x, y, marker="o", linewidth=2, label=ds.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_order, rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Tail eligible ratio")
    ax.set_title("Tail Eligible Ratio vs Threshold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_heatmap(
    summary_rows: list[dict[str, Any]],
    *,
    metric: str,
    threshold_order: list[str],
    datasets: list[str],
    title: str,
    out: Path,
) -> None:
    mat = np.full((len(threshold_order), len(datasets)), np.nan, dtype=float)
    lookup: dict[tuple[str, str], float] = {}
    for r in summary_rows:
        lookup[(str(r["threshold_label"]), str(r["dataset_id"]))] = float(r.get(metric) or np.nan)
    for i, t in enumerate(threshold_order):
        for j, d in enumerate(datasets):
            mat[i, j] = lookup.get((t, d), np.nan)

    fig_w = max(9, 1.1 * len(datasets) + 6)
    fig_h = max(8, 0.38 * len(threshold_order) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#ECEFF3")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([d.upper() for d in datasets], rotation=0)
    ax.set_yticks(np.arange(len(threshold_order)))
    ax.set_yticklabels(threshold_order)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                txt = "N/A"
                color = "#667085"
            else:
                txt = f"{v:.2f}"
                color = "white" if v >= 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=color, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(threshold_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset_submetric_lines(
    summary_rows: list[dict[str, Any]],
    dataset_id: str,
    threshold_order: list[str],
    out: Path,
) -> None:
    by_t = {str(r["threshold_label"]): r for r in summary_rows if str(r["dataset_id"]) == dataset_id}
    x = np.arange(len(threshold_order))
    y_set = [float(by_t.get(t, {}).get("tail_set_consistency_mean", np.nan)) for t in threshold_order]
    y_mass = [float(by_t.get(t, {}).get("tail_mass_similarity_mean", np.nan)) for t in threshold_order]
    y_conc = [float(by_t.get(t, {}).get("tail_concentration_consistency_mean", np.nan)) for t in threshold_order]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(x, y_set, marker="o", linewidth=2.2, label="Tail Set Consistency")
    ax.plot(x, y_mass, marker="o", linewidth=2.2, label="Tail Mass Similarity")
    ax.plot(x, y_conc, marker="o", linewidth=2.2, label="Tail Concentration Consistency")
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_order, rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset_id.upper()} Tail Sub-metrics vs Threshold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_boxplot(
    model_rows: list[dict[str, Any]],
    metric: str,
    threshold_order: list[str],
    title: str,
    out: Path,
) -> None:
    grouped: dict[str, list[float]] = {t: [] for t in threshold_order}
    for r in model_rows:
        t = str(r["threshold_label"])
        if t not in grouped:
            continue
        grouped[t].append(float(r.get(metric) or np.nan))
    data = [np.asarray([v for v in grouped[t] if not np.isnan(v)], dtype=float) for t in threshold_order]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    ax.set_xticks(np.arange(1, len(threshold_order) + 1))
    ax.set_xticklabels(threshold_order, rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    dataset_ids = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    synthetic_root = args.synthetic_root.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.selection_policy == "bottom_ratio":
        top_pct = int(round(float(args.subgroup_keep_ratio) * 100))
        tail_pct = max(0, 100 - top_pct)
        specs = [ThresholdSpec(f"split{top_pct}_tail{tail_pct}", "none")]
    else:
        specs = _threshold_specs(exclude_no_filter=not args.include_no_filter)
    threshold_order = [s.label for s in specs]
    tail_ratio = float(args.tail_ratio if args.tail_ratio is not None else (1.0 - float(args.subgroup_keep_ratio)))
    tail_ratio = max(0.0, min(1.0, tail_ratio))

    model_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []

    for ds in dataset_ids:
        real_csv = _find_real_csv(ds)
        if not real_csv:
            availability_rows.append({"dataset_id": ds, "status": "missing_real_csv"})
            continue

        cols_real, rows_real = _read_csv_rows(real_csv)
        if not cols_real or not rows_real:
            availability_rows.append({"dataset_id": ds, "status": "empty_real"})
            continue

        syn_files = _collect_synthetic_csvs(ds, synthetic_root, cols_real)
        if not syn_files:
            availability_rows.append({"dataset_id": ds, "status": "no_synthetic_match"})
            continue

        target_col = _load_target_column(ds, cols_real)
        feature_cols = [c for c in cols_real if c != target_col and not _is_id_like(c)]
        if not feature_cols:
            availability_rows.append({"dataset_id": ds, "status": "no_feature_columns"})
            continue

        transformers = _build_transformers(rows_real, feature_cols, numeric_bins=args.numeric_bins)
        real_counts = _build_key_counter(rows_real, feature_cols, transformers)
        n_real = len(rows_real)
        total_key_count = len([k for k, v in real_counts.items() if v > 0])

        syn_counters: dict[str, tuple[int, Counter]] = {}
        for model_id, spath in syn_files:
            _, rows_syn = _read_csv_rows(spath)
            syn_counts = _build_key_counter(rows_syn, feature_cols, transformers)
            syn_counters[model_id] = (len(rows_syn), syn_counts)

        availability_rows.append(
            {
                "dataset_id": ds,
                "status": "ok",
                "real_rows": n_real,
                "feature_col_count": len(feature_cols),
                "total_key_count": total_key_count,
                "synthetic_model_count": len(syn_counters),
                "target_column": target_col,
            }
        )

        for spec in specs:
            gate = spec.gate(n_real)
            if args.selection_policy == "bottom_ratio":
                real_items = sorted(((k, int(v)) for k, v in real_counts.items() if int(v) > 0), key=lambda kv: (kv[1], kv[0]))
                k_real = max(1, int(math.ceil(len(real_items) * tail_ratio))) if real_items else 0
                t_real = {k for k, _ in real_items[:k_real]}
                gate = int(real_items[k_real - 1][1]) if k_real > 0 else 0
            else:
                t_real = {k for k, v in real_counts.items() if v <= gate and v > 0}
            eligible_ratio = (len(t_real) / total_key_count) if total_key_count > 0 else 0.0
            eligible_rows.append(
                {
                    "dataset_id": ds,
                    "threshold_label": spec.label,
                    "gate": gate,
                    "tail_key_count_real": len(t_real),
                    "total_key_count": total_key_count,
                    "eligible_ratio": round(eligible_ratio, 6),
                }
            )

            metric_set: list[float] = []
            metric_mass: list[float] = []
            metric_conc: list[float] = []

            for model_id, (n_syn, syn_counts) in syn_counters.items():
                m = _tail_metrics_for_model(
                    real_counts=real_counts,
                    syn_counts=syn_counts,
                    n_real=n_real,
                    n_syn=n_syn,
                    gate=gate,
                    selection_policy=args.selection_policy,
                    tail_ratio=tail_ratio,
                )
                row = {
                    "dataset_id": ds,
                    "model_id": model_id,
                    "threshold_label": spec.label,
                    "gate": gate,
                    "eligible_ratio": round(eligible_ratio, 6),
                    **{k: round(float(v), 6) for k, v in m.items()},
                }
                model_rows.append(row)
                metric_set.append(float(m["tail_set_consistency"]))
                metric_mass.append(float(m["tail_mass_similarity"]))
                metric_conc.append(float(m["tail_concentration_consistency"]))

            summary_rows.append(
                {
                    "dataset_id": ds,
                    "threshold_label": spec.label,
                    "gate": gate,
                    "eligible_ratio": round(eligible_ratio, 6),
                    "tail_set_consistency_mean": round(mean(metric_set), 6) if metric_set else np.nan,
                    "tail_set_consistency_std": round(pstdev(metric_set), 6) if len(metric_set) >= 2 else 0.0,
                    "tail_mass_similarity_mean": round(mean(metric_mass), 6) if metric_mass else np.nan,
                    "tail_mass_similarity_std": round(pstdev(metric_mass), 6) if len(metric_mass) >= 2 else 0.0,
                    "tail_concentration_consistency_mean": round(mean(metric_conc), 6) if metric_conc else np.nan,
                    "tail_concentration_consistency_std": round(pstdev(metric_conc), 6) if len(metric_conc) >= 2 else 0.0,
                }
            )

    # Persist tables.
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(rows)

    _write_csv(out_dir / "tail_threshold_metrics_model_long.csv", model_rows)
    _write_csv(out_dir / "tail_threshold_metrics_summary.csv", summary_rows)
    _write_csv(out_dir / "tail_threshold_eligible_ratio.csv", eligible_rows)
    _write_csv(out_dir / "tail_threshold_dataset_availability.csv", availability_rows)

    valid_ds = [r["dataset_id"] for r in availability_rows if str(r.get("status")) == "ok"]
    valid_ds = [d for d in dataset_ids if d in set(valid_ds)]

    # Plots
    if eligible_rows:
        _plot_eligible_ratio_lines(eligible_rows, threshold_order, out_dir / "tail_eligible_ratio_lines.png")
    if summary_rows and valid_ds:
        _plot_metric_heatmap(
            summary_rows,
            metric="tail_set_consistency_mean",
            threshold_order=threshold_order,
            datasets=valid_ds,
            title="Tail Set Consistency (mean over models)",
            out=out_dir / "tail_set_consistency_heatmap.png",
        )
        _plot_metric_heatmap(
            summary_rows,
            metric="tail_mass_similarity_mean",
            threshold_order=threshold_order,
            datasets=valid_ds,
            title="Tail Mass Similarity (mean over models)",
            out=out_dir / "tail_mass_similarity_heatmap.png",
        )
        _plot_metric_heatmap(
            summary_rows,
            metric="tail_concentration_consistency_mean",
            threshold_order=threshold_order,
            datasets=valid_ds,
            title="Tail Concentration Consistency (mean over models)",
            out=out_dir / "tail_concentration_consistency_heatmap.png",
        )
        for ds in valid_ds:
            _plot_dataset_submetric_lines(
                summary_rows,
                dataset_id=ds,
                threshold_order=threshold_order,
                out=out_dir / f"tail_submetrics_lines_{ds}.png",
            )

    if model_rows:
        _plot_metric_boxplot(
            model_rows,
            metric="tail_set_consistency",
            threshold_order=threshold_order,
            title="Tail Set Consistency Distribution (all dataset-model points)",
            out=out_dir / "tail_set_consistency_boxplot.png",
        )
        _plot_metric_boxplot(
            model_rows,
            metric="tail_mass_similarity",
            threshold_order=threshold_order,
            title="Tail Mass Similarity Distribution (all dataset-model points)",
            out=out_dir / "tail_mass_similarity_boxplot.png",
        )
        _plot_metric_boxplot(
            model_rows,
            metric="tail_concentration_consistency",
            threshold_order=threshold_order,
            title="Tail Concentration Consistency Distribution (all dataset-model points)",
            out=out_dir / "tail_concentration_consistency_boxplot.png",
        )

    # Quick conclusions.
    conclusion_lines = [
        "# Tail Threshold Sensitivity - Quick Findings",
        "",
        "This report excludes `no_filter` unless explicitly enabled.",
        "",
    ]
    if summary_rows:
        by_thr: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in summary_rows:
            by_thr[str(r["threshold_label"])].append(r)
        conclusion_lines.append("## Threshold-level means across datasets")
        conclusion_lines.append("")
        conclusion_lines.append("| threshold | set_mean | mass_mean | conc_mean | eligible_ratio_mean |")
        conclusion_lines.append("|---|---:|---:|---:|---:|")
        for t in threshold_order:
            items = by_thr.get(t, [])
            if not items:
                continue
            set_m = mean(float(x["tail_set_consistency_mean"]) for x in items)
            mass_m = mean(float(x["tail_mass_similarity_mean"]) for x in items)
            conc_m = mean(float(x["tail_concentration_consistency_mean"]) for x in items)
            er_m = mean(float(x["eligible_ratio"]) for x in items)
            conclusion_lines.append(f"| {t} | {set_m:.3f} | {mass_m:.3f} | {conc_m:.3f} | {er_m:.3f} |")

    (out_dir / "README_tail_threshold_findings.md").write_text("\n".join(conclusion_lines), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "datasets_requested": dataset_ids,
        "datasets_ok": valid_ds,
        "threshold_labels": threshold_order,
        "selection_policy": args.selection_policy,
        "subgroup_keep_ratio": args.subgroup_keep_ratio,
        "tail_ratio": tail_ratio,
        "exclude_no_filter": bool(not args.include_no_filter),
        "synthetic_root": str(synthetic_root),
        "output_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "datasets_ok": valid_ds}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tail threshold sensitivity analysis.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="c2,c7,m1,m4,n1,n2",
        help="Comma-separated dataset ids.",
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / "SynData" / "synthetic_10ds_7models",
    )
    parser.add_argument("--numeric-bins", type=int, default=10)
    parser.add_argument(
        "--selection-policy",
        type=str,
        choices=["support_gate", "bottom_ratio"],
        default="bottom_ratio",
        help="bottom_ratio uses bottom tail-ratio keys by support.",
    )
    parser.add_argument(
        "--subgroup-keep-ratio",
        type=float,
        default=DEFAULT_SUBGROUP_KEEP_RATIO,
        help="Primary split ratio for subgroup bucket (tail ratio defaults to 1 - this value).",
    )
    parser.add_argument(
        "--tail-ratio",
        type=float,
        default=None,
        help="Optional explicit tail ratio. If omitted, tail_ratio = 1 - subgroup_keep_ratio.",
    )
    parser.add_argument(
        "--include-no-filter",
        action="store_true",
        help="Include no_filter baseline threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT.parent.parent / "Query" / "code" / "logs" / "analysis" / f"tail_threshold_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
