#!/usr/bin/env python3
"""Sensitivity study for subgroup selection policy.

Policies:
- support_gate (legacy): max(abs_threshold, pct_threshold * N_real)
- top_ratio (default): keep top-K groups by real support (default top 97%)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DatasetConfig:
    dataset_id: str
    run_id: str


DEFAULT_DATASET_CONFIGS = [
    DatasetConfig(dataset_id="c2", run_id="c2_20260402_004909"),
    DatasetConfig(dataset_id="m4", run_id="m4_20260412_010338"),
    DatasetConfig(dataset_id="n1", run_id="n1_20260402_013632"),
]
DEFAULT_SUBGROUP_KEEP_RATIO = 0.97


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        cols = [str(c) for c in (reader.fieldnames or [])]
    return cols, rows


def _load_field_registry(dataset_id: str) -> dict[str, Any]:
    path = PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / dataset_id / "metadata" / "field_registry.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_target_column(dataset_id: str) -> str:
    path = PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / dataset_id / "metadata" / "dataset_semantics.yaml"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("target_column:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"Cannot find target_column in {path}")


def _load_queryspecs(dataset_id: str, run_id: str) -> list[dict[str, Any]]:
    path = PROJECT_ROOT.parent.parent / "Query" / "code" / "logs" / "runs" / run_id / "benchmark_package" / "queryspecs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = payload.get("queryspecs")
    return specs if isinstance(specs, list) else []


def _parse_abs_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_pct_list(text: str) -> list[float]:
    out: list[float] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if token.endswith("%"):
            out.append(float(token[:-1]) / 100.0)
        else:
            out.append(float(token))
    return out


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quantile_edges(values: list[float], bins: int) -> list[float]:
    if not values:
        return []
    qs = np.linspace(0, 1, bins + 1)
    arr = np.asarray(values, dtype=float)
    edges = np.quantile(arr, qs).tolist()
    dedup: list[float] = []
    for x in edges:
        if not dedup or abs(dedup[-1] - x) > 1e-12:
            dedup.append(float(x))
    return dedup


def _bin_value(value: str, edges: list[float]) -> str:
    fv = _safe_float(value)
    if fv is None or len(edges) < 2:
        return "missing"
    for idx in range(len(edges) - 1):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == len(edges) - 2:
            if left <= fv <= right:
                return f"q{idx+1}"
        if left <= fv < right:
            return f"q{idx+1}"
    if fv < edges[0]:
        return "below_q1"
    return f"above_q{len(edges)-1}"


def _tv_similarity(real_counts: Counter[str], syn_counts: Counter[str], labels: list[str]) -> float:
    nr = sum(real_counts.values())
    ns = sum(syn_counts.values())
    if nr <= 0:
        return 0.0
    if ns <= 0:
        return 0.0
    tv = 0.0
    for lab in labels:
        pr = real_counts.get(lab, 0) / nr
        ps = syn_counts.get(lab, 0) / ns
        tv += abs(pr - ps)
    tv *= 0.5
    return max(0.0, min(1.0, 1.0 - tv))


def _kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    pos_a = {m: i for i, m in enumerate(order_a)}
    pos_b = {m: i for i, m in enumerate(order_b)}
    common = [m for m in order_a if m in pos_b]
    n = len(common)
    if n <= 1:
        return 0.0
    c = 0
    d = 0
    for i in range(n):
        for j in range(i + 1, n):
            mi = common[i]
            mj = common[j]
            sa = pos_a[mi] < pos_a[mj]
            sb = pos_b[mi] < pos_b[mj]
            if sa == sb:
                c += 1
            else:
                d += 1
    if c + d == 0:
        return 0.0
    return (c - d) / (c + d)


def _rank_models(scores: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _infer_model_id(path: Path, dataset_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(dataset_root.resolve())
        parts = rel.parts
        if parts:
            return parts[0]
    except Exception:  # noqa: BLE001
        pass
    return path.parent.name


def _collect_synthetic_csvs(dataset_id: str, synthetic_root: Path, expected_columns: list[str]) -> list[tuple[str, Path]]:
    root = synthetic_root / dataset_id
    if not root.exists():
        return []
    out: list[tuple[str, Path]] = []
    expected_set = set(expected_columns)
    for p in sorted(root.rglob("*.csv")):
        try:
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
            if set(header) != expected_set:
                continue
        except Exception:  # noqa: BLE001
            continue
        out.append((_infer_model_id(p, root), p))
    return out


def _build_subgroup_combos(
    *,
    dataset_id: str,
    run_id: str,
    field_registry: dict[str, Any],
    max_combos: int,
) -> list[tuple[str, ...]]:
    specs = _load_queryspecs(dataset_id, run_id)
    counter: Counter[tuple[str, ...]] = Counter()
    for spec in specs:
        fam = str(spec.get("family_id") or spec.get("family") or "")
        if fam not in {"subgroup_structure", "conditional_dependency_structure"}:
            continue
        cols = tuple(str(c) for c in (spec.get("subgroup_columns") or []) if str(c).strip())
        if cols and len(cols) <= 2:
            counter[cols] += 1
        # Also absorb feature columns as single-axis subgroup candidates.
        feat_cols = [str(c) for c in (spec.get("feature_columns") or []) if str(c).strip()]
        for c in feat_cols:
            counter[(c,)] += 1

    combos = [item for item, _ in counter.most_common(max_combos)]

    # fallback supplement: metadata-driven
    fields = field_registry.get("fields") if isinstance(field_registry, dict) else []
    groupby_cols: list[str] = []
    for f in fields if isinstance(fields, list) else []:
        if not isinstance(f, dict):
            continue
        name = str(f.get("name") or "").strip()
        if not name:
            continue
        use_gb = bool(f.get("use_for_groupby"))
        tags = [str(x) for x in (f.get("field_tags") or [])]
        sem = str(f.get("semantic_type") or f.get("declared_type") or "").lower()
        if (
            use_gb
            or "subgroup_candidate" in tags
            or "condition_candidate" in tags
            or "categorical" in sem
            or "discrete" in sem
            or "binary" in sem
            or "numeric_sparse_frequency" in sem
        ):
            groupby_cols.append(name)

    seen = set(combos)
    for c in groupby_cols:
        cand = (c,)
        if cand not in seen:
            combos.append(cand)
            seen.add(cand)
        if len(combos) >= max_combos:
            return combos[:max_combos]

    for p in combinations(groupby_cols, 2):
        cand = tuple(p)
        if cand not in seen:
            combos.append(cand)
            seen.add(cand)
        if len(combos) >= max_combos:
            break
    return combos[:max_combos]


def _prepare_binning_maps(
    *,
    rows_real: list[dict[str, str]],
    subgroup_cols_flat: list[str],
    target_col: str,
    field_registry: dict[str, Any],
    subgroup_bins: int,
    target_bins: int,
) -> tuple[dict[str, list[float]], list[str], dict[str, list[float]]]:
    field_map = {
        str(f.get("name")): f
        for f in (field_registry.get("fields") if isinstance(field_registry.get("fields"), list) else [])
        if isinstance(f, dict)
    }

    subgroup_edges: dict[str, list[float]] = {}
    for col in sorted(set(subgroup_cols_flat)):
        meta = field_map.get(col, {})
        sem = str(meta.get("semantic_type") or meta.get("declared_type") or "").lower()
        vals = [_safe_float(r.get(col)) for r in rows_real]
        nums = [v for v in vals if v is not None]
        distinct = len({float(v) for v in nums})
        if "numeric" in sem and distinct > 20:
            subgroup_edges[col] = _quantile_edges(nums, subgroup_bins)

    target_meta = field_map.get(target_col, {})
    target_sem = str(target_meta.get("semantic_type") or target_meta.get("declared_type") or "").lower()
    target_edges: dict[str, list[float]] = {}
    target_labels: list[str]
    if "categorical" in target_sem or "binary" in target_sem or "boolean" in target_sem:
        labels = sorted({str(r.get(target_col, "")) for r in rows_real})
        target_labels = labels
    else:
        vals = [_safe_float(r.get(target_col)) for r in rows_real]
        nums = [v for v in vals if v is not None]
        if nums:
            edges = _quantile_edges(nums, target_bins)
            target_edges[target_col] = edges
            target_labels = [f"q{i+1}" for i in range(max(1, len(edges) - 1))]
        else:
            target_labels = ["missing"]
    return subgroup_edges, target_labels, target_edges


def _transform_value(col: str, value: str, edges_map: dict[str, list[float]]) -> str:
    edges = edges_map.get(col)
    if edges:
        return _bin_value(value, edges)
    return "<NULL>" if value is None or value == "" else str(value)


def _transform_target(value: str, target_col: str, target_edges: dict[str, list[float]]) -> str:
    edges = target_edges.get(target_col)
    if edges:
        return _bin_value(value, edges)
    return "<NULL>" if value is None or value == "" else str(value)


def _build_group_stats(
    *,
    rows: list[dict[str, str]],
    combos: list[tuple[str, ...]],
    target_col: str,
    subgroup_edges: dict[str, list[float]],
    target_edges: dict[str, list[float]],
) -> dict[tuple[str, tuple[str, ...]], tuple[int, Counter[str]]]:
    out: dict[tuple[str, tuple[str, ...]], tuple[int, Counter[str]]] = {}
    support_counter: Counter[tuple[str, tuple[str, ...]]] = Counter()
    target_counter: dict[tuple[str, tuple[str, ...]], Counter[str]] = defaultdict(Counter)

    for row in rows:
        target_label = _transform_target(str(row.get(target_col, "")), target_col, target_edges)
        for combo in combos:
            combo_key = "|".join(combo)
            values = tuple(_transform_value(c, str(row.get(c, "")), subgroup_edges) for c in combo)
            gk = (combo_key, values)
            support_counter[gk] += 1
            target_counter[gk][target_label] += 1

    for gk, cnt in support_counter.items():
        out[gk] = (cnt, target_counter[gk])
    return out


def _score_threshold_for_model(
    *,
    records: list[dict[str, Any]],
    abs_threshold: int,
    pct_threshold: float,
    n_real: int,
    selection_policy: str,
    subgroup_keep_ratio: float,
) -> dict[str, Any]:
    total = len(records)
    keep_ratio = max(0.0, min(1.0, float(subgroup_keep_ratio)))

    if selection_policy == "top_ratio":
        if total <= 0:
            return {
                "selection_policy": selection_policy,
                "subgroup_keep_ratio": keep_ratio,
                "effective_threshold": 0,
                "eligible_groups": 0,
                "total_groups": 0,
                "eligible_ratio": 0.0,
                "profile_similarity": 0.0,
                "support_retention": 0.0,
                "coverage_stability": 0.0,
                "subgroup_score": 0.0,
            }
        keep_n = max(1, int(math.ceil(total * keep_ratio)))
        ranked = sorted(records, key=lambda r: int(r["support_real"]), reverse=True)
        eligible = ranked[:keep_n]
        gate = int(eligible[-1]["support_real"]) if eligible else 0
    else:
        gate = max(abs_threshold, int(math.ceil(pct_threshold * n_real)))
        eligible = [r for r in records if int(r["support_real"]) >= gate]

    if not eligible:
        return {
            "selection_policy": selection_policy,
            "subgroup_keep_ratio": keep_ratio,
            "effective_threshold": gate,
            "eligible_groups": 0,
            "total_groups": total,
            "eligible_ratio": 0.0,
            "profile_similarity": 0.0,
            "support_retention": 0.0,
            "coverage_stability": 0.0,
            "subgroup_score": 0.0,
        }

    ws = [float(r["weight"]) for r in eligible]
    sim = [float(r["similarity"]) for r in eligible]
    ret = [float(r["retention"]) for r in eligible]
    cov = [float(r["coverage"]) for r in eligible]
    sw = sum(ws) if sum(ws) > 0 else 1.0
    profile = sum(w * x for w, x in zip(ws, sim)) / sw
    support = sum(w * x for w, x in zip(ws, ret)) / sw
    coverage = sum(cov) / len(cov)
    score = (profile + support + coverage) / 3.0
    return {
        "selection_policy": selection_policy,
        "subgroup_keep_ratio": keep_ratio,
        "effective_threshold": gate,
        "eligible_groups": len(eligible),
        "total_groups": total,
        "eligible_ratio": len(eligible) / max(1, total),
        "profile_similarity": profile,
        "support_retention": support,
        "coverage_stability": coverage,
        "subgroup_score": score,
    }


def _plot_support_distribution(
    *,
    dataset_id: str,
    supports: list[int],
    n_real: int,
    subgroup_keep_ratio: float,
    output_path: Path,
) -> None:
    ratios = [s / max(1, n_real) for s in supports]
    ranked_supports = sorted((int(s) for s in supports if int(s) > 0), reverse=True)
    keep_ratio = max(0.0, min(1.0, float(subgroup_keep_ratio)))
    if ranked_supports:
        keep_n = max(1, int(math.ceil(len(ranked_supports) * keep_ratio)))
        split_gate_count = int(ranked_supports[keep_n - 1])
    else:
        split_gate_count = 0
    split_gate_ratio = split_gate_count / max(1, n_real)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].hist(supports, bins=40, color="#2A9D8F", alpha=0.9, edgecolor="white")
    axes[0].set_title(f"{dataset_id.upper()} subgroup support (count)")
    axes[0].set_xlabel("Support count")
    axes[0].set_ylabel("Group frequency")
    axes[0].axvline(
        split_gate_count,
        color="#E76F51",
        linestyle="--",
        linewidth=1.6,
        label=f"top{int(round(keep_ratio * 100))}% split gate≈{split_gate_count}",
    )
    axes[0].legend()

    axes[1].hist(ratios, bins=40, color="#264653", alpha=0.9, edgecolor="white")
    axes[1].set_title(f"{dataset_id.upper()} subgroup support (ratio)")
    axes[1].set_xlabel("Support / N_real")
    axes[1].set_ylabel("Group frequency")
    axes[1].axvline(
        split_gate_ratio,
        color="#E76F51",
        linestyle="--",
        linewidth=1.6,
        label=f"top{int(round(keep_ratio * 100))}% split gate≈{split_gate_ratio:.2%}",
    )
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    *,
    dataset_id: str,
    abs_list: list[int],
    pct_list: list[float],
    matrix: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(pct_list)))
    ax.set_xticklabels([f"{p*100:.1f}%" for p in pct_list], fontsize=9)
    ax.set_yticks(range(len(abs_list)))
    ax.set_yticklabels([str(a) for a in abs_list], fontsize=9)
    ax.set_xlabel("Percentage threshold")
    ax.set_ylabel("Absolute threshold")
    ax.set_title(f"{dataset_id.upper()} - {title}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)
    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            ax.text(xi, yi, f"{matrix[yi, xi]:.3f}", ha="center", va="center", fontsize=8, color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    synthetic_root = args.synthetic_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    abs_thresholds = _parse_abs_list(args.abs_thresholds)
    pct_thresholds = _parse_pct_list(args.pct_thresholds)
    if args.selection_policy == "top_ratio":
        # Fixed subgroup-vs-tail split mode: subgroup keeps top support groups.
        abs_thresholds = [0]
        pct_thresholds = [0.0]
    configs = DEFAULT_DATASET_CONFIGS

    threshold_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    group_dist_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "synthetic_root": str(synthetic_root),
        "abs_thresholds": abs_thresholds,
        "pct_thresholds": pct_thresholds,
        "selection_policy": args.selection_policy,
        "subgroup_keep_ratio": args.subgroup_keep_ratio,
        "datasets": [],
    }

    for cfg in configs:
        dataset_id = cfg.dataset_id
        real_csv = PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / dataset_id / "raw" / f"{dataset_id}-main.csv"
        columns, rows_real = _read_csv_rows(real_csv)
        n_real = len(rows_real)
        field_registry = _load_field_registry(dataset_id)
        target_col = _load_target_column(dataset_id)

        combos = _build_subgroup_combos(
            dataset_id=dataset_id,
            run_id=cfg.run_id,
            field_registry=field_registry,
            max_combos=args.max_combos,
        )
        combos = [c for c in combos if all(col in columns for col in c) and target_col not in c]
        if not combos:
            print(f"[warn] {dataset_id}: no subgroup combos found; skip.")
            continue

        subgroup_flat = [c for combo in combos for c in combo]
        subgroup_edges, target_labels, target_edges = _prepare_binning_maps(
            rows_real=rows_real,
            subgroup_cols_flat=subgroup_flat,
            target_col=target_col,
            field_registry=field_registry,
            subgroup_bins=args.subgroup_bins,
            target_bins=args.target_bins,
        )

        real_groups = _build_group_stats(
            rows=rows_real,
            combos=combos,
            target_col=target_col,
            subgroup_edges=subgroup_edges,
            target_edges=target_edges,
        )

        supports = [support for support, _ in real_groups.values()]
        for support in supports:
            group_dist_rows.append(
                {
                    "dataset_id": dataset_id,
                    "support_count": support,
                    "support_ratio": support / max(1, n_real),
                }
            )
        _plot_support_distribution(
            dataset_id=dataset_id,
            supports=supports,
            n_real=n_real,
            subgroup_keep_ratio=args.subgroup_keep_ratio,
            output_path=figures_dir / f"{dataset_id}_support_distribution.png",
        )

        syn_files = _collect_synthetic_csvs(dataset_id, synthetic_root, columns)
        if not syn_files:
            print(f"[warn] {dataset_id}: no usable synthetic files found.")
            continue

        # Precompute per-model per-group metrics.
        model_group_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for model_id, syn_path in syn_files:
            _, rows_syn = _read_csv_rows(syn_path)
            syn_groups = _build_group_stats(
                rows=rows_syn,
                combos=combos,
                target_col=target_col,
                subgroup_edges=subgroup_edges,
                target_edges=target_edges,
            )
            per_group_records: list[dict[str, Any]] = []
            for gk, (support_real, target_real) in real_groups.items():
                support_syn, target_syn = syn_groups.get(gk, (0, Counter()))
                sim = _tv_similarity(target_real, target_syn, target_labels)
                retention = min(1.0, (support_syn / support_real)) if support_real > 0 else 0.0
                coverage = 1.0 if support_syn > 0 else 0.0
                per_group_records.append(
                    {
                        "group_key": f"{gk[0]}::{gk[1]}",
                        "support_real": support_real,
                        "support_syn": support_syn,
                        "weight": math.sqrt(float(support_real)),
                        "similarity": sim,
                        "retention": retention,
                        "coverage": coverage,
                    }
                )
            model_group_records[model_id].extend(per_group_records)

        # Sweep thresholds.
        dataset_threshold_scores: dict[tuple[int, float], dict[str, float]] = defaultdict(dict)
        for abs_t in abs_thresholds:
            for pct_t in pct_thresholds:
                key = (abs_t, pct_t)
                model_scores_for_key: dict[str, float] = {}
                eligible_ratios: list[float] = []
                for model_id, records in sorted(model_group_records.items()):
                    stat = _score_threshold_for_model(
                        records=records,
                        abs_threshold=abs_t,
                        pct_threshold=pct_t,
                        n_real=n_real,
                        selection_policy=args.selection_policy,
                        subgroup_keep_ratio=args.subgroup_keep_ratio,
                    )
                    model_scores_for_key[model_id] = float(stat["subgroup_score"])
                    eligible_ratios.append(float(stat["eligible_ratio"]))
                    threshold_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "run_id": cfg.run_id,
                            "model_id": model_id,
                            "abs_threshold": abs_t,
                            "pct_threshold": pct_t,
                            "n_real": n_real,
                            **stat,
                        }
                    )

                dataset_threshold_scores[key] = model_scores_for_key
                vals = list(model_scores_for_key.values())
                aggregate_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "run_id": cfg.run_id,
                        "abs_threshold": abs_t,
                        "pct_threshold": pct_t,
                        "n_real": n_real,
                        "effective_threshold": max(abs_t, int(math.ceil(pct_t * n_real))),
                        "mean_score": float(np.mean(vals)) if vals else 0.0,
                        "std_score": float(np.std(vals)) if vals else 0.0,
                        "min_score": float(np.min(vals)) if vals else 0.0,
                        "max_score": float(np.max(vals)) if vals else 0.0,
                        "mean_eligible_ratio": float(np.mean(eligible_ratios)) if eligible_ratios else 0.0,
                    }
                )

        baseline_key = (0, 0.0) if args.selection_policy == "top_ratio" else (30, 0.01)
        if baseline_key not in dataset_threshold_scores:
            baseline_key = sorted(dataset_threshold_scores.keys())[0]
        baseline_scores = dataset_threshold_scores[baseline_key]
        baseline_rank = _rank_models(baseline_scores)

        for (abs_t, pct_t), score_map in sorted(dataset_threshold_scores.items()):
            rank = _rank_models(score_map)
            tau = _kendall_tau(baseline_rank, rank)
            top1_same = 1.0 if (baseline_rank and rank and baseline_rank[0] == rank[0]) else 0.0
            top3_overlap = len(set(baseline_rank[:3]) & set(rank[:3])) / max(1, min(3, len(rank)))
            rank_rows.append(
                {
                    "dataset_id": dataset_id,
                    "run_id": cfg.run_id,
                    "baseline_abs_threshold": baseline_key[0],
                    "baseline_pct_threshold": baseline_key[1],
                    "abs_threshold": abs_t,
                    "pct_threshold": pct_t,
                    "kendall_tau_vs_baseline": tau,
                    "top1_same_vs_baseline": top1_same,
                    "top3_overlap_vs_baseline": top3_overlap,
                }
            )

        # Dataset-level figures.
        agg_sub = [r for r in aggregate_rows if r["dataset_id"] == dataset_id]
        score_mat = np.zeros((len(abs_thresholds), len(pct_thresholds)), dtype=float)
        elig_mat = np.zeros((len(abs_thresholds), len(pct_thresholds)), dtype=float)
        for i, a in enumerate(abs_thresholds):
            for j, p in enumerate(pct_thresholds):
                row = next((x for x in agg_sub if x["abs_threshold"] == a and x["pct_threshold"] == p), None)
                if row:
                    score_mat[i, j] = float(row["mean_score"])
                    elig_mat[i, j] = float(row["mean_eligible_ratio"])

        _plot_heatmap(
            dataset_id=dataset_id,
            abs_list=abs_thresholds,
            pct_list=pct_thresholds,
            matrix=score_mat,
            title="mean subgroup score",
            output_path=figures_dir / f"{dataset_id}_score_heatmap.png",
            cmap="YlGnBu",
        )
        _plot_heatmap(
            dataset_id=dataset_id,
            abs_list=abs_thresholds,
            pct_list=pct_thresholds,
            matrix=elig_mat,
            title="mean eligible-group ratio",
            output_path=figures_dir / f"{dataset_id}_eligible_ratio_heatmap.png",
            cmap="YlOrBr",
        )

        manifest["datasets"].append(
            {
                "dataset_id": dataset_id,
                "run_id": cfg.run_id,
                "n_real": n_real,
                "target_column": target_col,
                "combo_count": len(combos),
                "combos": [list(c) for c in combos],
                "model_count": len(model_group_records),
                "synthetic_file_count": len(syn_files),
                "baseline_threshold": {"abs": baseline_key[0], "pct": baseline_key[1]},
            }
        )

    # Cross-dataset summary figure.
    if aggregate_rows:
        fig, ax = plt.subplots(figsize=(8.5, 5.1))
        datasets = sorted({r["dataset_id"] for r in aggregate_rows})
        x = np.arange(len(datasets))
        if args.selection_policy == "top_ratio":
            comparison_keys = [(0, 0.0)]
            labels = [f"top={args.subgroup_keep_ratio*100:.0f}% (tail={100-args.subgroup_keep_ratio*100:.0f}%)"]
        else:
            preferred = [(20, 0.01), (30, 0.01), (40, 0.01)]
            available = {(int(r["abs_threshold"]), float(r["pct_threshold"])) for r in aggregate_rows}
            comparison_keys = [k for k in preferred if k in available]
            if not comparison_keys:
                comparison_keys = sorted(available)[:3]
            labels = [f"abs={k[0]}, pct={k[1]*100:.0f}%" for k in comparison_keys]

        width = 0.70 if len(comparison_keys) <= 1 else min(0.24, 0.9 / len(comparison_keys))
        center_shift = (len(comparison_keys) - 1) / 2.0
        for idx, key in enumerate(comparison_keys):
            vals = []
            for d in datasets:
                row = next((r for r in aggregate_rows if r["dataset_id"] == d and r["abs_threshold"] == key[0] and abs(r["pct_threshold"] - key[1]) < 1e-12), None)
                vals.append(float(row["mean_score"]) if row else 0.0)
            ax.bar(x + (idx - center_shift) * width, vals, width=width, label=labels[idx])
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Mean subgroup score")
        if args.selection_policy == "top_ratio":
            ax.set_title(f"Subgroup split sensitivity (top={args.subgroup_keep_ratio*100:.0f}%)")
        else:
            ax.set_title("Threshold sensitivity (fixed pct=1%)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "cross_dataset_threshold_score_bars.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        headers = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    _write_csv(output_dir / "threshold_scores_by_model.csv", threshold_rows)
    _write_csv(output_dir / "threshold_aggregate.csv", aggregate_rows)
    _write_csv(output_dir / "threshold_rank_vs_baseline.csv", rank_rows)
    _write_csv(output_dir / "group_support_distribution.csv", group_dist_rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "datasets": [d.dataset_id for d in configs]}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep subgroup support thresholds and analyze score sensitivity.")
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / "SynData" / "synthetic_10ds_7models",
    )
    parser.add_argument("--abs-thresholds", type=str, default="5,10,20,30,40,60,80")
    parser.add_argument("--pct-thresholds", type=str, default="0.002,0.005,0.01,0.02,0.05")
    parser.add_argument("--max-combos", type=int, default=10)
    parser.add_argument("--subgroup-bins", type=int, default=4)
    parser.add_argument("--target-bins", type=int, default=5)
    parser.add_argument(
        "--selection-policy",
        type=str,
        choices=["support_gate", "top_ratio"],
        default="top_ratio",
        help="top_ratio keeps highest-support subgroups (default top97 split).",
    )
    parser.add_argument(
        "--subgroup-keep-ratio",
        type=float,
        default=DEFAULT_SUBGROUP_KEEP_RATIO,
        help="When selection-policy=top_ratio, keep this ratio of groups for subgroup scoring.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT.parent.parent / "Query" / "code" / "logs" / "analysis" / f"subgroup_threshold_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
