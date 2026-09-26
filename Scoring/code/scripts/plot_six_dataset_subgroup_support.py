#!/usr/bin/env python3
"""Plot subgroup support distributions for 6 datasets (c/m/n, two each)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ["c2", "c3", "m1", "m4", "n1", "n2"]


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        cols = [str(c) for c in (reader.fieldnames or [])]
    return cols, rows


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


def _load_field_registry(dataset_id: str) -> dict[str, Any]:
    path = PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / dataset_id / "metadata" / "field_registry.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_columns_without_metadata(columns: list[str], rows: list[dict[str, str]], max_columns: int) -> list[str]:
    sample = rows[: min(50000, len(rows))]
    scored: list[tuple[float, str]] = []
    for col in columns:
        vals = [str(r.get(col, "")) for r in sample]
        non_missing = [v for v in vals if v.strip() != ""]
        if not non_missing:
            continue
        distinct = len(set(non_missing))
        distinct_ratio = distinct / max(1, len(non_missing))
        parseable = 0
        for v in non_missing:
            try:
                float(v)
                parseable += 1
            except (TypeError, ValueError):
                pass
        numeric_ratio = parseable / max(1, len(non_missing))

        score = 0.0
        if numeric_ratio < 0.9:
            if 2 <= distinct <= 200:
                score += 8.0
            elif distinct <= 500:
                score += 4.0
        else:
            if distinct <= 30:
                score += 6.0
            elif distinct <= 100:
                score += 3.0
        if distinct_ratio > 0.95:
            score -= 4.0  # likely id-like / near-unique
        if col.lower() in {"id", "index", "row_id"}:
            score -= 6.0
        scored.append((score, col))

    scored.sort(key=lambda x: (-x[0], x[1]))
    cols = [name for score, name in scored if score > 0]
    if not cols:
        cols = [c for c in columns if c.lower() not in {"id", "index", "row_id"}]
    return cols[:max_columns]


def _select_subgroup_columns(
    field_registry: dict[str, Any],
    columns: list[str],
    rows: list[dict[str, str]],
    max_columns: int,
) -> list[str]:
    fields = field_registry.get("fields") if isinstance(field_registry, dict) else []
    if not fields:
        return _infer_columns_without_metadata(columns, rows, max_columns)

    scored: list[tuple[int, str]] = []
    for item in fields if isinstance(fields, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        role = str(item.get("role") or "").strip().lower()
        if role == "target":
            continue
        sem = str(item.get("semantic_type") or item.get("declared_type") or "").lower()
        tags = [str(x) for x in (item.get("field_tags") or [])]
        use_groupby = bool(item.get("use_for_groupby"))

        score = 0
        if use_groupby:
            score += 5
        if "subgroup_candidate" in tags:
            score += 4
        if "condition_candidate" in tags:
            score += 2
        if "categorical" in sem or "binary" in sem or "discrete" in sem:
            score += 3
        if "numeric_sparse_frequency" in sem:
            score += 2
        scored.append((score, name))

    scored.sort(key=lambda x: (-x[0], x[1]))
    cols = [name for score, name in scored if score > 0]
    cols = [c for c in cols if c in set(columns)]
    if not cols:
        return _infer_columns_without_metadata(columns, rows, max_columns)
    return cols[:max_columns]


def _build_combos(cols: list[str], max_combos: int) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = []
    for c in cols:
        combos.append((c,))
        if len(combos) >= max_combos:
            return combos
    for a, b in combinations(cols, 2):
        combos.append((a, b))
        if len(combos) >= max_combos:
            break
    return combos


def _build_edges(rows: list[dict[str, str]], cols: list[str], bins: int) -> dict[str, list[float]]:
    edges_map: dict[str, list[float]] = {}
    for c in cols:
        vals = [_safe_float(r.get(c)) for r in rows]
        nums = [v for v in vals if v is not None]
        distinct = len({float(v) for v in nums})
        if distinct > 20:
            edges_map[c] = _quantile_edges(nums, bins)
    return edges_map


def _transform_value(col: str, value: str, edges_map: dict[str, list[float]]) -> str:
    edges = edges_map.get(col)
    if edges:
        return _bin_value(value, edges)
    if value is None or str(value).strip() == "":
        return "<NULL>"
    return str(value)


def _compute_support_counts(
    rows: list[dict[str, str]],
    combos: list[tuple[str, ...]],
    edges_map: dict[str, list[float]],
) -> list[int]:
    counts: defaultdict[tuple[str, tuple[str, ...]], int] = defaultdict(int)
    for row in rows:
        for combo in combos:
            combo_key = "|".join(combo)
            values = tuple(_transform_value(c, row.get(c, ""), edges_map) for c in combo)
            counts[(combo_key, values)] += 1
    return list(counts.values())


def _dataset_color(dataset_id: str) -> str:
    if dataset_id.startswith("c"):
        return "#457B9D"  # blue
    if dataset_id.startswith("m"):
        return "#2A9D8F"  # teal
    return "#E76F51"  # orange/red


def _plot_box(
    *,
    labels: list[str],
    series: list[list[float]],
    colors: list[str],
    ylabel: str,
    title: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    bp = ax.boxplot(series, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("#1F2937")
    for median in bp["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.5)

    rng = np.random.default_rng(42)
    for i, values in enumerate(series, start=1):
        if not values:
            continue
        x = rng.normal(loc=i, scale=0.045, size=len(values))
        ax.scatter(x, values, s=10, alpha=0.25, color=colors[i - 1], edgecolors="none")

    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    dataset_ids = [x.strip() for x in args.datasets.split(",") if x.strip()]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    colors: list[str] = []
    count_series: list[list[float]] = []
    ratio_series: list[list[float]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset_id in dataset_ids:
        csv_path = PROJECT_ROOT.parent.parent / "Query" / "code" / "data" / dataset_id / "raw" / f"{dataset_id}-main.csv"
        field_registry = _load_field_registry(dataset_id)
        cols, rows = _read_csv_rows(csv_path)
        n_real = len(rows)

        subgroup_cols = _select_subgroup_columns(
            field_registry,
            columns=cols,
            rows=rows,
            max_columns=args.max_columns,
        )
        subgroup_cols = [c for c in subgroup_cols if c in cols]
        combos = _build_combos(subgroup_cols, max_combos=args.max_combos)
        edges_map = _build_edges(rows, subgroup_cols, bins=args.numeric_bins)
        supports = _compute_support_counts(rows, combos, edges_map)
        ratios = [x / max(1, n_real) for x in supports]

        labels.append(dataset_id.upper())
        colors.append(_dataset_color(dataset_id))
        count_series.append([float(v) for v in supports])
        ratio_series.append([float(v) for v in ratios])

        arr = np.asarray(supports, dtype=float)
        arr_r = np.asarray(ratios, dtype=float)
        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "row_count": n_real,
                "subgroup_columns_used": "|".join(subgroup_cols),
                "combo_count": len(combos),
                "group_count": len(supports),
                "count_p10": np.percentile(arr, 10),
                "count_p25": np.percentile(arr, 25),
                "count_p50": np.percentile(arr, 50),
                "count_p75": np.percentile(arr, 75),
                "count_p90": np.percentile(arr, 90),
                "ratio_p10": np.percentile(arr_r, 10),
                "ratio_p25": np.percentile(arr_r, 25),
                "ratio_p50": np.percentile(arr_r, 50),
                "ratio_p75": np.percentile(arr_r, 75),
                "ratio_p90": np.percentile(arr_r, 90),
            }
        )

    _plot_box(
        labels=labels,
        series=count_series,
        colors=colors,
        ylabel="Subgroup support count",
        title="Subgroup Support Distribution (Count) across 6 datasets",
        output_path=output_dir / "six_datasets_support_count.png",
        log_scale=True,
    )
    _plot_box(
        labels=labels,
        series=ratio_series,
        colors=colors,
        ylabel="Subgroup support ratio (support / N_real)",
        title="Subgroup Support Distribution (Ratio) across 6 datasets",
        output_path=output_dir / "six_datasets_support_ratio.png",
        log_scale=False,
    )

    _write_summary(output_dir / "six_datasets_support_summary.csv", summary_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "datasets": dataset_ids,
                "max_columns": args.max_columns,
                "max_combos": args.max_combos,
                "numeric_bins": args.numeric_bins,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "datasets": dataset_ids}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot subgroup support distributions for 6 datasets.")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--max-columns", type=int, default=6)
    parser.add_argument("--max-combos", type=int, default=18)
    parser.add_argument("--numeric-bins", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT.parent.parent / "Query" / "code" / "logs" / "analysis" / f"six_dataset_support_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
