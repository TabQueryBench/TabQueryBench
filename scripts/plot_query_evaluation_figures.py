#!/usr/bin/env python3
"""Generate slide-ready query evaluation figures (alignment/purity)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILY_ORDER = [
    "subgroup_structure",
    "conditional_dependency_structure",
    "tail_rarity_structure",
    "missingness_structure",
    "cardinality_structure",
]

FAMILY_LABEL = {
    "subgroup_structure": "Subgroup",
    "conditional_dependency_structure": "Conditional",
    "tail_rarity_structure": "Tail/Rarity",
    "missingness_structure": "Missingness",
    "cardinality_structure": "Cardinality",
}

FAMILY_COLOR = {
    "subgroup_structure": "#4C78A8",
    "conditional_dependency_structure": "#F58518",
    "tail_rarity_structure": "#54A24B",
    "missingness_structure": "#ECA82C",
    "cardinality_structure": "#B279A2",
}

STATUS_ORDER = [
    "High Align + High Purity",
    "High Align + Low Purity",
    "Low Align + High Purity",
    "Low Align + Low Purity",
]

STATUS_COLOR = {
    "High Align + High Purity": "#4C78A8",
    "High Align + Low Purity": "#F58518",
    "Low Align + High Purity": "#72B7B2",
    "Low Align + Low Purity": "#E45756",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _merge_query_eval(experiment_dir: Path) -> list[dict]:
    a_rows = _read_jsonl(experiment_dir / "self_evaluation" / "alignment_by_query.jsonl")
    p_rows = _read_jsonl(experiment_dir / "self_evaluation" / "purity_by_query.jsonl")
    p_map = {str(r.get("query_id")): r for r in p_rows}

    merged: list[dict] = []
    for a in a_rows:
        qid = str(a.get("query_id") or "")
        p = p_map.get(qid)
        if not p:
            continue
        fam = str(a.get("family_id") or "unknown")
        merged.append(
            {
                "query_id": qid,
                "family_id": fam,
                "alignment": float(a.get("alignment_score") or 0.0),
                "purity": float(p.get("purity_score") or 0.0),
                "high_contamination": bool(p.get("high_contamination")),
            }
        )
    return merged


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_quadrant(rows: list[dict], out_dir: Path, align_thr: float, purity_thr: float) -> None:
    bg = "#F5F6FA"
    fig, ax = plt.subplots(figsize=(8.5, 6.6), facecolor=bg)
    ax.set_facecolor(bg)

    for fam in FAMILY_ORDER:
        fam_rows = [r for r in rows if r["family_id"] == fam]
        if not fam_rows:
            continue
        ax.scatter(
            [r["alignment"] for r in fam_rows],
            [r["purity"] for r in fam_rows],
            s=38,
            alpha=0.78,
            color=FAMILY_COLOR[fam],
            label=FAMILY_LABEL[fam],
            edgecolors="white",
            linewidths=0.5,
        )

    ax.axvline(align_thr, color="#1F243A", linestyle="--", linewidth=1.4)
    ax.axhline(purity_thr, color="#1F243A", linestyle="--", linewidth=1.4)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Alignment Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Purity Score", fontsize=12, fontweight="bold")
    ax.set_title("Query-level Alignment vs Purity", fontsize=20, fontweight="bold", color="#1F243A", pad=10)

    # Quadrant counts (counts are correct; place labels in the correct quadrants):
    # HH = high alignment, high purity (top-right)
    # HL = high alignment, low purity (bottom-right)
    # LH = low alignment, high purity (top-left)
    # LL = low alignment, low purity (bottom-left)
    q1 = sum(1 for r in rows if r["alignment"] >= align_thr and r["purity"] >= purity_thr)
    q2 = sum(1 for r in rows if r["alignment"] >= align_thr and r["purity"] < purity_thr)
    q3 = sum(1 for r in rows if r["alignment"] < align_thr and r["purity"] >= purity_thr)
    q4 = sum(1 for r in rows if r["alignment"] < align_thr and r["purity"] < purity_thr)
    ax.text(0.84, 0.97, f"HH (count): {q1}", transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#1F243A")
    ax.text(0.84, 0.07, f"HL (count): {q2}", transform=ax.transAxes, va="bottom", ha="left", fontsize=10, color="#1F243A")
    ax.text(0.03, 0.97, f"LH (count): {q3}", transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#1F243A")
    ax.text(0.03, 0.07, f"LL (count): {q4}", transform=ax.transAxes, va="bottom", ha="left", fontsize=10, color="#1F243A")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    _save(fig, out_dir / "09_query_alignment_purity_quadrant.png")


def plot_family_means(experiment_dir: Path, out_dir: Path) -> None:
    a_report = _read_json(experiment_dir / "self_evaluation" / "alignment_report.json")
    p_report = _read_json(experiment_dir / "self_evaluation" / "purity_report.json")
    a_map = {str(r.get("family_id")): float(r.get("avg_alignment_score") or 0.0) for r in a_report.get("by_family", [])}
    p_map = {str(r.get("family_id")): float(r.get("avg_purity_score") or 0.0) for r in p_report.get("by_family", [])}

    labels = [FAMILY_LABEL[f] for f in FAMILY_ORDER]
    a_vals = [a_map.get(f, 0.0) for f in FAMILY_ORDER]
    p_vals = [p_map.get(f, 0.0) for f in FAMILY_ORDER]
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9.6, 5.6), facecolor="#F5F6FA")
    ax.set_facecolor("#F5F6FA")
    b1 = ax.bar(x - w / 2, a_vals, width=w, label="Alignment", color="#4C78A8")
    b2 = ax.bar(x + w / 2, p_vals, width=w, label="Purity", color="#72B7B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_title("Family-level Mean Scores", fontsize=20, fontweight="bold", color="#1F243A", pad=10)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    _save(fig, out_dir / "10_family_alignment_purity_means.png")


def plot_status_stacked(rows: list[dict], out_dir: Path, align_thr: float, purity_thr: float) -> None:
    def status(r: dict) -> str:
        a = r["alignment"] >= align_thr
        p = r["purity"] >= purity_thr
        if a and p:
            return "High Align + High Purity"
        if a and not p:
            return "High Align + Low Purity"
        if not a and p:
            return "Low Align + High Purity"
        return "Low Align + Low Purity"

    fam_counter: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        fam_counter[r["family_id"]][status(r)] += 1

    labels = [FAMILY_LABEL[f] for f in FAMILY_ORDER]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    fig, ax = plt.subplots(figsize=(10.4, 5.8), facecolor="#F5F6FA")
    ax.set_facecolor("#F5F6FA")
    for s in STATUS_ORDER:
        vals = np.array([fam_counter[f].get(s, 0) for f in FAMILY_ORDER], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=STATUS_COLOR[s], label=s, width=0.62)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Query Count")
    ax.set_title("Query Quality Segments by Family", fontsize=20, fontweight="bold", color="#1F243A", pad=10)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.22)
    _save(fig, out_dir / "11_query_quality_segments_stacked.png")


def plot_distributions(rows: list[dict], out_dir: Path) -> None:
    align = [r["alignment"] for r in rows]
    purity = [r["purity"] for r in rows]
    bins = np.linspace(0, 1, 16)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="#F5F6FA")
    for ax in axes:
        ax.set_facecolor("#F5F6FA")
        ax.grid(axis="y", alpha=0.25)

    axes[0].hist(align, bins=bins, color="#4C78A8", alpha=0.9, edgecolor="white")
    axes[0].set_title("Alignment Distribution", fontsize=14, fontweight="bold", color="#1F243A")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Alignment")
    axes[0].set_ylabel("Query Count")

    axes[1].hist(purity, bins=bins, color="#72B7B2", alpha=0.9, edgecolor="white")
    axes[1].set_title("Purity Distribution", fontsize=14, fontweight="bold", color="#1F243A")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Purity")

    fig.suptitle("Query-level Score Distributions", fontsize=19, fontweight="bold", color="#1F243A")
    _save(fig, out_dir / "12_query_score_distributions.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate query evaluation figures.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--alignment-threshold", type=float, default=0.45)
    parser.add_argument("--purity-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp = args.experiment_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else (exp / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _merge_query_eval(exp)
    if not rows:
        raise RuntimeError("No merged query evaluation rows found.")

    plot_scatter_quadrant(rows, out_dir, align_thr=args.alignment_threshold, purity_thr=args.purity_threshold)
    plot_family_means(exp, out_dir)
    plot_status_stacked(rows, out_dir, align_thr=args.alignment_threshold, purity_thr=args.purity_threshold)
    plot_distributions(rows, out_dir)
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "query_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
