#!/usr/bin/env python3
"""Generate polished C2 evaluation figures for slides.

Outputs:
- rank stability dashboard
- evidence/coverage dashboard
- four-metric overview (RankStability/Evidence/Alignment/Purity)
"""

from __future__ import annotations

import argparse
import json
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_rank_stability_dashboard(exp_dir: Path, out_dir: Path) -> Path:
    report = _read_json(exp_dir / "self_evaluation" / "rank_stability_report.json")
    domains = report.get("domains", {}) if isinstance(report.get("domains", {}), dict) else {}
    overall = domains.get("overall", {}) if isinstance(domains, dict) else {}
    summary = overall.get("summary", {}) if isinstance(overall, dict) else {}
    pairwise = overall.get("pairwise", []) if isinstance(overall, dict) else []

    # Left panel: overall pairwise metrics
    labels = [f"{p.get('left_build','?').replace('c2_','')}\nvs\n{p.get('right_build','?').replace('c2_','')}" for p in pairwise]
    kendall = [float(p.get("kendall_tau") or 0.0) for p in pairwise]
    spearman = [float(p.get("spearman_rho") or 0.0) for p in pairwise]
    topk = [float(p.get("top_k_overlap") or 0.0) for p in pairwise]
    rev_ok = [1.0 - float(p.get("pairwise_reversal_ratio") or 0.0) for p in pairwise]

    # Right panel: per-domain average Kendall/Spearman
    dom_labels = []
    dom_k = []
    dom_s = []
    for fam in ["overall"] + FAMILY_ORDER:
        row = domains.get(fam, {})
        s = row.get("summary", {}) if isinstance(row, dict) else {}
        if not s:
            continue
        dom_labels.append("Overall" if fam == "overall" else FAMILY_LABEL.get(fam, fam))
        dom_k.append(float(s.get("avg_kendall_tau") or 0.0))
        dom_s.append(float(s.get("avg_spearman_rho") or 0.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.6), facecolor="#F4F5FA")
    for ax in (ax1, ax2):
        ax.set_facecolor("#F4F5FA")
        ax.grid(axis="y", alpha=0.22)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    x = np.arange(len(labels))
    if len(labels) > 0:
        ax1.plot(x, kendall, marker="o", color="#2A9D8F", linewidth=2.3, label="Kendall tau")
        ax1.plot(x, spearman, marker="s", color="#457B9D", linewidth=2.3, label="Spearman rho")
        ax1.plot(x, topk, marker="^", color="#E9C46A", linewidth=2.3, label="Top-k overlap")
        ax1.plot(x, rev_ok, marker="D", color="#F4A261", linewidth=2.3, label="1 - reversal ratio")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0.0, 1.04)
    ax1.set_title("Pairwise Build Stability (Overall Domain)", fontsize=13, fontweight="bold", color="#1F243A")
    ax1.legend(frameon=False, fontsize=8, loc="lower right")

    x2 = np.arange(len(dom_labels))
    w = 0.38
    b1 = ax2.bar(x2 - w / 2, dom_k, width=w, color="#2A9D8F", label="Avg Kendall")
    b2 = ax2.bar(x2 + w / 2, dom_s, width=w, color="#457B9D", label="Avg Spearman")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(dom_labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylim(0.0, 1.04)
    ax2.set_title("Stability by Domain", fontsize=13, fontweight="bold", color="#1F243A")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"Rank Stability Dashboard  |  champion_retention={summary.get('champion_retention_rate', 0):.2f}",
        fontsize=18,
        fontweight="bold",
        color="#1F243A",
    )

    out = out_dir / "13_rank_stability_dashboard.png"
    _save(fig, out)
    return out


def plot_evidence_coverage_dashboard(exp_dir: Path, out_dir: Path) -> Path:
    report = _read_json(exp_dir / "self_evaluation" / "evidence_sufficiency_report.json")
    rows = report.get("by_family", [])
    row_map = {str(r.get("family_id")): r for r in rows}

    fams = [f for f in FAMILY_ORDER if f in row_map]
    labels = [FAMILY_LABEL[f] for f in fams]
    fam_score = [float(row_map[f].get("family_evidence_sufficient_score") or 0.0) for f in fams]
    cov = [float(row_map[f].get("facet_coverage_ratio") or 0.0) for f in fams]
    bal = [float(row_map[f].get("facet_balance_score") or 0.0) for f in fams]
    dis = [float(row_map[f].get("distinct_question_angle_ratio") or 0.0) for f in fams]
    qev = [float(row_map[f].get("avg_question_evidence_score") or 0.0) for f in fams]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.6), facecolor="#F4F5FA")
    for ax in (ax1, ax2):
        ax.set_facecolor("#F4F5FA")
        ax.grid(axis="y", alpha=0.22)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    x = np.arange(len(labels))
    bars = ax1.bar(x, fam_score, color=["#3A86FF", "#00A6FB", "#2EC4B6", "#90BE6D", "#BDB2FF"][: len(x)], width=0.62)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.set_ylim(0.0, 1.04)
    ax1.set_title("Family Evidence Sufficient Score", fontsize=13, fontweight="bold", color="#1F243A")
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # Component chart
    w = 0.2
    ax2.bar(x - 1.5 * w, cov, width=w, label="Facet coverage", color="#00B4D8")
    ax2.bar(x - 0.5 * w, bal, width=w, label="Facet balance", color="#90BE6D")
    ax2.bar(x + 0.5 * w, dis, width=w, label="Distinct angle", color="#F9C74F")
    ax2.bar(x + 1.5 * w, qev, width=w, label="Avg question evidence", color="#F9844A")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=18, ha="right")
    ax2.set_ylim(0.0, 1.04)
    ax2.set_title("Coverage Components by Family", fontsize=13, fontweight="bold", color="#1F243A")
    ax2.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    fig.suptitle(
        f"Evidence & Coverage Dashboard  |  workload={float(report.get('summary', {}).get('workload_evidence_sufficient_score', 0.0)):.3f}",
        fontsize=18,
        fontweight="bold",
        color="#1F243A",
    )
    out = out_dir / "14_evidence_coverage_dashboard.png"
    _save(fig, out)
    return out


def plot_four_metric_overview(exp_dir: Path, out_dir: Path) -> Path:
    e = _read_json(exp_dir / "self_evaluation" / "evidence_sufficiency_report.json")
    a = _read_json(exp_dir / "self_evaluation" / "alignment_report.json")
    p = _read_json(exp_dir / "self_evaluation" / "purity_report.json")
    r = _read_json(exp_dir / "self_evaluation" / "rank_stability_report.json")
    rd = r.get("domains", {}) if isinstance(r.get("domains", {}), dict) else {}
    rs_overall = float(rd.get("overall", {}).get("summary", {}).get("avg_kendall_tau") or 0.0)

    metrics = [
        ("Rank Stability", rs_overall),
        ("Evidence", float(e.get("summary", {}).get("workload_evidence_sufficient_score") or 0.0)),
        ("Alignment", float(a.get("summary", {}).get("workload_alignment_score") or 0.0)),
        ("Purity", float(p.get("summary", {}).get("workload_purity_score") or 0.0)),
    ]

    labels = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]
    colors = ["#2A9D8F", "#3A86FF", "#457B9D", "#72B7B2"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor="#F4F5FA")
    ax.set_facecolor("#F4F5FA")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    bars = ax.bar(x, vals, color=colors, width=0.56)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("C2 Evaluation Overview (4 Metrics)", fontsize=20, fontweight="bold", color="#1F243A", pad=8)
    ax.axhline(0.5, linestyle="--", linewidth=1.2, color="#9BA3B4", alpha=0.9)
    ax.text(3.4, 0.515, "0.5 reference", fontsize=8, color="#687083")
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    out = out_dir / "15_c2_four_eval_overview.png"
    _save(fig, out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate polished c2 evaluation overview figures.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <experiment-dir>/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp = args.experiment_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else (exp / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    f1 = plot_rank_stability_dashboard(exp, out_dir)
    f2 = plot_evidence_coverage_dashboard(exp, out_dir)
    f3 = plot_four_metric_overview(exp, out_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(out_dir),
                "files": [str(f1), str(f2), str(f3)],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
