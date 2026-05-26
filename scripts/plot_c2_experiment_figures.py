#!/usr/bin/env python3
"""Generate slide-ready figures from a c2 real-panel experiment directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FAMILIES = [
    "subgroup_structure",
    "conditional_dependency_structure",
    "tail_rarity_structure",
    "missingness_structure",
    "cardinality_structure",
]


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


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overall_ranking(experiment_dir: Path, out_dir: Path) -> None:
    ranking = _read_json(experiment_dir / "overall_ranking_c2.json")
    rows = ranking.get("ranking_primary", [])
    models = [str(row.get("model_id")) for row in rows]
    scores = [float(row.get("overall_score") or 0.0) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(models, scores, color="#4C78A8")
    ax.set_title("c2 Overall Ranking (Primary Workload)")
    ax.set_ylabel("Overall Score")
    ax.set_ylim(0.0, max(scores + [0.5]) * 1.15)
    ax.tick_params(axis="x", rotation=30)
    for i, score in enumerate(scores):
        ax.text(i, score + 0.005, f"{score:.3f}", ha="center", va="bottom", fontsize=8)
    _save(fig, out_dir / "01_overall_ranking_bar.png")


def plot_family_heatmap(experiment_dir: Path, out_dir: Path) -> None:
    ranking = _read_json(experiment_dir / "overall_ranking_c2.json")
    primary_run = str(ranking.get("primary_workload_run_id"))
    rows = _read_csv(experiment_dir / "model_scores_c2.csv")
    rows = [r for r in rows if str(r.get("workload_run_id")) == primary_run]
    rows.sort(key=lambda r: float(r.get("overall_score") or 0.0), reverse=True)

    models = [str(r["model_id"]) for r in rows]
    values = []
    for r in rows:
        values.append([float(r.get(f"{fam}_score") or 0.0) for fam in FAMILIES])

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_title(f"Family Score Heatmap ({primary_run})")
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels(
        ["subgroup", "conditional", "tail_rarity", "missingness", "cardinality"],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Family Score")
    _save(fig, out_dir / "02_family_score_heatmap.png")


def plot_rank_stability(experiment_dir: Path, out_dir: Path) -> None:
    report = _read_json(experiment_dir / "self_evaluation" / "rank_stability_report.json")
    domains = report.get("domains") if isinstance(report.get("domains"), dict) else {}
    overall = domains.get("overall", {}) if isinstance(domains, dict) else {}
    pairwise = overall.get("pairwise", []) if isinstance(overall, dict) else []
    labels = [f"{p.get('left_build')} vs {p.get('right_build')}" for p in pairwise]
    taus = [float(p.get("kendall_tau") or 0.0) for p in pairwise]
    rhos = [float(p.get("spearman_rho") or 0.0) for p in pairwise]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = list(range(len(labels)))
    ax.plot(x, taus, marker="o", label="Kendall tau", color="#F58518")
    ax.plot(x, rhos, marker="s", label="Spearman rho", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Correlation")
    ax.set_title("Rank Stability Across Workload Builds (Overall Domain)")
    ax.legend(loc="lower right")
    _save(fig, out_dir / "03_rank_stability_pairwise.png")


def plot_alignment_family(experiment_dir: Path, out_dir: Path) -> None:
    report = _read_json(experiment_dir / "self_evaluation" / "alignment_report.json")
    rows = report.get("by_family", [])
    families = [str(r.get("family_id")) for r in rows]
    align = [float(r.get("avg_alignment_score") or 0.0) for r in rows]
    agree = [float(r.get("family_agreement_rate") or 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = list(range(len(families)))
    w = 0.38
    ax.bar([i - w / 2 for i in x], align, width=w, label="Avg Alignment Score", color="#4C78A8")
    ax.bar([i + w / 2 for i in x], agree, width=w, label="Family Agreement Rate", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_structure", "") for f in families], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Alignment Diagnostics by Family")
    ax.legend()
    _save(fig, out_dir / "04_alignment_family_bars.png")


def plot_purity_family(experiment_dir: Path, out_dir: Path) -> None:
    report = _read_json(experiment_dir / "self_evaluation" / "purity_report.json")
    rows = report.get("by_family", [])
    families = [str(r.get("family_id")) for r in rows]
    purity = [float(r.get("avg_purity_score") or 0.0) for r in rows]
    high_contam = [float(r.get("high_contamination_query_ratio") or 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = list(range(len(families)))
    w = 0.38
    ax.bar([i - w / 2 for i in x], purity, width=w, label="Avg Purity Score", color="#72B7B2")
    ax.bar([i + w / 2 for i in x], high_contam, width=w, label="High Contamination Ratio", color="#B279A2")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_structure", "") for f in families], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Purity Diagnostics by Family")
    ax.legend()
    _save(fig, out_dir / "05_purity_family_bars.png")


def plot_alignment_vs_purity_scatter(experiment_dir: Path, out_dir: Path) -> None:
    a_rows = _read_jsonl(experiment_dir / "self_evaluation" / "alignment_by_query.jsonl")
    p_rows = _read_jsonl(experiment_dir / "self_evaluation" / "purity_by_query.jsonl")
    p_by_qid = {str(r.get("query_id")): r for r in p_rows}

    x_align = []
    y_purity = []
    colors = []
    for a in a_rows:
        qid = str(a.get("query_id"))
        p = p_by_qid.get(qid)
        if not p:
            continue
        x_align.append(float(a.get("alignment_score") or 0.0))
        y_purity.append(float(p.get("purity_score") or 0.0))
        colors.append(str(a.get("family_id") or "unknown"))

    palette = {
        "subgroup_structure": "#4C78A8",
        "conditional_dependency_structure": "#F58518",
        "tail_rarity_structure": "#54A24B",
        "missingness_structure": "#EECA3B",
        "cardinality_structure": "#B279A2",
        "unknown": "#9D9D9D",
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    for fam in sorted(set(colors)):
        idx = [i for i, c in enumerate(colors) if c == fam]
        ax.scatter(
            [x_align[i] for i in idx],
            [y_purity[i] for i in idx],
            s=24,
            alpha=0.75,
            label=fam.replace("_structure", ""),
            color=palette.get(fam, "#9D9D9D"),
        )
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Alignment Score")
    ax.set_ylabel("Purity Score")
    ax.set_title("Query-level Alignment vs Purity")
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, out_dir / "06_query_alignment_vs_purity_scatter.png")


def plot_evidence_components(experiment_dir: Path, out_dir: Path) -> None:
    report = _read_json(experiment_dir / "self_evaluation" / "evidence_sufficiency_report.json")
    rows = report.get("by_family", [])
    families = [str(r.get("family_id")) for r in rows]
    coverage = [float(r.get("facet_coverage_ratio") or 0.0) for r in rows]
    balance = [float(r.get("facet_balance_score") or 0.0) for r in rows]
    distinct = [float(r.get("distinct_question_angle_ratio") or 0.0) for r in rows]
    avg_q = [float(r.get("avg_question_evidence_score") or 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    x = list(range(len(families)))
    ax.bar(x, coverage, label="Facet Coverage", color="#4C78A8")
    ax.bar(x, balance, bottom=coverage, label="Facet Balance", color="#72B7B2")
    ax.bar(
        x,
        distinct,
        bottom=[coverage[i] + balance[i] for i in x],
        label="Distinct Angle",
        color="#F58518",
    )
    ax.bar(
        x,
        avg_q,
        bottom=[coverage[i] + balance[i] + distinct[i] for i in x],
        label="Avg Question Evidence",
        color="#54A24B",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_structure", "") for f in families], rotation=20, ha="right")
    ax.set_title("Evidence Components by Family (Raw Inputs)")
    ax.set_ylabel("Component Value (not weighted sum)")
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, out_dir / "07_evidence_family_components_stacked.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate c2 experiment figures.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to c2 real-panel experiment directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: <experiment-dir>/figures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    out_dir = (args.output_dir.expanduser().resolve() if args.output_dir else (experiment_dir / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_overall_ranking(experiment_dir, out_dir)
    plot_family_heatmap(experiment_dir, out_dir)
    plot_rank_stability(experiment_dir, out_dir)
    plot_alignment_family(experiment_dir, out_dir)
    plot_purity_family(experiment_dir, out_dir)
    plot_alignment_vs_purity_scatter(experiment_dir, out_dir)
    plot_evidence_components(experiment_dir, out_dir)

    print(json.dumps({"status": "ok", "figures_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

