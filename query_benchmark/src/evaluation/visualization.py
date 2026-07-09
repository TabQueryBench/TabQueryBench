"""Auto visualization for evaluation outputs (Rank Stability + CIR + ACR)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.benchmark.models import FIVE_FIXED_FAMILIES

FAMILY_LABEL = {
    "subgroup_structure": "Subgroup",
    "conditional_dependency_structure": "Conditional",
    "tail_rarity_structure": "Tail/Rarity",
    "missingness_structure": "Missingness",
    "cardinality_structure": "Cardinality",
}

FAMILY_COLOR = {
    "subgroup_structure": "#4C78A8",
    "conditional_dependency_structure": "#2A9D8F",
    "tail_rarity_structure": "#F4A261",
    "missingness_structure": "#6C757D",
    "cardinality_structure": "#B5179E",
}


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        except Exception:  # noqa: BLE001
            continue
    return rows


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _rank_stability_score(rank_report: dict[str, Any]) -> tuple[float, str]:
    top_summary = rank_report.get("summary") if isinstance(rank_report.get("summary"), dict) else {}
    if isinstance(top_summary, dict) and top_summary.get("rank_stability_score") is not None:
        return float(top_summary.get("rank_stability_score") or 0.0), "weighted (0.75 workload + 0.25 query)"
    domains = rank_report.get("domains") if isinstance(rank_report.get("domains"), dict) else {}
    overall = domains.get("overall", {}) if isinstance(domains, dict) else {}
    summary = overall.get("summary", {}) if isinstance(overall, dict) else {}
    if not summary:
        return 0.0, "N/A"
    return float(summary.get("avg_kendall_tau") or 0.0), "avg Kendall tau"


def _plot_metrics_overview(
    *,
    cir_report: dict[str, Any],
    acr_report: dict[str, Any],
    rank_report: dict[str, Any],
    out_dir: Path,
) -> Path:
    cir_summary = cir_report.get("summary", {}) if isinstance(cir_report.get("summary"), dict) else {}
    acr_summary = acr_report.get("summary", {}) if isinstance(acr_report.get("summary"), dict) else {}
    cir = float(cir_summary.get("workload_cir_score") or 0.0)
    acr = float(acr_summary.get("workload_acr_score") or 0.0)
    cir_eff = float(cir_summary.get("workload_cir_effective_score") or cir)
    acr_eff = float(acr_summary.get("workload_acr_effective_score") or acr)
    cir_cov = float(cir_summary.get("evaluable_query_ratio") or 0.0)
    acr_cov = float(acr_summary.get("evaluable_query_ratio") or 0.0)
    rank, rank_label = _rank_stability_score(rank_report)
    has_rank = rank_label != "N/A"

    labels = ["CIR", "ACR", "Rank Stability"]
    vals = [cir, acr, rank]
    colors = ["#2A9D8F", "#3A86FF", "#6D6875"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor="#F4F5FA")
    ax.set_facecolor("#F4F5FA")
    ax.grid(axis="y", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    bars = ax.bar(x, vals, color=colors, width=0.58)
    if not has_rank:
        bars[2].set_hatch("//")
        bars[2].set_alpha(0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(min(-0.25, min(vals) - 0.05), 1.05)
    ax.axhline(0.0, color="#9BA3B4", linewidth=1.0, alpha=0.8)
    ax.set_title("Evaluation Overview (Rank Stability + CIR + ACR)", fontsize=16, fontweight="bold", color="#1F243A")
    ax.set_ylabel("Score")
    subtitle = (
        f"Rank metric: {rank_label} | "
        f"CIR_eff={cir_eff:.3f} (cov={cir_cov:.2f}) | "
        f"ACR_eff={acr_eff:.3f} (cov={acr_cov:.2f})"
    )
    ax.text(0.98, 0.97, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#5A6275")
    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.015 if value >= 0 else -0.035),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
    out = out_dir / "01_metrics_overview.png"
    _save(fig, out)
    return out


def _plot_family_scores(
    *,
    cir_report: dict[str, Any],
    acr_report: dict[str, Any],
    out_dir: Path,
) -> Path:
    cir_map = {str(item.get("family_id")): float(item.get("avg_question_cir_score") or 0.0) for item in cir_report.get("by_family", [])}
    acr_map = {str(item.get("family_id")): float(item.get("avg_question_acr_score") or 0.0) for item in acr_report.get("by_family", [])}

    families = [family for family in FIVE_FIXED_FAMILIES if family in cir_map or family in acr_map]
    labels = [FAMILY_LABEL.get(family, family) for family in families]
    cir_vals = [cir_map.get(family, 0.0) for family in families]
    acr_vals = [acr_map.get(family, 0.0) for family in families]
    x = np.arange(len(families))
    w = 0.36

    fig, ax = plt.subplots(figsize=(10.4, 5.4), facecolor="#F4F5FA")
    ax.set_facecolor("#F4F5FA")
    ax.grid(axis="y", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    b1 = ax.bar(x - w / 2, cir_vals, width=w, color="#2A9D8F", label="CIR (question-level mean)")
    b2 = ax.bar(x + w / 2, acr_vals, width=w, color="#3A86FF", label="ACR (question-level mean)")
    ax.axhline(0.0, color="#9BA3B4", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(min(-0.25, min(cir_vals + acr_vals + [0.0]) - 0.05), 1.05)
    ax.set_title("Family-level CIR vs ACR", fontsize=16, fontweight="bold", color="#1F243A")
    ax.legend(frameon=False)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.012 if h >= 0 else -0.03),
                f"{h:.2f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8,
            )

    out = out_dir / "02_family_cir_acr.png"
    _save(fig, out)
    return out


def _plot_query_scatter(
    *,
    cir_rows: list[dict[str, Any]],
    acr_rows: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    acr_by_qid = {str(row.get("query_id") or ""): row for row in acr_rows}
    merged: list[dict[str, Any]] = []
    for row in cir_rows:
        qid = str(row.get("query_id") or "")
        if not qid or qid not in acr_by_qid:
            continue
        cir_score = row.get("cir_score")
        acr_score = acr_by_qid[qid].get("acr_score")
        if cir_score is None or acr_score is None:
            continue
        merged.append(
            {
                "query_id": qid,
                "family_id": str(row.get("family_id") or "unknown"),
                "cir": float(cir_score),
                "acr": float(acr_score),
            }
        )

    fig, ax = plt.subplots(figsize=(8.0, 6.0), facecolor="#F4F5FA")
    ax.set_facecolor("#F4F5FA")
    ax.grid(alpha=0.22)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    for family in FIVE_FIXED_FAMILIES:
        points = [row for row in merged if row["family_id"] == family]
        if not points:
            continue
        ax.scatter(
            [row["cir"] for row in points],
            [row["acr"] for row in points],
            s=36,
            alpha=0.78,
            color=FAMILY_COLOR.get(family, "#7A7A7A"),
            label=FAMILY_LABEL.get(family, family),
            edgecolors="white",
            linewidths=0.4,
        )

    ax.axvline(0.0, linestyle="--", linewidth=1.0, color="#9BA3B4")
    ax.axhline(0.5, linestyle="--", linewidth=1.0, color="#9BA3B4")
    ax.set_xlabel("CIR (target responsiveness - off-target leakage)")
    ax.set_ylabel("ACR (anti cherry-picking robustness)")
    ax.set_title("Query-level CIR vs ACR", fontsize=16, fontweight="bold", color="#1F243A")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    out = out_dir / "03_query_cir_acr_scatter.png"
    _save(fig, out)
    return out


def _plot_perturbation_validity(
    *,
    perturbation_manifest: dict[str, Any],
    out_dir: Path,
) -> Path:
    variants = [item for item in (perturbation_manifest.get("variants") or []) if isinstance(item, dict)]
    accepted = {family: 0 for family in FIVE_FIXED_FAMILIES}
    rejected = {family: 0 for family in FIVE_FIXED_FAMILIES}
    for row in variants:
        if str(row.get("kind") or "") != "family":
            continue
        family = str(row.get("family_id") or "")
        if family not in accepted:
            continue
        is_ok = bool((row.get("validity") or {}).get("accepted", False))
        if is_ok:
            accepted[family] += 1
        else:
            rejected[family] += 1

    labels = [FAMILY_LABEL.get(family, family) for family in FIVE_FIXED_FAMILIES]
    x = np.arange(len(labels))
    acc_vals = [accepted[family] for family in FIVE_FIXED_FAMILIES]
    rej_vals = [rejected[family] for family in FIVE_FIXED_FAMILIES]

    fig, ax = plt.subplots(figsize=(10.6, 5.3), facecolor="#F4F5FA")
    ax.set_facecolor("#F4F5FA")
    ax.grid(axis="y", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    b1 = ax.bar(x, acc_vals, color="#2A9D8F", label="Accepted variants")
    b2 = ax.bar(x, rej_vals, bottom=acc_vals, color="#E76F51", label="Rejected variants")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Variant count")
    ax.set_title("Perturbation Validity by Family", fontsize=16, fontweight="bold", color="#1F243A")
    ax.legend(frameon=False)

    for i, (a, r) in enumerate(zip(acc_vals, rej_vals)):
        total = a + r
        if total <= 0:
            continue
        rate = a / total
        ax.text(i, total + 0.05, f"{a}/{total} ({rate:.0%})", ha="center", va="bottom", fontsize=8)

    out = out_dir / "04_perturbation_validity.png"
    _save(fig, out)
    return out


def _plot_rank_domains(
    *,
    rank_report: dict[str, Any],
    out_dir: Path,
) -> Path | None:
    domains = rank_report.get("domains") if isinstance(rank_report.get("domains"), dict) else {}
    if not domains:
        return None

    labels: list[str] = []
    tau_vals: list[float] = []
    rho_vals: list[float] = []
    for domain in ["overall"] + list(FIVE_FIXED_FAMILIES):
        row = domains.get(domain, {})
        summary = row.get("summary", {}) if isinstance(row, dict) else {}
        if not summary:
            continue
        labels.append("Overall" if domain == "overall" else FAMILY_LABEL.get(domain, domain))
        tau_vals.append(float(summary.get("avg_kendall_tau") or 0.0))
        rho_vals.append(float(summary.get("avg_spearman_rho") or 0.0))
    if not labels:
        return None

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 5.2), facecolor="#F4F5FA")
    ax.set_facecolor("#F4F5FA")
    ax.grid(axis="y", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    b1 = ax.bar(x - w / 2, tau_vals, width=w, color="#2A9D8F", label="Avg Kendall tau")
    b2 = ax.bar(x + w / 2, rho_vals, width=w, color="#457B9D", label="Avg Spearman rho")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.04)
    ax.set_title("Rank Stability by Domain", fontsize=16, fontweight="bold", color="#1F243A")
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    out = out_dir / "05_rank_stability_domains.png"
    _save(fig, out)
    return out


def _plot_acr_rewrite_diagnostics(
    *,
    acr_rewrite_rows: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    templates = ["refinement_rewrite", "filter_neighborhood_rewrite", "population_neighborhood_rewrite"]
    generated = {template: 0 for template in templates}
    valid = {template: 0 for template in templates}
    for row in acr_rewrite_rows:
        template = str(row.get("template_type") or "")
        if template not in generated:
            continue
        generated[template] += 1
        if bool(row.get("valid_rewrite")):
            valid[template] += 1

    labels = ["Refinement", "Filter-neighborhood", "Population-neighborhood"]
    gen_vals = [generated[t] for t in templates]
    val_vals = [valid[t] for t in templates]
    ratio_vals = [(valid[t] / generated[t]) if generated[t] > 0 else 0.0 for t in templates]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax1 = plt.subplots(figsize=(10.2, 5.2), facecolor="#F4F5FA")
    ax1.set_facecolor("#F4F5FA")
    ax1.grid(axis="y", alpha=0.22)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    b1 = ax1.bar(x - w / 2, gen_vals, width=w, color="#94A3B8", label="Generated rewrites")
    b2 = ax1.bar(x + w / 2, val_vals, width=w, color="#3A86FF", label="Valid rewrites")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Rewrite count")
    ax1.set_title("ACR Rewrite Validity by Template", fontsize=16, fontweight="bold", color="#1F243A")

    ax2 = ax1.twinx()
    ax2.plot(x, ratio_vals, color="#2A9D8F", marker="o", linewidth=2.2, label="Valid ratio")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Valid ratio")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.7, f"{int(h)}", ha="center", va="bottom", fontsize=8)
    for xi, ratio in zip(x, ratio_vals):
        ax2.text(xi, ratio + 0.03, f"{ratio:.2f}", ha="center", va="bottom", fontsize=8, color="#2A9D8F")

    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, frameon=False, loc="upper right")

    out = out_dir / "06_acr_rewrite_validity.png"
    _save(fig, out)
    return out


def generate_standard_evaluation_figures(evaluation_dir: Path) -> dict[str, Any]:
    """Generate a standard figure pack under <evaluation_dir>/figures."""
    evaluation_dir = evaluation_dir.resolve()
    out_dir = evaluation_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    cir_report = _read_json(evaluation_dir / "cir_report.json", {})
    acr_report = _read_json(evaluation_dir / "acr_report.json", {})
    rank_report = _read_json(evaluation_dir / "rank_stability_report.json", {})
    perturbation_manifest = _read_json(evaluation_dir / "perturbation_manifest.json", {})
    cir_rows = _read_jsonl(evaluation_dir / "cir_by_query.jsonl")
    acr_rows = _read_jsonl(evaluation_dir / "acr_by_query.jsonl")
    acr_rewrite_rows = _read_jsonl(evaluation_dir / "acr_rewrite_trace.jsonl")

    files: list[str] = []
    files.append(str(_plot_metrics_overview(cir_report=cir_report, acr_report=acr_report, rank_report=rank_report, out_dir=out_dir)))
    files.append(str(_plot_family_scores(cir_report=cir_report, acr_report=acr_report, out_dir=out_dir)))
    files.append(str(_plot_query_scatter(cir_rows=cir_rows, acr_rows=acr_rows, out_dir=out_dir)))
    files.append(str(_plot_perturbation_validity(perturbation_manifest=perturbation_manifest, out_dir=out_dir)))
    rank_path = _plot_rank_domains(rank_report=rank_report, out_dir=out_dir)
    if rank_path is not None:
        files.append(str(rank_path))
    files.append(str(_plot_acr_rewrite_diagnostics(acr_rewrite_rows=acr_rewrite_rows, out_dir=out_dir)))

    return {
        "status": "ok",
        "figures_dir": str(out_dir),
        "files": files,
    }
