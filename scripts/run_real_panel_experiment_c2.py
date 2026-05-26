#!/usr/bin/env python3
"""Run one real end-to-end c2 experiment on synthetic model panel."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.common import SQL_SOURCE_VERSION_CHOICES
from src.evaluation.real_panel_experiment import run_real_panel_experiment_c2


def _format_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    out = []
    for ridx, row in enumerate(rows):
        line = "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"
        out.append(line)
        if ridx == 0:
            sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
            out.append(sep)
    return "\n".join(out)


def _write_report(output_dir: Path, args: argparse.Namespace) -> None:
    dataset_id = str(args.dataset_id)
    panel_inventory = json.loads((output_dir / f"panel_inventory_{dataset_id}.json").read_text(encoding="utf-8"))
    panel_manifest = json.loads((output_dir / f"model_panel_{dataset_id}.json").read_text(encoding="utf-8"))
    selected_workloads = json.loads((output_dir / f"selected_workloads_{dataset_id}.json").read_text(encoding="utf-8"))
    ranking = json.loads((output_dir / f"overall_ranking_{dataset_id}.json").read_text(encoding="utf-8"))

    self_eval_dir = output_dir / "self_evaluation"
    self_eval_status = str(selected_workloads.get("self_evaluation_status") or "completed")
    self_eval_skip_reason = str(selected_workloads.get("self_evaluation_skip_reason") or "")
    cir = {}
    acr = {}
    rank_stability = {}
    if self_eval_status == "completed":
        cir_path = self_eval_dir / "cir_report.json"
        acr_path = self_eval_dir / "acr_report.json"
        rank_path = self_eval_dir / "rank_stability_report.json"
        if cir_path.exists():
            cir = json.loads(cir_path.read_text(encoding="utf-8"))
        if acr_path.exists():
            acr = json.loads(acr_path.read_text(encoding="utf-8"))
        if rank_path.exists():
            rank_stability = json.loads(rank_path.read_text(encoding="utf-8"))
    model_scores_csv = output_dir / f"model_scores_{dataset_id}.csv"

    usable_records = [row for row in panel_inventory.get("records", []) if row.get("usable")]
    unusable_records = [row for row in panel_inventory.get("records", []) if not row.get("usable")]

    model_rows = [["Model", "Run Count"]]
    for model in panel_manifest.get("models", []):
        model_rows.append([str(model.get("model_id")), str(model.get("run_count"))])

    workload_rows = [["Run ID", "Status", "Queries", "Baseline Valid", "Models"]]
    for item in selected_workloads.get("workloads", []):
        workload_rows.append(
            [
                str(item.get("run_id")),
                str(item.get("status")),
                str(item.get("queryspec_count", "-")),
                str(item.get("baseline_valid_query_count", "-")),
                str(item.get("model_count", "-")),
            ]
        )

    ranking_rows = [["Rank", "Model", "Overall Score"]]
    for row in ranking.get("ranking_primary", []):
        ranking_rows.append([
            str(row.get("rank")),
            str(row.get("model_id")),
            str(row.get("overall_score")),
        ])

    primary_run_id = str(selected_workloads.get("primary_workload_run_id") or "")
    family_score_rows = [[
        "Model",
        "Subgroup",
        "Conditional",
        "Tail/Rarity",
        "Missingness",
        "Overall",
    ]]
    validation_rows = [[
        "Model",
        "Cardinality/Range",
        "Missing Introduction",
        "Uniqueness",
        "Impossible-State",
    ]]
    if model_scores_csv.exists():
        with model_scores_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            records = [row for row in reader if str(row.get("workload_run_id") or "") == primary_run_id]
        records.sort(key=lambda r: float(r.get("overall_score") or 0.0), reverse=True)
        for row in records:
            family_score_rows.append(
                [
                    str(row.get("model_id") or ""),
                    str(row.get("subgroup_structure_score") or ""),
                    str(row.get("conditional_dependency_structure_score") or ""),
                    str(row.get("tail_rarity_structure_score") or ""),
                    str(row.get("missingness_structure_score") or ""),
                    str(row.get("overall_score") or ""),
                ]
            )
            validation_rows.append(
                [
                    str(row.get("model_id") or ""),
                    str(row.get("validation_cardinality_range_score") or ""),
                    str(row.get("validation_missing_introduction_score") or ""),
                    str(row.get("validation_uniqueness_integrity_score") or ""),
                    str(row.get("validation_impossible_state_score") or "N/A"),
                ]
            )

    rank_domains = rank_stability.get("domains") if isinstance(rank_stability.get("domains"), dict) else {}
    overall_rank_stability = rank_domains.get("overall", {}) if isinstance(rank_domains, dict) else {}
    overall_rank_summary = overall_rank_stability.get("summary", {}) if isinstance(overall_rank_stability, dict) else {}

    report = []
    report.append(f"# REAL_PANEL_EVAL_{dataset_id.upper()}_REPORT")
    report.append("")
    report.append(f"Generated at: {datetime.now().isoformat()}")
    report.append("")
    report.append("## 1) Panel discovery summary")
    report.append("")
    report.append(f"- Synthetic root: `{args.synthetic_root}`")
    report.append(f"- Dataset: `{dataset_id}`")
    report.append(f"- Discovered files: {len(panel_inventory.get('records', []))}")
    report.append(f"- Usable files: {len(usable_records)}")
    report.append(f"- Unusable files: {len(unusable_records)}")
    report.append(f"- Models with usable data: {len(panel_manifest.get('models', []))}")
    report.append("")
    report.append(_format_table(model_rows))
    report.append("")

    if unusable_records:
        report.append("Unusable files:")
        for item in unusable_records:
            report.append(f"- {item.get('path')}: {item.get('notes')}")
        report.append("")

    report.append("## 2) Workload summary")
    report.append("")
    report.append(f"- Primary workload run: `{selected_workloads.get('primary_workload_run_id')}`")
    report.append(f"- Candidate workload runs: {args.workload_run_ids}")
    report.append(f"- SQL source line: `{selected_workloads.get('sql_source_label')}` (`{selected_workloads.get('sql_source_version')}`)")
    report.append("")
    report.append(_format_table(workload_rows))
    report.append("")

    report.append("## 3) Real synthetic-data benchmark results")
    report.append("")
    report.append(f"- Primary ranking run: `{ranking.get('primary_workload_run_id')}`")
    report.append(f"- Ranking size: {len(ranking.get('ranking_primary', []))} models")
    report.append("")
    report.append(_format_table(ranking_rows))
    report.append("")
    report.append("Family-level scores on primary workload:")
    report.append("")
    report.append(_format_table(family_score_rows))
    report.append("")
    report.append("Validation (v0.4 deterministic channels) on primary workload:")
    report.append("")
    report.append(_format_table(validation_rows))
    report.append("")

    report.append("## 4) Self-evaluation results")
    report.append("")
    report.append(f"- Self-evaluation status: {self_eval_status}")
    if self_eval_status != "completed":
        report.append(f"- Reason: {self_eval_skip_reason or 'No additional reason recorded.'}")
    else:
        report.append(f"- CIR score (workload): {cir.get('summary', {}).get('workload_cir_score')}")
        report.append(f"- CIR effective score (coverage-adjusted): {cir.get('summary', {}).get('workload_cir_effective_score')}")
        report.append(f"- CIR evaluable query ratio: {cir.get('summary', {}).get('evaluable_query_ratio')}")
        report.append(f"- ACR score (workload): {acr.get('summary', {}).get('workload_acr_score')}")
        report.append(f"- ACR effective score (coverage-adjusted): {acr.get('summary', {}).get('workload_acr_effective_score')}")
        report.append(f"- ACR evaluable query ratio: {acr.get('summary', {}).get('evaluable_query_ratio')}")
        report.append(f"- Rank stability status: {rank_stability.get('summary', {}).get('status')}")
        report.append(f"- Rank stability domains: {rank_stability.get('summary', {}).get('domains')}")
        if overall_rank_summary:
            report.append(f"- Rank stability avg Kendall tau (overall): {overall_rank_summary.get('avg_kendall_tau')}")
            report.append(f"- Rank stability avg Spearman rho (overall): {overall_rank_summary.get('avg_spearman_rho')}")
            report.append(f"- Rank stability champion retention (overall): {overall_rank_summary.get('champion_retention_rate')}")
            report.append(f"- Rank stability avg top-{overall_rank_summary.get('top_k')} overlap (overall): {overall_rank_summary.get('avg_top_k_overlap')}")
            report.append(f"- Rank stability avg pairwise reversal ratio (overall): {overall_rank_summary.get('avg_pairwise_reversal_ratio')}")
    report.append("")

    report.append("## 5) Diagnostics / caveats")
    report.append("")
    report.append("- Query-family synthetic scoring still uses query-result similarity heuristics for analytics channels.")
    report.append("- Validation v0.4 channels are deterministic and column-level, reported independently.")
    report.append(f"- Panel appears to contain one synthetic run per model for {dataset_id}; repeat-level uncertainty is limited.")
    report.append("- Under v0.4 policy, if no uniqueness-eligible columns exist, uniqueness integrity is assigned full score (no penalty path).")
    report.append("- Impossible-state channel is currently a v0.4 placeholder and reported separately as N/A.")
    report.append("- Some compared historical workload runs predate build_manifest_v2, so build-level metadata fields are partially missing in rank-stability report.")
    report.append("- Rank stability uses workload-level score tables generated in this experiment.")
    report.append("- CIR evaluates target-family responsiveness vs off-target leakage under controlled perturbations.")
    report.append("- ACR evaluates conclusion robustness under constrained local rewrites (anti cherry-picking).")
    report.append("- Raw CIR/ACR should be interpreted together with evaluable coverage ratios and effective scores.")
    report.append("")

    report.append("## 6) Commands executed")
    report.append("")
    report.append("```bash")
    report.append("# Real panel experiment end-to-end")
    report.append(
        "./.venv/bin/python scripts/run_real_panel_experiment_c2.py "
        f"--synthetic-root {args.synthetic_root} --sql-source-version {args.sql_source_version} "
        f"--workload-run-ids {' '.join(args.workload_run_ids)} "
        f"--self-eval-max-queries {args.self_eval_max_queries} --output-dir {output_dir}"
        + (" --skip-self-eval" if args.skip_self_eval else "")
    )
    report.append("```")
    report.append("")

    report_path = output_dir / f"REAL_PANEL_EVAL_{dataset_id.upper()}_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real synthetic-panel experiment for c2.")
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "SynData" / "synthetic_10ds_7models",
        help="Synthetic panel root.",
    )
    parser.add_argument("--dataset-id", type=str, default="c2")
    parser.add_argument(
        "--workload-run-ids",
        nargs="+",
        default=["c2_20260329_171325", "c2_20260329_162247", "c2_20260329_004153"],
        help="Existing benchmark workload run IDs to score against panel.",
    )
    parser.add_argument(
        "--sql-source-version",
        choices=list(SQL_SOURCE_VERSION_CHOICES),
        default="v1",
        help="Which workload SQL line the run ids belong to.",
    )
    parser.add_argument(
        "--self-eval-max-queries",
        type=int,
        default=80,
        help="Max queries for alignment/purity evaluation on primary workload (0 means all).",
    )
    parser.add_argument(
        "--skip-self-eval",
        action="store_true",
        help="Skip STEP2 self-evaluation and only produce workload scoring outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir. Default: logs/experiments/c2_real_panel_eval_<timestamp>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "logs" / "experiments" / f"{args.dataset_id}_real_panel_eval_{timestamp}"
    )
    output_dir = output_dir.expanduser().resolve()

    result = run_real_panel_experiment_c2(
        synthetic_root=args.synthetic_root.expanduser().resolve(),
        dataset_id=args.dataset_id,
        workload_run_ids=args.workload_run_ids,
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        self_eval_max_queries=args.self_eval_max_queries,
        sql_source_version=str(args.sql_source_version),
        skip_self_eval=bool(args.skip_self_eval),
    )

    _write_report(output_dir, args)

    print(json.dumps({
        "status": "ok",
        "dataset_id": result.get("dataset_id"),
        "sql_source_version": result.get("sql_source_version"),
        "output_dir": result.get("output_dir"),
        "usable_synthetic_file_count": result.get("usable_synthetic_file_count"),
        "model_count": result.get("model_count"),
        "primary_workload_run_id": result.get("primary_workload_run_id"),
        "self_evaluation_status": result.get("self_evaluation_status"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
