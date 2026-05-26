"""Rank-stability evaluation over SQL-derived model rankings."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.eval.analytics_contract import (
    CANONICAL_ANALYTICS_SUBITEMS,
    annotate_query_row_with_contract,
    canonical_subitem_score_field,
)
from src.eval.common import (
    DEFAULT_SQL_SOURCE_VERSION,
    PROVENANCE_CONTRACT_VERSION,
    TaskProgressTracker,
    list_dataset_ids,
    make_task_run_dir,
    mean_or_none,
    normalize_sql_source_version,
    now_run_tag,
    read_json,
    sql_source_family,
    sql_source_label,
    sql_source_line_version,
    write_csv,
    write_json,
)
from src.eval.final_outputs import (
    build_longtable_report_tex,
    compile_tex,
    copy_files,
    STANDARD_MODEL_ORDER,
    normalize_standard_model_id,
    render_pdf_to_png,
    task_version_final_dir,
    write_json as write_final_json,
    write_versioned_final_readme,
)
from src.evaluation.rank_stability import (
    _kendall_tau,
    _pairwise_reversal_ratio,
    _rank_map,
    _rank_models,
    _spearman_rho,
    _topk_overlap,
)

TASK_NAME = "sql_eval"
SQL_BIG_BLOCK_FIELDS = [
    "analysis_overall_score",
    "subgroup_structure_score",
    "conditional_dependency_structure_score",
    "tail_rarity_structure_score",
    "missingness_structure_score",
]
SQL_SUBITEM_FIELDS = [
    canonical_subitem_score_field(family_id, subitem_id)
    for family_id, subitems in CANONICAL_ANALYTICS_SUBITEMS.items()
    for subitem_id in subitems
]
SQL_SOURCE_CONTEXT_FIELDS = [
    "provenance_contract_version",
    "sql_source_family",
    "sql_source_line_version",
    "sql_source_version",
    "sql_source_label",
    "sql_source_description",
    "sql_source_root",
    "sql_source_registry_root",
]
REAL_DATASET_CONTEXT_FIELDS = [
    "real_reference_split",
    "real_source_kind",
    "real_source_dataset_id",
    "real_source_split",
    "real_source_path",
    "real_source_exists",
    "real_source_mtime_utc",
    "real_source_size_bytes",
]


def _merge_contexts(*contexts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for context in contexts:
        for key, value in context.items():
            if value in (None, ""):
                continue
            merged[key] = value
    return merged


def _dataset_source_context(sql_source_meta: dict[str, Any], sample_row: dict[str, Any]) -> dict[str, Any]:
    return _merge_contexts(
        {field: sql_source_meta.get(field) for field in SQL_SOURCE_CONTEXT_FIELDS},
        {field: sample_row.get(field) for field in REAL_DATASET_CONTEXT_FIELDS},
    )


def _load_analysis_rows(analysis_run_dir: Path) -> list[dict[str, Any]]:
    path = analysis_run_dir / "summaries" / "analysis_query_scores__all_datasets.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        import json

        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _dataset_sort_key(dataset_id: str) -> tuple[str, int, str]:
    text = str(dataset_id or "").strip().lower()
    prefix = text[:1]
    suffix = text[1:]
    try:
        numeric = int(suffix)
    except Exception:
        numeric = 10**9
    return (prefix, numeric, text)


def _overall_ranking(query_rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in query_rows:
        asset_key = str(row.get("asset_key") or "")
        if not asset_key:
            continue
        if row.get("synthetic_exec_ok") is False:
            continue
        scores[asset_key].append(float(row.get("query_score") or 0.0))
    averaged = {asset_key: mean(values) for asset_key, values in scores.items() if values}
    return _rank_models(averaged)


def _ranking_for_subset(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for row in rows:
        asset_key = str(row.get("asset_key") or "")
        if not asset_key:
            continue
        if row.get("synthetic_exec_ok") is False:
            continue
        scores[asset_key] = float(row.get("query_score") or 0.0)
    return _rank_models(scores)


def _compare_rankings(reference: list[tuple[str, float]], candidate: list[tuple[str, float]], top_k: int) -> dict[str, Any] | None:
    order_ref = [model for model, _ in reference]
    order_cand = [model for model, _ in candidate]
    common = [model for model in order_ref if model in set(order_cand)]
    if len(common) < 2:
        return None
    ref_scores = {model: score for model, score in reference if model in common}
    cand_scores = {model: score for model, score in candidate if model in common}
    ref_ranked = _rank_models(ref_scores)
    cand_ranked = _rank_models(cand_scores)
    ref_order = [model for model, _ in ref_ranked]
    cand_order = [model for model, _ in cand_ranked]
    ref_rank_map = _rank_map(ref_scores)
    cand_rank_map = _rank_map(cand_scores)
    reversal_ratio, _ = _pairwise_reversal_ratio(ref_order, cand_order)
    return {
        "kendall_tau": round(_kendall_tau(ref_order, cand_order), 6),
        "spearman_rho": round(_spearman_rho(ref_rank_map, cand_rank_map), 6),
        "champion_same": bool(ref_order and cand_order and ref_order[0] == cand_order[0]),
        "top_k_overlap": round(_topk_overlap(ref_order, cand_order, top_k), 6),
        "pairwise_reversal_ratio": round(reversal_ratio, 6),
        "reference_asset_count": len(ref_order),
        "candidate_asset_count": len(cand_order),
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolve_analysis_sql_source_metadata(analysis_run_dir: Path, query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = read_json(analysis_run_dir / "manifest.json", {}) or {}
    sample_row = query_rows[0] if query_rows else {}
    version = str(manifest.get("sql_source_version") or "")
    label = str(manifest.get("sql_source_label") or "")
    root = str(manifest.get("sql_source_root") or "")
    if not version and query_rows:
        version = str(sample_row.get("sql_source_version") or "")
    if not label and query_rows:
        label = str(sample_row.get("sql_source_label") or "")
    if not root and query_rows:
        root = str(sample_row.get("sql_source_root") or "")
    if not version and query_rows:
        has_explicit_source = any(str(row.get("sql_source_version") or "").strip() for row in query_rows[:10])
        if not has_explicit_source:
            version = "v1"
    normalized = normalize_sql_source_version(version or DEFAULT_SQL_SOURCE_VERSION)
    return {
        "provenance_contract_version": str(
            manifest.get("provenance_contract_version")
            or sample_row.get("provenance_contract_version")
            or PROVENANCE_CONTRACT_VERSION
        ),
        "real_reference_split": str(
            manifest.get("real_reference_split")
            or sample_row.get("real_reference_split")
            or sample_row.get("real_source_split")
            or "train"
        ),
        "sql_source_family": str(
            manifest.get("sql_source_family")
            or sample_row.get("sql_source_family")
            or sql_source_family(normalized)
        ),
        "sql_source_line_version": str(
            manifest.get("sql_source_line_version")
            or sample_row.get("sql_source_line_version")
            or sql_source_line_version(normalized)
        ),
        "sql_source_version": normalized,
        "sql_source_label": label or sql_source_label(normalized),
        "sql_source_description": str(
            manifest.get("sql_source_description") or sample_row.get("sql_source_description") or ""
        ),
        "sql_source_root": root,
        "sql_source_registry_root": str(
            manifest.get("sql_source_registry_root") or sample_row.get("sql_source_registry_root") or ""
        ),
    }


def _aggregate_group_scores(rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field) or "")].append(row)
    output: list[dict[str, Any]] = []
    for group_value, items in sorted(grouped.items()):
        output.append(
            {
                **_merge_contexts(
                    {
                        field: items[0].get(field)
                        for field in [
                            "provenance_contract_version",
                            "real_reference_split",
                            "sql_source_family",
                            "sql_source_line_version",
                            "sql_source_version",
                            "sql_source_label",
                            "sql_source_description",
                            "sql_source_root",
                            "sql_source_registry_root",
                        ]
                    }
                ),
                group_field: group_value,
                "dataset_count": len({str(item.get("dataset_id") or "") for item in items if item.get("dataset_id")}),
                "mean_kendall_tau": round(mean(float(item["avg_kendall_tau"]) for item in items), 6),
                "mean_spearman_rho": round(mean(float(item["avg_spearman_rho"]) for item in items), 6),
                "mean_champion_retention_rate": round(mean(float(item["champion_retention_rate"]) for item in items), 6),
                "mean_top_k_overlap": round(mean(float(item["avg_top_k_overlap"]) for item in items), 6),
                "mean_pairwise_reversal_ratio": round(
                    mean(float(item["avg_pairwise_reversal_ratio"]) for item in items),
                    6,
                ),
            }
        )
    return output


def _build_sql_dataset_model_rows(
    *,
    query_rows: list[dict[str, Any]],
    dataset_summary_rows: list[dict[str, Any]],
    sql_source_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    dataset_ids = sorted(list_dataset_ids(), key=_dataset_sort_key)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw_row in query_rows:
        dataset_id = str(raw_row.get("dataset_id") or "").strip()
        model_id = normalize_standard_model_id(raw_row.get("model_id"))
        if dataset_id not in dataset_ids:
            continue
        if model_id not in STANDARD_MODEL_ORDER:
            continue
        row = annotate_query_row_with_contract(raw_row)
        grouped[(dataset_id, model_id)].append(row)

    dataset_rank_summary = {str(row.get("dataset_id") or ""): row for row in dataset_summary_rows}
    output: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        dataset_rank = dataset_rank_summary.get(dataset_id, {})
        dataset_context = _dataset_source_context(sql_source_meta, dataset_rank)
        for model_id in STANDARD_MODEL_ORDER:
            rows = grouped.get((dataset_id, model_id), [])
            payload: dict[str, Any] = {
                **dataset_context,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "coverage_status": "ok" if rows else "missing_asset",
                "asset_count": len({str(row.get("asset_key") or "") for row in rows if row.get("asset_key")}),
                "query_count": len(rows),
                "analysis_query_success_rate": mean_or_none(
                    [1.0 if bool(row.get("synthetic_exec_ok")) else 0.0 for row in rows]
                ),
                "analysis_overall_score": mean_or_none(
                    [float(row.get("query_score")) for row in rows if row.get("query_score") is not None]
                ),
                "dataset_rank_stability_score": dataset_rank.get("rank_stability_score"),
                "dataset_avg_kendall_tau": dataset_rank.get("avg_kendall_tau"),
                "dataset_avg_spearman_rho": dataset_rank.get("avg_spearman_rho"),
                "dataset_champion_retention_rate": dataset_rank.get("champion_retention_rate"),
                "dataset_avg_top_k_overlap": dataset_rank.get("avg_top_k_overlap"),
                "dataset_avg_pairwise_reversal_ratio": dataset_rank.get("avg_pairwise_reversal_ratio"),
            }
            for family_id in CANONICAL_ANALYTICS_SUBITEMS:
                payload[f"{family_id}_score"] = mean_or_none(
                    [
                        float(row.get("query_score"))
                        for row in rows
                        if row.get("query_score") is not None and str(row.get("family_id") or "") == family_id
                    ]
                )
            for family_id, subitems in CANONICAL_ANALYTICS_SUBITEMS.items():
                for subitem_id in subitems:
                    field = canonical_subitem_score_field(family_id, subitem_id)
                    payload[field] = mean_or_none(
                        [
                            float(row.get("query_score"))
                            for row in rows
                            if row.get("query_score") is not None
                            and str(row.get("family_id") or "") == family_id
                            and str(row.get("canonical_subitem_id") or "") == subitem_id
                        ]
                    )
            output.append(payload)
    return output


def _build_sql_eval_summary_note(
    *,
    manifest: dict[str, Any],
    dataset_summary_rows: list[dict[str, Any]],
    family_rollup_rows: list[dict[str, Any]],
) -> str:
    dataset_lines = [
        f"- `{row['dataset_id']}`: rank_stability_score={row.get('rank_stability_score')}, queries={row.get('query_count')}, assets={row.get('asset_count')}"
        for row in dataset_summary_rows
    ]
    family_lines = [
        f"- `{row['family_id']}`: mean_kendall_tau={row.get('mean_kendall_tau')}, mean_top_k_overlap={row.get('mean_top_k_overlap')}, datasets={row.get('dataset_count')}"
        for row in family_rollup_rows
    ]
    return "\n".join(
        [
            "# SQL Rank Stability Final Bundle",
            "",
            f"- Analysis run dir: `{manifest['analysis_run_dir']}`",
            f"- SQL source: `{manifest['sql_source_label']}` (`{manifest['sql_source_version']}`)",
            f"- SQL source family: `{manifest.get('sql_source_family') or ''}`",
            f"- SQL source root: `{manifest.get('sql_source_root') or ''}`",
            f"- Dataset count: `{manifest['dataset_count']}`",
            f"- Top-k overlap setting: `{manifest['top_k']}`",
            "",
            "## Dataset summary",
            "",
            *(dataset_lines or ["- none"]),
            "",
            "## Family rollup",
            "",
            *(family_lines or ["- none"]),
            "",
        ]
    )


def _write_sql_eval_final_bundle(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    sql_dataset_model_rows: list[dict[str, Any]],
    dataset_summary_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    template_rows: list[dict[str, Any]],
    latex_engine: str | None,
) -> dict[str, Any]:
    sql_source_version = str(manifest.get("sql_source_version") or DEFAULT_SQL_SOURCE_VERSION)
    final_dir = task_version_final_dir(TASK_NAME, sql_source_version)
    final_dir.mkdir(parents=True, exist_ok=True)
    write_versioned_final_readme(
        task_name=TASK_NAME,
        title="sql_eval final outputs",
        summary="Versioned final bundles for SQL-derived rank-stability evaluation.",
        notes=[
            "This bundle is anchored to a specific analysis run and inherits that run's SQL source version.",
            "The query-level CSV is preserved here because it is the direct input to rank-stability diagnostics.",
        ],
    )

    family_rollup_rows = _aggregate_group_scores(family_rows, "family_id")
    template_rollup_rows = _aggregate_group_scores(template_rows, "template_id")
    summary_note = _build_sql_eval_summary_note(
        manifest=manifest,
        dataset_summary_rows=dataset_summary_rows,
        family_rollup_rows=family_rollup_rows,
    )

    summary_note_path = final_dir / "sql_rank_stability_summary.md"
    report_tex_path = final_dir / "sql_rank_stability_report.tex"
    report_png_path = final_dir / "sql_rank_stability_report.png"
    report_manifest_path = final_dir / "sql_eval_final_manifest.json"
    _write_text(summary_note_path, summary_note)

    key_files = [
        run_dir / "summaries" / "sql_rank_stability_summary__all_datasets.csv",
        run_dir / "summaries" / "sql_rank_stability_by_family__all_datasets.csv",
        run_dir / "summaries" / "sql_rank_stability_by_template__all_datasets.csv",
        run_dir / "summaries" / "sql_rank_stability_by_query__all_datasets.csv",
        run_dir / "summaries" / "sql_eval_dataset_model_metrics.csv",
    ]
    copy_files(final_dir, key_files)

    tables = [
        {
            "heading": "Dataset-Model SQL Metrics Grid",
            "columns": [
                ("dataset_id", "Dataset"),
                ("model_id", "Model"),
                ("coverage_status", "Coverage"),
                ("analysis_overall_score", "Overall"),
                ("subgroup_structure_score", "Subgroup"),
                ("conditional_dependency_structure_score", "Conditional"),
                ("tail_rarity_structure_score", "Tail"),
                ("missingness_structure_score", "Missingness"),
            ],
            "rows": sql_dataset_model_rows,
            "note": "This standardized grid is always expanded to current datasets x 11 paper-facing models, so missing assets stay visible instead of silently disappearing.",
        },
        {
            "heading": "Run Summary",
            "columns": [("field", "Field"), ("value", "Value")],
            "rows": [
                {"field": "run_tag", "value": manifest.get("run_tag")},
                {"field": "analysis_run_dir", "value": manifest.get("analysis_run_dir")},
                {"field": "provenance_contract_version", "value": manifest.get("provenance_contract_version")},
                {"field": "real_reference_split", "value": manifest.get("real_reference_split")},
                {"field": "sql_source_family", "value": manifest.get("sql_source_family")},
                {"field": "sql_source_line_version", "value": manifest.get("sql_source_line_version")},
                {"field": "sql_source_version", "value": manifest.get("sql_source_version")},
                {"field": "sql_source_label", "value": manifest.get("sql_source_label")},
                {"field": "dataset_count", "value": manifest.get("dataset_count")},
                {"field": "top_k", "value": manifest.get("top_k")},
            ],
            "widths": ["4.0cm", "10.0cm"],
        },
        {
            "heading": "Dataset Rank Stability Summary",
            "columns": [
                ("dataset_id", "Dataset"),
                ("query_count", "Queries"),
                ("asset_count", "Assets"),
                ("avg_kendall_tau", "Avg Kendall"),
                ("avg_spearman_rho", "Avg Spearman"),
                ("rank_stability_score", "Rank Stability"),
            ],
            "rows": dataset_summary_rows,
        },
        {
            "heading": "Family Rollup",
            "columns": [
                ("family_id", "Family"),
                ("dataset_count", "Datasets"),
                ("mean_kendall_tau", "Mean Kendall"),
                ("mean_spearman_rho", "Mean Spearman"),
                ("mean_top_k_overlap", "Mean Top-k"),
            ],
            "rows": family_rollup_rows,
        },
        {
            "heading": "Family Detail",
            "columns": [
                ("dataset_id", "Dataset"),
                ("family_id", "Family"),
                ("query_count", "Queries"),
                ("avg_kendall_tau", "Avg Kendall"),
                ("avg_spearman_rho", "Avg Spearman"),
                ("champion_retention_rate", "Champion Retention"),
            ],
            "rows": family_rows,
        },
        {
            "heading": "Template Rollup",
            "columns": [
                ("template_id", "Template"),
                ("dataset_count", "Datasets"),
                ("mean_kendall_tau", "Mean Kendall"),
                ("mean_spearman_rho", "Mean Spearman"),
                ("mean_top_k_overlap", "Mean Top-k"),
            ],
            "rows": template_rollup_rows,
        },
    ]
    report_tex = build_longtable_report_tex(
        title="SQL Rank Stability Final Report",
        subtitle="Paper-facing summary of how single-query rankings preserve the overall synthetic-model ordering.",
        intro_lines=[
            f"run_tag={manifest.get('run_tag')}",
            f"analysis_run_dir={manifest.get('analysis_run_dir')}",
            f"sql_source={manifest.get('sql_source_label')} ({manifest.get('sql_source_version')})",
            f"top_k={manifest.get('top_k')}",
        ],
        tables=tables,
    )
    _write_text(report_tex_path, report_tex)
    report_pdf_path, report_log_path = compile_tex(report_tex_path, latex_engine=latex_engine)
    render_pdf_to_png(report_pdf_path, report_png_path, densest_page=True)

    write_final_json(final_dir / "sql_eval_run_manifest.json", manifest)
    final_manifest = {
        "task": TASK_NAME,
        "run_tag": manifest.get("run_tag"),
        "run_dir": str(run_dir.resolve()),
        "final_dir": str(final_dir.resolve()),
        "provenance_contract_version": manifest.get("provenance_contract_version"),
        "real_reference_split": manifest.get("real_reference_split"),
        "sql_source_family": manifest.get("sql_source_family"),
        "sql_source_line_version": manifest.get("sql_source_line_version"),
        "sql_source_version": sql_source_version,
        "sql_source_label": manifest.get("sql_source_label"),
        "dataset_model_metrics_csv": str((final_dir / "sql_eval_dataset_model_metrics.csv").resolve()),
        "summary_note": str(summary_note_path.resolve()),
        "report_tex": str(report_tex_path.resolve()),
        "report_pdf": str(report_pdf_path.resolve()),
        "report_png": str(report_png_path.resolve()),
        "report_compile_log": str(report_log_path.resolve()),
    }
    write_final_json(report_manifest_path, final_manifest)
    return final_manifest


def run_sql_rank_stability(
    *,
    run_tag: str,
    analysis_run_dir: Path,
    top_k: int = 3,
    latex_engine: str | None = None,
    sql_source_version_override: str | None = None,
    publish_final: bool = True,
) -> dict[str, Any]:
    run_dir = make_task_run_dir(TASK_NAME, run_tag)
    query_rows = _load_analysis_rows(analysis_run_dir)
    sql_source_meta = _resolve_analysis_sql_source_metadata(analysis_run_dir, query_rows)
    if sql_source_version_override:
        normalized_override = normalize_sql_source_version(sql_source_version_override)
        actual_version = str(sql_source_meta.get("sql_source_version") or "")
        if normalized_override != actual_version:
            raise ValueError(
                "sql_source_version_override does not match the analysis run provenance: "
                f"override={normalized_override}, analysis={actual_version}, analysis_run_dir={analysis_run_dir.resolve()}"
            )
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        by_dataset[str(row.get("dataset_id") or "")].append(row)
    progress = TaskProgressTracker(
        task_name=TASK_NAME,
        total_steps=len(by_dataset),
        step_label="datasets",
        substep_label="queries",
        total_substeps=len(query_rows),
    )
    progress.print_start(extra=f"run_dir={run_dir.resolve()} | analysis_run_dir={analysis_run_dir.resolve()}")

    dataset_summary_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    query_metric_rows: list[dict[str, Any]] = []

    for dataset_id, rows in sorted(by_dataset.items()):
        overall = _overall_ranking(rows)
        if len(overall) < 2:
            progress.advance(step_name=dataset_id, substeps_done=len(rows), extra="skipped=insufficient_assets")
            continue
        by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_query[str(row.get("query_id") or "")].append(row)
            by_family[str(row.get("family_id") or "")].append(row)
            by_template[str(row.get("template_id") or "")].append(row)

        query_metrics_for_dataset: list[dict[str, Any]] = []
        for query_id, subset in sorted(by_query.items()):
            ranking = _ranking_for_subset(subset)
            metrics = _compare_rankings(overall, ranking, top_k)
            if metrics is None:
                continue
            sample = subset[0]
            dataset_context = _dataset_source_context(sql_source_meta, sample)
            row = {
                **dataset_context,
                "dataset_id": dataset_id,
                "query_id": query_id,
                "question_id": sample.get("question_id"),
                "template_id": sample.get("template_id"),
                "family_id": sample.get("family_id"),
                **metrics,
            }
            query_metric_rows.append(row)
            query_metrics_for_dataset.append(row)

        def _aggregate_group(group_map: dict[str, list[dict[str, Any]]], group_name: str) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for value, subset in sorted(group_map.items()):
                rankings_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in subset:
                    rankings_by_query[str(row.get("query_id") or "")].append(row)
                metrics_rows: list[dict[str, Any]] = []
                for ranking_rows in rankings_by_query.values():
                    ranking = _ranking_for_subset(ranking_rows)
                    metrics = _compare_rankings(overall, ranking, top_k)
                    if metrics is not None:
                        metrics_rows.append(metrics)
                if not metrics_rows:
                    continue
                dataset_context = _dataset_source_context(sql_source_meta, subset[0])
                out.append(
                    {
                        **dataset_context,
                        "dataset_id": dataset_id,
                        group_name: value,
                        "query_count": len(metrics_rows),
                        "avg_kendall_tau": round(mean(float(row["kendall_tau"]) for row in metrics_rows), 6),
                        "avg_spearman_rho": round(mean(float(row["spearman_rho"]) for row in metrics_rows), 6),
                        "champion_retention_rate": round(mean(1.0 if row["champion_same"] else 0.0 for row in metrics_rows), 6),
                        "avg_top_k_overlap": round(mean(float(row["top_k_overlap"]) for row in metrics_rows), 6),
                        "avg_pairwise_reversal_ratio": round(mean(float(row["pairwise_reversal_ratio"]) for row in metrics_rows), 6),
                    }
                )
            return out

        dataset_family_rows = _aggregate_group(by_family, "family_id")
        dataset_template_rows = _aggregate_group(by_template, "template_id")
        family_rows.extend(dataset_family_rows)
        template_rows.extend(dataset_template_rows)

        if query_metrics_for_dataset:
            dataset_context = _dataset_source_context(sql_source_meta, rows[0])
            dataset_summary_rows.append(
                {
                    **dataset_context,
                    "dataset_id": dataset_id,
                    "query_count": len(query_metrics_for_dataset),
                    "asset_count": len(overall),
                    "avg_kendall_tau": round(mean(float(row["kendall_tau"]) for row in query_metrics_for_dataset), 6),
                    "avg_spearman_rho": round(mean(float(row["spearman_rho"]) for row in query_metrics_for_dataset), 6),
                    "champion_retention_rate": round(mean(1.0 if row["champion_same"] else 0.0 for row in query_metrics_for_dataset), 6),
                    "avg_top_k_overlap": round(mean(float(row["top_k_overlap"]) for row in query_metrics_for_dataset), 6),
                    "avg_pairwise_reversal_ratio": round(mean(float(row["pairwise_reversal_ratio"]) for row in query_metrics_for_dataset), 6),
                    "rank_stability_score": round(
                        mean(
                            (
                                float(row["kendall_tau"])
                                + float(row["spearman_rho"])
                                + (1.0 if row["champion_same"] else 0.0)
                                + float(row["top_k_overlap"])
                                + (1.0 - float(row["pairwise_reversal_ratio"]))
                            )
                            / 5.0
                            for row in query_metrics_for_dataset
                        ),
                        6,
                    ),
                }
            )

        write_csv(run_dir / "datasets" / dataset_id / f"sql_rank_stability_by_query__{dataset_id}.csv", [row for row in query_metric_rows if row["dataset_id"] == dataset_id])
        write_csv(run_dir / "datasets" / dataset_id / f"sql_rank_stability_by_family__{dataset_id}.csv", dataset_family_rows)
        write_csv(run_dir / "datasets" / dataset_id / f"sql_rank_stability_by_template__{dataset_id}.csv", dataset_template_rows)
        progress.advance(step_name=dataset_id, substeps_done=len(rows), extra=f"ranked_assets={len(overall)}")

    write_csv(run_dir / "summaries" / "sql_rank_stability_summary__all_datasets.csv", dataset_summary_rows)
    write_csv(run_dir / "summaries" / "sql_rank_stability_by_family__all_datasets.csv", family_rows)
    write_csv(run_dir / "summaries" / "sql_rank_stability_by_template__all_datasets.csv", template_rows)
    write_csv(run_dir / "summaries" / "sql_rank_stability_by_query__all_datasets.csv", query_metric_rows)
    sql_dataset_model_rows = _build_sql_dataset_model_rows(
        query_rows=query_rows,
        dataset_summary_rows=dataset_summary_rows,
        sql_source_meta=sql_source_meta,
    )
    write_csv(run_dir / "summaries" / "sql_eval_dataset_model_metrics.csv", sql_dataset_model_rows)
    manifest = {
        "task": TASK_NAME,
        "run_tag": run_tag,
        "analysis_run_dir": str(analysis_run_dir.resolve()),
        "dataset_count": len(dataset_summary_rows),
        "top_k": top_k,
        **sql_source_meta,
    }
    if publish_final:
        final_manifest = _write_sql_eval_final_bundle(
            run_dir=run_dir,
            manifest=manifest,
            sql_dataset_model_rows=sql_dataset_model_rows,
            dataset_summary_rows=dataset_summary_rows,
            family_rows=family_rows,
            template_rows=template_rows,
            latex_engine=latex_engine,
        )
        manifest["final_outputs"] = final_manifest
    else:
        manifest["final_outputs"] = None
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": run_dir, "dataset_summary_rows": dataset_summary_rows, "manifest": manifest}


def resolve_latest_analysis_run_dir() -> Path | None:
    latest_path = Path(__file__).resolve().parents[3] / "Evaluation" / "analysis" / "LATEST_RUN.json"
    payload = read_json(latest_path, {}) or {}
    run_dir = payload.get("run_dir")
    if not run_dir:
        return None
    candidate = Path(str(run_dir))
    return candidate if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SQL rank-stability evaluation from analysis outputs.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag.")
    parser.add_argument("--analysis-run-dir", type=Path, default=None, help="Optional analysis run dir.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k overlap cutoff.")
    parser.add_argument("--latex-engine", type=str, default=None, help="Optional LaTeX engine.")
    parser.add_argument("--skip-final-publish", action="store_true", help="Skip writing shared final outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_run_dir = args.analysis_run_dir or resolve_latest_analysis_run_dir()
    if analysis_run_dir is None:
        raise FileNotFoundError("Could not resolve the latest analysis run dir.")
    result = run_sql_rank_stability(
        run_tag=args.run_tag or now_run_tag(),
        analysis_run_dir=analysis_run_dir,
        top_k=max(1, int(args.top_k)),
        latex_engine=args.latex_engine,
        publish_final=not args.skip_final_publish,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
