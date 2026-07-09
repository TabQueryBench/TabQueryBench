"""Run generated SQL against real-train and synthetic data and compare outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from src.benchmark.sql_exec import execute_sql
from src.eval.analytics_contract import (
    ANALYTICS_CONTRACT_VERSION,
    annotate_query_row_with_contract,
    build_subitem_and_family_rows,
)
from src.eval.common import (
    DEFAULT_SQL_SOURCE_VERSION,
    SyntheticAsset,
    TaskProgressTracker,
    discover_synthetic_assets,
    list_dataset_ids,
    load_latest_sql_queries,
    load_sql_result_role_annotations,
    make_task_run_dir,
    materialize_csv_to_sqlite,
    mean_or_none,
    real_split_provenance,
    resolve_real_split_path,
    sql_source_label,
    write_csv,
    write_json,
    write_jsonl,
)
from src.eval.final_outputs import (
    build_longtable_report_tex,
    compile_tex,
    copy_files,
    render_pdf_to_png,
    task_version_final_dir,
    version_label,
    write_json as write_final_json,
    write_versioned_final_readme,
)
from src.evaluation.real_panel_experiment import _compare_execution_results

TASK_NAME = "analysis"
DEFAULT_ANALYSIS_FAMILIES = (
    "subgroup_structure",
    "conditional_dependency_structure",
    "tail_rarity_structure",
    "missingness_structure",
    "cardinality_structure",
)


def _normalize_family_filter(families: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not families:
        return ()
    seen: list[str] = []
    for family in families:
        text = str(family or "").strip()
        if not text or text in seen:
            continue
        seen.append(text)
    return tuple(seen)


def _build_real_sqlite(dataset_id: str, cache_root: Path) -> tuple[Path, str]:
    real_csv = resolve_real_split_path(dataset_id, split="train")
    if not real_csv.exists():
        raise FileNotFoundError(f"Train split missing for {dataset_id}: {real_csv}")
    sqlite_path = cache_root / "real_sqlite" / f"{dataset_id}.sqlite"
    table_name = dataset_id
    materialize_csv_to_sqlite(real_csv, sqlite_path, table_name)
    return sqlite_path, table_name


def _build_synthetic_sqlite(asset: SyntheticAsset, cache_root: Path, table_name: str) -> Path:
    sqlite_path = cache_root / "synthetic_sqlite" / asset.dataset_id / f"{asset.asset_key}.sqlite"
    materialize_csv_to_sqlite(Path(asset.synthetic_csv_path), sqlite_path, table_name)
    return sqlite_path


def _sql_source_summary_fields(sql_queries: list[dict[str, Any]]) -> dict[str, Any]:
    if not sql_queries:
        return {}
    sample = sql_queries[0]
    return {
        "provenance_contract_version": sample.get("provenance_contract_version"),
        "sql_source_family": sample.get("sql_source_family"),
        "sql_source_line_version": sample.get("sql_source_line_version"),
        "sql_source_version": sample.get("sql_source_version"),
        "sql_source_label": sample.get("sql_source_label"),
        "sql_source_description": sample.get("sql_source_description"),
        "sql_source_root": sample.get("sql_source_root"),
        "sql_source_registry_root": sample.get("sql_source_registry_root"),
        "sql_source_kind": sample.get("sql_source_kind"),
        "sql_source_selection_mode": sample.get("sql_source_selection_mode"),
    }


def _attach_context(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    if not context:
        return rows
    return [{**context, **row} for row in rows]


def _aggregate_rows(rows: list[dict[str, Any]], group_key: str, score_field: str = "query_score") -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("dataset_id") or ""), str(row.get(group_key) or ""))
        grouped[key].append(row)
    out: list[dict[str, Any]] = []
    for (dataset_id, group_value), items in sorted(grouped.items()):
        success_values = [1.0 if item.get("synthetic_exec_ok") else 0.0 for item in items]
        score_values = [float(item.get(score_field) or 0.0) for item in items if item.get(score_field) is not None]
        asset_keys = sorted({str(item.get("asset_key") or "") for item in items})
        out.append(
            {
                "dataset_id": dataset_id,
                group_key: group_value,
                "query_count": len(items),
                "asset_count": len(asset_keys),
                "mean_query_score": round(mean(score_values), 6) if score_values else None,
                "mean_success_rate": round(mean(success_values), 6) if success_values else None,
            }
        )
    return out


def _aggregate_contract_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    score_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in group_keys)
        grouped[key].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        payload = {field: value for field, value in zip(group_keys, key)}
        scores = [float(item.get(score_field)) for item in items if item.get(score_field) is not None]
        query_counts = [int(item.get("query_count") or 0) for item in items]
        active_counts = [int(item.get("active_subitem_count") or 0) for item in items if item.get("active_subitem_count") is not None]
        applicable_flags = [bool(item.get("subitem_applicable")) for item in items if "subitem_applicable" in item]
        payload.update(
            {
                "row_count": len(items),
                "query_count": sum(query_counts) if query_counts else 0,
                score_field: round(mean(scores), 6) if scores else None,
                "contract_version": ANALYTICS_CONTRACT_VERSION,
            }
        )
        if active_counts:
            payload["active_subitem_count_mean"] = round(mean(active_counts), 6)
        if applicable_flags:
            payload["subitem_applicable"] = any(applicable_flags)
        out.append(payload)
    return out


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_analysis_model_summary_rows(asset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in asset_rows:
        grouped[str(row.get("model_id") or "")].append(row)

    output: list[dict[str, Any]] = []
    for model_id, items in sorted(grouped.items()):
        output.append(
            {
                "model_id": model_id,
                "asset_count": len(items),
                "dataset_count": len({str(item.get("dataset_id") or "") for item in items if item.get("dataset_id")}),
                "server_types": ",".join(sorted({str(item.get("server_type") or "") for item in items if item.get("server_type")})),
                "mean_overall_score": mean_or_none(
                    [float(item.get("overall_score")) for item in items if item.get("overall_score") is not None]
                ),
                "mean_query_success_rate": mean_or_none(
                    [float(item.get("query_success_rate")) for item in items if item.get("query_success_rate") is not None]
                ),
            }
        )
    return output


def _build_analysis_overall_family_rows(family_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in family_summary_rows:
        grouped[str(row.get("family_id") or "")].append(row)

    output: list[dict[str, Any]] = []
    for family_id, items in sorted(grouped.items()):
        output.append(
            {
                "family_id": family_id,
                "dataset_count": len({str(item.get("dataset_id") or "") for item in items if item.get("dataset_id")}),
                "mean_family_score": mean_or_none(
                    [float(item.get("family_score")) for item in items if item.get("family_score") is not None]
                ),
                "mean_active_subitem_count": mean_or_none(
                    [
                        float(item.get("active_subitem_count_mean"))
                        for item in items
                        if item.get("active_subitem_count_mean") is not None
                    ]
                ),
                "applicable_dataset_count": sum(1 for item in items if bool(item.get("subitem_applicable"))),
            }
        )
    return output


def _build_analysis_summary_note(
    *,
    manifest: dict[str, Any],
    dataset_manifest_rows: list[dict[str, Any]],
    model_summary_rows: list[dict[str, Any]],
    overall_family_rows: list[dict[str, Any]],
) -> str:
    dataset_lines = [
        f"- `{row['dataset_id']}`: assets={int(row.get('asset_count') or 0)}, sql_queries={int(row.get('sql_query_count') or 0)}"
        for row in dataset_manifest_rows
    ]
    model_lines = [
        f"- `{row['model_id']}`: mean_overall_score={row.get('mean_overall_score')}, mean_query_success_rate={row.get('mean_query_success_rate')}, assets={row.get('asset_count')}"
        for row in model_summary_rows
    ]
    family_lines = [
        f"- `{row['family_id']}`: mean_family_score={row.get('mean_family_score')}, datasets={row.get('dataset_count')}"
        for row in overall_family_rows
    ]
    return "\n".join(
        [
            "# SQL Analysis Final Bundle",
            "",
            f"- SQL source: `{manifest['sql_source_label']}` (`{manifest['sql_source_version']}`)",
            f"- SQL source root: `{manifest.get('sql_source_root') or ''}`",
            f"- Engine filter: `{','.join(manifest.get('engine_filter') or [])}`",
            f"- Dataset count: `{manifest['dataset_count']}`",
            f"- Asset count: `{manifest['asset_count']}`",
            f"- Query score count: `{manifest['query_score_count']}`",
            f"- Real reference split: `{manifest['real_reference_split']}`",
            "",
            "## Dataset coverage",
            "",
            *(dataset_lines or ["- none"]),
            "",
            "## Model summary",
            "",
            *(model_lines or ["- none"]),
            "",
            "## Family summary",
            "",
            *(family_lines or ["- none"]),
            "",
        ]
    )


def _write_analysis_final_bundle(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    dataset_manifest_rows: list[dict[str, Any]],
    asset_rows: list[dict[str, Any]],
    family_summary_rows: list[dict[str, Any]],
    subitem_summary_rows: list[dict[str, Any]],
    template_summary_rows: list[dict[str, Any]],
    latex_engine: str | None,
) -> dict[str, Any]:
    sql_source_version = str(manifest.get("sql_source_version") or DEFAULT_SQL_SOURCE_VERSION)
    final_dir = task_version_final_dir(TASK_NAME, sql_source_version)
    final_dir.mkdir(parents=True, exist_ok=True)
    write_versioned_final_readme(
        task_name=TASK_NAME,
        title="analysis final outputs",
        summary="Versioned final bundles for SQL execution scoring against real-train references.",
        notes=[
            "Raw query-level JSONL stays under `runs/<run_tag>/summaries/` because it can be large.",
            "The versioned bundle keeps the paper-facing summaries, report sources, compiled PDF, and PNG preview together.",
        ],
    )

    summary_note_path = final_dir / "analysis_summary.md"
    report_tex_path = final_dir / "analysis_summary_report.tex"
    report_png_path = final_dir / "analysis_summary_report.png"
    report_manifest_path = final_dir / "analysis_final_manifest.json"

    model_summary_rows = _build_analysis_model_summary_rows(asset_rows)
    overall_family_rows = _build_analysis_overall_family_rows(family_summary_rows)
    summary_note = _build_analysis_summary_note(
        manifest=manifest,
        dataset_manifest_rows=dataset_manifest_rows,
        model_summary_rows=model_summary_rows,
        overall_family_rows=overall_family_rows,
    )
    _write_text(summary_note_path, summary_note)

    key_files = [
        run_dir / "summaries" / "analysis_asset_scores__all_datasets.csv",
        run_dir / "summaries" / "analysis_template_mean_scores__all_datasets.csv",
        run_dir / "summaries" / "analysis_subitem_scores__all_datasets.csv",
        run_dir / "summaries" / "analysis_family_mean_scores__all_datasets.csv",
        run_dir / "summaries" / "analysis_dataset_manifest.csv",
    ]
    copy_files(final_dir, key_files)

    tables = [
        {
            "heading": "Run Summary",
            "columns": [
                ("field", "Field"),
                ("value", "Value"),
            ],
            "rows": [
                {"field": "run_tag", "value": manifest.get("run_tag")},
                {"field": "sql_source_version", "value": manifest.get("sql_source_version")},
                {"field": "sql_source_label", "value": version_label(sql_source_version)},
                {"field": "engine_filter", "value": ",".join(manifest.get("engine_filter") or [])},
                {"field": "dataset_count", "value": manifest.get("dataset_count")},
                {"field": "asset_count", "value": manifest.get("asset_count")},
                {"field": "query_score_count", "value": manifest.get("query_score_count")},
            ],
            "widths": ["4.0cm", "10.0cm"],
        },
        {
            "heading": "Dataset Coverage",
            "columns": [
                ("dataset_id", "Dataset"),
                ("asset_count", "Assets"),
                ("sql_query_count", "SQL Queries"),
                ("sql_source_label", "SQL Source"),
            ],
            "rows": dataset_manifest_rows,
        },
        {
            "heading": "Model Score Summary",
            "columns": [
                ("model_id", "Model"),
                ("asset_count", "Assets"),
                ("dataset_count", "Datasets"),
                ("mean_overall_score", "Mean Overall"),
                ("mean_query_success_rate", "Mean Query Success"),
            ],
            "rows": model_summary_rows,
        },
        {
            "heading": "Family Score Summary",
            "columns": [
                ("family_id", "Family"),
                ("dataset_count", "Datasets"),
                ("mean_family_score", "Mean Family Score"),
                ("mean_active_subitem_count", "Mean Active Subitems"),
            ],
            "rows": overall_family_rows,
        },
        {
            "heading": "Dataset-Family Detail",
            "columns": [
                ("dataset_id", "Dataset"),
                ("family_id", "Family"),
                ("family_score", "Family Score"),
                ("active_subitem_count_mean", "Mean Active Subitems"),
                ("query_count", "Query Count"),
            ],
            "rows": family_summary_rows,
        },
        {
            "heading": "Dataset-Subitem Detail",
            "columns": [
                ("dataset_id", "Dataset"),
                ("family_id", "Family"),
                ("subitem_id", "Subitem"),
                ("subitem_score", "Subitem Score"),
                ("query_count", "Query Count"),
            ],
            "rows": subitem_summary_rows,
        },
        {
            "heading": "Template Summary",
            "columns": [
                ("dataset_id", "Dataset"),
                ("template_id", "Template"),
                ("mean_query_score", "Mean Query Score"),
                ("mean_success_rate", "Mean Success Rate"),
                ("query_count", "Query Count"),
            ],
            "rows": template_summary_rows,
        },
    ]
    report_tex = build_longtable_report_tex(
        title="SQL Analysis Final Report",
        subtitle="Paper-facing summary of real-vs-synthetic SQL execution scoring.",
        intro_lines=[
            f"run_tag={manifest.get('run_tag')}",
            f"sql_source={manifest.get('sql_source_label')} ({manifest.get('sql_source_version')})",
            f"sql_source_root={manifest.get('sql_source_root') or ''}",
            f"real_reference_split={manifest.get('real_reference_split')}",
        ],
        tables=tables,
    )
    _write_text(report_tex_path, report_tex)
    report_pdf_path, report_log_path = compile_tex(report_tex_path, latex_engine=latex_engine)
    render_pdf_to_png(report_pdf_path, report_png_path, densest_page=True)

    final_manifest = {
        "task": TASK_NAME,
        "run_tag": manifest.get("run_tag"),
        "run_dir": str(run_dir.resolve()),
        "final_dir": str(final_dir.resolve()),
        "sql_source_version": sql_source_version,
        "sql_source_label": manifest.get("sql_source_label"),
        "summary_note": str(summary_note_path.resolve()),
        "report_tex": str(report_tex_path.resolve()),
        "report_pdf": str(report_pdf_path.resolve()),
        "report_png": str(report_png_path.resolve()),
        "report_compile_log": str(report_log_path.resolve()),
    }
    write_final_json(final_dir / "analysis_run_manifest.json", manifest)
    write_final_json(report_manifest_path, final_manifest)
    return final_manifest


def _run_analysis_dataset(
    dataset_id: str,
    dataset_assets: list[SyntheticAsset],
    run_dir_str: str,
    cache_root_str: str | None,
    engines: tuple[str, ...],
    sql_source_version: str,
    include_all_sql_statements: bool,
    max_sql_per_dataset: int,
    query_row_limit: int,
    family_filter: tuple[str, ...],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_dir = Path(run_dir_str)
    cache_root = Path(cache_root_str) if cache_root_str else (run_dir / "cache")
    real_provenance = real_split_provenance(dataset_id, split="train")
    sql_queries = load_latest_sql_queries(
        dataset_id=dataset_id,
        engines=engines,
        include_all_statements=include_all_sql_statements,
        sql_source_version=sql_source_version,
    )
    normalized_family_filter = _normalize_family_filter(family_filter)
    if normalized_family_filter:
        sql_queries = [row for row in sql_queries if str(row.get("family_id") or "") in set(normalized_family_filter)]
    annotation_map = load_sql_result_role_annotations(dataset_id, sql_source_version=sql_source_version)
    sql_source_summary = _sql_source_summary_fields(sql_queries)
    if max_sql_per_dataset > 0:
        sql_queries = sql_queries[:max_sql_per_dataset]
    dataset_manifest = {
        "dataset_id": dataset_id,
        "asset_count": len(dataset_assets),
        "sql_query_count": len(sql_queries),
        "engine_filter": ",".join(engines),
        **real_provenance,
        **sql_source_summary,
        "sql_source_version": sql_source_summary.get("sql_source_version") or sql_source_version,
        "sql_source_label": sql_source_summary.get("sql_source_label") or sql_source_label(sql_source_version),
        "sql_source_root": (sql_source_summary.get("sql_source_root") or ""),
        "query_row_limit": query_row_limit,
        "family_filter": list(normalized_family_filter),
    }
    if not dataset_assets or not sql_queries:
        return dataset_id, [], [], [], [], [], dataset_manifest

    real_sqlite_path, table_name = _build_real_sqlite(dataset_id, cache_root)
    baseline_real: dict[str, Any] = {}
    for query in sql_queries:
        baseline_real[query["query_id"]] = execute_sql(real_sqlite_path, query["sql"], row_limit=query_row_limit)

    dataset_query_rows: list[dict[str, Any]] = []
    dataset_asset_rows: list[dict[str, Any]] = []
    dataset_template_rows: list[dict[str, Any]] = []
    dataset_subitem_rows: list[dict[str, Any]] = []
    dataset_family_rows: list[dict[str, Any]] = []

    for asset in dataset_assets:
        synthetic_sqlite = _build_synthetic_sqlite(asset, cache_root, table_name)
        per_asset_query_scores: list[float] = []
        per_asset_success: list[float] = []
        per_template_scores: dict[str, list[float]] = defaultdict(list)
        asset_query_rows: list[dict[str, Any]] = []

        for query in sql_queries:
            real_exec = baseline_real.get(query["query_id"])
            syn_exec = execute_sql(synthetic_sqlite, query["sql"], row_limit=query_row_limit)
            annotation = annotation_map.get((str(query.get("sql_source_version") or sql_source_version), str(query["query_id"])))
            score, detail = _compare_execution_results(real_exec, syn_exec, result_role_annotation=annotation)
            per_asset_query_scores.append(score)
            per_asset_success.append(1.0 if syn_exec.ok else 0.0)
            per_template_scores[str(query["template_id"])].append(score)
            asset_query_rows.append(
                annotate_query_row_with_contract(
                    {
                        **real_provenance,
                        **asset.to_dict(),
                        "dataset_id": dataset_id,
                        "real_reference_split": "train",
                        "query_id": query["query_id"],
                        "question_id": query["question_id"],
                        "sql_index": query["sql_index"],
                        "question": query["question"],
                        "template_id": query["template_id"],
                        "template_name": query["template_name"],
                        "family_id": query["family_id"],
                        "intended_facet_id": query.get("intended_facet_id"),
                        "variant_semantic_role": query.get("variant_semantic_role"),
                        "stable_question_id": query.get("stable_question_id"),
                        "query_identity_stable_key": query.get("query_identity_stable_key"),
                        "source_sql_run_id": query["source_run_id"],
                        "sql_engine": query["engine"],
                        "sql_model": query["model"],
                        "sql_source_version": query.get("sql_source_version"),
                        "sql_source_label": query.get("sql_source_label"),
                        "sql_source_description": query.get("sql_source_description"),
                        "sql_source_root": query.get("sql_source_root"),
                        "sql_source_kind": query.get("sql_source_kind"),
                        "sql_source_selection_mode": query.get("sql_source_selection_mode"),
                        "sql_origin_path": query.get("sql_origin_path"),
                        "sql_source_manifest_path": query.get("sql_source_manifest_path"),
                        "sql_source_registry_path": query.get("sql_source_registry_path"),
                        "sql": query["sql"],
                        "query_score": round(score, 6),
                        "query_score_method": detail.get("query_score_method"),
                        "query_row_limit": query_row_limit,
                        "synthetic_exec_ok": syn_exec.ok,
                        "real_exec_ok": bool(getattr(real_exec, "ok", False)),
                        "details": detail,
                    }
                )
            )

        dataset_query_rows.extend(asset_query_rows)

        dataset_asset_rows.append(
            {
                **asset.to_dict(),
                **real_provenance,
                **sql_source_summary,
                "dataset_id": dataset_id,
                "real_reference_split": "train",
                "query_count": len(per_asset_query_scores),
                "query_success_rate": round(mean(per_asset_success), 6) if per_asset_success else None,
                "overall_score": round(mean(per_asset_query_scores), 6) if per_asset_query_scores else None,
            }
        )

        for template_id, scores in sorted(per_template_scores.items()):
            dataset_template_rows.append(
                {
                    **asset.to_dict(),
                    **real_provenance,
                    **sql_source_summary,
                    "dataset_id": dataset_id,
                    "template_id": template_id,
                    "query_count": len(scores),
                    "template_score": round(mean(scores), 6),
                }
            )
        asset_subitem_rows, asset_family_rows = build_subitem_and_family_rows(
            query_rows=asset_query_rows,
            context_fields={
                **asset.to_dict(),
                **real_provenance,
                **sql_source_summary,
                "dataset_id": dataset_id,
                "real_reference_split": "train",
            },
            score_field="query_score",
            missingness_applicable=True,
        )
        dataset_subitem_rows.extend(asset_subitem_rows)
        dataset_family_rows.extend(asset_family_rows)

    return dataset_id, dataset_asset_rows, dataset_query_rows, dataset_template_rows, dataset_subitem_rows, dataset_family_rows, dataset_manifest


def run_sql_analysis(
    *,
    run_tag: str,
    datasets: list[str] | None = None,
    latest_only: bool = True,
    engines: tuple[str, ...] = ("cli",),
    sql_source_version: str = DEFAULT_SQL_SOURCE_VERSION,
    include_all_sql_statements: bool = True,
    max_sql_per_dataset: int = 0,
    query_row_limit: int = 0,
    max_workers: int = 1,
    family_filter: tuple[str, ...] | list[str] | None = None,
    cache_root: Path | None = None,
    latex_engine: str | None = None,
    root_names: tuple[str, ...] | list[str] | None = None,
    publish_final: bool = True,
) -> dict[str, Any]:
    dataset_ids = datasets or list_dataset_ids()
    run_dir = make_task_run_dir(TASK_NAME, run_tag)
    normalized_family_filter = _normalize_family_filter(family_filter)
    normalized_root_names = tuple(str(item).strip() for item in (root_names or []) if str(item).strip())
    cache_root = cache_root.expanduser().resolve() if cache_root is not None else (run_dir / "cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    assets = discover_synthetic_assets(
        datasets=dataset_ids,
        latest_only=latest_only,
        root_names=normalized_root_names,
    )

    asset_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    subitem_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    dataset_manifest_rows: list[dict[str, Any]] = []

    dataset_asset_map = {dataset_id: [asset for asset in assets if asset.dataset_id == dataset_id] for dataset_id in dataset_ids}
    progress = TaskProgressTracker(
        task_name=TASK_NAME,
        total_steps=len(dataset_ids),
        step_label="datasets",
        substep_label="assets",
        total_substeps=sum(len(dataset_asset_map.get(dataset_id, [])) for dataset_id in dataset_ids),
    )
    progress.print_start(
        extra=(
            f"run_dir={run_dir.resolve()} | engines={','.join(engines)} "
            f"| sql_source={sql_source_label(sql_source_version)}"
            f" | families={','.join(normalized_family_filter) if normalized_family_filter else 'all'}"
            f" | roots={','.join(normalized_root_names) if normalized_root_names else 'all'}"
            f" | cache_root={cache_root}"
        )
    )

    def _consume_result(
        dataset_id: str,
        dataset_asset_rows: list[dict[str, Any]],
        dataset_query_rows: list[dict[str, Any]],
        dataset_template_rows: list[dict[str, Any]],
        dataset_subitem_rows: list[dict[str, Any]],
        dataset_family_rows: list[dict[str, Any]],
        dataset_manifest: dict[str, Any],
    ) -> None:
        dataset_manifest_rows.append(dataset_manifest)
        progress.advance(
            step_name=dataset_id,
            substeps_done=len(dataset_asset_rows),
            extra=f"queries={int(dataset_manifest.get('sql_query_count') or 0)}",
        )
        if not dataset_asset_rows and not dataset_query_rows:
            return
        asset_rows.extend(dataset_asset_rows)
        query_rows.extend(dataset_query_rows)
        template_rows.extend(dataset_template_rows)
        subitem_rows.extend(dataset_subitem_rows)
        family_rows.extend(dataset_family_rows)
        write_csv(run_dir / "datasets" / dataset_id / f"analysis_asset_scores__{dataset_id}.csv", dataset_asset_rows)
        write_jsonl(run_dir / "datasets" / dataset_id / f"analysis_query_scores__{dataset_id}.jsonl", dataset_query_rows)
        write_csv(run_dir / "datasets" / dataset_id / f"analysis_template_scores__{dataset_id}.csv", dataset_template_rows)
        write_csv(run_dir / "datasets" / dataset_id / f"analysis_subitem_scores__{dataset_id}.csv", dataset_subitem_rows)
        write_csv(run_dir / "datasets" / dataset_id / f"analysis_family_scores__{dataset_id}.csv", dataset_family_rows)

    if max_workers > 1 and len(dataset_ids) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_analysis_dataset,
                    dataset_id,
                    dataset_asset_map.get(dataset_id, []),
                    str(run_dir),
                    str(cache_root),
                    engines,
                    sql_source_version,
                    include_all_sql_statements,
                    max_sql_per_dataset,
                    query_row_limit,
                    normalized_family_filter,
                ): dataset_id
                for dataset_id in dataset_ids
            }
            for future in as_completed(futures):
                _consume_result(*future.result())
    else:
        for dataset_id in dataset_ids:
            _consume_result(
                *_run_analysis_dataset(
                    dataset_id,
                    dataset_asset_map.get(dataset_id, []),
                    str(run_dir),
                    str(cache_root),
                    engines,
                    sql_source_version,
                    include_all_sql_statements,
                    max_sql_per_dataset,
                    query_row_limit,
                    normalized_family_filter,
                )
            )

    template_summary_rows = _aggregate_rows(query_rows, "template_id")
    subitem_summary_rows = _aggregate_contract_rows(
        subitem_rows,
        group_keys=("dataset_id", "family_id", "subitem_id"),
        score_field="subitem_score",
    )
    family_summary_rows = _aggregate_contract_rows(
        family_rows,
        group_keys=("dataset_id", "family_id"),
        score_field="family_score",
    )
    summary_context = {
        "provenance_contract_version": query_rows[0].get("provenance_contract_version") if query_rows else None,
        "real_reference_split": "train",
        "sql_source_family": query_rows[0].get("sql_source_family") if query_rows else None,
        "sql_source_line_version": query_rows[0].get("sql_source_line_version") if query_rows else None,
        "sql_source_version": query_rows[0].get("sql_source_version") if query_rows else sql_source_version,
        "sql_source_label": query_rows[0].get("sql_source_label") if query_rows else sql_source_label(sql_source_version),
        "sql_source_root": query_rows[0].get("sql_source_root") if query_rows else "",
    }
    template_summary_rows = _attach_context(template_summary_rows, summary_context)
    subitem_summary_rows = _attach_context(subitem_summary_rows, summary_context)
    family_summary_rows = _attach_context(family_summary_rows, summary_context)

    write_csv(run_dir / "summaries" / "analysis_asset_scores__all_datasets.csv", asset_rows)
    write_jsonl(run_dir / "summaries" / "analysis_query_scores__all_datasets.jsonl", query_rows)
    write_csv(run_dir / "summaries" / "analysis_template_mean_scores__all_datasets.csv", template_summary_rows)
    write_csv(run_dir / "summaries" / "analysis_subitem_scores__all_datasets.csv", subitem_summary_rows)
    write_csv(run_dir / "summaries" / "analysis_family_mean_scores__all_datasets.csv", family_summary_rows)
    write_csv(run_dir / "summaries" / "analysis_dataset_manifest.csv", dataset_manifest_rows)

    manifest = {
        "task": TASK_NAME,
        "run_tag": run_tag,
        "dataset_count": len(dataset_ids),
        "asset_count": len(asset_rows),
        "query_score_count": len(query_rows),
        "real_reference_split": "train",
        "latest_only": latest_only,
        "engine_filter": list(engines),
        "sql_source_version": (query_rows[0].get("sql_source_version") if query_rows else sql_source_version),
        "sql_source_label": (query_rows[0].get("sql_source_label") if query_rows else sql_source_label(sql_source_version)),
        "sql_source_root": (query_rows[0].get("sql_source_root") if query_rows else ""),
        "sql_source_family": (query_rows[0].get("sql_source_family") if query_rows else None),
        "sql_source_line_version": (query_rows[0].get("sql_source_line_version") if query_rows else None),
        "provenance_contract_version": (query_rows[0].get("provenance_contract_version") if query_rows else None),
        "include_all_sql_statements": include_all_sql_statements,
        "max_sql_per_dataset": max_sql_per_dataset,
        "query_row_limit": query_row_limit,
        "max_workers": max_workers,
        "family_filter": list(normalized_family_filter),
        "synthetic_root_filter": list(normalized_root_names),
        "cache_root": str(cache_root),
        "analytics_contract_version": ANALYTICS_CONTRACT_VERSION,
    }
    if publish_final:
        try:
            final_manifest = _write_analysis_final_bundle(
                run_dir=run_dir,
                manifest=manifest,
                dataset_manifest_rows=dataset_manifest_rows,
                asset_rows=asset_rows,
                family_summary_rows=family_summary_rows,
                subitem_summary_rows=subitem_summary_rows,
                template_summary_rows=template_summary_rows,
                latex_engine=latex_engine,
            )
            manifest["final_outputs"] = final_manifest
        except RuntimeError as exc:
            manifest["final_outputs"] = None
            manifest["final_outputs_error"] = str(exc)
    else:
        manifest["final_outputs"] = None
    write_json(run_dir / "manifest.json", manifest)
    return {
        "run_dir": run_dir,
        "asset_rows": asset_rows,
        "query_rows": query_rows,
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SQL analysis against real-train and synthetic data.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset ids to limit evaluation.")
    parser.add_argument("--latest-only", action="store_true", help="Use only latest synthetic asset per model/dataset.")
    parser.add_argument("--engines", nargs="*", default=["cli"], help="SQL engine filter.")
    parser.add_argument("--sql-source-version", type=str, default=DEFAULT_SQL_SOURCE_VERSION, help="SQL source version.")
    parser.add_argument("--first-sql-only", action="store_true", help="Use only the first SQL statement per query file.")
    parser.add_argument("--max-sql-per-dataset", type=int, default=0, help="Optional cap on SQL statements per dataset.")
    parser.add_argument("--query-row-limit", type=int, default=0, help="Optional row limit passed to SQL execution.")
    parser.add_argument("--max-workers", type=int, default=1, help="Dataset-level process parallelism.")
    parser.add_argument("--families", nargs="*", default=None, help="Optional family ids to include.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Optional alternate cache root for sqlite artifacts.")
    parser.add_argument("--latex-engine", type=str, default=None, help="Optional LaTeX engine for final report.")
    parser.add_argument("--root-names", nargs="*", default=None, help="Optional synthetic root names to include.")
    parser.add_argument("--skip-final-publish", action="store_true", help="Do not publish into Evaluation/analysis/final.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_sql_analysis(
        run_tag=args.run_tag or now_run_tag(),
        datasets=args.datasets,
        latest_only=args.latest_only,
        engines=tuple(args.engines),
        sql_source_version=args.sql_source_version,
        include_all_sql_statements=not args.first_sql_only,
        max_sql_per_dataset=args.max_sql_per_dataset,
        query_row_limit=args.query_row_limit,
        max_workers=max(1, int(args.max_workers)),
        family_filter=args.families,
        cache_root=args.cache_root,
        latex_engine=args.latex_engine,
        root_names=args.root_names,
        publish_final=not args.skip_final_publish,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
