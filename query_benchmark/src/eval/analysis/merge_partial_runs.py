from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from src.eval.analytics_contract import ANALYTICS_CONTRACT_VERSION
from src.eval.analysis.runner import (
    TASK_NAME,
    _aggregate_contract_rows,
    _aggregate_rows,
    _attach_context,
    _normalize_family_filter,
    _write_analysis_final_bundle,
)
from src.eval.common import (
    DEFAULT_SQL_SOURCE_VERSION,
    make_task_run_dir,
    now_run_tag,
    read_json,
    sql_source_label,
    write_csv,
    write_json,
    write_jsonl,
)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            cleaned = {
                key: (None if value == "" else value)
                for key, value in dict(row).items()
            }
            rows.append(cleaned)
        return rows


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        item = json.loads(text)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _copy_dataset_dir(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def _dataset_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / "datasets"
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name)


def merge_analysis_runs(
    *,
    run_tag: str,
    source_run_dirs: list[Path],
    latex_engine: str | None = None,
    override_duplicate_datasets: bool = False,
) -> dict[str, Any]:
    if not source_run_dirs:
        raise ValueError("source_run_dirs must not be empty")

    source_manifests = [read_json(run_dir / "manifest.json", {}) or {} for run_dir in source_run_dirs]
    sql_source_version = str(
        next(
            (
                manifest.get("sql_source_version")
                for manifest in source_manifests
                if manifest.get("sql_source_version")
            ),
            DEFAULT_SQL_SOURCE_VERSION,
        )
    )
    family_filter = _normalize_family_filter(
        next((manifest.get("family_filter") for manifest in source_manifests if manifest.get("family_filter")), [])
    )
    latest_only = any(bool(manifest.get("latest_only")) for manifest in source_manifests)
    include_all_sql_statements = all(
        bool(manifest.get("include_all_sql_statements", True)) for manifest in source_manifests
    )
    max_sql_per_dataset = max(int(manifest.get("max_sql_per_dataset") or 0) for manifest in source_manifests)
    query_row_limit = max(int(manifest.get("query_row_limit") or 0) for manifest in source_manifests)
    engine_filter = next((manifest.get("engine_filter") for manifest in source_manifests if manifest.get("engine_filter")), ["cli"])
    cache_root = next((manifest.get("cache_root") for manifest in source_manifests if manifest.get("cache_root")), "")

    run_dir = make_task_run_dir(TASK_NAME, run_tag)

    asset_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    template_rows_raw: list[dict[str, Any]] = []
    subitem_rows_raw: list[dict[str, Any]] = []
    family_rows_raw: list[dict[str, Any]] = []
    dataset_manifest_rows: list[dict[str, Any]] = []

    selected_dataset_dirs: dict[str, Path] = {}

    for source_run_dir in source_run_dirs:
        for dataset_dir in _dataset_dirs(source_run_dir):
            dataset_id = dataset_dir.name
            if dataset_id in selected_dataset_dirs and not override_duplicate_datasets:
                raise ValueError(f"Duplicate dataset_id across source runs: {dataset_id}")
            selected_dataset_dirs[dataset_id] = dataset_dir

    for dataset_id, dataset_dir in sorted(selected_dataset_dirs.items()):
        _copy_dataset_dir(dataset_dir, run_dir / "datasets" / dataset_id)

        asset_rows.extend(_read_csv_rows(dataset_dir / f"analysis_asset_scores__{dataset_id}.csv"))
        query_rows.extend(_read_jsonl_rows(dataset_dir / f"analysis_query_scores__{dataset_id}.jsonl"))
        template_rows_raw.extend(_read_csv_rows(dataset_dir / f"analysis_template_scores__{dataset_id}.csv"))
        subitem_rows_raw.extend(_read_csv_rows(dataset_dir / f"analysis_subitem_scores__{dataset_id}.csv"))
        family_rows_raw.extend(_read_csv_rows(dataset_dir / f"analysis_family_scores__{dataset_id}.csv"))

    if not query_rows:
        raise ValueError("No query rows found in source runs")

    for dataset_id in sorted(selected_dataset_dirs):
        asset_count = sum(1 for row in asset_rows if str(row.get("dataset_id") or "") == dataset_id)
        dataset_queries = [row for row in query_rows if str(row.get("dataset_id") or "") == dataset_id]
        sample = dataset_queries[0] if dataset_queries else {}
        dataset_manifest_rows.append(
            {
                "dataset_id": dataset_id,
                "asset_count": asset_count,
                "sql_query_count": len({str(row.get("query_id") or "") for row in dataset_queries}),
                "engine_filter": ",".join(engine_filter),
                "real_reference_split": sample.get("real_reference_split") or "train",
                "real_source_kind": sample.get("real_source_kind"),
                "real_source_dataset_id": sample.get("real_source_dataset_id"),
                "real_source_split": sample.get("real_source_split"),
                "real_source_path": sample.get("real_source_path"),
                "real_source_exists": sample.get("real_source_exists"),
                "real_source_mtime_utc": sample.get("real_source_mtime_utc"),
                "real_source_size_bytes": sample.get("real_source_size_bytes"),
                "provenance_contract_version": sample.get("provenance_contract_version"),
                "sql_source_family": sample.get("sql_source_family"),
                "sql_source_line_version": sample.get("sql_source_line_version"),
                "sql_source_version": sample.get("sql_source_version") or sql_source_version,
                "sql_source_label": sample.get("sql_source_label") or sql_source_label(sql_source_version),
                "sql_source_root": sample.get("sql_source_root"),
                "query_row_limit": query_row_limit,
                "family_filter": list(family_filter),
            }
        )

    template_summary_rows = _aggregate_rows(query_rows, "template_id")
    subitem_summary_rows = _aggregate_contract_rows(
        subitem_rows_raw,
        group_keys=("dataset_id", "family_id", "subitem_id"),
        score_field="subitem_score",
    )
    family_summary_rows = _aggregate_contract_rows(
        family_rows_raw,
        group_keys=("dataset_id", "family_id"),
        score_field="family_score",
    )
    summary_context = {
        "provenance_contract_version": query_rows[0].get("provenance_contract_version"),
        "real_reference_split": "train",
        "sql_source_family": query_rows[0].get("sql_source_family"),
        "sql_source_line_version": query_rows[0].get("sql_source_line_version"),
        "sql_source_version": query_rows[0].get("sql_source_version") or sql_source_version,
        "sql_source_label": query_rows[0].get("sql_source_label") or sql_source_label(sql_source_version),
        "sql_source_root": query_rows[0].get("sql_source_root") or "",
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
        "dataset_count": len(selected_dataset_dirs),
        "asset_count": len(asset_rows),
        "query_score_count": len(query_rows),
        "real_reference_split": "train",
        "latest_only": latest_only,
        "engine_filter": list(engine_filter),
        "sql_source_version": query_rows[0].get("sql_source_version") or sql_source_version,
        "sql_source_label": query_rows[0].get("sql_source_label") or sql_source_label(sql_source_version),
        "sql_source_root": query_rows[0].get("sql_source_root") or "",
        "sql_source_family": query_rows[0].get("sql_source_family"),
        "sql_source_line_version": query_rows[0].get("sql_source_line_version"),
        "provenance_contract_version": query_rows[0].get("provenance_contract_version"),
        "include_all_sql_statements": include_all_sql_statements,
        "max_sql_per_dataset": max_sql_per_dataset,
        "query_row_limit": query_row_limit,
        "max_workers": 0,
        "family_filter": list(family_filter),
        "cache_root": str(cache_root),
        "analytics_contract_version": ANALYTICS_CONTRACT_VERSION,
        "source_run_dirs": [str(path.resolve()) for path in source_run_dirs],
        "merged_from_partial_runs": True,
        "override_duplicate_datasets": override_duplicate_datasets,
    }
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
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge partial analysis runs into one finalized run.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional merged run tag.")
    parser.add_argument("--source-run-dirs", nargs="+", type=Path, required=True, help="Source analysis run directories.")
    parser.add_argument("--latex-engine", type=str, default=None, help="Optional LaTeX engine for final report.")
    parser.add_argument(
        "--override-duplicate-datasets",
        action="store_true",
        help="Allow later source runs to override earlier datasets with the same dataset_id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = merge_analysis_runs(
        run_tag=args.run_tag or now_run_tag(),
        source_run_dirs=[path.expanduser().resolve() for path in args.source_run_dirs],
        latex_engine=args.latex_engine,
        override_duplicate_datasets=args.override_duplicate_datasets,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
