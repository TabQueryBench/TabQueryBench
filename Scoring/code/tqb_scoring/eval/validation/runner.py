"""Deterministic synthetic-data validation over train references."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqb_scoring.eval.common import (
    SyntheticAsset,
    TaskProgressTracker,
    discover_synthetic_assets,
    list_dataset_ids,
    make_task_run_dir,
    real_split_provenance,
    resolve_real_split_path,
    write_csv,
    write_json,
    write_jsonl,
)
from tqb_scoring.evaluation.synthetic_validation_v4 import build_validation_context_v4, evaluate_synthetic_validation_v4


def _evaluate_asset(dataset_id: str, asset: SyntheticAsset) -> tuple[dict, dict]:
    real_csv = resolve_real_split_path(dataset_id, split="train")
    if not real_csv.exists():
        raise FileNotFoundError(f"Train split missing for {dataset_id}: {real_csv}")
    import csv

    with real_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        expected_columns = [str(item) for item in next(reader, [])]

    context = build_validation_context_v4(
        dataset_id=dataset_id,
        project_root=Path(__file__).resolve().parents[3],
        real_csv_path=real_csv,
        expected_columns=expected_columns,
    )
    report = evaluate_synthetic_validation_v4(context=context, synthetic_csv_path=Path(asset.synthetic_csv_path))
    scores = report.get("validation_scores") if isinstance(report, dict) else {}
    real_provenance = real_split_provenance(dataset_id, split="train")
    row = {
        **asset.to_dict(),
        **real_provenance,
        "row_count_real": report.get("row_count_real"),
        "row_count_synthetic": report.get("row_count_synthetic"),
        "cardinality_range_score": scores.get("cardinality_range_score"),
        "missing_introduction_score": scores.get("missing_introduction_score"),
    }
    return row, {"asset": asset.to_dict(), "real_provenance": real_provenance, "report": report}


def _run_validation_dataset(dataset_id: str, dataset_assets: list[SyntheticAsset]) -> tuple[str, list[dict], list[dict]]:
    dataset_rows: list[dict] = []
    dataset_details: list[dict] = []
    for asset in dataset_assets:
        row, detail = _evaluate_asset(dataset_id, asset)
        dataset_rows.append(row)
        dataset_details.append(detail)
    return dataset_id, dataset_rows, dataset_details


def run_validation_evaluation(
    *,
    run_tag: str,
    datasets: list[str] | None = None,
    latest_only: bool = True,
    max_workers: int = 1,
    root_names: list[str] | None = None,
) -> dict:
    dataset_ids = datasets or list_dataset_ids()
    run_dir = make_task_run_dir("validation", run_tag)
    assets = discover_synthetic_assets(datasets=dataset_ids, latest_only=latest_only, root_names=root_names)
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    dataset_asset_map = {dataset_id: [asset for asset in assets if asset.dataset_id == dataset_id] for dataset_id in dataset_ids}
    dataset_asset_map = {k: v for k, v in dataset_asset_map.items() if v}
    progress = TaskProgressTracker(
        task_name="validation",
        total_steps=len(dataset_asset_map),
        step_label="datasets",
        substep_label="assets",
        total_substeps=sum(len(items) for items in dataset_asset_map.values()),
    )
    progress.print_start(extra=f"run_dir={run_dir.resolve()}")

    def _consume_result(dataset_id: str, dataset_rows: list[dict], dataset_details: list[dict]) -> None:
        summary_rows.extend(dataset_rows)
        detail_rows.extend(dataset_details)
        write_csv(run_dir / "datasets" / dataset_id / f"validation_summary__{dataset_id}.csv", dataset_rows)
        write_jsonl(run_dir / "datasets" / dataset_id / f"validation_details__{dataset_id}.jsonl", dataset_details)
        progress.advance(step_name=dataset_id, substeps_done=len(dataset_rows))

    if max_workers > 1 and len(dataset_asset_map) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_validation_dataset, dataset_id, dataset_assets): dataset_id
                for dataset_id, dataset_assets in dataset_asset_map.items()
            }
            for future in as_completed(futures):
                dataset_id, dataset_rows, dataset_details = future.result()
                _consume_result(dataset_id, dataset_rows, dataset_details)
    else:
        for dataset_id, dataset_assets in dataset_asset_map.items():
            dataset_id, dataset_rows, dataset_details = _run_validation_dataset(dataset_id, dataset_assets)
            _consume_result(dataset_id, dataset_rows, dataset_details)

    write_csv(run_dir / "summaries" / "validation_summary__all_datasets.csv", summary_rows)
    write_jsonl(run_dir / "summaries" / "validation_details__all_datasets.jsonl", detail_rows)
    manifest = {
        "task": "validation",
        "run_tag": run_tag,
        "dataset_count": len(dataset_ids),
        "asset_count": len(summary_rows),
        "provenance_contract_version": summary_rows[0].get("provenance_contract_version") if summary_rows else "",
        "real_reference_split": "train",
        "real_source_kind": "reference_split_csv",
        "latest_only": latest_only,
        "max_workers": max_workers,
        "synthetic_root_filter": list(root_names or []),
    }
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": run_dir, "summary_rows": summary_rows, "manifest": manifest}
