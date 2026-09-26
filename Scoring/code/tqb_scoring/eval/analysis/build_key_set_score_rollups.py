from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from tqb_query.analytics_contract import CANONICAL_ANALYTICS_SUBITEMS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_RUN = (
    PROJECT_ROOT.parent / "results"
    / "analysis"
    / "runs"
    / "eval_refresh_20260503_1730_zurich"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "results" / "analysis" / "final" / "v2"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(mean(values)), 6)


def _family_columns() -> list[str]:
    return [f"family__{family_id}" for family_id in CANONICAL_ANALYTICS_SUBITEMS]


def _subfamily_columns() -> list[str]:
    columns: list[str] = []
    for family_id, subitems in CANONICAL_ANALYTICS_SUBITEMS.items():
        for subitem_id in subitems:
            columns.append(f"subfamily__{family_id}__{subitem_id}")
    return columns


def _build_dataset_model_level(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str], dict[str, float]],
]:
    family_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    subfamily_scores: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)

    for row in rows:
        dataset_id = str(row.get("dataset_id") or "").strip().lower()
        model_id = str(row.get("model_id") or "").strip().lower()
        family_id = str(row.get("family_id") or "").strip().lower()
        subitem_id = str(row.get("canonical_subitem_id") or "").strip()
        details = row.get("details") or {}
        try:
            key_score = float(details.get("key_set_score"))
        except Exception:
            continue
        if not dataset_id or not model_id:
            continue
        if family_id not in CANONICAL_ANALYTICS_SUBITEMS:
            continue
        family_scores[(dataset_id, model_id, family_id)].append(key_score)
        if subitem_id and subitem_id in CANONICAL_ANALYTICS_SUBITEMS[family_id]:
            subfamily_scores[(dataset_id, model_id, family_id, subitem_id)].append(key_score)

    dataset_model_family: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (dataset_id, model_id, family_id), values in family_scores.items():
        dataset_model_family[(dataset_id, model_id)][f"family__{family_id}"] = round(float(mean(values)), 6)

    dataset_model_subfamily: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (dataset_id, model_id, family_id, subitem_id), values in subfamily_scores.items():
        dataset_model_subfamily[(dataset_id, model_id)][
            f"subfamily__{family_id}__{subitem_id}"
        ] = round(float(mean(values)), 6)

    return dataset_model_family, dataset_model_subfamily


def _aggregate_dataset_rows(
    dataset_model_family: dict[tuple[str, str], dict[str, float]],
    dataset_model_subfamily: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    by_dataset_family: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_dataset_subfamily: dict[tuple[str, str], list[float]] = defaultdict(list)

    for (dataset_id, _model_id), metrics in dataset_model_family.items():
        for column, value in metrics.items():
            by_dataset_family[(dataset_id, column)].append(float(value))

    for (dataset_id, _model_id), metrics in dataset_model_subfamily.items():
        for column, value in metrics.items():
            by_dataset_subfamily[(dataset_id, column)].append(float(value))

    dataset_ids = sorted({dataset_id for dataset_id, _ in by_dataset_family} | {dataset_id for dataset_id, _ in by_dataset_subfamily})
    columns = _family_columns() + _subfamily_columns()
    output: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        row: dict[str, Any] = {"dataset_id": dataset_id}
        for column in columns:
            family_values = by_dataset_family.get((dataset_id, column), [])
            subfamily_values = by_dataset_subfamily.get((dataset_id, column), [])
            values = family_values or subfamily_values
            row[column] = _mean_or_none(values)
        output.append(row)
    return output


def _aggregate_model_rows(
    dataset_model_family: dict[tuple[str, str], dict[str, float]],
    dataset_model_subfamily: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    by_model_family: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_model_subfamily: dict[tuple[str, str], list[float]] = defaultdict(list)

    for (_dataset_id, model_id), metrics in dataset_model_family.items():
        for column, value in metrics.items():
            by_model_family[(model_id, column)].append(float(value))

    for (_dataset_id, model_id), metrics in dataset_model_subfamily.items():
        for column, value in metrics.items():
            by_model_subfamily[(model_id, column)].append(float(value))

    model_ids = sorted({model_id for model_id, _ in by_model_family} | {model_id for model_id, _ in by_model_subfamily})
    columns = _family_columns() + _subfamily_columns()
    output: list[dict[str, Any]] = []
    for model_id in model_ids:
        row: dict[str, Any] = {"model_id": model_id}
        for column in columns:
            family_values = by_model_family.get((model_id, column), [])
            subfamily_values = by_model_subfamily.get((model_id, column), [])
            values = family_values or subfamily_values
            row[column] = _mean_or_none(values)
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def build_key_set_score_rollups(analysis_run_dir: Path, output_dir: Path) -> dict[str, Any]:
    query_scores_path = analysis_run_dir / "summaries" / "analysis_query_scores__all_datasets.jsonl"
    rows = _read_jsonl(query_scores_path)
    dataset_model_family, dataset_model_subfamily = _build_dataset_model_level(rows)
    dataset_rows = _aggregate_dataset_rows(dataset_model_family, dataset_model_subfamily)
    model_rows = _aggregate_model_rows(dataset_model_family, dataset_model_subfamily)

    family_columns = _family_columns()
    subfamily_columns = _subfamily_columns()
    dataset_csv = output_dir / "analysis_key_set_score_dataset_mean_by_family_subfamily.csv"
    model_csv = output_dir / "analysis_key_set_score_model_mean_by_family_subfamily.csv"

    _write_csv(dataset_csv, dataset_rows, ["dataset_id", *family_columns, *subfamily_columns])
    _write_csv(model_csv, model_rows, ["model_id", *family_columns, *subfamily_columns])

    return {
        "analysis_run_dir": str(analysis_run_dir.resolve()),
        "query_scores_path": str(query_scores_path.resolve()),
        "dataset_csv": str(dataset_csv.resolve()),
        "model_csv": str(model_csv.resolve()),
        "dataset_row_count": len(dataset_rows),
        "model_row_count": len(model_rows),
        "family_column_count": len(family_columns),
        "subfamily_column_count": len(subfamily_columns),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build key_set_score rollup CSVs from analysis query-level outputs.")
    parser.add_argument("--analysis-run-dir", type=Path, default=DEFAULT_ANALYSIS_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_key_set_score_rollups(
        analysis_run_dir=args.analysis_run_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
