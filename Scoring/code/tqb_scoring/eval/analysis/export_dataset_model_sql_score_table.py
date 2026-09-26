from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROBE_SUBITEM_RE = re.compile(r"\bprobe\s+([a-z_]+)\b", re.IGNORECASE)

FAMILY_TO_SUBITEMS: dict[str, list[str]] = {
    "subgroup_structure": [
        "internal_profile_stability",
        "subgroup_size_stability",
    ],
    "conditional_dependency_structure": [
        "dependency_strength_similarity",
        "direction_consistency",
        "slice_level_consistency",
    ],
    "tail_rarity_structure": [
        "tail_set_consistency",
        "tail_mass_similarity",
        "tail_concentration_consistency",
    ],
    "missingness_structure": [
        "marginal_missing_rate_consistency",
        "co_missingness_pattern_consistency",
    ],
    "cardinality_structure": [
        "support_rank_profile_consistency",
        "high_cardinality_response_stability",
    ],
}


def _read_asset_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_query_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _normalize_subitem_id(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "")
    match = PROBE_SUBITEM_RE.search(question)
    if match:
        return match.group(1).strip()
    canonical = str(row.get("canonical_subitem_id") or row.get("subitem_id") or "").strip()
    if canonical:
        return canonical
    template_id = str(row.get("template_id") or "").strip()
    if template_id == "tpl_cardinality_high_card_response_stability":
        return "high_cardinality_response_stability"
    if template_id in {
        "tpl_cardinality_support_rank_profile",
        "tpl_cardinality_distinct_share_profile",
        "tpl_cardinality_continuous_range_envelope",
    }:
        return "support_rank_profile_consistency"
    return ""


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mean_or_none(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(mean(clean))


def build_dataset_model_score_table(analysis_run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    summaries_dir = analysis_run_dir / "summaries"
    asset_rows = _read_asset_rows(summaries_dir / "analysis_asset_scores__all_datasets.csv")
    query_rows = _read_query_rows(summaries_dir / "analysis_query_scores__all_datasets.jsonl")

    query_scores_by_asset_subitem: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in query_rows:
        asset_key = str(row.get("asset_key") or "").strip()
        family_id = str(row.get("family_id") or "").strip()
        subitem_id = _normalize_subitem_id(row)
        score = _float_or_none(row.get("query_score"))
        if not asset_key or not family_id or not subitem_id or score is None:
            continue
        query_scores_by_asset_subitem[(asset_key, family_id, subitem_id)].append(score)

    subitem_score_by_asset: dict[tuple[str, str, str], float | None] = {}
    for key, scores in query_scores_by_asset_subitem.items():
        subitem_score_by_asset[key] = _mean_or_none(scores)

    output_rows: list[dict[str, Any]] = []
    for asset_row in asset_rows:
        asset_key = str(asset_row.get("asset_key") or "").strip()
        base_row: dict[str, Any] = {
            "dataset_id": asset_row.get("dataset_id") or "",
            "model_id": asset_row.get("model_id") or "",
            "asset_key": asset_key,
            "root_name": asset_row.get("root_name") or "",
            "run_id": asset_row.get("run_id") or "",
            "synthetic_csv_path": asset_row.get("synthetic_csv_path") or "",
            "overall_score": asset_row.get("overall_score") or "",
            "query_count": asset_row.get("query_count") or "",
            "query_success_rate": asset_row.get("query_success_rate") or "",
            "sql_source_version": asset_row.get("sql_source_version") or "",
            "sql_source_label": asset_row.get("sql_source_label") or "",
        }

        for family_id, subitems in FAMILY_TO_SUBITEMS.items():
            family_values: list[float] = []
            for subitem_id in subitems:
                value = subitem_score_by_asset.get((asset_key, family_id, subitem_id))
                if value is None:
                    base_row[f"{subitem_id}_score"] = ""
                else:
                    base_row[f"{subitem_id}_score"] = f"{value:.6f}"
                    family_values.append(value)
            family_score = _mean_or_none(family_values)
            if family_score is None:
                base_row[f"{family_id}_score"] = ""
            else:
                base_row[f"{family_id}_score"] = f"{family_score:.6f}"

        output_rows.append(base_row)

    fixed_columns = [
        "dataset_id",
        "model_id",
        "asset_key",
        "root_name",
        "run_id",
        "synthetic_csv_path",
        "overall_score",
        "query_count",
        "query_success_rate",
        "sql_source_version",
        "sql_source_label",
    ]
    family_columns = [f"{family_id}_score" for family_id in FAMILY_TO_SUBITEMS]
    subitem_columns = [f"{subitem_id}_score" for subitems in FAMILY_TO_SUBITEMS.values() for subitem_id in subitems]
    fieldnames = fixed_columns + family_columns + subitem_columns
    return output_rows, fieldnames


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dataset+model SQL score table from analysis outputs.")
    parser.add_argument("--analysis-run-dir", type=Path, required=True, help="Analysis run directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, fieldnames = build_dataset_model_score_table(args.analysis_run_dir.resolve())
    write_csv(args.output.resolve(), rows, fieldnames)
    print(
        json.dumps(
            {
                "analysis_run_dir": str(args.analysis_run_dir.resolve()),
                "output": str(args.output.resolve()),
                "row_count": len(rows),
                "column_count": len(fieldnames),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
