from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


FAMILY_ORDER = [
    "subgroup_structure",
    "conditional_dependency_structure",
    "tail_rarity_structure",
    "missingness_structure",
    "cardinality_structure",
]

FAMILY_LABELS = {
    "subgroup_structure": "subgroup",
    "conditional_dependency_structure": "conditional",
    "tail_rarity_structure": "tail",
    "missingness_structure": "missingness",
    "cardinality_structure": "cardinality",
}


def natural_dataset_key(dataset_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", dataset_id)
    if not match:
        return (dataset_id.lower(), -1)
    return (match.group(1).lower(), int(match.group(2)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset-level v2 query count and token summaries by family."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Directory containing query_registry_v2 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for generated summary artifacts.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_main_batch_files(registry_dir: Path) -> list[Path]:
    pattern = re.compile(r"v2_cli_(\d{8}_\d{6})_([a-z])_query_registry_v2\.csv$")
    by_stamp: dict[str, list[Path]] = defaultdict(list)
    for path in registry_dir.glob("v2_cli_*_query_registry_v2.csv"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        by_stamp[match.group(1)].append(path)
    if not by_stamp:
        raise FileNotFoundError(f"No formal v2 registry batches found in {registry_dir}")
    latest_stamp = sorted(by_stamp)[-1]
    return sorted(by_stamp[latest_stamp], key=lambda p: p.name)


def latest_failed_rerun_file(registry_dir: Path) -> Path | None:
    files = sorted(registry_dir.glob("subitem_workload_v2_failed_rerun_*_query_registry_v2.csv"))
    if not files:
        return None
    return files[-1]


def parse_int(value: str | None) -> int:
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


@dataclass
class FamilyAggregate:
    query_count: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def build_aggregates(registry_files: list[Path]) -> tuple[dict[str, dict[str, FamilyAggregate]], dict[str, dict[str, str]]]:
    accepted_by_query_id: dict[str, dict[str, str]] = {}
    for path in registry_files:
        for row in load_csv_rows(path):
            if row.get("accepted_for_eval", "").lower() != "true":
                continue
            query_id = row.get("query_record_id", "")
            if not query_id:
                continue
            accepted_by_query_id[query_id] = row

    dataset_family: dict[str, dict[str, FamilyAggregate]] = defaultdict(
        lambda: defaultdict(FamilyAggregate)
    )
    for row in accepted_by_query_id.values():
        dataset_id = row["dataset_id"]
        family_id = row["family_id"]
        agg = dataset_family[dataset_id][family_id]
        agg.query_count += 1
        agg.input_tokens += parse_int(row.get("usage_input_tokens"))
        agg.cached_input_tokens += parse_int(row.get("usage_cached_input_tokens"))
        agg.output_tokens += parse_int(row.get("usage_output_tokens"))
        agg.total_tokens += parse_int(row.get("usage_total_tokens"))

    return dataset_family, accepted_by_query_id


def write_long_csv(output_path: Path, dataset_family: dict[str, dict[str, FamilyAggregate]]) -> None:
    fieldnames = [
        "dataset_id",
        "family_id",
        "family_label",
        "query_count",
        "query_input_tokens",
        "query_cached_input_tokens",
        "query_output_tokens",
        "query_total_tokens",
    ]
    rows: list[dict[str, int | str]] = []
    for dataset_id in sorted(dataset_family, key=natural_dataset_key):
        family_map = dataset_family[dataset_id]
        for family_id in FAMILY_ORDER:
            agg = family_map.get(family_id, FamilyAggregate())
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "family_id": family_id,
                    "family_label": FAMILY_LABELS[family_id],
                    "query_count": agg.query_count,
                    "query_input_tokens": agg.input_tokens,
                    "query_cached_input_tokens": agg.cached_input_tokens,
                    "query_output_tokens": agg.output_tokens,
                    "query_total_tokens": agg.total_tokens,
                }
            )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_wide_csv(output_path: Path, dataset_family: dict[str, dict[str, FamilyAggregate]]) -> None:
    base_fields = [
        "dataset_id",
        "total_queries",
        "total_query_input_tokens",
        "total_query_cached_input_tokens",
        "total_query_output_tokens",
        "total_query_tokens",
    ]
    family_fields: list[str] = []
    for family_id in FAMILY_ORDER:
        label = FAMILY_LABELS[family_id]
        family_fields.extend(
            [
                f"{label}_query_count",
                f"{label}_query_input_tokens",
                f"{label}_query_cached_input_tokens",
                f"{label}_query_output_tokens",
                f"{label}_query_total_tokens",
            ]
        )
    fieldnames = base_fields + family_fields

    rows: list[dict[str, int | str]] = []
    for dataset_id in sorted(dataset_family, key=natural_dataset_key):
        family_map = dataset_family[dataset_id]
        row: dict[str, int | str] = {"dataset_id": dataset_id}
        total_queries = 0
        total_input = 0
        total_cached = 0
        total_output = 0
        total_tokens = 0
        for family_id in FAMILY_ORDER:
            label = FAMILY_LABELS[family_id]
            agg = family_map.get(family_id, FamilyAggregate())
            row[f"{label}_query_count"] = agg.query_count
            row[f"{label}_query_input_tokens"] = agg.input_tokens
            row[f"{label}_query_cached_input_tokens"] = agg.cached_input_tokens
            row[f"{label}_query_output_tokens"] = agg.output_tokens
            row[f"{label}_query_total_tokens"] = agg.total_tokens
            total_queries += agg.query_count
            total_input += agg.input_tokens
            total_cached += agg.cached_input_tokens
            total_output += agg.output_tokens
            total_tokens += agg.total_tokens
        row["total_queries"] = total_queries
        row["total_query_input_tokens"] = total_input
        row["total_query_cached_input_tokens"] = total_cached
        row["total_query_output_tokens"] = total_output
        row["total_query_tokens"] = total_tokens
        rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(
    output_path: Path,
    registry_files: list[Path],
    dataset_family: dict[str, dict[str, FamilyAggregate]],
    accepted_by_query_id: dict[str, dict[str, str]],
) -> None:
    total_queries = len(accepted_by_query_id)
    dataset_count = len(dataset_family)
    family_lines = []
    for family_id in FAMILY_ORDER:
        label = FAMILY_LABELS[family_id]
        qcount = 0
        tokens = 0
        for family_map in dataset_family.values():
            agg = family_map.get(family_id)
            if not agg:
                continue
            qcount += agg.query_count
            tokens += agg.total_tokens
        family_lines.append(f"- `{label}`: {qcount} queries, {tokens:,} total query tokens")

    text = "\n".join(
        [
            "# Dataset Query and Token Summary (v2)",
            "",
            "This report aggregates accepted v2 queries by dataset and by the five canonical families.",
            "",
            "## Sources",
            *(f"- `{path.name}`" for path in registry_files),
            "",
            "## Scope",
            f"- Datasets covered: `{dataset_count}`",
            f"- Accepted queries covered: `{total_queries}`",
            "- Token fields are aggregated from accepted registry rows (`usage_*` columns).",
            "- Deterministic queries usually contribute zero usage tokens because they do not call the SQL agent.",
            "",
            "## Family Totals",
            *family_lines,
            "",
            "## Output Files",
            "- `dataset_family_query_token_summary.csv`: long format (`dataset x family`).",
            "- `dataset_query_token_overview.csv`: wide format (`one row per dataset`).",
        ]
    )
    output_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    registry_dir = (
        args.registry_dir.resolve()
        if args.registry_dir
        else project_root / "data" / "workload_grounding_v2" / "registries"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "Evaluation" / "subitem_workload_v2" / "final"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_files = latest_main_batch_files(registry_dir)
    rerun_file = latest_failed_rerun_file(registry_dir)
    if rerun_file is not None:
        registry_files = registry_files + [rerun_file]

    dataset_family, accepted_by_query_id = build_aggregates(registry_files)
    write_long_csv(output_dir / "dataset_family_query_token_summary.csv", dataset_family)
    write_wide_csv(output_dir / "dataset_query_token_overview.csv", dataset_family)
    write_summary_md(
        output_dir / "dataset_query_token_overview.md",
        registry_files,
        dataset_family,
        accepted_by_query_id,
    )

    print(f"[query-token-summary] registry_files={len(registry_files)}")
    print(f"[query-token-summary] datasets={len(dataset_family)}")
    print(f"[query-token-summary] accepted_queries={len(accepted_by_query_id)}")
    print(f"[query-token-summary] output_dir={output_dir}")


if __name__ == "__main__":
    main()
