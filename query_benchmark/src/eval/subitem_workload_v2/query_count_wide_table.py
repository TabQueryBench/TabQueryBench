from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


FAMILY_ORDER = [
    ("subgroup_structure", "subgroup"),
    ("conditional_dependency_structure", "conditional"),
    ("tail_rarity_structure", "tail"),
    ("missingness_structure", "missingness"),
    ("cardinality_structure", "cardinality"),
]

PHYSICAL_BATCHED_DETERMINISTIC_SUBITEMS = {
    "marginal_missing_rate_consistency",
    "co_missingness_pattern_consistency",
    "support_rank_profile_consistency",
    "high_cardinality_response_stability",
}

SUBITEM_ORDER = [
    "internal_profile_stability",
    "subgroup_size_stability",
    "dependency_strength_similarity",
    "direction_consistency",
    "slice_level_consistency",
    "tail_set_consistency",
    "tail_mass_similarity",
    "tail_concentration_consistency",
    "marginal_missing_rate_consistency",
    "co_missingness_pattern_consistency",
    "support_rank_profile_consistency",
    "high_cardinality_response_stability",
]

SUBITEM_SHORT = {
    "internal_profile_stability": "internal_profile",
    "subgroup_size_stability": "subgroup_size",
    "dependency_strength_similarity": "dep_strength",
    "direction_consistency": "direction",
    "slice_level_consistency": "slice_level",
    "tail_set_consistency": "tail_set",
    "tail_mass_similarity": "tail_mass",
    "tail_concentration_consistency": "tail_concentration",
    "marginal_missing_rate_consistency": "missing_rate",
    "co_missingness_pattern_consistency": "co_missing_profile",
    "support_rank_profile_consistency": "support_rank",
    "high_cardinality_response_stability": "high_card_response",
}

PAPER_SUBITEM_SHORT = {
    "internal_profile_stability": "subgroup_correlation",
    "subgroup_size_stability": "subgroup_size",
    "dependency_strength_similarity": "dep_strength",
    "direction_consistency": "direction",
    "slice_level_consistency": "slice_level",
    "tail_set_consistency": "tail_set",
    "tail_mass_similarity": "tail_mass",
    "tail_concentration_consistency": "tail_concentration",
    "marginal_missing_rate_consistency": "missing_rate",
    "co_missingness_pattern_consistency": "co_missing_profile",
    "support_rank_profile_consistency": "support_rank",
    "high_cardinality_response_stability": "high_card_response",
}

SUBITEM_TO_FAMILY = {
    "internal_profile_stability": "subgroup_structure",
    "subgroup_size_stability": "subgroup_structure",
    "dependency_strength_similarity": "conditional_dependency_structure",
    "direction_consistency": "conditional_dependency_structure",
    "slice_level_consistency": "conditional_dependency_structure",
    "tail_set_consistency": "tail_rarity_structure",
    "tail_mass_similarity": "tail_rarity_structure",
    "tail_concentration_consistency": "tail_rarity_structure",
    "marginal_missing_rate_consistency": "missingness_structure",
    "co_missingness_pattern_consistency": "missingness_structure",
    "support_rank_profile_consistency": "cardinality_structure",
    "high_cardinality_response_stability": "cardinality_structure",
}


def natural_dataset_key(dataset_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", dataset_id)
    if not match:
        return (dataset_id.lower(), -1)
    return (match.group(1).lower(), int(match.group(2)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dataset x family x subitem accepted query count table for subitem workload v2."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_main_batch_files(registry_dir: Path) -> list[Path]:
    pattern = re.compile(r"v2_cli_(\d{8}_\d{6})_([a-z])_query_registry_v2\.csv$")
    by_stamp: dict[str, list[Path]] = defaultdict(list)
    for path in registry_dir.glob("v2_cli_*_query_registry_v2.csv"):
        match = pattern.fullmatch(path.name)
        if match:
            by_stamp[match.group(1)].append(path)
    if not by_stamp:
        raise FileNotFoundError(f"No formal v2 registry files found in {registry_dir}")
    latest_stamp = sorted(by_stamp)[-1]
    return sorted(by_stamp[latest_stamp], key=lambda p: p.name)


def latest_failed_rerun_file(registry_dir: Path) -> Path | None:
    files = sorted(registry_dir.glob("subitem_workload_v2_failed_rerun_*_query_registry_v2.csv"))
    return files[-1] if files else None


def collect_counts(registry_files: list[Path]) -> tuple[list[str], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    accepted_by_query_id: dict[str, dict[str, str]] = {}
    for path in registry_files:
        for row in load_csv_rows(path):
            if row.get("accepted_for_eval", "").lower() != "true":
                continue
            query_id = row.get("query_record_id", "")
            if query_id:
                accepted_by_query_id[query_id] = row

    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    subitem_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    dataset_ids: set[str] = set()
    for row in accepted_by_query_id.values():
        dataset_id = row["dataset_id"]
        family_id = row["family_id"]
        subitem_id = row["canonical_subitem_id"]
        dataset_ids.add(dataset_id)
        family_counts[dataset_id][family_id] += 1
        subitem_counts[dataset_id][subitem_id] += 1

    datasets = sorted(dataset_ids, key=natural_dataset_key)
    return datasets, family_counts, subitem_counts


def write_csv(
    output_path: Path,
    datasets: list[str],
    family_counts: dict[str, dict[str, int]],
    subitem_counts: dict[str, dict[str, int]],
) -> None:
    fieldnames = ["dataset_id"]
    fieldnames.extend([f"{label}_queries" for _, label in FAMILY_ORDER])
    fieldnames.extend([f"{SUBITEM_SHORT[subitem]}_queries" for subitem in SUBITEM_ORDER])

    rows: list[dict[str, int | str]] = []
    for dataset_id in datasets:
        row: dict[str, int | str] = {"dataset_id": dataset_id}
        for family_id, label in FAMILY_ORDER:
            row[f"{label}_queries"] = family_counts[dataset_id].get(family_id, 0)
        for subitem_id in SUBITEM_ORDER:
            row[f"{SUBITEM_SHORT[subitem_id]}_queries"] = subitem_counts[dataset_id].get(subitem_id, 0)
        rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def write_tex(
    output_path: Path,
    datasets: list[str],
    family_counts: dict[str, dict[str, int]],
    subitem_counts: dict[str, dict[str, int]],
) -> None:
    header = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Accepted query counts per dataset, first aggregated by the five query families and then by all canonical subitems in the v2 workload line.}",
        r"\label{tab:v2_dataset_family_subitem_query_counts}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "r" * (len(FAMILY_ORDER) + len(SUBITEM_ORDER)) + "}",
        r"\toprule",
        "Dataset & "
        + " & ".join(tex_escape(label) for _, label in FAMILY_ORDER)
        + " & "
        + " & ".join(tex_escape(SUBITEM_SHORT[sub]) for sub in SUBITEM_ORDER)
        + r" \\",
        r"\midrule",
    ]

    body: list[str] = []
    for dataset_id in datasets:
        values: list[str] = [tex_escape(dataset_id)]
        for family_id, _label in FAMILY_ORDER:
            values.append(str(family_counts[dataset_id].get(family_id, 0)))
        for subitem_id in SUBITEM_ORDER:
            values.append(str(subitem_counts[dataset_id].get(subitem_id, 0)))
        body.append(" & ".join(values) + r" \\")

    footer = [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]

    output_path.write_text("\n".join(header + body + footer) + "\n", encoding="utf-8")


def write_md(output_path: Path, registry_files: list[Path], datasets: list[str]) -> None:
    lines = [
        "# Dataset Query Count Table (v2)",
        "",
        "This table reports accepted query counts per dataset, first by the five canonical families and then by each canonical subitem.",
        "",
        "## Sources",
        *(f"- `{path.name}`" for path in registry_files),
        "",
        f"- Datasets covered: `{len(datasets)}`",
        "- Only `accepted_for_eval = true` rows are counted.",
        "",
        "## Output Files",
        "- `dataset_family_subitem_query_counts.csv`",
        "- `dataset_family_subitem_query_counts.tex`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paper_subitem_count(subitem_id: str, raw_count: int) -> int:
    if subitem_id in PHYSICAL_BATCHED_DETERMINISTIC_SUBITEMS:
        return 1 if raw_count > 0 else 0
    return raw_count


def _paper_family_count(
    dataset_id: str,
    family_id: str,
    family_counts: dict[str, dict[str, int]],
    subitem_counts: dict[str, dict[str, int]],
) -> int:
    if family_id not in {"missingness_structure", "cardinality_structure"}:
        return family_counts[dataset_id].get(family_id, 0)
    return sum(
        _paper_subitem_count(subitem_id, subitem_counts[dataset_id].get(subitem_id, 0))
        for subitem_id in SUBITEM_ORDER
        if SUBITEM_TO_FAMILY[subitem_id] == family_id
    )


def write_paper_csv(
    output_path: Path,
    datasets: list[str],
    family_counts: dict[str, dict[str, int]],
    subitem_counts: dict[str, dict[str, int]],
) -> None:
    fieldnames = ["dataset_id", "total_queries"]
    fieldnames.extend([f"{label}_queries" for _, label in FAMILY_ORDER])
    fieldnames.extend([f"{PAPER_SUBITEM_SHORT[subitem]}_queries" for subitem in SUBITEM_ORDER])

    rows: list[dict[str, int | str]] = []
    for dataset_id in datasets:
        paper_family_counts = {
            family_id: _paper_family_count(dataset_id, family_id, family_counts, subitem_counts)
            for family_id, _ in FAMILY_ORDER
        }
        total_queries = sum(paper_family_counts.values())
        row: dict[str, int | str] = {
            "dataset_id": dataset_id,
            "total_queries": total_queries,
        }
        for family_id, label in FAMILY_ORDER:
            row[f"{label}_queries"] = paper_family_counts[family_id]
        for subitem_id in SUBITEM_ORDER:
            row[f"{PAPER_SUBITEM_SHORT[subitem_id]}_queries"] = _paper_subitem_count(
                subitem_id,
                subitem_counts[dataset_id].get(subitem_id, 0),
            )
        rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_paper_tex(
    output_path: Path,
    datasets: list[str],
    family_counts: dict[str, dict[str, int]],
    subitem_counts: dict[str, dict[str, int]],
) -> None:
    family_headers = [label for _, label in FAMILY_ORDER]
    subitem_headers = [PAPER_SUBITEM_SHORT[subitem] for subitem in SUBITEM_ORDER]
    header = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Accepted query counts per dataset in the v2 workload line. Each row reports the total accepted SQL count, the five family totals, and then the fully expanded canonical subitem counts.}",
        r"\label{tab:v2_dataset_total_family_subitem_query_counts}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "r" * (1 + len(FAMILY_ORDER) + len(SUBITEM_ORDER)) + "}",
        r"\toprule",
        "Dataset & Total & "
        + " & ".join(tex_escape(label) for label in family_headers)
        + " & "
        + " & ".join(tex_escape(label) for label in subitem_headers)
        + r" \\",
        r"\midrule",
    ]

    body: list[str] = []
    for dataset_id in datasets:
        paper_family_counts = {
            family_id: _paper_family_count(dataset_id, family_id, family_counts, subitem_counts)
            for family_id, _ in FAMILY_ORDER
        }
        total_queries = sum(paper_family_counts.values())
        values: list[str] = [tex_escape(dataset_id), str(total_queries)]
        for family_id, _label in FAMILY_ORDER:
            values.append(str(paper_family_counts[family_id]))
        for subitem_id in SUBITEM_ORDER:
            values.append(str(_paper_subitem_count(subitem_id, subitem_counts[dataset_id].get(subitem_id, 0))))
        body.append(" & ".join(values) + r" \\")

    footer = [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]

    output_path.write_text("\n".join(header + body + footer) + "\n", encoding="utf-8")


def write_paper_md(output_path: Path, registry_files: list[Path], datasets: list[str]) -> None:
    lines = [
        "# Dataset Total / Family / Subitem Query Count Table (v2)",
        "",
        "This paper-facing table reports accepted SQL counts per dataset after deterministic batching.",
        "For the deterministic families, `missingness` contributes up to two physical SQL artifacts per dataset",
        "(one for `missing_rate`, one for `co_missing_profile`) and `cardinality` contributes up to two",
        "physical SQL artifacts per dataset (one for `support_rank`, one for `high_card_response`).",
        "The agent-generated families (`subgroup`, `conditional`, `tail`) remain direct accepted-query counts.",
        "",
        "## Sources",
        *(f"- `{path.name}`" for path in registry_files),
        "",
        f"- Datasets covered: `{len(datasets)}`",
        "- Only `accepted_for_eval = true` rows are counted.",
        "",
        "## Output Files",
        "- `dataset_total_family_subitem_query_counts.csv`",
        "- `dataset_total_family_subitem_query_counts.tex`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    datasets, family_counts, subitem_counts = collect_counts(registry_files)
    write_csv(
        output_dir / "dataset_family_subitem_query_counts.csv",
        datasets,
        family_counts,
        subitem_counts,
    )
    write_tex(
        output_dir / "dataset_family_subitem_query_counts.tex",
        datasets,
        family_counts,
        subitem_counts,
    )
    write_md(output_dir / "dataset_family_subitem_query_counts.md", registry_files, datasets)
    write_paper_csv(
        output_dir / "dataset_total_family_subitem_query_counts.csv",
        datasets,
        family_counts,
        subitem_counts,
    )
    write_paper_tex(
        output_dir / "dataset_total_family_subitem_query_counts.tex",
        datasets,
        family_counts,
        subitem_counts,
    )
    write_paper_md(output_dir / "dataset_total_family_subitem_query_counts.md", registry_files, datasets)

    print(f"[dataset-query-count-table] datasets={len(datasets)}")
    print(f"[dataset-query-count-table] output_dir={output_dir}")


if __name__ == "__main__":
    main()
