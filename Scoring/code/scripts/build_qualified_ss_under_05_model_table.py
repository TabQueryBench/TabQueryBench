from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MERGED_TABLE_PATH = REPO_ROOT / "synthetic_data_audit_20260503" / "table_merged.csv"
DISTANCE_ASSET_PATH = REPO_ROOT.parent / "results" / "distance" / "final" / "distance_summary__all_datasets.csv"
ANALYSIS_RUN_DIR = REPO_ROOT.parent / "results" / "analysis" / "runs" / "20260503_analysis_v2_merged"
ANALYSIS_ASSET_PATH = ANALYSIS_RUN_DIR / "summaries" / "analysis_asset_scores__all_datasets.csv"
ANALYSIS_DATASETS_DIR = ANALYSIS_RUN_DIR / "datasets"
OUTPUT_DIR = REPO_ROOT / "synthetic_data_audit_20260503" / "qualified_ss_under_0p5_model_summary_20260503"

ROOT_MAP = {
    "5": "SynOutput-5090",
    "6": "SynOutput",
}

DISTANCE_COLUMNS = {
    "distance_overall": "overall_fidelity_score",
    "distance_jsd": "jensen_shannon_distance",
    "distance_tvd": "total_variation_distance",
    "distance_ks": "kolmogorov_smirnov_distance",
    "distance_wasserstein": "wasserstein_distance",
}

FAMILY_COLUMNS = {
    "family_conditional_dependency_structure": "conditional_dependency_structure",
    "family_missingness_structure": "missingness_structure",
    "family_subgroup_structure": "subgroup_structure",
    "family_tail_rarity_structure": "tail_rarity_structure",
}

SUBITEM_COLUMNS = {
    "subitem_dependency_strength_similarity": "dependency_strength_similarity",
    "subitem_direction_consistency": "direction_consistency",
    "subitem_slice_level_consistency": "slice_level_consistency",
    "subitem_co_missingness_pattern_consistency": "co_missingness_pattern_consistency",
    "subitem_marginal_missing_rate_consistency": "marginal_missing_rate_consistency",
    "subitem_internal_profile_stability": "internal_profile_stability",
    "subitem_subgroup_size_stability": "subgroup_size_stability",
    "subitem_tail_concentration_consistency": "tail_concentration_consistency",
    "subitem_tail_mass_similarity": "tail_mass_similarity",
    "subitem_tail_set_consistency": "tail_set_consistency",
}


@dataclass(frozen=True)
class ParsedFragment:
    dataset_id: str
    model_id: str
    root_name: str
    source_code: str
    fragment: str
    row_status: str
    problem_code: str
    jsd_from_table: float | None
    is_qualified: bool


def parse_cell(dataset_id: str, model_id: str, cell_value: str) -> list[ParsedFragment]:
    if pd.isna(cell_value) or cell_value == "N":
        return []

    text = str(cell_value)
    if text.startswith("56-"):
        roots = ["5", "6"]
        fragments = text[3:].split("-")
    elif text.startswith("5-"):
        roots = ["5"]
        fragments = [text[2:]]
    elif text.startswith("6-"):
        roots = ["6"]
        fragments = [text[2:]]
    else:
        raise ValueError(f"Unrecognized merged-cell format: {dataset_id} / {model_id} / {text}")

    parsed: list[ParsedFragment] = []
    for source_code, fragment in zip(roots, fragments, strict=True):
        match = re.fullmatch(r"([SVW])([SCtdz])([0-9]+(?:\.[0-9]+)?|_)", fragment)
        if not match:
            raise ValueError(f"Could not parse fragment: {dataset_id} / {model_id} / {fragment}")
        row_status, problem_code, jsd_raw = match.groups()
        jsd_value = None if jsd_raw == "_" else float(jsd_raw)
        parsed.append(
            ParsedFragment(
                dataset_id=dataset_id,
                model_id=model_id,
                root_name=ROOT_MAP[source_code],
                source_code=source_code,
                fragment=fragment,
                row_status=row_status,
                problem_code=problem_code,
                jsd_from_table=jsd_value,
                is_qualified=(row_status == "S" and problem_code == "S" and jsd_value is not None and jsd_value <= 0.5),
            )
        )
    return parsed


def load_qualified_fragments() -> pd.DataFrame:
    merged = pd.read_csv(MERGED_TABLE_PATH)
    models = [col for col in merged.columns if col != "dataset_id"]
    rows: list[dict[str, object]] = []

    for _, row in merged.iterrows():
        dataset_id = row["dataset_id"]
        for model_id in models:
            for fragment in parse_cell(dataset_id, model_id, row[model_id]):
                rows.append(fragment.__dict__)

    fragments = pd.DataFrame(rows)
    fragments["qualifier_key"] = (
        fragments["dataset_id"] + "||" + fragments["model_id"] + "||" + fragments["root_name"]
    )
    return fragments


def load_family_rows(dataset_ids: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for dataset_id in dataset_ids:
        path = ANALYSIS_DATASETS_DIR / dataset_id / f"analysis_family_scores__{dataset_id}.csv"
        if path.exists():
            pieces.append(pd.read_csv(path))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def load_subitem_rows(dataset_ids: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for dataset_id in dataset_ids:
        path = ANALYSIS_DATASETS_DIR / dataset_id / f"analysis_subitem_scores__{dataset_id}.csv"
        if path.exists():
            pieces.append(pd.read_csv(path))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def pivot_metric_mean(df: pd.DataFrame, index: str, column: str, value: str, aliases: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        out = pd.DataFrame(index=sorted(set()))
        for alias in aliases:
            out[alias] = pd.NA
        return out

    grouped = df.groupby([index, column], dropna=False)[value].mean().reset_index()
    pivoted = grouped.pivot(index=index, columns=column, values=value)
    result = pd.DataFrame(index=sorted(df[index].dropna().unique().tolist()))
    for alias, raw_name in aliases.items():
        result[alias] = pivoted.get(raw_name)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fragments = load_qualified_fragments()
    qualified = fragments.loc[fragments["is_qualified"]].copy()
    dataset_ids = sorted(qualified["dataset_id"].unique().tolist())

    distance_assets = pd.read_csv(DISTANCE_ASSET_PATH)
    distance_assets["qualifier_key"] = (
        distance_assets["dataset_id"] + "||" + distance_assets["model_id"] + "||" + distance_assets["root_name"]
    )
    distance_join = qualified.merge(
        distance_assets[
            [
                "qualifier_key",
                "asset_key",
                "overall_fidelity_score",
                "jensen_shannon_distance",
                "total_variation_distance",
                "kolmogorov_smirnov_distance",
                "wasserstein_distance",
            ]
        ],
        on="qualifier_key",
        how="left",
    )

    analysis_assets = pd.read_csv(ANALYSIS_ASSET_PATH)
    analysis_assets["qualifier_key"] = (
        analysis_assets["dataset_id"] + "||" + analysis_assets["model_id"] + "||" + analysis_assets["root_name"]
    )
    sql_join = qualified.merge(
        analysis_assets[["qualifier_key", "asset_key", "overall_score"]],
        on="qualifier_key",
        how="left",
    )

    family_rows = load_family_rows(dataset_ids)
    family_rows["qualifier_key"] = (
        family_rows["dataset_id"] + "||" + family_rows["model_id"] + "||" + family_rows["root_name"]
    )
    family_join = qualified.merge(
        family_rows[["qualifier_key", "asset_key", "family_id", "family_score"]],
        on="qualifier_key",
        how="left",
    )

    subitem_rows = load_subitem_rows(dataset_ids)
    subitem_rows["qualifier_key"] = (
        subitem_rows["dataset_id"] + "||" + subitem_rows["model_id"] + "||" + subitem_rows["root_name"]
    )
    subitem_join = qualified.merge(
        subitem_rows[["qualifier_key", "asset_key", "subitem_id", "subitem_score"]],
        on="qualifier_key",
        how="left",
    )

    model_index = sorted(qualified["model_id"].unique().tolist())
    summary = pd.DataFrame(index=model_index)
    summary.index.name = "model_id"
    summary["qualified_source_combo_count"] = qualified.groupby("model_id").size()
    summary["qualified_dataset_count"] = qualified.groupby("model_id")["dataset_id"].nunique()
    summary["distance_asset_count"] = distance_join.groupby("model_id")["asset_key"].apply(lambda s: s.notna().sum())
    summary["sql_asset_count"] = sql_join.groupby("model_id")["asset_key"].apply(lambda s: s.notna().sum())

    for alias, raw_col in DISTANCE_COLUMNS.items():
        summary[alias] = distance_join.groupby("model_id")[raw_col].mean()

    summary["sql_overall"] = sql_join.groupby("model_id")["overall_score"].mean()

    family_pivot = pivot_metric_mean(family_join, "model_id", "family_id", "family_score", FAMILY_COLUMNS)
    subitem_pivot = pivot_metric_mean(subitem_join, "model_id", "subitem_id", "subitem_score", SUBITEM_COLUMNS)
    summary = summary.join(family_pivot, how="left")
    summary = summary.join(subitem_pivot, how="left")

    summary["rank_sql_overall"] = summary["sql_overall"].rank(method="min", ascending=False)
    summary["rank_distance_overall"] = summary["distance_overall"].rank(method="min", ascending=False)
    summary = summary.reset_index()
    summary = summary.sort_values(["rank_sql_overall", "rank_distance_overall", "model_id"], na_position="last")

    qualified_assets = qualified.merge(
        distance_assets[
            [
                "qualifier_key",
                "asset_key",
                "overall_fidelity_score",
                "jensen_shannon_distance",
                "total_variation_distance",
                "kolmogorov_smirnov_distance",
                "wasserstein_distance",
            ]
        ],
        on="qualifier_key",
        how="left",
    ).merge(
        analysis_assets[["qualifier_key", "overall_score"]],
        on="qualifier_key",
        how="left",
        suffixes=("", "_sql"),
    )
    qualified_assets = qualified_assets[
        [
            "dataset_id",
            "model_id",
            "root_name",
            "row_status",
            "problem_code",
            "jsd_from_table",
            "asset_key",
            "overall_fidelity_score",
            "jensen_shannon_distance",
            "total_variation_distance",
            "kolmogorov_smirnov_distance",
            "wasserstein_distance",
            "overall_score",
        ]
    ].rename(
        columns={
            "overall_fidelity_score": "distance_overall",
            "jensen_shannon_distance": "distance_jsd",
            "total_variation_distance": "distance_tvd",
            "kolmogorov_smirnov_distance": "distance_ks",
            "wasserstein_distance": "distance_wasserstein",
            "overall_score": "sql_overall",
        }
    ).sort_values(["model_id", "dataset_id", "root_name"])

    coverage_rows = []
    for _, row in summary.iterrows():
        coverage_rows.append(
            {
                "model_id": row["model_id"],
                "qualified_source_combo_count": int(row["qualified_source_combo_count"]),
                "distance_asset_count": int(row["distance_asset_count"]) if pd.notna(row["distance_asset_count"]) else 0,
                "sql_asset_count": int(row["sql_asset_count"]) if pd.notna(row["sql_asset_count"]) else 0,
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    summary_path = OUTPUT_DIR / "model_score_summary.csv"
    qualified_path = OUTPUT_DIR / "qualified_source_combinations.csv"
    coverage_path = OUTPUT_DIR / "coverage_by_model.csv"
    missing_path = OUTPUT_DIR / "missing_evaluation_assets.csv"
    workbook_path = OUTPUT_DIR / "qualified_ss_under_0p5_model_summary.xlsx"
    report_path = OUTPUT_DIR / "README.md"

    missing_eval = qualified_assets.loc[
        qualified_assets["distance_overall"].isna() | qualified_assets["sql_overall"].isna()
    ].copy()
    missing_eval["missing_distance"] = missing_eval["distance_overall"].isna()
    missing_eval["missing_sql"] = missing_eval["sql_overall"].isna()
    missing_eval = missing_eval.sort_values(["model_id", "dataset_id", "root_name"])

    summary.to_csv(summary_path, index=False)
    qualified_assets.to_csv(qualified_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    missing_eval.to_csv(missing_path, index=False)

    with pd.ExcelWriter(workbook_path) as writer:
        summary.to_excel(writer, sheet_name="ModelSummary", index=False)
        qualified_assets.to_excel(writer, sheet_name="QualifiedAssets", index=False)
        coverage.to_excel(writer, sheet_name="Coverage", index=False)
        missing_eval.to_excel(writer, sheet_name="MissingEval", index=False)

    family_count = len(FAMILY_COLUMNS)
    subitem_count = len(SUBITEM_COLUMNS)
    report_lines = [
        "# Qualified SS<=0.5 Model Summary",
        "",
        f"- Source merged table: `{MERGED_TABLE_PATH}`",
        "- Qualification rule: row status `S` and problem code `S` and table JSD `<= 0.5`.",
        f"- Qualified source-level combinations: `{len(qualified)}`",
        f"- Qualified dataset-model cells (deduplicated across roots): `{qualified[['dataset_id', 'model_id']].drop_duplicates().shape[0]}`",
        "- Distance source: `Evaluation/distance/final/distance_summary__all_datasets.csv`",
        "- SQL source: `Evaluation/analysis/runs/20260503_analysis_v2_merged`",
        "- SQL contract version in those files: `analytics_family_subitem_contract_v1`",
        f"- Actual SQL family count found in source files: `{family_count}`",
        f"- Actual SQL subitem count found in source files: `{subitem_count}`",
        "",
        "## Notes",
        "",
        "- The repository currently exposes 4 family columns and 10 subitem columns in the complete SQL analysis source used here, not 5 and 12.",
        "- Model rows are averaged over all qualified source-level assets for that model.",
        "- If a qualified asset is missing SQL or distance results, that metric is skipped from the model mean.",
        "- `rank_sql_overall` and `rank_distance_overall` are descending ranks among the model means.",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
