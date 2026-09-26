"""Statistical distance and fidelity evaluation against train splits."""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tqb_scoring.eval.common import (
    MISSING_TEXT,
    SyntheticAsset,
    TaskProgressTracker,
    discover_synthetic_assets,
    list_dataset_ids,
    load_field_type_hints,
    make_task_run_dir,
    mean_or_none,
    normalize_missing,
    real_split_provenance,
    read_json,
    resolve_real_split_path,
    write_csv,
    write_json,
    write_jsonl,
)
from tqb_scoring.eval.final_outputs import (
    STANDARD_MODEL_ORDER,
    build_longtable_report_tex,
    compile_tex,
    copy_files,
    normalize_standard_model_id,
    render_pdf_to_png,
)

MAX_CATEGORICAL_PAIRWISE_COLS = 48
MAX_NUMERIC_CORR_COLS = 64
MAX_MISSING_CORR_COLS = 96
MAX_MISSING_PATTERN_COLS = 128
MAX_CRAMERS_DISTINCT_PER_COLUMN = 1024
MAX_CRAMERS_DISTINCT_PRODUCT = 262144
MAX_CRAMERS_OBSERVED_PAIRS = 500000
TASK_NAME = "distance"
FINAL_DIR = Path(__file__).resolve().parents[2] / "results" / TASK_NAME / "final"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_CONTRACT_ROOTS = [
    PROJECT_ROOT / "data_train_as_main" / "artifacts" / "data_core" / "tabular",
    PROJECT_ROOT / "artifacts" / "data_core" / "tabular",
]

def _read_csv_with_delimiter_fallback(csv_path: Path, *, header: int | None | str = "infer") -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, dtype=str, keep_default_na=False, header=header)
    except pd.errors.ParserError:
        sample = csv_path.read_text(encoding="utf-8", errors="ignore")[:8192]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        return pd.read_csv(csv_path, dtype=str, keep_default_na=False, sep=delimiter, header=header)


def _load_expected_columns(dataset_id: str) -> list[str]:
    for root in DATASET_CONTRACT_ROOTS:
        path = root / dataset_id / f"{dataset_id}-dataset_contract_v1.json"
        payload = read_json(path, {}) or {}
        columns = payload.get("columns") or []
        names = [str(col.get("name") or "").strip() for col in columns if str(col.get("name") or "").strip()]
        if names:
            return names
    return []


def _normalize_text_token(value: Any) -> str:
    return str(value or "").strip()


def _row_matches_expected(values: list[Any], expected_columns: list[str]) -> bool:
    if len(values) != len(expected_columns):
        return False
    return all(_normalize_text_token(left) == _normalize_text_token(right) for left, right in zip(values, expected_columns))


def _load_csv_with_schema_repair(csv_path: Path, expected_columns: list[str]) -> pd.DataFrame:
    header_df = _read_csv_with_delimiter_fallback(csv_path, header="infer")
    if not expected_columns:
        return header_df

    actual_columns = [_normalize_text_token(col) for col in header_df.columns]
    if actual_columns == expected_columns:
        return header_df
    if len(actual_columns) == len(expected_columns) and set(actual_columns) == set(expected_columns):
        return header_df[expected_columns]

    no_header_df = _read_csv_with_delimiter_fallback(csv_path, header=None)
    if no_header_df.shape[1] == len(expected_columns):
        no_header_df.columns = expected_columns
        if not no_header_df.empty and _row_matches_expected(no_header_df.iloc[0].tolist(), expected_columns):
            no_header_df = no_header_df.iloc[1:].reset_index(drop=True)
        return no_header_df

    return header_df


def _infer_column_kind(series: pd.Series, hint: str) -> str:
    token = (hint or "").lower()
    if any(word in token for word in ["numeric", "integer", "float", "double", "decimal", "continuous"]):
        return "numeric"
    if any(word in token for word in ["categorical", "string", "text", "boolean", "ordinal"]):
        return "categorical"
    non_missing = series[~series.map(normalize_missing)]
    if non_missing.empty:
        return "categorical"
    parsed = pd.to_numeric(non_missing, errors="coerce")
    ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
    return "numeric" if ratio >= 0.95 else "categorical"


def _load_real_df(dataset_id: str) -> pd.DataFrame:
    real_path = resolve_real_split_path(dataset_id, split="train")
    if not real_path.exists():
        raise FileNotFoundError(f"Train split missing for {dataset_id}: {real_path}")
    return _load_csv_with_schema_repair(real_path, _load_expected_columns(dataset_id))


def _load_syn_df(synthetic_csv_path: Path, expected_columns: list[str]) -> pd.DataFrame:
    syn_df = _load_csv_with_schema_repair(synthetic_csv_path, expected_columns)
    for column in expected_columns:
        if column not in syn_df.columns:
            syn_df[column] = ""
    syn_df = syn_df[expected_columns]
    return syn_df


def _value_distribution(series: pd.Series) -> dict[str, float]:
    non_missing = series[~series.map(normalize_missing)]
    if non_missing.empty:
        return {}
    counts = non_missing.astype(str).value_counts(dropna=False)
    total = float(counts.sum())
    return {str(key): float(value) / total for key, value in counts.items()}


def _js_distance(dist_a: dict[str, float], dist_b: dict[str, float]) -> float | None:
    keys = sorted(set(dist_a) | set(dist_b))
    if not keys:
        return None
    p = np.array([dist_a.get(key, 0.0) for key in keys], dtype=float)
    q = np.array([dist_b.get(key, 0.0) for key in keys], dtype=float)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    js_div = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(math.sqrt(max(js_div, 0.0)))


def _tv_distance(dist_a: dict[str, float], dist_b: dict[str, float]) -> float | None:
    keys = sorted(set(dist_a) | set(dist_b))
    if not keys:
        return None
    return 0.5 * float(sum(abs(dist_a.get(key, 0.0) - dist_b.get(key, 0.0)) for key in keys))


def _ks_distance(real_values: pd.Series, syn_values: pd.Series) -> float | None:
    real_num = pd.to_numeric(real_values[~real_values.map(normalize_missing)], errors="coerce").dropna().to_numpy(dtype=float)
    syn_num = pd.to_numeric(syn_values[~syn_values.map(normalize_missing)], errors="coerce").dropna().to_numpy(dtype=float)
    if len(real_num) == 0 or len(syn_num) == 0:
        return None
    real_num.sort()
    syn_num.sort()
    grid = np.sort(np.unique(np.concatenate([real_num, syn_num])))
    real_cdf = np.searchsorted(real_num, grid, side="right") / len(real_num)
    syn_cdf = np.searchsorted(syn_num, grid, side="right") / len(syn_num)
    return float(np.max(np.abs(real_cdf - syn_cdf)))


def _range_coverage(real_values: pd.Series, syn_values: pd.Series) -> float | None:
    real_num = pd.to_numeric(real_values[~real_values.map(normalize_missing)], errors="coerce").dropna().to_numpy(dtype=float)
    syn_num = pd.to_numeric(syn_values[~syn_values.map(normalize_missing)], errors="coerce").dropna().to_numpy(dtype=float)
    if len(real_num) == 0 or len(syn_num) == 0:
        return None
    r_min, r_max = float(np.min(real_num)), float(np.max(real_num))
    s_min, s_max = float(np.min(syn_num)), float(np.max(syn_num))
    if r_max <= r_min:
        return 1.0 if s_min <= r_min <= s_max else 0.0
    overlap = max(0.0, min(r_max, s_max) - max(r_min, s_min))
    return overlap / (r_max - r_min)


def _wasserstein_distance_normalized(real_values: pd.Series, syn_values: pd.Series) -> float | None:
    real_num = pd.to_numeric(real_values[~real_values.map(normalize_missing)], errors="coerce").dropna().to_numpy(dtype=float)
    syn_num = pd.to_numeric(syn_values[~syn_values.map(normalize_missing)], errors="coerce").dropna().to_numpy(dtype=float)
    if len(real_num) == 0 or len(syn_num) == 0:
        return None
    q_count = int(max(32, min(256, max(len(real_num), len(syn_num)))))
    grid = np.linspace(0.0, 1.0, num=q_count)
    real_q = np.quantile(real_num, grid)
    syn_q = np.quantile(syn_num, grid)
    raw = float(np.mean(np.abs(real_q - syn_q)))
    real_range = float(np.max(real_num) - np.min(real_num))
    scale = real_range if real_range > 1e-12 else float(np.std(real_num))
    if scale <= 1e-12:
        scale = 1.0
    return min(raw / scale, 1.0)


def _cramers_v(series_a: pd.Series, series_b: pd.Series) -> float | None:
    mask = (~series_a.map(normalize_missing)) & (~series_b.map(normalize_missing))
    if not bool(mask.any()):
        return None
    clean_a = series_a[mask].astype(str)
    clean_b = series_b[mask].astype(str)
    if clean_a.empty or clean_b.empty:
        return None
    distinct_a = int(clean_a.nunique(dropna=True))
    distinct_b = int(clean_b.nunique(dropna=True))
    if distinct_a <= 1 or distinct_b <= 1:
        return None
    if distinct_a > MAX_CRAMERS_DISTINCT_PER_COLUMN or distinct_b > MAX_CRAMERS_DISTINCT_PER_COLUMN:
        return None
    if (distinct_a * distinct_b) > MAX_CRAMERS_DISTINCT_PRODUCT:
        return None

    row_counts: Counter[str] = Counter()
    col_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for value_a, value_b in zip(clean_a.to_numpy(), clean_b.to_numpy()):
        row_counts[str(value_a)] += 1
        col_counts[str(value_b)] += 1
        pair_counts[(str(value_a), str(value_b))] += 1
    if not pair_counts:
        return None
    if len(pair_counts) > MAX_CRAMERS_OBSERVED_PAIRS:
        return None

    total = float(sum(pair_counts.values()))
    if total <= 0:
        return None

    chi2 = 0.0
    for (value_a, value_b), observed in pair_counts.items():
        expected = (row_counts[value_a] * col_counts[value_b]) / total
        if expected > 0:
            chi2 += ((float(observed) - expected) ** 2) / expected
    r, k = len(row_counts), len(col_counts)
    denom = total * max(1, min(r - 1, k - 1))
    if denom <= 0:
        return None
    return float(math.sqrt(max(chi2 / denom, 0.0)))


def _corr_matrix_diff(df_real: pd.DataFrame, df_syn: pd.DataFrame, columns: list[str], missing: bool = False) -> tuple[float | None, int]:
    if len(columns) < 2:
        return None, 0
    if missing:
        real_num = pd.DataFrame({col: df_real[col].map(normalize_missing).astype(int) for col in columns})
        syn_num = pd.DataFrame({col: df_syn[col].map(normalize_missing).astype(int) for col in columns})
        scale = 2.0
    else:
        real_num = pd.DataFrame({col: pd.to_numeric(df_real[col], errors="coerce") for col in columns})
        syn_num = pd.DataFrame({col: pd.to_numeric(df_syn[col], errors="coerce") for col in columns})
        scale = 2.0
    real_corr = real_num.corr(method="pearson", min_periods=2)
    syn_corr = syn_num.corr(method="pearson", min_periods=2)
    diffs: list[float] = []
    for idx, col_a in enumerate(columns):
        for col_b in columns[idx + 1 :]:
            a = real_corr.get(col_a, pd.Series(dtype=float)).get(col_b)
            b = syn_corr.get(col_a, pd.Series(dtype=float)).get(col_b)
            if pd.isna(a) or pd.isna(b):
                continue
            diffs.append(abs(float(a) - float(b)) / scale)
    return (float(np.mean(diffs)) if diffs else None, len(diffs))


def _missing_pattern_jsd(df_real: pd.DataFrame, df_syn: pd.DataFrame, columns: list[str]) -> float | None:
    if not columns:
        return None
    real_patterns = df_real[columns].apply(lambda row: "|".join("1" if normalize_missing(v) else "0" for v in row), axis=1)
    syn_patterns = df_syn[columns].apply(lambda row: "|".join("1" if normalize_missing(v) else "0" for v in row), axis=1)
    return _js_distance(_value_distribution(real_patterns), _value_distribution(syn_patterns))


def _categorical_priority(df: pd.DataFrame, col: str) -> tuple[float, float, str]:
    series = df[col]
    non_missing = series[~series.map(normalize_missing)]
    distinct = float(non_missing.astype(str).nunique(dropna=True))
    coverage = float(len(non_missing) / max(1, len(series)))
    return (-distinct, -coverage, col)


def _numeric_priority(df: pd.DataFrame, col: str) -> tuple[float, float, str]:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return (float("inf"), float("inf"), col)
    std = float(series.std())
    coverage = float(len(series) / max(1, len(df)))
    return (-std, -coverage, col)


def _missing_priority(df: pd.DataFrame, col: str) -> tuple[float, float, str]:
    miss_rate = float(df[col].map(normalize_missing).mean())
    distinct = float(df[col][~df[col].map(normalize_missing)].astype(str).nunique(dropna=True))
    return (-miss_rate, -distinct, col)


def _select_columns(df: pd.DataFrame, columns: list[str], limit: int, kind: str) -> list[str]:
    if len(columns) <= limit:
        return list(columns)
    if kind == "categorical":
        ranked = sorted(columns, key=lambda col: _categorical_priority(df, col))
    elif kind == "numeric":
        ranked = sorted(columns, key=lambda col: _numeric_priority(df, col))
    else:
        ranked = sorted(columns, key=lambda col: _missing_priority(df, col))
    return ranked[:limit]


def _overall_fidelity_score(row: dict[str, Any]) -> float | None:
    score_candidates = [
        (1.0 - row["jensen_shannon_distance"]) if row.get("jensen_shannon_distance") is not None else None,
        (1.0 - row["kolmogorov_smirnov_distance"]) if row.get("kolmogorov_smirnov_distance") is not None else None,
        (1.0 - row["total_variation_distance"]) if row.get("total_variation_distance") is not None else None,
        (1.0 - row["wasserstein_distance"]) if row.get("wasserstein_distance") is not None else None,
    ]
    return mean_or_none(score_candidates)


def _evaluate_one_asset(
    dataset_id: str,
    asset: SyntheticAsset,
    *,
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    column_kinds: dict[str, str],
    categorical_cols: list[str],
    numeric_cols: list[str],
    categorical_assoc_cols: list[str],
    numeric_corr_cols: list[str],
    missing_corr_cols: list[str],
    missing_pattern_cols: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:

    js_values: list[float] = []
    tv_values: list[float] = []
    per_column: list[dict[str, Any]] = []
    for col in categorical_cols:
        real_dist = _value_distribution(real_df[col])
        syn_dist = _value_distribution(syn_df[col])
        jsd = _js_distance(real_dist, syn_dist)
        tvd = _tv_distance(real_dist, syn_dist)
        if jsd is not None:
            js_values.append(jsd)
        if tvd is not None:
            tv_values.append(tvd)
        per_column.append(
            {
                "column": col,
                "kind": "categorical",
                "jensen_shannon_distance": jsd,
                "total_variation_distance": tvd,
            }
        )

    ks_values: list[float] = []
    wasserstein_values: list[float] = []
    for col in numeric_cols:
        ks = _ks_distance(real_df[col], syn_df[col])
        wass = _wasserstein_distance_normalized(real_df[col], syn_df[col])
        if ks is not None:
            ks_values.append(ks)
        if wass is not None:
            wasserstein_values.append(wass)
        per_column.append(
            {
                "column": col,
                "kind": "numeric",
                "kolmogorov_smirnov_distance": ks,
                "wasserstein_distance": wass,
            }
        )

    real_provenance = real_split_provenance(dataset_id, split="train")
    row = {
        **asset.to_dict(),
        **real_provenance,
        "real_row_count": int(len(real_df)),
        "synthetic_row_count": int(len(syn_df)),
        "categorical_column_count": len(categorical_cols),
        "numeric_column_count": len(numeric_cols),
        "jensen_shannon_distance": mean_or_none(js_values),
        "kolmogorov_smirnov_distance": mean_or_none(ks_values),
        "total_variation_distance": mean_or_none(tv_values),
        "wasserstein_distance": mean_or_none(wasserstein_values),
    }
    row["overall_fidelity_score"] = _overall_fidelity_score(row)
    return row, {
        "asset": asset.to_dict(),
        "real_provenance": real_provenance,
        "column_kinds": column_kinds,
        "column_subsets": {},
        "per_column": per_column,
        "metric_contract": {
            "wasserstein_distance": "normalized by real-train range (fallback std) into [0,1]",
            "overall_fidelity_score": "mean of the four higher-is-better normalized distribution scores: 1-JSD, 1-KSD, 1-TVD, 1-Wasserstein",
        },
    }


def _run_distance_dataset(dataset_id: str, dataset_assets: list[SyntheticAsset]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    real_df = _load_real_df(dataset_id)
    hints = load_field_type_hints(dataset_id)
    column_kinds = {col: _infer_column_kind(real_df[col], hints.get(col, "")) for col in real_df.columns}
    categorical_cols = [col for col, kind in column_kinds.items() if kind == "categorical"]
    numeric_cols = [col for col, kind in column_kinds.items() if kind == "numeric"]
    categorical_assoc_cols = _select_columns(real_df, categorical_cols, MAX_CATEGORICAL_PAIRWISE_COLS, "categorical")
    numeric_corr_cols = _select_columns(real_df, numeric_cols, MAX_NUMERIC_CORR_COLS, "numeric")
    missing_corr_cols = _select_columns(real_df, list(real_df.columns), MAX_MISSING_CORR_COLS, "missing")
    missing_pattern_cols = _select_columns(real_df, list(real_df.columns), MAX_MISSING_PATTERN_COLS, "missing")
    per_dataset_summary: list[dict[str, Any]] = []
    per_dataset_details: list[dict[str, Any]] = []
    expected_columns = list(real_df.columns)
    for asset in dataset_assets:
        syn_df = _load_syn_df(Path(asset.synthetic_csv_path), expected_columns)
        row, detail = _evaluate_one_asset(
            dataset_id,
            asset,
            real_df=real_df,
            syn_df=syn_df,
            column_kinds=column_kinds,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            categorical_assoc_cols=categorical_assoc_cols,
            numeric_corr_cols=numeric_corr_cols,
            missing_corr_cols=missing_corr_cols,
            missing_pattern_cols=missing_pattern_cols,
        )
        per_dataset_summary.append(row)
        per_dataset_details.append(detail)
    return dataset_id, per_dataset_summary, per_dataset_details


def _dataset_sort_key(dataset_id: str) -> tuple[str, int, str]:
    text = str(dataset_id or "").strip().lower()
    prefix = text[:1]
    suffix = text[1:]
    try:
        numeric = int(suffix)
    except Exception:
        numeric = 10**9
    return (prefix, numeric, text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_distance_dataset_model_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset_ids = sorted(list_dataset_ids(), key=_dataset_sort_key)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    dataset_context_map: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        dataset_id = str(row.get("dataset_id") or "").strip()
        model_id = normalize_standard_model_id(row.get("model_id"))
        if dataset_id not in dataset_ids:
            continue
        if model_id not in STANDARD_MODEL_ORDER:
            continue
        dataset_context_map.setdefault(
            dataset_id,
            {
                field: row.get(field)
                for field in [
                    "provenance_contract_version",
                    "real_reference_split",
                    "real_source_kind",
                    "real_source_dataset_id",
                    "real_source_split",
                    "real_source_path",
                    "real_source_exists",
                    "real_source_mtime_utc",
                    "real_source_size_bytes",
                ]
                if row.get(field) not in (None, "")
            },
        )
        grouped[(dataset_id, model_id)].append(row)

    output: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        dataset_context = dataset_context_map.get(dataset_id, {})
        for model_id in STANDARD_MODEL_ORDER:
            rows = grouped.get((dataset_id, model_id), [])
            payload: dict[str, Any] = {
                **dataset_context,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "coverage_status": "ok" if rows else "missing_asset",
                "asset_count": len({str(row.get("asset_key") or "") for row in rows if row.get("asset_key")}),
                "server_types": ",".join(sorted({str(row.get("server_type") or "") for row in rows if row.get("server_type")})),
                "root_names": ",".join(sorted({str(row.get("root_name") or "") for row in rows if row.get("root_name")})),
                "real_row_count": mean_or_none([float(row.get("real_row_count")) for row in rows if row.get("real_row_count") not in (None, "")]),
                "synthetic_row_count": mean_or_none([float(row.get("synthetic_row_count")) for row in rows if row.get("synthetic_row_count") not in (None, "")]),
                "categorical_column_count": mean_or_none([float(row.get("categorical_column_count")) for row in rows if row.get("categorical_column_count") not in (None, "")]),
                "numeric_column_count": mean_or_none([float(row.get("numeric_column_count")) for row in rows if row.get("numeric_column_count") not in (None, "")]),
                "jensen_shannon_distance": mean_or_none([float(row.get("jensen_shannon_distance")) for row in rows if row.get("jensen_shannon_distance") not in (None, "")]),
                "kolmogorov_smirnov_distance": mean_or_none([float(row.get("kolmogorov_smirnov_distance")) for row in rows if row.get("kolmogorov_smirnov_distance") not in (None, "")]),
                "total_variation_distance": mean_or_none([float(row.get("total_variation_distance")) for row in rows if row.get("total_variation_distance") not in (None, "")]),
                "wasserstein_distance": mean_or_none([float(row.get("wasserstein_distance")) for row in rows if row.get("wasserstein_distance") not in (None, "")]),
                "overall_fidelity_score": mean_or_none([float(row.get("overall_fidelity_score")) for row in rows if row.get("overall_fidelity_score") not in (None, "")]),
            }
            output.append(payload)
    return output


def _build_distance_summary_note(manifest: dict[str, Any], dataset_model_rows: list[dict[str, Any]]) -> str:
    covered_rows = [row for row in dataset_model_rows if row["coverage_status"] == "ok"]
    missing_rows = [row for row in dataset_model_rows if row["coverage_status"] != "ok"]
    return "\n".join(
        [
            "# Distance Final Bundle",
            "",
            f"- run_tag: `{manifest['run_tag']}`",
            f"- dataset_count: `{manifest['dataset_count']}`",
            f"- raw_asset_count: `{manifest['asset_count']}`",
            f"- real_reference_split: `{manifest.get('real_reference_split') or ''}`",
            f"- standardized_grid_rows: `{len(dataset_model_rows)}`",
            f"- covered_rows: `{len(covered_rows)}`",
            f"- missing_rows: `{len(missing_rows)}`",
            "- standardized model grid: `arf, bayesnet, ctgan, forestdiffusion, realtabformer, tabbyflow, tabddpm, tabdiff, tabpfgen, tabsyn, tvae`",
            "",
        ]
    )


def _write_distance_final_bundle(run_dir: Path, manifest: dict[str, Any], dataset_model_rows: list[dict[str, Any]], latex_engine: str | None) -> dict[str, Any]:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    readme = "\n".join(
        [
            "# distance final outputs",
            "",
            "This directory stores the paper-facing final bundle for statistical distance evaluation.",
            "",
            "Main files:",
            "",
            "- `distance_dataset_model_metrics.csv`",
            "- `distance_summary_report.tex`",
            "- `distance_summary_report.pdf`",
            "- `distance_summary_report.png`",
            "",
        ]
    )
    _write_text(FINAL_DIR / "README.md", readme)

    standardized_csv_path = run_dir / "summaries" / "distance_dataset_model_metrics.csv"
    write_csv(standardized_csv_path, dataset_model_rows)
    copy_files(
        FINAL_DIR,
        [
            standardized_csv_path,
            run_dir / "summaries" / "distance_summary__all_datasets.csv",
            run_dir / "manifest.json",
        ],
    )

    note_path = FINAL_DIR / "distance_summary.md"
    tex_path = FINAL_DIR / "distance_summary_report.tex"
    png_path = FINAL_DIR / "distance_summary_report.png"
    note_text = _build_distance_summary_note(manifest, dataset_model_rows)
    _write_text(note_path, note_text)
    report_tex = build_longtable_report_tex(
        title="Distance Evaluation Final Report",
        subtitle="Standardized dataset-model grid for statistical distance and fidelity metrics.",
        intro_lines=[
            f"run_tag={manifest['run_tag']}",
            f"dataset_count={manifest['dataset_count']}",
            f"raw_asset_count={manifest['asset_count']}",
            f"standardized_grid_rows={len(dataset_model_rows)}",
        ],
        tables=[
            {
                "heading": "Dataset-Model Distance Metrics",
                "columns": [
                    ("dataset_id", "Dataset"),
                    ("model_id", "Model"),
                    ("coverage_status", "Coverage"),
                    ("overall_fidelity_score", "Overall"),
                    ("jensen_shannon_distance", "JSD"),
                    ("kolmogorov_smirnov_distance", "KSD"),
                    ("total_variation_distance", "TVD"),
                    ("wasserstein_distance", "Wasserstein"),
                ],
                "rows": dataset_model_rows,
                "note": "The final CSV always expands to current datasets x 11 standardized models, with missing combinations retained as explicit empty rows.",
            }
        ],
    )
    _write_text(tex_path, report_tex)
    pdf_path: Path | None = None
    log_path: Path | None = None
    try:
        pdf_path, log_path = compile_tex(tex_path, latex_engine=latex_engine)
        render_pdf_to_png(pdf_path, png_path, densest_page=True)
    except RuntimeError as exc:
        _write_text(FINAL_DIR / "distance_summary_report.compile_note.txt", str(exc).strip() + "\n")

    final_manifest = {
        "task": TASK_NAME,
        "run_tag": manifest["run_tag"],
        "run_dir": str(run_dir.resolve()),
        "final_dir": str(FINAL_DIR.resolve()),
        "provenance_contract_version": manifest.get("provenance_contract_version"),
        "real_reference_split": manifest.get("real_reference_split"),
        "real_source_kind": manifest.get("real_source_kind"),
        "dataset_model_metrics_csv": str((FINAL_DIR / standardized_csv_path.name).resolve()),
        "summary_note": str(note_path.resolve()),
        "report_tex": str(tex_path.resolve()),
        "report_pdf": str(pdf_path.resolve()) if pdf_path and pdf_path.exists() else None,
        "report_png": str(png_path.resolve()) if png_path.exists() else None,
        "report_compile_log": str(log_path.resolve()) if log_path and log_path.exists() else None,
        "row_count": len(dataset_model_rows),
    }
    write_json(FINAL_DIR / "distance_final_manifest.json", final_manifest)
    return final_manifest


def finalize_distance_run(*, run_dir: Path | str, latex_engine: str | None = None) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir).resolve()
    summary_csv_path = resolved_run_dir / "summaries" / "distance_summary__all_datasets.csv"
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"Distance summary CSV not found: {summary_csv_path}")
    summary_rows = pd.read_csv(summary_csv_path).replace({np.nan: None}).to_dict(orient="records")
    manifest_path = resolved_run_dir / "manifest.json"
    manifest = read_json(manifest_path, {}) or {}
    if not manifest:
        manifest = {
            "task": TASK_NAME,
            "run_tag": resolved_run_dir.name,
            "dataset_count": len({str(row.get("dataset_id") or "") for row in summary_rows if row.get("dataset_id")}),
            "asset_count": len(summary_rows),
            "provenance_contract_version": summary_rows[0].get("provenance_contract_version") if summary_rows else "",
            "real_reference_split": "train",
            "real_source_kind": "reference_split_csv",
            "latest_only": True,
            "max_workers": None,
        }
    else:
        manifest.setdefault("task", TASK_NAME)
        manifest.setdefault("run_tag", resolved_run_dir.name)
        manifest.setdefault(
            "dataset_count",
            len({str(row.get("dataset_id") or "") for row in summary_rows if row.get("dataset_id")}),
        )
        manifest.setdefault("asset_count", len(summary_rows))
        manifest.setdefault(
            "provenance_contract_version",
            summary_rows[0].get("provenance_contract_version") if summary_rows else "",
        )
        manifest.setdefault("real_reference_split", "train")
        manifest.setdefault("real_source_kind", "reference_split_csv")
    dataset_model_rows = _build_distance_dataset_model_rows(summary_rows)
    write_csv(resolved_run_dir / "summaries" / "distance_dataset_model_metrics.csv", dataset_model_rows)
    final_manifest = _write_distance_final_bundle(resolved_run_dir, manifest, dataset_model_rows, latex_engine)
    manifest["final_outputs"] = final_manifest
    write_json(manifest_path, manifest)
    return {
        "run_dir": resolved_run_dir,
        "manifest": manifest,
        "final_manifest": final_manifest,
        "dataset_model_row_count": len(dataset_model_rows),
    }


def run_distance_evaluation(
    *,
    run_tag: str,
    datasets: list[str] | None = None,
    latest_only: bool = True,
    max_workers: int = 1,
    latex_engine: str | None = None,
    root_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    dataset_ids = datasets or list_dataset_ids()
    run_dir = make_task_run_dir(TASK_NAME, run_tag)
    normalized_root_names = tuple(str(item).strip() for item in (root_names or []) if str(item).strip())
    assets = discover_synthetic_assets(
        datasets=dataset_ids,
        latest_only=latest_only,
        root_names=normalized_root_names,
    )
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    dataset_asset_map = {dataset_id: [asset for asset in assets if asset.dataset_id == dataset_id] for dataset_id in dataset_ids}
    dataset_asset_map = {k: v for k, v in dataset_asset_map.items() if v}
    progress = TaskProgressTracker(
        task_name="distance",
        total_steps=len(dataset_asset_map),
        step_label="datasets",
        substep_label="assets",
        total_substeps=sum(len(items) for items in dataset_asset_map.values()),
    )
    progress.print_start(
        extra=(
            f"run_dir={run_dir.resolve()}"
            f" | roots={','.join(normalized_root_names) if normalized_root_names else 'all'}"
        )
    )

    def _consume_result(dataset_id: str, per_dataset_summary: list[dict[str, Any]], per_dataset_details: list[dict[str, Any]]) -> None:
        summary_rows.extend(per_dataset_summary)
        detail_rows.extend(per_dataset_details)
        write_csv(run_dir / "datasets" / dataset_id / f"distance_summary__{dataset_id}.csv", per_dataset_summary)
        write_jsonl(run_dir / "datasets" / dataset_id / f"distance_details__{dataset_id}.jsonl", per_dataset_details)
        progress.advance(step_name=dataset_id, substeps_done=len(per_dataset_summary))

    if max_workers > 1 and len(dataset_asset_map) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_distance_dataset, dataset_id, dataset_assets): dataset_id
                for dataset_id, dataset_assets in dataset_asset_map.items()
            }
            for future in as_completed(futures):
                dataset_id, per_dataset_summary, per_dataset_details = future.result()
                _consume_result(dataset_id, per_dataset_summary, per_dataset_details)
    else:
        for dataset_id, dataset_assets in dataset_asset_map.items():
            dataset_id, per_dataset_summary, per_dataset_details = _run_distance_dataset(dataset_id, dataset_assets)
            _consume_result(dataset_id, per_dataset_summary, per_dataset_details)

    write_csv(run_dir / "summaries" / "distance_summary__all_datasets.csv", summary_rows)
    write_jsonl(run_dir / "summaries" / "distance_details__all_datasets.jsonl", detail_rows)
    manifest = {
        "task": TASK_NAME,
        "run_tag": run_tag,
        "dataset_count": len(dataset_ids),
        "asset_count": len(summary_rows),
        "provenance_contract_version": summary_rows[0].get("provenance_contract_version") if summary_rows else "",
        "real_reference_split": "train",
        "real_source_kind": "reference_split_csv",
        "latest_only": latest_only,
        "max_workers": max_workers,
        "synthetic_root_filter": list(normalized_root_names),
        "server_roots": {
            root_name: {
                "server_type": str(asset.server_type),
                "gpu_hour_ratio": None,
            }
            for root_name, asset in sorted(
                {
                    asset.root_name: asset
                    for asset in assets
                    if asset.root_name
                }.items()
            )
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    finalized = finalize_distance_run(run_dir=run_dir, latex_engine=latex_engine)
    return {"run_dir": run_dir, "summary_rows": summary_rows, "manifest": finalized["manifest"]}
