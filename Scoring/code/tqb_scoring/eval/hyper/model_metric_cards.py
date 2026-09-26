from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_DISPLAY = {
    "arf": "ARF",
    "bayesnet": "Bayes",
    "ctgan": "CTGAN",
    "forestdiffusion": "F-Diff",
    "realtabformer": "RTF",
    "tabbyflow": "T-Flow",
    "tabddpm": "T-DDPM",
    "tabdiff": "T-Diff",
    "tabpfgen": "TPF",
    "tabsyn": "T-Syn",
    "tvae": "TVAE",
}

MODEL_ORDER = [
    "arf",
    "bayesnet",
    "ctgan",
    "forestdiffusion",
    "realtabformer",
    "tabbyflow",
    "tabddpm",
    "tabdiff",
    "tabpfgen",
    "tabsyn",
    "tvae",
]

DATASET_ORDER = ["c2", "m4", "n3"]

FAMILY_COLUMNS = {
    "subgroup_structure": "subgroup_score",
    "conditional_dependency_structure": "conditional_score",
    "tail_rarity_structure": "tail_score",
    "missingness_structure": "missing_score",
    "cardinality_structure": "cardinality_score",
}

METRIC_SPECS = [
    ("train_sec", "Train(s)", "lower_better"),
    ("generate_sec", "Gen(s)", "lower_better"),
    ("subgroup_score", "Subgroup", "higher_better"),
    ("conditional_score", "Cond.", "higher_better"),
    ("tail_score", "Tail", "higher_better"),
    ("missing_score", "Missing", "higher_better"),
    ("cardinality_score", "Card.", "higher_better"),
    ("sql_overall_score", "SQL Avg", "higher_better"),
    ("jensen_shannon_distance", "JSD", "lower_better"),
    ("total_variation_distance", "TVD", "lower_better"),
    ("kolmogorov_smirnov_distance", "KS", "lower_better"),
    ("wasserstein_distance", "WD", "lower_better"),
    ("distance_overall_score", "Dist Avg", "higher_better"),
]


@dataclass
class HyperRunPaths:
    analysis_run_dir: Path
    distance_run_dir: Path
    validation_run_dir: Path | None = None


def _latest_run_dir(root: Path, suffix: str) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("hyperparameter_eval_") and path.name.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"No hyperparameter run directories with suffix {suffix!r} under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_latest_hyper_runs(project_root: Path) -> HyperRunPaths:
    analysis_root = project_root.parent / "results" / "analysis" / "runs"
    distance_root = project_root.parent / "results" / "distance" / "runs"
    validation_root = project_root.parent / "results" / "validation" / "runs"
    validation_run_dir = None
    if validation_root.exists():
        validation_candidates = [
            path
            for path in validation_root.iterdir()
            if path.is_dir() and path.name.startswith("hyperparameter_eval_") and path.name.endswith("_validation")
        ]
        if validation_candidates:
            validation_run_dir = max(validation_candidates, key=lambda path: path.stat().st_mtime)
    return HyperRunPaths(
        analysis_run_dir=_latest_run_dir(analysis_root, "_sql"),
        distance_run_dir=_latest_run_dir(distance_root, "_distance"),
        validation_run_dir=validation_run_dir,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_env_overrides(model_id: str, env_overrides: dict[str, Any]) -> str:
    clean = {
        str(key): str(value).strip()
        for key, value in sorted(env_overrides.items())
        if not str(key).startswith("BENCHMARK_") and str(value).strip()
    }
    if not clean:
        return "default"

    if model_id == "tabsyn":
        epoch = clean.get("TABSYN_DIFFUSION_MAX_EPOCHS") or clean.get("TABSYN_VAE_EPOCHS") or "?"
        batch = clean.get("TABSYN_VAE_BATCH_SIZE") or "?"
        return f"ep={epoch}, bs={batch}"
    if model_id == "tabbyflow":
        eval_n = clean.get("EFVFM_EVAL_NUM_SAMPLES", "?")
        sample_bs = clean.get("EFVFM_SAMPLE_BATCH_SIZE", "?")
        return f"eval={eval_n}, samp={sample_bs}"
    if model_id == "tvae":
        epoch = clean.get("TVAE_EPOCHS", "?")
        emb = clean.get("TVAE_EMBEDDING_DIM", "?")
        comp = clean.get("TVAE_COMPRESS_DIMS", "?").replace(",", "x")
        batch = clean.get("TVAE_BATCH_SIZE", "?")
        return f"ep={epoch}, emb={emb}, comp={comp}, bs={batch}"
    if model_id == "arf":
        return ", ".join(
            [
                f"it={clean.get('ARF_MAX_ITERS', '?')}",
                f"tr={clean.get('ARF_NUM_TREES', '?')}",
                f"min={clean.get('ARF_MIN_NODE_SIZE', '?')}",
                f"d={clean.get('ARF_DELTA', '?')}",
            ]
        )
    if model_id == "bayesnet":
        edge = (
            clean.get("BAYESNET_EDGE_WEIGHTS_FN", "?")
            .replace("mutual_info", "mi")
            .replace("adjusted_mutual_info", "ami")
            .replace("normalized_mutual_info", "nmi")
        )
        return ", ".join(
            [
                f"fit={clean.get('BAYESNET_FIT_ROWS', '?')}",
                f"struct={clean.get('BAYESNET_STRUCT_ROWS', '?')}",
                f"bins={clean.get('BAYESNET_MAX_BINS', '?')}",
                f"edge={edge}",
            ]
        )

    alias_map = {
        "EPOCHS": "ep",
        "MAX_EPOCHS": "ep",
        "BATCH_SIZE": "bs",
        "EMBEDDING_DIM": "emb",
        "NUM_SAMPLES": "ns",
        "SAMPLE_BATCH_SIZE": "samp",
        "STEPS": "steps",
    }
    items: list[str] = []
    for key, value in clean.items():
        token = key
        for prefix in ["CTGAN_", "TABPFGEN_", "TABDDPM_", "TABDIFF_", "FORESTDIFFUSION_", "REALTABFORMER_"]:
            token = token.replace(prefix, "")
        for old, new in alias_map.items():
            token = token.replace(old, new)
        token = token.lower()
        items.append(f"{token}={value}")
    return ", ".join(items[:4])


def _annotation_text(column_name: str, value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if column_name in {"train_sec", "generate_sec"}:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _column_color_values(series: pd.Series, direction: str) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    mask = np.isnan(values)
    valid = values[~mask]
    if valid.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    min_value = float(np.min(valid))
    max_value = float(np.max(valid))
    if max_value - min_value <= 1e-12:
        out = np.full_like(values, 0.5, dtype=float)
        out[mask] = np.nan
        return out
    normalized = (values - min_value) / (max_value - min_value)
    if direction == "lower_better":
        normalized = 1.0 - normalized
    normalized[mask] = np.nan
    return normalized


def build_hyper_merged_frame(project_root: Path, run_paths: HyperRunPaths) -> pd.DataFrame:
    analysis_asset = pd.read_csv(run_paths.analysis_run_dir / "summaries" / "analysis_asset_scores__all_datasets.csv")
    analysis_family = pd.concat(
        [pd.read_csv(path) for path in sorted((run_paths.analysis_run_dir / "datasets").glob("*" + "/analysis_family_scores__*.csv"))],
        ignore_index=True,
    )
    distance_asset = pd.read_csv(run_paths.distance_run_dir / "summaries" / "distance_summary__all_datasets.csv")
    validation_asset = None
    if run_paths.validation_run_dir is not None:
        validation_summary_path = run_paths.validation_run_dir / "summaries" / "validation_summary__all_datasets.csv"
        if validation_summary_path.exists():
            validation_asset = pd.read_csv(validation_summary_path)

    family_pivot = (
        analysis_family[["asset_key", "family_id", "family_score"]]
        .drop_duplicates()
        .pivot(index="asset_key", columns="family_id", values="family_score")
        .rename(columns=FAMILY_COLUMNS)
        .reset_index()
    )

    merged = (
        analysis_asset.rename(columns={"overall_score": "sql_overall_score"})
        .merge(family_pivot, on="asset_key", how="left")
        .merge(
            distance_asset[
                [
                    "asset_key",
                    "jensen_shannon_distance",
                    "total_variation_distance",
                    "kolmogorov_smirnov_distance",
                    "wasserstein_distance",
                    "overall_fidelity_score",
                ]
            ].rename(columns={"overall_fidelity_score": "distance_overall_score"}),
            on="asset_key",
            how="left",
        )
    )

    if validation_asset is not None:
        validation_columns = [column for column in ["asset_key", "cardinality_range_score", "missing_introduction_score"] if column in validation_asset.columns]
        merged = merged.merge(
            validation_asset[validation_columns].rename(
                columns={
                    "missing_introduction_score": "missing_score",
                    "cardinality_range_score": "cardinality_score",
                }
            ),
            on="asset_key",
            how="left",
            suffixes=("", "_validation"),
        )
        for metric_name in ["missing_score", "cardinality_score"]:
            validation_name = f"{metric_name}_validation"
            if validation_name in merged.columns:
                if metric_name not in merged.columns:
                    merged[metric_name] = merged[validation_name]
                else:
                    merged[metric_name] = merged[metric_name].fillna(merged[validation_name])
                merged = merged.drop(columns=[validation_name])

    for column_name in FAMILY_COLUMNS.values():
        if column_name not in merged.columns:
            merged[column_name] = np.nan

    runtime_rows: list[dict[str, Any]] = []
    for asset_dir_text in merged["asset_dir"].dropna().unique():
        asset_dir = Path(str(asset_dir_text))
        runtime_path = asset_dir / "runtime_result.json"
        run_config_path = asset_dir / "run_config.json"
        runtime_payload = _load_json(runtime_path) if runtime_path.exists() else {}
        run_config_payload = _load_json(run_config_path) if run_config_path.exists() else {}
        train_sec = float(((runtime_payload.get("timings") or {}).get("train") or {}).get("duration_sec") or 0.0)
        generate_sec = float(((runtime_payload.get("timings") or {}).get("generate") or {}).get("duration_sec") or 0.0)
        env_overrides = run_config_payload.get("env_overrides") if isinstance(run_config_payload, dict) else {}
        env_overrides = env_overrides if isinstance(env_overrides, dict) else {}
        runtime_rows.append(
            {
                "asset_dir": str(asset_dir),
                "train_sec": train_sec,
                "generate_sec": generate_sec,
                "total_sec": train_sec + generate_sec,
                "hyperparam_signature": json.dumps(env_overrides, ensure_ascii=False, sort_keys=True),
                "hyperparam_summary": _summarize_env_overrides(str(run_config_payload.get("model") or ""), env_overrides),
            }
        )
    runtime_df = pd.DataFrame(runtime_rows)
    merged = merged.merge(runtime_df, on="asset_dir", how="left")

    merged["dataset_sort"] = merged["dataset_id"].map({name: idx for idx, name in enumerate(DATASET_ORDER)}).fillna(999).astype(int)
    merged["train_rank_within_dataset_model"] = (
        merged.sort_values(["dataset_sort", "model_id", "train_sec", "generate_sec", "run_id"])
        .groupby(["dataset_id", "model_id"])["train_sec"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    merged["model_display"] = merged["model_id"].map(MODEL_DISPLAY).fillna(merged["model_id"])
    merged["row_label"] = merged.apply(
        lambda row: f"{str(row['dataset_id']).upper()} | t{int(row['train_rank_within_dataset_model'])} | {str(row['hyperparam_summary'])}",
        axis=1,
    )
    return merged


def _plot_model_card(model_id: str, model_df: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    plot_df = model_df.copy()
    plot_df = plot_df.sort_values(["dataset_sort", "train_sec", "generate_sec", "run_id"]).reset_index(drop=True)
    color_matrix = np.column_stack(
        [_column_color_values(plot_df[column_name], direction) for column_name, _, direction in METRIC_SPECS]
    )
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="white")

    fig_height = max(4.8, 0.55 * len(plot_df) + 1.8)
    fig_width = 20.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(color_matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(METRIC_SPECS)))
    ax.set_xticklabels([label for _, label, _ in METRIC_SPECS], rotation=35, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["row_label"].tolist(), fontsize=8.2)
    ax.set_title(f"{MODEL_DISPLAY.get(model_id, model_id)} hyperparameter runs", fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Time / Query / Distance metrics", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dataset + train-time rank + hyperparameter setting", fontsize=11, fontweight="bold")

    for row_idx in range(len(plot_df) + 1):
        ax.axhline(row_idx - 0.5, color="#d9d9d9", linewidth=0.7, zorder=3)
    for col_idx in range(len(METRIC_SPECS) + 1):
        ax.axvline(col_idx - 0.5, color="#d9d9d9", linewidth=0.7, zorder=3)

    for row_idx in range(len(plot_df)):
        for col_idx, (column_name, _, _) in enumerate(METRIC_SPECS):
            raw_value = plot_df.iloc[row_idx][column_name]
            text = _annotation_text(column_name, raw_value)
            if not text:
                continue
            color_value = color_matrix[row_idx, col_idx]
            text_color = "white" if not np.isnan(color_value) and color_value >= 0.6 else "#1f1f1f"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8.7, color=text_color)

    fig.subplots_adjust(left=0.33, right=0.985, top=0.92, bottom=0.14)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _build_contact_sheet(image_paths: list[Path], output_png: Path, output_pdf: Path) -> None:
    cols = 3
    rows = math.ceil(len(image_paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes_arr = np.atleast_2d(axes)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, image_path in zip(axes_arr.ravel(), image_paths):
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.set_title(image_path.stem.replace("hyper_model_card__", ""), fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_model_metric_cards(project_root: Path, output_root: Path, run_paths: HyperRunPaths | None = None) -> dict[str, Any]:
    resolved_runs = run_paths or resolve_latest_hyper_runs(project_root)
    merged = build_hyper_merged_frame(project_root, resolved_runs)

    output_root.mkdir(parents=True, exist_ok=True)
    merged_csv = output_root / "hyper_run_metric_matrix.csv"
    merged.to_csv(merged_csv, index=False)

    per_model_dir = output_root / "model_cards"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    generated_images: list[Path] = []
    summary_rows: list[dict[str, Any]] = []

    for model_id in MODEL_ORDER:
        model_df = merged[merged["model_id"] == model_id].copy()
        if model_df.empty:
            continue
        model_csv = per_model_dir / f"hyper_model_card__{model_id}.csv"
        model_png = per_model_dir / f"hyper_model_card__{model_id}.png"
        model_pdf = per_model_dir / f"hyper_model_card__{model_id}.pdf"
        model_df.to_csv(model_csv, index=False)
        _plot_model_card(model_id, model_df, model_png, model_pdf)
        generated_images.append(model_png)
        summary_rows.append(
            {
                "model_id": model_id,
                "model_display": MODEL_DISPLAY.get(model_id, model_id),
                "row_count": len(model_df),
                "datasets": ",".join(sorted(model_df["dataset_id"].astype(str).unique())),
                "png_path": str(model_png),
                "pdf_path": str(model_pdf),
                "csv_path": str(model_csv),
            }
        )

    summary_csv = output_root / "hyper_model_card_index.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    contact_png = output_root / "hyper_model_cards_contact_sheet.png"
    contact_pdf = output_root / "hyper_model_cards_contact_sheet.pdf"
    _build_contact_sheet(generated_images, contact_png, contact_pdf)

    summary_md = output_root / "hyper_model_cards_summary.md"
    lines = [
        "# Hyperparameter Model Cards",
        "",
        f"- Analysis run: `{resolved_runs.analysis_run_dir}`",
        f"- Distance run: `{resolved_runs.distance_run_dir}`",
        f"- Validation run: `{resolved_runs.validation_run_dir}`",
        f"- Merged table: `{merged_csv}`",
        f"- Contact sheet: `{contact_png}`",
        "",
        "Per-model outputs:",
    ]
    for row in summary_rows:
        lines.append(
            f"- `{row['model_display']}`: rows={row['row_count']} | csv=`{row['csv_path']}` | png=`{row['png_path']}`"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "analysis_run_dir": str(resolved_runs.analysis_run_dir),
        "distance_run_dir": str(resolved_runs.distance_run_dir),
        "merged_csv": str(merged_csv),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "contact_png": str(contact_png),
        "contact_pdf": str(contact_pdf),
        "model_count": len(summary_rows),
        "row_count": int(len(merged)),
    }
