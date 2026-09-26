from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
REMOTE_ROOT = REPO_ROOT / "remote-output-Benchmark-trainonly-v1"
OUT_DIR = REPO_ROOT.parent / "results" / "time_cost" / "timecost0506"

TARGET_DATASETS = ["c2", "c7", "c14", "m4", "m6", "m8", "n3", "n6", "n11"]
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
MODEL_LABELS = {
    "arf": "ARF",
    "bayesnet": "BayesNet",
    "ctgan": "CTGAN",
    "forestdiffusion": "ForestDiffusion",
    "realtabformer": "RealTabFormer",
    "tabbyflow": "TabbyFlow",
    "tabddpm": "TabDDPM",
    "tabdiff": "TabDiff",
    "tabpfgen": "TabPFGen",
    "tabsyn": "TabSyn",
    "tvae": "TVAE",
}


@dataclass
class RunRecord:
    dataset_id: str
    model_id: str
    model_label: str
    run_id: str
    runtime_result_path: str
    public_gate_status: str | None
    adapter_ready_status: str | None
    train_status: str | None
    generate_status: str | None
    train_started_at: str | None
    train_ended_at: str | None
    train_duration_sec: float | None
    generate_started_at: str | None
    generate_ended_at: str | None
    generate_duration_sec: float | None
    synthetic_csv_path: str | None
    model_path: str | None


def parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_timing_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    result: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("started_at_utc:"):
            result["started_at"] = line.split(":", 1)[1].strip().replace("Z", "+00:00")
        elif line.startswith("finished_at_utc:"):
            result["ended_at"] = line.split(":", 1)[1].strip().replace("Z", "+00:00")
        elif line.startswith("elapsed_seconds:"):
            value = line.split(":", 1)[1].strip()
            try:
                result["duration_sec"] = float(value)
            except ValueError:
                pass
    return result


def scan_runs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id in TARGET_DATASETS:
        for model_id in MODEL_ORDER:
            model_dir = REMOTE_ROOT / dataset_id / model_id
            if not model_dir.exists():
                continue
            runtime_paths = list(model_dir.glob("runtime_result.json")) + list(model_dir.glob("*/runtime_result.json"))
            for runtime_path in runtime_paths:
                payload = read_json(runtime_path)
                timings = payload.get("timings", {})
                train = timings.get("train", {}) or {}
                generate = timings.get("generate", {}) or {}
                run_dir = runtime_path.parent
                if not train.get("duration_sec"):
                    train_logs = sorted(run_dir.rglob("train_*.log"))
                    if train_logs:
                        train = {**parse_timing_log(train_logs[0]), **train}
                if not generate.get("duration_sec"):
                    gen_logs = sorted(run_dir.rglob("gen_*.log"))
                    if gen_logs:
                        generate = {**parse_timing_log(gen_logs[0]), **generate}
                artifacts = payload.get("artifacts", {}) or {}
                record = RunRecord(
                    dataset_id=payload.get("dataset_id", dataset_id),
                    model_id=payload.get("model", model_id),
                    model_label=MODEL_LABELS.get(payload.get("model", model_id), payload.get("model", model_id)),
                    run_id=payload.get("run_id", runtime_path.parent.name),
                    runtime_result_path=str(runtime_path),
                    public_gate_status=payload.get("public_gate_status"),
                    adapter_ready_status=payload.get("adapter_ready_status"),
                    train_status=payload.get("train_status"),
                    generate_status=payload.get("generate_status"),
                    train_started_at=train.get("started_at"),
                    train_ended_at=train.get("ended_at"),
                    train_duration_sec=train.get("duration_sec"),
                    generate_started_at=generate.get("started_at"),
                    generate_ended_at=generate.get("ended_at"),
                    generate_duration_sec=generate.get("duration_sec"),
                    synthetic_csv_path=artifacts.get("synthetic_csv"),
                    model_path=artifacts.get("model_path"),
                )
                rows.append(record.__dict__)
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("No runtime_result.json files found for the target datasets.")
    return df


def select_runs(all_runs: pd.DataFrame) -> pd.DataFrame:
    df = all_runs.copy()
    df["train_ended_dt"] = df["train_ended_at"].map(parse_iso)
    df["train_started_dt"] = df["train_started_at"].map(parse_iso)
    df["generate_ended_dt"] = df["generate_ended_at"].map(parse_iso)
    df["is_success"] = (
        df["public_gate_status"].eq("pass")
        & df["adapter_ready_status"].eq("pass")
        & df["train_status"].isin(["success", "skipped"])
        & df["generate_status"].eq("success")
        & df["train_duration_sec"].notna()
        & df["generate_duration_sec"].notna()
    )
    success = df[df["is_success"]].copy()
    if success.empty:
        raise RuntimeError("No successful train/generate runs found for the target datasets.")
    success = success.sort_values(
        ["dataset_id", "model_id", "train_ended_dt", "train_started_dt", "run_id"],
        ascending=[True, True, False, False, False],
    )
    selected = success.drop_duplicates(["dataset_id", "model_id"], keep="first").copy()

    missing_pairs = [
        (dataset_id, model_id)
        for dataset_id in TARGET_DATASETS
        for model_id in MODEL_ORDER
        if not ((selected["dataset_id"] == dataset_id) & (selected["model_id"] == model_id)).any()
    ]
    paired_rows: list[pd.Series] = []
    for dataset_id, model_id in missing_pairs:
        candidates = df[
            (df["dataset_id"] == dataset_id)
            & (df["model_id"] == model_id)
            & df["public_gate_status"].eq("pass")
            & df["adapter_ready_status"].eq("pass")
        ].copy()
        if candidates.empty:
            continue
        train_candidates = candidates[candidates["train_duration_sec"].notna()].sort_values(
            ["train_ended_dt", "train_started_dt", "run_id"], ascending=[False, False, False]
        )
        gen_candidates = candidates[
            candidates["generate_duration_sec"].notna() & candidates["generate_status"].eq("success")
        ].sort_values(["generate_ended_dt", "run_id"], ascending=[False, False])
        if train_candidates.empty or gen_candidates.empty:
            continue
        train_row = train_candidates.iloc[0]
        gen_row = gen_candidates.iloc[0]
        combined = gen_row.copy()
        combined["run_id"] = f"paired::{train_row['run_id']}+{gen_row['run_id']}"
        combined["runtime_result_path"] = (
            f"{train_row['runtime_result_path']} | {gen_row['runtime_result_path']}"
        )
        combined["train_status"] = train_row["train_status"]
        combined["train_started_at"] = train_row["train_started_at"]
        combined["train_ended_at"] = train_row["train_ended_at"]
        combined["train_duration_sec"] = train_row["train_duration_sec"]
        combined["train_started_dt"] = train_row["train_started_dt"]
        combined["train_ended_dt"] = train_row["train_ended_dt"]
        paired_rows.append(combined)

    if paired_rows:
        paired_df = pd.DataFrame(paired_rows)
        selected = pd.concat([selected, paired_df], ignore_index=True)
        selected = selected.sort_values(
            ["dataset_id", "model_id", "train_ended_dt", "train_started_dt", "run_id"],
            ascending=[True, True, False, False, False],
        ).drop_duplicates(["dataset_id", "model_id"], keep="first")

    selected["dataset_prefix"] = selected["dataset_id"].str[0]
    selected["train_duration_min"] = selected["train_duration_sec"] / 60.0
    selected["generate_duration_min"] = selected["generate_duration_sec"] / 60.0
    return selected


def build_model_summary(selected: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        selected.groupby(["model_id", "model_label"], as_index=False)
        .agg(
            dataset_count=("dataset_id", "nunique"),
            dataset_list=("dataset_id", lambda s: ",".join(sorted(s))),
            train_time_mean_sec=("train_duration_sec", "mean"),
            train_time_std_sec=("train_duration_sec", "std"),
            train_time_min_sec=("train_duration_sec", "min"),
            train_time_max_sec=("train_duration_sec", "max"),
            generation_time_mean_sec=("generate_duration_sec", "mean"),
            generation_time_std_sec=("generate_duration_sec", "std"),
            generation_time_min_sec=("generate_duration_sec", "min"),
            generation_time_max_sec=("generate_duration_sec", "max"),
        )
    )
    grouped["train_time_std_sec"] = grouped["train_time_std_sec"].fillna(0.0)
    grouped["generation_time_std_sec"] = grouped["generation_time_std_sec"].fillna(0.0)
    for col in [
        "train_time_mean_sec",
        "train_time_std_sec",
        "train_time_min_sec",
        "train_time_max_sec",
        "generation_time_mean_sec",
        "generation_time_std_sec",
        "generation_time_min_sec",
        "generation_time_max_sec",
    ]:
        grouped[col.replace("_sec", "_min")] = grouped[col] / 60.0
    grouped["model_order"] = grouped["model_id"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return grouped.sort_values("model_order").reset_index(drop=True)


def build_dataset_summary(selected: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        selected.groupby("dataset_id", as_index=False)
        .agg(
            model_count=("model_id", "nunique"),
            train_time_mean_sec=("train_duration_sec", "mean"),
            generation_time_mean_sec=("generate_duration_sec", "mean"),
        )
    )
    grouped["train_time_mean_min"] = grouped["train_time_mean_sec"] / 60.0
    grouped["generation_time_mean_min"] = grouped["generation_time_mean_sec"] / 60.0
    return grouped.sort_values("dataset_id").reset_index(drop=True)


def build_coverage_matrix(selected: pd.DataFrame) -> pd.DataFrame:
    selected_flag = selected.assign(selected_flag=1)
    matrix = (
        selected_flag.pivot_table(
            index="dataset_id",
            columns="model_id",
            values="selected_flag",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(index=TARGET_DATASETS, columns=MODEL_ORDER, fill_value=0)
        .reset_index()
    )
    return matrix


def render_markdown(model_summary: pd.DataFrame, dataset_summary: pd.DataFrame) -> str:
    lines = [
        "# Train-only v1 time-cost summary (timecost0506)",
        "",
        f"Target datasets: {', '.join(TARGET_DATASETS)}",
        "Selection rule: latest successful run per dataset-model pair with pass/pass/success/success statuses.",
        "",
        "## Model averages",
        "",
        "| Model | Datasets | Mean train (min) | Mean gen (min) | Mean train (sec) | Mean gen (sec) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in model_summary.iterrows():
        lines.append(
            f"| {row['model_label']} | {int(row['dataset_count'])} | "
            f"{row['train_time_mean_min']:.2f} | {row['generation_time_mean_min']:.2f} | "
            f"{row['train_time_mean_sec']:.1f} | {row['generation_time_mean_sec']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset averages",
            "",
            "| Dataset | Models | Mean train (min) | Mean gen (min) |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in dataset_summary.iterrows():
        lines.append(
            f"| {row['dataset_id']} | {int(row['model_count'])} | "
            f"{row['train_time_mean_min']:.2f} | {row['generation_time_mean_min']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_runs = scan_runs()
    selected = select_runs(all_runs)
    model_summary = build_model_summary(selected)
    dataset_summary = build_dataset_summary(selected)
    coverage = build_coverage_matrix(selected)

    all_runs.sort_values(["dataset_id", "model_id", "run_id"]).to_csv(
        OUT_DIR / "all_candidate_runs.csv", index=False
    )
    selected.sort_values(["dataset_id", "model_id"]).to_csv(
        OUT_DIR / "selected_dataset_model_runs.csv", index=False
    )
    model_summary.to_csv(OUT_DIR / "model_time_summary.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "dataset_time_summary.csv", index=False)
    coverage.to_csv(OUT_DIR / "selection_coverage_matrix.csv", index=False)
    (OUT_DIR / "time_cost_summary.md").write_text(
        render_markdown(model_summary, dataset_summary), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
