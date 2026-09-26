from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINONLY_ROOT = PROJECT_ROOT / "remote-output-Benchmark-trainonly-v1"
STATUS_ROOT = TRAINONLY_ROOT / "_status"
HYPERPARAM_AUDIT_CSV = STATUS_ROOT / "benchmark_hyperparams_audit.csv"
TRAINONLY_STATUS_CSV = STATUS_ROOT / "benchmark_trainonly_status.csv"
OUTPUT_DIR = PROJECT_ROOT.parent / "results" / "time_cost" / "final"

TARGET_DATASETS = ["c2", "c7", "c14", "m4", "m6", "m8", "n3", "n6", "n11"]
TARGET_MODELS = [
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_hms_to_seconds(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text or text == "-":
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
    except ValueError:
        return None
    return float(hours * 3600 + minutes * 60 + seconds)


def _format_total_hms(train_hms: str | None, generate_hms: str | None) -> str:
    total_seconds = (_parse_hms_to_seconds(train_hms) or 0.0) + (_parse_hms_to_seconds(generate_hms) or 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_iso_datetime(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _seconds_from_status_row(row: dict[str, str], prefix: str) -> float | None:
    direct = (row.get(f"{prefix}_duration_sec") or "").strip()
    if direct:
        try:
            return float(direct)
        except ValueError:
            pass
    start = _parse_iso_datetime(row.get(f"{prefix}_started_at"))
    end = _parse_iso_datetime(row.get(f"{prefix}_ended_at"))
    if start is not None and end is not None:
        return max(0.0, round((end - start).total_seconds(), 3))
    return None


def _format_seconds_to_hms(value: float | None) -> str:
    if value is None:
        return ""
    total_seconds = int(round(float(value)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _status_priority(row: dict[str, str]) -> tuple[int, int, datetime]:
    overall = (row.get("overall_status") or "").strip().lower()
    train_status = (row.get("train_status") or "").strip().lower()
    generate_status = (row.get("generate_status") or "").strip().lower()

    if overall == "success" and train_status == "success" and generate_status == "success":
        bucket = 5
    elif overall == "train_only_success" or (train_status == "success" and generate_status == "fail"):
        bucket = 4
    elif overall == "partial":
        bucket = 3
    elif overall == "in_progress":
        bucket = 2
    else:
        bucket = 1

    richness = 0
    for prefix in ("train", "generate"):
        if _seconds_from_status_row(row, prefix) is not None:
            richness += 1

    updated_at = _parse_iso_datetime(row.get("last_updated_at")) or datetime.min
    return (bucket, richness, updated_at)


def _dataset_sort_key(dataset_id: str) -> tuple[str, int]:
    text = (dataset_id or "").strip().lower()
    return text[:1], int(text[1:] or "0")


def build_subset_rows() -> list[dict[str, Any]]:
    audit_rows = _read_csv_rows(HYPERPARAM_AUDIT_CSV)
    status_rows = _read_csv_rows(TRAINONLY_STATUS_CSV)

    audit_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in audit_rows:
        dataset_id = (row.get("dataset") or "").strip().lower()
        model_id = (row.get("model") or "").strip().lower()
        if dataset_id in TARGET_DATASETS and model_id in TARGET_MODELS:
            audit_by_key[(dataset_id, model_id)] = row

    status_by_run: dict[tuple[str, str, str], dict[str, str]] = {}
    best_status_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in status_rows:
        dataset_id = (row.get("dataset_id") or "").strip().lower()
        model_id = (row.get("model_id") or "").strip().lower()
        run_id = (row.get("run_id") or "").strip()
        key = (dataset_id, model_id, run_id)
        status_by_run[key] = row
        if dataset_id in TARGET_DATASETS and model_id in TARGET_MODELS:
            current = best_status_by_key.get((dataset_id, model_id))
            if current is None or _status_priority(row) > _status_priority(current):
                best_status_by_key[(dataset_id, model_id)] = row

    output_rows: list[dict[str, Any]] = []
    for dataset_id in TARGET_DATASETS:
        for model_id in TARGET_MODELS:
            audit_row = audit_by_key.get((dataset_id, model_id))
            selected_status = best_status_by_key.get((dataset_id, model_id), {})

            if audit_row is not None:
                run_id = (audit_row.get("run_id") or "").strip()
                train_status = (audit_row.get("train_status") or "").strip()
                generate_status = (audit_row.get("generate_status") or "").strip()
                train_hms = (audit_row.get("train_duration_hms") or "").strip()
                generate_hms = (audit_row.get("generate_duration_hms") or "").strip()
                hyperparam_source = (audit_row.get("hyperparam_source") or "").strip()
                train_hyperparams_json = (audit_row.get("train_hyperparams") or "").strip()
                generate_hyperparams_json = (audit_row.get("generate_hyperparams") or "").strip()
                selection_strategy = "hyperparams_audit_preferred"
                status_match = status_by_run.get((dataset_id, model_id, run_id), {})
            elif selected_status:
                run_id = (selected_status.get("run_id") or "").strip()
                train_status = (selected_status.get("train_status") or "").strip()
                generate_status = (selected_status.get("generate_status") or "").strip()
                train_seconds_from_status = _seconds_from_status_row(selected_status, "train")
                generate_seconds_from_status = _seconds_from_status_row(selected_status, "generate")
                train_hms = _format_seconds_to_hms(train_seconds_from_status)
                generate_hms = _format_seconds_to_hms(generate_seconds_from_status)
                hyperparam_source = ""
                train_hyperparams_json = ""
                generate_hyperparams_json = ""
                selection_strategy = "status_best_available"
                status_match = selected_status
            else:
                continue

            train_seconds = _parse_hms_to_seconds(train_hms)
            generate_seconds = _parse_hms_to_seconds(generate_hms)
            if train_seconds is None:
                train_seconds = _seconds_from_status_row(status_match, "train")
                train_hms = _format_seconds_to_hms(train_seconds)
            if generate_seconds is None:
                generate_seconds = _seconds_from_status_row(status_match, "generate")
                generate_hms = _format_seconds_to_hms(generate_seconds)

            total_seconds = (train_seconds or 0.0) + (generate_seconds or 0.0)
            run_dir = TRAINONLY_ROOT / dataset_id / model_id / run_id

            output_rows.append(
                {
                    "dataset_id": dataset_id,
                    "model_id": model_id,
                    "run_id": run_id,
                    "subset_tag": "trainonly_serial_timecost_9datasets",
                    "recording_mode": "single_run_serial",
                    "source_root": str(TRAINONLY_ROOT.resolve()),
                    "selection_source_csv": str(HYPERPARAM_AUDIT_CSV.resolve()),
                    "detail_source_csv": str(TRAINONLY_STATUS_CSV.resolve()),
                    "selection_strategy": selection_strategy,
                    "train_status": train_status,
                    "generate_status": generate_status,
                    "complete_success": (
                        train_status == "success"
                        and generate_status == "success"
                    ),
                    "train_duration_hms": train_hms,
                    "generate_duration_hms": generate_hms,
                    "total_duration_hms": _format_total_hms(train_hms, generate_hms),
                    "train_duration_sec": train_seconds,
                    "generate_duration_sec": generate_seconds,
                    "total_duration_sec": total_seconds if train_seconds is not None or generate_seconds is not None else None,
                    "train_started_at": (status_match.get("train_started_at") or "").strip(),
                    "train_ended_at": (status_match.get("train_ended_at") or "").strip(),
                    "generate_started_at": (status_match.get("generate_started_at") or "").strip(),
                    "generate_ended_at": (status_match.get("generate_ended_at") or "").strip(),
                    "train_rows": (status_match.get("train_rows") or "").strip(),
                    "generate_rows": (status_match.get("generate_rows") or "").strip(),
                    "synthetic_csv": (status_match.get("synthetic_csv") or "").strip(),
                    "runtime_result_exists_in_status": (status_match.get("runtime_result_exists") or "").strip(),
                    "runtime_result_path": str((run_dir / "runtime_result.json").resolve()),
                    "run_dir": str(run_dir.resolve()),
                    "status_row_matched_exact_run_id": bool(status_match),
                    "hyperparam_source": hyperparam_source,
                    "train_hyperparams_json": train_hyperparams_json,
                    "generate_hyperparams_json": generate_hyperparams_json,
                }
            )

    output_rows.sort(key=lambda item: (_dataset_sort_key(str(item["dataset_id"])), str(item["model_id"])))
    return output_rows


def build_outputs() -> dict[str, Path]:
    rows = build_subset_rows()
    csv_path = OUTPUT_DIR / "trainonly_serial_timecost_subset_9datasets.csv"
    success_csv_path = OUTPUT_DIR / "trainonly_serial_timecost_subset_9datasets_success_only.csv"
    summary_path = OUTPUT_DIR / "trainonly_serial_timecost_subset_9datasets_summary.md"

    fieldnames = list(rows[0].keys()) if rows else []
    _write_csv(csv_path, rows, fieldnames)
    _write_csv(success_csv_path, [row for row in rows if row["complete_success"]], fieldnames)

    dataset_count = len({str(row["dataset_id"]) for row in rows})
    model_count = len({str(row["model_id"]) for row in rows})
    success_count = sum(1 for row in rows if row["complete_success"])
    total_count = len(rows)
    lines = [
        "# Train-only Serial Time-Cost Subset",
        "",
        f"- Source root: `{TRAINONLY_ROOT.as_posix()}`",
        f"- Representative run selector: `{HYPERPARAM_AUDIT_CSV.as_posix()}`",
        f"- Detailed status table: `{TRAINONLY_STATUS_CSV.as_posix()}`",
        f"- Target datasets: `{', '.join(TARGET_DATASETS)}`",
        f"- Total rows exported: `{total_count}`",
        f"- Complete success rows: `{success_count}`",
        f"- Dataset count: `{dataset_count}`",
        f"- Model count: `{model_count}`",
        "",
        "Notes:",
        "- This subset is intended for the serial time-cost analysis only.",
        "- `train_duration_sec` and `generate_duration_sec` are derived from the audited `HH:MM:SS` fields.",
        "- `status_row_matched_exact_run_id=false` means the representative run exists in the hyperparameter audit, but the older flat status CSV did not include the same run id locally.",
        "",
        f"- Full CSV: `{csv_path.as_posix()}`",
        f"- Success-only CSV: `{success_csv_path.as_posix()}`",
    ]
    _write_text(summary_path, "\n".join(lines) + "\n")
    return {"csv": csv_path, "success_csv": success_csv_path, "summary": summary_path}


def main() -> None:
    outputs = build_outputs()
    print(json.dumps({key: str(path) for key, path in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
