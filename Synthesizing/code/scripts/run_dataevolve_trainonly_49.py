#!/usr/bin/env python3
"""Run DataEvolve as a train-only synthetic-data generator for the 49 datasets.

The local benchmark datasets live as:

    data_train_as_main/<dataset_id>/<dataset_id>-train.csv

DataEvolve expects a dataset directory containing one CSV per table. For our
single-table benchmark, this runner stages each train split as:

    <out-root>/_inputs/<dataset_id>/<dataset_id>.csv

and writes DataEvolve outputs to:

    <out-root>/runs/<dataset_id>/synthetic/<dataset_id>.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data_train_as_main"
DEFAULT_DATAEVOLVE_ROOT = ROOT / "DataEvolve"
DEFAULT_OUT_ROOT = DEFAULT_DATAEVOLVE_ROOT / "workspace" / "tabquerybench_49_dataevolve_off"
DEFAULT_PUBLISH_ROOT = ROOT / "SynOutput-5090"
MODEL_ID = "dataevolve"
SUPPORTED_BACKENDS = {"python", "native"}
SUPPORTED_BUDGETS = {"compact", "balanced", "expanded", "auto"}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.worker_dataset:
        return _run_worker(args)
    return _run_parent(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataevolve-root", type=Path, default=DEFAULT_DATAEVOLVE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--publish-root", type=Path, default=DEFAULT_PUBLISH_ROOT)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset ids. Default: all discovered datasets.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N dataset limit for smoke runs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=sorted(SUPPORTED_BACKENDS), default="python")
    parser.add_argument("--parameter-budget", choices=sorted(SUPPORTED_BUDGETS), default="compact")
    parser.add_argument("--copy-inputs", action="store_true", help="Copy train CSVs instead of symlinking them.")
    parser.add_argument("--no-publish", action="store_true", help="Do not publish outputs into SynOutput-style layout.")
    parser.add_argument("--fresh", action="store_true", help="Delete existing per-dataset run directories before running.")
    parser.add_argument("--resume", action="store_true", help="Skip datasets with an existing synthetic CSV.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a dataset failure.")
    parser.add_argument("--timeout-sec", type=float, default=0.0, help="Per-dataset timeout. <=0 disables timeout.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-dataset", default="", help=argparse.SUPPRESS)
    parser.add_argument("--input-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default="", help=argparse.SUPPRESS)
    return parser


def _run_parent(args: argparse.Namespace) -> int:
    data_root = args.data_root.resolve()
    dataevolve_root = args.dataevolve_root.resolve()
    out_root = args.out_root.resolve()
    publish_root = args.publish_root.resolve()
    datasets = _select_datasets(data_root, args.datasets, args.limit)
    if not datasets:
        raise SystemExit(f"No datasets found under {data_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures = 0
    print(f"[dataevolve-49] datasets={len(datasets)} backend={args.backend} budget={args.parameter_budget}")
    print(f"[dataevolve-49] out_root={out_root}")

    for dataset_id in datasets:
        train_csv = data_root / dataset_id / f"{dataset_id}-train.csv"
        input_dir = out_root / "_inputs" / dataset_id
        run_dir = out_root / "runs" / dataset_id
        run_id = f"{MODEL_ID}-{dataset_id}-{_timestamp()}"
        synthetic_csv = run_dir / "synthetic" / f"{dataset_id}.csv"
        if args.resume and synthetic_csv.is_file():
            record = _summarize_existing(dataset_id, train_csv, synthetic_csv, run_dir)
            record["status"] = "skipped_existing"
            records.append(record)
            print(f"[dataevolve-49] skip existing {dataset_id}: {synthetic_csv}")
            continue

        if args.fresh and run_dir.exists():
            shutil.rmtree(run_dir)
        _stage_input(train_csv=train_csv, input_dir=input_dir, dataset_id=dataset_id, copy_inputs=bool(args.copy_inputs))

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-dataset",
            dataset_id,
            "--input-dir",
            str(input_dir),
            "--run-dir",
            str(run_dir),
            "--run-id",
            run_id,
            "--dataevolve-root",
            str(dataevolve_root),
            "--seed",
            str(int(args.seed)),
            "--backend",
            str(args.backend),
            "--parameter-budget",
            str(args.parameter_budget),
        ]
        if args.dry_run:
            print("[dataevolve-49] dry-run", " ".join(command))
            records.append(
                {
                    "dataset_id": dataset_id,
                    "status": "dry_run",
                    "train_csv": str(train_csv),
                    "input_dir": str(input_dir),
                    "run_dir": str(run_dir),
                    "synthetic_csv": str(synthetic_csv),
                }
            )
            continue

        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=(float(args.timeout_sec) if float(args.timeout_sec) > 0 else None),
            check=False,
        )
        elapsed = float(time.perf_counter() - started)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        status = "success" if completed.returncode == 0 and synthetic_csv.is_file() else "failed"
        record = _summarize_existing(dataset_id, train_csv, synthetic_csv, run_dir)
        record.update(
            {
                "status": status,
                "returncode": completed.returncode,
                "elapsed_sec_parent": elapsed,
            }
        )
        records.append(record)
        _write_summary(out_root, records)
        if status != "success":
            failures += 1
            if not args.keep_going:
                return completed.returncode or 1
        else:
            if not bool(args.no_publish):
                publish_record = _publish_synoutput_asset(
                    dataset_id=dataset_id,
                    run_id=run_id,
                    data_root=data_root,
                    publish_root=publish_root,
                    run_dir=run_dir,
                    synthetic_csv=synthetic_csv,
                    backend=str(args.backend),
                    parameter_budget=str(args.parameter_budget),
                    seed=int(args.seed),
                )
                record.update(publish_record)
                _write_summary(out_root, records)
            print(f"[dataevolve-49] done {dataset_id}: rows={record.get('synthetic_rows')} sec={elapsed:.2f}")

    _write_summary(out_root, records)
    print(f"[dataevolve-49] completed={len(records) - failures} failed={failures}")
    return 1 if failures else 0


def _run_worker(args: argparse.Namespace) -> int:
    dataset_id = str(args.worker_dataset)
    run_id = str(args.run_id or f"{MODEL_ID}-{dataset_id}-{_timestamp()}")
    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    dataevolve_root = Path(args.dataevolve_root).resolve()
    os.environ["DATAEVOLVE_FAST_TRAINONLY"] = "1"
    sys.path.insert(0, str(dataevolve_root / "src"))

    from dataevolve.multitable import export_evolving_engine, generate_from_evolving_engine

    run_dir.mkdir(parents=True, exist_ok=True)
    engine_dir = run_dir / "engine"
    started = time.perf_counter()
    export_result = export_evolving_engine(
        dataset_dir=input_dir,
        engine_dir=engine_dir,
        seed=int(args.seed),
        augmentation_mode="off",
        augmentation_strength=0.5,
        synthetic_mix_ratio_cap=1.0,
        parameter_budget=str(args.parameter_budget),
        backend=str(args.backend),
        native_artifact_mode=("minimal" if str(args.backend) == "native" else "compatibility"),
    )
    exported_sec = float(time.perf_counter() - started)
    generation_started = time.perf_counter()
    generation_result = generate_from_evolving_engine(
        engine_params_path=export_result["engine_params_path"],
        output_dir=run_dir,
        seed=int(args.seed),
        backend=str(args.backend),
        native_runtime_mode=("performance" if str(args.backend) == "native" else "parity"),
    )
    generated_sec = float(time.perf_counter() - generation_started)
    payload = {
        "dataset_id": dataset_id,
        "status": "success",
        "run_id": run_id,
        "seed": int(args.seed),
        "backend": str(args.backend),
        "augmentation_mode": "off",
        "parameter_budget": str(args.parameter_budget),
        "input_dir": str(input_dir),
        "run_dir": str(run_dir),
        "synthetic_dir": str(run_dir / "synthetic"),
        "engine_params_path": export_result.get("engine_params_path"),
        "table_order": export_result.get("table_order", []),
        "timing_sec": {
            "export_sec": exported_sec,
            "generate_sec": generated_sec,
            "total_sec": float(time.perf_counter() - started),
        },
        "generation_result_keys": sorted(generation_result.keys()),
    }
    (run_dir / "dataevolve_generation_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _select_datasets(data_root: Path, dataset_csv: str, limit: int | None) -> list[str]:
    if dataset_csv.strip():
        selected = [item.strip() for item in dataset_csv.split(",") if item.strip()]
    else:
        selected = [
            path.name
            for path in sorted(data_root.iterdir(), key=lambda p: _dataset_sort_key(p.name))
            if path.is_dir() and (path / f"{path.name}-train.csv").is_file()
        ]
    if limit is not None:
        selected = selected[: int(limit)]
    missing = [dataset_id for dataset_id in selected if not (data_root / dataset_id / f"{dataset_id}-train.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing train CSVs for datasets: {', '.join(missing)}")
    return selected


def _dataset_sort_key(name: str) -> tuple[str, int, str]:
    prefix = name[:1]
    suffix = name[1:]
    return (prefix, int(suffix) if suffix.isdigit() else 10**9, name)


def _stage_input(*, train_csv: Path, input_dir: Path, dataset_id: str, copy_inputs: bool) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    staged_csv = input_dir / f"{dataset_id}.csv"
    if staged_csv.exists() or staged_csv.is_symlink():
        staged_csv.unlink()
    if copy_inputs:
        shutil.copy2(train_csv, staged_csv)
        return
    try:
        os.symlink(train_csv.resolve(), staged_csv)
    except OSError:
        shutil.copy2(train_csv, staged_csv)


def _publish_synoutput_asset(
    *,
    dataset_id: str,
    run_id: str,
    data_root: Path,
    publish_root: Path,
    run_dir: Path,
    synthetic_csv: Path,
    backend: str,
    parameter_budget: str,
    seed: int,
) -> dict[str, Any]:
    dataset_dir = data_root / dataset_id
    train_csv = dataset_dir / f"{dataset_id}-train.csv"
    val_csv = dataset_dir / f"{dataset_id}-val.csv"
    test_csv = dataset_dir / f"{dataset_id}-test.csv"
    main_csv = dataset_dir / f"{dataset_id}-main.csv"
    row_count = _count_csv_rows(train_csv)
    generated_stamp = _timestamp()

    model_dir = publish_root / dataset_id / MODEL_ID
    synthetic_dir = model_dir / "synthetic_data"
    metadata_dir = model_dir / "metadata"
    logs_dir = model_dir / "logs"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{dataset_id}__{MODEL_ID}__{run_id}__"
    published_synthetic = synthetic_dir / f"{prefix}{MODEL_ID}-{dataset_id}-{row_count}-{generated_stamp}.csv"
    shutil.copy2(synthetic_csv, published_synthetic)
    _link_or_copy(train_csv, synthetic_dir / f"{prefix}train.csv")
    _link_or_copy(val_csv, synthetic_dir / f"{prefix}val.csv")
    _link_or_copy(test_csv, synthetic_dir / f"{prefix}test.csv")
    _link_or_copy(main_csv, synthetic_dir / f"{prefix}real.csv")

    generation_result_path = run_dir / "dataevolve_generation_result.json"
    generation_result = _read_json_or_empty(generation_result_path)
    runtime_result = {
        "dataset_id": dataset_id,
        "model": MODEL_ID,
        "run_id": run_id,
        "public_gate_status": "pass",
        "adapter_ready_status": "pass",
        "train_status": "success",
        "generate_status": "success",
        "reason_code": None,
        "reason_detail": None,
        "backend": backend,
        "augmentation_mode": "off",
        "parameter_budget": parameter_budget,
        "seed": seed,
        "timings": {
            "train": {"duration_sec": generation_result.get("timing_sec", {}).get("export_sec")},
            "generate": {"duration_sec": generation_result.get("timing_sec", {}).get("generate_sec")},
            "total": {"duration_sec": generation_result.get("timing_sec", {}).get("total_sec")},
        },
        "artifacts": {
            "synthetic_csv": str(published_synthetic.resolve()),
            "model_path": str(run_dir.resolve()),
            "engine_params": str((run_dir / "engine" / "engine_params.json").resolve()),
        },
    }
    input_snapshot = {
        "dataset_id": dataset_id,
        "model": MODEL_ID,
        "inputs": {
            "train_csv": _file_snapshot(train_csv),
            "val_csv": _file_snapshot(val_csv),
            "test_csv": _file_snapshot(test_csv),
            "main_csv": _file_snapshot(main_csv),
        },
    }
    staged_input_manifest = {
        "dataset_id": dataset_id,
        "model": MODEL_ID,
        "target_column": _target_column_from_schema(dataset_dir),
        "task_type": _task_type_from_schema(dataset_dir),
        "train_csv": str(train_csv.resolve()),
        "val_csv": str(val_csv.resolve()),
        "test_csv": str(test_csv.resolve()),
        "main_csv": str(main_csv.resolve()),
    }
    _write_json(metadata_dir / f"{prefix}runtime_result.json", runtime_result)
    _write_json(metadata_dir / f"{prefix}input_snapshot.json", input_snapshot)
    _write_json(metadata_dir / f"{prefix}staged_input_manifest.json", staged_input_manifest)
    field_registry = dataset_dir / "metadata_core" / "field_registry.json"
    if field_registry.is_file():
        shutil.copy2(field_registry, metadata_dir / f"{prefix}field_registry.json")
    return {
        "published_synthetic_csv": str(published_synthetic),
        "published_runtime_result": str(metadata_dir / f"{prefix}runtime_result.json"),
        "published_model_dir": str(model_dir),
        "run_id": run_id,
    }


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.symlink(source.resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def _file_snapshot(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256_file(path) if path.is_file() else None,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_column_from_schema(dataset_dir: Path) -> str | None:
    registry = _read_json_or_empty(dataset_dir / "metadata_core" / "field_registry.json")
    for key in ("target_column", "target"):
        value = registry.get(key)
        if isinstance(value, str) and value:
            return value
    fields = registry.get("fields") or registry.get("columns") or []
    if isinstance(fields, list):
        for field in fields:
            if isinstance(field, dict) and str(field.get("role", "")).lower() == "target":
                name = field.get("name")
                return str(name) if name else None
    return None


def _task_type_from_schema(dataset_dir: Path) -> str | None:
    registry = _read_json_or_empty(dataset_dir / "metadata_core" / "field_registry.json")
    value = registry.get("task_type")
    return str(value) if value else None


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _summarize_existing(dataset_id: str, train_csv: Path, synthetic_csv: Path, run_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "dataset_id": dataset_id,
        "train_csv": str(train_csv),
        "run_dir": str(run_dir),
        "synthetic_csv": str(synthetic_csv),
        "synthetic_exists": synthetic_csv.is_file(),
    }
    if train_csv.is_file():
        record["train_rows"] = _count_csv_rows(train_csv)
        record["train_bytes"] = train_csv.stat().st_size
    if synthetic_csv.is_file():
        record["synthetic_rows"] = _count_csv_rows(synthetic_csv)
        record["synthetic_bytes"] = synthetic_csv.stat().st_size
        record["row_count_matches_train"] = record.get("train_rows") == record.get("synthetic_rows")
    result_path = run_dir / "dataevolve_generation_result.json"
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            timing = payload.get("timing_sec", {})
            if isinstance(timing, dict):
                record.update({f"timing_{key}": value for key, value in timing.items()})
        except Exception:
            pass
    return record


def _count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _write_summary(out_root: Path, records: list[dict[str, Any]]) -> None:
    (out_root / "manifest.json").write_text(
        json.dumps({"records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record})
    with (out_root / "dataevolve_off_generation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    raise SystemExit(main())
