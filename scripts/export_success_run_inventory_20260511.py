#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_PROV_JSON = Path(r"F:\TabQueryBench\Data_HF\_LOCAL_ONLY_NOT_FOR_UPLOAD\final_csv_provenance_20260509.json")
HYPER_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\hyper_parameter")
TIME_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\time_cost")
OUT_DIR = REPO_ROOT / "tmp" / "success_run_inventory_20260511"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _repo_rel(path: Path) -> str:
    return _posix(path.relative_to(REPO_ROOT))


def _source_tag_from_root(rel: str) -> str:
    if rel.startswith("SynOutput-5090/"):
        return "5"
    if rel.startswith("SynOutput/"):
        return "6"
    if rel.startswith("hyperparameter/"):
        return "H"
    if rel.startswith("remote-output-Benchmark-trainonly-v1/"):
        return "R"
    return ""


def _extract_run_id_from_name(name: str) -> str:
    if "__" in name:
        parts = Path(name).stem.split("__")
        if len(parts) >= 3:
            return parts[2]
    stem = Path(name).stem
    if stem.endswith("_runtime_result"):
        return stem[: -len("_runtime_result")]
    if stem.endswith("_tabdiff_train"):
        return stem[: -len("_tabdiff_train")]
    return ""


def _common_ancestor(parts_a: Tuple[str, ...], parts_b: Tuple[str, ...]) -> Tuple[str, ...]:
    acc: List[str] = []
    for a, b in zip(parts_a, parts_b):
        if a != b:
            break
        acc.append(a)
    return tuple(acc)


def _run_dir_from_final_paths(csv_rel: str, meta_rel: str) -> str:
    common = _common_ancestor(Path(csv_rel).parts, Path(meta_rel).parts)
    return _posix(Path(*common))


def _resolve_final_sources(
    source_ref: str, csv_rel: Optional[str], meta_rel: Optional[str]
) -> Tuple[str, str, str, str]:
    if csv_rel and meta_rel:
        return _run_dir_from_final_paths(csv_rel, meta_rel), csv_rel, meta_rel, _extract_run_id_from_name(meta_rel)

    ref_rel = meta_rel or source_ref
    ref_path = Path(ref_rel)
    root_name = ref_path.parts[0]
    run_id = _extract_run_id_from_name(ref_path.name)

    if root_name in {"remote-output-Benchmark-trainonly-v1", "hyperparameter"}:
        abs_ref = REPO_ROOT / ref_rel
        run_dir = abs_ref.parent
        csv_candidates = sorted(run_dir.glob("*.csv"))
        runtime_path = run_dir / "runtime_result.json"
        meta_path = runtime_path if runtime_path.exists() else abs_ref
        csv_path = csv_candidates[0] if csv_candidates else None
        return (
            _repo_rel(run_dir),
            _repo_rel(csv_path) if csv_path else "",
            _repo_rel(meta_path),
            run_id,
        )

    if root_name in {"SynOutput", "SynOutput-5090"}:
        model_root = REPO_ROOT / Path(*ref_path.parts[:3])
        csv_candidates = sorted(
            [p for p in model_root.rglob("*.csv") if run_id and run_id in p.name]
        )
        if not csv_candidates:
            csv_candidates = sorted(model_root.rglob("*.csv"))
        runtime_candidates = sorted(
            [p for p in model_root.rglob("*runtime_result.json") if run_id and run_id in p.name]
        )
        if not runtime_candidates:
            runtime_candidates = sorted(model_root.rglob("*runtime_result.json"))
        csv_path = csv_candidates[0] if csv_candidates else None
        runtime_path = runtime_candidates[0] if runtime_candidates else (REPO_ROOT / ref_rel)
        if csv_path:
            try:
                common = Path(*_common_ancestor(csv_path.relative_to(REPO_ROOT).parts, runtime_path.relative_to(REPO_ROOT).parts))
                run_dir = REPO_ROOT / common
            except Exception:
                run_dir = model_root
        else:
            run_dir = model_root
        return (
            _repo_rel(run_dir),
            _repo_rel(csv_path) if csv_path else "",
            _repo_rel(runtime_path),
            run_id,
        )

    abs_ref = REPO_ROOT / ref_rel
    return (_repo_rel(abs_ref.parent), csv_rel or "", _repo_rel(abs_ref), run_id)


def _collect_final_records() -> List[Dict[str, Any]]:
    prov = _read_json(FINAL_PROV_JSON)
    records: List[Dict[str, Any]] = []
    for item in prov["items"]:
        if item.get("kind") != "synthetic_csv":
            continue
        dataset = item["dataset"]
        model = item["model"]
        source_entry = item["source_entry"]
        csv_rel = item.get("resolved_source_csv")
        meta_rel = item.get("resolved_source_metadata")
        run_dir_rel, csv_rel_resolved, meta_rel_resolved, run_id_hint = _resolve_final_sources(
            source_entry.get("source_ref", ""), csv_rel, meta_rel
        )
        run_dir = REPO_ROOT / run_dir_rel
        run_config = run_dir / "run_config.json"
        record = {
            "bucket": "final",
            "dataset": dataset,
            "model": model,
            "variant": "",
            "source_tag": source_entry.get("source", _source_tag_from_root(run_dir_rel)),
            "source_ref": source_entry.get("source_ref", ""),
            "local_run_dir": run_dir_rel,
            "local_csv": csv_rel_resolved,
            "local_runtime_result_json": meta_rel_resolved,
            "local_run_config_json": _repo_rel(run_config) if run_config.exists() else "",
            "local_json_files": [],
            "local_log_files": [],
            "final_selected": True,
            "hyper_variant": "",
            "time_variant": "",
            "run_id_hint": run_id_hint,
        }
        records.append(record)
    return records


def _collect_bucket_records(bucket: str, root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name == "manifests":
            continue
        dataset = dataset_dir.name
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            index_json = model_dir / "INDEX.json"
            if not index_json.exists():
                continue
            data = _read_json(index_json)
            if bucket == "hyper_parameter":
                for item in data.get("imported", []):
                    source_run = item["source_run"]
                    run_dir = REPO_ROOT / source_run
                    csvs = list(run_dir.glob("*.csv"))
                    runtime = run_dir / "runtime_result.json"
                    run_config = run_dir / "run_config.json"
                    records.append(
                        {
                            "bucket": bucket,
                            "dataset": dataset,
                            "model": model,
                            "variant": item["variant"],
                            "source_tag": _source_tag_from_root(source_run),
                            "source_ref": source_run,
                            "local_run_dir": source_run,
                            "local_csv": _repo_rel(csvs[0]) if csvs else "",
                            "local_runtime_result_json": _repo_rel(runtime) if runtime.exists() else "",
                            "local_run_config_json": _repo_rel(run_config) if run_config.exists() else "",
                            "local_json_files": [],
                            "local_log_files": [],
                            "final_selected": False,
                            "hyper_variant": item["variant"],
                            "time_variant": "",
                            "run_id_hint": Path(source_run).name,
                        }
                    )
            elif bucket == "time_cost":
                for item in data.get("runs", []):
                    source_run = item["source_run"]
                    run_dir = REPO_ROOT / source_run
                    csvs = list(run_dir.glob("*.csv"))
                    runtime = run_dir / "runtime_result.json"
                    run_config = run_dir / "run_config.json"
                    records.append(
                        {
                            "bucket": bucket,
                            "dataset": dataset,
                            "model": model,
                            "variant": item["variant"],
                            "source_tag": _source_tag_from_root(source_run),
                            "source_ref": source_run,
                            "local_run_dir": source_run,
                            "local_csv": _repo_rel(csvs[0]) if csvs else "",
                            "local_runtime_result_json": _repo_rel(runtime) if runtime.exists() else "",
                            "local_run_config_json": _repo_rel(run_config) if run_config.exists() else "",
                            "local_json_files": [],
                            "local_log_files": [],
                            "final_selected": False,
                            "hyper_variant": "",
                            "time_variant": item["variant"],
                            "run_id_hint": Path(source_run).name,
                        }
                    )
    return records


def _safe_rel_file(path: Path) -> str:
    try:
        return _repo_rel(path)
    except ValueError:
        return _posix(path)


def _augment_record(record: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = REPO_ROOT / record["local_run_dir"]
    runtime_path = REPO_ROOT / record["local_runtime_result_json"] if record["local_runtime_result_json"] else None
    run_config_path = REPO_ROOT / record["local_run_config_json"] if record["local_run_config_json"] else None

    run_id_hint = record.get("run_id_hint", "")
    if run_dir.exists():
        json_candidates = [p for p in run_dir.rglob("*.json") if p.is_file()]
        if run_id_hint and not (run_dir / "run_config.json").exists():
            json_candidates = [p for p in json_candidates if run_id_hint in p.name]
        json_files = sorted([_safe_rel_file(p) for p in json_candidates])
    else:
        json_files = []
    log_files: List[str] = []
    if run_dir.exists():
        for candidate in run_dir.rglob("*.log"):
            if candidate.is_file():
                if run_id_hint and not (run_dir / "run_config.json").exists() and run_id_hint not in candidate.name:
                    continue
                log_files.append(_safe_rel_file(candidate))
        log_files.sort()

    runtime_data: Dict[str, Any] = _read_json(runtime_path) if runtime_path and runtime_path.exists() else {}
    run_config_data: Dict[str, Any] = _read_json(run_config_path) if run_config_path and run_config_path.exists() else {}
    timings = runtime_data.get("timings") or {}
    train_timing = timings.get("train") or {}
    gen_timing = timings.get("generate") or {}

    record.update(
        {
            "run_dir_exists": run_dir.exists(),
            "csv_exists": bool(record["local_csv"]) and (REPO_ROOT / record["local_csv"]).exists(),
            "runtime_json_exists": bool(record["local_runtime_result_json"]) and (REPO_ROOT / record["local_runtime_result_json"]).exists(),
            "run_config_exists": bool(record["local_run_config_json"]) and (REPO_ROOT / record["local_run_config_json"]).exists(),
            "local_json_files": json_files,
            "local_log_files": log_files,
            "train_status": runtime_data.get("train_status", ""),
            "generate_status": runtime_data.get("generate_status", ""),
            "public_gate_status": runtime_data.get("public_gate_status", ""),
            "adapter_ready_status": runtime_data.get("adapter_ready_status", ""),
            "run_id": runtime_data.get("run_id", ""),
            "target_column": ((run_config_data.get("input_artifacts") or {}).get("target_column", "")),
            "task_type": ((run_config_data.get("input_artifacts") or {}).get("task_type", "")),
            "num_rows": ((run_config_data.get("resolved") or {}).get("num_rows")),
            "train_duration_sec": train_timing.get("duration_sec"),
            "generate_duration_sec": gen_timing.get("duration_sec"),
            "env_overrides": run_config_data.get("env_overrides", {}),
            "cli_args": run_config_data.get("cli_args", {}),
        }
    )
    return record


def _flatten_env(env: Dict[str, Any]) -> str:
    if not env:
        return ""
    return json.dumps(env, ensure_ascii=False, sort_keys=True)


def _flatten_list(values: Iterable[str]) -> str:
    return " | ".join(values)


def _build_run_level(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        key = rec["local_run_dir"]
        entry = grouped.setdefault(
            key,
            {
                "local_run_dir": key,
                "dataset": rec["dataset"],
                "model": rec["model"],
                "source_tag": rec["source_tag"],
                "local_csv": rec["local_csv"],
                "local_runtime_result_json": rec["local_runtime_result_json"],
                "local_run_config_json": rec["local_run_config_json"],
                "local_json_files": rec["local_json_files"],
                "local_log_files": rec["local_log_files"],
                "run_dir_exists": rec["run_dir_exists"],
                "csv_exists": rec["csv_exists"],
                "runtime_json_exists": rec["runtime_json_exists"],
                "run_config_exists": rec["run_config_exists"],
                "train_status": rec["train_status"],
                "generate_status": rec["generate_status"],
                "public_gate_status": rec["public_gate_status"],
                "adapter_ready_status": rec["adapter_ready_status"],
                "run_id": rec["run_id"],
                "target_column": rec["target_column"],
                "task_type": rec["task_type"],
                "num_rows": rec["num_rows"],
                "train_duration_sec": rec["train_duration_sec"],
                "generate_duration_sec": rec["generate_duration_sec"],
                "env_overrides": rec["env_overrides"],
                "cli_args": rec["cli_args"],
                "buckets": [],
                "final_models": [],
                "hyper_variants": [],
                "time_variants": [],
            },
        )
        entry["buckets"].append(rec["bucket"])
        if rec["bucket"] == "final":
            entry["final_models"].append(f"{rec['dataset']}/{rec['model']}")
        if rec["bucket"] == "hyper_parameter":
            entry["hyper_variants"].append(rec["variant"])
        if rec["bucket"] == "time_cost":
            entry["time_variants"].append(rec["variant"])

    rows: List[Dict[str, Any]] = []
    for entry in grouped.values():
        entry["buckets"] = sorted(set(entry["buckets"]))
        entry["final_models"] = sorted(set(entry["final_models"]))
        entry["hyper_variants"] = sorted(set(entry["hyper_variants"]))
        entry["time_variants"] = sorted(set(entry["time_variants"]))
        rows.append(entry)
    rows.sort(key=lambda r: (r["dataset"], r["model"], r["local_run_dir"]))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = _collect_final_records()
    records.extend(_collect_bucket_records("hyper_parameter", HYPER_ROOT))
    records.extend(_collect_bucket_records("time_cost", TIME_ROOT))
    records = [_augment_record(rec) for rec in records]

    final_records = [rec for rec in records if rec["bucket"] == "final"]
    hyper_records = [rec for rec in records if rec["bucket"] == "hyper_parameter"]
    time_records = [rec for rec in records if rec["bucket"] == "time_cost"]
    run_level_records = _build_run_level(records)

    summary = {
        "record_count_total": len(records),
        "run_count_unique": len(run_level_records),
        "final_record_count": len(final_records),
        "hyper_parameter_record_count": len(hyper_records),
        "time_cost_record_count": len(time_records),
        "generated_at": "2026-05-11",
        "note": "All paths are SQLagent-local relative paths. No F-drive asset paths included.",
    }

    _write_json(OUT_DIR / "summary.json", summary)
    _write_json(OUT_DIR / "final_success_records_20260511.json", final_records)
    _write_json(OUT_DIR / "hyper_parameter_success_records_20260511.json", hyper_records)
    _write_json(OUT_DIR / "time_cost_success_records_20260511.json", time_records)
    _write_json(OUT_DIR / "all_success_records_by_bucket_20260511.json", records)
    _write_json(OUT_DIR / "all_success_runs_by_run_20260511.json", run_level_records)

    bucket_csv_rows = []
    for rec in records:
        bucket_csv_rows.append(
            {
                "bucket": rec["bucket"],
                "dataset": rec["dataset"],
                "model": rec["model"],
                "variant": rec["variant"],
                "source_tag": rec["source_tag"],
                "source_ref": rec["source_ref"],
                "local_run_dir": rec["local_run_dir"],
                "local_csv": rec["local_csv"],
                "local_runtime_result_json": rec["local_runtime_result_json"],
                "local_run_config_json": rec["local_run_config_json"],
                "train_status": rec["train_status"],
                "generate_status": rec["generate_status"],
                "public_gate_status": rec["public_gate_status"],
                "adapter_ready_status": rec["adapter_ready_status"],
                "run_id": rec["run_id"],
                "target_column": rec["target_column"],
                "task_type": rec["task_type"],
                "num_rows": rec["num_rows"],
                "train_duration_sec": rec["train_duration_sec"],
                "generate_duration_sec": rec["generate_duration_sec"],
                "run_dir_exists": rec["run_dir_exists"],
                "csv_exists": rec["csv_exists"],
                "runtime_json_exists": rec["runtime_json_exists"],
                "run_config_exists": rec["run_config_exists"],
                "json_files": _flatten_list(rec["local_json_files"]),
                "log_files": _flatten_list(rec["local_log_files"]),
                "env_overrides_json": _flatten_env(rec["env_overrides"]),
                "cli_args_json": _flatten_env(rec["cli_args"]),
            }
        )

    run_csv_rows = []
    for rec in run_level_records:
        run_csv_rows.append(
            {
                "dataset": rec["dataset"],
                "model": rec["model"],
                "source_tag": rec["source_tag"],
                "local_run_dir": rec["local_run_dir"],
                "local_csv": rec["local_csv"],
                "local_runtime_result_json": rec["local_runtime_result_json"],
                "local_run_config_json": rec["local_run_config_json"],
                "buckets": ",".join(rec["buckets"]),
                "final_models": ",".join(rec["final_models"]),
                "hyper_variants": ",".join(rec["hyper_variants"]),
                "time_variants": ",".join(rec["time_variants"]),
                "train_status": rec["train_status"],
                "generate_status": rec["generate_status"],
                "public_gate_status": rec["public_gate_status"],
                "adapter_ready_status": rec["adapter_ready_status"],
                "run_id": rec["run_id"],
                "target_column": rec["target_column"],
                "task_type": rec["task_type"],
                "num_rows": rec["num_rows"],
                "train_duration_sec": rec["train_duration_sec"],
                "generate_duration_sec": rec["generate_duration_sec"],
                "run_dir_exists": rec["run_dir_exists"],
                "csv_exists": rec["csv_exists"],
                "runtime_json_exists": rec["runtime_json_exists"],
                "run_config_exists": rec["run_config_exists"],
                "json_files": _flatten_list(rec["local_json_files"]),
                "log_files": _flatten_list(rec["local_log_files"]),
                "env_overrides_json": _flatten_env(rec["env_overrides"]),
                "cli_args_json": _flatten_env(rec["cli_args"]),
            }
        )

    _write_csv(
        OUT_DIR / "all_success_records_by_bucket_20260511.csv",
        bucket_csv_rows,
        [
            "bucket",
            "dataset",
            "model",
            "variant",
            "source_tag",
            "source_ref",
            "local_run_dir",
            "local_csv",
            "local_runtime_result_json",
            "local_run_config_json",
            "train_status",
            "generate_status",
            "public_gate_status",
            "adapter_ready_status",
            "run_id",
            "target_column",
            "task_type",
            "num_rows",
            "train_duration_sec",
            "generate_duration_sec",
            "run_dir_exists",
            "csv_exists",
            "runtime_json_exists",
            "run_config_exists",
            "json_files",
            "log_files",
            "env_overrides_json",
            "cli_args_json",
        ],
    )
    _write_csv(
        OUT_DIR / "all_success_runs_by_run_20260511.csv",
        run_csv_rows,
        [
            "dataset",
            "model",
            "source_tag",
            "local_run_dir",
            "local_csv",
            "local_runtime_result_json",
            "local_run_config_json",
            "buckets",
            "final_models",
            "hyper_variants",
            "time_variants",
            "train_status",
            "generate_status",
            "public_gate_status",
            "adapter_ready_status",
            "run_id",
            "target_column",
            "task_type",
            "num_rows",
            "train_duration_sec",
            "generate_duration_sec",
            "run_dir_exists",
            "csv_exists",
            "runtime_json_exists",
            "run_config_exists",
            "json_files",
            "log_files",
            "env_overrides_json",
            "cli_args_json",
        ],
    )


if __name__ == "__main__":
    main()
