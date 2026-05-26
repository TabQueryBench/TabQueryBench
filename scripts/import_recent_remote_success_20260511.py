#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\final_csv")
MANIFESTS_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\manifests")
LOCAL_ONLY_ROOT = Path(r"F:\TabQueryBench\Data_HF\_LOCAL_ONLY_NOT_FOR_UPLOAD")
PROV_JSON = LOCAL_ONLY_ROOT / "final_csv_provenance_20260509.json"
PROV_MD = LOCAL_ONLY_ROOT / "final_csv_provenance_20260509.md"
MATRIX_CSV = REPO_ROOT / "tmp" / "generated_data_source_matrix_20260506.csv"
MATRIX_MD = REPO_ROOT / "tmp" / "generated_data_source_matrix_20260506.md"
FINAL_INVENTORY_CSV = FINAL_ROOT / "final_csv_dataset_model_inventory_20260510.csv"
IMPORT_REPORT = MANIFESTS_ROOT / "final_recent_remote_success_20260511.json"
REMOTE_ROOT = REPO_ROOT / "remote-output-Benchmark-trainonly-v1" / "output-Benchmark-trainonly-v1"

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
SOURCE_ORDER = ["5", "6", "H", "R"]
PREFIX_ORDER = ["F", "T", "H"]

IMPORT_TASKS: List[Dict[str, str]] = [
    {"dataset": "c18", "model": "forestdiffusion", "run": "forest-c18-20260510_220505"},
    {"dataset": "c18", "model": "tabbyflow", "run": "tabbyflow-c18-20260510_213427"},
    {"dataset": "c18", "model": "tvae", "run": "tvae-c18-20260511_061930"},
    {"dataset": "c4", "model": "forestdiffusion", "run": "forest-c4-20260510_204828"},
    {"dataset": "c5", "model": "forestdiffusion", "run": "forest-c5-20260511_040339"},
    {"dataset": "c5", "model": "tabpfgen", "run": "tabpfgen-c5-20260511_061054"},
    {"dataset": "m9", "model": "tabbyflow", "run": "tabbyflow-m9-20260510_205907"},
    {"dataset": "n17", "model": "forestdiffusion", "run": "forest-n17-20260510_205323"},
    {"dataset": "n4", "model": "forestdiffusion", "run": "forest-n4-20260511_130614"},
    {"dataset": "n8", "model": "forestdiffusion", "run": "forest-n8-20260511_132539"},
    {"dataset": "n8", "model": "tabbyflow", "run": "tabbyflow-n8-20260510_211859"},
    {"dataset": "n8", "model": "tabsyn", "run": "tabsyn-n8-20260510_202608"},
]
MATRIX_ONLY_TASKS: List[Dict[str, str]] = [
    {"dataset": "c4", "model": "tabsyn", "run": "tabsyn-c4-20260510_080926"},
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dataset_sort_key(name: str) -> Tuple[str, int]:
    return name[0], int(name[1:])


def _render_index_md(index_data: Dict[str, Any]) -> str:
    lines = [
        f"# {index_data['dataset']}",
        "",
        f"- imported: `{index_data['imported_count']}`",
        f"- skipped: `{index_data['skipped_count']}`",
        "",
        "## Imported",
    ]
    for item in index_data["imported"]:
        lines.append(f"- `{item['model']}` from `{item['source']}` -> `{item['source_ref']}`")
    lines.extend(["", "## Skipped"])
    if index_data.get("skipped"):
        for item in index_data["skipped"]:
            lines.append(f"- `{item['model']}` -> `{item['reason']}`")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _update_index(dataset: str, new_entries: List[Dict[str, Any]]) -> None:
    ds_dir = FINAL_ROOT / dataset
    index_json = ds_dir / "INDEX.json"
    index_md = ds_dir / "INDEX.md"
    data = _read_json(index_json) if index_json.exists() else {"dataset": dataset, "imported": [], "skipped": []}
    new_models = {entry["model"] for entry in new_entries}
    imported = [item for item in data.get("imported", []) if item["model"] not in new_models]
    imported.extend(new_entries)
    imported.sort(key=lambda x: x["model"])
    skipped = [item for item in data.get("skipped", []) if item["model"] not in new_models]
    data["imported"] = imported
    data["skipped"] = skipped
    data["imported_count"] = len(imported)
    data["skipped_count"] = len(skipped)
    _write_json(index_json, data)
    index_md.write_text(_render_index_md(data), encoding="utf-8")


def _render_provenance_md(items: List[Dict[str, Any]]) -> str:
    lines = [
        "# final_csv provenance",
        "",
        "- local_only: `true`",
        "- note: `Do not upload to Hugging Face.`",
        "",
        "| final_file | kind | dataset | model | source | source_ref | resolved_source_csv | resolved_source_metadata |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in items:
        src = item.get("source_entry", {})
        lines.append(
            f"| {item.get('final_file','')} | {item.get('kind','')} | {item.get('dataset','')} | {item.get('model','')} | "
            f"{src.get('source','')} | {src.get('source_ref','')} | {item.get('resolved_source_csv','')} | {item.get('resolved_source_metadata','')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _update_provenance(records: List[Dict[str, Any]]) -> None:
    prov = _read_json(PROV_JSON)
    items = prov["items"]
    affected_files = {rec["final_csv"] for rec in records} | {rec["final_json"] for rec in records}
    for dataset in sorted({rec["dataset"] for rec in records}, key=_dataset_sort_key):
        affected_files |= {
            str(FINAL_ROOT / dataset / "INDEX.json").replace("\\", "/"),
            str(FINAL_ROOT / dataset / "INDEX.md").replace("\\", "/"),
        }
    items = [item for item in items if item.get("final_file") not in affected_files]

    for dataset in sorted({rec["dataset"] for rec in records}, key=_dataset_sort_key):
        ds_dir = FINAL_ROOT / dataset
        derived = [str(p).replace("\\", "/") for p in sorted(ds_dir.glob(f"{dataset}-*.csv"))]
        for suffix in ("INDEX.json", "INDEX.md"):
            items.append(
                {
                    "final_file": str(ds_dir / suffix).replace("\\", "/"),
                    "kind": "generated_index",
                    "dataset": dataset,
                    "derived_from": derived,
                }
            )

    for rec in records:
        source_entry = {
            "model": rec["model"],
            "source": rec["source"],
            "source_ref": rec["source_ref"],
            "imported_csv": rec["final_csv"],
            "imported_metadata": rec["final_json"],
        }
        items.append(
            {
                "final_file": rec["final_csv"],
                "kind": "synthetic_csv",
                "dataset": rec["dataset"],
                "model": rec["model"],
                "source_entry": source_entry,
                "resolved_source_csv": rec["resolved_source_csv"],
                "resolved_source_metadata": rec["resolved_source_metadata"],
            }
        )
        items.append(
            {
                "final_file": rec["final_json"],
                "kind": "metadata_json",
                "dataset": rec["dataset"],
                "model": rec["model"],
                "source_entry": source_entry,
                "resolved_source_csv": rec["resolved_source_csv"],
                "resolved_source_metadata": rec["resolved_source_metadata"],
            }
        )

    items.sort(key=lambda x: x["final_file"])
    prov["items"] = items
    _write_json(PROV_JSON, prov)
    PROV_MD.write_text(_render_provenance_md(items), encoding="utf-8")


def _parse_cell(cell: str) -> Tuple[List[str], List[str]]:
    if not cell:
        return [], []
    if "=" not in cell:
        return [], [src for src in cell.split("-") if src]
    left, right = cell.split("=", 1)
    prefixes = [p for p in left.split("/") if p]
    sources = [s for s in right.split("-") if s]
    return prefixes, sources


def _format_cell(prefixes: List[str], sources: List[str]) -> str:
    prefixes = sorted(set(prefixes), key=PREFIX_ORDER.index)
    sources = sorted(set(sources), key=SOURCE_ORDER.index)
    if prefixes:
        return f"{'/'.join(prefixes)}={'-'.join(sources)}"
    return "-".join(sources)


def _update_matrix() -> None:
    with MATRIX_CSV.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    rows_by_ds = {row[0]: row for row in rows[1:]}
    for task in IMPORT_TASKS:
        row = rows_by_ds[task["dataset"]]
        idx = header.index(task["model"])
        prefixes, sources = _parse_cell(row[idx])
        prefixes.append("F")
        sources.append("R")
        row[idx] = _format_cell(prefixes, sources)
    for task in MATRIX_ONLY_TASKS:
        row = rows_by_ds[task["dataset"]]
        idx = header.index(task["model"])
        prefixes, sources = _parse_cell(row[idx])
        sources.append("R")
        row[idx] = _format_cell(prefixes, sources)

    with MATRIX_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    MATRIX_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _update_final_inventory() -> None:
    datasets = sorted([p.name for p in FINAL_ROOT.iterdir() if p.is_dir()], key=_dataset_sort_key)
    rows: List[List[str]] = [["dataset"] + MODEL_ORDER]
    for dataset in datasets:
        row = [dataset]
        ds_dir = FINAL_ROOT / dataset
        for model in MODEL_ORDER:
            row.append("1" if (ds_dir / f"{dataset}-{model}.csv").exists() else "")
        rows.append(row)
    with FINAL_INVENTORY_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def _int_env(env: Dict[str, str], key: str, default: int) -> int:
    return int(env.get(key, str(default)))


def _str_env(env: Dict[str, str], key: str, default: str) -> str:
    return str(env.get(key, default))


def _extract_hyperparams(model: str, run_dir: Path) -> Tuple[Dict[str, Any], str]:
    run_config = _read_json(run_dir / "run_config.json")
    cli_args = run_config.get("cli_args") or {}
    env = run_config.get("env_overrides") or {}

    if model == "forestdiffusion":
        return (
            {
                "max_train_rows": _int_env(env, "FORESTDIFFUSION_MAX_TRAIN_ROWS", 50000),
                "n_estimators": _int_env(env, "FORESTDIFFUSION_N_ESTIMATORS", 20),
                "n_t": _int_env(env, "FORESTDIFFUSION_N_T", 10),
                "duplicate_K": _int_env(env, "FORESTDIFFUSION_DUPLICATE_K", 5),
                "n_jobs": _int_env(env, "FORESTDIFFUSION_N_JOBS", 1),
                "max_depth": _int_env(env, "FORESTDIFFUSION_MAX_DEPTH", 4),
                "xgb_nthread": _int_env(env, "FORESTDIFFUSION_XGB_NTHREAD", 1),
                "xgb_verbosity": _int_env(env, "FORESTDIFFUSION_XGB_VERBOSITY", 1),
            },
            "exact_artifact",
        )

    if model == "tabbyflow":
        steps = cli_args.get("epochs")
        if steps is None:
            steps = 500
        return (
            {
                "steps": int(steps),
                "lr": 0.001,
                "weight_decay": 0,
                "ema_decay": 0.997,
                "batch_size": 4096,
                "warmup_epochs": 100,
                "num_layers": 2,
                "d_token": 4,
                "n_head": 1,
                "factor": 32,
                "dim_t": 1024,
                "activation": "gelu",
                "EFVFM_SAMPLE_BATCH_SIZE": _int_env(env, "EFVFM_SAMPLE_BATCH_SIZE", 128),
                "EFVFM_EVAL_NUM_SAMPLES": _int_env(env, "EFVFM_EVAL_NUM_SAMPLES", 512),
                "exp_name": "adapter_efvfm",
            },
            "artifact_plus_default",
        )

    if model == "tvae":
        return (
            {
                "epochs": _int_env(env, "TVAE_EPOCHS", 300),
                "batch_size": _int_env(env, "TVAE_BATCH_SIZE", 500),
                "embedding_dim": _int_env(env, "TVAE_EMBEDDING_DIM", 128),
                "compress_dims": [int(part) for part in _str_env(env, "TVAE_COMPRESS_DIMS", "128,128").split(",") if part],
                "decompress_dims": [int(part) for part in _str_env(env, "TVAE_DECOMPRESS_DIMS", "128,128").split(",") if part],
            },
            "exact_artifact",
        )

    if model == "tabpfgen":
        return (
            {
                "device": _str_env(env, "TABPFGEN_DEVICE", "auto"),
                "n_sgld_steps": 1000,
                "sgld_step_size": 0.01,
                "sgld_noise_scale": 0.01,
                "gen_chunk_rows": 256,
            },
            "artifact_plus_default",
        )

    if model == "tabsyn":
        data: Dict[str, Any] = {
            "TABSYN_RESUME": _int_env(env, "TABSYN_RESUME", 0),
            "TABSYN_VAE_BATCH_SIZE": _int_env(env, "TABSYN_VAE_BATCH_SIZE", 32),
            "TABSYN_VAE_EPOCHS": _int_env(env, "TABSYN_VAE_EPOCHS", 4000),
            "TABSYN_DIFFUSION_MAX_EPOCHS": _int_env(env, "TABSYN_DIFFUSION_MAX_EPOCHS", 10001),
            "TABSYN_VAE_ENCODE_BATCH_SIZE": _int_env(env, "TABSYN_VAE_ENCODE_BATCH_SIZE", 32),
            "TABSYN_VAE_EVAL_BATCH_SIZE": _int_env(env, "TABSYN_VAE_EVAL_BATCH_SIZE", 32),
            "TABSYN_VAE_INFER_BATCH_SIZE": _int_env(env, "TABSYN_VAE_INFER_BATCH_SIZE", 32),
            "TABSYN_VAE_NUM_WORKERS": _int_env(env, "TABSYN_VAE_NUM_WORKERS", 0),
        }
        if "TABSYN_DIFFUSION_BATCH_SIZE" in env:
            data["TABSYN_DIFFUSION_BATCH_SIZE"] = int(env["TABSYN_DIFFUSION_BATCH_SIZE"])
        return data, "exact_artifact"

    raise ValueError(f"Unsupported model: {model}")


def main() -> None:
    imported_records: List[Dict[str, str]] = []
    imported_index_entries: Dict[str, List[Dict[str, Any]]] = {}

    for task in IMPORT_TASKS:
        dataset = task["dataset"]
        model = task["model"]
        run = task["run"]
        run_dir = REMOTE_ROOT / dataset / model / run
        runtime_path = run_dir / "runtime_result.json"
        run_config_path = run_dir / "run_config.json"
        csv_candidates = sorted(run_dir.glob("*.csv"))
        if not runtime_path.exists() or not run_config_path.exists() or not csv_candidates:
            raise FileNotFoundError(f"Missing run artifacts for {dataset}/{model}/{run}")
        source_csv = csv_candidates[0]
        runtime = _read_json(runtime_path)
        train_hyperparams, hyperparam_source = _extract_hyperparams(model, run_dir)

        ds_dir = FINAL_ROOT / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        final_csv_path = ds_dir / f"{dataset}-{model}.csv"
        final_json_path = ds_dir / f"{dataset}-{model}.json"
        shutil.copy2(source_csv, final_csv_path)

        metadata = {
            "dataset": dataset,
            "model": model,
            "train_duration_sec": (((runtime.get("timings") or {}).get("train") or {}).get("duration_sec")),
            "generate_duration_sec": (((runtime.get("timings") or {}).get("generate") or {}).get("duration_sec")),
            "train_hyperparams": train_hyperparams,
            "hyperparam_source": hyperparam_source,
        }
        _write_json(final_json_path, metadata)

        source_ref = f"remote-output-Benchmark-trainonly-v1/output-Benchmark-trainonly-v1/{dataset}/{model}/{run}/runtime_result.json"
        index_entry = {
            "model": model,
            "source": "R",
            "source_ref": source_ref,
            "imported_csv": str(final_csv_path).replace("\\", "/"),
            "imported_metadata": str(final_json_path).replace("\\", "/"),
        }
        imported_index_entries.setdefault(dataset, []).append(index_entry)
        imported_records.append(
            {
                "dataset": dataset,
                "model": model,
                "run": run,
                "source": "R",
                "source_ref": source_ref,
                "final_csv": str(final_csv_path).replace("\\", "/"),
                "final_json": str(final_json_path).replace("\\", "/"),
                "resolved_source_csv": str(source_csv).replace("\\", "/"),
                "resolved_source_metadata": str(runtime_path).replace("\\", "/"),
            }
        )

    for dataset, entries in imported_index_entries.items():
        _update_index(dataset, entries)

    _update_provenance(imported_records)
    _update_matrix()
    _update_final_inventory()

    report = {
        "imported_count": len(imported_records),
        "matrix_only_updates": MATRIX_ONLY_TASKS,
        "imported": imported_records,
    }
    _write_json(IMPORT_REPORT, report)


if __name__ == "__main__":
    main()
