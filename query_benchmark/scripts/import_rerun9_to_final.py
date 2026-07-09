#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\final_csv")
MANIFESTS_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\manifests")
LOCAL_ONLY_ROOT = Path(r"F:\TabQueryBench\Data_HF\_LOCAL_ONLY_NOT_FOR_UPLOAD")
PROV_JSON = LOCAL_ONLY_ROOT / "final_csv_provenance_20260509.json"
PROV_MD = LOCAL_ONLY_ROOT / "final_csv_provenance_20260509.md"
MATRIX_CSV = REPO_ROOT / "tmp" / "generated_data_source_matrix_20260506.csv"
MATRIX_MD = REPO_ROOT / "tmp" / "generated_data_source_matrix_20260506.md"
IMPORT_REPORT = MANIFESTS_ROOT / "final_rerun9_imports_20260510.json"
RERUN_ROOT = REPO_ROOT / "hyperparameter" / "rerun9_20260510" / "output-Benchmark-trainonly-v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    imported = [item for item in data.get("imported", []) if item["model"] not in {e["model"] for e in new_entries}]
    imported.extend(new_entries)
    imported.sort(key=lambda x: x["model"])
    data["imported"] = imported
    data["skipped"] = data.get("skipped", [])
    data["imported_count"] = len(imported)
    data["skipped_count"] = len(data["skipped"])
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
    for dataset in sorted({rec["dataset"] for rec in records}):
        affected_files |= {
            str(FINAL_ROOT / dataset / "INDEX.json").replace("\\", "/"),
            str(FINAL_ROOT / dataset / "INDEX.md").replace("\\", "/"),
        }
    items = [item for item in items if item.get("final_file") not in affected_files]

    for dataset in sorted({rec["dataset"] for rec in records}):
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


def _update_matrix() -> None:
    with MATRIX_CSV.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    rows_by_ds = {row[0]: row for row in rows[1:]}
    updates = {
        ("c14", "ctgan"): "F/T/H=6-H-R",
        ("c14", "tvae"): "F/T/H=6-H-R",
        ("c2", "tabpfgen"): "F/T/H=5-6-H-R",
        ("c4", "tabsyn"): "F/H=5-6-H",
        ("m4", "ctgan"): "F/T/H=5-6-H-R",
        ("m4", "tvae"): "F/T/H=5-6-H-R",
        ("m5", "tabdiff"): "F/H=5-H",
        ("n3", "ctgan"): "F/T/H=5-6-H-R",
        ("n3", "tvae"): "F/T/H=5-6-H-R",
    }
    for (dataset, model), value in updates.items():
        row = rows_by_ds[dataset]
        row[header.index(model)] = value
    with MATRIX_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    MATRIX_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _parse_dims(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _extract_hyperparams(model: str, run_dir: Path) -> Dict[str, Any]:
    run_config = _read_json(run_dir / "run_config.json")
    env = run_config.get("env_overrides") or {}

    if model == "ctgan":
        result: Dict[str, Any] = {
            "epochs": int(env["CTGAN_DEFAULT_EPOCHS"]),
            "embedding_dim": int(env["CTGAN_EMBEDDING_DIM"]),
            "generator_dim": _parse_dims(env["CTGAN_GENERATOR_DIMS"]),
            "discriminator_dim": _parse_dims(env["CTGAN_DISCRIMINATOR_DIMS"]),
            "batch_size": int(env["CTGAN_BATCH_SIZE"]),
            "pac": int(env["CTGAN_PAC"]),
        }
        if "CTGAN_MAX_TRAIN_ROWS" in env:
            result["max_train_rows"] = int(env["CTGAN_MAX_TRAIN_ROWS"])
        return result

    if model == "tvae":
        return {
            "epochs": int(env["TVAE_EPOCHS"]),
            "batch_size": int(env["TVAE_BATCH_SIZE"]),
            "embedding_dim": int(env["TVAE_EMBEDDING_DIM"]),
            "compress_dims": _parse_dims(env["TVAE_COMPRESS_DIMS"]),
            "decompress_dims": _parse_dims(env["TVAE_DECOMPRESS_DIMS"]),
        }

    if model == "tabpfgen":
        return {
            "device": env["TABPFGEN_DEVICE"],
            "gen_chunk_rows": int(env["TABPFGEN_GEN_CHUNK_ROWS"]),
            "n_sgld_steps": int(env["TABPFGEN_N_SGLD_STEPS"]),
            "sgld_step_size": float(env["TABPFGEN_SGLD_STEP_SIZE"]),
            "sgld_noise_scale": float(env["TABPFGEN_SGLD_NOISE_SCALE"]),
        }

    if model == "tabsyn":
        result = {
            "TABSYN_VAE_BATCH_SIZE": int(env["TABSYN_VAE_BATCH_SIZE"]),
            "TABSYN_VAE_EPOCHS": int(env["TABSYN_VAE_EPOCHS"]),
            "TABSYN_DIFFUSION_MAX_EPOCHS": int(env["TABSYN_DIFFUSION_MAX_EPOCHS"]),
        }
        if "TABSYN_DIFFUSION_BATCH_SIZE" in env:
            result["TABSYN_DIFFUSION_BATCH_SIZE"] = int(env["TABSYN_DIFFUSION_BATCH_SIZE"])
        return result

    if model == "tabdiff":
        meta_path = run_dir / "tabdiff_train_meta.json"
        if meta_path.exists():
            meta = _read_json(meta_path)
            return {
                "dataname": meta.get("dataname"),
                "exp_name": meta.get("exp_name"),
                "steps": meta.get("steps"),
                "batch_size": meta.get("batch_size"),
                "lr": meta.get("lr"),
                "num_timesteps": meta.get("num_timesteps"),
            }
        return {
            "steps": int(env["TABDIFF_STEPS"]),
            "batch_size": int(env["TABDIFF_BATCH_SIZE"]),
            "lr": float(env["TABDIFF_LR"]),
            "num_timesteps": int(env["TABDIFF_NUM_TIMESTEPS"]),
        }

    raise ValueError(f"Unsupported model for rerun9 hyperparam extraction: {model}")


def main() -> None:
    tasks = [
        ("c14", "ctgan", "ctgan-c14-20260510_070847"),
        ("c14", "tvae", "tvae-c14-20260510_071949"),
        ("c2", "tabpfgen", "tabpfgen-c2-20260510_071135"),
        ("c4", "tabsyn", "tabsyn-c4-20260510_080926"),
        ("m4", "ctgan", "ctgan-m4-20260510_071244"),
        ("m4", "tvae", "tvae-m4-20260510_070849"),
        ("m5", "tabdiff", "tabdiff-m5-20260510_162741"),
        ("n3", "ctgan", "ctgan-n3-20260510_071506"),
        ("n3", "tvae", "tvae-n3-20260510_070951"),
    ]

    import_records: List[Dict[str, Any]] = []
    index_updates: Dict[str, List[Dict[str, Any]]] = {}

    for dataset, model, run_name in tasks:
        run_dir = RERUN_ROOT / dataset / model / run_name
        runtime_result = run_dir / "runtime_result.json"
        if not runtime_result.exists():
            raise FileNotFoundError(runtime_result)
        csv_files = sorted(run_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No csv found in {run_dir}")
        src_csv = csv_files[0]
        data = _read_json(runtime_result)
        if data.get("train_status") != "success" or data.get("generate_status") != "success":
            raise ValueError(f"{run_dir} is not a successful run")

        train_cols = list(pd.read_csv(REPO_ROOT / "data" / dataset / f"{dataset}-train.csv", nrows=0).columns)
        df = pd.read_csv(src_csv)
        if list(df.columns) != train_cols:
            raise ValueError(f"{src_csv} columns do not match {dataset}-train.csv")

        final_dir = FINAL_ROOT / dataset
        final_dir.mkdir(parents=True, exist_ok=True)
        final_csv = final_dir / f"{dataset}-{model}.csv"
        final_json = final_dir / f"{dataset}-{model}.json"
        df.to_csv(final_csv, index=False)

        timings = data.get("timings") or {}
        train_sec = ((timings.get("train") or {}).get("duration_sec"))
        gen_sec = ((timings.get("generate") or {}).get("duration_sec"))
        train_hyperparams = _extract_hyperparams(model, run_dir)
        meta = {
            "dataset": dataset,
            "model": model,
            "train_duration_sec": train_sec,
            "generate_duration_sec": gen_sec,
            "train_hyperparams": train_hyperparams,
            "hyperparam_source": "exact_artifact",
        }
        if final_json.exists():
            old = _read_json(final_json)
            old.update(meta)
            meta = old
        _write_json(final_json, meta)

        source_ref = f"hyperparameter/rerun9_20260510/output-Benchmark-trainonly-v1/{dataset}/{model}/{run_name}/runtime_result.json"
        resolved_csv = f"hyperparameter/rerun9_20260510/output-Benchmark-trainonly-v1/{dataset}/{model}/{run_name}/{src_csv.name}"
        resolved_meta = source_ref

        entry = {
            "model": model,
            "source": "H",
            "source_ref": source_ref,
            "imported_csv": str(final_csv).replace("\\", "/"),
            "imported_metadata": str(final_json).replace("\\", "/"),
        }
        index_updates.setdefault(dataset, []).append(entry)
        import_records.append(
            {
                "dataset": dataset,
                "model": model,
                "source": "H",
                "source_ref": source_ref,
                "resolved_source_csv": resolved_csv,
                "resolved_source_metadata": resolved_meta,
                "final_csv": str(final_csv).replace("\\", "/"),
                "final_json": str(final_json).replace("\\", "/"),
            }
        )

    for dataset, entries in index_updates.items():
        _update_index(dataset, entries)
    _update_provenance(import_records)
    _update_matrix()
    _write_json(IMPORT_REPORT, {"items": import_records})
    print(json.dumps({"imported": len(import_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
