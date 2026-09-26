#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\final_csv")
MANIFESTS_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\manifests")
LOCAL_ONLY_ROOT = Path(r"F:\TabQueryBench\Data_HF\_LOCAL_ONLY_NOT_FOR_UPLOAD")
PROV_JSON = LOCAL_ONLY_ROOT / "final_csv_provenance_20260509.json"
PROV_MD = LOCAL_ONLY_ROOT / "final_csv_provenance_20260509.md"
MATRIX_CSV = REPO_ROOT / "tmp" / "generated_data_source_matrix_20260506.csv"
MATRIX_MD = REPO_ROOT / "tmp" / "generated_data_source_matrix_20260506.md"
IMPORT_REPORT = MANIFESTS_ROOT / "final_schema_postprocess_fixed5_imports_20260510.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _norm_code(value: Any) -> str:
    text = str(value)
    try:
        num = float(text)
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
    except Exception:
        pass
    return text


def _build_mapping_from_tabbyflow(run_dir: Path, dataset: str) -> Dict[str, Dict[str, str]]:
    public_train = pd.read_csv(run_dir / "staged" / "public" / "train.csv")
    info_path = run_dir / "tabular_bundle" / f"pipeline_{dataset}" / "info.json"
    real_path = run_dir / "tabular_bundle" / f"pipeline_{dataset}" / "real.csv"
    info = _read_json(info_path)
    grouped_cols = info["column_names"]
    bundle_real = pd.read_csv(real_path)
    bundle_real.columns = grouped_cols

    mappings: Dict[str, Dict[str, str]] = {}
    cat_cols = [grouped_cols[i] for i in info["cat_col_idx"]]
    for col in cat_cols:
        pairs = pd.DataFrame(
            {
                "code": bundle_real[col].astype(str).map(_norm_code),
                "label": public_train[col].astype(str),
            }
        ).drop_duplicates()
        mappings[col] = dict(pairs[["code", "label"]].itertuples(index=False, name=None))

    if info["task_type"] in {"binclass", "multiclass"}:
        y_path = next(iter(sorted(run_dir.rglob("y_train.npy"))))
        import numpy as np

        y_train = np.load(y_path, allow_pickle=True).reshape(-1)
        target_col = grouped_cols[info["target_col_idx"][0]]
        pairs = pd.DataFrame(
            {"code": pd.Series(y_train).astype(str).map(_norm_code), "label": public_train[target_col].astype(str)}
        ).drop_duplicates()
        mappings[target_col] = dict(pairs[["code", "label"]].itertuples(index=False, name=None))

    return mappings


def _decode_series(series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    def _decode(value: Any) -> Any:
        text = str(value)
        norm = _norm_code(text)
        if norm in mapping:
            return mapping[norm]
        if text in mapping.values():
            return text
        return value

    return series.map(_decode)


def _repair_csv(src_csv: Path, train_cols: List[str], decode_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    df = pd.read_csv(src_csv)
    if set(df.columns) != set(train_cols):
        raise ValueError(f"{src_csv} has wrong columns; cannot reorder safely.")
    repaired = df[train_cols].copy()
    for col, mapping in decode_map.items():
        if col in repaired.columns:
            repaired[col] = _decode_series(repaired[col], mapping)
    repaired.to_csv(src_csv, index=False)
    return repaired


def _extract_timing_from_runtime(runtime_result: Path) -> Dict[str, Any]:
    data = _read_json(runtime_result)
    timings = data.get("timings") or {}
    train = ((timings.get("train") or {}).get("duration_sec"))
    generate = ((timings.get("generate") or {}).get("duration_sec"))
    return {"train_duration_sec": train, "generate_duration_sec": generate}


def _extract_tabbyflow_legacy_timing(train_log: Path) -> Dict[str, Any]:
    text = train_log.read_text(encoding="utf-8", errors="ignore")
    # Some legacy EF-VFM logs misspell "total" as "totoal".
    m_train = re.search(r"to(?:tal|toal)\s+training\s+time\s*=\s*([0-9.]+)", text, re.I)
    m_gen = re.search(r"to(?:tal|toal)\s+sampling\s+time\s*=\s*([0-9.]+)", text, re.I)
    return {
        "train_duration_sec": float(m_train.group(1)) if m_train else None,
        "generate_duration_sec": float(m_gen.group(1)) if m_gen else None,
    }


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
    if index_data["skipped"]:
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
    affected_files |= {
        str(FINAL_ROOT / "m4" / "INDEX.json").replace("\\", "/"),
        str(FINAL_ROOT / "m4" / "INDEX.md").replace("\\", "/"),
        str(FINAL_ROOT / "m6" / "INDEX.json").replace("\\", "/"),
        str(FINAL_ROOT / "m6" / "INDEX.md").replace("\\", "/"),
    }
    items = [item for item in items if item.get("final_file") not in affected_files]

    for dataset in {"m4", "m6"}:
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
        ("m4", "tabbyflow"): "F/T/H=5-H-R",
        ("m4", "tabddpm"): "F/T/H=5-6-H-R",
        ("m4", "tabdiff"): "F/T/H=5-6-H-R",
        ("m6", "tabbyflow"): "F/T=5-R",
        ("m6", "tabddpm"): "F/T=6-R",
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


def main() -> None:
    m4_map = _build_mapping_from_tabbyflow(
        REPO_ROOT / "remote-output-Benchmark-trainonly-v1" / "m4" / "tabbyflow" / "tabbyflow-m4-20260505_003507",
        "m4",
    )
    m6_map = _build_mapping_from_tabbyflow(
        REPO_ROOT / "remote-output-Benchmark-trainonly-v1" / "m6" / "tabbyflow" / "tabbyflow-m6-20260429_041029",
        "m6",
    )

    tasks = [
        {
            "dataset": "m4",
            "model": "tabbyflow",
            "source": "5",
            "source_ref": "SynOutput-5090/m4/tabbyflow/metadata/m4__tabbyflow__tabbyflow-m4-20260420_094753__runtime_result.json",
            "resolved_source_csv": "SynOutput-5090/m4/tabbyflow/synthetic_data/m4__tabbyflow__tabbyflow-m4-20260420_094753__tabbyflow-m4-2217-20260420_095345.csv",
            "resolved_source_metadata": "SynOutput-5090/m4/tabbyflow/metadata/m4__tabbyflow__tabbyflow-m4-20260420_094753__runtime_result.json",
            "csv_path": REPO_ROOT / "SynOutput-5090" / "m4" / "tabbyflow" / "synthetic_data" / "m4__tabbyflow__tabbyflow-m4-20260420_094753__tabbyflow-m4-2217-20260420_095345.csv",
            "timing_log": REPO_ROOT / "SynOutput-5090" / "m4" / "tabbyflow" / "logs" / "m4__tabbyflow__tabbyflow-m4-20260420_094753__train_20260420_094753.log",
            "decode_map": m4_map,
        },
        {
            "dataset": "m4",
            "model": "tabdiff",
            "source": "6",
            "source_ref": "SynOutput/m4/tabdiff/metadata/m4__tabdiff__tabdiff-m4-20260501_004659__runtime_result.json",
            "resolved_source_csv": "SynOutput/m4/tabdiff/synthetic_data/m4__tabdiff__tabdiff-m4-20260501_004659__tabdiff-m4-2217-20260501_005309.csv",
            "resolved_source_metadata": "SynOutput/m4/tabdiff/metadata/m4__tabdiff__tabdiff-m4-20260501_004659__runtime_result.json",
            "csv_path": REPO_ROOT / "SynOutput" / "m4" / "tabdiff" / "synthetic_data" / "m4__tabdiff__tabdiff-m4-20260501_004659__tabdiff-m4-2217-20260501_005309.csv",
            "runtime_result": REPO_ROOT / "SynOutput" / "m4" / "tabdiff" / "metadata" / "m4__tabdiff__tabdiff-m4-20260501_004659__runtime_result.json",
            "decode_map": m4_map,
        },
        {
            "dataset": "m4",
            "model": "tabddpm",
            "source": "H",
            "source_ref": "hyperparameter/output-Benchmark-trainonly-v1/m4/tabddpm/tabddpm-m4-20260504_183740/runtime_result.json",
            "resolved_source_csv": "hyperparameter/output-Benchmark-trainonly-v1/m4/tabddpm/tabddpm-m4-20260504_183740/tabddpm-m4-2217-20260504_184027.csv",
            "resolved_source_metadata": "hyperparameter/output-Benchmark-trainonly-v1/m4/tabddpm/tabddpm-m4-20260504_183740/runtime_result.json",
            "csv_path": REPO_ROOT / "hyperparameter" / "output-Benchmark-trainonly-v1" / "m4" / "tabddpm" / "tabddpm-m4-20260504_183740" / "tabddpm-m4-2217-20260504_184027.csv",
            "runtime_result": REPO_ROOT / "hyperparameter" / "output-Benchmark-trainonly-v1" / "m4" / "tabddpm" / "tabddpm-m4-20260504_183740" / "runtime_result.json",
            "decode_map": m4_map,
        },
        {
            "dataset": "m6",
            "model": "tabbyflow",
            "source": "R",
            "source_ref": "remote-output-Benchmark-trainonly-v1/m6/tabbyflow/tabbyflow-m6-20260429_041029/runtime_result.json",
            "resolved_source_csv": "remote-output-Benchmark-trainonly-v1/m6/tabbyflow/tabbyflow-m6-20260429_041029/tabbyflow-m6-9864-20260429_042639.csv",
            "resolved_source_metadata": "remote-output-Benchmark-trainonly-v1/m6/tabbyflow/tabbyflow-m6-20260429_041029/runtime_result.json",
            "csv_path": REPO_ROOT / "remote-output-Benchmark-trainonly-v1" / "m6" / "tabbyflow" / "tabbyflow-m6-20260429_041029" / "tabbyflow-m6-9864-20260429_042639.csv",
            "runtime_result": REPO_ROOT / "remote-output-Benchmark-trainonly-v1" / "m6" / "tabbyflow" / "tabbyflow-m6-20260429_041029" / "runtime_result.json",
            "decode_map": m6_map,
        },
        {
            "dataset": "m6",
            "model": "tabddpm",
            "source": "R",
            "source_ref": "remote-output-Benchmark-trainonly-v1/m6/tabddpm/tabddpm-m6-20260429_052038/runtime_result.json",
            "resolved_source_csv": "remote-output-Benchmark-trainonly-v1/m6/tabddpm/tabddpm-m6-20260429_052038/tabddpm-m6-9864-20260429_052144.csv",
            "resolved_source_metadata": "remote-output-Benchmark-trainonly-v1/m6/tabddpm/tabddpm-m6-20260429_052038/runtime_result.json",
            "csv_path": REPO_ROOT / "remote-output-Benchmark-trainonly-v1" / "m6" / "tabddpm" / "tabddpm-m6-20260429_052038" / "tabddpm-m6-9864-20260429_052144.csv",
            "runtime_result": REPO_ROOT / "remote-output-Benchmark-trainonly-v1" / "m6" / "tabddpm" / "tabddpm-m6-20260429_052038" / "runtime_result.json",
            "decode_map": m6_map,
        },
    ]

    import_records: List[Dict[str, Any]] = []
    index_updates: Dict[str, List[Dict[str, Any]]] = {"m4": [], "m6": []}

    for task in tasks:
        dataset = task["dataset"]
        model = task["model"]
        train_cols = list(pd.read_csv(REPO_ROOT / "data" / dataset / f"{dataset}-train.csv", nrows=0).columns)
        repaired = _repair_csv(task["csv_path"], train_cols, task["decode_map"])

        final_dir = FINAL_ROOT / dataset
        final_dir.mkdir(parents=True, exist_ok=True)
        final_csv = final_dir / f"{dataset}-{model}.csv"
        repaired.to_csv(final_csv, index=False)

        timing = {"train_duration_sec": None, "generate_duration_sec": None}
        if "runtime_result" in task:
            timing = _extract_timing_from_runtime(task["runtime_result"])
        elif "timing_log" in task:
            timing = _extract_tabbyflow_legacy_timing(task["timing_log"])

        final_json = final_dir / f"{dataset}-{model}.json"
        meta = {
            "dataset": dataset,
            "model": model,
            "train_duration_sec": timing["train_duration_sec"],
            "generate_duration_sec": timing["generate_duration_sec"],
            "train_hyperparams": None,
            "hyperparam_source": None,
        }
        if final_json.exists():
            old = _read_json(final_json)
            old.update(meta)
            meta = old
        _write_json(final_json, meta)

        entry = {
            "model": model,
            "source": task["source"],
            "source_ref": task["source_ref"],
            "imported_csv": str(final_csv).replace("\\", "/"),
            "imported_metadata": str(final_json).replace("\\", "/"),
        }
        index_updates[dataset].append(entry)
        import_records.append(
            {
                "dataset": dataset,
                "model": model,
                "source": task["source"],
                "source_ref": task["source_ref"],
                "resolved_source_csv": task["resolved_source_csv"],
                "resolved_source_metadata": task["resolved_source_metadata"],
                "final_csv": str(final_csv).replace("\\", "/"),
                "final_json": str(final_json).replace("\\", "/"),
                "repaired_source_csv": str(task["csv_path"]).replace("\\", "/"),
            }
        )

    for dataset, entries in index_updates.items():
        _update_index(dataset, entries)

    _update_provenance(import_records)
    _update_matrix()
    _write_json(IMPORT_REPORT, {"items": import_records})

    print(f"Imported {len(import_records)} repaired combos into final_csv.")


if __name__ == "__main__":
    main()
