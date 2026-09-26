import csv
import json
import shutil
from pathlib import Path


ROOT = Path(r"D:\dpan\Uni\Project\HKUNAISS\SQLagent")
DEST_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data")
FINAL_ROOT = DEST_ROOT / "final_csv"
TIME_COST_ROOT = DEST_ROOT / "time_cost"
HYPER_ROOT = DEST_ROOT / "hyper_parameter"
LOCAL_ONLY_ROOT = Path(r"F:\TabQueryBench\Data_HF\_LOCAL_ONLY_NOT_FOR_UPLOAD")

RECOMMEND_JSON = ROOT / "tmp" / "time_cost_only_final_recommendations_20260509.json"
MATRIX_CSV = ROOT / "tmp" / "generated_data_source_matrix_20260506.csv"
MATRIX_MD = ROOT / "tmp" / "generated_data_source_matrix_20260506.md"
MANIFEST_JSON = DEST_ROOT / "manifests" / "final_time_cost_now9_imports_20260509.json"
PROVENANCE_JSON = LOCAL_ONLY_ROOT / "final_csv_provenance_20260509.json"
PROVENANCE_MD = LOCAL_ONLY_ROOT / "final_csv_provenance_20260509.md"

SOURCE_ROOTS = {
    "5": ROOT / "SynOutput-5090",
    "6": ROOT / "SynOutput",
    "R": ROOT / "remote-output-Benchmark-trainonly-v1",
}
CSV_SUFFIX_BLACKLIST = ("__real.csv", "__train.csv", "__val.csv", "__test.csv")
IGNORE_TOP = {"manifests"}

SELECTED = {
    ("c7", "bayesnet"),
    ("c7", "tabddpm"),
    ("c7", "tabsyn"),
    ("c7", "tvae"),
    ("m6", "ctgan"),
    ("m6", "tvae"),
    ("n6", "ctgan"),
    ("n6", "realtabformer"),
    ("n6", "tvae"),
}


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def posix_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def find_runtime_result(run_dir: Path):
    direct = run_dir / "runtime_result.json"
    if direct.exists():
        return direct
    meta = run_dir / "meta" / "runtime_result.json"
    if meta.exists():
        return meta
    return None


def find_synthetic_csv(base_dir: Path, runtime: dict):
    basename = Path(runtime.get("artifacts", {}).get("synthetic_csv", "")).name
    if basename:
        direct = base_dir / basename
        if direct.exists():
            return direct
        for p in base_dir.rglob(basename):
            if p.is_file():
                return p
    candidates = [
        p
        for p in base_dir.rglob("*.csv")
        if p.is_file() and not any(p.name.endswith(suffix) for suffix in CSV_SUFFIX_BLACKLIST)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def extract_train_hparams_from_runtime_path(source: str, runtime_path: Path):
    if source in {"5", "6"}:
        normalized = runtime_path.parent / runtime_path.name.replace("__runtime_result.json", "__normalized_record.json")
        if normalized.exists():
            try:
                payload = read_json(normalized)
                return payload.get("train_hyperparams")
            except Exception:
                return None
    if source == "R":
        run_dir = runtime_path.parent if runtime_path.name == "runtime_result.json" else runtime_path.parent.parent
        run_config = run_dir / "run_config.json"
        if run_config.exists():
            try:
                payload = read_json(run_config)
                env = payload.get("env_overrides")
                if isinstance(env, dict) and env:
                    return env
                cli = payload.get("cli_args")
                if isinstance(cli, dict) and cli:
                    return cli
            except Exception:
                pass
        for meta_file in run_dir.glob("*_train_meta.json"):
            try:
                payload = read_json(meta_file)
                if isinstance(payload, dict) and payload:
                    return payload
            except Exception:
                pass
    return None


def resolve_source_artifacts(item: dict):
    dataset = item["dataset"]
    model = item["model"]
    recommended = item["recommended"]
    source = recommended["source"]
    run_ref = recommended["source_ref"]

    if source in {"5", "6"}:
        runtime_path = ROOT / Path(run_ref)
        runtime = read_json(runtime_path)
        model_dir = SOURCE_ROOTS[source] / dataset / model
        if not model_dir.exists() and model == "realtabformer":
            alias = SOURCE_ROOTS[source] / dataset / "rtf"
            if alias.exists():
                model_dir = alias
        synth_dir = model_dir
        synthetic_csv = find_synthetic_csv(synth_dir, runtime)
        timings = runtime.get("timings", {})
        return {
            "runtime_path": runtime_path,
            "synthetic_csv": synthetic_csv,
            "train_hyperparams": extract_train_hparams_from_runtime_path(source, runtime_path),
            "train_duration_sec": timings.get("train", {}).get("duration_sec"),
            "generate_duration_sec": timings.get("generate", {}).get("duration_sec"),
            "source_ref": repo_rel(runtime_path),
            "source_run": recommended["run_label"],
            "source": source,
        }

    if source == "R":
        run_dir = ROOT / Path(run_ref)
        runtime_path = find_runtime_result(run_dir)
        if runtime_path is None:
            return None
        runtime = read_json(runtime_path)
        synthetic_csv = find_synthetic_csv(run_dir, runtime)
        timings = runtime.get("timings", {})
        return {
            "runtime_path": runtime_path,
            "synthetic_csv": synthetic_csv,
            "train_hyperparams": extract_train_hparams_from_runtime_path(source, runtime_path),
            "train_duration_sec": timings.get("train", {}).get("duration_sec"),
            "generate_duration_sec": timings.get("generate", {}).get("duration_sec"),
            "source_ref": repo_rel(run_dir),
            "source_run": repo_rel(run_dir),
            "source": source,
        }

    return None


def load_dataset_index(dataset: str):
    dataset_dir = FINAL_ROOT / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    index_path = dataset_dir / "INDEX.json"
    if index_path.exists():
        return read_json(index_path)
    return {"dataset": dataset, "imported_count": 0, "skipped_count": 0, "imported": [], "skipped": []}


def dedupe_imported(items):
    dedup = {}
    for item in items:
        dedup[item["model"]] = item
    return [dedup[k] for k in sorted(dedup)]


def write_dataset_index(index_payload: dict):
    dataset = index_payload["dataset"]
    dataset_dir = FINAL_ROOT / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    index_payload["imported"] = dedupe_imported(index_payload.get("imported", []))
    index_payload["skipped"] = sorted(index_payload.get("skipped", []), key=lambda x: x["model"])
    index_payload["imported_count"] = len(index_payload["imported"])
    index_payload["skipped_count"] = len(index_payload["skipped"])
    write_json(dataset_dir / "INDEX.json", index_payload)

    lines = [
        f"# {dataset}",
        "",
        f"- imported: `{index_payload['imported_count']}`",
        f"- skipped: `{index_payload['skipped_count']}`",
        "",
        "## Imported",
    ]
    if index_payload["imported"]:
        for item in index_payload["imported"]:
            lines.append(f"- `{item['model']}` from `{item['source']}` -> `{item['source_ref']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Skipped"])
    if index_payload["skipped"]:
        for item in index_payload["skipped"]:
            source_ref = item.get("source_ref", "")
            if source_ref:
                lines.append(
                    f"- `{item['model']}` from `{item['source']}` skipped: `{item['reason']}` -> `{source_ref}`"
                )
            else:
                lines.append(f"- `{item['model']}` from `{item['source']}` skipped: `{item['reason']}`")
    else:
        lines.append("- none")
    (dataset_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def gather_final_imported():
    mapping = {}
    for dataset_dir in sorted(p for p in FINAL_ROOT.iterdir() if p.is_dir()):
        index_path = dataset_dir / "INDEX.json"
        if not index_path.exists():
            continue
        try:
            payload = read_json(index_path)
        except Exception:
            continue
        for item in payload.get("imported", []):
            mapping[(dataset_dir.name, item["model"])] = item["source"]
    return mapping


def gather_time_cost_models():
    combos = set()
    if not TIME_COST_ROOT.exists():
        return combos
    for dataset_dir in sorted(p for p in TIME_COST_ROOT.iterdir() if p.is_dir() and p.name not in IGNORE_TOP):
        for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            index_path = model_dir / "INDEX.json"
            run_count = None
            if index_path.exists():
                try:
                    payload = read_json(index_path)
                    run_count = payload.get("run_count")
                except Exception:
                    run_count = None
            if run_count is None:
                run_count = len(list(model_dir.glob("*.json"))) - 1
            if run_count and run_count > 0:
                combos.add((dataset_dir.name, model_dir.name))
    return combos


def gather_hyper_models():
    combos = set()
    if not HYPER_ROOT.exists():
        return combos
    for dataset_dir in sorted(p for p in HYPER_ROOT.iterdir() if p.is_dir() and p.name not in IGNORE_TOP):
        if dataset_dir.name == "manifests":
            continue
        for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            index_path = model_dir / "INDEX.json"
            run_count = None
            if index_path.exists():
                try:
                    payload = read_json(index_path)
                    run_count = payload.get("run_count")
                except Exception:
                    run_count = None
            if run_count is None:
                run_count = len([p for p in model_dir.glob("*.json") if p.name != "INDEX.json"])
            if run_count and run_count > 0:
                combos.add((dataset_dir.name, model_dir.name))
    return combos


def refresh_matrix():
    final_map = gather_final_imported()
    time_cost_set = gather_time_cost_models()
    hyper_set = gather_hyper_models()

    with MATRIX_CSV.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    updated_rows = []
    for row in rows:
        dataset = row["dataset"]
        updated = {"dataset": dataset}
        for model in fieldnames:
            if model == "dataset":
                continue
            cell = row.get(model, "")
            if not cell:
                updated[model] = cell
                continue
            source_part = cell.split("=", 1)[1] if "=" in cell else cell
            prefixes = []
            if (dataset, model) in final_map:
                prefixes.append("F")
            if (dataset, model) in time_cost_set:
                prefixes.append("T")
            if (dataset, model) in hyper_set:
                prefixes.append("H")
            updated[model] = ("/".join(prefixes) + "=" + source_part) if prefixes else source_part
        updated_rows.append(updated)

    with MATRIX_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    lines = [
        "# Generated Data Source Matrix (updated 2026-05-09)",
        "",
        "Legend: source codes `5=SynOutput-5090`, `6=SynOutput`, `H=hyperparameter`, `R=remote-output-Benchmark-trainonly-v1`; destination prefixes `F=final_csv`, `T=time_cost`, `H=hyper_parameter`.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in updated_rows:
        lines.append("| " + " | ".join(row[h] for h in fieldnames) + " |")
    MATRIX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_source_paths_from_entry(entry: dict):
    source = entry.get("source")
    source_ref = entry.get("source_ref")
    if not source_ref:
        return None, None
    source_path = ROOT / Path(source_ref)
    if source in {"5", "6"}:
        runtime_path = source_path
        try:
            runtime = read_json(runtime_path)
        except Exception:
            return None, repo_rel(runtime_path)
        dataset = Path(entry["imported_csv"]).stem.split("-")[0]
        model = Path(entry["imported_csv"]).stem[len(dataset) + 1 :]
        model_dir = SOURCE_ROOTS[source] / dataset / model
        if not model_dir.exists() and model == "realtabformer":
            alias = SOURCE_ROOTS[source] / dataset / "rtf"
            if alias.exists():
                model_dir = alias
        synthetic_csv = find_synthetic_csv(model_dir, runtime)
        return repo_rel(synthetic_csv) if synthetic_csv else None, repo_rel(runtime_path)
    if source == "R":
        run_dir = source_path
        runtime_path = find_runtime_result(run_dir)
        if runtime_path is None:
            return None, repo_rel(run_dir)
        try:
            runtime = read_json(runtime_path)
        except Exception:
            return None, repo_rel(runtime_path)
        synthetic_csv = find_synthetic_csv(run_dir, runtime)
        return repo_rel(synthetic_csv) if synthetic_csv else None, repo_rel(runtime_path)
    return None, repo_rel(source_path)


def refresh_provenance():
    items = []
    for dataset_dir in sorted(p for p in FINAL_ROOT.iterdir() if p.is_dir()):
        index_path = dataset_dir / "INDEX.json"
        if not index_path.exists():
            continue
        payload = read_json(index_path)
        imported = payload.get("imported", [])
        csv_targets = [item["imported_csv"] for item in imported]
        for index_name in ("INDEX.json", "INDEX.md"):
            items.append(
                {
                    "final_file": posix_path(dataset_dir / index_name),
                    "kind": "generated_index",
                    "dataset": dataset_dir.name,
                    "derived_from": csv_targets,
                }
            )
        for entry in imported:
            source_csv, source_meta = resolve_source_paths_from_entry(entry)
            dataset = dataset_dir.name
            model = Path(entry["imported_csv"]).stem[len(dataset) + 1 :]
            for key, kind in (("imported_csv", "synthetic_csv"), ("imported_metadata", "metadata_json")):
                items.append(
                    {
                        "final_file": entry[key],
                        "kind": kind,
                        "dataset": dataset,
                        "model": model,
                        "source_entry": entry,
                        "resolved_source_csv": source_csv,
                        "resolved_source_metadata": source_meta,
                    }
                )

    payload = {
        "scope": posix_path(FINAL_ROOT),
        "note": "Local-only provenance map. Do not upload to Hugging Face.",
        "items": items,
    }
    write_json(PROVENANCE_JSON, payload)

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
        if item["kind"] == "generated_index":
            lines.append(f"| {item['final_file']} | {item['kind']} | {item['dataset']} |  |  |  |  |  |")
            continue
        source_entry = item["source_entry"]
        lines.append(
            f"| {item['final_file']} | {item['kind']} | {item['dataset']} | {item['model']} | "
            f"{source_entry.get('source','')} | {source_entry.get('source_ref','')} | "
            f"{item.get('resolved_source_csv') or ''} | {item.get('resolved_source_metadata') or ''} |"
        )
    PROVENANCE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    recommendations = read_json(RECOMMEND_JSON)
    targets = [x for x in recommendations if (x["dataset"], x["model"]) in SELECTED]
    added = []
    dataset_indices = {}

    for item in targets:
        recommended = item["recommended"]
        if not recommended.get("healthy"):
            continue
        artifacts = resolve_source_artifacts(item)
        if artifacts is None or artifacts["synthetic_csv"] is None:
            continue

        dataset = item["dataset"]
        model = item["model"]
        dataset_dir = FINAL_ROOT / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        csv_dest = dataset_dir / f"{dataset}-{model}.csv"
        json_dest = dataset_dir / f"{dataset}-{model}.json"
        shutil.copy2(artifacts["synthetic_csv"], csv_dest)
        write_json(
            json_dest,
            {
                "dataset": dataset,
                "model": model,
                "train_duration_sec": artifacts["train_duration_sec"],
                "generate_duration_sec": artifacts["generate_duration_sec"],
                "train_hyperparams": artifacts["train_hyperparams"],
            },
        )

        if dataset not in dataset_indices:
            dataset_indices[dataset] = load_dataset_index(dataset)
        payload = dataset_indices[dataset]
        entry = {
            "model": model,
            "source": recommended["source"],
            "source_ref": artifacts["source_ref"],
            "imported_csv": posix_path(csv_dest),
            "imported_metadata": posix_path(json_dest),
        }
        payload["imported"] = [x for x in payload.get("imported", []) if x["model"] != model] + [entry]
        added.append(
            {
                "dataset": dataset,
                "model": model,
                "source": recommended["source"],
                "source_run": artifacts["source_run"],
            }
        )

    for payload in dataset_indices.values():
        write_dataset_index(payload)

    write_json(MANIFEST_JSON, {"added_count": len(added), "added": added})
    refresh_matrix()
    refresh_provenance()
    print(json.dumps({"added_count": len(added), "added": added}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
