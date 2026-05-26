import csv
import json
from pathlib import Path


ROOT = Path(r"D:\dpan\Uni\Project\HKUNAISS\SQLagent")
MATRIX = ROOT / "tmp" / "generated_data_source_matrix_20260506.csv"
SRC_ROOT = ROOT / "remote-output-Benchmark-trainonly-v1"
DEST_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\final_csv")
EXCLUDE = {"codi", "cdtd", "goggle"}


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_r_only_targets():
    targets = []
    with MATRIX.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            for model, val in row.items():
                if model == "dataset" or model in EXCLUDE:
                    continue
                if val == "R":
                    targets.append((ds, model))
    return sorted(set(targets))


def list_success_runs(model_dir: Path):
    runs = []
    for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        runtime_path = run_dir / "runtime_result.json"
        if not runtime_path.exists():
            continue
        try:
            runtime = read_json(runtime_path)
        except Exception:
            continue
        if runtime.get("generate_status") != "success":
            continue
        runs.append((run_dir, runtime))
    return runs


def find_synthetic_csv(run_dir: Path, runtime: dict):
    synthetic_name = Path(runtime.get("artifacts", {}).get("synthetic_csv", "")).name
    if synthetic_name:
        direct = run_dir / synthetic_name
        if direct.exists():
            return direct
        for p in run_dir.rglob(synthetic_name):
            if p.is_file():
                return p
    csvs = sorted(p for p in run_dir.glob("*.csv") if p.is_file())
    if len(csvs) == 1:
        return csvs[0]
    return None


def extract_train_hparams(run_dir: Path):
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        try:
            run_config = read_json(run_config_path)
        except Exception:
            run_config = {}
    else:
        run_config = {}

    env = run_config.get("env_overrides", {})
    cleaned = {}
    if isinstance(env, dict):
        for k, v in env.items():
            if k.endswith("_GPUS") or "GPUS" in k:
                continue
            cleaned[k] = v

    cli = run_config.get("cli_args", {})
    if isinstance(cli, dict):
        for k in ["epochs", "num_rows"]:
            if cli.get(k) is not None:
                cleaned[k] = cli.get(k)

    for meta_name in [
        "tabdiff_train_meta.json",
        "tabbyflow_train_meta.json",
        "tabpfgen_train_meta.json",
        "tvae_train_meta.json",
        "ctgan_train_meta.json",
        "arf_train_meta.json",
        "bayesnet_train_meta.json",
        "realtabformer_train_meta.json",
        "forestdiffusion_train_meta.json",
    ]:
        meta_path = run_dir / meta_name
        if meta_path.exists():
            try:
                meta = read_json(meta_path)
                if isinstance(meta, dict):
                    cleaned.update(meta)
            except Exception:
                pass
    return cleaned or None


def load_index(dataset_dir: Path):
    idx = dataset_dir / "INDEX.json"
    if idx.exists():
        return read_json(idx)
    return {
        "dataset": dataset_dir.name,
        "imported_count": 0,
        "skipped_count": 0,
        "imported": [],
        "skipped": [],
    }


def write_index(dataset_dir: Path, payload: dict):
    payload["imported"] = sorted(payload["imported"], key=lambda x: x["model"])
    payload["skipped"] = sorted(payload["skipped"], key=lambda x: x["model"])
    payload["imported_count"] = len(payload["imported"])
    payload["skipped_count"] = len(payload["skipped"])
    with (dataset_dir / "INDEX.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        f"# {dataset_dir.name}",
        "",
        f"- imported: `{payload['imported_count']}`",
        f"- skipped: `{payload['skipped_count']}`",
        "",
        "## Imported",
    ]
    if payload["imported"]:
        for item in payload["imported"]:
            lines.append(
                f"- `{item['model']}` from `{item['source']}` -> `{item['source_ref']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Skipped"])
    if payload["skipped"]:
        for item in payload["skipped"]:
            source_ref = item.get("source_ref", "")
            if source_ref:
                lines.append(
                    f"- `{item['model']}` from `{item['source']}` skipped: `{item['reason']}` -> `{source_ref}`"
                )
            else:
                lines.append(
                    f"- `{item['model']}` from `{item['source']}` skipped: `{item['reason']}`"
                )
    else:
        lines.append("- none")
    with (dataset_dir / "INDEX.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    targets = find_r_only_targets()
    added = []
    skipped = []

    for dataset, model in targets:
        dataset_dir = DEST_ROOT / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        csv_dest = dataset_dir / f"{dataset}-{model}.csv"
        json_dest = dataset_dir / f"{dataset}-{model}.json"

        # already there
        if csv_dest.exists() and json_dest.exists():
            continue

        model_dir = SRC_ROOT / dataset / model
        if not model_dir.exists():
            skipped.append((dataset, model, "source_model_dir_missing", None))
            continue

        success_runs = list_success_runs(model_dir)
        if not success_runs:
            skipped.append((dataset, model, "no_success_run", repo_rel(model_dir)))
            continue

        # choose latest successful run
        run_dir, runtime = success_runs[-1]
        synthetic_csv = find_synthetic_csv(run_dir, runtime)
        if synthetic_csv is None:
            skipped.append((dataset, model, "no_synthetic_csv_found", repo_rel(run_dir)))
            continue

        train_hyperparams = extract_train_hparams(run_dir)
        timings = runtime.get("timings", {})
        payload = {
            "dataset": dataset,
            "model": model,
            "train_duration_sec": timings.get("train", {}).get("duration_sec"),
            "generate_duration_sec": timings.get("generate", {}).get("duration_sec"),
            "train_hyperparams": train_hyperparams,
        }
        csv_dest.write_bytes(synthetic_csv.read_bytes())
        with json_dest.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        added.append((dataset, model, repo_rel(run_dir)))

    # update indexes
    affected = sorted(set([d for d, _, _ in added] + [d for d, _, _, _ in skipped]))
    for dataset in affected:
        dataset_dir = DEST_ROOT / dataset
        payload = load_index(dataset_dir)

        imported_map = {item["model"]: item for item in payload.get("imported", [])}
        skipped_map = {item["model"]: item for item in payload.get("skipped", [])}

        for ds, model, source_ref in added:
            if ds != dataset:
                continue
            imported_map[model] = {
                "model": model,
                "source": "R",
                "source_ref": source_ref,
                "imported_csv": str((dataset_dir / f"{dataset}-{model}.csv")).replace("\\", "/"),
                "imported_metadata": str((dataset_dir / f"{dataset}-{model}.json")).replace("\\", "/"),
            }
            skipped_map.pop(model, None)

        for ds, model, reason, source_ref in skipped:
            if ds != dataset:
                continue
            if model not in imported_map:
                skipped_map[model] = {
                    "model": model,
                    "source": "R",
                    "reason": reason,
                    "source_ref": source_ref,
                }

        payload["imported"] = list(imported_map.values())
        payload["skipped"] = list(skipped_map.values())
        write_index(dataset_dir, payload)

    summary_path = DEST_ROOT.parent / "manifests" / "r_only_supplement_summary.json"
    summary = {
        "target_count": len(targets),
        "added_count": len(added),
        "skipped_count": len(skipped),
        "added": [{"dataset": d, "model": m, "source_ref": s} for d, m, s in added],
        "skipped": [
            {"dataset": d, "model": m, "reason": r, "source_ref": s}
            for d, m, r, s in skipped
        ],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
