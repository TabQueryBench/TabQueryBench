import csv
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path.cwd()
DEST_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data")
MATRIX_CSV = ROOT / "tmp" / "generated_data_source_matrix_20260506.csv"
DUP_JSON = ROOT / "tmp" / "generated_data_duplicate_details_20260506.json"

EXCLUDE_MODELS = {"codi", "cdtd", "goggle"}
ALLOWED_SINGLE_SOURCE = {"5", "6", "R"}
SOURCE_PATHS = {
    "5": ROOT / "SynOutput-5090",
    "6": ROOT / "SynOutput",
    "R": ROOT / "remote-output-Benchmark-trainonly-v1",
}

RUN_DIR_PAT = re.compile(r"^[A-Za-z0-9]+-[A-Za-z0-9]+-\d{8}_\d{6}$")
META_RUN_PAT = re.compile(
    r"^(?P<ds>[a-z]\d+)__(?P<model>[a-z0-9]+)__(?P<run>[A-Za-z0-9-]+-\d{8}_\d{6})__"
)
CSV_SUFFIX_BLACKLIST = ("__real.csv", "__train.csv", "__val.csv", "__test.csv")


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def load_candidates():
    with MATRIX_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with DUP_JSON.open(encoding="utf-8") as f:
        dup = json.load(f)

    within = {
        (entry["dataset"], entry["model"], entry["source"]): entry["run_count"]
        for entry in dup["within_source_multi"]
    }

    candidates = []
    skipped_multi = []
    for row in rows:
        ds = row["dataset"]
        for model, source in row.items():
            if model == "dataset" or model in EXCLUDE_MODELS or not source:
                continue
            if source not in ALLOWED_SINGLE_SOURCE:
                continue
            run_count = within.get((ds, model, source), 1)
            rec = {
                "dataset": ds,
                "model": model,
                "source": source,
                "run_count": run_count,
            }
            if run_count == 1:
                candidates.append(rec)
            else:
                skipped_multi.append(
                    {
                        **rec,
                        "reason": f"multiple_runs_in_single_source:{run_count}",
                    }
                )
    return candidates, skipped_multi


def parse_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def clean_metadata(dataset, model, train_duration, generate_duration, train_hparams):
    return {
        "dataset": dataset,
        "model": model,
        "train_duration_sec": train_duration,
        "generate_duration_sec": generate_duration,
        "train_hyperparams": train_hparams,
    }


def find_single_run_dir(model_dir: Path):
    run_dirs = [p for p in model_dir.iterdir() if p.is_dir() and RUN_DIR_PAT.match(p.name)]
    if len(run_dirs) != 1:
        return None
    return run_dirs[0]


def resolve_consolidated(model_dir: Path, dataset: str, model: str):
    meta_dir = model_dir / "metadata"
    synth_dir = model_dir / "synthetic_data"
    if not meta_dir.exists() or not synth_dir.exists():
        return None

    normalized = list(meta_dir.glob("*__normalized_record.json"))
    if len(normalized) != 1:
        return None

    norm_path = normalized[0]
    norm = parse_json(norm_path)
    runtime_result_path = meta_dir / norm_path.name.replace("__normalized_record.json", "__runtime_result.json")
    if not runtime_result_path.exists():
        return None
    runtime = parse_json(runtime_result_path)

    synthetic_path = None
    basename = Path(runtime.get("artifacts", {}).get("synthetic_csv", "")).name
    if basename:
        for p in synth_dir.iterdir():
            if p.is_file() and p.name.endswith(basename):
                synthetic_path = p
                break
    if synthetic_path is None:
        run_id = norm["run_id"]
        candidates = [
            p
            for p in synth_dir.iterdir()
            if p.is_file()
            and run_id in p.name
            and not any(p.name.endswith(suffix) for suffix in CSV_SUFFIX_BLACKLIST)
        ]
        if len(candidates) == 1:
            synthetic_path = candidates[0]

    if synthetic_path is None:
        return None

    train_hparams = norm.get("train_hyperparams")

    return {
        "dataset": dataset,
        "model": model,
        "synthetic_path": synthetic_path,
        "source_ref": runtime_result_path,
        "run_ref": norm["run_id"],
        "metadata": clean_metadata(
            dataset,
            model,
            norm.get("train_duration_sec"),
            norm.get("generate_duration_sec"),
            train_hparams,
        ),
    }


def find_runtime_in_run_dir(run_dir: Path):
    direct = run_dir / "runtime_result.json"
    if direct.exists():
        return direct
    meta_runtime = run_dir / "meta" / "runtime_result.json"
    if meta_runtime.exists():
        return meta_runtime
    return None


def find_train_hparams_in_run_dir(run_dir: Path):
    run_config = run_dir / "run_config.json"
    if run_config.exists():
        cfg = parse_json(run_config)
        env_overrides = cfg.get("env_overrides")
        if isinstance(env_overrides, dict):
            return env_overrides, run_config
        cli_args = cfg.get("cli_args")
        if isinstance(cli_args, dict):
            return cli_args, run_config

    for meta_file in run_dir.glob("*_train_meta.json"):
        meta = parse_json(meta_file)
        if isinstance(meta, dict):
            return meta, meta_file

    return None, None


def find_synthetic_in_run_dir(run_dir: Path, runtime: dict):
    basename = Path(runtime.get("artifacts", {}).get("synthetic_csv", "")).name
    if basename:
        direct = run_dir / basename
        if direct.exists():
            return direct
        for p in run_dir.rglob(basename):
            if p.is_file():
                return p

    candidates = [
        p
        for p in run_dir.rglob("*.csv")
        if p.is_file() and not any(p.name.endswith(suffix) for suffix in CSV_SUFFIX_BLACKLIST)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def resolve_run_dir(model_dir: Path, dataset: str, model: str):
    run_dir = find_single_run_dir(model_dir)
    if run_dir is None:
        return None

    runtime_path = find_runtime_in_run_dir(run_dir)
    if runtime_path is None:
        return None
    runtime = parse_json(runtime_path)

    if runtime.get("generate_status") != "success":
        return None

    timings = runtime.get("timings", {})
    train_duration = timings.get("train", {}).get("duration_sec")
    generate_duration = timings.get("generate", {}).get("duration_sec")

    train_hparams, hparam_ref = find_train_hparams_in_run_dir(run_dir)

    synthetic_path = find_synthetic_in_run_dir(run_dir, runtime)
    if synthetic_path is None:
        return None

    return {
        "dataset": dataset,
        "model": model,
        "synthetic_path": synthetic_path,
        "source_ref": hparam_ref or runtime_path,
        "run_ref": runtime.get("run_id", run_dir.name),
        "metadata": clean_metadata(
            dataset,
            model,
            train_duration,
            generate_duration,
            train_hparams,
        ),
    }


def resolve_candidate(candidate):
    dataset = candidate["dataset"]
    model = candidate["model"]
    source = candidate["source"]
    model_dir = SOURCE_PATHS[source] / dataset / model
    if not model_dir.exists():
        # handle alias dir name for realtabformer
        if model == "realtabformer":
            alias_dir = SOURCE_PATHS[source] / dataset / "rtf"
            if alias_dir.exists():
                model_dir = alias_dir
            else:
                return None, "source_model_dir_missing"
        else:
            return None, "source_model_dir_missing"

    rec = resolve_consolidated(model_dir, dataset, model)
    if rec is not None:
        return rec, None

    rec = resolve_run_dir(model_dir, dataset, model)
    if rec is not None:
        return rec, None

    return None, "insufficient_clean_metadata_or_artifact"


def write_dataset_index(dataset_dir: Path, imported, skipped):
    index_json = {
        "dataset": dataset_dir.name,
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "imported": imported,
        "skipped": skipped,
    }
    with (dataset_dir / "INDEX.json").open("w", encoding="utf-8") as f:
        json.dump(index_json, f, ensure_ascii=False, indent=2)

    lines = [
        f"# {dataset_dir.name}",
        "",
        f"- imported: `{len(imported)}`",
        f"- skipped: `{len(skipped)}`",
        "",
        "## Imported",
    ]
    if imported:
        for item in imported:
            lines.append(
                f"- `{item['model']}` from `{item['source']}` -> `{item['source_ref']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Skipped"])
    if skipped:
        for item in skipped:
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
    candidates, skipped_multi = load_candidates()
    final_root = DEST_ROOT / "final_csv"
    manifest_root = DEST_ROOT / "manifests"
    final_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    imported_by_dataset = defaultdict(list)
    skipped_by_dataset = defaultdict(list)

    for item in skipped_multi:
        skipped_by_dataset[item["dataset"]].append(
            {
                "model": item["model"],
                "source": item["source"],
                "reason": item["reason"],
            }
        )

    imported_total = 0
    skipped_total = len(skipped_multi)

    for candidate in candidates:
        rec, reason = resolve_candidate(candidate)
        dataset = candidate["dataset"]
        model = candidate["model"]
        source = candidate["source"]

        if rec is None:
            skipped_total += 1
            skipped_by_dataset[dataset].append(
                {
                    "model": model,
                    "source": source,
                    "reason": reason,
                    "source_ref": repo_rel(SOURCE_PATHS[source] / dataset / model)
                    if (SOURCE_PATHS[source] / dataset / model).exists()
                    else repo_rel(SOURCE_PATHS[source] / dataset / "rtf"),
                }
            )
            continue

        dataset_dir = final_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        csv_dest = dataset_dir / f"{dataset}-{model}.csv"
        meta_dest = dataset_dir / f"{dataset}-{model}.json"
        shutil.copy2(rec["synthetic_path"], csv_dest)
        with meta_dest.open("w", encoding="utf-8") as f:
            json.dump(rec["metadata"], f, ensure_ascii=False, indent=2)

        imported_total += 1
        imported_by_dataset[dataset].append(
            {
                "model": model,
                "source": source,
                "source_ref": repo_rel(rec["source_ref"]),
                "imported_csv": repo_rel(csv_dest),
                "imported_metadata": repo_rel(meta_dest),
            }
        )

    datasets = sorted(set(imported_by_dataset) | set(skipped_by_dataset))
    for dataset in datasets:
        dataset_dir = final_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_index(
            dataset_dir,
            sorted(imported_by_dataset[dataset], key=lambda x: x["model"]),
            sorted(skipped_by_dataset[dataset], key=lambda x: x["model"]),
        )

    summary = {
        "rule": "only_single_source_without_codi_cdtd_goggle_allow_missing_time_skip_multi_run",
        "imported_total": imported_total,
        "skipped_total": skipped_total,
        "datasets_with_imports": sum(1 for items in imported_by_dataset.values() if items),
        "datasets_with_skips": sum(1 for items in skipped_by_dataset.values() if items),
    }
    with (manifest_root / "only_import_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (manifest_root / "only_import_summary.md").open("w", encoding="utf-8") as f:
        f.write("# only import summary\n\n")
        for k, v in summary.items():
            f.write(f"- {k}: `{v}`\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
