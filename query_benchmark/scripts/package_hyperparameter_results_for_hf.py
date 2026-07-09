import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path.cwd()
SRC_ROOT = ROOT / "hyperparameter" / "output-Benchmark-trainonly-v1"
LOG_ROOT = ROOT / "hyperparameter" / "logs"
DEST_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\hyper_parameter")


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_synthetic_csv(run_dir: Path, runtime: dict):
    art = runtime.get("artifacts", {})
    synthetic_name = Path(art.get("synthetic_csv", "")).name
    if synthetic_name:
        direct = run_dir / synthetic_name
        if direct.exists():
            return direct
    csvs = sorted(p for p in run_dir.glob("*.csv") if p.is_file())
    if len(csvs) == 1:
        return csvs[0]
    if synthetic_name:
        for p in csvs:
            if p.name == synthetic_name:
                return p
    return None


def extract_train_hparams(run_dir: Path, run_config: dict):
    env = run_config.get("env_overrides", {})
    cleaned_env = {}
    if isinstance(env, dict):
        for k, v in env.items():
            if "GPU" in k or "GPUS" in k or k.startswith("BENCHMARK_"):
                if "GPUS" in k or k.endswith("_GPUS"):
                    continue
            cleaned_env[k] = v

    cli_args = run_config.get("cli_args", {})
    extras = {}
    if isinstance(cli_args, dict):
        if cli_args.get("epochs") is not None:
            extras["epochs"] = cli_args.get("epochs")

    # model-specific small meta files
    for meta_name in [
        "tabbyflow_train_meta.json",
        "tabdiff_train_meta.json",
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
                meta_json = read_json(meta_path)
                if isinstance(meta_json, dict):
                    extras.update(meta_json)
            except Exception:
                pass

    if not cleaned_env and not extras:
        return None
    merged = {}
    merged.update(cleaned_env)
    merged.update(extras)
    return merged


def collect_runs():
    runs_by_combo = defaultdict(list)
    for dataset_dir in sorted(p for p in SRC_ROOT.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            model = model_dir.name
            for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                runtime_path = run_dir / "runtime_result.json"
                run_config_path = run_dir / "run_config.json"
                if not runtime_path.exists():
                    continue
                runtime = read_json(runtime_path)
                run_config = read_json(run_config_path) if run_config_path.exists() else {}
                synthetic_csv = find_synthetic_csv(run_dir, runtime)
                train_hparams = extract_train_hparams(run_dir, run_config)
                record = {
                    "dataset": dataset,
                    "model": model,
                    "run_dir": run_dir,
                    "runtime": runtime,
                    "run_config": run_config,
                    "synthetic_csv": synthetic_csv,
                    "train_hyperparams": train_hparams,
                }
                runs_by_combo[(dataset, model)].append(record)
    return runs_by_combo


def write_model_index(model_dest: Path, imported, skipped):
    payload = {
        "dataset": model_dest.parent.name,
        "model": model_dest.name,
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "imported": imported,
        "skipped": skipped,
    }
    with (model_dest / "INDEX.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        f"# {model_dest.parent.name} / {model_dest.name}",
        "",
        f"- imported: `{len(imported)}`",
        f"- skipped: `{len(skipped)}`",
        "",
        "## Imported",
    ]
    if imported:
        for item in imported:
            lines.append(
                f"- `{item['variant']}` from `{item['source_run']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Skipped"])
    if skipped:
        for item in skipped:
            lines.append(
                f"- `{item['source_run']}` skipped: `{item['reason']}`"
            )
    else:
        lines.append("- none")
    with (model_dest / "INDEX.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_dataset_index(dataset_dest: Path, model_counts):
    payload = {
        "dataset": dataset_dest.name,
        "models": model_counts,
    }
    with (dataset_dest / "INDEX.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [f"# {dataset_dest.name}", "", "## Models"]
    for item in model_counts:
        lines.append(
            f"- `{item['model']}`: imported `{item['imported_count']}`, skipped `{item['skipped_count']}`"
        )
    with (dataset_dest / "INDEX.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    runs_by_combo = collect_runs()

    imported_total = 0
    skipped_total = 0
    dataset_model_counts = defaultdict(list)

    for (dataset, model), runs in sorted(runs_by_combo.items()):
        dataset_dest = DEST_ROOT / dataset
        model_dest = dataset_dest / model
        model_dest.mkdir(parents=True, exist_ok=True)

        imported = []
        skipped = []
        success_runs = []
        for rec in runs:
            runtime = rec["runtime"]
            if runtime.get("generate_status") == "success" and rec["synthetic_csv"] is not None:
                success_runs.append(rec)
            else:
                skipped.append(
                    {
                        "source_run": repo_rel(rec["run_dir"]),
                        "reason": f"generate_status={runtime.get('generate_status')}",
                    }
                )
                skipped_total += 1

        for idx, rec in enumerate(success_runs, start=1):
            variant = f"hp{idx:02d}"
            csv_dest = model_dest / f"{dataset}-{model}-{variant}.csv"
            meta_dest = model_dest / f"{dataset}-{model}-{variant}.json"

            shutil.copy2(rec["synthetic_csv"], csv_dest)

            timings = rec["runtime"].get("timings", {})
            metadata = {
                "dataset": dataset,
                "model": model,
                "variant": variant,
                "train_duration_sec": timings.get("train", {}).get("duration_sec"),
                "generate_duration_sec": timings.get("generate", {}).get("duration_sec"),
                "train_hyperparams": rec["train_hyperparams"],
            }
            with meta_dest.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            imported.append(
                {
                    "variant": variant,
                    "source_run": repo_rel(rec["run_dir"]),
                    "imported_csv": repo_rel(csv_dest),
                    "imported_metadata": repo_rel(meta_dest),
                }
            )
            imported_total += 1

        write_model_index(model_dest, imported, skipped)
        dataset_model_counts[dataset].append(
            {
                "model": model,
                "imported_count": len(imported),
                "skipped_count": len(skipped),
            }
        )

    for dataset, model_counts in dataset_model_counts.items():
        write_dataset_index(DEST_ROOT / dataset, sorted(model_counts, key=lambda x: x["model"]))

    summary = {
        "dataset_count": len(dataset_model_counts),
        "model_combo_count": sum(len(v) for v in dataset_model_counts.values()),
        "imported_total": imported_total,
        "skipped_total": skipped_total,
        "source_root": repo_rel(SRC_ROOT),
        "log_root": repo_rel(LOG_ROOT),
    }
    manifests = DEST_ROOT / "manifests"
    manifests.mkdir(exist_ok=True)
    with (manifests / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (manifests / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# hyperparameter summary\n\n")
        for k, v in summary.items():
            f.write(f"- {k}: `{v}`\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
