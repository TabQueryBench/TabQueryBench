import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"D:\dpan\Uni\Project\HKUNAISS\SQLagent")
SRC_ROOT = ROOT / "remote-output-Benchmark-trainonly-v1"
DEST_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\time_cost")
IGNORE_TOP = {"Statistic", "_status", ".gitkeep"}
KEEP_DATASETS = {"c2", "c7", "c14", "m4", "m6", "m8", "n3", "n6", "n11"}


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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


def extract_train_hparams(run_dir: Path, run_config: dict):
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


def write_model_index(model_dest: Path, records):
    payload = {
        "dataset": model_dest.parent.name,
        "model": model_dest.name,
        "run_count": len(records),
        "runs": records,
    }
    with (model_dest / "INDEX.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        f"# {model_dest.parent.name} / {model_dest.name}",
        "",
        f"- run_count: `{len(records)}`",
        "",
        "## Runs",
    ]
    if records:
        for rec in records:
            lines.append(f"- `{rec['variant']}` from `{rec['source_run']}`")
    else:
        lines.append("- none")
    with (model_dest / "INDEX.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_dataset_index(dataset_dest: Path, model_counts):
    payload = {"dataset": dataset_dest.name, "models": model_counts}
    with (dataset_dest / "INDEX.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    lines = [f"# {dataset_dest.name}", "", "## Models"]
    for item in model_counts:
        lines.append(f"- `{item['model']}`: `{item['run_count']}` runs")
    with (dataset_dest / "INDEX.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if DEST_ROOT.exists():
        for child in DEST_ROOT.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    imported_total = 0
    dataset_model_counts = defaultdict(list)

    for dataset_dir in sorted(
        p
        for p in SRC_ROOT.iterdir()
        if p.is_dir() and p.name not in IGNORE_TOP and p.name in KEEP_DATASETS
    ):
        dataset = dataset_dir.name
        dataset_dest = DEST_ROOT / dataset
        dataset_dest.mkdir(parents=True, exist_ok=True)

        for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            model = model_dir.name
            model_dest = dataset_dest / model
            model_dest.mkdir(parents=True, exist_ok=True)

            run_dirs = sorted(p for p in model_dir.iterdir() if p.is_dir())
            success_entries = []
            for run_dir in run_dirs:
                runtime_path = run_dir / "runtime_result.json"
                if not runtime_path.exists():
                    continue
                try:
                    runtime = read_json(runtime_path)
                except Exception:
                    continue
                run_config_path = run_dir / "run_config.json"
                if run_config_path.exists():
                    try:
                        run_config = read_json(run_config_path)
                    except Exception:
                        run_config = {}
                else:
                    run_config = {}
                synthetic_csv = find_synthetic_csv(run_dir, runtime)
                train_hparams = extract_train_hparams(run_dir, run_config)
                if runtime.get("generate_status") == "success" and synthetic_csv is not None:
                    success_entries.append(
                        {
                            "run_dir": run_dir,
                            "runtime": runtime,
                            "train_hyperparams": train_hparams,
                            "synthetic_csv": synthetic_csv,
                        }
                    )

            records = []
            for idx, rec in enumerate(success_entries, start=1):
                variant = f"r{idx:02d}"
                json_dest = model_dest / f"{dataset}-{model}-{variant}.json"
                csv_dest = model_dest / f"{dataset}-{model}-{variant}.csv"
                timings = rec["runtime"].get("timings", {})
                payload = {
                    "dataset": dataset,
                    "model": model,
                    "variant": variant,
                    "train_status": rec["runtime"].get("train_status"),
                    "generate_status": rec["runtime"].get("generate_status"),
                    "train_duration_sec": timings.get("train", {}).get("duration_sec"),
                    "generate_duration_sec": timings.get("generate", {}).get("duration_sec"),
                    "train_hyperparams": rec["train_hyperparams"],
                }
                with json_dest.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                shutil.copy2(rec["synthetic_csv"], csv_dest)
                records.append(
                    {
                        "variant": variant,
                        "source_run": repo_rel(rec["run_dir"]),
                        "json": repo_rel(json_dest),
                        "csv": repo_rel(csv_dest),
                    }
                )
                imported_total += 1

            write_model_index(model_dest, records)
            dataset_model_counts[dataset].append(
                {"model": model, "run_count": len(records)}
            )

        write_dataset_index(dataset_dest, sorted(dataset_model_counts[dataset], key=lambda x: x["model"]))

    manifests = DEST_ROOT / "manifests"
    manifests.mkdir(exist_ok=True)
    summary = {
        "dataset_count": len(dataset_model_counts),
        "model_combo_count": sum(len(v) for v in dataset_model_counts.values()),
        "successful_run_total": imported_total,
        "source_root": repo_rel(SRC_ROOT),
    }
    with (manifests / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (manifests / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# remote time cost summary\n\n")
        for k, v in summary.items():
            f.write(f"- {k}: `{v}`\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
