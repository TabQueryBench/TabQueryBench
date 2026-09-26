#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import shlex
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_REMOTE = "hku172"
DEFAULT_REMOTE_ROOT = "/data/jialinzhang/SynthesizePipeline-server/output-Benchmark-trainonly-v1"
DEFAULT_REMOTE_AUDIT = "/data/jialinzhang/SynthesizePipeline-server/scripts/build_hyperparam_audit.py"
DEFAULT_DEST = Path(
    "/Users/jialinzhang/Documents/HKUNAISS/SyntheticNips/SQLagent/SynOutput"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--remote", default=DEFAULT_REMOTE)
    p.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    p.add_argument("--remote-audit-script", default=DEFAULT_REMOTE_AUDIT)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument(
        "--datasets",
        default="",
        help="Comma-separated datasets. Empty means auto-discover all remote datasets.",
    )
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval-sec", type=int, default=60)
    return p.parse_args()


def ssh_base_cmd() -> List[str]:
    raw = os.environ.get("SSH", "ssh")
    return shlex.split(raw)


def run_ssh_checked(cmd: List[str], *, input_bytes: bytes | None = None, stdout=None) -> subprocess.CompletedProcess[bytes]:
    last_exc = None
    for attempt in range(2):
        try:
            return subprocess.run(
                cmd,
                input=input_bytes,
                stdout=stdout if stdout is not None else subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            last_exc = e
            stderr = (e.stderr or b"").decode("utf-8", "ignore").strip()
            if attempt == 0:
                print(f"[sync] ssh failed once, retrying: {' '.join(cmd)}", flush=True)
                if stderr:
                    print(stderr, flush=True)
                time.sleep(2)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def run_ssh_python(remote: str, code: str) -> subprocess.CompletedProcess[bytes]:
    return run_ssh_checked(
        ssh_base_cmd() + [remote, "python3", "-"],
        input_bytes=code.encode("utf-8"),
    )


def get_datasets(args: argparse.Namespace) -> List[str]:
    if args.datasets.strip():
        return [x.strip() for x in args.datasets.split(",") if x.strip()]

    code = textwrap.dedent(
        f"""
        from pathlib import Path
        root = Path({args.remote_root!r})
        items = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name != "_status"])
        import json, sys
        sys.stdout.write(json.dumps(items))
        """
    )
    cp = run_ssh_python(args.remote, code)
    return json.loads(cp.stdout.decode("utf-8"))


def remote_program(remote_root: str, remote_audit_script: str, datasets: List[str], mode: str) -> str:
    return textwrap.dedent(
        f"""
        from __future__ import annotations
        import hashlib
        import importlib.util
        import io
        import json
        import os
        import sys
        import tarfile
        from pathlib import Path

        REMOTE_ROOT = Path({remote_root!r})
        AUDIT_SCRIPT = Path({remote_audit_script!r})
        DATASETS = {datasets!r}
        MODE = {mode!r}

        spec = importlib.util.spec_from_file_location("build_hyperparam_audit", AUDIT_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        def load_json(path: Path):
            return json.loads(path.read_text(encoding="utf-8"))

        def hms(sec):
            if sec is None:
                return "-"
            sec = int(round(float(sec)))
            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            return f"{{h:02d}}:{{m:02d}}:{{s:02d}}"

        rows = mod.build_rows(REMOTE_ROOT, DATASETS)
        rows_by_key = {{(r["dataset"], r["model"], r["run_id"]): r for r in rows}}

        candidates = []
        for ds in DATASETS:
            ds_dir = REMOTE_ROOT / ds
            if not ds_dir.exists():
                continue
            for model_dir in sorted([p for p in ds_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                res = mod.latest_run(model_dir)
                # We need every successful run, not only latest. Scan all runtime_result.json files.
                rr_files = []
                root_rr = model_dir / "runtime_result.json"
                if root_rr.exists():
                    rr_files.append((model_dir, root_rr))
                for child in sorted([p for p in model_dir.iterdir() if p.is_dir() and p.name not in {{"staged", "public_gate"}}], key=lambda p: p.name):
                    rr = child / "runtime_result.json"
                    if rr.exists():
                        rr_files.append((child, rr))
                for run_dir, rr in rr_files:
                    try:
                        data = load_json(rr)
                    except Exception:
                        continue
                    if data.get("generate_status") != "success":
                        continue
                    synthetic_csv = data.get("artifacts", {{}}).get("synthetic_csv")
                    if not synthetic_csv or not Path(synthetic_csv).exists():
                        continue
                    key = (ds, model_dir.name, run_dir.name)
                    row = rows_by_key.get(key)
                    if row is None:
                        train_hp, gen_hp, source = mod.PARSERS.get(model_dir.name, lambda _: ({{}}, {{}}, "unknown"))(run_dir)
                        row = {{
                            "dataset": ds,
                            "model": model_dir.name,
                            "run_id": run_dir.name,
                            "train_hyperparams": json.dumps(train_hp, ensure_ascii=False, sort_keys=True),
                            "generate_hyperparams": json.dumps(gen_hp, ensure_ascii=False, sort_keys=True),
                            "hyperparam_source": source,
                            "train_duration_hms": hms(data.get("timings", {{}}).get("train", {{}}).get("duration_sec")),
                            "generate_duration_hms": hms(data.get("timings", {{}}).get("generate", {{}}).get("duration_sec")),
                        }}
                    fingerprint = json.dumps({{
                        "dataset": ds,
                        "model": model_dir.name,
                        "train_hyperparams": row.get("train_hyperparams", "{{}}"),
                        "generate_hyperparams": row.get("generate_hyperparams", "{{}}"),
                    }}, ensure_ascii=False, sort_keys=True)
                    candidates.append({{
                        "dataset": ds,
                        "model": model_dir.name,
                        "run_id": run_dir.name,
                        "run_dir": str(run_dir),
                        "runtime_result": str(rr),
                        "synthetic_csv": synthetic_csv,
                        "runtime_mtime": rr.stat().st_mtime,
                        "raw_train_status": data.get("train_status"),
                        "raw_generate_status": data.get("generate_status"),
                        "train_duration_sec": data.get("timings", {{}}).get("train", {{}}).get("duration_sec"),
                        "generate_duration_sec": data.get("timings", {{}}).get("generate", {{}}).get("duration_sec"),
                        "hyperparam_source": row.get("hyperparam_source"),
                        "train_hyperparams": row.get("train_hyperparams", "{{}}"),
                        "generate_hyperparams": row.get("generate_hyperparams", "{{}}"),
                        "fingerprint": fingerprint,
                    }})

        selected = {{}}
        for item in candidates:
            key = (item["dataset"], item["model"], item["fingerprint"])
            old = selected.get(key)
            if old is None or item["runtime_mtime"] > old["runtime_mtime"]:
                selected[key] = item

        items = sorted(selected.values(), key=lambda x: (x["dataset"], x["model"], x["run_id"]))
        summary = {{
            "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "remote_root": str(REMOTE_ROOT),
            "datasets": DATASETS,
            "selected_run_count": len(items),
            "runs": [],
        }}

        for item in items:
            run_dir = Path(item["run_dir"])
            dataset = item["dataset"]
            model = item["model"]
            run_id = item["run_id"]
            prefix = f"{{dataset}}__{{model}}__{{run_id}}"
            norm = {{
                "dataset": dataset,
                "model": model,
                "run_id": run_id,
                "remote_run_dir": str(run_dir),
                "hyperparam_source": item["hyperparam_source"],
                "train_hyperparams": json.loads(item["train_hyperparams"]),
                "generate_hyperparams": json.loads(item["generate_hyperparams"]),
                "train_status_raw": item["raw_train_status"],
                "generate_status_raw": item["raw_generate_status"],
                "train_status_normalized": (
                    "reused_train_success"
                    if item["raw_train_status"] == "skipped" and item["raw_generate_status"] == "success"
                    else item["raw_train_status"]
                ),
                "generate_status_normalized": item["raw_generate_status"],
                "train_duration_sec": item["train_duration_sec"],
                "generate_duration_sec": item["generate_duration_sec"],
                "train_duration_hms": hms(item["train_duration_sec"]),
                "generate_duration_hms": hms(item["generate_duration_sec"]),
                "time_note": (
                    "generate_only_success_from_existing_model"
                    if item["raw_train_status"] == "skipped" and item["raw_generate_status"] == "success"
                    else None
                ),
            }}
            files = {{
                "synthetic_data": [],
                "logs": [],
                "metadata": [],
            }}
            syn = Path(item["synthetic_csv"])
            if syn.exists():
                files["synthetic_data"].append((syn, f"{{dataset}}/{{model}}/synthetic_data/{{prefix}}__{{syn.name}}"))
            for pat in ["train_*.log", "gen_*.log"]:
                for p in sorted(run_dir.glob(pat)):
                    files["logs"].append((p, f"{{dataset}}/{{model}}/logs/{{prefix}}__{{p.name}}"))
            keep_pats = [
                "runtime_result.json", "run_config.json", "input_snapshot.json",
                "*meta*.json", "*manifest*.json", "*features*.json",
                "*report*.json", "*schema*.json", "*transforms*.json", "*.toml"
            ]
            seen = set()
            for pat in keep_pats:
                for p in sorted(run_dir.glob(pat)):
                    if p.name in seen:
                        continue
                    seen.add(p.name)
                    files["metadata"].append((p, f"{{dataset}}/{{model}}/metadata/{{prefix}}__{{p.name}}"))
            norm_name = f"{{dataset}}/{{model}}/metadata/{{prefix}}__normalized_record.json"
            item["files"] = {{
                k: [dest for _, dest in v] for k, v in files.items()
            }}
            item["normalized_record"] = norm_name
            item["_file_specs"] = files
            item["_norm_obj"] = norm
            summary["runs"].append({{
                k: v for k, v in item.items()
                if k not in {{"_file_specs", "_norm_obj"}}
            }})

        if MODE == "manifest":
            sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2))
            raise SystemExit(0)

        tf = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
        summary_bytes = json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")
        ti = tarfile.TarInfo("summary.json")
        ti.size = len(summary_bytes)
        tf.addfile(ti, io.BytesIO(summary_bytes))

        for item in items:
            norm_bytes = json.dumps(item["_norm_obj"], ensure_ascii=False, indent=2).encode("utf-8")
            nti = tarfile.TarInfo(item["normalized_record"])
            nti.size = len(norm_bytes)
            tf.addfile(nti, io.BytesIO(norm_bytes))
            for kind in ["synthetic_data", "logs", "metadata"]:
                for src, dest in item["_file_specs"][kind]:
                    if not src.exists() or not src.is_file():
                        continue
                    tf.add(str(src), arcname=dest, recursive=False)
        tf.close()
        """
    )


def load_remote_manifest(args: argparse.Namespace, datasets: List[str]) -> Dict[str, Any]:
    code = remote_program(args.remote_root, args.remote_audit_script, datasets, "manifest")
    cp = run_ssh_python(args.remote, code)
    return json.loads(cp.stdout.decode("utf-8"))


def manifest_hash(manifest: Dict[str, Any]) -> str:
    stable = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(stable).hexdigest()


def load_local_summary(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_readme(dest: Path, manifest: Dict[str, Any]) -> None:
    datasets = ", ".join(manifest.get("datasets", [])) or "auto-discovered datasets"
    readme = f"""# SynOutput

This folder contains synchronized successful runs copied from the remote benchmark output:

- remote root: `{manifest.get("remote_root")}`
- generated at: `{manifest.get("generated_at")}`
- datasets in this sync: `{datasets}`
- selected successful runs: `{manifest.get("selected_run_count", 0)}`

## What is included

- Only runs with `generate_status=success`
- Failed runs are skipped
- We do **not** download model weights/checkpoints
- For repeated experiments with different hyperparameters, each distinct hyperparameter setting is kept once
- If the same hyperparameter setting succeeded multiple times, only the latest successful run is kept

## Runs from this benchmark/trainonly workflow

The runs downloaded by this script are the newer benchmark/trainonly runs collected from the server workflow I have been helping with.
That includes:

- normal train+generate runs
- generate-only repair runs
- repeated hyperparameter experiments

## Time fields

We keep time information in two places:

1. raw copied metadata files such as `runtime_result.json`
2. normalized records generated locally:
   - `dataset/model/metadata/*__normalized_record.json`

These normalized records are important because some repaired runs reused an existing trained model.
In those cases the raw server record may show:

- `train_status=skipped`
- `generate_status=success`

For consistency, the normalized record marks these as reused-training success and preserves available train/generate durations in a uniform format.

## Hyperparameters

Hyperparameter information comes from:

- `run_config.json` for newer runs
- bridge scripts and metadata files for older runs

This means multi-hyperparameter experiments are preserved and can be distinguished later.

## Structure

- `dataset/model/synthetic_data/`
- `dataset/model/logs/`
- `dataset/model/metadata/`
- `summary.json`

Use `summary.json` as the inventory for everything downloaded by this sync.
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")


def write_summary(dest: Path, manifest: Dict[str, Any]) -> None:
    (dest / "summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def single_run_program(item: Dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        import io
        import json
        import tarfile
        from pathlib import Path

        item = {json.dumps(item, ensure_ascii=False)}
        run_dir = Path(item["remote_run_dir"])
        synthetic_csv = Path(item["synthetic_csv"])
        dataset = item["dataset"]
        model = item["model"]
        run_id = item["run_id"]
        prefix = f"{{dataset}}__{{model}}__{{run_id}}"

        norm = {{
            "dataset": dataset,
            "model": model,
            "run_id": run_id,
            "remote_run_dir": str(run_dir),
            "hyperparam_source": item["hyperparam_source"],
            "train_hyperparams": item["train_hyperparams"],
            "generate_hyperparams": item["generate_hyperparams"],
            "train_status_raw": item["train_status_raw"],
            "generate_status_raw": item["generate_status_raw"],
            "train_status_normalized": (
                "reused_train_success"
                if item["train_status_raw"] == "skipped" and item["generate_status_raw"] == "success"
                else item["train_status_raw"]
            ),
            "generate_status_normalized": item["generate_status_raw"],
            "train_duration_sec": item.get("train_duration_sec"),
            "generate_duration_sec": item.get("generate_duration_sec"),
            "train_duration_hms": item.get("train_duration_hms"),
            "generate_duration_hms": item.get("generate_duration_hms"),
            "time_note": (
                "generate_only_success_from_existing_model"
                if item["train_status_raw"] == "skipped" and item["generate_status_raw"] == "success"
                else None
            ),
        }}

        files = []
        if synthetic_csv.exists():
            files.append((synthetic_csv, f"{{dataset}}/{{model}}/synthetic_data/{{prefix}}__{{synthetic_csv.name}}"))
        for pat in ["train_*.log", "gen_*.log"]:
            for p in sorted(run_dir.glob(pat)):
                files.append((p, f"{{dataset}}/{{model}}/logs/{{prefix}}__{{p.name}}"))
        keep_pats = [
            "runtime_result.json", "run_config.json", "input_snapshot.json",
            "*meta*.json", "*manifest*.json", "*features*.json",
            "*report*.json", "*schema*.json", "*transforms*.json", "*.toml"
        ]
        seen = set()
        for pat in keep_pats:
            for p in sorted(run_dir.glob(pat)):
                if p.name in seen:
                    continue
                seen.add(p.name)
                files.append((p, f"{{dataset}}/{{model}}/metadata/{{prefix}}__{{p.name}}"))

        tf = tarfile.open(fileobj=__import__("sys").stdout.buffer, mode="w|")
        norm_name = item["normalized_record"]
        norm_bytes = json.dumps(norm, ensure_ascii=False, indent=2).encode("utf-8")
        nti = tarfile.TarInfo(norm_name)
        nti.size = len(norm_bytes)
        tf.addfile(nti, io.BytesIO(norm_bytes))
        for src, dest in files:
            if src.exists() and src.is_file():
                tf.add(str(src), arcname=dest, recursive=False)
        tf.close()
        """
    )


def sync_runs(args: argparse.Namespace, manifest: Dict[str, Any], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    runs = manifest.get("runs", [])
    total = len(runs)
    for idx, item in enumerate(runs, 1):
        label = f'{item["dataset"]}/{item["model"]}/{item["run_id"]}'
        print(f"[sync] downloading {idx}/{total} {label}", flush=True)
        code = single_run_program(item)
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with tmp_path.open("wb") as out:
                run_ssh_checked(
                    ssh_base_cmd() + [args.remote, "python3", "-"],
                    input_bytes=code.encode("utf-8"),
                    stdout=out,
                )
            with tarfile.open(tmp_path, "r") as tf:
                tf.extractall(dest)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    write_summary(dest, manifest)


def main() -> None:
    args = parse_args()
    while True:
        datasets = get_datasets(args)
        manifest = load_remote_manifest(args, datasets)
        current_hash = manifest_hash(manifest)
        args.dest.mkdir(parents=True, exist_ok=True)
        hash_path = args.dest / ".sync_manifest.sha256"
        old_hash = hash_path.read_text(encoding="utf-8").strip() if hash_path.exists() else ""
        if current_hash != old_hash:
            print(f"[sync] change detected, downloading {manifest.get('selected_run_count', 0)} runs", flush=True)
            sync_runs(args, manifest, args.dest)
            hash_path.write_text(current_hash + "\n", encoding="utf-8")
            write_readme(args.dest, manifest)
            print(f"[sync] updated {args.dest}", flush=True)
        else:
            write_summary(args.dest, manifest)
            write_readme(args.dest, manifest)
            print("[sync] no change", flush=True)
        if not args.watch:
            break
        time.sleep(max(1, args.interval_sec))


if __name__ == "__main__":
    main()
