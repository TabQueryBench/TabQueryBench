"""Environment check before running synthetic generation.

  PYTHONPATH=Synthesizing/code/src python -m core.doctor [--models all] [--gpu-test]

Checks host Python dependencies, Docker access, model images, GPUs, prepared datasets,
and model-specific readiness (an adapter module may define ``check_ready() -> (ok, message)``,
e.g. TabPFGen weights / Hugging Face token).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from core.runner.config import (
    MODEL_DOCKER_MAP,
    SUPPORTED_MODELS,
    get_new_tabular_dataset_root,
    get_output_base,
)

OK, WARN, FAIL = "ok", "warn", "FAIL"


def _run(cmd: List[str], timeout: int = 60) -> Tuple[int, str]:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return cp.returncode, (cp.stdout + cp.stderr).strip()
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def check_python() -> List[Tuple[str, str, str]]:
    rows = []
    for mod in ("pandas", "numpy", "yaml"):
        spec = importlib.util.find_spec(mod)
        rows.append((f"python: {mod}", OK if spec else FAIL, "" if spec else "pip install pandas numpy pyyaml"))
    return rows


def check_docker(models: List[str], gpu_test: bool) -> List[Tuple[str, str, str]]:
    rows = []
    if not shutil.which("docker"):
        return [("docker CLI", FAIL, "docker not installed")]
    code, out = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if code != 0:
        return [("docker daemon", FAIL, out.splitlines()[-1] if out else "cannot talk to docker (in docker group?)")]
    rows.append(("docker daemon", OK, f"server {out}"))
    for model in models:
        image = MODEL_DOCKER_MAP.get(model, "")
        code, out = _run(["docker", "image", "inspect", "--format", "{{.Id}}", image])
        rows.append((f"image {model}", OK if code == 0 else FAIL, image if code == 0 else f"{image} missing (docker load / tag it, or set BENCHMARK_{model.upper()}_IMAGE)"))
    if gpu_test:
        image = MODEL_DOCKER_MAP.get("ctgan") or next(iter(MODEL_DOCKER_MAP.values()))
        code, out = _run(
            ["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "sh", image, "-c",
             "python -c 'import torch; print(torch.cuda.device_count(), torch.cuda.is_available())'"],
            timeout=300,
        )
        rows.append(("GPU inside container", OK if code == 0 and "True" in out else FAIL, out.splitlines()[-1] if out else ""))
    return rows


def check_gpus() -> List[Tuple[str, str, str]]:
    code, out = _run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"])
    if code != 0:
        return [("nvidia-smi", WARN, "no NVIDIA GPU visible; GPU models will fail")]
    return [(f"GPU {line.split(',')[0].strip()}", OK, line.strip()) for line in out.splitlines() if line.strip()]


def check_datasets() -> List[Tuple[str, str, str]]:
    from core.runner.batch import list_prepared_datasets

    root = get_new_tabular_dataset_root()
    ds = list_prepared_datasets(root)
    rows = [("dataset root", OK if root.is_dir() else WARN, str(root))]
    rows.append(("prepared datasets", OK if ds else WARN, ", ".join(ds) if ds else "none; run python -m core.prepare run <dataset_dir>"))
    out = get_output_base()
    rows.append(("output root", OK, str(out)))
    return rows


def check_models(models: List[str]) -> List[Tuple[str, str, str]]:
    from core.runner.adapters import _load_adapter_class

    rows = []
    for model in models:
        try:
            # Goes through the registry so the per-model compatibility module aliases are installed.
            module = importlib.import_module(_load_adapter_class(model).__module__)
        except Exception as exc:  # pragma: no cover - reported to the user
            rows.append((f"adapter {model}", FAIL, f"import failed: {exc}"))
            continue
        check = getattr(module, "check_ready", None)
        if check is None:
            rows.append((f"adapter {model}", OK, "imported"))
            continue
        try:
            ready, message = check()
        except Exception as exc:
            ready, message = False, f"check_ready raised: {exc}"
        rows.append((f"adapter {model}", OK if ready else FAIL, message))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m core.doctor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="all")
    p.add_argument("--gpu-test", action="store_true", help="run a tiny CUDA check inside a model image")
    args = p.parse_args(argv)
    models = list(SUPPORTED_MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]

    from core.runner.adapters import get_adapter  # noqa: F401  (registers adapter module aliases)

    rows = check_python() + check_docker(models, args.gpu_test) + check_gpus() + check_datasets() + check_models(models)
    width = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"{name:{width}}  {status:4}  {detail}")
    failed = [r for r in rows if r[1] == FAIL]
    print(f"\n{len(failed)} failing check(s)" if failed else "\nall required checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
