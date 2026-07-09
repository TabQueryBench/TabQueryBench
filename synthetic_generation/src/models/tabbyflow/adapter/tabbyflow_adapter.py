"""TabbyFlow (ef-vfm) 适配器；与 TabDiff 共用数据准备，工作目录为 /workspace/ef-vfm。"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base_adapter import BaseModelAdapter, _write_docker_log
from .pipeline_npy_bundle import prepare_tabular_npy_bundle, tabular_bundle_slug_from_manifest
from ..config import MODEL_DOCKER_MAP, get_synthetic_benchmark_root


class TabbyFlowAdapter(BaseModelAdapter):
    _CONTAINER_ROOT = "/workspace/ef-vfm"

    @property
    def model_name(self) -> str:
        return "tabbyflow"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["tabbyflow"]

    def _extra_volumes(self):
        r = get_synthetic_benchmark_root()
        return [(r / "third_party" / "ef-vfm", self._CONTAINER_ROOT)]

    @staticmethod
    def _host_efvfm_file_to_container(host_file: Path) -> str:
        root = (get_synthetic_benchmark_root() / "third_party" / "ef-vfm").resolve()
        rel = Path(host_file).resolve().relative_to(root)
        return f"/workspace/ef-vfm/{rel.as_posix()}"

    @staticmethod
    def _efvfm_runtime_dir(work_dir: Path) -> Path:
        return Path(work_dir) / "_efvfm_runtime"

    def train(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        epochs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        csv_path, json_path = self._resolve_model_inputs(csv_path, json_path, kwargs)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._validate_model_inputs(csv_path, json_path, require_target=True, strict_numeric_cast=True)
        if not kwargs.get("model_input_manifest"):
            raise ValueError("tabbyflow 需要 model_input_manifest")
        with open(kwargs["model_input_manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        dataname = tabular_bundle_slug_from_manifest(manifest)
        prepare_tabular_npy_bundle(
            work_dir,
            Path(manifest["train_csv"]),
            Path(manifest["val_csv"]),
            Path(manifest["test_csv"]),
            json_path,
            slug=dataname,
            task_type=manifest.get("task_type"),
            target_column=manifest.get("target_column"),
        )
        c_bundle = self._to_container_path(work_dir / "tabular_bundle" / dataname)
        runtime_dir = self._efvfm_runtime_dir(work_dir)
        c_runtime = self._to_container_path(runtime_dir)
        steps = max(1, int(epochs or 500))
        exp_name = kwargs.get("tabbyflow_exp_name") or "adapter_efvfm"

        script = textwrap.dedent(
            f"""
            import os, shutil, subprocess, sys
            root = r"{self._CONTAINER_ROOT}"
            rt = r"{c_runtime}"
            name = r"{dataname}"
            src = r"{c_bundle}"

            shutil.rmtree(rt, ignore_errors=True)

            def _ignore(_, names):
                skip = {{"__pycache__", "data", "synthetic", "result", "results", "ckpt"}}
                return [n for n in names if n in skip or n.endswith(".pyc")]

            shutil.copytree(root, rt, ignore=_ignore)
            pkg_cfg = os.path.join(rt, "ef_vfm", "configs")
            root_cfg = os.path.join(rt, "configs")
            if not os.path.isdir(root_cfg) and os.path.isdir(pkg_cfg):
                shutil.copytree(pkg_cfg, root_cfg)
            dst_data = os.path.join(rt, "data", name)
            dst_syn = os.path.join(rt, "synthetic", name)
            shutil.rmtree(dst_data, ignore_errors=True)
            os.makedirs(os.path.dirname(dst_data), exist_ok=True)
            shutil.copytree(src, dst_data)
            os.makedirs(dst_syn, exist_ok=True)
            for fn in ("real.csv", "test.csv", "val.csv"):
                shutil.copy(os.path.join(src, fn), os.path.join(dst_syn, fn))
            os.chdir(rt)
            os.environ["PYTHONPATH"] = rt + os.pathsep + os.environ.get("PYTHONPATH", "")
            os.environ["EFVFM_SMOKE_STEPS"] = "{steps}"
            os.environ["EFVFM_ADAPTER_TRAIN"] = "1"
            os.environ.setdefault("EFVFM_TRAIN_BATCH_SIZE", "64")
            os.environ.setdefault("EFVFM_SAMPLE_BATCH_SIZE", "64")
            os.environ.setdefault("EFVFM_EVAL_NUM_SAMPLES", "512")
            os.environ.setdefault("EFVFM_ODE_FALLBACK", "1")
            os.environ.setdefault("EFVFM_RK4_STEPS", "32")
            subprocess.check_call([
                sys.executable, os.path.join(rt, "main.py"),
                "--dataname", name, "--mode", "train", "--gpu", "0",
                "--no_wandb", "--exp_name", r"{exp_name}",
            ])
            """
        )
        bridge = self._write_bridge_script(work_dir, "_tabbyflow_train.py", script)
        log = work_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        extra_env = {"WANDB_MODE": "disabled"}
        for key, value in os.environ.items():
            if key.startswith("EFVFM_"):
                extra_env[key] = value
        try:
            r = self._run_docker(["python", self._to_container_path(bridge)], extra_env=extra_env)
            _write_docker_log(
                log,
                r.stdout or "",
                r.stderr or "",
                getattr(r, "bench_timing", None),
            )
        except Exception as e:
            _write_docker_log(
                log,
                getattr(e, "stdout", "") or "",
                getattr(e, "stderr", "") or "",
                getattr(e, "bench_timing", None),
            )
            raise
        (work_dir / "tabbyflow_train_meta.json").write_text(
            json.dumps(
                {
                    "exp_name": exp_name,
                    "dataname": dataname,
                    "steps": steps,
                    "runtime_dir": str(runtime_dir),
                    "ckpt_dir": str(runtime_dir / "ckpt" / dataname / exp_name),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        md = work_dir / "models_tabbyflow"
        md.mkdir(parents=True, exist_ok=True)
        mp = md / "trained.pt"
        mp.write_text("tabbyflow: see tabbyflow_train_meta.json", encoding="utf-8")
        return {"model_path": mp, "work_dir": work_dir}

    def generate(
        self,
        model_path: Path,
        output_csv: Path,
        num_rows: int = 1000,
        csv_path: Optional[Path] = None,
        json_path: Optional[Path] = None,
        **kwargs,
    ) -> Path:
        model_path, output_csv = Path(model_path), Path(output_csv)
        if model_path.is_dir():
            work_dir = model_path
        else:
            work_dir = model_path.parent.parent if model_path.parent.name.startswith("models_") else model_path.parent

        meta_path = work_dir / "tabbyflow_train_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            exp_name = meta["exp_name"]
            dataname = meta.get("dataname") or "pipeline_ds"
            runtime_dir = Path(meta.get("runtime_dir") or self._efvfm_runtime_dir(work_dir))
        else:
            manifest_path = kwargs.get("model_input_manifest")
            if not manifest_path:
                raise FileNotFoundError(f"tabbyflow缺少 train meta，且未提供 model_input_manifest: {work_dir}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            dataname = tabular_bundle_slug_from_manifest(manifest)
            exp_name = kwargs.get("tabbyflow_exp_name") or "adapter_efvfm"
            runtime_dir = self._efvfm_runtime_dir(work_dir)
            bundle_dir = work_dir / "tabular_bundle" / dataname
            if not bundle_dir.exists():
                prepare_tabular_npy_bundle(
                    work_dir,
                    Path(manifest["train_csv"]),
                    Path(manifest["val_csv"]),
                    Path(manifest["test_csv"]),
                    Path(json_path) if json_path is not None else Path(manifest["features_json"]),
                    slug=dataname,
                    task_type=manifest.get("task_type"),
                    target_column=manifest.get("target_column"),
                )
            meta_path.write_text(
                json.dumps(
                    {
                        "exp_name": exp_name,
                        "dataname": dataname,
                        "runtime_dir": str(runtime_dir),
                        "ckpt_dir": str(runtime_dir / "ckpt" / dataname / exp_name),
                        "restored_for_generate_only": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        host_ckpt_dir = Path(meta.get("ckpt_dir") or runtime_dir / "ef_vfm" / "ckpt" / dataname / exp_name)
        if not host_ckpt_dir.exists():
            host_ckpt_dir = (
                get_synthetic_benchmark_root()
                / "third_party"
                / "ef-vfm"
                / "ef_vfm"
                / "ckpt"
                / dataname
                / exp_name
            )
        ckpts = sorted(host_ckpt_dir.glob("model_*.pt"))
        if not ckpts:
            ckpts = sorted(host_ckpt_dir.glob("best*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"tabbyflow无 checkpoint: {host_ckpt_dir}")
        if runtime_dir.resolve() in ckpts[-1].resolve().parents:
            c_ckpt = self._to_container_path(ckpts[-1])
        else:
            c_ckpt = self._host_efvfm_file_to_container(ckpts[-1])
        c_bundle = self._to_container_path(work_dir / "tabular_bundle" / dataname)
        c_out = self._to_container_path(output_csv)
        c_runtime = self._to_container_path(runtime_dir)

        script = textwrap.dedent(
            f"""
            import os, shutil, subprocess, sys
            root = r"{self._CONTAINER_ROOT}"
            rt = r"{c_runtime}"
            name = r"{dataname}"
            src = r"{c_bundle}"

            if not os.path.exists(rt):
                def _ignore(_, names):
                    skip = {{"__pycache__", "data", "synthetic", "result", "results", "ckpt"}}
                    return [n for n in names if n in skip or n.endswith(".pyc")]
                shutil.copytree(root, rt, ignore=_ignore)

            dst_data = os.path.join(rt, "data", name)
            shutil.rmtree(dst_data, ignore_errors=True)
            os.makedirs(os.path.dirname(dst_data), exist_ok=True)
            shutil.copytree(src, dst_data)
            dst_syn = os.path.join(rt, "synthetic", name)
            os.makedirs(dst_syn, exist_ok=True)
            for fn in ("real.csv", "test.csv", "val.csv"):
                shutil.copy(os.path.join(src, fn), os.path.join(dst_syn, fn))
            os.chdir(rt)
            os.environ["PYTHONPATH"] = rt + os.pathsep + os.environ.get("PYTHONPATH", "")
            os.environ.setdefault("EFVFM_SAMPLE_BATCH_SIZE", "64")
            os.environ.setdefault("EFVFM_ODE_FALLBACK", "1")
            os.environ.setdefault("EFVFM_RK4_STEPS", "32")
            subprocess.check_call([
                sys.executable, os.path.join(rt, "main.py"),
                "--dataname", name, "--mode", "test", "--gpu", "0",
                "--no_wandb", "--exp_name", r"{exp_name}",
                "--ckpt_path", r"{c_ckpt}",
                "--num_samples_to_generate", str(int({num_rows})),
            ])
            search_roots = [
                os.path.join(rt, "result", name, r"{exp_name}"),
                os.path.join(rt, "ef_vfm", "result", name, r"{exp_name}"),
            ]
            best = None
            best_t = -1.0
            for base in search_roots:
                if not os.path.isdir(base):
                    continue
                for r, _, files in os.walk(base):
                    if "samples.csv" in files:
                        p = os.path.join(r, "samples.csv")
                        t = os.path.getmtime(p)
                        if t > best_t:
                            best_t, best = t, p
            if not best:
                raise SystemExit("tabbyflow: no samples.csv in " + " | ".join(search_roots))
            shutil.copy(best, r"{c_out}")
            """
        )
        bridge = self._write_bridge_script(work_dir, "_tabbyflow_gen.py", script)
        gen_log = work_dir / f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            extra_env = {"WANDB_MODE": "disabled", "EFVFM_ADAPTER_SAMPLE_ONLY": "1"}
            for key, value in os.environ.items():
                if key.startswith("EFVFM_"):
                    extra_env[key] = value
            r = self._run_docker(
                ["python", self._to_container_path(bridge)],
                extra_env=extra_env,
            )
            _write_docker_log(
                gen_log,
                r.stdout or "",
                r.stderr or "",
                getattr(r, "bench_timing", None),
            )
        except Exception as e:
            _write_docker_log(
                gen_log,
                getattr(e, "stdout", "") or "",
                getattr(e, "stderr", "") or "",
                getattr(e, "bench_timing", None),
            )
            raise
        return self._postprocess_generated_csv(
            output_csv, csv_path, json_path, num_rows, **kwargs
        )
