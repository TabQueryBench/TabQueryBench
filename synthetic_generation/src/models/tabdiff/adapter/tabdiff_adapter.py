"""
TabDiff 适配器：staging CSV + Features -> TabDiff data 布局；Docker 内训练/采样。
镜像见 docker_images.json；挂载 synthetic_benchmark/third_party/TabDiff。
"""

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


class TabDiffAdapter(BaseModelAdapter):
    _CONTAINER_TABDIFF = "/workspace/TabDiff"

    @property
    def model_name(self) -> str:
        return "tabdiff"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["tabdiff"]

    def _extra_volumes(self):
        root = get_synthetic_benchmark_root()
        return [(root / "third_party" / "TabDiff", self._CONTAINER_TABDIFF)]

    @staticmethod
    def _host_tabdiff_file_to_container(host_file: Path) -> str:
        root = (get_synthetic_benchmark_root() / "third_party" / "TabDiff").resolve()
        rel = Path(host_file).resolve().relative_to(root)
        return f"{TabDiffAdapter._CONTAINER_TABDIFF}/{rel.as_posix()}"

    @staticmethod
    def _tabdiff_runtime_dir(work_dir: Path) -> Path:
        return Path(work_dir) / "_tabdiff_runtime"

    @staticmethod
    def _train_overrides(epochs: Optional[int]) -> Dict[str, Any]:
        default_steps = int(epochs or 500)
        steps = max(2, int(os.environ.get("TABDIFF_STEPS", str(default_steps))))
        batch_size = max(1, int(os.environ.get("TABDIFF_BATCH_SIZE", "4096")))
        sample_batch_size = max(1, int(os.environ.get("TABDIFF_SAMPLE_BATCH_SIZE", str(min(batch_size, 2048)))))
        lr = float(os.environ.get("TABDIFF_LR", "0.001"))
        num_timesteps = max(1, int(os.environ.get("TABDIFF_NUM_TIMESTEPS", "50")))
        num_workers = max(0, int(os.environ.get("TABDIFF_NUM_WORKERS", "0")))
        return {
            "steps": steps,
            "batch_size": batch_size,
            "sample_batch_size": sample_batch_size,
            "lr": lr,
            "num_timesteps": num_timesteps,
            "num_workers": num_workers,
        }

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
            raise ValueError("tabdiff 需要 model_input_manifest（含 val_csv/test_csv）")
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
        runtime_dir = self._tabdiff_runtime_dir(work_dir)
        c_runtime = self._to_container_path(runtime_dir)

        train_cfg = self._train_overrides(epochs)
        steps = train_cfg["steps"]
        exp_name = kwargs.get("tabdiff_exp_name") or "adapter_learnable"

        script = textwrap.dedent(
            f"""
            import os, shutil, subprocess, sys
            td = r"{self._CONTAINER_TABDIFF}"
            name = r"{dataname}"
            src = r"{c_bundle}"
            rt = r"{c_runtime}"
            shutil.rmtree(rt, ignore_errors=True)

            def _ignore(_, names):
                skip = {{"__pycache__", "data", "synthetic", "result", "results", "ckpt"}}
                return [n for n in names if n in skip or n.endswith(".pyc")]

            shutil.copytree(td, rt, ignore=_ignore)

            def _replace_once(path, old, new):
                text = open(path, "r", encoding="utf-8").read()
                if old not in text:
                    raise RuntimeError(f"patch anchor not found in {{path}}")
                text = text.replace(old, new, 1)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)

            _replace_once(
                os.path.join(rt, "utils_train.py"),
                "        X_train_num, X_test_num = X_num['train'], X_num['test']\\n        X_train_cat, X_test_cat = X_cat['train'], X_cat['test']\\n        \\n        categories = src.get_categories(X_train_cat)\\n",
                "        X_train_num, X_test_num = X_num['train'], X_num['test']\\n        if X_cat is None:\\n            X_train_cat = np.empty((X_train_num.shape[0], 0), dtype=np.int64)\\n            X_test_cat = np.empty((X_test_num.shape[0], 0), dtype=np.int64)\\n            categories = []\\n        else:\\n            X_train_cat, X_test_cat = X_cat['train'], X_cat['test']\\n            categories = src.get_categories(X_train_cat)\\n",
            )
            _replace_once(
                os.path.join(rt, "utils_train.py"),
                "        X_cat = {{}} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None\\n        X_num = {{}} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None\\n        y = {{}} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None\\n",
                "        has_cat = os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))\\n        X_cat = {{}} if (has_cat or concat) else None\\n        X_num = {{}} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None\\n        y = {{}} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None\\n",
            )
            _replace_once(
                os.path.join(rt, "src", "data.py"),
                "        num_workers=1,\\n",
                "        num_workers=int(os.environ.get('TABDIFF_NUM_WORKERS', '0')),\\n",
            )
            _replace_once(
                os.path.join(rt, "src", "data.py"),
                "    loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)\\n",
                "    loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(os.environ.get('TABDIFF_NUM_WORKERS', '0')))\\n",
            )
            _replace_once(
                os.path.join(rt, "tabdiff", "main.py"),
                "    if os.environ.get(\\"TABDIFF_ADAPTER_TRAIN\\", \\"\\").strip() and args.mode == \\"train\\":\\n        raw_config[\\"train\\"][\\"main\\"][\\"check_val_every\\"] = int(raw_config[\\"train\\"][\\"main\\"][\\"steps\\"])\\n\\n    ## Load training data\\n",
                "    if os.environ.get(\\"TABDIFF_ADAPTER_TRAIN\\", \\"\\").strip() and args.mode == \\"train\\":\\n        raw_config[\\"train\\"][\\"main\\"][\\"check_val_every\\"] = int(raw_config[\\"train\\"][\\"main\\"][\\"steps\\"])\\n\\n    _train_batch = os.environ.get(\\"TABDIFF_BATCH_SIZE\\", \\"\\").strip() or os.environ.get(\\"TABDIFF_TRAIN_BATCH_SIZE\\", \\"\\").strip()\\n    if _train_batch:\\n        raw_config[\\"train\\"][\\"main\\"][\\"batch_size\\"] = max(1, int(_train_batch))\\n    _sample_batch = os.environ.get(\\"TABDIFF_SAMPLE_BATCH_SIZE\\", \\"\\").strip()\\n    if _sample_batch:\\n        raw_config[\\"sample\\"][\\"batch_size\\"] = max(1, int(_sample_batch))\\n    _train_lr = os.environ.get(\\"TABDIFF_LR\\", \\"\\").strip() or os.environ.get(\\"TABDIFF_LEARNING_RATE\\", \\"\\").strip()\\n    if _train_lr:\\n        raw_config[\\"train\\"][\\"main\\"][\\"lr\\"] = float(_train_lr)\\n    _num_timesteps = os.environ.get(\\"TABDIFF_NUM_TIMESTEPS\\", \\"\\").strip() or os.environ.get(\\"TABDIFF_TIMESTEPS\\", \\"\\").strip()\\n    if _num_timesteps:\\n        raw_config[\\"diffusion_params\\"][\\"num_timesteps\\"] = max(1, int(_num_timesteps))\\n\\n    ## Load training data\\n",
            )
            _replace_once(
                os.path.join(rt, "tabdiff", "main.py"),
                "        num_workers = 4,\\n",
                "        num_workers = int(os.environ.get('TABDIFF_NUM_WORKERS', '0')),\\n",
            )
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
            os.environ["TABDIFF_SMOKE_STEPS"] = "{steps}"
            os.environ["TABDIFF_STEPS"] = "{steps}"
            os.environ["TABDIFF_BATCH_SIZE"] = "{train_cfg["batch_size"]}"
            os.environ["TABDIFF_TRAIN_BATCH_SIZE"] = "{train_cfg["batch_size"]}"
            os.environ["TABDIFF_SAMPLE_BATCH_SIZE"] = "{train_cfg["sample_batch_size"]}"
            os.environ["TABDIFF_LR"] = "{train_cfg["lr"]}"
            os.environ["TABDIFF_LEARNING_RATE"] = "{train_cfg["lr"]}"
            os.environ["TABDIFF_NUM_TIMESTEPS"] = "{train_cfg["num_timesteps"]}"
            os.environ["TABDIFF_TIMESTEPS"] = "{train_cfg["num_timesteps"]}"
            os.environ["TABDIFF_NUM_WORKERS"] = "{train_cfg["num_workers"]}"
            os.environ["TABDIFF_ADAPTER_TRAIN"] = "1"
            subprocess.check_call([
                sys.executable, "-m", "tabdiff.main",
                "--dataname", name, "--mode", "train", "--gpu", "0",
                "--no_wandb", "--exp_name", r"{exp_name}",
            ])
            """
        )
        bridge = self._write_bridge_script(work_dir, "_tabdiff_train.py", script)
        c_bridge = self._to_container_path(bridge)
        train_log = work_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        extra_env = {"WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"}
        try:
            result = self._run_docker(["python", c_bridge], extra_env=extra_env)
            _write_docker_log(
                train_log,
                result.stdout or "",
                result.stderr or "",
                getattr(result, "bench_timing", None),
            )
        except Exception as e:
            _write_docker_log(
                train_log,
                getattr(e, "stdout", "") or "",
                getattr(e, "stderr", "") or "",
                getattr(e, "bench_timing", None),
            )
            raise

        meta = {
            "exp_name": exp_name,
            "dataname": dataname,
            "steps": steps,
            "batch_size": train_cfg["batch_size"],
            "sample_batch_size": train_cfg["sample_batch_size"],
            "lr": train_cfg["lr"],
            "num_timesteps": train_cfg["num_timesteps"],
            "num_workers": train_cfg["num_workers"],
            "runtime_dir": str(runtime_dir),
            "ckpt_dir": str(runtime_dir / "tabdiff" / "ckpt" / dataname / exp_name),
        }
        (work_dir / "tabdiff_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        marker_dir = work_dir / "models_tabdiff"
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker_pt = marker_dir / "trained.pt"
        marker_pt.write_text("tabdiff: see tabdiff_train_meta.json and third_party/TabDiff/tabdiff/ckpt/", encoding="utf-8")
        return {"model_path": marker_pt, "work_dir": work_dir}

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
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        work_dir = model_path.parent.parent if model_path.parent.name.startswith("models_") else model_path.parent
        meta_path = work_dir / "tabdiff_train_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError("缺少 tabdiff_train_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        exp_name = meta["exp_name"]
        dataname = meta.get("dataname") or "pipeline_ds"
        runtime_dir = Path(meta.get("runtime_dir") or self._tabdiff_runtime_dir(work_dir))
        c_runtime = self._to_container_path(runtime_dir)

        host_ckpt_dir = Path(meta.get("ckpt_dir") or runtime_dir / "tabdiff" / "ckpt" / dataname / exp_name)
        if not host_ckpt_dir.exists():
            host_ckpt_dir = (
                get_synthetic_benchmark_root()
                / "third_party"
                / "TabDiff"
                / "tabdiff"
                / "ckpt"
                / dataname
                / exp_name
            )
        ckpts = sorted(host_ckpt_dir.glob("model_*.pt"))
        if not ckpts:
            ckpts = sorted(host_ckpt_dir.glob("best*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"tabdiff 未找到 checkpoint: {host_ckpt_dir}")
        if runtime_dir in ckpts[-1].resolve().parents:
            c_ckpt = self._to_container_path(ckpts[-1])
        else:
            c_ckpt = self._host_tabdiff_file_to_container(ckpts[-1])
        c_bundle = self._to_container_path(work_dir / "tabular_bundle" / dataname)
        c_out = self._to_container_path(output_csv)

        script = textwrap.dedent(
            f"""
            import os, shutil, subprocess, sys
            td = r"{self._CONTAINER_TABDIFF}"
            name = r"{dataname}"
            src = r"{c_bundle}"
            rt = r"{c_runtime}"
            if not os.path.exists(rt):
                def _ignore(_, names):
                    skip = {{"__pycache__", "data", "synthetic", "result", "results", "ckpt"}}
                    return [n for n in names if n in skip or n.endswith(".pyc")]
                shutil.copytree(td, rt, ignore=_ignore)
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
            subprocess.check_call([
                sys.executable, "-m", "tabdiff.main",
                "--dataname", name, "--mode", "test", "--gpu", "0",
                "--no_wandb", "--exp_name", r"{exp_name}",
                "--ckpt_path", r"{c_ckpt}",
                "--num_samples_to_generate", str(int({num_rows})),
            ])
            # test() 写入 tabdiff/result/<dataname>/<exp>/<epoch>/samples.csv
            base = os.path.join(rt, "tabdiff", "result", name, r"{exp_name}")
            best = None
            best_t = -1.0
            for root, _, files in os.walk(base):
                if "samples.csv" in files:
                    p = os.path.join(root, "samples.csv")
                    t = os.path.getmtime(p)
                    if t > best_t:
                        best_t = t
                        best = p
            if not best:
                raise SystemExit("tabdiff: no samples.csv under " + base)
            shutil.copy(best, r"{c_out}")
            """
        )
        bridge = self._write_bridge_script(work_dir, "_tabdiff_gen.py", script)
        gen_log = work_dir / f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        extra_env = {
            "WANDB_MODE": "disabled",
            "TABDIFF_ADAPTER_SAMPLE_ONLY": "1",
        }
        for k in (
            "TABDIFF_STEPS",
            "TABDIFF_BATCH_SIZE",
            "TABDIFF_TRAIN_BATCH_SIZE",
            "TABDIFF_LR",
            "TABDIFF_LEARNING_RATE",
            "TABDIFF_NUM_TIMESTEPS",
            "TABDIFF_TIMESTEPS",
        ):
            v = (os.environ.get(k) or "").strip()
            if v:
                extra_env[k] = v
        try:
            result = self._run_docker(
                ["python", self._to_container_path(bridge)],
                extra_env=extra_env,
            )
            _write_docker_log(
                gen_log,
                result.stdout or "",
                result.stderr or "",
                getattr(result, "bench_timing", None),
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
