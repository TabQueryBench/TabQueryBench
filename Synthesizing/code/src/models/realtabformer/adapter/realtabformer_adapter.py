"""
REaLTabFormer 模型适配器

通过 Docker 调用配置的 REaLTabFormer 镜像。

GPU：容器内 HF Trainer 看到多卡会走 DataParallel（曾触发 NCCL Error 2），因此必须单卡：
- ``BENCHMARK_REALTABFORMER_GPUS=device=N``：docker 只暴露该卡，容器内不再覆盖 CUDA_VISIBLE_DEVICES；
- ``REALTABFORMER_CUDA_VISIBLE_DEVICES=<idx>``：显式指定容器内可见卡（优先级最高）；
- 两者都没设（``--gpus all``）：退回容器内 ``CUDA_VISIBLE_DEVICES=0`` 并打印警告。

训练超参（env 可覆盖，括号内为默认）：
- ``REALTABFORMER_BATCH_SIZE`` (64)、``REALTABFORMER_GRAD_ACCUM`` (1)：有效 batch 64（上游 8x4=32）；
- ``REALTABFORMER_EPOCHS``：显式 epochs；未设且 runner 未传时按行数缩放
  ``ceil(REALTABFORMER_TARGET_STEPS(20000) / (rows // eff_batch))``，夹在
  ``[REALTABFORMER_MIN_EPOCHS(3), REALTABFORMER_MAX_EPOCHS(100)]``；
- ``REALTABFORMER_SAVE_STEPS`` (2000)、``REALTABFORMER_LOGGING_STEPS`` (100)、
  ``REALTABFORMER_GEN_BATCH``（默认按列数：≤40 列 512、≤80 列 128、≤160 列 32、更宽 16，防止生成时 KV 缓存 OOM）、
  ``REALTABFORMER_NUMERIC_MAX_LEN`` (10，浮点大数自动上调)。
- 仍使用 ``n_critic=0``（不做 sensitivity 早停，避免大表上反复采样评估）。
"""

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import load_features_json
from core.runner.config import MODEL_DOCKER_MAP


class REaLTabFormerAdapter(BaseModelAdapter):
    @staticmethod
    def _resolve_numeric_max_len(kwargs: Dict[str, Any]) -> Optional[int]:
        raw = kwargs.get("numeric_max_len")
        if raw in (None, ""):
            raw = os.getenv("REALTABFORMER_NUMERIC_MAX_LEN")
        if raw in (None, ""):
            return None
        return int(raw)

    @property
    def model_name(self) -> str:
        return "realtabformer"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["realtabformer"]

    # HOME=/tmp 防止 --user 模式下 HuggingFace cache 写入失败
    # CUDA_VISIBLE_DEVICES 由 _docker_env() 按 GPU 选择决定（见模块 docstring）。
    _EXTRA_ENV = {
        "NCCL_P2P_DISABLE": "1",
        "HOME": "/tmp",
        # 优先导入挂载到 /work 的补丁版 realtabformer 包，避免依赖镜像内旧代码。
        "PYTHONPATH": "/work/src/models/realtabformer/upstream",
    }

    def _docker_env(self) -> Dict[str, str]:
        env = dict(self._EXTRA_ENV)
        explicit = (os.environ.get("REALTABFORMER_CUDA_VISIBLE_DEVICES") or "").strip()
        gpus = (os.environ.get(self._gpu_env_key()) or "all").strip()
        if explicit:
            env["CUDA_VISIBLE_DEVICES"] = explicit
        elif not gpus or gpus == "all":
            env["CUDA_VISIBLE_DEVICES"] = "0"
            print(
                f"[realtabformer] WARNING: {self._gpu_env_key()} unset/all -> container sees all GPUs; "
                "pinning CUDA_VISIBLE_DEVICES=0 to avoid multi-GPU DataParallel. Set "
                f"{self._gpu_env_key()}=device=N (or REALTABFORMER_CUDA_VISIBLE_DEVICES) to choose a GPU."
            )
        return env

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        raw = (os.environ.get(key) or "").strip()
        return int(raw) if raw else int(default)

    def _resolve_train_hparams(self, epochs: Optional[int], csv_path: Path) -> Dict[str, int]:
        batch = max(self._env_int("REALTABFORMER_BATCH_SIZE", 64), 1)
        accum = max(self._env_int("REALTABFORMER_GRAD_ACCUM", 1), 1)
        if not epochs:
            raw = (os.environ.get("REALTABFORMER_EPOCHS") or "").strip()
            if raw:
                epochs = int(raw)
        if not epochs:
            import pandas as pd

            n_rows = len(pd.read_csv(csv_path, usecols=[0], encoding="utf-8-sig", low_memory=False))
            target = self._env_int("REALTABFORMER_TARGET_STEPS", 20000)
            lo = self._env_int("REALTABFORMER_MIN_EPOCHS", 3)
            hi = self._env_int("REALTABFORMER_MAX_EPOCHS", 100)
            steps_per_epoch = max(n_rows // (batch * accum), 1)
            epochs = int(min(max(int(math.ceil(target / steps_per_epoch)), lo), hi))
            print(
                f"[realtabformer] auto epochs={epochs} (rows={n_rows}, eff_batch={batch * accum}, "
                f"target_steps={target}, clamp=[{lo},{hi}])"
            )
        return {
            "epochs": int(epochs),
            "batch_size": batch,
            "grad_accum": accum,
            "save_steps": max(self._env_int("REALTABFORMER_SAVE_STEPS", 2000), 1),
            "logging_steps": max(self._env_int("REALTABFORMER_LOGGING_STEPS", 100), 1),
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
        self._validate_model_inputs(
            csv_path=csv_path,
            json_path=json_path,
            require_target=True,
            strict_numeric_cast=False,
        )
        hp = self._resolve_train_hparams(epochs, csv_path)
        epochs = hp["epochs"]
        numeric_max_len = self._resolve_numeric_max_len(kwargs)
        models_dir = work_dir / f"models_{epochs}epochs"
        models_dir.mkdir(parents=True, exist_ok=True)
        resume = bool(kwargs.get("resume", False))
        resume_checkpoint = kwargs.get("resume_checkpoint")

        c_csv = self._to_container_path(csv_path)
        c_json = ""
        if json_path.exists():
            # 兼容新版 dataset_profile.json：先归一为旧版 features 列表格式再传给 benchmark_cli
            normalized_json = work_dir / "realtabformer_features.json"
            features = load_features_json(json_path)
            with open(normalized_json, "w", encoding="utf-8") as f:
                json.dump(features, f, ensure_ascii=False, indent=2)
            c_json = self._to_container_path(normalized_json)
        c_model = self._to_container_path(models_dir)

        args = [
            "python", "-m", "realtabformer.benchmark_cli",
            "--csv", c_csv,
            "--model-dir", c_model,
            "--train-only",
            "--epochs", str(epochs),
            "--batch-size", str(hp["batch_size"]),
            "--gradient-accumulation-steps", str(hp["grad_accum"]),
            "--save-steps", str(hp["save_steps"]),
            "--logging-steps", str(hp["logging_steps"]),
        ]
        if c_json:
            args.extend(["--features-json", c_json])
        if numeric_max_len is not None:
            args.extend(["--numeric-max-len", str(numeric_max_len)])
        if resume:
            args.append("--resume")
        if resume_checkpoint:
            args.extend(["--resume-checkpoint", str(resume_checkpoint)])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 日志始终写到 work_dir（用户创建的目录），避免写入 Docker 创建的 id* 目录权限不足
        train_log = work_dir / f"train_{ts}.log"
        try:
            result = self._run_docker(
                args,
                extra_env=self._docker_env(),
                container_workdir=self._to_container_path(work_dir),
            )
            _write_docker_log(
                train_log,
                result.stdout or "",
                result.stderr or "",
                getattr(result, "bench_timing", None),
            )
        except Exception as e:
            stdout = getattr(e, "stdout", "") or ""
            stderr = getattr(e, "stderr", "") or ""
            _write_docker_log(train_log, stdout, stderr, getattr(e, "bench_timing", None))
            raise

        return {
            "model_path": models_dir,
            "work_dir": work_dir,
        }

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

        # benchmark_cli 的 --model-dir 须为含 id* 子目录的 models_* 目录，不能传 id* 本身
        mp = model_path
        if mp.is_file():
            par = mp.parent
            if par.name.startswith("id") and par.parent.name.startswith("models_"):
                model_dir_for_cli = par.parent
                run_root = model_dir_for_cli.parent
            else:
                model_dir_for_cli = par
                run_root = model_dir_for_cli.parent
        elif mp.is_dir():
            if mp.name.startswith("id") and mp.parent.name.startswith("models_"):
                model_dir_for_cli = mp.parent
                run_root = model_dir_for_cli.parent
            elif mp.name.startswith("models_"):
                model_dir_for_cli = mp
                run_root = mp.parent
            else:
                model_dir_for_cli = mp
                run_root = mp.parent
        else:
            raise ValueError(f"无效的 model_path: {mp}")

        if kwargs.get("model_input_manifest"):
            csv_path, _ = self._resolve_model_inputs(csv_path or Path("."), json_path or Path("."), kwargs)
        csv_path = Path(csv_path) if csv_path else None
        if not csv_path or not csv_path.exists():
            raise ValueError("generate 需要有效的 csv_path")

        c_csv = self._to_container_path(csv_path)
        c_model = self._to_container_path(model_dir_for_cli)
        c_out = self._to_container_path(output_csv)

        # The KV cache grows with batch x sequence length, and the sequence grows with the column
        # count: 512 rows at a time fit for ~30 columns but ran out of 32 GB at 338 columns.
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            n_cols = len(next(csv.reader(fh)))
        auto_gen_batch = 512 if n_cols <= 40 else 128 if n_cols <= 80 else 32 if n_cols <= 160 else 16
        gen_batch = max(self._env_int("REALTABFORMER_GEN_BATCH", auto_gen_batch), 1)
        print(f"[REaLTabFormer] generation batch={gen_batch} ({n_cols} columns)")

        args = [
            "python", "-m", "realtabformer.benchmark_cli",
            "--csv", c_csv,
            "--model-dir", c_model,
            "--output-csv", c_out,
            "--generate-only",
            "--num-rows", str(num_rows),
            "--gen-batch", str(gen_batch),
        ]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_log = output_csv.parent / f"gen_{ts}.log"
        try:
            result = self._run_docker(
                args,
                extra_env=self._docker_env(),
                container_workdir=self._to_container_path(run_root),
            )
            _write_docker_log(
                gen_log,
                result.stdout or "",
                result.stderr or "",
                getattr(result, "bench_timing", None),
            )
        except Exception as e:
            stdout = getattr(e, "stdout", "") or ""
            stderr = getattr(e, "stderr", "") or ""
            _write_docker_log(gen_log, stdout, stderr, getattr(e, "bench_timing", None))
            raise
        return self._postprocess_generated_csv(
            output_csv, csv_path, json_path, num_rows, **kwargs
        )
