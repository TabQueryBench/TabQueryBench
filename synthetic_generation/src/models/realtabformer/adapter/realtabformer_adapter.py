"""
REaLTabFormer 模型适配器

通过 Docker 调用配置的 REaLTabFormer 镜像。
已修复：强制 CUDA_VISIBLE_DEVICES=0 避免 NCCL 多卡通信错误。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import load_features_json
from ..config import MODEL_DOCKER_MAP


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

    # 修复 NCCL Error 2：强制单卡，禁用 NCCL
    # HOME=/tmp 防止 --user 模式下 HuggingFace cache 写入失败
    _EXTRA_ENV = {
        "CUDA_VISIBLE_DEVICES": "0",
        "NCCL_P2P_DISABLE": "1",
        "HOME": "/tmp",
        # 优先导入挂载到 /work 的补丁版 realtabformer 包，避免依赖镜像内旧代码。
        "PYTHONPATH": "/work/src/models/realtabformer/upstream",
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
        epochs = epochs or 100
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
                extra_env=self._EXTRA_ENV,
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

        args = [
            "python", "-m", "realtabformer.benchmark_cli",
            "--csv", c_csv,
            "--model-dir", c_model,
            "--output-csv", c_out,
            "--generate-only",
            "--num-rows", str(num_rows),
        ]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_log = output_csv.parent / f"gen_{ts}.log"
        try:
            result = self._run_docker(
                args,
                extra_env=self._EXTRA_ENV,
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
