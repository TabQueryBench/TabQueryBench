"""
CTGAN 模型适配器

通过 Docker 调用镜像（默认见 docker_images.json，可用环境变量覆盖）。
已修复：添加 OPENBLAS_NUM_THREADS 防止大数据集段错误。
"""

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import convert_features_to_ctgan_metadata
from ..config import MODEL_DOCKER_MAP


class CTGANAdapter(BaseModelAdapter):
    @property
    def model_name(self) -> str:
        return "ctgan"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["ctgan"]

    # 默认额外环境变量：修复 OpenBLAS 线程数超限导致 SIGSEGV
    # HOME=/tmp 防止 --user 模式下缓存目录写入失败
    # expandable_segments 缓解 CUDA 显存碎片（大表 CTGAN 训练 OOM 时可略有帮助）
    _EXTRA_ENV = {
        "OPENBLAS_NUM_THREADS": os.environ.get("CTGAN_BLAS_THREADS", "4"),
        "MKL_NUM_THREADS": os.environ.get("CTGAN_BLAS_THREADS", "4"),
        "HOME": "/tmp",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    # Conservative defaults for very wide/high-cardinality tables.
    _TRAINING_ARGS = {
        "embedding_dim": 4,
        "generator_dim": (8, 8),
        "discriminator_dim": (8, 8),
        "batch_size": 2,
        "pac": 1,
    }

    @classmethod
    def _effective_training_args(cls) -> Dict[str, Any]:
        def _parse_dims(env_key: str, default: tuple[int, int]) -> tuple[int, int]:
            raw = (os.environ.get(env_key) or '').strip()
            if not raw:
                return default
            parts = [int(x.strip()) for x in raw.split(',') if x.strip()]
            if not parts:
                return default
            if len(parts) == 1:
                return (parts[0], parts[0])
            return tuple(parts[:2])

        return {
            'embedding_dim': int(os.environ.get('CTGAN_EMBEDDING_DIM', cls._TRAINING_ARGS['embedding_dim'])),
            'generator_dim': _parse_dims('CTGAN_GENERATOR_DIMS', cls._TRAINING_ARGS['generator_dim']),
            'discriminator_dim': _parse_dims('CTGAN_DISCRIMINATOR_DIMS', cls._TRAINING_ARGS['discriminator_dim']),
            'batch_size': int(os.environ.get('CTGAN_BATCH_SIZE', cls._TRAINING_ARGS['batch_size'])),
            'pac': int(os.environ.get('CTGAN_PAC', cls._TRAINING_ARGS['pac'])),
        }

    @staticmethod
    def _columns_from_ctgan_meta(meta_path: Path) -> Dict[str, List[str]]:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cols = meta.get("columns") or []
        continuous: List[str] = []
        discrete: List[str] = []
        for c in cols:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            typ = str(c.get("type") or "").lower()
            if not name:
                continue
            if typ == "continuous":
                continuous.append(str(name))
            else:
                discrete.append(str(name))
        return {"continuous": continuous, "discrete": discrete}

    def _write_train_csv_with_continuous_impute(
        self,
        df: pd.DataFrame,
        continuous_cols: List[str],
        out_path: Path,
    ) -> bool:
        """
        CTGAN 不接受连续列中的 NaN。对 metadata 中标为 continuous 的列用训练集的中位数填补。
        若没有任何填补则返回 False（不写文件）。
        """
        work = df.copy()
        changed = False
        for col in continuous_cols:
            if col not in work.columns:
                continue
            num = pd.to_numeric(work[col], errors="coerce")
            if not num.isna().any():
                continue
            med = num.median()
            if pd.isna(med):
                med = 0.0
            work[col] = num.fillna(med)
            changed = True
        if not changed:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        work.to_csv(out_path, index=False, encoding="utf-8")
        return True

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
        self._validate_model_inputs(
            csv_path=csv_path,
            json_path=json_path,
            require_target=True,
            strict_numeric_cast=True,
        )
        default_epochs = int(os.environ.get("CTGAN_DEFAULT_EPOCHS", "50"))
        epochs = epochs or default_epochs
        models_dir = work_dir / f"models_{epochs}epochs"
        model_file = f"ctgan_{epochs}epochs.pt"
        models_dir.mkdir(parents=True, exist_ok=True)

        # 转换 Features JSON -> CTGAN metadata
        meta_path = work_dir / "ctgan_metadata.json"
        convert_features_to_ctgan_metadata(json_path, meta_path)

        train_df = self.read_staged_csv(csv_path)
        max_train_rows = int(os.environ.get("CTGAN_MAX_TRAIN_ROWS", "50000"))
        sampled = False
        if max_train_rows > 0 and len(train_df) > max_train_rows:
            train_df = train_df.sample(n=max_train_rows, random_state=42).reset_index(drop=True)
            sampled = True
        meta_cols = self._columns_from_ctgan_meta(meta_path)
        cont_cols = meta_cols["continuous"]
        discrete_cols = meta_cols["discrete"]
        prepared_csv = work_dir / "ctgan_train_prepared.csv"
        imputed = self._write_train_csv_with_continuous_impute(train_df, cont_cols, prepared_csv)
        if imputed or sampled:
            if not imputed:
                prepared_csv.parent.mkdir(parents=True, exist_ok=True)
                train_df.to_csv(prepared_csv, index=False, encoding="utf-8")
            csv_path = prepared_csv

        c_csv = self._to_container_path(csv_path)
        c_model = self._to_container_path(models_dir / model_file)

        train_args = self._effective_training_args()

        script = textwrap.dedent(
            f"""
            import pandas as pd
            from ctgan.synthesizers.ctgan import CTGAN

            data = pd.read_csv("{c_csv}")
            discrete_columns = {discrete_cols!r}
            model = CTGAN(
                embedding_dim={train_args['embedding_dim']!r},
                generator_dim={train_args['generator_dim']!r},
                discriminator_dim={train_args['discriminator_dim']!r},
                batch_size={train_args['batch_size']!r},
                pac={train_args['pac']!r},
                epochs={epochs!r},
                verbose=True,
            )
            model.fit(data, discrete_columns)
            model.save("{c_model}")
            print("[CTGAN] Saved model ->", "{c_model}")
            """
        ).strip()
        bridge = self._write_bridge_script(work_dir, "_ctgan_train.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log = models_dir / f"train_{ts}.log"
        try:
            result = self._run_docker(["python", c_bridge], extra_env=self._EXTRA_ENV)
            _write_docker_log(train_log, result.stdout or "", result.stderr or "")
        except Exception as e:
            stdout = getattr(e, "stdout", "") or ""
            stderr = getattr(e, "stderr", "") or ""
            _write_docker_log(train_log, stdout, stderr)
            raise

        return {
            "model_path": models_dir / model_file,
            "work_dir": work_dir,
            "metadata_path": meta_path,
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

        work_dir = model_path.parent.parent if model_path.suffix == ".pt" else model_path.parent
        meta_path = work_dir / "ctgan_metadata.json"
        if kwargs.get("model_input_manifest"):
            csv_path, _ = self._resolve_model_inputs(csv_path or Path("."), json_path or Path("."), kwargs)
        csv_path = Path(csv_path) if csv_path else None
        if not csv_path or not csv_path.exists():
            raise ValueError("generate 需要有效的 csv_path（训练数据路径）")

        c_model = self._to_container_path(model_path)
        c_out = self._to_container_path(output_csv)

        # 与 TVAE 相同：修补 ctgan 对 RDT 多分量连续列逆变换硬编码 [:2] 的问题（见 ctgan_rdt_inverse_fix）。
        # 不再使用 ``python -m ctgan --generate-only``，以便在 sample 前挂载 patch。
        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, "/work")
            from src.models.ctgan.adapter.ctgan_rdt_inverse_fix import apply_ctgan_inverse_fix
            apply_ctgan_inverse_fix()
            import pandas as pd
            from ctgan.synthesizers.ctgan import CTGAN
            model = CTGAN.load("{c_model}")
            total = {num_rows}
            chunk = min(50000, total) if total > 50000 else total
            parts = []
            left = total
            while left > 0:
                take = min(chunk, left)
                parts.append(model.sample(take))
                left -= take
            sampled = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
            sampled.to_csv("{c_out}", index=False)
            print("[CTGAN] Generated", total, "rows in", len(parts), "chunks ->", "{c_out}")
            """
        ).strip()
        bridge = self._write_bridge_script(work_dir, "_ctgan_generate.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_log = output_csv.parent / f"gen_{ts}.log"
        try:
            result = self._run_docker(["python", c_bridge], extra_env=self._EXTRA_ENV)
            _write_docker_log(gen_log, result.stdout or "", result.stderr or "")
        except Exception as e:
            stdout = getattr(e, "stdout", "") or ""
            stderr = getattr(e, "stderr", "") or ""
            _write_docker_log(gen_log, stdout, stderr)
            raise
        return self._postprocess_generated_csv(
            output_csv, csv_path, json_path, num_rows, **kwargs
        )
