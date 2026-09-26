"""
CTGAN 模型适配器

通过 Docker 调用镜像（默认见 docker_images.json，可用环境变量覆盖）。
已修复：添加 OPENBLAS_NUM_THREADS 防止大数据集段错误。

宽离散列 holdout（规则与实现见 models/shared/high_cardinality_holdout.py，与 TVAE /
TabDiff / TabbyFlow 共用）：CTGAN 对每个离散列在数据矩阵与 conditional vector 中都做
one-hot，变换后宽度是 ``sum(levels)``，因此少数高基数列即可撑爆显存。满足以下任一条件
的离散列在训练前剔除、采样后用训练列原值回填：

  - id 型：levels > ``CTGAN_ID_MAX_LEVELS``（默认 5000）且 levels >
    ``CTGAN_ID_UNIQUE_RATIO``（默认 0.5）× 行数；
  - 高基数文本：levels > ``CTGAN_HIGH_CARD_MAX_LEVELS``（默认 5000）。

clickbench_hits_0_10（50,000 行 / 77 个离散列）上命中 10 个 id 列
（WatchID 50,000、HID 48,735、UserID 44,819、ClientIP 42,264、FUniqID 40,222、
RemoteIP 39,905、URLHash 33,231、URL 30,679、RefererHash 26,967、Referer 26,276）
与 5 个高基数文本列（Title 22,719、WindowName 12,619、IPNetworkID 8,251、
OriginalURL 5,790、SearchPhrase 5,595）；``sum(levels)`` 从 441,858 降到 3,786，
否则梯度惩罚一次就要 13.04 GiB 显存。目标列永不剔除。
"""

import json
import math
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import convert_features_to_ctgan_metadata
from models.ctgan.adapter.ctgan_missing_indicator import (
    add_missing_indicators,
    apply_missing_indicators,
    read_indicator_map,
    write_indicator_map,
)
from models.shared.high_cardinality_holdout import (
    DEFAULT_MAX_LEVELS,
    DEFAULT_UNIQUE_RATIO,
    describe_selection,
    read_holdout_columns,
    refill_holdout_csv,
    select_holdout_columns,
    target_columns,
    write_holdout_columns,
)
from core.runner.config import MODEL_DOCKER_MAP


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
        # torch >= 2.9 renamed the variable and prints a deprecation warning per worker for the old
        # one; on the 105-column ClickBench table those 800 warnings buried the real OOM traceback.
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }

    # Upstream CTGAN defaults (ctgan 0.11 / Xu et al. 2019). Every value can be
    # overridden through env: CTGAN_EMBEDDING_DIM, CTGAN_GENERATOR_DIMS ("256,256"),
    # CTGAN_DISCRIMINATOR_DIMS, CTGAN_BATCH_SIZE, CTGAN_PAC.
    _TRAINING_ARGS = {
        "embedding_dim": 128,
        "generator_dim": (256, 256),
        "discriminator_dim": (256, 256),
        "batch_size": 500,
        "pac": 10,
    }

    # Epoch auto-scaling (used only when neither ``epochs`` nor CTGAN_EPOCHS /
    # CTGAN_DEFAULT_EPOCHS is given): aim for ~CTGAN_TARGET_STEPS generator
    # updates, clamped to [CTGAN_MIN_EPOCHS, CTGAN_MAX_EPOCHS].
    _EPOCH_TARGET_STEPS = 20000
    _EPOCH_MIN = 10
    _EPOCH_MAX = 300

    # A discrete column with more than CTGAN_ID_MAX_LEVELS levels that are mostly unique
    # (levels > CTGAN_ID_UNIQUE_RATIO * rows) is an identifier: CTGAN one-hot encodes it both in the
    # data and in the conditional vector, so a 50k-level id makes the networks too large for a GPU
    # (1123_bigill_rnaseq_exon: 19 GiB for the parameters alone) and there is no distribution to
    # learn. The same one-hot blow-up happens for repeating free text once it passes
    # CTGAN_HIGH_CARD_MAX_LEVELS levels even though it is not an identifier (clickbench_hits_0_10:
    # Title 22,719 levels over 50,000 rows). Both kinds are left out of training and filled after
    # sampling with training values drawn without replacement (with replacement once they run out),
    # i.e. "a value seen in training". See models/shared/high_cardinality_holdout.py.
    _ID_MAX_LEVELS = DEFAULT_MAX_LEVELS
    _ID_UNIQUE_RATIO = DEFAULT_UNIQUE_RATIO
    _HIGH_CARD_MAX_LEVELS = DEFAULT_MAX_LEVELS
    _ID_COLUMNS_FILE = "ctgan_id_columns.json"

    @classmethod
    def _effective_training_args(cls) -> Dict[str, Any]:
        def _parse_dims(env_key: str, default: tuple) -> tuple:
            raw = (os.environ.get(env_key) or '').strip()
            if not raw:
                return default
            parts = [int(x.strip()) for x in raw.split(',') if x.strip()]
            if not parts:
                return default
            if len(parts) == 1:
                return (parts[0], parts[0])
            return tuple(parts)

        pac = max(1, int(os.environ.get('CTGAN_PAC', cls._TRAINING_ARGS['pac'])))
        batch_size = int(os.environ.get('CTGAN_BATCH_SIZE', cls._TRAINING_ARGS['batch_size']))
        # ctgan asserts batch_size % 2 == 0 and batch_size % pac == 0.
        step = pac * 2 // math.gcd(pac, 2)
        if batch_size < step or batch_size % step != 0:
            fixed = max(step, int(math.ceil(batch_size / step)) * step)
            print(
                f"[CTGAN] batch_size={batch_size} is not a positive multiple of lcm(2, pac={pac}); "
                f"using batch_size={fixed}"
            )
            batch_size = fixed
        return {
            'embedding_dim': int(os.environ.get('CTGAN_EMBEDDING_DIM', cls._TRAINING_ARGS['embedding_dim'])),
            'generator_dim': _parse_dims('CTGAN_GENERATOR_DIMS', cls._TRAINING_ARGS['generator_dim']),
            'discriminator_dim': _parse_dims('CTGAN_DISCRIMINATOR_DIMS', cls._TRAINING_ARGS['discriminator_dim']),
            'batch_size': batch_size,
            'pac': pac,
        }

    @classmethod
    def _resolve_epochs(cls, epochs: Optional[int], n_rows: int, batch_size: int) -> int:
        """explicit arg > CTGAN_EPOCHS > CTGAN_DEFAULT_EPOCHS (legacy) > auto-scaled by rows."""
        if epochs:
            return int(epochs)
        for key in ("CTGAN_EPOCHS", "CTGAN_DEFAULT_EPOCHS"):
            raw = (os.environ.get(key) or "").strip()
            if raw:
                return int(raw)
        target = int(os.environ.get("CTGAN_TARGET_STEPS", cls._EPOCH_TARGET_STEPS))
        lo = int(os.environ.get("CTGAN_MIN_EPOCHS", cls._EPOCH_MIN))
        hi = int(os.environ.get("CTGAN_MAX_EPOCHS", cls._EPOCH_MAX))
        steps_per_epoch = max(n_rows // max(batch_size, 1), 1)
        auto = int(math.ceil(target / steps_per_epoch))
        return int(min(max(auto, lo), hi))

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
        try:
            self._validate_model_inputs(
                csv_path=csv_path,
                json_path=json_path,
                require_target=True,
                strict_numeric_cast=True,
            )
        except ValueError as exc:
            if "numeric cast failed" not in str(exc):
                raise
            raise ValueError(
                f"{exc}. CTGAN models features with data_type continuous/integer as numeric, so every "
                "non-missing value in those columns must parse as a number. Fix the staged input: "
                "convert datetimes to epoch integers, encode missing values as empty cells (real NaN), "
                "or mark the column as categorical/ordinal in staged_features.json."
            ) from exc

        # 转换 Features JSON -> CTGAN metadata
        meta_path = work_dir / "ctgan_metadata.json"
        convert_features_to_ctgan_metadata(json_path, meta_path)

        train_df = self.read_staged_csv(csv_path)
        n_rows_full = len(train_df)
        # Row cap is OFF by default (global max_train_rows subsampling happens upstream).
        # CTGAN_MAX_TRAIN_ROWS=<n> keeps an explicit, logged opt-in cap.
        max_train_rows = int(os.environ.get("CTGAN_MAX_TRAIN_ROWS", "0") or 0)
        sampled = False
        if max_train_rows > 0 and len(train_df) > max_train_rows:
            train_df = train_df.sample(n=max_train_rows, random_state=42).reset_index(drop=True)
            sampled = True
            print(f"[CTGAN] CTGAN_MAX_TRAIN_ROWS={max_train_rows}: subsampled {n_rows_full} -> {len(train_df)} rows")

        train_args = self._effective_training_args()
        epochs = self._resolve_epochs(epochs, len(train_df), train_args["batch_size"])
        models_dir = work_dir / f"models_{epochs}epochs"
        model_file = f"ctgan_{epochs}epochs.pt"
        models_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[CTGAN] rows={len(train_df)} epochs={epochs} "
            f"steps/epoch={max(len(train_df) // train_args['batch_size'], 1)} args={train_args}"
        )

        meta_cols = self._columns_from_ctgan_meta(meta_path)
        cont_cols = meta_cols["continuous"]
        discrete_cols = list(meta_cols["discrete"])
        # CTGAN cannot model NaN in continuous columns: median-impute + learn a <col>__isna flag.
        train_df, indicator_map = add_missing_indicators(train_df, cont_cols)
        write_indicator_map(work_dir, indicator_map)
        discrete_cols += list(indicator_map.values())
        if indicator_map:
            print(f"[CTGAN] missing-value indicators: {indicator_map}")
        selection = select_holdout_columns(
            train_df,
            discrete_cols,
            max_levels=int(os.environ.get("CTGAN_ID_MAX_LEVELS", self._ID_MAX_LEVELS)),
            unique_ratio=float(os.environ.get("CTGAN_ID_UNIQUE_RATIO", self._ID_UNIQUE_RATIO)),
            high_cardinality_max_levels=int(
                os.environ.get("CTGAN_HIGH_CARD_MAX_LEVELS", self._HIGH_CARD_MAX_LEVELS)
            ),
            protected=target_columns(json_path),
        )
        id_cols = sorted(selection)
        write_holdout_columns(work_dir / self._ID_COLUMNS_FILE, selection)
        if id_cols:
            train_df = train_df.drop(columns=id_cols)
            discrete_cols = [c for c in discrete_cols if c not in id_cols]
            print(
                "[CTGAN] wide discrete columns left out of training, resampled after generation: "
                + describe_selection(selection)
            )
        if indicator_map or sampled or id_cols:
            prepared_csv = work_dir / "ctgan_train_prepared.csv"
            train_df.to_csv(prepared_csv, index=False, encoding="utf-8")
            csv_path = prepared_csv

        c_csv = self._to_container_path(csv_path)
        c_model = self._to_container_path(models_dir / model_file)

        script = textwrap.dedent(
            f"""
            import pandas as pd
            from ctgan.synthesizers.ctgan import CTGAN

            data = pd.read_csv("{c_csv}", encoding="utf-8-sig", low_memory=False)
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

    def _fill_id_columns(self, output_csv: Path, train_csv: Path, work_dir: Path) -> None:
        id_cols = read_holdout_columns(work_dir / self._ID_COLUMNS_FILE)
        if not id_cols:
            return
        refill_holdout_csv(
            output_csv,
            self.read_staged_csv(train_csv),
            id_cols,
            seed=int(os.environ.get("CTGAN_ID_SEED", "42")),
        )
        print(f"[CTGAN] refilled held-out columns from training values: {id_cols}")

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
        apply_missing_indicators(output_csv, read_indicator_map(work_dir))
        self._fill_id_columns(output_csv, csv_path, work_dir)
        return self._postprocess_generated_csv(
            output_csv, csv_path, json_path, num_rows, **kwargs
        )
