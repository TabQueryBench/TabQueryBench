"""
TVAE 模型适配器

与 CTGAN 共用同一仓库（ctgan 包），但使用 TVAE 类。
由于 CTGAN CLI (__main__.py) 仅支持 CTGAN，此适配器通过 bridge 脚本调用 TVAE API。
镜像与 CTGAN 相同（默认见 docker_images.json，可用环境变量覆盖）。
"""

import json
import math
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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


def _dataset_id_from_tvae_work_dir(work_dir: Path) -> str:
    """…/output-…/<dataset>/tvae/tvae-<id>/ → dataset id."""
    try:
        wd = Path(work_dir).resolve()
        if wd.name.startswith("tvae-") and wd.parent.name == "tvae":
            return wd.parent.parent.name
    except OSError:
        pass
    return ""


def _jobtrans_cap_dataset_ids() -> set[str]:
    raw = (os.environ.get("TVAE_JOBTRANS_CAP_DATASETS") or "").strip()
    if not raw:
        return set()
    return {p.strip() for p in raw.split(",") if p.strip()}


class TVAEAdapter(BaseModelAdapter):
    """TVAE 适配器。

    CTGAN ``DataTransformer`` 在 ``fit`` 里会用 ``joblib.Parallel(n_jobs=-1)``，宽表易 OOM。
    默认**不**限制并发；仅在以下任一条件满足时，才向容器注入 ``TVAE_CTGAN_JOBTRANS_N_JOBS`` 并打 patch：

    - **全局**（当前 shell / runner 环境）：``TVAE_CTGAN_JOBTRANS_N_JOBS=<正整数>`` — 本次所有 TVAE 训练/生成都限制到该上限。
    - **仅部分数据集**：``TVAE_JOBTRANS_CAP_DATASETS=c21,c13``（逗号分隔）— 仅这些数据集的运行会注入 cap；
      上限取 ``TVAE_JOBTRANS_CAP_N_JOBS``（默认 ``1``），除非同时设置了上面的全局变量（全局优先）。

    可选：宿主若设置 ``TVAE_LOKY_MAX_CPU_COUNT``，会覆盖容器内 ``LOKY_MAX_CPU_COUNT``（默认 ``8``），
    用于进一步压低 joblib/loky 在 ``n_jobs=-1`` 时的进程池规模。

    Epochs：``train(epochs=N)`` > ``TVAE_EPOCHS`` > 按行数自动缩放
    ``ceil(TVAE_TARGET_STEPS / (rows // batch))``，夹在 ``[TVAE_MIN_EPOCHS, TVAE_MAX_EPOCHS]``
    （默认 30000 步、[20, 300]）。例：5k 行→300 epochs，100k 行→150，616k 行→25。
    不对训练行数设上限（上游 max_train_rows 负责）。

    可选：当本次运行**启用了** ``TVAE_CTGAN_JOBTRANS_N_JOBS``（全局或 cap 列表命中）时，若设置
    ``TVAE_CAP_BLAS_THREADS``（例如 ``2``），会把容器内 ``OPENBLAS/MKL/OMP/NUMEXPR`` 线程数
    全部改为该值，减轻单列 transform 里 BLAS 峰值（略增墙钟）。

    宽离散列 holdout（规则与实现见 models/shared/high_cardinality_holdout.py，与 CTGAN /
    TabDiff / TabbyFlow 共用同一套阈值与回填逻辑）：TVAE 与 CTGAN 共用 ``DataTransformer``，
    one-hot 后的宽度是 ``sum(levels)``，``joblib.Parallel`` 里单个 worker 就要物化整张
    ``rows x sum(levels)`` 的 float32 矩阵。满足以下任一条件的离散列在训练前剔除、采样后用
    训练列原值回填：

      - id 型：levels > ``TVAE_ID_MAX_LEVELS``（默认 5000）且 levels >
        ``TVAE_ID_UNIQUE_RATIO``（默认 0.5）× 行数；
      - 高基数文本：levels > ``TVAE_HIGH_CARD_MAX_LEVELS``（默认 5000）。

    clickbench_hits_0_10（50,000 行 / 77 个离散列）上命中 10 个 id 列 + 5 个高基数文本列
    （Title 22,719、WindowName 12,619、IPNetworkID 8,251、OriginalURL 5,790、
    SearchPhrase 5,595），``sum(levels)`` 从 441,858 降到 3,786；否则 transform 要
    50,000 x 441,858 x 4B ≈ 82 GiB，joblib worker 被宿主 OOM killer SIGKILL，报成
    "A worker process managed by the executor was unexpectedly terminated"。目标列永不剔除。
    """

    # See the class docstring and models/shared/high_cardinality_holdout.py.
    _ID_MAX_LEVELS = DEFAULT_MAX_LEVELS
    _ID_UNIQUE_RATIO = DEFAULT_UNIQUE_RATIO
    _HIGH_CARD_MAX_LEVELS = DEFAULT_MAX_LEVELS
    _HOLDOUT_COLUMNS_FILE = "tvae_holdout_columns.json"

    @property
    def model_name(self) -> str:
        return "tvae"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["tvae"]

    # Keep CPU thread usage conservative to avoid host-wide overload / OOM killer
    # when multiple TVAE jobs run concurrently.
    _EXTRA_ENV = {
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OMP_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        # joblib/loky respects this cap when n_jobs=-1
        "LOKY_MAX_CPU_COUNT": "8",
    }

    def _tvae_docker_env(self, work_dir: Path) -> Dict[str, str]:
        env = dict(self._EXTRA_ENV)
        ds = _dataset_id_from_tvae_work_dir(work_dir)
        explicit = (os.environ.get("TVAE_CTGAN_JOBTRANS_N_JOBS") or "").strip()
        cap_ds = _jobtrans_cap_dataset_ids()
        if explicit:
            env["TVAE_CTGAN_JOBTRANS_N_JOBS"] = explicit
        elif ds and ds in cap_ds:
            n = (os.environ.get("TVAE_JOBTRANS_CAP_N_JOBS") or "1").strip() or "1"
            env["TVAE_CTGAN_JOBTRANS_N_JOBS"] = n
        loky = (os.environ.get("TVAE_LOKY_MAX_CPU_COUNT") or "").strip()
        if loky:
            env["LOKY_MAX_CPU_COUNT"] = loky

        cap_active = bool(explicit) or (bool(ds) and ds in cap_ds)
        blas = (os.environ.get("TVAE_CAP_BLAS_THREADS") or "").strip()
        if cap_active and blas:
            for k in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[k] = blas

        for k in (
            "TVAE_EPOCHS",
            "TVAE_BATCH_SIZE",
            "TVAE_EMBEDDING_DIM",
            "TVAE_COMPRESS_DIMS",
            "TVAE_DECOMPRESS_DIMS",
            "TVAE_L2SCALE",
            "TVAE_LOSS_FACTOR",
        ):
            v = (os.environ.get(k) or "").strip()
            if v:
                env[k] = v
        return env

    @staticmethod
    def _resolve_epochs(epochs: Optional[int], csv_path: Path) -> int:
        if epochs:
            return int(epochs)
        raw = (os.environ.get("TVAE_EPOCHS") or "").strip()
        if raw:
            return int(raw)
        n_rows = len(pd.read_csv(csv_path, usecols=[0], encoding="utf-8-sig", low_memory=False))
        batch = max(int(os.environ.get("TVAE_BATCH_SIZE", "500") or 500), 1)
        target = int(os.environ.get("TVAE_TARGET_STEPS", "30000"))
        lo = int(os.environ.get("TVAE_MIN_EPOCHS", "20"))
        hi = int(os.environ.get("TVAE_MAX_EPOCHS", "300"))
        auto = int(math.ceil(target / max(n_rows // batch, 1)))
        resolved = int(min(max(auto, lo), hi))
        print(f"[TVAE] auto epochs={resolved} (rows={n_rows}, batch={batch}, target_steps={target}, clamp=[{lo},{hi}])")
        return resolved

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
                f"{exc}. TVAE models features with data_type continuous/integer as numeric, so every "
                "non-missing value in those columns must parse as a number. Fix the staged input: "
                "convert datetimes to epoch integers, encode missing values as empty cells (real NaN), "
                "or mark the column as categorical/ordinal in staged_features.json."
            ) from exc
        epochs = self._resolve_epochs(epochs, csv_path)
        models_dir = work_dir / f"models_{epochs}epochs"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_file = models_dir / f"tvae_{epochs}epochs.pt"

        # 转换 metadata
        meta_path = work_dir / "tvae_metadata.json"
        convert_features_to_ctgan_metadata(json_path, meta_path)

        # NaN in continuous columns: median-impute + learn a discrete <col>__isna flag
        # (ctgan's DataTransformer drops RDT's is_null output, see ctgan_missing_indicator).
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cont_cols = [c["name"] for c in meta["columns"] if c.get("type") == "continuous"]
        discrete_cols = [c["name"] for c in meta["columns"] if c.get("type") != "continuous"]
        train_df, indicator_map = add_missing_indicators(self.read_staged_csv(csv_path), cont_cols)
        write_indicator_map(work_dir, indicator_map)
        if indicator_map:
            print(f"[TVAE] missing-value indicators: {indicator_map}")
            meta["columns"] += [{"name": v, "type": "categorical"} for v in indicator_map.values()]
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        # Discrete columns too wide to one-hot are dropped here and refilled in generate().
        selection = select_holdout_columns(
            train_df,
            discrete_cols,
            max_levels=int(os.environ.get("TVAE_ID_MAX_LEVELS", self._ID_MAX_LEVELS)),
            unique_ratio=float(os.environ.get("TVAE_ID_UNIQUE_RATIO", self._ID_UNIQUE_RATIO)),
            high_cardinality_max_levels=int(
                os.environ.get("TVAE_HIGH_CARD_MAX_LEVELS", self._HIGH_CARD_MAX_LEVELS)
            ),
            protected=target_columns(json_path),
        )
        holdout_cols = sorted(selection)
        write_holdout_columns(work_dir / self._HOLDOUT_COLUMNS_FILE, selection)
        if holdout_cols:
            train_df = train_df.drop(columns=holdout_cols)
            print(
                "[TVAE] wide discrete columns left out of training, resampled after generation: "
                + describe_selection(selection)
            )
        if indicator_map or holdout_cols:
            csv_path = work_dir / "tvae_train_prepared.csv"
            train_df.to_csv(csv_path, index=False, encoding="utf-8")

        c_csv = self._to_container_path(csv_path)
        c_meta = self._to_container_path(meta_path)
        c_model = self._to_container_path(model_file)

        script = textwrap.dedent(f"""\
            import json, os, sys
            sys.path.insert(0, "/work")
            from src.models.ctgan.adapter.ctgan_joblib_parallel_cap import apply_parallel_cap_from_env
            apply_parallel_cap_from_env()
            import pandas as pd
            import torch
            from torch.optim import Adam
            from torch.utils.data import DataLoader, TensorDataset
            from tqdm import tqdm
            from ctgan.data_transformer import DataTransformer
            from ctgan.synthesizers.tvae import TVAE, Encoder, Decoder, _loss_function

            # Keep transform stage parallelism bounded for stability on shared host.
            os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
            os.environ.setdefault("MKL_NUM_THREADS", "8")
            _nj = (os.environ.get("TVAE_CTGAN_JOBTRANS_N_JOBS") or "").strip()
            if _nj:
                print("[TVAE] joblib Parallel cap ON, TVAE_CTGAN_JOBTRANS_N_JOBS=" + _nj)
            else:
                print("[TVAE] joblib Parallel cap OFF (unset TVAE_CTGAN_JOBTRANS_N_JOBS)")
            print("[TVAE] LOKY_MAX_CPU_COUNT=" + str(os.environ.get("LOKY_MAX_CPU_COUNT", "")))

            csv_path = "{c_csv}"
            meta_path = "{c_meta}"
            save_path = "{c_model}"
            epochs = {epochs}
            batch_size = int(os.environ.get("TVAE_BATCH_SIZE", "500"))
            embedding_dim = int(os.environ.get("TVAE_EMBEDDING_DIM", "128"))
            l2scale = float(os.environ.get("TVAE_L2SCALE", "1e-5"))
            loss_factor = float(os.environ.get("TVAE_LOSS_FACTOR", "2"))
            def _parse_dims(name, default):
                raw = (os.environ.get(name) or "").strip()
                if not raw:
                    return default
                return tuple(int(x.strip()) for x in raw.split(",") if x.strip())
            compress_dims = _parse_dims("TVAE_COMPRESS_DIMS", (128, 128))
            decompress_dims = _parse_dims("TVAE_DECOMPRESS_DIMS", (128, 128))

            def _fit_cpu_staged(self, train_data, discrete_columns=()):
                self.transformer = DataTransformer()
                self.transformer.fit(train_data, discrete_columns)
                train_data = self.transformer.transform(train_data)
                pin_memory = getattr(self._device, "type", str(self._device)) == "cuda"
                dataset = TensorDataset(torch.from_numpy(train_data.astype("float32")))
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=False,
                    pin_memory=pin_memory,
                )

                data_dim = self.transformer.output_dimensions
                encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
                self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
                optimizerAE = Adam(
                    list(encoder.parameters()) + list(self.decoder.parameters()),
                    weight_decay=self.l2scale,
                )

                self.loss_values = pd.DataFrame(columns=["Epoch", "Batch", "Loss"])
                iterator = tqdm(range(self.epochs), disable=(not self.verbose))
                if self.verbose:
                    iterator_description = "Loss: {{loss:.3f}}"
                    iterator.set_description(iterator_description.format(loss=0))

                for i in iterator:
                    loss_values = []
                    batch = []
                    for id_, data in enumerate(loader):
                        optimizerAE.zero_grad()
                        real = data[0].to(self._device, non_blocking=pin_memory)
                        mu, std, logvar = encoder(real)
                        eps = torch.randn_like(std)
                        emb = eps * std + mu
                        rec, sigmas = self.decoder(emb)
                        loss_1, loss_2 = _loss_function(
                            rec,
                            real,
                            sigmas,
                            mu,
                            logvar,
                            self.transformer.output_info_list,
                            self.loss_factor,
                        )
                        loss = loss_1 + loss_2
                        loss.backward()
                        optimizerAE.step()
                        self.decoder.sigma.data.clamp_(0.01, 1.0)

                        batch.append(id_)
                        loss_values.append(loss.detach().cpu().item())

                    epoch_loss_df = pd.DataFrame({{
                        "Epoch": [i] * len(batch),
                        "Batch": batch,
                        "Loss": loss_values,
                    }})
                    if not self.loss_values.empty:
                        self.loss_values = pd.concat(
                            [self.loss_values, epoch_loss_df]
                        ).reset_index(drop=True)
                    else:
                        self.loss_values = epoch_loss_df

                    if self.verbose:
                        iterator.set_description(
                            iterator_description.format(loss=loss.detach().cpu().item())
                        )

            # ctgan.data.read_csv has no BOM handling; staged CSVs may carry a utf-8 BOM.
            data = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
            with open(meta_path, "r", encoding="utf-8") as _mf:
                _meta = json.load(_mf)
            discrete_columns = [c["name"] for c in _meta["columns"] if c["type"] != "continuous" and c["name"] in data.columns]
            print(f"[TVAE] Training on {{len(data)}} rows, {{len(data.columns)}} cols, epochs={{epochs}}, batch_size={{batch_size}}, embedding_dim={{embedding_dim}}, compress_dims={{compress_dims}}, decompress_dims={{decompress_dims}}, l2scale={{l2scale}}, loss_factor={{loss_factor}}")
            _orig_fit = TVAE.fit
            TVAE.fit = _fit_cpu_staged
            try:
                model = TVAE(
                    epochs=epochs,
                    batch_size=batch_size,
                    embedding_dim=embedding_dim,
                    compress_dims=compress_dims,
                    decompress_dims=decompress_dims,
                    l2scale=l2scale,
                    loss_factor=loss_factor,
                )
                model.fit(data, discrete_columns)
            finally:
                TVAE.fit = _orig_fit
            model.save(save_path)
            print(f"[TVAE] Model saved -> {{save_path}}")
        """)
        bridge = self._write_bridge_script(work_dir, "_tvae_train.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log = models_dir / f"train_{ts}.log"
        try:
            result = self._run_docker(
                ["python", c_bridge], extra_env=self._tvae_docker_env(work_dir)
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
            "model_path": model_file,
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
        if kwargs.get("model_input_manifest"):
            csv_path, json_path = self._resolve_model_inputs(
                Path(csv_path or "."), Path(json_path or "."), kwargs
            )

        # TVAE.load() 需要 .pt 文件路径，若传入目录则查找 models_* 下的 .pt
        if model_path.is_dir():
            found = None
            for d in sorted(model_path.iterdir(), reverse=True):
                if d.is_dir() and d.name.startswith("models_"):
                    for f in d.iterdir():
                        if f.is_file() and f.suffix in (".pt", ".pth", ".pkl"):
                            found = f
                            break
                    if found is not None:
                        break
            if found is not None:
                model_path = found
            else:
                raise FileNotFoundError(f"目录下未找到模型文件 .pt/.pth: {model_path}")

        work_dir = model_path.parent.parent if model_path.suffix in (".pt", ".pth", ".pkl") else model_path.parent
        c_model = self._to_container_path(model_path)
        c_out = self._to_container_path(output_csv)

        # 超大 num_rows 时一次 sample 易在逆变换 / concat 阶段 OOM（宿主见 exit 137），分块生成再拼接。
        script = textwrap.dedent(f"""\
            import os, sys
            sys.path.insert(0, "/work")
            from src.models.ctgan.adapter.ctgan_joblib_parallel_cap import apply_parallel_cap_from_env
            apply_parallel_cap_from_env()
            from src.models.ctgan.adapter.ctgan_rdt_inverse_fix import apply_ctgan_inverse_fix
            apply_ctgan_inverse_fix()
            import pandas as pd
            from ctgan.synthesizers.tvae import TVAE
            def _fit_cpu_staged_compat(self, *args, **kwargs):
                raise RuntimeError("_fit_cpu_staged_compat should never be called during generate-only load")
            if not hasattr(TVAE, "_fit_cpu_staged"):
                TVAE._fit_cpu_staged = _fit_cpu_staged_compat
            os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
            os.environ.setdefault("MKL_NUM_THREADS", "8")
            model = TVAE.load("{c_model}")
            total = {num_rows}
            chunk = min(50000, total) if total > 50000 else total
            parts = []
            left = total
            while left > 0:
                take = min(chunk, left)
                parts.append(model.sample(take))
                left -= take
            samples = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
            samples.to_csv("{c_out}", index=False)
            print(f"[TVAE] Generated {{total}} rows (chunks={{len(parts)}}) -> {c_out}")
        """)
        bridge = self._write_bridge_script(work_dir, "_tvae_generate.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_log = output_csv.parent / f"gen_{ts}.log"
        try:
            result = self._run_docker(
                ["python", c_bridge], extra_env=self._tvae_docker_env(work_dir)
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
        apply_missing_indicators(output_csv, read_indicator_map(work_dir))
        holdout_cols = read_holdout_columns(work_dir / self._HOLDOUT_COLUMNS_FILE)
        if holdout_cols:
            if not csv_path or not Path(csv_path).exists():
                raise ValueError("tvae: refilling held-out columns needs the training csv_path")
            refill_holdout_csv(
                output_csv,
                self.read_staged_csv(Path(csv_path)),
                holdout_cols,
                seed=int(os.environ.get("TVAE_HOLDOUT_SEED", "42")),
            )
            print(f"[TVAE] refilled held-out columns from training values: {holdout_cols}")
        return self._postprocess_generated_csv(
            output_csv, csv_path, json_path, num_rows, **kwargs
        )
