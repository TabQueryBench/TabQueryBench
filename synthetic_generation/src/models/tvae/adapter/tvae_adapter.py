"""
TVAE 模型适配器

与 CTGAN 共用同一仓库（ctgan 包），但使用 TVAE 类。
由于 CTGAN CLI (__main__.py) 仅支持 CTGAN，此适配器通过 bridge 脚本调用 TVAE API。
镜像与 CTGAN 相同（默认见 docker_images.json，可用环境变量覆盖）。
"""

import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import convert_features_to_ctgan_metadata
from ..config import MODEL_DOCKER_MAP


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

    可选：当本次运行**启用了** ``TVAE_CTGAN_JOBTRANS_N_JOBS``（全局或 cap 列表命中）时，若设置
    ``TVAE_CAP_BLAS_THREADS``（例如 ``2``），会把容器内 ``OPENBLAS/MKL/OMP/NUMEXPR`` 线程数
    全部改为该值，减轻单列 transform 里 BLAS 峰值（略增墙钟）。
    """

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
        epochs = epochs or 300
        models_dir = work_dir / f"models_{epochs}epochs"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_file = models_dir / f"tvae_{epochs}epochs.pt"

        # 转换 metadata
        meta_path = work_dir / "tvae_metadata.json"
        convert_features_to_ctgan_metadata(json_path, meta_path)

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
            from ctgan.data import read_csv
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
            epochs = int(os.environ.get("TVAE_EPOCHS", {epochs}))
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

            data, discrete_columns = read_csv(csv_path, meta_path, header=True, discrete=None)
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
        return self._postprocess_generated_csv(
            output_csv, csv_path, json_path, num_rows, **kwargs
        )
