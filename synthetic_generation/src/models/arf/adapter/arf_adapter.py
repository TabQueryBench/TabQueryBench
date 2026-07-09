"""
ARF (Adversarial Random Forest) 模型适配器

通过 Docker 调用配置的 ARF 镜像。
ARF 仅有 Python API（arfpy 包），使用 bridge 脚本调用。
CPU-only 模型。
"""

import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base_adapter import BaseModelAdapter, _write_docker_log
from ..config import MODEL_DOCKER_MAP


class ARFAdapter(BaseModelAdapter):
    @property
    def model_name(self) -> str:
        return "arf"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["arf"]

    @property
    def _needs_gpu(self) -> bool:
        return False

    def train(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        epochs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        ARF 训练。ARF 是 fit+forge 一体的模型，训练阶段保存训练数据路径，
        实际 fit 在 generate 时一起完成（ARF 不产生可序列化的 checkpoint）。

        为了与统一接口兼容，这里执行 fit 并用 pickle 保存模型。
        """
        csv_path, json_path = self._resolve_model_inputs(csv_path, json_path, kwargs)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._validate_model_inputs(
            csv_path=csv_path,
            json_path=json_path,
            require_target=True,
            strict_numeric_cast=False,
        )
        model_file = work_dir / "arf_model.pkl"

        c_csv = self._to_container_path(csv_path)
        c_model = self._to_container_path(model_file)

        script = textwrap.dedent(f"""\
            import os
            import pickle
            import numpy as np
            import pandas as pd
            from arfpy import arf

            def _sanitize_for_arf(df: pd.DataFrame) -> pd.DataFrame:
                \"\"\"缓解 forge 阶段 scipy.stats.truncnorm / 除零：处理 inf、NaN 与极端尾部。\"\"\"
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna(axis=1, how="all")
                for col in df.select_dtypes(include=[np.number]).columns:
                    med = df[col].median()
                    if pd.isna(med):
                        med = 0.0
                    df[col] = df[col].fillna(med)
                    nu = int(df[col].nunique(dropna=True))
                    if nu <= 1:
                        continue
                    q_low = float(os.environ.get("ARF_CLIP_QUANTILE_LOW", "0.001"))
                    q_high = float(os.environ.get("ARF_CLIP_QUANTILE_HIGH", "0.999"))
                    lo, hi = df[col].quantile(q_low), df[col].quantile(q_high)
                    if pd.notna(lo) and pd.notna(hi) and lo < hi:
                        df[col] = df[col].clip(lo, hi)
                return df

            df = pd.read_csv("{c_csv}")
            df = _sanitize_for_arf(df)
            num_trees = int(os.environ.get("ARF_NUM_TREES", "30"))
            delta = float(os.environ.get("ARF_DELTA", "0"))
            max_iters = int(os.environ.get("ARF_MAX_ITERS", "10"))
            early_stop = (os.environ.get("ARF_EARLY_STOP", "true").strip().lower() in ("1", "true", "yes"))
            verbose = (os.environ.get("ARF_VERBOSE", "true").strip().lower() in ("1", "true", "yes"))
            min_node_size = int(os.environ.get("ARF_MIN_NODE_SIZE", "5"))
            print(f"[ARF] Training on {{len(df)}} rows, {{len(df.columns)}} cols")
            print(f"[ARF] Config num_trees={{num_trees}} delta={{delta}} max_iters={{max_iters}} early_stop={{early_stop}} min_node_size={{min_node_size}}")

            model = arf.arf(x=df, num_trees=num_trees, delta=delta, max_iters=max_iters, early_stop=early_stop, verbose=verbose, min_node_size=min_node_size)
            if hasattr(model, "fit"):
                model.fit()
            elif hasattr(model, "forde"):
                model.forde()
            else:
                raise RuntimeError("arfpy API: no fit() / forde()")

            with open("{c_model}", "wb") as f:
                pickle.dump(model, f)
            print(f"[ARF] Model saved -> {c_model}")
        """)
        bridge = self._write_bridge_script(work_dir, "_arf_train.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log = work_dir / f"train_{ts}.log"
        try:
            result = self._run_docker(["python", c_bridge], extra_env={k: v for k, v in os.environ.items() if k.startswith("ARF_")})
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

        return {"model_path": model_file, "work_dir": work_dir}

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
        work_dir = model_path.parent

        c_model = self._to_container_path(model_path)
        c_out = self._to_container_path(output_csv)
        # 供 bridge 内数据集特例（仅 c19 使用）；其它模型忽略该 kwarg。
        _ds_id_lit = repr((kwargs.get("dataset_id") or ""))

        script = textwrap.dedent(f"""\
            import pickle
            import pandas as pd

            def _safe_forge(model, n_target: int):
                # arfpy 在部分分布上会 ZeroDivisionError；n=1 在部分版本会触发
                # AttributeError（不要用 n=1）。失败返回 None，由外层 hard fail。
                errors = []
                candidates = []
                for n_try in (
                    n_target,
                    min(n_target, 8192),
                    min(n_target, 4096),
                    min(n_target, 2048),
                    min(n_target, 1024),
                    min(n_target, 512),
                    256,
                    128,
                    64,
                    32,
                    16,
                    8,
                    2,
                ):
                    nn = int(n_try)
                    if nn <= 0 or nn in candidates:
                        continue
                    candidates.append(nn)
                for n_try in candidates:
                    try:
                        out = model.forge(n=n_try).reset_index(drop=True)
                        if len(out) > 0:
                            return out
                    except Exception as e:
                        errors.append(f"n={{n_try}}: {{type(e).__name__}}: {{e}}")
                print("[ARF] forge failed after retries; last errors:", " | ".join(errors[-4:]))
                return None

            n_target = int({num_rows})
            with open("{c_model}", "rb") as f:
                model = pickle.load(f)

            syn = _safe_forge(model, n_target)
            if syn is None or len(syn) == 0:
                raise RuntimeError("ARF forge failed; refusing to emit train-data fallback")
            else:
                if len(syn) > n_target:
                    syn = syn.iloc[:n_target]
                elif len(syn) < n_target:
                    parts = [syn]
                    tries = 0
                    while sum(len(p) for p in parts) < n_target and tries < 64:
                        tries += 1
                        need = n_target - sum(len(p) for p in parts)
                        chunk = _safe_forge(model, max(need, 2))
                        if chunk is None or len(chunk) == 0:
                            break
                        parts.append(chunk)
                    syn = pd.concat(parts, ignore_index=True).iloc[:n_target]
                if len(syn) < n_target:
                    raise RuntimeError(
                        f"ARF generated only {{len(syn)}}/{{n_target}} rows; "
                        "refusing to pad with train data"
                    )

            _ds_id = {_ds_id_lit}
            if _ds_id == "c19":
                # 仅 c19：object 列内裸换行会使 pivot 用 csv.reader 统计到的「记录数」大于 DataFrame 行数 → Sw。
                for _col in syn.columns:
                    if syn[_col].dtype == object:
                        syn[_col] = (
                            syn[_col]
                            .astype(str)
                            .str.replace("\\r\\n", " ", regex=False)
                            .str.replace("\\n", " ", regex=False)
                            .str.replace("\\r", " ", regex=False)
                        )
                syn = syn.iloc[:n_target].reset_index(drop=True)

            syn.to_csv("{c_out}", index=False)
            print(f"[ARF] Generated {{len(syn)}} rows (requested {{n_target}}) -> {c_out}")
        """)
        bridge = self._write_bridge_script(work_dir, "_arf_generate.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_log = output_csv.parent / f"gen_{ts}.log"
        try:
            result = self._run_docker(["python", c_bridge])
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
