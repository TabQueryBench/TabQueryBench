"""
TabDDPM adapter

Docker image: configured through docker_images.json or BENCHMARK_TABDDPM_IMAGE.
TabDDPM uses TOML config + preprocessed .npy data.
Adapter converts unified CSV+Features JSON to TabDDPM format.

CLI: python scripts/pipeline.py --config <config.toml> --train --sample
"""

import json
import os
import re
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import load_features_json
from ..config import MODEL_DOCKER_MAP, get_synthetic_benchmark_root

_SYNTHETIC_BENCHMARK_ROOT = get_synthetic_benchmark_root()
_TABDDPM_HOST_PATH = _SYNTHETIC_BENCHMARK_ROOT / "tabddpm" / "code"
_TABDDPM_CONTAINER_PATH = "/workspace/tabddpm/code"

# Docker 采样失败时是否重试（降低 batch_size、换 seed）；仅对可恢复类错误重试
_TABDDPM_SAMPLE_RETRY_MARKERS = (
    "FoundNANsError",
    "CUDA out of memory",
    "OutOfMemoryError",
)


class TabDDPMAdapter(BaseModelAdapter):
    @property
    def model_name(self) -> str:
        return "tabddpm"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["tabddpm"]

    def _extra_volumes(self):
        return [(_TABDDPM_HOST_PATH, _TABDDPM_CONTAINER_PATH)]

    @staticmethod
    def _tabddpm_runtime_dir(work_dir: Path) -> Path:
        return Path(work_dir) / "_tabddpm_runtime"

    @staticmethod
    def _parse_sample_batch_and_seed(content: str) -> Tuple[int, int]:
        m = re.search(r"(?ms)^\[sample\]\s*\r?\n(.*?)(?=^\[|\Z)", content)
        if not m:
            return 1000, 0
        block = m.group(1)
        bs_m = re.search(r"(?m)^batch_size\s*=\s*(\d+)", block)
        seed_m = re.search(r"(?m)^seed\s*=\s*(\d+)", block)
        bs = int(bs_m.group(1)) if bs_m else 1000
        sd = int(seed_m.group(1)) if seed_m else 0
        return bs, sd

    @staticmethod
    def _patch_tabddpm_sample_section(
        content: str,
        *,
        num_samples: int,
        sample_batch_size: int,
        sample_seed: int,
        sample_num_timesteps: Optional[int] = None,
    ) -> str:
        def repl(m) -> str:
            prefix = m.group(1)
            block = m.group(2)
            block = re.sub(r"(?m)^num_samples\s*=\s*\d+", f"num_samples = {num_samples}", block)
            block = re.sub(r"(?m)^batch_size\s*=\s*\d+", f"batch_size = {sample_batch_size}", block)
            block = re.sub(r"(?m)^seed\s*=\s*\d+", f"seed = {sample_seed}", block)
            return prefix + block

        new_content, n = re.subn(
            r"(?ms)^(\[sample\]\s*\r?\n)(.*?)(?=^\[|\Z)",
            repl,
            content,
            count=1,
        )
        if n == 0:
            new_content = re.sub(r"num_samples\s*=\s*\d+", f"num_samples = {num_samples}", content)
        else:
            new_content = new_content

        if sample_num_timesteps is not None:
            new_content = re.sub(
                r"(?m)^num_timesteps\s*=\s*\d+",
                f"num_timesteps = {sample_num_timesteps}",
                new_content,
            )
        return new_content

    @staticmethod
    def _sample_batch_schedule(orig_bs: int, num_rows: int, max_attempts: int) -> List[int]:
        ordered: List[int] = []
        default_cap = int(os.environ.get("TABDDPM_GEN_DEFAULT_MAX_BATCH", "256"))
        first = max(1, min(orig_bs, num_rows, default_cap))
        ordered.append(first)
        for b in (256, 128, 64, 32, 16, 8, 4, 2, 1):
            v = min(b, num_rows)
            if v >= 1 and v < ordered[-1]:
                ordered.append(v)
        while len(ordered) < max_attempts and ordered[-1] > 1:
            nxt = max(1, ordered[-1] // 2)
            if nxt < ordered[-1]:
                ordered.append(nxt)
            else:
                break
        return ordered[:max_attempts]

    @staticmethod
    def _sample_timestep_schedule(orig_steps: int, max_attempts: int) -> List[int]:
        ordered = [max(1, orig_steps)]
        for cand in (100, 50, 20, 10, 5):
            cand = max(1, min(cand, ordered[0]))
            if cand not in ordered:
                ordered.append(cand)
        while len(ordered) < max_attempts:
            ordered.append(ordered[-1])
        return ordered[:max_attempts]

    @staticmethod
    def _tabddpm_sample_output_retryable(output: str) -> bool:
        if not output:
            return False
        if any(marker in output for marker in _TABDDPM_SAMPLE_RETRY_MARKERS):
            return True
        return "returned non-zero exit status" in output and "sample_r" in output

    @staticmethod
    def _parse_num_timesteps(content: str) -> int:
        m = re.search(r"(?m)^num_timesteps\s*=\s*(\d+)", content)
        return int(m.group(1)) if m else 100

    def _tabddpm_sample_bridge_script(
        self,
        c_config: str,
        c_out: str,
        c_npy_dir: str,
        c_info: str,
        c_runtime: str,
        num_rows: int,
    ) -> str:
        return textwrap.dedent(f"""\
            import os, sys, subprocess, json
            import numpy as np
            import pandas as pd

            tabddpm_root = "{_TABDDPM_CONTAINER_PATH}"
            runtime_root = "{c_runtime}"
            assert os.path.isdir(tabddpm_root), f"TabDDPM source not mounted: {{tabddpm_root}}"

            if not os.path.exists(runtime_root):
                def _ignore(_, names):
                    skip = {{"__pycache__", "data", "synthetic", "result", "results", "ckpt"}}
                    return [n for n in names if n in skip or n.endswith(".pyc")]
                import shutil
                shutil.copytree(tabddpm_root, runtime_root, ignore=_ignore)

            env = os.environ.copy()
            env["PYTHONPATH"] = runtime_root + (os.pathsep + env.get("PYTHONPATH", ""))

            # Reuse the compat wrapper (patches collections.Sequence for skorch)
            wrapper = os.path.join(runtime_root, "_compat_run.py")
            if not os.path.exists(wrapper):
                with open(wrapper, "w") as f:
                    f.write(
                        "import collections, collections.abc\\n"
                        "for _a in ('Sequence','MutableSequence','MutableMapping','Mapping',"
                        "'MutableSet','Set','Callable','Iterable','Iterator'):\\n"
                        "    if not hasattr(collections, _a): setattr(collections, _a, getattr(collections.abc, _a, None))\\n"
                        "import sys, runpy\\n"
                        "sys.argv = sys.argv[1:]\\n"
                        "runpy.run_path(sys.argv[0], run_name='__main__')\\n"
                    )

            print(f"[TabDDPM] Sampling {num_rows} rows")
            ret = subprocess.run(
                [sys.executable, wrapper, "scripts/pipeline.py",
                 "--config", "{c_config}",
                 "--sample"],
                cwd=runtime_root,
                env=env
            )
            if ret.returncode != 0:
                sys.exit(ret.returncode)

            # 将 .npy 输出转为 CSV（npy 在 TabDDPM 的 parent_dir，即 npy_dir）
            info_path = "{c_info}"
            with open(info_path) as f:
                info = json.load(f)

            output_dir = "{c_npy_dir}"
            col_names = info.get("column_names", [])

            parts = []
            x_num_path = os.path.join(output_dir, "X_num_train.npy")
            x_cat_path = os.path.join(output_dir, "X_cat_train.npy")
            y_path = os.path.join(output_dir, "y_train.npy")

            if os.path.exists(x_num_path):
                parts.append(np.load(x_num_path, allow_pickle=True))
            if os.path.exists(x_cat_path):
                parts.append(np.load(x_cat_path, allow_pickle=True).astype(float))
            if os.path.exists(y_path):
                y = np.load(y_path, allow_pickle=True)
                parts.append(y.reshape(-1, 1) if y.ndim == 1 else y)

            if parts:
                combined = np.concatenate(parts, axis=1)
                if col_names and len(col_names) == combined.shape[1]:
                    df = pd.DataFrame(combined, columns=col_names)
                else:
                    df = pd.DataFrame(combined)
                for col, levels in (info.get("categorical_levels") or {{}}).items():
                    if col not in df.columns:
                        continue
                    vals = pd.to_numeric(df[col], errors="coerce").round().fillna(0).astype(int)
                    vals = vals.clip(lower=0, upper=max(0, len(levels) - 1))
                    df[col] = [levels[i] for i in vals]
                target_col = info.get("target_col")
                target_levels = info.get("target_categories")
                if target_col and target_levels and target_col in df.columns:
                    vals = pd.to_numeric(df[target_col], errors="coerce").round().fillna(0).astype(int)
                    vals = vals.clip(lower=0, upper=max(0, len(target_levels) - 1))
                    df[target_col] = [target_levels[i] for i in vals]
                df.to_csv("{c_out}", index=False)
                print(f"[TabDDPM] Saved {{len(df)}} rows -> {c_out}")
            else:
                print("[TabDDPM] WARNING: No output .npy files found")
                sys.exit(1)
        """)

    def _prepare_data(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        val_csv_path: Optional[Path] = None,
        test_csv_path: Optional[Path] = None,
        num_rows_to_generate: int = 1000,
    ) -> Path:
        """
        将 CSV + Features JSON 转换为 TabDDPM 所需的 .npy 格式。
        返回数据目录路径。
        """
        import pandas as pd

        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        df_train = self.read_staged_csv(csv_path)
        features = load_features_json(json_path)

        num_cols = []
        cat_cols = []
        target_col = None
        target_dtype = "continuous"

        for feat in features:
            name = feat.get("feature_name")
            if not name or name not in df_train.columns:
                continue
            dtype = str(feat.get("data_type", "continuous") or "continuous").strip().lower()
            is_target = feat.get("is_target", False)

            if is_target:
                target_col = name
                target_dtype = dtype
            # NOTE:
            # features_converter 会把 inferred_type="id_like" 归一化为 data_type="ID"。
            # TabDDPM 侧数值列会被强制 astype(float32)，如果把 ID/日期 当成数值列就会
            # 抛出 ValueError: could not convert string to float。datetime/timestamp 按类别处理。
            elif dtype in (
                "categorical", "binary", "ordinal", "id", "id_like",
                "datetime", "datetime_like", "timestamp",
            ):
                cat_cols.append(name)
            else:
                num_cols.append(name)

        if target_col is None:
            raise ValueError("TabDDPM requires explicit target column in features (is_target=true)")

        # 数值列（严格 train-only：统计量仅由 train 计算）
        if num_cols:
            X_num_train = df_train[num_cols].values.astype(np.float32)
            col_means = np.nanmean(X_num_train, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)

            def _fill(arr):
                for i in range(arr.shape[1]):
                    mask = np.isnan(arr[:, i])
                    arr[mask, i] = col_means[i]

            _fill(X_num_train)
        else:
            X_num_train = None

        # NOTE: pd.Categorical 对 NaN 会赋 codes=-1，导致 F.one_hot/Embedding 越界。
        # 必须先填充 NaN 为显式类别 '__nan__'，再编码。
        _NAN_SENTINEL = "__nan__"

        # 类别列（严格 train-only：类别空间仅由 train 决定）
        categorical_levels: Dict[str, List[str]] = {}
        if cat_cols:
            for c in cat_cols:
                s_all = df_train[c].fillna(_NAN_SENTINEL).astype(str)
                # 确保 fillna 后的字符串不会与原值冲突（原值若是 '__nan__' 则保留）
                s_all = s_all.replace("nan", _NAN_SENTINEL)
                categories = pd.unique(s_all)
                categories = sorted(categories)  # 稳定顺序
                categorical_levels[c] = [str(v) for v in categories]
                s = df_train[c].fillna(_NAN_SENTINEL).astype(str).replace("nan", _NAN_SENTINEL)
                df_train[c] = pd.Categorical(s, categories=categories).codes
            X_cat_train = df_train[cat_cols].values.astype(np.int64)
        else:
            X_cat_train = None

        # 目标列（严格 train-only：编码空间仅由 train 决定）
        # 同样避免 NaN -> -1：分类目标需 fillna 为显式类别
        y_all = df_train[target_col]
        is_cls_target = target_dtype in ("categorical", "binary", "ordinal", "id", "id_like")
        # n2 等：target 为 continuous 但 nunique<=20 时被判为 multiclass，必须 label-encode 为 0..K-1
        is_classification_target = (
            is_cls_target
            or y_all.dtype == object
            or str(y_all.dtype) in ("category", "bool")
            or y_all.nunique() <= 20
        )
        if is_classification_target:
            y_filled = y_all.fillna(_NAN_SENTINEL).astype(str).replace("nan", _NAN_SENTINEL)
            y_categories = sorted(pd.unique(y_filled))
            def _encode_y(s):
                return pd.Categorical(
                    s.fillna(_NAN_SENTINEL).astype(str).replace("nan", _NAN_SENTINEL),
                    categories=y_categories,
                ).codes
            y_train = _encode_y(df_train[target_col])
            y_train = np.asarray(y_train, dtype=np.int64)
        else:
            y_categories = None
            y_train = np.asarray(pd.to_numeric(df_train[target_col], errors="raise"), dtype=np.float32)

        if X_num_train is not None:
            np.save(data_dir / "X_num_train.npy", X_num_train)
        if X_cat_train is not None:
            np.save(data_dir / "X_cat_train.npy", X_cat_train)
        np.save(data_dir / "y_train.npy", y_train)

        # info.json
        is_classification = is_classification_target
        info = {
            "name": "benchmark_dataset",
            "task_type": "multiclass" if is_classification else "regression",
            "n_num_features": len(num_cols),
            "n_cat_features": len(cat_cols),
            "train_size": len(df_train),
            "num_col_idx": list(range(len(num_cols))),
            "cat_col_idx": list(range(len(num_cols), len(num_cols) + len(cat_cols))),
            "target_col_idx": [len(num_cols) + len(cat_cols)],
            "column_names": num_cols + cat_cols + [target_col],
            "target_col": target_col,
            "categorical_levels": categorical_levels,
        }
        if y_categories is not None:
            info["target_categories"] = [str(v) for v in y_categories]
        if is_classification:
            # 必须与 y_train 编码一致：nunique(原始列) 可能与 Categorical codes 的基数不一致（如 c16/c18），
            # 会导致 TabDDPM 里 embedding 索引越界（CUDA indexSelectLargeIndex assertion）。
            info["num_classes"] = int(y_train.max()) + 1 if y_train.size else 0

        with open(data_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        return data_dir

    def _write_config_toml(self, work_dir: Path, data_dir: Path, epochs: int, num_samples: int) -> Path:
        """生成 TabDDPM 的 config.toml (matches format from tabddpm repo)"""
        config_path = work_dir / "config.toml"
        c_data = self._to_container_path(data_dir)
        c_parent = self._to_container_path(work_dir / "output")

        # Load info.json for dataset metadata
        import json
        info_path = data_dir / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        n_num = info.get("n_num_features", 0)
        n_cat = info.get("n_cat_features", 0)
        is_clf = info.get("task_type", "regression") != "regression"
        n_classes = info.get("num_classes", 2) if is_clf else 0
        # d_in = total feature dimension (num + cat + maybe target if not y_cond)
        d_in = n_num + n_cat

        config_content = f"""\
seed = 0
parent_dir = "{c_parent}"
real_data_path = "{c_data}"
model_type = "mlp"
num_numerical_features = {n_num}
device = "cuda:0"

[model_params]
d_in = {d_in}
num_classes = {n_classes}
is_y_cond = true

[model_params.rtdl_params]
d_layers = [256, 256]
dropout = 0.0

[diffusion_params]
num_timesteps = {int(os.environ.get("TABDDPM_NUM_TIMESTEPS", "100"))}
gaussian_loss_type = "mse"

[train.main]
steps = {max(10, epochs * int(os.environ.get("TABDDPM_STEPS_PER_EPOCH", "20")))}
lr = {float(os.environ.get("TABDDPM_TRAIN_LR", "0.0005"))}
weight_decay = 0.0
batch_size = {int(os.environ.get("TABDDPM_TRAIN_BATCH_SIZE", "64"))}

[train.T]
seed = 0
normalization = "quantile"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

[sample]
num_samples = {num_samples}
batch_size = {min(num_samples, int(os.environ.get("TABDDPM_SAMPLE_BATCH_SIZE", "64")))}
seed = 0
"""
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

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
        epochs = epochs or 50

        # 准备数据
        if not kwargs.get("model_input_manifest"):
            raise ValueError("TabDDPM requires model_input_manifest with staged train/features inputs")
        with open(kwargs["model_input_manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        data_dir = self._prepare_data(
            csv_path,
            json_path,
            work_dir,
        )
        runtime_dir = self._tabddpm_runtime_dir(work_dir)
        c_runtime = self._to_container_path(runtime_dir)

        # 写配置
        config_path = self._write_config_toml(work_dir, data_dir, epochs, num_samples=1000)
        c_config = self._to_container_path(config_path)

        script = textwrap.dedent(f"""\
            import os, sys, subprocess

            tabddpm_root = "{_TABDDPM_CONTAINER_PATH}"
            runtime_root = "{c_runtime}"
            assert os.path.isdir(tabddpm_root), f"TabDDPM source not mounted: {{tabddpm_root}}"

            def _ignore(_, names):
                skip = {{"__pycache__", "data", "synthetic", "result", "results", "ckpt"}}
                return [n for n in names if n in skip or n.endswith(".pyc")]

            import shutil
            shutil.rmtree(runtime_root, ignore_errors=True)
            shutil.copytree(tabddpm_root, runtime_root, ignore=_ignore)

            env = os.environ.copy()
            env["PYTHONPATH"] = runtime_root + (os.pathsep + env.get("PYTHONPATH", ""))

            # Write a wrapper that patches collections.Sequence (removed in Python 3.10+)
            # before running pipeline.py - needed because skorch uses old API
            wrapper = os.path.join(runtime_root, "_compat_run.py")
            with open(wrapper, "w") as f:
                f.write(
                    "import collections, collections.abc\\n"
                    "for _a in ('Sequence','MutableSequence','MutableMapping','Mapping',"
                    "'MutableSet','Set','Callable','Iterable','Iterator'):\\n"
                    "    if not hasattr(collections, _a): setattr(collections, _a, getattr(collections.abc, _a, None))\\n"
                    "import sys, runpy\\n"
                    "sys.argv = sys.argv[1:]\\n"
                    "runpy.run_path(sys.argv[0], run_name='__main__')\\n"
                )

            print(f"[TabDDPM] Training, config={c_config}")
            ret = subprocess.run(
                [sys.executable, wrapper, "scripts/pipeline.py",
                 "--config", "{c_config}",
                 "--train"],
                cwd=runtime_root,
                env=env
            )
            if ret.returncode != 0:
                sys.exit(ret.returncode)
            print("[TabDDPM] Training complete")
        """)
        bridge = self._write_bridge_script(work_dir, "_tabddpm_train.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log = work_dir / f"train_{ts}.log"
        try:
            result = self._run_docker(["python", c_bridge])
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

        return {"model_path": work_dir, "work_dir": work_dir, "config_path": config_path}

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
        work_dir = model_path if model_path.is_dir() else model_path.parent
        runtime_dir = self._tabddpm_runtime_dir(work_dir)
        c_runtime = self._to_container_path(runtime_dir)
        # generate-only 时 model_path 常为 .../output/model.pt，此时 work_dir 已是「output 子目录」，
        # 采样产物 X_num_train.npy 等写在 work_dir 下，而不是 work_dir/output/。
        if work_dir.name == "output" and (work_dir / "model.pt").is_file():
            npy_dir = work_dir
            run_root = work_dir.parent
        else:
            npy_dir = work_dir / "output"
            run_root = work_dir
        info_json = run_root / "data" / "info.json"
        if not info_json.is_file():
            info_json = npy_dir / "data" / "info.json"
        if not info_json.is_file():
            raise FileNotFoundError(
                f"TabDDPM info.json 未找到（已尝试 {run_root / 'data' / 'info.json'} 与 {npy_dir / 'data' / 'info.json'}）"
            )

        # 更新配置中的 num_samples / sample.batch_size / seed：
        # historical run dirs may be read-only; write patched configs under output_csv.parent.
        config_src = npy_dir / "config.toml"
        if not config_src.is_file():
            config_src = work_dir / "config.toml"
        if not config_src.exists():
            raise FileNotFoundError(f"TabDDPM config not found: {config_src}")

        base_content = config_src.read_text(encoding="utf-8")
        orig_bs, base_seed = self._parse_sample_batch_and_seed(base_content)
        _retry_env = os.environ.get("TABDDPM_GEN_NAN_RETRY_MAX") or os.environ.get(
            "TABDDPM_GEN_NAN_MAX_RETRIES"
        )
        max_retries = max(1, int(_retry_env or "10"))
        sizes_env = (os.environ.get("TABDDPM_GEN_NAN_BATCH_SIZES") or "").strip()
        if sizes_env:
            batch_candidates = []
            for part in sizes_env.split(","):
                part = part.strip()
                if not part:
                    continue
                batch_candidates.append(max(1, min(int(part), num_rows)))
            if not batch_candidates:
                batch_candidates = self._sample_batch_schedule(
                    orig_bs, num_rows, max_retries
                )
        else:
            batch_candidates = self._sample_batch_schedule(orig_bs, num_rows, max_retries)
        seed_offset = int(os.environ.get("TABDDPM_GEN_NAN_SEED_BASE", "0"))
        ts_base = datetime.now().strftime("%Y%m%d_%H%M%S")
        orig_steps = self._parse_num_timesteps(base_content)
        timestep_candidates = self._sample_timestep_schedule(
            orig_steps, len(batch_candidates)
        )
        retry_trace_path = output_csv.parent / f"tabddpm_sample_retry_trace_{ts_base}.jsonl"

        c_out = self._to_container_path(output_csv)
        c_npy_dir = self._to_container_path(npy_dir)
        c_info = self._to_container_path(info_json)

        for attempt, sample_bs in enumerate(batch_candidates):
            sample_bs = max(1, min(sample_bs, num_rows))
            sample_seed = base_seed + seed_offset + attempt
            sample_steps = timestep_candidates[min(attempt, len(timestep_candidates) - 1)]
            # Switch to DDIM earlier once the first DDPM attempt fails; this is
            # materially more stable on the wide datasets that tend to hit NANs.
            use_ddim = attempt >= 1 or sample_steps <= 50
            content = self._patch_tabddpm_sample_section(
                base_content,
                num_samples=num_rows,
                sample_batch_size=sample_bs,
                sample_seed=sample_seed,
                sample_num_timesteps=sample_steps,
            )
            config_path = output_csv.parent / f"config_sample_{ts_base}_r{attempt}.toml"
            config_path.write_text(content, encoding="utf-8")
            c_config = self._to_container_path(config_path)

            script = self._tabddpm_sample_bridge_script(
                c_config, c_out, c_npy_dir, c_info, c_runtime, num_rows
            )
            bridge = self._write_bridge_script(
                output_csv.parent, f"_tabddpm_sample_r{attempt}.py", script
            )
            c_bridge = self._to_container_path(bridge)

            gen_log = output_csv.parent / f"gen_{ts_base}_r{attempt}.log"
            trace = {
                "attempt": attempt,
                "sample_batch_size": sample_bs,
                "sample_seed": sample_seed,
                "sample_num_timesteps": sample_steps,
                "use_ddim": use_ddim,
                "config_path": str(config_path),
                "gen_log": str(gen_log),
            }
            if attempt > 0:
                print(
                    f"[TabDDPM] 采样重试 {attempt + 1}/{len(batch_candidates)}: "
                    f"batch_size={sample_bs}, seed={sample_seed}, "
                    f"num_timesteps={sample_steps}, ddim={use_ddim}"
                )
            try:
                extra_env = {"TABDDPM_SAMPLE_DDIM": "1"} if use_ddim else None
                result = self._run_docker(["python", c_bridge], extra_env=extra_env)
                _write_docker_log(
                    gen_log,
                    result.stdout or "",
                    result.stderr or "",
                    getattr(result, "bench_timing", None),
                )
                trace["status"] = "success"
                retry_trace_path.write_text(
                    retry_trace_path.read_text(encoding="utf-8") + json.dumps(trace, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                ) if retry_trace_path.exists() else retry_trace_path.write_text(
                    json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                return self._postprocess_generated_csv(
                    output_csv, csv_path, json_path, num_rows, **kwargs
                )
            except Exception as e:
                stdout = getattr(e, "stdout", "") or ""
                stderr = getattr(e, "stderr", "") or ""
                combined = stdout + stderr
                _write_docker_log(
                    gen_log, stdout, stderr, getattr(e, "bench_timing", None)
                )
                retryable = self._tabddpm_sample_output_retryable(combined)
                is_subproc = isinstance(e, subprocess.CalledProcessError)
                trace.update(
                    {
                        "status": "fail",
                        "retryable": retryable,
                        "exception_type": type(e).__name__,
                        "exception": str(e),
                        "stdout_tail": stdout[-1500:],
                        "stderr_tail": stderr[-1500:],
                    }
                )
                retry_trace_path.write_text(
                    retry_trace_path.read_text(encoding="utf-8") + json.dumps(trace, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                ) if retry_trace_path.exists() else retry_trace_path.write_text(
                    json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                if retryable and is_subproc and attempt + 1 < len(batch_candidates):
                    continue
                raise

        raise RuntimeError("TabDDPM generate: no batch candidates (internal error)")
