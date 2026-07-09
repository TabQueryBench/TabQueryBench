"""
TabSyn 模型适配器

通过 Docker 调用镜像（默认见 docker_images.json，可用环境变量覆盖）。
TabSyn 使用自己的数据格式 (data/{dataname}/ + info.json)，
适配器负责将 Pipeline 的 CSV+Features JSON 转换为 TabSyn 格式。

TabSyn CLI:
  训练: python main.py --dataname <name> --mode train --method tabsyn --gpu 0
  采样: python main.py --dataname <name> --mode sample --method tabsyn --gpu 0 --save_path <path>
"""

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import load_features_json
from ..config import MODEL_DOCKER_MAP, get_synthetic_benchmark_root

# TabSyn 源码在 synthetic_benchmark 中，需要额外挂载
_SYNTHETIC_BENCHMARK_ROOT = get_synthetic_benchmark_root()
_TABSYN_HOST_PATH = _SYNTHETIC_BENCHMARK_ROOT / "tabsyn"
_TABSYN_CONTAINER_PATH = "/workspace/tabsyn"


def _features_to_tabsyn_info(features: List[Dict[str, Any]], num_rows: int) -> Dict:
    """
    将 Pipeline Features JSON 转换为 TabSyn 的 info.json 格式。
    """
    num_idx = []
    cat_idx = []
    target_idx = []

    # TabSyn 的 info 与 process_dataset.get_column_name_mapping 要求 num / cat / target
    # 三类列索引两两互斥：目标列只能出现在 target_col_idx，不得再出现在 cat_col_idx，
    # 否则 recover_data 里 range(n_num+n_cat+n_target) 会多算一列（KeyError）。
    for i, feat in enumerate(features):
        dtype = feat.get("data_type", "continuous").lower()
        is_target = feat.get("is_target", False)
        if is_target:
            target_idx.append(i)
            continue
        # NOTE:
        # Features JSON 里可能会有 id_like，经 features_converter 归一化后变为 data_type="ID".
        # TabSyn 这里应把 ID 当作类别列处理，而不是当成连续数值列，否则会在
        # `astype(np.float32)` 时把文本强转 float 导致 ValueError。
        if dtype in ("categorical", "binary", "ordinal", "id", "id_like"):
            cat_idx.append(i)
        else:
            num_idx.append(i)

    if len(target_idx) != 1:
        raise ValueError(f"TabSyn requires exactly one explicit target column, got {target_idx}")

    # TabSyn 的数值预处理（QuantileTransformer）要求至少有 1 个 numerical feature。
    # 纯分类数据集无连续列时，将第一个非目标 categorical 提升为 pseudo-numerical
    # （label-encode 后作为数值通道），以满足 QuantileTransformer 约束。
    if len(num_idx) == 0:
        promote = None
        for i in cat_idx:
            if i not in target_idx:
                promote = i
                break
        if promote is None:
            raise ValueError("TabSyn requires at least one non-target column (all cols are target)")
        cat_idx = [x for x in cat_idx if x != promote]
        num_idx = [promote]

    # 推断任务类型
    target_feat = features[target_idx[0]]
    target_dtype = target_feat.get("data_type", "continuous").lower()
    if target_dtype in ("categorical", "binary", "id", "id_like"):
        n_unique = target_feat.get("unique_count", 3)
        task_type = "binclass" if n_unique == 2 else "multiclass"
    else:
        task_type = "regression"

    # column names
    column_names = [f.get("feature_name", f"col_{i}") for i, f in enumerate(features)]

    return {
        "name": "benchmark_dataset",
        "task_type": task_type,
        "n_num_features": len(num_idx),
        "n_cat_features": len(cat_idx),
        "train_size": num_rows,
        "num_col_idx": num_idx,
        "cat_col_idx": cat_idx,
        "target_col_idx": target_idx,
        "column_names": column_names,
    }


def _get_column_name_mapping(
    num_col_idx: List[int],
    cat_col_idx: List[int],
    target_col_idx: List[int],
    column_names: List[str],
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    """
    与 synthetic_benchmark/tabsyn/process_dataset.get_column_name_mapping 一致。
    TabSyn 的 sample.recover_data 与 sample.rename 依赖 idx_mapping / idx_name_mapping。
    """
    num_set = set(num_col_idx)
    cat_set = set(cat_col_idx)

    idx_mapping: Dict[int, int] = {}
    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):
        if idx in num_set:
            idx_mapping[idx] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_set:
            idx_mapping[idx] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[idx] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {int(v): int(k) for k, v in idx_mapping.items()}
    idx_name_mapping = {int(i): column_names[i] for i in range(len(column_names))}

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


class TabSynAdapter(BaseModelAdapter):
    @property
    def model_name(self) -> str:
        return "tabsyn"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["tabsyn"]

    def _extra_volumes(self):
        """挂载 TabSyn 源码（Python 依赖由配置的 TabSyn 镜像提供，避免宿主机 pip_cache 污染 import）。"""
        return [
            (_TABSYN_HOST_PATH, _TABSYN_CONTAINER_PATH),
        ]

    def _tabsyn_dataname(self, work_dir: Path) -> str:
        """
        Build a unique TabSyn dataname per run to avoid reusing stale checkpoints
        across hyperparameter sweeps or retried jobs.
        """
        try:
            dataset_id = work_dir.parent.parent.name
        except Exception:
            dataset_id = "benchmark_ds"
        run_id = work_dir.name.replace("-", "_")
        return f"tabsyn_{dataset_id}_{run_id}"

    def _prepare_data_dir(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        dataname: str,
        val_csv_path: Optional[Path] = None,
    ) -> Path:
        """
        将 CSV + Features JSON 转换为 TabSyn 期望的 data/{dataname}/ 目录。
        生成 .npy 文件、train/test CSV、以及完整的 info.json。

        仅使用 staging 的 **train**；**不使用** pipeline 的 held-out test。
        TabSyn 上游仍要求 train/test 两套 .npy，这里将 **test** 设为 **train 的副本**
       （仅满足文件格式与 DataLoader，不把真实 test 用于拟合）。
        """
        import numpy as np
        import pandas as pd
        data_dir = work_dir / "data" / dataname
        data_dir.mkdir(parents=True, exist_ok=True)

        if val_csv_path is None:
            raise ValueError("TabSyn requires model_input_manifest with val_csv (test is not used for TabSyn data)")
        if not Path(val_csv_path).exists():
            raise FileNotFoundError(f"val_csv missing: {val_csv_path}")
        train_df = self.read_staged_csv(csv_path)
        # Use staged train split only for TabSyn fitting (do not merge val into train).
        df = train_df.reset_index(drop=True).copy()
        features = load_features_json(json_path)
        info = _features_to_tabsyn_info(features, len(df))
        info["name"] = dataname

        num_col_idx = info["num_col_idx"]
        cat_col_idx = info["cat_col_idx"]
        target_col_idx = info["target_col_idx"]
        column_names = info["column_names"]

        # Ensure column order matches features
        if list(df.columns) != column_names:
            df.columns = column_names

        num_columns = [column_names[i] for i in num_col_idx]
        cat_columns = [column_names[i] for i in cat_col_idx]
        target_columns = [column_names[i] for i in target_col_idx]

        # TabSyn internal "train" = staged train; "test" tensors = copy of train (no pipeline test).
        train_df = df.reset_index(drop=True)
        test_df = train_df.copy().reset_index(drop=True)

        # Label-encode categorical columns to integer indices（基于 train；test 与 train 同源，无未见类问题）.
        categorical_levels: Dict[str, List[str]] = {}
        cols_to_encode = list(cat_columns)  # 只对 cat_columns 编码；回归目标的数值列不编码
        for col in cols_to_encode:
            train_vals = train_df[col].astype(str)
            test_vals = test_df[col].astype(str)

            uniq = train_vals.unique().tolist()
            categorical_levels[col] = [str(v) for v in uniq]
            mapping = {v: i for i, v in enumerate(uniq)}
            unk_index = 0

            train_df[col] = train_vals.map(mapping).astype(np.int64)
            test_df[col] = test_vals.map(lambda v: mapping.get(v, unk_index)).astype(np.int64)

        # 分类目标不再属于 cat_col_idx，需单独编码为整数（与 TabSyn concat_y / y 张量一致）
        if info["task_type"] in ("binclass", "multiclass"):
            for col in target_columns:
                train_vals = train_df[col].astype(str)
                test_vals = test_df[col].astype(str)
                uniq = train_vals.unique().tolist()
                categorical_levels[col] = [str(v) for v in uniq]
                mapping = {v: i for i, v in enumerate(uniq)}
                unk_index = 0
                train_df[col] = train_vals.map(mapping).astype(np.int64)
                test_df[col] = test_vals.map(lambda v: mapping.get(v, unk_index)).astype(np.int64)

        # Numerical features must be real-valued. If TabSyn info forces some
        # previously-categorical columns into num (e.g. to avoid 0 numerical features),
        # we label-encode them into float codes here.
        for col in num_columns:
            # Use "is_numeric_dtype" to catch pandas string dtypes as well.
            if not pd.api.types.is_numeric_dtype(train_df[col]):
                train_vals = train_df[col].astype(str)
                test_vals = test_df[col].astype(str)
                uniq = train_vals.unique().tolist()
                categorical_levels[col] = [str(v) for v in uniq]
                mapping = {v: i for i, v in enumerate(uniq)}
                unk_index = 0
                train_df[col] = train_vals.map(mapping).astype(np.int64)
                test_df[col] = test_vals.map(lambda v: mapping.get(v, unk_index)).astype(np.int64)

        # Numerical features
        X_num_train = (
            train_df[num_columns].to_numpy().astype(np.float32)
            if num_columns
            else np.zeros((len(train_df), 0), dtype=np.float32)
        )
        X_num_test = (
            test_df[num_columns].to_numpy().astype(np.float32)
            if num_columns
            else np.zeros((len(test_df), 0), dtype=np.float32)
        )

        # Fill NaNs in numerical features (TabSyn diffusion/attention 不稳定，先按列均值处理)
        if X_num_train.size:
            col_means = np.nanmean(X_num_train, axis=0)
            # If a column is all-NaN, fallback mean to 0.0
            col_means = np.where(np.isnan(col_means), 0.0, col_means)

            nan_train = np.isnan(X_num_train)
            if nan_train.any():
                X_num_train[nan_train] = np.take(col_means, np.where(nan_train)[1])

            nan_test = np.isnan(X_num_test)
            if nan_test.any():
                X_num_test[nan_test] = np.take(col_means, np.where(nan_test)[1])

        # Categorical features
        X_cat_train = train_df[cat_columns].to_numpy() if cat_columns else np.zeros((len(train_df), 0))
        X_cat_test = test_df[cat_columns].to_numpy() if cat_columns else np.zeros((len(test_df), 0))

        # Target
        y_train = train_df[target_columns].to_numpy()
        y_test = test_df[target_columns].to_numpy()

        # Save .npy files
        np.save(data_dir / "X_num_train.npy", X_num_train)
        np.save(data_dir / "X_num_test.npy", X_num_test)
        np.save(data_dir / "X_cat_train.npy", X_cat_train)
        np.save(data_dir / "X_cat_test.npy", X_cat_test)
        np.save(data_dir / "y_train.npy", y_train)
        np.save(data_dir / "y_test.npy", y_test)

        # Save CSVs
        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        # Also create synthetic/{dataname}/ dir with real.csv and test.csv
        syn_dir = work_dir / "synthetic" / dataname
        syn_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(syn_dir / "real.csv", index=False)
        test_df.to_csv(syn_dir / "test.csv", index=False)

        # Enrich info.json
        info["train_num"] = len(train_df)
        info["test_num"] = len(test_df)
        info["header"] = 0
        info["file_type"] = "csv"
        info["data_path"] = f"data/{dataname}/train.csv"
        info["test_path"] = None

        idx_mapping, inverse_idx_mapping, idx_name_mapping = _get_column_name_mapping(
            num_col_idx, cat_col_idx, target_col_idx, column_names
        )
        info["idx_mapping"] = idx_mapping
        info["inverse_idx_mapping"] = inverse_idx_mapping
        info["idx_name_mapping"] = idx_name_mapping
        info["categorical_levels"] = categorical_levels

        # n_classes for classification
        if info["task_type"] in ("binclass", "multiclass"):
            info["n_classes"] = int(y_train.max()) + 1

        # metadata for TabSyn
        metadata = {"columns": {}}
        for i in num_col_idx:
            metadata["columns"][str(i)] = {"sdtype": "numerical", "computer_representation": "Float"}
        for i in cat_col_idx:
            metadata["columns"][str(i)] = {"sdtype": "categorical"}
        for i in target_col_idx:
            if info["task_type"] == "regression":
                metadata["columns"][str(i)] = {"sdtype": "numerical", "computer_representation": "Float"}
            else:
                metadata["columns"][str(i)] = {"sdtype": "categorical"}
        info["metadata"] = metadata

        with open(data_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        return data_dir

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

        # 准备数据
        if not kwargs.get("model_input_manifest"):
            raise ValueError("TabSyn requires model_input_manifest (val_csv 用于门禁；不使用 test_csv 建表)")
        with open(kwargs["model_input_manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        dataname = self._tabsyn_dataname(work_dir)
        data_dir = self._prepare_data_dir(
            csv_path,
            json_path,
            work_dir,
            dataname=dataname,
            val_csv_path=Path(manifest["val_csv"]),
        )
        c_work = self._to_container_path(work_dir)

        # TabSyn 训练分两步：先训 VAE，再训 diffusion
        # 使用 bridge script 来在容器内切换工作目录并调用
        script = textwrap.dedent(f"""\
            import json, os, sys, subprocess

            work_dir = "{c_work}"
            dataname = "{dataname}"
            tabsyn_root = "{_TABSYN_CONTAINER_PATH}"

            assert os.path.exists(tabsyn_root), f"TabSyn source not mounted: {{tabsyn_root}}"

            old = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = tabsyn_root + (os.pathsep + old if old else "")
            sys.path.insert(0, tabsyn_root)

            os.chdir(tabsyn_root)

            # Symlink data dir into TabSyn data/
            data_link = os.path.join(tabsyn_root, "data", dataname)
            data_src = os.path.join(work_dir, "data", dataname)
            os.makedirs(os.path.join(tabsyn_root, "data"), exist_ok=True)
            if os.path.exists(data_link):
                os.remove(data_link)
            os.symlink(data_src, data_link)

            env = os.environ.copy()
            env.setdefault("TABSYN_RESUME", "0")
            env.setdefault("TABSYN_VAE_BATCH_SIZE", "32")
            env.setdefault("TABSYN_VAE_NUM_WORKERS", "0")
            env.setdefault("TABSYN_VAE_EVAL_BATCH_SIZE", env["TABSYN_VAE_BATCH_SIZE"])
            env.setdefault("TABSYN_VAE_INFER_BATCH_SIZE", env["TABSYN_VAE_BATCH_SIZE"])
            env.setdefault("TABSYN_VAE_ENCODE_BATCH_SIZE", env["TABSYN_VAE_BATCH_SIZE"])
            # Safer defaults for wide tables on Docker: reduce shared-memory pressure in diffusion DataLoader.
            env.setdefault("TABSYN_DIFFUSION_NUM_WORKERS", "0")
            _te = {repr(epochs)}
            if _te is not None:
                env["TABSYN_VAE_EPOCHS"] = str(_te)
                env["TABSYN_DIFFUSION_MAX_EPOCHS"] = str(max(_te + 1, 2))

            # Data preprocessing is done on the host side (_prepare_data_dir)
            # which creates .npy files, train/test CSVs, and info.json

            # Step 1: Train VAE (produces latent embeddings)
            print(f"[TabSyn] Step 1/2: Training VAE in {{tabsyn_root}}, dataname={{dataname}}")
            ret = subprocess.run(
                [sys.executable, "main.py",
                 "--dataname", dataname,
                 "--mode", "train",
                 "--method", "vae",
                 "--gpu", "0"],
                cwd=tabsyn_root,
                env=env
            )
            if ret.returncode != 0:
                print("[TabSyn] VAE training failed")
                sys.exit(ret.returncode)

            # Step 2: Train diffusion model on latent space
            print(f"[TabSyn] Step 2/2: Training diffusion model")
            ret = subprocess.run(
                [sys.executable, "main.py",
                 "--dataname", dataname,
                 "--mode", "train",
                 "--method", "tabsyn",
                 "--gpu", "0"],
                cwd=tabsyn_root,
                env=env
            )
            if ret.returncode != 0:
                print("[TabSyn] Diffusion training failed")
                sys.exit(ret.returncode)
            print("[TabSyn] Training complete (VAE + Diffusion)")
        """)
        bridge = self._write_bridge_script(work_dir, "_tabsyn_train.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log = work_dir / f"train_{ts}.log"
        try:
            env = {k: v for k, v in os.environ.items() if k.startswith("TABSYN_")}
            env.setdefault("TABSYN_CKPT_BASE", f"{self._to_container_path(work_dir)}/tabsyn_ckpt")
            result = self._run_docker(["python", c_bridge], extra_env=env)
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

        return {"model_path": work_dir, "work_dir": work_dir}

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
        dataname = self._tabsyn_dataname(work_dir)

        c_work = self._to_container_path(work_dir)
        c_out = self._to_container_path(output_csv)

        script = textwrap.dedent(f"""\
            import os, sys, subprocess

            work_dir = "{c_work}"
            dataname = "{dataname}"
            output_csv = "{c_out}"
            tabsyn_root = "{_TABSYN_CONTAINER_PATH}"

            assert os.path.exists(tabsyn_root), f"TabSyn source not mounted: {{tabsyn_root}}"

            old = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = tabsyn_root + (os.pathsep + old if old else "")
            sys.path.insert(0, tabsyn_root)

            os.chdir(tabsyn_root)

            # Ensure data symlink exists
            data_link = os.path.join(tabsyn_root, "data", dataname)
            data_src = os.path.join(work_dir, "data", dataname)
            os.makedirs(os.path.join(tabsyn_root, "data"), exist_ok=True)
            if os.path.exists(data_link):
                os.remove(data_link)
            os.symlink(data_src, data_link)

            print(f"[TabSyn] Sampling {num_rows} rows")
            env = os.environ.copy()
            env.setdefault("TABSYN_RESUME", "0")
            env.setdefault("TABSYN_VAE_BATCH_SIZE", "32")
            env.setdefault("TABSYN_VAE_EVAL_BATCH_SIZE", env["TABSYN_VAE_BATCH_SIZE"])
            env.setdefault("TABSYN_VAE_INFER_BATCH_SIZE", env["TABSYN_VAE_BATCH_SIZE"])
            env.setdefault("TABSYN_VAE_ENCODE_BATCH_SIZE", env["TABSYN_VAE_BATCH_SIZE"])
            ret = subprocess.run(
                [sys.executable, "main.py",
                 "--dataname", dataname,
                 "--mode", "sample",
                 "--method", "tabsyn",
                 "--gpu", "0",
                 "--save_path", output_csv],
                cwd=tabsyn_root,
                env=env
            )
            if ret.returncode != 0:
                sys.exit(ret.returncode)
            info_path = os.path.join(data_src, "info.json")
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            levels_map = info.get("categorical_levels") or {{}}
            if levels_map:
                import pandas as pd
                df = pd.read_csv(output_csv)
                for col, levels in levels_map.items():
                    if col not in df.columns:
                        continue
                    vals = pd.to_numeric(df[col], errors="coerce").round().fillna(0).astype(int)
                    vals = vals.clip(lower=0, upper=max(0, len(levels) - 1))
                    df[col] = [levels[i] for i in vals]
                df.to_csv(output_csv, index=False)
            print(f"[TabSyn] Saved -> {{output_csv}}")
        """)
        bridge = self._write_bridge_script(work_dir, "_tabsyn_sample.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_log = output_csv.parent / f"gen_{ts}.log"
        try:
            env = {k: v for k, v in os.environ.items() if k.startswith("TABSYN_")}
            env.setdefault("TABSYN_CKPT_BASE", f"{self._to_container_path(work_dir)}/tabsyn_ckpt")
            result = self._run_docker(["python", c_bridge], extra_env=env)
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
