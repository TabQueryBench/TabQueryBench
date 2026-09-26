"""
TabSyn 模型适配器

Docker 镜像: MODEL_DOCKER_MAP["tabsyn"]（tabquerybench/tabsyn:latest, torch 2.9+cu128）。
源码: synthetic_benchmark/tabsyn（只读挂载到 /workspace/tabsyn，扁平 import 布局，PYTHONPATH=/workspace/tabsyn）。
容器内 cwd = 本次 run 的 work_dir，TabSyn 的相对路径 data/<dataname>/ 就指向 work_dir/data/<dataname>/。

训练: vae/main.py（阶段 1）→ main.py（阶段 2，潜空间扩散）；生成: sample.py --num_samples N。

输入契约（staging 之后）:
- 列类型: continuous / integer（含已转为 epoch 整数的 datetime）→ 数值通道；
  categorical / binary / ordinal / boolean（以及任何无法数值化的列）→ 类别通道。
- 缺失值: 类别列 NaN 编码为独立类别 "__nan__"，生成后还原为 NaN；
  数值列 NaN 用均值填充 + 内部 0/1 指示列 __tabsyn_na__<col>（类别通道），生成后按指示列还原 NaN。
- 恰好一个 target；task_type 优先取 manifest / field_registry 的 task_type，否则按 target 列类型推断。
- 不做任何训练行数截断（全量 train 训练）。

Epoch 映射（`--epochs E`）:
- VAE epochs = E；diffusion epochs = round(2.5 * E)（上游 4000 : 10000 的比例），diffusion 仍保留上游早停 patience=500。
- E 为空时使用上游默认: VAE 4000，diffusion 10001。
- 显式环境变量优先级最高: TABSYN_VAE_EPOCHS / TABSYN_DIFFUSION_EPOCHS（兼容旧名 TABSYN_DIFFUSION_MAX_EPOCHS）。

其它环境变量（均可选）:
- TABSYN_VAE_BATCH_SIZE: 默认上游 4096，token 数 > 64 时按 1/T^2 自动缩小（最小 256）。
- TABSYN_DIFFUSION_BATCH_SIZE (4096), TABSYN_DIFFUSION_PATIENCE (500), TABSYN_DIFFUSION_SAVE_EVERY (0=关)。
- TABSYN_VAE_VAL_ROWS: VAE 监控 / checkpoint 选择用的 train 随机子集行数（默认 min(n, 10000)；不参与拟合之外的任何事）。
- TABSYN_SAMPLE_BATCH_SIZE, TABSYN_SAMPLE_STEPS (50), TABSYN_GPU_DATA_MAX_GB (4), TABSYN_LOG_EVERY。
- TABSYN_NUMERIC_NAN_INDICATORS=0 关闭数值列缺失指示列（则数值 NaN 仅均值填充，不再还原）。
"""

import json
import math
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import load_features_json
from core.runner.config import MODEL_DOCKER_MAP, get_synthetic_benchmark_root

_SYNTHETIC_BENCHMARK_ROOT = get_synthetic_benchmark_root()
_TABSYN_HOST_PATH = _SYNTHETIC_BENCHMARK_ROOT / "tabsyn"
_TABSYN_CONTAINER_PATH = "/workspace/tabsyn"

DEFAULT_VAE_EPOCHS = 4000
DEFAULT_DIFFUSION_EPOCHS = 10001
DIFFUSION_EPOCH_RATIO = 2.5
DEFAULT_VAE_VAL_ROWS = 10000

MISSING_SENTINEL = "__nan__"
_NA_INDICATOR_PREFIX = "__tabsyn_na__"
_DUMMY_NUM_COL = "__tabsyn_dummy_num__"
_DUMMY_CAT_COL = "__tabsyn_dummy_cat__"

_CAT_DTYPES = {
    "categorical", "binary", "ordinal", "boolean", "bool", "id", "id_like",
    "text", "others", "multilabel",
}
_NUM_DTYPES = {
    "continuous", "integer", "numerical", "numeric", "float", "int",
    "datetime", "timestamp", "datetime_like", "date",
}


def _log(msg: str) -> None:
    print(f"[TabSyn] {msg}", flush=True)


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def _resolve_epochs(epochs: Optional[int]) -> Tuple[int, int, str]:
    if epochs is not None and int(epochs) > 0:
        vae = int(epochs)
        diff = max(1, int(round(DIFFUSION_EPOCH_RATIO * vae)))
        src = f"--epochs={epochs} (vae=E, diffusion=round({DIFFUSION_EPOCH_RATIO}*E))"
    else:
        vae, diff = DEFAULT_VAE_EPOCHS, DEFAULT_DIFFUSION_EPOCHS
        src = "upstream defaults"
    env_vae = _env_int("TABSYN_VAE_EPOCHS")
    env_diff = _env_int("TABSYN_DIFFUSION_EPOCHS") or _env_int("TABSYN_DIFFUSION_MAX_EPOCHS")
    if env_vae:
        vae = env_vae
        src += "; TABSYN_VAE_EPOCHS override"
    if env_diff:
        diff = env_diff
        src += "; TABSYN_DIFFUSION_EPOCHS override"
    return vae, diff, src


def _encode_categorical(series) -> Tuple["Any", List[str]]:
    """Label-encode to int64 codes; NaN becomes its own level MISSING_SENTINEL."""
    import numpy as np
    import pandas as pd

    mask = series.isna()
    non_null = series[~mask]
    if pd.api.types.is_bool_dtype(series):
        keys = series.astype(object).map(lambda v: str(bool(v)))
    elif pd.api.types.is_float_dtype(series) and len(non_null) and bool(
        np.all(np.isfinite(non_null.to_numpy())) and np.all(np.mod(non_null.to_numpy(), 1) == 0)
    ):
        keys = series.astype("Int64").astype(str)
    else:
        keys = series.astype(str)
    keys = keys.astype(object)
    keys[mask.to_numpy()] = MISSING_SENTINEL
    codes, uniques = pd.factorize(keys, sort=False)
    return codes.astype(np.int64), [str(u) for u in uniques]


def _numeric_castable(series) -> bool:
    import pandas as pd

    casted = pd.to_numeric(series, errors="coerce")
    return not bool((series.notna() & casted.isna()).any())


def _normalize_task_type(raw: Optional[str]) -> Optional[str]:
    t = str(raw or "").strip().lower()
    if not t:
        return None
    if "regress" in t:
        return "regression"
    if any(k in t for k in ("class", "binary", "binclass", "multiclass")):
        return "classification"
    return None


def _get_column_name_mapping(
    num_col_idx: List[int],
    cat_col_idx: List[int],
    target_col_idx: List[int],
    column_names: List[str],
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    """与上游 process_dataset.get_column_name_mapping 一致（sample.recover_data / rename 依赖）。"""
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
        # 只读挂载：避免容器写 __pycache__ / 数据软链接污染共享源码目录
        return [(_TABSYN_HOST_PATH, f"{_TABSYN_CONTAINER_PATH}:ro")]

    def _tabsyn_dataname(self, work_dir: Path) -> str:
        """每个 run 唯一的 dataname，避免复用陈旧 checkpoint。"""
        work_dir = Path(work_dir)
        dataset_id = work_dir.parent.parent.name or "benchmark_ds"
        raw = f"tabsyn_{dataset_id}_{work_dir.name}"
        return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)

    # ------------------------------------------------------------------ data

    def _load_task_hint(self, kwargs: Dict[str, Any], target_col: str) -> Optional[str]:
        manifest_path = kwargs.get("model_input_manifest")
        if manifest_path:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            hint = _normalize_task_type(manifest.get("task_type"))
            if hint:
                return hint
        registry_path = self._resolve_field_registry_path(kwargs)
        if registry_path is not None:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            primary = registry.get("adapter_primary_target")
            if primary and primary != target_col:
                _log(f"WARNING registry adapter_primary_target={primary!r} != features target={target_col!r}")
            return _normalize_task_type(registry.get("task_type"))
        return None

    def _prepare_data_dir(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        dataname: str,
        task_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        staged train CSV + features JSON → work_dir/data/<dataname>/{X_num,X_cat,y}_{train,test}.npy + info.json。
        TabSyn 上游需要 test split（仅用于 VAE 每 epoch 的监控 / LR 调度 / checkpoint 选择）；这里用 train 的随机子集，
        全部 train 行都参与拟合。
        """
        import numpy as np
        import pandas as pd

        data_dir = Path(work_dir) / "data" / dataname
        data_dir.mkdir(parents=True, exist_ok=True)

        df = self.read_staged_csv(csv_path)
        features = load_features_json(json_path)
        names = [str(f.get("feature_name")) for f in features]
        missing = [c for c in names if c not in df.columns]
        extra = [c for c in df.columns if c not in names]
        if missing or extra:
            raise ValueError(f"TabSyn: CSV/features mismatch; missing={missing[:10]}, extra={extra[:10]}")
        df = df[names].reset_index(drop=True)
        n = len(df)
        if n < 2:
            raise ValueError(f"TabSyn: need at least 2 training rows, got {n}")

        target_idx = [i for i, f in enumerate(features) if f.get("is_target")]
        if len(target_idx) != 1:
            raise ValueError(f"TabSyn requires exactly one explicit target column, got {target_idx}")
        t_idx = target_idx[0]
        target_col = names[t_idx]

        def kind_of(i: int) -> str:
            dtype = str(features[i].get("data_type", "")).strip().lower()
            col = names[i]
            if dtype in _CAT_DTYPES:
                return "cat"
            if _numeric_castable(df[col]):
                return "num"
            if dtype in _NUM_DTYPES:
                _log(f"column {col!r} declared {dtype} but not numeric-castable -> categorical")
            return "cat"

        target_kind = kind_of(t_idx)
        task = task_hint or ("regression" if target_kind == "num" else "classification")
        if task == "regression" and not _numeric_castable(df[target_col]):
            _log(f"task hint regression but target {target_col!r} is non-numeric -> classification")
            task = "classification"

        use_na_ind = os.environ.get("TABSYN_NUMERIC_NAN_INDICATORS", "1").strip().lower() not in {"0", "false", "no"}

        column_names: List[str] = list(names)
        num_cols: List[str] = []
        cat_cols: List[str] = []
        num_data: Dict[str, np.ndarray] = {}
        cat_data: Dict[str, np.ndarray] = {}
        categorical_levels: Dict[str, List[str]] = {}
        na_indicators: Dict[str, str] = {}
        internal_cols: List[str] = []

        def add_numeric(col: str, values: np.ndarray) -> np.ndarray:
            nan = np.isnan(values)
            if nan.any():
                fill = float(np.nanmean(values)) if (~nan).any() else 0.0
                values = np.where(nan, fill, values)
                if use_na_ind:
                    ind = f"{_NA_INDICATOR_PREFIX}{col}"
                    column_names.append(ind)
                    cat_cols.append(ind)
                    cat_data[ind] = nan.astype(np.int64)
                    na_indicators[ind] = col
                    internal_cols.append(ind)
            return values

        for i, col in enumerate(names):
            if i == t_idx:
                continue
            if kind_of(i) == "cat":
                codes, levels = _encode_categorical(df[col])
                cat_cols.append(col)
                cat_data[col] = codes
                categorical_levels[col] = levels
            else:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
                num_cols.append(col)
                num_data[col] = add_numeric(col, vals)

        if task == "classification":
            y, levels = _encode_categorical(df[target_col])
            categorical_levels[target_col] = levels
            n_classes = len(levels)
            task_type = "binclass" if n_classes <= 2 else "multiclass"
        else:
            yv = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
            y = add_numeric(target_col, yv)
            task_type = "regression"
            n_classes = None

        # 上游约束: classification 需要 ≥1 个数值列（QuantileTransformer）；regression 需要 ≥1 个类别列（OrdinalEncoder）。
        if task_type != "regression" and not num_cols:
            column_names.append(_DUMMY_NUM_COL)
            num_cols.append(_DUMMY_NUM_COL)
            num_data[_DUMMY_NUM_COL] = np.zeros(n, dtype=np.float64)
            internal_cols.append(_DUMMY_NUM_COL)
            _log(f"no numeric feature: added constant internal column {_DUMMY_NUM_COL}")
        if task_type == "regression" and not cat_cols:
            column_names.append(_DUMMY_CAT_COL)
            cat_cols.append(_DUMMY_CAT_COL)
            cat_data[_DUMMY_CAT_COL] = np.zeros(n, dtype=np.int64)
            internal_cols.append(_DUMMY_CAT_COL)
            _log(f"no categorical feature: added constant internal column {_DUMMY_CAT_COL}")

        pos = {c: i for i, c in enumerate(column_names)}
        num_cols.sort(key=pos.get)
        cat_cols.sort(key=pos.get)
        num_col_idx = [pos[c] for c in num_cols]
        cat_col_idx = [pos[c] for c in cat_cols]
        target_col_idx = [t_idx]

        X_num = (
            np.column_stack([num_data[c] for c in num_cols]).astype(np.float64)
            if num_cols else np.zeros((n, 0), dtype=np.float64)
        )
        X_cat = (
            np.column_stack([cat_data[c] for c in cat_cols]).astype(np.int64)
            if cat_cols else np.zeros((n, 0), dtype=np.int64)
        )
        y = np.asarray(y)

        val_rows = _env_int("TABSYN_VAE_VAL_ROWS") or DEFAULT_VAE_VAL_ROWS
        val_rows = max(1, min(n, val_rows))
        rng = np.random.default_rng(0)
        val_idx = np.sort(rng.choice(n, size=val_rows, replace=False)) if val_rows < n else np.arange(n)

        np.save(data_dir / "X_num_train.npy", X_num)
        np.save(data_dir / "X_cat_train.npy", X_cat)
        np.save(data_dir / "y_train.npy", y)
        np.save(data_dir / "X_num_test.npy", X_num[val_idx])
        np.save(data_dir / "X_cat_test.npy", X_cat[val_idx])
        np.save(data_dir / "y_test.npy", y[val_idx])

        idx_mapping, inverse_idx_mapping, idx_name_mapping = _get_column_name_mapping(
            num_col_idx, cat_col_idx, target_col_idx, column_names
        )
        metadata = {"columns": {}}
        for i in num_col_idx:
            metadata["columns"][str(i)] = {"sdtype": "numerical", "computer_representation": "Float"}
        for i in cat_col_idx:
            metadata["columns"][str(i)] = {"sdtype": "categorical"}
        metadata["columns"][str(t_idx)] = (
            {"sdtype": "numerical", "computer_representation": "Float"}
            if task_type == "regression" else {"sdtype": "categorical"}
        )

        info: Dict[str, Any] = {
            "name": dataname,
            "task_type": task_type,
            "n_num_features": len(num_col_idx),
            "n_cat_features": len(cat_col_idx),
            "train_size": n,
            "train_num": n,
            "test_num": int(len(val_idx)),
            "num_col_idx": num_col_idx,
            "cat_col_idx": cat_col_idx,
            "target_col_idx": target_col_idx,
            "column_names": column_names,
            "header": 0,
            "file_type": "csv",
            "data_path": None,
            "test_path": None,
            "idx_mapping": idx_mapping,
            "inverse_idx_mapping": inverse_idx_mapping,
            "idx_name_mapping": idx_name_mapping,
            "categorical_levels": categorical_levels,
            "missing_sentinel": MISSING_SENTINEL,
            "tabsyn_na_indicators": na_indicators,
            "tabsyn_internal_cols": internal_cols,
            "metadata": metadata,
        }
        if n_classes is not None:
            info["n_classes"] = int(n_classes)

        with open(data_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        summary = {
            "rows": n,
            "val_rows": int(len(val_idx)),
            "task_type": task_type,
            "task_type_source": "manifest/registry" if task_hint else "target data_type",
            "n_num": len(num_col_idx),
            "n_cat": len(cat_col_idx),
            "na_indicators": len(na_indicators),
            "internal_cols": internal_cols,
        }
        _log(f"prepared data dir {data_dir}: {summary}")
        return summary

    # ------------------------------------------------------------------ bridges

    @staticmethod
    def _bridge_header(c_work: str) -> str:
        return textwrap.dedent(f"""\
            import os, subprocess, sys
            work_dir = {c_work!r}
            tabsyn_root = {_TABSYN_CONTAINER_PATH!r}
            assert os.path.isdir(tabsyn_root), f"TabSyn source not mounted: {{tabsyn_root}}"
            env = os.environ.copy()
            env["PYTHONPATH"] = tabsyn_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

            def run(script, *args):
                cmd = [sys.executable, os.path.join(tabsyn_root, script), *args]
                print("[TabSyn] $ " + " ".join(cmd), flush=True)
                ret = subprocess.run(cmd, cwd=work_dir, env=env)
                if ret.returncode != 0:
                    print(f"[TabSyn] {{script}} failed with exit code {{ret.returncode}}", flush=True)
                    sys.exit(ret.returncode)
            """)

    def _docker_env(self, work_dir: Path, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k.startswith("TABSYN_")}
        env.setdefault("TABSYN_CKPT_BASE", f"{self._to_container_path(work_dir)}/tabsyn_ckpt")
        env.setdefault("TABSYN_RESUME", "0")
        if extra:
            env.update(extra)
        return env

    def _run_logged(self, script_path: Path, log_path: Path, env: Dict[str, str]) -> None:
        try:
            result = self._run_docker(["python", self._to_container_path(script_path)], extra_env=env)
            _write_docker_log(log_path, result.stdout or "", result.stderr or "", getattr(result, "bench_timing", None))
        except Exception as e:
            _write_docker_log(
                log_path, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or "", getattr(e, "bench_timing", None)
            )
            raise

    # ------------------------------------------------------------------ API

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
        checks = self._validate_model_inputs(
            csv_path=csv_path, json_path=json_path, require_target=True, strict_numeric_cast=True
        )

        dataname = self._tabsyn_dataname(work_dir)
        task_hint = self._load_task_hint(kwargs, checks["target_col"])
        summary = self._prepare_data_dir(csv_path, json_path, work_dir, dataname, task_hint=task_hint)

        vae_epochs, diff_epochs, epoch_src = _resolve_epochs(epochs)
        env = self._docker_env(
            work_dir,
            {"TABSYN_VAE_EPOCHS": str(vae_epochs), "TABSYN_DIFFUSION_EPOCHS": str(diff_epochs)},
        )
        _log(f"epochs: vae={vae_epochs} diffusion={diff_epochs} ({epoch_src})")
        config = {
            "dataname": dataname,
            "vae_epochs": vae_epochs,
            "diffusion_epochs": diff_epochs,
            "epoch_source": epoch_src,
            "data": summary,
            "tabsyn_env": {k: v for k, v in env.items()},
        }
        (work_dir / "tabsyn_train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        c_work = self._to_container_path(work_dir)
        script = self._bridge_header(c_work) + textwrap.dedent(f"""\
            dataname = {dataname!r}
            print("[TabSyn] Step 1/2: VAE", flush=True)
            run("vae/main.py", "--dataname", dataname, "--gpu", "0")
            print("[TabSyn] Step 2/2: latent diffusion", flush=True)
            run("main.py", "--dataname", dataname, "--gpu", "0")
            print("[TabSyn] Training complete (VAE + Diffusion)", flush=True)
            """)
        bridge = self._write_bridge_script(work_dir, "_tabsyn_train.py", script)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_logged(bridge, work_dir / f"train_{ts}.log", env)
        return {"model_path": work_dir, "work_dir": work_dir, "train_meta": config}

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
        if not (work_dir / "data" / dataname / "info.json").exists():
            raise FileNotFoundError(f"TabSyn: missing {work_dir / 'data' / dataname / 'info.json'}; run train first")

        steps = _env_int("TABSYN_SAMPLE_STEPS") or 50
        c_work = self._to_container_path(work_dir)
        c_out = self._to_container_path(output_csv)
        script = self._bridge_header(c_work) + textwrap.dedent(f"""\
            import json
            dataname = {dataname!r}
            output_csv = {c_out!r}
            num_rows = {int(num_rows)!r}
            run("sample.py", "--dataname", dataname, "--gpu", "0", "--save_path", output_csv,
                "--num_samples", str(num_rows), "--steps", {str(steps)!r})

            import numpy as np
            import pandas as pd
            with open(os.path.join(work_dir, "data", dataname, "info.json"), "r", encoding="utf-8") as f:
                info = json.load(f)
            sentinel = info.get("missing_sentinel", "__nan__")
            df = pd.read_csv(output_csv)
            for col, levels in (info.get("categorical_levels") or {{}}).items():
                if col not in df.columns or not levels:
                    continue
                codes = pd.to_numeric(df[col], errors="coerce").round().fillna(0)
                codes = codes.clip(0, len(levels) - 1).astype(int).to_numpy()
                vals = np.asarray(levels, dtype=object)[codes]
                vals[vals == sentinel] = None
                df[col] = vals
            for ind, src in (info.get("tabsyn_na_indicators") or {{}}).items():
                if ind in df.columns and src in df.columns:
                    flag = pd.to_numeric(df[ind], errors="coerce").round().fillna(0).astype(int) == 1
                    df.loc[flag.to_numpy(), src] = np.nan
            drop = [c for c in (info.get("tabsyn_internal_cols") or []) if c in df.columns]
            df = df.drop(columns=drop)
            df.to_csv(output_csv, index=False)
            print(f"[TabSyn] Saved {{len(df)}} rows x {{df.shape[1]}} cols -> {{output_csv}}", flush=True)
            """)
        bridge = self._write_bridge_script(work_dir, "_tabsyn_sample.py", script)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_logged(bridge, output_csv.parent / f"gen_{ts}.log", self._docker_env(work_dir))
        return self._postprocess_generated_csv(output_csv, csv_path, json_path, num_rows, **kwargs)
