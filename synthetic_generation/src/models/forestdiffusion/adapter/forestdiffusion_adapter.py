"""ForestDiffusion（CPU）适配器：数值列 + 类别列整数编码后拟合树扩散，joblib 保存模型。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import load_features_json
from ..config import MODEL_DOCKER_MAP, get_synthetic_benchmark_root


class ForestDiffusionAdapter(BaseModelAdapter):
    @property
    def model_name(self) -> str:
        return "forestdiffusion"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["forestdiffusion"]

    @property
    def _needs_gpu(self) -> bool:
        return False

    def _extra_volumes(self):
        r = get_synthetic_benchmark_root()
        return [
            (
                r / "third_party" / "ForestDiffusion" / "Python-Package" / "base-ForestDiffusion",
                "/workspace/base-ForestDiffusion",
            )
        ]

    @staticmethod
    def _encode_frame(
        df: pd.DataFrame,
        features: List[dict],
    ) -> tuple[np.ndarray, List[str], List[int], Dict[str, List[str]]]:
        """返回 X(float32)、列名顺序、类别列在 X 中的下标（用于 ForestDiffusion cat_indexes）。"""
        cols_order: List[str] = []
        cat_idx_in_x: List[int] = []
        categorical_levels: Dict[str, List[str]] = {}
        parts: List[np.ndarray] = []
        pos = 0
        for feat in features:
            name = feat.get("feature_name")
            if not name or name not in df.columns:
                continue
            dtype = str(feat.get("data_type", "continuous")).lower()
            if feat.get("is_target"):
                continue
            if dtype in ("categorical", "binary", "ordinal", "id", "id_like", "timestamp", "datetime_like"):
                s = df[name].fillna("__nan__").astype(str)
                codes, levels = pd.factorize(s, sort=True)
                categorical_levels[name] = [str(v) for v in levels.tolist()]
                parts.append(codes.astype(np.float32).reshape(-1, 1))
                if s.nunique(dropna=False) > 1:
                    cat_idx_in_x.append(pos)
            else:
                v = pd.to_numeric(df[name], errors="coerce").astype(np.float32).values.reshape(-1, 1)
                parts.append(v)
            cols_order.append(name)
            pos += 1
        target = next(f["feature_name"] for f in features if f.get("is_target"))
        y = df[target].fillna("__nan__").astype(str)
        y_codes, y_levels = pd.factorize(y, sort=True)
        categorical_levels[target] = [str(v) for v in y_levels.tolist()]
        parts.append(y_codes.astype(np.float32).reshape(-1, 1))
        cols_order.append(target)
        X = np.hstack(parts).astype(np.float32)
        return X, cols_order, cat_idx_in_x, categorical_levels

    def train(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        epochs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        csv_path, json_path = self._resolve_model_inputs(csv_path, json_path, kwargs)
        # Hard guard: training must only consume staged train split.
        csv_name = Path(csv_path).name.lower()
        if csv_name in {"val.csv", "test.csv"}:
            raise ValueError(
                f"{self.model_name}: training input must be train split, got '{csv_name}'"
            )
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._validate_model_inputs(csv_path, json_path, require_target=True, strict_numeric_cast=False)
        df = self.read_staged_csv(csv_path)
        max_train_rows = int(os.environ.get("FORESTDIFFUSION_MAX_TRAIN_ROWS", "50000"))
        if max_train_rows > 0 and len(df) > max_train_rows:
            df = df.sample(n=max_train_rows, random_state=42).reset_index(drop=True)
        features = load_features_json(json_path)
        X, col_names, cat_indexes, categorical_levels = self._encode_frame(df, features)
        meta = {
            "column_names": col_names,
            "cat_indexes": cat_indexes,
            "categorical_levels": categorical_levels,
        }

        n_est = max(1, min(int(os.environ.get("FORESTDIFFUSION_N_ESTIMATORS", str(epochs or 20))), 400))
        fd_n_t = max(2, int(os.environ.get("FORESTDIFFUSION_N_T", "10")))
        fd_dup_k = max(1, int(os.environ.get("FORESTDIFFUSION_DUPLICATE_K", "5")))
        fd_n_jobs = max(1, int(os.environ.get("FORESTDIFFUSION_N_JOBS", "1")))
        fd_max_depth = max(2, int(os.environ.get("FORESTDIFFUSION_MAX_DEPTH", "4")))
        xgb_verbosity = int(os.environ.get("FORESTDIFFUSION_XGB_VERBOSITY", "1"))
        xgb_nthread = max(1, int(os.environ.get("FORESTDIFFUSION_XGB_NTHREAD", "1")))
        if xgb_verbosity < 0 or xgb_verbosity > 3:
            raise ValueError("FORESTDIFFUSION_XGB_VERBOSITY must be one of 0, 1, 2, 3")
        script = f"""
import numpy as np, joblib, json, os
from ForestDiffusion import ForestDiffusionModel
X = np.load("/tmp/fd_X.npy")
with open("/tmp/fd_meta.json") as f:
    meta = json.load(f)
cat_indexes = meta["cat_indexes"]
print(
    "[ForestDiffusion] train config: "
    f"rows={{X.shape[0]}} cols={{X.shape[1]}} n_t={fd_n_t} "
    f"n_estimators={n_est} duplicate_K={fd_dup_k} n_jobs={fd_n_jobs} "
    f"max_depth={fd_max_depth} xgb_verbosity={xgb_verbosity} xgb_nthread={xgb_nthread}",
    flush=True,
)
m = ForestDiffusionModel(
    X, n_t={fd_n_t}, n_estimators={n_est}, duplicate_K={fd_dup_k}, n_jobs={fd_n_jobs},
    model="xgboost", max_depth={fd_max_depth}, tree_method="hist", cat_indexes=cat_indexes,
    verbosity={xgb_verbosity}, nthread={xgb_nthread},
)
joblib.dump((m, meta), "/tmp/fd_model.joblib")
print("ForestDiffusion train OK")
"""
        np.save(work_dir / "_fd_X_host.npy", X)
        (work_dir / "_fd_meta_host.json").write_text(json.dumps(meta), encoding="utf-8")
        c_x = self._to_container_path(work_dir / "_fd_X_host.npy")
        c_meta = self._to_container_path(work_dir / "_fd_meta_host.json")
        c_out = self._to_container_path(work_dir / "forestdiffusion_model.joblib")

        full_script = f"""
import os, shutil, json
with open('/tmp/pgrep', 'w') as _f:
    _f.write("#!/usr/bin/env python3\\n")
    _f.write("import subprocess, sys\\n")
    _f.write("ppid = sys.argv[-1]\\n")
    _f.write("out = subprocess.check_output(['ps', '-o', 'pid=', '--ppid', str(ppid)], text=True)\\n")
    _f.write("print(out, end='')\\n")
os.chmod('/tmp/pgrep', 0o755)
os.environ['PATH'] = '/tmp:' + os.environ.get('PATH', '')
shutil.copy(r'{c_x}', '/tmp/fd_X.npy')
with open(r'{c_meta}') as f:
    open('/tmp/fd_meta.json','w').write(f.read())
{script}
shutil.copy('/tmp/fd_model.joblib', r'{c_out}')
"""
        bridge = self._write_bridge_script(work_dir, "_fd_train.py", full_script)
        log = work_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            r = self._run_docker(
                ["python", self._to_container_path(bridge)],
                extra_env={"PYTHONUNBUFFERED": "1"},
            )
            _write_docker_log(
                log,
                r.stdout or "",
                r.stderr or "",
                getattr(r, "bench_timing", None),
            )
        except Exception as e:
            _write_docker_log(
                log,
                getattr(e, "stdout", "") or "",
                getattr(e, "stderr", "") or "",
                getattr(e, "bench_timing", None),
            )
            raise
        md = work_dir / "models_fd"
        md.mkdir(parents=True, exist_ok=True)
        mp = md / "model.joblib"
        mp.write_bytes((work_dir / "forestdiffusion_model.joblib").read_bytes())
        return {"model_path": mp, "work_dir": work_dir}

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
        work_dir = model_path.parent.parent if model_path.parent.name.startswith("models_") else model_path.parent
        c_model = self._to_container_path(work_dir / "forestdiffusion_model.joblib")
        c_out = self._to_container_path(output_csv)
        script = f"""
import joblib, pandas as pd
m, meta = joblib.load(r'{c_model}')
# generate：batch_size 为样本数
arr = m.generate(batch_size=int({num_rows}))
df = pd.DataFrame(arr, columns=meta["column_names"])
for col, levels in (meta.get("categorical_levels") or {{}}).items():
    if col not in df.columns:
        continue
    vals = pd.to_numeric(df[col], errors="coerce").round().fillna(0).astype(int)
    vals = vals.clip(lower=0, upper=max(0, len(levels) - 1))
    df[col] = [levels[i] for i in vals]
df.to_csv(r'{c_out}', index=False)
print("saved", len(df))
"""
        bridge = self._write_bridge_script(work_dir, "_fd_gen.py", script)
        gen_log = output_csv.parent / f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            r = self._run_docker(["python", self._to_container_path(bridge)])
            _write_docker_log(
                gen_log,
                r.stdout or "",
                r.stderr or "",
                getattr(r, "bench_timing", None),
            )
        except Exception as e:
            _write_docker_log(
                gen_log,
                getattr(e, "stdout", "") or "",
                getattr(e, "stderr", "") or "",
                getattr(e, "bench_timing", None),
            )
            raise
        return self._postprocess_generated_csv(
            output_csv, csv_path, json_path, num_rows, **kwargs
        )
