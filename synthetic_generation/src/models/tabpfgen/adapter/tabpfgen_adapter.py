"""
TabPFGen adapter

Docker 镜像默认见 docker_images.json，可用 BENCHMARK_TABPFGEN_IMAGE 覆盖。
TabPFGen uses TabPFN internally -- no traditional training step.
Adapter splits CSV into X/y arrays, calls TabPFGen API for generation.
Source code mounted from synthetic_benchmark/tabpfgen/src.
"""

import json
import os
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import load_features_json
from ..config import MODEL_DOCKER_MAP, get_synthetic_benchmark_root


_TABPFGEN_HOST_PATH = get_synthetic_benchmark_root() / "tabpfgen" / "src"
_TABPFGEN_CONTAINER_PATH = "/workspace/tabpfgen_src"


class TabPFGenAdapter(BaseModelAdapter):
    @property
    def model_name(self) -> str:
        return "tabpfgen"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["tabpfgen"]

    def _extra_volumes(self):
        return [(_TABPFGEN_HOST_PATH, _TABPFGEN_CONTAINER_PATH)]

    @property
    def _extra_env(self) -> Dict[str, str]:
        env: Dict[str, str] = {
            "PYTHONPATH": _TABPFGEN_CONTAINER_PATH,
        }
        tok = os.environ.get("HF_TOKEN", "").strip()
        if tok:
            env["HF_TOKEN"] = tok
        return env

    def _find_target_col(self, features, df_columns) -> str:
        target_cols = []
        for feat in features:
            if feat.get("is_target", False):
                name = feat.get("feature_name")
                if name and name in df_columns:
                    target_cols.append(name)
        if len(target_cols) != 1:
            raise ValueError(
                f"TabPFGen requires exactly one explicit target column, got: {target_cols}"
            )
        return target_cols[0]

    def _load_model_manifest(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        manifest_path = kwargs.get("model_input_manifest")
        if not manifest_path:
            return {}
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _is_classification(
        self,
        features,
        target_col: str,
        task_type: Optional[str] = None,
    ) -> bool:
        task = str(task_type or "").strip().lower()
        if task == "classification":
            return True
        if task == "regression":
            return False
        for feat in features:
            if feat.get("feature_name") == target_col:
                dtype = feat.get("data_type", "continuous").lower()
                return dtype in ("categorical", "binary", "ordinal")
        return False

    def train(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        epochs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """TabPFGen has no training (uses pretrained TabPFN). Save metadata."""
        csv_path, json_path = self._resolve_model_inputs(csv_path, json_path, kwargs)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._validate_model_inputs(
            csv_path=csv_path,
            json_path=json_path,
            require_target=True,
            strict_numeric_cast=False,
        )

        import pandas as pd

        df = self.read_staged_csv(csv_path)
        features = load_features_json(json_path)
        manifest = self._load_model_manifest(kwargs)
        target_col = self._find_target_col(features, list(df.columns))
        task_type = manifest.get("task_type")
        is_clf = self._is_classification(features, target_col, task_type=task_type)

        meta = {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "target_col": target_col,
            "is_classification": is_clf,
            "task_type": task_type or ("classification" if is_clf else "regression"),
            "n_rows": len(df),
            "n_cols": len(df.columns),
        }
        meta_path = work_dir / "tabpfgen_meta.json"
        t_train0 = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log = work_dir / f"train_{ts}.log"
        t_done = datetime.now(timezone.utc).isoformat()
        elapsed = round(time.perf_counter() - t_train0, 3)
        train_log.write_text(
            "=== benchmark train phase timing (host) ===\n"
            f"adapter_model: tabpfgen\n"
            f"phase: train (no Docker — pretrained TabPFN)\n"
            f"started_at_utc: {started_at}\n"
            f"finished_at_utc: {t_done}\n"
            f"elapsed_seconds: {elapsed}\n"
            f"note: Meta saved to {meta_path}\n"
            "=== message ===\n"
            f"[TabPFGen] No training needed (pretrained). Meta saved to {meta_path}\n"
        )

        return {"model_path": work_dir, "work_dir": work_dir, "meta_path": meta_path}

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

        manifest = self._load_model_manifest(kwargs)
        meta_path = work_dir / "tabpfgen_meta.json"
        task_type = str(manifest.get("task_type") or "").strip().lower() or None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            target_col = meta["target_col"]
            is_clf = meta["is_classification"]
            task_type = task_type or str(meta.get("task_type") or "").strip().lower() or None
            if csv_path is None:
                csv_path = Path(meta["csv_path"])
        else:
            target_col = None
            is_clf = task_type != "regression"

        manifest_target_col = str(manifest.get("target_column") or "").strip()
        if manifest_target_col:
            target_col = manifest_target_col
        if task_type:
            is_clf = self._is_classification([], target_col or "", task_type=task_type)

        csv_path = Path(csv_path) if csv_path else None
        if not csv_path or not csv_path.exists():
            raise ValueError("generate requires a valid csv_path")
        if not target_col:
            raise ValueError(
                "TabPFGen generate requires explicit target_col from training metadata"
            )

        c_csv = self._to_container_path(csv_path)
        c_out = self._to_container_path(output_csv)
        target_col_str = target_col
        gen_method = "generate_classification" if is_clf else "generate_regression"

        script = textwrap.dedent(
            f"""\
            import os
            import numpy as np
            import pandas as pd
            import json
            from tabpfgen import TabPFGen

            df = pd.read_csv("{c_csv}")
            target_col = "{target_col_str}"

            target_missing = df[target_col].isna()
            if target_missing.any():
                dropped = int(target_missing.sum())
                df = df.loc[~target_missing].copy()
                print(
                    f"[TabPFGen] Dropped {{dropped}} rows with missing target '{{target_col}}'"
                )
            if df.empty:
                raise ValueError(
                    f"[TabPFGen] No rows remain after dropping missing target '{{target_col}}'"
                )

            feature_cols = [c for c in df.columns if c != target_col]

            cat_encodings = {{}}
            for col in feature_cols:
                if df[col].dtype == object or str(df[col].dtype) == 'category':
                    cats = sorted(df[col].dropna().unique().tolist(), key=str)
                    cat_map = {{v: i for i, v in enumerate(cats)}}
                    df[col] = df[col].map(cat_map).astype(float)
                    cat_encodings[col] = cats
                    print(f"[TabPFGen] Label-encoded '{{col}}' ({{len(cats)}} categories)")

            target_cats = None
            if df[target_col].dtype == object or str(df[target_col].dtype) == 'category':
                cats = sorted(df[target_col].dropna().unique().tolist(), key=str)
                t_map = {{v: i for i, v in enumerate(cats)}}
                df[target_col] = df[target_col].map(t_map).astype(float)
                target_cats = cats
                print(f"[TabPFGen] Label-encoded target '{{target_col}}' ({{len(cats)}} categories)")

            X = df[feature_cols].values.astype(np.float32)
            y = df[target_col].values
            fit_rows_cap = max(1, int(os.environ.get("TABPFGEN_FIT_MAX_ROWS", "50000")))
            if len(X) > fit_rows_cap:
                rng = np.random.default_rng(42)
                idx = np.sort(rng.choice(len(X), size=fit_rows_cap, replace=False))
                X = X[idx]
                y = y[idx]
                print(f"[TabPFGen] Downsampled fit rows -> {{len(X)}} (cap={{fit_rows_cap}})")
            target_n = int({num_rows})

            for i in range(X.shape[1]):
                col_vals = X[:, i]
                mask = np.isnan(col_vals)
                if mask.any():
                    mean_val = np.nanmean(col_vals)
                    X[mask, i] = mean_val if not np.isnan(mean_val) else 0.0

            chunk_rows = max(1, int(os.environ.get("TABPFGEN_GEN_CHUNK_ROWS", "256")))
            device = (os.environ.get("TABPFGEN_DEVICE") or "auto").strip() or "auto"

            n_sgld_steps = max(1, int(os.environ.get("TABPFGEN_N_SGLD_STEPS", "1000")))
            sgld_step_size = float(os.environ.get("TABPFGEN_SGLD_STEP_SIZE", "0.01"))
            sgld_noise_scale = float(os.environ.get("TABPFGEN_SGLD_NOISE_SCALE", "0.01"))

            # TabPFGen v0.1.x API：仅支持 n_sgld_steps / sgld_* / device。
            # （旧版脚本中的 energy_*_chunk 与上游 TabPFGen 不一致，会导致 TypeError。）
            gen = TabPFGen(
                n_sgld_steps=n_sgld_steps,
                sgld_step_size=sgld_step_size,
                sgld_noise_scale=sgld_noise_scale,
                device=device,
            )

            print(
                f"[TabPFGen] Generating {{target_n}} rows via {gen_method} "
                f"(chunk_rows={{chunk_rows}}, device={{device}}, "
                f"n_sgld_steps={{n_sgld_steps}}, sgld_step_size={{sgld_step_size}}, "
                f"sgld_noise_scale={{sgld_noise_scale}})"
            )
            x_parts = []
            y_parts = []
            remaining = target_n
            while remaining > 0:
                take = min(chunk_rows, remaining)
                X_part, y_part = gen.{gen_method}(X, y, n_samples=take)
                x_parts.append(np.asarray(X_part))
                y_parts.append(np.asarray(y_part))
                remaining -= take
                print(f"[TabPFGen] chunk done: take={{take}}, remaining={{remaining}}")

            X_syn = np.concatenate(x_parts, axis=0)
            y_syn = np.concatenate(y_parts, axis=0)

            syn_df = pd.DataFrame(X_syn, columns=feature_cols)
            syn_df[target_col] = y_syn

            for col, cats in cat_encodings.items():
                codes = np.round(syn_df[col].values).astype(int)
                codes = np.clip(codes, 0, len(cats) - 1)
                syn_df[col] = [cats[c] for c in codes]

            if target_cats is not None:
                codes = np.round(syn_df[target_col].values).astype(int)
                codes = np.clip(codes, 0, len(target_cats) - 1)
                syn_df[target_col] = [target_cats[c] for c in codes]

            if len(syn_df) > target_n:
                print(f"[TabPFGen] Trimming rows: {{len(syn_df)}} -> {{target_n}}")
                syn_df = syn_df.iloc[:target_n].copy()
            elif len(syn_df) < target_n:
                raise RuntimeError(
                    f"[TabPFGen] Generated only {{len(syn_df)}}/{{target_n}} rows; "
                    "refusing to pad from generated or original training rows"
                )

            syn_df = syn_df[list(df.columns)]
            if len(syn_df) != target_n:
                raise RuntimeError(
                    f"[TabPFGen] Row alignment failed: got {{len(syn_df)}}, expected {{target_n}}"
                )
            syn_df.to_csv("{c_out}", index=False)
            print(f"[TabPFGen] Saved {{len(syn_df)}} rows -> {c_out}")
        """
        )
        bridge = self._write_bridge_script(
            output_csv.parent, "_tabpfgen_generate.py", script
        )
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_log = output_csv.parent / f"gen_{ts}.log"
        extra_env = dict(self._extra_env)
        for k in (
            "TABPFGEN_ENERGY_SYNTH_CHUNK",
            "TABPFGEN_ENERGY_TRAIN_CHUNK",
            "TABPFGEN_FIT_MAX_ROWS",
            "TABPFGEN_GEN_CHUNK_ROWS",
            "TABPFGEN_DEVICE",
            "TABPFGEN_N_SGLD_STEPS",
            "TABPFGEN_SGLD_STEP_SIZE",
            "TABPFGEN_SGLD_NOISE_SCALE",
        ):
            v = os.environ.get(k)
            if v is not None and str(v).strip() != "":
                extra_env[k] = str(v).strip()
        try:
            result = self._run_docker(["python", c_bridge], extra_env=extra_env)
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
