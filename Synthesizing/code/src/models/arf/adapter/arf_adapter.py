"""
ARF (Adversarial Random Forest) 模型适配器

通过 Docker 调用配置的 ARF 镜像（arfpy 0.1.1，CPU-only）。bridge 脚本调用 Python API。

Encoding:
- categorical/binary/ordinal/boolean/... (features data_type) -> object dtype so arfpy
  treats them as factors (even when values look numeric); NaN stays NaN (arfpy code -1
  round-trips to NaN).
- target: categorical for classification, numeric for regression (field_registry
  task_type > target data_type).
- numeric NaN -> hot-deck imputation + factor column `__arf_isna__<col>`; NaN is
  re-injected in generated rows whose indicator is "1". All-NaN columns are emitted as NaN.

Env:
  ARF_NUM_TREES(30) ARF_DELTA(0) ARF_MAX_ITERS(10) ARF_EARLY_STOP(true) ARF_MIN_NODE_SIZE(5)
  ARF_N_JOBS(-1 = all cores; passed to sklearn RandomForestClassifier via arfpy **kwargs)
  ARF_FAST(1): use arf_fastpath.FastARF (same algorithm; vectorized adversarial resampling,
    leaf pruning, forde and forge); 0 = upstream arfpy (single-threaded pandas loops)
  ARF_SEED(0) ARF_GEN_CHUNK_ROWS(100000)
  ARF_CLIP_QUANTILE_LOW/HIGH: clip numeric training tails. Default 0/1 (off) with ARF_FAST=1;
    0.001/0.999 with ARF_FAST=0 (only guarded upstream forge crashes).
No dataset-specific branches: generated CSVs are written by pandas (quoted), so values with
embedded newlines round-trip and stay inside the training domain.
"""

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base_adapter import BaseModelAdapter, _write_docker_log
from core.runner.config import MODEL_DOCKER_MAP

_NUMERIC_DTYPES = {"continuous", "integer", "numeric", "numerical", "float", "int"}
_FASTPATH_DIR = Path(__file__).resolve().parent


def _resolve_task_type(
    manifest_path: Optional[str], target_col: str, target_dtype: str
) -> Tuple[str, str]:
    """Return ("classification"|"regression", source)."""
    if manifest_path:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            reg_path = manifest.get("field_registry_json")
            if reg_path and Path(reg_path).is_file():
                with open(reg_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                tt = str(registry.get("task_type") or "").strip().lower()
                if "regress" in tt:
                    return "regression", "field_registry.task_type"
                if any(k in tt for k in ("class", "binary", "multiclass")):
                    return "classification", "field_registry.task_type"
                for field in registry.get("fields") or []:
                    if field.get("name") == target_col:
                        sem = str(field.get("semantic_type") or "").lower()
                        if sem in ("continuous", "integer"):
                            return "regression", "field_registry.target.semantic_type"
                        if sem:
                            return "classification", "field_registry.target.semantic_type"
        except Exception as exc:
            print(f"[ARF] WARNING: could not read task_type from registry: {exc}")
    if target_dtype in ("continuous", "integer"):
        return "regression", "features.data_type"
    return "classification", "features.data_type"


def _arf_env() -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.startswith("ARF_")}
    env["PYTHONUNBUFFERED"] = "1"
    return env


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
        ARF 训练：arf() 对抗训练 + forde() 密度估计，pickle 保存模型。`epochs` 不适用（忽略）。
        """
        csv_path, json_path = self._resolve_model_inputs(csv_path, json_path, kwargs)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        info = self._validate_model_inputs(
            csv_path=csv_path,
            json_path=json_path,
            require_target=True,
            strict_numeric_cast=False,
        )
        features = info["features"]
        target_col = info["target_col"]
        target_dtype = str(
            next(f for f in features if f.get("is_target")).get("data_type", "")
        ).strip().lower()
        task_type, task_source = _resolve_task_type(
            kwargs.get("model_input_manifest"), target_col, target_dtype
        )
        numeric_cols = []
        for f in features:
            name = f.get("feature_name")
            dtype = str(f.get("data_type", "") or "").strip().lower()
            if f.get("is_target"):
                if task_type == "regression":
                    numeric_cols.append(name)
            elif dtype in _NUMERIC_DTYPES:
                numeric_cols.append(name)
        schema = {
            "target_col": target_col,
            "task_type": task_type,
            "task_type_source": task_source,
            "numeric_cols": numeric_cols,
        }
        schema_path = work_dir / "arf_schema.json"
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        print(f"[ARF] target={target_col} task_type={task_type} (source={task_source})")

        model_file = work_dir / "arf_model.pkl"
        c_csv = self._to_container_path(csv_path)
        c_model = self._to_container_path(model_file)
        c_schema = self._to_container_path(schema_path)
        c_fast = self._to_container_path(_FASTPATH_DIR)

        script = textwrap.dedent(f"""\
            import json, os, pickle, sys, time
            import numpy as np
            import pandas as pd
            sys.path.insert(0, "{c_fast}")

            ISNA = "__arf_isna__"
            with open("{c_schema}") as f:
                schema = json.load(f)
            seed = int(os.environ.get("ARF_SEED", "0"))
            rng = np.random.default_rng(seed)
            np.random.seed(seed)

            df = pd.read_csv("{c_csv}", low_memory=False, encoding="utf-8-sig")
            output_columns = list(df.columns)
            numeric_cols = [c for c in schema["numeric_cols"] if c in df.columns]
            fast = os.environ.get("ARF_FAST", "1").strip().lower() in ("1", "true", "yes")
            # Tail clipping only guarded upstream arfpy forge crashes; the fast path does not need it.
            q_low = float(os.environ.get("ARF_CLIP_QUANTILE_LOW", "0" if fast else "0.001"))
            q_high = float(os.environ.get("ARF_CLIP_QUANTILE_HIGH", "1" if fast else "0.999"))
            all_nan_cols, indicators = [], {{}}
            for col in list(df.columns):
                if col in numeric_cols:
                    v = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    mask = v.isna().to_numpy()
                    if mask.all():
                        all_nan_cols.append(col)
                        df = df.drop(columns=[col])
                        continue
                    if mask.any():
                        obs = v.to_numpy()[~mask]
                        vals = v.to_numpy().copy()
                        vals[mask] = rng.choice(obs, size=int(mask.sum()), replace=True)
                        v = pd.Series(vals, index=df.index)
                        indicators[col] = ISNA + col
                        df[ISNA + col] = np.where(mask, "1", "0").astype(object)
                    if v.nunique() > 1 and 0.0 < q_low < q_high < 1.0:
                        lo, hi = v.quantile(q_low), v.quantile(q_high)
                        if pd.notna(lo) and pd.notna(hi) and lo < hi:
                            v = v.clip(lo, hi)
                    df[col] = v.astype(float)
                else:
                    s = df[col]
                    df[col] = s.astype(object).where(s.notna(), None).map(lambda x: None if x is None else str(x))
                    if df[col].isna().all():
                        all_nan_cols.append(col)
                        df = df.drop(columns=[col])
            print(f"[ARF] missing indicators: {{sorted(indicators)}}; all-NaN cols: {{all_nan_cols}}")

            num_trees = int(os.environ.get("ARF_NUM_TREES", "30"))
            delta = float(os.environ.get("ARF_DELTA", "0"))
            max_iters = int(os.environ.get("ARF_MAX_ITERS", "10"))
            early_stop = os.environ.get("ARF_EARLY_STOP", "true").strip().lower() in ("1", "true", "yes")
            verbose = os.environ.get("ARF_VERBOSE", "true").strip().lower() in ("1", "true", "yes")
            min_node_size = int(os.environ.get("ARF_MIN_NODE_SIZE", "5"))
            n_jobs = int(os.environ.get("ARF_N_JOBS", "-1"))
            print(f"[ARF] Training on {{len(df)}} rows, {{len(df.columns)}} cols (cpus={{os.cpu_count()}})")
            print(f"[ARF] Config num_trees={{num_trees}} delta={{delta}} max_iters={{max_iters}} early_stop={{early_stop}} "
                  f"min_node_size={{min_node_size}} n_jobs={{n_jobs}} fast={{fast}} seed={{seed}}", flush=True)

            t0 = time.time()
            if fast:
                from arf_fastpath import FastARF
                model = FastARF(x=df, num_trees=num_trees, delta=delta, max_iters=max_iters, early_stop=early_stop,
                                verbose=verbose, min_node_size=min_node_size, seed=seed, n_jobs=n_jobs)
            else:
                from arfpy import arf
                model = arf.arf(x=df, num_trees=num_trees, delta=delta, max_iters=max_iters, early_stop=early_stop,
                                verbose=verbose, min_node_size=min_node_size, n_jobs=n_jobs)
            print(f"[ARF] adversarial fit seconds={{time.time() - t0:.1f}}", flush=True)
            t1 = time.time()
            model.forde()
            print(f"[ARF] forde seconds={{time.time() - t1:.1f}}", flush=True)

            model._bench_meta = {{"output_columns": output_columns, "missing_indicators": indicators,
                                  "all_nan_cols": all_nan_cols, "fast": fast}}
            with open("{c_model}", "wb") as f:
                pickle.dump(model, f)
            print(f"[ARF] Model saved -> {c_model}")
        """)
        bridge = self._write_bridge_script(work_dir, "_arf_train.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log = work_dir / f"train_{ts}.log"
        try:
            result = self._run_docker(["python", c_bridge], extra_env=_arf_env())
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
        c_fast = self._to_container_path(_FASTPATH_DIR)

        script = textwrap.dedent(f"""\
            import os, pickle, sys, time
            import numpy as np
            import pandas as pd
            sys.path.insert(0, "{c_fast}")

            def _safe_forge(model, n_target, seed):
                # upstream arfpy can raise (ZeroDivisionError / probabilities not summing to 1);
                # retry with smaller n. n=1 triggers AttributeError in some versions -> never use it.
                errors = []
                candidates = []
                for n_try in (n_target, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 2):
                    nn = int(min(n_try, n_target))
                    if nn > 1 and nn not in candidates:
                        candidates.append(nn)
                for n_try in candidates:
                    try:
                        if getattr(model, "_bench_meta", {{}}).get("fast"):
                            out = model.forge(n=n_try, seed=seed)
                        else:
                            out = model.forge(n=n_try)
                        out = out.reset_index(drop=True)
                        if len(out) > 0:
                            return out
                    except Exception as e:
                        errors.append(f"n={{n_try}}: {{type(e).__name__}}: {{e}}")
                print("[ARF] forge failed after retries; last errors:", " | ".join(errors[-4:]))
                return None

            n_target = int({int(num_rows)})
            with open("{c_model}", "rb") as f:
                model = pickle.load(f)
            meta = getattr(model, "_bench_meta", {{}})
            seed = int(os.environ.get("ARF_SEED", "0"))
            np.random.seed(seed)
            chunk = max(2, int(os.environ.get("ARF_GEN_CHUNK_ROWS", "100000")))

            parts, total, tries, t0 = [], 0, 0, time.time()
            while total < n_target and tries < 10000:
                need = max(2, min(chunk, n_target - total))
                out = _safe_forge(model, need, seed + tries)
                tries += 1
                if out is None or len(out) == 0:
                    break
                parts.append(out)
                total += len(out)
                print(f"[ARF] forged {{min(total, n_target)}}/{{n_target}} rows ({{time.time() - t0:.1f}}s)", flush=True)
            if total < n_target:
                raise RuntimeError(f"ARF generated only {{total}}/{{n_target}} rows; refusing to pad with train data")
            syn = pd.concat(parts, ignore_index=True).iloc[:n_target].reset_index(drop=True)

            for col, ind in (meta.get("missing_indicators") or {{}}).items():
                if col in syn.columns and ind in syn.columns:
                    syn.loc[syn[ind].astype(str) == "1", col] = np.nan
            for col in meta.get("all_nan_cols") or []:
                syn[col] = np.nan
            out_cols = meta.get("output_columns") or [c for c in syn.columns if not str(c).startswith("__arf_isna__")]
            syn = syn[out_cols]
            syn.to_csv("{c_out}", index=False)
            print(f"[ARF] Generated {{len(syn)}} rows (requested {{n_target}}) -> {c_out}; null rates={{syn.isna().mean().round(4).to_dict()}}")
        """)
        bridge = self._write_bridge_script(work_dir, "_arf_generate.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_log = output_csv.parent / f"gen_{ts}.log"
        try:
            result = self._run_docker(["python", c_bridge], extra_env=_arf_env())
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
