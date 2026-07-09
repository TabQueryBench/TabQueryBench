"""
BayesNet adapter

Docker image: configured through docker_images.json or BENCHMARK_BAYESNET_IMAGE.

使用 pgmpy 1.x（DiscreteBayesianNetwork + TreeSearch + BayesianModelSampling）在离散化后的表格上学习与采样，
语义接近 synthcity 的 bayesian_network 插件，但不导入 synthcity（避免与镜像内 torch/numpy 版本冲突及 pip_libs 污染）。
"""

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import features_to_ctgan_metadata, load_features_json
from ..config import MODEL_DOCKER_MAP


class BayesNetAdapter(BaseModelAdapter):
    @property
    def model_name(self) -> str:
        return "bayesnet"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["bayesnet"]

    @property
    def _needs_gpu(self) -> bool:
        return False

    def _run_docker(
        self,
        extra_args: List[str],
        extra_env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        container_workdir: Optional[str] = None,
    ):
        """降低宿主机 site-packages / 用户路径污染容器 import 的概率；限制 BLAS 线程避免部分 OOM。"""
        merged: Dict[str, str] = {
            "PYTHONNOUSERSITE": "1",
            "OPENBLAS_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OMP_NUM_THREADS": "8",
        }
        if extra_env:
            merged.update(extra_env)
        return super()._run_docker(
            extra_args,
            extra_env=merged,
            timeout=timeout,
            container_workdir=container_workdir,
        )

    def _write_colmeta(self, work_dir: Path, json_path: Path) -> Path:
        features = load_features_json(json_path)
        colmeta = features_to_ctgan_metadata(features)
        colmeta["integer_columns"] = [
            str(f.get("feature_name"))
            for f in features
            if str(f.get("data_type", "")).strip().lower() == "integer"
            and f.get("feature_name")
        ]
        out = work_dir / "bayesnet_coltypes.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(colmeta, f, indent=2, ensure_ascii=False)
        return out

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
            strict_numeric_cast=False,
        )
        colmeta_path = self._write_colmeta(work_dir, json_path)

        model_file = work_dir / "bayesnet_model.pkl"
        c_csv = self._to_container_path(csv_path)
        c_model = self._to_container_path(model_file)
        c_colmeta = self._to_container_path(colmeta_path)

        script = textwrap.dedent(
            f"""
            import json
            import os
            import pickle
            import subprocess
            import sys
            import warnings

            import numpy as np
            import pandas as pd
            from pgmpy.estimators import TreeSearch
            from pgmpy.models import DiscreteBayesianNetwork
            warnings.filterwarnings("ignore", category=FutureWarning)

            def _ensure_cloudpickle():
                try:
                    import cloudpickle  # noqa: F401
                except ModuleNotFoundError:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "--quiet", "cloudpickle"],
                    )

            _ensure_cloudpickle()

            with open("{c_colmeta}", "r", encoding="utf-8") as _f:
                colmeta = json.load(_f)
            integer_columns = set(colmeta.get("integer_columns") or [])

            df = pd.read_csv("{c_csv}")
            df = df.dropna(axis=1, how="all")
            full_column_order = list(df.columns)

            const_cols = {{}}
            for col in list(df.columns):
                if df[col].nunique(dropna=True) <= 1:
                    const_cols[col] = df[col].iloc[0] if len(df) > 0 else None
                    df = df.drop(columns=[col])
                    print(f"[BayesNet] Dropped zero-variance column '{{col}}'")

            const_path = "{c_model}".replace("bayesnet_model.pkl", "const_cols.json")
            with open(const_path, "w", encoding="utf-8") as _f:
                json.dump({{k: str(v) for k, v in const_cols.items()}}, _f)

            inverse = {{"categorical": {{}}, "continuous": {{}}}}
            enc = pd.DataFrame(index=df.index)
            _n_samples = len(df)
            _n_plan = sum(
                1 for e in colmeta["columns"] if str(e.get("name", "")) in df.columns
            )
            max_bins = int(os.environ.get("BAYESNET_MAX_BINS", "0"))
            max_cat_levels = int(os.environ.get("BAYESNET_MAX_CAT_LEVELS", "0"))
            if max_bins <= 0:
                max_bins = 10
            if max_cat_levels <= 0:
                max_cat_levels = 256
            auto_caps = os.environ.get("BAYESNET_DISABLE_AUTO_CAPS", "0").strip().lower() not in ("1", "true", "yes")
            if auto_caps and max_bins == 10 and max_cat_levels == 256:
                if _n_plan > 35 or _n_samples > 200000:
                    max_bins = 5
                    max_cat_levels = 64
                if _n_plan > 55:
                    max_bins = 4
                    max_cat_levels = 32
            struct_rows = int(os.environ.get("BAYESNET_STRUCT_ROWS", "25000"))
            fit_rows = int(os.environ.get("BAYESNET_FIT_ROWS", "120000"))
            estimator_type = (os.environ.get("BAYESNET_ESTIMATOR_TYPE", "chow-liu") or "chow-liu").strip()
            edge_weights_fn = (os.environ.get("BAYESNET_EDGE_WEIGHTS_FN", "mutual_info") or "mutual_info").strip()
            root_node = (os.environ.get("BAYESNET_ROOT_NODE", "") or "").strip() or None
            n_jobs = int(os.environ.get("BAYESNET_N_JOBS", "1"))
            print(
                f"[BayesNet] max_bins={{max_bins}}, max_cat_levels={{max_cat_levels}}, struct_rows={{struct_rows}}, fit_rows={{fit_rows}}, estimator_type={{estimator_type}}, edge_weights_fn={{edge_weights_fn}}, root_node={{root_node}}, n_jobs={{n_jobs}} "
                f"(cols_in_df={{_n_plan}}, rows={{_n_samples}})"
            )

            for entry in colmeta["columns"]:
                name = entry["name"]
                if name not in df.columns:
                    continue
                kind = entry["type"]
                s = df[name]
                if kind == "categorical":
                    s2 = s.astype(str).fillna("__NA__")
                    counts = s2.value_counts(dropna=False)
                    if len(counts) > max_cat_levels:
                        keep = set(counts.index[: max_cat_levels - 1].tolist())
                        s2 = s2.map(lambda x: x if x in keep else "__OTHER__")
                    uniques = sorted(s2.dropna().unique(), key=lambda x: str(x))
                    mapping = {{str(v): i for i, v in enumerate(uniques)}}
                    inverse["categorical"][name] = [uniques[i] for i in range(len(uniques))]
                    enc[name] = s2.map(lambda x, m=mapping: m.get(str(x), 0)).astype(int)
                else:
                    s_num = pd.to_numeric(s, errors="coerce")
                    nu = int(s_num.nunique(dropna=True))
                    q = min(max_bins, max(2, nu))
                    if nu < 2:
                        enc[name] = np.zeros(len(s_num), dtype=int)
                        lo, hi = float(s_num.min()), float(s_num.max())
                        inverse["continuous"][name] = [lo, hi]
                    else:
                        try:
                            _, bins = pd.qcut(
                                s_num, q=q, retbins=True, duplicates="drop"
                            )
                        except Exception:
                            med = float(s_num.median())
                            s2 = s_num.fillna(med)
                            _, bins = pd.qcut(
                                s2, q=min(q, 3), retbins=True, duplicates="drop"
                            )
                        bins = np.asarray(bins, dtype=float)
                        lab = pd.cut(
                            s_num, bins=bins, labels=False, include_lowest=True
                        )
                        enc[name] = lab.fillna(0).astype(int)
                        inverse["continuous"][name] = bins.tolist()

            print(f"[BayesNet] Training on {{len(enc)}} rows, {{len(enc.columns)}} cols (encoded)")

            enc_struct = enc
            if len(enc) > struct_rows:
                enc_struct = enc.sample(n=struct_rows, random_state=0, replace=False)
                print(f"[BayesNet] TreeSearch on {{len(enc_struct)}} rows (subsample; full n={{len(enc)}})")
            dag = TreeSearch(enc_struct, root_node=root_node, n_jobs=n_jobs).estimate(estimator_type=estimator_type, edge_weights_fn=edge_weights_fn, show_progress=False)
            for col in enc.columns:
                if col not in dag.nodes():
                    dag.add_node(col)
                    print(f"[BayesNet] Added isolated node to DAG: {{col}}")
            network = DiscreteBayesianNetwork(dag)
            enc_fit = enc
            if len(enc) > fit_rows:
                enc_fit = enc.sample(n=fit_rows, random_state=1, replace=False)
                print(f"[BayesNet] fit() on {{len(enc_fit)}} rows (full n={{len(enc)}})")
            network.fit(enc_fit)

            bundle = {{
                "network": network,
                "inverse": inverse,
                "column_order": list(enc.columns),
                "full_column_order": full_column_order,
                "integer_columns": list(integer_columns),
                "original_dtypes": {{c: str(df[c].dtype) for c in enc.columns}},
                "const_cols": const_cols,
            }}
            with open("{c_model}", "wb") as _f:
                pickle.dump(bundle, _f)
            print(f"[BayesNet] Model saved -> {c_model}")
            """
        )
        bridge = self._write_bridge_script(work_dir, "_bayesnet_train.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log = work_dir / f"train_{ts}.log"
        try:
            result = self._run_docker(["python", c_bridge], extra_env={k: v for k, v in os.environ.items() if k.startswith("BAYESNET_")})
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

        c_model = self._to_container_path(model_path)
        c_out = self._to_container_path(output_csv)

        script = textwrap.dedent(
            f"""
            import pickle
            import subprocess
            import sys
            import warnings

            import numpy as np
            import pandas as pd
            from pgmpy.sampling import BayesianModelSampling

            warnings.filterwarnings("ignore", category=FutureWarning)

            def _ensure_cloudpickle():
                try:
                    import cloudpickle  # noqa: F401
                except ModuleNotFoundError:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "--quiet", "cloudpickle"],
                    )

            _ensure_cloudpickle()

            with open("{c_model}", "rb") as f:
                bundle = pickle.load(f)

            network = bundle["network"]
            inverse = bundle["inverse"]
            cols = bundle["column_order"]
            integer_columns = set(bundle.get("integer_columns") or [])
            full_order = bundle.get("full_column_order") or cols
            const_cols = bundle.get("const_cols") or {{}}

            num_rows = int({num_rows})
            sampler = BayesianModelSampling(network)
            raw = sampler.forward_sample(size=num_rows, show_progress=False)
            raw = raw.reset_index(drop=True)
            if len(raw) > num_rows:
                raw = raw.iloc[:num_rows]
            _tries = 0
            while len(raw) < num_rows and _tries < 64:
                _tries += 1
                nextra = min(10000, num_rows - len(raw))
                more = sampler.forward_sample(size=max(nextra, 1), show_progress=False)
                more = more.reset_index(drop=True)
                if len(more) == 0:
                    break
                raw = pd.concat([raw, more], ignore_index=True)
                if len(raw) > num_rows:
                    raw = raw.iloc[:num_rows]

            out = pd.DataFrame(index=raw.index)
            rng = np.random.default_rng()

            for c in cols:
                if c in inverse["categorical"]:
                    levels = inverse["categorical"][c]
                    idx = raw[c].astype(int).to_numpy()
                    idx = np.clip(idx, 0, max(0, len(levels) - 1))
                    out[c] = [levels[i] for i in idx]
                else:
                    edges = np.asarray(inverse["continuous"][c], dtype=float)
                    if edges.size < 2:
                        out[c] = 0.0
                    else:
                        nbin = edges.size - 1
                        res = []
                        for k in raw[c].astype(int).to_numpy():
                            k = int(k)
                            if k < 0:
                                k = 0
                            if k >= nbin:
                                k = nbin - 1
                            lo, hi = float(edges[k]), float(edges[k + 1])
                            if hi < lo:
                                lo, hi = hi, lo
                            v = rng.uniform(lo, hi)
                            if c in integer_columns:
                                v = int(round(v))
                            res.append(v)
                        out[c] = res

            final = pd.DataFrame(index=out.index)
            for c in full_order:
                if c in const_cols:
                    final[c] = const_cols[c]
                elif c in out.columns:
                    final[c] = out[c]

            dtypes = bundle.get("original_dtypes") or {{}}
            for c, dts in dtypes.items():
                if c not in final.columns:
                    continue
                try:
                    if "int" in dts:
                        final[c] = pd.to_numeric(final[c], errors="coerce").astype("Int64")
                    elif "float" in dts:
                        final[c] = pd.to_numeric(final[c], errors="coerce")
                except Exception:
                    pass

            if len(final) != num_rows:
                final = final.iloc[:num_rows].copy()
            final = final.reset_index(drop=True)
            final.to_csv("{c_out}", index=False)
            print(f"[BayesNet] Generated {{len(final)}} rows (requested {{num_rows}}) -> {c_out}")
            """
        )
        work_dir = model_path.parent
        bridge = self._write_bridge_script(work_dir, "_bayesnet_generate.py", script)
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
