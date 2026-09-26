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

import pandas as pd

from .base_adapter import BaseModelAdapter, _write_docker_log
from .features_converter import features_to_ctgan_metadata, load_features_json
from core.runner.config import MODEL_DOCKER_MAP


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

    def _write_colmeta(self, work_dir: Path, json_path: Path, csv_path: Path) -> Path:
        """
        features_converter maps ``ordinal`` to a non-categorical type, which the bridge would
        discretize with ``pd.to_numeric`` (string ordinals -> all NaN). Resolve ordinals here:
        string-valued or low-cardinality ordinals are modelled as categorical levels; only
        purely numeric ordinals with many distinct values are binned like continuous columns.
        """
        features = load_features_json(json_path)
        colmeta = features_to_ctgan_metadata(features)
        dtype_by_name = {
            str(f.get("feature_name")): str(f.get("data_type", "")).strip().lower()
            for f in features
            if f.get("feature_name")
        }
        ordinal_cols = [c["name"] for c in colmeta["columns"] if c.get("type") == "ordinal"]
        integer_columns = [n for n, t in dtype_by_name.items() if t == "integer"]
        if ordinal_cols:
            df = self.read_staged_csv(csv_path)
            max_levels = int(os.environ.get("BAYESNET_MAX_CAT_LEVELS", "0") or 0) or 256
            for entry in colmeta["columns"]:
                if entry.get("type") != "ordinal":
                    continue
                name = entry["name"]
                s = df[name].dropna() if name in df.columns else None
                num = None if s is None else pd.to_numeric(s, errors="coerce")
                numeric = num is not None and len(s) > 0 and bool(num.notna().all())
                if numeric and int(num.nunique()) > max_levels:
                    entry["type"] = "continuous"
                    if bool((num % 1 == 0).all()):
                        integer_columns.append(name)
                else:
                    entry["type"] = "categorical"
                print(f"[BayesNet] ordinal column {name!r} -> {entry['type']}")
        colmeta["integer_columns"] = integer_columns
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
        colmeta_path = self._write_colmeta(work_dir, json_path, csv_path)

        model_file = work_dir / "bayesnet_model.pkl"
        c_csv = self._to_container_path(csv_path)
        c_model = self._to_container_path(model_file)
        c_colmeta = self._to_container_path(colmeta_path)

        script = textwrap.dedent(
            f"""
            import json
            import os
            import pickle
            import warnings

            import numpy as np
            import pandas as pd
            from pgmpy.estimators import TreeSearch
            from pgmpy.models import DiscreteBayesianNetwork
            warnings.filterwarnings("ignore", category=FutureWarning)

            with open("{c_colmeta}", "r", encoding="utf-8") as _f:
                colmeta = json.load(_f)
            integer_columns = set(colmeta.get("integer_columns") or [])

            df = pd.read_csv("{c_csv}", encoding="utf-8-sig", low_memory=False)
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

            inverse = {{"categorical": {{}}, "categorical_fallback": {{}}, "continuous": {{}}, "continuous_nan_code": {{}}}}
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
                if _n_plan > 35:
                    max_bins = 5
                    max_cat_levels = 64
                if _n_plan > 55:
                    max_bins = 4
                    max_cat_levels = 32
            # Row subsampling for structure learning / CPD fitting is OFF by default (0);
            # BAYESNET_STRUCT_ROWS / BAYESNET_FIT_ROWS remain explicit, logged opt-ins.
            struct_rows = int(os.environ.get("BAYESNET_STRUCT_ROWS", "0") or 0)
            fit_rows = int(os.environ.get("BAYESNET_FIT_ROWS", "0") or 0)
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
                    # NaN -> "__NA__" level (decoded back to NaN at generation time).
                    s2 = s.astype(object).where(s.notna(), "__NA__").astype(str)
                    counts = s2.value_counts(dropna=False)
                    fallback = str(counts.index[0]) if len(counts) else ""
                    if len(counts) > max_cat_levels:
                        keep = set(counts.index[: max_cat_levels - 1].tolist())
                        s2 = s2.map(lambda x: x if x in keep else "__OTHER__")
                    uniques = sorted(s2.dropna().unique(), key=lambda x: str(x))
                    mapping = {{str(v): i for i, v in enumerate(uniques)}}
                    inverse["categorical"][name] = [uniques[i] for i in range(len(uniques))]
                    inverse["categorical_fallback"][name] = fallback
                    enc[name] = s2.map(lambda x, m=mapping: m.get(str(x), 0)).astype(int)
                else:
                    s_num = pd.to_numeric(s, errors="coerce")
                    nu = int(s_num.nunique(dropna=True))
                    q = min(max_bins, max(2, nu))
                    if nu < 2:
                        if s_num.isna().any() and nu == 1:
                            enc[name] = s_num.isna().astype(int).to_numpy()
                            inverse["continuous_nan_code"][name] = 1
                        else:
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
                        if lab.isna().any():
                            # Dedicated code for missing values so missingness is modelled,
                            # instead of silently inflating bin 0.
                            nan_code = int(len(bins) - 1)
                            enc[name] = lab.fillna(nan_code).astype(int)
                            inverse["continuous_nan_code"][name] = nan_code
                        else:
                            enc[name] = lab.astype(int)
                        inverse["continuous"][name] = bins.tolist()

            print(f"[BayesNet] Training on {{len(enc)}} rows, {{len(enc.columns)}} cols (encoded)")

            enc_struct = enc
            if struct_rows > 0 and len(enc) > struct_rows:
                enc_struct = enc.sample(n=struct_rows, random_state=0, replace=False)
                print(f"[BayesNet] TreeSearch on {{len(enc_struct)}} rows (subsample; full n={{len(enc)}})")
            dag = TreeSearch(enc_struct, root_node=root_node, n_jobs=n_jobs).estimate(estimator_type=estimator_type, edge_weights_fn=edge_weights_fn, show_progress=False)
            for col in enc.columns:
                if col not in dag.nodes():
                    dag.add_node(col)
                    print(f"[BayesNet] Added isolated node to DAG: {{col}}")
            network = DiscreteBayesianNetwork(dag)
            enc_fit = enc
            if fit_rows > 0 and len(enc) > fit_rows:
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
            import warnings

            import numpy as np
            import pandas as pd
            from pgmpy.sampling import BayesianModelSampling

            warnings.filterwarnings("ignore", category=FutureWarning)

            with open("{c_model}", "rb") as f:
                bundle = pickle.load(f)

            network = bundle["network"]
            inverse = bundle["inverse"]
            categorical_fallback = inverse.get("categorical_fallback") or {{}}
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

            nan_codes = inverse.get("continuous_nan_code") or {{}}
            for c in cols:
                codes = raw[c].astype(int).to_numpy()
                if c in inverse["categorical"]:
                    levels = inverse["categorical"][c]
                    fallback = categorical_fallback.get(c)
                    if fallback is None:
                        fallback = next((v for v in levels if str(v) not in ("__OTHER__", "__NA__")), "")
                    lv = np.array(
                        [fallback if str(v) == "__OTHER__" else v for v in levels] or [fallback],
                        dtype=object,
                    )
                    lv[lv == "__NA__"] = np.nan
                    idx = np.clip(codes, 0, len(lv) - 1)
                    out[c] = lv[idx]
                else:
                    edges = np.asarray(inverse["continuous"][c], dtype=float)
                    nan_code = nan_codes.get(c)
                    is_nan = codes == int(nan_code) if nan_code is not None else np.zeros(len(codes), dtype=bool)
                    if edges.size < 2:
                        vals = np.zeros(len(codes), dtype=float)
                    else:
                        nbin = edges.size - 1
                        k = np.clip(codes, 0, nbin - 1)
                        lo = np.minimum(edges[k], edges[k + 1])
                        hi = np.maximum(edges[k], edges[k + 1])
                        vals = rng.uniform(lo, hi)
                        if c in integer_columns:
                            vals = np.round(vals)
                    vals = vals.astype(float)
                    vals[is_nan] = np.nan
                    out[c] = vals

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
