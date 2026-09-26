"""
TabDDPM adapter

Docker image: configured through docker_images.json or BENCHMARK_TABDDPM_IMAGE.
TabDDPM uses TOML config + preprocessed .npy data.
Adapter converts unified CSV+Features JSON to TabDDPM format.

CLI: python scripts/pipeline.py --config <config.toml> --train --sample

Data handling (see `_prepare_data`):
- task type: field_registry `task_type` (via model_input_manifest) > target data_type.
  Classification -> y-conditional model (is_y_cond=true, num_classes=K);
  regression -> upstream convention is_y_cond=false (y is modelled as the first
  numerical column and sampled jointly).
- numeric NaN: a binary missing-indicator column `__tabddpm_isna__<col>` is added
  to X_cat, the NaN cells are hot-deck imputed with random observed values, and
  generated rows whose indicator is 1 get NaN re-injected. All-NaN columns are
  dropped from the model and emitted as all-NaN.
- categorical NaN: explicit "__nan__" level, decoded back to NaN.
- high cardinality: TABDDPM_MAX_CAT_LEVELS (default 100, 0 = unlimited). Levels
  beyond the top (cap-1) most frequent are bucketed into "__other__"; at decode
  time "__other__" is replaced by a draw from the empirical distribution of the
  bucketed rare levels (marginals preserved, only rare-level joint structure lost).
  Logged per column. Why a cap at all: every level is an MLP input dim fed as
  log-one-hot (zeros ~ -69); on 1267_h2_w_2_csv (pdb_id 1366, chain 1850 levels)
  cap 1000 (~2k dims) diverged to NaN within 200 steps even at lr 2e-4, cap 100 trained fine.

Hyper-parameters (see `_resolve_hparams`, written to <work_dir>/tabddpm_hparams.json):
  TABDDPM_STEPS, TABDDPM_TRAIN_BATCH_SIZE, TABDDPM_D_LAYERS ("1024,1024"),
  TABDDPM_TRAIN_LR, TABDDPM_WEIGHT_DECAY, TABDDPM_DROPOUT, TABDDPM_NUM_TIMESTEPS,
  TABDDPM_SAMPLE_BATCH_SIZE, TABDDPM_MIN_STEPS, TABDDPM_MAX_STEPS,
  TABDDPM_STEPS_PER_EPOCH (legacy: steps = epochs * value).

Sampling (scripts/sample.py + tab_ddpm/gaussian_multinomial_diffsuion.py, benchmark patches):
  TABDDPM_SAMPLE_X0_CLIP: bound for the x0 prediction in every reverse step. Default 5.2
    (support of the quantile->normal transform); 0/off disables. Without it the first
    cosine-schedule step (beta=0.999) amplifies eps errors ~31x and sampling diverges
    (nearly all numeric values collapse onto the train min/max).
  TABDDPM_SAMPLE_DDIM=1 forces DDIM; retries after a failed attempt use DDIM automatically.

Numerical stability of the multinomial posterior (see `_STABLE_SLICED_LSE_PATCH`,
applied to the per-run runtime copy in `_runtime_setup_snippet`; TABDDPM_STABLE_SLICED_LSE=0 disables):
  q_posterior normalises each categorical column with upstream `sliced_logsumexp`, which
  takes a logcumsumexp over ALL one-hot dims and subtracts prefix sums
  (log_sub_exp(cum[end], cum[start])). In float32 this cancels to log(0) = -inf / NaN whenever
  a column's unnormalised mass is < ~1e-7 of the mass of all columns before it, e.g. at t=0
  when x_t != x_0 in a 100-level column (mass ~(1-alpha_0)/K ~ e^-14.7). The -inf enters the
  KL / decoder NLL, so MLoss (the multinomial loss; GLoss is the Gaussian one) becomes nan and
  the inf gradients later blow up GLoss and the weights. Risk grows with #categorical columns
  and cardinality, not with lr: 188_xstal_tracker_csv (22 cat cols, five with 100 levels) had
  MLoss nan from step ~250 at lr 1e-3/5e-4/2.5e-4; 813_table_oser11_csv (257 cat cols, 20 rows)
  nan by step 3500. The patch computes the same quantity per slice with a max shift
  (float64-exact to ~1e-6), so model semantics are unchanged; it also covers sampling
  (p_sample -> q_posterior). The earlier "cap 1000 diverged" note above is likely the same effect.
  The lr-halving retry in `train` stays as a last-resort safety net.
"""

import json
import math
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
from core.runner.config import MODEL_DOCKER_MAP, get_synthetic_benchmark_root

_SYNTHETIC_BENCHMARK_ROOT = get_synthetic_benchmark_root()
_TABDDPM_HOST_PATH = _SYNTHETIC_BENCHMARK_ROOT / "tabddpm" / "code"
_TABDDPM_CONTAINER_PATH = "/workspace/tabddpm/code"

# Docker 采样失败时是否重试（降低 batch_size、换 seed）；仅对可恢复类错误重试
_TABDDPM_SAMPLE_RETRY_MARKERS = (
    "FoundNANsError",
    "CUDA out of memory",
    "OutOfMemoryError",
)

_NAN_SENTINEL = "__nan__"
_OTHER_TOKEN = "__other__"
_ISNA_PREFIX = "__tabddpm_isna__"

# Appended to the runtime copy of tab_ddpm/utils.py (see module docstring,
# "Numerical stability of the multinomial posterior"). Redefining the name at the
# end of utils.py is enough: gaussian_multinomial_diffsuion does `from .utils import *`
# after utils has fully executed.
_STABLE_SLICED_LSE_PATCH = '''

# --- TabQueryBench patch (tabddpm_adapter._STABLE_SLICED_LSE_PATCH) ---
# Upstream sliced_logsumexp = log_sub_exp(cumLSE[end], cumLSE[start]) over ALL one-hot
# dims concatenated; in float32 the difference cancels to log(0) = -inf (or NaN) once a
# slice's mass is < ~1e-7 of the prefix mass. Per-slice max-shifted logsumexp: same math.
def sliced_logsumexp(x, slices):
    sizes = slices[1:] - slices[:-1]
    seg = torch.repeat_interleave(torch.arange(sizes.numel(), device=x.device), sizes)
    idx = seg.unsqueeze(0).expand(x.shape[0], -1)
    m = torch.full((x.shape[0], sizes.numel()), -float('inf'), device=x.device, dtype=x.dtype)
    m = m.scatter_reduce(1, idx, x, reduce='amax', include_self=True).detach()
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = torch.zeros_like(m).scatter_add(1, idx, torch.exp(x - m.gather(1, idx)))
    return (m + torch.log(s)).gather(1, idx)
'''

_CATEGORICAL_DTYPES = {
    "categorical", "binary", "ordinal", "boolean", "bool", "id", "id_like",
    "datetime", "datetime_like", "timestamp", "text", "others", "string",
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else float(default)


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
        except Exception as exc:  # registry is optional for task type
            print(f"[TabDDPM] WARNING: could not read task_type from registry: {exc}")
    if target_dtype in ("continuous", "integer"):
        return "regression", "features.data_type"
    return "classification", "features.data_type"


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

        if sample_num_timesteps is not None:
            new_content = re.sub(
                r"(?m)^num_timesteps\s*=\s*\d+",
                f"num_timesteps = {sample_num_timesteps}",
                new_content,
            )
        return new_content

    @staticmethod
    def _sample_batch_schedule(orig_bs: int, num_rows: int, max_attempts: int) -> List[int]:
        """First attempt uses the configured sample batch (capped by
        TABDDPM_GEN_DEFAULT_MAX_BATCH, default 8192); retries halve it."""
        default_cap = _env_int("TABDDPM_GEN_DEFAULT_MAX_BATCH", 8192)
        first = max(1, min(orig_bs, num_rows, default_cap))
        ordered: List[int] = [first]
        while len(ordered) < max_attempts and ordered[-1] > 1:
            ordered.append(max(1, ordered[-1] // 2))
        return ordered[:max_attempts]

    @staticmethod
    def _sample_timestep_schedule(orig_steps: int, max_attempts: int) -> List[int]:
        """The denoiser is trained for a fixed diffusion length; sampling with a
        different num_timesteps changes the noise schedule and yields garbage.
        Keep the trained value unless TABDDPM_GEN_RETRY_REDUCE_TIMESTEPS=1
        (legacy behaviour)."""
        ordered = [max(1, orig_steps)]
        reduce = os.environ.get("TABDDPM_GEN_RETRY_REDUCE_TIMESTEPS", "").strip().lower() in ("1", "true", "yes")
        if reduce:
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

    @staticmethod
    def _runtime_setup_snippet(c_runtime: str) -> str:
        # Always refresh the runtime copy so fixes to the mounted source (e.g.
        # TABDDPM_SAMPLE_DDIM support in scripts/sample.py) reach old run dirs.
        # The stable sliced_logsumexp patch is applied to that copy only (the mounted
        # source stays untouched); TABDDPM_STABLE_SLICED_LSE=0 runs upstream as-is.
        stable_lse = os.environ.get("TABDDPM_STABLE_SLICED_LSE", "1").strip().lower() not in ("0", "false", "no")
        patch = _STABLE_SLICED_LSE_PATCH if stable_lse else ""
        return textwrap.dedent(f"""\
            import os, sys, shutil
            tabddpm_root = "{_TABDDPM_CONTAINER_PATH}"
            runtime_root = "{c_runtime}"
            assert os.path.isdir(tabddpm_root), f"TabDDPM source not mounted: {{tabddpm_root}}"

            def _ignore(_, names):
                skip = {{"__pycache__", "data", "synthetic", "result", "results", "ckpt"}}
                return [n for n in names if n in skip or n.endswith(".pyc")]

            shutil.rmtree(runtime_root, ignore_errors=True)
            shutil.copytree(tabddpm_root, runtime_root, ignore=_ignore)

            _patch = {patch!r}
            if _patch:
                _utils = os.path.join(runtime_root, "tab_ddpm", "utils.py")
                with open(_utils) as f:
                    assert "def sliced_logsumexp(" in f.read(), "upstream sliced_logsumexp not found; patch outdated"
                with open(_utils, "a") as f:
                    f.write(_patch)
                print("[TabDDPM] applied stable sliced_logsumexp patch to runtime tab_ddpm/utils.py")

            env = os.environ.copy()
            env["PYTHONPATH"] = runtime_root + (os.pathsep + env.get("PYTHONPATH", ""))

            # Wrapper that patches collections.Sequence (removed in Python 3.10+) for skorch
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
        """)

    def _tabddpm_sample_bridge_script(
        self,
        c_config: str,
        c_out: str,
        c_npy_dir: str,
        c_info: str,
        c_runtime: str,
        num_rows: int,
        decode_seed: int = 0,
    ) -> str:
        return self._runtime_setup_snippet(c_runtime) + textwrap.dedent(f"""\
            import subprocess, json
            import numpy as np
            import pandas as pd

            print(f"[TabDDPM] Sampling {num_rows} rows, ddim={{os.environ.get('TABDDPM_SAMPLE_DDIM', '0')}}")
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
            with open("{c_info}") as f:
                info = json.load(f)

            output_dir = "{c_npy_dir}"
            col_names = info.get("column_names", [])

            parts = []
            x_num_path = os.path.join(output_dir, "X_num_train.npy")
            x_cat_path = os.path.join(output_dir, "X_cat_train.npy")
            y_path = os.path.join(output_dir, "y_train.npy")

            if info.get("n_num_features", 0) and os.path.exists(x_num_path):
                parts.append(np.load(x_num_path, allow_pickle=True).astype(float))
            if info.get("n_cat_features", 0) and os.path.exists(x_cat_path):
                parts.append(np.load(x_cat_path, allow_pickle=True).astype(float))
            if os.path.exists(y_path):
                y = np.load(y_path, allow_pickle=True).astype(float)
                parts.append(y.reshape(-1, 1) if y.ndim == 1 else y)

            if not parts:
                print("[TabDDPM] WARNING: No output .npy files found")
                sys.exit(1)
            combined = np.concatenate(parts, axis=1)
            if len(col_names) != combined.shape[1]:
                raise RuntimeError(f"column count mismatch: info={{len(col_names)}} generated={{combined.shape[1]}}")
            df = pd.DataFrame(combined, columns=col_names)
            rng = np.random.default_rng({int(decode_seed)})

            def _decode(series, levels):
                vals = pd.to_numeric(series, errors="coerce").round().fillna(0).astype(int)
                vals = vals.clip(lower=0, upper=max(0, len(levels) - 1))
                return pd.Series([levels[i] for i in vals], index=series.index, dtype=object)

            rare = info.get("rare_levels") or {{}}
            cat_levels = dict(info.get("categorical_levels") or {{}})
            target_col = info.get("target_col")
            if target_col and info.get("target_categories"):
                cat_levels[target_col] = info["target_categories"]
            for col, levels in cat_levels.items():
                if col not in df.columns:
                    continue
                s = _decode(df[col], levels)
                if col in rare:
                    mask = (s == "{_OTHER_TOKEN}").to_numpy()
                    if mask.any():
                        vals = np.asarray(rare[col]["values"], dtype=object)
                        p = np.asarray(rare[col]["probs"], dtype=float)
                        s.loc[mask] = rng.choice(vals, size=int(mask.sum()), p=p / p.sum())
                s = s.where(s != "{_NAN_SENTINEL}", other=np.nan)
                df[col] = s
            for col, ind in (info.get("missing_indicators") or {{}}).items():
                if col in df.columns and ind in df.columns:
                    df.loc[df[ind].astype(str) == "1", col] = np.nan
            for col in info.get("all_nan_cols") or []:
                df[col] = np.nan
            out_cols = info.get("output_columns") or [c for c in df.columns if not c.startswith("{_ISNA_PREFIX}")]
            df = df[out_cols]
            df.to_csv("{c_out}", index=False)
            print(f"[TabDDPM] Saved {{len(df)}} rows -> {c_out}; null rates={{df.isna().mean().round(4).to_dict()}}")
        """)

    def _prepare_data(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        val_csv_path: Optional[Path] = None,
        test_csv_path: Optional[Path] = None,
        num_rows_to_generate: int = 1000,
        manifest_path: Optional[str] = None,
    ) -> Path:
        """
        将 CSV + Features JSON 转换为 TabDDPM 所需的 .npy 格式。
        返回数据目录路径。严格 train-only：所有统计量/类别空间仅由 train 计算。
        """
        import pandas as pd

        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        df_train = self.read_staged_csv(csv_path)
        output_columns = list(df_train.columns)
        features = load_features_json(json_path)
        seed = _env_int("TABDDPM_PREP_SEED", 0)
        rng = np.random.default_rng(seed)

        num_cols: List[str] = []
        cat_cols: List[str] = []
        target_col = None
        target_dtype = "continuous"

        for feat in features:
            name = feat.get("feature_name")
            if not name or name not in df_train.columns:
                continue
            dtype = str(feat.get("data_type", "continuous") or "continuous").strip().lower()
            if feat.get("is_target", False):
                target_col = name
                target_dtype = dtype
            elif dtype in _CATEGORICAL_DTYPES:
                cat_cols.append(name)
            else:
                num_cols.append(name)

        if target_col is None:
            raise ValueError("TabDDPM requires explicit target column in features (is_target=true)")

        task_type, task_source = _resolve_task_type(manifest_path, target_col, target_dtype)
        is_classification = task_type == "classification"
        print(f"[TabDDPM] target={target_col} task_type={task_type} (source={task_source})")

        missing_indicators: Dict[str, str] = {}
        all_nan_cols: List[str] = []

        def _impute_with_indicator(col: str, values: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            """Hot-deck impute NaN; return (imputed_values, indicator or None)."""
            mask = ~np.isfinite(values)
            if not mask.any():
                return values, None
            if mask.all():
                return None, None
            observed = values[~mask]
            values = values.copy()
            values[mask] = rng.choice(observed, size=int(mask.sum()), replace=True)
            return values, mask.astype(np.int64)

        # 数值列：float64 保留 epoch 类大整数精度；NaN -> 缺失指示列 + hot-deck 插补
        indicator_cols: List[Tuple[str, np.ndarray]] = []
        num_arrays: List[np.ndarray] = []
        kept_num_cols: List[str] = []
        for c in num_cols:
            vals = pd.to_numeric(df_train[c], errors="coerce").to_numpy(dtype=np.float64)
            imputed, indicator = _impute_with_indicator(c, vals)
            if imputed is None:
                all_nan_cols.append(c)
                print(f"[TabDDPM] column '{c}' is entirely NaN; excluded from model and emitted as NaN")
                continue
            kept_num_cols.append(c)
            num_arrays.append(imputed)
            if indicator is not None:
                ind_name = f"{_ISNA_PREFIX}{c}"
                missing_indicators[c] = ind_name
                indicator_cols.append((ind_name, indicator))
        num_cols = kept_num_cols
        X_num_train = np.stack(num_arrays, axis=1) if num_arrays else None

        # 类别列：NaN -> "__nan__"；高基数列 -> top (cap-1) + "__other__"
        categorical_levels: Dict[str, List[str]] = {}
        rare_levels: Dict[str, Dict[str, List[Any]]] = {}
        max_cat_levels = _env_int("TABDDPM_MAX_CAT_LEVELS", 100)
        cat_code_arrays: List[np.ndarray] = []
        for c in cat_cols:
            s = df_train[c].astype(object).where(df_train[c].notna(), _NAN_SENTINEL).astype(str)
            n_levels = int(s.nunique(dropna=False))
            if max_cat_levels > 1 and n_levels > max_cat_levels:
                counts = s.value_counts(dropna=False)
                keep = counts.index[: max_cat_levels - 1]
                rare = counts.iloc[max_cat_levels - 1:]
                rare_levels[c] = {
                    "values": [str(v) for v in rare.index.tolist()],
                    "probs": [float(x) / float(rare.sum()) for x in rare.tolist()],
                }
                s = s.where(s.isin(set(keep)), _OTHER_TOKEN)
                print(
                    f"[TabDDPM] high-cardinality column '{c}': {n_levels} levels > "
                    f"TABDDPM_MAX_CAT_LEVELS={max_cat_levels}; kept top {len(keep)} "
                    f"(covering {float(counts.iloc[: max_cat_levels - 1].sum()) / len(s):.2%} of rows), "
                    f"{len(rare)} rare levels bucketed into '{_OTHER_TOKEN}' and resampled at decode"
                )
            categories = sorted(pd.unique(s))
            categorical_levels[c] = [str(v) for v in categories]
            cat_code_arrays.append(pd.Categorical(s, categories=categories).codes.astype(np.int64))

        # 目标列
        y_series = df_train[target_col]
        y_categories = None
        if is_classification:
            y_filled = y_series.astype(object).where(y_series.notna(), _NAN_SENTINEL).astype(str)
            y_categories = sorted(pd.unique(y_filled))
            y_train = np.asarray(pd.Categorical(y_filled, categories=y_categories).codes, dtype=np.int64)
        else:
            y_vals = pd.to_numeric(y_series, errors="raise").to_numpy(dtype=np.float64)
            y_imputed, y_ind = _impute_with_indicator(target_col, y_vals)
            if y_imputed is None:
                raise ValueError(f"TabDDPM: regression target '{target_col}' is entirely NaN")
            y_train = y_imputed
            if y_ind is not None:
                ind_name = f"{_ISNA_PREFIX}{target_col}"
                missing_indicators[target_col] = ind_name
                indicator_cols.append((ind_name, y_ind))

        for ind_name, ind in indicator_cols:
            cat_cols.append(ind_name)
            categorical_levels[ind_name] = ["0", "1"]
            cat_code_arrays.append(ind)
        if missing_indicators:
            print(f"[TabDDPM] missing-indicator columns added for: {sorted(missing_indicators)}")
        onehot_width = int(sum(len(v) for v in categorical_levels.values()))
        print(f"[TabDDPM] model input width: {len(num_cols)} numeric + {onehot_width} one-hot dims")
        if onehot_width > 1000:
            print(
                f"[TabDDPM] WARNING: {onehot_width} one-hot dims; the MLP input is log-one-hot "
                "(zeros = log(1e-30) ~ -69) and training diverged to NaN at ~2k dims in tests. "
                "Consider lowering TABDDPM_MAX_CAT_LEVELS."
            )

        X_cat_train = np.stack(cat_code_arrays, axis=1) if cat_code_arrays else None

        for stale in ("X_num_train.npy", "X_cat_train.npy", "y_train.npy"):
            (data_dir / stale).unlink(missing_ok=True)
        if X_num_train is not None:
            np.save(data_dir / "X_num_train.npy", X_num_train)
        if X_cat_train is not None:
            np.save(data_dir / "X_cat_train.npy", X_cat_train)
        np.save(data_dir / "y_train.npy", y_train)

        info = {
            "name": "benchmark_dataset",
            "task_type": "multiclass" if is_classification else "regression",
            "task_type_source": task_source,
            "n_num_features": len(num_cols),
            "n_cat_features": len(cat_cols),
            "train_size": len(df_train),
            "num_col_idx": list(range(len(num_cols))),
            "cat_col_idx": list(range(len(num_cols), len(num_cols) + len(cat_cols))),
            "target_col_idx": [len(num_cols) + len(cat_cols)],
            "column_names": num_cols + cat_cols + [target_col],
            "output_columns": output_columns,
            "target_col": target_col,
            "categorical_levels": categorical_levels,
            "rare_levels": rare_levels,
            "missing_indicators": missing_indicators,
            "all_nan_cols": all_nan_cols,
            "max_cat_levels": max_cat_levels,
        }
        if y_categories is not None:
            info["target_categories"] = [str(v) for v in y_categories]
            # 必须与 y_train 编码一致，否则 embedding 索引越界
            info["num_classes"] = int(y_train.max()) + 1 if y_train.size else 0

        with open(data_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        return data_dir

    @staticmethod
    def _resolve_hparams(n_rows: int, epochs: Optional[int]) -> Dict[str, Any]:
        """
        Size-scaled defaults (upstream TabDDPM tuned configs use batch 4096,
        ~20k-30k steps, 1000 diffusion steps, 2-4 wide MLP layers):

          rows < 10k   : d_layers [512, 512],               steps 10000
          rows < 100k  : d_layers [1024, 1024, 1024],       steps 20000
          rows >= 100k : d_layers [1024, 1024, 1024, 1024], steps 30000
          batch = clamp(2**floor(log2(rows/8)), 256, 4096); lr 1e-3; T = 1000

        If `epochs` is given (runner --epochs), steps = ceil(epochs * rows / batch)
        clamped to [TABDDPM_MIN_STEPS=100, TABDDPM_MAX_STEPS=100000].
        Every value can be overridden with the matching TABDDPM_* env var.
        """
        n = max(1, int(n_rows))
        reasons: Dict[str, str] = {}
        if os.environ.get("TABDDPM_TRAIN_BATCH_SIZE", "").strip():
            batch = _env_int("TABDDPM_TRAIN_BATCH_SIZE", 256)
            reasons["batch_size"] = "env"
        else:
            batch = int(min(4096, max(256, 2 ** int(math.floor(math.log2(max(1, n // 8)))) if n >= 16 else 256)))
            reasons["batch_size"] = "auto: clamp(2**floor(log2(rows/8)), 256, 4096)"

        if n < 10_000:
            tier_layers, tier_steps = [512, 512], 10_000
        elif n < 100_000:
            tier_layers, tier_steps = [1024, 1024, 1024], 20_000
        else:
            tier_layers, tier_steps = [1024, 1024, 1024, 1024], 30_000

        min_steps = _env_int("TABDDPM_MIN_STEPS", 100)
        max_steps = _env_int("TABDDPM_MAX_STEPS", 100_000)
        if os.environ.get("TABDDPM_STEPS", "").strip():
            steps = _env_int("TABDDPM_STEPS", tier_steps)
            reasons["steps"] = "env TABDDPM_STEPS"
        elif epochs and os.environ.get("TABDDPM_STEPS_PER_EPOCH", "").strip():
            steps = int(epochs) * _env_int("TABDDPM_STEPS_PER_EPOCH", 20)
            reasons["steps"] = "legacy: epochs * TABDDPM_STEPS_PER_EPOCH"
        elif epochs:
            steps = int(math.ceil(int(epochs) * n / batch))
            reasons["steps"] = f"epochs={epochs} * rows / batch"
        else:
            steps = tier_steps
            reasons["steps"] = "auto size tier"
        clamped = max(min_steps, min(max_steps, steps))
        if clamped != steps:
            reasons["steps"] += f" (clamped {steps} -> {clamped} by TABDDPM_MIN/MAX_STEPS)"
        steps = clamped

        layers_env = os.environ.get("TABDDPM_D_LAYERS", "").strip()
        if layers_env:
            d_layers = [int(x) for x in layers_env.replace("[", "").replace("]", "").split(",") if x.strip()]
            reasons["d_layers"] = "env"
        else:
            d_layers = tier_layers
            reasons["d_layers"] = "auto size tier"

        return {
            "train_rows": n,
            "steps": int(steps),
            "batch_size": int(batch),
            "d_layers": d_layers,
            "dropout": _env_float("TABDDPM_DROPOUT", 0.0),
            "lr": _env_float("TABDDPM_TRAIN_LR", 0.001),
            "weight_decay": _env_float("TABDDPM_WEIGHT_DECAY", 0.0),
            "num_timesteps": _env_int("TABDDPM_NUM_TIMESTEPS", 1000),
            "sample_batch_size": _env_int("TABDDPM_SAMPLE_BATCH_SIZE", 8192),
            "reasons": reasons,
        }

    def _write_config_toml(
        self,
        work_dir: Path,
        data_dir: Path,
        epochs: Optional[int],
        num_samples: int,
    ) -> Path:
        """生成 TabDDPM 的 config.toml (matches format from tabddpm repo)"""
        config_path = work_dir / "config.toml"
        c_data = self._to_container_path(data_dir)
        c_parent = self._to_container_path(work_dir / "output")

        with open(data_dir / "info.json") as f:
            info = json.load(f)
        n_num = info.get("n_num_features", 0)
        n_cat = info.get("n_cat_features", 0)
        is_clf = info.get("task_type", "regression") != "regression"
        n_classes = info.get("num_classes", 2) if is_clf else 0
        # regression: upstream convention is_y_cond=false (y joins X_num); the
        # y-conditional path would sample y as class indices of unique values.
        is_y_cond = "true" if is_clf else "false"
        d_in = n_num + n_cat

        hp = self._resolve_hparams(int(info.get("train_size", 0)), epochs)
        print(f"[TabDDPM] resolved hyper-parameters: {json.dumps(hp)}")
        (work_dir / "tabddpm_hparams.json").write_text(json.dumps(hp, indent=2), encoding="utf-8")

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
is_y_cond = {is_y_cond}

[model_params.rtdl_params]
d_layers = {json.dumps(hp["d_layers"])}
dropout = {hp["dropout"]}

[diffusion_params]
num_timesteps = {hp["num_timesteps"]}
gaussian_loss_type = "mse"

[train.main]
steps = {hp["steps"]}
lr = {hp["lr"]}
weight_decay = {hp["weight_decay"]}
batch_size = {hp["batch_size"]}

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
batch_size = {max(1, min(num_samples, hp["sample_batch_size"]))}
seed = 0
"""
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

    def _train_once(self, work_dir: Path, data_dir: Path, epochs: Optional[int], c_runtime: str) -> Tuple[Path, Path]:
        config_path = self._write_config_toml(work_dir, data_dir, epochs, num_samples=1000)
        c_config = self._to_container_path(config_path)

        script = self._runtime_setup_snippet(c_runtime) + textwrap.dedent(f"""\
            import subprocess
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
        return config_path, train_log

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

        if not kwargs.get("model_input_manifest"):
            raise ValueError("TabDDPM requires model_input_manifest with staged train/features inputs")
        data_dir = self._prepare_data(
            csv_path,
            json_path,
            work_dir,
            manifest_path=str(kwargs["model_input_manifest"]),
        )
        runtime_dir = self._tabddpm_runtime_dir(work_dir)
        c_runtime = self._to_container_path(runtime_dir)

        # Safety net only: the nan seen on 188_xstal_tracker_csv / 813_table_oser11_csv came from the
        # float32 sliced_logsumexp (fixed by _STABLE_SLICED_LSE_PATCH), not from lr. A diverged
        # model only yields FoundNANsError at sampling, after every sampling retry has run. Check the
        # training log instead and retrain with the learning rate halved (TABDDPM_NAN_TRAIN_RETRIES, default 2).
        nan_retries = max(0, _env_int("TABDDPM_NAN_TRAIN_RETRIES", 2))
        base_lr = _env_float("TABDDPM_TRAIN_LR", 0.001)
        saved_lr = os.environ.get("TABDDPM_TRAIN_LR")
        try:
            for train_attempt in range(nan_retries + 1):
                os.environ["TABDDPM_TRAIN_LR"] = repr(base_lr / (2 ** train_attempt))
                config_path, train_log = self._train_once(work_dir, data_dir, epochs, c_runtime)
                if not re.search(r"Loss: nan", train_log.read_text(encoding="utf-8", errors="replace")):
                    break
                print(f"[TabDDPM] training diverged (nan loss) at lr={os.environ['TABDDPM_TRAIN_LR']}")
                if train_attempt == nan_retries:
                    raise RuntimeError(
                        f"TabDDPM training diverged to nan in {nan_retries + 1} attempts "
                        f"(lr {base_lr} halved each time); see {train_log}"
                    )
        finally:
            if saved_lr is None:
                os.environ.pop("TABDDPM_TRAIN_LR", None)
            else:
                os.environ["TABDDPM_TRAIN_LR"] = saved_lr

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
        # generate-only 时 model_path 常为 .../output/model.pt，此时 work_dir 已是「output 子目录」，
        # 采样产物 X_num_train.npy 等写在 work_dir 下，而不是 work_dir/output/。
        if work_dir.name == "output" and (work_dir / "model.pt").is_file():
            npy_dir = work_dir
            run_root = work_dir.parent
        else:
            npy_dir = work_dir / "output"
            run_root = work_dir
        runtime_dir = self._tabddpm_runtime_dir(run_root)
        c_runtime = self._to_container_path(runtime_dir)
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
        if os.environ.get("TABDDPM_SAMPLE_BATCH_SIZE", "").strip():
            orig_bs = _env_int("TABDDPM_SAMPLE_BATCH_SIZE", orig_bs)
        _retry_env = os.environ.get("TABDDPM_GEN_NAN_RETRY_MAX") or os.environ.get(
            "TABDDPM_GEN_NAN_MAX_RETRIES"
        )
        max_retries = max(1, int(_retry_env or "6"))
        sizes_env = (os.environ.get("TABDDPM_GEN_NAN_BATCH_SIZES") or "").strip()
        batch_candidates: List[int] = []
        if sizes_env:
            for part in sizes_env.split(","):
                part = part.strip()
                if part:
                    batch_candidates.append(max(1, min(int(part), num_rows)))
        if not batch_candidates:
            batch_candidates = self._sample_batch_schedule(orig_bs, num_rows, max_retries)
        seed_offset = _env_int("TABDDPM_GEN_NAN_SEED_BASE", 0)
        ts_base = datetime.now().strftime("%Y%m%d_%H%M%S")
        orig_steps = self._parse_num_timesteps(base_content)
        timestep_candidates = self._sample_timestep_schedule(orig_steps, len(batch_candidates))
        retry_trace_path = output_csv.parent / f"tabddpm_sample_retry_trace_{ts_base}.jsonl"
        force_ddim = os.environ.get("TABDDPM_SAMPLE_DDIM", "").strip().lower() in ("1", "true", "yes")

        c_out = self._to_container_path(output_csv)
        c_npy_dir = self._to_container_path(npy_dir)
        c_info = self._to_container_path(info_json)

        def _append_trace(trace: Dict[str, Any]) -> None:
            with open(retry_trace_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")

        for attempt, sample_bs in enumerate(batch_candidates):
            sample_bs = max(1, min(sample_bs, num_rows))
            sample_seed = base_seed + seed_offset + attempt
            sample_steps = timestep_candidates[min(attempt, len(timestep_candidates) - 1)]
            # First attempt: ancestral DDPM sampling (upstream default). After a
            # failure switch to DDIM, which is more stable on wide datasets that
            # hit NaNs. scripts/sample.py reads TABDDPM_SAMPLE_DDIM.
            use_ddim = force_ddim or attempt >= 1
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
                c_config, c_out, c_npy_dir, c_info, c_runtime, num_rows, decode_seed=sample_seed
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
                extra_env = {"TABDDPM_SAMPLE_DDIM": "1" if use_ddim else "0"}
                if os.environ.get("TABDDPM_SAMPLE_X0_CLIP", "").strip():
                    extra_env["TABDDPM_SAMPLE_X0_CLIP"] = os.environ["TABDDPM_SAMPLE_X0_CLIP"].strip()
                result = self._run_docker(["python", c_bridge], extra_env=extra_env)
                _write_docker_log(
                    gen_log,
                    result.stdout or "",
                    result.stderr or "",
                    getattr(result, "bench_timing", None),
                )
                trace["status"] = "success"
                _append_trace(trace)
                return self._postprocess_generated_csv(
                    output_csv, csv_path, json_path, num_rows, **kwargs
                )
            except Exception as e:
                stdout = getattr(e, "stdout", "") or ""
                stderr = getattr(e, "stderr", "") or ""
                combined = stdout + stderr
                if isinstance(e, subprocess.CalledProcessError):
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
                _append_trace(trace)
                if retryable and is_subproc and attempt + 1 < len(batch_candidates):
                    continue
                raise

        raise RuntimeError("TabDDPM generate: no batch candidates (internal error)")
