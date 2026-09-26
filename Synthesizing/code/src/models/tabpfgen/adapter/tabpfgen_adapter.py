"""
TabPFGen adapter

Docker 镜像默认见 docker_images.json，可用 BENCHMARK_TABPFGEN_IMAGE 覆盖。
TabPFGen uses pretrained TabPFN -- no training step (train() only validates inputs,
checks weights and writes metadata). generate() runs
`synthetic_benchmark/tabpfgen/src/tabpfgen/bench_bridge.py` inside the container.

Weights / auth (see `check_ready`):
  TABPFGEN_MODEL_VERSION   v2 (default) | v2.5
  TABPFGEN_CKPT_DIR        host dir with default checkpoint files
                           (default: <code>/.home/.cache/tabpfn == container $XDG_CACHE_HOME/tabpfn)
  TABPFGEN_CLASSIFIER_CKPT / TABPFGEN_REGRESSOR_CKPT   explicit host checkpoint files
  TABPFGEN_ALLOW_DOWNLOAD  0 (default: fail fast if weights missing) | 1 (download in container)
  HF_TOKEN / TABPFN_TOKEN  passed through by name only (never logged)

Preflight / download CLI (PYTHONPATH=src):
  python -m models.tabpfgen.adapter.tabpfgen_adapter check [--task classification|regression]
  python -m models.tabpfgen.adapter.tabpfgen_adapter download [--version v2|v2.5]

Generation knobs (env):
  TABPFGEN_N_SGLD_STEPS (1000)  TABPFGEN_SGLD_STEP_SIZE (0.01)  TABPFGEN_SGLD_NOISE_SCALE (0.01)
  TABPFGEN_GEN_CHUNK_ROWS (auto: pairwise budget / fit rows, clamp [256, 16384])
  TABPFGEN_SGLD_PAIRWISE_BUDGET (64000000)  TABPFGEN_N_ESTIMATORS (4)  TABPFGEN_FIT_MODE (fit_preprocessors)
  TABPFGEN_FIT_MAX_ROWS (model limit: v2=10000, v2.5=50000)  TABPFGEN_BALANCE_CLASSES (0)
  TABPFGEN_LABEL_SAMPLING (sample|argmax=upstream)  TABPFGEN_REGRESSION_SAMPLING (predictive|upstream|median)
  TABPFGEN_MANY_CLASS (auto|off)  TABPFGEN_REINJECT_MISSING (1)  TABPFGEN_CLIP_TO_TRAIN_RANGE (1)
  TABPFGEN_SEED (42)  TABPFGEN_DEVICE (auto)  TABPFGEN_PROGRESS_EVERY (0)
"""

import json
import math
import os
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .base_adapter import BaseModelAdapter, _write_docker_log
    from .features_converter import load_features_json
except ImportError:  # direct `python -m` use without the runner's compat aliases
    from models.shared.base_adapter import BaseModelAdapter, _write_docker_log
    from models.shared.features_converter import load_features_json
from core.runner.config import MODEL_DOCKER_MAP, get_synthetic_benchmark_root


_TABPFGEN_HOST_PATH = get_synthetic_benchmark_root() / "tabpfgen" / "src"
_TABPFGEN_CONTAINER_PATH = "/workspace/tabpfgen_src"
_CKPT_MOUNT_BASE = "/workspace/tabpfn_ckpt"

# Mirrors tabpfn 6.4.1 ModelSource defaults + InferenceConfig limits.
MODEL_VERSIONS: Dict[str, Dict[str, Any]] = {
    "v2": {
        "classifier": "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
        "regressor": "tabpfn-v2-regressor.ckpt",
        "repos": {"classifier": "Prior-Labs/TabPFN-v2-clf", "regressor": "Prior-Labs/TabPFN-v2-reg"},
        "max_samples": 10_000,
        "max_features": 500,
    },
    "v2.5": {
        "classifier": "tabpfn-v2.5-classifier-v2.5_default.ckpt",
        "regressor": "tabpfn-v2.5-regressor-v2.5_default.ckpt",
        "repos": {"classifier": "Prior-Labs/tabpfn_2_5", "regressor": "Prior-Labs/tabpfn_2_5"},
        "max_samples": 50_000,
        "max_features": 2_000,
    },
}
_VERSION_ALIASES = {
    "2": "v2", "v2": "v2", "v2.0": "v2",
    "2.5": "v2.5", "v2.5": "v2.5", "v2_5": "v2.5", "2_5": "v2.5",
}
MAX_TABPFN_CLASSES = 10
_MIN_CKPT_BYTES = 1_000_000
_TASK_KIND = {"classification": "classifier", "regression": "regressor"}


# ---------------------------------------------------------------- env helpers

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: Optional[int], minimum: int = 1) -> Optional[int]:
    value = _env(name)
    if value is None or value.lower() == "auto":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _env_choice(name: str, default: str, choices: Tuple[str, ...]) -> str:
    value = (_env(name) or default).lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


def _code_root() -> Path:
    from core.runner.config import get_dataset_root

    return get_dataset_root().parent


# ------------------------------------------------------------ model/weights spec

def resolve_model_spec() -> Dict[str, Any]:
    """Resolve backend / version / checkpoint host paths from env (no I/O besides stat)."""
    backend = (_env("TABPFGEN_BACKEND", "local") or "local").lower()
    raw_version = (_env("TABPFGEN_MODEL_VERSION", "v2") or "v2").lower()
    version = _VERSION_ALIASES.get(raw_version)
    if version is None:
        raise ValueError(
            f"TABPFGEN_MODEL_VERSION={raw_version!r} is not supported; use v2 or v2.5"
        )
    ckpt_dir = Path(_env("TABPFGEN_CKPT_DIR") or (_code_root() / ".home" / ".cache" / "tabpfn"))
    checkpoints: Dict[str, Path] = {}
    explicit: Dict[str, bool] = {}
    for kind, env_name in (
        ("classifier", "TABPFGEN_CLASSIFIER_CKPT"),
        ("regressor", "TABPFGEN_REGRESSOR_CKPT"),
    ):
        override = _env(env_name)
        explicit[kind] = bool(override)
        checkpoints[kind] = (
            Path(override).expanduser() if override else ckpt_dir / MODEL_VERSIONS[version][kind]
        )
    return {
        "backend": backend,
        "version": version,
        "ckpt_dir": ckpt_dir,
        "checkpoints": checkpoints,
        "explicit": explicit,
        "allow_download": _env_bool("TABPFGEN_ALLOW_DOWNLOAD", False),
    }


def _effective_version(spec: Dict[str, Any], kind: str) -> str:
    # tabpfn infers v2.5 iff "v2.5" appears in the checkpoint filename.
    return "v2.5" if "v2.5" in spec["checkpoints"][kind].name else "v2"


def check_ready(task: Optional[str] = None) -> Tuple[bool, str]:
    """
    Lightweight preflight (no Docker, no network): are TabPFN weights / auth in place?

    Args:
        task: "classification" | "regression" | None (check both).
    Returns:
        (ok, human-readable message). Never includes secret values.
    """
    try:
        spec = resolve_model_spec()
    except ValueError as exc:
        return False, f"TabPFGen: invalid configuration: {exc}"

    token_state = (
        f"HF_TOKEN set={'yes' if _env('HF_TOKEN') else 'no'}, "
        f"TABPFN_TOKEN set={'yes' if _env('TABPFN_TOKEN') else 'no'}"
    )
    if spec["backend"] == "client":
        if _env("TABPFN_TOKEN"):
            return True, f"TabPFGen: backend=client (Prior Labs API), {token_state}"
        return False, (
            "TabPFGen: TABPFGEN_BACKEND=client requires TABPFN_TOKEN "
            "(get one at https://ux.priorlabs.ai); or unset TABPFGEN_BACKEND to use local weights."
        )

    if task is not None and task not in _TASK_KIND:
        return False, f"TabPFGen: unknown task {task!r}"
    tasks = [task] if task else list(_TASK_KIND)
    version = spec["version"]
    lines = [f"TabPFGen: model_version={version}, ckpt_dir={spec['ckpt_dir']}, {token_state}"]
    ok = True
    for t in tasks:
        kind = _TASK_KIND[t]
        path = spec["checkpoints"][kind]
        eff = _effective_version(spec, kind)
        note = ""
        if spec["explicit"][kind] and eff != version:
            note = f" (filename implies {eff}; tabpfn will load it as {eff})"
        if path.is_file() and path.stat().st_size >= _MIN_CKPT_BYTES:
            size_mb = path.stat().st_size / 1e6
            lines.append(f"  [ok] {kind}: {path} ({size_mb:.1f} MB){note}")
            continue
        if spec["allow_download"] and not spec["explicit"][kind]:
            lines.append(
                f"  [download] {kind}: {path} missing; will be downloaded in-container from "
                f"https://huggingface.co/{MODEL_VERSIONS[version]['repos'][kind]} "
                "(TABPFGEN_ALLOW_DOWNLOAD=1)"
            )
            continue
        ok = False
        state = "too small / corrupt" if path.is_file() else "not found"
        lines.append(f"  [missing] {kind}: {path} ({state}){note}")
    if not ok:
        repo = MODEL_VERSIONS[version]["repos"]
        lines.append(
            "Fix (any one):\n"
            f"  1) download once: PYTHONPATH=src python -m models.tabpfgen.adapter.tabpfgen_adapter download --version {version}\n"
            "  2) point to existing files: export TABPFGEN_CLASSIFIER_CKPT=/abs/x.ckpt TABPFGEN_REGRESSOR_CKPT=/abs/y.ckpt\n"
            "     (or TABPFGEN_CKPT_DIR=/dir containing the default filenames)\n"
            "  3) allow download at generate time: export TABPFGEN_ALLOW_DOWNLOAD=1\n"
            f"Source repos: {sorted(set(repo.values()))}. v2.5 weights are under the TabPFN-2.5 "
            "license (non-commercial): accept it on the HF model page and export HF_TOKEN=<read token> "
            "if the repo is gated for your account. v2 is used by default."
        )
    return ok, "\n".join(lines)


# -------------------------------------------------------------------- adapter

class TabPFGenAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._ckpt_mounts: List[Tuple[Path, str]] = []
        self._cpu_only = False

    @property
    def model_name(self) -> str:
        return "tabpfgen"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["tabpfgen"]

    @property
    def _needs_gpu(self) -> bool:
        return not (self._cpu_only or (_env("TABPFGEN_DEVICE", "auto") or "").lower() == "cpu")

    def _extra_volumes(self):
        return [(_TABPFGEN_HOST_PATH, _TABPFGEN_CONTAINER_PATH)] + [
            (str(h), c) for h, c in self._ckpt_mounts
        ]

    def _base_env(self, allow_network: bool) -> Dict[str, str]:
        env: Dict[str, str] = {
            "PYTHONPATH": _TABPFGEN_CONTAINER_PATH,
            "PYTHONUNBUFFERED": "1",
            "TABPFN_DISABLE_TELEMETRY": _env("TABPFN_DISABLE_TELEMETRY", "1"),
            "HF_HUB_DISABLE_TELEMETRY": _env("HF_HUB_DISABLE_TELEMETRY", "1"),
            "HF_HUB_OFFLINE": "0" if allow_network else "1",
        }
        # Secrets / pass-through: only names reach the docker command line.
        for key in (
            "HF_TOKEN",
            "TABPFN_TOKEN",
            "TABPFN_NO_BROWSER",
            "TABPFN_CLIENT_NO_BROWSER",
            "TABPFGEN_BACKEND",
            "TABPFN_ALLOW_CPU_LARGE_DATASET",
        ):
            value = _env(key)
            if value:
                env[key] = value
        return env

    @property
    def _extra_env(self) -> Dict[str, str]:
        spec = resolve_model_spec()
        return self._base_env(allow_network=spec["allow_download"] or spec["backend"] == "client")

    def _container_path_for(self, host_path: Path) -> str:
        host_path = Path(host_path).expanduser().resolve()
        root = self._get_project_root().resolve()
        try:
            host_path.relative_to(root)
            return self._to_container_path(host_path)
        except ValueError:
            pass
        data_root = _env("BENCHMARK_DOCKER_HOST_DATA_ROOT", "/data/jialinzhang")
        if data_root:
            dr = Path(data_root).expanduser().resolve()
            if dr.exists():
                try:
                    host_path.relative_to(dr)
                    return str(host_path)
                except ValueError:
                    pass
        parent = host_path.parent
        for mounted_host, mounted_c in self._ckpt_mounts:
            if mounted_host == parent:
                return f"{mounted_c}/{host_path.name}"
        c_dir = f"{_CKPT_MOUNT_BASE}/{len(self._ckpt_mounts)}"
        self._ckpt_mounts.append((parent, c_dir))
        return f"{c_dir}/{host_path.name}"

    def _require_ready(self, task: str) -> Dict[str, Any]:
        ok, message = check_ready(task)
        print(message)
        if not ok:
            raise RuntimeError(message)
        return resolve_model_spec()

    # ----------------------------------------------------------- helpers

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

    def _load_registry(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        path = self._resolve_field_registry_path(kwargs)
        if path is None:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _is_classification(
        self,
        features,
        target_col: str,
        task_type: Optional[str] = None,
    ) -> bool:
        task = str(task_type or "").strip().lower()
        if task in ("classification", "binary_classification", "multiclass_classification"):
            return True
        if task == "regression":
            return False
        for feat in features:
            if feat.get("feature_name") == target_col:
                dtype = feat.get("data_type", "continuous").lower()
                return dtype in ("categorical", "binary", "ordinal", "boolean")
        return False

    @staticmethod
    def _column_kinds(features, registry: Dict[str, Any]) -> Dict[str, str]:
        kinds: Dict[str, str] = {}
        for field in registry.get("fields") or []:
            name = field.get("name")
            if name:
                kinds[name] = str(field.get("semantic_type") or "").lower()
        for feat in features:
            name = feat.get("feature_name")
            if name and feat.get("data_type"):
                kinds[name] = str(feat["data_type"]).lower()
        return kinds

    def _gen_settings(self, version: str, n_fit_rows: int, n_features: int) -> Dict[str, Any]:
        limit = int(MODEL_VERSIONS[version]["max_samples"])
        fit_cap = _env_int("TABPFGEN_FIT_MAX_ROWS", limit)
        ignore_limits = _env_bool("TABPFGEN_IGNORE_PRETRAINING_LIMITS", False)
        if fit_cap > limit and min(n_fit_rows, fit_cap) > limit:
            print(
                f"[tabpfgen] WARNING: TABPFGEN_FIT_MAX_ROWS={fit_cap} exceeds TabPFN {version} "
                f"pretraining limit {limit}; enabling ignore_pretraining_limits (slower, may degrade)"
            )
            ignore_limits = True
        if n_features > int(MODEL_VERSIONS[version]["max_features"]):
            print(
                f"[tabpfgen] WARNING: {n_features} features > TabPFN {version} limit "
                f"{MODEL_VERSIONS[version]['max_features']}; enabling ignore_pretraining_limits"
            )
            ignore_limits = True
        return {
            "model_sample_limit": limit,
            "fit_max_rows": fit_cap,
            "ignore_pretraining_limits": ignore_limits,
            "n_sgld_steps": _env_int("TABPFGEN_N_SGLD_STEPS", 1000),
            "sgld_step_size": _env_float("TABPFGEN_SGLD_STEP_SIZE", 0.01),
            "sgld_noise_scale": _env_float("TABPFGEN_SGLD_NOISE_SCALE", 0.01),
            "chunk_rows": _env_int("TABPFGEN_GEN_CHUNK_ROWS", None),
            "pairwise_budget": _env_int("TABPFGEN_SGLD_PAIRWISE_BUDGET", 64_000_000),
            "n_estimators": _env_int("TABPFGEN_N_ESTIMATORS", 4),
            "fit_mode": _env_choice(
                "TABPFGEN_FIT_MODE",
                "fit_preprocessors",
                ("fit_preprocessors", "fit_with_cache", "low_memory"),
            ),
            "balance_classes": _env_bool("TABPFGEN_BALANCE_CLASSES", False),
            "label_sampling": _env_choice("TABPFGEN_LABEL_SAMPLING", "sample", ("sample", "argmax")),
            "regression_sampling": _env_choice(
                "TABPFGEN_REGRESSION_SAMPLING", "predictive", ("predictive", "upstream", "median")
            ),
            "many_class": _env_choice("TABPFGEN_MANY_CLASS", "auto", ("auto", "off")),
            "reinject_missing": _env_bool("TABPFGEN_REINJECT_MISSING", True),
            "clip_to_train_range": _env_bool("TABPFGEN_CLIP_TO_TRAIN_RANGE", True),
            "seed": _env_int("TABPFGEN_SEED", 42, minimum=0),
            "device": _env("TABPFGEN_DEVICE", "auto"),
            "progress_every": _env_int("TABPFGEN_PROGRESS_EVERY", 0, minimum=0),
        }

    # ------------------------------------------------------------- train

    def train(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        epochs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """TabPFGen has no training (uses pretrained TabPFN). Validate, preflight, save metadata."""
        t_train0 = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        csv_path, json_path = self._resolve_model_inputs(csv_path, json_path, kwargs)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._validate_model_inputs(
            csv_path=csv_path,
            json_path=json_path,
            require_target=True,
            strict_numeric_cast=False,
        )

        df = self.read_staged_csv(csv_path)
        features = load_features_json(json_path)
        manifest = self._load_model_manifest(kwargs)
        registry = self._load_registry(kwargs)
        target_col = self._find_target_col(features, list(df.columns))
        task_type = manifest.get("task_type") or registry.get("task_type")
        is_clf = self._is_classification(features, target_col, task_type=task_type)
        task = "classification" if is_clf else "regression"

        spec = self._require_ready(task)
        n_classes = int(df[target_col].dropna().nunique()) if is_clf else None
        messages = [f"[TabPFGen] task={task} target={target_col!r} rows={len(df)} cols={len(df.columns)}"]
        if is_clf:
            if n_classes < 2:
                raise ValueError(f"TabPFGen: classification target {target_col!r} has {n_classes} class(es)")
            if n_classes > MAX_TABPFN_CLASSES:
                if _env_choice("TABPFGEN_MANY_CLASS", "auto", ("auto", "off")) == "off":
                    raise ValueError(
                        f"TabPFGen: target {target_col!r} has {n_classes} classes > TabPFN limit "
                        f"{MAX_TABPFN_CLASSES}; set TABPFGEN_MANY_CLASS=auto"
                    )
                messages.append(
                    f"[TabPFGen] {n_classes} classes > {MAX_TABPFN_CLASSES}: ManyClassClassifier will be used"
                )
        settings = self._gen_settings(spec["version"], len(df), len(df.columns) - 1)
        if len(df) > settings["fit_max_rows"]:
            messages.append(
                f"[TabPFGen] model limit: TabPFN fit set will be stratified-subsampled "
                f"{len(df)} -> {settings['fit_max_rows']} rows (TABPFGEN_FIT_MAX_ROWS)"
            )
        for m in messages:
            print(m)

        meta = {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "target_col": target_col,
            "is_classification": is_clf,
            "task_type": task_type or task,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_classes": n_classes,
            "model_version": spec["version"],
            "backend": spec["backend"],
        }
        meta_path = work_dir / "tabpfgen_meta.json"
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
            + "\n".join(messages)
            + f"\n[TabPFGen] No training needed (pretrained). Meta saved to {meta_path}\n"
        )

        return {"model_path": work_dir, "work_dir": work_dir, "meta_path": meta_path}

    # ---------------------------------------------------------- generate

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
        registry = self._load_registry(kwargs)
        if manifest:
            csv_path, json_path = self._resolve_model_inputs(
                Path(csv_path or "."), Path(json_path or "."), kwargs
            )
        meta_path = work_dir / "tabpfgen_meta.json"
        task_type = str(manifest.get("task_type") or registry.get("task_type") or "").strip().lower() or None
        target_col = None
        is_clf = None
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            target_col = meta["target_col"]
            is_clf = meta["is_classification"]
            if csv_path is None:
                csv_path = Path(meta["csv_path"])
            if json_path is None:
                json_path = Path(meta["json_path"])

        manifest_target_col = str(manifest.get("target_column") or "").strip()
        if manifest_target_col:
            target_col = manifest_target_col

        csv_path = Path(csv_path) if csv_path else None
        if not csv_path or not csv_path.exists():
            raise ValueError("generate requires a valid csv_path")
        features = load_features_json(json_path) if json_path and Path(json_path).exists() else []
        if not target_col and features:
            import pandas as pd

            header = list(pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns)
            target_col = self._find_target_col(features, header)
        if not target_col:
            raise ValueError("TabPFGen generate requires an explicit target column")
        if task_type or is_clf is None:
            is_clf = self._is_classification(features, target_col, task_type=task_type)
        task = "classification" if is_clf else "regression"

        spec = self._require_ready(task)
        kind = _TASK_KIND[task]

        train_df = self.read_staged_csv(csv_path)
        header = list(train_df.columns)
        n_rows_train = int(train_df[target_col].notna().sum())
        del train_df
        settings = self._gen_settings(spec["version"], n_rows_train, len(header) - 1)

        self._ckpt_mounts = []
        checkpoints: Dict[str, str] = {}
        if spec["backend"] != "client":
            checkpoints[kind] = self._container_path_for(spec["checkpoints"][kind])
        cfg = {
            "train_csv": self._to_container_path(csv_path),
            "output_csv": self._to_container_path(output_csv),
            "target_col": target_col,
            "is_classification": bool(is_clf),
            "num_rows": int(num_rows),
            "column_kinds": self._column_kinds(features, registry),
            "backend": spec["backend"],
            "model_version": _effective_version(spec, kind),
            "checkpoints": checkpoints,
            "allow_download": spec["allow_download"],
            **settings,
        }

        fit_rows = min(n_rows_train, settings["fit_max_rows"])
        chunk = settings["chunk_rows"] or int(
            min(16384, max(256, settings["pairwise_budget"] // max(1, fit_rows)))
        )
        chunk = max(1, min(chunk, int(num_rows)))
        print(
            f"[tabpfgen] generate: task={task} rows={num_rows} model={cfg['model_version']} "
            f"fit_rows={fit_rows}/{n_rows_train} (cap={settings['fit_max_rows']}) "
            f"chunk_rows={chunk} chunks={math.ceil(int(num_rows) / chunk)} "
            f"sgld_steps={settings['n_sgld_steps']} n_estimators={settings['n_estimators']} "
            "(calibrated ETA is logged by the container)"
        )

        cfg_path = output_csv.parent / "_tabpfgen_generate_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
        script = textwrap.dedent(
            f"""\
            from tabpfgen.bench_bridge import main
            main({self._to_container_path(cfg_path)!r})
            """
        )
        bridge = self._write_bridge_script(output_csv.parent, "_tabpfgen_generate.py", script)
        c_bridge = self._to_container_path(bridge)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_log = output_csv.parent / f"gen_{ts}.log"
        try:
            result = self._run_docker(["python", c_bridge], extra_env=self._extra_env)
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

    # ---------------------------------------------------------- weights

    def download_weights(self, version: Optional[str] = None, tasks: Optional[List[str]] = None) -> None:
        """Download default TabPFN checkpoints into TABPFGEN_CKPT_DIR using tabpfn's own loader."""
        if version:
            os.environ["TABPFGEN_MODEL_VERSION"] = version
        spec = resolve_model_spec()
        tasks = tasks or list(_TASK_KIND)
        jobs = []
        self._ckpt_mounts = []
        for t in tasks:
            kind = _TASK_KIND[t]
            host = spec["checkpoints"][kind]
            host.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((self._container_path_for(host), _effective_version(spec, kind), kind))
        script = textwrap.dedent(
            f"""\
            import sys
            from pathlib import Path
            from tabpfn.constants import ModelVersion
            from tabpfn.model_loading import download_model
            failed = 0
            for path, version, which in {jobs!r}:
                p = Path(path)
                if p.exists():
                    print(f"[TabPFGen] already present: {{p}}")
                    continue
                res = download_model(p, version=ModelVersion(version), which=which, model_name=p.name)
                if res != "ok":
                    failed += 1
                    print(f"[TabPFGen] download FAILED for {{p}}:")
                    for err in res:
                        print(err)
                else:
                    print(f"[TabPFGen] downloaded {{p}} ({{p.stat().st_size}} bytes)")
            sys.exit(1 if failed else 0)
            """
        )
        bridge_dir = self._get_project_root() / ".home" / ".cache" / "tabpfgen_bridge"
        bridge = self._write_bridge_script(bridge_dir, "_tabpfgen_download.py", script)
        self._cpu_only = True
        try:
            self._run_docker(
                ["python", self._to_container_path(bridge)],
                extra_env=self._base_env(allow_network=True),
            )
        finally:
            self._cpu_only = False


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="tabpfgen_adapter", description="TabPFGen weights preflight")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_check = sub.add_parser("check", help="check weights/auth without Docker")
    p_check.add_argument("--task", choices=sorted(_TASK_KIND), default=None)
    p_dl = sub.add_parser("download", help="download default checkpoints via Docker")
    p_dl.add_argument("--version", default=None, help="v2 | v2.5 (default: TABPFGEN_MODEL_VERSION or v2)")
    p_dl.add_argument("--task", choices=sorted(_TASK_KIND), default=None)
    args = parser.parse_args(argv)
    if args.cmd == "download":
        TabPFGenAdapter().download_weights(args.version, [args.task] if args.task else None)
    ok, message = check_ready(getattr(args, "task", None))
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_cli())
