"""
TabDiff 适配器。

运行时（每个 run 独立、确定）：
  1. 容器内把镜像自带的完整 /workspace/TabDiff 复制到 <work_dir>/_tabdiff_runtime；
  2. 用 src/models/tabdiff/upstream 下的已打补丁文件覆盖（先校验镜像原文件 md5，镜像漂移直接报错）；
  3. 子进程 PYTHONPATH 只含 runtime，并校验 tabdiff / src / utils_train 均从 runtime 导入；
  4. train:    main.py --mode train，TABDIFF_ADAPTER_TRAIN=1 -> 只在最后一步保存 model_N.pt + ema_state_N.pt，不做评估；
     generate: main.py --mode test，TABDIFF_ADAPTER_SAMPLE_ONLY=1 -> 只采样写 samples.csv（无 density/MLE/C2ST）。
生成的类别 code 在宿主机按 train-only 类别表解码回原标签，再走 _postprocess_generated_csv。

超参（显式 epochs > 环境变量 > 默认值；实际取值写入 tabdiff_train_meta.json）：
  TABDIFF_STEPS (=epochs, 默认 8000, 上游论文设置)  TABDIFF_BATCH_SIZE (4096)  TABDIFF_LR (1e-3)
  TABDIFF_NUM_TIMESTEPS (50)  TABDIFF_SAMPLE_BATCH_SIZE (10000)  TABDIFF_NUM_WORKERS (0)
  TABDIFF_USE_EMA (auto: steps>=2000 用 EMA 权重)  TABDIFF_MAX_EVAL_ROWS (4096, 仅 val-loss 日志用的 held-out 行数上限)

宽离散列 holdout（规则与实现见 models/shared/high_cardinality_holdout.py，与 CTGAN / TVAE /
TabbyFlow 共用）：``unified_ctime_diffusion._subs_parameterization`` 会把**每个**类别头 pad 到
``max(levels)``，显存 ≈ ``batch x max(levels) x n_cat``。满足以下任一条件的离散列在构建 npy
bundle 前从 train/val/test 三个 CSV 中剔除，采样解码后用训练列原值回填：

  - id 型：levels > ``TABDIFF_ID_MAX_LEVELS``（默认 5000）且 levels >
    ``TABDIFF_ID_UNIQUE_RATIO``（默认 0.5）× 行数；
  - 高基数文本：levels > ``TABDIFF_HIGH_CARD_MAX_LEVELS``（默认 5000）。

clickbench_hits_0_10（50,000 行 / 77 个离散列）上命中 10 个 id 列 + 5 个高基数文本列，
``max(levels)`` 从 50,000 降到 1,131：原先 batch=4096 时单个 pad 张量就要 782 MiB、
77 个头共 ~63 GiB，日志里正是 "Tried to allocate 782.00 MiB ... 678.31 MiB is free"。目标列永不剔除。

训练期 CUDA OOM 回退（与 generate 的 sample_batch 回退对称）：训练进程报 CUDA OOM 时把
``TABDIFF_BATCH_SIZE`` 减半重试，直到 ``TABDIFF_MIN_TRAIN_BATCH_SIZE``（默认 64）。
``steps``（=epoch 数）保持不变：上游没有梯度累积，而 epoch 的语义是"过一遍训练集"，与 batch
无关，减半 batch 只改变每个 epoch 的更新次数，不改变过数据的轮数。实际生效的 batch 写回
tabdiff_train_meta.json 的 config.batch_size。
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .base_adapter import BaseModelAdapter, _write_docker_log
from .pipeline_npy_bundle import (
    DEFAULT_MAX_EVAL_ROWS,
    decode_generated_csv,
    prepare_tabular_npy_bundle,
    render_runtime_setup_code,
    tabular_bundle_slug_from_manifest,
)
from models.shared.high_cardinality_holdout import (
    DEFAULT_MAX_LEVELS,
    DEFAULT_UNIQUE_RATIO,
    prepare_bundle_holdout,
    read_holdout_columns,
    refill_holdout_csv,
)
from core.runner.config import MODEL_DOCKER_MAP

_UPSTREAM_DIR = Path(__file__).resolve().parents[1] / "upstream"
# runtime-relative path -> (vendored patched copy under upstream/, md5 of the image original it replaces)
_OVERLAYS: Dict[str, Tuple[str, str]] = {
    "tabdiff/main.py": ("tabdiff/main.py", "30a29d84b358f2d5ff17065fab66e17c"),
    "tabdiff/trainer.py": ("tabdiff/trainer.py", "d0cd8ce24f9a75c5da94d2a0814737e6"),
    "utils_train.py": ("utils_train.py", "e2bcea14c0c7b3894b13fbad2d9ac377"),
    "src/data.py": ("src/data.py", "ab15609f45e4692f5ac43967fcb77993"),
}
_IMPORT_CHECKS = ("tabdiff.main", "tabdiff.trainer", "utils_train", "src")

DEFAULT_STEPS = 8000
EMA_AUTO_MIN_STEPS = 2000


def _env_value(keys: Tuple[str, ...], cast: Callable[[str], Any], default: Any) -> Tuple[Any, str]:
    for key in keys:
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return cast(raw), f"env:{key}"
    return default, "default"


def _resolve_use_ema(raw: str, steps: int) -> bool:
    raw = (raw or "auto").strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return steps >= EMA_AUTO_MIN_STEPS


class TabDiffAdapter(BaseModelAdapter):
    _CONTAINER_TABDIFF = "/workspace/TabDiff"

    # Wide-discrete-column holdout; see the module docstring and
    # models/shared/high_cardinality_holdout.py.
    _ID_MAX_LEVELS = DEFAULT_MAX_LEVELS
    _ID_UNIQUE_RATIO = DEFAULT_UNIQUE_RATIO
    _HIGH_CARD_MAX_LEVELS = DEFAULT_MAX_LEVELS
    _HOLDOUT_COLUMNS_FILE = "tabdiff_holdout_columns.json"
    # Floor for the training-side CUDA-OOM batch-halving fallback.
    _MIN_TRAIN_BATCH_SIZE = 64

    @property
    def model_name(self) -> str:
        return "tabdiff"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["tabdiff"]

    def _extra_volumes(self):
        # Nothing is mounted over the image: the runtime is a per-run copy of the image tree.
        return []

    @staticmethod
    def _tabdiff_runtime_dir(work_dir: Path) -> Path:
        return Path(work_dir) / "_tabdiff_runtime"

    @staticmethod
    def resolve_train_config(epochs: Optional[int]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        source: Dict[str, str] = {}
        if epochs is not None and int(epochs) > 0:
            cfg["steps"], source["steps"] = int(epochs), "epochs_arg"
        else:
            cfg["steps"], source["steps"] = _env_value(("TABDIFF_STEPS",), int, DEFAULT_STEPS)
        for key, env_keys, cast, default in (
            ("batch_size", ("TABDIFF_BATCH_SIZE", "TABDIFF_TRAIN_BATCH_SIZE"), int, 4096),
            ("lr", ("TABDIFF_LR", "TABDIFF_LEARNING_RATE"), float, 1e-3),
            ("num_timesteps", ("TABDIFF_NUM_TIMESTEPS", "TABDIFF_TIMESTEPS"), int, 50),
            ("sample_batch_size", ("TABDIFF_SAMPLE_BATCH_SIZE",), int, 10000),
            ("num_workers", ("TABDIFF_NUM_WORKERS",), int, 0),
            ("max_eval_rows", ("TABDIFF_MAX_EVAL_ROWS",), int, DEFAULT_MAX_EVAL_ROWS),
            ("use_ema", ("TABDIFF_USE_EMA",), str, "auto"),
        ):
            cfg[key], source[key] = _env_value(env_keys, cast, default)
        if cfg["steps"] < 1 or cfg["batch_size"] < 1 or cfg["sample_batch_size"] < 1 or cfg["num_timesteps"] < 1:
            raise ValueError(f"tabdiff: invalid hyperparameters {cfg}")
        cfg["use_ema"] = _resolve_use_ema(cfg["use_ema"], cfg["steps"])
        cfg["source"] = source
        return cfg

    def _load_manifest(self, csv_path: Path, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {"train_csv": str(csv_path)}
        if kwargs.get("model_input_manifest"):
            with open(kwargs["model_input_manifest"], "r", encoding="utf-8") as f:
                manifest.update(json.load(f))
        if not manifest.get("task_type"):
            registry_path = self._resolve_field_registry_path(kwargs)
            if registry_path is not None:
                manifest["task_type"] = json.loads(registry_path.read_text(encoding="utf-8")).get("task_type")
        return manifest

    def _run_logged(self, bridge: Path, log_path: Path, extra_env: Dict[str, str]) -> None:
        try:
            result = self._run_docker(["python", self._to_container_path(bridge)], extra_env=extra_env)
            _write_docker_log(log_path, result.stdout or "", result.stderr or "", getattr(result, "bench_timing", None))
        except Exception as e:
            _write_docker_log(
                log_path, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or "", getattr(e, "bench_timing", None)
            )
            raise

    def _runtime_setup(self, work_dir: Path, dataname: str, fresh: bool) -> str:
        return render_runtime_setup_code(
            tag="TabDiff",
            image_root=self._CONTAINER_TABDIFF,
            runtime_dir=self._to_container_path(self._tabdiff_runtime_dir(work_dir)),
            overlay_src_dir=self._to_container_path(_UPSTREAM_DIR),
            overlays=_OVERLAYS,
            bundle_dir=self._to_container_path(work_dir / "tabular_bundle" / dataname),
            dataname=dataname,
            fresh=fresh,
            import_checks=_IMPORT_CHECKS,
            ensure_init=("tabdiff",),
        )

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
        self._validate_model_inputs(csv_path, json_path, require_target=True, strict_numeric_cast=True)
        manifest = self._load_manifest(csv_path, kwargs)
        dataname = tabular_bundle_slug_from_manifest(manifest)
        cfg = self.resolve_train_config(epochs)
        print(f"[tabdiff] train config: {json.dumps(cfg)}")

        bundle_train, bundle_val, bundle_test, holdout_cols = prepare_bundle_holdout(
            work_dir,
            csv_path,
            manifest.get("val_csv"),
            manifest.get("test_csv"),
            json_path,
            self.read_staged_csv,
            tag="tabdiff",
            holdout_file=self._HOLDOUT_COLUMNS_FILE,
            max_levels=int(os.environ.get("TABDIFF_ID_MAX_LEVELS", self._ID_MAX_LEVELS)),
            unique_ratio=float(os.environ.get("TABDIFF_ID_UNIQUE_RATIO", self._ID_UNIQUE_RATIO)),
            high_cardinality_max_levels=int(
                os.environ.get("TABDIFF_HIGH_CARD_MAX_LEVELS", self._HIGH_CARD_MAX_LEVELS)
            ),
        )
        cfg["holdout_columns"] = holdout_cols

        prepare_tabular_npy_bundle(
            work_dir,
            bundle_train,
            bundle_val,
            bundle_test,
            json_path,
            slug=dataname,
            task_type=manifest.get("task_type"),
            target_column=manifest.get("target_column"),
            max_eval_rows=cfg["max_eval_rows"],
        )
        exp_name = kwargs.get("tabdiff_exp_name") or "adapter_learnable"
        script = self._runtime_setup(work_dir, dataname, fresh=True) + textwrap.dedent(
            f"""
            subprocess.check_call([
                sys.executable, os.path.join(RT, "main.py"),
                "--dataname", name, "--mode", "train", "--gpu", "0",
                "--no_wandb", "--exp_name", {exp_name!r},
            ], env=ENV, cwd=RT)
            """
        )
        bridge = self._write_bridge_script(work_dir, "_tabdiff_train.py", script)
        extra_env = {
            "WANDB_MODE": "disabled",
            "PYTHONUNBUFFERED": "1",
            "TABDIFF_ADAPTER_TRAIN": "1",
            "TABDIFF_STEPS": str(cfg["steps"]),
            "TABDIFF_BATCH_SIZE": str(cfg["batch_size"]),
            "TABDIFF_LR": str(cfg["lr"]),
            "TABDIFF_NUM_TIMESTEPS": str(cfg["num_timesteps"]),
            "TABDIFF_SAMPLE_BATCH_SIZE": str(cfg["sample_batch_size"]),
            "TABDIFF_NUM_WORKERS": str(cfg["num_workers"]),
        }
        # Training memory grows with batch x max(levels) x n_cat (every categorical head is
        # padded to max(levels)). Mirror the sampling-side fallback: on CUDA OOM halve the
        # training batch and retry. ``steps`` (= epochs) is deliberately left alone -- upstream
        # has no gradient accumulation, and an epoch is one pass over the data whatever the batch.
        train_batch = int(cfg["batch_size"])
        min_batch = max(1, int(os.environ.get("TABDIFF_MIN_TRAIN_BATCH_SIZE", self._MIN_TRAIN_BATCH_SIZE)))
        while True:
            extra_env["TABDIFF_BATCH_SIZE"] = str(train_batch)
            train_log = work_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_b{train_batch}.log"
            try:
                self._run_logged(bridge, train_log, extra_env)
                break
            except Exception:
                oom = "CUDA out of memory" in train_log.read_text(encoding="utf-8", errors="replace")
                if not oom or train_batch <= min_batch:
                    raise
                train_batch = max(min_batch, train_batch // 2)
                print(f"[tabdiff] CUDA OOM while training; retrying with batch_size={train_batch}")
        if train_batch != int(cfg["batch_size"]):
            cfg["batch_size"] = train_batch
            cfg["source"]["batch_size"] = "cuda_oom_fallback"

        ckpt_dir = self._tabdiff_runtime_dir(work_dir) / "tabdiff" / "ckpt" / dataname / exp_name
        model_ckpt = ckpt_dir / f"model_{cfg['steps']}.pt"
        ema_ckpt = ckpt_dir / f"ema_state_{cfg['steps']}.pt"
        for p in (model_ckpt, ema_ckpt, ckpt_dir / "config.pkl"):
            if not p.is_file():
                raise FileNotFoundError(f"tabdiff: training finished without expected artifact {p}")
        meta = {
            "exp_name": exp_name,
            "dataname": dataname,
            "config": cfg,
            "runtime_dir": str(self._tabdiff_runtime_dir(work_dir)),
            "ckpt_dir": str(ckpt_dir),
            "model_ckpt": str(model_ckpt),
            "ema_ckpt": str(ema_ckpt),
        }
        (work_dir / "tabdiff_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        marker_dir = work_dir / "models_tabdiff"
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker_pt = marker_dir / "trained.pt"
        marker_pt.write_text(f"tabdiff: see {work_dir / 'tabdiff_train_meta.json'}", encoding="utf-8")
        return {"model_path": marker_pt, "work_dir": work_dir, "train_meta": meta}

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
        if model_path.is_dir():
            work_dir = model_path
        else:
            work_dir = model_path.parent.parent if model_path.parent.name.startswith("models_") else model_path.parent
        meta_path = work_dir / "tabdiff_train_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"tabdiff: missing {meta_path} (train first)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dataname = meta["dataname"]
        cfg = meta["config"]
        use_ema_env = (os.environ.get("TABDIFF_USE_EMA") or "").strip()
        use_ema = _resolve_use_ema(use_ema_env, int(cfg["steps"])) if use_ema_env else bool(cfg["use_ema"])
        ckpt = Path(meta["ema_ckpt"] if use_ema else meta["model_ckpt"])
        if not ckpt.is_file():
            raise FileNotFoundError(f"tabdiff: checkpoint not found: {ckpt}")
        sample_batch, _ = _env_value(("TABDIFF_SAMPLE_BATCH_SIZE",), int, cfg["sample_batch_size"])
        raw_csv = work_dir / "tabdiff_raw_samples.csv"
        raw_csv.unlink(missing_ok=True)
        print(f"[tabdiff] generate {num_rows} rows from {ckpt.name} (ema={use_ema}, sample_batch_size={sample_batch})")

        script = self._runtime_setup(work_dir, dataname, fresh=False) + textwrap.dedent(
            f"""
            ckpt = {self._to_container_path(ckpt)!r}
            epoch = os.path.basename(ckpt).split("_")[-1].split(".")[0]
            samples = os.path.join(os.path.dirname(ckpt).replace("ckpt", "result"), epoch, "samples.csv")
            if os.path.exists(samples):
                os.remove(samples)
            subprocess.check_call([
                sys.executable, os.path.join(RT, "main.py"),
                "--dataname", name, "--mode", "test", "--gpu", "0",
                "--no_wandb", "--exp_name", {meta["exp_name"]!r},
                "--ckpt_path", ckpt, "--num_samples_to_generate", str({int(num_rows)}),
            ], env=ENV, cwd=RT)
            if not os.path.isfile(samples):
                raise SystemExit("tabdiff: sampling finished without " + samples)
            shutil.copyfile(samples, {self._to_container_path(raw_csv)!r})
            """
        )
        bridge = self._write_bridge_script(work_dir, "_tabdiff_gen.py", script)
        extra_env = {
            "WANDB_MODE": "disabled",
            "PYTHONUNBUFFERED": "1",
            "TABDIFF_ADAPTER_SAMPLE_ONLY": "1",
            "TABDIFF_SAMPLE_BATCH_SIZE": str(sample_batch),
            "TABDIFF_NUM_WORKERS": str(cfg.get("num_workers", 0)),
        }
        # Sampling memory grows with batch x one-hot width: 10,000 rows of 1123_bigill_rnaseq_exon
        # (~70k category levels) asked for 86 GiB. On CUDA OOM, halve the batch and retry (down to 64).
        attempt = 0
        while True:
            extra_env["TABDIFF_SAMPLE_BATCH_SIZE"] = str(sample_batch)
            gen_log = work_dir / f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_b{sample_batch}.log"
            try:
                self._run_logged(bridge, gen_log, extra_env)
                break
            except Exception:
                oom = "CUDA out of memory" in gen_log.read_text(encoding="utf-8", errors="replace")
                if not oom or sample_batch <= 64:
                    raise
                attempt += 1
                sample_batch = max(64, sample_batch // 2)
                print(f"[tabdiff] CUDA OOM while sampling; retrying with sample_batch_size={sample_batch}")
        if not raw_csv.is_file():
            raise FileNotFoundError(f"tabdiff: raw samples not produced: {raw_csv}")
        summary = decode_generated_csv(work_dir / "tabular_bundle" / dataname, raw_csv, output_csv)
        print(f"[tabdiff] decoded generated categories: {summary}")
        holdout_cols = read_holdout_columns(work_dir / self._HOLDOUT_COLUMNS_FILE)
        if holdout_cols:
            if kwargs.get("model_input_manifest"):
                csv_path, json_path = self._resolve_model_inputs(
                    Path(csv_path or "."), Path(json_path or "."), kwargs
                )
            if not csv_path or not Path(csv_path).exists():
                raise ValueError("tabdiff: refilling held-out columns needs the training csv_path")
            refill_holdout_csv(
                output_csv,
                self.read_staged_csv(Path(csv_path)),
                holdout_cols,
                seed=int(os.environ.get("TABDIFF_HOLDOUT_SEED", "42")),
            )
            print(f"[tabdiff] refilled held-out columns from training values: {holdout_cols}")
        return self._postprocess_generated_csv(output_csv, csv_path, json_path, num_rows, **kwargs)
