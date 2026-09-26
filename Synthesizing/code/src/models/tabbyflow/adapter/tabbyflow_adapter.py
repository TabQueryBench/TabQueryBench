"""
TabbyFlow (ef-vfm) 适配器；与 TabDiff 共用 tabular bundle 与 runtime 组装逻辑。

运行时（每个 run 独立、确定）：
  - 不再把 synthetic_benchmark/third_party/ef-vfm（只有 ~8 个文件的旧补丁副本）挂载覆盖镜像；
  - 容器内把镜像自带的完整 /workspace/ef-vfm（ef_vfm/、src/、eval/、utils_train.py）复制到
    <work_dir>/_efvfm_runtime，再用 src/models/tabbyflow/upstream 下的补丁文件覆盖
    ef_vfm/main.py、ef_vfm/trainer.py、ef_vfm/models/flow_model.py、utils_train.py（先校验镜像原文件 md5）；
  - 子进程 PYTHONPATH 只含 runtime（去掉镜像的 PYTHONPATH=/workspace/TabDiff），并校验 ef_vfm/src/utils_train 从 runtime 导入。
  train:    main.py --mode train，EFVFM_ADAPTER_TRAIN=1 -> 最后一步保存 model_N.pt + ema_state_N.pt，不评估；
  generate: main.py --mode test，EFVFM_ADAPTER_SAMPLE_ONLY=1 -> 只采样写 samples.csv。

超参（显式 epochs > 环境变量 > 默认值；实际取值写入 tabbyflow_train_meta.json）：
  EFVFM_STEPS (=epochs, 默认 8000, 上游设置)  EFVFM_TRAIN_BATCH_SIZE (4096)  EFVFM_LR (1e-3)
  EFVFM_WARMUP_EPOCHS (上游 100)  EFVFM_SAMPLE_BATCH_SIZE (10000)  EFVFM_TRAIN_NUM_WORKERS (0)
  EFVFM_ODE_SOLVER (dopri5)  EFVFM_ODE_RTOL/ATOL (1e-5)  EFVFM_ODE_FALLBACK (1)  EFVFM_RK4_STEPS (32)
  EFVFM_ODE_MAX_STEPS (2000, dopri5 超过该步数即抛错并回退 rk4；<=0 不限)
  EFVFM_USE_EMA (auto: steps>=2000 用 EMA 权重)  EFVFM_MAX_EVAL_ROWS (4096)

宽离散列 holdout（规则与实现见 models/shared/high_cardinality_holdout.py，与 CTGAN / TVAE /
TabDiff 共用）：``modules/main_modules.py`` 会把所有类别头的 logits 拼成一个
``batch x sum(levels)`` 张量，显存随 ``sum(levels)`` 线性增长。满足以下任一条件的离散列在构建
npy bundle 前从 train/val/test 三个 CSV 中剔除，采样解码后用训练列原值回填：

  - id 型：levels > ``EFVFM_ID_MAX_LEVELS``（默认 5000）且 levels >
    ``EFVFM_ID_UNIQUE_RATIO``（默认 0.5）× 行数；
  - 高基数文本：levels > ``EFVFM_HIGH_CARD_MAX_LEVELS``（默认 5000）。

clickbench_hits_0_10（50,000 行 / 77 个离散列）上命中 10 个 id 列 + 5 个高基数文本列，
``sum(levels)`` 从 441,858 降到 3,786：原先 batch=4096 时仅这一次 ``torch.cat`` 就要
441,858 x 4096 x 4B ≈ 6.7 GiB，日志里正是 "Tried to allocate 6.74 GiB ... 3.64 GiB is free"。
目标列永不剔除。

训练期 CUDA OOM 回退（与 generate 的 sample_batch 回退对称）：训练进程报 CUDA OOM 时把
``EFVFM_TRAIN_BATCH_SIZE`` 减半重试，直到 ``EFVFM_MIN_TRAIN_BATCH_SIZE``（默认 64）。
``steps``（=epoch 数）保持不变：上游没有梯度累积，epoch 的语义是"过一遍训练集"，与 batch 无关。
实际生效的 batch 写回 tabbyflow_train_meta.json 的 config.batch_size。
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
# ef_vfm/metrics.py, modules/*, configs/* are byte-identical to the image and are not overlaid.
_OVERLAYS: Dict[str, Tuple[str, str]] = {
    "ef_vfm/main.py": ("main.py", "22a5d093b17ed74f0d7c098d6156f03a"),
    "ef_vfm/trainer.py": ("trainer.py", "87993ed3d6d8a06b362690b69dd9c49c"),
    "ef_vfm/models/flow_model.py": ("models/flow_model.py", "6b6a0468778162cf8f0f34e282ab1ddf"),
    "utils_train.py": ("utils_train.py", "a8a6160119d1abecb9d18ee80edad19c"),
}
_IMPORT_CHECKS = ("ef_vfm.main", "ef_vfm.trainer", "ef_vfm.models.flow_model", "utils_train", "src")
# Sampling-side knobs forwarded from the host environment when set.
_SAMPLING_ENV = (
    "EFVFM_ODE_SOLVER", "EFVFM_ODE_RTOL", "EFVFM_ODE_ATOL", "EFVFM_ODE_FALLBACK", "EFVFM_RK4_STEPS", "EFVFM_ODE_MAX_STEPS",
)

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


class TabbyFlowAdapter(BaseModelAdapter):
    _CONTAINER_ROOT = "/workspace/ef-vfm"

    # Wide-discrete-column holdout; see the module docstring and
    # models/shared/high_cardinality_holdout.py.
    _ID_MAX_LEVELS = DEFAULT_MAX_LEVELS
    _ID_UNIQUE_RATIO = DEFAULT_UNIQUE_RATIO
    _HIGH_CARD_MAX_LEVELS = DEFAULT_MAX_LEVELS
    _HOLDOUT_COLUMNS_FILE = "tabbyflow_holdout_columns.json"
    # Floor for the training-side CUDA-OOM batch-halving fallback.
    _MIN_TRAIN_BATCH_SIZE = 64

    @property
    def model_name(self) -> str:
        return "tabbyflow"

    @property
    def docker_image(self) -> str:
        return MODEL_DOCKER_MAP["tabbyflow"]

    def _extra_volumes(self):
        # Nothing is mounted over the image: the runtime is a per-run copy of the full image tree.
        return []

    @staticmethod
    def _efvfm_runtime_dir(work_dir: Path) -> Path:
        return Path(work_dir) / "_efvfm_runtime"

    @staticmethod
    def resolve_train_config(epochs: Optional[int]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        source: Dict[str, str] = {}
        if epochs is not None and int(epochs) > 0:
            cfg["steps"], source["steps"] = int(epochs), "epochs_arg"
        else:
            cfg["steps"], source["steps"] = _env_value(("EFVFM_STEPS",), int, DEFAULT_STEPS)
        for key, env_keys, cast, default in (
            ("batch_size", ("EFVFM_TRAIN_BATCH_SIZE", "EFVFM_BATCH_SIZE"), int, 4096),
            ("lr", ("EFVFM_LR",), float, 1e-3),
            ("warmup_epochs", ("EFVFM_WARMUP_EPOCHS",), int, None),
            ("sample_batch_size", ("EFVFM_SAMPLE_BATCH_SIZE",), int, 10000),
            ("num_workers", ("EFVFM_TRAIN_NUM_WORKERS",), int, 0),
            ("max_eval_rows", ("EFVFM_MAX_EVAL_ROWS",), int, DEFAULT_MAX_EVAL_ROWS),
            ("use_ema", ("EFVFM_USE_EMA",), str, "auto"),
        ):
            cfg[key], source[key] = _env_value(env_keys, cast, default)
        if cfg["steps"] < 1 or cfg["batch_size"] < 1 or cfg["sample_batch_size"] < 1:
            raise ValueError(f"tabbyflow: invalid hyperparameters {cfg}")
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
            tag="TabbyFlow",
            image_root=self._CONTAINER_ROOT,
            runtime_dir=self._to_container_path(self._efvfm_runtime_dir(work_dir)),
            overlay_src_dir=self._to_container_path(_UPSTREAM_DIR),
            overlays=_OVERLAYS,
            bundle_dir=self._to_container_path(work_dir / "tabular_bundle" / dataname),
            dataname=dataname,
            fresh=fresh,
            import_checks=_IMPORT_CHECKS,
            ensure_init=("ef_vfm",),
        )

    @staticmethod
    def _sampling_env() -> Dict[str, str]:
        env = {"EFVFM_ODE_FALLBACK": "1", "EFVFM_RK4_STEPS": "32"}
        for key in _SAMPLING_ENV:
            value = (os.environ.get(key) or "").strip()
            if value:
                env[key] = value
        return env

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
        print(f"[tabbyflow] train config: {json.dumps(cfg)}")

        bundle_train, bundle_val, bundle_test, holdout_cols = prepare_bundle_holdout(
            work_dir,
            csv_path,
            manifest.get("val_csv"),
            manifest.get("test_csv"),
            json_path,
            self.read_staged_csv,
            tag="tabbyflow",
            holdout_file=self._HOLDOUT_COLUMNS_FILE,
            max_levels=int(os.environ.get("EFVFM_ID_MAX_LEVELS", self._ID_MAX_LEVELS)),
            unique_ratio=float(os.environ.get("EFVFM_ID_UNIQUE_RATIO", self._ID_UNIQUE_RATIO)),
            high_cardinality_max_levels=int(
                os.environ.get("EFVFM_HIGH_CARD_MAX_LEVELS", self._HIGH_CARD_MAX_LEVELS)
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
        exp_name = kwargs.get("tabbyflow_exp_name") or "adapter_efvfm"
        script = self._runtime_setup(work_dir, dataname, fresh=True) + textwrap.dedent(
            f"""
            subprocess.check_call([
                sys.executable, os.path.join(RT, "main.py"),
                "--dataname", name, "--mode", "train", "--gpu", "0",
                "--no_wandb", "--exp_name", {exp_name!r},
            ], env=ENV, cwd=RT)
            """
        )
        bridge = self._write_bridge_script(work_dir, "_tabbyflow_train.py", script)
        extra_env = {
            "WANDB_MODE": "disabled",
            "PYTHONUNBUFFERED": "1",
            "EFVFM_ADAPTER_TRAIN": "1",
            "EFVFM_STEPS": str(cfg["steps"]),
            "EFVFM_TRAIN_BATCH_SIZE": str(cfg["batch_size"]),
            "EFVFM_LR": str(cfg["lr"]),
            "EFVFM_SAMPLE_BATCH_SIZE": str(cfg["sample_batch_size"]),
            "EFVFM_TRAIN_NUM_WORKERS": str(cfg["num_workers"]),
        }
        if cfg["warmup_epochs"] is not None:
            extra_env["EFVFM_WARMUP_EPOCHS"] = str(cfg["warmup_epochs"])
        # Training memory grows with batch x sum(levels) (all categorical logits are
        # concatenated). Mirror the sampling-side fallback: on CUDA OOM halve the training
        # batch and retry. ``steps`` (= epochs) is deliberately left alone -- upstream has no
        # gradient accumulation, and an epoch is one pass over the data whatever the batch.
        train_batch = int(cfg["batch_size"])
        min_batch = max(1, int(os.environ.get("EFVFM_MIN_TRAIN_BATCH_SIZE", self._MIN_TRAIN_BATCH_SIZE)))
        while True:
            extra_env["EFVFM_TRAIN_BATCH_SIZE"] = str(train_batch)
            train_log = work_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_b{train_batch}.log"
            try:
                self._run_logged(bridge, train_log, extra_env)
                break
            except Exception:
                oom = "CUDA out of memory" in train_log.read_text(encoding="utf-8", errors="replace")
                if not oom or train_batch <= min_batch:
                    raise
                train_batch = max(min_batch, train_batch // 2)
                print(f"[tabbyflow] CUDA OOM while training; retrying with batch_size={train_batch}")
        if train_batch != int(cfg["batch_size"]):
            cfg["batch_size"] = train_batch
            cfg["source"]["batch_size"] = "cuda_oom_fallback"

        ckpt_dir = self._efvfm_runtime_dir(work_dir) / "ef_vfm" / "ckpt" / dataname / exp_name
        model_ckpt = ckpt_dir / f"model_{cfg['steps']}.pt"
        ema_ckpt = ckpt_dir / f"ema_state_{cfg['steps']}.pt"
        for p in (model_ckpt, ema_ckpt, ckpt_dir / "config.pkl"):
            if not p.is_file():
                raise FileNotFoundError(f"tabbyflow: training finished without expected artifact {p}")
        meta = {
            "exp_name": exp_name,
            "dataname": dataname,
            "config": cfg,
            "runtime_dir": str(self._efvfm_runtime_dir(work_dir)),
            "ckpt_dir": str(ckpt_dir),
            "model_ckpt": str(model_ckpt),
            "ema_ckpt": str(ema_ckpt),
        }
        (work_dir / "tabbyflow_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        md = work_dir / "models_tabbyflow"
        md.mkdir(parents=True, exist_ok=True)
        mp = md / "trained.pt"
        mp.write_text(f"tabbyflow: see {work_dir / 'tabbyflow_train_meta.json'}", encoding="utf-8")
        return {"model_path": mp, "work_dir": work_dir, "train_meta": meta}

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
        meta_path = work_dir / "tabbyflow_train_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"tabbyflow: missing {meta_path} (train first)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dataname = meta["dataname"]
        cfg = meta["config"]
        use_ema_env = (os.environ.get("EFVFM_USE_EMA") or "").strip()
        use_ema = _resolve_use_ema(use_ema_env, int(cfg["steps"])) if use_ema_env else bool(cfg["use_ema"])
        ckpt = Path(meta["ema_ckpt"] if use_ema else meta["model_ckpt"])
        if not ckpt.is_file():
            raise FileNotFoundError(f"tabbyflow: checkpoint not found: {ckpt}")
        sample_batch, _ = _env_value(("EFVFM_SAMPLE_BATCH_SIZE",), int, cfg["sample_batch_size"])
        raw_csv = work_dir / "tabbyflow_raw_samples.csv"
        raw_csv.unlink(missing_ok=True)
        sampling_env = self._sampling_env()
        print(f"[tabbyflow] generate {num_rows} rows from {ckpt.name} (ema={use_ema}, "
              f"sample_batch_size={sample_batch}, sampling_env={sampling_env})")

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
                raise SystemExit("tabbyflow: sampling finished without " + samples)
            shutil.copyfile(samples, {self._to_container_path(raw_csv)!r})
            """
        )
        bridge = self._write_bridge_script(work_dir, "_tabbyflow_gen.py", script)
        extra_env = {
            "WANDB_MODE": "disabled",
            "PYTHONUNBUFFERED": "1",
            "EFVFM_ADAPTER_SAMPLE_ONLY": "1",
            "EFVFM_SAMPLE_BATCH_SIZE": str(sample_batch),
            **sampling_env,
        }
        # Sampling memory grows with batch x one-hot width: 10,000 rows of 1123_bigill_rnaseq_exon
        # (~70k category levels) asked for 86 GiB. On CUDA OOM, halve the batch and retry (down to 64).
        attempt = 0
        while True:
            extra_env["EFVFM_SAMPLE_BATCH_SIZE"] = str(sample_batch)
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
                print(f"[tabbyflow] CUDA OOM while sampling; retrying with sample_batch_size={sample_batch}")
        if not raw_csv.is_file():
            raise FileNotFoundError(f"tabbyflow: raw samples not produced: {raw_csv}")
        summary = decode_generated_csv(work_dir / "tabular_bundle" / dataname, raw_csv, output_csv)
        print(f"[tabbyflow] decoded generated categories: {summary}")
        holdout_cols = read_holdout_columns(work_dir / self._HOLDOUT_COLUMNS_FILE)
        if holdout_cols:
            if kwargs.get("model_input_manifest"):
                csv_path, json_path = self._resolve_model_inputs(
                    Path(csv_path or "."), Path(json_path or "."), kwargs
                )
            if not csv_path or not Path(csv_path).exists():
                raise ValueError("tabbyflow: refilling held-out columns needs the training csv_path")
            refill_holdout_csv(
                output_csv,
                self.read_staged_csv(Path(csv_path)),
                holdout_cols,
                seed=int(os.environ.get("EFVFM_HOLDOUT_SEED", "42")),
            )
            print(f"[tabbyflow] refilled held-out columns from training values: {holdout_cols}")
        return self._postprocess_generated_csv(output_csv, csv_path, json_path, num_rows, **kwargs)
