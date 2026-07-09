#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark 专用模型统一入口

支持 --train、--generate 布尔参数控制执行阶段。
用法:
  PYTHONPATH=synthetic_generation/src python -m core.runner.runner --model ctgan --dataset c1 --train --generate
  PYTHONPATH=synthetic_generation/src python -m core.runner.runner --model ctgan --dataset c1 --train
  PYTHONPATH=synthetic_generation/src python -m core.runner.runner --model ctgan --dataset c1 --generate --model-dir ...
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

from .config import (
    get_legacy_dataset_root,
    get_new_tabular_dataset_root,
    get_default_dataset_source,
    get_output_base,
    SUPPORTED_MODELS,
    MODEL_RUN_PREFIX,
)
from .dataset_loader import DatasetLoader
from .adapters import get_adapter
from .staging import AdapterStagingManager, StagingError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 专用模型：训练与生成统一入口"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        help="模型名称",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="数据集 ID，如 Tab-Cate1 或 c2/m1/n1",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["auto", "old", "new"],
        default=get_default_dataset_source(),
        help="数据集来源：auto(自动优先新目录) / old(Dataset) / new(DatasetNew)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="执行训练阶段",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="执行生成阶段",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=1000,
        help="Number of rows to generate. 0 = auto (same as training data rows)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数（可选，模型有默认值）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出根目录；默认见 config.get_output_base()（output-Benchmark-trainonly-v1，"
        "或 BENCHMARK_OUTPUT_ROOT / BENCHMARK_OUTPUT_LEGACY）",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="仅生成时：已有模型目录/文件路径；续训时可指向 models_* 目录或其上级 run 目录",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="显式指定已有 run 目录（与 --resume 联用）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="REaLTabFormer 断点续训：使用 work_dir/rtf_checkpoints 下 HuggingFace checkpoint",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="跳过自动生成统计报告",
    )
    return parser.parse_args()


def resolve_paths(dataset_id: str, loader: DatasetLoader) -> Tuple[Path, Path]:
    """
    根据 dataset_id 解析 csv_path 和 features_json_path。
    Returns:
        (csv_path, json_path)
    """
    csv_path = loader.get_train_path(dataset_id)
    json_path = loader.get_features_path(dataset_id)
    if not json_path.exists():
        raise FileNotFoundError(
            f"特征文件不存在: {json_path}\n"
            f"请先运行 Preprocess 生成 Features"
        )
    return csv_path, json_path


def _dataset_slug(dataset_id: str) -> str:
    """数据集 ID 转为小写 slug，如 Tab-Cate1 -> tab-cate1"""
    return dataset_id.lower().strip()


def _find_model_file_in_run(run_dir: Path) -> Optional[Path]:
    """在 run 目录下查找模型文件 (.pt/.pkl/.pth)，优先 models_* 子目录（含 REaLTabFormer 的 id*/rtf_model.pt）"""
    if (run_dir / "tabpfgen_meta.json").is_file():
        # TabPFGen 无权重文件；adapter.generate 以 work_dir 为 model_path。
        return run_dir
    if (run_dir / "tabbyflow_train_meta.json").is_file() or (run_dir / "_tabbyflow_train.py").is_file():
        # TabbyFlow checkpoint 保存在上游 ef-vfm/ckpt；generate-only 以 run_dir 为锚点恢复。
        sentinel = run_dir / "_tabbyflow_train.py"
        return sentinel if sentinel.is_file() else (run_dir / "tabbyflow_train_meta.json")
    model_exts = {".pkl", ".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
    models_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("models_")], reverse=True)
    for md in models_dirs:
        rtf_pts = sorted(md.glob("**/rtf_model.pt"))
        if rtf_pts:
            return rtf_pts[-1]
        for f in md.rglob("*"):
            if not f.is_file() or f.suffix not in model_exts:
                continue
            if "checkpoint-" in f.as_posix():
                continue
            return f
    # TabDDPM 等：权重在 run_dir/output/model.pt（非顶层、无 models_*）
    out_dir = run_dir / "output"
    if out_dir.is_dir():
        for name in ("model.pt", "model_ema.pt"):
            p = out_dir / name
            if p.is_file():
                return p
    for f in run_dir.iterdir():
        if f.is_file() and f.suffix in model_exts:
            return f
    return None


def _find_latest_run(output_base: Path, dataset_id: str, model_name: str) -> Optional[Path]:
    """在 output_base/{dataset}/{model}/ 下查找最新时间戳 run 目录"""
    model_dir = output_base / dataset_id / model_name
    if not model_dir.is_dir():
        return None
    candidates = []
    for d in model_dir.iterdir():
        if d.is_dir():
            m = re.search(r"(\d{8}_\d{6})", d.name)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                    candidates.append((ts, d))
                except ValueError:
                    pass
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def _run_has_rtf_checkpoint(run_dir: Path) -> bool:
    """run 目录下是否存在可续训的 REaLTabFormer HuggingFace checkpoint。"""
    cp_root = run_dir / "rtf_checkpoints"
    if not cp_root.is_dir():
        return False
    return any(p.is_dir() for p in cp_root.glob("checkpoint-*"))


def _find_latest_run_with_rtf_checkpoint(
    output_base: Path, dataset_id: str, model_name: str
) -> Optional[Path]:
    """在 output_base/{dataset}/{model}/ 下查找含 rtf_checkpoints/checkpoint-* 的最新 run。"""
    model_root = output_base / dataset_id / model_name
    if not model_root.is_dir():
        return None
    candidates = []
    for d in model_root.iterdir():
        if not d.is_dir():
            continue
        m = re.search(r"(\d{8}_\d{6})", d.name)
        if not m or not _run_has_rtf_checkpoint(d):
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            candidates.append((ts, d))
        except ValueError:
            pass
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def _find_latest_run_with_model(output_base: Path, dataset_id: str, model_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    """查找最新且包含模型文件的 run，返回 (work_dir, model_file)"""
    model_dir = output_base / dataset_id / model_name
    if not model_dir.is_dir():
        return None, None
    candidates = []
    for d in model_dir.iterdir():
        if d.is_dir():
            m = re.search(r"(\d{8}_\d{6})", d.name)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                    model_file = _find_model_file_in_run(d)
                    if model_file:
                        candidates.append((ts, d, model_file))
                except ValueError:
                    pass
    if not candidates:
        return None, None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def make_work_dir(
    output_base: Path,
    dataset_id: str,
    model_name: str,
) -> Path:
    """创建带时间戳的工作目录：output_base/{dataset}/{model}/{prefix}-{dataset}-{timestamp}"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _dataset_slug(dataset_id)
    prefix = MODEL_RUN_PREFIX.get(model_name, model_name)
    run_name = f"{prefix}-{slug}-{ts}"
    work_dir = output_base / dataset_id / model_name / run_name
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _assert_manifest_train_only(staged: dict) -> None:
    """
    确保 model_input_manifest 指向的 train 即 staged public gate 的 train.csv，
    且文件名明确为 train 分割（防止误把 val/test 登记为训练输入）。
    """
    mp = Path(staged["model_manifest"])
    with open(mp, encoding="utf-8") as f:
        man = json.load(f)
    st_train = Path(staged["train_csv"]).resolve()
    mt_train = Path(man["train_csv"]).resolve()
    if mt_train != st_train:
        raise ValueError(
            "model_input_manifest.train_csv 必须与 staged train 一致: "
            f"{mt_train} != {st_train}"
        )
    if mt_train.name.lower() != "train.csv":
        raise ValueError(
            "训练输入必须是 train 分割（文件名 train.csv），禁止 val/test："
            f"{mt_train}"
        )


def make_synthetic_csv_name(model_name: str, dataset_id: str, gen_ts: str, num_rows: int = 1000) -> str:
    """合成 CSV 文件名：{prefix}-{dataset}-{num_rows}-{gen_timestamp}.csv"""
    slug = _dataset_slug(dataset_id)
    prefix = MODEL_RUN_PREFIX.get(model_name, model_name)
    return f"{prefix}-{slug}-{num_rows}-{gen_ts}.csv"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _collect_run_env_snapshot(model_name: str) -> Dict[str, str]:
    prefixes = [
        'BENCHMARK_', 'ARF_', 'BAYESNET_', 'CTGAN_', 'FORESTDIFFUSION_', 'REALTABFORMER_',
        'TABBYFLOW_', 'TABDDPM_', 'TABDIFF_', 'TABPFGEN_', 'TABSYN_', 'TVAE_',
        'EFVFM_', 'WANDB_', 'OPENBLAS_', 'MKL_', 'OMP_', 'NUMEXPR_',
        'PYTORCH_', 'LOKY_', 'HF_', 'NCCL_', 'CUDA_',
    ]
    seen = {}
    for key, value in os.environ.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            seen[key] = value
    return dict(sorted(seen.items()))


def _build_run_config(
    args: argparse.Namespace,
    adapter_model: str,
    dataset_id: str,
    effective_source: str,
    work_dir: Path,
    staged: Optional[dict] = None,
    num_rows: Optional[int] = None,
    model_path: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    return {
        'schema_version': 1,
        'recorded_at': datetime.now().isoformat(timespec='seconds'),
        'dataset_id': dataset_id,
        'model': adapter_model,
        'work_dir': str(work_dir),
        'dataset_source_requested': args.dataset_source,
        'dataset_source_resolved': effective_source,
        'cli_args': _jsonable(vars(args)),
        'resolved': {
            'num_rows': num_rows,
            'model_path': str(model_path) if model_path else None,
            'output_csv': str(output_csv) if output_csv else None,
        },
        'input_artifacts': _jsonable(staged or {}),
        'env_overrides': _collect_run_env_snapshot(adapter_model),
    }


def _write_run_config(work_dir: Path, payload: Dict[str, Any]) -> Path:
    out = Path(work_dir) / 'run_config.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def run(args: argparse.Namespace) -> dict:
    """执行训练和/或生成"""
    if not args.train and not args.generate:
        raise ValueError("请至少指定 --train 或 --generate")
    if args.resume and not args.train:
        raise ValueError("--resume 需与 --train 同时使用")

    output_base = args.output_dir or get_output_base()
    loader = DatasetLoader(
        dataset_root=str(get_legacy_dataset_root()),
        new_dataset_root=str(get_new_tabular_dataset_root()),
        dataset_source=args.dataset_source,
    )
    adapter = get_adapter(args.model)

    dataset_id = args.dataset
    if not loader.validate_dataset_id(dataset_id):
        raise ValueError(
            f"未知数据集: {dataset_id} (dataset_source={args.dataset_source})"
        )

    effective_source = loader.get_effective_source(dataset_id)
    print(f"[Runner] dataset_source={args.dataset_source}, resolved={effective_source}")
    if effective_source != "new":
        raise ValueError("当前 staging/public-gate 流程仅支持 DatasetNew (dataset_source=new/auto->new)")

    # --num-rows 0 = auto: match the training CSV row count
    num_rows = args.num_rows
    result = {}
    runtime_result = {
        "dataset_id": dataset_id,
        "model": adapter.model_name,
        "run_id": None,
        "public_gate_status": "fail",
        "adapter_ready_status": "fail",
        "train_status": "skipped",
        "generate_status": "skipped",
        "reason_code": None,
        "reason_detail": None,
        "artifacts": {},
        "timings": {
            "train": {"started_at": None, "ended_at": None, "duration_sec": None},
            "generate": {"started_at": None, "ended_at": None, "duration_sec": None},
        },
    }

    if args.train:
        if args.resume:
            if args.model != "realtabformer":
                raise ValueError("--resume 当前仅支持 --model realtabformer (RTF)")
            wd: Optional[Path] = None
            if args.work_dir is not None:
                wd = Path(args.work_dir).resolve()
            elif args.model_dir is not None:
                mp = Path(args.model_dir).resolve()
                wd = mp.parent if mp.name.startswith("models_") else mp
            else:
                wd = _find_latest_run_with_rtf_checkpoint(
                    output_base, dataset_id, adapter.model_name
                )
            if wd is None or not wd.is_dir():
                raise ValueError(
                    "断点续训需指定 --work-dir 或 --model-dir（run 或 models_*），"
                    "或 output 下存在含 rtf_checkpoints/checkpoint-* 的历史 run"
                )
            if not _run_has_rtf_checkpoint(wd):
                raise ValueError(
                    f"目录中未找到可续训 checkpoint: {wd / 'rtf_checkpoints'}"
                )
            work_dir = wd
            print(f"[Runner] 断点续训，沿用 work_dir={work_dir}")
        else:
            work_dir = make_work_dir(output_base, dataset_id, adapter.model_name)
    elif args.model_dir:
        # Allow caller to override work_dir in generate-only mode, useful when
        # historical run directories are read-only.
        if args.work_dir is not None:
            work_dir = Path(args.work_dir).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            mp0 = Path(args.model_dir).resolve()
            if mp0.is_file() and mp0.suffix in {".pt", ".pth", ".pkl", ".ckpt", ".bin"}:
                work_dir = mp0.parent.parent if mp0.parent.name.startswith("models_") else mp0.parent
            elif mp0.is_dir():
                if mp0.name.startswith("id") and mp0.parent.name.startswith("models_"):
                    work_dir = mp0.parent.parent
                elif mp0.name.startswith("models_"):
                    work_dir = mp0.parent
                elif any(c.is_dir() and c.name.startswith("models_") for c in mp0.iterdir()):
                    work_dir = mp0
                elif (mp0 / "tabpfgen_meta.json").is_file():
                    work_dir = mp0
                else:
                    work_dir = mp0.parent
            else:
                work_dir = mp0.parent
    else:
        # 仅生成时未指定 model-dir：自动查找最新含模型文件的 run
        work_dir, _auto_model_file = _find_latest_run_with_model(output_base, dataset_id, adapter.model_name)
        if work_dir is None:
            raise ValueError(
                f"仅生成时需提供 --model-dir，或 output 下无已有 run（含模型文件）: "
                f"{output_base / dataset_id / adapter.model_name}"
            )
        print(f"[Runner] 使用已有 run: {work_dir.name}")

    runtime_result["run_id"] = work_dir.name

    staging = AdapterStagingManager(loader)
    try:
        staged = staging.prepare(dataset_id=dataset_id, model_name=adapter.model_name, work_dir=work_dir)
        runtime_result["public_gate_status"] = "pass"
        runtime_result["adapter_ready_status"] = "pass"
    except StagingError as e:
        runtime_result["reason_code"] = e.reason_code
        runtime_result["reason_detail"] = e.detail
        with open(work_dir / "runtime_result.json", "w", encoding="utf-8") as f:
            json.dump(runtime_result, f, ensure_ascii=False, indent=2)
        raise

    csv_path = Path(staged["train_csv"])
    json_path = Path(staged["features_json"])
    model_manifest = staged["model_manifest"]
    _assert_manifest_train_only(staged)

    run_config = _build_run_config(
        args=args,
        adapter_model=adapter.model_name,
        dataset_id=dataset_id,
        effective_source=effective_source,
        work_dir=work_dir,
        staged=staged,
        num_rows=num_rows,
    )

    if num_rows <= 0:
        import pandas as pd

        if args.model == "arf" and dataset_id == "c19":
            # 仅 c19+ARF：与 pivot / statistics._train_rows_for_dataset 使用相同的 csv.reader 口径，
            # 避免 staging train 上 pandas 行数与 DatasetNew 合同路径统计不一致导致 Sw。
            from .statistics import _train_rows_for_dataset

            _repo_root = Path(__file__).resolve().parent.parent.parent
            pivot_train_rows = _train_rows_for_dataset("c19", _repo_root)
            if pivot_train_rows >= 0:
                num_rows = pivot_train_rows
                print(
                    f"[Runner] Auto num_rows = {num_rows} "
                    f"(ARF c19: aligned with pivot csv.reader train row count)"
                )
            else:
                num_rows = len(
                    pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
                )
                print(
                    f"[Runner] Auto num_rows = {num_rows} "
                    f"(ARF c19: pivot train row count unavailable, fallback pandas len on staging train)"
                )
        else:
            num_rows = len(
                pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
            )
            print(f"[Runner] Auto num_rows = {num_rows} (same as training data)")

    run_config["resolved"]["num_rows"] = num_rows
    _write_run_config(work_dir, run_config)

    if args.train:
        print(f"[Runner] 训练 {adapter.model_name} on {dataset_id}, work_dir={work_dir}")
        train_started_at = datetime.now().isoformat(timespec="seconds")
        train_t0 = perf_counter()
        runtime_result["timings"]["train"]["started_at"] = train_started_at
        try:
            train_kw = {}
            if args.resume:
                train_kw["resume"] = True
            train_result = adapter.train(
                csv_path=csv_path,
                json_path=json_path,
                work_dir=work_dir,
                epochs=args.epochs,
                model_input_manifest=model_manifest,
                **train_kw,
            )
            runtime_result["train_status"] = "success"
        except Exception as e:
            runtime_result["train_status"] = "fail"
            runtime_result["reason_code"] = "adapter_runtime_error"
            runtime_result["reason_detail"] = str(e)
            runtime_result["timings"]["train"]["ended_at"] = datetime.now().isoformat(timespec="seconds")
            runtime_result["timings"]["train"]["duration_sec"] = round(perf_counter() - train_t0, 3)
            with open(work_dir / "runtime_result.json", "w", encoding="utf-8") as f:
                json.dump(runtime_result, f, ensure_ascii=False, indent=2)
            raise
        runtime_result["timings"]["train"]["ended_at"] = datetime.now().isoformat(timespec="seconds")
        runtime_result["timings"]["train"]["duration_sec"] = round(perf_counter() - train_t0, 3)
        result["train"] = train_result
        result["work_dir"] = work_dir
        model_path = train_result.get("model_path")
        if model_path:
            result["model_path"] = Path(model_path)
            run_config["resolved"]["model_path"] = str(Path(model_path))
            _write_run_config(work_dir, run_config)

    if args.generate:
        if args.train:
            model_path = result.get("model_path")
            if not model_path:
                model_path = result["work_dir"] / "models"
            work_dir = result["work_dir"]
        elif args.model_dir:
            mp = Path(args.model_dir)
            if mp.is_dir():
                mp = _find_model_file_in_run(mp)
                if mp is None:
                    raise ValueError(f"--model-dir 目录下未找到模型文件: {args.model_dir}")
            model_path = mp
        else:
            # generate-only，work_dir 由 _find_latest_run_with_model 得到，需解析出 .pt 路径
            model_path = _find_model_file_in_run(work_dir)
            if model_path is None:
                raise ValueError(f"run 目录下未找到模型文件: {work_dir}")

        gen_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = make_synthetic_csv_name(adapter.model_name, dataset_id, gen_ts, num_rows)
        output_csv = work_dir / csv_name

        run_config["resolved"]["model_path"] = str(Path(model_path))
        run_config["resolved"]["output_csv"] = str(output_csv)
        _write_run_config(work_dir, run_config)

        print(f"[Runner] Generating {num_rows} rows -> {output_csv}")
        gen_started_at = datetime.now().isoformat(timespec="seconds")
        gen_t0 = perf_counter()
        runtime_result["timings"]["generate"]["started_at"] = gen_started_at
        try:
            out_path = adapter.generate(
                model_path=model_path,
                output_csv=output_csv,
                num_rows=num_rows,
                csv_path=csv_path,
                json_path=json_path,
                model_input_manifest=model_manifest,
                dataset_id=dataset_id,
            )
            runtime_result["generate_status"] = "success"
        except Exception as e:
            runtime_result["generate_status"] = "fail"
            runtime_result["reason_code"] = "adapter_runtime_error"
            runtime_result["reason_detail"] = str(e)
            runtime_result["timings"]["generate"]["ended_at"] = datetime.now().isoformat(timespec="seconds")
            runtime_result["timings"]["generate"]["duration_sec"] = round(perf_counter() - gen_t0, 3)
            with open(work_dir / "runtime_result.json", "w", encoding="utf-8") as f:
                json.dump(runtime_result, f, ensure_ascii=False, indent=2)
            raise
        runtime_result["timings"]["generate"]["ended_at"] = datetime.now().isoformat(timespec="seconds")
        runtime_result["timings"]["generate"]["duration_sec"] = round(perf_counter() - gen_t0, 3)
        result["generate"] = {"output_csv": str(out_path)}
        result["synthetic_csv"] = out_path
        runtime_result["artifacts"]["synthetic_csv"] = str(out_path)

    model_path_obj = result.get("model_path")
    if model_path_obj:
        runtime_result["artifacts"]["model_path"] = str(model_path_obj)

    with open(work_dir / "runtime_result.json", "w", encoding="utf-8") as f:
        json.dump(runtime_result, f, ensure_ascii=False, indent=2)
    run_config["resolved"]["model_path"] = str(model_path_obj) if model_path_obj else run_config["resolved"].get("model_path")
    run_config["resolved"]["output_csv"] = str(out_path) if args.generate else run_config["resolved"].get("output_csv")
    _write_run_config(work_dir, run_config)
    result["runtime_result"] = work_dir / "runtime_result.json"

    return result


def _auto_stats(output_base: Path, no_stats: bool = False) -> None:
    """Automatically generate a statistics report after train/generate."""
    if no_stats:
        return
    try:
        from .statistics import generate_report
        generate_report(output_base)
    except ImportError:
        print("[Runner] openpyxl not installed, skipping stats (pip install openpyxl)")
    except Exception as e:
        # Stats failure should not block the main workflow
        print(f"[Runner] Stats report generation failed: {e}")


def main() -> None:
    args = parse_args()
    output_base = args.output_dir or get_output_base()
    result = run(args)
    print("[Runner] 完成:", result)
    _auto_stats(output_base, no_stats=args.no_stats)


if __name__ == "__main__":
    main()
