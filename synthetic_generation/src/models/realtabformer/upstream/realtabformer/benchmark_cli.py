#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark CLI for REaLTabFormer

Unified entry for pipeline: --train-only / --generate-only
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from .realtabformer import REaLTabFormer

# Pipeline / Public Gate 中应作为离散列处理的 data_type（pandas 可能读成 int/float）
_DISCRETE_FEATURE_DTYPES = frozenset(
    {
        "categorical",
        "binary",
        "ordinal",
        "id",
        "id_like",
        "text",
        "timestamp",
        "datetime",
        "datetime_like",
    }
)


def _normalize_feature_dtype(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _iter_feature_dicts(features_blob: Any) -> List[dict]:
    if isinstance(features_blob, list):
        return [x for x in features_blob if isinstance(x, dict)]
    if isinstance(features_blob, dict):
        cols = features_blob.get("columns", [])
        if isinstance(cols, list) and cols and isinstance(cols[0], dict):
            return [x for x in cols if isinstance(x, dict)]
    return []


def _find_latest_model_dir(model_dir: Path) -> Path:
    """Find latest id* subdir in model_dir"""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    candidates = sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("id")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No id* subdir in: {model_dir}. Run train first."
        )
    return candidates[0]


def _apply_features_dtype_correction(df: pd.DataFrame, features_path: Path) -> pd.DataFrame:
    """
    按 Features JSON 将离散语义列从 pandas 数值 dtype 转为 str，
    避免 REaLTabFormer 把本应分词/类别的整数列当成连续 float 训练。
    （staging 会产出 data_type=ID 等；此前仅处理 categorical/binary 不够。）
    """
    with open(features_path, "r", encoding="utf-8") as f:
        features_blob = json.load(f)
    rows = _iter_feature_dicts(features_blob)
    if not rows and isinstance(features_blob, dict):
        # 兜底：顶层为 list 包在 key 下
        for k in ("features", "items"):
            v = features_blob.get(k)
            if isinstance(v, list):
                rows = _iter_feature_dicts(v)
                break
    df = df.copy()
    for feat in rows:
        name = feat.get("feature_name") or feat.get("name")
        if not name or name not in df.columns:
            continue
        dt = _normalize_feature_dtype(feat.get("data_type"))
        if dt in _DISCRETE_FEATURE_DTYPES and pd.api.types.is_numeric_dtype(df[name]):
            df[name] = df[name].astype(str)
    return df


def main():
    parser = argparse.ArgumentParser(description="REaLTabFormer Benchmark CLI")
    parser.add_argument("--csv", required=True, help="Training CSV path")
    parser.add_argument("--features-json", default=None, help="Pipeline Features JSON (optional)")
    parser.add_argument("--model-dir", required=True, help="Model save/load directory")
    parser.add_argument("--output-csv", default=None, help="Output synthetic CSV path")
    parser.add_argument("--train-only", action="store_true", help="Only train")
    parser.add_argument("--generate-only", action="store_true", help="Only generate")
    parser.add_argument("--num-rows", type=int, default=1000, help="Number of rows to generate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--numeric-max-len",
        type=int,
        default=int(os.getenv("REALTABFORMER_NUMERIC_MAX_LEN", "10")),
        help="Max token length used when encoding numeric values",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续训：从当前工作目录下 rtf_checkpoints/ 中最新 HuggingFace checkpoint 继续",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="可选：指定 checkpoint 目录路径（含 trainer_state.json）；不设则自动选最新",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    model_dir = Path(args.model_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")
    df = df.replace([np.inf, -np.inf], np.nan)
    if args.features_json and Path(args.features_json).exists():
        df = _apply_features_dtype_correction(df, Path(args.features_json))

    if args.generate_only:
        load_path = _find_latest_model_dir(model_dir)
        rtf = REaLTabFormer.load_from_dir(path=load_path)
        samples = rtf.sample(n_samples=args.num_rows)
        out_path = Path(args.output_csv) if args.output_csv else (model_dir.parent / "synthetic.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        samples.to_csv(out_path, index=False)
        print(f"[benchmark_cli] Generated {args.num_rows} rows -> {out_path}")
        return

    rtf = REaLTabFormer(
        model_type="tabular",
        epochs=args.epochs,
        batch_size=args.batch_size,
        numeric_max_len=args.numeric_max_len,
        gradient_accumulation_steps=4,
        logging_steps=100,
    )
    resume_kw = False
    if args.resume:
        resume_kw = args.resume_checkpoint if args.resume_checkpoint else True
    rtf.fit(df, n_critic=0, resume_from_checkpoint=resume_kw)
    model_dir.mkdir(parents=True, exist_ok=True)
    rtf.save(model_dir)
    print(f"[benchmark_cli] Training done, model saved -> {model_dir}")

    if not args.train_only:
        samples = rtf.sample(n_samples=args.num_rows)
        out_path = Path(args.output_csv) if args.output_csv else (model_dir.parent / "synthetic.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        samples.to_csv(out_path, index=False)
        print(f"[benchmark_cli] Generated {args.num_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
