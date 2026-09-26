"""
Host-side missing-value handling for CTGAN / TVAE continuous columns.

ctgan's DataTransformer drops RDT's ``is_null`` output, so NaN in continuous columns is
either lost (CTGAN) or decoded incorrectly (TVAE). Instead we:

- train: median-impute each NaN-bearing continuous column and add a discrete
  ``<col>__isna`` (0/1) column that the model learns jointly with the rest;
- generate: set ``<col>`` to NaN where the sampled indicator is 1, then drop indicators.

The column mapping is stored as ``missing_indicators.json`` in the run work_dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

INDICATOR_FILE = "missing_indicators.json"


def add_missing_indicators(
    df: pd.DataFrame, continuous_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    work = df.copy()
    mapping: Dict[str, str] = {}
    for col in continuous_cols:
        if col not in work.columns:
            continue
        num = pd.to_numeric(work[col], errors="coerce")
        mask = num.isna()
        if not mask.any():
            continue
        med = num.median()
        med = 0.0 if pd.isna(med) else float(med)
        ind = f"{col}__isna"
        while ind in work.columns:
            ind += "_"
        work[col] = num.fillna(med)
        work[ind] = mask.astype(int)
        mapping[col] = ind
    return work, mapping


def write_indicator_map(work_dir: Path, mapping: Dict[str, str]) -> Path:
    path = Path(work_dir) / INDICATOR_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_indicator_map(work_dir: Path) -> Dict[str, str]:
    path = Path(work_dir) / INDICATOR_FILE
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")) or {}


def apply_missing_indicators(output_csv: Path, mapping: Dict[str, str]) -> None:
    if not mapping:
        return
    output_csv = Path(output_csv)
    out = pd.read_csv(output_csv, encoding="utf-8-sig", low_memory=False)
    drop = []
    for col, ind in mapping.items():
        if ind not in out.columns:
            continue
        if col in out.columns:
            flag = pd.to_numeric(out[ind], errors="coerce").fillna(0) >= 0.5
            out.loc[flag, col] = float("nan")
        drop.append(ind)
    if drop:
        out = out.drop(columns=drop)
    out.to_csv(output_csv, index=False, encoding="utf-8")
