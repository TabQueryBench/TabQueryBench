"""
Shared helper for adapters that need the TabDiff / TabbyFlow tabular bundle.

It materializes a compact bundle containing:
- split CSVs in the original schema order
- synthetic evaluation CSVs (`real.csv`, `val.csv`, `test.csv`)
- grouped Num/Cat/Target numpy arrays
- `info.json` with enough metadata for recovery and evaluation
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features_converter import load_features_json


_CAT_DTYPES = {
    "binary",
    "categorical",
    "datetime",
    "datetime_like",
    "id",
    "id_like",
    "ordinal",
    "others",
    "text",
    "timestamp",
}
_INT_DTYPES = {"integer"}
_NAN_SENTINEL = "__nan__"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def tabular_bundle_slug_from_manifest(manifest: Dict[str, Any]) -> str:
    dataset_id = str(manifest.get("dataset_id") or "").strip()
    if dataset_id:
        return f"pipeline_{_slugify(dataset_id)}"

    train_csv = str(manifest.get("train_csv") or "").strip()
    if train_csv:
        stem = Path(train_csv).stem
        stem = re.sub(r"-(train|val|test|main)$", "", stem, flags=re.IGNORECASE)
        if stem:
            return f"pipeline_{_slugify(stem)}"

    return "pipeline_ds"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _normalize_task_type(task_type: Optional[str], y_train: pd.Series) -> str:
    t = str(task_type or "").strip().lower()
    if t == "regression":
        return "regression"
    if t in {"classification", "binclass", "binary"}:
        return "binclass" if y_train.nunique(dropna=True) <= 2 else "multiclass"
    if t == "multiclass":
        return "multiclass"
    return "regression"


def _normalize_cat_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.where(~s.isna(), _NAN_SENTINEL)
    s = s.astype(str)
    return s.replace("nan", _NAN_SENTINEL)


def _categorical_categories(*series_list: Iterable[pd.Series]) -> List[str]:
    values: List[str] = []
    for series in series_list:
        if series is None:
            continue
        normalized = _normalize_cat_series(series)
        values.extend(normalized.tolist())
    if not values:
        return [_NAN_SENTINEL]
    return list(pd.unique(pd.Series(values, dtype="object")))


def _encode_categorical(series: pd.Series, categories: Sequence[str]) -> np.ndarray:
    normalized = _normalize_cat_series(series)
    codes = pd.Categorical(normalized, categories=list(categories)).codes
    if (codes < 0).any():
        missing = normalized[pd.Series(codes) < 0].astype(str).head(5).tolist()
        raise ValueError(f"Unencodable categorical values: {missing}")
    return np.asarray(codes, dtype=np.int64)


def _numeric_array(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_num = pd.to_numeric(train, errors="coerce")
    val_num = pd.to_numeric(val, errors="coerce")
    test_num = pd.to_numeric(test, errors="coerce")
    fill_value = train_num.mean(skipna=True)
    if pd.isna(fill_value):
        fill_value = 0.0
    return (
        np.asarray(train_num.fillna(fill_value), dtype=np.float32),
        np.asarray(val_num.fillna(fill_value), dtype=np.float32),
        np.asarray(test_num.fillna(fill_value), dtype=np.float32),
    )


def _write_split_csvs(
    bundle_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_df.to_csv(bundle_dir / "train.csv", index=False)
    val_df.to_csv(bundle_dir / "val.csv", index=False)
    test_df.to_csv(bundle_dir / "test.csv", index=False)
    train_df.to_csv(bundle_dir / "real.csv", index=False)


def _build_metadata(
    num_col_idx: List[int],
    cat_col_idx: List[int],
    target_col_idx: List[int],
    task_type: str,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"columns": {}}

    for idx in num_col_idx:
        metadata["columns"][idx] = {
            "sdtype": "numerical",
            "computer_representation": "Float",
        }
    for idx in cat_col_idx:
        metadata["columns"][idx] = {"sdtype": "categorical"}
    for idx in target_col_idx:
        if task_type == "regression":
            metadata["columns"][idx] = {
                "sdtype": "numerical",
                "computer_representation": "Float",
            }
        else:
            metadata["columns"][idx] = {"sdtype": "categorical"}
    return metadata


def prepare_tabular_npy_bundle(
    work_dir: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    features_json_path: Path,
    slug: str,
    task_type: Optional[str] = None,
    target_column: Optional[str] = None,
) -> Path:
    work_dir = Path(work_dir)
    bundle_dir = work_dir / "tabular_bundle" / slug
    bundle_dir.mkdir(parents=True, exist_ok=True)

    train_df = _read_csv(Path(train_csv))
    val_df = _read_csv(Path(val_csv))
    test_df = _read_csv(Path(test_csv))

    column_names = list(train_df.columns)
    if list(val_df.columns) != column_names or list(test_df.columns) != column_names:
        raise ValueError("Train/val/test CSV headers must match exactly for bundle build")

    features = load_features_json(Path(features_json_path))
    feature_map = {row.get("feature_name"): row for row in features if row.get("feature_name")}

    inferred_targets = [name for name, row in feature_map.items() if row.get("is_target")]
    resolved_target = target_column or (inferred_targets[0] if len(inferred_targets) == 1 else None)
    if resolved_target is None or resolved_target not in column_names:
        raise ValueError("Could not resolve target column for tabular bundle")

    target_idx = column_names.index(resolved_target)
    target_feature = feature_map.get(resolved_target, {})

    normalized_task = _normalize_task_type(task_type, train_df[resolved_target])
    if task_type is None:
        target_dtype = str(target_feature.get("data_type", "")).strip().lower()
        if target_dtype in _CAT_DTYPES or target_dtype == "binary":
            normalized_task = (
                "binclass"
                if train_df[resolved_target].nunique(dropna=True) <= 2
                else "multiclass"
            )

    num_cols: List[str] = []
    cat_cols: List[str] = []
    num_col_idx: List[int] = []
    cat_col_idx: List[int] = []
    int_columns: List[str] = []
    int_col_idx: List[int] = []
    int_col_idx_wrt_num: List[int] = []

    for idx, name in enumerate(column_names):
        if name == resolved_target:
            continue
        dtype = str(feature_map.get(name, {}).get("data_type", "continuous")).strip().lower()
        if dtype in _CAT_DTYPES:
            cat_cols.append(name)
            cat_col_idx.append(idx)
        else:
            num_cols.append(name)
            num_col_idx.append(idx)
            if dtype in _INT_DTYPES:
                int_columns.append(name)
                int_col_idx.append(idx)
                int_col_idx_wrt_num.append(len(num_cols) - 1)

    idx_mapping: Dict[int, int] = {}
    for grouped_pos, original_idx in enumerate(num_col_idx):
        idx_mapping[int(original_idx)] = grouped_pos
    grouped_offset = len(num_col_idx)
    for grouped_pos, original_idx in enumerate(cat_col_idx, start=grouped_offset):
        idx_mapping[int(original_idx)] = grouped_pos
    idx_mapping[int(target_idx)] = len(num_col_idx) + len(cat_col_idx)
    inverse_idx_mapping = {int(v): int(k) for k, v in idx_mapping.items()}
    idx_name_mapping = {int(i): name for i, name in enumerate(column_names)}

    x_num_train_parts: List[np.ndarray] = []
    x_num_val_parts: List[np.ndarray] = []
    x_num_test_parts: List[np.ndarray] = []
    for col in num_cols:
        train_arr, val_arr, test_arr = _numeric_array(train_df[col], val_df[col], test_df[col])
        x_num_train_parts.append(train_arr.reshape(-1, 1))
        x_num_val_parts.append(val_arr.reshape(-1, 1))
        x_num_test_parts.append(test_arr.reshape(-1, 1))

    if x_num_train_parts:
        x_num_train = np.concatenate(x_num_train_parts, axis=1).astype(np.float32)
        x_num_val = np.concatenate(x_num_val_parts, axis=1).astype(np.float32)
        x_num_test = np.concatenate(x_num_test_parts, axis=1).astype(np.float32)
    else:
        x_num_train = x_num_val = x_num_test = None

    x_cat_train_parts: List[np.ndarray] = []
    x_cat_val_parts: List[np.ndarray] = []
    x_cat_test_parts: List[np.ndarray] = []
    for col in cat_cols:
        categories = _categorical_categories(train_df[col], val_df[col], test_df[col])
        x_cat_train_parts.append(_encode_categorical(train_df[col], categories).reshape(-1, 1))
        x_cat_val_parts.append(_encode_categorical(val_df[col], categories).reshape(-1, 1))
        x_cat_test_parts.append(_encode_categorical(test_df[col], categories).reshape(-1, 1))

    if x_cat_train_parts:
        x_cat_train = np.concatenate(x_cat_train_parts, axis=1).astype(np.int64)
        x_cat_val = np.concatenate(x_cat_val_parts, axis=1).astype(np.int64)
        x_cat_test = np.concatenate(x_cat_test_parts, axis=1).astype(np.int64)
    else:
        x_cat_train = x_cat_val = x_cat_test = None

    if normalized_task == "regression":
        y_train_arr, y_val_arr, y_test_arr = _numeric_array(
            train_df[resolved_target],
            val_df[resolved_target],
            test_df[resolved_target],
        )
        y_train = y_train_arr.astype(np.float32)
        y_val = y_val_arr.astype(np.float32)
        y_test = y_test_arr.astype(np.float32)
        n_classes = None
    else:
        target_categories = _categorical_categories(
            train_df[resolved_target],
            val_df[resolved_target],
            test_df[resolved_target],
        )
        y_train = _encode_categorical(train_df[resolved_target], target_categories)
        y_val = _encode_categorical(val_df[resolved_target], target_categories)
        y_test = _encode_categorical(test_df[resolved_target], target_categories)
        n_classes = len(target_categories)
        if normalized_task == "binclass" and n_classes > 2:
            normalized_task = "multiclass"

    if x_num_train is not None:
        np.save(bundle_dir / "X_num_train.npy", x_num_train)
        np.save(bundle_dir / "X_num_val.npy", x_num_val)
        np.save(bundle_dir / "X_num_test.npy", x_num_test)
    if x_cat_train is not None:
        np.save(bundle_dir / "X_cat_train.npy", x_cat_train)
        np.save(bundle_dir / "X_cat_val.npy", x_cat_val)
        np.save(bundle_dir / "X_cat_test.npy", x_cat_test)
    np.save(bundle_dir / "y_train.npy", y_train)
    np.save(bundle_dir / "y_val.npy", y_val)
    np.save(bundle_dir / "y_test.npy", y_test)

    _write_split_csvs(bundle_dir, train_df, val_df, test_df)
    shutil.copy2(Path(features_json_path), bundle_dir / "staged_features.json")

    info: Dict[str, Any] = {
        "name": slug,
        "task_type": normalized_task,
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": [target_idx],
        "column_names": column_names,
        "idx_mapping": idx_mapping,
        "inverse_idx_mapping": inverse_idx_mapping,
        "idx_name_mapping": idx_name_mapping,
        "int_columns": int_columns,
        "int_col_idx": int_col_idx,
        "int_col_idx_wrt_num": int_col_idx_wrt_num,
        "metadata": _build_metadata(num_col_idx, cat_col_idx, [target_idx], normalized_task),
    }
    if n_classes is not None:
        info["n_classes"] = int(n_classes)

    with open(bundle_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return bundle_dir
