"""
Shared helper for adapters that need the TabDiff / TabbyFlow tabular bundle.

It materializes a compact bundle containing:
- split CSVs in the original schema order
- synthetic evaluation CSVs (`real.csv`, `val.csv`, `test.csv`)
- grouped Num/Cat/Target numpy arrays
- `info.json` with enough metadata for recovery and evaluation
- `category_maps.json`: train-only code -> original label tables used to decode
  generated code columns back into labels (`decode_generated_csv`)

It also renders the in-container "runtime setup" prelude shared by both
adapters (`render_runtime_setup_code`): copy the image's complete upstream tree
into a per-run runtime dir, overlay vendored patched files (md5-guarded against
image drift), stage the bundle and verify imports resolve inside the runtime.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features_converter import load_features_json


_CAT_DTYPES = {
    "binary",
    "bool",
    "boolean",
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
DEFAULT_MAX_EVAL_ROWS = 4096


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


def _canon(value: Any) -> str:
    """Canonical string key of a categorical cell (1 == 1.0 == "1"; NaN -> sentinel)."""
    if value is None:
        return _NAN_SENTINEL
    try:
        if pd.isna(value):
            return _NAN_SENTINEL
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isfinite(v) and v.is_integer():
            return str(int(v))
        return repr(v)
    s = str(value)
    return _NAN_SENTINEL if s == "nan" else s


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _train_category_table(series: pd.Series) -> Tuple[List[str], List[Any], int]:
    """Categories are built from TRAIN only. Returns (keys, original values, mode code)."""
    keys = series.map(_canon)
    uniq = list(pd.unique(keys))
    if not uniq:
        return [_NAN_SENTINEL], [None], 0
    originals: List[Any] = []
    first_idx = keys.reset_index(drop=True).drop_duplicates()
    raw = series.reset_index(drop=True)
    for key, pos in zip(first_idx.tolist(), first_idx.index.tolist()):
        originals.append(None if key == _NAN_SENTINEL else _json_scalar(raw.iloc[pos]))
    mode_key = keys.value_counts().idxmax()
    return uniq, originals, uniq.index(mode_key)


def _encode_with_table(
    series: pd.Series,
    keys: Sequence[str],
    fallback_code: int,
    column: str,
    split: str,
    report: Dict[str, Any],
) -> np.ndarray:
    lookup = {k: i for i, k in enumerate(keys)}
    mapped = series.map(_canon).map(lookup)
    unseen = mapped.isna()
    if unseen.any():
        if split == "train":
            raise ValueError(f"train categorical column {column!r} has unencodable values")
        examples = series[unseen].astype(str).head(5).tolist()
        report.setdefault("unseen_categories", []).append(
            {"column": column, "split": split, "cells": int(unseen.sum()), "examples": examples}
        )
        mapped = mapped.fillna(fallback_code)
    return np.asarray(mapped, dtype=np.int64)


def _numeric_arrays(
    train: pd.Series, others: Sequence[pd.Series]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    train_num = pd.to_numeric(train, errors="coerce")
    fill_value = train_num.mean(skipna=True)
    if pd.isna(fill_value):
        fill_value = 0.0
    out = [
        np.asarray(pd.to_numeric(s, errors="coerce").fillna(fill_value), dtype=np.float32)
        for s in others
    ]
    return np.asarray(train_num.fillna(fill_value), dtype=np.float32), out


def _build_metadata(
    num_col_idx: List[int],
    cat_col_idx: List[int],
    target_col_idx: List[int],
    task_type: str,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"columns": {}}
    for idx in num_col_idx:
        metadata["columns"][idx] = {"sdtype": "numerical", "computer_representation": "Float"}
    for idx in cat_col_idx:
        metadata["columns"][idx] = {"sdtype": "categorical"}
    for idx in target_col_idx:
        if task_type == "regression":
            metadata["columns"][idx] = {"sdtype": "numerical", "computer_representation": "Float"}
        else:
            metadata["columns"][idx] = {"sdtype": "categorical"}
    return metadata


def _eval_split(
    path: Optional[Path],
    train_df: pd.DataFrame,
    max_rows: int,
    seed: int,
    name: str,
    report: Dict[str, Any],
) -> pd.DataFrame:
    """Held-out split used only for the upstream val-loss logging (never for training).

    Missing split -> a train subsample (the upcoming staging contract is train-only).
    Large split -> subsampled to `max_rows` to bound the per-epoch val-loss cost.
    """
    if path is not None and str(path).strip() and Path(path).is_file():
        df = _read_csv(Path(path))
        source = str(path)
    else:
        df = train_df
        source = "train_subsample"
    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    report.setdefault("eval_splits", {})[name] = {"source": source, "rows": int(len(df))}
    return df


def prepare_tabular_npy_bundle(
    work_dir: Path,
    train_csv: Path,
    val_csv: Optional[Path],
    test_csv: Optional[Path],
    features_json_path: Path,
    slug: str,
    task_type: Optional[str] = None,
    target_column: Optional[str] = None,
    max_eval_rows: int = DEFAULT_MAX_EVAL_ROWS,
    seed: int = 0,
) -> Path:
    work_dir = Path(work_dir)
    bundle_dir = work_dir / "tabular_bundle" / slug
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}

    train_df = _read_csv(Path(train_csv))
    column_names = list(train_df.columns)
    val_df = _eval_split(val_csv, train_df, max_eval_rows, seed, "val", report)
    test_df = _eval_split(test_csv, train_df, max_eval_rows, seed + 1, "test", report)
    if list(val_df.columns) != column_names or list(test_df.columns) != column_names:
        raise ValueError("Train/val/test CSV headers must match exactly for bundle build")

    features = load_features_json(Path(features_json_path))
    feature_map = {row.get("feature_name"): row for row in features if row.get("feature_name")}

    inferred_targets = [name for name, row in feature_map.items() if row.get("is_target")]
    resolved_target = target_column or (inferred_targets[0] if len(inferred_targets) == 1 else None)
    if resolved_target is None or resolved_target not in column_names:
        raise ValueError("Could not resolve target column for tabular bundle")
    target_idx = column_names.index(resolved_target)
    target_dtype = str(feature_map.get(resolved_target, {}).get("data_type", "")).strip().lower()

    if task_type is None:
        if target_dtype in _CAT_DTYPES:
            n_uniq = train_df[resolved_target].nunique(dropna=True)
            normalized_task = "binclass" if n_uniq <= 2 else "multiclass"
        else:
            normalized_task = "regression"
    else:
        normalized_task = _normalize_task_type(task_type, train_df[resolved_target])

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
    for grouped_pos, original_idx in enumerate(cat_col_idx, start=len(num_col_idx)):
        idx_mapping[int(original_idx)] = grouped_pos
    idx_mapping[int(target_idx)] = len(num_col_idx) + len(cat_col_idx)
    inverse_idx_mapping = {int(v): int(k) for k, v in idx_mapping.items()}
    idx_name_mapping = {int(i): name for i, name in enumerate(column_names)}

    splits = {"train": train_df, "val": val_df, "test": test_df}
    arrays: Dict[str, Dict[str, List[np.ndarray]]] = {s: {"num": [], "cat": []} for s in splits}
    for col in num_cols:
        tr, (va, te) = _numeric_arrays(train_df[col], [val_df[col], test_df[col]])
        for split, arr in (("train", tr), ("val", va), ("test", te)):
            arrays[split]["num"].append(arr.reshape(-1, 1))

    category_maps: Dict[str, Any] = {}
    for col in cat_cols + ([] if normalized_task == "regression" else [resolved_target]):
        keys, originals, mode_code = _train_category_table(train_df[col])
        category_maps[col] = {"keys": keys, "values": originals}
        encoded = {
            split: _encode_with_table(df[col], keys, mode_code, col, split, report)
            for split, df in splits.items()
        }
        if col == resolved_target:
            y = encoded
        else:
            for split in splits:
                arrays[split]["cat"].append(encoded[split].reshape(-1, 1))

    n_classes = None
    if normalized_task == "regression":
        tr, (va, te) = _numeric_arrays(
            train_df[resolved_target], [val_df[resolved_target], test_df[resolved_target]]
        )
        y = {"train": tr, "val": va, "test": te}
    else:
        n_classes = len(category_maps[resolved_target]["keys"])
        if normalized_task == "binclass" and n_classes > 2:
            normalized_task = "multiclass"

    for split in splits:
        if arrays[split]["num"]:
            np.save(bundle_dir / f"X_num_{split}.npy", np.concatenate(arrays[split]["num"], axis=1).astype(np.float32))
        if arrays[split]["cat"]:
            np.save(bundle_dir / f"X_cat_{split}.npy", np.concatenate(arrays[split]["cat"], axis=1).astype(np.int64))
        y_arr = y[split]
        np.save(bundle_dir / f"y_{split}.npy", y_arr.astype(np.float32 if normalized_task == "regression" else np.int64))

    train_df.to_csv(bundle_dir / "train.csv", index=False)
    train_df.to_csv(bundle_dir / "real.csv", index=False)
    val_df.to_csv(bundle_dir / "val.csv", index=False)
    test_df.to_csv(bundle_dir / "test.csv", index=False)
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
        "adapter_bundle_report": report,
    }
    if n_classes is not None:
        info["n_classes"] = int(n_classes)

    with open(bundle_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    with open(bundle_dir / "category_maps.json", "w", encoding="utf-8") as f:
        json.dump({"nan_sentinel": _NAN_SENTINEL, "columns": category_maps}, f, ensure_ascii=False, indent=2)

    for item in report.get("unseen_categories", []):
        print(
            f"[tabular_bundle] {item['split']} column {item['column']!r}: {item['cells']} cells with "
            f"categories unseen in train mapped to the train mode (examples={item['examples']})"
        )
    print(f"[tabular_bundle] {slug}: task={normalized_task} num={len(num_cols)} cat={len(cat_cols)} "
          f"train_rows={len(train_df)} eval_splits={report.get('eval_splits')}")
    return bundle_dir


def decode_generated_csv(bundle_dir: Path, raw_csv: Path, out_csv: Path) -> Dict[str, Any]:
    """Map generated categorical codes back to train labels and restore column order."""
    bundle_dir = Path(bundle_dir)
    info = json.loads((bundle_dir / "info.json").read_text(encoding="utf-8"))
    maps = json.loads((bundle_dir / "category_maps.json").read_text(encoding="utf-8"))["columns"]
    df = pd.read_csv(raw_csv, low_memory=False)
    column_names = list(info["column_names"])
    missing = [c for c in column_names if c not in df.columns]
    if missing:
        raise ValueError(f"generated CSV lacks columns {missing[:10]} (got {list(df.columns)[:10]})")
    summary: Dict[str, Any] = {"rows": int(len(df)), "clipped_codes": {}}
    for col, table in maps.items():
        values = table["values"]
        codes = pd.to_numeric(df[col], errors="coerce")
        if codes.isna().any():
            bad = df.loc[codes.isna(), col].astype(str).head(5).tolist()
            raise ValueError(f"generated categorical column {col!r} has non-code values: {bad}")
        rounded = codes.round()
        clipped = rounded.clip(0, len(values) - 1).astype(int)
        n_clipped = int((rounded != clipped).sum())
        if n_clipped:
            summary["clipped_codes"][col] = n_clipped
        decoded = [values[i] for i in clipped.tolist()]
        df[col] = pd.Series(decoded, index=df.index, dtype="object").where(
            pd.Series([v is not None for v in decoded], index=df.index), np.nan
        )
    df[column_names].to_csv(out_csv, index=False)
    return summary


# ---------------------------------------------------------------------------
# In-container runtime assembly shared by the TabDiff / TabbyFlow adapters.
# ---------------------------------------------------------------------------

_RUNTIME_SETUP_TEMPLATE = r'''
import hashlib, json, os, shutil, subprocess, sys
_P = json.loads(__PARAMS__)
TAG = _P["tag"]
IMG = _P["image_root"]
RT = _P["runtime_dir"]
OV = _P["overlay_src_dir"]
name = _P["dataname"]

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

_SKIP = set(_P["skip_dirs"])
def _ignore(_, names):
    return [n for n in names if n in _SKIP or n.endswith(".pyc")]

if _P["fresh"] or not os.path.isdir(RT):
    shutil.rmtree(RT, ignore_errors=True)
    shutil.copytree(IMG, RT, ignore=_ignore)
    print(f"[{TAG}] copied image tree {IMG} -> {RT}", flush=True)

applied = {}
for rel, (src_rel, expected_image_md5) in sorted(_P["overlays"].items()):
    img_file = os.path.join(IMG, rel)
    if not os.path.isfile(img_file):
        raise RuntimeError(f"[{TAG}] overlay target missing in image: {img_file}")
    got = _md5(img_file)
    if got != expected_image_md5:
        raise RuntimeError(
            f"[{TAG}] image drift: {img_file} md5={got}, vendored patch was made against "
            f"{expected_image_md5}. Re-vendor {os.path.join(OV, src_rel)} from the new image."
        )
    src_file = os.path.join(OV, src_rel)
    if not os.path.isfile(src_file):
        raise RuntimeError(f"[{TAG}] vendored overlay missing: {src_file}")
    dst_file = os.path.join(RT, rel)
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copyfile(src_file, dst_file)
    applied[rel] = _md5(dst_file)
for pkg in _P["ensure_init"]:
    open(os.path.join(RT, pkg, "__init__.py"), "a").close()

dst_data = os.path.join(RT, "data", name)
shutil.rmtree(dst_data, ignore_errors=True)
os.makedirs(os.path.dirname(dst_data), exist_ok=True)
shutil.copytree(_P["bundle_dir"], dst_data)
dst_syn = os.path.join(RT, "synthetic", name)
os.makedirs(dst_syn, exist_ok=True)
for fn in ("real.csv", "test.csv", "val.csv"):
    shutil.copy(os.path.join(_P["bundle_dir"], fn), os.path.join(dst_syn, fn))

ENV = dict(os.environ)
ENV["PYTHONPATH"] = RT
ENV["PYTHONNOUSERSITE"] = "1"
os.chdir(RT)
_CHECK = (
    "import importlib, os, sys\n"
    "rt = os.path.realpath(sys.argv[1])\n"
    "for m in sys.argv[2:]:\n"
    "    f = os.path.realpath(getattr(importlib.import_module(m), '__file__', None) or '<namespace>')\n"
    "    if not f.startswith(rt + os.sep):\n"
    "        raise SystemExit(f'module {m} resolved outside runtime: {f}')\n"
    "print('[runtime] imports resolved inside', rt, ':', ' '.join(sys.argv[2:]), flush=True)\n"
)
subprocess.check_call([sys.executable, "-c", _CHECK, RT] + list(_P["import_checks"]), env=ENV, cwd=RT)
with open(os.path.join(RT, "_adapter_runtime.json"), "w", encoding="utf-8") as fh:
    json.dump({"image_root": IMG, "overlay_src_dir": OV, "overlays_md5": applied}, fh, indent=2)
print(f"[{TAG}] runtime ready at {RT}; overlays: {sorted(applied)}", flush=True)
'''


def render_runtime_setup_code(
    *,
    tag: str,
    image_root: str,
    runtime_dir: str,
    overlay_src_dir: str,
    overlays: Dict[str, Tuple[str, str]],
    bundle_dir: str,
    dataname: str,
    fresh: bool,
    import_checks: Sequence[str],
    ensure_init: Sequence[str] = (),
    skip_dirs: Sequence[str] = (
        "__pycache__", "data", "synthetic", "ckpt", "result", "results",
        "tests", "images", "debug", "wandb", "impute",
    ),
) -> str:
    """Python source (for the bridge script) that assembles a deterministic runtime.

    After it runs, the bridge has `RT` (runtime root, cwd), `ENV` (subprocess env with
    PYTHONPATH=RT only) and `name` (dataname) in scope.
    """
    params = {
        "tag": tag,
        "image_root": image_root,
        "runtime_dir": runtime_dir,
        "overlay_src_dir": overlay_src_dir,
        "overlays": {k: list(v) for k, v in overlays.items()},
        "bundle_dir": bundle_dir,
        "dataname": dataname,
        "fresh": bool(fresh),
        "import_checks": list(import_checks),
        "ensure_init": list(ensure_init),
        "skip_dirs": list(skip_dirs),
    }
    return _RUNTIME_SETUP_TEMPLATE.replace("__PARAMS__", repr(json.dumps(params)))
