"""Model view: the model-facing representation of a prepared dataset, and its inverse.

Applied during staging for datasets prepared by ``core.prepare`` (or when
``BENCHMARK_MODEL_VIEW=1``). Every adapter then sees the same simple table:

- temporal columns (date/datetime/time) become integers (days / seconds since
  1970-01-01, seconds since midnight) and are decoded back to ISO 8601 strings;
- entirely-missing and constant columns are removed and re-inserted afterwards;
- column names are sanitized to ``^[A-Za-z_][A-Za-z0-9_]*$`` and renamed back;
- categorical values that pandas would parse as NaN (``NA``, ``None``, ...) are
  escaped so adapters cannot lose them;
- id/text/multilabel columns are presented as categorical;
- training rows can be subsampled (stratified for classification) when
  ``max_train_rows`` is set. It is off by default.

``restore_generated_csv`` inverts the view on a generated CSV after the model
postprocess has accepted it.
"""

from __future__ import annotations

import copy
import json
import keyword
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SPEC_VERSION = 1
SAFE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ESCAPE_PREFIX = "__tqb__"
# pandas.read_csv default NA strings (pandas 2.x); a real category equal to one of these would be lost.
PANDAS_NA_STRINGS = {
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
}
CATEGORICAL_SEMANTICS = {"categorical", "boolean", "ordinal", "id", "text", "multilabel"}
TEMPORAL_UNITS = {"date": "day", "datetime": "second", "time": "second_of_day"}
TEMPORAL_FORMATS = {"date": "%Y-%m-%d", "datetime": "%Y-%m-%d %H:%M:%S", "time": "%H:%M:%S"}
EPOCH = pd.Timestamp("1970-01-01")


def is_enabled_for(registry: Dict[str, Any]) -> bool:
    import os

    flag = os.environ.get("BENCHMARK_MODEL_VIEW", "").strip().lower()
    if flag in {"0", "false", "no"}:
        return False
    if flag in {"1", "true", "yes"}:
        return True
    contract = registry.get("preprocessing_contract") or {}
    return str(contract.get("prepared_by") or "").startswith("tqb_prepare")


def _sanitize(name: str, taken: set) -> str:
    base = name if SAFE_NAME_RE.match(name) and not keyword.iskeyword(name) else re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not base or not re.match(r"[A-Za-z_]", base[0]):
        base = f"c_{base}"
    if keyword.iskeyword(base):
        base = f"{base}_"
    candidate, i = base, 2
    while candidate.lower() in taken:
        candidate = f"{base}_{i}"
        i += 1
    taken.add(candidate.lower())
    return candidate


def _encode_temporal(series: pd.Series, temporal_type: str) -> pd.Series:
    fmt = TEMPORAL_FORMATS[temporal_type]
    ts = pd.to_datetime(series, format=fmt, errors="coerce")
    bad = series.notna() & ts.isna()
    if bad.any():
        raise ValueError(f"temporal column has values not in {fmt!r}: {series[bad].head(5).tolist()}")
    if temporal_type == "date":
        num = (ts - EPOCH).dt.days
    elif temporal_type == "datetime":
        num = (ts - EPOCH).dt.total_seconds()
    else:
        num = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    return num.round().astype("Int64")


def _decode_temporal(series: pd.Series, temporal_type: str) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce").round()
    if temporal_type == "date":
        ts = EPOCH + pd.to_timedelta(num, unit="D")
    elif temporal_type == "datetime":
        ts = EPOCH + pd.to_timedelta(num, unit="s")
    else:
        ts = EPOCH + pd.to_timedelta(num.clip(lower=0, upper=86399), unit="s")
    out = ts.dt.strftime(TEMPORAL_FORMATS[temporal_type])
    return out.where(num.notna(), "")


def _stratified_sample(df: pd.DataFrame, target: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = df.groupby(target, dropna=False, sort=True).indices
    total = len(df)
    sizes = {k: len(v) for k, v in groups.items()}
    exact = {k: sizes[k] * n / total for k in groups}
    alloc = {k: int(np.floor(exact[k])) for k in groups}
    if n >= len(groups):
        for k in groups:
            alloc[k] = max(alloc[k], 1)
    remainder = n - sum(alloc.values())
    order = sorted(groups, key=lambda k: exact[k] - np.floor(exact[k]), reverse=True)
    i = 0
    while remainder > 0 and order:
        k = order[i % len(order)]
        if alloc[k] < sizes[k]:
            alloc[k] += 1
            remainder -= 1
        i += 1
    while remainder < 0:
        k = max(alloc, key=lambda key: alloc[key])
        alloc[k] -= 1
        remainder += 1
    picked = [rng.choice(groups[k], size=min(alloc[k], sizes[k]), replace=False) for k in groups if alloc[k] > 0]
    return np.sort(df.index.to_numpy()[np.concatenate(picked)])


def subsample_train(
    train_df: pd.DataFrame, target: str, task_type: str, max_rows: Optional[int], seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info = {"train_rows_original": int(len(train_df)), "max_train_rows": max_rows, "sampling": "none", "seed": seed}
    if not max_rows or len(train_df) <= max_rows:
        info["train_rows_used"] = int(len(train_df))
        return train_df, info
    if task_type == "classification":
        idx = _stratified_sample(train_df, target, int(max_rows), seed)
        info["sampling"] = "stratified"
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(train_df.index.to_numpy(), size=int(max_rows), replace=False))
        info["sampling"] = "random"
    out = train_df.loc[idx].reset_index(drop=True)
    info["train_rows_used"] = int(len(out))
    return out, info


def build_model_view(
    splits: Dict[str, pd.DataFrame],
    columns: List[Dict[str, Any]],
    registry: Dict[str, Any],
    target_column: str,
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Return (model-view splits, model-view contract columns, model-view registry, inverse spec)."""
    field_map = {str(f.get("name")): f for f in registry.get("fields") or []}
    original_columns = [c["name"] for c in columns]
    full = pd.concat([splits[k] for k in ("train", "val", "test")], ignore_index=True)

    spec: Dict[str, Any] = {
        "version": SPEC_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_columns": original_columns,
        "renames": {},
        "reinserted_columns": {},
        "temporal": {},
        "escaped_categorical": [],
    }

    taken: set = set()
    new_columns: List[Dict[str, Any]] = []
    new_fields: List[Dict[str, Any]] = []
    out = {k: v.copy() for k, v in splits.items()}
    keep_order: List[str] = []

    for col in columns:
        name = col["name"]
        field = copy.deepcopy(field_map.get(name, {"name": name}))
        series_all = full[name]
        # Decide on the training split: models only ever see train, and a column that is empty or
        # constant there (small tables, or after max_train_rows sampling) cannot be modelled.
        train_col = splits["train"][name]
        train_non_null = train_col.dropna()
        if name != target_column:
            if train_non_null.empty:
                spec["reinserted_columns"][name] = {"kind": "all_missing", "value": None}
                continue
            if not train_col.isna().any() and train_non_null.astype(str).nunique() == 1:
                spec["reinserted_columns"][name] = {"kind": "constant", "value": str(train_non_null.iloc[0])}
                continue

        new_name = _sanitize(name, taken)
        if new_name != name:
            spec["renames"][new_name] = name
        new_col = dict(col)
        new_col["name"] = new_name
        field["name"] = new_name
        semantic = str(col.get("semantic_type") or "").lower()
        temporal_type = field.get("temporal_type")

        if temporal_type in TEMPORAL_FORMATS:
            for k in out:
                out[k][name] = _encode_temporal(out[k][name], temporal_type)
            encoded_all = pd.concat([out[k][name] for k in out], ignore_index=True)
            spec["temporal"][new_name] = {"temporal_type": temporal_type, "unit": TEMPORAL_UNITS[temporal_type]}
            new_col.update({"semantic_type": "integer", "integer_like": True, "numeric_like": True, "domain_values": []})
            policies = {"round_integer", "clip_range", "warn_null_rate"}
            if encoded_all.isna().any():
                policies.add("preserve_missing_sentinel")
            field.update(
                {
                    "semantic_type": "integer",
                    "storage_type": "integer_token",
                    "integer_like": True,
                    "numeric_like": True,
                    "datetime_like": False,
                    "domain_values": [],
                    "numeric_min": int(encoded_all.min()) if encoded_all.notna().any() else None,
                    "numeric_max": int(encoded_all.max()) if encoded_all.notna().any() else None,
                    "postprocess_policy": sorted(policies),
                    "model_view_encoding": {"kind": "temporal", "temporal_type": temporal_type, "unit": TEMPORAL_UNITS[temporal_type]},
                }
            )
            new_col["postprocess_policy"] = field["postprocess_policy"]
        elif semantic in CATEGORICAL_SEMANTICS:
            needs_escape = series_all.dropna().astype(str)
            needs_escape = needs_escape.isin(PANDAS_NA_STRINGS) | needs_escape.str.startswith(ESCAPE_PREFIX)
            if needs_escape.any():
                for k in out:
                    s = out[k][name]
                    mask = s.notna() & (s.astype(str).isin(PANDAS_NA_STRINGS) | s.astype(str).str.startswith(ESCAPE_PREFIX))
                    out[k][name] = s.where(~mask, ESCAPE_PREFIX + s.astype(str))
                field["domain_values"] = [
                    ESCAPE_PREFIX + v if (v in PANDAS_NA_STRINGS or v.startswith(ESCAPE_PREFIX)) else v
                    for v in (str(x) for x in field.get("domain_values") or [])
                ]
                new_col["domain_values"] = field["domain_values"]
                spec["escaped_categorical"].append(new_name)
            if semantic in {"id", "text", "multilabel"}:
                new_col["semantic_type"] = "categorical"
                field["semantic_type"] = "categorical"
                field.setdefault("model_view_encoding", {"kind": "as_categorical", "original_semantic_type": semantic})

        if name != new_name:
            for k in out:
                out[k] = out[k].rename(columns={name: new_name})
        keep_order.append(new_name)
        new_columns.append(new_col)
        new_fields.append(field)

    for k in out:
        out[k] = out[k][keep_order]

    rename_fwd = {v: k for k, v in spec["renames"].items()}
    view_registry = copy.deepcopy(registry)
    view_registry["fields"] = new_fields
    view_registry["field_count"] = len(new_fields)
    for key in ("target_column", "official_target_column", "adapter_primary_target"):
        if view_registry.get(key):
            view_registry[key] = rename_fwd.get(view_registry[key], view_registry[key])
    view_registry["label_columns"] = [rename_fwd.get(c, c) for c in view_registry.get("label_columns") or []]
    policy = view_registry.get("target_policy") or {}
    for key in ("official_target_column", "adapter_primary_target"):
        if policy.get(key):
            policy[key] = rename_fwd.get(policy[key], policy[key])
    policy["label_columns"] = [rename_fwd.get(c, c) for c in policy.get("label_columns") or []]
    view_registry["model_view"] = {"spec_version": SPEC_VERSION, "source_dataset_id": registry.get("dataset_id")}
    spec["target_column"] = rename_fwd.get(target_column, target_column)
    spec["model_columns"] = keep_order
    return out, new_columns, view_registry, spec


def restore_generated_csv(output_csv: Path, spec_path: Path) -> Dict[str, Any]:
    """Invert the model view on a generated CSV in place; returns a summary."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    df = pd.read_csv(output_csv, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig", low_memory=False)
    expected = spec.get("model_columns") or []
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"generated CSV lacks model-view columns: {missing}")
    actions: List[Dict[str, Any]] = []

    for col, info in (spec.get("temporal") or {}).items():
        df[col] = _decode_temporal(df[col].replace("", np.nan), info["temporal_type"])
        actions.append({"action": "decode_temporal", "column": col, "temporal_type": info["temporal_type"]})
    for col in spec.get("escaped_categorical") or []:
        s = df[col]
        mask = s.str.startswith(ESCAPE_PREFIX)
        df[col] = s.where(~mask, s.str.slice(len(ESCAPE_PREFIX)))
        actions.append({"action": "unescape_categorical", "column": col, "cells": int(mask.sum())})
    if spec.get("renames"):
        df = df.rename(columns=spec["renames"])
        actions.append({"action": "restore_column_names", "count": len(spec["renames"])})
    for col, info in (spec.get("reinserted_columns") or {}).items():
        df[col] = "" if info.get("value") is None else info["value"]
        actions.append({"action": "reinsert_column", "column": col, "kind": info.get("kind")})

    original = spec["original_columns"]
    df = df[original]
    df.to_csv(output_csv, index=False, encoding="utf-8", lineterminator="\n")
    summary = {
        "status": "restored",
        "output_csv": str(output_csv),
        "rows": int(len(df)),
        "columns": len(original),
        "actions": actions,
        "restored_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(output_csv).with_name("model_view_restore.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
