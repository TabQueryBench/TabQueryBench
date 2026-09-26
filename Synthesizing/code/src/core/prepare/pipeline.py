"""Turn a dataset package (data file + profile.yaml) into runner-ready inputs.

Output layout under ``<out_root>/<dataset_id>/``::

    <id>-main.csv  <id>-train.csv  <id>-val.csv  <id>-test.csv
    metadata_core/field_registry.json
    metadata_core/dataset_semantics.yaml
    metadata_core/profile.yaml          (copy of the source profile)
    prepare_report.json

Prepared CSVs are UTF-8, comma-separated, missing values written as empty cells,
temporal columns normalized to ISO 8601 (``YYYY-MM-DD``, ``YYYY-MM-DD HH:MM:SS``,
``HH:MM:SS``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .profile import (
    TEMPORAL_OUTPUT_FORMATS,
    TEMPORAL_TYPES,
    ColumnSpec,
    DatasetProfile,
    ProfileError,
    load_profile,
)

PREPARED_BY = "tqb_prepare_v1"
HIGH_CARDINALITY = 1000
LARGE_TABLE_ROWS = 100_000


class PrepareError(RuntimeError):
    """Raised when the data cannot be prepared under the given profile."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _read_raw(profile: DatasetProfile) -> pd.DataFrame:
    if not profile.data_file.is_file():
        raise PrepareError(f"data_file not found: {profile.data_file}")
    encoding = "utf-8-sig" if profile.encoding.lower().replace("_", "-") in {"utf-8", "utf8"} else profile.encoding
    try:
        return pd.read_csv(
            profile.data_file,
            sep=profile.delimiter,
            encoding=encoding,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            low_memory=False,
        )
    except Exception as exc:  # pragma: no cover - surfaced to the user
        raise PrepareError(f"failed to read {profile.data_file}: {exc}") from exc


def _parse_temporal(values: pd.Series, spec: ColumnSpec) -> pd.Series:
    s = values
    if spec.pad:
        s = s.str.zfill(spec.pad)
    fmt = spec.format or ""
    if fmt in {"unix_s", "unix_ms"}:
        num = pd.to_numeric(s, errors="coerce")
        ts = pd.to_datetime(num, unit="s" if fmt == "unix_s" else "ms", errors="coerce", utc=True)
    elif fmt == "iso8601":
        ts = pd.to_datetime(s, format="ISO8601", errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
    return ts.dt.tz_convert(None)


def _clean_column(raw: pd.Series, spec: ColumnSpec, profile: DatasetProfile) -> Tuple[pd.Series, Dict[str, Any]]:
    stripped = raw.str.strip()
    tokens = set(profile.missing_tokens_for(spec))
    missing = stripped.isin(tokens)
    values = stripped.mask(missing)
    info: Dict[str, Any] = {
        "type": spec.type,
        "missing_tokens": sorted(tokens),
        "missing": int(missing.sum()),
        "whitespace_trimmed": int((raw != stripped).sum()),
    }
    on_invalid = profile.on_invalid_for(spec)

    def invalid(mask: pd.Series, what: str) -> None:
        count = int(mask.sum())
        info["invalid"] = count
        if count == 0:
            return
        examples = values[mask].drop_duplicates().head(8).tolist()
        info["invalid_examples"] = examples
        if on_invalid == "error":
            raise PrepareError(
                f"column {spec.name!r} ({spec.type}): {count} values are not valid {what}, examples={examples}. "
                f"Fix the profile (type/format/missing_tokens) or set on_invalid: null for this column."
            )

    if spec.type in {"continuous", "integer"}:
        num = pd.to_numeric(values, errors="coerce")
        bad = values.notna() & (num.isna() | ~np.isfinite(num.fillna(0)))
        if spec.type == "integer":
            bad = bad | (num.notna() & ((num - num.round()).abs() > 1e-9))
        invalid(bad, "integers" if spec.type == "integer" else "finite numbers")
        num = num.mask(bad)
        out = num.round().astype("Int64") if spec.type == "integer" else num.astype("float64")
    elif spec.type in TEMPORAL_TYPES:
        ts = _parse_temporal(values, spec)
        bad = values.notna() & ts.isna()
        invalid(bad, f"{spec.type} values in format {spec.format!r}")
        out = ts.dt.strftime(TEMPORAL_OUTPUT_FORMATS[spec.type]).astype("object")
        out = out.where(ts.notna(), None)
        info["output_format"] = TEMPORAL_OUTPUT_FORMATS[spec.type]
    else:
        out = values.astype("object")
        if spec.type == "boolean":
            distinct = out.dropna().unique().tolist()
            if len(distinct) > 2:
                raise PrepareError(f"column {spec.name!r} (boolean) has more than 2 distinct values: {distinct[:8]}")
        if spec.type == "ordinal" and spec.order:
            bad = out.notna() & ~out.isin(spec.order)
            invalid(bad, f"levels of order {spec.order}")
            out = out.mask(bad)
        info.setdefault("invalid", 0)

    info["missing_after_clean"] = int(out.isna().sum())
    info["unique"] = int(out.nunique(dropna=True))
    return out, info


def _split_indices(df: pd.DataFrame, profile: DatasetProfile) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    target = profile.target_column
    eligible = df.index[df[target].notna()].to_numpy()
    notes: Dict[str, Any] = {"rows_missing_target_excluded": int(len(df) - len(eligible))}
    rng = np.random.default_rng(profile.split.seed)
    ratios = profile.split
    parts: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}

    def allocate(idx: np.ndarray) -> None:
        idx = rng.permutation(idx)
        n = len(idx)
        n_val = int(round(n * ratios.val))
        n_test = int(round(n * ratios.test))
        if n - n_val - n_test < 1:
            n_val, n_test = 0, 0
        parts["val"].append(idx[:n_val])
        parts["test"].append(idx[n_val:n_val + n_test])
        parts["train"].append(idx[n_val + n_test:])

    stratified = profile.task_type == "classification" and profile.split.stratify
    notes["stratified"] = stratified
    if stratified:
        tiny = []
        groups = df.loc[eligible].groupby(target, sort=True).indices
        for label, pos in groups.items():
            idx = eligible[pos]
            if len(idx) < 3:
                parts["train"].append(idx)
                tiny.append(str(label))
            else:
                allocate(idx)
        if tiny:
            notes["classes_with_lt3_rows_train_only"] = tiny[:50]
            notes["classes_with_lt3_rows_count"] = len(tiny)
    else:
        allocate(eligible)

    out = {k: np.sort(np.concatenate(v)) if v else np.array([], dtype=int) for k, v in parts.items()}
    return out, notes


def _field_entry(name: str, spec: ColumnSpec, series: pd.Series, info: Dict[str, Any], profile: DatasetProfile) -> Dict[str, Any]:
    ctype = spec.type
    semantic = "datetime" if ctype in TEMPORAL_TYPES else ctype
    nullable = bool(series.isna().any())
    field: Dict[str, Any] = {
        "name": name,
        "profile_type": ctype,
        "semantic_type": semantic,
        "storage_type": {
            "continuous": "float_token",
            "integer": "integer_token",
        }.get(ctype, "datetime_string_token" if ctype in TEMPORAL_TYPES else "string_token"),
        "nullable": nullable,
        "missing_tokens": [""],
        "missing_tokens_exact": True,
        "missing_sentinel": "",
        "raw_null_rate": round(float(series.isna().mean()), 6) if len(series) else 0.0,
        "integer_like": ctype == "integer",
        "numeric_like": ctype in {"continuous", "integer"},
        "datetime_like": ctype in TEMPORAL_TYPES,
        "unique_count": info["unique"],
        "domain_values": [],
        "numeric_min": None,
        "numeric_max": None,
        "examples": [str(v) for v in series.dropna().drop_duplicates().head(8).tolist()],
    }
    policies = {"strip_whitespace", "warn_null_rate"}
    if ctype in {"continuous", "integer"}:
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().any():
            field["numeric_min"] = _json_number(num.min())
            field["numeric_max"] = _json_number(num.max())
        policies.add("warn_range")
    if ctype == "integer":
        policies.add("round_integer")
    if ctype in {"categorical", "boolean", "ordinal"}:
        values = series.dropna().unique().tolist()
        if ctype == "ordinal" and spec.order:
            field["domain_values"] = [v for v in spec.order if v in set(values)]
            field["ordinal_order"] = list(spec.order)
        else:
            field["domain_values"] = sorted(str(v) for v in values)
        policies.add("restore_domain")
    if ctype == "id":
        policies.add("restore_domain")
    if ctype in TEMPORAL_TYPES:
        field["temporal_type"] = ctype
        field["temporal_output_format"] = TEMPORAL_OUTPUT_FORMATS[ctype]
        field["temporal_source_format"] = spec.format
    if nullable:
        policies.add("preserve_missing_sentinel")
    field["postprocess_policy"] = sorted(policies)
    if spec.description:
        field["description"] = spec.description
    return field


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")


def prepare_dataset(profile_path: Path, out_root: Path) -> Dict[str, Any]:
    """Validate, clean, split and describe one dataset. Returns the prepare report."""
    profile = load_profile(Path(profile_path))
    out_dir = Path(out_root).resolve() / profile.dataset_id
    report: Dict[str, Any] = {
        "status": "fail",
        "prepared_by": PREPARED_BY,
        "started_at_utc": _now(),
        "dataset_id": profile.dataset_id,
        "profile": str(profile.path),
        "data_file": str(profile.data_file),
        "out_dir": str(out_dir),
        "warnings": [],
    }

    raw = _read_raw(profile)
    header = list(raw.columns)
    declared = [c.name for c in profile.columns]
    missing_cols = [c for c in header if c not in declared]
    extra_cols = [c for c in declared if c not in header]
    if missing_cols or extra_cols:
        raise ProfileError(
            f"{profile.path}: columns must list every data column exactly once. "
            f"In data but not in profile: {missing_cols}; in profile but not in data: {extra_cols}"
        )
    report["input"] = {"rows": int(len(raw)), "columns": len(header), "sha256": _sha256(profile.data_file)}

    specs = profile.column_map
    cleaned: Dict[str, pd.Series] = {}
    column_reports: Dict[str, Any] = {}
    dropped = []
    for name in header:
        spec = specs[name]
        if spec.drop:
            dropped.append(name)
            column_reports[name] = {"type": spec.type, "dropped": True}
            continue
        series, info = _clean_column(raw[name], spec, profile)
        cleaned[name] = series
        column_reports[name] = info
    df = pd.DataFrame(cleaned)
    kept = list(df.columns)

    warnings: List[str] = report["warnings"]
    for name in kept:
        s = df[name]
        info = column_reports[name]
        if s.isna().all():
            warnings.append(f"{name}: entirely missing; generated data will reproduce it as all-missing")
        elif s.nunique(dropna=True) == 1 and not s.isna().any():
            warnings.append(f"{name}: constant value; generated data will reproduce the constant")
        if specs[name].type in {"categorical", "id", "text"} and info["unique"] > HIGH_CARDINALITY:
            warnings.append(
                f"{name}: {info['unique']} distinct values; high-cardinality columns are hard for most models "
                "(consider type id/text, dropping, or coarsening upstream)"
            )
    if len(df) > LARGE_TABLE_ROWS:
        warnings.append(
            f"{len(df)} rows: training every model on the full table can take many hours for "
            "realtabformer/tabsyn/arf/tabpfgen; set generation.max_train_rows or --max-train-rows to cap training rows"
        )

    target = profile.target_column
    if profile.task_type == "classification":
        n_classes = int(df[target].nunique(dropna=True))
        if n_classes < 2:
            raise PrepareError(f"target {target!r} has {n_classes} distinct values; classification needs at least 2")
        if n_classes > 256:
            warnings.append(f"target {target!r} has {n_classes} classes; several models cap or degrade above 256 classes")

    splits, split_notes = _split_indices(df, profile)
    report["split"] = {
        "ratios": {"train": profile.split.train, "val": profile.split.val, "test": profile.split.test},
        "seed": profile.split.seed,
        **split_notes,
        "rows": {k: int(len(v)) for k, v in splits.items()},
    }
    if split_notes.get("rows_missing_target_excluded"):
        warnings.append(
            f"{split_notes['rows_missing_target_excluded']} rows have a missing target and are excluded from "
            "train/val/test (kept in main.csv)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "metadata_core"
    meta_dir.mkdir(parents=True, exist_ok=True)
    did = profile.dataset_id
    _write_csv(df, out_dir / f"{did}-main.csv")
    for split_name, idx in splits.items():
        _write_csv(df.loc[idx], out_dir / f"{did}-{split_name}.csv")

    fields = []
    for name in kept:
        entry = _field_entry(name, specs[name], df[name], column_reports[name], profile)
        is_target = name == target
        entry.update(
            {
                "role": "target" if is_target else "feature",
                "adapter_role": "target" if is_target else "feature",
                "use_as_target": is_target,
                "is_label_column": is_target,
            }
        )
        fields.append(entry)

    row_counts = {k: int(len(v)) for k, v in splits.items()}
    profile_sha = _sha256(profile.path)
    registry = {
        "schema_version": "dataset_understanding_field_registry_v2",
        "dataset_id": did,
        "generated_at": _now(),
        "generated_by": "core.prepare",
        "ai_provider": "none",
        "target_column": target,
        "official_target_column": target,
        "adapter_primary_target": target,
        "label_columns": [target],
        "task_type": profile.task_type,
        "target_policy": {
            "kind": "profile_target",
            "official_target_column": target,
            "adapter_primary_target": target,
            "task_type": profile.task_type,
            "label_columns": [target],
            "source_note": "Declared in profile.yaml.",
        },
        "expected_rows": row_counts["train"],
        "row_counts": row_counts,
        "main_rows": int(len(df)),
        "field_count": len(fields),
        "dropped_columns": dropped,
        "generation": {
            "max_train_rows": profile.generation.max_train_rows,
            "num_rows": profile.generation.num_rows,
        },
        "preprocessing_contract": {
            "prepared_by": PREPARED_BY,
            "missing_encoding": "empty_cell",
            "exact_missing_tokens": True,
            "temporal_normalization": dict(TEMPORAL_OUTPUT_FORMATS),
            "profile_sha256": profile_sha,
            "data_sha256": report["input"]["sha256"],
        },
        "fields": fields,
    }
    (meta_dir / "field_registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    semantics = {
        "dataset_id": did,
        "description": profile.description,
        "target_policy_kind": "profile_target",
        "official_target_column": target,
        "label_columns": [target],
        "adapter_primary_target": target,
        "task_type": profile.task_type,
        "row_count": row_counts["train"],
        "notes": ["Generated by core.prepare from profile.yaml."],
    }
    (meta_dir / "dataset_semantics.yaml").write_text(yaml.safe_dump(semantics, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copyfile(profile.path, meta_dir / "profile.yaml")

    report.update(
        {
            "status": "pass",
            "finished_at_utc": _now(),
            "target": {"column": target, "task_type": profile.task_type},
            "output": {
                "main_rows": int(len(df)),
                "columns": len(kept),
                "dropped_columns": dropped,
                "files": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()),
            },
            "columns": column_reports,
        }
    )
    (out_dir / "prepare_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def find_profiles(paths: List[Path], profile_name: str = "profile.yaml") -> List[Path]:
    """Accept profile files, dataset directories, or parent directories of dataset directories."""
    found: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            found.append(p)
        elif (p / profile_name).is_file():
            found.append(p / profile_name)
        elif p.is_dir():
            found.extend(sorted(child / profile_name for child in p.iterdir() if (child / profile_name).is_file()))
    if not found:
        raise PrepareError(f"no {profile_name} found under: {[str(p) for p in paths]}")
    return found


def default_out_root() -> Path:
    from core.runner.config import get_new_tabular_dataset_root

    return get_new_tabular_dataset_root()


def prepare_many(paths: List[Path], out_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    out_root = Path(out_root) if out_root else default_out_root()
    results = []
    for profile_path in find_profiles(paths):
        try:
            results.append(prepare_dataset(profile_path, out_root))
        except (ProfileError, PrepareError) as exc:
            results.append({"status": "fail", "profile": str(profile_path), "error": str(exc)})
    return results
