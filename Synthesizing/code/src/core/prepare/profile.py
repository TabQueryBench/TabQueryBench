"""Dataset profile (``profile.yaml``) schema, validation and draft generation.

A dataset package is a directory containing a data file plus ``profile.yaml``::

    schema_version: tqb_dataset_profile_v1
    dataset_id: 1010_birds_csv
    data_file: 1010_birds_csv.csv
    missing_tokens: ["", "NULL"]
    target: {column: Species, task_type: classification}
    columns:
      - {name: Species, type: categorical}
      - {name: Date, type: date, format: "%Y-%m-%d"}

See ``Synthesizing/code/docs/DATASET_PROFILE.md`` for the full specification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

SCHEMA_VERSION = "tqb_dataset_profile_v1"
PROFILE_FILENAME = "profile.yaml"

NUMERIC_TYPES = {"continuous", "integer"}
TEMPORAL_TYPES = {"date", "datetime", "time"}
CATEGORICAL_TYPES = {"categorical", "ordinal", "boolean", "id", "text"}
COLUMN_TYPES = NUMERIC_TYPES | TEMPORAL_TYPES | CATEGORICAL_TYPES
TASK_TYPES = {"classification", "regression"}
TARGET_TYPES = {
    "classification": {"categorical", "ordinal", "boolean", "integer"},
    "regression": {"continuous", "integer"},
}
SPECIAL_TEMPORAL_FORMATS = {"iso8601", "unix_s", "unix_ms"}
ON_INVALID = {"error", "null"}
DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Output formats for normalized temporal columns.
TEMPORAL_OUTPUT_FORMATS = {
    "date": "%Y-%m-%d",
    "datetime": "%Y-%m-%d %H:%M:%S",
    "time": "%H:%M:%S",
}


class ProfileError(ValueError):
    """Raised when profile.yaml is invalid or inconsistent with the data file."""


@dataclass
class ColumnSpec:
    name: str
    type: str
    format: Optional[str] = None
    pad: Optional[int] = None
    missing_tokens: Optional[List[str]] = None
    order: Optional[List[str]] = None
    on_invalid: Optional[str] = None
    drop: bool = False
    description: Optional[str] = None


@dataclass
class SplitSpec:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    stratify: bool = True


@dataclass
class GenerationSpec:
    max_train_rows: Optional[int] = None
    num_rows: Any = "train"  # "train" (same as the train split) or a positive int


@dataclass
class DatasetProfile:
    path: Path
    dataset_id: str
    data_file: Path
    delimiter: str
    encoding: str
    missing_tokens: List[str]
    on_invalid: str
    target_column: str
    task_type: str
    split: SplitSpec
    generation: GenerationSpec
    columns: List[ColumnSpec]
    description: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def column_map(self) -> Dict[str, ColumnSpec]:
        return {c.name: c for c in self.columns}

    def missing_tokens_for(self, column: ColumnSpec) -> List[str]:
        return list(column.missing_tokens) if column.missing_tokens is not None else list(self.missing_tokens)

    def on_invalid_for(self, column: ColumnSpec) -> str:
        return column.on_invalid or self.on_invalid


def _as_str_list(value: Any, where: str) -> List[str]:
    if not isinstance(value, list):
        raise ProfileError(f"{where} must be a list of strings")
    return ["" if v is None else str(v) for v in value]


def _positive_int_or_none(value: Any, where: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProfileError(f"{where} must be a positive integer or null, got {value!r}")
    return value


def load_profile(path: Path) -> DatasetProfile:
    """Load and validate a profile.yaml (structure only; data checks happen in prepare)."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ProfileError(f"profile not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileError(f"{path}: top level must be a mapping")

    errors: List[str] = []

    def need(key: str) -> Any:
        if key not in raw or raw[key] in (None, ""):
            errors.append(f"missing required key: {key}")
            return None
        return raw[key]

    version = need("schema_version")
    if version is not None and version != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}, got {version!r}")

    dataset_id = str(need("dataset_id") or "")
    if dataset_id and not DATASET_ID_RE.match(dataset_id):
        errors.append(f"dataset_id {dataset_id!r} must match {DATASET_ID_RE.pattern}")

    data_file_raw = need("data_file")
    data_file = (path.parent / str(data_file_raw)).resolve() if data_file_raw else path.parent

    fmt = raw.get("format") or {}
    if not isinstance(fmt, dict):
        errors.append("format must be a mapping")
        fmt = {}
    delimiter = str(fmt.get("delimiter", ","))
    encoding = str(fmt.get("encoding", "utf-8"))

    missing_tokens = [""]
    if "missing_tokens" in raw:
        try:
            missing_tokens = _as_str_list(raw["missing_tokens"], "missing_tokens")
        except ProfileError as exc:
            errors.append(str(exc))

    on_invalid = str(raw.get("on_invalid", "error"))
    if on_invalid not in ON_INVALID:
        errors.append(f"on_invalid must be one of {sorted(ON_INVALID)}")

    target = raw.get("target")
    target_column, task_type = "", ""
    if not isinstance(target, dict):
        errors.append("missing required key: target (mapping with column and task_type)")
    else:
        target_column = str(target.get("column") or "")
        task_type = str(target.get("task_type") or "")
        if not target_column:
            errors.append("target.column is required")
        if task_type not in TASK_TYPES:
            errors.append(f"target.task_type must be one of {sorted(TASK_TYPES)}, got {task_type!r}")

    split_raw = raw.get("split") or {}
    split = SplitSpec()
    if not isinstance(split_raw, dict):
        errors.append("split must be a mapping")
    else:
        for key in ("train", "val", "test"):
            if key in split_raw:
                try:
                    setattr(split, key, float(split_raw[key]))
                except (TypeError, ValueError):
                    errors.append(f"split.{key} must be a number")
        if "seed" in split_raw:
            if not isinstance(split_raw["seed"], int) or isinstance(split_raw["seed"], bool):
                errors.append("split.seed must be an integer")
            else:
                split.seed = split_raw["seed"]
        if "stratify" in split_raw:
            split.stratify = bool(split_raw["stratify"])
        ratios = (split.train, split.val, split.test)
        if any(r < 0 for r in ratios) or split.train <= 0 or abs(sum(ratios) - 1.0) > 1e-6:
            errors.append(f"split ratios must be >= 0, train > 0 and sum to 1, got {ratios}")

    gen_raw = raw.get("generation") or {}
    generation = GenerationSpec()
    if not isinstance(gen_raw, dict):
        errors.append("generation must be a mapping")
    else:
        try:
            generation.max_train_rows = _positive_int_or_none(gen_raw.get("max_train_rows"), "generation.max_train_rows")
        except ProfileError as exc:
            errors.append(str(exc))
        num_rows = gen_raw.get("num_rows", "train")
        if num_rows != "train" and (isinstance(num_rows, bool) or not isinstance(num_rows, int) or num_rows <= 0):
            errors.append(f"generation.num_rows must be 'train' or a positive integer, got {num_rows!r}")
        generation.num_rows = num_rows

    columns: List[ColumnSpec] = []
    cols_raw = raw.get("columns")
    if not isinstance(cols_raw, list) or not cols_raw:
        errors.append("missing required key: columns (non-empty list, one entry per data column)")
        cols_raw = []
    seen = set()
    for idx, item in enumerate(cols_raw):
        where = f"columns[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be a mapping")
            continue
        name = item.get("name")
        if name is None or str(name) == "":
            errors.append(f"{where}.name is required")
            continue
        name = str(name)
        where = f"column {name!r}"
        if name in seen:
            errors.append(f"{where} is listed twice")
        seen.add(name)
        ctype = str(item.get("type") or "")
        if ctype not in COLUMN_TYPES:
            errors.append(f"{where}: type must be one of {sorted(COLUMN_TYPES)}, got {ctype!r}")
        spec = ColumnSpec(name=name, type=ctype, drop=bool(item.get("drop", False)), description=item.get("description"))
        if "format" in item and item["format"] is not None:
            spec.format = str(item["format"])
        if ctype in TEMPORAL_TYPES and not spec.format:
            errors.append(
                f"{where}: type {ctype} requires format (strptime pattern such as '%m/%d/%Y', or one of "
                f"{sorted(SPECIAL_TEMPORAL_FORMATS)})"
            )
        if spec.format and ctype not in TEMPORAL_TYPES:
            errors.append(f"{where}: format is only allowed for {sorted(TEMPORAL_TYPES)} columns")
        if "pad" in item and item["pad"] is not None:
            try:
                spec.pad = _positive_int_or_none(item["pad"], f"{where}.pad")
            except ProfileError as exc:
                errors.append(str(exc))
        if "missing_tokens" in item and item["missing_tokens"] is not None:
            try:
                spec.missing_tokens = _as_str_list(item["missing_tokens"], f"{where}.missing_tokens")
            except ProfileError as exc:
                errors.append(str(exc))
        if "order" in item and item["order"] is not None:
            if ctype != "ordinal":
                errors.append(f"{where}: order is only allowed for ordinal columns")
            try:
                spec.order = _as_str_list(item["order"], f"{where}.order")
            except ProfileError as exc:
                errors.append(str(exc))
        if "on_invalid" in item and item["on_invalid"] is not None:
            if item["on_invalid"] not in ON_INVALID:
                errors.append(f"{where}: on_invalid must be one of {sorted(ON_INVALID)}")
            else:
                spec.on_invalid = str(item["on_invalid"])
        unknown = set(item) - {"name", "type", "format", "pad", "missing_tokens", "order", "on_invalid", "drop", "description"}
        if unknown:
            errors.append(f"{where}: unknown keys {sorted(unknown)}")
        columns.append(spec)

    if target_column and columns:
        tspec = next((c for c in columns if c.name == target_column), None)
        if tspec is None:
            errors.append(f"target.column {target_column!r} is not listed in columns")
        else:
            if tspec.drop:
                errors.append(f"target.column {target_column!r} cannot be dropped")
            allowed = TARGET_TYPES.get(task_type, set())
            if task_type in TASK_TYPES and tspec.type not in allowed:
                errors.append(
                    f"target.column {target_column!r} has type {tspec.type}; {task_type} targets must be one of {sorted(allowed)}"
                )

    unknown_top = set(raw) - {
        "schema_version", "dataset_id", "description", "data_file", "format", "missing_tokens",
        "on_invalid", "target", "split", "generation", "columns", "source", "notes",
    }
    if unknown_top:
        errors.append(f"unknown top-level keys {sorted(unknown_top)}")

    if errors:
        raise ProfileError(f"{path}: invalid profile:\n  - " + "\n  - ".join(errors))

    return DatasetProfile(
        path=path,
        dataset_id=dataset_id,
        data_file=data_file,
        delimiter=delimiter,
        encoding=encoding,
        missing_tokens=missing_tokens,
        on_invalid=on_invalid,
        target_column=target_column,
        task_type=task_type,
        split=split,
        generation=generation,
        columns=columns,
        description=raw.get("description"),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Draft generation: infer a starting profile from a raw CSV for a human to review.
# ---------------------------------------------------------------------------

_DRAFT_TEMPORAL_FORMATS = [
    ("date", "%Y-%m-%d"),
    ("date", "%m/%d/%Y"),
    ("date", "%d/%m/%Y"),
    ("datetime", "%Y-%m-%d %H:%M:%S"),
    ("datetime", "%m/%d/%Y %H:%M"),
    ("datetime", "%m/%d/%Y %H:%M:%S"),
    ("datetime", "%m/%d/%Y %I:%M:%S %p"),
    ("datetime", "iso8601"),
    ("time", "%H:%M"),
    ("time", "%H:%M:%S"),
]
_DRAFT_MISSING = ["", "NULL", "null", "NA", "N/A", "nan", "NaN", "None"]


def _draft_temporal(values: pd.Series) -> Optional[tuple]:
    sample = values.head(2000)
    for ctype, fmt in _DRAFT_TEMPORAL_FORMATS:
        if fmt == "iso8601":
            if not sample.str.match(r"^\d{4}-\d{2}-\d{2}T").all():
                continue
            parsed = pd.to_datetime(sample, format="ISO8601", errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if parsed.notna().mean() >= 0.99:
            return ctype, fmt
    return None


def draft_profile(data_file: Path, dataset_id: str, target: Optional[str] = None, delimiter: str = ",") -> str:
    """Infer a draft profile.yaml (as text). Every inferred choice must be reviewed by a human."""
    data_file = Path(data_file)
    df = pd.read_csv(data_file, sep=delimiter, dtype=str, keep_default_na=False, encoding="utf-8-sig", low_memory=False)
    present_missing = sorted({tok for tok in _DRAFT_MISSING if (df == tok).to_numpy().any()}, key=_DRAFT_MISSING.index)
    missing = present_missing or [""]
    columns = []
    for name in df.columns:
        s = df[name].str.strip()
        s = s[~s.isin(missing)]
        entry: Dict[str, Any] = {"name": name}
        if s.empty:
            entry["type"] = "continuous"
            entry["description"] = "TODO: column is entirely missing"
        else:
            num = pd.to_numeric(s, errors="coerce")
            uniq = s.nunique()
            if num.notna().all():
                is_int = ((num - num.round()).abs() < 1e-9).all()
                entry["type"] = "integer" if is_int else "continuous"
                if is_int and uniq <= 20 and len(s) > 100:
                    entry["description"] = "TODO: few distinct integer values; categorical/ordinal code?"
            else:
                temporal = _draft_temporal(s)
                if temporal:
                    entry["type"], entry["format"] = temporal
                elif uniq <= 2:
                    entry["type"] = "boolean" if uniq == 2 else "categorical"
                elif uniq / max(1, len(s)) > 0.9 and uniq > 1000:
                    entry["type"] = "id"
                    entry["description"] = "TODO: near-unique strings; id or free text?"
                else:
                    entry["type"] = "categorical"
        columns.append(entry)

    target_col = target or ""
    task = "classification"
    if target_col:
        t = next((c for c in columns if c["name"] == target_col), None)
        if t and t["type"] == "continuous":
            task = "regression"
    doc = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "description": "TODO: one-line description",
        "data_file": data_file.name,
        "missing_tokens": missing,
        "target": {"column": target_col or "TODO", "task_type": task},
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42, "stratify": True},
        "generation": {"max_train_rows": None, "num_rows": "train"},
        "columns": columns,
    }
    header = (
        "# DRAFT generated by `python -m core.prepare draft`. Review every column type,\n"
        "# temporal format, missing token and the target before running prepare.\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120)
