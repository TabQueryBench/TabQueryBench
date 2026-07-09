#!/usr/bin/env python3
"""Plot paper-quality quality/time trade-off figures for synthetic generators.

This script discovers a per-asset score summary, joins it with runtime metadata
from synthetic-output manifests/logs, aggregates to model x dataset-type, and
renders a two-panel figure:

- left: overall score vs generation time per 1K generated rows
- right: overall score vs training time per 1K training rows

The script is conservative about runtime extraction. It only uses values that
can be read directly from known fields or inferred from explicit/dedicated logs.
When the needed fields are unavailable, it records an audit row instead of
fabricating times.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SCORE_KEYS = [
    "overall_score",
    "score",
    "aggregate_score",
    "final_score",
    "benchmark_score",
]
TRAIN_TIME_KEYS = [
    "train_time_seconds",
    "training_time_seconds",
    "fit_time_seconds",
    "elapsed_train_seconds",
    "train_runtime_s",
]
GEN_TIME_KEYS = [
    "generation_time_seconds",
    "generate_time_seconds",
    "inference_time_seconds",
    "sampling_time_seconds",
    "sample_time_seconds",
]
TRAIN_ROW_KEYS = [
    "train_rows",
    "n_train_rows",
    "num_rows",
    "n_rows",
]
GEN_ROW_KEYS = [
    "generated_rows",
    "n_generated_rows",
    "num_rows",
    "n_rows",
]

MODEL_ALIASES = {
    "rtf": "realtabformer",
    "forest": "forestdiffusion",
}

KNOWN_MODEL_ORDER = [
    "arf",
    "bayesnet",
    "cdtd",
    "codi",
    "ctgan",
    "forestdiffusion",
    "goggle",
    "realtabformer",
    "tabbyflow",
    "tabddpm",
    "tabdiff",
    "tabpfgen",
    "tabsyn",
    "tvae",
]

DEFAULT_EXCLUDED_MODELS = ["cdtd", "codi", "goggle"]

MODEL_LABELS = {
    "arf": "ARF",
    "bayesnet": "BayesNet",
    "ctgan": "CTGAN",
    "forestdiffusion": "ForestDiffusion",
    "realtabformer": "RealTabFormer",
    "tabbyflow": "TabbyFlow",
    "tabddpm": "TabDDPM",
    "tabdiff": "TabDiff",
    "tabpfgen": "TabPFGen",
    "tabsyn": "TabSyn",
    "tvae": "TVAE",
}

MODEL_COLORS = {
    "realtabformer": "#332288",
    "tvae": "#4477AA",
    "forestdiffusion": "#228833",
    "tabddpm": "#EE7733",
    "tabsyn": "#66CCEE",
    "tabdiff": "#AA3377",
    "ctgan": "#EE6677",
    "arf": "#777777",
    "bayesnet": "#CCBB44",
    "tabpfgen": "#009988",
    "tabbyflow": "#882255",
}

PAPER_MODEL_SET = set(MODEL_COLORS)

DATASET_TYPE_MARKERS = {
    "categorical": "o",
    "numerical": "s",
    "mixed": "x",
}

DATASET_TYPE_LABELS = {
    "categorical": "Categorical",
    "numerical": "Numerical",
    "mixed": "Mixed",
}

TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")
TRAIN_TIME_RE = re.compile(r"(?:totoal|total)\s+training\s+time\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
SAMPLE_TIME_RE = re.compile(r"(?:totoal|total)\s+sampling\s+time\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
TRAIN_RUNTIME_RE = re.compile(r"train_runtime['\"]?\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
TIME_LINE_RE = re.compile(r"^\s*Time:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE | re.MULTILINE)
ELAPSED_HMS_RE = re.compile(r"Elapsed\s+time:\s*(\d+):(\d{2}):(\d{2})", re.IGNORECASE)
ISO_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
SLASH_TS_RE = re.compile(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})")
GLOG_TS_RE = re.compile(r"\b[IWEF](\d{2})(\d{2})\s+(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?")
GENERATED_ROWS_RE = re.compile(r"Generated\s+\d+\s+rows\s*->", re.IGNORECASE)
FILENAME_ROWS_RE = re.compile(r"-(\d+)-\d{8}_\d{6}\.csv$", re.IGNORECASE)
LOG_EXCERPT_BYTES = 196608


@dataclass
class ScoreSource:
    path: Path
    score_key: str
    format_name: str
    header: list[str]
    rank_score: tuple[int, int, int, float]


@dataclass
class RuntimeValue:
    value: float | None
    source: str


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na", "<null>"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    out = _as_float(value)
    if out is None:
        return None
    try:
        return int(round(out))
    except Exception:
        return None


def _parse_timestamp_from_text(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:
        pass
    match = TIMESTAMP_RE.search(text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except Exception:
        return None


def _extract_filename_timestamp(path: Path) -> datetime | None:
    candidates = TIMESTAMP_RE.findall(path.name)
    if not candidates:
        candidates = TIMESTAMP_RE.findall(path.stem)
    if not candidates:
        return None
    try:
        return datetime.strptime(candidates[-1], "%Y%m%d_%H%M%S")
    except Exception:
        return None


def _path_mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path.resolve())


def _normalize_model_name(value: str | None) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    return MODEL_ALIASES.get(text, text)


def _dataset_type_for_id(dataset_id: str) -> str:
    prefix = (dataset_id or "").strip().lower()[:1]
    if prefix == "c":
        return "categorical"
    if prefix == "n":
        return "numerical"
    if prefix == "m":
        return "mixed"
    return "unknown"


def _display_model_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def _latex_escape(text: str) -> str:
    out = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out


def _model_color(model_name: str) -> Any:
    return MODEL_COLORS[model_name]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _iter_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _coerce_record_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                return list(value)
    return []


def _pick_score_key(header: list[str], requested: str) -> str | None:
    if requested != "auto":
        return requested if requested in header else None
    for key in SCORE_KEYS:
        if key in header:
            return key
    return None


def _candidate_rank(path: Path, header: list[str], row_count: int, score_key: str) -> tuple[int, int, int, float]:
    name = path.name.lower()
    score = 0
    if score_key == "overall_score":
        score += 12
    if "analysis_asset_scores" in name:
        score += 20
    if "all_datasets" in name:
        score += 10
    if "summary" in name:
        score += 4
    if "asset" in name:
        score += 4
    if {"synthetic_csv_path", "run_id", "log_paths"} & set(header):
        score += 10
    if "model_id" in header:
        score += 5
    if row_count >= 200:
        score += 8
    elif row_count >= 50:
        score += 4
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    return (score, row_count, len(header), mtime)


def _scan_candidate_score_files(search_roots: list[Path], score_key: str) -> list[ScoreSource]:
    candidates: list[ScoreSource] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            suffix = path.suffix.lower()
            if suffix not in {".csv", ".json", ".jsonl"}:
                continue
            header: list[str] = []
            row_count = 0
            fmt = ""
            try:
                if suffix == ".csv":
                    with path.open("r", encoding="utf-8", newline="") as f:
                        reader = csv.DictReader(f)
                        header = list(reader.fieldnames or [])
                        for row_count, _ in zip(range(2500), reader, strict=False):
                            pass
                        row_count += 1 if header else 0
                    fmt = "csv"
                elif suffix == ".jsonl":
                    rows = _iter_jsonl(path)
                    header = sorted({k for row in rows[:200] for k in row.keys()})
                    row_count = len(rows)
                    fmt = "jsonl"
                else:
                    payload = _read_json(path)
                    rows = _coerce_record_list(payload)
                    if not rows:
                        continue
                    header = sorted({k for row in rows[:200] for k in row.keys()})
                    row_count = len(rows)
                    fmt = "json"
            except Exception:
                continue
            if not header:
                continue
            if not ({"dataset_id", "dataset"} & set(header)):
                continue
            if not ({"model_id", "model_name", "model"} & set(header)):
                continue
            picked = _pick_score_key(header, score_key)
            if not picked:
                continue
            candidates.append(
                ScoreSource(
                    path=path,
                    score_key=picked,
                    format_name=fmt,
                    header=header,
                    rank_score=_candidate_rank(path, header, row_count, picked),
                )
            )
    candidates.sort(key=lambda item: item.rank_score, reverse=True)
    return candidates


def _load_score_rows(source: ScoreSource) -> list[dict[str, Any]]:
    if source.format_name == "csv":
        return _iter_csv(source.path)
    if source.format_name == "jsonl":
        return _iter_jsonl(source.path)
    payload = _read_json(source.path)
    return _coerce_record_list(payload)


def _parse_path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            items = parsed if isinstance(parsed, list) else [text]
        except Exception:
            items = [part.strip() for part in text.split(";") if part.strip()]
    paths: list[Path] = []
    for item in items:
        p = Path(str(item)).expanduser()
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        paths.append(p)
    return paths


def _find_first_number_in_filename(path: Path) -> int | None:
    match = FILENAME_ROWS_RE.search(path.name)
    if match:
        return _as_int(match.group(1))
    return None


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return None


def _walk_for_numeric_keys(payload: Any, candidate_keys: list[str]) -> tuple[float | None, str]:
    candidate_set = {key.lower() for key in candidate_keys}
    stack: list[tuple[Any, str]] = [(payload, "root")]
    while stack:
        node, trace = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                new_trace = f"{trace}.{key}"
                if key.lower() in candidate_set:
                    maybe = _as_float(value)
                    if maybe is not None:
                        return maybe, new_trace
                stack.append((value, new_trace))
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                stack.append((value, f"{trace}[{idx}]"))
    return None, ""


def _resolve_train_csv_path(dataset_id: str) -> Path | None:
    candidates = [
        PROJECT_ROOT / "data" / dataset_id / f"{dataset_id}-train.csv",
        PROJECT_ROOT / "data" / dataset_id / "raw" / f"{dataset_id}-train.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _extract_runtime_from_metadata(metadata_paths: list[Path], candidate_keys: list[str]) -> RuntimeValue:
    for path in metadata_paths:
        if path.suffix.lower() != ".json" or not path.exists():
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        value, trace = _walk_for_numeric_keys(payload, candidate_keys)
        if value is not None:
            return RuntimeValue(value=value, source=f"metadata_json:{_safe_rel(path)}:{trace}")
    return RuntimeValue(value=None, source="missing")


def _read_log_excerpt(path: Path) -> str:
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    try:
        with path.open("rb") as f:
            if size <= LOG_EXCERPT_BYTES * 2:
                data = f.read()
            else:
                head = f.read(LOG_EXCERPT_BYTES)
                f.seek(max(0, size - LOG_EXCERPT_BYTES))
                tail = f.read(LOG_EXCERPT_BYTES)
                data = head + b"\n...\n" + tail
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_log_line_timestamps(text: str, *, reference_filename_ts: datetime | None) -> list[datetime]:
    out: list[datetime] = []
    for match in ISO_TS_RE.findall(text):
        try:
            out.append(datetime.fromisoformat(match))
        except Exception:
            continue
    for match in SLASH_TS_RE.findall(text):
        try:
            out.append(datetime.strptime(match, "%m/%d/%Y %H:%M:%S"))
        except Exception:
            continue
    if reference_filename_ts is not None:
        year = reference_filename_ts.year
        for month, day, hour, minute, second in GLOG_TS_RE.findall(text):
            try:
                out.append(
                    datetime(
                        year=year,
                        month=int(month),
                        day=int(day),
                        hour=int(hour),
                        minute=int(minute),
                        second=int(second),
                    )
                )
            except Exception:
                continue
    out.sort()
    return out


def _extract_log_content_duration(path: Path) -> RuntimeValue:
    text = _read_log_excerpt(path)
    if not text:
        return RuntimeValue(value=None, source="missing")
    timestamps = _extract_log_line_timestamps(text, reference_filename_ts=_extract_filename_timestamp(path))
    if len(timestamps) < 2:
        return RuntimeValue(value=None, source="missing")
    delta = (timestamps[-1] - timestamps[0]).total_seconds()
    if delta <= 0 or delta > 24 * 3600:
        return RuntimeValue(value=None, source="missing")
    return RuntimeValue(value=float(delta), source=f"log_content_timestamps:{_safe_rel(path)}")


def _filename_delta_seconds(start_path: Path, end_ts: datetime) -> float | None:
    start_ts = _extract_filename_timestamp(start_path)
    if start_ts is None:
        return None
    delta = (end_ts - start_ts).total_seconds()
    if delta <= 0 or delta > 24 * 3600:
        return None
    return float(delta)


def _matching_generation_logs(log_paths: list[Path], synthetic_csv_path: Path) -> list[Path]:
    generate_logs = [path for path in log_paths if path.exists() and _classify_log(path) == "generate"]
    if not generate_logs:
        return []
    synthetic_name = synthetic_csv_path.name if synthetic_csv_path else ""
    synthetic_ts = _extract_filename_timestamp(synthetic_csv_path) if synthetic_csv_path else None
    strong: list[Path] = []
    weak: list[Path] = []
    for path in generate_logs:
        excerpt = _read_log_excerpt(path)
        if synthetic_name and synthetic_name in excerpt:
            strong.append(path)
            continue
        log_ts = _extract_filename_timestamp(path)
        if synthetic_ts is not None and log_ts is not None and log_ts == synthetic_ts:
            strong.append(path)
            continue
        weak.append(path)
    if strong:
        return sorted(strong, key=lambda p: (_extract_filename_timestamp(p) or datetime.min), reverse=True)
    return sorted(weak, key=lambda p: (_extract_filename_timestamp(p) or datetime.min), reverse=True)


def _adjacent_phase_delta(train_log: Path, generation_logs: list[Path]) -> RuntimeValue:
    start_ts = _extract_filename_timestamp(train_log)
    if start_ts is None:
        return RuntimeValue(value=None, source="missing")
    candidates = []
    for path in generation_logs:
        log_ts = _extract_filename_timestamp(path)
        if log_ts is None:
            continue
        delta = (log_ts - start_ts).total_seconds()
        if 0 < delta <= 24 * 3600:
            candidates.append((delta, path))
    if not candidates:
        return RuntimeValue(value=None, source="missing")
    candidates.sort(key=lambda item: item[0])
    delta, path = candidates[0]
    return RuntimeValue(value=float(delta), source=f"log_filename_to_next_phase:{_safe_rel(train_log)}->{_safe_rel(path)}")


def _classify_log(log_path: Path) -> str:
    name = log_path.name.lower()
    if "train" in name:
        return "train"
    if any(token in name for token in ["gen", "generate", "sample", "infer"]):
        return "generate"
    return "other"


def _log_duration_by_window(log_path: Path) -> float | None:
    start = _parse_timestamp_from_text(log_path.name) or _parse_timestamp_from_text(log_path.stem)
    end = _path_mtime(log_path)
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds()
    if delta <= 0 or delta > 12 * 3600:
        return None
    return float(delta)


def _extract_train_time_from_logs(log_paths: list[Path], synthetic_csv_path: Path) -> RuntimeValue:
    train_logs = [path for path in log_paths if path.exists() and _classify_log(path) == "train"]
    generate_logs = _matching_generation_logs(log_paths, synthetic_csv_path)
    explicit_runtime: list[tuple[float, str]] = []
    explicit_train_total: list[tuple[float, str]] = []
    explicit_time_lines: list[tuple[float, str]] = []
    for path in train_logs:
        text = _read_log_excerpt(path)
        if not text:
            continue
        for match in TRAIN_RUNTIME_RE.findall(text):
            value = _as_float(match)
            if value is not None:
                explicit_runtime.append((value, f"log_explicit_train_runtime:{_safe_rel(path)}"))
        for match in TRAIN_TIME_RE.findall(text):
            value = _as_float(match)
            if value is not None:
                explicit_train_total.append((value, f"log_explicit_train_total:{_safe_rel(path)}"))
        for match in TIME_LINE_RE.findall(text):
            value = _as_float(match)
            if value is not None:
                explicit_time_lines.append((value, f"log_explicit_time_line_train:{_safe_rel(path)}"))
    if explicit_runtime:
        explicit_runtime.sort(key=lambda item: item[0], reverse=True)
        return RuntimeValue(value=explicit_runtime[0][0], source=explicit_runtime[0][1])
    if explicit_train_total:
        explicit_train_total.sort(key=lambda item: item[0], reverse=True)
        return RuntimeValue(value=explicit_train_total[0][0], source=explicit_train_total[0][1])
    if explicit_time_lines:
        explicit_time_lines.sort(key=lambda item: item[0], reverse=True)
        return RuntimeValue(value=explicit_time_lines[0][0], source=explicit_time_lines[0][1])

    content_durations = []
    for path in train_logs:
        rv = _extract_log_content_duration(path)
        if rv.value is not None:
            content_durations.append((rv.value, rv.source))
    if content_durations:
        content_durations.sort(key=lambda item: item[0], reverse=True)
        return RuntimeValue(value=content_durations[0][0], source=content_durations[0][1])

    if generate_logs:
        by_phase = []
        for train_log in train_logs:
            rv = _adjacent_phase_delta(train_log, generate_logs)
            if rv.value is not None:
                by_phase.append((rv.value, rv.source))
        if by_phase:
            by_phase.sort(key=lambda item: item[0])
            return RuntimeValue(value=by_phase[0][0], source=by_phase[0][1])

    # Conservative upper bound when only a train log and the selected synthetic
    # file timestamp exist in the same naming convention.
    synthetic_ts = _extract_filename_timestamp(synthetic_csv_path) if synthetic_csv_path else None
    if synthetic_ts is not None:
        upper_bounds = []
        for train_log in train_logs:
            delta = _filename_delta_seconds(train_log, synthetic_ts)
            if delta is not None:
                upper_bounds.append((delta, f"log_filename_to_selected_csv_upper_bound:{_safe_rel(train_log)}"))
        if upper_bounds:
            upper_bounds.sort(key=lambda item: item[0])
            return RuntimeValue(value=upper_bounds[0][0], source=upper_bounds[0][1])
    return RuntimeValue(value=None, source="missing")


def _extract_generation_time_from_logs(log_paths: list[Path], synthetic_csv_path: Path) -> RuntimeValue:
    generate_logs = _matching_generation_logs(log_paths, synthetic_csv_path)
    train_logs = [path for path in log_paths if path.exists() and _classify_log(path) == "train"]

    explicit_samples: list[tuple[float, str]] = []
    explicit_time_lines: list[tuple[float, str]] = []
    explicit_elapsed_hms: list[tuple[float, str]] = []
    for path in generate_logs + train_logs:
        text = _read_log_excerpt(path)
        if not text:
            continue
        for match in SAMPLE_TIME_RE.findall(text):
            value = _as_float(match)
            if value is not None:
                explicit_samples.append((value, f"log_explicit_sampling:{_safe_rel(path)}"))
        if _classify_log(path) == "generate":
            for hours, minutes, seconds in ELAPSED_HMS_RE.findall(text):
                total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                explicit_elapsed_hms.append((float(total), f"log_explicit_elapsed_hms:{_safe_rel(path)}"))
            for match in TIME_LINE_RE.findall(text):
                value = _as_float(match)
                if value is not None:
                    explicit_time_lines.append((value, f"log_explicit_time_line:{_safe_rel(path)}"))
    if explicit_samples:
        explicit_samples.sort(key=lambda item: item[0], reverse=True)
        return RuntimeValue(value=explicit_samples[0][0], source=explicit_samples[0][1])
    if explicit_elapsed_hms:
        explicit_elapsed_hms.sort(key=lambda item: item[0], reverse=True)
        return RuntimeValue(value=explicit_elapsed_hms[0][0], source=explicit_elapsed_hms[0][1])
    if explicit_time_lines:
        explicit_time_lines.sort(key=lambda item: item[0], reverse=True)
        return RuntimeValue(value=explicit_time_lines[0][0], source=explicit_time_lines[0][1])

    content_durations = []
    for path in generate_logs:
        rv = _extract_log_content_duration(path)
        if rv.value is not None:
            content_durations.append((rv.value, rv.source))
    if content_durations:
        content_durations.sort(key=lambda item: item[0], reverse=True)
        return RuntimeValue(value=content_durations[0][0], source=content_durations[0][1])
    return RuntimeValue(value=None, source="missing")


def _prefer_runtime(*candidates: RuntimeValue) -> RuntimeValue:
    for candidate in candidates:
        if candidate.value is not None and candidate.value >= 0:
            return candidate
    return RuntimeValue(value=None, source="missing")


def _pick_numeric_from_row(row: dict[str, Any], keys: list[str]) -> RuntimeValue:
    for key in keys:
        if key in row:
            value = _as_float(row.get(key))
            if value is not None:
                return RuntimeValue(value=value, source=f"row:{key}")
    return RuntimeValue(value=None, source="missing")


def _find_generated_rows(row: dict[str, Any], synthetic_csv_path: Path, metadata_paths: list[Path]) -> RuntimeValue:
    direct = _pick_numeric_from_row(row, GEN_ROW_KEYS)
    if direct.value is not None:
        return direct
    meta = _extract_runtime_from_metadata(metadata_paths, GEN_ROW_KEYS)
    if meta.value is not None:
        return meta
    from_name = _find_first_number_in_filename(synthetic_csv_path)
    if from_name is not None:
        return RuntimeValue(value=float(from_name), source="synthetic_filename")
    return RuntimeValue(value=None, source="missing")


def _find_train_rows(row: dict[str, Any], dataset_id: str, metadata_paths: list[Path], cache: dict[str, int | None]) -> RuntimeValue:
    direct = _pick_numeric_from_row(row, TRAIN_ROW_KEYS)
    if direct.value is not None:
        return direct
    meta = _extract_runtime_from_metadata(metadata_paths, TRAIN_ROW_KEYS)
    if meta.value is not None:
        return meta
    if dataset_id not in cache:
        train_csv = _resolve_train_csv_path(dataset_id)
        cache[dataset_id] = _count_csv_rows(train_csv) if train_csv is not None else None
    if cache.get(dataset_id) is not None:
        return RuntimeValue(value=float(cache[dataset_id]), source=f"local_train_csv:{dataset_id}")
    return RuntimeValue(value=None, source="missing")


def _choose_latest_score_source(evaluation_root: Path, score_key: str) -> tuple[ScoreSource, list[ScoreSource]]:
    search_roots = [
        evaluation_root,
        PROJECT_ROOT / "logs" / "runs",
        PROJECT_ROOT / "logs" / "experiments",
    ]
    candidates = _scan_candidate_score_files(search_roots=search_roots, score_key=score_key)
    if not candidates:
        raise FileNotFoundError("No score summary with dataset/model/score fields was found in the expected repository roots.")
    return candidates[0], candidates


def _score_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if value <= 1.5 else value


def _panel_long_rows(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        panel = str(row["panel"])
        model_name = str(row["model_name"])
        dataset_type = str(row["dataset_type"])
        grouped[(panel, model_name, dataset_type)].append(row)

    out: list[dict[str, Any]] = []
    for (panel, model_name, dataset_type), items in sorted(grouped.items()):
        score_values = [float(item["overall_score_pct"]) for item in items if item.get("overall_score_pct") is not None]
        time_values = [float(item["time_per_1k_seconds"]) for item in items if item.get("time_per_1k_seconds") is not None]
        if not score_values or not time_values:
            continue
        out.append(
            {
                "panel": panel,
                "model_name": model_name,
                "dataset_type": dataset_type,
                "overall_score_pct_median": round(float(median(score_values)), 6),
                "time_per_1k_seconds_median": round(float(median(time_values)), 6),
                "n_datasets": len(items),
                "dataset_ids": "|".join(sorted({str(item["dataset_id"]) for item in items})),
                "source_rows": len(items),
                "aggregation": "median",
                "is_imputed": False,
                "imputation_basis": "",
                "imputation_ratio": "",
            }
        )
    return out


def _impute_inference_rows_from_training(aggregated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    training_rows = {
        (str(row["model_name"]), str(row["dataset_type"])): row
        for row in aggregated_rows
        if str(row["panel"]) == "training"
    }
    inference_rows = {
        (str(row["model_name"]), str(row["dataset_type"])): row
        for row in aggregated_rows
        if str(row["panel"]) == "inference"
    }

    ratios_by_type: dict[str, list[float]] = defaultdict(list)
    global_ratios: list[float] = []
    for key, train_row in training_rows.items():
        infer_row = inference_rows.get(key)
        if infer_row is None:
            continue
        train_time = float(train_row["time_per_1k_seconds_median"])
        infer_time = float(infer_row["time_per_1k_seconds_median"])
        if train_time <= 0 or infer_time <= 0:
            continue
        ratio = infer_time / train_time
        dataset_type = str(train_row["dataset_type"])
        ratios_by_type[dataset_type].append(ratio)
        global_ratios.append(ratio)

    if not global_ratios:
        return aggregated_rows

    inferred: list[dict[str, Any]] = list(aggregated_rows)
    global_ratio = float(median(global_ratios))

    for key, train_row in sorted(training_rows.items()):
        if key in inference_rows:
            continue
        dataset_type = str(train_row["dataset_type"])
        type_ratios = ratios_by_type.get(dataset_type) or []
        ratio = float(median(type_ratios)) if type_ratios else global_ratio
        train_time = float(train_row["time_per_1k_seconds_median"])
        inferred.append(
            {
                "panel": "inference",
                "model_name": str(train_row["model_name"]),
                "dataset_type": dataset_type,
                "overall_score_pct_median": float(train_row["overall_score_pct_median"]),
                "time_per_1k_seconds_median": round(train_time * ratio, 6),
                "n_datasets": int(train_row["n_datasets"]),
                "dataset_ids": str(train_row["dataset_ids"]),
                "source_rows": int(train_row["source_rows"]),
                "aggregation": "median",
                "is_imputed": True,
                "imputation_basis": "median_generation_to_training_ratio_by_dataset_type",
                "imputation_ratio": round(ratio, 6),
            }
        )
    return inferred


def _pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_points = sorted(points, key=lambda item: (float(item["time_per_1k_seconds_median"]), -float(item["overall_score_pct_median"])))
    frontier: list[dict[str, Any]] = []
    best_y = -math.inf
    for point in sorted_points:
        y = float(point["overall_score_pct_median"])
        if y > best_y:
            frontier.append(point)
            best_y = y
    return frontier


def _make_figure(
    aggregated_rows: list[dict[str, Any]],
    output_pdf: Path,
    output_png: Path,
    export_pgf: bool,
    show_labels: str,
    included_models: list[str],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    panel_specs = [
        ("inference", "Generation Time per 1K Samples (s)"),
        ("training", "Training Time per 1K Train Rows (s)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=False)
    fig.patch.set_facecolor("white")

    plotted_models = {str(row["model_name"]) for row in aggregated_rows}
    all_models = [model for model in included_models if model in plotted_models]

    for ax, (panel_key, xlabel) in zip(axes, panel_specs, strict=False):
        panel_rows = [row for row in aggregated_rows if row["panel"] == panel_key]
        for row in panel_rows:
            model_name = str(row["model_name"])
            dataset_type = str(row["dataset_type"])
            x = float(row["time_per_1k_seconds_median"])
            y = float(row["overall_score_pct_median"])
            marker = DATASET_TYPE_MARKERS.get(dataset_type, "o")
            color = _model_color(model_name)
            is_imputed = bool(row.get("is_imputed"))
            alpha = 0.62 if is_imputed else 0.96
            if marker == "x":
                ax.scatter(x, y, marker=marker, s=70, linewidths=1.5, color=color, alpha=alpha, zorder=3)
            else:
                ax.scatter(
                    x,
                    y,
                    marker=marker,
                    s=62,
                    facecolor=color,
                    edgecolor=("#222222" if is_imputed else "white"),
                    linewidth=(0.9 if is_imputed else 0.8),
                    alpha=alpha,
                    zorder=3,
                )

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Overall Score (%)")
        ax.grid(True, which="major", alpha=0.22, linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.10, linewidth=0.45)
        ax.set_axisbelow(True)
        ax.set_title("Faster and Better is Top-Left")

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6.5,
            markerfacecolor=_model_color(model_name),
            markeredgecolor=_model_color(model_name),
            label=_display_model_name(model_name),
            alpha=1.0,
        )
        for model_name in all_models
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markersize=6.5,
            markerfacecolor=("none" if marker == "x" else "#666666"),
            markeredgecolor="#444444",
            color="#444444",
            markeredgewidth=1.2,
            label=label.capitalize(),
        )
        for label, marker in DATASET_TYPE_MARKERS.items()
    ]

    fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=7,
        frameon=False,
        title="Model",
        title_fontsize=9,
    )
    fig.legend(
        handles=marker_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        title="Dataset Type",
        title_fontsize=9,
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.25, wspace=0.18)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    if export_pgf:
        fig.savefig(output_pdf.with_suffix(".pgf"), bbox_inches="tight")
    plt.close(fig)


def _tikz_color_name(model_name: str) -> str:
    return "model" + re.sub(r"[^a-zA-Z0-9]+", "", model_name.lower())


def _tikz_marker_style(dataset_type: str, is_imputed: bool) -> str:
    opacity = "0.62" if is_imputed else "0.96"
    if dataset_type == "categorical":
        return f"mark=*, mark size=2.4pt, draw=white, fill opacity={opacity}, draw opacity={opacity}, opacity={opacity}"
    if dataset_type == "numerical":
        return f"mark=square*, mark size=2.5pt, draw=white, fill opacity={opacity}, draw opacity={opacity}, opacity={opacity}"
    return f"mark=x, mark size=2.9pt, line width=0.95pt, opacity={opacity}"


def _legend_model_cells(models: list[str], plotted_models: set[str], ncols: int = 6) -> str:
    entries: list[str] = []
    for model_name in models:
        color_name = _tikz_color_name(model_name)
        label = _latex_escape(_display_model_name(model_name))
        entries.append(
            rf"\tikz[baseline=-0.6ex] \fill[{color_name}, opacity=1.0] (0,0) circle (2.2pt);~{label}"
        )
    rows = []
    for idx in range(0, len(entries), ncols):
        row = entries[idx : idx + ncols]
        if len(row) < ncols:
            row = row + [""] * (ncols - len(row))
        rows.append(" & ".join(row) + r" \\")
    return "\n".join(rows)


def _legend_dataset_cells() -> str:
    return "\n".join(
        [
            r"\tikz[baseline=-0.6ex] \fill[black!60] (0,0) circle (2.2pt);~Categorical"
            + r" & "
            + r"\tikz[baseline=-0.6ex] \fill[black!60] (-0.11,-0.11) rectangle (0.11,0.11);~Numerical"
            + r" & "
            + r"\tikz[baseline=-0.6ex] \draw[line width=0.95pt] (-0.12,-0.12) -- (0.12,0.12) (-0.12,0.12) -- (0.12,-0.12);~Mixed \\"
        ]
    )


def _build_tikz_tex(aggregated_rows: list[dict[str, Any]], included_models: list[str], title: str) -> str:
    color_defs = []
    for model_name in included_models:
        color_name = _tikz_color_name(model_name)
        hex_color = MODEL_COLORS.get(model_name)
        if hex_color and hex_color.startswith("#"):
            color_defs.append(rf"\definecolor{{{color_name}}}{{HTML}}{{{hex_color[1:]}}}")
    plotted_models = {str(row["model_name"]) for row in aggregated_rows}
    legend_models = [model for model in included_models if model in plotted_models]
    panels = [
        ("inference", "Generation Time per 1K Samples (s)"),
        ("training", "Training Time per 1K Train Rows (s)"),
    ]

    panel_blocks: list[str] = []
    for panel_key, xlabel in panels:
        lines = [
            r"\nextgroupplot[",
            rf"title={{{_latex_escape(title)}}},",
            rf"xlabel={{{_latex_escape(xlabel)}}},",
            r"ylabel={Overall Score (\%)},",
            r"xmode=log,",
            r"grid=major,",
            r"minor grid style={draw=gray!12},",
            r"major grid style={draw=gray!22},",
            r"tick label style={font=\small},",
            r"label style={font=\small},",
            r"title style={font=\normalsize},",
            r"]",
        ]
        panel_rows = [row for row in aggregated_rows if str(row["panel"]) == panel_key]
        grouped: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
        for row in panel_rows:
            key = (str(row["model_name"]), str(row["dataset_type"]), bool(row.get("is_imputed")))
            grouped[key].append(row)
        for model_name in legend_models:
            for dataset_type in DATASET_TYPE_MARKERS:
                matching_keys = [key for key in grouped if key[0] == model_name and key[1] == dataset_type]
                for key in sorted(matching_keys, key=lambda item: item[2]):
                    rows = grouped[key]
                    coords = " ".join(
                        f"({float(row['time_per_1k_seconds_median']):.6f},{float(row['overall_score_pct_median']):.6f})"
                        for row in rows
                    )
                    color_name = _tikz_color_name(model_name)
                    marker_style = _tikz_marker_style(dataset_type, key[2])
                    lines.append(rf"\addplot+[only marks, color={color_name}, {marker_style}] coordinates {{{coords}}};")
        panel_blocks.append("\n".join(lines))

    color_defs_block = "\n".join(color_defs)
    return f"""\\documentclass[tikz,border=6pt]{{standalone}}
\\usepackage{{pgfplots}}
\\usepackage{{tikz}}
\\usetikzlibrary{{calc,positioning}}
\\usepgfplotslibrary{{groupplots}}
\\pgfplotsset{{compat=1.18}}
{color_defs_block}
\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
group style={{group size=2 by 1, horizontal sep=1.6cm}},
width=0.44\\textwidth,
height=0.31\\textwidth,
ymin=17.5,
]
{chr(10).join(panel_blocks)}
\\end{{groupplot}}

\\node[anchor=north] at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-1.25cm)$) {{
\\begin{{tabular}}{{{'c' * min(max(len(legend_models), 1), 6)}}}
\\multicolumn{{{min(max(len(legend_models), 1), 6)}}}{{c}}{{\\textbf{{Model}}}}\\\\[2pt]
{_legend_model_cells(legend_models, plotted_models, ncols=6)}
\\end{{tabular}}
}};

\\node[anchor=north] at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-2.35cm)$) {{
\\begin{{tabular}}{{ccc}}
\\multicolumn{{3}}{{c}}{{\\textbf{{Dataset Type}}}}\\\\[2pt]
{_legend_dataset_cells()}
\\end{{tabular}}
}};
\\end{{tikzpicture}}
\\end{{document}}
"""


def _compile_tex_to_pdf(tex_path: Path) -> tuple[bool, str]:
    engine = shutil.which("tectonic")
    if not engine:
        return False, "tectonic not found"
    command = [engine, "--outdir", str(tex_path.parent), str(tex_path)]
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "").strip().splitlines()[-20:]
        return False, "\n".join(tail)
    return True, ""


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)
        fieldnames = ordered
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_readme(
    path: Path,
    *,
    command: str,
    score_source: ScoreSource,
    candidate_sources: list[ScoreSource],
    used_panel_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    plot_rows: list[dict[str, Any]],
    tex_path: Path,
    excluded_models: list[str],
) -> None:
    used_count = sum(1 for row in audit_rows if row.get("status") == "valid_both")
    dropped_count = len(audit_rows) - used_count
    inference_used = sum(1 for row in audit_rows if str(row.get("panel_inference_status")) == "valid")
    training_used = sum(1 for row in audit_rows if str(row.get("panel_training_status")) == "valid")
    candidate_lines = "\n".join(
        f"- `{_safe_rel(item.path)}` using score key `{item.score_key}`"
        for item in candidate_sources[:5]
    )

    text = f"""# Time/Score Trade-Off Figure

## Files Read

- Selected score source: `{_safe_rel(score_source.path)}` using score key `{score_source.score_key}`.
- Runtime metadata: per-asset log and metadata paths referenced by the selected score source, under `SynOutput/` and `SynOutput-5090/`.
- Train split row counts: local `data/<dataset_id>/<dataset_id>-train.csv` or `data/<dataset_id>/raw/<dataset_id>-train.csv` when available.
- TeX source: `{_safe_rel(tex_path)}` compiled with `tectonic` when available.

Top score-source candidates considered:
{candidate_lines if candidate_lines else "- none"}

## Aggregation

- Aggregation unit: `model_name x dataset_type`.
- Excluded models: `{", ".join(excluded_models) if excluded_models else "none"}`.
- Dataset type mapping: dataset id prefix `c -> categorical`, `n -> numerical`, `m -> mixed`.
- Panel aggregation: median score and median time are computed separately per panel over the rows that are valid for that panel.
- Left panel rows require valid score, generation time, and generated row count.
- Right panel rows require valid score, training time, and training row count.

## Time Normalization

- `inference_time_per_1k = generation_time_seconds / generated_rows * 1000`
- `training_time_per_1k = training_time_seconds / train_rows * 1000`

## Inclusion Summary

- Rows with both train and generation times available: `{used_count}`
- Rows dropped from at least one panel: `{dropped_count}`
- Rows contributing to the generation panel: `{inference_used}`
- Rows contributing to the training panel: `{training_used}`
- Aggregated plotted points: `{len(plot_rows)}`

## Reproduce

```bash
{command}
```
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot score/time trade-off for synthetic generators.")
    parser.add_argument("--evaluation-root", type=Path, default=Path("Evaluation"))
    parser.add_argument("--synoutput-roots", nargs="+", default=["SynOutput", "SynOutput-5090"])
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/time_score_tradeoff"))
    parser.add_argument("--score-key", type=str, default="auto")
    parser.add_argument("--min-datasets-per-point", type=int, default=1)
    parser.add_argument("--export-pgf", action="store_true")
    parser.add_argument("--show-labels", choices=["none", "pareto"], default="none")
    parser.add_argument("--demo-impute-generation-from-training", action="store_true")
    parser.add_argument("--output-stem", type=str, default="time_score_tradeoff")
    parser.add_argument("--exclude-models", nargs="+", default=list(DEFAULT_EXCLUDED_MODELS))
    parser.add_argument("--command-name", type=str, default="python src/eval/time_score_tradeoff/runner.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_root = (PROJECT_ROOT / args.evaluation_root).resolve() if not args.evaluation_root.is_absolute() else args.evaluation_root.resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    score_source, candidates = _choose_latest_score_source(evaluation_root=evaluation_root, score_key=args.score_key)
    score_rows = _load_score_rows(score_source)
    excluded_models = {_normalize_model_name(item) for item in args.exclude_models}
    included_models = [model for model in KNOWN_MODEL_ORDER if model in PAPER_MODEL_SET and model not in excluded_models]

    train_row_cache: dict[str, int | None] = {}
    audit_rows: list[dict[str, Any]] = []
    panel_valid_rows: list[dict[str, Any]] = []

    for row in score_rows:
        dataset_id = str(row.get("dataset_id") or row.get("dataset") or "").strip().lower()
        model_name = _normalize_model_name(str(row.get("model_name") or row.get("model_id") or row.get("model") or ""))
        score_value = _as_float(row.get(score_source.score_key))
        score_pct = _score_percent(score_value)
        if not dataset_id or not model_name:
            continue
        if model_name not in PAPER_MODEL_SET:
            continue
        if model_name in excluded_models:
            continue

        synthetic_csv_path = Path(str(row.get("synthetic_csv_path") or "")).expanduser()
        if synthetic_csv_path and not synthetic_csv_path.is_absolute():
            synthetic_csv_path = (PROJECT_ROOT / synthetic_csv_path).resolve()
        metadata_paths = _parse_path_list(row.get("metadata_paths"))
        log_paths = _parse_path_list(row.get("log_paths"))

        dataset_type = _dataset_type_for_id(dataset_id)
        train_rows = _find_train_rows(row=row, dataset_id=dataset_id, metadata_paths=metadata_paths, cache=train_row_cache)
        generated_rows = _find_generated_rows(row=row, synthetic_csv_path=synthetic_csv_path, metadata_paths=metadata_paths)

        direct_train = _pick_numeric_from_row(row, TRAIN_TIME_KEYS)
        meta_train = _extract_runtime_from_metadata(metadata_paths, TRAIN_TIME_KEYS)
        log_train = _extract_train_time_from_logs(log_paths, synthetic_csv_path)
        training_time = _prefer_runtime(direct_train, meta_train, log_train)

        direct_gen = _pick_numeric_from_row(row, GEN_TIME_KEYS)
        meta_gen = _extract_runtime_from_metadata(metadata_paths, GEN_TIME_KEYS)
        log_gen = _extract_generation_time_from_logs(log_paths, synthetic_csv_path)
        generation_time = _prefer_runtime(direct_gen, meta_gen, log_gen)

        inference_time_per_1k = None
        if generation_time.value is not None and generated_rows.value not in {None, 0}:
            inference_time_per_1k = float(generation_time.value) / float(generated_rows.value) * 1000.0

        training_time_per_1k = None
        if training_time.value is not None and train_rows.value not in {None, 0}:
            training_time_per_1k = float(training_time.value) / float(train_rows.value) * 1000.0

        missing_fields: list[str] = []
        if score_pct is None:
            missing_fields.append("score")
        if training_time.value is None:
            missing_fields.append("training_time_seconds")
        if generation_time.value is None:
            missing_fields.append("generation_time_seconds")
        if train_rows.value is None:
            missing_fields.append("train_rows")
        if generated_rows.value is None:
            missing_fields.append("generated_rows")

        status = "valid_both" if score_pct is not None and inference_time_per_1k is not None and training_time_per_1k is not None else "partial_or_missing"
        inference_status = "valid" if score_pct is not None and inference_time_per_1k is not None else "dropped"
        training_status = "valid" if score_pct is not None and training_time_per_1k is not None else "dropped"

        audit_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_type": dataset_type,
                "model_name": model_name,
                "run_id": row.get("run_id"),
                "server_type": row.get("server_type"),
                "status": status,
                "panel_inference_status": inference_status,
                "panel_training_status": training_status,
                "overall_score_raw": score_value,
                "overall_score_pct": score_pct,
                "score_key": score_source.score_key,
                "training_time_seconds": training_time.value,
                "training_time_source": training_time.source,
                "generation_time_seconds": generation_time.value,
                "generation_time_source": generation_time.source,
                "train_rows": _as_int(train_rows.value),
                "train_rows_source": train_rows.source,
                "generated_rows": _as_int(generated_rows.value),
                "generated_rows_source": generated_rows.source,
                "training_time_per_1k_seconds": training_time_per_1k,
                "inference_time_per_1k_seconds": inference_time_per_1k,
                "missing_fields": "|".join(missing_fields),
                "synthetic_csv_path": str(synthetic_csv_path) if synthetic_csv_path else "",
                "log_paths": "|".join(str(path) for path in log_paths),
                "metadata_paths": "|".join(str(path) for path in metadata_paths),
                "score_source_path": str(score_source.path.resolve()),
            }
        )

        if score_pct is not None and inference_time_per_1k is not None:
            panel_valid_rows.append(
                {
                    "panel": "inference",
                    "dataset_id": dataset_id,
                    "dataset_type": dataset_type,
                    "model_name": model_name,
                    "overall_score_pct": score_pct,
                    "time_per_1k_seconds": inference_time_per_1k,
                }
            )
        if score_pct is not None and training_time_per_1k is not None:
            panel_valid_rows.append(
                {
                    "panel": "training",
                    "dataset_id": dataset_id,
                    "dataset_type": dataset_type,
                    "model_name": model_name,
                    "overall_score_pct": score_pct,
                    "time_per_1k_seconds": training_time_per_1k,
                }
            )

    aggregated_rows = [
        row
        for row in _panel_long_rows(panel_valid_rows)
        if int(row.get("n_datasets") or 0) >= int(args.min_datasets_per_point)
        and str(row.get("dataset_type")) in DATASET_TYPE_MARKERS
    ]

    if args.demo_impute_generation_from_training:
        aggregated_rows = _impute_inference_rows_from_training(aggregated_rows)

    stem = args.output_stem
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    tex_path = output_dir / f"{stem}.tex"
    data_csv_path = output_dir / f"{stem}_data.csv"
    audit_csv_path = output_dir / f"{stem}_audit.csv"
    readme_path = output_dir / f"{stem}_README.md"

    _write_csv(data_csv_path, aggregated_rows)
    _write_csv(audit_csv_path, audit_rows)

    if aggregated_rows:
        _make_figure(
            aggregated_rows=aggregated_rows,
            output_pdf=pdf_path,
            output_png=png_path,
            export_pgf=args.export_pgf,
            show_labels=args.show_labels,
            included_models=included_models,
        )
        tex_path.write_text(_build_tikz_tex(aggregated_rows, included_models, title="Faster and Better is Top-Left"), encoding="utf-8")
        ok, tex_log = _compile_tex_to_pdf(tex_path)
        if not ok:
            print(f"Warning: TeX compilation failed for {tex_path.name}. Keeping matplotlib PDF fallback.\n{tex_log}", file=sys.stderr)
    else:
        print("No aggregated rows satisfied the plotting requirements; figure files were not created.", file=sys.stderr)

    command = args.command_name
    if args.evaluation_root != Path("Evaluation"):
        command += f" --evaluation-root {args.evaluation_root}"
    if args.synoutput_roots != ["SynOutput", "SynOutput-5090"]:
        command += " --synoutput-roots " + " ".join(args.synoutput_roots)
    if args.output_dir != Path("Evaluation/time_score_tradeoff"):
        command += f" --output-dir {args.output_dir}"
    if args.score_key != "auto":
        command += f" --score-key {args.score_key}"
    if args.min_datasets_per_point != 1:
        command += f" --min-datasets-per-point {args.min_datasets_per_point}"
    if args.export_pgf:
        command += " --export-pgf"
    if args.show_labels != "none":
        command += f" --show-labels {args.show_labels}"
    if args.demo_impute_generation_from_training:
        command += " --demo-impute-generation-from-training"
    if args.output_stem != "time_score_tradeoff":
        command += f" --output-stem {args.output_stem}"
    if args.exclude_models != list(DEFAULT_EXCLUDED_MODELS):
        command += " --exclude-models " + " ".join(args.exclude_models)

    _write_readme(
        readme_path,
        command=command,
        score_source=score_source,
        candidate_sources=candidates,
        used_panel_rows=panel_valid_rows,
        audit_rows=audit_rows,
        plot_rows=aggregated_rows,
        tex_path=tex_path,
        excluded_models=sorted(excluded_models),
    )

    inference_used = sum(1 for row in audit_rows if row["panel_inference_status"] == "valid")
    inference_dropped = len(audit_rows) - inference_used
    training_used = sum(1 for row in audit_rows if row["panel_training_status"] == "valid")
    training_dropped = len(audit_rows) - training_used

    print(f"Selected score source: {_safe_rel(score_source.path)}")
    print(f"Score key: {score_source.score_key}")
    print(f"Inference panel rows used: {inference_used}; dropped: {inference_dropped}")
    print(f"Training panel rows used: {training_used}; dropped: {training_dropped}")
    print(f"Aggregated plot points: {len(aggregated_rows)}")
    print(f"Data CSV: {data_csv_path}")
    print(f"Audit CSV: {audit_csv_path}")
    if aggregated_rows:
        print(f"Figure PDF: {pdf_path}")
        print(f"Figure PNG: {png_path}")
    print(f"README: {readme_path}")


if __name__ == "__main__":
    main()
