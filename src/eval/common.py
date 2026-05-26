"""Shared utilities for synthetic-data evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.eval.subitem_workload_v2.paths import (
    SUPPORTED_LINE_VERSIONS,
    normalize_line_version,
    registry_dir,
    run_manifest_dir,
    runs_root,
)
from src.eval.subitem_workload_v2.registry import load_registry_rows

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


DATA_ROOT = _env_path("EVAL_REAL_DATA_ROOT", PROJECT_ROOT / "data")
LOGS_ROOT = _env_path("EVAL_LOGS_ROOT", PROJECT_ROOT / "logs" / "runs")
OUTPUT_ROOT = _env_path("EVAL_OUTPUT_ROOT", PROJECT_ROOT / "Evaluation")
SQL_RESULT_ROLE_ANNOTATION_ROOT = DATA_ROOT / "sql_result_role_annotations_v1" / "datasets"

PROVENANCE_CONTRACT_VERSION = "evaluation_source_provenance_v1"
SQL_SOURCE_VERSION_ENV_VAR = "EVAL_SQL_SOURCE_VERSION"
SQL_SOURCE_VERSION_V1 = "v1"
SQL_SOURCE_VERSION_V2 = "v2"
SQL_SOURCE_VERSION_V3 = "v3"
SQL_SOURCE_VERSION_V4 = "v4"
CURRENT_SQL_SOURCE_VERSIONS = tuple(SUPPORTED_LINE_VERSIONS)
SQL_SOURCE_VERSION_CHOICES = (
    SQL_SOURCE_VERSION_V1,
    *CURRENT_SQL_SOURCE_VERSIONS,
)
DEFAULT_SQL_SOURCE_VERSION = SQL_SOURCE_VERSION_V2

_SQL_SOURCE_LABELS = {
    SQL_SOURCE_VERSION_V1: "v1_legacy",
    SQL_SOURCE_VERSION_V2: "v2_current",
    SQL_SOURCE_VERSION_V3: "v3_current",
    SQL_SOURCE_VERSION_V4: "v4_current",
}
_SQL_SOURCE_DESCRIPTIONS = {
    SQL_SOURCE_VERSION_V1: "legacy grounded SQL runs under logs/runs/",
    SQL_SOURCE_VERSION_V2: "current registry-backed workload SQL under logs/subitem_workload_v2/",
    SQL_SOURCE_VERSION_V3: "current registry-backed workload SQL under logs/subitem_workload_v3/",
    SQL_SOURCE_VERSION_V4: "current registry-backed workload SQL under logs/subitem_workload_v4/",
}
_SQL_SOURCE_ALIASES = {
    "v1": SQL_SOURCE_VERSION_V1,
    "legacy": SQL_SOURCE_VERSION_V1,
    "v1_legacy": SQL_SOURCE_VERSION_V1,
    "logs/runs": SQL_SOURCE_VERSION_V1,
    "logs\\runs": SQL_SOURCE_VERSION_V1,
    "v2": SQL_SOURCE_VERSION_V2,
    "query_registry_v2": SQL_SOURCE_VERSION_V2,
    "current": SQL_SOURCE_VERSION_V2,
    "v2_current": SQL_SOURCE_VERSION_V2,
    "subitem_workload_v2": SQL_SOURCE_VERSION_V2,
    "logs/subitem_workload_v2": SQL_SOURCE_VERSION_V2,
    "logs\\subitem_workload_v2": SQL_SOURCE_VERSION_V2,
    "v3": SQL_SOURCE_VERSION_V3,
    "v3_current": SQL_SOURCE_VERSION_V3,
    "query_registry_v3": SQL_SOURCE_VERSION_V3,
    "subitem_workload_v3": SQL_SOURCE_VERSION_V3,
    "logs/subitem_workload_v3": SQL_SOURCE_VERSION_V3,
    "logs\\subitem_workload_v3": SQL_SOURCE_VERSION_V3,
    "v4": SQL_SOURCE_VERSION_V4,
    "v4_current": SQL_SOURCE_VERSION_V4,
    "query_registry_v4": SQL_SOURCE_VERSION_V4,
    "subitem_workload_v4": SQL_SOURCE_VERSION_V4,
    "logs/subitem_workload_v4": SQL_SOURCE_VERSION_V4,
    "logs\\subitem_workload_v4": SQL_SOURCE_VERSION_V4,
}

ROOT_CONFIGS = {
    "SynOutput": {
        "path": _env_path("EVAL_SYNOUTPUT_ROOT", PROJECT_ROOT / "SynOutput"),
        "server_type": "rtx_pro_6000",
        "gpu_hour_ratio": 1.0,
    },
    "SynOutput-5090": {
        "path": _env_path("EVAL_SYNOUTPUT_5090_ROOT", PROJECT_ROOT / "SynOutput-5090"),
        "server_type": "rtx_5090",
        "gpu_hour_ratio": 1.0,
    },
    "Benchmark-trainonly-v1": {
        "path": _env_path("EVAL_BENCHMARK_TRAINONLY_ROOT", PROJECT_ROOT / "remote-output-Benchmark-trainonly-v1"),
        "server_type": "trainonly_serial",
        "gpu_hour_ratio": 1.0,
    },
    "Hyperparameter-trainonly-v1": {
        "path": _env_path(
            "EVAL_HYPERPARAMETER_TRAINONLY_ROOT",
            PROJECT_ROOT / "hyperparameter" / "output-Benchmark-trainonly-v1",
        ),
        "server_type": "hyperparameter_trainonly",
        "gpu_hour_ratio": 1.0,
    },
    "TabQueryBench-SynDataSuccess-main": {
        "path": _env_path(
            "EVAL_TABQUERYBENCH_MAIN_ROOT",
            Path("/data/jialinzhang/TabQueryBench/SynDataSuccess/main"),
        ),
        "server_type": "server_authoritative_main",
        "gpu_hour_ratio": 1.0,
    },
}

USD_PER_GPU_HOUR = 1.0
MAX_FALLBACK_GPU_SECONDS = 12 * 3600
MISSING_TEXT = {"", "null", "none", "nan", "na", "n/a", "<null>"}
TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")
RUNTIME_RESULT_RE = re.compile(r"(?P<prefix>.+?)__runtime_result\.json$", re.IGNORECASE)
TRAIN_TIME_RE = re.compile(
    r"(?:totoal|total)\s+training\s+time\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
SAMPLE_TIME_RE = re.compile(
    r"(?:totoal|total)\s+sampling\s+time\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
GENERIC_SECONDS_RE = re.compile(
    r"(?:elapsed|duration|runtime|wall\s*time|completed\s+in|finished\s+in)\D+([0-9]+(?:\.[0-9]+)?)\s*(?:seconds|secs|s)?",
    re.IGNORECASE,
)
SUBITEM_RUNS_PATH_RE = re.compile(
    r"/logs/subitem_workload_(v[234])/runs/(?P<suffix>.+)$",
    re.IGNORECASE,
)


@dataclass
class SyntheticAsset:
    dataset_id: str
    model_id: str
    server_type: str
    root_name: str
    root_path: str
    asset_dir: str
    run_id: str
    synthetic_csv_path: str
    metadata_paths: list[str]
    log_paths: list[str]
    discovered_via: str
    timestamp_utc: str | None
    synthetic_source_mtime_utc: str | None
    synthetic_source_size_bytes: int | None
    gpu_seconds_raw: float
    gpu_hours_equivalent: float
    gpu_hours_source: str
    cost_usd: float

    @property
    def asset_key(self) -> str:
        return f"{self.dataset_id}__{self.server_type}__{self.model_id}__{self.run_id}"

    @property
    def model_server_key(self) -> str:
        return f"{self.model_id}__{self.server_type}"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["asset_key"] = self.asset_key
        row["model_server_key"] = self.model_server_key
        row["provenance_contract_version"] = PROVENANCE_CONTRACT_VERSION
        row["synthetic_source_path"] = row["synthetic_csv_path"]
        row["synthetic_source_root_name"] = row["root_name"]
        row["synthetic_source_root_path"] = row["root_path"]
        row["synthetic_source_asset_dir"] = row["asset_dir"]
        row["synthetic_source_run_id"] = row["run_id"]
        row["synthetic_source_discovered_via"] = row["discovered_via"]
        return row


def now_run_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: set[str] = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "--:--:--"
    total_seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class TaskProgressTracker:
    task_name: str
    total_steps: int
    step_label: str = "datasets"
    substep_label: str = "assets"
    total_substeps: int = 0
    completed_steps: int = 0
    completed_substeps: int = 0

    def __post_init__(self) -> None:
        self._start_ts = time.monotonic()
        self._last_print_ts = self._start_ts

    def print_start(self, extra: str = "") -> None:
        parts = [
            f"[{self.task_name}] start",
            f"{self.step_label}=0/{self.total_steps}",
        ]
        if self.total_substeps > 0:
            parts.append(f"{self.substep_label}=0/{self.total_substeps}")
        if extra:
            parts.append(extra)
        print(" | ".join(parts), flush=True)

    def advance(self, *, step_name: str, substeps_done: int = 0, extra: str = "") -> None:
        self.completed_steps += 1
        self.completed_substeps += max(0, int(substeps_done))
        elapsed = time.monotonic() - self._start_ts
        avg_per_step = (elapsed / self.completed_steps) if self.completed_steps > 0 else None
        remaining_steps = max(0, self.total_steps - self.completed_steps)
        eta_seconds = (avg_per_step * remaining_steps) if avg_per_step is not None else None

        parts = [
            f"[{self.task_name}] {self.step_label}={self.completed_steps}/{self.total_steps}",
        ]
        if self.total_substeps > 0:
            parts.append(f"{self.substep_label}={self.completed_substeps}/{self.total_substeps}")
        parts.extend(
            [
                f"elapsed={format_duration(elapsed)}",
                f"eta={format_duration(eta_seconds)}",
                f"done={step_name}",
            ]
        )
        if extra:
            parts.append(extra)
        print(" | ".join(parts), flush=True)


def make_task_run_dir(task_name: str, run_tag: str) -> Path:
    run_dir = OUTPUT_ROOT / task_name / "runs" / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / task_name / "LATEST_RUN.json", {"run_tag": run_tag, "run_dir": str(run_dir.resolve())})
    return run_dir


def list_dataset_ids() -> list[str]:
    out: list[str] = []
    if not DATA_ROOT.exists():
        return out
    for path in sorted(DATA_ROOT.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        train_csv = resolve_real_split_path(path.name, split="train")
        if train_csv.exists():
            out.append(path.name)
    return out


def resolve_dataset_dir(dataset_id: str) -> Path:
    return DATA_ROOT / dataset_id


def resolve_real_split_path(dataset_id: str, split: str = "train") -> Path:
    candidates = [
        DATA_ROOT / dataset_id / f"{dataset_id}-{split}.csv",
        DATA_ROOT / dataset_id / "raw" / f"{dataset_id}-{split}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_field_registry_path(dataset_id: str) -> Path | None:
    candidates = [
        DATA_ROOT / dataset_id / "metadata_core" / "field_registry.json",
        DATA_ROOT / dataset_id / "metadata" / "field_registry.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_field_registry(dataset_id: str) -> dict[str, Any]:
    path = resolve_field_registry_path(dataset_id)
    if path is None:
        return {}
    return read_json(path, {}) or {}


def load_field_type_hints(dataset_id: str) -> dict[str, str]:
    payload = load_field_registry(dataset_id)
    hints: dict[str, str] = {}
    for item in payload.get("fields", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        semantic = str(item.get("semantic_type") or "").strip().lower()
        declared = str(item.get("declared_type") or "").strip().lower()
        hints[name] = semantic or declared
    return hints


def resolve_sql_result_role_annotation_path(dataset_id: str) -> Path:
    return SQL_RESULT_ROLE_ANNOTATION_ROOT / dataset_id / "outputs" / "sql_result_roles_ai_v1.json"


def load_sql_result_role_annotations(
    dataset_id: str,
    *,
    sql_source_version: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    path = resolve_sql_result_role_annotation_path(dataset_id)
    payload = read_json(path, {}) or {}
    query_annotations = payload.get("query_annotations") if isinstance(payload, dict) else []
    requested_version = normalize_sql_source_version(sql_source_version) if sql_source_version else None

    output: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(query_annotations, list):
        return output

    for item in query_annotations:
        if not isinstance(item, dict):
            continue
        version_text = str(item.get("sql_source_version") or "").strip()
        query_id = str(item.get("query_id") or "").strip()
        if not query_id:
            continue
        try:
            normalized_version = normalize_sql_source_version(version_text or requested_version or DEFAULT_SQL_SOURCE_VERSION)
        except Exception:
            continue
        if requested_version and normalized_version != requested_version:
            continue
        output[(normalized_version, query_id)] = item
    return output


def parse_timestamp_text(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass
    match = TIMESTAMP_RE.search(text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _candidate_timestamps(*values: str | Path | None) -> list[datetime]:
    out: list[datetime] = []
    for value in values:
        if value is None:
            continue
        parsed = parse_timestamp_text(str(value))
        if parsed is not None:
            out.append(parsed)
    return out


def _stat_mtime_ts(path: Path | None) -> datetime | None:
    if path is None or not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _stat_size_bytes(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        return int(path.stat().st_size)
    except Exception:
        return None


def _resolved_path_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.expanduser().resolve())
    except Exception:
        return str(path)


def _path_provenance_fields(prefix: str, path: Path | None) -> dict[str, Any]:
    mtime = _stat_mtime_ts(path)
    return {
        f"{prefix}_path": _resolved_path_text(path),
        f"{prefix}_exists": bool(path and path.exists()),
        f"{prefix}_mtime_utc": (mtime.isoformat() if mtime is not None else None),
        f"{prefix}_size_bytes": _stat_size_bytes(path),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_registry_backed_path(raw_path: str | Path | None) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return Path("")
    candidate = Path(text).expanduser()
    if candidate.exists():
        return candidate

    normalized = text.replace("\\", "/")
    marker = "/SQLagent/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1].lstrip("/")
        rebased = (PROJECT_ROOT / suffix).resolve()
        if rebased.exists():
            return rebased

    if normalized.startswith("SQLagent/"):
        rebased = (PROJECT_ROOT / normalized[len("SQLagent/"):]).resolve()
        if rebased.exists():
            return rebased

    match = SUBITEM_RUNS_PATH_RE.search(normalized)
    if match:
        version = match.group(1).lower()
        suffix = match.group("suffix").lstrip("/")
        rebased = (runs_root(version) / suffix).resolve()
        if rebased.exists():
            return rebased

    return candidate


def sql_source_family(version: str | None) -> str:
    normalized = normalize_sql_source_version(version)
    return "legacy" if normalized == SQL_SOURCE_VERSION_V1 else "current"


def sql_source_line_version(version: str | None) -> str:
    normalized = normalize_sql_source_version(version)
    return normalized if normalized in CURRENT_SQL_SOURCE_VERSIONS else ""


def sql_source_registry_root(version: str | None) -> Path | None:
    normalized = normalize_sql_source_version(version)
    if normalized == SQL_SOURCE_VERSION_V1:
        return None
    return registry_dir(normalized)


def is_current_sql_source_version(version: str | None) -> bool:
    return normalize_sql_source_version(version) in CURRENT_SQL_SOURCE_VERSIONS


def real_split_provenance(dataset_id: str, split: str = "train") -> dict[str, Any]:
    real_path = resolve_real_split_path(dataset_id, split=split)
    return {
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "real_reference_split": split,
        "real_source_kind": "reference_split_csv",
        "real_source_dataset_id": dataset_id,
        "real_source_split": split,
        **_path_provenance_fields("real_source", real_path),
    }


def resolve_latest_task_run_dir(task_name: str) -> Path | None:
    latest_path = OUTPUT_ROOT / task_name / "LATEST_RUN.json"
    payload = read_json(latest_path, {}) or {}
    run_dir = payload.get("run_dir")
    if not run_dir:
        return None
    candidate = Path(str(run_dir))
    return candidate if candidate.exists() else None


def resolve_requested_sql_source_version(
    task_name: str | None = None,
    default: str = DEFAULT_SQL_SOURCE_VERSION,
) -> str:
    override = str(os.environ.get(SQL_SOURCE_VERSION_ENV_VAR) or "").strip()
    if override:
        return normalize_sql_source_version(override)
    if task_name:
        return resolve_latest_task_sql_source_version(task_name, default=default)
    return normalize_sql_source_version(default)


def resolve_latest_task_sql_source_version(task_name: str, default: str = DEFAULT_SQL_SOURCE_VERSION) -> str:
    run_dir = resolve_latest_task_run_dir(task_name)
    if run_dir is None:
        return normalize_sql_source_version(default)
    manifest = read_json(run_dir / "manifest.json", {}) or {}
    try:
        return normalize_sql_source_version(str(manifest.get("sql_source_version") or default))
    except Exception:
        return normalize_sql_source_version(default)


def resolve_task_run_dir_for_sql_source(
    task_name: str,
    sql_source_version: str | None = None,
    *,
    default: str = DEFAULT_SQL_SOURCE_VERSION,
) -> Path | None:
    requested = resolve_requested_sql_source_version(task_name=task_name, default=default)
    target_version = normalize_sql_source_version(sql_source_version or requested)
    latest_run_dir = resolve_latest_task_run_dir(task_name)
    if latest_run_dir is not None:
        latest_manifest = read_json(latest_run_dir / "manifest.json", {}) or {}
        latest_version = str(latest_manifest.get("sql_source_version") or "").strip()
        if latest_version:
            try:
                if normalize_sql_source_version(latest_version) == target_version:
                    return latest_run_dir
            except Exception:
                pass

    runs_root_dir = OUTPUT_ROOT / task_name / "runs"
    if not runs_root_dir.exists():
        return None

    ranked: list[tuple[int, int, str, Path]] = []
    for candidate in runs_root_dir.iterdir():
        if not candidate.is_dir():
            continue
        manifest_path = candidate / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path, {}) or {}
        manifest_version = str(manifest.get("sql_source_version") or "").strip()
        if not manifest_version:
            continue
        try:
            if normalize_sql_source_version(manifest_version) != target_version:
                continue
        except Exception:
            continue
        ranked.append(
            (
                int(manifest.get("dataset_count") or 0),
                int(manifest.get("asset_count") or 0),
                candidate.name,
                candidate.resolve(),
            )
        )
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][3]


def build_sql_source_provenance(
    *,
    sql_source_version: str,
    sql_source_kind: str,
    sql_source_selection_mode: str,
    source_run_id: str = "",
    sql_file_path: Path | None = None,
    manifest_path: Path | None = None,
    registry_path: Path | None = None,
    run_dir: Path | None = None,
    dataset_dir: Path | None = None,
    registry_version: str = "",
    declared_version: str = "",
    declared_label: str = "",
    sql_file_sha256: str = "",
) -> dict[str, Any]:
    normalized = normalize_sql_source_version(sql_source_version)
    registry_root = sql_source_registry_root(normalized)
    return {
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "sql_source_family": sql_source_family(normalized),
        "sql_source_line_version": sql_source_line_version(normalized),
        "sql_source_version": normalized,
        "sql_source_label": sql_source_label(normalized),
        "sql_source_description": sql_source_description(normalized),
        "sql_source_root": _resolved_path_text(sql_source_root(normalized)),
        "sql_source_registry_root": _resolved_path_text(registry_root),
        "sql_source_kind": sql_source_kind,
        "sql_source_selection_mode": sql_source_selection_mode,
        "sql_source_registry_version": str(registry_version or ""),
        "sql_source_declared_version": str(declared_version or ""),
        "sql_source_declared_label": str(declared_label or ""),
        "sql_source_file_sha256": str(sql_file_sha256 or ""),
        "source_run_id": str(source_run_id or ""),
        "sql_origin_path": _resolved_path_text(sql_file_path),
        **_path_provenance_fields("sql_source_file", sql_file_path),
        **_path_provenance_fields("sql_source_manifest", manifest_path),
        **_path_provenance_fields("sql_source_registry", registry_path),
        **_path_provenance_fields("sql_source_run_dir", run_dir),
        **_path_provenance_fields("sql_source_dataset_dir", dataset_dir),
    }


def _find_local_artifact_by_name(search_root: Path, name: str) -> Path | None:
    if not name:
        return None
    for path in search_root.rglob(name):
        if path.is_file():
            return path
    return None


def _choose_synthetic_csv(candidates: list[Path]) -> Path | None:
    filtered = _list_synthetic_csv_candidates(candidates)
    if not filtered:
        return None
    filtered.sort(key=lambda p: (parse_timestamp_text(p.name) or _stat_mtime_ts(p) or datetime.min.replace(tzinfo=timezone.utc)))
    return filtered[-1]


def _list_synthetic_csv_candidates(candidates: Iterable[Path]) -> list[Path]:
    return [path for path in candidates if _is_synthetic_candidate_csv(path)]


def _is_synthetic_candidate_csv(path: Path) -> bool:
    lname = path.name.lower()
    stem = path.stem.lower()
    if "train_continuous_imputed" in lname:
        return False
    for suffix in ("real", "test", "val", "train"):
        if f"__{suffix}.csv" in lname or lname.endswith(f"_{suffix}.csv") or stem.endswith(f"_{suffix}"):
            return False
    return True


def _synthetic_candidate_sort_key(path: Path) -> datetime:
    return parse_timestamp_text(path.name) or _stat_mtime_ts(path) or datetime.min.replace(tzinfo=timezone.utc)


def _runtime_result_prefix(path: Path) -> str:
    match = RUNTIME_RESULT_RE.match(path.name)
    if match:
        return str(match.group("prefix") or "").strip()
    return path.stem


def _match_runtime_payload_for_synthetic_csv(runtime_files: list[Path], synthetic_csv_path: Path) -> tuple[dict[str, Any], Path | None]:
    synthetic_name = synthetic_csv_path.name
    for runtime_file in sorted(runtime_files, reverse=True):
        prefix = _runtime_result_prefix(runtime_file)
        if prefix and synthetic_name.startswith(prefix):
            return read_json(runtime_file, {}) or {}, runtime_file
    if runtime_files:
        chosen = sorted(runtime_files)[-1]
        return read_json(chosen, {}) or {}, chosen
    return {}, None


def _derive_run_id_for_candidate(runtime_run_id: str, synthetic_csv_path: Path) -> str:
    stem = synthetic_csv_path.stem
    if runtime_run_id and runtime_run_id in stem:
        suffix = stem.split(runtime_run_id, 1)[1].strip("_-")
        if suffix:
            return f"{runtime_run_id}__{suffix}"
        return runtime_run_id
    if runtime_run_id:
        return runtime_run_id
    return stem


def _extract_gpu_seconds_from_logs(log_paths: list[Path], synthetic_csv_path: Path | None = None) -> tuple[float, str]:
    explicit_seconds = 0.0
    saw_explicit = False
    for path in log_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for regex in [TRAIN_TIME_RE, SAMPLE_TIME_RE, GENERIC_SECONDS_RE]:
            for match in regex.findall(text):
                try:
                    explicit_seconds += float(match)
                    saw_explicit = True
                except Exception:
                    continue
    if saw_explicit and explicit_seconds > 0:
        return explicit_seconds, "explicit_log_seconds"

    inferred_seconds = 0.0
    for path in log_paths:
        start_ts = parse_timestamp_text(path.name) or parse_timestamp_text(path.stem)
        end_ts = _stat_mtime_ts(path)
        if start_ts is not None and end_ts is not None:
            delta = (end_ts - start_ts).total_seconds()
            if 0 < delta <= MAX_FALLBACK_GPU_SECONDS:
                inferred_seconds += delta
    if inferred_seconds > 0:
        return inferred_seconds, "log_mtime_fallback"

    if log_paths and synthetic_csv_path is not None and synthetic_csv_path.exists():
        start_candidates = [parse_timestamp_text(path.name) for path in log_paths]
        start_candidates = [item for item in start_candidates if item is not None]
        end_ts = _stat_mtime_ts(synthetic_csv_path)
        if start_candidates and end_ts is not None:
            delta = (end_ts - min(start_candidates)).total_seconds()
            if 0 < delta <= MAX_FALLBACK_GPU_SECONDS:
                return delta, "artifact_mtime_fallback"

    return 0.0, "unavailable_zero"


def _extract_gpu_seconds_from_runtime_payload(runtime_payload: dict[str, Any] | None) -> tuple[float, str] | None:
    if not isinstance(runtime_payload, dict):
        return None
    timings = runtime_payload.get("timings")
    if not isinstance(timings, dict):
        return None
    total_seconds = 0.0
    saw_duration = False
    for stage_name in ("train", "generate"):
        stage_payload = timings.get(stage_name)
        if not isinstance(stage_payload, dict):
            continue
        raw_value = stage_payload.get("duration_sec")
        if raw_value is None:
            continue
        try:
            duration_sec = float(raw_value)
        except Exception:
            continue
        if duration_sec > 0:
            total_seconds += duration_sec
            saw_duration = True
    if saw_duration:
        return total_seconds, "runtime_result_timings"
    return None


def _hyperparameter_tabsyn_is_consistent_batch(env_overrides: dict[str, Any]) -> bool:
    # Accept any successful Tabsyn hyperparameter run that explicitly varies
    # training knobs. Older code only admitted one very specific sweep shape,
    # which filtered out newer smoke/BO runs (e.g. smaller batch sizes).
    keys = {str(k): v for k, v in env_overrides.items()}
    has_batch = any(
        str(keys.get(name) or "").strip()
        for name in (
            "TABSYN_VAE_BATCH_SIZE",
            "TABSYN_DIFFUSION_BATCH_SIZE",
            "TABSYN_VAE_ENCODE_BATCH_SIZE",
            "TABSYN_VAE_EVAL_BATCH_SIZE",
            "TABSYN_VAE_INFER_BATCH_SIZE",
        )
    )
    has_epoch = any(
        str(keys.get(name) or "").strip()
        for name in (
            "TABSYN_VAE_EPOCHS",
            "TABSYN_DIFFUSION_MAX_EPOCHS",
        )
    )
    if not (has_batch and has_epoch):
        return False
    num_workers = str(keys.get("TABSYN_VAE_NUM_WORKERS") or "").strip()
    if num_workers and num_workers != "0":
        return False
    return True


def _should_keep_hyperparameter_run(*, model_id: str, run_config_payload: dict[str, Any], runtime_payload: dict[str, Any]) -> bool:
    if str(runtime_payload.get("train_status") or "").strip().lower() != "success":
        return False
    if str(runtime_payload.get("generate_status") or "").strip().lower() != "success":
        return False
    env_overrides = run_config_payload.get("env_overrides")
    if not isinstance(env_overrides, dict) or not env_overrides:
        return False
    if str(model_id or "").strip().lower() == "tabsyn":
        if _hyperparameter_tabsyn_is_consistent_batch(env_overrides):
            return True
        cli_args = run_config_payload.get("cli_args")
        cli_args = cli_args if isinstance(cli_args, dict) else {}
        has_epoch_signal = bool(str(cli_args.get("epochs") or "").strip()) or any(
            str(env_overrides.get(name) or "").strip()
            for name in ("TABSYN_VAE_EPOCHS", "TABSYN_DIFFUSION_MAX_EPOCHS")
        )
        has_batch_signal = any(
            str(env_overrides.get(name) or "").strip()
            for name in (
                "TABSYN_VAE_BATCH_SIZE",
                "TABSYN_DIFFUSION_BATCH_SIZE",
                "TABSYN_VAE_ENCODE_BATCH_SIZE",
                "TABSYN_VAE_EVAL_BATCH_SIZE",
                "TABSYN_VAE_INFER_BATCH_SIZE",
            )
        )
        return has_epoch_signal and has_batch_signal
    return True


def _has_substantive_hyperparameter_overrides(env_overrides: dict[str, Any]) -> bool:
    for key, value in env_overrides.items():
        if str(key).startswith("BENCHMARK_"):
            continue
        if value is None:
            continue
        if str(value).strip():
            return True
    return False


def _build_asset(
    *,
    dataset_id: str,
    model_id: str,
    root_name: str,
    asset_dir: Path,
    run_id: str,
    synthetic_csv_path: Path,
    metadata_paths: list[Path],
    log_paths: list[Path],
    discovered_via: str,
    runtime_payload: dict[str, Any] | None = None,
) -> SyntheticAsset:
    cfg = ROOT_CONFIGS[root_name]
    timestamp_candidates = []
    timestamp_candidates.extend(_candidate_timestamps(run_id, synthetic_csv_path.name))
    timestamp_candidates.extend(item for item in (_stat_mtime_ts(synthetic_csv_path), _stat_mtime_ts(asset_dir)) if item is not None)
    timestamp = max(timestamp_candidates) if timestamp_candidates else None
    runtime_timing = _extract_gpu_seconds_from_runtime_payload(runtime_payload)
    if runtime_timing is not None:
        gpu_seconds_raw, gpu_source = runtime_timing
    else:
        gpu_seconds_raw, gpu_source = _extract_gpu_seconds_from_logs(log_paths, synthetic_csv_path)
    gpu_hours_equivalent = (gpu_seconds_raw / 3600.0) * float(cfg["gpu_hour_ratio"])
    return SyntheticAsset(
        dataset_id=dataset_id,
        model_id=model_id,
        server_type=str(cfg["server_type"]),
        root_name=root_name,
        root_path=str(Path(cfg["path"]).resolve()),
        asset_dir=str(asset_dir.resolve()),
        run_id=run_id,
        synthetic_csv_path=str(synthetic_csv_path.resolve()),
        metadata_paths=[str(path.resolve()) for path in metadata_paths],
        log_paths=[str(path.resolve()) for path in log_paths],
        discovered_via=discovered_via,
        timestamp_utc=(timestamp.isoformat() if timestamp is not None else None),
        synthetic_source_mtime_utc=(_stat_mtime_ts(synthetic_csv_path).isoformat() if _stat_mtime_ts(synthetic_csv_path) is not None else None),
        synthetic_source_size_bytes=_stat_size_bytes(synthetic_csv_path),
        gpu_seconds_raw=round(gpu_seconds_raw, 6),
        gpu_hours_equivalent=round(gpu_hours_equivalent, 6),
        gpu_hours_source=gpu_source,
        cost_usd=round(gpu_hours_equivalent * USD_PER_GPU_HOUR, 6),
    )


def _discover_assets_in_synoutput(dataset_id: str, root_name: str) -> list[SyntheticAsset]:
    root = Path(ROOT_CONFIGS[root_name]["path"])
    dataset_root = root / dataset_id
    if not dataset_root.exists():
        return []
    assets: list[SyntheticAsset] = []
    for model_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        model_id = model_dir.name
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = read_json(manifest_path, {}) or {}
            runtime_result = manifest.get("runtime_result") if isinstance(manifest, dict) else {}
            artifacts = runtime_result.get("artifacts") if isinstance(runtime_result, dict) else {}
            desired_name = Path(str(artifacts.get("synthetic_csv") or "")).name
            synthetic_csv_path = _find_local_artifact_by_name(run_dir, desired_name) if desired_name else None
            if synthetic_csv_path is None:
                synthetic_csv_path = _choose_synthetic_csv(list((run_dir / "synthetic").glob("*.csv")))
            if synthetic_csv_path is None:
                continue
            run_id = str(runtime_result.get("run_id") or manifest.get("run_id") or run_dir.name)
            log_paths = sorted((run_dir / "logs").glob("*.log"))
            metadata_paths = [manifest_path] + sorted((run_dir / "meta").glob("*.json"))
            assets.append(
                _build_asset(
                    dataset_id=dataset_id,
                    model_id=model_id,
                    root_name=root_name,
                    asset_dir=run_dir,
                    run_id=run_id,
                    synthetic_csv_path=synthetic_csv_path,
                    metadata_paths=metadata_paths,
                    log_paths=log_paths,
                    discovered_via="manifest_json",
                )
            )
    return assets


def _discover_assets_in_synoutput_5090(dataset_id: str, root_name: str) -> list[SyntheticAsset]:
    root = Path(ROOT_CONFIGS[root_name]["path"])
    dataset_root = root / dataset_id
    if not dataset_root.exists():
        return []
    assets: list[SyntheticAsset] = []
    for model_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        model_id = model_dir.name
        runtime_files = sorted((model_dir / "metadata").glob("*__runtime_result.json"))
        synthetic_candidates = sorted(
            _list_synthetic_csv_candidates((model_dir / "synthetic_data").glob("*.csv")),
            key=_synthetic_candidate_sort_key,
        )
        if not synthetic_candidates:
            continue
        metadata_paths_all = sorted((model_dir / "metadata").glob("*.json"))
        log_paths = sorted((model_dir / "logs").glob("*.log"))

        for synthetic_csv_path in synthetic_candidates:
            runtime_payload, matched_runtime = _match_runtime_payload_for_synthetic_csv(runtime_files, synthetic_csv_path)
            runtime_run_id = str(runtime_payload.get("run_id") or model_dir.name)
            run_id = _derive_run_id_for_candidate(runtime_run_id, synthetic_csv_path)
            metadata_paths = list(metadata_paths_all)
            if matched_runtime is not None and matched_runtime not in metadata_paths:
                metadata_paths = [matched_runtime] + metadata_paths
            assets.append(
                _build_asset(
                    dataset_id=dataset_id,
                    model_id=model_id,
                    root_name=root_name,
                    asset_dir=model_dir,
                    run_id=run_id,
                    synthetic_csv_path=synthetic_csv_path,
                    metadata_paths=metadata_paths,
                    log_paths=log_paths,
                    discovered_via=("runtime_result_json_matched" if matched_runtime is not None else "synthetic_csv_scan"),
                )
            )
    return assets


def _discover_assets_in_trainonly_root(dataset_id: str, root_name: str) -> list[SyntheticAsset]:
    root = Path(ROOT_CONFIGS[root_name]["path"])
    dataset_root = root / dataset_id
    if not dataset_root.exists():
        return []
    assets: list[SyntheticAsset] = []
    for model_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        model_id = model_dir.name
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            runtime_path = run_dir / "runtime_result.json"
            runtime_payload = read_json(runtime_path, {}) or {}
            if not isinstance(runtime_payload, dict):
                continue
            artifacts = runtime_payload.get("artifacts") if isinstance(runtime_payload.get("artifacts"), dict) else {}
            desired_name = Path(str(artifacts.get("synthetic_csv") or "")).name
            candidate_files = list(run_dir.glob("*.csv"))
            synthetic_csv_path = _find_local_artifact_by_name(run_dir, desired_name) if desired_name else None
            if synthetic_csv_path is None:
                synthetic_csv_path = _choose_synthetic_csv(candidate_files)
            if synthetic_csv_path is None:
                continue

            run_id = str(runtime_payload.get("run_id") or run_dir.name)
            log_paths = sorted(run_dir.glob("*.log"))
            metadata_paths = [runtime_path] if runtime_path.exists() else []
            for extra in [
                run_dir / "input_snapshot.json",
                run_dir / "run_config.json",
                run_dir / "public_gate" / "public_gate_report.json",
                run_dir / "public_gate" / "normalized_schema_snapshot.json",
                run_dir / "public_gate" / "staged_input_manifest.json",
            ]:
                if extra.exists() and extra not in metadata_paths:
                    metadata_paths.append(extra)
            assets.append(
                _build_asset(
                    dataset_id=dataset_id,
                    model_id=model_id,
                    root_name=root_name,
                    asset_dir=run_dir,
                    run_id=run_id,
                    synthetic_csv_path=synthetic_csv_path,
                    metadata_paths=metadata_paths,
                    log_paths=log_paths,
                    discovered_via="runtime_result_json",
                    runtime_payload=runtime_payload,
                )
            )
    return assets


def _discover_assets_in_hyperparameter_trainonly_root(dataset_id: str, root_name: str) -> list[SyntheticAsset]:
    root = Path(ROOT_CONFIGS[root_name]["path"])
    dataset_root = root / dataset_id
    if not dataset_root.exists():
        return []
    assets: list[SyntheticAsset] = []
    for model_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        model_id = model_dir.name
        candidate_runs: list[tuple[Path, dict[str, Any], dict[str, Any], bool]] = []
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            runtime_path = run_dir / "runtime_result.json"
            run_config_path = run_dir / "run_config.json"
            runtime_payload = read_json(runtime_path, {}) or {}
            run_config_payload = read_json(run_config_path, {}) or {}
            if not isinstance(runtime_payload, dict) or not isinstance(run_config_payload, dict):
                continue
            if not _should_keep_hyperparameter_run(
                model_id=model_id,
                run_config_payload=run_config_payload,
                runtime_payload=runtime_payload,
            ):
                continue
            env_overrides = run_config_payload.get("env_overrides")
            env_overrides = env_overrides if isinstance(env_overrides, dict) else {}
            candidate_runs.append(
                (
                    run_dir,
                    runtime_payload,
                    run_config_payload,
                    _has_substantive_hyperparameter_overrides(env_overrides),
                )
            )

        if not candidate_runs:
            continue
        keep_only_substantive = any(item[3] for item in candidate_runs)
        for run_dir, runtime_payload, run_config_payload, has_substantive_overrides in candidate_runs:
            if keep_only_substantive and not has_substantive_overrides:
                continue
            artifacts = runtime_payload.get("artifacts") if isinstance(runtime_payload.get("artifacts"), dict) else {}
            desired_name = Path(str(artifacts.get("synthetic_csv") or "")).name
            candidate_files = list(run_dir.glob("*.csv"))
            synthetic_csv_path = _find_local_artifact_by_name(run_dir, desired_name) if desired_name else None
            if synthetic_csv_path is None:
                synthetic_csv_path = _choose_synthetic_csv(candidate_files)
            if synthetic_csv_path is None:
                continue

            run_id = str(runtime_payload.get("run_id") or run_dir.name)
            log_paths = sorted(run_dir.glob("*.log"))
            metadata_paths = [runtime_path] if runtime_path.exists() else []
            for extra in [
                run_config_path,
                run_dir / "input_snapshot.json",
                run_dir / "public_gate" / "public_gate_report.json",
                run_dir / "public_gate" / "normalized_schema_snapshot.json",
                run_dir / "public_gate" / "staged_input_manifest.json",
            ]:
                if extra.exists() and extra not in metadata_paths:
                    metadata_paths.append(extra)
            assets.append(
                _build_asset(
                    dataset_id=dataset_id,
                    model_id=model_id,
                    root_name=root_name,
                    asset_dir=run_dir,
                    run_id=run_id,
                    synthetic_csv_path=synthetic_csv_path,
                    metadata_paths=metadata_paths,
                    log_paths=log_paths,
                    discovered_via="runtime_result_json_hyperparameter",
                    runtime_payload=runtime_payload,
                )
            )
    return assets


def discover_synthetic_assets(
    *,
    datasets: list[str] | None = None,
    latest_only: bool = True,
    root_names: list[str] | tuple[str, ...] | None = None,
) -> list[SyntheticAsset]:
    dataset_ids = datasets or list_dataset_ids()
    requested_roots = [str(item).strip() for item in (root_names or []) if str(item).strip()]
    if requested_roots:
        invalid = sorted(set(requested_roots) - set(ROOT_CONFIGS.keys()))
        if invalid:
            raise ValueError(f"Unsupported synthetic root names: {invalid}. Available: {sorted(ROOT_CONFIGS.keys())}")
    active_roots = requested_roots or list(ROOT_CONFIGS.keys())
    assets: list[SyntheticAsset] = []
    for dataset_id in dataset_ids:
        for root_name in active_roots:
            if root_name == "SynOutput":
                assets.extend(_discover_assets_in_synoutput(dataset_id, root_name))
            elif root_name == "SynOutput-5090":
                assets.extend(_discover_assets_in_synoutput_5090(dataset_id, root_name))
            elif root_name == "Benchmark-trainonly-v1":
                assets.extend(_discover_assets_in_trainonly_root(dataset_id, root_name))
            elif root_name == "Hyperparameter-trainonly-v1":
                assets.extend(_discover_assets_in_hyperparameter_trainonly_root(dataset_id, root_name))
            elif root_name == "TabQueryBench-SynDataSuccess-main":
                assets.extend(_discover_assets_in_trainonly_root(dataset_id, root_name))
    if not latest_only:
        return sorted(assets, key=lambda item: (item.dataset_id, item.server_type, item.model_id, item.timestamp_utc or ""))

    latest_map: dict[tuple[str, str, str], SyntheticAsset] = {}
    for asset in assets:
        key = (asset.dataset_id, asset.server_type, asset.model_id)
        current = latest_map.get(key)
        asset_ts = parse_timestamp_text(asset.timestamp_utc or "")
        current_ts = parse_timestamp_text(current.timestamp_utc or "") if current else None
        if current is None or ((asset_ts or datetime.min.replace(tzinfo=timezone.utc)) >= (current_ts or datetime.min.replace(tzinfo=timezone.utc))):
            latest_map[key] = asset
    return sorted(latest_map.values(), key=lambda item: (item.dataset_id, item.server_type, item.model_id))


def split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    prev = ""
    for ch in sql_text:
        if ch == "'" and not in_double and prev != "\\":
            in_single = not in_single
        elif ch == '"' and not in_single and prev != "\\":
            in_double = not in_double
        if ch == ";" and not in_single and not in_double:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
        prev = ch
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    cleaned = []
    for stmt in statements:
        lines = [line for line in stmt.splitlines() if not line.strip().startswith("--")]
        candidate = "\n".join(lines).strip()
        if candidate:
            cleaned.append(candidate)
    return cleaned


def normalize_sql_source_version(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return DEFAULT_SQL_SOURCE_VERSION
    match = re.search(r"(v[1-4])", text)
    if match and match.group(1) in SQL_SOURCE_VERSION_CHOICES:
        candidate = match.group(1)
        if candidate == SQL_SOURCE_VERSION_V1 and "subitem_workload" in text:
            candidate = ""
        if candidate:
            return candidate
    version = _SQL_SOURCE_ALIASES.get(text)
    if version is None:
        raise ValueError(
            f"Unsupported sql source version: {value!r}. Expected one of: {', '.join(SQL_SOURCE_VERSION_CHOICES)}"
        )
    return version


def sql_source_label(version: str | None) -> str:
    normalized = normalize_sql_source_version(version)
    return _SQL_SOURCE_LABELS[normalized]


def sql_source_description(version: str | None) -> str:
    normalized = normalize_sql_source_version(version)
    return _SQL_SOURCE_DESCRIPTIONS[normalized]


def sql_source_root(version: str | None) -> Path:
    normalized = normalize_sql_source_version(version)
    if normalized == SQL_SOURCE_VERSION_V1:
        return LOGS_ROOT
    if normalized in CURRENT_SQL_SOURCE_VERSIONS:
        return runs_root(normalized)
    raise ValueError(f"Unsupported sql source version: {version!r}")


def resolve_sql_run_dir(*, sql_source_version: str, run_id: str, dataset_id: str | None = None) -> Path:
    normalized = normalize_sql_source_version(sql_source_version)
    if normalized == SQL_SOURCE_VERSION_V1:
        return LOGS_ROOT / run_id
    if not dataset_id:
        raise ValueError("dataset_id is required when resolving a current workload run directory.")
    return runs_root(normalized) / run_id / dataset_id


def _load_latest_v1_sql_query_groups(
    *,
    dataset_ids: Iterable[str] | None = None,
    engines: tuple[str, ...] = ("cli",),
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    if not LOGS_ROOT.exists():
        return grouped

    dataset_filter = {str(item).strip() for item in dataset_ids or [] if str(item).strip()}
    for manifest_path in LOGS_ROOT.rglob("run_manifest.json"):
        payload = read_json(manifest_path, {}) or {}
        if str(payload.get("status") or "") != "completed":
            continue
        if str(payload.get("mode") or "") != "template_grounded_sql_qa":
            continue
        dataset_id = str(payload.get("dataset_id") or "").strip()
        if not dataset_id:
            continue
        if dataset_filter and dataset_id not in dataset_filter:
            continue
        engine = str(payload.get("engine") or "").strip()
        if engines and engine not in engines:
            continue
        question_record = payload.get("question_record")
        if not isinstance(question_record, dict):
            continue
        question_id = str(question_record.get("question_id") or "").strip()
        if not question_id:
            continue
        sql_path = manifest_path.parent / "generated_sql.sql"
        if not sql_path.exists():
            continue
        ended_at = str(payload.get("ended_at") or payload.get("started_at") or "")
        key = (dataset_id, question_id)
        current = grouped.get(key)
        if current is None:
            grouped[key] = {
                "payload": payload,
                "sql_path": sql_path,
                "sort_dt": parse_timestamp_text(ended_at) or _stat_mtime_ts(sql_path) or datetime.min.replace(tzinfo=timezone.utc),
                "manifest_path": manifest_path,
            }
            continue
        new_dt = parse_timestamp_text(ended_at) or _stat_mtime_ts(sql_path) or datetime.min.replace(tzinfo=timezone.utc)
        if new_dt >= current.get("sort_dt", datetime.min.replace(tzinfo=timezone.utc)):
            grouped[key] = {
                "payload": payload,
                "sql_path": sql_path,
                "sort_dt": new_dt,
                "manifest_path": manifest_path,
            }
    return grouped


def _current_query_manifest_path(
    *,
    run_id: str,
    dataset_id: str,
    query_record_id: str,
    sql_source_version: str,
) -> Path:
    normalized = normalize_line_version(sql_source_version)
    return run_manifest_dir(run_id, dataset_id, line_version=normalized) / query_record_id / "run_manifest.json"


def _load_latest_current_sql_query_groups(
    *,
    sql_source_version: str,
    dataset_ids: Iterable[str] | None = None,
    engines: tuple[str, ...] = ("cli",),
    require_accepted_for_eval: bool = True,
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    normalized = normalize_sql_source_version(sql_source_version)
    registry_root = registry_dir(normalized)
    if not registry_root.exists():
        return grouped

    dataset_filter = {str(item).strip() for item in dataset_ids or [] if str(item).strip()}
    for registry_path in sorted(registry_root.glob(f"*_query_registry_{normalized}.jsonl")):
        for row in load_registry_rows(registry_path):
            dataset_id = str(row.get("dataset_id") or "").strip()
            if not dataset_id:
                continue
            if dataset_filter and dataset_id not in dataset_filter:
                continue
            engine = str(row.get("engine") or "").strip()
            if engines and engine not in engines:
                continue
            if require_accepted_for_eval and not bool(row.get("accepted_for_eval")):
                continue
            query_record_id = str(row.get("query_record_id") or "").strip()
            if not query_record_id:
                continue
            sql_path = _resolve_registry_backed_path(row.get("sql_path"))
            if not sql_path.exists():
                continue
            run_id = str(row.get("round_id") or "").strip()
            manifest_path = _current_query_manifest_path(
                run_id=run_id,
                dataset_id=dataset_id,
                query_record_id=query_record_id,
                sql_source_version=normalized,
            )
            manifest = read_json(manifest_path, {}) or {}
            sort_dt = (
                parse_timestamp_text(str(manifest.get("ended_at") or manifest.get("started_at") or ""))
                or _stat_mtime_ts(sql_path)
                or _stat_mtime_ts(manifest_path)
                or _stat_mtime_ts(registry_path)
                or datetime.min.replace(tzinfo=timezone.utc)
            )
            key = (dataset_id, query_record_id)
            current = grouped.get(key)
            if current is None or sort_dt >= current.get("sort_dt", datetime.min.replace(tzinfo=timezone.utc)):
                grouped[key] = {
                    "row": row,
                    "sql_path": sql_path,
                    "registry_path": registry_path,
                    "manifest_path": manifest_path,
                    "manifest": manifest,
                    "sql_source_version": normalized,
                    "sort_dt": sort_dt,
                }
    return grouped


def load_latest_sql_queries_by_dataset(
    *,
    dataset_ids: Iterable[str],
    engines: tuple[str, ...] = ("cli",),
    include_all_statements: bool = True,
    sql_source_version: str = DEFAULT_SQL_SOURCE_VERSION,
) -> dict[str, list[dict[str, Any]]]:
    dataset_ids = [str(item).strip() for item in dataset_ids if str(item).strip()]
    normalized_source = normalize_sql_source_version(sql_source_version)
    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if normalized_source == SQL_SOURCE_VERSION_V1:
        grouped = _load_latest_v1_sql_query_groups(dataset_ids=dataset_ids, engines=engines)
        for (dataset_id, question_id), item in sorted(grouped.items()):
            payload = item["payload"]
            sql_text = item["sql_path"].read_text(encoding="utf-8", errors="ignore")
            sql_file_hash = _sha256_text(sql_text)
            statements = split_sql_statements(sql_text)
            if not statements:
                continue
            if not include_all_statements:
                statements = statements[:1]
            question_record = payload.get("question_record") or {}
            provenance = build_sql_source_provenance(
                sql_source_version=SQL_SOURCE_VERSION_V1,
                sql_source_kind="legacy_grounded_run_manifest",
                sql_source_selection_mode="latest_per_question_id",
                source_run_id=str(payload.get("run_id") or ""),
                sql_file_path=item["sql_path"],
                manifest_path=item["manifest_path"],
                run_dir=item["manifest_path"].parent,
                declared_version=str(payload.get("sql_source_version") or ""),
                declared_label=str(payload.get("sql_source_label") or ""),
                sql_file_sha256=sql_file_hash,
            )
            for idx, statement in enumerate(statements, start=1):
                rows_by_dataset[dataset_id].append(
                    {
                        "dataset_id": dataset_id,
                        "question_id": question_id,
                        "query_id": f"{question_id}__sql{idx}",
                        "sql_index": idx,
                        "question": str(payload.get("question") or question_record.get("question") or ""),
                        "template_id": str(question_record.get("template_id") or ""),
                        "template_name": str(question_record.get("template_name") or ""),
                        "family_id": str(question_record.get("primary_family") or ""),
                        "canonical_subitem_id": str(question_record.get("canonical_subitem_id") or ""),
                        "intended_facet_id": str(question_record.get("intended_facet_id") or ""),
                        "variant_semantic_role": str(question_record.get("variant_semantic_role") or ""),
                        "stable_question_id": str(question_record.get("stable_question_id") or ""),
                        "query_identity_stable_key": str(question_record.get("query_identity_stable_key") or ""),
                        "source_run_id": str(payload.get("run_id") or ""),
                        "engine": str(payload.get("engine") or ""),
                        "model": str(payload.get("model") or ""),
                        "sql": statement,
                        **provenance,
                    }
                )
    else:
        grouped = _load_latest_current_sql_query_groups(
            sql_source_version=normalized_source,
            dataset_ids=dataset_ids,
            engines=engines,
            require_accepted_for_eval=True,
        )
        for (dataset_id, query_record_id), item in sorted(grouped.items()):
            row = item["row"]
            manifest = item["manifest"] if isinstance(item.get("manifest"), dict) else {}
            question_record = manifest.get("question_record") if isinstance(manifest, dict) else {}
            sql_text = item["sql_path"].read_text(encoding="utf-8", errors="ignore")
            sql_file_hash = str(row.get("sql_sha256") or "") or _sha256_text(sql_text)
            statements = split_sql_statements(sql_text)
            if not statements:
                continue
            if not include_all_statements:
                statements = statements[:1]
            declared_version = str(row.get("sql_source_version") or manifest.get("sql_source_version") or "")
            declared_label = str(row.get("sql_source_label") or manifest.get("sql_source_label") or "")
            run_id = str(row.get("round_id") or "")
            current_runs_root = runs_root(normalized_source)
            run_root = current_runs_root / run_id
            dataset_dir = run_root / dataset_id
            provenance = build_sql_source_provenance(
                sql_source_version=normalized_source,
                sql_source_kind="current_query_registry",
                sql_source_selection_mode="latest_per_query_record_id",
                source_run_id=run_id,
                sql_file_path=item["sql_path"],
                manifest_path=item["manifest_path"],
                registry_path=item["registry_path"],
                run_dir=run_root,
                dataset_dir=dataset_dir,
                registry_version=str(row.get("registry_version") or ""),
                declared_version=declared_version,
                declared_label=declared_label,
                sql_file_sha256=sql_file_hash,
            )
            for idx, statement in enumerate(statements, start=1):
                query_id = query_record_id if len(statements) == 1 else f"{query_record_id}__sql{idx}"
                rows_by_dataset[dataset_id].append(
                    {
                        "dataset_id": dataset_id,
                        "question_id": query_record_id,
                        "query_id": query_id,
                        "sql_index": idx,
                        "question": str(row.get("question_text") or question_record.get("question") or ""),
                        "template_id": str(row.get("template_id") or question_record.get("template_id") or ""),
                        "template_name": str(row.get("template_name") or question_record.get("template_name") or ""),
                        "family_id": str(row.get("family_id") or question_record.get("family_id") or ""),
                        "canonical_subitem_id": str(row.get("canonical_subitem_id") or question_record.get("canonical_subitem_id") or ""),
                        "intended_facet_id": str(row.get("intended_facet_id") or question_record.get("intended_facet_id") or ""),
                        "variant_semantic_role": str(row.get("variant_semantic_role") or question_record.get("variant_semantic_role") or ""),
                        "stable_question_id": query_record_id,
                        "query_identity_stable_key": str(row.get("query_identity_stable_key") or f"{dataset_id}::{query_record_id}"),
                        "source_run_id": run_id,
                        "engine": str(row.get("engine") or manifest.get("engine") or ""),
                        "model": str(manifest.get("model") or ""),
                        "sql": statement,
                        "accepted_for_eval": bool(row.get("accepted_for_eval")),
                        **provenance,
                    }
                )
    return {dataset_id: rows_by_dataset.get(dataset_id, []) for dataset_id in dataset_ids}


def load_latest_sql_queries(
    *,
    dataset_id: str,
    engines: tuple[str, ...] = ("cli",),
    include_all_statements: bool = True,
    sql_source_version: str = DEFAULT_SQL_SOURCE_VERSION,
) -> list[dict[str, Any]]:
    return load_latest_sql_queries_by_dataset(
        dataset_ids=[dataset_id],
        engines=engines,
        include_all_statements=include_all_statements,
        sql_source_version=sql_source_version,
    ).get(dataset_id, [])


def materialize_csv_to_sqlite(csv_path: Path, sqlite_path: Path, table_name: str) -> None:
    if sqlite_path.exists():
        sqlite_path.unlink()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    def _sqlite_ident(name: str) -> str:
        return f'"{str(name).replace("\"", "\"\"")}"'

    def _sniff_delimiter(path: Path) -> str:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                sample = handle.read(4096)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except Exception:
            return ","

    def _repair_single_field_row(row: list[str], delimiter: str) -> list[str]:
        if len(row) != 1:
            return row
        cell = str(row[0] or "")
        if delimiter not in cell:
            return row
        repaired = cell.strip()
        if repaired.startswith('"') and repaired.endswith('"') and len(repaired) >= 2:
            repaired = repaired[1:-1]
        repaired = repaired.replace('""', '"')
        try:
            return next(csv.reader([repaired], delimiter=delimiter))
        except Exception:
            return repaired.split(delimiter)

    def _infer_header_from_synthetic(dataset_id: str, width: int) -> list[str] | None:
        try:
            assets = discover_synthetic_assets(
                datasets=[dataset_id],
                root_names=["TabQueryBench-SynDataSuccess-main"],
            )
        except Exception:
            return None
        for asset in assets:
            synthetic_path = Path(asset.synthetic_csv_path)
            if not synthetic_path.exists():
                continue
            try:
                delimiter = _sniff_delimiter(synthetic_path)
                with synthetic_path.open("r", encoding="utf-8-sig", newline="") as synthetic_file:
                    synthetic_reader = csv.reader(synthetic_file, delimiter=delimiter)
                    synthetic_headers = next(synthetic_reader, [])
            except Exception:
                continue
            normalized = [str(header or "").strip() for header in synthetic_headers]
            if len(normalized) == width and all(normalized):
                return normalized
        return None

    def _normalize_headers(first_row: list[str]) -> tuple[list[str], bool]:
        cleaned = [str(header or "").strip() for header in first_row]
        counts = Counter(cleaned)
        has_duplicates = any(name and count > 1 for name, count in counts.items())
        has_empty = any(not name for name in cleaned)
        if has_duplicates or has_empty:
            inferred = _infer_header_from_synthetic(table_name, len(first_row))
            if inferred:
                return inferred, True
            return [f"col_{idx}" for idx in range(1, len(first_row) + 1)], True
        return cleaned, False

    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.cursor()
        delimiter = _sniff_delimiter(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            first_row = _repair_single_field_row(next(reader, []), delimiter)
            if not first_row:
                raise ValueError(f"Empty header: {csv_path}")
            headers, headerless = _normalize_headers(first_row)
            col_defs = ", ".join([f"{_sqlite_ident(header)} TEXT" for header in headers])
            cur.execute(f"DROP TABLE IF EXISTS {_sqlite_ident(table_name)}")
            cur.execute(f"CREATE TABLE {_sqlite_ident(table_name)} ({col_defs})")
            placeholders = ",".join(["?" for _ in headers])
            insert_sql = f"INSERT INTO {_sqlite_ident(table_name)} VALUES ({placeholders})"
            batch: list[list[str]] = []
            if headerless:
                row = list(first_row)
                if len(row) < len(headers):
                    row = row + [""] * (len(headers) - len(row))
                elif len(row) > len(headers):
                    row = row[: len(headers)]
                batch.append(row)
            for row in reader:
                row = _repair_single_field_row(row, delimiter)
                if len(row) < len(headers):
                    row = row + [""] * (len(headers) - len(row))
                elif len(row) > len(headers):
                    row = row[: len(headers)]
                batch.append(row)
                if len(batch) >= 1000:
                    cur.executemany(insert_sql, batch)
                    batch.clear()
            if batch:
                cur.executemany(insert_sql, batch)
        conn.commit()
    finally:
        conn.close()


def normalize_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_TEXT


def mean_or_none(values: Iterable[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)
