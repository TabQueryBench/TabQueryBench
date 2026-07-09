"""Filesystem layout helpers for versioned subitem workload lines."""

from __future__ import annotations

from pathlib import Path

from src.config.settings import LOGS_DIR, PROJECT_ROOT


SUPPORTED_LINE_VERSIONS = ("v2", "v3", "v4")
DEFAULT_V3_V4_DATASETS = ("c2", "c7", "c14", "m4", "m6", "m8", "n3", "n6", "n11")


def normalize_line_version(line_version: str) -> str:
    version = (line_version or "v2").strip().lower()
    if version not in SUPPORTED_LINE_VERSIONS:
        raise ValueError(f"Unsupported line version: {line_version}")
    return version


def workload_data_root(line_version: str = "v2") -> Path:
    version = normalize_line_version(line_version)
    return PROJECT_ROOT / "data" / f"workload_grounding_{version}"


def contracts_dir(line_version: str = "v2") -> Path:
    return workload_data_root(line_version) / "contracts"


def inventory_dir(line_version: str = "v2") -> Path:
    return workload_data_root(line_version) / "inventories"


def portability_dir(line_version: str = "v2") -> Path:
    return workload_data_root(line_version) / "portability"


def registry_dir(line_version: str = "v2") -> Path:
    return workload_data_root(line_version) / "registries"


def rerun_dir(line_version: str = "v2") -> Path:
    return workload_data_root(line_version) / "reruns"


def logs_root(line_version: str = "v2") -> Path:
    version = normalize_line_version(line_version)
    return LOGS_DIR / f"subitem_workload_{version}"


def runs_root(line_version: str = "v2") -> Path:
    return logs_root(line_version) / "runs"


def evaluation_root(line_version: str = "v2") -> Path:
    version = normalize_line_version(line_version)
    return PROJECT_ROOT / "Evaluation" / f"subitem_workload_{version}"


def evaluation_final_dir(line_version: str = "v2") -> Path:
    return evaluation_root(line_version) / "final"


def template_library_path(line_version: str = "v2") -> Path:
    version = normalize_line_version(line_version)
    return workload_data_root(version) / f"template_library_{version}.jsonl"


def ensure_line_dirs(line_version: str = "v2") -> None:
    for path in (
        workload_data_root(line_version),
        contracts_dir(line_version),
        inventory_dir(line_version),
        portability_dir(line_version),
        registry_dir(line_version),
        rerun_dir(line_version),
        logs_root(line_version),
        runs_root(line_version),
        evaluation_root(line_version),
        evaluation_final_dir(line_version),
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_v2_dirs() -> None:
    ensure_line_dirs("v2")


def dataset_inventory_path(dataset_id: str, line_version: str = "v2") -> Path:
    version = normalize_line_version(line_version)
    return inventory_dir(version) / f"{dataset_id}_inventory_{version}.json"


def combined_inventory_path(line_version: str = "v2") -> Path:
    version = normalize_line_version(line_version)
    return inventory_dir(version) / f"inventory_summary_{version}.json"


def registry_jsonl_path(run_id: str, line_version: str = "v2") -> Path:
    version = normalize_line_version(line_version)
    return registry_dir(version) / f"{run_id}_query_registry_{version}.jsonl"


def registry_csv_path(run_id: str, line_version: str = "v2") -> Path:
    version = normalize_line_version(line_version)
    return registry_dir(version) / f"{run_id}_query_registry_{version}.csv"


def run_sql_dir(run_id: str, dataset_id: str, line_version: str = "v2") -> Path:
    return runs_root(line_version) / run_id / dataset_id / "sql"


def run_manifest_dir(run_id: str, dataset_id: str, line_version: str = "v2") -> Path:
    return runs_root(line_version) / run_id / dataset_id / "artifacts"


def default_dataset_ids_for_line_version(line_version: str) -> tuple[str, ...]:
    version = normalize_line_version(line_version)
    if version in {"v3", "v4"}:
        return DEFAULT_V3_V4_DATASETS
    return ("c2", "m4", "n1")


V2_DATA_ROOT = workload_data_root("v2")
V2_CONTRACTS_DIR = contracts_dir("v2")
V2_INVENTORY_DIR = inventory_dir("v2")
V2_PORTABILITY_DIR = portability_dir("v2")
V2_REGISTRY_DIR = registry_dir("v2")
V2_LOGS_ROOT = logs_root("v2")
V2_RUNS_ROOT = runs_root("v2")
V2_EVALUATION_ROOT = evaluation_root("v2")
V2_EVALUATION_FINAL_DIR = evaluation_final_dir("v2")
