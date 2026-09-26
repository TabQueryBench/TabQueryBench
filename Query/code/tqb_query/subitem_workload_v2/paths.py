"""Filesystem layout helpers for versioned subitem workload lines."""

from __future__ import annotations

import os
from pathlib import Path

from tqb_query.config.settings import LOGS_DIR, PROJECT_ROOT, TABQUERYBENCH_ROOT
from tqb_query.workload_grounding.v10_versions import GROUNDING_VERSION_RE


SUPPORTED_LINE_VERSIONS = ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11")
# Model-specific artifact versions of the model-grounded lines (V10 bind-only, V11 select+bind),
# e.g. v11.2.2_glm-5.3; see workload_grounding/v10_versions.py.
MODEL_GROUNDING_VERSION_RE = GROUNDING_VERSION_RE
DEFAULT_V3_V4_DATASETS = ("c2", "c7", "c14", "m4", "m6", "m8", "n3", "n6", "n11")
DEFAULT_ALL_DATASETS = (
    "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11",
    "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19",
    "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10", "m11",
    "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11",
    "n12", "n13", "n14", "n15", "n16", "n17", "n18", "n19",
)


TABQUERYBENCH_LAYOUT = {
    "v2": "v2_current",
    "v3": "v3_experimental",
    "v4": "v4_experimental",
    "v5": "v5_experimental",
    "v6": "v6_experimental",
    "v7": "v7_experimental",
    "v8": "v8_experimental",
    "v9": "v9_experimental",
    "v10": "v10_experimental",
    "v11": "v11_experimental",
}


def normalize_line_version(line_version: str) -> str:
    version = (line_version or "v2").strip().lower()
    if version not in SUPPORTED_LINE_VERSIONS and not MODEL_GROUNDING_VERSION_RE.fullmatch(version):
        raise ValueError(f"Unsupported line version: {line_version}")
    return version


def line_version_family(line_version: str) -> str:
    """Return the workload family while preserving dotted V10/V11 artifact versions elsewhere."""
    version = normalize_line_version(line_version)
    return version.split(".", 1)[0] if MODEL_GROUNDING_VERSION_RE.fullmatch(version) else version


def _tabquerybench_root() -> Path | None:
    explicit = os.getenv("TABQUERYBENCH_ROOT")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.exists():
            return candidate
    return TABQUERYBENCH_ROOT


def _tabquerybench_line_root(line_version: str) -> Path | None:
    root = _tabquerybench_root()
    if root is None:
        return None
    version = line_version_family(line_version)
    candidate = root / "sql_workloads" / TABQUERYBENCH_LAYOUT[version]
    return candidate if candidate.exists() else None


def workload_data_root(line_version: str = "v2") -> Path:
    version = normalize_line_version(line_version)
    tb_root = _tabquerybench_line_root(version)
    if tb_root is not None:
        base = tb_root / "grounding"
    else:
        base = PROJECT_ROOT / "data" / f"workload_grounding_{line_version_family(version)}"
    return base / "variants" / version if MODEL_GROUNDING_VERSION_RE.fullmatch(version) else base


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
    tb_root = _tabquerybench_line_root(version)
    if tb_root is not None:
        base = tb_root / "runs_and_launches"
    else:
        base = LOGS_DIR / f"subitem_workload_{line_version_family(version)}"
    return base / "variants" / version if MODEL_GROUNDING_VERSION_RE.fullmatch(version) else base


def runs_root(line_version: str = "v2") -> Path:
    return logs_root(line_version) / "runs"


def evaluation_root(line_version: str = "v2") -> Path:
    version = normalize_line_version(line_version)
    tb_root = _tabquerybench_line_root(version)
    if tb_root is not None:
        base = tb_root
    else:
        base = PROJECT_ROOT / "Evaluation" / f"subitem_workload_{line_version_family(version)}"
    return base / "variants" / version if MODEL_GROUNDING_VERSION_RE.fullmatch(version) else base


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
    version = line_version_family(line_version)
    if version in {"v3", "v4"}:
        return DEFAULT_V3_V4_DATASETS
    if version in {"v5", "v6", "v7", "v8", "v9", "v10", "v11"}:
        return DEFAULT_ALL_DATASETS
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
