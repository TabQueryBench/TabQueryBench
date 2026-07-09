"""Project-level settings and default paths."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.getenv("SQL_AGENT_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()


def _detect_tabquerybench_root(project_root: Path) -> Path | None:
    explicit = os.getenv("TABQUERYBENCH_ROOT")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.exists():
            return candidate
    parent = project_root.parent
    if (parent / "sql_workloads").exists() and (parent / "RowData").exists():
        return parent
    return None


TABQUERYBENCH_ROOT = _detect_tabquerybench_root(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
RUNS_DIR = LOGS_DIR / "runs"

DEFAULT_DB_PATH = DATA_DIR / "Chinook.db"
DEFAULT_USAGE_CSV_PATH = LOGS_DIR / "usage_log.csv"
DEFAULT_MODEL = os.getenv("SQL_AGENT_DEFAULT_MODEL", "gpt-5.2")
MODEL_PRICING_CONFIG_PATH = CONFIG_DIR / "model_pricing.json"
FAMILY_FACET_CATALOG_PATH = CONFIG_DIR / "family_facet_catalog_v0_1.yaml"
DEFAULT_SQL_EXEMPLAR_POOL_PATH = (
    LOGS_DIR
    / "sql_high_corpus_build_20260404"
    / "v2_refinement"
    / "preprocessed_sql_exemplars"
    / "benchmark_sql_exemplar_pool.csv"
)
DEFAULT_DATASET_ID = os.getenv("SQL_AGENT_DEFAULT_DATASET_ID", "c2")


def ensure_runtime_dirs() -> None:
    """Create runtime directories if they do not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
