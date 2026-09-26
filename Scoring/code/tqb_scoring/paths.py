"""Repository locations used by scoring code."""

from __future__ import annotations

import os
from pathlib import Path

SCORING_CODE_ROOT = Path(__file__).resolve().parents[1]
SCORING_ROOT = SCORING_CODE_ROOT.parent
REPO_ROOT = SCORING_ROOT.parent

# Scoring outputs (formerly ``code/Evaluation``).
SCORING_RESULTS_ROOT = Path(os.getenv("TQB_SCORING_RESULTS_ROOT", SCORING_ROOT / "results")).resolve()
# Versioned scoring definitions and implementations.
SCORING_STANDARD_ROOT = SCORING_ROOT / "Scoring_Standard"

# Query generation working data and logs (formerly ``code/data`` and ``code/logs``).
QUERY_CODE_ROOT = REPO_ROOT / "Query" / "code"
QUERY_RELEASES_ROOT = REPO_ROOT / "Query" / "Queries"

# Real and synthetic tabular data.
RAW_DATA_ROOT = REPO_ROOT / "Synthesizing" / "raw_data" / "tabular_datasets"
SYNTHETIC_DATA_ROOT = REPO_ROOT / "Synthesizing" / "synthetic_data"
