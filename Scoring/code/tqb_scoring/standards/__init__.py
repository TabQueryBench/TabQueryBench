"""Registry of versioned scoring standards.

Each standard lives in ``Scoring/Scoring_Standard/<version>/`` next to its
definition documents, with the implementation in ``scorer.py``. That directory
is added to this package's search path, so ``tqb_scoring.standards.<version>.scorer``
imports the file directly and the standard folder stays the single source of
truth.

To add a new standard (for example ``sv2``), create
``Scoring_Standard/sv2/scorer.py`` exposing ``compare_semantic_execution_results``
and register it in ``STANDARDS``.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import ModuleType

from tqb_scoring.paths import SCORING_STANDARD_ROOT

__path__.append(str(SCORING_STANDARD_ROOT))  # type: ignore[name-defined]  # noqa: F821


@dataclass(frozen=True)
class StandardInfo:
    version: str
    module: str
    score_version: str
    description: str


STANDARDS: dict[str, StandardInfo] = {
    "legacy_composite": StandardInfo(
        "legacy_composite",
        "legacy_composite.scorer",
        "composite_key_profile_rowcount_column",
        "Legacy fixed composite of key match, profile, row-count, and column similarity.",
    ),
    "semantic_v8": StandardInfo(
        "semantic_v8",
        "semantic_v8.scorer",
        "semantic_scorer_v1_parallel",
        "Template-specific semantic scorers with 100/50_50/50_25_25 weight rules (V8 run3).",
    ),
    "standard_v1": StandardInfo(
        "standard_v1",
        "standard_v1.scorer",
        "semantic_scoring_standard_v1",
        "Semantic Scoring Standard v1 (used by the SPQ v8 runs and review bundle).",
    ),
    "spq_v9": StandardInfo(
        "spq_v9",
        "spq_v9.scorer",
        "spq_v9_single_primary",
        "SPQ v9 single-primary scorer, repository implementation (SEMANTIC_SCORING_V9_SINGLE_PRIMARY.md).",
    ),
    "spq_v9_sqlagent_batch": StandardInfo(
        "spq_v9_sqlagent_batch",
        "spq_v9.sqlagent_batch_20260914.scorer",
        "spq_v9_single_primary",
        "SPQ v9 variant from SQLagent used for the semantic_spq_v9_batch_20260914* runs.",
    ),
    "sv2": StandardInfo(
        "sv2",
        "sv2.scorer",
        "sv2_four_primary_v1",
        "Semantic Scoring V2: numeric magnitude (sMAPE), count distribution (TVD), rate (MAE), ranking (RBO).",
    ),
}

DEFAULT_SEMANTIC_STANDARD = "spq_v9"
# Environment switch kept compatible with the SQLagent runner (TQB_SCORING_MODE=spq_v9).
SCORING_MODE_ENV = "TQB_SCORING_MODE"


def load_standard(version: str) -> ModuleType:
    """Return the ``scorer`` module registered for ``version``."""
    if version not in STANDARDS:
        raise KeyError(f"Unknown scoring standard {version!r}; known: {sorted(STANDARDS)}")
    return importlib.import_module(f"{__name__}.{STANDARDS[version].module}")


def selected_semantic_standard() -> str:
    """Semantic standard used by analysis runs; ``TQB_SCORING_MODE`` overrides the default."""
    mode = os.environ.get(SCORING_MODE_ENV)
    if mode and mode in STANDARDS and mode != "legacy_composite":
        return mode
    return DEFAULT_SEMANTIC_STANDARD


SINGLE_PRIMARY_STANDARDS = frozenset({"spq_v9", "spq_v9_sqlagent_batch", "sv2"})


def semantic_primary_scoring_requested() -> bool:
    """True when ``TQB_SCORING_MODE`` selects a single-primary standard, which then is the primary score."""
    return os.environ.get(SCORING_MODE_ENV) in SINGLE_PRIMARY_STANDARDS
