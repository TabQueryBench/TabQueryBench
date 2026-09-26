"""Legacy composite scorer.

The legacy score is still computed inline by the analysis runner for every
query (reported as ``legacy_query_score``), so its implementation remains in
``tqb_scoring.evaluation.real_panel_experiment``. This module exposes it under
the standard interface.
"""

from __future__ import annotations

from typing import Any

from tqb_scoring.evaluation.real_panel_experiment import _compare_execution_results

SCORE_VERSION = "composite_key_profile_rowcount_column"
QUERY_SCORE_METHOD = SCORE_VERSION


def compare_execution_results(real_exec: Any, syn_exec: Any, *, result_role_annotation: dict[str, Any] | None = None):
    return _compare_execution_results(real_exec, syn_exec, result_role_annotation=result_role_annotation)
