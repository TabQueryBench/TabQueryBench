"""SV2 standard entry point used by the analysis runner (``TQB_SCORING_MODE=sv2``)."""

from __future__ import annotations

from typing import Any

from .api import score_execution_results, score_query  # noqa: F401  (re-exported)
from .common import CONTRACT_VERSION, SCORE_METHOD

SCORE_VERSION = CONTRACT_VERSION
SEMANTIC_QUERY_SCORE_METHOD = SCORE_METHOD


def compare_semantic_execution_results(
    real_exec: Any,
    syn_exec: Any,
    *,
    query: dict[str, Any] | None = None,
    legacy_detail: dict[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Runner interface: returns (score or None when invalid, detail)."""
    result = score_execution_results(real_exec, syn_exec, query or {})
    valid = bool(result["valid"])
    detail = {
        **result,
        "score_version": CONTRACT_VERSION,
        "validity_status": "ok" if valid else "invalid",
        "validity_failures": [] if valid else [result.get("error_code")],
        "primary_metric": result["metric"],
        "primary_score": result["semantic_query_score"],
        "weight_rule": "single_primary",
        "component_scores": {result["metric"]: result["semantic_query_score"]} if valid else {},
        "legacy_query_score_method": (legacy_detail or {}).get("query_score_method"),
    }
    return result["semantic_query_score"], detail
