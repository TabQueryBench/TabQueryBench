"""Public SV2 entry points: score raw answers or SQL execution results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .canonicalize import answers_from_results, build_column_plan, canonicalize_answer
from .common import (
    CONTRACT_VERSION,
    COUNT_SUPPORT_DISTRIBUTION,
    NUMERIC_MAGNITUDE,
    RATE_SHARE_PROPORTION,
    SCORE_METHOD,
    TOPK_RANKING,
    ScoringError,
    checked_unit_interval,
    metric_id_for,
)
from .count_support_distribution import score_count_support_distribution
from .numeric_magnitude import score_numeric_magnitude
from .rate_share_proportion import score_rate_share_proportion
from .routing import resolve_sv2_scorer
from .topk_ranking import score_topk_ranking

SCORERS = {
    NUMERIC_MAGNITUDE: score_numeric_magnitude,
    COUNT_SUPPORT_DISTRIBUTION: score_count_support_distribution,
    RATE_SHARE_PROPORTION: score_rate_share_proportion,
    TOPK_RANKING: score_topk_ranking,
}


def _result(
    *,
    scorer_type: str | None,
    score: float | None = None,
    diagnostics: dict[str, Any] | None = None,
    error: ScoringError | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": error is None,
        "semantic_query_score_method": SCORE_METHOD,
        "score_contract_version": CONTRACT_VERSION,
        "scorer_type": scorer_type,
        "metric": metric_id_for(scorer_type),
        "semantic_query_score": score,
        "query_score": score,
        "diagnostics": diagnostics or {},
    }
    if error is not None:
        result["error_code"] = error.code
        result["error_message"] = str(error)
    return result


def score_query(
    real_answer: Any,
    synthetic_answer: Any,
    scorer_type: str | None = None,
    scorer_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score two raw answers.

    ``scorer_config`` may set ``requested_depth`` (ranking) and ``rate_scale``
    (``"unit"`` or an explicitly declared ``"percent"``).
    """
    routing_metadata = dict(metadata or {})
    if scorer_type is not None:
        routing_metadata["sv2_scorer_type"] = scorer_type
    resolved: str | None = None
    try:
        resolved, routing_source = resolve_sv2_scorer(routing_metadata)
        config = {**routing_metadata, **(scorer_config or {})}
        real = canonicalize_answer(real_answer, scorer_type=resolved, metadata=config)
        syn = canonicalize_answer(synthetic_answer, scorer_type=resolved, metadata=config)
        score, diagnostics = SCORERS[resolved](real, syn, config)
        diagnostics["routing_source"] = routing_source
        return _result(scorer_type=resolved, score=checked_unit_interval(score), diagnostics=diagnostics)
    except ScoringError as exc:
        return _result(scorer_type=resolved, error=exc)


def score_execution_results(real_exec: Any, syn_exec: Any, query: Mapping[str, Any]) -> dict[str, Any]:
    """Score one query executed on real and synthetic data (``ExecutionResult``-like objects)."""
    resolved: str | None = None
    diagnostics: dict[str, Any] = {"template_id": query.get("template_id")}
    try:
        resolved, routing_source = resolve_sv2_scorer(query)
        diagnostics["routing_source"] = routing_source
        if not getattr(real_exec, "ok", False):
            raise ScoringError("REAL_EXECUTION_FAILED", str(getattr(real_exec, "error", "") or "real query failed"))
        if not getattr(syn_exec, "ok", False):
            raise ScoringError("SYNTHETIC_EXECUTION_FAILED", str(getattr(syn_exec, "error", "") or "synthetic query failed"))

        real_columns = [str(column) for column in getattr(real_exec, "columns", []) or []]
        syn_columns = [str(column) for column in getattr(syn_exec, "columns", []) or []]
        if [column.lower() for column in real_columns] != [column.lower() for column in syn_columns]:
            raise ScoringError("COLUMN_MISMATCH", f"Real columns {real_columns} differ from synthetic columns {syn_columns}.")

        plan = build_column_plan(real_columns, scorer_type=resolved, query=query)
        diagnostics.update(plan.describe())
        real, syn, answer_diagnostics = answers_from_results(
            list(getattr(real_exec, "rows", []) or []),
            list(getattr(syn_exec, "rows", []) or []),
            plan,
        )
        diagnostics.update(answer_diagnostics)

        config = {"requested_depth": plan.requested_depth, "rate_scale": plan.rate_scale}
        score, scorer_diagnostics = SCORERS[resolved](real, syn, config)
        diagnostics.update(scorer_diagnostics)
        return _result(scorer_type=resolved, score=checked_unit_interval(score), diagnostics=diagnostics)
    except ScoringError as exc:
        return _result(scorer_type=resolved, diagnostics=diagnostics, error=exc)
