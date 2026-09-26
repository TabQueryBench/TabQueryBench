"""SV2 — Semantic Scoring V2: one semantic scorer and one established metric per query."""

from .aggregate import summarize_query_scores
from .api import score_execution_results, score_query
from .common import CONTRACT_VERSION, SCORE_METHOD, SCORER_TYPES, ScoringError
from .routing import resolve_sv2_scorer

__all__ = [
    "CONTRACT_VERSION",
    "SCORE_METHOD",
    "SCORER_TYPES",
    "ScoringError",
    "resolve_sv2_scorer",
    "score_execution_results",
    "score_query",
    "summarize_query_scores",
]
