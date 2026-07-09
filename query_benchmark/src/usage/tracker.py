"""Collect token usage from streamed LangGraph/LangChain messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UsageSnapshot:
    api_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_usage(message: Any) -> tuple[int, int, int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    input_tokens = _safe_int(usage.get("input_tokens"))
    output_tokens = _safe_int(usage.get("output_tokens"))
    total_tokens = _safe_int(usage.get("total_tokens"))
    input_details = usage.get("input_token_details", {}) or {}
    cached_input_tokens = _safe_int(input_details.get("cache_read"), _safe_int(input_details.get("cached_tokens")))

    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {})
    prompt_token_details = token_usage.get("prompt_tokens_details", {}) or {}

    if input_tokens == 0:
        input_tokens = _safe_int(token_usage.get("prompt_tokens"))
    if output_tokens == 0:
        output_tokens = _safe_int(token_usage.get("completion_tokens"))
    if total_tokens == 0:
        total_tokens = _safe_int(token_usage.get("total_tokens"), input_tokens + output_tokens)
    if cached_input_tokens == 0:
        cached_input_tokens = _safe_int(prompt_token_details.get("cached_tokens"))

    if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
        total_tokens = input_tokens + output_tokens

    return input_tokens, cached_input_tokens, output_tokens, total_tokens


def _looks_like_model_message(message: Any) -> bool:
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    return bool(usage) or bool(response_metadata)


class UsageTracker:
    """Aggregate usage over one user question/run."""

    def __init__(self) -> None:
        self.snapshot = UsageSnapshot()

    def add_message(self, message: Any) -> None:
        # Keep this tracker importable even when LangChain is not installed.
        if not hasattr(message, "usage_metadata") and not hasattr(message, "response_metadata"):
            return
        if not _looks_like_model_message(message):
            return

        input_tokens, cached_input_tokens, output_tokens, total_tokens = _extract_usage(message)
        self.snapshot.api_calls += 1
        self.snapshot.input_tokens += input_tokens
        self.snapshot.cached_input_tokens += cached_input_tokens
        self.snapshot.output_tokens += output_tokens
        self.snapshot.total_tokens += total_tokens

    def has_model_usage(self) -> bool:
        return self.snapshot.api_calls > 0
