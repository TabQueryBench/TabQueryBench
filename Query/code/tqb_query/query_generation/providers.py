"""Binding-model adapters used by the query-generation pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class BindingResponse:
    bindings: dict[str, Any]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw_response: str = ""


class BindingProvider(Protocol):
    def bind(self, request: dict[str, Any]) -> BindingResponse: ...


class BindingProviderError(RuntimeError):
    """Base class for model-provider failures."""


class BindingProviderUnavailable(BindingProviderError):
    """Authentication, transport, quota, or provider infrastructure failure."""


class BindingResponseError(BindingProviderError):
    """A recoverable malformed model response."""


def candidate_first_bindings(request: dict[str, Any]) -> dict[str, Any]:
    """Build a valid deterministic response for tests and deterministic templates."""
    expected = list(request["output_contract"]["expected_binding_keys"])
    allowed = request["allowed_bindings"]
    column_roles = allowed.get("column_roles") or {}
    result: dict[str, Any] = {}
    for role in expected:
        options = list(column_roles.get(role) or [])
        if options:
            result[role] = options[0]
    predicates = list(allowed.get("predicate_candidates") or [])
    if {"predicate_col", "predicate_op", "predicate_value"}.issubset(expected) and predicates:
        predicate = predicates[0]
        result.update(predicate_col=predicate["column"], predicate_op=predicate["operator"], predicate_value=predicate["value"])
    observed = allowed.get("observed_values_by_column") or {}
    for value_role, column_role, index in (
        ("condition_value", "condition_col", 0),
        ("positive_value", "condition_col", 0),
        ("negative_value", "condition_col", 1),
        ("target_value", "target_col", 0),
    ):
        if value_role in expected and column_role in result:
            options = list(observed.get(result[column_role]) or [])
            if options:
                result[value_role] = options[min(index, len(options) - 1)]
    numeric = allowed.get("numeric_candidates_by_column") or {}
    for value_role, column_role, index in (
        ("band_cut_1", "band_col", 1),
        ("band_cut_2", "band_col", -2),
        ("lower_bound", "band_col", 0),
        ("upper_bound", "band_col", -1),
        ("measure_threshold", "measure_col", 2),
    ):
        if value_role in expected and column_role in result:
            options = list(numeric.get(result[column_role]) or [])
            if options:
                result[value_role] = options[max(-len(options), min(index, len(options) - 1))]
    for role, options in (allowed.get("parameter_candidates") or {}).items():
        if role in expected and options:
            result[role] = options[0]
    distinct_pairs = ((request.get("template") or {}).get("role_constraints") or {}).get("distinct_roles") or []
    for left, right in distinct_pairs:
        if left in result and right in result and result[left] == result[right]:
            alternatives = [value for value in column_roles.get(right, []) if value != result[left]]
            if alternatives:
                result[right] = alternatives[0]
    return result


class HeuristicBindingProvider:
    """Deterministic provider for tests; production must configure a model provider."""

    def bind(self, request: dict[str, Any]) -> BindingResponse:
        return BindingResponse(bindings=candidate_first_bindings(request), model="heuristic-test-provider")


class LangChainBindingProvider:
    """Provider-neutral API adapter backed by LangChain's configured chat model."""

    def __init__(self, model: str) -> None:
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as exc:  # pragma: no cover - dependency installation path
            raise RuntimeError("LangChain dependencies are required for model grounding") from exc
        self.model_name = model
        self.model = init_chat_model(model)

    def bind(self, request: dict[str, Any]) -> BindingResponse:
        system = (
            "Ground the immutable SQL template using only allowed_bindings. "
            "Return exactly one JSON object with one top-level field named bindings. "
            "Select every expected key; copy candidate scalars exactly as provided, without adding SQL quotes. "
            "Do not invent columns, values, or parameters and do not rewrite SQL."
        )
        started = time.perf_counter()
        try:
            response = self.model.invoke([
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ])
        except Exception as exc:  # provider/auth/quota failures must fail the job
            raise BindingProviderUnavailable(str(exc)) from exc
        latency = (time.perf_counter() - started) * 1000
        raw_content = response.content
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            parts: list[str] = []
            for block in raw_content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            content = "".join(parts)
        else:
            content = str(raw_content or "")
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BindingResponseError("model response is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"bindings"} or not isinstance(payload["bindings"], dict):
            raise BindingResponseError("model response must contain exactly one bindings object")
        usage = dict(getattr(response, "usage_metadata", None) or {})
        return BindingResponse(payload["bindings"], self.model_name, usage, latency, content)


class ClaudeCliBindingProvider:
    """Local-development adapter for Claude CLI; no tools or shell are used."""

    def __init__(self, model: str, *, timeout_seconds: int = 420) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._:-]+", model):
            raise ValueError("invalid Claude CLI model name")
        executable = shutil.which("claude")
        if executable is None:
            raise BindingProviderUnavailable("claude executable was not found")
        self.model_name = model
        self.timeout_seconds = timeout_seconds
        self.command = [
            executable, "--print", "--input-format", "text", "--permission-mode", "dontAsk",
            "--tools", "", "--model", model,
        ]

    def bind(self, request: dict[str, Any]) -> BindingResponse:
        prompt = (
            "Ground the immutable SQL template using only allowed_bindings. "
            "Return exactly one JSON object with one top-level field named bindings. "
            "Select every expected key; copy candidate scalars exactly as provided, without adding SQL quotes. "
            "Do not invent columns, values, or parameters and do not rewrite SQL.\n\n"
            + json.dumps(request, ensure_ascii=False)
        )
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                self.command,
                input=prompt,
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BindingProviderUnavailable(str(exc)) from exc
        latency = (time.perf_counter() - started) * 1000
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:1000]
            raise BindingProviderUnavailable(f"Claude CLI failed with exit code {completed.returncode}: {detail}")
        raw = completed.stdout.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BindingResponseError("Claude CLI response is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"bindings"} or not isinstance(payload["bindings"], dict):
            raise BindingResponseError("Claude CLI response must contain exactly one bindings object")
        return BindingResponse(payload["bindings"], self.model_name, {}, latency, completed.stdout)


def provider_from_environment() -> BindingProvider:
    provider = os.getenv("QUERY_GENERATION_PROVIDER", "").strip().lower()
    model = os.getenv("QUERY_GENERATION_MODEL", "").strip()
    if provider == "langchain" and model:
        return LangChainBindingProvider(model)
    if provider == "claude-cli" and model:
        return ClaudeCliBindingProvider(model)
    raise RuntimeError(
        "No AI binding provider configured. Set QUERY_GENERATION_PROVIDER=langchain or claude-cli "
        "and QUERY_GENERATION_MODEL=<model>."
    )
