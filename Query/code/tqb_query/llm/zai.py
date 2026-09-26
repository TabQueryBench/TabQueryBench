"""Z.AI GLM-5.3 client over the OpenAI-compatible chat completions API.

The API key is read from ``ZAI_API_KEY``.  When the variable is absent (for
example in a detached tmux job that did not source the shell profile), the
client falls back to ``ZAI_API_KEY_FILE`` (default ``~/.config/zai/env``), a
file outside every repository containing ``export ZAI_API_KEY=...``.  The key
is never logged, written to artifacts, or included in raised errors.
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/"
# GLM Coding Plan quota is only billable through this endpoint; on ZAI_BASE_URL
# a plan-only key gets 429 code=1113 ("Insufficient balance") for glm-5.3.
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4/"
ZAI_BASE_URLS = {"general": ZAI_BASE_URL, "coding": ZAI_CODING_BASE_URL}
ZAI_API_BASE_ENV = "ZAI_API_BASE"
ZAI_MODEL = "glm-5.3"
# The Z.AI models this project grounds with; each maps to its own v11.5.y artifact version.
ZAI_MODELS = frozenset({"glm-5.2", "glm-5.3", "glm-5.3-flash"})
ZAI_API_KEY_ENV = "ZAI_API_KEY"
ZAI_API_KEY_FILE_ENV = "ZAI_API_KEY_FILE"
DEFAULT_KEY_FILE = Path.home() / ".config" / "zai" / "env"
REASONING_EFFORTS = frozenset({"low", "high", "max"})

# Business codes from https://docs.z.ai/api-reference/api-code.
# Balance/plan/usage-window exhaustion: retrying cannot help, stop the run.
QUOTA_ERROR_CODES = frozenset({"1113", "1308", "1309", "1310", "1311", "1313", "1314", "1315"} | {str(c) for c in range(1316, 1322)})
# Transient throttling/overload: retry with backoff.
RETRYABLE_ERROR_CODES = frozenset({"1302", "1305", "1200", "1230", "1234"})


class ZAIError(RuntimeError):
    """A Z.AI API call failed; the message never contains the API key."""

    def __init__(self, message: str, *, status_code: int | None = None, error_code: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class ZAIQuotaError(ZAIError):
    """Account balance, plan, or usage window is exhausted."""


def _read_key_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(rf"^\s*(?:export\s+)?{ZAI_API_KEY_ENV}\s*=\s*['\"]?([^'\"\s]+)", text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def load_zai_api_key() -> str:
    key = os.environ.get(ZAI_API_KEY_ENV, "").strip()
    if key:
        return key
    key = _read_key_file(Path(os.environ.get(ZAI_API_KEY_FILE_ENV, "") or DEFAULT_KEY_FILE).expanduser())
    if key:
        return key
    raise ZAIError(
        f"{ZAI_API_KEY_ENV} is not set; export it or put `export {ZAI_API_KEY_ENV}=...` in {DEFAULT_KEY_FILE}"
    )


def resolve_zai_base_url() -> str:
    choice = os.environ.get(ZAI_API_BASE_ENV, "").strip().lower() or "coding"
    if choice not in ZAI_BASE_URLS:
        raise ZAIError(f"{ZAI_API_BASE_ENV} must be one of {sorted(ZAI_BASE_URLS)}, got {choice!r}")
    return ZAI_BASE_URLS[choice]


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def _error_details(exc: Exception) -> tuple[int | None, str, str]:
    """Return (http_status, business_code, message) for an OpenAI SDK exception."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    code = ""
    message = str(exc)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = str(error.get("code") or "")
            message = str(error.get("message") or message)
    return status, code, message


@dataclass
class ZAIChatResult:
    text: str
    reasoning_text: str
    usage: dict[str, int]
    response_id: str
    finish_reason: str
    raw: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    elapsed_ms: int = 0


class ZAIChatClient:
    """Thin wrapper around ``openai.OpenAI`` pinned to Z.AI and ``glm-5.3``."""

    def __init__(
        self,
        *,
        model: str = ZAI_MODEL,
        timeout_seconds: float = 300.0,
        max_attempts: int = 5,
        max_backoff_seconds: float = 60.0,
        log_stream: Any = None,
    ) -> None:
        from openai import OpenAI

        if model not in ZAI_MODELS:
            raise ZAIError(f"Unsupported Z.AI model {model!r}; expected one of {sorted(ZAI_MODELS)}")
        self._api_key = load_zai_api_key()
        self.model = model
        self.base_url = resolve_zai_base_url()
        self.max_attempts = max(1, int(max_attempts))
        self.max_backoff_seconds = max_backoff_seconds
        self.log_stream = log_stream if log_stream is not None else sys.stderr
        # Retries are handled here so every failure is logged with its API error code.
        self._client = OpenAI(api_key=self._api_key, base_url=self.base_url, timeout=timeout_seconds, max_retries=0)

    def _log(self, message: str) -> None:
        print(f"[zai] {_redact(message, self._api_key)}", file=self.log_stream, flush=True)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
    ) -> ZAIChatResult:
        import httpx
        from openai import APIConnectionError, APIError, APITimeoutError

        # Streaming keeps bytes flowing: non-streaming replies slower than ~60s were cut by the network path
        # and surfaced as APIConnectionError, so long GLM thinking calls only succeeded after retries.
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # GLM-5.3 always thinks; depth is reasoning_effort (server default "max").
        if reasoning_effort:
            if reasoning_effort not in REASONING_EFFORTS:
                raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}, got {reasoning_effort!r}")
            kwargs["reasoning_effort"] = reasoning_effort

        started = time.monotonic()
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._collect_stream(
                    self._client.chat.completions.create(**kwargs), attempts=attempt, started=started
                )
            except (APITimeoutError, APIConnectionError, httpx.HTTPError) as exc:
                # Includes a stream dropped mid-reply (raw httpx errors raised while iterating chunks).
                cause = f" (cause: {type(exc.__cause__).__name__}: {exc.__cause__})" if exc.__cause__ else ""
                detail = _redact(f"{type(exc).__name__}: {exc}{cause}", self._api_key)
                if attempt >= self.max_attempts:
                    self._log(f"request failed (attempt {attempt}/{self.max_attempts}): {detail}")
                    raise ZAIError(detail) from None
                self._backoff(attempt, detail)
            except APIError as exc:
                # HTTP errors (APIStatusError) and error events sent inside a stream (no status code).
                status, code, message = _error_details(exc)
                detail = _redact(f"HTTP {status} code={code or '-'} message={message}", self._api_key)
                if code in QUOTA_ERROR_CODES:
                    if code == "1113" and self.base_url == ZAI_BASE_URL:
                        detail += f" (if this key is a GLM Coding Plan, set {ZAI_API_BASE_ENV}=coding)"
                    self._log(f"quota/balance error, not retrying: {detail}")
                    raise ZAIQuotaError(detail, status_code=status, error_code=code) from None
                retryable = code in RETRYABLE_ERROR_CODES or status is None or status in {408, 409, 429} or status >= 500
                if not retryable or attempt >= self.max_attempts:
                    self._log(f"request failed (attempt {attempt}/{self.max_attempts}): {detail}")
                    raise ZAIError(detail, status_code=status, error_code=code) from None
                self._backoff(attempt, detail)
        raise ZAIError("unreachable")  # pragma: no cover

    def _backoff(self, attempt: int, detail: str) -> None:
        delay = min(self.max_backoff_seconds, 2.0**attempt) + random.uniform(0, 1)
        self._log(f"attempt {attempt}/{self.max_attempts} failed: {detail}; retrying in {delay:.1f}s")
        time.sleep(delay)

    @staticmethod
    def _collect_stream(stream: Any, *, attempts: int, started: float) -> ZAIChatResult:
        content: list[str] = []
        reasoning: list[str] = []
        response_id = ""
        model = ""
        finish_reason = ""
        usage = None
        for chunk in stream:
            response_id = response_id or str(getattr(chunk, "id", "") or "")
            model = model or str(getattr(chunk, "model", "") or "")
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                content.append(str(getattr(delta, "content", None) or ""))
                reasoning.append(str(getattr(delta, "reasoning_content", None) or ""))
                finish_reason = str(getattr(choice, "finish_reason", None) or finish_reason)
        details = getattr(usage, "prompt_tokens_details", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        usage_payload = {
            "input_tokens": prompt_tokens,
            "cached_input_tokens": int(getattr(details, "cached_tokens", 0) or 0),
            "output_tokens": completion_tokens,
            "total_tokens": int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)),
        }
        text = "".join(content)
        reasoning_text = "".join(reasoning)
        return ZAIChatResult(
            text=text,
            reasoning_text=reasoning_text,
            usage=usage_payload,
            response_id=response_id,
            finish_reason=finish_reason,
            raw={
                "id": response_id,
                "model": model,
                "stream": True,
                "finish_reason": finish_reason,
                "content": text,
                "reasoning_content": reasoning_text,
                "usage": usage_payload,
            },
            attempts=attempts,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
