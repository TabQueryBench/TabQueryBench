"""LLM runtime wrapper with phase-level usage and trace logging."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from src.benchmark.contracts import hash_prompt
from src.logging.run_artifacts import RunArtifactWriter
from src.usage.logger import UsageCSVLogger, UsageLogRecord
from src.usage.pricing import ModelPricing, calculate_cost_usd, resolve_model_pricing
from src.usage.tracker import UsageTracker


@dataclass
class CallUsage:
    api_calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float


class BenchmarkLLMRuntime:
    def __init__(
        self,
        *,
        model_name: str,
        dataset_id: str,
        run_id: str,
        usage_logger: UsageCSVLogger,
        pricing_config: dict[str, Any],
        artifact_writer: RunArtifactWriter,
        enforce_model: str = "gpt-4.1-mini",
    ) -> None:
        normalized = model_name.strip()
        if normalized != enforce_model:
            raise ValueError(
                f"Benchmark v1 requires model '{enforce_model}'. Received: '{model_name}'."
            )

        self.model_name = normalized
        self.dataset_id = dataset_id
        self.run_id = run_id
        self.usage_logger = usage_logger
        self.pricing = resolve_model_pricing(self.model_name, pricing_config)
        self.artifact_writer = artifact_writer
        self.request_timeout_seconds = int(os.getenv("BENCHMARK_LLM_TIMEOUT_SECONDS", "60"))
        self.provider_retries = int(os.getenv("BENCHMARK_LLM_PROVIDER_RETRIES", "2"))
        self.invoke_retries = max(1, int(os.getenv("BENCHMARK_LLM_INVOKE_RETRIES", "3")))
        self.model = init_chat_model(
            self.model_name,
            timeout=self.request_timeout_seconds,
            max_retries=self.provider_retries,
        )

        self.summary = {
            "model": self.model_name,
            "api_calls": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "by_phase": {},
            "prompt_signatures": {},
            "invoke_config": {
                "request_timeout_seconds": self.request_timeout_seconds,
                "provider_retries": self.provider_retries,
                "invoke_retries": self.invoke_retries,
            },
        }

    def _record_prompt_signature(self, *, phase: str, module: str, system_prompt: str) -> None:
        key = f"{phase}:{module}"
        prompt_hash = hash_prompt(system_prompt)
        self.summary["prompt_signatures"][key] = {
            "phase": phase,
            "module": module,
            "system_prompt_hash": prompt_hash,
            "prompt_hash_algo": "sha256_20",
        }

    def _extract_usage(self, response: AIMessage) -> CallUsage:
        tracker = UsageTracker()
        tracker.add_message(response)
        snapshot = tracker.snapshot
        cost = calculate_cost_usd(
            snapshot.input_tokens,
            snapshot.output_tokens,
            self.pricing,
            cached_input_tokens=snapshot.cached_input_tokens,
        )
        return CallUsage(
            api_calls=snapshot.api_calls,
            input_tokens=snapshot.input_tokens,
            cached_input_tokens=snapshot.cached_input_tokens,
            output_tokens=snapshot.output_tokens,
            total_tokens=snapshot.total_tokens,
            cost_usd=cost,
        )

    def _append_usage_record(self, *, phase: str, module: str, question: str, usage: CallUsage) -> None:
        if usage.api_calls == 0:
            return

        self.usage_logger.append(
            UsageLogRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                run_id=self.run_id,
                dataset_id=self.dataset_id,
                phase=phase,
                module=module,
                question=question,
                model=self.model_name,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=usage.cost_usd,
            )
        )

        self.summary["api_calls"] += usage.api_calls
        self.summary["input_tokens"] += usage.input_tokens
        self.summary["cached_input_tokens"] += usage.cached_input_tokens
        self.summary["output_tokens"] += usage.output_tokens
        self.summary["total_tokens"] += usage.total_tokens
        self.summary["cost_usd"] += usage.cost_usd

        by_phase = self.summary["by_phase"]
        if phase not in by_phase:
            by_phase[phase] = {
                "api_calls": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        by_phase[phase]["api_calls"] += usage.api_calls
        by_phase[phase]["input_tokens"] += usage.input_tokens
        by_phase[phase]["cached_input_tokens"] += usage.cached_input_tokens
        by_phase[phase]["output_tokens"] += usage.output_tokens
        by_phase[phase]["total_tokens"] += usage.total_tokens
        by_phase[phase]["cost_usd"] += usage.cost_usd

    def invoke_text(
        self,
        *,
        phase: str,
        module: str,
        system_prompt: str,
        user_prompt: str,
        question_for_usage: str,
    ) -> str:
        self._record_prompt_signature(phase=phase, module=module, system_prompt=system_prompt)

        last_exc: Exception | None = None
        response: AIMessage | None = None
        for attempt in range(1, self.invoke_retries + 1):
            try:
                response = self.model.invoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.invoke_retries:
                    raise
                wait_seconds = min(2 ** (attempt - 1), 8)
                self.artifact_writer.append_trace(
                    {
                        "event_type": "llm_call_retry",
                        "phase": phase,
                        "module": module,
                        "question": question_for_usage,
                        "attempt": attempt,
                        "max_attempts": self.invoke_retries,
                        "wait_seconds": wait_seconds,
                        "error": str(exc),
                    }
                )
                time.sleep(wait_seconds)

        if response is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("LLM invocation failed with unknown error.")

        content = response.content if isinstance(response.content, str) else json.dumps(response.content, ensure_ascii=False)

        usage = self._extract_usage(response)
        self._append_usage_record(phase=phase, module=module, question=question_for_usage, usage=usage)

        self.artifact_writer.append_trace(
            {
                "event_type": "llm_call",
                "phase": phase,
                "module": module,
                "question": question_for_usage,
                "usage": {
                    "api_calls": usage.api_calls,
                    "input_tokens": usage.input_tokens,
                    "cached_input_tokens": usage.cached_input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "cost_usd": usage.cost_usd,
                },
                "response_preview": content[:500],
            }
        )
        return content

    def invoke_json(
        self,
        *,
        phase: str,
        module: str,
        system_prompt: str,
        user_prompt: str,
        question_for_usage: str,
    ) -> dict[str, Any]:
        text = self.invoke_text(
            phase=phase,
            module=module,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            question_for_usage=question_for_usage,
        )
        return parse_json_response(text)


def parse_json_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        return {}

    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\\n", "", raw)
        raw = re.sub(r"\\n```$", "", raw)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"items": parsed}
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
            return {"items": parsed}
        except json.JSONDecodeError:
            return {}

    return {}
