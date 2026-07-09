"""LLM-assisted planning over constrained template and problem candidate spaces."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.local_sql_runner import invoke_ai_cli, resolve_ai_cli_command
from src.logging.run_artifacts import RunArtifactWriter
from src.usage.logger import UsageCSVLogger, UsageLogRecord
from src.usage.pricing import calculate_cost_usd, resolve_model_pricing
from src.usage.tracker import UsageTracker

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage


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


@dataclass
class ProblemPlannerConfig:
    model_name: str
    request_timeout_seconds: int = 60
    provider_retries: int = 2
    invoke_retries: int = 2


@dataclass
class CLIProblemPlannerConfig:
    model_name: str
    command: str
    cwd: Path
    request_timeout_seconds: int = 420
    invoke_retries: int = 2


class LLMProblemPlanner:
    """Use an LLM as a constrained selector, not an open-ended enumerator."""

    def __init__(
        self,
        *,
        model_name: str,
        dataset_id: str,
        run_id: str,
        usage_logger: UsageCSVLogger | None = None,
        pricing_config: dict[str, Any] | None = None,
    ) -> None:
        timeout_seconds = int(os.getenv("GROUNDING_PLANNER_TIMEOUT_SECONDS", "60"))
        provider_retries = int(os.getenv("GROUNDING_PLANNER_PROVIDER_RETRIES", "2"))
        invoke_retries = max(1, int(os.getenv("GROUNDING_PLANNER_INVOKE_RETRIES", "2")))
        self.config = ProblemPlannerConfig(
            model_name=model_name,
            request_timeout_seconds=timeout_seconds,
            provider_retries=provider_retries,
            invoke_retries=invoke_retries,
        )
        from langchain.chat_models import init_chat_model

        self.model = init_chat_model(
            model_name,
            timeout=timeout_seconds,
            max_retries=provider_retries,
        )
        self.dataset_id = dataset_id
        self.run_id = run_id
        self.usage_logger = usage_logger
        self.pricing = resolve_model_pricing(model_name, pricing_config) if pricing_config is not None else None

    def _record_usage(self, *, response: Any, module: str, question: str) -> None:
        if self.usage_logger is None or self.pricing is None:
            return
        tracker = UsageTracker()
        tracker.add_message(response)
        snapshot = tracker.snapshot
        if snapshot.api_calls <= 0:
            return
        cost = calculate_cost_usd(
            snapshot.input_tokens,
            snapshot.output_tokens,
            self.pricing,
            cached_input_tokens=snapshot.cached_input_tokens,
        )
        self.usage_logger.append(
            UsageLogRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                run_id=self.run_id,
                dataset_id=self.dataset_id,
                phase="grounding_planner",
                module=module,
                question=question,
                model=self.config.model_name,
                input_tokens=snapshot.input_tokens,
                output_tokens=snapshot.output_tokens,
                total_tokens=snapshot.total_tokens,
                cost_usd=cost,
            )
        )

    def _invoke_json(self, *, system_prompt: str, user_prompt: str, module: str, question: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.config.invoke_retries + 1):
            try:
                response = self.model.invoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                self._record_usage(response=response, module=module, question=question)
                content = response.content if isinstance(response.content, str) else json.dumps(response.content, ensure_ascii=False)
                return parse_json_response(content)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.config.invoke_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        if last_exc is not None:
            raise last_exc
        return {}

    def select_templates(
        self,
        *,
        dataset_id: str,
        dataset_summary: dict[str, Any],
        candidates: list[dict[str, Any]],
        min_templates: int,
        target_templates: int,
    ) -> list[str]:
        system_prompt = (
            "You are selecting dataset-specific workload templates from a constrained candidate pool.\n"
            "Your task is NOT to invent new templates. Only choose from the provided template_ids.\n"
            "Prefer templates that are:\n"
            "- natural for the dataset structure,\n"
            "- production-like,\n"
            "- collectively diverse across families,\n"
            "- capable of producing multiple non-trivial problem instances.\n"
            "Do not over-select niche, overly statistical, or overly redundant templates.\n"
            "Return strict JSON with one field: selected_template_ids."
        )
        user_prompt = json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_summary": dataset_summary,
                "selection_rule": {
                    "min_templates": min_templates,
                    "target_templates": target_templates,
                    "must_select_only_from_candidates": True,
                },
                "candidate_templates": candidates,
                "output_schema": {
                    "selected_template_ids": ["template_id_1", "template_id_2"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self._invoke_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            module="select_templates",
            question=f"select_templates:{dataset_id}",
        )
        selected = payload.get("selected_template_ids") or []
        if not isinstance(selected, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        valid_ids = {row["template_id"] for row in candidates}
        for value in selected:
            if not isinstance(value, str):
                continue
            if value not in valid_ids or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def select_problem_ids(
        self,
        *,
        dataset_id: str,
        template_summary: dict[str, Any],
        candidate_items: list[dict[str, Any]],
        min_problems: int,
        max_problems: int,
    ) -> list[str]:
        system_prompt = (
            "You are selecting problem instances for a single SQL template from a constrained candidate pool.\n"
            "Do NOT invent new problem instances. Only choose from the provided question_ids.\n"
            "Prefer a set that:\n"
            "- respects the template's can_vary/must_fix contract,\n"
            "- spans multiple meaningful parameter or binding choices,\n"
            "- avoids near-duplicates,\n"
            "- remains production-like rather than synthetic or repetitive.\n"
            "Return strict JSON with one field: selected_question_ids."
        )
        user_prompt = json.dumps(
            {
                "dataset_id": dataset_id,
                "template_summary": template_summary,
                "selection_rule": {
                    "min_problems": min_problems,
                    "max_problems": max_problems,
                    "must_select_only_from_candidates": True,
                },
                "candidate_problem_instances": candidate_items,
                "output_schema": {
                    "selected_question_ids": ["question_id_1", "question_id_2"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self._invoke_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            module="select_problem_ids",
            question=f"select_problems:{dataset_id}:{template_summary.get('template_id')}",
        )
        selected = payload.get("selected_question_ids") or []
        if not isinstance(selected, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        valid_ids = {row["question_id"] for row in candidate_items}
        for value in selected:
            if not isinstance(value, str):
                continue
            if value not in valid_ids or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized


class CLIProblemPlanner:
    """Use a local AI CLI to select templates and generate problem instances."""

    def __init__(
        self,
        *,
        model_name: str,
        dataset_id: str,
        run_id: str,
        project_root: Path,
        ai_cli_preset: str = "codex",
        ai_cli_command: str = "",
        usage_logger: UsageCSVLogger | None = None,
        pricing_config: dict[str, Any] | None = None,
        artifact_writer: RunArtifactWriter | None = None,
        request_timeout_seconds: int | None = None,
        invoke_retries: int | None = None,
    ) -> None:
        timeout_seconds = request_timeout_seconds or int(os.getenv("GROUNDING_PLANNER_TIMEOUT_SECONDS", "420"))
        retry_count = max(1, invoke_retries or int(os.getenv("GROUNDING_PLANNER_INVOKE_RETRIES", "2")))
        self.config = CLIProblemPlannerConfig(
            model_name=model_name,
            command=resolve_ai_cli_command(
                preset=ai_cli_preset,
                custom_command=ai_cli_command,
                project_root=project_root,
                model=model_name,
            ),
            cwd=project_root,
            request_timeout_seconds=timeout_seconds,
            invoke_retries=retry_count,
        )
        self.dataset_id = dataset_id
        self.run_id = run_id
        self.usage_logger = usage_logger
        self.pricing = resolve_model_pricing(model_name, pricing_config) if pricing_config is not None else None
        self.artifact_writer = artifact_writer
        self._call_counter = 0
        self.summary: dict[str, Any] = {
            "planner_kind": "cli",
            "model": model_name,
            "command": self.config.command,
            "request_timeout_seconds": timeout_seconds,
            "invoke_retries": retry_count,
            "calls": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "by_module": {},
        }

    def _record_usage(self, *, usage: dict[str, Any] | None, module: str, question: str) -> None:
        if not isinstance(usage, dict):
            return
        input_tokens = int(usage.get("input_tokens") or 0)
        cached_input_tokens = int(usage.get("cached_input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
        cost_usd = 0.0
        if self.pricing is not None:
            cost_usd = calculate_cost_usd(
                input_tokens,
                output_tokens,
                self.pricing,
                cached_input_tokens=cached_input_tokens,
            )
        if self.usage_logger is not None:
            self.usage_logger.append(
                UsageLogRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    run_id=self.run_id,
                    dataset_id=self.dataset_id,
                    phase="grounding_planner_cli",
                    module=module,
                    question=question,
                    model=self.config.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                )
            )

        self.summary["calls"] += 1
        self.summary["input_tokens"] += input_tokens
        self.summary["cached_input_tokens"] += cached_input_tokens
        self.summary["output_tokens"] += output_tokens
        self.summary["total_tokens"] += total_tokens
        self.summary["cost_usd"] += cost_usd
        by_module = self.summary["by_module"]
        if module not in by_module:
            by_module[module] = {
                "calls": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        by_module[module]["calls"] += 1
        by_module[module]["input_tokens"] += input_tokens
        by_module[module]["cached_input_tokens"] += cached_input_tokens
        by_module[module]["output_tokens"] += output_tokens
        by_module[module]["total_tokens"] += total_tokens
        by_module[module]["cost_usd"] += cost_usd

    def _write_artifacts(
        self,
        *,
        call_id: int,
        module: str,
        prompt: str,
        result: dict[str, Any],
        payload: dict[str, Any] | None,
    ) -> None:
        if self.artifact_writer is None:
            return
        stem = f"planner/{call_id:02d}_{module}"
        self.artifact_writer.write_text(f"{stem}_prompt.txt", prompt)
        self.artifact_writer.write_text(f"{stem}_response.raw.txt", result.get("stdout", ""))
        self.artifact_writer.write_text(
            f"{stem}_response.txt",
            str(((result.get("parsed_output") or {}).get("text")) or ""),
        )
        self.artifact_writer.write_text(f"{stem}_stderr.txt", result.get("stderr", ""))
        metadata = {
            "module": module,
            "command": result.get("command"),
            "returncode": result.get("returncode"),
            "elapsed_ms": result.get("elapsed_ms"),
            "started_at": result.get("started_at"),
            "ended_at": result.get("ended_at"),
            "prompt_metrics": result.get("prompt_metrics"),
            "stdout_metrics": result.get("stdout_metrics"),
            "stderr_metrics": result.get("stderr_metrics"),
            "parsed_output": result.get("parsed_output"),
            "parsed_payload": payload or {},
        }
        self.artifact_writer.write_json(f"{stem}.metadata.json", metadata)

    def _invoke_json(self, *, system_prompt: str, user_prompt: str, module: str, question: str) -> dict[str, Any]:
        prompt = (
            f"{system_prompt}\n\n"
            "Return only a single JSON object that matches the requested schema. "
            "Do not wrap it in markdown fences or add commentary.\n\n"
            f"{user_prompt}"
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.config.invoke_retries + 1):
            call_id = self._call_counter + 1
            self._call_counter = call_id
            try:
                result = invoke_ai_cli(
                    command=self.config.command,
                    prompt=prompt,
                    cwd=self.config.cwd,
                    timeout_seconds=self.config.request_timeout_seconds,
                    model_hint=self.config.model_name,
                )
                parsed_text = str(((result.get("parsed_output") or {}).get("text")) or "")
                payload = parse_json_response(parsed_text)
                self._write_artifacts(
                    call_id=call_id,
                    module=module,
                    prompt=prompt,
                    result=result,
                    payload=payload,
                )
                if not payload:
                    raise ValueError(f"planner CLI returned non-JSON payload for module `{module}`")
                self._record_usage(
                    usage=(result.get("parsed_output") or {}).get("usage"),
                    module=module,
                    question=question,
                )
                return payload
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.config.invoke_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        if last_exc is not None:
            raise last_exc
        return {}

    def select_templates(
        self,
        *,
        dataset_id: str,
        dataset_summary: dict[str, Any],
        candidates: list[dict[str, Any]],
        min_templates: int,
        target_templates: int,
    ) -> list[str]:
        system_prompt = (
            "You are choosing SQL workload templates for a dataset.\n"
            "The preprocessing shortlist is only a reference, not a hard decision.\n"
            "You must choose only from the provided 36 core template_ids.\n"
            "Select a production-like, diverse set that fits the dataset and can support multiple realistic problems.\n"
            "Avoid redundant templates unless they cover clearly distinct workload shapes.\n"
            "Return JSON with one field: selected_template_ids."
        )
        user_prompt = json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_summary": dataset_summary,
                "selection_rule": {
                    "min_templates": min_templates,
                    "target_templates": target_templates,
                    "must_select_only_from_candidates": True,
                },
                "candidate_templates": candidates,
                "output_schema": {
                    "selected_template_ids": ["template_id_1", "template_id_2"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self._invoke_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            module="select_templates_cli",
            question=f"cli_select_templates:{dataset_id}",
        )
        selected = payload.get("selected_template_ids") or []
        if not isinstance(selected, list):
            return []
        valid_ids = {row["template_id"] for row in candidates}
        normalized: list[str] = []
        seen: set[str] = set()
        for value in selected:
            if not isinstance(value, str):
                continue
            if value not in valid_ids or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def generate_problems(
        self,
        *,
        dataset_id: str,
        dataset_summary: dict[str, Any],
        template_summary: dict[str, Any],
        reference_items: list[dict[str, Any]],
        min_problems: int,
        max_problems: int,
    ) -> list[dict[str, Any]]:
        system_prompt = (
            "You are generating SQL problem instances for one already-chosen SQL template.\n"
            "The heuristic reference problems are suggestions only; do not simply copy them unless they are already ideal.\n"
            "You may create new bindings, but they must stay inside the provided dataset schema and template contract.\n"
            "Each generated problem must be realistic, distinct, and executable under SQLite-style single-table semantics.\n"
            "Respect required_roles, can_vary, and must_fix.\n"
            "Return JSON with one field: problems."
        )
        user_prompt = json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_summary": dataset_summary,
                "template_summary": template_summary,
                "generation_rule": {
                    "min_problems": min_problems,
                    "max_problems": max_problems,
                    "expected_sql_count_per_problem": 1,
                    "reference_problems_are_hints_only": True,
                },
                "reference_problem_candidates": reference_items,
                "output_schema": {
                    "problems": [
                        {
                            "bindings": {"group_col": "example_col"},
                            "variation_axes": ["group_col"],
                            "notes": ["optional short note"],
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self._invoke_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            module=f"generate_problems_{template_summary.get('template_id')}",
            question=f"cli_generate_problems:{dataset_id}:{template_summary.get('template_id')}",
        )
        raw_items = payload.get("problems")
        if not isinstance(raw_items, list):
            raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            bindings = item.get("bindings")
            if not isinstance(bindings, dict):
                continue
            variation_axes = item.get("variation_axes")
            if not isinstance(variation_axes, list):
                variation_axes = []
            notes = item.get("notes")
            if not isinstance(notes, list):
                notes = []
            normalized.append(
                {
                    "bindings": bindings,
                    "variation_axes": [str(value).strip() for value in variation_axes if str(value).strip()],
                    "notes": [str(value).strip() for value in notes if str(value).strip()],
                }
            )
        return normalized
