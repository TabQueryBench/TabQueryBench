from __future__ import annotations

import io
import os

import httpx
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tqb_query.llm.zai import (
    ZAI_BASE_URL,
    ZAI_CODING_BASE_URL,
    ZAIChatClient,
    ZAIChatResult,
    ZAIError,
    ZAIQuotaError,
    _redact,
    load_zai_api_key,
    resolve_zai_base_url,
)
from tqb_query.workload_grounding.problem_planner import PlannerAbortError, PlannerQuotaError, ZAIProblemPlanner


class FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _reply(text: str) -> ZAIChatResult:
    return ZAIChatResult(
        text=text,
        reasoning_text="",
        usage={"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 5, "total_tokens": 15},
        response_id="r1",
        finish_reason="stop",
    )


def _planner(client: FakeClient) -> ZAIProblemPlanner:
    return ZAIProblemPlanner(model_name="glm-5.3", dataset_id="c-test", run_id="run", client=client, invoke_retries=2)


class ZAIPlannerTest(unittest.TestCase):
    def test_binding_uses_json_mode_and_records_usage(self) -> None:
        client = FakeClient([_reply('{"bindings": {"group_col": "category"}}')])
        planner = _planner(client)
        bindings = planner.bind_template_placeholders({"case_id": "x", "template": {"template_id": "t"}})
        self.assertEqual(bindings, {"group_col": "category"})
        self.assertTrue(client.calls[0]["json_mode"])
        self.assertEqual(client.calls[0]["messages"][0]["role"], "system")
        self.assertEqual(planner.summary["calls"], 1)
        self.assertEqual(planner.summary["total_tokens"], 15)

    def test_non_json_reply_is_retried(self) -> None:
        client = FakeClient([_reply("not json"), _reply('{"bindings": {}}')])
        with patch("tqb_query.workload_grounding.problem_planner.time.sleep"):
            self.assertEqual(_planner(client).bind_template_placeholders({"case_id": "x", "template": {}}), {})
        self.assertEqual(len(client.calls), 2)

    def test_error_classes_map_to_run_control(self) -> None:
        with self.assertRaises(PlannerQuotaError):
            _planner(FakeClient([ZAIQuotaError("HTTP 429 code=1113", status_code=429, error_code="1113")]))._invoke_json(
                system_prompt="s", user_prompt="u", module="m", question="q"
            )
        with self.assertRaises(PlannerAbortError):
            _planner(FakeClient([ZAIError("HTTP 401 code=1000", status_code=401, error_code="1000")]))._invoke_json(
                system_prompt="s", user_prompt="u", module="m", question="q"
            )
        with self.assertRaises(ZAIError) as ctx:
            _planner(FakeClient([ZAIError("HTTP 400 code=1261", status_code=400, error_code="1261")]))._invoke_json(
                system_prompt="s", user_prompt="u", module="m", question="q"
            )
        self.assertNotIsInstance(ctx.exception, PlannerQuotaError)

    def test_single_wrapper_key_is_unwrapped(self) -> None:
        client = FakeClient([_reply('{"answer": {"selected_template_ids": ["tpl_a", "tpl_b"]}}')])
        selected = _planner(client).select_templates_for_binding(
            dataset_id="c-test", dataset_summary={}, candidates=[], min_templates=1, target_templates=2, family_minimums={}
        )
        self.assertEqual(selected, ["tpl_a", "tpl_b"])
        client = FakeClient([_reply('{"result": {"bindings": {"group_col": "category"}}}')])
        self.assertEqual(
            _planner(client).bind_template_placeholders({"case_id": "x", "template": {}}), {"group_col": "category"}
        )

    def test_selection_without_id_list_fails_loudly(self) -> None:
        client = FakeClient([_reply('{"templates": ["tpl_a"]}')])
        with self.assertRaises(ValueError):
            _planner(client).select_templates_for_binding(
                dataset_id="c-test", dataset_summary={}, candidates=[], min_templates=1, target_templates=2, family_minimums={}
            )

    def test_accepts_every_zai_model_and_rejects_others(self) -> None:
        for model in ["glm-5.2", "glm-5.3", "glm-5.3-flash"]:
            planner = ZAIProblemPlanner(model_name=model, dataset_id="c", run_id="r", client=FakeClient([]))
            self.assertEqual(planner.config.model_name, model)
            self.assertIn(f"model={model}", planner.config.command)
            self.assertEqual(planner.summary["model"], model)
        with self.assertRaises(ValueError):
            ZAIProblemPlanner(model_name="glm-4.5", dataset_id="c", run_id="r", client=FakeClient([]))


class ZAIClientTest(unittest.TestCase):
    def test_key_from_env_then_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            key_file = Path(temp_dir) / "env"
            key_file.write_text("export ZAI_API_KEY='file-key'\n", encoding="utf-8")
            with patch.dict(os.environ, {"ZAI_API_KEY": "env-key", "ZAI_API_KEY_FILE": str(key_file)}):
                self.assertEqual(load_zai_api_key(), "env-key")
            with patch.dict(os.environ, {"ZAI_API_KEY": "", "ZAI_API_KEY_FILE": str(key_file)}):
                self.assertEqual(load_zai_api_key(), "file-key")
            with patch.dict(os.environ, {"ZAI_API_KEY": "", "ZAI_API_KEY_FILE": str(key_file) + ".missing"}), patch(
                "tqb_query.llm.zai.DEFAULT_KEY_FILE", Path(temp_dir) / "none"
            ):
                with self.assertRaises(ZAIError):
                    load_zai_api_key()

    def test_base_url_is_limited_to_official_endpoints(self) -> None:
        with patch.dict(os.environ, {"ZAI_API_BASE": ""}):
            self.assertEqual(resolve_zai_base_url(), ZAI_CODING_BASE_URL)
        with patch.dict(os.environ, {"ZAI_API_BASE": "General"}):
            self.assertEqual(resolve_zai_base_url(), ZAI_BASE_URL)
        with patch.dict(os.environ, {"ZAI_API_BASE": "https://evil.example/v1"}):
            with self.assertRaises(ZAIError):
                resolve_zai_base_url()

    def test_retries_rate_limit_and_never_logs_key(self) -> None:
        from openai import RateLimitError

        secret = "sk-secret-value"
        response = [
            SimpleNamespace(
                id="r1",
                model="glm-5.3",
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="o", reasoning_content="think"), finish_reason=None)],
            ),
            SimpleNamespace(
                id="r1",
                model="glm-5.3",
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4, prompt_tokens_details=None),
                choices=[SimpleNamespace(delta=SimpleNamespace(content="k", reasoning_content=None), finish_reason="stop")],
            ),
        ]

        def broken_stream():
            yield response[0]
            raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

        http_response = SimpleNamespace(status_code=429, headers={}, request=None)
        rate_limited = RateLimitError(
            f"rate limited {secret}",
            response=http_response,
            body={"error": {"code": "1302", "message": f"Rate limit reached {secret}"}},
        )
        log = io.StringIO()
        client = object.__new__(ZAIChatClient)
        client._api_key = secret
        client.model = "glm-5.3"
        client.max_attempts = 3
        client.max_backoff_seconds = 0
        client.log_stream = log
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=FakeClient([rate_limited, broken_stream(), response]).chat))
        )
        with patch("tqb_query.llm.zai.time.sleep"), patch("tqb_query.llm.zai.random.uniform", return_value=0):
            result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result.text, "ok")
        self.assertEqual(result.reasoning_text, "think")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage["total_tokens"], 4)
        self.assertEqual(result.attempts, 3)
        self.assertIn("code=1302", log.getvalue())
        self.assertIn("RemoteProtocolError", log.getvalue())
        self.assertNotIn(secret, log.getvalue())
        self.assertEqual(_redact(f"a{secret}b", secret), "a***b")


if __name__ == "__main__":
    unittest.main()
