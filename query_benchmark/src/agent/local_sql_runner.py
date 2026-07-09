"""Local SQL runners for template-grounded questions.

This module intentionally avoids LangChain imports.  It supports two non-API
execution paths:

- ``cli``: ask an external AI CLI to produce SQLite SQL, then execute locally.
- ``template``: deterministically instantiate the planned template skeleton.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.logging.run_artifacts import RunArtifactWriter

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


SQL_CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
TRANSIENT_PATH_UPDATE_PATTERNS = (
    "could not update path",
    "os error 5",
)
NETWORK_BLOCKED_AI_CLI_PATTERNS = (
    "os error 10013",
    "error sending request for url",
    "failed to connect to websocket",
    "stream disconnected before completion",
)

COUNT_AGGREGATE_FALLBACK_IDS = {
    "tpl_h2o_group_sum",
    "tpl_h2o_two_dimensional_group_sum",
    "tpl_tpcds_topk_group_sum",
    "tpl_tpcds_within_group_share",
    "tpl_tpch_relative_total_threshold",
    "tpl_tpch_max_aggregate_winner",
    "tpl_tpch_thresholded_group_ranking",
    "tpl_tail_weighted_topk_sum",
}


@dataclass
class LocalRunnerResult:
    final_answer: str
    generated_sqls: list[str]
    query_results: list[dict[str, Any]]
    usage_summary: dict[str, Any]


class AISQLCommandError(RuntimeError):
    """Raised when an external AI CLI command fails."""

    def __init__(self, message: str, *, event: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.event = event or {}


def is_transient_ai_cli_bootstrap_error(message: str) -> bool:
    lowered = str(message or "").lower()
    return any(pattern in lowered for pattern in TRANSIENT_PATH_UPDATE_PATTERNS)


def retry_sleep_seconds_for_ai_cli_error(*, attempt: int, error_message: str) -> float:
    if is_transient_ai_cli_bootstrap_error(error_message):
        return min(15.0 * attempt, 45.0)
    return float(min(2 ** (attempt - 1), 8))


def is_network_blocked_ai_cli_error(message: str) -> bool:
    lowered = str(message or "").lower()
    return any(pattern in lowered for pattern in NETWORK_BLOCKED_AI_CLI_PATTERNS)


def estimate_token_count(text: str, model_hint: str = "") -> int | None:
    if not text or tiktoken is None:
        return None
    encoding = None
    if model_hint:
        try:
            encoding = tiktoken.encoding_for_model(model_hint)
        except KeyError:
            encoding = None
    if encoding is None:
        try:
            encoding = tiktoken.get_encoding("o200k_base")
        except Exception:  # pragma: no cover - defensive fallback
            return None
    return len(encoding.encode(text))


def text_metrics(text: str, model_hint: str = "") -> dict[str, Any]:
    return {
        "chars": len(text),
        "bytes_utf8": len(text.encode("utf-8")),
        "lines": len(text.splitlines()),
        "estimated_tokens": estimate_token_count(text, model_hint=model_hint),
    }


def parse_ai_cli_json_events(text: str) -> list[dict[str, Any]] | None:
    events: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "type" not in obj:
            return None
        events.append(obj)
    return events or None


def extract_text_from_ai_output(text: str) -> str:
    events = parse_ai_cli_json_events(text)
    if events:
        agent_messages: list[str] = []
        for event in events:
            if event.get("type") != "item.completed":
                continue
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                agent_messages.append(item["text"])
        if agent_messages:
            return agent_messages[-1].strip()
    return text.strip()


def extract_usage_from_ai_output(text: str) -> dict[str, Any] | None:
    events = parse_ai_cli_json_events(text)
    if not events:
        return None
    usage: dict[str, Any] | None = None
    for event in events:
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
    return usage


def quote_identifier(identifier: Any) -> str:
    escaped = str(identifier).replace('"', '""')
    return f'"{escaped}"'


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def numeric_literal(value: Any, fallback: float = 0.0) -> str:
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        return repr(float(fallback))


def resolve_ai_cli_command(
    *,
    preset: str,
    custom_command: str,
    project_root: Path,
    model: str,
) -> str:
    """Build a shell command that reads the prompt from stdin."""
    if custom_command.strip():
        return custom_command.strip()

    model_arg = ""
    if model.strip():
        model_arg = f" -m {model.strip()}"

    if preset == "codex":
        return f'codex exec --disable plugins --sandbox read-only --cd "{project_root}"{model_arg} --json -'
    if preset == "claude":
        claude_model_arg = f" --model {model.strip()}" if model.strip() else ""
        return f"claude --print --input-format text --permission-mode dontAsk --tools \"\"{claude_model_arg}"
    if preset == "gemini":
        # Gemini CLI flags vary by version; use a conservative stdin-oriented
        # command.  Users can override with --ai-cli-command when needed.
        gemini_model_arg = f" --model {model.strip()}" if model.strip() else ""
        return f"gemini{gemini_model_arg}"
    raise ValueError("custom AI CLI command is required when --ai-cli-preset custom is used")


def invoke_ai_cli(
    *,
    command: str,
    prompt: str,
    cwd: Path,
    timeout_seconds: int,
    model_hint: str = "",
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    if os.getenv("SQLAGENT_ASSUME_CODEX_BLOCKED", "").strip() == "1" and "codex" in command.lower():
        ended_at = datetime.now(timezone.utc).isoformat()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        stderr_text = (
            "stream disconnected before completion: "
            "error sending request for url (https://api.openai.com/v1/responses); os error 10013"
        )
        result = {
            "command": command,
            "returncode": 1,
            "stdout": "",
            "stderr": stderr_text,
            "elapsed_ms": elapsed_ms,
            "started_at": started_at,
            "ended_at": ended_at,
            "prompt_metrics": text_metrics(prompt, model_hint=model_hint),
            "stdout_metrics": text_metrics("", model_hint=model_hint),
            "stderr_metrics": text_metrics(stderr_text, model_hint=model_hint),
            "parsed_output": {
                "format": "plain_text",
                "text": "",
                "text_metrics": text_metrics("", model_hint=model_hint),
                "usage": None,
            },
        }
        raise AISQLCommandError(
            f"AI CLI command failed with exit code 1: {stderr_text}",
            event=result,
        )
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd),
        shell=True,
        timeout=timeout_seconds,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    ended_at = datetime.now(timezone.utc).isoformat()
    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    parsed_text = extract_text_from_ai_output(stdout_text)
    parsed_usage = extract_usage_from_ai_output(stdout_text)
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "elapsed_ms": elapsed_ms,
        "started_at": started_at,
        "ended_at": ended_at,
        "prompt_metrics": text_metrics(prompt, model_hint=model_hint),
        "stdout_metrics": text_metrics(stdout_text, model_hint=model_hint),
        "stderr_metrics": text_metrics(stderr_text, model_hint=model_hint),
        "parsed_output": {
            "format": "jsonl_events" if parse_ai_cli_json_events(stdout_text) else "plain_text",
            "text": parsed_text,
            "text_metrics": text_metrics(parsed_text, model_hint=model_hint),
            "usage": parsed_usage,
        },
    }
    if completed.returncode != 0:
        raise AISQLCommandError(
            f"AI CLI command failed with exit code {completed.returncode}: {stderr_text.strip()}",
            event=result,
        )
    return result


def _strip_markdown_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = JSON_CODE_BLOCK_RE.search(stripped)
        if match:
            return match.group(1).strip()
    return stripped


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = _strip_markdown_json(text)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def extract_sql_from_ai_output(text: str) -> str:
    payload = _parse_json_object(text)
    if payload:
        for key in ("sql", "query", "sqlite_sql"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        value = payload.get("sql_queries")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()

    block = SQL_CODE_BLOCK_RE.search(text)
    if block:
        return block.group(1).strip()

    lines = [line for line in text.splitlines() if line.strip()]
    sql_lines: list[str] = []
    capture = False
    for line in lines:
        lowered = line.strip().lower()
        if lowered.startswith(("select ", "with ", "-- template_id:")):
            capture = True
        if capture:
            sql_lines.append(line)
    if sql_lines:
        return "\n".join(sql_lines).strip()
    return text.strip()


def _without_leading_comments(sql: str) -> str:
    lines = []
    for line in sql.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _split_leading_comments(sql: str) -> tuple[str, str]:
    prefix_lines: list[str] = []
    body_lines: list[str] = []
    in_prefix = True
    for line in sql.splitlines():
        stripped = line.strip()
        if in_prefix and (not stripped or stripped.startswith("--")):
            prefix_lines.append(line)
            continue
        in_prefix = False
        body_lines.append(line)
    return "\n".join(prefix_lines).strip(), "\n".join(body_lines).strip()


def normalize_sql_for_sqlite(sql: str) -> tuple[str, list[str]]:
    prefix, body = _split_leading_comments(sql)
    if not body:
        return sql, []

    normalized = body
    notes: list[str] = []
    if re.search(r"(?is)^\s*with\b", normalized) and not re.search(r"(?is)^\s*with\s+recursive\b", normalized):
        if re.search(r"(?is),\s*recursive\s+[A-Za-z_\"(]", normalized):
            normalized = re.sub(r"(?is)^\s*with\b", "WITH RECURSIVE", normalized, count=1)
            normalized = re.sub(r"(?is),\s*recursive\s+", ", ", normalized)
            notes.append("moved_recursive_keyword_to_with_clause")

    rebuilt = normalized if not prefix else f"{prefix}\n{normalized}"
    return rebuilt.strip(), notes


def validate_readonly_sql(sql: str) -> None:
    body = _without_leading_comments(sql).lstrip(" \ufeff").lower()
    if not body.startswith(("select", "with")):
        raise ValueError("only SELECT/WITH SQL is allowed in local runners")
    blocked = re.search(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum)\b", body)
    if blocked:
        raise ValueError(f"blocked non-readonly SQL token: {blocked.group(1)}")


def _sqlite_row_to_jsonable(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def execute_sqlite_query(
    *,
    db_path: Path,
    sql: str,
    row_limit: int,
    timeout_ms: int,
) -> dict[str, Any]:
    validate_readonly_sql(sql)
    started = time.perf_counter()
    deadline = started + max(timeout_ms, 1) / 1000
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.create_function("SQRT", 1, lambda x: None if x is None else math.sqrt(float(x)))
    conn.create_function("sqrt", 1, lambda x: None if x is None else math.sqrt(float(x)))

    def progress_handler() -> int:
        return 1 if time.perf_counter() > deadline else 0

    conn.set_progress_handler(progress_handler, 1000)
    try:
        cursor = conn.execute(sql)
        rows = [_sqlite_row_to_jsonable(row) for row in cursor.fetchmany(row_limit)]
        columns = [description[0] for description in cursor.description or []]
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "query": sql,
            "columns": columns,
            "rows": rows,
            "row_count_returned": len(rows),
            "row_limit": row_limit,
            "truncated": len(rows) >= row_limit,
            "elapsed_ms": elapsed_ms,
        }
    finally:
        conn.close()


def build_schema_snapshot(
    *,
    db_path: Path,
    table_name: str,
    sample_rows: int = 5,
) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    quoted_table = quote_identifier(table_name)
    try:
        columns = [
            {
                "name": row["name"],
                "type": row["type"],
                "notnull": bool(row["notnull"]),
                "pk": bool(row["pk"]),
            }
            for row in conn.execute(f"PRAGMA table_info({quoted_table})")
        ]
        rows = [
            _sqlite_row_to_jsonable(row)
            for row in conn.execute(f"SELECT * FROM {quoted_table} LIMIT ?", (sample_rows,))
        ]
        row_count = conn.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0]
        return {
            "table_name": table_name,
            "quoted_table_name": quoted_table,
            "row_count": row_count,
            "columns": columns,
            "sample_rows": rows,
        }
    finally:
        conn.close()


def ensure_template_comment(sql: str, template_id: str | None) -> str:
    if not template_id:
        return sql.strip()
    if re.search(r"^\s*--\s*template_id:", sql, flags=re.MULTILINE):
        return sql.strip()
    return f"-- template_id: {template_id}\n{sql.strip()}"


def _short_template_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in selection.get("shortlist", []):
        rows.append(
            {
                "template_id": item.get("template_id"),
                "template_name": item.get("template_name"),
                "primary_family": item.get("primary_family"),
                "portability": item.get("portability"),
                "sql_skeleton": item.get("sql_skeleton"),
                "required_roles": item.get("required_roles"),
            }
        )
    return rows


def build_cli_sql_prompt(
    *,
    dataset_id: str,
    question: str,
    dataset_context: str,
    selection: dict[str, Any],
    question_record: dict[str, Any] | None,
    schema_snapshot: dict[str, Any],
    previous_sql: str | None = None,
    previous_error: str | None = None,
) -> str:
    problem_payload = {
        "dataset_id": dataset_id,
        "question": question,
        "planned_template_id": (question_record or {}).get("template_id"),
        "bindings": (question_record or {}).get("bindings") or {},
        "can_vary": (question_record or {}).get("can_vary") or [],
        "must_fix": (question_record or {}).get("must_fix") or [],
        "runtime_sql_skeleton": (question_record or {}).get("runtime_sql_skeleton"),
    }
    repair_payload = {}
    if previous_error:
        repair_payload = {
            "previous_sql": previous_sql,
            "sqlite_error": previous_error,
            "repair_instruction": "Return a corrected SQLite query that fixes the error.",
        }
    return (
        "You are generating one SQLite SELECT query for a single-table SQL QA task.\n"
        "Return strict JSON only, with this schema: {\"sql\": \"...\", \"notes\": \"...\"}.\n"
        "Rules:\n"
        "- Use only the provided table and columns.\n"
        "- Do not write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH, DETACH, or VACUUM.\n"
        "- Prefer the planned template and bound roles when provided.\n"
        "- Add a leading SQL comment exactly like: -- template_id: <planned_template_id>.\n"
        "- Generate SQLite-compatible SQL. SQLite does not support PERCENTILE_CONT or STDDEV.\n"
        "- Quote identifiers with double quotes.\n"
        "- Return no markdown and no extra prose.\n\n"
        f"Dataset context:\n{dataset_context}\n\n"
        f"SQLite schema snapshot:\n{json.dumps(schema_snapshot, ensure_ascii=False, indent=2)}\n\n"
        f"Shortlisted templates:\n{json.dumps(_short_template_rows(selection), ensure_ascii=False, indent=2)}\n\n"
        f"Problem instance:\n{json.dumps(problem_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Repair context:\n{json.dumps(repair_payload, ensure_ascii=False, indent=2)}\n"
    )


def build_cli_answer_prompt(
    *,
    question: str,
    sql: str,
    execution: dict[str, Any],
) -> str:
    return (
        "Answer the user's question using only the SQL result preview below.\n"
        "Return concise plain text. Do not invent values not present in the result.\n\n"
        f"Question:\n{question}\n\n"
        f"SQL:\n{sql}\n\n"
        f"Result:\n{json.dumps(execution, ensure_ascii=False, indent=2)}\n"
    )


def local_result_answer(question: str, execution: dict[str, Any]) -> str:
    rows = execution.get("rows") or []
    if not rows:
        return f"No rows were returned for: {question}"
    preview = json.dumps(rows[:5], ensure_ascii=False)
    suffix = " Results were truncated." if execution.get("truncated") else ""
    return f"SQL executed successfully for: {question}\nResult preview: {preview}{suffix}"


def run_ai_cli_sql_question(
    *,
    command: str,
    dataset_id: str,
    question: str,
    dataset_context: str,
    selection: dict[str, Any],
    question_record: dict[str, Any] | None,
    db_path: Path,
    table_name: str,
    artifact_writer: RunArtifactWriter,
    timeout_seconds: int,
    max_retries: int,
    row_limit: int,
    sql_timeout_ms: int,
    answer_mode: str,
    cwd: Path,
    engine_label: str = "cli",
    model_hint: str = "",
) -> LocalRunnerResult:
    schema_snapshot = build_schema_snapshot(db_path=db_path, table_name=table_name)
    generated_sqls: list[str] = []
    query_results: list[dict[str, Any]] = []
    previous_sql: str | None = None
    previous_error: str | None = None
    cli_calls = 0
    template_id = str((question_record or {}).get("template_id") or "")
    cli_elapsed_ms_total = 0.0
    sql_execution_elapsed_ms_total = 0.0
    estimated_input_tokens = 0
    estimated_output_tokens = 0
    actual_input_tokens = 0
    actual_cached_input_tokens = 0
    actual_output_tokens = 0
    actual_usage_available = False
    conversation_rows: list[dict[str, Any]] = []
    last_cli_error: str | None = None

    attempts = max(1, max_retries + 1)
    execution: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        prompt = build_cli_sql_prompt(
            dataset_id=dataset_id,
            question=question,
            dataset_context=dataset_context,
            selection=selection,
            question_record=question_record,
            schema_snapshot=schema_snapshot,
            previous_sql=previous_sql,
            previous_error=previous_error,
        )
        artifact_writer.write_text(f"cli/sql_prompt_attempt_{attempt}.txt", prompt)
        try:
            cli_event = invoke_ai_cli(
                command=command,
                prompt=prompt,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                model_hint=model_hint,
            )
        except AISQLCommandError as exc:
            last_cli_error = str(exc)
            cli_event = exc.event or {}
            cli_calls += 1
            cli_elapsed_ms_total += float(cli_event.get("elapsed_ms") or 0.0)
            prompt_metrics = cli_event.get("prompt_metrics") or text_metrics(prompt, model_hint=model_hint)
            parsed_output = cli_event.get("parsed_output") or {
                "format": "plain_text",
                "text": "",
                "text_metrics": text_metrics("", model_hint=model_hint),
                "usage": {},
            }
            parsed_usage = parsed_output.get("usage") or {}
            estimated_input_tokens += int(prompt_metrics.get("estimated_tokens") or 0)
            estimated_output_tokens += int((parsed_output.get("text_metrics") or {}).get("estimated_tokens") or 0)
            if parsed_usage:
                actual_usage_available = True
                actual_input_tokens += int(parsed_usage.get("input_tokens") or 0)
                actual_cached_input_tokens += int(parsed_usage.get("cached_input_tokens") or 0)
                actual_output_tokens += int(parsed_usage.get("output_tokens") or 0)
            artifact_writer.write_text(f"cli/sql_response_attempt_{attempt}.txt", str(parsed_output.get("text") or ""))
            artifact_writer.write_text(f"cli/sql_response_attempt_{attempt}.raw.txt", str(cli_event.get("stdout") or ""))
            artifact_writer.write_text(f"cli/sql_stderr_attempt_{attempt}.txt", str(cli_event.get("stderr") or ""))
            artifact_writer.write_json(
                f"cli/sql_attempt_{attempt}.metadata.json",
                {
                    "attempt": attempt,
                    "phase": "sql_generation",
                    "command": command,
                    "started_at": cli_event.get("started_at"),
                    "ended_at": cli_event.get("ended_at"),
                    "elapsed_ms": cli_event.get("elapsed_ms"),
                    "returncode": cli_event.get("returncode"),
                    "prompt_metrics": prompt_metrics,
                    "stdout_metrics": cli_event.get("stdout_metrics") or text_metrics(str(cli_event.get("stdout") or ""), model_hint=model_hint),
                    "stderr_metrics": cli_event.get("stderr_metrics") or text_metrics(str(cli_event.get("stderr") or ""), model_hint=model_hint),
                    "parsed_output": {
                        "format": parsed_output.get("format"),
                        "text_metrics": parsed_output.get("text_metrics") or text_metrics("", model_hint=model_hint),
                        "usage": parsed_usage,
                    },
                    "status": "failed",
                    "error": str(exc),
                    "prompt_path": f"cli/sql_prompt_attempt_{attempt}.txt",
                    "response_path": f"cli/sql_response_attempt_{attempt}.txt",
                    "raw_response_path": f"cli/sql_response_attempt_{attempt}.raw.txt",
                    "stderr_path": f"cli/sql_stderr_attempt_{attempt}.txt",
                },
            )
            conversation_rows.append(
                {
                    "attempt": attempt,
                    "phase": "sql_generation",
                    "role": "user",
                    "content_path": f"cli/sql_prompt_attempt_{attempt}.txt",
                    "metrics": prompt_metrics,
                }
            )
            conversation_rows.append(
                {
                    "attempt": attempt,
                    "phase": "sql_generation",
                    "role": "assistant",
                    "content_path": f"cli/sql_response_attempt_{attempt}.txt",
                    "raw_content_path": f"cli/sql_response_attempt_{attempt}.raw.txt",
                    "stderr_path": f"cli/sql_stderr_attempt_{attempt}.txt",
                    "metrics": parsed_output.get("text_metrics") or text_metrics("", model_hint=model_hint),
                    "usage": parsed_usage,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            artifact_writer.append_trace(
                {
                    "event_type": "ai_cli_sql_generation_error",
                    "engine": engine_label,
                    "attempt": attempt,
                    "command": command,
                    "returncode": cli_event.get("returncode"),
                    "elapsed_ms": cli_event.get("elapsed_ms"),
                    "started_at": cli_event.get("started_at"),
                    "ended_at": cli_event.get("ended_at"),
                    "prompt_metrics": prompt_metrics,
                    "response_metrics": parsed_output.get("text_metrics") or text_metrics("", model_hint=model_hint),
                    "usage": parsed_usage,
                    "stderr_preview": str(cli_event.get("stderr") or "")[:1000],
                    "stdout_preview": str(parsed_output.get("text") or "")[:1000],
                    "error": str(exc),
                }
            )
            if attempt >= attempts:
                if question_record and question_record.get("template_id") and is_network_blocked_ai_cli_error(str(exc)):
                    break
                raise
            time.sleep(
                retry_sleep_seconds_for_ai_cli_error(
                    attempt=attempt,
                    error_message=str(exc),
                )
            )
            continue
        cli_calls += 1
        cli_elapsed_ms_total += float(cli_event["elapsed_ms"])
        estimated_input_tokens += int(cli_event["prompt_metrics"].get("estimated_tokens") or 0)
        estimated_output_tokens += int(cli_event["parsed_output"]["text_metrics"].get("estimated_tokens") or 0)
        parsed_usage = cli_event["parsed_output"].get("usage") or {}
        if parsed_usage:
            actual_usage_available = True
            actual_input_tokens += int(parsed_usage.get("input_tokens") or 0)
            actual_cached_input_tokens += int(parsed_usage.get("cached_input_tokens") or 0)
            actual_output_tokens += int(parsed_usage.get("output_tokens") or 0)
        artifact_writer.write_text(f"cli/sql_response_attempt_{attempt}.txt", cli_event["parsed_output"]["text"])
        artifact_writer.write_text(f"cli/sql_response_attempt_{attempt}.raw.txt", cli_event["stdout"])
        artifact_writer.write_text(f"cli/sql_stderr_attempt_{attempt}.txt", cli_event["stderr"])
        artifact_writer.write_json(
            f"cli/sql_attempt_{attempt}.metadata.json",
            {
                "attempt": attempt,
                "phase": "sql_generation",
                "command": command,
                "started_at": cli_event["started_at"],
                "ended_at": cli_event["ended_at"],
                "elapsed_ms": cli_event["elapsed_ms"],
                "prompt_metrics": cli_event["prompt_metrics"],
                "stdout_metrics": cli_event["stdout_metrics"],
                "stderr_metrics": cli_event["stderr_metrics"],
                "parsed_output": {
                    "format": cli_event["parsed_output"]["format"],
                    "text_metrics": cli_event["parsed_output"]["text_metrics"],
                    "usage": parsed_usage,
                },
                "prompt_path": f"cli/sql_prompt_attempt_{attempt}.txt",
                "response_path": f"cli/sql_response_attempt_{attempt}.txt",
                "raw_response_path": f"cli/sql_response_attempt_{attempt}.raw.txt",
                "stderr_path": f"cli/sql_stderr_attempt_{attempt}.txt",
            },
        )
        conversation_rows.append(
            {
                "attempt": attempt,
                "phase": "sql_generation",
                "role": "user",
                "content_path": f"cli/sql_prompt_attempt_{attempt}.txt",
                "metrics": cli_event["prompt_metrics"],
            }
        )
        conversation_rows.append(
            {
                "attempt": attempt,
                "phase": "sql_generation",
                "role": "assistant",
                "content_path": f"cli/sql_response_attempt_{attempt}.txt",
                "raw_content_path": f"cli/sql_response_attempt_{attempt}.raw.txt",
                "stderr_path": f"cli/sql_stderr_attempt_{attempt}.txt",
                "metrics": cli_event["parsed_output"]["text_metrics"],
                "usage": parsed_usage,
            }
        )
        artifact_writer.append_trace(
            {
                "event_type": "ai_cli_sql_generation",
                "engine": engine_label,
                "attempt": attempt,
                "command": command,
                "returncode": cli_event["returncode"],
                "elapsed_ms": cli_event["elapsed_ms"],
                "started_at": cli_event["started_at"],
                "ended_at": cli_event["ended_at"],
                "prompt_metrics": cli_event["prompt_metrics"],
                "response_metrics": cli_event["parsed_output"]["text_metrics"],
                "usage": parsed_usage,
                "stderr_preview": cli_event["stderr"][:1000],
                "stdout_preview": cli_event["parsed_output"]["text"][:1000],
            }
        )
        raw_sql = ensure_template_comment(extract_sql_from_ai_output(cli_event["parsed_output"]["text"]), template_id or None)
        sql, normalization_notes = normalize_sql_for_sqlite(raw_sql)
        if normalization_notes:
            artifact_writer.append_trace(
                {
                    "event_type": "sqlite_sql_normalized",
                    "engine": engine_label,
                    "attempt": attempt,
                    "notes": normalization_notes,
                    "original_query": raw_sql,
                    "normalized_query": sql,
                }
            )
        previous_sql = sql
        generated_sqls.append(sql)
        try:
            execution = execute_sqlite_query(
                db_path=db_path,
                sql=sql,
                row_limit=row_limit,
                timeout_ms=sql_timeout_ms,
            )
            query_results.append(
                {
                    "step_index": attempt,
                    "message_index": 0,
                    "node_name": engine_label,
                    "tool_name": "sqlite_query",
                    "query": sql,
                    "result": json.dumps(execution, ensure_ascii=False),
                }
            )
            sql_execution_elapsed_ms_total += float(execution.get("elapsed_ms") or 0.0)
            break
        except Exception as exc:  # noqa: BLE001
            previous_error = str(exc)
            artifact_writer.append_trace(
                {
                    "event_type": "sqlite_query_error",
                    "engine": engine_label,
                    "attempt": attempt,
                    "query": sql,
                    "error": previous_error,
                }
            )
            if attempt >= attempts:
                raise

    if execution is None:
        if question_record and question_record.get("template_id") and is_network_blocked_ai_cli_error(last_cli_error or ""):
            template_lookup = {
                str(item.get("template_id")): dict(item)
                for item in selection.get("shortlist", [])
                if item.get("template_id")
            }
            template_result = run_template_sql_question(
                dataset_id=dataset_id,
                question=question,
                question_record=question_record,
                db_path=db_path,
                table_name=table_name,
                template_lookup=template_lookup,
                artifact_writer=artifact_writer,
                row_limit=row_limit,
                sql_timeout_ms=sql_timeout_ms,
            )
            artifact_writer.append_trace(
                {
                    "event_type": "ai_cli_fallback_to_template",
                    "engine": engine_label,
                    "reason": last_cli_error,
                    "template_id": question_record.get("template_id"),
                }
            )
            fallback_usage = dict(template_result.usage_summary)
            fallback_usage.update(
                {
                    "dataset_id": dataset_id,
                    "model": engine_label,
                    "run_id": artifact_writer.run_id,
                    "ai_cli_calls": cli_calls,
                    "input_tokens": actual_input_tokens if actual_usage_available else estimated_input_tokens,
                    "cached_input_tokens": actual_cached_input_tokens if actual_usage_available else 0,
                    "output_tokens": actual_output_tokens if actual_usage_available else estimated_output_tokens,
                    "total_tokens": (
                        actual_input_tokens + actual_output_tokens
                        if actual_usage_available
                        else estimated_input_tokens + estimated_output_tokens
                    ),
                    "estimated_input_tokens": estimated_input_tokens,
                    "estimated_output_tokens": estimated_output_tokens,
                    "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
                    "usage_source": "ai_cli_network_blocked_template_fallback",
                    "cli_elapsed_ms_total": round(cli_elapsed_ms_total, 2),
                    "sql_execution_elapsed_ms_total": round(
                        sum(float((row.get("result") and json.loads(row["result"]).get("elapsed_ms")) or 0.0) for row in template_result.query_results),
                        2,
                    ),
                    "conversation_log_path": str((artifact_writer.run_dir / "cli" / "conversation.jsonl").resolve()),
                    "note": "Codex CLI was unreachable from the sandbox; executed the planned template deterministically.",
                }
            )
            artifact_writer.write_jsonl("cli/conversation.jsonl", conversation_rows)
            artifact_writer.write_json(
                "cli/session_summary.json",
                {
                    "engine": engine_label,
                    "command": command,
                    "ai_cli_calls": cli_calls,
                    "fallback_reason": last_cli_error,
                    "usage_summary": fallback_usage,
                },
            )
            artifact_writer.write_usage_summary(fallback_usage)
            return LocalRunnerResult(
                final_answer=template_result.final_answer,
                generated_sqls=template_result.generated_sqls,
                query_results=template_result.query_results,
                usage_summary=fallback_usage,
            )
        raise RuntimeError("AI CLI did not produce an executable SQL query.")

    if answer_mode == "ai":
        answer_prompt = build_cli_answer_prompt(question=question, sql=generated_sqls[-1], execution=execution)
        artifact_writer.write_text("cli/answer_prompt.txt", answer_prompt)
        answer_event = invoke_ai_cli(
            command=command,
            prompt=answer_prompt,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            model_hint=model_hint,
        )
        cli_calls += 1
        cli_elapsed_ms_total += float(answer_event["elapsed_ms"])
        estimated_input_tokens += int(answer_event["prompt_metrics"].get("estimated_tokens") or 0)
        estimated_output_tokens += int(answer_event["parsed_output"]["text_metrics"].get("estimated_tokens") or 0)
        answer_usage = answer_event["parsed_output"].get("usage") or {}
        if answer_usage:
            actual_usage_available = True
            actual_input_tokens += int(answer_usage.get("input_tokens") or 0)
            actual_cached_input_tokens += int(answer_usage.get("cached_input_tokens") or 0)
            actual_output_tokens += int(answer_usage.get("output_tokens") or 0)
        artifact_writer.write_text("cli/answer_response.txt", answer_event["parsed_output"]["text"])
        artifact_writer.write_text("cli/answer_response.raw.txt", answer_event["stdout"])
        artifact_writer.write_text("cli/answer_stderr.txt", answer_event["stderr"])
        artifact_writer.write_json(
            "cli/answer_attempt.metadata.json",
            {
                "phase": "answer_generation",
                "command": command,
                "started_at": answer_event["started_at"],
                "ended_at": answer_event["ended_at"],
                "elapsed_ms": answer_event["elapsed_ms"],
                "prompt_metrics": answer_event["prompt_metrics"],
                "stdout_metrics": answer_event["stdout_metrics"],
                "stderr_metrics": answer_event["stderr_metrics"],
                "parsed_output": {
                    "format": answer_event["parsed_output"]["format"],
                    "text_metrics": answer_event["parsed_output"]["text_metrics"],
                    "usage": answer_usage,
                },
                "prompt_path": "cli/answer_prompt.txt",
                "response_path": "cli/answer_response.txt",
                "raw_response_path": "cli/answer_response.raw.txt",
                "stderr_path": "cli/answer_stderr.txt",
            },
        )
        conversation_rows.append(
            {
                "phase": "answer_generation",
                "role": "user",
                "content_path": "cli/answer_prompt.txt",
                "metrics": answer_event["prompt_metrics"],
            }
        )
        conversation_rows.append(
            {
                "phase": "answer_generation",
                "role": "assistant",
                "content_path": "cli/answer_response.txt",
                "raw_content_path": "cli/answer_response.raw.txt",
                "stderr_path": "cli/answer_stderr.txt",
                "metrics": answer_event["parsed_output"]["text_metrics"],
                "usage": answer_usage,
            }
        )
        artifact_writer.append_trace(
            {
                "event_type": "ai_cli_answer_generation",
                "engine": engine_label,
                "command": command,
                "elapsed_ms": answer_event["elapsed_ms"],
                "started_at": answer_event["started_at"],
                "ended_at": answer_event["ended_at"],
                "prompt_metrics": answer_event["prompt_metrics"],
                "response_metrics": answer_event["parsed_output"]["text_metrics"],
                "usage": answer_usage,
                "stderr_preview": answer_event["stderr"][:1000],
                "stdout_preview": answer_event["parsed_output"]["text"][:1000],
            }
        )
        final_answer = answer_event["parsed_output"]["text"].strip()
    else:
        final_answer = local_result_answer(question, execution)

    input_tokens = actual_input_tokens if actual_usage_available else 0
    cached_input_tokens = actual_cached_input_tokens if actual_usage_available else 0
    output_tokens = actual_output_tokens if actual_usage_available else 0
    usage_summary = {
        "dataset_id": dataset_id,
        "model": engine_label,
        "run_id": artifact_writer.run_id,
        "api_calls": 0,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": 0.0,
        "ai_cli_calls": cli_calls,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
        "usage_source": "ai_cli_json_usage" if actual_usage_available else "estimated_only",
        "cli_elapsed_ms_total": round(cli_elapsed_ms_total, 2),
        "sql_execution_elapsed_ms_total": round(sql_execution_elapsed_ms_total, 2),
        "conversation_log_path": str((artifact_writer.run_dir / "cli" / "conversation.jsonl").resolve()),
        "note": (
            "Executed through a local AI CLI with structured usage metadata."
            if actual_usage_available
            else "Executed through a local AI CLI; exact token usage was unavailable, so only estimated token counts are recorded."
        ),
    }
    artifact_writer.write_jsonl("cli/conversation.jsonl", conversation_rows)
    artifact_writer.write_json(
        "cli/session_summary.json",
        {
            "engine": engine_label,
            "command": command,
            "ai_cli_calls": cli_calls,
            "usage_summary": usage_summary,
        },
    )
    artifact_writer.write_generated_sql(generated_sqls)
    artifact_writer.write_query_results(query_results)
    artifact_writer.write_final_answer(final_answer)
    artifact_writer.write_usage_summary(usage_summary)
    return LocalRunnerResult(
        final_answer=final_answer,
        generated_sqls=generated_sqls,
        query_results=query_results,
        usage_summary=usage_summary,
    )


def _count_fallback_sql(template_id: str, bindings: dict[str, Any], table: str) -> str | None:
    if template_id not in COUNT_AGGREGATE_FALLBACK_IDS:
        return None
    group_col = quote_identifier(bindings.get("group_col"))
    group_col_2 = quote_identifier(bindings.get("group_col_2")) if bindings.get("group_col_2") else None
    item_col = quote_identifier(bindings.get("item_col")) if bindings.get("item_col") else None
    predicate_col = quote_identifier(bindings.get("predicate_col")) if bindings.get("predicate_col") else None
    predicate_op = _safe_operator(bindings.get("predicate_op"))
    predicate_value = sql_literal(bindings.get("predicate_value"))
    top_k = int(float(bindings.get("top_k") or 5))
    fraction_threshold = numeric_literal(bindings.get("fraction_threshold"), 0.1)
    threshold = numeric_literal(bindings.get("measure_threshold") or bindings.get("min_support"), 1)
    min_support = numeric_literal(bindings.get("min_support"), 1)

    if template_id == "tpl_h2o_group_sum":
        return f"SELECT {group_col}, COUNT(*) AS total_rows FROM {table} GROUP BY {group_col} ORDER BY total_rows DESC"
    if template_id == "tpl_h2o_two_dimensional_group_sum" and group_col_2:
        return (
            f"SELECT {group_col}, {group_col_2}, COUNT(*) AS total_rows "
            f"FROM {table} GROUP BY {group_col}, {group_col_2} ORDER BY total_rows DESC"
        )
    if template_id == "tpl_tpcds_topk_group_sum" and predicate_col:
        return (
            f"SELECT {group_col}, COUNT(*) AS total_rows FROM {table} "
            f"WHERE {predicate_col} {predicate_op} {predicate_value} "
            f"GROUP BY {group_col} ORDER BY total_rows DESC LIMIT {top_k}"
        )
    if template_id == "tpl_tpcds_within_group_share" and item_col:
        return (
            f"SELECT {group_col}, {item_col}, COUNT(*) AS total_rows, "
            f"COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY {group_col}) AS share_within_group "
            f"FROM {table} GROUP BY {group_col}, {item_col} ORDER BY share_within_group DESC"
        )
    if template_id == "tpl_tpch_relative_total_threshold":
        return (
            "WITH grouped AS ("
            f"SELECT {group_col}, COUNT(*) AS group_value FROM {table} GROUP BY {group_col}"
            "), total AS (SELECT SUM(group_value) AS total_value FROM grouped) "
            f"SELECT g.{group_col}, g.group_value FROM grouped AS g CROSS JOIN total AS t "
            f"WHERE g.group_value > t.total_value * {fraction_threshold} ORDER BY g.group_value DESC"
        )
    if template_id == "tpl_tpch_max_aggregate_winner":
        return (
            "WITH grouped AS ("
            f"SELECT {group_col}, COUNT(*) AS total_rows FROM {table} GROUP BY {group_col}"
            f") SELECT {group_col}, total_rows FROM grouped "
            f"WHERE total_rows = (SELECT MAX(total_rows) FROM grouped) ORDER BY {group_col}"
        )
    if template_id == "tpl_tpch_thresholded_group_ranking":
        return (
            f"SELECT {group_col}, COUNT(*) AS total_rows FROM {table} GROUP BY {group_col} "
            f"HAVING COUNT(*) > {threshold} ORDER BY total_rows DESC LIMIT {top_k}"
        )
    if template_id == "tpl_tail_weighted_topk_sum":
        return (
            f"SELECT {group_col}, COUNT(*) AS weighted_total, COUNT(*) AS support "
            f"FROM {table} GROUP BY {group_col} HAVING COUNT(*) >= {min_support} "
            f"ORDER BY weighted_total DESC LIMIT {top_k}"
        )
    return None


def _sqlite_special_template_sql(template_id: str, bindings: dict[str, Any], table: str) -> str | None:
    if bindings.get("aggregate_measure_mode") == "count_rows":
        fallback = _count_fallback_sql(template_id, bindings, table)
        if fallback:
            return fallback

    group_col = quote_identifier(bindings.get("group_col")) if bindings.get("group_col") else None
    measure_col = quote_identifier(bindings.get("measure_col")) if bindings.get("measure_col") else None
    if template_id == "tpl_m4_group_dispersion_rank" and group_col and measure_col:
        top_k = int(float(bindings.get("top_k") or 5))
        measure = f"CAST({measure_col} AS REAL)"
        return (
            f"SELECT {group_col}, "
            f"(AVG({measure} * {measure}) - AVG({measure}) * AVG({measure})) AS measure_variance "
            f"FROM {table} WHERE {measure_col} IS NOT NULL AND {measure_col} != '' "
            f"GROUP BY {group_col} ORDER BY measure_variance DESC LIMIT {top_k}"
        )
    if template_id == "tpl_grouped_percentile_point" and group_col and measure_col:
        percentile = numeric_literal(bindings.get("percentile_value"), 0.95)
        measure = f"CAST({measure_col} AS REAL)"
        return (
            "WITH ranked AS ("
            f"SELECT {group_col} AS group_value, {measure} AS measure_value, "
            f"ROW_NUMBER() OVER (PARTITION BY {group_col} ORDER BY {measure}) AS rn, "
            f"COUNT(*) OVER (PARTITION BY {group_col}) AS cnt "
            f"FROM {table} WHERE {measure_col} IS NOT NULL AND {measure_col} != ''"
            ") "
            "SELECT group_value, MIN(measure_value) AS percentile_measure "
            f"FROM ranked WHERE rn >= cnt * {percentile} GROUP BY group_value ORDER BY percentile_measure DESC"
        )
    if template_id == "tpl_conditional_group_quantiles" and group_col and measure_col:
        percentile = numeric_literal(bindings.get("percentile_value"), 0.95)
        measure = f"CAST({measure_col} AS REAL)"
        condition_col = quote_identifier(bindings.get("condition_col"))
        condition_value = sql_literal(bindings.get("condition_value"))
        return (
            "WITH ranked AS ("
            f"SELECT {group_col} AS group_value, {measure} AS measure_value, "
            f"ROW_NUMBER() OVER (PARTITION BY {group_col} ORDER BY {measure}) AS rn, "
            f"COUNT(*) OVER (PARTITION BY {group_col}) AS cnt "
            f"FROM {table} WHERE {condition_col} = {condition_value} "
            f"AND {measure_col} IS NOT NULL AND {measure_col} != ''"
            ") "
            "SELECT group_value, MIN(measure_value) AS conditional_percentile "
            f"FROM ranked WHERE rn >= cnt * {percentile} GROUP BY group_value ORDER BY conditional_percentile DESC"
        )
    return None


def _safe_operator(value: Any) -> str:
    op = str(value or "=").strip().upper()
    return op if op in {"=", "!=", "<>", "<", "<=", ">", ">=", "LIKE"} else "="


def instantiate_template_sql(
    *,
    template_id: str,
    template_lookup: dict[str, dict[str, Any]],
    question_record: dict[str, Any],
    table_name: str,
) -> str:
    template = template_lookup[template_id]
    bindings = dict(question_record.get("bindings") or {})
    table = quote_identifier(table_name)
    special = _sqlite_special_template_sql(template_id, bindings, table)
    if special:
        return ensure_template_comment(special, template_id)

    skeleton = str(question_record.get("runtime_sql_skeleton") or template.get("sql_skeleton") or "")
    if not skeleton.strip():
        raise ValueError(f"template {template_id} has no SQL skeleton")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "table":
            return table
        if key == "predicate_op":
            return _safe_operator(bindings.get(key))
        if key.endswith("_col") or key in {
            "group_col",
            "group_col_2",
            "measure_col",
            "entity_col",
            "item_col",
            "band_col",
            "condition_col",
            "predicate_col",
            "target_col",
        }:
            if key not in bindings:
                raise KeyError(f"missing binding `{key}` for template {template_id}")
            return quote_identifier(bindings[key])
        if key.endswith("_value") or key in {
            "predicate_value",
            "target_value",
            "condition_value",
            "positive_value",
            "negative_value",
        }:
            return sql_literal(bindings.get(key))
        if key in {"top_k", "top_n", "num_tiles"}:
            return str(int(float(bindings.get(key) or 1)))
        if key in {
            "percentile_value",
            "z_threshold",
            "fraction_threshold",
            "baseline_multiplier",
            "baseline_fraction",
            "min_support",
            "min_group_size",
            "measure_threshold",
            "band_cut_1",
            "band_cut_2",
            "lower_bound",
            "upper_bound",
        }:
            return numeric_literal(bindings.get(key))
        return str(bindings.get(key, ""))

    sql = re.sub(r"\{([A-Za-z0-9_]+)\}", replace, skeleton).strip()
    return ensure_template_comment(sql, template_id)


def run_template_sql_question(
    *,
    dataset_id: str,
    question: str,
    question_record: dict[str, Any] | None,
    db_path: Path,
    table_name: str,
    template_lookup: dict[str, dict[str, Any]],
    artifact_writer: RunArtifactWriter,
    row_limit: int,
    sql_timeout_ms: int,
) -> LocalRunnerResult:
    if not question_record or not question_record.get("template_id"):
        raise ValueError("template engine requires question records with `template_id`")
    template_id = str(question_record["template_id"])
    sql = instantiate_template_sql(
        template_id=template_id,
        template_lookup=template_lookup,
        question_record=question_record,
        table_name=table_name,
    )
    execution = execute_sqlite_query(
        db_path=db_path,
        sql=sql,
        row_limit=row_limit,
        timeout_ms=sql_timeout_ms,
    )
    query_results = [
        {
            "step_index": 0,
            "message_index": 0,
            "node_name": "template",
            "tool_name": "sqlite_query",
            "query": sql,
            "result": json.dumps(execution, ensure_ascii=False),
        }
    ]
    final_answer = local_result_answer(question, execution)
    usage_summary = {
        "dataset_id": dataset_id,
        "model": "template",
        "run_id": artifact_writer.run_id,
        "api_calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "ai_cli_calls": 0,
        "note": "Executed deterministically from the planned SQL template.",
    }
    artifact_writer.append_trace(
        {
            "event_type": "template_sql_generation",
            "template_id": template_id,
            "query": sql,
            "row_count_returned": execution["row_count_returned"],
            "elapsed_ms": execution["elapsed_ms"],
        }
    )
    artifact_writer.write_generated_sql([sql])
    artifact_writer.write_query_results(query_results)
    artifact_writer.write_final_answer(final_answer)
    artifact_writer.write_usage_summary(usage_summary)
    return LocalRunnerResult(
        final_answer=final_answer,
        generated_sqls=[sql],
        query_results=query_results,
        usage_summary=usage_summary,
    )
