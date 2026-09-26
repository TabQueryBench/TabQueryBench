#!/usr/bin/env python3
"""CLI entrypoint for dataset-mode LangGraph SQL QA agent."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.agent.local_sql_runner import (
    resolve_ai_cli_command,
    run_ai_cli_sql_question,
)
from tqb_query.config.settings import (
    DATA_DIR,
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL,
    DEFAULT_USAGE_CSV_PATH,
    MODEL_PRICING_CONFIG_PATH,
    RUNS_DIR,
    ensure_runtime_dirs,
)
from tqb_query.data.bundle import DatasetBundle, load_dataset_bundle
from tqb_query.data.context import build_dataset_context
from tqb_query.db.csv_sqlite import SqliteMaterializationResult, materialize_dataset_to_sqlite
from tqb_query.logging.run_artifacts import RunArtifactWriter
from tqb_query.usage.logger import UsageCSVLogger, UsageLogRecord
from tqb_query.usage.pricing import calculate_cost_usd, load_pricing_config, resolve_model_pricing
from tqb_query.usage.tracker import UsageTracker


def _safe_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def _extract_sql_queries(message: Any) -> list[str]:
    sql_list: list[str] = []
    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        if tool_call.get("name") != "sql_db_query":
            continue
        args = tool_call.get("args", {}) or {}
        query = args.get("query")
        if isinstance(query, str) and query.strip():
            sql_list.append(query.strip())
    return sql_list


def _message_to_trace_event(message: Any, node_name: str, step_index: int, message_index: int) -> dict[str, Any]:
    event = {
        "step_index": step_index,
        "message_index": message_index,
        "node_name": node_name,
        "message_type": getattr(message, "type", type(message).__name__),
        "content": _safe_content(getattr(message, "content", "")),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        event["tool_calls"] = tool_calls
    if getattr(message, "type", "") == "tool":
        event["tool_name"] = message.name
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            event["tool_call_id"] = tool_call_id
    return event


def _print_message(message: Any, node_name: str, verbose: bool) -> None:
    if verbose:
        message.pretty_print()
        return

    if getattr(message, "type", "") == "tool":
        preview = _safe_content(message.content)
        if len(preview) > 180:
            preview = preview[:177] + "..."
        print(f"[{node_name}] tool:{message.name} -> {preview}")
        return

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        names = [tool_call.get("name", "") for tool_call in tool_calls]
        print(f"[{node_name}] ai tool_calls: {names}")
        return

    if hasattr(message, "content"):
        print(f"[{node_name}] ai: {_safe_content(message.content)}")
        return

    print(f"[{node_name}] {getattr(message, 'type', type(message).__name__)}")


def build_run_id(dataset_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset_id}_{timestamp}"


def run_single_question(
    *,
    agent,
    question: str,
    model_name: str,
    dataset_id: str,
    usage_logger: UsageCSVLogger,
    pricing_config: dict[str, Any],
    artifact_writer: RunArtifactWriter,
    verbose: bool,
) -> dict[str, Any]:
    run_id = artifact_writer.run_id
    tracker = UsageTracker()
    run_error: Exception | None = None
    generated_sqls: list[str] = []
    query_results: list[dict[str, Any]] = []
    sql_query_by_tool_call_id: dict[str, str] = {}
    final_answer = ""

    try:
        for step_index, update in enumerate(
            agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="updates",
            )
        ):
            for node_name, payload in update.items():
                messages = payload.get("messages", []) if isinstance(payload, dict) else []
                for message_index, message in enumerate(messages):
                    _print_message(message, node_name=node_name, verbose=verbose)
                    tracker.add_message(message)
                    generated_sqls.extend(_extract_sql_queries(message))
                    tool_calls = getattr(message, "tool_calls", None) or []
                    for tool_call in tool_calls:
                        if tool_call.get("name") != "sql_db_query":
                            continue
                        tool_call_id = tool_call.get("id")
                        query = (tool_call.get("args") or {}).get("query")
                        if isinstance(tool_call_id, str) and isinstance(query, str) and query.strip():
                            sql_query_by_tool_call_id[tool_call_id] = query.strip()
                    if getattr(message, "type", "") == "tool" and getattr(message, "name", "") == "sql_db_query":
                        tool_call_id = getattr(message, "tool_call_id", None)
                        query_text = sql_query_by_tool_call_id.get(tool_call_id) if tool_call_id else None
                        if query_text is None and generated_sqls:
                            query_text = generated_sqls[-1]
                        query_results.append(
                            {
                                "step_index": step_index,
                                "message_index": message_index,
                                "node_name": node_name,
                                "tool_name": "sql_db_query",
                                "tool_call_id": tool_call_id,
                                "query": query_text or "",
                                "result": _safe_content(message.content),
                            }
                        )
                    artifact_writer.append_trace(
                        _message_to_trace_event(
                            message=message,
                            node_name=node_name,
                            step_index=step_index,
                            message_index=message_index,
                        )
                    )

                    if not tool_calls and hasattr(message, "content"):
                        candidate_answer = _safe_content(message.content).strip()
                        if candidate_answer:
                            final_answer = candidate_answer
    except Exception as exc:  # noqa: BLE001
        run_error = exc

    snapshot = tracker.snapshot
    usage_summary: dict[str, Any]
    if tracker.has_model_usage():
        pricing = resolve_model_pricing(model_name, pricing_config)
        cost_usd = calculate_cost_usd(
            snapshot.input_tokens,
            snapshot.output_tokens,
            pricing,
            cached_input_tokens=snapshot.cached_input_tokens,
        )
        usage_summary = {
            "dataset_id": dataset_id,
            "model": model_name,
            "run_id": run_id,
            "api_calls": snapshot.api_calls,
            "input_tokens": snapshot.input_tokens,
            "cached_input_tokens": snapshot.cached_input_tokens,
            "output_tokens": snapshot.output_tokens,
            "total_tokens": snapshot.total_tokens,
            "cost_usd": cost_usd,
        }
        usage_logger.append(
            UsageLogRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                run_id=run_id,
                model=model_name,
                input_tokens=snapshot.input_tokens,
                output_tokens=snapshot.output_tokens,
                total_tokens=snapshot.total_tokens,
                cost_usd=cost_usd,
                question=question,
                dataset_id=dataset_id,
                phase="sql_qa",
                module="run_sql_agent",
            )
        )
    else:
        usage_summary = {
            "dataset_id": dataset_id,
            "model": model_name,
            "run_id": run_id,
            "api_calls": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "note": "No model API usage detected.",
        }

    unique_generated_sqls = []
    seen = set()
    for sql in generated_sqls:
        if sql not in seen:
            unique_generated_sqls.append(sql)
            seen.add(sql)

    artifact_writer.write_generated_sql(unique_generated_sqls)
    artifact_writer.write_query_results(query_results)
    artifact_writer.write_final_answer(final_answer)
    artifact_writer.write_usage_summary(usage_summary)

    if run_error is not None:
        raise run_error

    return {
        "final_answer": final_answer,
        "generated_sqls": unique_generated_sqls,
        "usage_summary": usage_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a LangGraph SQL QA agent on standardized single-table datasets.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=DEFAULT_DATASET_ID,
        help="Dataset ID under data root (example: c2).",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Question to ask. If omitted, runs in interactive mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model passed to the API runner or forwarded to the local AI CLI preset.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["api", "cli"],
        default="api",
        help="Execution engine: API LangGraph agent or local AI CLI.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_DIR,
        help="Root directory containing standardized datasets.",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to reuse cached SQLite database when source CSV has not changed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full LangChain message objects instead of compact logs.",
    )
    parser.add_argument(
        "--usage-csv",
        type=Path,
        default=DEFAULT_USAGE_CSV_PATH,
        help="CSV path for usage logs.",
    )
    parser.add_argument(
        "--pricing-config",
        type=Path,
        default=MODEL_PRICING_CONFIG_PATH,
        help="JSON config path for model pricing.",
    )
    parser.add_argument(
        "--ai-cli-preset",
        type=str,
        choices=["codex", "claude", "gemini", "custom"],
        default="codex",
        help="Preset command used when --engine cli is selected.",
    )
    parser.add_argument(
        "--ai-cli-command",
        type=str,
        default="",
        help="Optional custom shell command for the local AI CLI. Prompt will be piped through stdin.",
    )
    parser.add_argument(
        "--ai-cli-timeout-seconds",
        type=int,
        default=120,
        help="Timeout for each local AI CLI call.",
    )
    parser.add_argument(
        "--ai-cli-retries",
        type=int,
        default=1,
        help="How many retry rounds to allow after a failed SQL execution in CLI mode.",
    )
    parser.add_argument(
        "--ai-cli-answer-mode",
        type=str,
        choices=["local", "ai"],
        default="local",
        help="Whether CLI mode answers locally from query results or makes a second AI CLI call.",
    )
    parser.add_argument(
        "--local-sql-row-limit",
        type=int,
        default=50,
        help="Maximum number of result rows to materialize in local runner outputs.",
    )
    parser.add_argument(
        "--local-sql-timeout-ms",
        type=int,
        default=10000,
        help="SQLite execution timeout used by local runners.",
    )
    return parser.parse_args()


def _prepare_runtime(args: argparse.Namespace):
    bundle = load_dataset_bundle(dataset_id=args.dataset_id, data_root=args.data_root, strict=True)
    sqlite_result = materialize_dataset_to_sqlite(bundle=bundle, use_cache=args.use_cache)
    dataset_context = build_dataset_context(bundle=bundle, table_name=sqlite_result.table_name)

    print(f"[dataset] dataset_id={bundle.dataset_id}")
    print(f"[dataset] main_csv={bundle.main_csv_path}")
    print(f"[dataset] sqlite_db={sqlite_result.db_path} table={sqlite_result.table_name} cache_hit={sqlite_result.cache_hit}")

    agent = None
    if args.engine == "api":
        try:
            from tqb_query.agent.sql_agent import build_sql_agent
        except ModuleNotFoundError as exc:  # noqa: PERF203
            raise ModuleNotFoundError(
                "API engine requires LangChain/LangGraph dependencies. "
                "Install requirements.txt or run with --engine cli."
            ) from exc

        agent = build_sql_agent(
            model_name=args.model,
            db_uri=sqlite_result.sqlite_uri,
            dataset_context=dataset_context,
            primary_table_name=sqlite_result.table_name,
        )
    return bundle, sqlite_result, dataset_context, agent


def _build_manifest(
    *,
    run_id: str,
    args: argparse.Namespace,
    question: str,
    bundle: DatasetBundle,
    sqlite_result: SqliteMaterializationResult,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "running",
        "dataset_id": bundle.dataset_id,
        "question": question,
        "model": args.model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "loaded_files_summary": bundle.loaded_files_summary(),
        "sqlite": {
            "db_path": str(sqlite_result.db_path),
            "table_name": sqlite_result.table_name,
            "row_count": sqlite_result.row_count,
            "cache_hit": sqlite_result.cache_hit,
            "cache_manifest_path": str(sqlite_result.manifest_path),
        },
        "cli_options": {
            "data_root": str(args.data_root),
            "use_cache": args.use_cache,
            "verbose": args.verbose,
            "engine": args.engine,
            "ai_cli_preset": args.ai_cli_preset,
            "ai_cli_command": args.ai_cli_command,
            "ai_cli_timeout_seconds": args.ai_cli_timeout_seconds,
            "ai_cli_retries": args.ai_cli_retries,
            "ai_cli_answer_mode": args.ai_cli_answer_mode,
            "local_sql_row_limit": args.local_sql_row_limit,
            "local_sql_timeout_ms": args.local_sql_timeout_ms,
        },
        "sql_source_version": "v1",
        "sql_source_label": "v1_legacy",
        "sql_source_layout": "legacy_single_run",
        "artifacts_dir": str((RUNS_DIR / run_id)),
    }


def main() -> None:
    ensure_runtime_dirs()
    args = parse_args()
    usage_logger = UsageCSVLogger(args.usage_csv)
    pricing_config = load_pricing_config(args.pricing_config)

    bundle, sqlite_result, _dataset_context, agent = _prepare_runtime(args)

    if args.question:
        questions = [args.question]
    else:
        print("Interactive mode. Enter an empty line to exit.")
        questions = []
        while True:
            question = input("\nQuestion: ").strip()
            if not question:
                break
            questions.append(question)

    for question in questions:
        run_id = build_run_id(bundle.dataset_id)
        artifact_writer = RunArtifactWriter(RUNS_DIR, run_id)
        artifact_writer.set_sql_header_metadata(
            {
                "sql_source_version": "v1",
                "sql_source_label": "v1_legacy",
                "sql_source_run_id": run_id,
                "sql_source_dataset_id": bundle.dataset_id,
                "sql_source_engine": args.engine,
            }
        )

        manifest = _build_manifest(
            run_id=run_id,
            args=args,
            question=question,
            bundle=bundle,
            sqlite_result=sqlite_result,
        )
        artifact_writer.write_manifest(manifest)

        print(f"\n[run] run_id={run_id}")
        print(f"[run] question={question}")

        try:
            if args.engine == "api":
                result = run_single_question(
                    agent=agent,
                    question=question,
                    model_name=args.model,
                    dataset_id=bundle.dataset_id,
                    usage_logger=usage_logger,
                    pricing_config=pricing_config,
                    artifact_writer=artifact_writer,
                    verbose=args.verbose,
                )
            else:
                cli_command = resolve_ai_cli_command(
                    preset=args.ai_cli_preset,
                    custom_command=args.ai_cli_command,
                    project_root=PROJECT_ROOT,
                    model=args.model,
                )
                local_result = run_ai_cli_sql_question(
                    command=cli_command,
                    dataset_id=bundle.dataset_id,
                    question=question,
                    dataset_context=_dataset_context,
                    selection={"shortlist": []},
                    question_record=None,
                    db_path=sqlite_result.db_path,
                    table_name=sqlite_result.table_name,
                    artifact_writer=artifact_writer,
                    timeout_seconds=args.ai_cli_timeout_seconds,
                    max_retries=args.ai_cli_retries,
                    row_limit=args.local_sql_row_limit,
                    sql_timeout_ms=args.local_sql_timeout_ms,
                    answer_mode=args.ai_cli_answer_mode,
                    cwd=PROJECT_ROOT,
                    engine_label=f"cli:{args.ai_cli_preset}",
                )
                result = {
                    "final_answer": local_result.final_answer,
                    "generated_sqls": local_result.generated_sqls,
                    "usage_summary": local_result.usage_summary,
                }
            manifest["status"] = "completed"
            manifest["final_answer"] = result["final_answer"]
            manifest["generated_sql_count"] = len(result["generated_sqls"])
            manifest["usage_summary"] = result["usage_summary"]
            print(f"[run] final_answer={result['final_answer']}")
            print(f"[run] artifacts={artifact_writer.run_dir}")
            print(f"[run] usage_summary={result['usage_summary']}")
        except Exception as exc:  # noqa: BLE001
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            print(f"[run] failed: {exc}")
            raise
        finally:
            manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
            artifact_writer.write_manifest(manifest)


if __name__ == "__main__":
    main()
