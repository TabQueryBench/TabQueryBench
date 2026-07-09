#!/usr/bin/env python3
"""Build an enriched summary for prior local CLI runs.

This script does not modify individual run artifacts. It scans existing run
directories and produces a consolidated JSON summary with:

- wall-clock timing (from manifest timestamps)
- CLI invocation latency (from trace.jsonl)
- SQL execution latency (from query_results.jsonl)
- prompt/response file references
- estimated token counts for legacy runs that did not record exact usage
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "logs" / "runs" / "trial6_codex_gpt54_xhigh_final_summary.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "logs" / "runs" / "trial6_codex_gpt54_xhigh_enriched_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill enriched telemetry for prior CLI runs.")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model-hint", type=str, default="gpt-5.4")
    return parser.parse_args()


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
        encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))


def text_metrics(path: Path, model_hint: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "chars": 0,
            "bytes_utf8": 0,
            "lines": 0,
            "estimated_tokens": None,
        }
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "exists": True,
        "chars": len(text),
        "bytes_utf8": len(text.encode("utf-8")),
        "lines": len(text.splitlines()),
        "estimated_tokens": estimate_token_count(text, model_hint=model_hint),
    }


def duration_ms(started_at: str | None, ended_at: str | None) -> float | None:
    if not started_at or not ended_at:
        return None
    try:
        start_dt = datetime.fromisoformat(started_at)
        end_dt = datetime.fromisoformat(ended_at)
    except ValueError:
        return None
    return round((end_dt - start_dt).total_seconds() * 1000, 2)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _run_dirs_for_prefixes(logs_dir: Path, prefixes: list[str]) -> list[Path]:
    return sorted(
        [
            path
            for path in logs_dir.iterdir()
            if path.is_dir() and any(path.name.startswith(prefix) for prefix in prefixes)
        ]
    )


def _trace_latency_summary(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sql_cli_elapsed = 0.0
    answer_cli_elapsed = 0.0
    sql_attempts = 0
    answer_attempts = 0
    for row in trace_rows:
        event_type = row.get("event_type")
        elapsed = float(row.get("elapsed_ms") or 0.0)
        if event_type == "ai_cli_sql_generation":
            sql_cli_elapsed += elapsed
            sql_attempts += 1
        elif event_type == "ai_cli_answer_generation":
            answer_cli_elapsed += elapsed
            answer_attempts += 1
    return {
        "sql_cli_elapsed_ms_total": round(sql_cli_elapsed, 2),
        "answer_cli_elapsed_ms_total": round(answer_cli_elapsed, 2),
        "cli_elapsed_ms_total": round(sql_cli_elapsed + answer_cli_elapsed, 2),
        "sql_attempts": sql_attempts,
        "answer_attempts": answer_attempts,
    }


def _sql_elapsed_total(query_rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in query_rows:
        raw_result = row.get("result")
        if not isinstance(raw_result, str):
            continue
        try:
            parsed = json.loads(raw_result)
        except json.JSONDecodeError:
            continue
        total += float(parsed.get("elapsed_ms") or 0.0)
    return round(total, 2)


def _conversation_paths(run_dir: Path) -> dict[str, Any]:
    cli_dir = run_dir / "cli"
    prompt_paths = sorted(str(path.resolve()) for path in cli_dir.glob("*prompt*.txt"))
    response_paths = sorted(str(path.resolve()) for path in cli_dir.glob("*response*.txt") if ".raw." not in path.name)
    raw_response_paths = sorted(str(path.resolve()) for path in cli_dir.glob("*response*.raw.*"))
    stderr_paths = sorted(str(path.resolve()) for path in cli_dir.glob("*stderr*.txt"))
    return {
        "prompt_paths": prompt_paths,
        "response_paths": response_paths,
        "raw_response_paths": raw_response_paths,
        "stderr_paths": stderr_paths,
        "final_answer_path": str((run_dir / "final_answer.txt").resolve()),
        "generated_sql_path": str((run_dir / "generated_sql.sql").resolve()),
        "question_record_path": str((run_dir / "grounding" / "question_record.json").resolve()),
        "selection_path": str((run_dir / "grounding" / "selection.json").resolve()),
        "adherence_path": str((run_dir / "grounding" / "template_adherence.json").resolve()),
    }


def build_run_row(run_dir: Path, model_hint: str) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    usage_summary = manifest.get("usage_summary") or {}
    trace_rows = _jsonl_rows(run_dir / "trace.jsonl")
    query_rows = _jsonl_rows(run_dir / "query_results.jsonl")
    prompt_metrics_total = {"chars": 0, "bytes_utf8": 0, "lines": 0, "estimated_tokens": 0}
    response_metrics_total = {"chars": 0, "bytes_utf8": 0, "lines": 0, "estimated_tokens": 0}
    for prompt_path in sorted((run_dir / "cli").glob("*prompt*.txt")):
        metrics = text_metrics(prompt_path, model_hint=model_hint)
        for key in prompt_metrics_total:
            prompt_metrics_total[key] += int(metrics.get(key) or 0)
    for response_path in sorted((run_dir / "cli").glob("*response*.txt")):
        if ".raw." in response_path.name:
            continue
        metrics = text_metrics(response_path, model_hint=model_hint)
        for key in response_metrics_total:
            response_metrics_total[key] += int(metrics.get(key) or 0)

    return {
        "run_id": manifest.get("run_id"),
        "dataset_id": manifest.get("dataset_id"),
        "status": manifest.get("status"),
        "question_id": ((manifest.get("question_record") or {}).get("question_id")),
        "question": manifest.get("question"),
        "template_id": ((manifest.get("question_record") or {}).get("template_id")),
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "duration_ms": manifest.get("duration_ms") or duration_ms(manifest.get("started_at"), manifest.get("ended_at")),
        "input_tokens": usage_summary.get("input_tokens"),
        "cached_input_tokens": usage_summary.get("cached_input_tokens"),
        "output_tokens": usage_summary.get("output_tokens"),
        "total_tokens": usage_summary.get("total_tokens"),
        "estimated_input_tokens": usage_summary.get("estimated_input_tokens", prompt_metrics_total["estimated_tokens"]),
        "estimated_output_tokens": usage_summary.get("estimated_output_tokens", response_metrics_total["estimated_tokens"]),
        "estimated_total_tokens": usage_summary.get(
            "estimated_total_tokens",
            prompt_metrics_total["estimated_tokens"] + response_metrics_total["estimated_tokens"],
        ),
        "usage_source": usage_summary.get("usage_source", "estimated_only"),
        "trace_latency": _trace_latency_summary(trace_rows),
        "sql_execution_elapsed_ms_total": usage_summary.get(
            "sql_execution_elapsed_ms_total",
            _sql_elapsed_total(query_rows),
        ),
        "prompt_metrics_total": prompt_metrics_total,
        "response_metrics_total": response_metrics_total,
        "conversation_paths": _conversation_paths(run_dir),
        "artifacts_dir": str(run_dir.resolve()),
    }


def main() -> None:
    args = parse_args()
    summary_payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    logs_dir = PROJECT_ROOT / "logs" / "runs"
    enriched = {
        "source_summary": str(args.summary_json.resolve()),
        "model_hint": args.model_hint,
        "datasets": [],
    }

    for dataset_row in summary_payload.get("datasets", []):
        prefixes = dataset_row.get("run_prefixes") or []
        run_dirs = _run_dirs_for_prefixes(logs_dir, prefixes)
        run_rows = [
            build_run_row(run_dir, model_hint=args.model_hint)
            for run_dir in run_dirs
            if (run_dir / "run_manifest.json").exists()
        ]
        dataset_summary = dict(dataset_row)
        dataset_summary["run_rows"] = run_rows
        dataset_summary["duration_ms_total"] = round(
            sum(float(row.get("duration_ms") or 0.0) for row in run_rows if row.get("status") == "completed"),
            2,
        )
        dataset_summary["cli_elapsed_ms_total"] = round(
            sum(float((row.get("trace_latency") or {}).get("cli_elapsed_ms_total") or 0.0) for row in run_rows),
            2,
        )
        dataset_summary["sql_execution_elapsed_ms_total"] = round(
            sum(float(row.get("sql_execution_elapsed_ms_total") or 0.0) for row in run_rows),
            2,
        )
        dataset_summary["actual_total_tokens"] = sum(int(row.get("total_tokens") or 0) for row in run_rows)
        dataset_summary["estimated_total_tokens"] = sum(int(row.get("estimated_total_tokens") or 0) for row in run_rows)
        dataset_summary["runs_with_exact_usage"] = sum(
            1 for row in run_rows if row.get("usage_source") == "ai_cli_json_usage"
        )
        dataset_summary["runs_with_estimated_only"] = sum(
            1 for row in run_rows if row.get("usage_source") != "ai_cli_json_usage"
        )
        enriched["datasets"].append(dataset_summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
