"""Execution runner for the isolated v2 workload line."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import time
from typing import Any

from tqb_query.agent.local_sql_runner import (
    execute_sqlite_query,
    instantiate_template_sql,
    resolve_ai_cli_command,
    run_ai_cli_sql_question,
)
from tqb_query.config.settings import DATA_DIR
from tqb_query.data.context import build_dataset_context
from tqb_query.logging.run_artifacts import RunArtifactWriter

from .catalog import load_template_lookup
from .dataset_profile import load_dataset_role_profile
from .paths import (
    ensure_line_dirs,
    registry_csv_path,
    registry_jsonl_path,
    run_manifest_dir,
    run_sql_dir,
    template_library_path,
)
from .registry import append_registry_rows, load_registry_rows, write_registry_csv
from .sql_metadata import prepend_sql_metadata


PROGRESS_EVERY_ITEMS = max(1, int(os.getenv("V2_RUNNER_PROGRESS_EVERY", "10")))
PROGRESS_EVERY_SECONDS = max(1.0, float(os.getenv("V2_RUNNER_PROGRESS_SECONDS", "30")))


def _log(line_version: str, message: str) -> None:
    """Progress goes out unbuffered so `tail -f` on a detached run stays current."""
    print(f"[{line_version}-run] {message}", flush=True)


def _duration(seconds: float) -> str:
    """Seconds below a minute: a fast dataset would otherwise just read 0.0m."""
    return f"{seconds:.0f}s" if seconds < 60 else f"{seconds / 60:.1f}m"


def _elapsed_eta(done: int, total: int, started: float) -> str:
    elapsed = time.monotonic() - started
    if done <= 0 or done >= total:
        return f"elapsed={_duration(elapsed)}"
    return f"elapsed={_duration(elapsed)} eta={_duration(elapsed / done * (total - done))}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _selection_from_template(template_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "shortlist": [
            {
                "template_id": template_row.get("template_id"),
                "template_name": template_row.get("template_name"),
                "primary_family": template_row.get("family_id"),
                "portability": template_row.get("single_table_portable", "yes"),
                "sql_skeleton": template_row.get("sql_skeleton"),
                "required_roles": template_row.get("required_roles", []),
            }
        ]
    }


def _read_inventory(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inventory_shard_items(
    items: list[dict[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [item for item_index, item in enumerate(items) if item_index % shard_count == shard_index]


def _write_sql_copy(path: Path, sql_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sql_text.rstrip() + "\n", encoding="utf-8")


@contextmanager
def _registry_file_lock(registry_path: Path):
    """Serialize registry jsonl/csv updates when datasets run in parallel."""
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _manifest_base(*, run_id: str, dataset_id: str, item: dict[str, Any], engine: str, line_version: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "started_at": now,
        "ended_at": now,
        "status": "started",
        "engine": engine,
        "question_record": item,
        "mode": f"subitem_workload_{line_version}",
        "sql_source_version": line_version,
        "sql_source_label": f"{line_version}_current",
    }


def run_inventory(
    *,
    inventory_path: Path,
    run_id: str,
    line_version: str = "v2",
    data_root: Path = DATA_DIR,
    engine: str = "template",
    model: str = "",
    ai_cli_preset: str = "codex",
    ai_cli_command: str = "",
    ai_cli_timeout_seconds: int = 120,
    ai_cli_retries: int = 1,
    ai_cli_answer_mode: str = "local",
    allow_ai_cli_template_fallback: bool = True,
    row_limit: int = 50,
    sql_timeout_ms: int = 10000,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    if engine not in {"template", "cli"}:
        raise ValueError("v2 runner currently supports engine=template or engine=cli")

    ensure_line_dirs(line_version)
    inventory = _read_inventory(inventory_path)
    dataset_id = str(inventory["dataset_id"])
    profile = load_dataset_role_profile(dataset_id, data_root=data_root, use_cache=True)
    template_library = template_library_path(line_version)
    if not template_library.exists():
        from .catalog import write_template_library_jsonl

        write_template_library_jsonl(template_library)
    template_lookup = load_template_lookup(template_library)
    dataset_context = build_dataset_context(profile.bundle, profile.sqlite_result.table_name)
    rows_to_append: list[dict[str, Any]] = []
    cli_command = ""
    if engine == "cli":
        cli_command = resolve_ai_cli_command(
            preset=ai_cli_preset,
            custom_command=ai_cli_command,
            project_root=Path.cwd(),
            model=model,
        )

    inventory_items = _inventory_shard_items(
        list(inventory.get("items") or []),
        shard_index=shard_index,
        shard_count=shard_count,
    )
    total_items = len(inventory_items)
    # Shards of one dataset write into the same log, so every line has to say which shard it came from.
    shard_label = f"[{shard_index + 1}/{shard_count}] " if shard_count > 1 else ""
    _log(line_version, f"{shard_label}{dataset_id} start items={total_items} engine={engine} run_id={run_id}")
    started_at = time.monotonic()
    last_progress_at = started_at
    ok_count = 0
    failed_count = 0
    for item in inventory_items:
        query_record_id = str(item["query_record_id"])
        artifact_root = run_manifest_dir(run_id, dataset_id, line_version=line_version)
        artifact_writer = RunArtifactWriter(artifact_root, query_record_id)
        manifest = _manifest_base(
            run_id=run_id,
            dataset_id=dataset_id,
            item=item,
            engine=engine,
            line_version=line_version,
        )
        template_row = template_lookup[item["template_id"]]
        try:
            if engine == "template" or str(item.get("realization_mode")) == "deterministic":
                raw_sql = instantiate_template_sql(
                    template_id=str(item["template_id"]),
                    template_lookup=template_lookup,
                    question_record=item,
                    table_name=profile.sqlite_result.table_name,
                    strict=str(item.get("realization_mode")) == "agent_bind",
                )
                sql_text = prepend_sql_metadata(
                    raw_sql,
                    {
                        **item,
                        "sql_source_version": line_version,
                        "sql_source_label": f"{line_version}_current",
                        "sql_source_run_id": run_id,
                        "sql_source_dataset_id": dataset_id,
                    },
                )
                execution = execute_sqlite_query(
                    db_path=profile.sqlite_result.db_path,
                    sql=sql_text,
                    row_limit=row_limit,
                    timeout_ms=sql_timeout_ms,
                )
                final_answer = json.dumps(
                    {
                        "row_count": execution.get("row_count"),
                        "preview_rows": execution.get("rows", [])[:5],
                    },
                    ensure_ascii=False,
                )
                artifact_writer.write_generated_sql([sql_text])
                artifact_writer.write_query_results(
                    [
                        {
                            "node_name": "v2_template",
                            "tool_name": "sqlite_query",
                            "query": sql_text,
                            "result": json.dumps(execution, ensure_ascii=False),
                        }
                    ]
                )
                artifact_writer.write_final_answer(final_answer)
                usage_summary = {
                    "engine": "template",
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_total_tokens": 0,
                    "usage_source": "none",
                }
                artifact_writer.write_usage_summary(usage_summary)
                exec_ok_real = True
                reject_reason_codes: list[str] = []
            else:
                local_result = run_ai_cli_sql_question(
                    command=cli_command,
                    dataset_id=dataset_id,
                    question=str(item["question"]),
                    dataset_context=dataset_context,
                    selection=_selection_from_template(template_row),
                    question_record=item,
                    db_path=profile.sqlite_result.db_path,
                    table_name=profile.sqlite_result.table_name,
                    artifact_writer=artifact_writer,
                    timeout_seconds=ai_cli_timeout_seconds,
                    max_retries=ai_cli_retries,
                    row_limit=row_limit,
                    sql_timeout_ms=sql_timeout_ms,
                    answer_mode=ai_cli_answer_mode,
                    cwd=Path.cwd(),
                    engine_label=f"v2-cli:{ai_cli_preset}",
                    model_hint=model,
                    allow_network_template_fallback=allow_ai_cli_template_fallback,
                )
                raw_sql = (local_result.generated_sqls or [""])[0]
                sql_text = prepend_sql_metadata(
                    raw_sql,
                    {
                        **item,
                        "sql_source_version": line_version,
                        "sql_source_label": f"{line_version}_current",
                        "sql_source_run_id": run_id,
                        "sql_source_dataset_id": dataset_id,
                    },
                )
                artifact_writer.write_generated_sql([sql_text])
                final_answer = local_result.final_answer
                usage_summary = local_result.usage_summary
                exec_ok_real = True
                reject_reason_codes = []

            sql_copy_path = run_sql_dir(run_id, dataset_id, line_version=line_version) / f"{query_record_id}.sql"
            _write_sql_copy(sql_copy_path, sql_text)
            manifest["status"] = "completed"
            manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
            manifest["generated_sql_path"] = str(sql_copy_path.resolve())
            manifest["usage_summary"] = usage_summary
        except Exception as exc:  # noqa: BLE001
            sql_text = ""
            final_answer = str(exc)
            exec_ok_real = False
            reject_reason_codes = ["exec_failed"]
            usage_summary = {
                "engine": engine,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_total_tokens": 0,
                "usage_source": "none",
            }
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        finally:
            artifact_writer.write_manifest(manifest)

        loader_visible = bool(sql_text)
        accepted_for_eval = exec_ok_real and loader_visible and bool(item.get("family_id")) and bool(item.get("canonical_subitem_id"))
        cli_dir = artifact_writer.run_dir / "cli"
        raw_response_paths = sorted(cli_dir.glob("*response*.raw.txt")) if cli_dir.exists() else []
        registry_row = {
            "registry_version": f"query_registry_{line_version}",
            "dataset_id": dataset_id,
            "round_id": run_id,
            "query_record_id": query_record_id,
            "problem_id": item["problem_id"],
            "source_kind": item["source_kind"],
            "realization_mode": item["realization_mode"],
            "template_id": item["template_id"],
            "generator_id": f"deterministic_{line_version}" if item["realization_mode"] == "deterministic" else "",
            "family_id": item["family_id"],
            "canonical_subitem_id": item["canonical_subitem_id"],
            "intended_facet_id": item["intended_facet_id"],
            "variant_semantic_role": item["variant_semantic_role"],
            "subitem_assignment_source": item["subitem_assignment_source"],
            "extended_family": bool(item.get("extended_family")),
            "question_text": item["question"],
            "sql_path": str((run_sql_dir(run_id, dataset_id, line_version=line_version) / f"{query_record_id}.sql").resolve()) if sql_text else "",
            "sql_sha256": _sha256_text(sql_text) if sql_text else "",
            "exec_ok_real": exec_ok_real,
            "accepted_for_eval": accepted_for_eval,
            "reject_reason_codes": reject_reason_codes,
            "loader_visible": loader_visible,
            "coverage_key": f"{dataset_id}::{item['canonical_subitem_id']}",
            "coverage_target_min": item["coverage_target_min"],
            "subitem_inference_source": "explicit",
            "subitem_inference_note": "canonical_subitem_id",
            "engine": engine,
            "sql_source_version": line_version,
            "sql_source_label": f"{line_version}_current",
            "template_name": item["template_name"],
            "semantic_result_contract": item.get("semantic_result_contract") or {},
            "role_constraints": item.get("role_constraints") or {},
            "scorer_type": (item.get("semantic_result_contract") or {}).get("scorer_type", ""),
            "final_answer": final_answer,
            "usage_input_tokens": usage_summary.get("input_tokens", 0),
            "usage_cached_input_tokens": usage_summary.get("cached_input_tokens", 0),
            "usage_output_tokens": usage_summary.get("output_tokens", 0),
            "usage_total_tokens": usage_summary.get("total_tokens", 0),
            "usage_estimated_total_tokens": usage_summary.get("estimated_total_tokens", 0),
            "usage_source": usage_summary.get("usage_source", "none"),
            "ai_cli_calls": usage_summary.get("ai_cli_calls", 0),
            "artifact_dir": str(artifact_writer.run_dir.resolve()),
            "usage_summary_path": str((artifact_writer.run_dir / "usage_summary.json").resolve()),
            "conversation_log_path": str((cli_dir / "conversation.jsonl").resolve()) if cli_dir.exists() else "",
            "cli_raw_response_paths": ";".join(str(path.resolve()) for path in raw_response_paths),
        }
        rows_to_append.append(registry_row)

        if exec_ok_real:
            ok_count += 1
        else:
            failed_count += 1
            # Surface the failure now: a whole dataset timing out used to be visible only at the end.
            _log(
                line_version,
                f"{shard_label}{dataset_id} FAILED {query_record_id} template={item['template_id']} "
                f"error={str(manifest.get('error') or '; '.join(reject_reason_codes))[:160]}",
            )
        done = len(rows_to_append)
        now = time.monotonic()
        if done % PROGRESS_EVERY_ITEMS == 0 or now - last_progress_at >= PROGRESS_EVERY_SECONDS or done == total_items:
            last_progress_at = now
            _log(
                line_version,
                f"{shard_label}{dataset_id} {done}/{total_items} ok={ok_count} failed={failed_count} "
                f"{_elapsed_eta(done, total_items, started_at)}",
            )

    _log(
        line_version,
        f"{shard_label}{dataset_id} done items={len(rows_to_append)} ok={ok_count} failed={failed_count} "
        f"accepted={sum(1 for row in rows_to_append if row['accepted_for_eval'])} "
        f"{_elapsed_eta(len(rows_to_append), len(rows_to_append), started_at)}",
    )
    registry_path = registry_jsonl_path(run_id, line_version=line_version)
    with _registry_file_lock(registry_path):
        append_registry_rows(registry_path, rows_to_append)
        all_rows = load_registry_rows(registry_path)
        write_registry_csv(registry_csv_path(run_id, line_version=line_version), all_rows)
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "inventory_path": str(inventory_path.resolve()),
        "registry_path": str(registry_path.resolve()),
        "row_count": len(rows_to_append),
        "accepted_count": sum(1 for row in rows_to_append if row["accepted_for_eval"]),
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
