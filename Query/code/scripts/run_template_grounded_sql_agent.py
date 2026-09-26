#!/usr/bin/env python3
"""Run the dataset SQL agent with workload-grounded all-core template guidance."""

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

from scripts.run_sql_agent import build_run_id, run_single_question
from tqb_query.agent.local_sql_runner import (
    resolve_ai_cli_command,
    run_ai_cli_sql_question,
    run_template_sql_question,
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
from tqb_query.usage.logger import UsageCSVLogger
from tqb_query.usage.pricing import load_pricing_config
from tqb_query.workload_grounding.adherence import analyze_sql_queries
from tqb_query.workload_grounding.question_inventory import build_cli_all_question_inventory
from tqb_query.workload_grounding.runtime import (
    extract_template_ids_from_sql,
    format_grounding_prompt,
    select_grounded_templates,
)


def _safe_console_text(text: Any) -> str:
    rendered = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        rendered.encode(encoding)
        return rendered
    except UnicodeEncodeError:
        return rendered.encode(encoding, errors="replace").decode(encoding, errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SQL QA agent with workload-grounded all-core template guidance.",
    )
    parser.add_argument("--dataset-id", type=str, default=DEFAULT_DATASET_ID, help="Dataset ID under data root.")
    parser.add_argument("--question", "-q", type=str, help="Question to ask. If omitted, runs in interactive mode.")
    parser.add_argument(
        "--questions-json",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of questions or question records with a `question` field.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model passed to the API runner or AI CLI preset.")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["api", "cli", "cli-all", "template"],
        default="api",
        help="Execution engine: API LangGraph agent, local AI CLI, CLI-all AI planning + SQL, or deterministic template SQL.",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_DIR, help="Root directory containing datasets.")
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to reuse cached SQLite database when source CSV has not changed.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full LangChain message objects.")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Optional cap for batch execution. Use 0 to disable the cap.",
    )
    parser.add_argument("--usage-csv", type=Path, default=DEFAULT_USAGE_CSV_PATH, help="CSV path for usage logs.")
    parser.add_argument(
        "--pricing-config",
        type=Path,
        default=MODEL_PRICING_CONFIG_PATH,
        help="JSON config path for model pricing.",
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "agent_candidate_spec_all_core_v1.json",
        help="Path to curated agent candidate spec JSON.",
    )
    parser.add_argument(
        "--spec-bucket",
        type=str,
        default="all_core",
        help="List bucket inside the candidate spec to use as the runtime candidate pool.",
    )
    parser.add_argument(
        "--template-library",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "template_library_v1.jsonl",
        help="Path to core template library JSONL.",
    )
    parser.add_argument(
        "--portability-report",
        type=Path,
        default=PROJECT_ROOT / "data" / "workload_grounding" / "template_portability_report_v1.csv",
        help="Path to core portability report CSV.",
    )
    parser.add_argument(
        "--min-grounded-templates",
        type=int,
        default=10,
        help="Minimum number of shortlisted grounded templates to retain after strict screening.",
    )
    parser.add_argument(
        "--max-grounded-templates",
        type=int,
        default=0,
        help="Optional upper cap on shortlisted grounded templates. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="",
        help="Optional run id prefix. Defaults to <dataset_id>_tg.",
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
    parser.add_argument(
        "--cli-all-min-templates",
        type=int,
        default=10,
        help="Minimum template count when --engine cli-all builds its own inventory.",
    )
    parser.add_argument(
        "--cli-all-target-templates",
        type=int,
        default=12,
        help="Target template count when --engine cli-all builds its own inventory.",
    )
    parser.add_argument(
        "--cli-all-min-problems-per-template",
        type=int,
        default=4,
        help="Minimum problem count per selected template when --engine cli-all builds its own inventory.",
    )
    parser.add_argument(
        "--cli-all-max-problems-per-template",
        type=int,
        default=12,
        help="Maximum problem count per selected template when --engine cli-all builds its own inventory.",
    )
    return parser.parse_args()


def _load_question_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unsupported questions JSON payload in {path}")

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, str):
            records.append({"question": item, "batch_index": idx})
            continue
        if isinstance(item, dict) and str(item.get("question") or "").strip():
            record = dict(item)
            record.setdefault("batch_index", idx)
            records.append(record)
            continue
        raise ValueError(f"Question record at index {idx} is missing a usable `question` field.")
    return records


def _load_template_lookup(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[obj["template_id"]] = obj
    return rows


def _apply_problem_runtime_override(
    template_lookup: dict[str, dict[str, Any]],
    question_record: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not question_record or not question_record.get("template_id"):
        return template_lookup
    runtime_sql_skeleton = question_record.get("runtime_sql_skeleton")
    if not runtime_sql_skeleton:
        return template_lookup
    template_id = str(question_record["template_id"])
    if template_id not in template_lookup:
        return template_lookup
    updated = dict(template_lookup)
    template_obj = dict(updated[template_id])
    template_obj["sql_skeleton"] = runtime_sql_skeleton
    updated[template_id] = template_obj
    return updated


def _format_problem_prompt_block(question_record: dict[str, Any] | None) -> str:
    if not question_record or not question_record.get("template_id"):
        return ""
    lines = [
        "Problem-instance block:",
        f"- planned_template_id: {question_record.get('template_id')}",
        f"- planned_problem_index: {question_record.get('problem_index_within_template')}",
        f"- expected_sql_count: {question_record.get('expected_sql_count', 2)}",
        f"- can_vary: {question_record.get('can_vary', [])}",
        f"- must_fix: {question_record.get('must_fix', [])}",
        f"- variation_axes_used_for_this_problem: {question_record.get('variation_axes', [])}",
        "- problem_rule: this question was instantiated from the planned template above. Stay on this template family unless it is clearly impossible.",
        "- binding_rule: respect the bound roles and values below; do not silently switch to a different template family.",
    ]
    bindings = question_record.get("bindings") or {}
    if bindings:
        lines.append(f"- bound_roles: {bindings}")
    runtime_sql_skeleton = question_record.get("runtime_sql_skeleton")
    if runtime_sql_skeleton:
        lines.append("- runtime_sql_shape_override:")
        for sql_line in str(runtime_sql_skeleton).splitlines():
            lines.append(f"  {sql_line}")
    return "\n".join(lines)


def _prepare_runtime(
    args: argparse.Namespace,
    question: str,
    question_record: dict[str, Any] | None = None,
) -> tuple[DatasetBundle, SqliteMaterializationResult, str, dict[str, Any], dict[str, dict[str, Any]], Any]:
    bundle = load_dataset_bundle(dataset_id=args.dataset_id, data_root=args.data_root, strict=True)
    sqlite_result = materialize_dataset_to_sqlite(bundle=bundle, use_cache=args.use_cache)
    dataset_context = build_dataset_context(bundle=bundle, table_name=sqlite_result.table_name)
    selection = select_grounded_templates(
        dataset_id=bundle.dataset_id,
        question=question,
        spec_path=args.spec_path,
        spec_bucket=args.spec_bucket,
        core_library_path=args.template_library,
        portability_report_path=args.portability_report,
        min_templates=args.min_grounded_templates,
        max_templates=args.max_grounded_templates,
        preferred_template_id=str(question_record.get("template_id")) if question_record else None,
    )
    grounding_block = format_grounding_prompt(selection)
    problem_block = _format_problem_prompt_block(question_record)
    grounded_context = dataset_context + "\n\n" + grounding_block
    if problem_block:
        grounded_context += "\n\n" + problem_block
    template_lookup = _apply_problem_runtime_override(
        _load_template_lookup(args.template_library),
        question_record,
    )

    print(f"[dataset] dataset_id={bundle.dataset_id}")
    print(f"[dataset] main_csv={bundle.main_csv_path}")
    print(f"[dataset] sqlite_db={sqlite_result.db_path} table={sqlite_result.table_name} cache_hit={sqlite_result.cache_hit}")
    print(
        f"[grounding] spec_bucket={selection['spec_bucket']} "
        f"candidate_pool={selection['candidate_pool_count']} "
        f"applicable={selection['applicable_count']} "
        f"strict_keep={selection['strict_keep_count']} "
        f"shortlist={selection['shortlist_count']}"
    )

    agent = None
    if args.engine == "api":
        try:
            from tqb_query.agent.sql_agent import build_sql_agent
        except ModuleNotFoundError as exc:  # noqa: PERF203
            raise ModuleNotFoundError(
                "API engine requires LangChain/LangGraph dependencies. "
                "Install requirements.txt or run with --engine cli/--engine cli-all/--engine template."
            ) from exc

        agent = build_sql_agent(
            model_name=args.model,
            db_uri=sqlite_result.sqlite_uri,
            dataset_context=grounded_context,
            primary_table_name=sqlite_result.table_name,
        )
    return bundle, sqlite_result, grounded_context, selection, template_lookup, agent


def _build_manifest(
    *,
    run_id: str,
    runs_root: Path,
    args: argparse.Namespace,
    question: str,
    bundle: DatasetBundle,
    sqlite_result: SqliteMaterializationResult,
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "running",
        "mode": "template_grounded_sql_qa",
        "dataset_id": bundle.dataset_id,
        "question": question,
        "model": args.model,
        "engine": args.engine,
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
            "spec_path": str(args.spec_path),
            "spec_bucket": args.spec_bucket,
            "template_library": str(args.template_library),
            "portability_report": str(args.portability_report),
            "min_grounded_templates": args.min_grounded_templates,
            "max_grounded_templates": args.max_grounded_templates,
            "max_questions": args.max_questions,
            "ai_cli_preset": args.ai_cli_preset,
            "ai_cli_command": args.ai_cli_command,
            "ai_cli_timeout_seconds": args.ai_cli_timeout_seconds,
            "ai_cli_retries": args.ai_cli_retries,
            "ai_cli_answer_mode": args.ai_cli_answer_mode,
            "local_sql_row_limit": args.local_sql_row_limit,
            "local_sql_timeout_ms": args.local_sql_timeout_ms,
            "cli_all_min_templates": args.cli_all_min_templates,
            "cli_all_target_templates": args.cli_all_target_templates,
            "cli_all_min_problems_per_template": args.cli_all_min_problems_per_template,
            "cli_all_max_problems_per_template": args.cli_all_max_problems_per_template,
        },
        "grounding": {
            "spec_bucket": selection["spec_bucket"],
            "candidate_pool_count": selection["candidate_pool_count"],
            "applicable_count": selection["applicable_count"],
            "strict_keep_count": selection["strict_keep_count"],
            "backfill_candidate_count": selection["backfill_candidate_count"],
            "excluded_count": selection["excluded_count"],
            "shortlist_count": selection["shortlist_count"],
            "question_intents": selection["question_intents"],
            "preferred_template_id": selection.get("preferred_template_id"),
            "preferred_template_applied": selection.get("preferred_template_applied", False),
            "candidate_pool_template_ids": [row["template_id"] for row in selection["candidate_pool"]],
            "shortlist_template_ids": [row["template_id"] for row in selection["shortlist"]],
        },
        "sql_source_version": "v1",
        "sql_source_label": "v1_legacy",
        "sql_source_layout": "legacy_grounded_template_run",
        "artifacts_dir": str((runs_root / run_id)),
    }


def _duration_ms(started_at: str | None, ended_at: str | None) -> float | None:
    if not started_at or not ended_at:
        return None
    try:
        start_dt = datetime.fromisoformat(started_at)
        end_dt = datetime.fromisoformat(ended_at)
    except ValueError:
        return None
    return round((end_dt - start_dt).total_seconds() * 1000, 2)


def main() -> None:
    ensure_runtime_dirs()
    args = parse_args()
    usage_logger = UsageCSVLogger(args.usage_csv)
    pricing_config = load_pricing_config(args.pricing_config)

    batch_summary_dir: Path | None = None
    question_source_path: Path | None = None
    cli_all_inventory: dict[str, Any] | None = None
    if args.engine == "cli-all":
        if args.questions_json is not None:
            raise ValueError("--engine cli-all builds its own inventory; do not combine it with --questions-json")
        if args.question:
            raise ValueError("--engine cli-all expects dataset-level planning, not a single ad hoc --question")
        batch_run_id = build_run_id((args.run_prefix.strip() or f"{args.dataset_id}_cli_all_batch"))
        batch_summary_dir = RUNS_DIR / batch_run_id
        batch_summary_dir.mkdir(parents=True, exist_ok=True)
        batch_writer = RunArtifactWriter(RUNS_DIR, batch_run_id)
        print(f"[cli-all] building dataset inventory for {args.dataset_id}")
        cli_all_inventory = build_cli_all_question_inventory(
            dataset_id=args.dataset_id,
            spec_path=args.spec_path,
            spec_bucket=args.spec_bucket,
            core_library_path=args.template_library,
            portability_report_path=args.portability_report,
            planner_model=args.model,
            project_root=PROJECT_ROOT,
            data_root=args.data_root,
            min_templates=args.cli_all_min_templates,
            target_templates=args.cli_all_target_templates,
            min_problems_per_template=args.cli_all_min_problems_per_template,
            max_problems_per_template=args.cli_all_max_problems_per_template,
            ai_cli_preset=args.ai_cli_preset,
            ai_cli_command=args.ai_cli_command,
            planner_timeout_seconds=args.ai_cli_timeout_seconds,
            planner_invoke_retries=max(1, args.ai_cli_retries),
            planner_run_id=batch_run_id,
            usage_logger=usage_logger,
            pricing_config=pricing_config,
            artifact_writer=batch_writer,
        )
        batch_writer.write_json("planning/cli_all_inventory.json", cli_all_inventory)
        question_records = []
        for idx, item in enumerate(cli_all_inventory.get("items") or [], start=1):
            if not isinstance(item, dict):
                continue
            record = dict(item)
            record.setdefault("batch_index", idx)
            question_records.append(record)
        question_source_path = batch_writer.write_json("planning/generated_questions.json", {"items": question_records})
        print(
            f"[cli-all] selected_templates={cli_all_inventory.get('selected_template_count')} "
            f"generated_problems={cli_all_inventory.get('inventory_count')}"
        )
        print(f"[cli-all] output_root={batch_summary_dir}")
    elif args.questions_json is not None:
        question_records = _load_question_records(args.questions_json)
        question_source_path = args.questions_json
    elif args.question:
        question_records = [{"question": args.question, "batch_index": 1}]
    else:
        print("Interactive mode. Enter an empty line to exit.")
        question_records = []
        while True:
            question = input("\nQuestion: ").strip()
            if not question:
                break
            question_records.append({"question": question, "batch_index": len(question_records) + 1})

    if args.max_questions > 0:
        question_records = question_records[: args.max_questions]

    total_questions = len(question_records)
    batch_records: list[dict[str, Any]] = []
    if args.questions_json is not None and batch_summary_dir is None:
        batch_run_id = build_run_id((args.run_prefix.strip() or f"{args.dataset_id}_tg_batch"))
        batch_summary_dir = RUNS_DIR / batch_run_id
        batch_summary_dir.mkdir(parents=True, exist_ok=True)
    question_runs_root = (batch_summary_dir / "question_runs") if batch_summary_dir is not None else RUNS_DIR
    if batch_summary_dir is not None:
        question_runs_root.mkdir(parents=True, exist_ok=True)
        print(f"[batch] output_root={batch_summary_dir}")
    for question_record in question_records:
        question = str(question_record["question"]).strip()
        bundle, sqlite_result, grounded_context, selection, template_lookup, agent = _prepare_runtime(
            args,
            question,
            question_record=question_record,
        )
        run_stem = args.run_prefix.strip() or f"{bundle.dataset_id}_tg"
        run_id = build_run_id(run_stem)
        artifact_writer = RunArtifactWriter(question_runs_root, run_id)
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
            runs_root=question_runs_root,
            args=args,
            question=question,
            bundle=bundle,
            sqlite_result=sqlite_result,
            selection=selection,
        )
        artifact_writer.write_manifest(manifest)
        artifact_writer.write_json("grounding/selection.json", selection)
        artifact_writer.write_text("grounding/prompt_block.txt", grounded_context)
        if question_record:
            artifact_writer.write_json("grounding/question_record.json", question_record)

        print(
            f"\n[run] run_id={run_id} "
            f"question_index={question_record.get('batch_index', 1)}/{total_questions}"
        )
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
            elif args.engine in {"cli", "cli-all"}:
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
                    dataset_context=grounded_context,
                    selection=selection,
                    question_record=question_record,
                    db_path=sqlite_result.db_path,
                    table_name=sqlite_result.table_name,
                    artifact_writer=artifact_writer,
                    timeout_seconds=args.ai_cli_timeout_seconds,
                    max_retries=args.ai_cli_retries,
                    row_limit=args.local_sql_row_limit,
                    sql_timeout_ms=args.local_sql_timeout_ms,
                    answer_mode=args.ai_cli_answer_mode,
                    cwd=PROJECT_ROOT,
                    engine_label=f"{args.engine}:{args.ai_cli_preset}",
                    model_hint=args.model,
                )
                result = {
                    "final_answer": local_result.final_answer,
                    "generated_sqls": local_result.generated_sqls,
                    "usage_summary": local_result.usage_summary,
                }
            else:
                local_result = run_template_sql_question(
                    dataset_id=bundle.dataset_id,
                    question=question,
                    question_record=question_record,
                    db_path=sqlite_result.db_path,
                    table_name=sqlite_result.table_name,
                    template_lookup=template_lookup,
                    artifact_writer=artifact_writer,
                    row_limit=args.local_sql_row_limit,
                    sql_timeout_ms=args.local_sql_timeout_ms,
                )
                result = {
                    "final_answer": local_result.final_answer,
                    "generated_sqls": local_result.generated_sqls,
                    "usage_summary": local_result.usage_summary,
                }
            chosen_template_ids = extract_template_ids_from_sql(result["generated_sqls"])
            artifact_writer.write_json("grounding/chosen_template_ids.json", {"chosen_template_ids": chosen_template_ids})
            adherence = analyze_sql_queries(
                sql_queries=result["generated_sqls"],
                template_lookup=template_lookup,
                shortlist_ids=[item["template_id"] for item in selection["shortlist"]],
            )
            artifact_writer.write_json("grounding/template_adherence.json", adherence)

            manifest["status"] = "completed"
            manifest["final_answer"] = result["final_answer"]
            manifest["generated_sql_count"] = len(result["generated_sqls"])
            manifest["usage_summary"] = result["usage_summary"]
            manifest["question_record"] = question_record
            manifest["grounding"]["chosen_template_ids"] = chosen_template_ids
            completed_at = datetime.now(timezone.utc).isoformat()
            manifest["ended_at"] = completed_at
            manifest["duration_ms"] = _duration_ms(manifest.get("started_at"), completed_at)
            manifest["grounding"]["adherence"] = {
                "overall_adherence_score": adherence["overall_adherence_score"],
                "commented_query_count": adherence["commented_query_count"],
                "analyzed_query_count": adherence["analyzed_query_count"],
                "shortlist_violation_count": adherence["shortlist_violation_count"],
                "label_counts": adherence["label_counts"],
                "artifact_path": str((artifact_writer.run_dir / "grounding" / "template_adherence.json").resolve()),
            }

            print(f"[run] chosen_template_ids={chosen_template_ids}")
            print(f"[run] adherence={adherence['overall_adherence_score']} labels={adherence['label_counts']}")
            print(f"[run] final_answer={_safe_console_text(result['final_answer'])}")
            print(f"[run] artifacts={artifact_writer.run_dir}")
            batch_records.append(
                {
                    "run_id": run_id,
                    "dataset_id": bundle.dataset_id,
                    "question_index": question_record.get("batch_index", 1),
                    "question_id": question_record.get("question_id"),
                    "question": question,
                    "template_id": question_record.get("template_id"),
                    "primary_family": question_record.get("primary_family"),
                    "variation_axes": question_record.get("variation_axes", []),
                    "expected_sql_count": question_record.get("expected_sql_count", 2),
                    "engine": args.engine,
                    "generated_sql_count": len(result["generated_sqls"]),
                    "chosen_template_ids": chosen_template_ids,
                    "adherence_score": adherence["overall_adherence_score"],
                    "adherence_labels": adherence["label_counts"],
                    "duration_ms": manifest.get("duration_ms"),
                    "input_tokens": result["usage_summary"].get("input_tokens", 0),
                    "cached_input_tokens": result["usage_summary"].get("cached_input_tokens", 0),
                    "output_tokens": result["usage_summary"].get("output_tokens", 0),
                    "total_tokens": result["usage_summary"].get("total_tokens", 0),
                    "estimated_total_tokens": result["usage_summary"].get("estimated_total_tokens", 0),
                    "usage_source": result["usage_summary"].get("usage_source"),
                    "artifacts_dir": str(artifact_writer.run_dir),
                }
            )
        except Exception as exc:  # noqa: BLE001
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            print(f"[run] failed: {exc}")
            batch_records.append(
                {
                    "run_id": run_id,
                    "dataset_id": args.dataset_id,
                    "question_index": question_record.get("batch_index", 1),
                    "question_id": question_record.get("question_id"),
                    "question": question,
                    "template_id": question_record.get("template_id"),
                    "primary_family": question_record.get("primary_family"),
                    "variation_axes": question_record.get("variation_axes", []),
                    "expected_sql_count": question_record.get("expected_sql_count", 2),
                    "engine": args.engine,
                    "status": "failed",
                    "error": str(exc),
                    "artifacts_dir": str(artifact_writer.run_dir),
                }
            )
            raise
        finally:
            manifest.setdefault("ended_at", datetime.now(timezone.utc).isoformat())
            manifest["duration_ms"] = _duration_ms(manifest.get("started_at"), manifest.get("ended_at"))
            artifact_writer.write_manifest(manifest)

    if batch_summary_dir is not None:
        template_counts: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        total_sql = 0
        completed = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_estimated_tokens = 0
        for row in batch_records:
            template_id = row.get("template_id")
            family = row.get("primary_family")
            if template_id:
                template_counts[template_id] = template_counts.get(template_id, 0) + 1
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1
            if row.get("status") != "failed":
                completed += 1
                total_sql += int(row.get("generated_sql_count") or 0)
                total_input_tokens += int(row.get("input_tokens") or 0)
                total_output_tokens += int(row.get("output_tokens") or 0)
                total_estimated_tokens += int(row.get("estimated_total_tokens") or 0)
        summary = {
            "dataset_id": args.dataset_id,
            "model": args.model,
            "engine": args.engine,
            "sql_source_version": "v1",
            "sql_source_label": "v1_legacy",
            "question_source": str(question_source_path.resolve()) if question_source_path else None,
            "question_count": len(batch_records),
            "completed_question_count": completed,
            "failed_question_count": len(batch_records) - completed,
            "total_generated_sql_count": total_sql,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_estimated_tokens": total_estimated_tokens,
            "template_problem_counts": template_counts,
            "family_problem_counts": family_counts,
            "cli_all_inventory": cli_all_inventory,
            "records": batch_records,
        }
        (batch_summary_dir / "batch_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (batch_summary_dir / "batch_records.jsonl").open("w", encoding="utf-8") as handle:
            for row in batch_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[batch] summary={batch_summary_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
