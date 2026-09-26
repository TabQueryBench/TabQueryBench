"""Job-local all-applicable-template query-generation pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tqb_query.agent.local_sql_runner import instantiate_template_sql
from .binding_contract import (
    COLUMN_ROLES,
    build_agent_binding_request,
    validate_agent_bindings,
)

from .preprocessing import ExternalDatasetProfile, preprocess_csv, sanitize_identifier
from .providers import BindingProvider, BindingProviderUnavailable, BindingResponse, candidate_first_bindings


ProgressCallback = Callable[[dict[str, Any]], None]
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
BUNDLE_VERSION = "tabquerybench.ai_query_bundle.v1"


@dataclass(frozen=True)
class QueryGenerationOptions:
    dataset_id: str = "uploaded_dataset"
    target_column: str | None = None
    excluded_columns: tuple[str, ...] = ()
    max_rows: int = 1_000_000
    max_columns: int = 256
    max_file_bytes: int = 256 * 1024 * 1024
    bindings_per_template: int = 1
    grounding_attempts: int = 2
    query_timeout_seconds: float = 10.0
    max_grounding_workers: int = 4

    def __post_init__(self) -> None:
        if self.bindings_per_template != 1:
            raise ValueError("the MVP supports exactly one binding per applicable template")
        if self.grounding_attempts < 1 or self.grounding_attempts > 5:
            raise ValueError("grounding_attempts must be between 1 and 5")
        if self.max_grounding_workers < 1 or self.max_grounding_workers > 16:
            raise ValueError("max_grounding_workers must be between 1 and 16")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_templates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("template_id") or not row.get("sql_skeleton"):
                raise ValueError(f"invalid template at line {line_number}")
            rows.append(row)
    return rows


def _temporal_parameters(profile: ExternalDatasetProfile) -> dict[str, list[str]]:
    observed: list[str] = []
    for column in profile.temporal_cols:
        observed.extend(str(value) for value in profile.field_stats[column].sample_values)
    ordered = sorted(set(observed))
    if len(ordered) < 4:
        return {}
    indices = [0, len(ordered) // 3, (len(ordered) * 2) // 3, len(ordered) - 1]
    boundaries = ["'" + ordered[index].replace("'", "''") + "'" for index in indices]
    return {
        "previous_period_start": [boundaries[0]],
        "previous_period_end": [boundaries[1]],
        "current_period_start": [boundaries[2]],
        "current_period_end": [boundaries[3]],
    }


def build_grounding_request(
    profile: ExternalDatasetProfile,
    template: dict[str, Any],
    grounding_index: int,
) -> dict[str, Any]:
    request = build_agent_binding_request(
        dataset_id=profile.dataset_id,
        template_row=template,
        profile=profile,
        grounding_index=grounding_index,
    )
    parameters = request["allowed_bindings"]["parameter_candidates"]
    parameters.update(_temporal_parameters(profile))
    # Do not send raw examples for identifier/high-cardinality columns. Values
    # are only needed for categorical condition, target, and predicate choices.
    column_roles = request["allowed_bindings"].get("column_roles") or {}
    value_columns = set(column_roles.get("condition_col") or []) | set(column_roles.get("target_col") or [])
    value_columns.update(
        row["column"] for row in request["allowed_bindings"].get("predicate_candidates") or []
        if not profile.field_stats[row["column"]].is_numeric
    )
    for name in profile.high_card_cols:
        if name not in value_columns:
            request["allowed_bindings"]["observed_values_by_column"][name] = []
    for column in request["dataset"]["schema"]:
        if column["name"] not in value_columns:
            column["top_values"] = []
    return request


def applicability_issues(request: dict[str, Any]) -> list[str]:
    expected = set(request["output_contract"]["expected_binding_keys"])
    allowed = request["allowed_bindings"]
    column_roles = allowed.get("column_roles") or {}
    issues: list[str] = []
    for role in sorted(expected & COLUMN_ROLES):
        if not column_roles.get(role):
            issues.append(f"no_candidate_for_column_role:{role}")
    constraints = (request.get("template") or {}).get("role_constraints") or {}
    for pair in constraints.get("distinct_roles") or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left, right = str(pair[0]), str(pair[1])
        if left not in expected or right not in expected:
            continue
        if not any(a != b for a in column_roles.get(left, []) for b in column_roles.get(right, [])):
            issues.append(f"cannot_satisfy_distinct_roles:{left},{right}")
    if {"predicate_col", "predicate_op", "predicate_value"}.issubset(expected) and not allowed.get("predicate_candidates"):
        issues.append("no_predicate_candidate")
    if "condition_value" in expected and not any(allowed.get("observed_values_by_column", {}).get(column) for column in column_roles.get("condition_col", [])):
        issues.append("no_condition_value_candidate")
    if {"positive_value", "negative_value"} & expected:
        if not any(len(allowed.get("observed_values_by_column", {}).get(column, [])) >= 2 for column in column_roles.get("condition_col", [])):
            issues.append("no_binary_condition_candidates")
    if "target_value" in expected and not any(allowed.get("observed_values_by_column", {}).get(column) for column in column_roles.get("target_col", [])):
        issues.append("no_target_value_candidate")
    known = set(column_roles) | set(allowed.get("parameter_candidates") or {}) | {
        "predicate_col", "predicate_op", "predicate_value", "condition_value", "positive_value",
        "negative_value", "target_value", "band_cut_1", "band_cut_2", "lower_bound",
        "upper_bound", "measure_threshold",
    }
    for role in sorted(expected - known):
        issues.append(f"unsupported_placeholder:{role}")
    return issues


def _validate_temporal_order(bindings: dict[str, Any]) -> str | None:
    roles = ("previous_period_start", "previous_period_end", "current_period_start", "current_period_end")
    if not all(role in bindings for role in roles):
        return None
    values = [str(bindings[role]) for role in roles]
    return None if values == sorted(values) and len(set(values)) == 4 else "invalid_temporal_period_order"


def _execute_readonly(db_path: Path, sql: str, timeout_seconds: float) -> dict[str, Any]:
    if not re.match(r"^\s*(?:--[^\n]*\n\s*)*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("rendered SQL must start with SELECT or WITH")
    deadline = time.monotonic() + timeout_seconds
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 2_000)
    started = time.perf_counter()
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(6)
        columns = [item[0] for item in cursor.description or []]
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise TimeoutError(f"query exceeded {timeout_seconds} seconds") from exc
        raise
    finally:
        conn.close()
    if not rows:
        raise ValueError("query returned no rows")
    if all(all(value is None for value in row) for row in rows):
        raise ValueError("query returned only NULL values")
    return {
        "columns": columns,
        "preview_rows": [list(row) for row in rows[:5]],
        "preview_truncated": len(rows) > 5,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _question(template: dict[str, Any], bindings: dict[str, Any]) -> str:
    detail = ", ".join(f"{key}={value}" for key, value in bindings.items() if key.endswith("_col"))
    intent = str(template.get("intent") or template.get("template_name") or template["template_id"]).strip()
    return intent + (f" ({detail})" if detail else "")


class QueryGenerationPipeline:
    def __init__(
        self,
        *,
        binding_provider: BindingProvider,
        template_library_path: Path | None = None,
    ) -> None:
        self.binding_provider = binding_provider
        code_root = Path(__file__).resolve().parents[2]
        self.template_library_path = template_library_path or code_root / "data/workload_grounding_v8/template_library_v8.jsonl"

    def run(
        self,
        *,
        csv_path: Path,
        output_dir: Path,
        options: QueryGenerationOptions,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        started_at = _utc_now()
        csv_path = csv_path.resolve()
        output_dir = output_dir.resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        if csv_path.stat().st_size > options.max_file_bytes:
            raise ValueError(f"upload exceeds {options.max_file_bytes} bytes")
        dataset_id = sanitize_identifier(options.dataset_id, "uploaded_dataset")
        output_dir.mkdir(parents=True, exist_ok=True)
        emit = progress or (lambda _event: None)
        emit({"phase": "preprocessing", "processed_templates": 0, "accepted_queries": 0})
        profile = preprocess_csv(
            csv_path,
            output_dir=output_dir,
            dataset_id=dataset_id,
            target_column=options.target_column,
            excluded_columns=options.excluded_columns,
            max_rows=options.max_rows,
            max_columns=options.max_columns,
        )
        profile_payload = profile.summary()
        (output_dir / "dataset_profile.json").write_text(json.dumps(profile_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        templates = [row for row in _load_templates(self.template_library_path) if row.get("status") == "ready"]
        applicable: list[tuple[dict[str, Any], dict[str, Any]]] = []
        skipped: list[dict[str, Any]] = []
        for template in templates:
            request = build_grounding_request(profile, template, 0)
            issues = applicability_issues(request)
            if issues:
                skipped.append({"template_id": template["template_id"], "stage": "applicability", "reasons": issues})
            else:
                applicable.append((template, request))
        emit({"phase": "grounding", "applicable_templates": len(applicable), "processed_templates": 0, "accepted_queries": 0})

        template_lookup = {row["template_id"]: row for row in templates}
        def ground_one(item: tuple[int, dict[str, Any], dict[str, Any]]) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
            template_index, template, request = item
            records: list[dict[str, Any]] = []
            deterministic = template.get("realization_mode") == "deterministic"
            last_failure = "grounding_failed"
            for attempt in range(1, options.grounding_attempts + 1):
                try:
                    response = (
                        BindingResponse(candidate_first_bindings(request), "deterministic")
                        if deterministic else self.binding_provider.bind(request)
                    )
                    validation = validate_agent_bindings(response.bindings, request)
                    temporal_issue = _validate_temporal_order(validation.bindings)
                    if not validation.valid or temporal_issue:
                        details = [asdict(issue) for issue in validation.issues]
                        if temporal_issue:
                            details.append({"code": temporal_issue, "role": "", "detail": ""})
                        last_failure = "binding_validation_failed"
                        records.append({
                            "template_id": template["template_id"], "binding_index": 0,
                            "attempt": attempt, "accepted": False, "stage": "binding_validation",
                            "model": response.model, "bindings": response.bindings, "issues": details,
                            "usage": response.usage, "latency_ms": response.latency_ms,
                        })
                        request["retry_context"] = {
                            "previous_attempt": attempt, "reason": last_failure, "issues": details,
                            "instruction": "Choose a different binding that resolves every issue.",
                        }
                        continue
                    sql = instantiate_template_sql(
                        template_id=template["template_id"], template_lookup=template_lookup,
                        question_record={"bindings": validation.bindings, "runtime_sql_skeleton": template["sql_skeleton"]},
                        table_name=profile.sqlite_result.table_name,
                    )
                    execution = _execute_readonly(profile.sqlite_result.db_path, sql, options.query_timeout_seconds)
                    query = {
                        "template_id": template["template_id"], "template_name": template.get("template_name"),
                        "question": _question(template, validation.bindings), "bindings": validation.bindings,
                        "sql": sql, "execution": execution,
                        "semantic_result_contract": template.get("semantic_result_contract") or {},
                        "grounding": {"model": response.model, "attempt": attempt, "usage": response.usage, "latency_ms": response.latency_ms},
                    }
                    records.append({
                        "template_id": template["template_id"], "binding_index": 0,
                        "attempt": attempt, "accepted": True, "stage": "execution",
                        "model": response.model, "bindings": validation.bindings,
                        "usage": response.usage, "latency_ms": response.latency_ms,
                    })
                    return template_index, query, None, records
                except BindingProviderUnavailable:
                    raise
                except Exception as exc:  # isolate bad output/bindings/SQL for one template
                    last_failure = f"{type(exc).__name__}:{str(exc)[:300]}"
                    records.append({
                        "template_id": template["template_id"], "binding_index": 0,
                        "attempt": attempt, "accepted": False, "stage": "grounding_or_execution",
                        "error": last_failure,
                    })
                    request["retry_context"] = {
                        "previous_attempt": attempt, "reason": last_failure,
                        "instruction": "Choose a different valid binding; the previous binding did not execute successfully.",
                    }
            return template_index, None, {"template_id": template["template_id"], "stage": "grounding", "reasons": [last_failure]}, records

        work = [(index, template, request) for index, (template, request) in enumerate(applicable)]
        completed_results: dict[int, tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]] = {}
        with ThreadPoolExecutor(max_workers=min(options.max_grounding_workers, max(1, len(work)))) as executor:
            futures = {executor.submit(ground_one, item): item[0] for item in work}
            for completed_count, future in enumerate(as_completed(futures), start=1):
                index, query, skip, records = future.result()
                completed_results[index] = (query, skip, records)
                emit({
                    "phase": "grounding", "applicable_templates": len(applicable),
                    "processed_templates": completed_count,
                    "accepted_queries": sum(1 for result in completed_results.values() if result[0] is not None),
                    "skipped_templates": len(skipped) + sum(1 for result in completed_results.values() if result[1] is not None),
                })

        accepted: list[dict[str, Any]] = []
        grounding_records: list[dict[str, Any]] = []
        for index in sorted(completed_results):
            query, skip, records = completed_results[index]
            grounding_records.extend(records)
            if skip is not None:
                skipped.append(skip)
            if query is not None:
                query["query_id"] = f"q_{len(accepted) + 1:04d}_{query['template_id']}"
                accepted.append(query)

        (output_dir / "grounding_records.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in grounding_records), encoding="utf-8"
        )
        model_records = [row for row in grounding_records if row.get("model") not in {None, "deterministic"}]
        usage_totals = {
            key: sum(int((row.get("usage") or {}).get(key) or 0) for row in model_records)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        manifest = {
            "schema_version": BUNDLE_VERSION,
            "dataset_id": dataset_id,
            "created_at": _utc_now(),
            "input_sha256": _sha256(csv_path),
            "template_registry_sha256": _sha256(self.template_library_path),
            "template_count": len(templates),
            "applicable_template_count": len(applicable),
            "accepted_query_count": len(accepted),
            "skipped_template_count": len(skipped),
            "grounding": {
                "strategy": "all_applicable_templates_ai_bindings",
                "model_calls": len(model_records),
                "models": sorted({str(row["model"]) for row in model_records}),
                "usage": usage_totals,
            },
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "started_at": started_at,
        }
        bundle = {
            "schema_version": BUNDLE_VERSION,
            "manifest": manifest,
            "dataset": profile_payload,
            "queries": accepted,
            "skipped_templates": skipped,
        }
        (output_dir / "query_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "selected_queries.sql").write_text("\n\n".join(row["sql"] for row in accepted) + ("\n" if accepted else ""), encoding="utf-8")
        emit({"phase": "completed", "processed_templates": len(applicable), "accepted_queries": len(accepted)})
        return bundle
