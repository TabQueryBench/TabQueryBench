"""Real synthetic-panel experiment runner for c2."""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.benchmark.models import FIVE_FIXED_FAMILIES
from src.benchmark.sql_exec import execute_sql
from src.eval.analytics_contract import (
    ANALYTICS_CONTRACT_VERSION,
    all_canonical_subitem_score_fields,
    annotate_query_row_with_contract,
    build_subitem_and_family_rows,
    canonical_subitem_score_field,
)
from src.eval.common import (
    SQL_SOURCE_VERSION_V1,
    build_sql_source_provenance,
    normalize_sql_source_version,
    read_json,
    resolve_sql_run_dir,
    split_sql_statements,
    sql_source_label,
)
from src.eval.subitem_workload_v2.paths import registry_jsonl_path, run_manifest_dir, run_sql_dir
from src.eval.subitem_workload_v2.registry import load_registry_rows
from src.evaluation.io import load_evaluation_context, write_json, write_jsonl
from src.evaluation.pipeline import run_evaluation_step2_v0_1
from src.evaluation.synthetic_validation_v4 import (
    ValidationContextV4,
    build_validation_context_v4,
    evaluate_synthetic_validation_v4,
)

ANALYTICS_FAMILIES = [family for family in FIVE_FIXED_FAMILIES if family != "cardinality_structure"]


@dataclass
class SyntheticFileRecord:
    path: Path
    dataset_id: str
    model_id: str
    synthetic_run_id: str
    file_type: str
    row_count: int
    column_names: list[str]
    schema_match_status: str
    usable: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "synthetic_run_id": self.synthetic_run_id,
            "file_type": self.file_type,
            "row_count": self.row_count,
            "column_names": self.column_names,
            "schema_match_status": self.schema_match_status,
            "usable": self.usable,
            "notes": self.notes,
        }


def _read_csv_header_and_count(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        count = 0
        for _ in reader:
            count += 1
    return [str(item) for item in header], count


def _infer_model_id(path: Path, dataset_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(dataset_root.resolve())
        parts = rel.parts
        # dataset_root is .../<panel>/<dataset_id>, so first segment is model folder.
        if len(parts) >= 1:
            return parts[0]
    except Exception:  # noqa: BLE001
        try:
            parts = path.parts
            dataset_name = dataset_root.name
            for idx, part in enumerate(parts[:-1]):
                if part == dataset_name and idx + 1 < len(parts):
                    return parts[idx + 1]
        except Exception:  # noqa: BLE001
            pass
    return path.parent.name


def _infer_synthetic_run_id(path: Path, model_id: str, dataset_id: str) -> str:
    stem = path.stem
    pattern = re.compile(rf"(?:{re.escape(model_id)}|rtf|bayesnet|ctgan|arf|tvae|tabddpm|tabpfgen)-{re.escape(dataset_id)}-\d+-(\d{{8}}_\d{{6}})")
    match = pattern.search(stem)
    if match:
        return match.group(1)
    return stem


def _schema_status(expected: list[str], observed: list[str]) -> tuple[str, list[str], bool]:
    expected_set = set(expected)
    observed_set = set(observed)

    notes: list[str] = []
    if observed == expected:
        return "exact_order_match", notes, True

    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    if not missing and not extra:
        notes.append("column_order_differs")
        return "set_match_order_diff", notes, True

    if missing:
        notes.append("missing_columns=" + ",".join(missing))
    if extra:
        notes.append("extra_columns=" + ",".join(extra))
    return "mismatch", notes, False


def inventory_panel_dataset(
    *,
    synthetic_root: Path,
    dataset_id: str,
    expected_columns: list[str],
) -> list[SyntheticFileRecord]:
    dataset_root = synthetic_root / dataset_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found under panel root: {dataset_root}")
    scan_root = dataset_root.resolve()

    records: list[SyntheticFileRecord] = []
    for root, _dirs, files in os.walk(scan_root, followlinks=True):
        for filename in sorted(files):
            path = Path(root) / filename
            if path.name.startswith("."):
                continue
            if path.suffix.lower() != ".csv":
                continue

            model_id = _infer_model_id(path, scan_root)
            synthetic_run_id = _infer_synthetic_run_id(path, model_id=model_id, dataset_id=dataset_id)

            notes: list[str] = []
            try:
                columns, row_count = _read_csv_header_and_count(path)
                schema_status, schema_notes, schema_ok = _schema_status(expected_columns, columns)
                notes.extend(schema_notes)
                usable = schema_ok and row_count > 0
                if row_count <= 0:
                    notes.append("empty_or_no_data_rows")
            except Exception as exc:  # noqa: BLE001
                columns, row_count = [], 0
                schema_status = "unreadable"
                usable = False
                notes.append(f"read_error={exc}")

            records.append(
                SyntheticFileRecord(
                    path=path,
                    dataset_id=dataset_id,
                    model_id=model_id,
                    synthetic_run_id=synthetic_run_id,
                    file_type="csv",
                    row_count=row_count,
                    column_names=columns,
                    schema_match_status=schema_status,
                    usable=usable,
                    notes=notes,
                )
            )

    return records


# Backward-compatible alias.
def inventory_panel_c2(
    *,
    synthetic_root: Path,
    dataset_id: str,
    expected_columns: list[str],
) -> list[SyntheticFileRecord]:
    return inventory_panel_dataset(
        synthetic_root=synthetic_root,
        dataset_id=dataset_id,
        expected_columns=expected_columns,
    )


def _write_inventory_csv(path: Path, records: list[SyntheticFileRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "path",
        "dataset_id",
        "model_id",
        "synthetic_run_id",
        "file_type",
        "row_count",
        "column_names",
        "schema_match_status",
        "usable",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for record in records:
            row = record.to_dict()
            row["column_names"] = json.dumps(row["column_names"], ensure_ascii=False)
            row["notes"] = json.dumps(row["notes"], ensure_ascii=False)
            writer.writerow(row)


def build_model_panel(records: list[SyntheticFileRecord], dataset_id: str) -> dict[str, Any]:
    grouped: dict[str, list[SyntheticFileRecord]] = defaultdict(list)
    for record in records:
        if record.usable:
            grouped[record.model_id].append(record)

    models: list[dict[str, Any]] = []
    for model_id in sorted(grouped.keys()):
        runs = sorted(grouped[model_id], key=lambda item: item.synthetic_run_id)
        models.append(
            {
                "model_id": model_id,
                "run_count": len(runs),
                "runs": [
                    {
                        "synthetic_run_id": item.synthetic_run_id,
                        "path": str(item.path.resolve()),
                        "row_count": item.row_count,
                        "schema_match_status": item.schema_match_status,
                    }
                    for item in runs
                ],
            }
        )

    unusable = [item for item in records if not item.usable]
    return {
        "dataset_id": dataset_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_discovered_files": len(records),
        "usable_file_count": sum(1 for item in records if item.usable),
        "unusable_file_count": len(unusable),
        "models": models,
        "unusable_files": [item.to_dict() for item in unusable],
    }


def _load_workload_queries(run_dir: Path) -> list[dict[str, Any]]:
    pkg = run_dir / "benchmark_package"
    queryspec_path = pkg / "queryspecs.json"
    queries: list[dict[str, Any]] = []

    if queryspec_path.exists():
        try:
            payload = json.loads(queryspec_path.read_text(encoding="utf-8"))
            candidates = payload.get("queryspecs") if isinstance(payload, dict) else []
            if isinstance(candidates, list):
                for item in candidates:
                    if isinstance(item, dict):
                        queries.append(item)
        except Exception:  # noqa: BLE001
            pass

    if queries:
        return queries

    # Fallback from question bundles.
    bundles_path = pkg / "question_bundles.json"
    if bundles_path.exists():
        payload = json.loads(bundles_path.read_text(encoding="utf-8"))
        bundles = payload.get("bundles") if isinstance(payload, dict) else []
        if isinstance(bundles, list):
            for bundle in bundles:
                if not isinstance(bundle, dict):
                    continue
                variants = bundle.get("variants") or []
                if not isinstance(variants, list):
                    continue
                for variant in variants:
                    if not isinstance(variant, dict):
                        continue
                    if not bool(variant.get("accepted_local", True)):
                        continue
                    spec = variant.get("query_spec")
                    if isinstance(spec, dict):
                        queries.append(spec)

    return queries


def _load_current_workload_queries(*, dataset_id: str, run_id: str, sql_source_version: str) -> list[dict[str, Any]]:
    normalized_source = normalize_sql_source_version(sql_source_version)
    registry_path = registry_jsonl_path(run_id, line_version=normalized_source)
    if not registry_path.exists():
        raise FileNotFoundError(f"Current workload registry not found for run {run_id}: {registry_path}")

    queries: list[dict[str, Any]] = []
    for row in load_registry_rows(registry_path):
        if str(row.get("dataset_id") or "").strip() != dataset_id:
            continue
        if not bool(row.get("accepted_for_eval")):
            continue
        query_record_id = str(row.get("query_record_id") or "").strip()
        sql_path = Path(str(row.get("sql_path") or "")).expanduser()
        if query_record_id and not sql_path.exists():
            sql_path = run_sql_dir(run_id, dataset_id, line_version=normalized_source) / f"{query_record_id}.sql"
        if not query_record_id or not sql_path.exists():
            continue
        manifest_path = run_manifest_dir(run_id, dataset_id, line_version=normalized_source) / query_record_id / "run_manifest.json"
        manifest = read_json(manifest_path, {}) or {}
        sql_text = sql_path.read_text(encoding="utf-8", errors="ignore")
        statements = split_sql_statements(sql_text)
        if not statements:
            continue
        dataset_dir = run_manifest_dir(run_id, dataset_id, line_version=normalized_source).parent
        run_root = dataset_dir.parent
        provenance = build_sql_source_provenance(
            sql_source_version=normalized_source,
            sql_source_kind="current_query_registry",
            sql_source_selection_mode="explicit_run_id",
            source_run_id=run_id,
            sql_file_path=sql_path,
            manifest_path=manifest_path,
            registry_path=registry_path,
            run_dir=run_root,
            dataset_dir=dataset_dir,
            registry_version=str(row.get("registry_version") or ""),
            declared_version=str(row.get("sql_source_version") or manifest.get("sql_source_version") or ""),
            declared_label=str(row.get("sql_source_label") or manifest.get("sql_source_label") or ""),
            sql_file_sha256=str(row.get("sql_sha256") or ""),
        )
        queries.append(
            {
                "query_id": query_record_id,
                "family": str(row.get("family_id") or ""),
                "family_id": str(row.get("family_id") or ""),
                "research_question": str(row.get("question_text") or ""),
                "question": str(row.get("question_text") or ""),
                "expected_output_shape": "",
                "intended_structure_claim": "",
                "sql": statements[0],
                "status": "accepted_for_eval",
                "variant_semantic_role": str(row.get("variant_semantic_role") or ""),
                "intended_facet_id": str(row.get("intended_facet_id") or ""),
                "stable_query_id": query_record_id,
                "stable_question_id": query_record_id,
                "question_id": query_record_id,
                "query_identity_stable_key": f"{dataset_id}::{query_record_id}",
                "template_id": str(row.get("template_id") or ""),
                "template_name": str(row.get("template_name") or ""),
                "canonical_subitem_id": str(row.get("canonical_subitem_id") or ""),
                "source_run_id": run_id,
                **provenance,
            }
        )
    queries.sort(key=lambda item: str(item.get("query_id") or ""))
    return queries


def _canonical_cell(value: Any) -> str:
    if value is None:
        return "<NULL>"
    return str(value)


def _rows_counter(rows: list[list[Any]]) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        key = tuple(_canonical_cell(cell) for cell in row)
        counter[key] += 1
    return counter


def _weighted_jaccard(a: Counter, b: Counter) -> float:
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 1.0
    num = 0.0
    den = 0.0
    for key in keys:
        av = float(a.get(key, 0))
        bv = float(b.get(key, 0))
        num += min(av, bv)
        den += max(av, bv)
    if den <= 0:
        return 0.0
    return num / den


_MEASURE_COL_RE = re.compile(
    r"(count|cnt|support|rate|ratio|pct|percent|prob|avg|mean|sum|min|max|std|var|score|total|share|frequency)",
    re.IGNORECASE,
)


def _project_rows_counter(rows: list[list[Any]], indices: list[int]) -> Counter:
    counter: Counter = Counter()
    if not indices:
        return counter
    for row in rows:
        key = tuple(_canonical_cell(row[idx]) if idx < len(row) else "<MISSING_COL>" for idx in indices)
        counter[key] += 1
    return counter


def _column_profile_score(real_rows: list[list[Any]], syn_rows: list[list[Any]], indices: list[int]) -> float:
    if not indices:
        return 1.0
    per_col_scores: list[float] = []
    for idx in indices:
        real_counter: Counter = Counter()
        syn_counter: Counter = Counter()
        for row in real_rows:
            real_counter[_canonical_cell(row[idx]) if idx < len(row) else "<MISSING_COL>"] += 1
        for row in syn_rows:
            syn_counter[_canonical_cell(row[idx]) if idx < len(row) else "<MISSING_COL>"] += 1
        per_col_scores.append(_weighted_jaccard(real_counter, syn_counter))
    return float(mean(per_col_scores)) if per_col_scores else 1.0


def _infer_key_column_indices(columns: list[str]) -> tuple[list[int], list[int]]:
    if not columns:
        return [], []
    measure_indices = [idx for idx, name in enumerate(columns) if _MEASURE_COL_RE.search(str(name))]
    key_indices = [idx for idx in range(len(columns)) if idx not in set(measure_indices)]

    # Fallback: if every column looks like a measure, keep dimensional columns by dropping only final column.
    if not key_indices:
        if len(columns) >= 2:
            key_indices = list(range(len(columns) - 1))
            measure_indices = [len(columns) - 1]
        else:
            key_indices = [0]
            measure_indices = []
    return key_indices, measure_indices


def _resolve_column_indices_by_name(columns: list[str], names: list[str]) -> list[int]:
    requested = [str(name) for name in names if str(name).strip()]
    if not requested:
        return []
    remaining: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(columns):
        remaining[str(name)].append(idx)

    indices: list[int] = []
    for name in requested:
        options = remaining.get(name) or []
        if not options:
            continue
        indices.append(options.pop(0))
    return indices


def _resolve_explicit_key_measure_indices(
    columns: list[str],
    annotation: dict[str, Any] | None,
) -> tuple[list[int], list[int]]:
    if not annotation:
        return [], []
    key_indices = _resolve_column_indices_by_name(columns, list(annotation.get("result_key_columns") or []))
    measure_indices = _resolve_column_indices_by_name(columns, list(annotation.get("result_measure_columns") or []))
    return key_indices, measure_indices


def _compare_execution_results(real_exec, syn_exec, *, result_role_annotation: dict[str, Any] | None = None) -> tuple[float, dict[str, Any]]:
    if not real_exec.ok:
        return 0.0, {"reason": "real_query_failed", "real_error": real_exec.error}
    if not syn_exec.ok:
        return 0.0, {"reason": "synthetic_query_failed", "synthetic_error": syn_exec.error}

    real_cols = [str(col) for col in real_exec.columns]
    syn_cols = [str(col) for col in syn_exec.columns]
    real_counter = _rows_counter(real_exec.rows)
    syn_counter = _rows_counter(syn_exec.rows)

    strict_set_score = _weighted_jaccard(real_counter, syn_counter)
    n_real = len(real_exec.rows)
    n_syn = len(syn_exec.rows)
    row_count_score = 1.0 - (abs(n_real - n_syn) / max(1, n_real, n_syn))
    row_count_score = max(0.0, min(1.0, row_count_score))

    col_inter = len(set(real_cols) & set(syn_cols))
    col_union = len(set(real_cols) | set(syn_cols))
    col_score = (col_inter / col_union) if col_union else 1.0

    key_indices, measure_indices = _resolve_explicit_key_measure_indices(real_cols, result_role_annotation)
    key_column_source = "explicit_annotation" if key_indices else "regex_fallback"
    if not key_indices:
        key_indices, measure_indices = _infer_key_column_indices(real_cols)
    key_real_counter = _project_rows_counter(real_exec.rows, key_indices)
    key_syn_counter = _project_rows_counter(syn_exec.rows, key_indices)
    key_set_score = _weighted_jaccard(key_real_counter, key_syn_counter)
    profile_score = _column_profile_score(real_exec.rows, syn_exec.rows, key_indices)

    score_weights = {
        "strict_set_score": 0.45,
        "key_set_score": 0.2,
        "profile_score": 0.15,
        "row_count_score": 0.1,
        "column_score": 0.1,
    }
    score = (
        (strict_set_score * score_weights["strict_set_score"])
        + (key_set_score * score_weights["key_set_score"])
        + (profile_score * score_weights["profile_score"])
        + (row_count_score * score_weights["row_count_score"])
        + (col_score * score_weights["column_score"])
    )
    score = max(0.0, min(1.0, score))

    return score, {
        "set_score": strict_set_score,  # backward-compatible field
        "strict_set_score": strict_set_score,
        "key_set_score": key_set_score,
        "profile_score": profile_score,
        "row_count_score": row_count_score,
        "column_score": col_score,
        "key_columns": [real_cols[idx] for idx in key_indices if idx < len(real_cols)],
        "measure_columns": [real_cols[idx] for idx in measure_indices if idx < len(real_cols)],
        "result_role_annotation_key": str(result_role_annotation.get("annotation_key") or "") if result_role_annotation else "",
        "result_role_annotation_confidence": result_role_annotation.get("confidence") if result_role_annotation else None,
        "result_role_annotation_contract_version": (
            "sql_result_role_annotation_v1" if result_role_annotation else ""
        ),
        "key_column_source": key_column_source,
        "real_rows": n_real,
        "synthetic_rows": n_syn,
        "real_columns": real_cols,
        "synthetic_columns": syn_cols,
        "query_score_method": "composite_key_profile_rowcount_column",
        "score_contract_version": "real_vs_synthetic_sql_result_v2",
        "score_weights": score_weights,
        "composite_score": score,
    }


def _mean_optional(values: list[Any]) -> float | None:
    cleaned: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            cleaned.append(float(value))
        except Exception:  # noqa: BLE001
            continue
    if not cleaned:
        return None
    return mean(cleaned)


def _has_native_missing_signal(context: ValidationContextV4) -> bool:
    # "Native missing signal" means at least one real-data column has missing rows.
    for stat in context.real_stats.values():
        if int(stat.missing_count) > 0:
            return True
    return False


def _materialize_synthetic_csv_to_sqlite(csv_path: Path, sqlite_path: Path, table_name: str) -> None:
    if sqlite_path.exists():
        sqlite_path.unlink()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.cursor()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            if not headers:
                raise ValueError(f"Empty header for synthetic CSV: {csv_path}")

            col_defs = ", ".join([f'"{h}" TEXT' for h in headers])
            cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            cur.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

            placeholders = ",".join(["?" for _ in headers])
            insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
            batch: list[list[str]] = []
            for row in reader:
                if len(row) < len(headers):
                    row = row + [""] * (len(headers) - len(row))
                elif len(row) > len(headers):
                    row = row[: len(headers)]
                batch.append(row)
                if len(batch) >= 1000:
                    cur.executemany(insert_sql, batch)
                    batch = []
            if batch:
                cur.executemany(insert_sql, batch)

        conn.commit()
    finally:
        conn.close()


def _run_workload_scoring(
    *,
    workload_run_id: str,
    workload_queries: list[dict[str, Any]],
    real_db_path: Path,
    table_name: str,
    sql_source_version: str,
    records: list[SyntheticFileRecord],
    output_dir: Path,
    validation_context_v4: ValidationContextV4,
    missingness_applicable: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    queries = workload_queries
    if not queries:
        raise RuntimeError(f"No workload queries found for run {workload_run_id}")

    # Build baseline real execution cache once per workload.
    baseline_real: dict[str, Any] = {}
    baseline_valid_query_ids: list[str] = []
    for query in queries:
        query_id = str(query.get("query_id") or "")
        sql = str(query.get("sql") or "")
        if not query_id or not sql.strip():
            continue
        exec_result = execute_sql(db_path=real_db_path, sql=sql, row_limit=1000)
        baseline_real[query_id] = exec_result
        if exec_result.ok:
            baseline_valid_query_ids.append(query_id)

    query_meta: dict[str, dict[str, Any]] = {}
    for query in queries:
        query_id = str(query.get("query_id") or "")
        if not query_id:
            continue
        family = str(query.get("family_id") or query.get("family") or "unknown")
        query_meta[query_id] = {
            "family": family,
            "sql": str(query.get("sql") or ""),
            "question": str(query.get("research_question") or query.get("question") or ""),
            "expected_output_shape": str(query.get("expected_output_shape") or ""),
            "intended_structure_claim": str(query.get("intended_structure_claim") or ""),
            "stable_query_id": str(query.get("stable_query_id") or ""),
            "stable_question_id": str(query.get("stable_question_id") or ""),
            "question_id": str(query.get("question_id") or ""),
            "variant_semantic_role": str(query.get("variant_semantic_role") or ""),
            "intended_facet_id": str(query.get("intended_facet_id") or ""),
            "query_identity_stable_key": str(query.get("query_identity_stable_key") or ""),
        }

    usable_records = [item for item in records if item.usable]

    run_level_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    subitem_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    score_table_dir = output_dir / "score_tables" / workload_run_id
    score_table_dir.mkdir(parents=True, exist_ok=True)

    for record in usable_records:
        synth_sqlite = output_dir / "sqlite_cache" / workload_run_id / record.model_id / f"{record.synthetic_run_id}.sqlite"
        _materialize_synthetic_csv_to_sqlite(record.path, synth_sqlite, table_name=table_name)

        query_scores: list[float] = []
        success_flags: list[float] = []
        record_query_rows: list[dict[str, Any]] = []

        for query_id, meta in query_meta.items():
            # v0.4.1 policy:
            # If real data has no native missing signal, analytics missingness is N/A.
            # Missingness queries are excluded from analytics scoring in this case.
            if (not missingness_applicable) and str(meta.get("family") or "") == "missingness_structure":
                continue

            real_exec = baseline_real.get(query_id)
            if real_exec is None or not real_exec.ok:
                continue
            sql = meta["sql"]
            syn_exec = execute_sql(db_path=synth_sqlite, sql=sql, row_limit=1000)
            score, detail = _compare_execution_results(real_exec, syn_exec)
            query_scores.append(score)
            success_flags.append(1.0 if syn_exec.ok else 0.0)

            record_query_rows.append(
                annotate_query_row_with_contract(
                    {
                        "workload_run_id": workload_run_id,
                        "model_id": record.model_id,
                        "synthetic_run_id": record.synthetic_run_id,
                        "query_id": query_id,
                        "stable_query_id": str(meta.get("stable_query_id") or ""),
                        "stable_question_id": str(meta.get("stable_question_id") or ""),
                        "question_id": str(meta.get("question_id") or ""),
                        "query_identity_stable_key": str(meta.get("query_identity_stable_key") or ""),
                        "family_id": meta["family"],
                        "intended_facet_id": str(meta.get("intended_facet_id") or ""),
                        "variant_semantic_role": str(meta.get("variant_semantic_role") or ""),
                        "question": str(meta.get("question") or ""),
                        "expected_output_shape": str(meta.get("expected_output_shape") or ""),
                        "intended_structure_claim": str(meta.get("intended_structure_claim") or ""),
                        "sql_source_version": sql_source_version,
                        "sql_source_label": sql_source_label(sql_source_version),
                        "sql": sql,
                        "query_score": round(score, 6),
                        "synthetic_exec_ok": syn_exec.ok,
                        "details": detail,
                    }
                )
            )

        per_query_rows.extend(record_query_rows)

        record_subitem_rows, record_family_rows = build_subitem_and_family_rows(
            query_rows=record_query_rows,
            context_fields={
                "workload_run_id": workload_run_id,
                "model_id": record.model_id,
                "synthetic_run_id": record.synthetic_run_id,
                "file_path": str(record.path.resolve()),
            },
            score_field="query_score",
            missingness_applicable=missingness_applicable,
        )
        subitem_rows.extend(record_subitem_rows)
        family_rows.extend(record_family_rows)

        overall_score = mean(query_scores) if query_scores else 0.0
        success_rate = mean(success_flags) if success_flags else 0.0

        run_row = {
            "workload_run_id": workload_run_id,
            "model_id": record.model_id,
            "synthetic_run_id": record.synthetic_run_id,
            "file_path": str(record.path.resolve()),
            "query_count": len(query_scores),
            "query_success_rate": round(success_rate, 6),
            "overall_score": round(overall_score, 6),
        }

        validation_report = evaluate_synthetic_validation_v4(
            context=validation_context_v4,
            synthetic_csv_path=record.path,
        )
        validation_scores = validation_report.get("validation_scores") if isinstance(validation_report, dict) else {}
        if not isinstance(validation_scores, dict):
            validation_scores = {}
        run_row["validation_cardinality_range_score"] = validation_scores.get("cardinality_range_score")
        run_row["validation_missing_introduction_score"] = validation_scores.get("missing_introduction_score")
        run_row["analytics_contract_version"] = ANALYTICS_CONTRACT_VERSION

        validation_rows.append(
            {
                "workload_run_id": workload_run_id,
                "model_id": record.model_id,
                "synthetic_run_id": record.synthetic_run_id,
                "file_path": str(record.path.resolve()),
                **validation_report,
            }
        )

        record_family_score_map = {
            str(row.get("family_id") or ""): row.get("family_score")
            for row in record_family_rows
        }
        for family in ANALYTICS_FAMILIES:
            fscore = record_family_score_map.get(family)
            run_row[f"{family}_score"] = fscore
        for row in record_subitem_rows:
            subitem_id = str(row.get("subitem_id") or "")
            family_id = str(row.get("family_id") or "")
            run_row[canonical_subitem_score_field(family_id, subitem_id)] = row.get("subitem_score")
        # Legacy cardinality analytics channel is now removed from real-panel model score exports
        # to avoid confusion with deterministic validation_cardinality_range_score.
        if "cardinality_structure_score" in run_row:
            run_row.pop("cardinality_structure_score", None)

        run_level_rows.append(run_row)

    # Aggregate to model-level (preserving repeats by reporting n_repeats).
    model_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_level_rows:
        model_group[row["model_id"]].append(row)

    model_level_rows: list[dict[str, Any]] = []
    for model_id, items in sorted(model_group.items(), key=lambda x: x[0]):
        agg = {
            "workload_run_id": workload_run_id,
            "model_id": model_id,
            "n_repeats": len(items),
            "overall_score": round(mean([float(item["overall_score"]) for item in items]), 6),
            "query_success_rate": round(mean([float(item["query_success_rate"]) for item in items]), 6),
            "query_count_mean": round(mean([float(item["query_count"]) for item in items]), 6),
            "analytics_contract_version": ANALYTICS_CONTRACT_VERSION,
        }
        for family in ANALYTICS_FAMILIES:
            f_mean = _mean_optional([item.get(f"{family}_score") for item in items])
            agg[f"{family}_score"] = (round(f_mean, 6) if f_mean is not None else None)
        for field in all_canonical_subitem_score_fields():
            s_mean = _mean_optional([item.get(field) for item in items])
            agg[field] = (round(s_mean, 6) if s_mean is not None else None)

        v_card = _mean_optional([item.get("validation_cardinality_range_score") for item in items])
        v_miss = _mean_optional([item.get("validation_missing_introduction_score") for item in items])
        agg["validation_cardinality_range_score"] = round(v_card, 6) if v_card is not None else None
        agg["validation_missing_introduction_score"] = round(v_miss, 6) if v_miss is not None else None
        model_level_rows.append(agg)

    # Rank by overall score.
    ranking = sorted(model_level_rows, key=lambda item: (-float(item["overall_score"]), item["model_id"]))

    # Export per-workload score table for rank-stability module.
    score_table_path = score_table_dir / "model_scores.csv"
    fieldnames = [
        "model_id",
        "overall_score",
    ] + [f"{family}_score" for family in ANALYTICS_FAMILIES]
    fieldnames += all_canonical_subitem_score_fields()
    fieldnames += [
        "analytics_contract_version",
        "validation_cardinality_range_score",
        "validation_missing_introduction_score",
    ]
    with score_table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in model_level_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    write_jsonl(score_table_dir / "model_scores_by_run.jsonl", run_level_rows)
    write_jsonl(score_table_dir / "query_scores.jsonl", per_query_rows)
    write_jsonl(score_table_dir / "subitem_scores.jsonl", subitem_rows)
    write_jsonl(score_table_dir / "validation_v4_by_run.jsonl", validation_rows)

    summary = {
        "workload_run_id": workload_run_id,
        "queryspec_count": len(query_meta),
        "baseline_valid_query_count": len(baseline_valid_query_ids),
        "model_count": len(model_level_rows),
        "score_table_path": str(score_table_path.resolve()),
        "subitem_score_table_path": str((score_table_dir / "subitem_scores.jsonl").resolve()),
        "validation_v4_by_run_path": str((score_table_dir / "validation_v4_by_run.jsonl").resolve()),
        "sql_source_version": sql_source_version,
        "sql_source_label": sql_source_label(sql_source_version),
        "analytics_contract_version": ANALYTICS_CONTRACT_VERSION,
        "ranking": [
            {
                "rank": idx + 1,
                "model_id": row["model_id"],
                "overall_score": row["overall_score"],
            }
            for idx, row in enumerate(ranking)
        ],
    }

    return model_level_rows, family_rows, subitem_rows, summary


def run_real_panel_experiment_c2(
    *,
    synthetic_root: Path,
    dataset_id: str,
    workload_run_ids: list[str],
    project_root: Path,
    output_dir: Path,
    self_eval_max_queries: int,
    sql_source_version: str = SQL_SOURCE_VERSION_V1,
    skip_self_eval: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sql_source_version = normalize_sql_source_version(sql_source_version)
    sql_source_name = sql_source_label(sql_source_version)

    real_csv_candidates = [
        project_root / "data" / dataset_id / "raw" / f"{dataset_id}-main.csv",
        project_root / "data" / dataset_id / f"{dataset_id}-main.csv",
    ]
    real_csv = next((path for path in real_csv_candidates if path.exists()), real_csv_candidates[0])
    if not real_csv.exists():
        raise FileNotFoundError(
            "Real dataset CSV not found. Checked: "
            + ", ".join(str(path) for path in real_csv_candidates)
        )
    expected_columns, real_row_count = _read_csv_header_and_count(real_csv)
    real_sqlite_path = output_dir / "sqlite_cache" / "real" / f"{dataset_id}.sqlite"
    _materialize_synthetic_csv_to_sqlite(real_csv, real_sqlite_path, table_name=dataset_id)
    validation_context_v4 = build_validation_context_v4(
        dataset_id=dataset_id,
        project_root=project_root,
        real_csv_path=real_csv,
        expected_columns=expected_columns,
    )
    missingness_applicable = _has_native_missing_signal(validation_context_v4)

    inventory = inventory_panel_dataset(
        synthetic_root=synthetic_root,
        dataset_id=dataset_id,
        expected_columns=expected_columns,
    )
    panel_manifest = build_model_panel(inventory, dataset_id=dataset_id)

    panel_inventory_payload = {
        "dataset_id": dataset_id,
        "real_schema": expected_columns,
        "real_row_count": real_row_count,
        "sql_source_version": sql_source_version,
        "sql_source_label": sql_source_name,
        "records": [item.to_dict() for item in inventory],
    }
    write_json(output_dir / f"panel_inventory_{dataset_id}.json", panel_inventory_payload)
    _write_inventory_csv(output_dir / f"panel_inventory_{dataset_id}.csv", inventory)
    write_json(output_dir / f"model_panel_{dataset_id}.json", panel_manifest)
    # Legacy c2 filenames for backward compatibility with older docs/scripts.
    if dataset_id == "c2":
        write_json(output_dir / "panel_inventory_c2.json", panel_inventory_payload)
        _write_inventory_csv(output_dir / "panel_inventory_c2.csv", inventory)
        write_json(output_dir / "model_panel_c2.json", panel_manifest)

    workload_summaries: list[dict[str, Any]] = []
    all_model_rows: list[dict[str, Any]] = []
    all_family_rows: list[dict[str, Any]] = []
    all_subitem_rows: list[dict[str, Any]] = []
    score_table_by_run: dict[str, Path] = {}

    for run_id in workload_run_ids:
        run_dir = resolve_sql_run_dir(sql_source_version=sql_source_version, run_id=run_id, dataset_id=dataset_id)
        if not run_dir.exists():
            workload_summaries.append(
                {
                    "run_id": run_id,
                    "status": "missing_run_dir",
                    "sql_source_version": sql_source_version,
                    "sql_source_label": sql_source_name,
                    "source_run_dir": str(run_dir.resolve()),
                }
            )
            continue

        try:
            if sql_source_version == SQL_SOURCE_VERSION_V1:
                workload_queries = _load_workload_queries(run_dir)
            else:
                workload_queries = _load_current_workload_queries(
                    dataset_id=dataset_id,
                    run_id=run_id,
                    sql_source_version=sql_source_version,
                )
            model_rows, family_rows, subitem_rows, summary = _run_workload_scoring(
                workload_run_id=run_id,
                workload_queries=workload_queries,
                real_db_path=real_sqlite_path,
                table_name=dataset_id,
                sql_source_version=sql_source_version,
                records=inventory,
                output_dir=output_dir,
                validation_context_v4=validation_context_v4,
                missingness_applicable=missingness_applicable,
            )
            for row in model_rows:
                all_model_rows.append(row)
            for row in family_rows:
                all_family_rows.append(row)
            for row in subitem_rows:
                all_subitem_rows.append(row)
            workload_summaries.append(
                {
                    "run_id": run_id,
                    "status": "ok",
                    "source_run_dir": str(run_dir.resolve()),
                    **summary,
                }
            )
            score_table_by_run[run_id] = Path(summary["score_table_path"])
        except Exception as exc:  # noqa: BLE001
            workload_summaries.append(
                {
                    "run_id": run_id,
                    "status": "error",
                    "error": str(exc),
                    "sql_source_version": sql_source_version,
                    "sql_source_label": sql_source_name,
                    "source_run_dir": str(run_dir.resolve()),
                }
            )

    if not all_model_rows:
        raise RuntimeError("No model scores were produced for any workload run.")

    # Persist global score tables.
    model_csv_path = output_dir / f"model_scores_{dataset_id}.csv"
    model_headers = [
        "workload_run_id",
        "model_id",
        "n_repeats",
        "overall_score",
        "query_success_rate",
        "query_count_mean",
    ] + [f"{family}_score" for family in ANALYTICS_FAMILIES]
    model_headers += all_canonical_subitem_score_fields()
    model_headers += [
        "analytics_contract_version",
        "validation_cardinality_range_score",
        "validation_missing_introduction_score",
    ]
    with model_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=model_headers)
        writer.writeheader()
        for row in all_model_rows:
            writer.writerow({key: row.get(key) for key in model_headers})

    validation_csv_path = output_dir / f"model_validation_v4_{dataset_id}.csv"
    validation_headers = [
        "workload_run_id",
        "model_id",
        "n_repeats",
        "validation_cardinality_range_score",
        "validation_missing_introduction_score",
    ]
    with validation_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=validation_headers)
        writer.writeheader()
        for row in all_model_rows:
            writer.writerow({key: row.get(key) for key in validation_headers})

    family_csv_path = output_dir / f"family_scores_{dataset_id}.csv"
    family_headers = [
        "workload_run_id",
        "model_id",
        "synthetic_run_id",
        "family_id",
        "family_score",
        "query_count",
    ]
    with family_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=family_headers)
        writer.writeheader()
        for row in all_family_rows:
            writer.writerow({key: row.get(key) for key in family_headers})

    subitem_csv_path = output_dir / f"subitem_scores_{dataset_id}.csv"
    subitem_headers = [
        "workload_run_id",
        "model_id",
        "synthetic_run_id",
        "family_id",
        "subitem_id",
        "subitem_order",
        "subitem_score",
        "query_count",
        "subitem_applicable",
        "subitem_inference_sources",
        "contract_version",
    ]
    with subitem_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=subitem_headers)
        writer.writeheader()
        for row in all_subitem_rows:
            writer.writerow({key: row.get(key) for key in subitem_headers})

    usable_workloads = [item for item in workload_summaries if item.get("status") == "ok"]
    if not usable_workloads:
        raise RuntimeError("No usable workloads for experiment.")

    # Pick primary workload with largest query count.
    primary = sorted(usable_workloads, key=lambda x: int(x.get("queryspec_count", 0)), reverse=True)[0]
    primary_run_id = str(primary["run_id"])

    by_workload_rankings: dict[str, list[dict[str, Any]]] = {}
    for run_id in sorted({row["workload_run_id"] for row in all_model_rows}):
        subset = [row for row in all_model_rows if row["workload_run_id"] == run_id]
        ranking = sorted(subset, key=lambda x: (-float(x["overall_score"]), x["model_id"]))
        by_workload_rankings[run_id] = [
            {
                "rank": idx + 1,
                "model_id": row["model_id"],
                "overall_score": row["overall_score"],
            }
            for idx, row in enumerate(ranking)
        ]

    overall_ranking_payload = {
        "dataset_id": dataset_id,
        "sql_source_version": sql_source_version,
        "sql_source_label": sql_source_name,
        "primary_workload_run_id": primary_run_id,
        "ranking_primary": by_workload_rankings.get(primary_run_id, []),
        "ranking_by_workload": by_workload_rankings,
    }
    self_eval_status = "completed"
    self_eval_skip_reason = ""
    eval_result: dict[str, Any] = {"summary": {}}
    if skip_self_eval:
        self_eval_status = "skipped_by_flag"
        self_eval_skip_reason = "User requested --skip-self-eval."
    elif sql_source_version != SQL_SOURCE_VERSION_V1:
        self_eval_status = "skipped_unsupported_source"
        self_eval_skip_reason = (
            "STEP2 self-evaluation currently requires legacy/v1 benchmark run directories "
            "with benchmark_package and build manifests; v2 registry-backed workload runs do not include those assets."
        )
    else:
        primary_context = load_evaluation_context(resolve_sql_run_dir(sql_source_version=SQL_SOURCE_VERSION_V1, run_id=primary_run_id))
        compare_runs = [
            resolve_sql_run_dir(sql_source_version=SQL_SOURCE_VERSION_V1, run_id=item["run_id"])
            for item in usable_workloads
            if item["run_id"] != primary_run_id
        ]
        self_eval_output = output_dir / "self_evaluation"
        eval_result = run_evaluation_step2_v0_1(
            context=primary_context,
            output_dir=self_eval_output,
            compare_run_dirs=compare_runs,
            score_table_overrides=score_table_by_run,
            perturb_intensities=[0.3, 0.6],
            perturb_repeats=2,
            perturb_seed=42,
            max_eval_queries=(None if self_eval_max_queries <= 0 else self_eval_max_queries),
            include_null_variant=True,
            include_boot_variant=True,
            top_k=3,
            near_duplicate_jaccard_threshold=0.92,
            alignment_pass_threshold=0.45,
            high_contamination_threshold=0.8,
        )

    selected_workloads_payload = {
        "dataset_id": dataset_id,
        "sql_source_version": sql_source_version,
        "sql_source_label": sql_source_name,
        "primary_workload_run_id": primary_run_id,
        "workloads": workload_summaries,
        "score_table_paths": {run_id: str(path) for run_id, path in score_table_by_run.items()},
        "self_evaluation_status": self_eval_status,
        "self_evaluation_skip_reason": self_eval_skip_reason,
    }
    write_json(output_dir / f"overall_ranking_{dataset_id}.json", overall_ranking_payload)
    write_json(output_dir / f"selected_workloads_{dataset_id}.json", selected_workloads_payload)
    if dataset_id == "c2":
        write_json(output_dir / "overall_ranking_c2.json", overall_ranking_payload)
        write_json(output_dir / "selected_workloads_c2.json", selected_workloads_payload)

    return {
        "dataset_id": dataset_id,
        "sql_source_version": sql_source_version,
        "sql_source_label": sql_source_name,
        "output_dir": str(output_dir.resolve()),
        "inventory_record_count": len(inventory),
        "usable_synthetic_file_count": sum(1 for item in inventory if item.usable),
        "model_count": len(panel_manifest.get("models", [])),
        "primary_workload_run_id": primary_run_id,
        "workload_summaries": workload_summaries,
        "self_evaluation_status": self_eval_status,
        "self_evaluation_skip_reason": self_eval_skip_reason,
        "self_evaluation_summary": eval_result.get("summary", {}),
    }
