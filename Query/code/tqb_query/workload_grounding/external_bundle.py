"""Public, dependency-light query-bundle generation for a user supplied CSV.

This module intentionally does not import the benchmark scorer.  It materializes a
single CSV in SQLite, binds compatible entries from the public v8 template registry,
and keeps only SQL that SQLite can execute against the supplied data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUNDLE_SCHEMA_VERSION = "tabquerybench.query_bundle.v1"
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_IDENTIFIER_ROLES = {
    "table", "group_col", "group_col_2", "measure_col", "key_col", "entity_col",
    "predicate_col", "condition_col", "target_col", "band_col", "missing_col", "item_col",
}
_LITERAL_ROLES = {"predicate_value", "condition_value", "positive_value", "negative_value"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    return value


def _load_templates(path: Path) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not row.get("template_id") or not row.get("sql_skeleton"):
                    raise ValueError(f"invalid template row at {path}:{line_number}")
                templates.append(row)
    return templates


def _is_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def inspect_csv(csv_path: Path) -> dict[str, Any]:
    """Return a compact, JSON-serializable profile used for automatic bindings."""
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must have a header row")
        names = [str(name).strip() for name in reader.fieldnames]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("CSV headers must be non-empty and unique")
        values: dict[str, list[str]] = {name: [] for name in names}
        missing: Counter[str] = Counter()
        row_count = 0
        for row in reader:
            row_count += 1
            for name in names:
                value = (row.get(name) or "").strip()
                if value == "":
                    missing[name] += 1
                else:
                    values[name].append(value)
    if row_count == 0:
        raise ValueError("CSV contains no data rows")
    columns = []
    for name in names:
        observed = values[name]
        numeric = bool(observed) and sum(_is_number(value) for value in observed) / len(observed) >= 0.95
        counts = Counter(observed)
        columns.append({
            "name": name,
            "inferred_type": "numeric" if numeric else "categorical",
            "non_null_count": len(observed),
            "missing_count": missing[name],
            "distinct_count": len(counts),
            "top_values": [value for value, _count in counts.most_common(8)],
            "numeric_values": [float(value) for value in observed if _is_number(value)] if numeric else [],
        })
    return {"row_count": row_count, "columns": columns}


def _metadata_roles(metadata: dict[str, Any]) -> dict[str, str]:
    roles = metadata.get("roles") or metadata.get("bindings") or {}
    if not isinstance(roles, dict):
        raise ValueError("metadata.roles must be an object mapping template roles to CSV column names")
    return {str(key): str(value) for key, value in roles.items() if value is not None}


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _default_bindings(profile: dict[str, Any], roles: dict[str, str]) -> dict[str, Any]:
    columns = profile["columns"]
    numeric = [column for column in columns if column["inferred_type"] == "numeric"]
    categorical = [column for column in columns if column["inferred_type"] != "numeric"]
    groupable = sorted(categorical, key=lambda column: (column["distinct_count"] > 50, column["distinct_count"], column["name"]))
    if not groupable:
        groupable = sorted(columns, key=lambda column: (column["distinct_count"], column["name"]))
    high_cardinality = sorted(columns, key=lambda column: (-column["distinct_count"], column["name"]))
    measure = numeric[0]["name"] if numeric else None
    group = groupable[0]["name"] if groupable else None
    second_group = next((column["name"] for column in groupable if column["name"] != group), None)
    key = next((column["name"] for column in high_cardinality if column["name"] != group), group)
    predicate = next((column for column in groupable if column["top_values"] and column["name"] != group), groupable[0] if groupable else None)
    predicate_value = predicate["top_values"][0] if predicate and predicate["top_values"] else None
    condition = next((column for column in groupable if column["distinct_count"] == 2), predicate)
    condition_values = condition["top_values"] if condition else []
    measure_values = numeric[0]["numeric_values"] if numeric else []
    defaults: dict[str, Any] = {
        "group_col": group, "group_col_2": second_group, "measure_col": measure,
        "key_col": key, "entity_col": key, "item_col": key, "predicate_col": predicate["name"] if predicate else None,
        "condition_col": condition["name"] if condition else None, "target_col": group or measure,
        "band_col": measure, "missing_col": columns[0]["name"] if columns else None,
        "predicate_op": "=", "predicate_value": predicate_value, "condition_value": condition_values[0] if condition_values else None,
        "positive_value": condition_values[0] if condition_values else None,
        "negative_value": condition_values[1] if len(condition_values) > 1 else None,
        "top_k": min(10, max(1, len(groupable))), "min_support": max(1, min(20, profile["row_count"] // 5)),
        "min_group_size": max(1, min(20, profile["row_count"] // 5)), "percentile": 0.95,
        "threshold": _quantile(measure_values, 0.95), "band_cut_1": _quantile(measure_values, 0.33),
        "band_cut_2": _quantile(measure_values, 0.66), "band_cut_3": _quantile(measure_values, 0.75),
    }
    defaults.update(roles)
    return {key: value for key, value in defaults.items() if value is not None}


def _render(template: dict[str, Any], bindings: dict[str, Any], table_name: str) -> str:
    values = dict(bindings)
    values["table"] = table_name
    unresolved = sorted(set(_PLACEHOLDER_RE.findall(template["sql_skeleton"])))
    missing = [key for key in unresolved if key not in values]
    if missing:
        raise ValueError("unsupported_or_unbound_roles:" + ",".join(missing))
    rendered = template["sql_skeleton"]
    for key in unresolved:
        value = values[key]
        if key in _IDENTIFIER_ROLES:
            replacement = _quote_identifier(str(value))
        elif key in _LITERAL_ROLES:
            replacement = _sql_literal(value)
        else:
            replacement = str(value)
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered.strip().rstrip(";") + ";"


def _materialize_csv(csv_path: Path, db_path: Path, table_name: str, columns: list[dict[str, Any]]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        definitions = ", ".join(
            f"{_quote_identifier(column['name'])} {'REAL' if column['inferred_type'] == 'numeric' else 'TEXT'}"
            for column in columns
        )
        conn.execute(f"CREATE TABLE {_quote_identifier(table_name)} ({definitions})")
        names = [column["name"] for column in columns]
        sql = f"INSERT INTO {_quote_identifier(table_name)} ({', '.join(_quote_identifier(name) for name in names)}) VALUES ({', '.join('?' for _ in names)})"
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = []
            numeric_names = {column["name"] for column in columns if column["inferred_type"] == "numeric"}
            for row in reader:
                rows.append(tuple(None if (row.get(name) or "").strip() == "" else (float(row[name]) if name in numeric_names else row[name]) for name in names))
            conn.executemany(sql, rows)
        conn.commit()
    finally:
        conn.close()


def generate_query_bundle(
    *, csv_path: str | Path, output_dir: str | Path, metadata_path: str | Path | None = None,
    dataset_id: str | None = None, template_library_path: str | Path | None = None, max_queries: int = 25,
) -> dict[str, Any]:
    """Generate a portable query bundle and write ``query_bundle.json`` to ``output_dir``."""
    started = time.perf_counter()
    started_at = _utc_now()
    csv_file = Path(csv_path).expanduser().resolve()
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_file = Path(metadata_path).expanduser().resolve() if metadata_path else None
    metadata = _load_json(metadata_file)
    code_root = Path(__file__).resolve().parents[2]
    library_file = Path(template_library_path).expanduser().resolve() if template_library_path else code_root / "data" / "workload_grounding_v8" / "template_library_v8.jsonl"
    if not library_file.is_file():
        raise FileNotFoundError(f"template library not found: {library_file}")
    profile = inspect_csv(csv_file)
    bundle_id = dataset_id or str(metadata.get("dataset_id") or csv_file.stem)
    table_name = str(metadata.get("table_name") or "source_data")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError("metadata.table_name must be a simple SQL identifier")
    bindings = _default_bindings(profile, _metadata_roles(metadata))
    db_path = output / "input.sqlite"
    if db_path.exists():
        db_path.unlink()
    _materialize_csv(csv_file, db_path, table_name, profile["columns"])
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    conn = sqlite3.connect(db_path)
    try:
        for template in _load_templates(library_file):
            if len(accepted) >= max_queries:
                break
            if template.get("status") != "ready":
                continue
            try:
                sql = _render(template, bindings, table_name)
                cursor = conn.execute(sql)
                preview = cursor.fetchmany(5)
                columns = [item[0] for item in cursor.description or []]
                if not preview:
                    skipped.append({"template_id": template["template_id"], "reason": "empty_result"})
                    continue
            except (sqlite3.Error, ValueError) as exc:
                skipped.append({"template_id": template["template_id"], "reason": str(exc)[:240]})
                continue
            used_roles = sorted(set(_PLACEHOLDER_RE.findall(template["sql_skeleton"])))
            accepted.append({
                "query_id": f"q_{len(accepted) + 1:04d}_{template['template_id']}",
                "question": template.get("intent") or template.get("template_name") or template["template_id"],
                "sql": "-- template_id: " + template["template_id"] + "\n" + sql,
                "output": {"columns": columns, "preview_rows": [list(row) for row in preview]},
                "provenance": {
                    "template_id": template["template_id"], "template_name": template.get("template_name"),
                    "template_sha256": hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest(),
                    "source_workload_id": template.get("source_workload_id"), "source_catalog": template.get("source_catalog"),
                    "source": template.get("provenance"), "bindings": {role: bindings[role] for role in used_roles if role in bindings},
                },
                "semantic_result_contract": template.get("semantic_result_contract", {}),
            })
    finally:
        conn.close()
    finished = _utc_now()
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION, "dataset_id": bundle_id, "created_at": finished,
        "inputs": {"csv": {"path": str(csv_file), "sha256": _sha256(csv_file)}, "metadata": None if metadata_file is None else {"path": str(metadata_file), "sha256": _sha256(metadata_file)}},
        "template_registry": {"path": str(library_file), "sha256": _sha256(library_file)},
        "execution": {"engine": "sqlite", "sqlite_version": sqlite3.sqlite_version, "table_name": table_name, "row_count": profile["row_count"]},
        "usage": {"network_calls": 0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "timing": {"started_at": started_at, "completed_at": finished, "elapsed_seconds": round(time.perf_counter() - started, 6)},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    payload = {"schema_version": BUNDLE_SCHEMA_VERSION, "manifest": manifest, "dataset": {"columns": [{key: value for key, value in column.items() if key != "numeric_values"} for column in profile["columns"]]}, "queries": accepted, "skipped_templates": skipped}
    bundle_path = output / "query_bundle.json"
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "selected_queries.sql").write_text("\n\n".join(query["sql"] for query in accepted) + ("\n" if accepted else ""), encoding="utf-8")
    return {"bundle_path": bundle_path, "manifest_path": output / "manifest.json", "query_count": len(accepted), "skipped_template_count": len(skipped), "bundle": payload}
