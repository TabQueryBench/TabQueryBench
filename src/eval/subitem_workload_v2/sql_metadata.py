"""Helpers for machine-readable SQL metadata headers."""

from __future__ import annotations

from typing import Any, Mapping


SQL_METADATA_FIELDS: tuple[str, ...] = (
    "sql_source_version",
    "sql_source_label",
    "sql_source_run_id",
    "sql_source_dataset_id",
    "family_id",
    "canonical_subitem_id",
    "intended_facet_id",
    "variant_semantic_role",
    "template_id",
    "query_record_id",
    "problem_id",
    "realization_mode",
    "source_kind",
)


def prepend_sql_metadata(sql: str, metadata: Mapping[str, Any]) -> str:
    sql_body = str(sql or "").strip()
    body_lines = sql_body.splitlines()
    while body_lines:
        stripped = body_lines[0].strip()
        if not stripped.startswith("--"):
            break
        lowered = stripped.lower()
        if any(lowered.startswith(f"-- {field}:") for field in SQL_METADATA_FIELDS):
            body_lines.pop(0)
            continue
        break
    sql_body = "\n".join(body_lines).strip()
    header_lines: list[str] = []
    for field in SQL_METADATA_FIELDS:
        value = str(metadata.get(field) or "").strip()
        if value:
            header_lines.append(f"-- {field}: {value}")
    if not header_lines:
        return sql_body
    return "\n".join(header_lines + [sql_body])
