"""Registry-backed loader for v2 query rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import load_registry_rows


def load_v2_query_rows(registry_path: Path, *, include_sql_text: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_registry_rows(registry_path):
        query_row = dict(row)
        if include_sql_text:
            sql_path = Path(str(row.get("sql_path") or ""))
            query_row["sql"] = sql_path.read_text(encoding="utf-8") if sql_path.exists() else ""
        query_row["subitem_inference_source"] = "explicit"
        query_row["subitem_inference_note"] = "canonical_subitem_id"
        rows.append(query_row)
    return rows
