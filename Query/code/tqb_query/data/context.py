"""Build lightweight dataset context text for SQL QA prompting."""

from __future__ import annotations

from typing import Any

from tqb_query.data.bundle import DatasetBundle


def _truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _format_field_line(field: dict[str, Any]) -> str:
    name = field.get("name", "")
    role = field.get("role", "")
    semantic_type = field.get("semantic_type", "")
    ordered = field.get("ordered", False)
    value_order = field.get("value_order") or []
    desc = field.get("description", "")
    tags = field.get("field_tags") or []

    order_text = ""
    if ordered and value_order:
        order_text = f" ordered={value_order}"

    tag_text = f" tags={tags}" if tags else ""
    desc_text = f" desc={desc}" if desc else ""
    return f"- {name}: role={role}, type={semantic_type}.{order_text}{tag_text}{desc_text}"


def build_dataset_context(bundle: DatasetBundle, table_name: str) -> str:
    semantics = bundle.dataset_semantics or {}
    profile = bundle.dataset_profile or {}
    contract = bundle.dataset_contract or {}
    query_policy = bundle.query_policy or {}
    validation_policy = bundle.validation_policy or {}
    source_info = bundle.source_info or {}

    dataset_name = semantics.get("dataset_name") or profile.get("dataset_name") or bundle.dataset_id
    row_semantics = semantics.get("row_semantics") or _truncate_text(bundle.dataset_description, 400)
    task_type = semantics.get("task_type") or contract.get("task_type") or profile.get("task_type")
    target_column = semantics.get("target_column") or contract.get("target_column") or profile.get("target_column")

    lines: list[str] = []
    lines.append("Dataset context for SQL QA:")
    lines.append(f"- dataset_id: {bundle.dataset_id}")
    lines.append(f"- dataset_name: {dataset_name}")
    lines.append(f"- table_name: {table_name}")
    lines.append("- table_layout: single-table dataset (do not assume joins).")
    if row_semantics:
        lines.append(f"- row_semantics: {row_semantics}")
    if task_type:
        lines.append(f"- task_type: {task_type}")
    if target_column:
        lines.append(f"- target_column: {target_column}")

    row_counts = profile.get("row_counts") or {}
    main_rows = row_counts.get("main")
    if main_rows is not None:
        lines.append(f"- main_row_count: {main_rows}")

    fields = (bundle.field_registry or {}).get("fields") or []
    if fields:
        lines.append("- important_fields:")
        for field in fields:
            lines.append(_format_field_line(field))

    useful_combinations = query_policy.get("useful_field_combinations") or []
    if useful_combinations:
        lines.append(f"- useful_field_combinations: {useful_combinations}")

    caution_fields = validation_policy.get("fields_requiring_caution") or []
    if caution_fields:
        lines.append(f"- fields_requiring_caution: {caution_fields}")

    source_url = source_info.get("source_url")
    if source_url:
        lines.append(f"- source_url: {source_url}")

    return "\n".join(lines)
