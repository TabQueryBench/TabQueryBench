"""Template catalog helpers for the v2 workload line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.settings import DATA_DIR

from .contract_spec import TEMPLATE_CONTRACTS, default_facet_ids_for_subitem


LEGACY_TEMPLATE_PATHS: tuple[Path, ...] = (
    DATA_DIR / "workload_grounding" / "template_library_v1.jsonl",
    DATA_DIR / "workload_grounding" / "template_library_extensions_v1.jsonl",
)


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _legacy_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in LEGACY_TEMPLATE_PATHS:
        if not path.exists():
            continue
        for row in _load_jsonl_rows(path):
            lookup[row["template_id"]] = row
    return lookup


def _local_sql_skeleton(template_id: str) -> str:
    skeletons = {
        "tpl_tail_low_support_group_count_v2": (
            "SELECT\n"
            "    {group_col},\n"
            "    COUNT(*) AS support\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY support ASC, {group_col}\n"
            "LIMIT {top_k};"
        ),
        "tpl_tail_pairwise_sparse_slice_v2": (
            "SELECT\n"
            "    {group_col},\n"
            "    {group_col_2},\n"
            "    COUNT(*) AS support\n"
            "FROM {table}\n"
            "GROUP BY {group_col}, {group_col_2}\n"
            "ORDER BY support ASC, {group_col}, {group_col_2}\n"
            "LIMIT {top_k};"
        ),
        "tpl_tail_target_rate_extremes_v2": (
            "SELECT\n"
            "    {group_col},\n"
            "    COUNT(*) AS support,\n"
            "    AVG(CASE WHEN {target_col} = {target_value} THEN 1 ELSE 0 END) AS focus_rate\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "HAVING COUNT(*) >= {min_support}\n"
            "ORDER BY focus_rate DESC, support ASC\n"
            "LIMIT {top_k};"
        ),
        "tpl_missing_marginal_rate_profile": (
            "SELECT\n"
            "    COUNT(*) AS total_rows,\n"
            "    SUM(CASE WHEN {missing_col} IS NULL THEN 1 ELSE 0 END) AS missing_rows,\n"
            "    AVG(CASE WHEN {missing_col} IS NULL THEN 1.0 ELSE 0.0 END) AS missing_rate\n"
            "FROM {table};"
        ),
        "tpl_missing_rate_by_subgroup": (
            "SELECT\n"
            "    {group_col},\n"
            "    COUNT(*) AS total_rows,\n"
            "    SUM(CASE WHEN {missing_col} IS NULL THEN 1 ELSE 0 END) AS missing_rows,\n"
            "    AVG(CASE WHEN {missing_col} IS NULL THEN 1.0 ELSE 0.0 END) AS missing_rate\n"
            "FROM {table}\n"
            "GROUP BY {group_col}\n"
            "ORDER BY missing_rate DESC, total_rows DESC;"
        ),
        "tpl_missing_target_interaction": (
            "SELECT\n"
            "    {target_col},\n"
            "    COUNT(*) AS total_rows,\n"
            "    SUM(CASE WHEN {missing_col} IS NULL THEN 1 ELSE 0 END) AS missing_rows,\n"
            "    AVG(CASE WHEN {missing_col} IS NULL THEN 1.0 ELSE 0.0 END) AS missing_rate\n"
            "FROM {table}\n"
            "GROUP BY {target_col}\n"
            "ORDER BY missing_rate DESC, total_rows DESC;"
        ),
        "tpl_cardinality_support_rank_profile": (
            "WITH grouped AS (\n"
            "    SELECT {group_col} AS value_label, COUNT(*) AS support\n"
            "    FROM {table}\n"
            "    GROUP BY {group_col}\n"
            ")\n"
            "SELECT\n"
            "    value_label,\n"
            "    support,\n"
            "    CAST(support AS FLOAT) / NULLIF(SUM(support) OVER (), 0) AS support_share,\n"
            "    ROW_NUMBER() OVER (ORDER BY support DESC, value_label) AS support_rank\n"
            "FROM grouped\n"
            "ORDER BY support DESC, value_label;"
        ),
        "tpl_cardinality_distinct_share_profile": (
            "WITH grouped AS (\n"
            "    SELECT {group_col} AS value_label, COUNT(*) AS support\n"
            "    FROM {table}\n"
            "    GROUP BY {group_col}\n"
            "), ranked AS (\n"
            "    SELECT\n"
            "        value_label,\n"
            "        support,\n"
            "        CAST(support AS FLOAT) / NULLIF(SUM(support) OVER (), 0) AS support_share,\n"
            "        SUM(support) OVER (ORDER BY support DESC, value_label ROWS UNBOUNDED PRECEDING) AS cumulative_support\n"
            "    FROM grouped\n"
            ")\n"
            "SELECT *\n"
            "FROM ranked\n"
            "ORDER BY support DESC, value_label;"
        ),
        "tpl_cardinality_high_card_response_stability": (
            "SELECT\n"
            "    {key_col},\n"
            "    COUNT(*) AS support,\n"
            "    AVG({measure_col}) AS avg_response\n"
            "FROM {table}\n"
            "GROUP BY {key_col}\n"
            "HAVING COUNT(*) >= {min_support}\n"
            "ORDER BY support DESC, avg_response DESC;"
        ),
    }
    return skeletons[template_id]


def _local_template_row(template_contract: Any) -> dict[str, Any]:
    provenance = {
        "url": "local://subitem_workload_v2",
        "title": "Locally authored v2 template",
        "source_query_id": template_contract.template_id,
    }
    materialization_bucket = "v2_deterministic" if template_contract.realization_mode == "deterministic" else "v2_agent"
    return {
        "template_id": template_contract.template_id,
        "template_name": template_contract.template_name,
        "source_workload_id": "subitem_workload_v2",
        "primary_family": template_contract.family_id,
        "secondary_family": None,
        "intent": template_contract.notes or f"Locally authored v2 template for {template_contract.template_name}.",
        "sql_skeleton": _local_sql_skeleton(template_contract.template_id),
        "required_roles": list(template_contract.binding_roles),
        "optional_roles": [],
        "constraints": ["single_table_only", f"v2_{template_contract.realization_mode}_template"],
        "single_table_portable": "yes",
        "provenance": provenance,
        "provenance_sources": [provenance],
        "status": "ready",
        "notes": template_contract.notes,
        "materialization_bucket": materialization_bucket,
        "activation_tier": "v2",
        "dialect_sensitive": False,
    }


def build_template_library_rows() -> list[dict[str, Any]]:
    legacy_lookup = _legacy_lookup()
    rows: list[dict[str, Any]] = []
    for contract in TEMPLATE_CONTRACTS:
        if contract.source_catalog == "template_library_v2":
            base_row = _local_template_row(contract)
        else:
            base_row = dict(legacy_lookup[contract.template_id])
        row = dict(base_row)
        row["family_id"] = contract.family_id
        row["realization_mode"] = contract.realization_mode
        row["binding_roles"] = list(contract.binding_roles)
        row["supported_canonical_subitem_ids"] = list(contract.supported_canonical_subitem_ids)
        row["allowed_variant_roles"] = list(contract.allowed_variant_roles)
        row["default_facet_ids"] = list(
            {
                facet_id
                for subitem_id in contract.supported_canonical_subitem_ids
                for facet_id in default_facet_ids_for_subitem(subitem_id)
            }
        )
        row["gate_priority"] = contract.gate_priority
        row["source_catalog"] = contract.source_catalog
        row["extended_family"] = contract.extended_family
        rows.append(row)
    rows.sort(key=lambda item: (str(item["family_id"]), str(item["template_id"])))
    return rows


def write_template_library_jsonl(output_path: Path) -> list[dict[str, Any]]:
    rows = build_template_library_rows()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_template_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {row["template_id"]: row for row in _load_jsonl_rows(path)}
