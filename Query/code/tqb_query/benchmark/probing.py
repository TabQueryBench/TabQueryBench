"""Exploratory SQL probing for evidence-grounded understanding."""

from __future__ import annotations

from typing import Any

from tqb_query.benchmark.models import ProbeResult, StaticDatasetUnderstanding
from tqb_query.benchmark.sql_exec import execute_sql


def _sql_literal(value: Any) -> str:
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _order_case_sql(column: str, values: list[str]) -> str:
    if not values:
        return column
    clauses = [f"WHEN {_sql_literal(value)} THEN {idx + 1}" for idx, value in enumerate(values)]
    return f"CASE {column} {' '.join(clauses)} ELSE {len(values) + 1} END"


def run_exploratory_sql_probes(
    *,
    db_path,
    table_name: str,
    static_understanding: StaticDatasetUnderstanding,
    useful_field_combinations: list[list[str]],
    max_field_target_probes: int = 6,
    max_pair_probes: int = 4,
    max_ordered_checks: int = 4,
) -> list[ProbeResult]:
    probes: list[ProbeResult] = []
    target = static_understanding.target_column

    target_sql = f"""
SELECT {target} AS target_label,
       COUNT(*) AS row_count,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM {table_name}), 2) AS pct
FROM {table_name}
GROUP BY {target}
ORDER BY row_count DESC;
""".strip()
    target_exec = execute_sql(db_path, target_sql)
    probes.append(
        ProbeResult(
            probe_id="target_distribution",
            probe_type="target_distribution",
            description="Target distribution overview",
            sql=target_sql,
            row_count=len(target_exec.rows),
            columns=target_exec.columns,
            rows=target_exec.rows,
            error=target_exec.error,
        )
    )

    candidate_fields = [field for field in static_understanding.key_fields if field != target][:max_field_target_probes]
    for field in candidate_fields:
        sql = f"""
SELECT {field} AS field_value,
       {target} AS target_label,
       COUNT(*) AS row_count
FROM {table_name}
GROUP BY {field}, {target}
ORDER BY row_count DESC
LIMIT 60;
""".strip()
        result = execute_sql(db_path, sql)
        probes.append(
            ProbeResult(
                probe_id=f"field_target_{field}",
                probe_type="field_target_distribution",
                description=f"Field-target distribution for {field}",
                sql=sql,
                row_count=len(result.rows),
                columns=result.columns,
                rows=result.rows,
                error=result.error,
            )
        )

    pair_probe_count = 0
    for combo in useful_field_combinations:
        if pair_probe_count >= max_pair_probes:
            break
        fields = [str(item) for item in combo if isinstance(item, str)]
        pair_fields = [field for field in fields if field != target]
        if len(pair_fields) < 2:
            continue
        a, b = pair_fields[0], pair_fields[1]
        sql = f"""
SELECT {a} AS field_a,
       {b} AS field_b,
       {target} AS target_label,
       COUNT(*) AS support
FROM {table_name}
GROUP BY {a}, {b}, {target}
ORDER BY support DESC
LIMIT 80;
""".strip()
        result = execute_sql(db_path, sql)
        probes.append(
            ProbeResult(
                probe_id=f"pair_target_{a}_{b}",
                probe_type="pair_target_support",
                description=f"Pair-target support for {a} and {b}",
                sql=sql,
                row_count=len(result.rows),
                columns=result.columns,
                rows=result.rows,
                error=result.error,
            )
        )
        pair_probe_count += 1

    ordered_fields = [(name, values) for name, values in static_understanding.ordered_fields.items() if values]
    for field, order in ordered_fields[:max_ordered_checks]:
        order_case = _order_case_sql(field, order)
        sql = f"""
SELECT {field} AS field_value,
       COUNT(*) AS row_count
FROM {table_name}
GROUP BY {field}
ORDER BY {order_case};
""".strip()
        result = execute_sql(db_path, sql)
        probes.append(
            ProbeResult(
                probe_id=f"ordered_values_{field}",
                probe_type="ordered_values",
                description=f"Ordered-category support check for {field}",
                sql=sql,
                row_count=len(result.rows),
                columns=result.columns,
                rows=result.rows,
                error=result.error,
            )
        )

    return probes
