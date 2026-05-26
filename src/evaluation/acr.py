"""Anti-Cherry-picking Robustness (ACR) evaluation."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any

from src.benchmark.sql_exec import execute_sql

SUPPORT_HINTS = {"support", "count", "total", "n", "freq"}
METRIC_HINTS = {"rate", "ratio", "pct", "percent", "score", "mean", "avg"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _trimmed_mean(values: list[float], trim_ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n <= 2:
        return _mean(ordered)
    k = int(math.floor(n * max(0.0, min(0.49, trim_ratio))))
    if k <= 0:
        return _mean(ordered)
    kept = ordered[k : n - k]
    if not kept:
        return _mean(ordered)
    return _mean(kept)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _normalize_claim_type(raw: str) -> str:
    text = (raw or "").strip().lower()
    mapping = {
        "distribution": "higher_lower_comparison",
        "rate": "higher_lower_comparison",
        "contrast": "higher_lower_comparison",
        "ranking": "higher_lower_comparison",
        "higher_lower": "higher_lower_comparison",
        "higher_lower_comparison": "higher_lower_comparison",
        "monotonic_trend": "monotonic_trend",
        "trend": "monotonic_trend",
        "association": "association_direction",
        "association_direction": "association_direction",
        "rare": "rare_pattern_presence",
        "rare_pattern_presence": "rare_pattern_presence",
    }
    return mapping.get(text, "higher_lower_comparison")


def _fallback_question_id(spec: dict[str, Any], query_id: str) -> str:
    explicit = str(spec.get("question_id") or "").strip()
    if explicit:
        return explicit
    stable = str(spec.get("stable_question_id") or "").strip()
    if stable:
        return stable
    rq = str(spec.get("research_question") or "").strip()
    if rq:
        normalized = " ".join(rq.lower().split())
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        return f"rq_{digest}"
    if query_id:
        return f"qgrp_{query_id.split('_v')[0]}"
    return "unknown"


def _first_match(pattern: str, text: str) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return (m.group(1) or "").strip()


def _extract_groupby_columns(sql: str) -> list[str]:
    clause = _first_match(r"\bgroup\s+by\s+(.+?)(?:\border\s+by\b|\bhaving\b|\blimit\b|$)", sql)
    if not clause:
        return []
    parts = [item.strip() for item in clause.split(",")]
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        token = re.sub(r"\s+as\s+\w+$", "", part, flags=re.IGNORECASE).strip()
        if token and token not in out:
            out.append(token)
    return out


def _extract_where_clause(sql: str) -> str:
    return _first_match(r"\bwhere\b\s+(.+?)(?:\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|$)", sql)


def _split_filters(where_clause: str) -> list[str]:
    if not where_clause:
        return []
    parts = re.split(r"\s+and\s+", where_clause, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def _replace_where_clause(sql: str, new_where: str) -> str:
    if re.search(r"\bwhere\b", sql, flags=re.IGNORECASE):
        return re.sub(
            r"(\bwhere\b\s+)(.+?)(?=(\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|$))",
            lambda m: f"{m.group(1)}{new_where.strip()} ",
            sql,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )

    insert_after_from = re.search(r"\bfrom\b\s+\S+", sql, flags=re.IGNORECASE)
    if not insert_after_from:
        return sql
    idx = insert_after_from.end()
    return sql[:idx] + f" WHERE {new_where.strip()} " + sql[idx:]


def _append_where_condition(sql: str, condition: str) -> str:
    where_clause = _extract_where_clause(sql)
    if where_clause:
        new_where = f"({where_clause}) AND ({condition})"
        return _replace_where_clause(sql, new_where)
    return _replace_where_clause(sql, condition)


def _rewrite_shift_having_count(sql: str, factor: float) -> str:
    pattern = re.compile(r"(\bhaving\b\s+count\s*\(\s*\*\s*\)\s*>=\s*)(\d+)", flags=re.IGNORECASE)
    m = pattern.search(sql)
    if not m:
        return ""
    base = int(m.group(2))
    shifted = max(1, int(round(base * factor)))
    return sql[: m.start(2)] + str(shifted) + sql[m.end(2) :]


def _rewrite_shift_limit(sql: str, factor: float) -> str:
    pattern = re.compile(r"(\blimit\b\s+)(\d+)", flags=re.IGNORECASE)
    m = pattern.search(sql)
    if not m:
        return ""
    base = int(m.group(2))
    shifted = max(1, int(round(base * factor)))
    return sql[: m.start(2)] + str(shifted) + sql[m.end(2) :]


def _infer_contract(
    *,
    spec: dict[str, Any],
    refinement_catalog: dict[str, list[str]],
) -> dict[str, Any]:
    sql = str(spec.get("sql") or "")
    family_id = str(spec.get("family_id") or spec.get("family") or "unknown")

    groupby_columns = list(spec.get("groupby_columns") or [])
    if not groupby_columns:
        groupby_columns = _extract_groupby_columns(sql)

    where_clause = _extract_where_clause(sql)
    parsed_filters = _split_filters(where_clause)

    base_filters = list(spec.get("base_filters") or [])
    optional_filters = list(spec.get("optional_filters") or [])
    if not base_filters and parsed_filters:
        if len(parsed_filters) == 1:
            optional_filters = optional_filters or [parsed_filters[0]]
        else:
            base_filters = [parsed_filters[0]]
            optional_filters = optional_filters or parsed_filters[1:]

    allowed_refinement_columns = list(spec.get("allowed_refinement_columns") or [])
    if not allowed_refinement_columns:
        allowed_refinement_columns = list(refinement_catalog.get(family_id) or [])

    claim_type = _normalize_claim_type(str(spec.get("claim_type") or ""))

    frozen_slots = list(spec.get("frozen_slots") or [])
    if not frozen_slots:
        frozen_slots = [
            "base_table",
            "join_graph",
            "aggregate_type",
            "measure_column",
            "comparison_entities",
            "direction_semantics",
            "mandatory_filters",
            "family_label",
        ]
    editable_slots = list(spec.get("editable_slots") or [])
    if not editable_slots:
        editable_slots = ["optional_filter", "threshold_adjacent_bin", "refinement_column", "population_step"]

    return {
        "claim_type": claim_type,
        "groupby_columns": groupby_columns,
        "base_filters": base_filters,
        "optional_filters": optional_filters,
        "allowed_refinement_columns": allowed_refinement_columns,
        "frozen_slots": frozen_slots,
        "editable_slots": editable_slots,
        "direction": str(spec.get("direction") or "unknown"),
    }


def build_refinement_catalog(
    *,
    static_understanding: dict[str, Any],
    query_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    key_fields = [str(item) for item in (static_understanding.get("key_fields") or []) if item]
    target_column = str(static_understanding.get("target_column") or "")

    family_columns: dict[str, list[str]] = defaultdict(list)
    for spec in query_specs:
        if not isinstance(spec, dict):
            continue
        family_id = str(spec.get("family_id") or spec.get("family") or "unknown")
        cols = []
        cols.extend([str(c) for c in (spec.get("related_columns") or []) if c])
        cols.extend([str(c) for c in (spec.get("source_columns") or []) if c])
        cols.extend([str(c) for c in (spec.get("subgroup_columns") or []) if c])
        cols.extend([str(c) for c in (spec.get("feature_columns") or []) if c])
        for col in cols:
            if not col or col == target_column:
                continue
            if col not in family_columns[family_id]:
                family_columns[family_id].append(col)

    # Keep deterministic, compact family-specific candidates.
    catalog: dict[str, list[str]] = {}
    for family_id, cols in sorted(family_columns.items(), key=lambda x: x[0]):
        merged = list(cols)
        for col in key_fields:
            if col == target_column:
                continue
            if col not in merged:
                merged.append(col)
        catalog[family_id] = merged[:8]

    return {
        "contract_version": "acr_refinement_catalog_v0_1",
        "target_column": target_column,
        "by_family": catalog,
    }


def _top_value_for_column(db_path: str, table_name: str, column: str) -> str | None:
    sql = (
        f"SELECT {column}, COUNT(*) AS support FROM {table_name} "
        f"WHERE {column} IS NOT NULL GROUP BY {column} ORDER BY support DESC LIMIT 1"
    )
    res = execute_sql(db_path=db_path, sql=sql)
    if not res.ok or not res.rows:
        return None
    value = res.rows[0][0] if res.rows[0] else None
    if value is None:
        return None
    return str(value)


def _quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _generate_rewrites_for_query(
    *,
    spec: dict[str, Any],
    contract: dict[str, Any],
    db_path: str,
    table_name: str,
) -> list[dict[str, Any]]:
    sql = str(spec.get("sql") or "").strip()
    if not sql:
        return []

    rewrites: list[dict[str, Any]] = []
    seen_sql: set[str] = set()

    def _append(template_type: str, candidate_sql: str, note: str) -> None:
        normalized = " ".join(candidate_sql.strip().split())
        if not normalized:
            return
        key = normalized.lower().rstrip(";")
        if key in seen_sql:
            return
        if key == " ".join(sql.strip().split()).lower().rstrip(";"):
            return
        seen_sql.add(key)
        rewrites.append(
            {
                "template_type": template_type,
                "sql": candidate_sql if candidate_sql.strip().endswith(";") else candidate_sql.strip() + ";",
                "note": note,
            }
        )

    # Template 1: refinement rewrite (up to 4)
    groupby_cols = [str(col) for col in (contract.get("groupby_columns") or []) if col]
    allowed_refinements = [str(col) for col in (contract.get("allowed_refinement_columns") or []) if col]
    refinement_candidates = [col for col in allowed_refinements if col not in groupby_cols][:4]
    for col in refinement_candidates:
        _append(
            "refinement_rewrite",
            _append_where_condition(sql, f"{col} IS NOT NULL"),
            f"refinement_slice:{col}_not_null",
        )

    # Template 2: filter-neighborhood rewrite (up to 2)
    shifted_up = _rewrite_shift_having_count(sql, factor=1.2)
    shifted_dn = _rewrite_shift_having_count(sql, factor=0.8)
    if shifted_up:
        _append("filter_neighborhood_rewrite", shifted_up, "having_count_shift_up")
    if shifted_dn:
        _append("filter_neighborhood_rewrite", shifted_dn, "having_count_shift_down")
    if len([row for row in rewrites if row["template_type"] == "filter_neighborhood_rewrite"]) < 2:
        limit_up = _rewrite_shift_limit(sql, factor=1.25)
        limit_dn = _rewrite_shift_limit(sql, factor=0.8)
        if limit_up:
            _append("filter_neighborhood_rewrite", limit_up, "limit_shift_up")
        if limit_dn:
            _append("filter_neighborhood_rewrite", limit_dn, "limit_shift_down")

    # Template 3: population-neighborhood rewrite (up to 2)
    optional_filters = [str(item) for item in (contract.get("optional_filters") or []) if item]
    if optional_filters:
        where_clause = _extract_where_clause(sql)
        filters = _split_filters(where_clause)
        if filters:
            dropped = [item for item in filters if item != optional_filters[0]]
            if dropped:
                _append(
                    "population_neighborhood_rewrite",
                    _replace_where_clause(sql, " AND ".join(dropped)),
                    "drop_one_optional_filter",
                )

    population_col = ""
    for col in groupby_cols + allowed_refinements:
        if col:
            population_col = col
            break
    if population_col:
        top_value = _top_value_for_column(db_path=db_path, table_name=table_name, column=population_col)
        if top_value is not None:
            cond = f"{population_col} = {_quote_sql(top_value)}"
            _append("population_neighborhood_rewrite", _append_where_condition(sql, cond), f"high_support_subgroup:{population_col}")

    # Enforce template caps: 4 + 2 + 2
    out: list[dict[str, Any]] = []
    caps = {
        "refinement_rewrite": 4,
        "filter_neighborhood_rewrite": 2,
        "population_neighborhood_rewrite": 2,
    }
    counts = defaultdict(int)
    for row in rewrites:
        t = row["template_type"]
        if counts[t] >= caps.get(t, 0):
            continue
        counts[t] += 1
        out.append(row)
    return out


def _numeric_columns(columns: list[str], rows: list[list[Any]]) -> list[int]:
    idxs: list[int] = []
    for idx, _ in enumerate(columns):
        ok = False
        for row in rows[: min(50, len(rows))]:
            if idx >= len(row):
                continue
            try:
                float(row[idx])
                ok = True
                break
            except (TypeError, ValueError):
                continue
        if ok:
            idxs.append(idx)
    return idxs


def _support_from_result(columns: list[str], rows: list[list[Any]]) -> float:
    if not rows:
        return 0.0
    numeric_idx = _numeric_columns(columns, rows)
    if not numeric_idx:
        return float(len(rows))

    support_idx = []
    for idx, name in enumerate(columns):
        lower = str(name).lower()
        if any(token in lower for token in SUPPORT_HINTS):
            support_idx.append(idx)
    candidates = [idx for idx in support_idx if idx in numeric_idx] or numeric_idx
    values = []
    for row in rows:
        for idx in candidates:
            if idx >= len(row):
                continue
            try:
                values.append(float(row[idx]))
            except (TypeError, ValueError):
                continue
    if not values:
        return float(len(rows))
    return float(sum(max(0.0, value) for value in values))


def _result_non_trivial(columns: list[str], rows: list[list[Any]]) -> bool:
    if not rows:
        return False
    numeric_idx = _numeric_columns(columns, rows)
    if not numeric_idx:
        return len(rows) >= 2
    values: list[float] = []
    for row in rows[:500]:
        for idx in numeric_idx:
            if idx >= len(row):
                continue
            try:
                values.append(float(row[idx]))
            except (TypeError, ValueError):
                continue
    if not values:
        return len(rows) >= 2
    return (max(values) - min(values)) > 1e-9


def _extract_direction(
    *,
    claim_type: str,
    columns: list[str],
    rows: list[list[Any]],
    ordered_fields: dict[str, list[str]],
) -> str:
    if not rows:
        return "indeterminate"

    numeric_idx = _numeric_columns(columns, rows)
    if not numeric_idx:
        return "indeterminate"

    metric_idx = numeric_idx[0]
    for idx, name in enumerate(columns):
        lower = str(name).lower()
        if any(token in lower for token in METRIC_HINTS):
            if idx in numeric_idx:
                metric_idx = idx
                break
    if claim_type == "rare_pattern_presence":
        # Prefer support-like metric for rarity presence checks.
        for idx, name in enumerate(columns):
            if idx in numeric_idx and "support" in str(name).lower():
                metric_idx = idx
                break

    key_idx = next((idx for idx in range(len(columns)) if idx not in numeric_idx), None)
    if key_idx is None:
        vals = []
        for row in rows:
            if metric_idx < len(row):
                vals.append(_to_float(row[metric_idx]))
        if len(vals) < 2:
            return "indeterminate"
        diff = max(vals) - min(vals)
        if abs(diff) <= 1e-9:
            return "indeterminate"
        return "positive" if diff > 0 else "negative"

    # Aggregate metric by key.
    agg: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if key_idx >= len(row) or metric_idx >= len(row):
            continue
        key = "<NULL>" if row[key_idx] is None else str(row[key_idx])
        agg[key].append(_to_float(row[metric_idx]))
    if len(agg) < 2:
        return "indeterminate"
    metric_by_key = {k: _mean(vs) for k, vs in agg.items()}

    if claim_type == "monotonic_trend":
        field_name = str(columns[key_idx])
        order = ordered_fields.get(field_name) or ordered_fields.get(field_name.lower()) or []
        if not order:
            sorted_items = sorted(metric_by_key.items(), key=lambda x: x[0])
            if len(sorted_items) < 2:
                return "indeterminate"
            delta = sorted_items[-1][1] - sorted_items[0][1]
            if abs(delta) < 1e-9:
                return "indeterminate"
            return "increasing" if delta > 0 else "decreasing"
        index_map = {str(v): idx for idx, v in enumerate(order)}
        pairs = [(index_map[k], v) for k, v in metric_by_key.items() if k in index_map]
        if len(pairs) < 2:
            return "indeterminate"
        pairs.sort(key=lambda x: x[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        x_mean = _mean([float(x) for x in xs])
        y_mean = _mean(ys)
        cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        var_x = sum((x - x_mean) ** 2 for x in xs)
        var_y = sum((y - y_mean) ** 2 for y in ys)
        if var_x <= 1e-12 or var_y <= 1e-12:
            return "indeterminate"
        corr = cov / math.sqrt(var_x * var_y)
        if abs(corr) < 0.05:
            return "indeterminate"
        return "increasing" if corr > 0 else "decreasing"

    sorted_items = sorted(metric_by_key.items(), key=lambda x: x[1], reverse=True)
    top_key, top_val = sorted_items[0]
    second_val = sorted_items[1][1]
    denom = max(abs(top_val), abs(second_val), 1e-9)
    if abs(top_val - second_val) / denom < 0.05:
        return "indeterminate"
    return f"top:{top_key}"


def evaluate_acr(
    *,
    query_specs: list[dict[str, Any]],
    db_path: str,
    table_name: str,
    static_understanding: dict[str, Any],
    max_eval_queries: int | None = None,
    support_min_ratio: float = 0.20,
    support_min_abs: float = 3.0,
    support_weight_clip_min: float = 0.25,
    support_weight_clip_max: float = 1.0,
    min_evaluable_valid_rewrites: int = 2,
    confidence_valid_rewrites: int = 4,
    question_trim_ratio: float = 0.20,
    query_floor_threshold: float = 0.15,
    question_floor_cap: float = 0.60,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ordered_specs = [item for item in query_specs if isinstance(item, dict)]
    ordered_specs.sort(key=lambda x: str(x.get("query_id") or ""))
    if max_eval_queries is not None and max_eval_queries > 0:
        ordered_specs = ordered_specs[:max_eval_queries]

    refinement_catalog = build_refinement_catalog(static_understanding=static_understanding, query_specs=ordered_specs)
    by_family_catalog = refinement_catalog.get("by_family", {}) if isinstance(refinement_catalog, dict) else {}
    if not isinstance(by_family_catalog, dict):
        by_family_catalog = {}

    ordered_fields = static_understanding.get("ordered_fields") or {}
    if not isinstance(ordered_fields, dict):
        ordered_fields = {}

    query_rows: list[dict[str, Any]] = []
    rewrite_rows: list[dict[str, Any]] = []

    for spec in ordered_specs:
        query_id = str(spec.get("query_id") or "")
        stable_query_id = str(spec.get("stable_query_id") or "")
        question_id = _fallback_question_id(spec, query_id=query_id)
        stable_question_id = str(spec.get("stable_question_id") or question_id)
        family_id = str(spec.get("family_id") or spec.get("family") or "unknown")
        intended_facet_id = str(spec.get("intended_facet_id") or "unknown")
        sql = str(spec.get("sql") or "").strip()
        if not sql:
            continue

        contract = _infer_contract(spec=spec, refinement_catalog=by_family_catalog)
        claim_type = contract["claim_type"]

        original_exec = execute_sql(db_path=db_path, sql=sql)
        if not original_exec.ok:
            query_rows.append(
                {
                    "query_id": query_id,
                    "stable_query_id": stable_query_id,
                    "question_id": question_id,
                    "stable_question_id": stable_question_id,
                    "family_id": family_id,
                    "intended_facet_id": intended_facet_id,
                    "execution_ok": False,
                    "query_evaluable": False,
                    "claim_type": claim_type,
                    "direction_original": "indeterminate",
                    "valid_rewrite_count": 0,
                    "acr_score": None,
                    "query_pass": False,
                    "reason_codes": ["ACR_ORIGINAL_QUERY_EXECUTION_FAILED"],
                }
            )
            continue

        original_support = _support_from_result(original_exec.columns, original_exec.rows)
        original_direction = _extract_direction(
            claim_type=claim_type,
            columns=original_exec.columns,
            rows=original_exec.rows,
            ordered_fields=ordered_fields,
        )

        rewrites = _generate_rewrites_for_query(
            spec=spec,
            contract=contract,
            db_path=db_path,
            table_name=table_name,
        )

        weighted_num = 0.0
        weighted_den = 0.0
        valid_count = 0  # directional-valid rewrites
        same_direction_count = 0
        structural_valid_count = 0
        reasons: list[str] = []
        valid_template_counts: dict[str, int] = defaultdict(int)  # directional-valid templates
        structural_template_counts: dict[str, int] = defaultdict(int)

        for ridx, rewrite in enumerate(rewrites, start=1):
            rewrite_sql = str(rewrite.get("sql") or "")
            template_type = str(rewrite.get("template_type") or "unknown")
            note = str(rewrite.get("note") or "")

            exec_res = execute_sql(db_path=db_path, sql=rewrite_sql)
            rewrite_record = {
                "query_id": query_id,
                "rewrite_id": f"{query_id}_rw{ridx}",
                "template_type": template_type,
                "note": note,
                "sql": rewrite_sql,
                "execution_ok": exec_res.ok,
                "claim_compatible": True,
                "support_sufficient": False,
                "non_trivial": False,
                "valid_rewrite": False,
                "directional_valid": False,
                "structural_valid": False,
                "support_original": round(original_support, 6),
                "support_rewrite": 0.0,
                "support_weight": 0.0,
                "direction_original": original_direction,
                "direction_rewrite": "indeterminate",
                "direction_agree": False,
                "validity_reason_codes": [],
            }
            if not exec_res.ok:
                rewrite_record["validity_reason_codes"].append("RW_EXECUTION_FAILED")
                rewrite_rows.append(rewrite_record)
                continue

            rewrite_support = _support_from_result(exec_res.columns, exec_res.rows)
            min_support = max(support_min_abs, support_min_ratio * max(1.0, original_support))
            support_ok = rewrite_support >= min_support
            non_trivial = _result_non_trivial(exec_res.columns, exec_res.rows)
            direction_rewrite = _extract_direction(
                claim_type=claim_type,
                columns=exec_res.columns,
                rows=exec_res.rows,
                ordered_fields=ordered_fields,
            )

            rewrite_record["support_rewrite"] = round(rewrite_support, 6)
            rewrite_record["support_sufficient"] = support_ok
            rewrite_record["non_trivial"] = non_trivial
            rewrite_record["direction_rewrite"] = direction_rewrite

            if not support_ok:
                rewrite_record["validity_reason_codes"].append("RW_SUPPORT_INSUFFICIENT")
            if not non_trivial:
                rewrite_record["validity_reason_codes"].append("RW_NON_TRIVIAL_FAILED")

            structural_valid = bool(support_ok and non_trivial and rewrite_record["claim_compatible"])
            rewrite_record["structural_valid"] = structural_valid
            if structural_valid:
                structural_valid_count += 1
                structural_template_counts[template_type] += 1

            if (
                original_direction != "indeterminate"
                and direction_rewrite != "indeterminate"
                and support_ok
                and non_trivial
            ):
                ratio = rewrite_support / max(1e-9, original_support)
                weight = _clip(math.sqrt(max(0.0, ratio)), support_weight_clip_min, support_weight_clip_max)
                agree = direction_rewrite == original_direction
                weighted_den += weight
                weighted_num += weight * (1.0 if agree else 0.0)
                valid_count += 1
                same_direction_count += 1 if agree else 0
                valid_template_counts[template_type] += 1

                rewrite_record["support_weight"] = round(weight, 6)
                rewrite_record["direction_agree"] = bool(agree)
                rewrite_record["valid_rewrite"] = True
                rewrite_record["directional_valid"] = True
            else:
                if original_direction == "indeterminate":
                    rewrite_record["validity_reason_codes"].append("RW_ORIGINAL_DIRECTION_INDETERMINATE")
                if direction_rewrite == "indeterminate":
                    rewrite_record["validity_reason_codes"].append("RW_DIRECTION_INDETERMINATE")

            rewrite_rows.append(rewrite_record)

        query_evaluable = True
        if original_direction == "indeterminate":
            reasons.append("ACR_ORIGINAL_DIRECTION_INDETERMINATE")
            query_evaluable = False
        if valid_count == 0:
            reasons.append("ACR_NO_VALID_REWRITES")
            query_evaluable = False
        elif valid_count < max(1, int(min_evaluable_valid_rewrites)):
            reasons.append("ACR_VALID_REWRITES_TOO_FEW")
            query_evaluable = False

        acr_raw = (weighted_num / weighted_den) if weighted_den > 1e-12 else None
        coverage_factor = _clip(
            valid_count / max(1.0, float(confidence_valid_rewrites)),
            0.0,
            1.0,
        )
        acr_score = None
        if acr_raw is not None:
            acr_score = _clip(acr_raw * coverage_factor, 0.0, 1.0)

        structural_coverage_factor = _clip(
            structural_valid_count / max(1.0, float(confidence_valid_rewrites)),
            0.0,
            1.0,
        )
        structural_template_diversity = _clip(
            len(structural_template_counts) / 3.0,
            0.0,
            1.0,
        )
        structural_score = _clip(structural_coverage_factor * structural_template_diversity, 0.0, 1.0)
        structural_evaluable = bool(len(rewrites) > 0)
        if structural_valid_count == 0:
            reasons.append("ACR_STRUCTURAL_NO_VALID_REWRITES")

        if not reasons:
            reasons.append("ACR_QUERY_OK")

        query_rows.append(
            {
                "query_id": query_id,
                "stable_query_id": stable_query_id,
                "question_id": question_id,
                "stable_question_id": stable_question_id,
                "family_id": family_id,
                "intended_facet_id": intended_facet_id,
                "execution_ok": True,
                "query_evaluable": query_evaluable,  # directional evaluability (legacy compatibility)
                "query_evaluable_directional": query_evaluable,
                "query_evaluable_structural": structural_evaluable,
                "claim_type": claim_type,
                "direction_original": original_direction,
                "generated_rewrite_count": len(rewrites),
                "valid_rewrite_count": valid_count,  # directional-valid rewrites (legacy compatibility)
                "directional_valid_rewrite_count": valid_count,
                "structural_valid_rewrite_count": structural_valid_count,
                "coverage_factor": round(coverage_factor, 6),
                "directional_coverage_factor": round(coverage_factor, 6),
                "structural_coverage_factor": round(structural_coverage_factor, 6),
                "raw_agreement_ratio": round((same_direction_count / valid_count) if valid_count > 0 else 0.0, 6),
                "structural_template_count": len(structural_template_counts),
                "structural_template_diversity": round(structural_template_diversity, 6),
                "acr_directional_score": (round(acr_score, 6) if acr_score is not None else None),
                "acr_structural_score": round(structural_score, 6),
                "acr_score": (round(acr_score, 6) if acr_score is not None else None),
                "query_pass": bool(query_evaluable and acr_score is not None and acr_score >= 0.5),
                "valid_template_counts": dict(valid_template_counts),  # directional templates
                "directional_valid_template_counts": dict(valid_template_counts),
                "structural_valid_template_counts": dict(structural_template_counts),
                "reason_codes": reasons,
                "query_contract_v1": {
                    "frozen_slots": contract.get("frozen_slots", []),
                    "editable_slots": contract.get("editable_slots", []),
                    "allowed_refinement_columns": contract.get("allowed_refinement_columns", []),
                },
            }
        )

    question_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        question_group[str(row.get("question_id") or "unknown")].append(row)

    question_rows: list[dict[str, Any]] = []
    family_query_scores: dict[str, list[float]] = defaultdict(list)  # directional
    family_question_scores: dict[str, list[float]] = defaultdict(list)  # directional
    family_query_scores_struct: dict[str, list[float]] = defaultdict(list)
    family_question_scores_struct: dict[str, list[float]] = defaultdict(list)

    for question_id, rows in sorted(question_group.items(), key=lambda x: x[0]):
        family_id = str(rows[0].get("family_id") or "unknown")
        directional_rows = [
            item
            for item in rows
            if item.get("execution_ok") and item.get("query_evaluable_directional") and item.get("acr_directional_score") is not None
        ]
        structural_rows = [
            item
            for item in rows
            if item.get("execution_ok") and item.get("query_evaluable_structural") and item.get("acr_structural_score") is not None
        ]

        directional_scores = [_to_float(item.get("acr_directional_score")) for item in directional_rows]
        structural_scores = [_to_float(item.get("acr_structural_score")) for item in structural_rows]

        for value in directional_scores:
            family_query_scores[family_id].append(value)
        for value in structural_scores:
            family_query_scores_struct[family_id].append(value)

        trimmed = _trimmed_mean(directional_scores, question_trim_ratio)
        min_score = min(directional_scores) if directional_scores else 0.0
        floor_triggered = False
        adjusted = trimmed
        if directional_scores and min_score < query_floor_threshold:
            adjusted = min(adjusted, question_floor_cap)
            floor_triggered = True

        valid_rewrite_count = sum(int(item.get("valid_rewrite_count") or 0) for item in rows)
        structural_valid_rewrite_count = sum(int(item.get("structural_valid_rewrite_count") or 0) for item in rows)
        generated_rewrite_count = sum(int(item.get("generated_rewrite_count") or 0) for item in rows)

        structural_question_score = _trimmed_mean(structural_scores, question_trim_ratio)

        question_rows.append(
            {
                "question_id": question_id,
                "family_id": family_id,
                "query_count": len(rows),
                "evaluable_query_count": len(directional_rows),  # directional evaluable (legacy compatibility)
                "directional_evaluable_query_count": len(directional_rows),
                "structural_evaluable_query_count": len(structural_rows),
                "avg_acr_score": round(_mean(directional_scores), 6),
                "trimmed_acr_score": round(trimmed, 6),
                "question_acr_score": (round(adjusted, 6) if directional_rows else None),
                "question_acr_directional_score": (round(adjusted, 6) if directional_rows else None),
                "question_acr_structural_score": (round(structural_question_score, 6) if structural_rows else None),
                "min_query_acr_score": round(min_score, 6),
                "floor_guard_triggered": floor_triggered,
                "generated_rewrite_count": generated_rewrite_count,
                "valid_rewrite_count": valid_rewrite_count,  # directional valid rewrites
                "directional_valid_rewrite_count": valid_rewrite_count,
                "structural_valid_rewrite_count": structural_valid_rewrite_count,
                "question_evaluable": bool(directional_rows),
                "question_evaluable_directional": bool(directional_rows),
                "question_evaluable_structural": bool(structural_rows),
            }
        )
        if directional_rows:
            family_question_scores[family_id].append(adjusted)
        if structural_rows:
            family_question_scores_struct[family_id].append(structural_question_score)

    family_rows: list[dict[str, Any]] = []
    all_family_keys = set(list(family_query_scores.keys()) + list(family_question_scores.keys()))
    all_family_keys.update(list(family_query_scores_struct.keys()) + list(family_question_scores_struct.keys()))
    for family_id in sorted(all_family_keys):
        qscores = family_query_scores.get(family_id, [])
        qscores2 = family_question_scores.get(family_id, [])
        qscores_struct = family_query_scores_struct.get(family_id, [])
        qscores2_struct = family_question_scores_struct.get(family_id, [])
        family_rows.append(
            {
                "family_id": family_id,
                "query_count": len(qscores),
                "question_count": len(qscores2),
                "avg_query_acr_score": round(_mean(qscores), 6),
                "trimmed_query_acr_score": round(_trimmed_mean(qscores, question_trim_ratio), 6),
                "avg_question_acr_score": round(_mean(qscores2), 6),
                "structural_query_count": len(qscores_struct),
                "structural_question_count": len(qscores2_struct),
                "avg_query_acr_structural_score": round(_mean(qscores_struct), 6),
                "trimmed_query_acr_structural_score": round(_trimmed_mean(qscores_struct, question_trim_ratio), 6),
                "avg_question_acr_structural_score": round(_mean(qscores2_struct), 6),
            }
        )

    evaluable_question_scores = [  # directional branch
        _to_float(item.get("question_acr_score"))
        for item in question_rows
        if item.get("question_evaluable_directional") and item.get("question_acr_score") is not None
    ]

    structural_question_scores = [
        _to_float(item.get("question_acr_structural_score"))
        for item in question_rows
        if item.get("question_evaluable_structural") and item.get("question_acr_structural_score") is not None
    ]

    workload_acr_directional = _trimmed_mean(evaluable_question_scores, question_trim_ratio)
    workload_acr_structural = _trimmed_mean(structural_question_scores, question_trim_ratio)

    evaluable_query_count = sum(1 for item in query_rows if item.get("query_evaluable_directional"))
    evaluable_query_ratio = evaluable_query_count / max(1, len(query_rows))
    structural_evaluable_query_count = sum(1 for item in query_rows if item.get("query_evaluable_structural"))
    structural_evaluable_query_ratio = structural_evaluable_query_count / max(1, len(query_rows))

    workload_acr_directional_effective = workload_acr_directional * evaluable_query_ratio
    workload_acr_structural_effective = workload_acr_structural * structural_evaluable_query_ratio

    # Primary branch remains directional; structural branch is reported as diagnostic.
    workload_acr = workload_acr_directional
    workload_acr_effective = workload_acr_directional_effective

    report = {
        "contract_version": "acr_report_v0_1",
        "formula": "Directional ACR(q) = support-weighted agreement of valid rewrites preserving original conclusion direction",
        "formula_structural": "Structural ACR(q) = directional-agnostic rewrite robustness from valid rewrite coverage x template diversity",
        "templates": [
            "refinement_rewrite",
            "filter_neighborhood_rewrite",
            "population_neighborhood_rewrite",
        ],
        "config": {
            "support_min_ratio": support_min_ratio,
            "support_min_abs": support_min_abs,
            "support_weight_formula": "w = clip(sqrt(support_rewrite/support_original), 0.25, 1.0)",
            "support_weight_clip_min": support_weight_clip_min,
            "support_weight_clip_max": support_weight_clip_max,
            "min_evaluable_valid_rewrites": min_evaluable_valid_rewrites,
            "confidence_valid_rewrites": confidence_valid_rewrites,
            "score_adjustment": "acr_score = raw_agreement * min(1, valid_rewrite_count/confidence_valid_rewrites)",
            "structural_score_adjustment": "acr_structural_score = min(1, structural_valid_rewrite_count/confidence_valid_rewrites) * (structural_template_count/3)",
            "question_trim_ratio": question_trim_ratio,
            "query_floor_threshold": query_floor_threshold,
            "question_floor_cap": question_floor_cap,
        },
        "summary": {
            "query_count": len(query_rows),
            "evaluable_query_count": evaluable_query_count,
            "evaluable_query_ratio": round(evaluable_query_ratio, 6),
            "directional_evaluable_query_count": evaluable_query_count,
            "directional_evaluable_query_ratio": round(evaluable_query_ratio, 6),
            "structural_evaluable_query_count": structural_evaluable_query_count,
            "structural_evaluable_query_ratio": round(structural_evaluable_query_ratio, 6),
            "question_count": len(question_rows),
            "evaluable_question_count": sum(1 for item in question_rows if item.get("question_evaluable_directional")),
            "directional_evaluable_question_count": sum(
                1 for item in question_rows if item.get("question_evaluable_directional")
            ),
            "structural_evaluable_question_count": sum(
                1 for item in question_rows if item.get("question_evaluable_structural")
            ),
            "family_count": len(family_rows),
            "workload_acr_score": round(workload_acr, 6),
            "workload_acr_effective_score": round(workload_acr_effective, 6),
            "workload_acr_directional_score": round(workload_acr_directional, 6),
            "workload_acr_directional_effective_score": round(workload_acr_directional_effective, 6),
            "workload_acr_structural_score": round(workload_acr_structural, 6),
            "workload_acr_structural_effective_score": round(workload_acr_structural_effective, 6),
            "valid_rewrite_count": sum(int(item.get("valid_rewrite_count") or 0) for item in query_rows),
            "structural_valid_rewrite_count": sum(
                int(item.get("structural_valid_rewrite_count") or 0) for item in query_rows
            ),
            "generated_rewrite_count": sum(int(item.get("generated_rewrite_count") or 0) for item in query_rows),
        },
        "by_question": question_rows,
        "by_family": family_rows,
    }
    return report, query_rows, rewrite_rows, refinement_catalog
