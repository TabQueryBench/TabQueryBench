"""Structural adherence checks for template-grounded generated SQL."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any


STRUCTURE_FLAG_ORDER = [
    "filtered",
    "count",
    "distinct",
    "avg",
    "sum",
    "percentile",
    "window",
    "case_when",
    "topk",
    "two_dimensional",
    "support_guard",
]
CRITICAL_FLAGS = {
    "count",
    "distinct",
    "avg",
    "sum",
    "percentile",
    "window",
    "case_when",
    "two_dimensional",
    "support_guard",
}
TEMPLATE_ID_COMMENT_RE = re.compile(r"^\s*--\s*template_id:\s*([A-Za-z0-9_\-]+)\s*$", re.MULTILINE)


def normalize_sql(sql: str) -> str:
    return " ".join((sql or "").lower().split())


def groupby_arity(sql: str) -> int:
    normalized = normalize_sql(sql)
    match = re.search(r"group by\s+(.+?)(having|order by|limit|$)", normalized)
    if not match:
        return 0
    return len([part.strip() for part in match.group(1).split(",") if part.strip()])


def structure_flags(sql: str) -> dict[str, bool]:
    normalized = normalize_sql(sql)
    return {
        "filtered": " where " in f" {normalized} " or " having " in f" {normalized} ",
        "count": "count(" in normalized,
        "distinct": "count(distinct" in normalized,
        "avg": "avg(" in normalized,
        "sum": "sum(" in normalized,
        "percentile": any(
            token in normalized
            for token in [
                "percentile_cont(",
                "approx_percentile(",
                "approx_quantile",
                "quantile(",
                "quantiles",
                "ds_rank(",
            ]
        ),
        "window": "over (" in normalized or "row_number()" in normalized or "rank()" in normalized,
        "case_when": "case when" in normalized,
        "topk": " limit " in f" {normalized} ",
        "two_dimensional": groupby_arity(normalized) >= 2,
        "support_guard": "having count(*) >" in normalized,
    }


def extract_template_ids_from_sql(sql_queries: list[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for sql in sql_queries:
        for match in TEMPLATE_ID_COMMENT_RE.findall(sql):
            if match not in seen:
                seen.add(match)
                found.append(match)
    return found


def _adherence_label(
    *,
    adherence_score: float,
    comment_in_shortlist: bool,
    missing_expected_flags: list[str],
    groupby_arity_match: bool,
) -> str:
    if not comment_in_shortlist:
        return "low"
    critical_missing = [flag for flag in missing_expected_flags if flag in CRITICAL_FLAGS]
    if adherence_score >= 0.85 and not critical_missing and groupby_arity_match:
        return "high"
    if adherence_score >= 0.6:
        return "medium"
    return "low"


def analyze_sql_queries(
    *,
    sql_queries: list[str],
    template_lookup: dict[str, dict[str, Any]],
    shortlist_ids: list[str] | None = None,
) -> dict[str, Any]:
    shortlist_set = set(shortlist_ids or [])
    analyses: list[dict[str, Any]] = []
    adherence_scores: list[float] = []
    label_counts = {"high": 0, "medium": 0, "low": 0}
    shortlist_violation_count = 0

    for sql_index, sql in enumerate(sql_queries):
        comment_template_ids = TEMPLATE_ID_COMMENT_RE.findall(sql or "")
        claimed_template_id = comment_template_ids[0] if comment_template_ids else None
        template = template_lookup.get(claimed_template_id) if claimed_template_id else None

        analysis: dict[str, Any] = {
            "sql_index": sql_index,
            "claimed_template_ids": comment_template_ids,
            "claimed_template_id": claimed_template_id,
            "template_found": bool(template),
            "comment_in_shortlist": claimed_template_id in shortlist_set if claimed_template_id else False,
            "expected_flags": [],
            "observed_flags": [],
            "matched_flags": [],
            "missing_expected_flags": [],
            "unexpected_flags": [],
            "expected_groupby_arity": None,
            "observed_groupby_arity": groupby_arity(sql),
            "groupby_arity_match": None,
            "adherence_score": 0.0,
            "adherence_label": "low",
            "notes": [],
        }

        if claimed_template_id and claimed_template_id not in shortlist_set and shortlist_set:
            shortlist_violation_count += 1
            analysis["notes"].append("claimed_template_not_in_shortlist")

        if not claimed_template_id:
            analysis["notes"].append("missing_template_comment")
            analyses.append(analysis)
            label_counts["low"] += 1
            continue

        if template is None:
            analysis["notes"].append("claimed_template_not_found_in_library")
            analyses.append(analysis)
            label_counts["low"] += 1
            continue

        expected_flag_map = structure_flags(template["sql_skeleton"])
        observed_flag_map = structure_flags(sql)
        expected_flags = [flag for flag in STRUCTURE_FLAG_ORDER if expected_flag_map.get(flag)]
        observed_flags = [flag for flag in STRUCTURE_FLAG_ORDER if observed_flag_map.get(flag)]
        expected_set = set(expected_flags)
        observed_set = set(observed_flags)
        matched_flags = [flag for flag in STRUCTURE_FLAG_ORDER if flag in expected_set & observed_set]
        missing_expected_flags = [flag for flag in STRUCTURE_FLAG_ORDER if flag in expected_set - observed_set]
        unexpected_flags = [flag for flag in STRUCTURE_FLAG_ORDER if flag in observed_set - expected_set]
        expected_arity = groupby_arity(template["sql_skeleton"])
        observed_arity = groupby_arity(sql)
        flag_recall = 1.0 if not expected_flags else len(matched_flags) / len(expected_flags)
        arity_component = 1.0 if expected_arity == observed_arity else 0.0
        adherence_score = round((0.8 * flag_recall) + (0.2 * arity_component), 4)
        adherence_label = _adherence_label(
            adherence_score=adherence_score,
            comment_in_shortlist=analysis["comment_in_shortlist"] or not shortlist_set,
            missing_expected_flags=missing_expected_flags,
            groupby_arity_match=(expected_arity == observed_arity),
        )

        analysis.update(
            {
                "expected_flags": expected_flags,
                "observed_flags": observed_flags,
                "matched_flags": matched_flags,
                "missing_expected_flags": missing_expected_flags,
                "unexpected_flags": unexpected_flags,
                "expected_groupby_arity": expected_arity,
                "observed_groupby_arity": observed_arity,
                "groupby_arity_match": expected_arity == observed_arity,
                "adherence_score": adherence_score,
                "adherence_label": adherence_label,
            }
        )
        if missing_expected_flags:
            analysis["notes"].append("missing_expected_structure")
        if unexpected_flags:
            analysis["notes"].append("additional_structure_present")

        analyses.append(analysis)
        adherence_scores.append(adherence_score)
        label_counts[adherence_label] += 1

    return {
        "total_sql_queries": len(sql_queries),
        "commented_query_count": sum(1 for row in analyses if row["claimed_template_id"]),
        "analyzed_query_count": sum(1 for row in analyses if row["template_found"]),
        "shortlist_violation_count": shortlist_violation_count,
        "overall_adherence_score": round(mean(adherence_scores), 4) if adherence_scores else 0.0,
        "label_counts": label_counts,
        "query_analyses": analyses,
    }
