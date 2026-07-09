"""Purity evaluation built on alignment perturbation responses."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_purity(
    *,
    alignment_by_query: list[dict[str, Any]],
    high_contamination_threshold: float = 0.8,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, float]]]:
    query_rows: list[dict[str, Any]] = []

    matrix_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in alignment_by_query:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("execution_ok", True)):
            continue

        family_id = str(row.get("family_id") or "unknown")
        by_family = row.get("by_family_response_summary") or {}
        if not isinstance(by_family, dict):
            by_family = {}

        target_response = _to_float(row.get("target_response_mean"))
        non_target_candidates: dict[str, float] = {}
        for fam, summary in by_family.items():
            if fam == family_id:
                continue
            if not isinstance(summary, dict):
                continue
            non_target_candidates[str(fam)] = _to_float(summary.get("mean"))

        max_non_target_family = "none"
        max_non_target_response = 0.0
        if non_target_candidates:
            max_non_target_family, max_non_target_response = max(non_target_candidates.items(), key=lambda x: x[1])

        non_target_mean = _to_float(row.get("non_target_response_mean"))

        # v0.1 formulas:
        # contamination_ratio = max_non_target_response / (target_response + eps)
        # overall_non_target_ratio = mean_non_target_response / (target_response + eps)
        # purity_score = 1 / (1 + overall_non_target_ratio)
        contamination_ratio = max_non_target_response / (target_response + 1e-9)
        overall_non_target_ratio = non_target_mean / (target_response + 1e-9)
        purity_score = 1.0 / (1.0 + max(0.0, overall_non_target_ratio))

        high_contamination = contamination_ratio >= high_contamination_threshold

        query_rows.append(
            {
                "query_id": str(row.get("query_id") or ""),
                "stable_query_id": str(row.get("stable_query_id") or ""),
                "question_id": str(row.get("question_id") or ""),
                "stable_question_id": str(row.get("stable_question_id") or ""),
                "family_id": family_id,
                "target_response": round(target_response, 6),
                "max_non_target_response": round(max_non_target_response, 6),
                "mean_non_target_response": round(non_target_mean, 6),
                "max_non_target_family": max_non_target_family,
                "contamination_ratio": round(contamination_ratio, 6),
                "overall_non_target_ratio": round(overall_non_target_ratio, 6),
                "purity_score": round(purity_score, 6),
                "high_contamination": high_contamination,
                "purity_evidence_codes": [],
            }
        )

        if max_non_target_family != "none":
            matrix_values[family_id][max_non_target_family].append(contamination_ratio)

    question_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        question_group[row["question_id"]].append(row)
        family_group[row["family_id"]].append(row)

    question_rows: list[dict[str, Any]] = []
    for question_id, rows in sorted(question_group.items(), key=lambda x: x[0]):
        contamination_values = [_to_float(item.get("contamination_ratio")) for item in rows]
        purity_values = [_to_float(item.get("purity_score")) for item in rows]
        high_ratio = _mean([1.0 if item.get("high_contamination") else 0.0 for item in rows])
        question_rows.append(
            {
                "question_id": question_id,
                "family_id": rows[0].get("family_id") if rows else "unknown",
                "query_count": len(rows),
                "avg_contamination_ratio": round(_mean(contamination_values), 6),
                "avg_purity_score": round(_mean(purity_values), 6),
                "high_contamination_query_ratio": round(high_ratio, 6),
            }
        )

    family_rows: list[dict[str, Any]] = []
    for family_id, rows in sorted(family_group.items(), key=lambda x: x[0]):
        contamination_values = [_to_float(item.get("contamination_ratio")) for item in rows]
        purity_values = [_to_float(item.get("purity_score")) for item in rows]
        high_ratio = _mean([1.0 if item.get("high_contamination") else 0.0 for item in rows])
        family_rows.append(
            {
                "family_id": family_id,
                "query_count": len(rows),
                "avg_contamination_ratio": round(_mean(contamination_values), 6),
                "avg_purity_score": round(_mean(purity_values), 6),
                "high_contamination_query_ratio": round(high_ratio, 6),
            }
        )

    contamination_matrix: dict[str, dict[str, float]] = {}
    for src_family, dst_map in matrix_values.items():
        contamination_matrix[src_family] = {}
        for dst_family, values in dst_map.items():
            contamination_matrix[src_family][dst_family] = round(_mean(values), 6)

    workload_purity = _mean([_to_float(row.get("purity_score")) for row in query_rows])

    report = {
        "contract_version": "purity_report_v0_1",
        "formulas": {
            "contamination_ratio": "max_non_target_response / (target_response + 1e-9)",
            "overall_non_target_ratio": "mean_non_target_response / (target_response + 1e-9)",
            "purity_score": "1 / (1 + overall_non_target_ratio)",
        },
        "config": {
            "high_contamination_threshold": high_contamination_threshold,
        },
        "summary": {
            "query_count": len(query_rows),
            "question_count": len(question_rows),
            "family_count": len(family_rows),
            "workload_purity_score": round(workload_purity, 6),
            "high_contamination_query_count": sum(1 for row in query_rows if row.get("high_contamination")),
        },
        "by_question": question_rows,
        "by_family": family_rows,
    }

    return report, query_rows, contamination_matrix
