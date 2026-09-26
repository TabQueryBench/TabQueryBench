"""Alignment evaluation using perturbation substrate responses."""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from typing import Any

from tqb_query.benchmark.sql_exec import execute_sql


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _query_signal(columns: list[str], rows: list[list[Any]]) -> float:
    numeric: list[float] = []
    for row in rows:
        for cell in row:
            try:
                numeric.append(float(cell))
            except (TypeError, ValueError):
                continue
    if numeric:
        return sum(abs(v) for v in numeric) / len(numeric)
    return float(len(rows))


def _delta(signal_real: float, signal_variant: float) -> float:
    return abs(signal_variant - signal_real) / (abs(signal_real) + 1e-9)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _summ(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": _mean(values),
        "std": _std(values),
        "min": min(values),
        "max": max(values),
    }


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
        # Last-resort grouping for legacy queryspecs lacking question linkage.
        return f"qgrp_{query_id.split('_v')[0]}"

    return "unknown"


def evaluate_alignment(
    *,
    query_specs: list[dict[str, Any]],
    perturbation_manifest: dict[str, Any],
    query_execution_summaries: list[dict[str, Any]] | None = None,
    max_eval_queries: int | None = None,
    alignment_pass_threshold: float = 0.45,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    variants = [item for item in (perturbation_manifest.get("variants") or []) if isinstance(item, dict)]
    if not variants:
        return {
            "contract_version": "alignment_report_v0_1",
            "summary": {
                "query_count": 0,
                "question_count": 0,
                "family_count": 0,
                "workload_alignment_score": 0.0,
                "warnings": ["no_perturbation_variants"],
            },
            "config": {
                "alignment_pass_threshold": alignment_pass_threshold,
            },
            "by_question": [],
            "by_family": [],
        }, [], []

    ordered_specs = [item for item in query_specs if isinstance(item, dict)]
    ordered_specs.sort(key=lambda x: str(x.get("query_id") or ""))
    if max_eval_queries is not None and max_eval_queries > 0:
        ordered_specs = ordered_specs[:max_eval_queries]

    query_rows: list[dict[str, Any]] = []
    execution_trace_rows: list[dict[str, Any]] = []

    real_variant = None
    for item in variants:
        if str(item.get("kind")) == "real":
            real_variant = item
            break
    if real_variant is None:
        real_variant = variants[0]

    baseline_summary_by_query: dict[str, dict[str, Any]] = {}
    for row in query_execution_summaries or []:
        if not isinstance(row, dict):
            continue
        query_id = str(row.get("query_id") or "")
        if query_id:
            baseline_summary_by_query[query_id] = row

    for spec in ordered_specs:
        query_id = str(spec.get("query_id") or "")
        stable_query_id = str(spec.get("stable_query_id") or "")
        question_id = _fallback_question_id(spec, query_id=query_id)
        stable_question_id = str(spec.get("stable_question_id") or question_id)
        family_id = str(spec.get("family_id") or spec.get("family") or "unknown")
        intended_facet_id = str(spec.get("intended_facet_id") or "unknown")
        sql = str(spec.get("sql") or "")
        if not sql.strip():
            continue

        real_exec = execute_sql(db_path=real_variant["db_path"], sql=sql)
        if not real_exec.ok:
            query_rows.append(
                {
                    "query_id": query_id,
                    "stable_query_id": stable_query_id,
                    "question_id": question_id,
                    "stable_question_id": stable_question_id,
                    "family_id": family_id,
                    "intended_facet_id": intended_facet_id,
                    "execution_ok": False,
                    "error": real_exec.error,
                    "alignment_score": 0.0,
                    "dominance_margin": -1.0,
                    "primary_activated_family": "execution_failed",
                    "query_alignment_pass": False,
                    "by_family_response_summary": {},
                    "target_response_summary": {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                    "non_target_response_summary": {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                    "null_response_summary": {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                    "boot_response_summary": {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                    "execution_failures": 1,
                    "total_variant_evaluations": len(variants),
                }
            )
            continue

        signal_real = _query_signal(real_exec.columns, real_exec.rows)

        family_deltas: dict[str, list[float]] = defaultdict(list)
        null_deltas: list[float] = []
        boot_deltas: list[float] = []
        execution_failures = 0

        for variant in variants:
            variant_id = str(variant.get("variant_id") or "")
            kind = str(variant.get("kind") or "")
            pert_family = str(variant.get("family_id") or "")
            db_path = variant.get("db_path")
            exec_result = execute_sql(db_path=db_path, sql=sql)
            if not exec_result.ok:
                execution_failures += 1
                execution_trace_rows.append(
                    {
                        "query_id": query_id,
                        "variant_id": variant_id,
                        "kind": kind,
                        "family_id": pert_family,
                        "execution_ok": False,
                        "error": exec_result.error,
                    }
                )
                continue

            signal_variant = _query_signal(exec_result.columns, exec_result.rows)
            response_delta = _delta(signal_real, signal_variant)

            execution_trace_rows.append(
                {
                    "query_id": query_id,
                    "variant_id": variant_id,
                    "kind": kind,
                    "family_id": pert_family,
                    "execution_ok": True,
                    "signal_real": signal_real,
                    "signal_variant": signal_variant,
                    "response_delta": response_delta,
                }
            )

            if kind == "family":
                family_deltas[pert_family].append(response_delta)
            elif kind == "null":
                null_deltas.append(response_delta)
            elif kind == "boot":
                boot_deltas.append(response_delta)

        target_values = family_deltas.get(family_id, [])
        target_summary = _summ(target_values)

        non_target_map = {fam: values for fam, values in family_deltas.items() if fam != family_id}
        non_target_flat = [value for values in non_target_map.values() for value in values]
        non_target_summary = _summ(non_target_flat)
        null_summary = _summ(null_deltas)
        boot_summary = _summ(boot_deltas)

        by_family_summary = {fam: _summ(values) for fam, values in family_deltas.items()}

        target_mean = _to_float(target_summary["mean"])
        non_target_mean = _to_float(non_target_summary["mean"])
        null_mean = _to_float(null_summary["mean"])

        # v0.1 explicit alignment formula:
        # alignment_score = target_mean / (target_mean + non_target_mean + null_mean + eps)
        alignment_score = target_mean / (target_mean + non_target_mean + null_mean + 1e-9)

        family_mean_map = {fam: _to_float(summary["mean"]) for fam, summary in by_family_summary.items()}
        if family_mean_map:
            primary_family = max(family_mean_map.items(), key=lambda x: x[1])[0]
            max_non_target = max((value for fam, value in family_mean_map.items() if fam != family_id), default=0.0)
        else:
            primary_family = "none"
            max_non_target = 0.0
        dominance_margin = target_mean - max_non_target

        query_alignment_pass = (primary_family == family_id) and (alignment_score >= alignment_pass_threshold)

        query_rows.append(
            {
                "query_id": query_id,
                "stable_query_id": stable_query_id,
                "question_id": question_id,
                "stable_question_id": stable_question_id,
                "family_id": family_id,
                "intended_facet_id": intended_facet_id,
                "execution_ok": True,
                "alignment_score": round(alignment_score, 6),
                "dominance_margin": round(dominance_margin, 6),
                "primary_activated_family": primary_family,
                "query_alignment_pass": query_alignment_pass,
                "target_response_mean": round(target_mean, 6),
                "non_target_response_mean": round(non_target_mean, 6),
                "null_response_mean": round(null_mean, 6),
                "boot_response_mean": round(_to_float(boot_summary["mean"]), 6),
                "max_non_target_response": round(max_non_target, 6),
                "max_non_target_family": (
                    max(non_target_map.keys(), key=lambda fam: _to_float(_summ(non_target_map[fam])["mean"]))
                    if non_target_map
                    else "none"
                ),
                "by_family_response_summary": by_family_summary,
                "target_response_summary": target_summary,
                "non_target_response_summary": non_target_summary,
                "null_response_summary": null_summary,
                "boot_response_summary": boot_summary,
                "execution_failures": execution_failures,
                "total_variant_evaluations": len(variants),
                "alignment_evidence_codes": [],
                "baseline_execution_summary_v2": baseline_summary_by_query.get(query_id, {}),
            }
        )

    # Aggregate at question level.
    question_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        question_group[str(row.get("question_id") or "unknown")].append(row)

    question_rows: list[dict[str, Any]] = []
    family_scores: dict[str, list[float]] = defaultdict(list)
    family_agreements: dict[str, list[float]] = defaultdict(list)

    for question_id, rows in sorted(question_group.items(), key=lambda x: x[0]):
        if not rows:
            continue
        family = str(rows[0].get("family_id") or "unknown")
        scores = [_to_float(row.get("alignment_score")) for row in rows if row.get("execution_ok")]
        primary_match = [1.0 if row.get("primary_activated_family") == family else 0.0 for row in rows if row.get("execution_ok")]
        pass_rate = [1.0 if row.get("query_alignment_pass") else 0.0 for row in rows if row.get("execution_ok")]

        question_rows.append(
            {
                "question_id": question_id,
                "family_id": family,
                "query_count": len(rows),
                "avg_alignment_score": round(_mean(scores), 6),
                "question_agreement_rate": round(_mean(primary_match), 6),
                "alignment_pass_rate": round(_mean(pass_rate), 6),
            }
        )

        family_scores[family].extend(scores)
        family_agreements[family].extend(primary_match)

    family_rows: list[dict[str, Any]] = []
    for family, scores in sorted(family_scores.items(), key=lambda x: x[0]):
        agreements = family_agreements.get(family, [])
        family_rows.append(
            {
                "family_id": family,
                "query_count": len(scores),
                "avg_alignment_score": round(_mean(scores), 6),
                "family_agreement_rate": round(_mean(agreements), 6),
            }
        )

    workload_alignment = _mean([_to_float(row.get("alignment_score")) for row in query_rows if row.get("execution_ok")])

    report = {
        "contract_version": "alignment_report_v0_1",
        "formulas": {
            "query_alignment_score": {
                "definition": "target_mean / (target_mean + non_target_mean + null_mean + 1e-9)",
                "range": "[0,1]",
            },
            "query_dominance_margin": {
                "definition": "target_mean - max_non_target_mean",
            },
            "question_agreement_rate": {
                "definition": "fraction of child queries where primary_activated_family == intended family",
            },
        },
        "config": {
            "alignment_pass_threshold": alignment_pass_threshold,
            "variant_count": len(variants),
        },
        "summary": {
            "query_count": len(query_rows),
            "question_count": len(question_rows),
            "family_count": len(family_rows),
            "workload_alignment_score": round(workload_alignment, 6),
            "execution_failure_count": sum(int(row.get("execution_failures") or 0) for row in query_rows),
        },
        "by_question": question_rows,
        "by_family": family_rows,
    }

    return report, query_rows, execution_trace_rows
