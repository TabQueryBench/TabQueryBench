"""Controlled Intervention Responsiveness (CIR) evaluation."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

from src.benchmark.models import FIVE_FIXED_FAMILIES
from src.benchmark.sql_exec import execute_sql

DEPENDENCY_MEMBERS = {"subgroup_structure", "conditional_dependency_structure"}


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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def _numeric_indices(columns: list[str], rows: list[list[Any]]) -> list[int]:
    if not rows:
        return []
    idxs: list[int] = []
    width = max(len(row) for row in rows) if rows else 0
    for idx in range(min(width, len(columns))):
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


def _row_key(row: list[Any], key_indices: list[int]) -> tuple[str, ...]:
    if not key_indices:
        return tuple()
    out: list[str] = []
    for idx in key_indices:
        if idx >= len(row):
            out.append("<MISSING>")
            continue
        value = row[idx]
        out.append("<NULL>" if value is None else str(value))
    return tuple(out)


def _set_jaccard_distance(rows_a: list[list[Any]], rows_b: list[list[Any]]) -> float:
    set_a = {tuple("<NULL>" if cell is None else str(cell) for cell in row) for row in rows_a}
    set_b = {tuple("<NULL>" if cell is None else str(cell) for cell in row) for row in rows_b}
    union = set_a | set_b
    if not union:
        return 0.0
    inter = set_a & set_b
    return _clip01(1.0 - (len(inter) / len(union)))


def _scalar_distance(rows_a: list[list[Any]], rows_b: list[list[Any]]) -> float:
    def _first_numeric(rows: list[list[Any]]) -> float:
        for row in rows:
            for cell in row:
                try:
                    return float(cell)
                except (TypeError, ValueError):
                    continue
        return 0.0

    a = _first_numeric(rows_a)
    b = _first_numeric(rows_b)
    dist = abs(a - b) / (abs(a) + 1e-9)
    return _clip01(dist)


def _grouped_distance(
    *,
    columns_a: list[str],
    rows_a: list[list[Any]],
    columns_b: list[str],
    rows_b: list[list[Any]],
) -> float:
    cols = list(columns_a) if columns_a else list(columns_b)
    if not cols:
        return _set_jaccard_distance(rows_a, rows_b)

    numeric_idx = _numeric_indices(cols, rows_a + rows_b)
    if not numeric_idx:
        return _set_jaccard_distance(rows_a, rows_b)

    key_idx = [idx for idx in range(len(cols)) if idx not in numeric_idx]

    def _to_map(rows: list[list[Any]]) -> dict[tuple[str, ...], list[float]]:
        out: dict[tuple[str, ...], list[float]] = {}
        for row in rows:
            key = _row_key(row, key_idx)
            vec = out.setdefault(key, [0.0 for _ in numeric_idx])
            for pos, col_idx in enumerate(numeric_idx):
                if col_idx >= len(row):
                    continue
                try:
                    vec[pos] += float(row[col_idx])
                except (TypeError, ValueError):
                    continue
        return out

    map_a = _to_map(rows_a)
    map_b = _to_map(rows_b)
    all_keys = set(map_a.keys()) | set(map_b.keys())
    if not all_keys:
        return 0.0

    total_a = sum(abs(v) for vec in map_a.values() for v in vec)
    total_b = sum(abs(v) for vec in map_b.values() for v in vec)
    total_a = total_a if total_a > 0 else 1.0
    total_b = total_b if total_b > 0 else 1.0

    l1 = 0.0
    for key in all_keys:
        vec_a = map_a.get(key, [0.0 for _ in numeric_idx])
        vec_b = map_b.get(key, [0.0 for _ in numeric_idx])
        for va, vb in zip(vec_a, vec_b):
            pa = abs(va) / total_a
            pb = abs(vb) / total_b
            l1 += abs(pa - pb)
    return _clip01(0.5 * l1)


def _infer_output_mode(
    *,
    spec: dict[str, Any],
    columns: list[str],
    rows: list[list[Any]],
) -> str:
    output_semantics = str(spec.get("output_semantics") or spec.get("expected_result_schema") or "").lower()
    claim_type = str(spec.get("claim_type") or "").lower()
    sql = str(spec.get("sql") or "").lower()

    if "set" in output_semantics or "top" in output_semantics or "rank" in output_semantics or "ranking" in claim_type:
        return "set_topk"
    if len(rows) == 1:
        numeric_idx = _numeric_indices(columns, rows)
        if len(numeric_idx) == 1 and len(columns) == 1:
            return "scalar"
    if " limit " in sql and " order by " in sql and "group by" not in sql:
        return "set_topk"
    return "grouped_table"


def _map_eval_family(family_id: str, merge_dependency_bucket: bool) -> str:
    fid = str(family_id or "").strip()
    if merge_dependency_bucket and fid in DEPENDENCY_MEMBERS:
        return "dependency_structure"
    return fid


def _build_eval_responses(
    *,
    base_responses: dict[str, float],
    merge_dependency_bucket: bool,
    active_base_families: set[str],
) -> dict[str, float]:
    if not merge_dependency_bucket:
        return {fam: float(base_responses.get(fam, 0.0)) for fam in sorted(active_base_families)}

    out: dict[str, float] = {}
    if active_base_families & DEPENDENCY_MEMBERS:
        out["dependency_structure"] = max(
            float(base_responses.get("subgroup_structure", 0.0)),
            float(base_responses.get("conditional_dependency_structure", 0.0)),
        )
    for fam in sorted(active_base_families):
        if fam in DEPENDENCY_MEMBERS:
            continue
        out[fam] = float(base_responses.get(fam, 0.0))
    return out


def _output_distance(
    *,
    spec: dict[str, Any],
    columns_real: list[str],
    rows_real: list[list[Any]],
    columns_var: list[str],
    rows_var: list[list[Any]],
) -> tuple[float, str]:
    mode = _infer_output_mode(spec=spec, columns=columns_real, rows=rows_real)
    if mode == "scalar":
        return _scalar_distance(rows_real, rows_var), mode
    if mode == "set_topk":
        return _set_jaccard_distance(rows_real, rows_var), mode
    return (
        _grouped_distance(
            columns_a=columns_real,
            rows_a=rows_real,
            columns_b=columns_var,
            rows_b=rows_var,
        ),
        mode,
    )


def evaluate_cir(
    *,
    query_specs: list[dict[str, Any]],
    perturbation_manifest: dict[str, Any],
    max_eval_queries: int | None = None,
    cir_lambda: float = 1.0,
    min_target_family_variants: int = 2,
    merge_dependency_bucket: bool = True,
    include_cardinality: bool = False,
    include_missingness: bool | None = None,
    missingness_auto_threshold: float = 1e-9,
    query_floor_threshold: float = 0.15,
    question_floor_cap: float = 0.60,
    question_trim_ratio: float = 0.20,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    variants = [item for item in (perturbation_manifest.get("variants") or []) if isinstance(item, dict)]
    if not variants:
        return {
            "contract_version": "cir_report_v0_1",
            "summary": {
                "query_count": 0,
                "question_count": 0,
                "family_count": 0,
                "workload_cir_score": 0.0,
                "warnings": ["no_perturbation_variants"],
            },
            "config": {
                "cir_lambda": cir_lambda,
            },
            "by_question": [],
            "by_family": [],
        }, [], []

    real_variant = next((v for v in variants if str(v.get("kind") or "") == "real"), None)
    if real_variant is None:
        real_variant = variants[0]

    ordered_specs = [item for item in query_specs if isinstance(item, dict)]
    ordered_specs.sort(key=lambda x: str(x.get("query_id") or ""))
    if max_eval_queries is not None and max_eval_queries > 0:
        ordered_specs = ordered_specs[:max_eval_queries]

    query_rows: list[dict[str, Any]] = []
    execution_trace_rows: list[dict[str, Any]] = []

    baseline_stats = perturbation_manifest.get("baseline_family_statistics") or {}
    if not isinstance(baseline_stats, dict):
        baseline_stats = {}
    baseline_missingness = _to_float(baseline_stats.get("missingness_structure"), default=0.0)
    include_missingness_effective = (
        bool(include_missingness)
        if include_missingness is not None
        else bool(baseline_missingness > float(missingness_auto_threshold))
    )

    active_base_families: set[str] = set(FIVE_FIXED_FAMILIES)
    if not include_cardinality:
        active_base_families.discard("cardinality_structure")
    if not include_missingness_effective:
        active_base_families.discard("missingness_structure")

    accepted_family_variant_count: dict[str, int] = defaultdict(int)
    for variant in variants:
        if str(variant.get("kind") or "") != "family":
            continue
        fam = str(variant.get("family_id") or "")
        if bool((variant.get("validity") or {}).get("accepted", True)):
            accepted_family_variant_count[fam] += 1

    accepted_eval_variant_count: dict[str, int] = {}
    if merge_dependency_bucket:
        if active_base_families & DEPENDENCY_MEMBERS:
            accepted_eval_variant_count["dependency_structure"] = max(
                int(accepted_family_variant_count.get("subgroup_structure", 0)),
                int(accepted_family_variant_count.get("conditional_dependency_structure", 0)),
            )
        for fam in sorted(active_base_families):
            if fam in DEPENDENCY_MEMBERS:
                continue
            accepted_eval_variant_count[fam] = int(accepted_family_variant_count.get(fam, 0))
    else:
        for fam in sorted(active_base_families):
            accepted_eval_variant_count[fam] = int(accepted_family_variant_count.get(fam, 0))

    eval_buckets_active = set(accepted_eval_variant_count.keys())
    evaluable_buckets = {
        fam
        for fam, count in accepted_eval_variant_count.items()
        if count >= max(1, int(min_target_family_variants))
    }

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
                    "query_evaluable": False,
                    "error": real_exec.error,
                    "output_mode": "unknown",
                    "noise_floor": 0.0,
                    "target_response": 0.0,
                    "offtarget_response": 0.0,
                    "cir_score": None,
                    "query_pass": False,
                    "primary_activated_family": "execution_failed",
                    "dominance_margin": -1.0,
                    "by_family_response": {},
                    "reason_codes": ["CIR_QUERY_EXECUTION_FAILED"],
                }
            )
            continue

        family_distances: dict[str, list[float]] = defaultdict(list)
        boot_distances: list[float] = []
        null_distances: list[float] = []
        execution_failures = 0
        output_mode = "grouped_table"

        for variant in variants:
            kind = str(variant.get("kind") or "")
            if kind == "real":
                continue
            variant_id = str(variant.get("variant_id") or "")
            pert_family = str(variant.get("family_id") or "")
            db_path = variant.get("db_path")
            valid = bool(variant.get("validity", {}).get("accepted", True))
            if kind == "family" and not valid:
                execution_trace_rows.append(
                    {
                        "query_id": query_id,
                        "variant_id": variant_id,
                        "kind": kind,
                        "family_id": pert_family,
                        "skipped": True,
                        "skip_reason": "variant_failed_validity",
                    }
                )
                continue

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

            distance, output_mode = _output_distance(
                spec=spec,
                columns_real=real_exec.columns,
                rows_real=real_exec.rows,
                columns_var=exec_result.columns,
                rows_var=exec_result.rows,
            )
            execution_trace_rows.append(
                {
                    "query_id": query_id,
                    "variant_id": variant_id,
                    "kind": kind,
                    "family_id": pert_family,
                    "execution_ok": True,
                    "distance": round(distance, 6),
                    "output_mode": output_mode,
                }
            )
            if kind == "family":
                family_distances[pert_family].append(distance)
            elif kind == "boot":
                boot_distances.append(distance)
            elif kind == "null":
                null_distances.append(distance)

        boot_mean = _mean(boot_distances)
        null_mean = _mean(null_distances)
        if boot_distances and null_distances:
            noise_floor = 0.5 * (boot_mean + null_mean)
        elif boot_distances:
            noise_floor = boot_mean
        elif null_distances:
            noise_floor = null_mean
        else:
            noise_floor = 0.0

        responses: dict[str, float] = {}
        for fam in FIVE_FIXED_FAMILIES:
            deltas = family_distances.get(fam, [])
            if not deltas:
                responses[fam] = 0.0
                continue
            responses[fam] = _mean([max(0.0, delta - noise_floor) for delta in deltas])

        eval_family_id = _map_eval_family(family_id, merge_dependency_bucket=merge_dependency_bucket)
        eval_responses = _build_eval_responses(
            base_responses=responses,
            merge_dependency_bucket=merge_dependency_bucket,
            active_base_families=active_base_families,
        )

        scoring_families = sorted(evaluable_buckets) if evaluable_buckets else sorted(eval_buckets_active)
        family_in_profile = eval_family_id in eval_buckets_active
        query_evaluable = bool(family_in_profile and eval_family_id in evaluable_buckets)

        target_response = eval_responses.get(eval_family_id, 0.0)
        off_values = [eval_responses.get(fam, 0.0) for fam in scoring_families if fam != eval_family_id]
        offtarget_response = _mean(off_values) if off_values else 0.0
        cir_score = (target_response - cir_lambda * offtarget_response) if query_evaluable else None

        response_view = {fam: eval_responses.get(fam, 0.0) for fam in scoring_families}
        primary_family = max(response_view.items(), key=lambda x: x[1])[0] if response_view else "none"
        max_non_target = max([response_view.get(fam, 0.0) for fam in scoring_families if fam != eval_family_id] or [0.0])
        dominance_margin = target_response - max_non_target

        reason_codes: list[str] = []
        if not family_in_profile:
            reason_codes.append("CIR_FAMILY_NOT_IN_EVAL_PROFILE")
        elif not query_evaluable:
            reason_codes.append("CIR_TARGET_FAMILY_NOT_EVALUABLE")
        else:
            if target_response <= 1e-9:
                reason_codes.append("CIR_TARGET_RESPONSE_WEAK")
            if offtarget_response > target_response:
                reason_codes.append("CIR_OFFTARGET_DOMINANT")
            if primary_family != eval_family_id:
                reason_codes.append("CIR_PRIMARY_FAMILY_MISMATCH")
        if not reason_codes:
            reason_codes.append("CIR_QUERY_OK")

        query_rows.append(
            {
                "query_id": query_id,
                "stable_query_id": stable_query_id,
                "question_id": question_id,
                "stable_question_id": stable_question_id,
                "family_id": family_id,
                "evaluation_family_id": eval_family_id,
                "intended_facet_id": intended_facet_id,
                "execution_ok": True,
                "query_evaluable": query_evaluable,
                "output_mode": output_mode,
                "noise_floor": round(noise_floor, 6),
                "target_response": round(target_response, 6),
                "offtarget_response": round(offtarget_response, 6),
                "cir_score": (round(cir_score, 6) if cir_score is not None else None),
                "query_pass": bool(query_evaluable and cir_score is not None and cir_score > 0 and primary_family == eval_family_id),
                "primary_activated_family": primary_family,
                "dominance_margin": round(dominance_margin, 6),
                "by_family_response": {fam: round(val, 6) for fam, val in responses.items()},
                "by_evaluation_family_response": {fam: round(val, 6) for fam, val in eval_responses.items()},
                "evaluable_families": sorted(scoring_families),
                "boot_mean_distance": round(boot_mean, 6),
                "null_mean_distance": round(null_mean, 6),
                "execution_failures": execution_failures,
                "reason_codes": reason_codes,
            }
        )

    question_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        question_group[str(row.get("question_id") or "unknown")].append(row)

    question_rows: list[dict[str, Any]] = []
    family_query_scores: dict[str, list[float]] = defaultdict(list)
    family_question_scores: dict[str, list[float]] = defaultdict(list)

    for question_id, rows in sorted(question_group.items(), key=lambda x: x[0]):
        ok_rows = [row for row in rows if row.get("execution_ok")]
        evaluable_rows = [row for row in ok_rows if row.get("query_evaluable") and row.get("cir_score") is not None]
        scores = [_to_float(row.get("cir_score")) for row in evaluable_rows]
        family_id = str((ok_rows[0].get("evaluation_family_id") if ok_rows else rows[0].get("evaluation_family_id")) or "unknown")
        original_family_ids = sorted(
            {
                str(item.get("family_id") or "").strip()
                for item in ok_rows
                if str(item.get("family_id") or "").strip()
            }
        )

        for score in scores:
            family_query_scores[family_id].append(score)

        trimmed = _trimmed_mean(scores, question_trim_ratio)
        floor_triggered = False
        min_score = min(scores) if scores else 0.0
        adjusted = trimmed
        if scores and min_score < query_floor_threshold:
            adjusted = min(adjusted, question_floor_cap)
            floor_triggered = True

        agreement = _mean(
            [
                1.0
                if str(item.get("primary_activated_family") or "") == family_id
                else 0.0
                for item in evaluable_rows
            ]
        )
        pass_rate = _mean([1.0 if bool(item.get("query_pass")) else 0.0 for item in evaluable_rows])

        row = {
            "question_id": question_id,
            "family_id": family_id,
            "original_family_ids": original_family_ids,
            "query_count": len(rows),
            "ok_query_count": len(ok_rows),
            "evaluable_query_count": len(evaluable_rows),
            "avg_cir_score": round(_mean(scores), 6),
            "trimmed_cir_score": round(trimmed, 6),
            "question_cir_score": (round(adjusted, 6) if evaluable_rows else None),
            "query_pass_rate": round(pass_rate, 6),
            "question_agreement_rate": round(agreement, 6),
            "min_query_cir_score": round(min_score, 6),
            "floor_guard_triggered": floor_triggered,
            "question_evaluable": bool(evaluable_rows),
        }
        question_rows.append(row)
        if evaluable_rows:
            family_question_scores[family_id].append(adjusted)

    family_rows: list[dict[str, Any]] = []
    for family_id in sorted(set(list(family_query_scores.keys()) + list(family_question_scores.keys()))):
        q_scores = family_query_scores.get(family_id, [])
        qq_scores = family_question_scores.get(family_id, [])
        family_rows.append(
            {
                "family_id": family_id,
                "query_count": len(q_scores),
                "question_count": len(qq_scores),
                "avg_query_cir_score": round(_mean(q_scores), 6),
                "trimmed_query_cir_score": round(_trimmed_mean(q_scores, question_trim_ratio), 6),
                "avg_question_cir_score": round(_mean(qq_scores), 6),
                "family_evaluable": bool(family_id in evaluable_buckets),
                "accepted_variant_count": int(accepted_eval_variant_count.get(family_id, 0)),
            }
        )

    evaluable_question_scores = [
        _to_float(row.get("question_cir_score"))
        for row in question_rows
        if row.get("question_evaluable") and row.get("question_cir_score") is not None
    ]
    workload_cir = _trimmed_mean(evaluable_question_scores, question_trim_ratio)
    evaluable_query_count = sum(1 for row in query_rows if row.get("query_evaluable"))
    evaluable_query_ratio = evaluable_query_count / max(1, len(query_rows))
    workload_cir_effective = workload_cir * evaluable_query_ratio

    report = {
        "contract_version": "cir_report_v0_1",
        "formula": "CIR(q) = R_target(q) - lambda * R_offtarget(q)",
        "details": {
            "noise_floor": "0.5 * (mean_dist_boot + mean_dist_null), fallback to available control",
            "family_response": "mean(max(0, dist_family_variant - noise_floor))",
            "output_distance_modes": {
                "scalar": "|a-b|/(|a|+eps), clipped [0,1]",
                "grouped_table": "normalized L1/TV over aligned grouped numeric vectors",
                "set_topk": "1 - JaccardOverlap",
            },
        },
        "config": {
            "cir_lambda": cir_lambda,
            "min_target_family_variants": min_target_family_variants,
            "merge_dependency_bucket": bool(merge_dependency_bucket),
            "include_cardinality": bool(include_cardinality),
            "include_missingness": bool(include_missingness_effective),
            "missingness_auto_threshold": float(missingness_auto_threshold),
            "baseline_missingness_rate": round(float(baseline_missingness), 6),
            "question_trim_ratio": question_trim_ratio,
            "query_floor_threshold": query_floor_threshold,
            "question_floor_cap": question_floor_cap,
        },
        "summary": {
            "query_count": len(query_rows),
            "evaluable_query_count": evaluable_query_count,
            "evaluable_query_ratio": round(evaluable_query_ratio, 6),
            "question_count": len(question_rows),
            "evaluable_question_count": sum(1 for row in question_rows if row.get("question_evaluable")),
            "family_count": len(family_rows),
            "workload_cir_score": round(workload_cir, 6),
            "workload_cir_effective_score": round(workload_cir_effective, 6),
            "evaluable_families": sorted(evaluable_buckets),
            "evaluation_buckets_active": sorted(eval_buckets_active),
        },
        "by_question": question_rows,
        "by_family": family_rows,
    }
    return report, query_rows, execution_trace_rows
