"""Rank stability evaluation across multiple benchmark builds."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqb_query.benchmark.models import FIVE_FIXED_FAMILIES

DEPENDENCY_MEMBERS = {"subgroup_structure", "conditional_dependency_structure"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_score_table(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("models"), list):
            return [item for item in payload.get("models", []) if isinstance(item, dict)]
        if isinstance(payload.get("rows"), list):
            return [item for item in payload.get("rows", []) if isinstance(item, dict)]
    return []


def _extract_domain_scores(row: dict[str, Any]) -> tuple[str, dict[str, float]]:
    model_id = str(row.get("model_id") or row.get("model") or row.get("id") or "").strip()
    if not model_id:
        return "", {}

    scores: dict[str, float] = {}

    for key in ["overall_score", "overall", "score"]:
        if key in row:
            scores["overall"] = _to_float(row.get(key), default=0.0)
            break

    for family in FIVE_FIXED_FAMILIES:
        candidates = [family, f"{family}_score", f"family__{family}", f"score__{family}"]
        for key in candidates:
            if key in row:
                scores[family] = _to_float(row.get(key), default=0.0)
                break

    if (scores.get("subgroup_structure") is not None) or (scores.get("conditional_dependency_structure") is not None):
        scores["dependency_structure"] = max(
            _to_float(scores.get("subgroup_structure"), default=0.0),
            _to_float(scores.get("conditional_dependency_structure"), default=0.0),
        )

    return model_id, scores


def _discover_query_score_path(score_table_path: Path) -> Path | None:
    candidates = [
        score_table_path.parent / "query_scores.jsonl",
        score_table_path.with_name("query_scores.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_query_score_table(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _normalize_eval_family(family_id: str) -> str:
    fid = str(family_id or "").strip()
    if fid in DEPENDENCY_MEMBERS:
        return "dependency_structure"
    return fid


def _extract_query_scores_by_domain(query_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    # domain -> query_id -> model_id -> score
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in query_rows:
        model_id = str(row.get("model_id") or "").strip()
        query_id = str(
            row.get("query_identity_stable_key")
            or row.get("stable_query_id")
            or row.get("query_id")
            or ""
        ).strip()
        if not model_id or not query_id:
            continue
        if row.get("synthetic_exec_ok") is False:
            continue
        score = _to_float(row.get("query_score"), default=0.0)
        family = str(row.get("family_id") or "").strip()
        eval_family = _normalize_eval_family(family)

        out["overall"][query_id][model_id] = score
        if family:
            out[family][query_id][model_id] = score
        if eval_family and eval_family != family:
            out[eval_family][query_id][model_id] = score
    return out


def _rank_models(model_scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(model_scores.items(), key=lambda x: (-x[1], x[0]))


def _rank_map(model_scores: dict[str, float]) -> dict[str, int]:
    ordered = _rank_models(model_scores)
    return {model_id: idx + 1 for idx, (model_id, _) in enumerate(ordered)}


def _kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    pos_a = {model: idx for idx, model in enumerate(order_a)}
    pos_b = {model: idx for idx, model in enumerate(order_b)}
    common = [model for model in order_a if model in pos_b]
    n = len(common)
    if n <= 1:
        return 0.0

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_i = common[i]
            a_j = common[j]
            sign_a = 1 if pos_a[a_i] < pos_a[a_j] else -1
            sign_b = 1 if pos_b[a_i] < pos_b[a_j] else -1
            if sign_a == sign_b:
                concordant += 1
            else:
                discordant += 1

    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _spearman_rho(rank_a: dict[str, int], rank_b: dict[str, int]) -> float:
    common = [model for model in rank_a if model in rank_b]
    n = len(common)
    if n <= 1:
        return 0.0
    vals_a = [rank_a[model] for model in common]
    vals_b = [rank_b[model] for model in common]

    mean_a = sum(vals_a) / n
    mean_b = sum(vals_b) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(vals_a, vals_b))
    var_a = sum((a - mean_a) ** 2 for a in vals_a)
    var_b = sum((b - mean_b) ** 2 for b in vals_b)
    if var_a <= 0 or var_b <= 0:
        return 0.0
    return cov / math.sqrt(var_a * var_b)


def _topk_overlap(order_a: list[str], order_b: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_a = set(order_a[:k])
    top_b = set(order_b[:k])
    denom = min(k, len(order_a), len(order_b))
    if denom <= 0:
        return 0.0
    return len(top_a & top_b) / denom


def _pairwise_reversal_ratio(order_a: list[str], order_b: list[str]) -> tuple[float, dict[tuple[str, str], bool]]:
    pos_a = {model: idx for idx, model in enumerate(order_a)}
    pos_b = {model: idx for idx, model in enumerate(order_b)}
    common = [model for model in order_a if model in pos_b]
    n = len(common)
    if n <= 1:
        return 0.0, {}

    total = 0
    reversals = 0
    flags: dict[tuple[str, str], bool] = {}
    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = common[i], common[j]
            sign_a = pos_a[m1] < pos_a[m2]
            sign_b = pos_b[m1] < pos_b[m2]
            total += 1
            is_reversed = sign_a != sign_b
            if is_reversed:
                reversals += 1
            pair = tuple(sorted((m1, m2)))
            flags[pair] = is_reversed

    return (reversals / total if total else 0.0), flags


def _extract_build_meta(build_meta: dict[str, Any]) -> dict[str, Any]:
    dataset_fingerprint = None
    if isinstance(build_meta, dict):
        fp = build_meta.get("dataset_fingerprint")
        if isinstance(fp, dict):
            dataset_fingerprint = fp.get("fingerprint_sha256")
    return {
        "run_id": build_meta.get("run_id") if isinstance(build_meta, dict) else None,
        "build_id": build_meta.get("build_id") if isinstance(build_meta, dict) else None,
        "dataset_id": build_meta.get("dataset_id") if isinstance(build_meta, dict) else None,
        "dataset_fingerprint": dataset_fingerprint,
        "pipeline_version": build_meta.get("pipeline_version") if isinstance(build_meta, dict) else None,
    }


def _metrics_from_pairwise(pairwise_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not pairwise_rows:
        return {
            "avg_kendall_tau": 0.0,
            "avg_spearman_rho": 0.0,
            "champion_retention_rate": 0.0,
            "avg_top_k_overlap": 0.0,
            "avg_pairwise_reversal_ratio": 1.0,
        }
    return {
        "avg_kendall_tau": sum(float(row.get("kendall_tau") or 0.0) for row in pairwise_rows) / len(pairwise_rows),
        "avg_spearman_rho": sum(float(row.get("spearman_rho") or 0.0) for row in pairwise_rows) / len(pairwise_rows),
        "champion_retention_rate": sum(1.0 if bool(row.get("champion_same")) else 0.0 for row in pairwise_rows)
        / len(pairwise_rows),
        "avg_top_k_overlap": sum(float(row.get("top_k_overlap") or 0.0) for row in pairwise_rows) / len(pairwise_rows),
        "avg_pairwise_reversal_ratio": sum(float(row.get("pairwise_reversal_ratio") or 0.0) for row in pairwise_rows)
        / len(pairwise_rows),
    }


def _stability_score(summary: dict[str, float]) -> float:
    tau = _to_float(summary.get("avg_kendall_tau"), default=0.0)
    rho = _to_float(summary.get("avg_spearman_rho"), default=0.0)
    champion = _to_float(summary.get("champion_retention_rate"), default=0.0)
    topk = _to_float(summary.get("avg_top_k_overlap"), default=0.0)
    reversal = _to_float(summary.get("avg_pairwise_reversal_ratio"), default=1.0)
    return (tau + rho + champion + topk + (1.0 - reversal)) / 5.0


def _compute_query_component_for_domain(
    *,
    domain: str,
    build_rankings: dict[str, dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    # build_rankings: build_key -> {"query_domain_scores": domain -> query_id -> model_id -> score}
    pairwise_rows: list[dict[str, Any]] = []
    build_keys = sorted(build_rankings.keys())
    total_comparable_queries = 0

    for left_key, right_key in itertools.combinations(build_keys, 2):
        left_query_map = (
            (build_rankings[left_key].get("query_domain_scores") or {}).get(domain)
            if isinstance(build_rankings[left_key].get("query_domain_scores"), dict)
            else None
        )
        right_query_map = (
            (build_rankings[right_key].get("query_domain_scores") or {}).get(domain)
            if isinstance(build_rankings[right_key].get("query_domain_scores"), dict)
            else None
        )
        if not isinstance(left_query_map, dict) or not isinstance(right_query_map, dict):
            continue

        common_queries = sorted(set(left_query_map.keys()) & set(right_query_map.keys()))
        if not common_queries:
            continue

        query_metrics: list[dict[str, float]] = []
        for query_id in common_queries:
            left_scores = left_query_map.get(query_id) or {}
            right_scores = right_query_map.get(query_id) or {}
            if not isinstance(left_scores, dict) or not isinstance(right_scores, dict):
                continue
            common_models = sorted(set(left_scores.keys()) & set(right_scores.keys()))
            if len(common_models) < 2:
                continue
            left_model_scores = {model: _to_float(left_scores.get(model), 0.0) for model in common_models}
            right_model_scores = {model: _to_float(right_scores.get(model), 0.0) for model in common_models}
            left_order = [m for m, _ in _rank_models(left_model_scores)]
            right_order = [m for m, _ in _rank_models(right_model_scores)]
            tau = _kendall_tau(left_order, right_order)
            rho = _spearman_rho(_rank_map(left_model_scores), _rank_map(right_model_scores))
            overlap = _topk_overlap(left_order, right_order, top_k)
            reversal_ratio, _ = _pairwise_reversal_ratio(left_order, right_order)
            query_metrics.append(
                {
                    "kendall_tau": tau,
                    "spearman_rho": rho,
                    "champion_same": 1.0 if left_order[0] == right_order[0] else 0.0,
                    "top_k_overlap": overlap,
                    "pairwise_reversal_ratio": reversal_ratio,
                }
            )

        if not query_metrics:
            continue

        total_comparable_queries += len(query_metrics)
        pairwise_rows.append(
            {
                "left_build": left_key,
                "right_build": right_key,
                "comparable_query_count": len(query_metrics),
                "kendall_tau": round(sum(item["kendall_tau"] for item in query_metrics) / len(query_metrics), 6),
                "spearman_rho": round(sum(item["spearman_rho"] for item in query_metrics) / len(query_metrics), 6),
                "champion_same": (
                    sum(item["champion_same"] for item in query_metrics) / len(query_metrics)
                )
                >= 0.5,
                "top_k_overlap": round(sum(item["top_k_overlap"] for item in query_metrics) / len(query_metrics), 6),
                "pairwise_reversal_ratio": round(
                    sum(item["pairwise_reversal_ratio"] for item in query_metrics) / len(query_metrics), 6
                ),
            }
        )

    summary = _metrics_from_pairwise(pairwise_rows)
    return {
        "status": ("ok" if pairwise_rows else "insufficient_query_scores"),
        "pairwise_comparisons": len(pairwise_rows),
        "comparable_query_count": total_comparable_queries,
        "summary": {key: round(val, 6) for key, val in summary.items()},
        "pairwise": pairwise_rows,
    }


def evaluate_rank_stability(
    *,
    scored_builds: list[dict[str, Any]],
    top_k: int = 3,
    rs_workload_weight: float = 0.75,
    rs_query_weight: float = 0.25,
) -> dict[str, Any]:
    normalized_builds: list[dict[str, Any]] = []
    warnings: list[str] = []

    for entry in scored_builds:
        score_path = Path(str(entry.get("score_table_path") or ""))
        rows = _load_score_table(score_path)
        model_scores: dict[str, dict[str, float]] = {}
        for row in rows:
            model_id, scores = _extract_domain_scores(row)
            if not model_id or not scores:
                continue
            model_scores[model_id] = scores

        query_score_path = Path(str(entry.get("query_score_path") or "")) if entry.get("query_score_path") else None
        if query_score_path is None or not str(query_score_path):
            query_score_path = _discover_query_score_path(score_path)
        query_domain_scores: dict[str, dict[str, dict[str, float]]] = {}
        if query_score_path and query_score_path.exists():
            query_rows = _load_query_score_table(query_score_path)
            query_domain_scores = _extract_query_scores_by_domain(query_rows)

        build_meta = entry.get("build_manifest_v2") or {}
        normalized_builds.append(
            {
                "run_id": str(entry.get("run_id") or build_meta.get("run_id") or ""),
                "build_id": str(entry.get("build_id") or build_meta.get("build_id") or ""),
                "score_table_path": str(score_path),
                "query_score_path": str(query_score_path) if query_score_path else "",
                "model_scores": model_scores,
                "query_domain_scores": query_domain_scores,
                "build_meta": _extract_build_meta(build_meta),
            }
        )

    valid_builds = [item for item in normalized_builds if item.get("model_scores")]
    if len(valid_builds) < 2:
        return {
            "contract_version": "rank_stability_report_v0_1",
            "summary": {
                "status": "insufficient_builds",
                "build_count": len(valid_builds),
                "required_min_builds": 2,
            },
            "warnings": ["Need at least 2 scored builds for rank stability."] + warnings,
            "builds": [
                {
                    "run_id": item.get("run_id"),
                    "build_id": item.get("build_id"),
                    "score_table_path": item.get("score_table_path"),
                    "query_score_path": item.get("query_score_path"),
                    "model_count": len(item.get("model_scores") or {}),
                }
                for item in normalized_builds
            ],
            "domains": {},
        }

    dataset_ids = {item["build_meta"].get("dataset_id") for item in valid_builds if item["build_meta"].get("dataset_id")}
    if len(dataset_ids) > 1:
        warnings.append("Builds contain different dataset_id values; comparability may be invalid.")

    fingerprints = {
        item["build_meta"].get("dataset_fingerprint")
        for item in valid_builds
        if item["build_meta"].get("dataset_fingerprint")
    }
    if len(fingerprints) > 1:
        warnings.append("Builds contain different dataset fingerprints; rank comparison is not strictly controlled.")

    domain_set = {"overall", "dependency_structure"}
    for build in valid_builds:
        for score_map in (build.get("model_scores") or {}).values():
            if isinstance(score_map, dict):
                domain_set.update(str(k) for k in score_map.keys())
        for domain in (build.get("query_domain_scores") or {}).keys():
            domain_set.add(str(domain))
    domains = ["overall"] + sorted(d for d in domain_set if d != "overall")
    domain_results: dict[str, Any] = {}

    for domain in domains:
        build_rankings: dict[str, dict[str, Any]] = {}
        for build in valid_builds:
            model_scores = build["model_scores"]
            domain_scores = {
                model_id: score_map[domain]
                for model_id, score_map in model_scores.items()
                if domain in score_map
            }
            if len(domain_scores) < 2:
                continue
            ordered = _rank_models(domain_scores)
            order_ids = [model_id for model_id, _ in ordered]
            build_rankings[build["build_id"] or build["run_id"]] = {
                "run_id": build["run_id"],
                "build_id": build["build_id"],
                "order": order_ids,
                "rank_map": _rank_map(domain_scores),
                "champion": order_ids[0] if order_ids else None,
                "model_scores": domain_scores,
                "query_domain_scores": build.get("query_domain_scores") or {},
            }

        if len(build_rankings) < 2:
            continue

        pairwise_rows: list[dict[str, Any]] = []
        reversal_counter: dict[tuple[str, str], int] = defaultdict(int)
        pair_count = 0

        build_keys = sorted(build_rankings.keys())
        for left_key, right_key in itertools.combinations(build_keys, 2):
            left = build_rankings[left_key]
            right = build_rankings[right_key]

            common_models = [model for model in left["order"] if model in right["rank_map"]]
            if len(common_models) < 2:
                continue

            left_order = [model for model in left["order"] if model in common_models]
            right_order = [model for model in right["order"] if model in common_models]

            tau = _kendall_tau(left_order, right_order)
            rho = _spearman_rho(
                {model: left["rank_map"][model] for model in common_models},
                {model: right["rank_map"][model] for model in common_models},
            )
            overlap = _topk_overlap(left_order, right_order, top_k)
            reversal_ratio, reversal_flags = _pairwise_reversal_ratio(left_order, right_order)
            for pair, flag in reversal_flags.items():
                if flag:
                    reversal_counter[pair] += 1
            pair_count += 1

            pairwise_rows.append(
                {
                    "left_build": left_key,
                    "right_build": right_key,
                    "common_model_count": len(common_models),
                    "kendall_tau": round(tau, 6),
                    "spearman_rho": round(rho, 6),
                    "champion_same": left.get("champion") == right.get("champion"),
                    "top_k_overlap": round(overlap, 6),
                    "pairwise_reversal_ratio": round(reversal_ratio, 6),
                }
            )

        if not pairwise_rows:
            continue

        reference_build = build_rankings[build_keys[0]]
        ref_champion = reference_build.get("champion")
        champions = [build_rankings[key].get("champion") for key in build_keys]
        champion_retention = sum(1 for champ in champions if champ == ref_champion) / max(1, len(champions))

        avg_tau = sum(row["kendall_tau"] for row in pairwise_rows) / len(pairwise_rows)
        avg_rho = sum(row["spearman_rho"] for row in pairwise_rows) / len(pairwise_rows)
        avg_topk = sum(row["top_k_overlap"] for row in pairwise_rows) / len(pairwise_rows)
        avg_reversal = sum(row["pairwise_reversal_ratio"] for row in pairwise_rows) / len(pairwise_rows)

        top_reversals = [
            {
                "model_pair": list(pair),
                "reversal_count": count,
                "reversal_rate": round(count / max(1, pair_count), 6),
            }
            for pair, count in sorted(reversal_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        workload_summary = {
            "avg_kendall_tau": round(avg_tau, 6),
            "avg_spearman_rho": round(avg_rho, 6),
            "champion_retention_rate": round(champion_retention, 6),
            "avg_top_k_overlap": round(avg_topk, 6),
            "avg_pairwise_reversal_ratio": round(avg_reversal, 6),
            "reference_champion": ref_champion,
            "top_k": top_k,
        }
        workload_score = _stability_score(workload_summary)

        query_component = _compute_query_component_for_domain(
            domain=domain,
            build_rankings=build_rankings,
            top_k=top_k,
        )
        query_summary = query_component.get("summary") if isinstance(query_component, dict) else {}
        query_status = str(query_component.get("status") or "") if isinstance(query_component, dict) else ""
        if query_status != "ok":
            if domain == "overall":
                warnings.append("RS_query unavailable for overall domain; fallback to RS_workload only.")
            effective_w_workload = 1.0
            effective_w_query = 0.0
            query_score = None
        else:
            effective_w_workload = _to_float(rs_workload_weight, 0.75)
            effective_w_query = _to_float(rs_query_weight, 0.25)
            total_w = effective_w_workload + effective_w_query
            if total_w <= 1e-9:
                effective_w_workload, effective_w_query = 1.0, 0.0
            else:
                effective_w_workload /= total_w
                effective_w_query /= total_w
            query_score = _stability_score(query_summary)

        combined_summary = {
            "avg_kendall_tau": round(
                effective_w_workload * _to_float(workload_summary.get("avg_kendall_tau"), 0.0)
                + effective_w_query * _to_float((query_summary or {}).get("avg_kendall_tau"), 0.0),
                6,
            ),
            "avg_spearman_rho": round(
                effective_w_workload * _to_float(workload_summary.get("avg_spearman_rho"), 0.0)
                + effective_w_query * _to_float((query_summary or {}).get("avg_spearman_rho"), 0.0),
                6,
            ),
            "champion_retention_rate": round(
                effective_w_workload * _to_float(workload_summary.get("champion_retention_rate"), 0.0)
                + effective_w_query * _to_float((query_summary or {}).get("champion_retention_rate"), 0.0),
                6,
            ),
            "avg_top_k_overlap": round(
                effective_w_workload * _to_float(workload_summary.get("avg_top_k_overlap"), 0.0)
                + effective_w_query * _to_float((query_summary or {}).get("avg_top_k_overlap"), 0.0),
                6,
            ),
            "avg_pairwise_reversal_ratio": round(
                effective_w_workload * _to_float(workload_summary.get("avg_pairwise_reversal_ratio"), 0.0)
                + effective_w_query * _to_float((query_summary or {}).get("avg_pairwise_reversal_ratio"), 0.0),
                6,
            ),
            "reference_champion": ref_champion,
            "top_k": top_k,
            "rs_workload_score": round(workload_score, 6),
            "rs_query_score": (round(query_score, 6) if query_score is not None else None),
            "rank_stability_score": round(
                effective_w_workload * workload_score + effective_w_query * (query_score or 0.0), 6
            ),
            "rs_workload_weight": round(effective_w_workload, 6),
            "rs_query_weight": round(effective_w_query, 6),
            "rs_query_status": query_status or "insufficient_query_scores",
        }

        domain_results[domain] = {
            "build_count": len(build_rankings),
            "pairwise_comparisons": len(pairwise_rows),
            "summary": combined_summary,
            "workload_component": {
                "summary": workload_summary,
                "pairwise": pairwise_rows,
                "top_reversal_pairs": top_reversals,
            },
            "query_component": query_component,
            "pairwise": pairwise_rows,
            "top_reversal_pairs": top_reversals,
        }

    overall_summary = domain_results.get("overall", {}).get("summary") if isinstance(domain_results.get("overall"), dict) else {}
    rank_overall = _to_float((overall_summary or {}).get("rank_stability_score"), default=0.0)

    return {
        "contract_version": "rank_stability_report_v0_1",
        "summary": {
            "status": "ok" if domain_results else "no_comparable_domains",
            "build_count": len(valid_builds),
            "domain_count": len(domain_results),
            "domains": sorted(domain_results.keys()),
            "rank_stability_score": round(rank_overall, 6),
            "rank_stability_formula": (
                f"RankStability = {rs_workload_weight:.3f}*RS_workload + "
                f"{rs_query_weight:.3f}*RS_query (fallback to workload-only when RS_query unavailable)"
            ),
            "requested_rs_workload_weight": round(float(rs_workload_weight), 6),
            "requested_rs_query_weight": round(float(rs_query_weight), 6),
        },
        "warnings": warnings,
        "builds": [
            {
                "run_id": item.get("run_id"),
                "build_id": item.get("build_id"),
                "score_table_path": item.get("score_table_path"),
                "query_score_path": item.get("query_score_path"),
                "model_count": len(item.get("model_scores") or {}),
                "build_meta": item.get("build_meta"),
            }
            for item in valid_builds
        ],
        "domains": domain_results,
    }
