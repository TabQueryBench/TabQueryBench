"""Runtime utilities for template-grounded agent selection."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqb_query.workload_grounding.adherence import extract_template_ids_from_sql, structure_flags

PORTABILITY_SCORE = {"yes": 2, "partial": 1, "no": 0}
PRIORITY_SCORE = {"p0": 2, "p1": 1, "extension": 0}

TEMPLATE_KEYWORD_HINTS: dict[str, list[str]] = {
    "tpl_clickbench_group_count": ["count", "distribution", "breakdown", "how many", "number"],
    "tpl_clickbench_filtered_topk_group_count": ["most", "least", "top", "common", "frequent", "among", "within", "filter"],
    "tpl_clickbench_group_distinct_topk": ["distinct", "unique", "coverage", "users", "entities"],
    "tpl_clickbench_filtered_distinct_topk": ["distinct", "unique", "coverage", "users", "entities", "among", "within", "filter"],
    "tpl_clickbench_group_summary_topk": ["summary", "compare", "support", "average", "avg", "mean", "distinct"],
    "tpl_clickbench_two_dimensional_topk_count": ["combination", "combinations", "pair", "pairs", "joint", "across both"],
    "tpl_m4_group_condition_rate": ["rate", "percentage", "percent", "proportion", "likelihood", "associated"],
    "tpl_m4_group_ratio_two_conditions": ["ratio", "versus", "vs", "compared", "relative"],
    "tpl_c2_filtered_group_count_2d": ["combination", "combinations", "pair", "pairs", "among", "target class"],
    "tpl_h2o_group_sum": ["sum", "total", "amount", "charges", "revenue"],
    "tpl_h2o_topn_within_group": ["top", "highest", "largest", "rank", "within each", "within group"],
    "tpl_m4_support_guarded_group_avg": ["average", "avg", "mean", "robust", "stable", "enough support"],
    "tpl_m4_group_avg_numeric": ["average", "avg", "mean"],
    "tpl_m4_two_dimensional_group_avg": ["combination", "combinations", "interaction", "across both", "across sex and smoker"],
    "tpl_m4_binned_numeric_group_avg": ["band", "bands", "bucket", "bucketed", "range", "across age bands", "age band"],
    "tpl_m4_median_filtered_numeric": ["median"],
    "tpl_tpcds_within_group_share": ["share", "contribution", "fraction", "percent of total"],
    "tpl_grouped_percentile_point": ["p95", "p99", "percentile", "quantile", "median"],
    "tpl_conditional_group_quantiles": ["p95", "p99", "percentile", "quantile", "under", "successful", "failed"],
    "tpl_threshold_rarity_cdf": ["how rare", "rarity", "threshold", "cdf"],
}

QUESTION_INTENT_HINTS: dict[str, list[str]] = {
    "count": ["count", "how many", "distribution", "breakdown", "most common", "frequent", "records"],
    "sum": ["sum", "total", "total charges", "total revenue", "aggregate mass"],
    "avg": ["average", "avg", "mean"],
    "rate": ["rate", "percentage", "percent", "proportion", "likelihood", "associated"],
    "ratio": ["ratio", "versus", "vs", "relative"],
    "share": ["share", "contribution", "fraction", "percent of total"],
    "distinct": ["distinct", "unique", "coverage", "users", "entities"],
    "bucketed": ["band", "bands", "bucket", "bucketed", "range", "ranges"],
    "percentile": ["p95", "p99", "percentile", "quantile", "median"],
    "tail": ["rare", "rarity", "threshold", "outlier", "extreme", "tail", "p95", "p99", "percentile", "quantile", "median"],
    "topk": ["top", "highest", "largest", "rank", "most common", "least common"],
    "window": ["within each", "within group", "rank", "top"],
}

SPECIALIZATION_TAGS: dict[str, list[str]] = {
    "tpl_clickbench_group_distinct_topk": ["distinct"],
    "tpl_clickbench_filtered_distinct_topk": ["distinct"],
    "tpl_m4_group_condition_rate": ["rate"],
    "tpl_m4_group_ratio_two_conditions": ["ratio"],
    "tpl_tpcds_within_group_share": ["share"],
    "tpl_m4_binned_numeric_group_avg": ["bucketed"],
    "tpl_m4_median_filtered_numeric": ["percentile", "tail"],
    "tpl_m4_quantile_tail_slice": ["percentile", "tail"],
    "tpl_m4_global_zscore_outliers": ["tail"],
    "tpl_tpch_relative_total_threshold": ["tail", "threshold"],
    "tpl_tpch_thresholded_group_ranking": ["tail", "threshold"],
    "tpl_tpcds_subgroup_baseline_outlier": ["tail"],
    "tpl_tpcds_baseline_gated_extreme_ranking": ["tail", "threshold"],
    "tpl_tail_weighted_topk_sum": ["tail", "topk"],
    "tpl_grouped_percentile_point": ["percentile", "tail"],
    "tpl_conditional_group_quantiles": ["percentile", "tail", "rate"],
    "tpl_threshold_rarity_cdf": ["threshold", "tail"],
}


@dataclass
class GroundedTemplateCandidate:
    rank: int | None
    template_id: str
    template_name: str
    primary_family: str
    secondary_family: str | None
    priority: str
    portability: str
    portability_summary: dict[str, int]
    activation_tier: str
    dialect_sensitive: bool
    why_pick: str
    use_when: str
    avoid_when: str
    required_roles: list[str]
    constraints: list[str]
    sql_skeleton: str
    provenance: dict[str, Any]
    question_match_score: int
    screening_tags: list[str]
    screening_reasons: list[str]
    screening_stage: str


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _question_intents(question: str) -> dict[str, bool]:
    lowered = _normalize_text(question)
    intents = {
        name: any(hint in lowered for hint in hints)
        for name, hints in QUESTION_INTENT_HINTS.items()
    }
    if re.search(r"\b(combination|combinations|pair|pairs|joint|interaction)\b", lowered):
        intents["two_dimensional"] = True
    elif re.search(r"\bacross\b.+\band\b", lowered):
        intents["two_dimensional"] = True
    else:
        intents["two_dimensional"] = False
    return intents


def _candidate_tags(template_id: str, sql_skeleton: str) -> list[str]:
    tags = set(SPECIALIZATION_TAGS.get(template_id, []))
    flags = structure_flags(sql_skeleton)
    if flags.get("two_dimensional"):
        tags.add("two_dimensional")
    if flags.get("window"):
        tags.add("window")
    if flags.get("percentile"):
        tags.add("percentile")
        tags.add("tail")
    if flags.get("support_guard"):
        tags.add("support_guard")
    return sorted(tags)


def _strict_screen_reasons(
    *,
    template_id: str,
    portability: str,
    dialect_sensitive: bool,
    sql_skeleton: str,
    intents: dict[str, bool],
) -> list[str]:
    reasons: list[str] = []
    tags = _candidate_tags(template_id, sql_skeleton)

    if portability == "no":
        reasons.append("portable=no")
        return reasons

    if dialect_sensitive and not intents["percentile"]:
        reasons.append("dialect_sensitive_without_explicit_percentile_signal")

    if "two_dimensional" in tags and not intents["two_dimensional"]:
        reasons.append("two_dimensional_template_without_joint_question_signal")
    if "bucketed" in tags and not intents["bucketed"]:
        reasons.append("bucketed_template_without_band_or_bucket_signal")
    if "share" in tags and not intents["share"]:
        reasons.append("share_template_without_share_signal")
    if "ratio" in tags and not intents["ratio"]:
        reasons.append("ratio_template_without_ratio_signal")
    if "rate" in tags and not intents["rate"]:
        reasons.append("rate_template_without_rate_signal")
    if "distinct" in tags and not intents["distinct"]:
        reasons.append("distinct_template_without_distinct_signal")
    if "percentile" in tags and not intents["percentile"]:
        reasons.append("percentile_template_without_percentile_signal")
    if "threshold" in tags and not intents["tail"]:
        reasons.append("threshold_tail_template_without_tail_signal")
    if "window" in tags and not (intents["topk"] or intents["window"]):
        reasons.append("window_template_without_topk_signal")

    return reasons


def _candidate_sort_key(candidate: GroundedTemplateCandidate) -> tuple[int, int, int, int]:
    return (
        -PORTABILITY_SCORE.get(candidate.portability, 0),
        -candidate.question_match_score,
        -PRIORITY_SCORE.get(candidate.priority, 0),
        candidate.rank if candidate.rank is not None else 999,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[obj["template_id"]] = obj
    return rows


def _load_portability(path: Path) -> dict[tuple[str, str], str]:
    rows: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(row["dataset_id"], row["template_id"])] = row["portable"]
    return rows


def _question_score(template_id: str, question: str) -> int:
    lowered = _normalize_text(question)
    score = 0
    for hint in TEMPLATE_KEYWORD_HINTS.get(template_id, []):
        if hint in lowered:
            score += 1
    return score


def _build_candidate(
    item: dict[str, Any],
    templates: dict[str, dict[str, Any]],
    portability_lookup: dict[tuple[str, str], str],
    dataset_id: str,
    question: str,
) -> GroundedTemplateCandidate:
    template_id = item["template_id"]
    template = templates[template_id]
    portability = portability_lookup.get((dataset_id, template_id), "no")
    screening_tags = _candidate_tags(template_id, template["sql_skeleton"])
    return GroundedTemplateCandidate(
        rank=item.get("rank"),
        template_id=template_id,
        template_name=item["template_name"],
        primary_family=item["primary_family"],
        secondary_family=item.get("secondary_family"),
        priority=item["priority"],
        portability=portability,
        portability_summary=item.get("portability_summary", {}),
        activation_tier=item.get("activation_tier", template.get("activation_tier", "core")),
        dialect_sensitive=bool(item.get("dialect_sensitive", template.get("dialect_sensitive", False))),
        why_pick=item["why_pick"],
        use_when=item["use_when"],
        avoid_when=item["avoid_when"],
        required_roles=list(item.get("required_roles", [])),
        constraints=list(item.get("constraints", [])),
        sql_skeleton=template["sql_skeleton"],
        provenance=dict(item.get("provenance", {})),
        question_match_score=_question_score(template_id, question),
        screening_tags=screening_tags,
        screening_reasons=[],
        screening_stage="unreviewed",
    )


def select_grounded_templates(
    *,
    dataset_id: str,
    question: str,
    spec_path: Path,
    spec_bucket: str = "all_core",
    core_library_path: Path,
    portability_report_path: Path,
    min_templates: int = 10,
    max_templates: int | None = None,
    preferred_template_id: str | None = None,
) -> dict[str, Any]:
    spec = _load_json(spec_path)
    templates = _load_jsonl_by_id(core_library_path)
    portability_lookup = _load_portability(portability_report_path)
    if spec_bucket not in spec:
        available_buckets = [key for key, value in spec.items() if isinstance(value, list)]
        raise KeyError(f"Spec bucket `{spec_bucket}` not found in {spec_path}. Available list buckets: {available_buckets}")

    intents = _question_intents(question)
    raw_pool = [
        _build_candidate(item, templates, portability_lookup, dataset_id, question)
        for item in spec.get(spec_bucket, [])
    ]
    strict_keep: list[GroundedTemplateCandidate] = []
    backfill_pool: list[GroundedTemplateCandidate] = []
    excluded_pool: list[GroundedTemplateCandidate] = []

    for candidate in raw_pool:
        reasons = _strict_screen_reasons(
            template_id=candidate.template_id,
            portability=candidate.portability,
            dialect_sensitive=candidate.dialect_sensitive,
            sql_skeleton=candidate.sql_skeleton,
            intents=intents,
        )
        candidate.screening_reasons = reasons
        if candidate.portability == "no":
            candidate.screening_stage = "excluded"
            excluded_pool.append(candidate)
        elif not reasons:
            candidate.screening_stage = "strict_keep"
            strict_keep.append(candidate)
        else:
            candidate.screening_stage = "backfill_candidate"
            backfill_pool.append(candidate)

    shortlisted = sorted(strict_keep, key=_candidate_sort_key)
    effective_max = None if max_templates is None or max_templates <= 0 else max_templates
    if effective_max is not None:
        shortlisted = shortlisted[:effective_max]

    if len(shortlisted) < min_templates:
        needed = min_templates - len(shortlisted)
        ranked_backfill = sorted(
            backfill_pool,
            key=lambda candidate: (len(candidate.screening_reasons),) + _candidate_sort_key(candidate),
        )
        for candidate in ranked_backfill:
            if candidate.template_id in {row.template_id for row in shortlisted}:
                continue
            candidate.screening_stage = "backfill_keep"
            shortlisted.append(candidate)
            needed -= 1
            if needed <= 0:
                break
            if effective_max is not None and len(shortlisted) >= effective_max:
                break

    if effective_max is not None and len(shortlisted) > effective_max:
        shortlisted = shortlisted[:effective_max]

    preferred_applied = False
    if preferred_template_id:
        preferred = next((candidate for candidate in raw_pool if candidate.template_id == preferred_template_id), None)
        if preferred is not None:
            preferred.screening_reasons = list(preferred.screening_reasons)
            preferred.screening_reasons.append("preferred_problem_template")
            preferred.screening_stage = "preferred_force_include"
            shortlisted = [candidate for candidate in shortlisted if candidate.template_id != preferred_template_id]
            shortlisted.insert(0, preferred)
            preferred_applied = True
            if effective_max is not None:
                shortlisted = shortlisted[:effective_max]

    return {
        "dataset_id": dataset_id,
        "question": question,
        "question_intents": intents,
        "spec_bucket": spec_bucket,
        "candidate_pool_count": len(raw_pool),
        "applicable_count": len([candidate for candidate in raw_pool if candidate.portability in {"yes", "partial"}]),
        "strict_keep_count": len(strict_keep),
        "backfill_candidate_count": len(backfill_pool),
        "excluded_count": len(excluded_pool),
        "shortlist_count": len(shortlisted),
        "min_templates": min_templates,
        "max_templates": effective_max,
        "preferred_template_id": preferred_template_id,
        "preferred_template_applied": preferred_applied,
        "candidate_pool": [asdict(candidate) for candidate in raw_pool],
        "excluded_candidates": [asdict(candidate) for candidate in excluded_pool],
        "backfill_candidates": [asdict(candidate) for candidate in backfill_pool],
        "shortlist": [asdict(candidate) for candidate in shortlisted],
    }


def format_grounding_prompt(selection: dict[str, Any]) -> str:
    shortlist = selection.get("shortlist", [])
    intents = selection.get("question_intents", {})
    positive_intents = [key for key, value in intents.items() if value]
    lines = [
        "Template-grounding block:",
        f"- dataset_id: {selection['dataset_id']}",
        f"- question: {selection['question']}",
        f"- spec_bucket: {selection.get('spec_bucket', 'all_core')}",
        f"- inferred_question_intents: {positive_intents or ['generic']}",
        f"- shortlist_stats: strict_keep={selection.get('strict_keep_count', 0)} backfill_candidates={selection.get('backfill_candidate_count', 0)} excluded={selection.get('excluded_count', 0)} final_shortlist={selection.get('shortlist_count', 0)}",
        f"- preferred_template: {selection.get('preferred_template_id') or 'none'} applied={selection.get('preferred_template_applied', False)}",
        "- approved_pool: use only the shortlisted workload-grounded templates below as structural priors. Templates not listed here have already been screened out or deprioritized.",
        "- screening_rule: first identify the question shape (count / sum / avg / rate / ratio / share / two-dimensional / percentile-tail / bucketed / threshold-rarity).",
        "- selection_rule: choose the shortlisted template whose use_when best matches the question and whose avoid_when is clearly not triggered.",
        "- simplicity_rule: if multiple shortlisted templates fit, prefer the simplest one that directly answers the question; do not over-specialize.",
        "- dialect_rule: use dialect-sensitive templates only when the question explicitly asks percentile/quantile/tail behavior and no simpler non-dialect-sensitive template answers it.",
        "- sql_rule: stay structurally close to the chosen template skeleton; do not invent joins or unrelated SQL shapes.",
        "- trace_rule: when calling sql_db_query, prepend exactly one SQL comment line in the form `-- template_id: <template_id>`.",
        "- fallback_rule: if none of the shortlisted templates fit, prefer the simplest shortlist member instead of inventing a new query family.",
    ]
    for idx, item in enumerate(shortlist, start=1):
        lines.append(
            f"{idx}. template_id={item['template_id']} "
            f"| name={item['template_name']} "
            f"| portability={item['portability']} "
            f"| family={item['primary_family']} "
            f"| priority={item['priority']}"
        )
        lines.append(f"   why_pick: {item['why_pick']}")
        lines.append(f"   use_when: {item['use_when']}")
        lines.append(f"   avoid_when: {item['avoid_when']}")
        if item.get("screening_tags"):
            lines.append(f"   screening_tags: {item['screening_tags']}")
        if item.get("screening_reasons"):
            lines.append(f"   screening_note: kept after backfill despite {item['screening_reasons']}")
        lines.append(f"   required_roles: {item['required_roles']}")
        if item.get("dialect_sensitive"):
            lines.append("   dialect_note: This template is dialect-sensitive; avoid it unless necessary.")
        lines.append("   sql_skeleton:")
        for sql_line in item["sql_skeleton"].splitlines():
            lines.append(f"     {sql_line}")
    return "\n".join(lines)
