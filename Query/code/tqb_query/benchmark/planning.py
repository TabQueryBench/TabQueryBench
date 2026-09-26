"""Family planning and research question generation."""

from __future__ import annotations

import hashlib
from itertools import combinations
import re
from uuid import uuid4

from tqb_query.benchmark.canonical_sql import stable_question_identity
from tqb_query.benchmark.facets import choose_facet_id, facets_for_family
from tqb_query.benchmark.llm_runtime import BenchmarkLLMRuntime
from tqb_query.benchmark.models import FIVE_FIXED_FAMILIES, FamilyPlan, OperationalUnderstanding, ResearchQuestion, StaticDatasetUnderstanding


def _family_floor_by_status(status: str) -> int:
    if status in {"applicable", "likely_applicable"}:
        return 1
    if status == "uncertain":
        return 1
    return 0


def _family_cap_by_status(status: str, total_budget: int) -> int:
    if status in {"applicable", "likely_applicable"}:
        return max(1, total_budget)
    if status == "uncertain":
        return max(1, total_budget // 2)
    # likely_not_applicable
    return 1


def _dedupe_preserve(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        token = str(item).strip()
        if not token or token in out:
            continue
        out.append(token)
    return out


def _stable_hash_token(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _format_field_combo(fields: list[str]) -> str:
    if not fields:
        return "the selected fields"
    if len(fields) == 1:
        return fields[0]
    if len(fields) == 2:
        return f"{fields[0]} and {fields[1]}"
    if len(fields) == 3:
        return f"{fields[0]}, {fields[1]}, and {fields[2]}"
    head = ", ".join(fields[:-1])
    return f"{head}, and {fields[-1]}"


def _build_field_combo_pool(
    *,
    static_understanding: StaticDatasetUnderstanding,
    operational_understanding: OperationalUnderstanding,
    family: str,
    max_fields: int = 12,
    max_combos: int = 64,
) -> list[list[str]]:
    target = static_understanding.target_column
    key_fields = [field for field in static_understanding.key_fields if field and field != target]

    if not key_fields:
        key_fields = [
            field
            for field in static_understanding.field_roles.keys()
            if field and field != target
        ]
    key_fields = _dedupe_preserve(key_fields)

    boosted_fields: list[str] = []
    boosted_combos: list[list[str]] = []
    for combo in operational_understanding.promising_field_combinations:
        if not isinstance(combo, list):
            continue
        cleaned = _dedupe_preserve([str(field) for field in combo if isinstance(field, str)])
        cleaned = [field for field in cleaned if field and field != target and field in key_fields]
        if not cleaned:
            continue
        trimmed = cleaned[:4]
        boosted_combos.append(trimmed)
        for field in trimmed:
            if field not in boosted_fields:
                boosted_fields.append(field)

    ordered_fields = _dedupe_preserve(boosted_fields + key_fields)
    if len(ordered_fields) > max_fields:
        protected = ordered_fields[: min(8, len(ordered_fields))]
        tail = ordered_fields[len(protected) :]
        tail = sorted(
            tail,
            key=lambda field: _stable_hash_token(f"{static_understanding.dataset_id}|{family}|field|{field}"),
        )
        ordered_fields = protected + tail[: max(0, max_fields - len(protected))]

    auto_combos: list[list[str]] = []
    for size in (1, 2, 3, 4):
        if len(ordered_fields) < size:
            continue
        for combo in combinations(ordered_fields, size):
            auto_combos.append(list(combo))

    auto_combos = sorted(
        auto_combos,
        key=lambda combo: _stable_hash_token(f"{static_understanding.dataset_id}|{family}|combo|{'|'.join(combo)}"),
    )

    buckets: dict[int, list[list[str]]] = {1: [], 2: [], 3: [], 4: []}
    seen: set[tuple[str, ...]] = set()
    for combo in boosted_combos + auto_combos:
        normalized = tuple(_dedupe_preserve(combo)[:4])
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        size = len(normalized)
        if size in buckets:
            buckets[size].append(list(normalized))

    combo_pool: list[list[str]] = []
    cursors = {1: 0, 2: 0, 3: 0, 4: 0}
    cycle = [2, 1, 3, 2, 1, 3, 4]
    while len(combo_pool) < max_combos:
        progressed = False
        for size in cycle:
            idx = cursors[size]
            if idx >= len(buckets[size]):
                continue
            combo_pool.append(buckets[size][idx])
            cursors[size] += 1
            progressed = True
            if len(combo_pool) >= max_combos:
                break
        if not progressed:
            break
    return combo_pool


def build_family_plan(
    *,
    static_understanding: StaticDatasetUnderstanding,
    operational_understanding: OperationalUnderstanding,
    round_index: int,
    max_questions: int,
    focus_families: list[str] | None = None,
) -> FamilyPlan:
    families = list(FIVE_FIXED_FAMILIES)
    if focus_families:
        focus_set = set(focus_families)
        families = [family for family in FIVE_FIXED_FAMILIES if family in focus_set]

    if not families:
        return FamilyPlan(round_index=round_index, attempts_by_family={}, rationale={})

    status_map = static_understanding.family_applicability_summary
    attempts_by_family: dict[str, int] = {}
    rationale: dict[str, str] = {}

    for family in families:
        status = status_map.get(family, "uncertain")
        floor = _family_floor_by_status(status)
        attempts_by_family[family] = floor
        rationale[family] = (
            f"status={status};floor={floor};score={operational_understanding.family_scores.get(family, 0.5):.3f}"
        )

    budget = max(max_questions, 0)
    current = sum(attempts_by_family.values())
    if current > budget:
        # Prefer keeping higher-priority families when trimming.
        priority = [f for f in operational_understanding.family_priority_order if f in attempts_by_family]
        idx = len(priority) - 1
        while current > budget and idx >= 0:
            family = priority[idx]
            if attempts_by_family[family] > 0:
                attempts_by_family[family] -= 1
                current -= 1
            idx -= 1
    else:
        remaining = budget - current
        priority = [f for f in operational_understanding.family_priority_order if f in attempts_by_family]
        if not priority:
            priority = families

        idx = 0
        while remaining > 0 and priority:
            family = priority[idx % len(priority)]
            status = status_map.get(family, "uncertain")
            cap = _family_cap_by_status(status, total_budget=budget)
            if attempts_by_family[family] < cap:
                attempts_by_family[family] += 1
                remaining -= 1
            idx += 1
            # Safeguard from infinite loop when all caps reached.
            if idx > 10_000:
                break

    # If this round focuses on missing families, guarantee at least one attempt each,
    # but still respect low-signal families by capping to one.
    if focus_families:
        for family in families:
            if attempts_by_family.get(family, 0) <= 0:
                attempts_by_family[family] = 1
                rationale[family] += ";focus_force=1"

    return FamilyPlan(
        round_index=round_index,
        attempts_by_family=attempts_by_family,
        rationale=rationale,
    )


def _fallback_questions(
    *,
    family: str,
    target_column: str,
    candidate_field_sets: list[list[str]],
    available_facet_ids: list[str],
    count: int,
) -> list[dict]:
    pool = [combo for combo in (candidate_field_sets or []) if combo]
    if not pool:
        pool = [[target_column]]

    items: list[dict] = []
    for idx in range(count):
        combo = list(pool[idx % len(pool)])
        combo_text = _format_field_combo(combo)
        field_a = combo[0] if combo else target_column
        field_b = combo[1] if len(combo) > 1 else field_a

        templates = {
            "subgroup_structure": f"How does {target_column} distribution vary across {combo_text}?",
            "conditional_dependency_structure": f"How does {target_column} change across combinations of {combo_text}?",
            "tail_rarity_structure": f"Which values of {combo_text} are most associated with rare {target_column} labels?",
            "missingness_structure": f"Are there missingness-related patterns by {combo_text} that relate to {target_column}?",
            "cardinality_structure": f"How concentrated is {target_column} across support patterns of {combo_text}?",
        }

        question = templates.get(family, f"What structure does {target_column} show with {combo_text}?")
        related = combo if combo else [field_a, field_b] if field_b != field_a else [field_a]
        items.append(
            {
                "question": question,
                "related_fields": related[:4],
                "target": target_column,
                "intent": f"family_probe:{family}",
                "rationale": f"Fallback template generated for family={family}.",
                "evidence_expectation": "Grouped support and target-relevant summary table.",
                "comparator_type": "distribution",
                "intended_facet_id": available_facet_ids[idx % len(available_facet_ids)] if available_facet_ids else "unknown",
                "reason_codes": ["RQ_FALLBACK_TEMPLATE", "RQ_FAMILY_ALIGNED"],
            }
        )
    return items


def _infer_comparator_type(question_text: str, fallback: str = "distribution") -> str:
    text = (question_text or "").lower()
    if any(token in text for token in ["top", "bottom", "rank", "highest", "lowest"]):
        return "ranking"
    if any(token in text for token in ["difference", "compare", "versus", "vs"]):
        return "contrast"
    if any(token in text for token in ["rate", "ratio", "percentage", "proportion"]):
        return "rate"
    return fallback


def generate_research_questions_for_family(
    *,
    llm_runtime: BenchmarkLLMRuntime,
    static_understanding: StaticDatasetUnderstanding,
    operational_understanding: OperationalUnderstanding,
    family: str,
    family_facet_catalog: dict | None = None,
    num_questions: int,
) -> list[ResearchQuestion]:
    if num_questions <= 0:
        return []

    facet_defs = facets_for_family(family_facet_catalog or {}, family)
    facet_ids = [str(item.get("facet_id")) for item in facet_defs if item.get("facet_id")]
    if not facet_ids:
        facet_ids = [f"{family}_general"]
    field_combo_pool = _build_field_combo_pool(
        static_understanding=static_understanding,
        operational_understanding=operational_understanding,
        family=family,
    )
    allowed_related_fields = set(
        _dedupe_preserve(
            [
                field
                for field in (
                    static_understanding.key_fields
                    or list(static_understanding.field_roles.keys())
                )
                if field and field != static_understanding.target_column
            ]
        )
    )

    system_prompt = """
You generate benchmark research questions for tabular SQL evaluation.
Rules:
- Family is fixed and must be respected.
- Return strict JSON only.
- Keep questions answerable using a single table.
- Avoid requiring joins or external data.
- Prefer subgroup/target-relevant and non-trivial questions.
- related_fields must contain 1 to 4 fields.
- Avoid always reusing the same 2-field pair; diversify related_fields combinations across returned questions.
- For likely_not_applicable families, generate only lightweight diagnostics.
JSON format:
{
  "questions": [
    {
      "question": "...",
      "related_fields": ["..."],
      "target": "...",
      "intent": "...",
      "rationale": "...",
      "evidence_expectation": "...",
      "comparator_type": "...",
      "intended_facet_id": "...",
      "reason_codes": ["..."]
    }
  ]
}
""".strip()

    family_status = static_understanding.family_applicability_summary.get(family, "uncertain")
    user_prompt = (
        f"dataset_id={static_understanding.dataset_id}\n"
        f"family={family}\n"
        f"family_status={family_status}\n"
        f"num_questions={num_questions}\n"
        f"task_type={static_understanding.task_type}\n"
        f"target_column={static_understanding.target_column}\n"
        f"key_fields={static_understanding.key_fields}\n"
        f"ordered_fields={static_understanding.ordered_fields}\n"
        f"family_scores={operational_understanding.family_scores}\n"
        f"promising_field_combinations={operational_understanding.promising_field_combinations}\n"
        f"field_combo_pool_1_to_4={field_combo_pool}\n"
        f"low_support_signals={operational_understanding.low_support_signals}\n"
        f"triviality_signals={operational_understanding.triviality_signals}\n"
        f"available_facets={facet_defs}\n"
        "Generate concise, distinct research questions."
    )

    payload = llm_runtime.invoke_json(
        phase="research_question_generation",
        module="planning.generate_research_questions_for_family",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        question_for_usage=f"generate_rq:{family}",
    )

    items = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        items = _fallback_questions(
            family=family,
            target_column=static_understanding.target_column,
            candidate_field_sets=field_combo_pool,
            available_facet_ids=facet_ids,
            count=num_questions,
        )

    research_questions: list[ResearchQuestion] = []
    for item_index, item in enumerate(items[:num_questions]):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        combo_fallback = field_combo_pool[item_index % len(field_combo_pool)] if field_combo_pool else []
        related_fields_raw = [str(v) for v in (item.get("related_fields") or []) if isinstance(v, str)]
        target = str(item.get("target") or static_understanding.target_column)
        related_fields = _dedupe_preserve(
            [
                field
                for field in related_fields_raw
                if field and field != target and (not allowed_related_fields or field in allowed_related_fields)
            ]
        )[:4]
        if not related_fields:
            related_fields = [field for field in combo_fallback if field != target][:4]
        intent = str(item.get("intent") or f"family_probe:{family}")
        rationale = str(item.get("rationale") or f"Family-oriented probe for {family}.")
        evidence_expectation = str(
            item.get("evidence_expectation")
            or "Grouped support and/or target-conditioned summary with interpretable structure."
        )
        suggested_facet = str(item.get("intended_facet_id") or "")
        intended_facet_id = choose_facet_id(
            catalog=family_facet_catalog or {},
            family=family,
            item_index=item_index,
            suggested_facet_id=suggested_facet,
        )
        comparator_type = str(item.get("comparator_type") or _infer_comparator_type(question))
        reason_codes = [str(v) for v in (item.get("reason_codes") or []) if isinstance(v, str)]
        if "RQ_FAMILY_ALIGNED" not in reason_codes:
            reason_codes.append("RQ_FAMILY_ALIGNED")
        if intended_facet_id != "unknown":
            facet_code = re.sub(r"[^A-Za-z0-9]+", "_", intended_facet_id.upper()).strip("_")
            reason_codes.append(f"RQ_FACET_{facet_code}")

        stable_question_id = stable_question_identity(
            dataset_id=static_understanding.dataset_id,
            family_id=family,
            intended_facet_id=intended_facet_id,
            question_text=question,
        )

        research_questions.append(
            ResearchQuestion(
                question_id=f"rq_{family}_{uuid4().hex[:8]}",
                family=family,
                question=question,
                related_fields=related_fields,
                target=target,
                intent=intent,
                reason_codes=reason_codes,
                family_id=family,
                intended_facet_id=intended_facet_id,
                question_text=question,
                target_columns=[target] if target else [],
                related_columns=related_fields,
                rationale=rationale,
                evidence_expectation=evidence_expectation,
                comparator_type=comparator_type,
                risk_tags=[],
                uncertainty_tags=[],
                stable_question_id=stable_question_id,
            )
        )

    if not research_questions:
        fallback_items = _fallback_questions(
            family=family,
            target_column=static_understanding.target_column,
            candidate_field_sets=field_combo_pool,
            available_facet_ids=facet_ids,
            count=num_questions,
        )
        for item_index, item in enumerate(fallback_items):
            intended_facet_id = choose_facet_id(
                catalog=family_facet_catalog or {},
                family=family,
                item_index=item_index,
                suggested_facet_id=str(item.get("intended_facet_id") or ""),
            )
            stable_question_id = stable_question_identity(
                dataset_id=static_understanding.dataset_id,
                family_id=family,
                intended_facet_id=intended_facet_id,
                question_text=item["question"],
            )
            research_questions.append(
                ResearchQuestion(
                    question_id=f"rq_{family}_{uuid4().hex[:8]}",
                    family=family,
                    question=item["question"],
                    related_fields=item["related_fields"],
                    target=item["target"],
                    intent=item["intent"],
                    reason_codes=item["reason_codes"],
                    family_id=family,
                    intended_facet_id=intended_facet_id,
                    question_text=item["question"],
                    target_columns=[item["target"]] if item.get("target") else [],
                    related_columns=item["related_fields"],
                    rationale=str(item.get("rationale") or f"Fallback template generated for family={family}."),
                    evidence_expectation=str(
                        item.get("evidence_expectation")
                        or "Grouped support and/or target-conditioned summary with interpretable structure."
                    ),
                    comparator_type=str(item.get("comparator_type") or _infer_comparator_type(item["question"])),
                    risk_tags=[],
                    uncertainty_tags=[],
                    stable_question_id=stable_question_id,
                )
            )

    return research_questions
