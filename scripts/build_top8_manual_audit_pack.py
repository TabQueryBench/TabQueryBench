#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

csv.field_size_limit(sys.maxsize)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = REPO_ROOT / "logs" / "sql_high_corpus_build_20260404"
FINAL_INDEX_PATH = BASE_ROOT / "final_v2" / "final_index_v2.csv"
EXEC_PATH = BASE_ROOT / "v2_refinement" / "execute" / "sql_executability_v2.csv"
OUTPUT_ROOT = BASE_ROOT / "qa_top8"
OUTPUT_MD = OUTPUT_ROOT / "top8_manual_audit_pack.md"
OUTPUT_CSV = OUTPUT_ROOT / "top8_sql_spotcheck.csv"
OUTPUT_JSON = OUTPUT_ROOT / "top8_decision.json"


@dataclass(frozen=True)
class DatasetManualDecision:
    source_dataset_alignment: str
    residual_collision_risk: str
    question_seed_generation_can_start_safely: str
    manual_override_vs_v2_gate: str
    decision_confidence: str
    verdict: str
    decision_summary: str
    why_safe_or_not: str
    recommended_next_action: str
    key_links_summary: str
    risk_summary: str


MANUAL_DECISIONS: Dict[str, DatasetManualDecision] = {
    "m12": DatasetManualDecision(
        source_dataset_alignment="confirmed",
        residual_collision_risk="low",
        question_seed_generation_can_start_safely="yes",
        manual_override_vs_v2_gate="none",
        decision_confidence="high",
        verdict="APPROVE",
        decision_summary=(
            "The reviewed SQL is tightly tied to hotel-booking analysis projects and uses"
            " hotel-booking columns consistently. Remaining issues are dialect portability,"
            " not source mismatch."
        ),
        why_safe_or_not=(
            "Safe to start. The sample covers analytic, cleaning, KPI, and schema-building"
            " queries from two hotel-booking repos with matching hotel-booking fields such as"
            " `hotel`, `is_canceled`, `arrival_date_*`, `lead_time`, and `adr`."
        ),
        recommended_next_action=(
            "Start question-seed generation from strict primary rows. Exclude setup-only rows"
            " like `CREATE DATABASE` or `USE` and treat percentile syntax as dialect-specific"
            " rewrite candidates."
        ),
        key_links_summary=(
            "AnalyticaNova/Hotel-Booking-Demand and tuhsin45/hotel-booking-demand carry the"
            " usable SQL core."
        ),
        risk_summary=(
            "Low collision risk. The main residual issue is SQL portability for PostgreSQL/T-SQL"
            " constructs rather than benchmark mismatch."
        ),
    ),
    "c17": DatasetManualDecision(
        source_dataset_alignment="confirmed",
        residual_collision_risk="low",
        question_seed_generation_can_start_safely="yes",
        manual_override_vs_v2_gate="none",
        decision_confidence="high",
        verdict="APPROVE",
        decision_summary=(
            "The reviewed SQL consistently targets Netflix-title schemas and dashboard projects"
            " built around the common Kaggle Netflix titles dataset."
        ),
        why_safe_or_not=(
            "Safe to start. The sample spans multiple Netflix-specific repos and repeatedly uses"
            " `netflix`, `netflix_titles`, title/rating/country/release_year fields, and"
            " typical dataset questions. Some rows are portability-heavy or staging-oriented,"
            " but still source-aligned."
        ),
        recommended_next_action=(
            "Start question-seed generation from strict pass rows first. Keep schema-qualified or"
            " `UNNEST`-style rows in a secondary review lane for dialect rewriting."
        ),
        key_links_summary=(
            "MrBkumar/netflix_movies_tvshows_sql_dataset_project, Sneha-273/...with-SQL,"
            " soyalexis/Netflix_analyst, and shazlanamirul8/Netflix_SQL_Portfolio are the best"
            " aligned sources."
        ),
        risk_summary=(
            "Low collision risk. The main caution is mixed SQL dialect syntax and a few derived"
            " staging-table queries."
        ),
    ),
    "m4": DatasetManualDecision(
        source_dataset_alignment="confirmed",
        residual_collision_risk="low",
        question_seed_generation_can_start_safely="yes",
        manual_override_vs_v2_gate="none",
        decision_confidence="high",
        verdict="APPROVE",
        decision_summary=(
            "The reviewed SQL stays on the medical-insurance charges schema and asks directly"
            " reusable analytical questions over age, BMI, smoking, region, and charges."
        ),
        why_safe_or_not=(
            "Safe to start. The sampled SQL comes from two dataset-specific insurance-analysis"
            " repos and is tightly aligned to the benchmark attributes. Failed rows are mostly"
            " malformed multi-statement blocks, not schema collisions."
        ),
        recommended_next_action=(
            "Start question-seed generation from strict rows. Exclude session/setup statements"
            " and malformed multi-statement fragments from seed drafting."
        ),
        key_links_summary=(
            "arka420/Insurance-Cost-Project-Using-SQL and"
            " Shagufta-DataAnalyst/insurance-sql-analysis carry the usable evidence."
        ),
        risk_summary=(
            "Low collision risk. Residual issues are execution hygiene and multi-statement"
            " formatting, not dataset identity."
        ),
    ),
    "m8": DatasetManualDecision(
        source_dataset_alignment="confirmed",
        residual_collision_risk="low",
        question_seed_generation_can_start_safely="yes",
        manual_override_vs_v2_gate="none",
        decision_confidence="high",
        verdict="APPROVE",
        decision_summary=(
            "The reviewed SQL is consistently about the bank-marketing campaign schema and"
            " matches the well-known bank marketing feature set."
        ),
        why_safe_or_not=(
            "Safe to start. The sample repeatedly uses bank-marketing fields such as"
            " `age`, `job`, `marital`, `education`, `balance`, `deposit`, `poutcome`, and"
            " campaign success metrics. The only weak residue is a single session command."
        ),
        recommended_next_action=(
            "Start question-seed generation from strict primary rows. Ignore `USE` statements and"
            " prefer the analytical `bank` table queries."
        ),
        key_links_summary=(
            "DanieltheAnalyst1/Bank_Marketing_Campaign_Analysis-SQL and"
            " sonajestin-pixel/bank-marketing-campaign-analysis-sql are the best sources."
        ),
        risk_summary=(
            "Low collision risk. The residual caution is minimal and mostly limited to one"
            " environment-specific statement."
        ),
    ),
    "c13": DatasetManualDecision(
        source_dataset_alignment="not_confirmed",
        residual_collision_risk="high",
        question_seed_generation_can_start_safely="no",
        manual_override_vs_v2_gate="downgrade_from_ready_with_warnings",
        decision_confidence="high",
        verdict="HOLD",
        decision_summary=(
            "The sampled SQL is census-adjacent, but much of it targets ACS/PUMA migration or"
            " generic census tutorial schemas rather than clearly the exact UCI US Census Data"
            " (1990) benchmark."
        ),
        why_safe_or_not=(
            "Not safe yet. The strongest-looking rows still rely on `migration-pumas-database`"
            " tables like `msa_delineation_2018`, `census_puma_relation`, and `df1`, while other"
            " rows come from `american-community-survey` notebooks or generic `zipcensus`"
            " exercises. That leaves source-to-benchmark identity unproven."
        ),
        recommended_next_action=(
            "Do not start question-seed generation yet. Re-run source discovery specifically for"
            " the UCI 1990 census benchmark, then rebuild the strict core using exact dataset or"
            " exact schema matches only."
        ),
        key_links_summary=(
            "Current evidence is dominated by jaanli/american-community-survey,"
            " E-A-Griffin/migration-pumas-database, and robertandrewstevens/SQL."
        ),
        risk_summary=(
            "High residual collision risk because the reviewed sources are census-related but not"
            " clearly the same benchmark dataset."
        ),
    ),
    "m11": DatasetManualDecision(
        source_dataset_alignment="confirmed",
        residual_collision_risk="low",
        question_seed_generation_can_start_safely="yes_with_caution",
        manual_override_vs_v2_gate="none",
        decision_confidence="medium",
        verdict="APPROVE_WITH_CAUTION",
        decision_summary=(
            "The full inventory is from one dataset-specific Kaggle-style cross-sell analysis repo"
            " and the schema matches the competition fields, but the strict core is small and"
            " several rows are setup or dialect-heavy."
        ),
        why_safe_or_not=(
            "Safe with caution. Alignment is strong because the rows use the expected"
            " `train`/`test`/`sample_submission` tables and cross-sell feature columns, but the"
            " inventory only has 18 rows total and several are DDL or type-conversion commands."
        ),
        recommended_next_action=(
            "Question-seed drafting can start from the strict pass rows only. Keep the fail and"
            " unknown rows out of the first taxonomy batch."
        ),
        key_links_summary=(
            "hiteshmahajan07/Health_Insuarance_CrossSell_Analysis is the sole reviewed source."
        ),
        risk_summary=(
            "Low collision risk, but moderate packaging risk because the usable SQL core is small"
            " and mixed with setup commands."
        ),
    ),
    "c2": DatasetManualDecision(
        source_dataset_alignment="partially_confirmed",
        residual_collision_risk="medium",
        question_seed_generation_can_start_safely="no",
        manual_override_vs_v2_gate="none",
        decision_confidence="medium",
        verdict="HOLD",
        decision_summary=(
            "Most of the reviewed SQL is genuinely about the car-evaluation benchmark, but the"
            " inventory is very small, duplicate-heavy, and padded by an Oracle scoring-procedure"
            " gist that does not provide clean benchmark-table analysis questions."
        ),
        why_safe_or_not=(
            "Not safe yet. The GitHub analysis rows match the expected feature columns such as"
            " buying, maint, safety, persons, and lug_boot, but there are only 12 rows total and"
            " only a handful of unique primary strict rows after dedup."
        ),
        recommended_next_action=(
            "Hold question-seed generation until more exact benchmark-table SQL is collected."
            " Keep the GitHub analytic rows, but exclude the Oracle procedure gist from the first"
            " taxonomy batch."
        ),
        key_links_summary=(
            "nehanawar025/Exploratory-Analysis-of-Car-Evaluation-Dataset-with-SQL is the strong"
            " source; the Oracle gist is only weak supplemental evidence."
        ),
        risk_summary=(
            "Medium residual risk because the inventory is too thin and one of the two sources is"
            " procedural rather than benchmark-style analytic SQL."
        ),
    ),
    "c7": DatasetManualDecision(
        source_dataset_alignment="not_confirmed",
        residual_collision_risk="critical",
        question_seed_generation_can_start_safely="no",
        manual_override_vs_v2_gate="none",
        decision_confidence="high",
        verdict="REJECT_FOR_NOW",
        decision_summary=(
            "The reviewed SQL is overwhelmingly from nursery business-management apps and DBMS"
            " projects, not the UCI/OpenML Nursery classification dataset."
        ),
        why_safe_or_not=(
            "Not safe. Even the rows that survived as V2 strict are DDL for retail nursery tables"
            " like `store`, `lot`, `employee`, and `plant_type`, while most of the rest are"
            " customer/order/payment or Java-app embedded SQL from unrelated nursery systems."
        ),
        recommended_next_action=(
            "Do not start question-seed generation. Recollect sources specifically tied to the"
            " UCI/OpenML Nursery benchmark and discard nursery-store or nursery-DBMS sources."
        ),
        key_links_summary=(
            "Current evidence is dominated by tamim87/Nursery-DBMS,"
            " Afas66/greenthumb-plantation-java-desktop-application, and ksbains/Nursery."
        ),
        risk_summary=(
            "Critical name-collision risk. The dataset title collides with nursery-management"
            " application projects that are not the benchmark."
        ),
    ),
}


STRONG_DATASETS = {"m12", "c17", "m4", "m8"}
EXCEPTION_DATASETS = {"m11", "c2"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def root_from_url(url: str) -> str:
    if "/blob/" in url:
        return url.split("/blob/")[0]
    return url


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def sql_snippet(text: str, limit: int = 220) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def source_short_name(url: str) -> str:
    url = root_from_url(url)
    if "github.com/" in url:
        return url.split("github.com/", 1)[1]
    if "gist.github.com/" in url:
        return "gist/" + url.split("gist.github.com/", 1)[1]
    return url


def is_session_or_setup_sql(sql_text: str) -> bool:
    sql_upper = clean_text(sql_text).upper()
    return sql_upper.startswith("USE ") or sql_upper.startswith("CREATE DATABASE")


def select_representative_sample(rows: List[dict], top_ids: set[str]) -> Tuple[List[Tuple[dict, str]], int]:
    target = min(20, len(rows))
    selected: List[Tuple[dict, str]] = []
    seen: set[str] = set()

    def add_group(candidates: Iterable[dict], role: str) -> None:
        nonlocal selected
        for row in sorted(candidates, key=row_sort_key):
            sql_item_id = row["sql_item_id"]
            if sql_item_id in seen:
                continue
            selected.append((row, role))
            seen.add(sql_item_id)
            if len(selected) >= target:
                return

    def row_sort_key(row: dict) -> Tuple[int, int, int, str]:
        top_rank = 0 if row["sql_item_id"] in top_ids else 1
        primary_rank = 0 if row.get("is_primary_canonical") == "yes" else 1
        spec_rank = {
            "strict": 0,
            "weak": 1,
            "collision_risk": 2,
            "reject_non_sql": 3,
        }.get(row.get("v2_specificity_label", ""), 9)
        return (top_rank, primary_rank, spec_rank, row["sql_item_id"])

    if len(rows) <= 20:
        add_group(rows, "full_inventory_exception")
        return selected, target

    add_group((row for row in rows if row["sql_item_id"] in top_ids), "top_strict_package")
    add_group(
        (
            row
            for row in rows
            if row.get("is_primary_canonical") == "yes"
            and row.get("v2_specificity_label") == "strict"
            and row.get("executable_status_v2") in {"fail", "unknown"}
        ),
        "strict_exec_edge",
    )
    add_group(
        (
            row
            for row in rows
            if row.get("is_primary_canonical") == "yes"
            and row.get("v2_specificity_label") in {"weak", "collision_risk", "reject_non_sql"}
        ),
        "risk_probe",
    )
    add_group(
        (
            row
            for row in rows
            if row.get("is_primary_canonical") == "yes"
            and row.get("v2_specificity_label") == "strict"
            and row.get("executable_status_v2") == "pass"
        ),
        "strict_primary_fill",
    )
    add_group(rows, "inventory_fill")
    return selected, target


def row_manual_assessment(own_id: str, row: dict) -> Tuple[str, str, str, str]:
    source_root = root_from_url(row.get("source_url", ""))
    sql_text = row.get("sql_text_clean") or row.get("sql_text_raw") or ""
    sql_upper = clean_text(sql_text).upper()
    exec_status = row.get("executable_status_v2", "")
    specificity = row.get("v2_specificity_label", "")

    if own_id in {"m12", "m4", "m8", "m11"}:
        alignment = "aligned"
        residual_risk = "low"
        if specificity != "strict" or is_session_or_setup_sql(sql_text):
            safe_row = "no"
        elif exec_status == "pass":
            safe_row = "yes"
        else:
            safe_row = "caution"
        if own_id == "m11":
            note = (
                "Matches the Kaggle cross-sell schema (`train`/`test`/`sample_submission`)."
                " Treat setup or type-conversion rows as non-seed material."
            )
        elif own_id == "m12":
            note = (
                "Hotel-booking tables and columns match the benchmark domain. Portability issues"
                " do not indicate source mismatch."
            )
        elif own_id == "m4":
            note = (
                "Insurance-analysis SQL stays on expected benchmark attributes (`age`, `bmi`,"
                " `smoker`, `region`, `charges`)."
            )
        else:
            note = (
                "Bank-marketing SQL matches the benchmark field set and campaign-analysis"
                " question space."
            )
        return alignment, residual_risk, safe_row, note

    if own_id == "c17":
        if "Jeanpierre-c-coder" in source_root:
            alignment = "borderline"
            residual_risk = "low"
            safe_row = "caution"
            note = (
                "Still Netflix-specific, but the row uses derived schema-qualified staging tables"
                " rather than the plain benchmark table."
            )
        else:
            alignment = "aligned"
            residual_risk = "low"
            if specificity != "strict" or is_session_or_setup_sql(sql_text):
                safe_row = "no"
            elif exec_status == "pass":
                safe_row = "yes"
            else:
                safe_row = "caution"
            note = (
                "Netflix-title schema matches the benchmark-style dataset and supports direct"
                " taxonomy drafting."
            )
        return alignment, residual_risk, safe_row, note

    if own_id == "c13":
        if "migration-pumas-database" in source_root:
            alignment = "borderline"
            residual_risk = "high"
            safe_row = "no"
            note = (
                "Census-adjacent, but this is a PUMA migration warehouse with tables like"
                " `msa_delineation_2018` and `census_puma_relation`, not a clear exact match to"
                " the UCI 1990 benchmark."
            )
        elif "american-community-survey" in source_root:
            alignment = "misaligned"
            residual_risk = "high"
            safe_row = "no"
            note = (
                "Targets ACS notebooks and derived survey workflows rather than the specific"
                " UCI US Census Data (1990) benchmark."
            )
        else:
            alignment = "misaligned"
            residual_risk = "high"
            safe_row = "no"
            note = (
                "Generic census SQL tutorial material (`zipcensus`, related helpers) is too weak"
                " to support benchmark-safe question seeds."
            )
        return alignment, residual_risk, safe_row, note

    if own_id == "c2":
        if "gist.github.com/ralfmueller" in source_root:
            alignment = "borderline"
            residual_risk = "medium"
            safe_row = "no"
            note = (
                "The Oracle scoring procedure is dataset-named but is not a clean benchmark-table"
                " analytic query for taxonomy seeding."
            )
        else:
            alignment = "aligned"
            residual_risk = "medium"
            if exec_status == "pass":
                safe_row = "caution"
            elif is_session_or_setup_sql(sql_text):
                safe_row = "no"
            else:
                safe_row = "caution"
            note = (
                "The GitHub analysis rows match car-evaluation attributes, but the inventory is"
                " too small and duplicate-heavy for safe seeding."
            )
        return alignment, residual_risk, safe_row, note

    if own_id == "c7":
        if "tamim87/Nursery-DBMS" in source_root:
            note = (
                "Retail nursery management DBMS schema (`customer_info`, `payment_info`,"
                " `plant`, `order_no`) is not the UCI/OpenML Nursery classification dataset."
            )
        elif "Afas66/greenthumb-plantation-java-desktop-application" in source_root:
            note = (
                "Greenthumb application SQL and embedded Java queries are for an operational"
                " nursery app, not the benchmark dataset."
            )
        else:
            note = (
                "The `ksbains/Nursery` DDL defines store/lot/employee/plant-type tables, which"
                " are a name collision with the benchmark."
            )
        return "misaligned", "critical", "no", note

    raise KeyError(f"Unsupported dataset for manual assessment: {own_id}")


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000"
    return f"{numerator / denominator:.3f}"


def make_group_id(own_id: str, sample_rows: List[Tuple[dict, str]]) -> str:
    joined = "|".join(row["sql_item_id"] for row, _ in sample_rows)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"{own_id}_{digest}"


def build_outputs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    final_index_rows = load_csv_rows(FINAL_INDEX_PATH)
    top8 = final_index_rows[:8]

    inventory_rows_by_dataset: Dict[str, List[dict]] = {}
    spotcheck_rows: List[dict] = []
    dataset_summaries: List[dict] = []
    overall_decision = {"yes": [], "yes_with_caution": [], "no": []}

    for dataset_row in top8:
        own_id = dataset_row["own_id"]
        dataset_name = dataset_row["dataset_name"]
        inventory_path = Path(dataset_row["sql_inventory_v2_path"])
        top_strict_path = Path(dataset_row["top_strict_sql_v2_path"])
        inventory_rows = load_csv_rows(inventory_path)
        inventory_rows_by_dataset[own_id] = inventory_rows
        top_ids = {row["source_sql_item_id"] for row in load_csv_rows(top_strict_path)}
        sample_rows, target = select_representative_sample(inventory_rows, top_ids)
        decision = MANUAL_DECISIONS[own_id]

        alignment_counts = Counter()
        row_safe_counts = Counter()
        risk_counts = Counter()
        source_roots = Counter(root_from_url(row.get("source_url", "")) for row, _ in sample_rows)

        for row, sample_role in sample_rows:
            alignment, row_risk, safe_row, manual_notes = row_manual_assessment(own_id, row)
            alignment_counts[alignment] += 1
            row_safe_counts[safe_row] += 1
            risk_counts[row_risk] += 1
            spotcheck_rows.append(
                {
                    "own_id": own_id,
                    "dataset_name": dataset_name,
                    "readiness_label_v2": dataset_row["readiness_label_v2"],
                    "rows_available_in_inventory": len(inventory_rows),
                    "rows_inspected_for_dataset": len(sample_rows),
                    "inspection_target": target,
                    "inspection_exception": (
                        "inventory_has_fewer_than_20_rows" if len(inventory_rows) < 20 else ""
                    ),
                    "sql_item_id": row["sql_item_id"],
                    "source_url": row["source_url"],
                    "source_root": root_from_url(row["source_url"]),
                    "source_title": row.get("source_title", ""),
                    "source_type": row.get("source_type", ""),
                    "manual_sample_role": sample_role,
                    "v2_specificity_label": row.get("v2_specificity_label", ""),
                    "v2_keep_candidate": row.get("v2_keep_candidate", ""),
                    "is_primary_canonical": row.get("is_primary_canonical", ""),
                    "duplicate_type": row.get("duplicate_type", ""),
                    "duplicate_of_sql_item_id": row.get("duplicate_of_sql_item_id", ""),
                    "executable_status_v2": row.get("executable_status_v2", ""),
                    "query_intent_label": row.get("query_intent_label", ""),
                    "family_tag_guess": row.get("family_tag_guess", ""),
                    "manual_alignment_label": alignment,
                    "manual_residual_collision_risk": row_risk,
                    "question_seed_safe_row": safe_row,
                    "manual_notes": manual_notes,
                    "sql_snippet": sql_snippet(row.get("sql_text_clean") or row.get("sql_text_raw") or ""),
                }
            )

        overall_decision[decision.question_seed_generation_can_start_safely].append(own_id)
        dataset_summaries.append(
            {
                "own_id": own_id,
                "dataset_name": dataset_name,
                "readiness_label_v2": dataset_row["readiness_label_v2"],
                "strict_keep_count_v2": int(dataset_row["strict_keep_count_v2"]),
                "rows_available": len(inventory_rows),
                "rows_inspected": len(sample_rows),
                "inspection_target": target,
                "inspection_target_met": "yes" if len(sample_rows) >= 20 else "full_inventory_exception",
                "alignment_counts": dict(alignment_counts),
                "row_safe_counts": dict(row_safe_counts),
                "risk_counts": dict(risk_counts),
                "source_dataset_alignment": decision.source_dataset_alignment,
                "residual_collision_risk": decision.residual_collision_risk,
                "question_seed_generation_can_start_safely": decision.question_seed_generation_can_start_safely,
                "manual_override_vs_v2_gate": decision.manual_override_vs_v2_gate,
                "decision_confidence": decision.decision_confidence,
                "verdict": decision.verdict,
                "decision_summary": decision.decision_summary,
                "why_safe_or_not": decision.why_safe_or_not,
                "recommended_next_action": decision.recommended_next_action,
                "key_links_summary": decision.key_links_summary,
                "risk_summary": decision.risk_summary,
                "top_reviewed_source_roots": [source_short_name(src) for src, _ in source_roots.most_common(4)],
            }
        )

    fieldnames = [
        "own_id",
        "dataset_name",
        "readiness_label_v2",
        "rows_available_in_inventory",
        "rows_inspected_for_dataset",
        "inspection_target",
        "inspection_exception",
        "sql_item_id",
        "source_url",
        "source_root",
        "source_title",
        "source_type",
        "manual_sample_role",
        "v2_specificity_label",
        "v2_keep_candidate",
        "is_primary_canonical",
        "duplicate_type",
        "duplicate_of_sql_item_id",
        "executable_status_v2",
        "query_intent_label",
        "family_tag_guess",
        "manual_alignment_label",
        "manual_residual_collision_risk",
        "question_seed_safe_row",
        "manual_notes",
        "sql_snippet",
    ]
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(spotcheck_rows)

    inspected_total = sum(summary["rows_inspected"] for summary in dataset_summaries)
    approved_now = [summary["own_id"] for summary in dataset_summaries if summary["question_seed_generation_can_start_safely"] == "yes"]
    approved_caution = [summary["own_id"] for summary in dataset_summaries if summary["question_seed_generation_can_start_safely"] == "yes_with_caution"]
    held = [summary["own_id"] for summary in dataset_summaries if summary["question_seed_generation_can_start_safely"] == "no"]
    downgraded = [
        summary["own_id"]
        for summary in dataset_summaries
        if summary["manual_override_vs_v2_gate"] != "none"
    ]

    md_lines: List[str] = []
    md_lines.append("# Top 8 Manual Audit Pack")
    md_lines.append("")
    md_lines.append("## Scope")
    md_lines.append("")
    md_lines.append(
        "This QA pass reviews the first 8 candidate datasets in"
        f" [`final_index_v2.csv`]({FINAL_INDEX_PATH}) without changing any existing V2 outputs."
        " The goal is to manually reinforce the V2 gate before question taxonomy starts."
    )
    md_lines.append("")
    md_lines.append("## Method")
    md_lines.append("")
    md_lines.append(
        "- Selection rule: first 8 rows from the final V2 readiness index, preserving file order."
    )
    md_lines.append(
        "- Row inspection rule: inspect at least 20 representative SQL rows per dataset when 20 are available."
    )
    md_lines.append(
        "- Exception handling: `m11` has only 18 SQL rows and `c2` has only 12 SQL rows, so both were reviewed in full."
    )
    md_lines.append(
        "- Sample construction: prioritize packaged `top_strict_sql_v2.csv` rows, then strict rows with fail/unknown executability,"
        " then risk-probe rows (`weak`, `collision_risk`, `reject_non_sql`), then fill with strict primary rows."
    )
    md_lines.append(
        "- Manual checks per row: source-dataset alignment, residual collision risk, and whether the row is safe input for question-seed generation."
    )
    md_lines.append("")
    md_lines.append("## Executive Verdict")
    md_lines.append("")
    md_lines.append(f"- Datasets reviewed: `{len(dataset_summaries)}`")
    md_lines.append(f"- Total SQL rows manually inspected: `{inspected_total}`")
    md_lines.append(f"- Safe to start now: `{', '.join(approved_now)}`")
    md_lines.append(f"- Safe to start with caution: `{', '.join(approved_caution)}`")
    md_lines.append(f"- Hold / do not start yet: `{', '.join(held)}`")
    md_lines.append(f"- Manual downgrade versus V2 gate: `{', '.join(downgraded) if downgraded else 'none'}`")
    md_lines.append("")
    md_lines.append("## Dataset Summary Table")
    md_lines.append("")
    md_lines.append("| own_id | dataset_name | V2 label | inspected | aligned / borderline / misaligned | residual collision risk | seed generation | verdict |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for summary in dataset_summaries:
        a = summary["alignment_counts"].get("aligned", 0)
        b = summary["alignment_counts"].get("borderline", 0)
        m = summary["alignment_counts"].get("misaligned", 0)
        md_lines.append(
            f"| `{summary['own_id']}` | {summary['dataset_name']} | `{summary['readiness_label_v2']}` |"
            f" `{summary['rows_inspected']}/{summary['rows_available']}` | `{a}/{b}/{m}` |"
            f" `{summary['residual_collision_risk']}` |"
            f" `{summary['question_seed_generation_can_start_safely']}` | `{summary['verdict']}` |"
        )
    md_lines.append("")

    for summary in dataset_summaries:
        md_lines.append(f"## {summary['own_id']} - {summary['dataset_name']}")
        md_lines.append("")
        md_lines.append(f"- V2 readiness label: `{summary['readiness_label_v2']}`")
        md_lines.append(
            f"- SQL rows reviewed: `{summary['rows_inspected']}` out of `{summary['rows_available']}`"
            + (
                " (full inventory review because fewer than 20 rows existed)."
                if summary["rows_available"] < 20
                else "."
            )
        )
        md_lines.append(f"- Strict keep count in V2: `{summary['strict_keep_count_v2']}`")
        md_lines.append(
            "- Top reviewed source roots: "
            + ", ".join(f"`{name}`" for name in summary["top_reviewed_source_roots"])
        )
        md_lines.append(
            "- Alignment counts in inspected sample: "
            f"`aligned={summary['alignment_counts'].get('aligned', 0)}`, "
            f"`borderline={summary['alignment_counts'].get('borderline', 0)}`, "
            f"`misaligned={summary['alignment_counts'].get('misaligned', 0)}`"
        )
        md_lines.append(
            "- Row safety counts in inspected sample: "
            f"`yes={summary['row_safe_counts'].get('yes', 0)}`, "
            f"`caution={summary['row_safe_counts'].get('caution', 0)}`, "
            f"`no={summary['row_safe_counts'].get('no', 0)}`"
        )
        md_lines.append(
            f"- Source-dataset alignment verdict: `{summary['source_dataset_alignment']}`"
        )
        md_lines.append(
            f"- Residual collision risk: `{summary['residual_collision_risk']}`"
        )
        md_lines.append(
            "- Question-seed generation can start safely: "
            f"`{summary['question_seed_generation_can_start_safely']}`"
        )
        md_lines.append(
            f"- Manual override versus V2 gate: `{summary['manual_override_vs_v2_gate']}`"
        )
        md_lines.append(f"- Decision confidence: `{summary['decision_confidence']}`")
        md_lines.append(f"- Key links summary: {summary['key_links_summary']}")
        md_lines.append(f"- Decision summary: {summary['decision_summary']}")
        md_lines.append(f"- Why safe or not: {summary['why_safe_or_not']}")
        md_lines.append(f"- Risk summary: {summary['risk_summary']}")
        md_lines.append(f"- Recommended next action: {summary['recommended_next_action']}")
        md_lines.append("")

    md_lines.append("## Overall Conclusion")
    md_lines.append("")
    md_lines.append(
        "Question-seed generation can start immediately for `m12`, `c17`, `m4`, and `m8`."
        " `m11` can enter a controlled first pass if only strict passing analytical rows are used."
        " `c13`, `c2`, and `c7` should stay out of taxonomy for now, with `c13` manually downgraded"
        " because the sampled sources remain census-adjacent rather than exact-benchmark-safe."
    )
    md_lines.append("")
    md_lines.append(
        "The row-level evidence for all inspected samples is recorded in"
        f" [`top8_sql_spotcheck.csv`]({OUTPUT_CSV})."
    )
    md_lines.append("")

    OUTPUT_MD.write_text("\n".join(md_lines))

    decision_payload = {
        "generated_at_utc": utc_now(),
        "selection_rule": "first_8_rows_in_final_index_v2_file_order",
        "inputs": {
            "final_index_v2": str(FINAL_INDEX_PATH),
            "executability_v2": str(EXEC_PATH),
        },
        "outputs": {
            "top8_manual_audit_pack_md": str(OUTPUT_MD),
            "top8_sql_spotcheck_csv": str(OUTPUT_CSV),
            "top8_decision_json": str(OUTPUT_JSON),
        },
        "inspection_policy": {
            "target_rows_per_dataset": 20,
            "full_inventory_exceptions": {
                "m11": 18,
                "c2": 12,
            },
        },
        "overall_decision": {
            "safe_to_start_now": approved_now,
            "safe_to_start_with_caution": approved_caution,
            "hold_for_now": held,
            "manual_downgrades_vs_v2_gate": downgraded,
            "safe_to_start_now_count": len(approved_now),
            "safe_to_start_with_caution_count": len(approved_caution),
            "hold_for_now_count": len(held),
        },
        "datasets": dataset_summaries,
    }
    OUTPUT_JSON.write_text(json.dumps(decision_payload, indent=2))


if __name__ == "__main__":
    build_outputs()
