#!/usr/bin/env python3
"""Analyze the current core_top10 template strategy and write a research report."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.workload_grounding.adherence import analyze_sql_queries, groupby_arity, structure_flags


TOP10_SPEC_PATH = PROJECT_ROOT / "data" / "workload_grounding" / "agent_candidate_spec_top10_v1.json"
CORE_LIBRARY_PATH = PROJECT_ROOT / "data" / "workload_grounding" / "template_library_v1.jsonl"
PORTABILITY_PATH = PROJECT_ROOT / "data" / "workload_grounding" / "template_portability_report_v1.csv"
OUTPUT_JSON = PROJECT_ROOT / "data" / "workload_grounding" / "top10_research_summary_v1.json"
OUTPUT_MD = PROJECT_ROOT / "docs" / "workload_grounding" / "TOP10_RESEARCH_REPORT.md"
M4_QUERYSET_COMPARISON = (
    PROJECT_ROOT
    / "logs"
    / "runs"
    / "m4_tgset_20260419_000606"
    / "comparison"
    / "against_m4_20260412_011231.json"
)
M4_QUERYSET_INSTANCE_CATALOG = (
    PROJECT_ROOT / "logs" / "runs" / "m4_tgset_20260419_000606" / "template_instance_catalog.json"
)
SMOKE_RUNS = {
    "c2": PROJECT_ROOT / "logs" / "runs" / "c2_tg_20260418_234818",
    "m4": PROJECT_ROOT / "logs" / "runs" / "m4_tg_20260418_234841",
    "n1": PROJECT_ROOT / "logs" / "runs" / "n1_tg_20260418_234841",
}

SECOND_TIER_RECOMMENDATIONS = [
    {
        "template_id": "tpl_m4_two_dimensional_group_avg",
        "why": "补强双轴 subgroup interaction，是 top10 里目前缺失但生产分析里很常见的核心结构。",
    },
    {
        "template_id": "tpl_clickbench_two_dimensional_topk_count",
        "why": "补强 joint heavy-hitter workload，结构简单、生产感强、且 c2/m4 均可自然绑定。",
    },
    {
        "template_id": "tpl_m4_binned_numeric_group_avg",
        "why": "补强 bucketed numeric analytics，能避免对高基数数值列直接 group by。",
    },
    {
        "template_id": "tpl_m4_median_filtered_numeric",
        "why": "补强 robust summary / tail-aware slice，让 production core 不只停留在 mean/sum。",
    },
    {
        "template_id": "tpl_tpcds_within_group_share",
        "why": "补强 share-of-total / contribution 这一类真实 BI 很常见但 top10 当前缺失的结构。",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _portability_score(portable: str) -> float:
    return {"yes": 1.0, "partial": 0.5, "no": 0.0}.get(portable, 0.0)


def _template_portability_rows() -> dict[str, list[dict[str, str]]]:
    rows_by_template: dict[str, list[dict[str, str]]] = {}
    with PORTABILITY_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_template.setdefault(row["template_id"], []).append(row)
    return rows_by_template


def _build_top10_records() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    spec = _load_json(TOP10_SPEC_PATH)
    library = {row["template_id"]: row for row in _load_jsonl(CORE_LIBRARY_PATH)}
    portability = _template_portability_rows()
    records: list[dict[str, Any]] = []
    for item in spec["core_top10"]:
        tid = item["template_id"]
        template = library[tid]
        rows = portability.get(tid, [])
        portability_by_dataset = {row["dataset_id"]: row["portable"] for row in rows}
        partial_or_no = [row for row in rows if row["portable"] != "yes"]
        natural_binding_score = round(
            sum(_portability_score(row["portable"]) for row in rows) / max(1, len(rows)), 4
        )
        flags = structure_flags(template["sql_skeleton"])
        records.append(
            {
                "rank": item["rank"],
                "template_id": tid,
                "template_name": item["template_name"],
                "primary_family": item["primary_family"],
                "secondary_family": item.get("secondary_family"),
                "priority": item["priority"],
                "required_roles": template["required_roles"],
                "portability_by_dataset": portability_by_dataset,
                "natural_binding_score": natural_binding_score,
                "binding_risks": [
                    {
                        "dataset_id": row["dataset_id"],
                        "portable": row["portable"],
                        "review_flag": row["review_flag"],
                        "failure_reason": row["failure_reason"],
                    }
                    for row in partial_or_no
                ],
                "structure_flags": flags,
                "groupby_arity": groupby_arity(template["sql_skeleton"]),
                "why_pick": item["why_pick"],
                "use_when": item["use_when"],
                "avoid_when": item["avoid_when"],
            }
        )
    return records, library


def _compute_overlap(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for i, left in enumerate(records):
        for right in records[i + 1 :]:
            role_score = _jaccard(set(left["required_roles"]), set(right["required_roles"]))
            flag_score = _jaccard(
                {k for k, v in left["structure_flags"].items() if v},
                {k for k, v in right["structure_flags"].items() if v},
            )
            avg_score = round((role_score + flag_score) / 2.0, 4)
            if avg_score < 0.55:
                continue
            overlaps.append(
                {
                    "left_template_id": left["template_id"],
                    "right_template_id": right["template_id"],
                    "role_overlap": round(role_score, 4),
                    "structure_overlap": round(flag_score, 4),
                    "average_overlap": avg_score,
                }
            )
    overlaps.sort(key=lambda item: item["average_overlap"], reverse=True)
    return overlaps


def _analyze_smoke_adherence(library: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id, run_dir in SMOKE_RUNS.items():
        selection = _load_json(run_dir / "grounding" / "selection.json")
        sql = (run_dir / "generated_sql.sql").read_text(encoding="utf-8")
        adherence = analyze_sql_queries(
            sql_queries=[sql],
            template_lookup=library,
            shortlist_ids=[item["template_id"] for item in selection["shortlist"]],
        )
        query = adherence["query_analyses"][0] if adherence["query_analyses"] else {}
        rows.append(
            {
                "dataset_id": dataset_id,
                "run_id": run_dir.name,
                "question": selection["question"],
                "chosen_template_id": query.get("claimed_template_id"),
                "shortlist_ids": [item["template_id"] for item in selection["shortlist"]],
                "comment_match": bool(query.get("claimed_template_id")),
                "expected_flags": query.get("expected_flags", []),
                "observed_flags": query.get("observed_flags", []),
                "adherence_score": query.get("adherence_score", 0.0),
                "notes": query.get("notes", []),
            }
        )
    return rows


def _second_tier_analysis(records: list[dict[str, Any]], library: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    top10_ids = {record["template_id"] for record in records}
    portability = _template_portability_rows()
    pack_instances = _load_json(M4_QUERYSET_INSTANCE_CATALOG).get("instances", [])
    used_in_pack = {row["template_id"] for row in pack_instances if row.get("accepted_local")}
    results: list[dict[str, Any]] = []
    for item in SECOND_TIER_RECOMMENDATIONS:
        tid = item["template_id"]
        template = library[tid]
        rows = portability.get(tid, [])
        portability_by_dataset = {row["dataset_id"]: row["portable"] for row in rows}
        results.append(
            {
                "template_id": tid,
                "template_name": template["template_name"],
                "primary_family": template["primary_family"],
                "secondary_family": template.get("secondary_family"),
                "required_roles": template["required_roles"],
                "portability_by_dataset": portability_by_dataset,
                "used_in_m4_production_pack": tid in used_in_pack,
                "why_promote_next": item["why"],
                "activation_tier": template.get("activation_tier"),
                "dialect_sensitive": bool(template.get("dialect_sensitive")),
            }
        )
    return results


def _coverage_gaps(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    top10_ids = {record["template_id"] for record in records}
    m4_pack_used = {
        row["template_id"]
        for row in _load_json(M4_QUERYSET_INSTANCE_CATALOG).get("instances", [])
        if row.get("accepted_local")
    }
    gaps: list[dict[str, str]] = []
    if "tpl_clickbench_two_dimensional_topk_count" not in top10_ids:
        gaps.append(
            {
                "gap": "two_dimensional_heavy_hitter",
                "why_it_matters": "生产 dashboard 很常见 joint heavy-hitter，但 top10 当前没有明确的二维 count/top-k 模板。",
                "candidate_template": "tpl_clickbench_two_dimensional_topk_count",
            }
        )
    if "tpl_m4_binned_numeric_group_avg" not in top10_ids:
        gaps.append(
            {
                "gap": "bucketed_numeric_analysis",
                "why_it_matters": "真实分析常会先分箱再聚合，避免直接按高基数数值列分组。",
                "candidate_template": "tpl_m4_binned_numeric_group_avg",
            }
        )
    if "tpl_tpcds_within_group_share" not in top10_ids:
        gaps.append(
            {
                "gap": "share_of_total_or_contribution",
                "why_it_matters": "贡献占比是非常常见的业务汇总模式，top10 当前没有显式覆盖。",
                "candidate_template": "tpl_tpcds_within_group_share",
            }
        )
    if "tpl_m4_median_filtered_numeric" not in top10_ids:
        gaps.append(
            {
                "gap": "robust_tail_summary",
                "why_it_matters": "当前 numeric core 以 mean/sum 为主，robust summary 还没进入首批。",
                "candidate_template": "tpl_m4_median_filtered_numeric",
            }
        )
    if "tpl_m4_quantile_tail_slice" not in top10_ids:
        gaps.append(
            {
                "gap": "tail_specific_pack",
                "why_it_matters": "top10 只有一个 tail 模板，长尾与极值监控仍然偏薄。",
                "candidate_template": "tpl_m4_quantile_tail_slice",
            }
        )
    if any(instance for instance in m4_pack_used if instance not in top10_ids):
        gaps.append(
            {
                "gap": "top10_not_yet_sufficient_for_dense_production_pack",
                "why_it_matters": "m4 的生产型 query set 已经自然动用了 5 个第二梯队模板，说明 top10 更像启动集而不是完整核心包。",
                "candidate_template": "see_second_tier_recommendations",
            }
        )
    return gaps


def _verdict(records: list[dict[str, Any]], adherence_rows: list[dict[str, Any]], comparison: dict[str, Any]) -> dict[str, str]:
    avg_binding = sum(record["natural_binding_score"] for record in records) / max(1, len(records))
    avg_adherence = sum(row["adherence_score"] for row in adherence_rows) / max(1, len(adherence_rows))
    production_like = float(comparison["grounded_metrics"]["production_like_query_rate"])
    if avg_binding >= 0.75 and avg_adherence >= 0.95 and production_like >= 0.75:
        phase1 = "top10 适合作为 agent 的 phase-1 production core starter set。"
    else:
        phase1 = "top10 仍需先打磨后再作为稳定的 phase-1 starter set。"
    full_pack = "top10 还不足以单独构成完整的 virtual production workload pack，需要第二梯队补齐二维、分箱、share、robust-tail。"
    return {
        "phase1": phase1,
        "full_pack": full_pack,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    inv = summary["inventory"]
    overlaps = summary["high_overlap_pairs"]
    adherence = summary["smoke_run_adherence"]
    second_tier = summary["recommended_second_tier"]
    gaps = summary["coverage_gaps"]
    comparison = summary["m4_grounded_vs_baseline"]
    verdict = summary["verdict"]

    lines = [
        "# Top10 模板策略研究报告",
        "",
        f"- 生成时间：`{summary['generated_at']}`",
        "",
        "## 结论先行",
        "",
        f"- `{verdict['phase1']}`",
        f"- `{verdict['full_pack']}`",
        f"- `m4` 上 grounded production-like query rate 为 `{comparison['grounded_metrics']['production_like_query_rate']:.3f}`，高于 baseline 的 `{comparison['baseline_metrics']['production_like_query_rate']:.3f}`。",
        "",
        "## Top10 清单与绑定自然性",
        "",
        "| Rank | Template | Family | Binding Score | 备注 |",
        "|---|---|---|---:|---|",
    ]
    for row in sorted(inv, key=lambda item: item["rank"]):
        note = "clean"
        if row["binding_risks"]:
            note = "; ".join(
                f"{item['dataset_id']}:{item['portable']}:{item['failure_reason'] or 'review'}"
                for item in row["binding_risks"]
            )
        lines.append(
            f"| {row['rank']} | `{row['template_id']}` | `{row['primary_family']}` | {row['natural_binding_score']:.2f} | {note} |"
        )

    lines.extend(
        [
            "",
            "## 当前发现的几个关键判断",
            "",
            "- `group_count` 和 `filtered_topk_group_count` 是最稳的 universal anchors，三套数据都能自然绑定。",
            "- `distinct` 系模板在 `c2` 上会退化到 target fallback，这说明它们适合做通用补充，但不一定是所有数据集的首选。",
            "- `group_sum` / `support_guarded_group_avg` / `topn_within_group` 都无法落到 `c2`，说明 numeric-heavy 模板天然偏向 mixed 或 numeric datasets。",
            "- `group_condition_rate` 与 `group_ratio_two_conditions` 在 `n1` 上依赖高基数字段 fallback，说明这两条虽然有代表性，但需要更强 gating。",
            "",
            "## 模板重叠",
            "",
            "| Left | Right | Role Overlap | Structure Overlap | Avg |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in overlaps[:10]:
        lines.append(
            f"| `{item['left_template_id']}` | `{item['right_template_id']}` | {item['role_overlap']:.2f} | {item['structure_overlap']:.2f} | {item['average_overlap']:.2f} |"
        )

    lines.extend(
        [
            "",
            "解释：",
            "- `group_count` / `filtered_topk_group_count` / `distinct topk` / `group_summary_topk` 明显形成一个 heavy-hitter / dashboard family cluster。",
            "- `group_sum` 与 `support_guarded_group_avg` 在 required roles 上接近，但 support guard 让它更像 production-safe 版本，而不是完全重复。",
            "- `group_condition_rate` 与 `group_ratio_two_conditions` 结构接近，但语义不同：前者像 KPI rate，后者像对照比值。",
            "",
            "## 现有 smoke runs 的模板遵循度",
            "",
            "| Dataset | Chosen Template | Comment Match | Adherence Score |",
            "|---|---|---|---:|",
        ]
    )
    for row in adherence:
        lines.append(
            f"| `{row['dataset_id']}` | `{row['chosen_template_id']}` | `{row['comment_match']}` | {row['adherence_score']:.2f} |"
        )

    lines.extend(
        [
            "",
            "解释：",
            "- 三个 smoke runs 的 comment trace 都能正确回收 chosen template id。",
            "- `c2` 与 `m4` 的模板遵循度都很高；但 `n1` 那次运行把 `support_guard` 漂掉并改成了 `LIMIT`，说明当前还需要正式的 adherence checker 来约束结构偏移。",
            "",
            "## m4 生产型 query set 侧证",
            "",
            f"- grounded run: `{summary['m4_grounded_vs_baseline']['grounded_run_id']}`",
            f"- baseline run: `{summary['m4_grounded_vs_baseline']['baseline_run_id']}`",
            f"- grounded `production_like_query_rate = {comparison['grounded_metrics']['production_like_query_rate']:.3f}`",
            f"- baseline `production_like_query_rate = {comparison['baseline_metrics']['production_like_query_rate']:.3f}`",
            f"- grounded `traceable_query_rate = {comparison['grounded_metrics']['traceable_query_rate']:.3f}`",
            f"- baseline `traceable_query_rate = {comparison['baseline_metrics']['traceable_query_rate']:.3f}`",
            "",
            "进一步观察：",
            "- m4 的 production pack 最终用了 12 条模板，其中只有 7 条来自 top10，另外 5 条来自第二梯队。",
            "- 这说明 top10 适合作为 agent 初接入的 starter set，但对“完整生产型 query pack”来说仍然偏薄。",
            "",
            "## 建议优先升级的第二梯队模板",
            "",
            "| Template | Family | Why Promote |",
            "|---|---|---|",
        ]
    )
    for row in second_tier:
        lines.append(
            f"| `{row['template_id']}` | `{row['primary_family']}` | {row['why_promote_next']} |"
        )

    lines.extend(
        [
            "",
            "## 当前缺口",
            "",
        ]
    )
    for row in gaps:
        lines.append(f"- `{row['gap']}`: {row['why_it_matters']} 推荐候选：`{row['candidate_template']}`")

    lines.extend(
        [
            "",
            "## 在扩到 26 条之前建议先做的事",
            "",
            "1. 做 template adherence checker，把 chosen template 与最终 SQL 的结构一致性纳入常规评估。",
            "2. 做 template ranking / gating 研究，特别是 condition/rate 类模板在 `n1` 上的 fallback 风险。",
            "3. 先用 `top10 + second-tier shortlist` 做增量实验，而不是一次性放开全部 26 条。",
            "4. 把评价口径分成两套：`benchmark diversity` 与 `virtual production realism`。",
            "5. 单独补一个 tail / share / bucketed numeric 的小扩展包，再决定是否进入主 candidate pool。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    inventory, library = _build_top10_records()
    overlaps = _compute_overlap(inventory)
    adherence = _analyze_smoke_adherence(library)
    second_tier = _second_tier_analysis(inventory, library)
    coverage_gaps = _coverage_gaps(inventory)
    comparison = _load_json(M4_QUERYSET_COMPARISON)
    verdict = _verdict(inventory, adherence, comparison)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inventory": inventory,
        "high_overlap_pairs": overlaps,
        "smoke_run_adherence": adherence,
        "recommended_second_tier": second_tier,
        "coverage_gaps": coverage_gaps,
        "m4_grounded_vs_baseline": comparison,
        "verdict": verdict,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(_render_markdown(summary), encoding="utf-8")

    print(f"[top10-research] summary={OUTPUT_JSON}")
    print(f"[top10-research] report={OUTPUT_MD}")


if __name__ == "__main__":
    main()
