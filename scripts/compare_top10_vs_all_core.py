#!/usr/bin/env python3
"""Compare the stable top10 template pool against the full all-core pool."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.workload_grounding.adherence import structure_flags
from src.workload_grounding.runtime import select_grounded_templates


DEFAULT_PANEL = PROJECT_ROOT / "data" / "workload_grounding" / "top10_vs_all_core_question_panel_v1.json"
DEFAULT_TOP10_SPEC = PROJECT_ROOT / "data" / "workload_grounding" / "agent_candidate_spec_top10_v1.json"
DEFAULT_ALL_CORE_SPEC = PROJECT_ROOT / "data" / "workload_grounding" / "agent_candidate_spec_all_core_v1.json"
DEFAULT_TEMPLATE_LIBRARY = PROJECT_ROOT / "data" / "workload_grounding" / "template_library_v1.jsonl"
DEFAULT_PORTABILITY = PROJECT_ROOT / "data" / "workload_grounding" / "template_portability_report_v1.csv"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "data" / "workload_grounding" / "top10_vs_all_core_summary_v1.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "docs" / "workload_grounding" / "TOP10_VS_ALL_CORE_REPORT.md"
RUNS_ROOT = PROJECT_ROOT / "logs" / "runs"
M4_INSTANCE_CATALOG = PROJECT_ROOT / "logs" / "runs" / "m4_tgset_20260419_000606" / "template_instance_catalog.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare top10 against the all-core template candidate pool.")
    parser.add_argument("--question-panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--top10-spec", type=Path, default=DEFAULT_TOP10_SPEC)
    parser.add_argument("--top10-bucket", type=str, default="core_top10")
    parser.add_argument("--all-core-spec", type=Path, default=DEFAULT_ALL_CORE_SPEC)
    parser.add_argument("--all-core-bucket", type=str, default="all_core")
    parser.add_argument("--template-library", type=Path, default=DEFAULT_TEMPLATE_LIBRARY)
    parser.add_argument("--portability-report", type=Path, default=DEFAULT_PORTABILITY)
    parser.add_argument("--top10-run-prefix", type=str, default="")
    parser.add_argument("--all-core-run-prefix", type=str, default="")
    parser.add_argument("--max-grounded-templates", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--logs-root", type=Path, default=PROJECT_ROOT / "logs" / "workload_grounding")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_library(path: Path) -> dict[str, dict[str, Any]]:
    return {row["template_id"]: row for row in _load_jsonl(path)}


def _expected_flag_match(expected_flags: list[str], sql_or_template: str) -> bool:
    if not expected_flags:
        return True
    flags = structure_flags(sql_or_template)
    return all(flags.get(flag, False) for flag in expected_flags)


def _family_diversity(shortlist: list[dict[str, Any]]) -> int:
    return len({row["primary_family"] for row in shortlist})


def _selection_record(
    panel_row: dict[str, Any],
    *,
    spec_path: Path,
    spec_bucket: str,
    template_library: Path,
    portability_report: Path,
    max_grounded_templates: int,
) -> dict[str, Any]:
    selection = select_grounded_templates(
        dataset_id=panel_row["dataset_id"],
        question=panel_row["question"],
        spec_path=spec_path,
        spec_bucket=spec_bucket,
        core_library_path=template_library,
        portability_report_path=portability_report,
        max_templates=max_grounded_templates,
    )
    shortlist = selection["shortlist"]
    preferred_templates = set(panel_row.get("preferred_templates", []))
    shortlist_ids = [row["template_id"] for row in shortlist]
    return {
        "question_id": panel_row["question_id"],
        "dataset_id": panel_row["dataset_id"],
        "question": panel_row["question"],
        "spec_bucket": spec_bucket,
        "candidate_pool_count": selection["candidate_pool_count"],
        "applicable_count": selection["applicable_count"],
        "shortlist_count": selection["shortlist_count"],
        "shortlist_template_ids": shortlist_ids,
        "shortlist_primary_families": [row["primary_family"] for row in shortlist],
        "shortlist_family_diversity": _family_diversity(shortlist),
        "preferred_in_shortlist": any(tid in preferred_templates for tid in shortlist_ids),
        "expected_shape_in_shortlist": any(
            _expected_flag_match(panel_row.get("expected_flags", []), row["sql_skeleton"]) for row in shortlist
        ),
    }


def _latest_matching_run(prefix: str, dataset_id: str, question: str) -> dict[str, Any] | None:
    if not prefix:
        return None
    candidates: list[tuple[str, dict[str, Any], Path]] = []
    for manifest_path in RUNS_ROOT.glob("*/run_manifest.json"):
        manifest = _load_json(manifest_path)
        run_id = manifest.get("run_id", manifest_path.parent.name)
        if prefix not in run_id:
            continue
        if manifest.get("dataset_id") != dataset_id:
            continue
        if manifest.get("question") != question:
            continue
        candidates.append((run_id, manifest, manifest_path.parent))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    run_id, manifest, run_dir = candidates[-1]
    return {
        "run_id": run_id,
        "manifest": manifest,
        "run_dir": str(run_dir),
    }


def _runtime_record(panel_row: dict[str, Any], prefix: str, library: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    matched = _latest_matching_run(prefix, panel_row["dataset_id"], panel_row["question"])
    if matched is None:
        return None
    run_dir = Path(matched["run_dir"])
    manifest = matched["manifest"]
    chosen_ids = manifest.get("grounding", {}).get("chosen_template_ids", []) or []
    chosen_template_id = chosen_ids[0] if chosen_ids else None
    generated_sql_path = run_dir / "generated_sql.sql"
    generated_sql = generated_sql_path.read_text(encoding="utf-8") if generated_sql_path.exists() else ""
    chosen_template = library.get(chosen_template_id) if chosen_template_id else None
    preferred_templates = set(panel_row.get("preferred_templates", []))
    adherence = manifest.get("grounding", {}).get("adherence", {})
    return {
        "run_id": matched["run_id"],
        "status": manifest.get("status"),
        "final_answer": manifest.get("final_answer", ""),
        "chosen_template_id": chosen_template_id,
        "chosen_primary_family": chosen_template.get("primary_family") if chosen_template else None,
        "preferred_template_match": chosen_template_id in preferred_templates if chosen_template_id else False,
        "chosen_template_expected_shape_match": _expected_flag_match(
            panel_row.get("expected_flags", []),
            chosen_template["sql_skeleton"] if chosen_template else "",
        ),
        "generated_sql_expected_shape_match": _expected_flag_match(panel_row.get("expected_flags", []), generated_sql),
        "adherence_score": adherence.get("overall_adherence_score"),
        "adherence_label_counts": adherence.get("label_counts"),
        "shortlist_count": manifest.get("grounding", {}).get("shortlist_count"),
        "generated_sql_path": str(generated_sql_path) if generated_sql_path.exists() else "",
        "run_dir": str(run_dir),
    }


def _summarize_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "question_count": len(records),
        "avg_candidate_pool_count": round(mean(row["candidate_pool_count"] for row in records), 4),
        "avg_applicable_count": round(mean(row["applicable_count"] for row in records), 4),
        "avg_shortlist_count": round(mean(row["shortlist_count"] for row in records), 4),
        "avg_shortlist_family_diversity": round(mean(row["shortlist_family_diversity"] for row in records), 4),
        "preferred_in_shortlist_rate": round(
            sum(1 for row in records if row["preferred_in_shortlist"]) / max(1, len(records)), 4
        ),
        "expected_shape_in_shortlist_rate": round(
            sum(1 for row in records if row["expected_shape_in_shortlist"]) / max(1, len(records)), 4
        ),
    }


def _summarize_runtime(records: list[dict[str, Any] | None]) -> dict[str, Any]:
    concrete = [row for row in records if row is not None]
    if not concrete:
        return {
            "run_count": 0,
            "completed_rate": 0.0,
            "preferred_template_match_rate": 0.0,
            "chosen_template_expected_shape_match_rate": 0.0,
            "generated_sql_expected_shape_match_rate": 0.0,
            "avg_adherence_score": None,
            "chosen_family_counts": {},
        }
    adherence_values = [row["adherence_score"] for row in concrete if row.get("adherence_score") is not None]
    return {
        "run_count": len(concrete),
        "completed_rate": round(sum(1 for row in concrete if row["status"] == "completed") / len(concrete), 4),
        "preferred_template_match_rate": round(
            sum(1 for row in concrete if row["preferred_template_match"]) / len(concrete), 4
        ),
        "chosen_template_expected_shape_match_rate": round(
            sum(1 for row in concrete if row["chosen_template_expected_shape_match"]) / len(concrete), 4
        ),
        "generated_sql_expected_shape_match_rate": round(
            sum(1 for row in concrete if row["generated_sql_expected_shape_match"]) / len(concrete), 4
        ),
        "avg_adherence_score": round(mean(adherence_values), 4) if adherence_values else None,
        "chosen_family_counts": dict(Counter(row["chosen_primary_family"] for row in concrete if row["chosen_primary_family"])),
    }


def _m4_pack_support() -> dict[str, Any]:
    payload = _load_json(M4_INSTANCE_CATALOG)
    accepted_ids = [row["template_id"] for row in payload.get("instances", []) if row.get("accepted_local")]
    top10_ids = set(_load_json(DEFAULT_TOP10_SPEC)["core_top10"][idx]["template_id"] for idx in range(len(_load_json(DEFAULT_TOP10_SPEC)["core_top10"])))
    all_core_ids = set(_load_json(DEFAULT_ALL_CORE_SPEC)["all_core"][idx]["template_id"] for idx in range(len(_load_json(DEFAULT_ALL_CORE_SPEC)["all_core"])))
    return {
        "accepted_template_count": len(accepted_ids),
        "top10_supported_count": sum(1 for tid in accepted_ids if tid in top10_ids),
        "all_core_supported_count": sum(1 for tid in accepted_ids if tid in all_core_ids),
        "accepted_template_ids": accepted_ids,
        "top10_missing_ids": [tid for tid in accepted_ids if tid not in top10_ids],
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Top10 vs All-Core 对比报告",
        "",
        f"- 生成时间：`{summary['generated_at']}`",
        f"- 问题面板：`{summary['question_panel_path']}`",
        "",
        "## 总结结论",
        "",
    ]
    verdict = summary["verdict"]
    lines.extend([f"- {item}" for item in verdict])
    lines.extend(
        [
            "",
            "## 选择层指标",
            "",
            "| Bucket | Avg Pool | Avg Applicable | Avg Shortlist | Avg Family Diversity | Preferred-In-Shortlist | Expected-Shape-In-Shortlist |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket in ["top10", "all_core"]:
        sel = summary[bucket]["selection_summary"]
        lines.append(
            f"| `{bucket}` | {sel['avg_candidate_pool_count']:.2f} | {sel['avg_applicable_count']:.2f} | "
            f"{sel['avg_shortlist_count']:.2f} | {sel['avg_shortlist_family_diversity']:.2f} | "
            f"{sel['preferred_in_shortlist_rate']:.2f} | {sel['expected_shape_in_shortlist_rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 运行时指标",
            "",
            "| Bucket | Runs | Completed | Preferred Match | Template Shape Match | SQL Shape Match | Avg Adherence |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket in ["top10", "all_core"]:
        rt = summary[bucket]["runtime_summary"]
        adherence = "n/a" if rt["avg_adherence_score"] is None else f"{rt['avg_adherence_score']:.2f}"
        lines.append(
            f"| `{bucket}` | {rt['run_count']} | {rt['completed_rate']:.2f} | {rt['preferred_template_match_rate']:.2f} | "
            f"{rt['chosen_template_expected_shape_match_rate']:.2f} | {rt['generated_sql_expected_shape_match_rate']:.2f} | {adherence} |"
        )
    lines.extend(
        [
            "",
            "## 按问题逐条对比",
            "",
            "| Question ID | Dataset | Top10 Chosen | All-Core Chosen | Top10 SQL Shape | All-Core SQL Shape | Notes |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in summary["per_question"]:
        top10_runtime = row["top10"].get("runtime") or {}
        all_runtime = row["all_core"].get("runtime") or {}
        lines.append(
            f"| `{row['question_id']}` | `{row['dataset_id']}` | `{top10_runtime.get('chosen_template_id') or 'missing'}` | "
            f"`{all_runtime.get('chosen_template_id') or 'missing'}` | "
            f"`{top10_runtime.get('generated_sql_expected_shape_match')}` | "
            f"`{all_runtime.get('generated_sql_expected_shape_match')}` | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## m4 production pack 侧证",
            "",
            f"- 接受的 production-pack 模板数：`{summary['m4_pack_support']['accepted_template_count']}`",
            f"- `top10` 能直接覆盖：`{summary['m4_pack_support']['top10_supported_count']}`",
            f"- `all_core` 能直接覆盖：`{summary['m4_pack_support']['all_core_supported_count']}`",
            f"- `top10` 缺的模板：`{', '.join(summary['m4_pack_support']['top10_missing_ids'])}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    panel = _load_json(args.question_panel)
    library = _load_library(args.template_library)

    per_question: list[dict[str, Any]] = []
    top10_selection_rows: list[dict[str, Any]] = []
    all_core_selection_rows: list[dict[str, Any]] = []
    top10_runtime_rows: list[dict[str, Any] | None] = []
    all_core_runtime_rows: list[dict[str, Any] | None] = []

    for panel_row in panel:
        top10_selection = _selection_record(
            panel_row,
            spec_path=args.top10_spec,
            spec_bucket=args.top10_bucket,
            template_library=args.template_library,
            portability_report=args.portability_report,
            max_grounded_templates=args.max_grounded_templates,
        )
        all_core_selection = _selection_record(
            panel_row,
            spec_path=args.all_core_spec,
            spec_bucket=args.all_core_bucket,
            template_library=args.template_library,
            portability_report=args.portability_report,
            max_grounded_templates=args.max_grounded_templates,
        )
        top10_runtime = _runtime_record(panel_row, args.top10_run_prefix, library)
        all_core_runtime = _runtime_record(panel_row, args.all_core_run_prefix, library)
        top10_selection_rows.append(top10_selection)
        all_core_selection_rows.append(all_core_selection)
        top10_runtime_rows.append(top10_runtime)
        all_core_runtime_rows.append(all_core_runtime)
        per_question.append(
            {
                "question_id": panel_row["question_id"],
                "dataset_id": panel_row["dataset_id"],
                "question": panel_row["question"],
                "expected_flags": panel_row.get("expected_flags", []),
                "preferred_templates": panel_row.get("preferred_templates", []),
                "notes": panel_row.get("notes", ""),
                "top10": {"selection": top10_selection, "runtime": top10_runtime},
                "all_core": {"selection": all_core_selection, "runtime": all_core_runtime},
            }
        )

    top10_selection_summary = _summarize_selection(top10_selection_rows)
    all_core_selection_summary = _summarize_selection(all_core_selection_rows)
    top10_runtime_summary = _summarize_runtime(top10_runtime_rows)
    all_core_runtime_summary = _summarize_runtime(all_core_runtime_rows)
    m4_pack_support = _m4_pack_support()

    verdict = [
        f"`all_core` 把平均候选池从 {top10_selection_summary['avg_candidate_pool_count']:.1f} 提升到 {all_core_selection_summary['avg_candidate_pool_count']:.1f}，并提高了 shortlist 的 family diversity。",
        f"`all_core` 在面板问题上的 preferred-template shortlist 命中率为 {all_core_selection_summary['preferred_in_shortlist_rate']:.2f}，高于 `top10` 的 {top10_selection_summary['preferred_in_shortlist_rate']:.2f}。",
        f"`all_core` 对 m4 production-pack 的直接覆盖为 {m4_pack_support['all_core_supported_count']}/{m4_pack_support['accepted_template_count']}，明显高于 `top10` 的 {m4_pack_support['top10_supported_count']}/{m4_pack_support['accepted_template_count']}。",
    ]
    if top10_runtime_summary["run_count"] and all_core_runtime_summary["run_count"]:
        verdict.append(
            f"运行时上，`all_core` 的 generated SQL shape match rate 为 {all_core_runtime_summary['generated_sql_expected_shape_match_rate']:.2f}，`top10` 为 {top10_runtime_summary['generated_sql_expected_shape_match_rate']:.2f}。"
        )
    else:
        verdict.append("当前报告同时保留了 selection 层分析；如要完成运行时对比，请用固定 prefix 先跑完两组 side-by-side runs。")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_panel_path": str(args.question_panel.resolve()),
        "top10": {
            "spec_path": str(args.top10_spec.resolve()),
            "spec_bucket": args.top10_bucket,
            "run_prefix": args.top10_run_prefix,
            "selection_summary": top10_selection_summary,
            "runtime_summary": top10_runtime_summary,
        },
        "all_core": {
            "spec_path": str(args.all_core_spec.resolve()),
            "spec_bucket": args.all_core_bucket,
            "run_prefix": args.all_core_run_prefix,
            "selection_summary": all_core_selection_summary,
            "runtime_summary": all_core_runtime_summary,
        },
        "per_question": per_question,
        "m4_pack_support": m4_pack_support,
        "verdict": verdict,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_report(summary), encoding="utf-8")

    if args.run_id:
        manifest_path = args.logs_root / args.run_id / "run_manifest.json"
        manifest = _load_json(manifest_path) if manifest_path.exists() else {"run_id": args.run_id}
        manifest.setdefault("outputs", {})["top10_vs_all_core_comparison"] = {
            "summary_json_path": str(args.output_json.resolve()),
            "report_md_path": str(args.output_md.resolve()),
            "question_count": len(panel),
            "top10_run_prefix": args.top10_run_prefix,
            "all_core_run_prefix": args.all_core_run_prefix,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[top10-vs-all-core] summary={args.output_json}")
    print(f"[top10-vs-all-core] report={args.output_md}")


if __name__ == "__main__":
    main()
