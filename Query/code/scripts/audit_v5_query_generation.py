#!/usr/bin/env python3
"""Audit versioned query generation outputs against inventories and run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = [
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "c7",
    "c8",
    "c9",
    "c10",
    "c11",
    "c12",
    "c13",
    "c14",
    "c15",
    "c16",
    "c17",
    "c18",
    "c19",
    "c1",
    "m1",
    "m2",
    "m4",
    "m5",
    "m6",
    "m7",
    "m8",
    "m9",
    "m10",
    "m11",
    "m3",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "n10",
    "n11",
    "n12",
    "n14",
    "n15",
    "n16",
    "n17",
    "n18",
    "n19",
    "n13",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def query_id_from_manifest(manifest: dict[str, Any], manifest_path: Path) -> str:
    record = manifest.get("question_record") or {}
    query_id = record.get("query_record_id") or manifest.get("query_record_id")
    if query_id:
        return str(query_id)
    return manifest_path.parent.name


def template_id_from_manifest(manifest: dict[str, Any]) -> str:
    record = manifest.get("question_record") or {}
    return str(record.get("template_id") or manifest.get("template_id") or "")


def dataset_from_manifest_path(path: Path) -> str:
    return path.parts[-4]


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def build_dataset_rows(
    *,
    project_root: Path,
    run_ids: list[str],
    datasets: list[str],
    line_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory_dir = project_root / "data" / f"workload_grounding_{line_version}" / "inventories"
    registry_dir = project_root / "data" / f"workload_grounding_{line_version}" / "registries"
    run_root = project_root / "logs" / f"subitem_workload_{line_version}" / "runs"

    manifests_by_dataset: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for run_id in run_ids:
        for manifest_path in (run_root / run_id).glob("*/artifacts/*/run_manifest.json"):
            try:
                manifest = load_json(manifest_path)
            except Exception as exc:  # pragma: no cover - audit output captures this.
                manifest = {"status": "unreadable", "error": str(exc)}
            manifests_by_dataset[dataset_from_manifest_path(manifest_path)].append((manifest_path, manifest))

    registry_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_id in run_ids:
        registry_path = registry_dir / f"{run_id}_query_registry_{line_version}.jsonl"
        for row in load_jsonl(registry_path):
            registry_by_dataset[str(row.get("dataset_id") or "")].append(row)

    rows: list[dict[str, Any]] = []
    missing_manifest_examples: dict[str, list[str]] = {}
    duplicate_manifest_examples: dict[str, list[str]] = {}
    failed_examples: dict[str, list[str]] = {}

    for dataset_id in datasets:
        inventory_path = inventory_dir / f"{dataset_id}_inventory_{line_version}.json"
        inventory = load_json(inventory_path)
        items = inventory.get("items") or []
        expected_ids = [str(item.get("query_record_id")) for item in items]
        expected_id_set = set(expected_ids)
        expected_templates = {str(item.get("template_id") or "") for item in items}

        manifests = manifests_by_dataset.get(dataset_id, [])
        manifest_ids = [query_id_from_manifest(manifest, path) for path, manifest in manifests]
        manifest_id_counts = Counter(manifest_ids)
        manifest_id_set = set(manifest_ids)
        completed = 0
        failed = 0
        unreadable = 0
        missing_usage = 0
        missing_raw = 0
        missing_sql = 0
        ai_cli_calls = 0
        input_tokens = 0
        cached_input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        manifest_templates: set[str] = set()

        for manifest_path, manifest in manifests:
            status = str(manifest.get("status") or "")
            if status == "completed":
                completed += 1
            elif status == "failed":
                failed += 1
                failed_examples.setdefault(dataset_id, []).append(query_id_from_manifest(manifest, manifest_path))
            elif status == "unreadable":
                unreadable += 1
            usage = manifest.get("usage_summary") or {}
            if not usage:
                missing_usage += 1
            calls = int(usage.get("ai_cli_calls") or 0)
            ai_cli_calls += calls
            input_tokens += int(usage.get("input_tokens") or 0)
            cached_input_tokens += int(usage.get("cached_input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
            manifest_templates.add(template_id_from_manifest(manifest))

            sql_path = manifest.get("generated_sql_path")
            if sql_path and not Path(sql_path).exists():
                missing_sql += 1
            if calls > 0 and not list(manifest_path.parent.glob("cli/*response*.raw.txt")):
                missing_raw += 1

        registry_rows = registry_by_dataset.get(dataset_id, [])
        registry_ids = [str(row.get("query_record_id") or "") for row in registry_rows]
        registry_id_set = set(registry_ids)
        accepted_registry_rows = sum(1 for row in registry_rows if boolish(row.get("accepted_for_eval")))
        registry_templates = {str(row.get("template_id") or "") for row in registry_rows}

        missing_manifest = sorted(expected_id_set - manifest_id_set)
        duplicate_manifest = sorted(query_id for query_id, count in manifest_id_counts.items() if count > 1)
        if missing_manifest:
            missing_manifest_examples[dataset_id] = missing_manifest[:10]
        if duplicate_manifest:
            duplicate_manifest_examples[dataset_id] = duplicate_manifest[:10]

        rows.append(
            {
                "dataset_id": dataset_id,
                "expected_queries": len(expected_ids),
                "inventory_selected_template_count": inventory.get("selected_template_count", len(expected_templates)),
                "inventory_unique_template_count": len(expected_templates),
                "manifest_queries": len(manifests),
                "completed_queries": completed,
                "failed_queries": failed,
                "unreadable_manifests": unreadable,
                "missing_manifest_count": len(missing_manifest),
                "duplicate_manifest_query_count": len(duplicate_manifest),
                "unexpected_manifest_count": len(manifest_id_set - expected_id_set),
                "manifest_unique_template_count": len({tid for tid in manifest_templates if tid}),
                "registry_rows": len(registry_rows),
                "registry_accepted_rows": accepted_registry_rows,
                "registry_unique_query_count": len(registry_id_set),
                "registry_missing_query_count": len(expected_id_set - registry_id_set),
                "registry_unexpected_query_count": len(registry_id_set - expected_id_set),
                "registry_unique_template_count": len({tid for tid in registry_templates if tid}),
                "ai_cli_calls": ai_cli_calls,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "missing_usage_summary_count": missing_usage,
                "missing_raw_response_count": missing_raw,
                "missing_generated_sql_file_count": missing_sql,
                "ok": (
                    len(expected_ids) == len(manifests)
                    and completed == len(expected_ids)
                    and failed == 0
                    and unreadable == 0
                    and len(missing_manifest) == 0
                    and len(duplicate_manifest) == 0
                    and len(manifest_id_set - expected_id_set) == 0
                    and missing_usage == 0
                    and missing_raw == 0
                    and missing_sql == 0
                    and len(expected_templates) == int(inventory.get("selected_template_count", len(expected_templates)))
                ),
            }
        )

    summary = {
        "line_version": line_version,
        "run_ids": run_ids,
        "dataset_count": len(rows),
        "all_ok": all(bool(row["ok"]) for row in rows),
        "expected_queries": sum(int(row["expected_queries"]) for row in rows),
        "manifest_queries": sum(int(row["manifest_queries"]) for row in rows),
        "completed_queries": sum(int(row["completed_queries"]) for row in rows),
        "failed_queries": sum(int(row["failed_queries"]) for row in rows),
        "missing_manifest_count": sum(int(row["missing_manifest_count"]) for row in rows),
        "duplicate_manifest_query_count": sum(int(row["duplicate_manifest_query_count"]) for row in rows),
        "registry_rows": sum(int(row["registry_rows"]) for row in rows),
        "registry_accepted_rows": sum(int(row["registry_accepted_rows"]) for row in rows),
        "ai_cli_calls": sum(int(row["ai_cli_calls"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "cached_input_tokens": sum(int(row["cached_input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "missing_usage_summary_count": sum(int(row["missing_usage_summary_count"]) for row in rows),
        "missing_raw_response_count": sum(int(row["missing_raw_response_count"]) for row in rows),
        "missing_generated_sql_file_count": sum(int(row["missing_generated_sql_file_count"]) for row in rows),
        "missing_manifest_examples": missing_manifest_examples,
        "duplicate_manifest_examples": duplicate_manifest_examples,
        "failed_examples": failed_examples,
    }
    return sorted(rows, key=lambda row: str(row["dataset_id"])), summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    bad_rows = [row for row in rows if not bool(row["ok"])]
    lines = [
        f"# {str(summary['line_version']).upper()} Query Generation Audit",
        "",
        f"- Line version: `{summary['line_version']}`",
        f"- Run IDs: `{'; '.join(summary['run_ids'])}`",
        f"- Datasets: `{summary['dataset_count']}`",
        f"- Expected queries: `{summary['expected_queries']}`",
        f"- Completed manifests: `{summary['completed_queries']}` / `{summary['manifest_queries']}`",
        f"- Failed manifests: `{summary['failed_queries']}`",
        f"- Registry accepted rows: `{summary['registry_accepted_rows']}` / `{summary['registry_rows']}`",
        f"- AI CLI calls: `{summary['ai_cli_calls']}`",
        f"- Total tokens: `{summary['total_tokens']}`",
        f"- Missing usage summaries: `{summary['missing_usage_summary_count']}`",
        f"- Missing raw responses: `{summary['missing_raw_response_count']}`",
        f"- Missing generated SQL files: `{summary['missing_generated_sql_file_count']}`",
        f"- Overall OK: `{summary['all_ok']}`",
        "",
    ]
    if bad_rows:
        lines.extend(["## Datasets Needing Attention", ""])
        for row in bad_rows:
            lines.append(
                "- "
                f"`{row['dataset_id']}`: expected={row['expected_queries']}, "
                f"manifest={row['manifest_queries']}, completed={row['completed_queries']}, "
                f"failed={row['failed_queries']}, missing_manifest={row['missing_manifest_count']}, "
                f"registry_missing={row['registry_missing_query_count']}, "
                f"missing_usage={row['missing_usage_summary_count']}, "
                f"missing_raw={row['missing_raw_response_count']}"
            )
        lines.append("")
    else:
        lines.extend(["All audited datasets passed the manifest/artifact checks.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--line-version", default="v5")
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    rows, summary = build_dataset_rows(
        project_root=project_root,
        run_ids=args.run_ids,
        datasets=args.datasets,
        line_version=args.line_version,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.line_version}_generation_audit"
    write_csv(args.output_dir / f"{prefix}_by_dataset.csv", rows)
    (args.output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "README.md", rows, summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
