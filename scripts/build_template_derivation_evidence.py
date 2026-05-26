#!/usr/bin/env python3
"""Build a paper-friendly derivation evidence asset for materialized templates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build template derivation evidence assets.")
    parser.add_argument("--mapping", default="data/workload_grounding/workload_to_family_mapping_v1.csv")
    parser.add_argument("--source-bank", default="data/workload_grounding/source_query_bank_v1.jsonl")
    parser.add_argument("--template-library", default="data/workload_grounding/template_library_v1.jsonl")
    parser.add_argument("--extension-library", default="data/workload_grounding/template_library_extensions_v1.jsonl")
    parser.add_argument("--output-csv", default="data/workload_grounding/template_derivation_evidence_v1.csv")
    parser.add_argument("--output-jsonl", default="data/workload_grounding/template_derivation_evidence_v1.jsonl")
    parser.add_argument("--include-prior-only", action="store_true", help="Include prior-only mappings even when they are not materialized.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--logs-root", default="logs/workload_grounding")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sql_preview(sql_text: str | None, limit: int = 220) -> str:
    text = " ".join((sql_text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _normalize_query_key(raw_value: str | None) -> str:
    return "".join(ch for ch in (raw_value or "").lower() if ch.isalnum())


def _query_ids_match(left: str | None, right: str | None) -> bool:
    left_key = _normalize_query_key(left)
    right_key = _normalize_query_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_digits = "".join(ch for ch in left_key if ch.isdigit())
    right_digits = "".join(ch for ch in right_key if ch.isdigit())
    left_alpha = "".join(ch for ch in left_key if not ch.isdigit()).replace("query", "").replace("q", "")
    right_alpha = "".join(ch for ch in right_key if not ch.isdigit()).replace("query", "").replace("q", "")
    return bool(left_digits) and left_digits == right_digits and left_alpha == right_alpha


def _provenance_matches_source_row(provenance_query_id: str | None, source: dict[str, Any] | None) -> bool:
    if not source:
        return False
    return any(
        _query_ids_match(provenance_query_id, candidate)
        for candidate in [
            source.get("source_query_id"),
            source.get("source_query_label"),
        ]
        if candidate
    )


def _evidence_status(template: dict[str, Any] | None, mapping: dict[str, str], source: dict[str, Any] | None) -> str:
    if template and mapping.get("evidence_url") and source:
        return "complete"
    if template and mapping.get("evidence_url"):
        return "mapping_only"
    if template:
        return "template_only"
    return "prior_only"


def main() -> None:
    args = parse_args()
    mapping_rows = load_mapping(Path(args.mapping))
    source_rows = {
        row["source_query_id"]: row
        for row in load_jsonl(Path(args.source_bank))
        if row.get("source_query_id")
    }
    template_rows = {
        row["template_id"]: row
        for row in load_jsonl(Path(args.template_library)) + load_jsonl(Path(args.extension_library))
        if row.get("template_id")
    }

    csv_rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []

    for mapping in mapping_rows:
        template_id = (mapping.get("template_id") or "").strip()
        template = template_rows.get(template_id)
        if not template_id:
            continue
        if template is None and not args.include_prior_only:
            continue
        source_query_id = (mapping.get("source_query_id") or "").strip()
        source = source_rows.get(source_query_id)
        evidence_status = _evidence_status(template, mapping, source)
        provenance = template.get("provenance", {}) if template else {}
        provenance_sources = template.get("provenance_sources", [provenance] if provenance else []) if template else []
        provenance_query_id = provenance.get("source_query_id", "")
        provenance_matches_source_bank = "yes" if _provenance_matches_source_row(provenance_query_id, source) else "no"

        csv_row = {
            "template_id": template_id,
            "template_name": (template or {}).get("template_name", mapping.get("template_name")),
            "materialization_bucket": (template or {}).get("materialization_bucket", mapping.get("materialization_bucket", "")),
            "activation_tier": (template or {}).get("activation_tier", ""),
            "status": (template or {}).get("status", ""),
            "source_workload_id": (template or {}).get("source_workload_id", mapping.get("workload_id", "")),
            "primary_family": (template or {}).get("primary_family", mapping.get("primary_family", "")),
            "secondary_family": (template or {}).get("secondary_family", mapping.get("secondary_family", "")),
            "template_kind": mapping.get("template_kind", ""),
            "mapping_id": mapping.get("mapping_id", ""),
            "pattern_name": mapping.get("pattern_name", ""),
            "pattern_description": mapping.get("pattern_description", ""),
            "evidence_url": mapping.get("evidence_url", ""),
            "evidence_snippet": mapping.get("evidence_snippet", ""),
            "source_query_id": source_query_id,
            "source_query_label": (source or {}).get("source_query_label", ""),
            "source_title": (source or {}).get("source_title", ""),
            "source_url": (source or {}).get("source_url", ""),
            "source_sql_preview": _sql_preview((source or {}).get("sql_text")),
            "template_provenance_url": provenance.get("url", ""),
            "template_provenance_query_id": provenance_query_id,
            "provenance_source_count": len(provenance_sources),
            "provenance_sources_json": json.dumps(provenance_sources, ensure_ascii=False),
            "template_notes": mapping.get("template_notes", (template or {}).get("notes", "")),
            "portability_notes": mapping.get("portability_notes", ""),
            "confidence": mapping.get("confidence", ""),
            "retrieval_notes": (source or {}).get("retrieval_notes", ""),
            "evidence_chain_status": evidence_status,
            "provenance_matches_source_bank": provenance_matches_source_bank,
        }
        csv_rows.append(csv_row)
        jsonl_rows.append(
            {
                "template": template,
                "mapping": mapping,
                "source_query": source,
                "derivation_summary": {
                    "template_id": template_id,
                    "template_name": csv_row["template_name"],
                    "evidence_chain_status": evidence_status,
                    "provenance_matches_source_bank": csv_row["provenance_matches_source_bank"] == "yes",
                    "provenance_source_count": len(provenance_sources),
                    "abstraction_basis": "source query + mapping evidence + template registry normalization",
                    "notes": [
                        "This asset is intended for paper-writing and audit trails.",
                        "Each row preserves both the original source evidence and the normalized template metadata.",
                    ],
                },
            }
        )

    csv_output_path = Path(args.output_csv)
    jsonl_output_path = Path(args.output_jsonl)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "template_id",
        "template_name",
        "materialization_bucket",
        "activation_tier",
        "status",
        "source_workload_id",
        "primary_family",
        "secondary_family",
        "template_kind",
        "mapping_id",
        "pattern_name",
        "pattern_description",
        "evidence_url",
        "evidence_snippet",
        "source_query_id",
        "source_query_label",
        "source_title",
        "source_url",
        "source_sql_preview",
        "template_provenance_url",
        "template_provenance_query_id",
        "provenance_source_count",
        "provenance_sources_json",
        "template_notes",
        "portability_notes",
        "confidence",
        "retrieval_notes",
        "evidence_chain_status",
        "provenance_matches_source_bank",
    ]
    with csv_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    with jsonl_output_path.open("w", encoding="utf-8") as handle:
        for row in jsonl_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.run_id:
        manifest_path = Path(args.logs_root) / args.run_id / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"run_id": args.run_id}
        manifest.setdefault("outputs", {})["template_derivation_evidence"] = {
            "csv_path": str(csv_output_path.resolve()),
            "jsonl_path": str(jsonl_output_path.resolve()),
            "row_count": len(csv_rows),
            "include_prior_only": args.include_prior_only,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "csv_path": str(csv_output_path.resolve()),
                "jsonl_path": str(jsonl_output_path.resolve()),
                "row_count": len(csv_rows),
                "include_prior_only": args.include_prior_only,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
