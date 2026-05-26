#!/usr/bin/env python3
"""Build a non-invasive shadow plan for template preprocessing assets.

This script does not move or rename any files. It audits current dataset-level
bundle readiness for the template-grounded pipeline and emits a suggested
logical layout for future cleanup.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
WG_ROOT = DATA_ROOT / "workload_grounding"
OUTPUT_DIR = WG_ROOT / "preprocessing_shadow_v1"

EXCLUDED_DATASET_DIRS = {"artifacts", "workload_grounding", "SynData", "splits"}
CORE_BUNDLE_ASSETS = [
    "main_csv",
    "dataset_profile",
    "dataset_contract",
    "dataset_description",
    "dataset_semantics",
    "field_registry",
    "query_policy",
    "validation_policy",
    "source_info",
]
OPTIONAL_ASSETS = [
    "family_applicability",
    "risk_register",
    "uncertainty_register",
]


def _required_asset_candidates(dataset_dir: Path, dataset_id: str) -> dict[str, list[Path]]:
    global_artifacts_dir = DATA_ROOT / "artifacts" / "data_core" / "tabular" / dataset_id
    return {
        "main_csv": [
            dataset_dir / "raw" / f"{dataset_id}-main.csv",
            dataset_dir / f"{dataset_id}-main.csv",
        ],
        "dataset_profile": [
            dataset_dir / "existing_artifacts" / f"{dataset_id}-dataset_profile.json",
            global_artifacts_dir / f"{dataset_id}-dataset_profile.json",
        ],
        "dataset_contract": [
            dataset_dir / "existing_artifacts" / f"{dataset_id}-dataset_contract_v1.json",
            global_artifacts_dir / f"{dataset_id}-dataset_contract_v1.json",
        ],
        "dataset_description": [dataset_dir / "metadata" / "dataset_description.txt"],
        "dataset_semantics": [dataset_dir / "metadata" / "dataset_semantics.yaml"],
        "field_registry": [dataset_dir / "metadata" / "field_registry.json"],
        "query_policy": [dataset_dir / "metadata" / "query_policy.yaml"],
        "validation_policy": [dataset_dir / "metadata" / "validation_policy.yaml"],
        "source_info": [dataset_dir / "source" / "source_info.json"],
        "family_applicability": [dataset_dir / "metadata" / "family_applicability.json"],
        "risk_register": [dataset_dir / "metadata" / "risk_register.json"],
        "uncertainty_register": [dataset_dir / "metadata" / "uncertainty_register.json"],
    }


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _load_portability_coverage(path: Path) -> set[str]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return {str(row.get("dataset_id") or "").strip() for row in rows if str(row.get("dataset_id") or "").strip()}


@dataclass
class DatasetAuditRow:
    dataset_id: str
    core_ready: bool
    strict_loader_ready: bool
    optional_ready_count: int
    portability_covered: bool
    end_to_end_template_ready: bool
    present_assets: list[str]
    missing_assets: list[str]
    missing_core_assets: list[str]
    missing_optional_assets: list[str]
    resolved_paths: dict[str, str]

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "core_ready": self.core_ready,
            "strict_loader_ready": self.strict_loader_ready,
            "optional_ready_count": self.optional_ready_count,
            "portability_covered": self.portability_covered,
            "end_to_end_template_ready": self.end_to_end_template_ready,
            "present_assets": ",".join(self.present_assets),
            "missing_assets": ",".join(self.missing_assets),
            "missing_core_assets": ",".join(self.missing_core_assets),
            "missing_optional_assets": ",".join(self.missing_optional_assets),
        }


def _logical_status(row: DatasetAuditRow) -> str:
    if row.end_to_end_template_ready:
        return "ready_now"
    if row.strict_loader_ready and not row.portability_covered:
        return "needs_portability_generation"
    if not row.core_ready:
        return "needs_metadata_core"
    return "needs_optional_enrichment"


def _audit_datasets() -> tuple[list[DatasetAuditRow], set[str]]:
    portability_covered = _load_portability_coverage(WG_ROOT / "template_portability_report_v1.csv")
    rows: list[DatasetAuditRow] = []

    dataset_dirs = [
        path
        for path in DATA_ROOT.iterdir()
        if path.is_dir() and path.name not in EXCLUDED_DATASET_DIRS and not path.name.startswith(".")
    ]
    for dataset_dir in sorted(dataset_dirs, key=lambda path: path.name):
        dataset_id = dataset_dir.name
        candidates = _required_asset_candidates(dataset_dir, dataset_id)
        resolved_paths: dict[str, str] = {}
        present_assets: list[str] = []
        missing_assets: list[str] = []
        missing_core_assets: list[str] = []
        missing_optional_assets: list[str] = []

        for asset_name, candidate_paths in candidates.items():
            hit = _first_existing(candidate_paths)
            if hit is not None:
                present_assets.append(asset_name)
                resolved_paths[asset_name] = str(hit.resolve())
            else:
                missing_assets.append(asset_name)
                if asset_name in CORE_BUNDLE_ASSETS:
                    missing_core_assets.append(asset_name)
                else:
                    missing_optional_assets.append(asset_name)

        core_ready = not missing_core_assets
        strict_loader_ready = core_ready
        optional_ready_count = len(OPTIONAL_ASSETS) - len(missing_optional_assets)
        covered = dataset_id in portability_covered
        end_to_end_template_ready = strict_loader_ready and covered
        rows.append(
            DatasetAuditRow(
                dataset_id=dataset_id,
                core_ready=core_ready,
                strict_loader_ready=strict_loader_ready,
                optional_ready_count=optional_ready_count,
                portability_covered=covered,
                end_to_end_template_ready=end_to_end_template_ready,
                present_assets=present_assets,
                missing_assets=missing_assets,
                missing_core_assets=missing_core_assets,
                missing_optional_assets=missing_optional_assets,
                resolved_paths=resolved_paths,
            )
        )
    return rows, portability_covered


def _current_to_proposed_mapping() -> list[tuple[str, str, str]]:
    return [
        ("data/workload_grounding/template_library_v1.jsonl", "workload_grounding/library/core/template_library_v1.jsonl", "Core canonical template library"),
        ("data/workload_grounding/template_library_extensions_v1.jsonl", "workload_grounding/library/extensions/template_library_extensions_v1.jsonl", "Extension-only template library"),
        ("data/workload_grounding/template_policy_v1.jsonl", "workload_grounding/policy/template_policy_v1.jsonl", "Per-template can_vary / must_fix policy"),
        ("data/workload_grounding/source_query_bank_v1.jsonl", "workload_grounding/provenance/source_query_bank_v1.jsonl", "Full upstream source-query bank"),
        ("data/workload_grounding/template_derivation_evidence_v1.csv", "workload_grounding/provenance/template_derivation_evidence_v1.csv", "Paper-facing provenance evidence table"),
        ("data/workload_grounding/template_derivation_evidence_v1.jsonl", "workload_grounding/provenance/template_derivation_evidence_v1.jsonl", "Machine-readable provenance evidence"),
        ("data/workload_grounding/workload_catalog.csv", "workload_grounding/provenance/workload_catalog.csv", "Upstream workload catalog"),
        ("data/workload_grounding/workload_to_family_mapping_v1.csv", "workload_grounding/provenance/workload_to_family_mapping_v1.csv", "Workload-to-family mapping"),
        ("data/workload_grounding/template_portability_report_v1.csv", "workload_grounding/portability/global/template_portability_report_v1.csv", "Core portability report"),
        ("data/workload_grounding/template_extension_portability_report_v1.csv", "workload_grounding/portability/extensions/template_extension_portability_report_v1.csv", "Extension portability report"),
        ("data/workload_grounding/agent_candidate_spec_all_core_v1.json", "workload_grounding/runtime_specs/agent_candidate_spec_all_core_v1.json", "Default runtime candidate bucket"),
        ("data/workload_grounding/agent_candidate_spec_top10_v1.json", "workload_grounding/runtime_specs/agent_candidate_spec_top10_v1.json", "Reference top10 bucket"),
        ("data/workload_grounding/agent_candidate_spec_top10_plus5_v1.json", "workload_grounding/runtime_specs/agent_candidate_spec_top10_plus5_v1.json", "Reference top10+5 bucket"),
        ("data/workload_grounding/full_question_inventory_v1/", "workload_grounding/inventories/full_question_inventory_v1/", "Heuristic or earlier inventory build"),
        ("data/workload_grounding/full_question_inventory_v2_policy_gpt54/", "workload_grounding/inventories/full_question_inventory_v2_policy_gpt54/", "Current policy-grounded GPT-5.4 inventory build"),
        ("data/workload_grounding/policyfull54_comparison_summary_v1.json", "workload_grounding/reports/policyfull54_comparison_summary_v1.json", "Inventory comparison summary"),
        ("data/workload_grounding/top10_research_summary_v1.json", "workload_grounding/reports/top10_research_summary_v1.json", "Top10 research summary"),
        ("data/workload_grounding/top10_vs_all_core_summary_v1.json", "workload_grounding/reports/top10_vs_all_core_summary_v1.json", "Top10 vs all-core comparison summary"),
        ("data/workload_grounding/top10_vs_all_core_question_panel_v1.json", "workload_grounding/reports/top10_vs_all_core_question_panel_v1.json", "Per-question comparison panel"),
    ]


def _shadow_tree_text() -> str:
    return "\n".join(
        [
            "data_shadow/",
            "  datasets/",
            "    <dataset_id>/",
            "      raw/",
            "        <dataset_id>-main.csv",
            "      source/",
            "        source_info.json",
            "      metadata_core/",
            "        dataset_description.txt",
            "        dataset_semantics.yaml",
            "        field_registry.json",
            "        query_policy.yaml",
            "        validation_policy.yaml",
            "      metadata_optional/",
            "        family_applicability.json",
            "        risk_register.json",
            "        uncertainty_register.json",
            "      contracts/",
            "        dataset_profile.json",
            "        dataset_contract_v1.json",
            "      cache/",
            "        <dataset_id>.sqlite",
            "  workload_grounding/",
            "    library/",
            "      core/",
            "      extensions/",
            "    policy/",
            "    provenance/",
            "    portability/",
            "      global/",
            "      extensions/",
            "      by_dataset/",
            "    runtime_specs/",
            "    inventories/",
            "    reports/",
            "    preprocessing_shadow_v1/",
        ]
    )


def _write_csv(path: Path, rows: list[DatasetAuditRow]) -> None:
    fieldnames = [
        "dataset_id",
        "core_ready",
        "strict_loader_ready",
        "optional_ready_count",
        "portability_covered",
        "end_to_end_template_ready",
        "present_assets",
        "missing_assets",
        "missing_core_assets",
        "missing_optional_assets",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_flat_dict())


def _write_json(path: Path, rows: list[DatasetAuditRow], portability_covered: set[str]) -> None:
    payload = {
        "summary": {
            "dataset_count": len(rows),
            "strict_loader_ready_count": sum(1 for row in rows if row.strict_loader_ready),
            "portability_covered_count": sum(1 for row in rows if row.portability_covered),
            "end_to_end_template_ready_count": sum(1 for row in rows if row.end_to_end_template_ready),
            "status_buckets": Counter(_logical_status(row) for row in rows),
            "missing_signature_buckets": Counter(
                "|".join(row.missing_core_assets + row.missing_optional_assets) for row in rows
            ),
        },
        "portability_covered_datasets": sorted(portability_covered),
        "datasets": [
            {
                "dataset_id": row.dataset_id,
                "status": _logical_status(row),
                "core_ready": row.core_ready,
                "strict_loader_ready": row.strict_loader_ready,
                "optional_ready_count": row.optional_ready_count,
                "portability_covered": row.portability_covered,
                "end_to_end_template_ready": row.end_to_end_template_ready,
                "present_assets": row.present_assets,
                "missing_assets": row.missing_assets,
                "missing_core_assets": row.missing_core_assets,
                "missing_optional_assets": row.missing_optional_assets,
                "resolved_paths": row.resolved_paths,
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_markdown(path: Path, rows: list[DatasetAuditRow], portability_covered: set[str]) -> None:
    status_buckets = Counter(_logical_status(row) for row in rows)
    missing_signature_buckets: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        signature = ", ".join(row.missing_core_assets + row.missing_optional_assets) or "(none)"
        missing_signature_buckets[signature].append(row.dataset_id)

    ready_now = [row.dataset_id for row in rows if _logical_status(row) == "ready_now"]
    needs_portability = [row.dataset_id for row in rows if _logical_status(row) == "needs_portability_generation"]
    needs_metadata = [row.dataset_id for row in rows if _logical_status(row) == "needs_metadata_core"]

    lines: list[str] = []
    lines.append("# Template Preprocessing Shadow Plan (v1)")
    lines.append("")
    lines.append("This is a non-invasive review artifact. No files were moved or renamed.")
    lines.append("")
    lines.append("## Why This Exists")
    lines.append("")
    lines.append("For the template-grounded pipeline, the real gating sequence is:")
    lines.append("")
    lines.append("1. dataset-level preprocessing assets must exist")
    lines.append("2. portability must then be regenerated on top of those assets")
    lines.append("3. inventories and SQL workloads can then be batch-built")
    lines.append("")
    lines.append("So `all preprocessing data` is the right unblocker, but it is not the final runtime artifact by itself.")
    lines.append("")
    lines.append("## Current Coverage")
    lines.append("")
    lines.append(f"- total dataset folders scanned: `{len(rows)}`")
    lines.append(f"- strict loader ready: `{sum(1 for row in rows if row.strict_loader_ready)}`")
    lines.append(f"- portability covered right now: `{len(portability_covered)}`")
    lines.append(f"- end-to-end template ready right now: `{sum(1 for row in rows if row.end_to_end_template_ready)}`")
    lines.append("")
    lines.append("Status buckets:")
    for status, count in sorted(status_buckets.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.append("")
    lines.append("Ready now:")
    lines.append(f"- `{', '.join(ready_now) if ready_now else 'none'}`")
    lines.append("")
    lines.append("Strict-loader ready but still missing portability generation:")
    lines.append(f"- `{', '.join(needs_portability) if needs_portability else 'none'}`")
    lines.append("")
    lines.append("Missing metadata-core assets:")
    lines.append(f"- `{', '.join(needs_metadata) if needs_metadata else 'none'}`")
    lines.append("")
    lines.append("## Dominant Gap Pattern")
    lines.append("")
    top_signature = max(missing_signature_buckets.items(), key=lambda item: len(item[1]))
    lines.append(
        f"The dominant missing signature is shared by `{len(top_signature[1])}` datasets:"
    )
    lines.append("")
    lines.append(f"- missing: `{top_signature[0]}`")
    lines.append(f"- datasets: `{', '.join(top_signature[1])}`")
    lines.append("")
    lines.append("This means the cleanup is structurally simple: most non-ready datasets are missing the same metadata bundle, not arbitrary one-off files.")
    lines.append("")
    lines.append("## Proposed Logical Layout")
    lines.append("")
    lines.append("Suggested shadow tree:")
    lines.append("")
    lines.append("```text")
    lines.append(_shadow_tree_text())
    lines.append("```")
    lines.append("")
    lines.append("## Current -> Proposed Mapping")
    lines.append("")
    lines.append("| Current | Proposed logical path | Why |")
    lines.append("| --- | --- | --- |")
    for current, proposed, why in _current_to_proposed_mapping():
        lines.append(f"| `{current}` | `{proposed}` | {why} |")
    lines.append("")
    lines.append("## Dataset-Level Recommendation")
    lines.append("")
    lines.append("Keep each `data/<dataset_id>/` folder as the source-of-truth dataset package, but treat it logically as four layers:")
    lines.append("")
    lines.append("1. `raw/`: real table input")
    lines.append("2. `contracts/`: profile + normalized contract")
    lines.append("3. `metadata_core/`: the minimum template-grounding metadata bundle")
    lines.append("4. `metadata_optional/`: family/risk/uncertainty enrichments")
    lines.append("")
    lines.append("This keeps template grounding auditable because source data and preprocessing assets stay attached to the dataset, while workload-grounding outputs stay centralized.")
    lines.append("")
    lines.append("## Next Batchable Step")
    lines.append("")
    lines.append("Once the metadata-core bundle is present for all datasets, the follow-up batch can be:")
    lines.append("")
    lines.append("1. regenerate `template_portability_report` for all datasets")
    lines.append("2. derive per-dataset portability slices")
    lines.append("3. build `all_core` question inventories for all datasets")
    lines.append("4. run the grounded SQL agent from those inventories")
    lines.append("")
    lines.append("## Dataset Audit Table")
    lines.append("")
    lines.append("| Dataset | Status | Missing core assets | Missing optional assets |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| `{row.dataset_id}` | `{_logical_status(row)}` | "
            f"`{', '.join(row.missing_core_assets) if row.missing_core_assets else 'none'}` | "
            f"`{', '.join(row.missing_optional_assets) if row.missing_optional_assets else 'none'}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, portability_covered = _audit_datasets()
    _write_csv(OUTPUT_DIR / "dataset_readiness.csv", rows)
    _write_json(OUTPUT_DIR / "dataset_readiness.json", rows, portability_covered)
    _write_markdown(OUTPUT_DIR / "proposed_shadow_layout.md", rows, portability_covered)
    print(f"[shadow-plan] output_dir={OUTPUT_DIR}")
    print(f"[shadow-plan] dataset_count={len(rows)}")
    print(f"[shadow-plan] strict_loader_ready={sum(1 for row in rows if row.strict_loader_ready)}")
    print(f"[shadow-plan] portability_covered={sum(1 for row in rows if row.portability_covered)}")
    print(f"[shadow-plan] end_to_end_template_ready={sum(1 for row in rows if row.end_to_end_template_ready)}")


if __name__ == "__main__":
    main()
