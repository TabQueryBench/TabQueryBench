"""Artifact schema definitions for the isolated v2 workload line."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArtifactNode:
    artifact_key: str
    layer: str
    relative_path: str
    producer: str
    format: str
    description: str


ARTIFACT_SCHEMA_NODES: tuple[ArtifactNode, ...] = (
    ArtifactNode(
        artifact_key="template_library_v2",
        layer="data",
        relative_path="data/workload_grounding_v2/template_library_v2.jsonl",
        producer="build_subitem_workload_v2_template_library.py",
        format="jsonl",
        description="Unified v2 template library including legacy agent templates and new local v2 templates.",
    ),
    ArtifactNode(
        artifact_key="contract_tables",
        layer="data",
        relative_path="data/workload_grounding_v2/contracts/",
        producer="export_contract_tables.py",
        format="csv",
        description="Machine-readable contract tables for template mapping, deterministic rules, and registry fields.",
    ),
    ArtifactNode(
        artifact_key="inventory_v2",
        layer="data",
        relative_path="data/workload_grounding_v2/inventories/{dataset_id}_inventory_v2.json",
        producer="build_subitem_workload_v2_inventory.py",
        format="json",
        description="Subitem-aware workload inventory with explicit family, subitem, facet, role, and bindings.",
    ),
    ArtifactNode(
        artifact_key="query_registry_v2",
        layer="data",
        relative_path="data/workload_grounding_v2/registries/{run_id}_query_registry_v2.jsonl",
        producer="run_subitem_workload_v2.py",
        format="jsonl",
        description="Explicit registry of all realized queries, metadata, execution status, and coverage labels.",
    ),
    ArtifactNode(
        artifact_key="sql_artifacts_v2",
        layer="logs",
        relative_path="logs/subitem_workload_v2/runs/{run_id}/{dataset_id}/sql/",
        producer="run_subitem_workload_v2.py",
        format="sql",
        description="Per-query SQL artifacts with machine-readable metadata comment headers.",
    ),
    ArtifactNode(
        artifact_key="run_artifacts_v2",
        layer="logs",
        relative_path="logs/subitem_workload_v2/runs/{run_id}/{dataset_id}/artifacts/",
        producer="run_subitem_workload_v2.py",
        format="json/txt",
        description="Per-query manifests and local execution artifacts for the v2 line.",
    ),
    ArtifactNode(
        artifact_key="coverage_audit_v2",
        layer="evaluation",
        relative_path="Evaluation/subitem_workload_v2/final/",
        producer="audit_subitem_workload_v2_coverage.py",
        format="csv/json",
        description="Coverage counts, deficits, and summary audit outputs for the v2 registry.",
    ),
    ArtifactNode(
        artifact_key="deficit_round_v2",
        layer="data",
        relative_path="data/workload_grounding_v2/deficit_rounds/{run_id}/",
        producer="plan_subitem_workload_v2_deficit_round.py",
        format="json",
        description="Planned next-round inventories restricted to uncovered dataset x subitem deficits.",
    ),
    ArtifactNode(
        artifact_key="evaluation_exports_v2",
        layer="evaluation",
        relative_path="Evaluation/subitem_workload_v2/final/evaluation_{run_id}/",
        producer="evaluate_subitem_workload_v2_registry.py",
        format="csv/json/md",
        description="Registry-backed evaluation summaries and metadata completeness reports.",
    ),
)


def artifact_schema_rows() -> list[dict[str, str]]:
    return [asdict(node) for node in ARTIFACT_SCHEMA_NODES]
