"""Core data models for benchmark construction pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FIVE_FIXED_FAMILIES = [
    "subgroup_structure",
    "conditional_dependency_structure",
    "tail_rarity_structure",
    "missingness_structure",
    "cardinality_structure",
]


@dataclass
class StaticDatasetUnderstanding:
    dataset_id: str
    dataset_name: str
    task_type: str
    row_semantics: str
    target_column: str
    target_labels: list[str]
    field_roles: dict[str, str]
    ordered_fields: dict[str, list[str]]
    family_applicability_summary: dict[str, str]
    policy_summary: dict[str, Any]
    risk_summary: list[dict[str, Any]]
    uncertainty_summary: list[dict[str, Any]]
    key_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeResult:
    probe_id: str
    probe_type: str
    description: str
    sql: str
    row_count: int
    columns: list[str]
    rows: list[list[Any]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperationalUnderstanding:
    dataset_id: str
    family_scores: dict[str, float]
    family_priority_order: list[str]
    promising_field_combinations: list[list[str]]
    low_support_signals: list[str]
    triviality_signals: list[str]
    notes: list[str]
    updates_from_validation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FamilyPlan:
    round_index: int
    attempts_by_family: dict[str, int]
    rationale: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchQuestion:
    question_id: str
    family: str
    question: str
    related_fields: list[str]
    target: str
    intent: str
    reason_codes: list[str]
    family_id: str = ""
    intended_facet_id: str = "unknown"
    question_text: str = ""
    target_columns: list[str] = field(default_factory=list)
    related_columns: list[str] = field(default_factory=list)
    rationale: str = ""
    evidence_expectation: str = "unknown"
    comparator_type: str | None = None
    risk_tags: list[str] = field(default_factory=list)
    uncertainty_tags: list[str] = field(default_factory=list)
    stable_question_id: str = ""

    def __post_init__(self) -> None:
        if not self.family_id:
            self.family_id = self.family
        if not self.question_text:
            self.question_text = self.question
        if not self.target_columns:
            self.target_columns = [self.target] if self.target else []
        if not self.related_columns:
            self.related_columns = list(self.related_fields)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuerySpec:
    query_id: str
    family: str
    research_question: str
    claim_type: str
    target_columns: list[str]
    subgroup_columns: list[str]
    feature_columns: list[str]
    expected_output_shape: str
    sql: str
    status: str
    reason_codes: list[str]
    variant_semantic_role: str = ""
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    question_id: str = ""
    family_id: str = ""
    intended_facet_id: str = "unknown"
    variant_id: str = ""
    diversity_intent_tag: str = "unknown"
    intended_structure_claim: str = "unknown"
    source_columns: list[str] = field(default_factory=list)
    expected_result_schema: str = "unknown"
    canonical_sql: str = ""
    canonical_sql_hash: str = ""
    stable_query_id: str = ""
    stable_question_id: str = ""
    secondary_family_candidates: list[str] = field(default_factory=list)
    contamination_risk_hints: list[str] = field(default_factory=list)
    comparator_type: str | None = None
    output_semantics: str = "unknown"
    aggregate_type: str = "unknown"
    measure_column: str = "unknown"
    base_filters: list[str] = field(default_factory=list)
    optional_filters: list[str] = field(default_factory=list)
    groupby_columns: list[str] = field(default_factory=list)
    comparison_target: str = "unknown"
    direction: str = "unknown"
    editable_slots: list[str] = field(default_factory=list)
    frozen_slots: list[str] = field(default_factory=list)
    allowed_refinement_columns: list[str] = field(default_factory=list)
    query_spec_contract_version: str = "query_spec_acr_v1"
    sql_origin_mode: str = "de_novo"
    exemplar_sql_item_id: str = ""
    exemplar_own_id: str = ""
    exemplar_source_url: str = ""
    exemplar_match_score: float = 0.0
    exemplar_transform_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.family_id:
            self.family_id = self.family
        if not self.expected_result_schema:
            self.expected_result_schema = self.expected_output_shape or "unknown"
        if not self.source_columns:
            dedup: list[str] = []
            for column in self.target_columns + self.subgroup_columns + self.feature_columns:
                if column and column not in dedup:
                    dedup.append(column)
            self.source_columns = dedup
        if not self.measure_column:
            self.measure_column = self.target_columns[0] if self.target_columns else "unknown"
        if not self.groupby_columns:
            self.groupby_columns = list(
                dict.fromkeys([col for col in (self.subgroup_columns + self.feature_columns) if col and col != self.measure_column])
            )
        if not self.allowed_refinement_columns:
            self.allowed_refinement_columns = list(
                dict.fromkeys([col for col in self.source_columns if col and col not in self.target_columns])
            )
        if not self.frozen_slots:
            self.frozen_slots = [
                "base_table",
                "join_graph",
                "aggregate_type",
                "measure_column",
                "comparison_entities",
                "direction_semantics",
                "mandatory_filters",
                "family_label",
            ]
        if not self.editable_slots:
            self.editable_slots = ["optional_filter", "threshold_adjacent_bin", "refinement_column", "population_step"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationCategoryResult:
    passed: bool
    reason_codes: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    static_validation: ValidationCategoryResult
    execution_validation: ValidationCategoryResult
    sanity_validation: ValidationCategoryResult
    overall_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "static_validation": self.static_validation.to_dict(),
            "execution_validation": self.execution_validation.to_dict(),
            "sanity_validation": self.sanity_validation.to_dict(),
            "overall_passed": self.overall_passed,
        }


@dataclass
class ExecutionResult:
    ok: bool
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateRecord:
    query_spec: QuerySpec
    validation: ValidationResult
    execution: ExecutionResult
    accepted_local: bool
    rejected_reason_codes: list[str]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_spec": self.query_spec.to_dict(),
            "validation": self.validation.to_dict(),
            "execution": self.execution.to_dict(),
            "accepted_local": self.accepted_local,
            "rejected_reason_codes": self.rejected_reason_codes,
            "provenance": self.provenance,
        }


@dataclass
class QuestionBundleRecord:
    bundle_id: str
    research_question: ResearchQuestion
    family: str
    variants: list[CandidateRecord]
    bundle_validation: ValidationCategoryResult
    accepted_local: bool
    rejected_reason_codes: list[str]
    provenance: dict[str, Any]
    bundle_quality: dict[str, Any] = field(default_factory=dict)

    def accepted_variant_count(self) -> int:
        return sum(1 for item in self.variants if item.accepted_local)

    def accepted_variants(self) -> list[CandidateRecord]:
        return [item for item in self.variants if item.accepted_local]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "research_question": self.research_question.to_dict(),
            "family": self.family,
            "variants": [item.to_dict() for item in self.variants],
            "bundle_validation": self.bundle_validation.to_dict(),
            "bundle_quality": self.bundle_quality,
            "accepted_local": self.accepted_local,
            "rejected_reason_codes": self.rejected_reason_codes,
            "provenance": self.provenance,
            "accepted_variant_count": self.accepted_variant_count(),
        }


@dataclass
class SetCurationResult:
    selected_bundle_ids: list[str]
    family_coverage: dict[str, int]
    notes: list[str]
    rejected_bundle_ids: list[str]
    audit_v2: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
