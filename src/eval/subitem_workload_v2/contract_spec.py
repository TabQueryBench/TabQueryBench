"""Structured contract tables for the v2 subitem workload design.

This module is intentionally configuration-like: the next implementation step
can import these tables directly when wiring the v2 inventory, registry, and
coverage gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class TemplateContract:
    template_id: str
    template_name: str
    family_id: str
    realization_mode: str
    binding_roles: tuple[str, ...]
    supported_canonical_subitem_ids: tuple[str, ...]
    allowed_variant_roles: tuple[str, ...]
    gate_priority: str
    source_catalog: str
    extended_family: bool = False
    notes: str = ""


@dataclass(frozen=True)
class DeterministicEnumerationRule:
    family_id: str
    canonical_subitem_id: str
    driver_template_ids: tuple[str, ...]
    applicability_rule: str
    enumeration_unit: str
    coverage_policy: str
    cap_rule: str
    na_rule: str


@dataclass(frozen=True)
class RegistryFieldSpec:
    field_name: str
    required: bool
    producer: str
    description: str
    notes: str = ""


CORE_AGENT_SUBITEMS: tuple[str, ...] = (
    "internal_profile_stability",
    "subgroup_size_stability",
    "dependency_strength_similarity",
    "direction_consistency",
    "slice_level_consistency",
    "tail_set_consistency",
    "tail_mass_similarity",
    "tail_concentration_consistency",
)

DETERMINISTIC_SUBITEMS: tuple[str, ...] = (
    "marginal_missing_rate_consistency",
    "co_missingness_pattern_consistency",
    "support_rank_profile_consistency",
    "high_cardinality_response_stability",
)

SUBITEM_DEFAULT_FACETS: dict[str, tuple[str, ...]] = {
    "internal_profile_stability": (
        "subgroup_distribution_shift",
        "subgroup_rank_order",
        "subgroup_conditional_contrast",
    ),
    "subgroup_size_stability": ("subgroup_distribution_shift",),
    "dependency_strength_similarity": ("pairwise_conditional_dependency",),
    "direction_consistency": ("conditional_rate_shift",),
    "slice_level_consistency": ("conditional_interaction_hotspots",),
    "tail_set_consistency": ("low_support_extremes",),
    "tail_mass_similarity": ("tail_ranked_signal",),
    "tail_concentration_consistency": ("rare_target_concentration",),
    "marginal_missing_rate_consistency": ("missing_indicator_distribution",),
    "co_missingness_pattern_consistency": (
        "missing_rate_by_subgroup",
        "missing_target_interaction",
    ),
    "support_rank_profile_consistency": (
        "support_concentration",
        "value_imbalance_profile",
    ),
    "high_cardinality_response_stability": (
        "target_cardinality_cross_section",
    ),
}

SUBITEM_TO_FAMILY: dict[str, str] = {
    "internal_profile_stability": "subgroup_structure",
    "subgroup_size_stability": "subgroup_structure",
    "dependency_strength_similarity": "conditional_dependency_structure",
    "direction_consistency": "conditional_dependency_structure",
    "slice_level_consistency": "conditional_dependency_structure",
    "tail_set_consistency": "tail_rarity_structure",
    "tail_mass_similarity": "tail_rarity_structure",
    "tail_concentration_consistency": "tail_rarity_structure",
    "marginal_missing_rate_consistency": "missingness_structure",
    "co_missingness_pattern_consistency": "missingness_structure",
    "support_rank_profile_consistency": "cardinality_structure",
    "high_cardinality_response_stability": "cardinality_structure",
}


TEMPLATE_CONTRACTS: tuple[TemplateContract, ...] = (
    TemplateContract(
        template_id="tpl_c2_filtered_group_count_2d",
        template_name="Filtered Two-Dimensional Group Count",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("group_col", "group_col_2", "predicate_col"),
        supported_canonical_subitem_ids=("slice_level_consistency",),
        allowed_variant_roles=("count_distribution",),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_c2_two_dim_target_rate",
        template_name="Two-Axis Target Rate Surface",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("group_col", "group_col_2", "target_col"),
        supported_canonical_subitem_ids=(
            "dependency_strength_similarity",
            "direction_consistency",
        ),
        allowed_variant_roles=("within_group_proportion", "ranked_signal_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_conditional_group_quantiles",
        template_name="Conditional Group Quantiles",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col", "condition_col"),
        supported_canonical_subitem_ids=(
            "slice_level_consistency",
            "direction_consistency",
        ),
        allowed_variant_roles=("focused_target_view", "filtered_stable_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_binned_numeric_group_avg",
        template_name="Binned Numeric Group Average",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("band_col", "measure_col"),
        supported_canonical_subitem_ids=("slice_level_consistency",),
        allowed_variant_roles=("collapsed_target_view", "filtered_stable_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_group_condition_rate",
        template_name="Grouped Condition Rate",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("group_col", "condition_col"),
        supported_canonical_subitem_ids=(
            "dependency_strength_similarity",
            "direction_consistency",
        ),
        allowed_variant_roles=("within_group_proportion", "focused_target_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_group_dispersion_rank",
        template_name="Grouped Dispersion Rank",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=("dependency_strength_similarity",),
        allowed_variant_roles=("ranked_signal_view", "focused_target_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_group_ratio_two_conditions",
        template_name="Grouped Ratio of Two Conditions",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("group_col", "condition_col"),
        supported_canonical_subitem_ids=("direction_consistency",),
        allowed_variant_roles=("contrastive_conditional_view",),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_median_filtered_numeric",
        template_name="Filtered Median Numeric Slice",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("measure_col", "predicate_col"),
        supported_canonical_subitem_ids=("slice_level_consistency",),
        allowed_variant_roles=("focused_target_view", "filtered_stable_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_window_partition_avg",
        template_name="Window Partition Average",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=(
            "slice_level_consistency",
            "direction_consistency",
        ),
        allowed_variant_roles=("filtered_stable_view", "ranked_signal_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_rtabench_time_bucket_filtered_count",
        template_name="Time-Bucket Filtered Count",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("time_col", "predicate_col"),
        supported_canonical_subitem_ids=("slice_level_consistency",),
        allowed_variant_roles=("filtered_stable_view", "count_distribution"),
        gate_priority="support",
        source_catalog="template_library_extensions_v1",
    ),
    TemplateContract(
        template_id="tpl_rtabench_time_bucket_group_moving_avg",
        template_name="Time-Bucket Group Moving Average",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("time_col", "group_col", "predicate_col"),
        supported_canonical_subitem_ids=(
            "slice_level_consistency",
            "direction_consistency",
        ),
        allowed_variant_roles=("filtered_stable_view", "ranked_signal_view"),
        gate_priority="support",
        source_catalog="template_library_extensions_v1",
    ),
    TemplateContract(
        template_id="tpl_tail_drift_ratio",
        template_name="Tail Drift Ratio",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("group_col", "time_col"),
        supported_canonical_subitem_ids=("direction_consistency",),
        allowed_variant_roles=(
            "contrastive_conditional_view",
            "ranked_signal_view",
        ),
        gate_priority="review",
        source_catalog="template_library_extensions_v1",
        notes="Keep in v2 but re-check facet binding before allowing it into gate.",
    ),
    TemplateContract(
        template_id="tpl_tpcds_baseline_gated_extreme_ranking",
        template_name="Baseline-Gated Extreme Ranking",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("group_col", "item_col", "measure_col"),
        supported_canonical_subitem_ids=("dependency_strength_similarity",),
        allowed_variant_roles=("ranked_signal_view", "focused_target_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_tpcds_within_group_share",
        template_name="Within-Group Share of Total",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("group_col", "item_col", "measure_col"),
        supported_canonical_subitem_ids=("dependency_strength_similarity",),
        allowed_variant_roles=("within_group_proportion", "focused_target_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_tpch_filtered_sum_band",
        template_name="Filtered Sum in Numeric Band",
        family_id="conditional_dependency_structure",
        realization_mode="agent",
        binding_roles=("measure_col", "band_col"),
        supported_canonical_subitem_ids=("slice_level_consistency",),
        allowed_variant_roles=("filtered_stable_view", "collapsed_target_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_clickbench_filtered_distinct_topk",
        template_name="Filtered Top-k Distinct Coverage",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "entity_col", "predicate_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("filtered_stable_view", "ranked_signal_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_clickbench_filtered_topk_group_count",
        template_name="Filtered Top-k Group Count",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "predicate_col"),
        supported_canonical_subitem_ids=("subgroup_size_stability",),
        allowed_variant_roles=("count_distribution", "filtered_stable_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_clickbench_group_count",
        template_name="Grouped Count by Category",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col",),
        supported_canonical_subitem_ids=("subgroup_size_stability",),
        allowed_variant_roles=("count_distribution",),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_clickbench_group_distinct_topk",
        template_name="Top-k Groups by Distinct Entity Coverage",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "entity_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("ranked_signal_view",),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_clickbench_group_summary_topk",
        template_name="Grouped Summary Top-k",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col", "entity_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("ranked_signal_view", "collapsed_target_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_clickbench_two_dimensional_topk_count",
        template_name="Two-Dimensional Top-k Count",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "group_col_2"),
        supported_canonical_subitem_ids=("subgroup_size_stability",),
        allowed_variant_roles=("count_distribution",),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_h2o_group_sum",
        template_name="Grouped Numeric Sum",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("collapsed_target_view",),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_h2o_two_dimensional_group_sum",
        template_name="Two-Dimensional Group Sum",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "group_col_2", "measure_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("collapsed_target_view",),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_h2o_two_dimensional_robust_summary",
        template_name="Two-Dimensional Robust Summary",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "group_col_2", "measure_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("collapsed_target_view", "filtered_stable_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_group_avg_numeric",
        template_name="Grouped Numeric Mean",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("collapsed_target_view",),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_support_guarded_group_avg",
        template_name="Support-Guarded Group Average",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("collapsed_target_view", "filtered_stable_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_two_dimensional_group_avg",
        template_name="Two-Dimensional Group Average",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "group_col_2", "measure_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("collapsed_target_view",),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_tail_weighted_topk_sum",
        template_name="Weighted Top-k Sum",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("ranked_signal_view", "filtered_stable_view"),
        gate_priority="review",
        source_catalog="template_library_v1",
        notes="Template name is tail-flavored but current family stays subgroup until rebind review.",
    ),
    TemplateContract(
        template_id="tpl_tpcds_topk_group_sum",
        template_name="Top-k Group Sum with Filter",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col", "predicate_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("ranked_signal_view", "filtered_stable_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_tpch_max_aggregate_winner",
        template_name="Max Aggregate Winner Selection",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("focused_target_view", "ranked_signal_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_tpch_two_dimensional_summary",
        template_name="Two-Dimensional Summary with Filter",
        family_id="subgroup_structure",
        realization_mode="agent",
        binding_roles=("group_col", "group_col_2", "measure_col", "predicate_col"),
        supported_canonical_subitem_ids=("internal_profile_stability",),
        allowed_variant_roles=("collapsed_target_view", "filtered_stable_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_tail_low_support_group_count_v2",
        template_name="Low-Support Group Count",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("group_col",),
        supported_canonical_subitem_ids=(
            "tail_set_consistency",
            "tail_mass_similarity",
        ),
        allowed_variant_roles=("rare_extreme_view", "count_distribution"),
        gate_priority="primary",
        source_catalog="template_library_v2",
        notes="New v2 agent template for count-based tail coverage on non-numeric datasets.",
    ),
    TemplateContract(
        template_id="tpl_tail_pairwise_sparse_slice_v2",
        template_name="Pairwise Sparse Slice Count",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("group_col", "group_col_2"),
        supported_canonical_subitem_ids=(
            "tail_set_consistency",
            "tail_mass_similarity",
        ),
        allowed_variant_roles=("rare_extreme_view", "filtered_stable_view"),
        gate_priority="support",
        source_catalog="template_library_v2",
        notes="New v2 agent template for sparse pairwise tail slices.",
    ),
    TemplateContract(
        template_id="tpl_tail_target_rate_extremes_v2",
        template_name="Tail Target-Rate Extremes",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("group_col", "target_col"),
        supported_canonical_subitem_ids=("tail_concentration_consistency",),
        allowed_variant_roles=("focused_target_view", "ranked_signal_view"),
        gate_priority="primary",
        source_catalog="template_library_v2",
        notes="New v2 agent template for classification-style tail concentration coverage.",
    ),
    TemplateContract(
        template_id="tpl_grouped_percentile_point",
        template_name="Grouped Percentile Point",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=("tail_concentration_consistency",),
        allowed_variant_roles=("focused_target_view", "ranked_signal_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_h2o_topn_within_group",
        template_name="Top-N Within Group by Measure",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=(
            "tail_set_consistency",
            "tail_mass_similarity",
        ),
        allowed_variant_roles=("rare_extreme_view", "ranked_signal_view"),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_global_zscore_outliers",
        template_name="Global Z-score Outlier Scan",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("measure_col",),
        supported_canonical_subitem_ids=("tail_set_consistency",),
        allowed_variant_roles=("rare_extreme_view",),
        gate_priority="support",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_m4_quantile_tail_slice",
        template_name="Quantile Tail Slice",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("measure_col",),
        supported_canonical_subitem_ids=("tail_set_consistency",),
        allowed_variant_roles=("rare_extreme_view",),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_threshold_rarity_cdf",
        template_name="Threshold Rarity CDF",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("measure_col",),
        supported_canonical_subitem_ids=("tail_set_consistency",),
        allowed_variant_roles=("rare_extreme_view",),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_tpcds_subgroup_baseline_outlier",
        template_name="Subgroup Baseline Outlier",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("group_col", "item_col", "measure_col"),
        supported_canonical_subitem_ids=("tail_concentration_consistency",),
        allowed_variant_roles=(
            "focused_target_view",
            "contrastive_conditional_view",
        ),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_tpch_relative_total_threshold",
        template_name="Relative-to-Total Extreme Threshold",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=("tail_mass_similarity",),
        allowed_variant_roles=("count_distribution", "filtered_stable_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_tpch_thresholded_group_ranking",
        template_name="Thresholded Group Ranking",
        family_id="tail_rarity_structure",
        realization_mode="agent",
        binding_roles=("group_col", "measure_col"),
        supported_canonical_subitem_ids=(
            "tail_set_consistency",
            "tail_mass_similarity",
        ),
        allowed_variant_roles=("rare_extreme_view", "filtered_stable_view"),
        gate_priority="primary",
        source_catalog="template_library_v1",
    ),
    TemplateContract(
        template_id="tpl_missing_marginal_rate_profile",
        template_name="Marginal Missing Rate Profile",
        family_id="missingness_structure",
        realization_mode="deterministic",
        binding_roles=("missing_col",),
        supported_canonical_subitem_ids=(
            "marginal_missing_rate_consistency",
        ),
        allowed_variant_roles=("missing_indicator_view",),
        gate_priority="deterministic",
        source_catalog="template_library_v2",
        notes="New deterministic template for v2.",
    ),
    TemplateContract(
        template_id="tpl_missing_rate_by_subgroup",
        template_name="Missing Rate by Subgroup",
        family_id="missingness_structure",
        realization_mode="deterministic",
        binding_roles=("missing_col", "group_col"),
        supported_canonical_subitem_ids=(
            "co_missingness_pattern_consistency",
        ),
        allowed_variant_roles=("missing_rate_by_subgroup",),
        gate_priority="deterministic",
        source_catalog="template_library_v2",
        notes="New deterministic template for v2.",
    ),
    TemplateContract(
        template_id="tpl_missing_target_interaction",
        template_name="Missingness-Target Interaction",
        family_id="missingness_structure",
        realization_mode="deterministic",
        binding_roles=("missing_col", "target_col"),
        supported_canonical_subitem_ids=(
            "co_missingness_pattern_consistency",
        ),
        allowed_variant_roles=("missing_target_interaction",),
        gate_priority="deterministic",
        source_catalog="template_library_v2",
        notes="New deterministic template for v2.",
    ),
    TemplateContract(
        template_id="tpl_cardinality_support_rank_profile",
        template_name="Cardinality Support Rank Profile",
        family_id="cardinality_structure",
        realization_mode="deterministic",
        binding_roles=("group_col",),
        supported_canonical_subitem_ids=(
            "support_rank_profile_consistency",
        ),
        allowed_variant_roles=("count_distribution",),
        gate_priority="deterministic",
        source_catalog="template_library_v2",
        extended_family=True,
        notes="New deterministic template for v2.",
    ),
    TemplateContract(
        template_id="tpl_cardinality_distinct_share_profile",
        template_name="Cardinality Distinct Share Profile",
        family_id="cardinality_structure",
        realization_mode="deterministic",
        binding_roles=("group_col",),
        supported_canonical_subitem_ids=(
            "support_rank_profile_consistency",
        ),
        allowed_variant_roles=("ranked_signal_view",),
        gate_priority="deterministic",
        source_catalog="template_library_v2",
        extended_family=True,
        notes="New deterministic template for v2.",
    ),
    TemplateContract(
        template_id="tpl_cardinality_high_card_response_stability",
        template_name="High-Cardinality Response Stability",
        family_id="cardinality_structure",
        realization_mode="deterministic",
        binding_roles=("key_col", "target_col"),
        supported_canonical_subitem_ids=(
            "high_cardinality_response_stability",
        ),
        allowed_variant_roles=("focused_target_view",),
        gate_priority="deterministic",
        source_catalog="template_library_v2",
        extended_family=True,
        notes="New deterministic template for v2.",
    ),
)


DETERMINISTIC_ENUMERATION_RULES: tuple[DeterministicEnumerationRule, ...] = (
    DeterministicEnumerationRule(
        family_id="missingness_structure",
        canonical_subitem_id="marginal_missing_rate_consistency",
        driver_template_ids=("tpl_missing_marginal_rate_profile",),
        applicability_rule="Dataset exposes at least one native missing column.",
        enumeration_unit="One missing_col produces one query.",
        coverage_policy="enumerate_all_applicable",
        cap_rule="No cap beyond the set of native missing columns.",
        na_rule="Mark family NA when the dataset has zero native missing columns.",
    ),
    DeterministicEnumerationRule(
        family_id="missingness_structure",
        canonical_subitem_id="co_missingness_pattern_consistency",
        driver_template_ids=(
            "tpl_missing_rate_by_subgroup",
            "tpl_missing_target_interaction",
        ),
        applicability_rule=(
            "Dataset exposes at least one missing_col and at least one usable "
            "group_col, target_col, or condition_col."
        ),
        enumeration_unit=(
            "Enumerate missing_col x group_col and missing_col x target_or_condition."
        ),
        coverage_policy="enumerate_all_applicable",
        cap_rule="For each missing_col, keep at most the top 10 conditioning fields.",
        na_rule="Mark family NA when no valid missingness pairings can be formed.",
    ),
    DeterministicEnumerationRule(
        family_id="cardinality_structure",
        canonical_subitem_id="support_rank_profile_consistency",
        driver_template_ids=(
            "tpl_cardinality_support_rank_profile",
            "tpl_cardinality_distinct_share_profile",
        ),
        applicability_rule=(
            "Dataset exposes at least one eligible groupable categorical column."
        ),
        enumeration_unit="One eligible groupable column produces two queries.",
        coverage_policy="enumerate_all_applicable",
        cap_rule="No cap beyond the eligible groupable columns list.",
        na_rule="Mark family NA when no eligible groupable column exists.",
    ),
    DeterministicEnumerationRule(
        family_id="cardinality_structure",
        canonical_subitem_id="high_cardinality_response_stability",
        driver_template_ids=("tpl_cardinality_high_card_response_stability",),
        applicability_rule=(
            "Dataset exposes at least one high-cardinality key_col and one target "
            "or numeric response column."
        ),
        enumeration_unit="Enumerate high-card key_col x response_col pairs.",
        coverage_policy="enumerate_all_applicable",
        cap_rule="For each dataset, keep at most the top 20 high-cardinality keys.",
        na_rule="Mark family NA when no valid key-response pair exists.",
    ),
)


QUERY_REGISTRY_FIELDS: tuple[RegistryFieldSpec, ...] = (
    RegistryFieldSpec(
        field_name="registry_version",
        required=True,
        producer="registry_writer",
        description="Schema stamp for the v2 registry.",
        notes="Fixed value: query_registry_v2.",
    ),
    RegistryFieldSpec(
        field_name="dataset_id",
        required=True,
        producer="planner",
        description="Dataset identifier.",
    ),
    RegistryFieldSpec(
        field_name="round_id",
        required=True,
        producer="coverage_loop",
        description="Coverage round that produced the query.",
    ),
    RegistryFieldSpec(
        field_name="query_record_id",
        required=True,
        producer="planner",
        description="Stable global query identifier.",
    ),
    RegistryFieldSpec(
        field_name="problem_id",
        required=True,
        producer="planner",
        description="Stable problem instance identifier.",
    ),
    RegistryFieldSpec(
        field_name="source_kind",
        required=True,
        producer="planner",
        description="Origin class: agent or deterministic.",
    ),
    RegistryFieldSpec(
        field_name="realization_mode",
        required=True,
        producer="planner",
        description="Execution path: agent or deterministic.",
    ),
    RegistryFieldSpec(
        field_name="template_id",
        required=True,
        producer="planner",
        description="Template contract identifier.",
    ),
    RegistryFieldSpec(
        field_name="generator_id",
        required=False,
        producer="deterministic_generator",
        description="Optional deterministic generator identifier.",
    ),
    RegistryFieldSpec(
        field_name="family_id",
        required=True,
        producer="planner",
        description="Canonical family label.",
    ),
    RegistryFieldSpec(
        field_name="canonical_subitem_id",
        required=True,
        producer="planner",
        description="Canonical subitem label.",
    ),
    RegistryFieldSpec(
        field_name="intended_facet_id",
        required=True,
        producer="planner",
        description="Facet label used during problem planning.",
    ),
    RegistryFieldSpec(
        field_name="variant_semantic_role",
        required=True,
        producer="planner",
        description="Semantic role label used during problem planning.",
    ),
    RegistryFieldSpec(
        field_name="subitem_assignment_source",
        required=True,
        producer="planner",
        description="Assignment mode: planner_selected or template_fixed.",
    ),
    RegistryFieldSpec(
        field_name="extended_family",
        required=True,
        producer="planner",
        description="Whether the query belongs to an extended family outside the core-10 contract.",
    ),
    RegistryFieldSpec(
        field_name="question_text",
        required=False,
        producer="planner",
        description="Human-readable question text.",
        notes="Deterministic queries may store a normalized description instead.",
    ),
    RegistryFieldSpec(
        field_name="sql_path",
        required=True,
        producer="runner",
        description="Filesystem path to the realized SQL artifact.",
    ),
    RegistryFieldSpec(
        field_name="sql_sha256",
        required=True,
        producer="runner",
        description="Content hash of the SQL artifact.",
    ),
    RegistryFieldSpec(
        field_name="exec_ok_real",
        required=True,
        producer="runner",
        description="Whether the SQL executed successfully on the real dataset.",
    ),
    RegistryFieldSpec(
        field_name="accepted_for_eval",
        required=True,
        producer="coverage_gate",
        description="Whether the query is admitted into formal evaluation.",
    ),
    RegistryFieldSpec(
        field_name="reject_reason_codes",
        required=False,
        producer="coverage_gate",
        description="Semicolon-delimited rejection reason codes.",
    ),
    RegistryFieldSpec(
        field_name="loader_visible",
        required=True,
        producer="loader",
        description="Whether the v2 loader can recover the registered query.",
    ),
    RegistryFieldSpec(
        field_name="coverage_key",
        required=True,
        producer="coverage_gate",
        description="Coverage bucket key, usually dataset_id + canonical_subitem_id.",
    ),
    RegistryFieldSpec(
        field_name="coverage_target_min",
        required=True,
        producer="coverage_gate",
        description="Target threshold: 5 for agent-backed subitems, enumerate_all_applicable for deterministic families.",
    ),
)


def _rows(items: Iterable[object]) -> list[dict[str, object]]:
    return [asdict(item) for item in items]


def template_contract_rows() -> list[dict[str, object]]:
    return _rows(TEMPLATE_CONTRACTS)


def deterministic_rule_rows() -> list[dict[str, object]]:
    return _rows(DETERMINISTIC_ENUMERATION_RULES)


def query_registry_field_rows() -> list[dict[str, object]]:
    return _rows(QUERY_REGISTRY_FIELDS)


def default_facet_ids_for_subitem(subitem_id: str) -> tuple[str, ...]:
    return SUBITEM_DEFAULT_FACETS.get(subitem_id, ())


def family_for_subitem(subitem_id: str) -> str:
    return SUBITEM_TO_FAMILY.get(subitem_id, "")
