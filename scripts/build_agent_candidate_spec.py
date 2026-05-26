#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CORE_TOP10 = [
    'tpl_clickbench_group_count',
    'tpl_clickbench_filtered_topk_group_count',
    'tpl_clickbench_group_distinct_topk',
    'tpl_clickbench_filtered_distinct_topk',
    'tpl_clickbench_group_summary_topk',
    'tpl_m4_group_condition_rate',
    'tpl_m4_group_ratio_two_conditions',
    'tpl_h2o_group_sum',
    'tpl_h2o_topn_within_group',
    'tpl_m4_support_guarded_group_avg',
]

EXPERIMENTAL_PLUS5 = [
    'tpl_m4_two_dimensional_group_avg',
    'tpl_clickbench_two_dimensional_topk_count',
    'tpl_m4_binned_numeric_group_avg',
    'tpl_m4_median_filtered_numeric',
    'tpl_tpcds_within_group_share',
]

TIME_AWARE_EXTENSIONS = [
    'tpl_rtabench_time_bucket_filtered_count',
    'tpl_rtabench_time_bucket_group_moving_avg',
    'tpl_tail_drift_ratio',
]

FAMILY_ORDER = {
    'subgroup_structure': 0,
    'conditional_dependency_structure': 1,
    'tail_rarity_structure': 2,
}

CURATION = {
    'tpl_clickbench_group_count': {
        'priority': 'p0',
        'why_pick': 'Most universal subgroup baseline; extremely easy for an agent to bind and explain.',
        'use_when': 'Any dataset has at least one groupable categorical or ordinal field.',
        'avoid_when': 'Skip only when the task explicitly needs a numeric measure or a filtered slice.'
    },
    'tpl_clickbench_filtered_topk_group_count': {
        'priority': 'p0',
        'why_pick': 'Matches real dashboard heavy-hitter analysis after a slice or filter.',
        'use_when': 'There is a groupable field and at least one sensible filterable field.',
        'avoid_when': 'Avoid if the filter would be arbitrary or if all fields are already extremely low cardinality.'
    },
    'tpl_clickbench_group_distinct_topk': {
        'priority': 'p0',
        'why_pick': 'Distinct-coverage ranking is common in web, product, and user analytics.',
        'use_when': 'The table exposes a reasonably high-cardinality entity or identifier-like column.',
        'avoid_when': 'Avoid on datasets without a meaningful entity-like column.'
    },
    'tpl_clickbench_filtered_distinct_topk': {
        'priority': 'p0',
        'why_pick': 'Adds the common pattern of ranking distinct coverage inside a filtered slice.',
        'use_when': 'There is both a meaningful filter and a usable entity-like column.',
        'avoid_when': 'Avoid when the only possible entity fallback would be semantically weak.'
    },
    'tpl_clickbench_group_summary_topk': {
        'priority': 'p0',
        'why_pick': 'High information density: support, average, and distinct coverage in one query shape.',
        'use_when': 'The dataset has a groupable field, a numeric measure, and a distinct-entity candidate.',
        'avoid_when': 'Avoid on measure-free datasets or when the distinct role is too weak.'
    },
    'tpl_m4_group_condition_rate': {
        'priority': 'p0',
        'why_pick': 'Condition rates are one of the most reusable analytical questions across domains.',
        'use_when': 'There is a low-cardinality condition column and a clean subgroup axis.',
        'avoid_when': 'Avoid when all candidate condition columns are high-cardinality or numeric-only.'
    },
    'tpl_m4_group_ratio_two_conditions': {
        'priority': 'p0',
        'why_pick': 'Directly captures KPI-style comparisons that agents often need to propose.',
        'use_when': 'A binary or low-cardinality condition field exists and ratio semantics are meaningful.',
        'avoid_when': 'Avoid when the denominator condition would be unstable or poorly defined.'
    },
    'tpl_h2o_group_sum': {
        'priority': 'p0',
        'why_pick': 'Grouped sums are missing surprisingly often in template libraries despite being universal.',
        'use_when': 'There is any numeric measure and one stable group axis.',
        'avoid_when': 'Avoid on purely categorical tables with no meaningful numeric measure.'
    },
    'tpl_h2o_topn_within_group': {
        'priority': 'p1',
        'why_pick': 'Provides a clean, agent-friendly window ranking primitive that the current core needed.',
        'use_when': 'There is a numeric measure and a natural subgroup field.',
        'avoid_when': 'Avoid when within-group ranking would be noisy because groups are too small.'
    },
    'tpl_m4_support_guarded_group_avg': {
        'priority': 'p1',
        'why_pick': 'Adds a broadly useful support guard so agents can prefer subgroup summaries that are less likely to be noise.',
        'use_when': 'There is a numeric measure, a sensible subgroup axis, and sparse small groups are a real concern.',
        'avoid_when': 'Avoid when the dataset is tiny or when every subgroup should be reported regardless of support.'
    },
    'tpl_m4_two_dimensional_group_avg': {
        'priority': 'p1',
        'why_pick': 'Adds the missing two-axis subgroup interaction pattern that frequently appears in production dashboards.',
        'use_when': 'There are two distinct subgroup axes and a stable numeric measure worth comparing across their grid.',
        'avoid_when': 'Avoid when the second group axis would be arbitrary or when the subgroup matrix would be extremely sparse.'
    },
    'tpl_clickbench_two_dimensional_topk_count': {
        'priority': 'p1',
        'why_pick': 'Captures joint heavy-hitter analysis without introducing numeric-measure dependencies.',
        'use_when': 'Two subgroup dimensions matter jointly and the question is about the most common combinations.',
        'avoid_when': 'Avoid when the task only needs a single grouping axis or when the second axis has no analytical meaning.'
    },
    'tpl_m4_binned_numeric_group_avg': {
        'priority': 'p1',
        'why_pick': 'Adds bucketed numeric analysis so the agent can avoid unnatural raw grouping on continuous fields.',
        'use_when': 'A numeric field should be summarized in coarse bands before comparing average outcomes.',
        'avoid_when': 'Avoid when there is no meaningful numeric banding variable or when the dataset is purely categorical.'
    },
    'tpl_m4_median_filtered_numeric': {
        'priority': 'p1',
        'why_pick': 'Adds a robust filtered summary that is less sensitive to skew than mean-only templates.',
        'use_when': 'The question is about a filtered numeric slice and a robust center is more natural than a raw average.',
        'avoid_when': 'Avoid when the filtered slice would be too small or when the question clearly asks for a count or sum.'
    },
    'tpl_tpcds_within_group_share': {
        'priority': 'p1',
        'why_pick': 'Adds share-of-total / contribution reasoning, which is common in BI and reporting workloads.',
        'use_when': 'The task asks how much each subgroup contributes relative to the whole within a broader grouping.',
        'avoid_when': 'Avoid when the user only needs absolute subgroup totals or when denominator semantics are unclear.'
    },
    'tpl_rtabench_time_bucket_filtered_count': {
        'priority': 'extension',
        'why_pick': 'Canonical temporal dashboard query for event logs and time-aware fact tables.',
        'use_when': 'The dataset exposes a real timestamp/date field plus a sensible filter.',
        'avoid_when': 'Do not force on non-temporal datasets or on ordinal fields that are not true time.'
    },
    'tpl_rtabench_time_bucket_group_moving_avg': {
        'priority': 'extension',
        'why_pick': 'Adds temporal smoothing and trend-reading behavior that simple counts cannot capture.',
        'use_when': 'The dataset has a real time field and one subgroup dimension worth trending.',
        'avoid_when': 'Avoid if the dataset lacks time, or if the series would be too sparse to support rolling averages.'
    },
    'tpl_tail_drift_ratio': {
        'priority': 'extension',
        'why_pick': 'Encodes material negative drift relative to a prior period, which is one of the clearest production tail-movement patterns.',
        'use_when': 'The dataset exposes a real temporal field and the task is about current-vs-prior decline by subgroup.',
        'avoid_when': 'Do not use on non-temporal datasets or when period boundaries would be arbitrary.'
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build curated agent candidate spec from the template library.')
    parser.add_argument('--template-library', default='data/workload_grounding/template_library_v1.jsonl')
    parser.add_argument('--extension-library', default='data/workload_grounding/template_library_extensions_v1.jsonl')
    parser.add_argument('--portability-report', default='data/workload_grounding/template_portability_report_v1.csv')
    parser.add_argument('--extension-portability-report', default='data/workload_grounding/template_extension_portability_report_v1.csv')
    parser.add_argument('--output', default='data/workload_grounding/agent_candidate_spec_top10_v1.json')
    parser.add_argument('--experimental-output', default='data/workload_grounding/agent_candidate_spec_top10_plus5_v1.json')
    parser.add_argument('--all-core-output', default='data/workload_grounding/agent_candidate_spec_all_core_v1.json')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--logs-root', default='logs/workload_grounding')
    return parser.parse_args()


def load_templates(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                obj = json.loads(line)
                rows[obj['template_id']] = obj
    return rows


def load_portability(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if not path.exists():
        return out
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row['template_id'], {'yes': 0, 'partial': 0, 'no': 0})[row['portable']] += 1
    return out


def _fallback_curation(template_id: str, template: dict[str, Any]) -> dict[str, str]:
    template_name = template.get('template_name', template_id).rstrip('.')
    primary_family = template.get('primary_family', '')
    required_roles = ', '.join(template.get('required_roles', [])) or 'standard analytical roles'
    priority = 'p1'
    if template.get('activation_tier') == 'optional' or template.get('dialect_sensitive'):
        priority = 'p1'
    if template.get('materialization_bucket') == 'extension':
        priority = 'extension'
    family_summary = {
        'subgroup_structure': 'a reusable subgroup-structure pattern',
        'conditional_dependency_structure': 'a reusable conditional-dependency pattern',
        'tail_rarity_structure': 'a reusable tail-or-rarity pattern',
    }.get(primary_family, 'a reusable analytical pattern')
    return {
        'priority': priority,
        'why_pick': f'Adds {family_summary} grounded by public evidence: {template_name}.',
        'use_when': f'Use when the question naturally maps to {template_name.lower()} and the dataset can bind roles such as {required_roles}.',
        'avoid_when': 'Avoid when the question can be answered by a simpler, more universal template or when the required roles would be forced.',
    }


def all_core_template_ids(templates: dict[str, dict[str, Any]]) -> list[str]:
    core_ids = [
        tid for tid, template in templates.items()
        if template.get('materialization_bucket', 'core') == 'core'
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    for tid in CORE_TOP10 + EXPERIMENTAL_PLUS5:
        if tid in core_ids and tid not in seen:
            ordered.append(tid)
            seen.add(tid)
    remaining = sorted(
        [tid for tid in core_ids if tid not in seen],
        key=lambda tid: (
            1 if templates[tid].get('activation_tier') == 'optional' else 0,
            FAMILY_ORDER.get(templates[tid].get('primary_family', ''), 9),
            tid,
        ),
    )
    ordered.extend(remaining)
    return ordered


def build_entry(template_id: str, rank: int | None, bucket: str, templates: dict[str, dict[str, Any]], portability: dict[str, dict[str, int]]) -> dict[str, Any]:
    template = templates[template_id]
    curated = CURATION.get(template_id, _fallback_curation(template_id, template))
    portability_summary = portability.get(template_id, {'yes': 0, 'partial': 0, 'no': 0})
    return {
        'rank': rank,
        'bucket': bucket,
        'template_id': template_id,
        'template_name': template['template_name'],
        'source_workload_id': template['source_workload_id'],
        'primary_family': template['primary_family'],
        'secondary_family': template.get('secondary_family'),
        'status': template['status'],
        'materialization_bucket': template.get('materialization_bucket', 'core'),
        'activation_tier': template.get('activation_tier', 'core'),
        'required_roles': template['required_roles'],
        'constraints': template['constraints'],
        'portability_summary': portability_summary,
        'priority': curated['priority'],
        'why_pick': curated['why_pick'],
        'use_when': curated['use_when'],
        'avoid_when': curated['avoid_when'],
        'dialect_sensitive': template.get('dialect_sensitive', False),
        'dialect_notes': template.get('dialect_notes'),
        'provenance': template['provenance'],
        'provenance_sources': template.get('provenance_sources', [template['provenance']]),
    }


def build_spec(
    *,
    selection_intent: str,
    core_templates: list[str],
    extension_templates: list[str],
    templates: dict[str, dict[str, Any]],
    portability: dict[str, dict[str, int]],
    extra_sections: dict[str, list[str]] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = {
        'spec_version': 'v1',
        'selection_intent': selection_intent,
        'selection_principles': [
            'Prefer templates with clear provenance and query-level evidence.',
            'Prefer templates that cover common analytical intents rather than corner cases.',
            'Prefer templates whose role binding is strong on current datasets unless they are explicitly marked as temporal extensions.',
            'Avoid near-duplicate templates that do not add a new analytical shape.',
        ],
        'core_top10': [build_entry(tid, idx, 'core_top10', templates, portability) for idx, tid in enumerate(core_templates, start=1)],
        'time_aware_extensions': [build_entry(tid, None, 'time_aware_extension', templates, portability) for tid in extension_templates],
    }
    if extra_sections:
        for bucket_name, bucket_templates in extra_sections.items():
            spec[bucket_name] = [
                build_entry(tid, idx, bucket_name, templates, portability) for idx, tid in enumerate(bucket_templates, start=1)
            ]
    if extra_metadata:
        spec.update(extra_metadata)
    return spec


def main() -> None:
    args = parse_args()
    template_path = Path(args.template_library)
    extension_template_path = Path(args.extension_library)
    portability_path = Path(args.portability_report)
    extension_portability_path = Path(args.extension_portability_report)
    output_path = Path(args.output)
    experimental_output_path = Path(args.experimental_output)
    all_core_output_path = Path(args.all_core_output)

    templates = load_templates(template_path)
    templates.update(load_templates(extension_template_path))
    portability = load_portability(portability_path)
    portability.update(load_portability(extension_portability_path))

    missing = [tid for tid in CORE_TOP10 + EXPERIMENTAL_PLUS5 + TIME_AWARE_EXTENSIONS if tid not in templates]
    if missing:
        raise KeyError(f'Missing template ids for candidate spec: {missing}')

    all_core = all_core_template_ids(templates)

    spec = build_spec(
        selection_intent='Curated candidate set for future agent integration over the single-table analytics template library.',
        core_templates=CORE_TOP10,
        extension_templates=TIME_AWARE_EXTENSIONS,
        templates=templates,
        portability=portability,
    )
    experimental_spec = build_spec(
        selection_intent='Experimental expansion of the stable core_top10 with a second-tier plus-five shortlist for ablation and candidate-pool studies.',
        core_templates=CORE_TOP10,
        extension_templates=TIME_AWARE_EXTENSIONS,
        templates=templates,
        portability=portability,
        extra_sections={
            'experimental_plus5': EXPERIMENTAL_PLUS5,
            'experimental_top15': CORE_TOP10 + EXPERIMENTAL_PLUS5,
        },
        extra_metadata={
            'experimental_design': {
                'base_bucket': 'core_top10',
                'goal': 'Test whether a small second-tier expansion improves coverage without opening the full 26-template core library.',
                'notes': [
                    'The plus-five shortlist comes from the top10 research review and the m4 production-pack analysis.',
                    'These additions are intentionally diverse: two-dimensional subgrouping, joint heavy hitters, bucketed numeric analysis, robust filtered summary, and share-of-total.',
                ],
            },
        },
    )
    all_core_spec = build_spec(
        selection_intent='Default all-core candidate set for runtime agent integration over the full materialized core template library.',
        core_templates=CORE_TOP10,
        extension_templates=TIME_AWARE_EXTENSIONS,
        templates=templates,
        portability=portability,
        extra_sections={
            'all_core': all_core,
        },
        extra_metadata={
            'runtime_design': {
                'default_bucket': 'all_core',
                'fallback_reference_bucket': 'core_top10',
                'notes': [
                    'The all-core bucket keeps every materialized core template in one candidate pool.',
                    'This asset is now the default runtime candidate pool for the template-grounded SQL agent.',
                    'The stable top10 remains available as a smaller comparison and fallback slice.',
                ],
            },
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    experimental_output_path.parent.mkdir(parents=True, exist_ok=True)
    experimental_output_path.write_text(json.dumps(experimental_spec, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    all_core_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_core_output_path.write_text(json.dumps(all_core_spec, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    if args.run_id:
        manifest_path = Path(args.logs_root) / args.run_id / 'run_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        else:
            manifest = {'run_id': args.run_id}
        manifest.setdefault('outputs', {})['agent_candidate_spec'] = {
            'path': str(output_path.resolve()),
            'core_top10_count': len(spec['core_top10']),
            'time_aware_extension_count': len(spec['time_aware_extensions']),
            'core_library_path': str(template_path.resolve()),
            'extension_library_path': str(extension_template_path.resolve()),
        }
        manifest.setdefault('outputs', {})['agent_candidate_spec_top10_plus5'] = {
            'path': str(experimental_output_path.resolve()),
            'core_top10_count': len(experimental_spec['core_top10']),
            'experimental_plus5_count': len(experimental_spec['experimental_plus5']),
            'experimental_top15_count': len(experimental_spec['experimental_top15']),
            'time_aware_extension_count': len(experimental_spec['time_aware_extensions']),
            'core_library_path': str(template_path.resolve()),
            'extension_library_path': str(extension_template_path.resolve()),
        }
        manifest.setdefault('outputs', {})['agent_candidate_spec_all_core'] = {
            'path': str(all_core_output_path.resolve()),
            'core_top10_count': len(all_core_spec['core_top10']),
            'all_core_count': len(all_core_spec['all_core']),
            'time_aware_extension_count': len(all_core_spec['time_aware_extensions']),
            'core_library_path': str(template_path.resolve()),
            'extension_library_path': str(extension_template_path.resolve()),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(json.dumps({
        'output_path': str(output_path.resolve()),
        'experimental_output_path': str(experimental_output_path.resolve()),
        'all_core_output_path': str(all_core_output_path.resolve()),
        'core_top10_count': len(spec['core_top10']),
        'experimental_plus5_count': len(experimental_spec['experimental_plus5']),
        'experimental_top15_count': len(experimental_spec['experimental_top15']),
        'all_core_count': len(all_core_spec['all_core']),
        'time_aware_extension_count': len(spec['time_aware_extensions']),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
