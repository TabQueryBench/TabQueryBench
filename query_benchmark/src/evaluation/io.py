"""I/O helpers for benchmark self-evaluation stage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.benchmark.facets import load_family_facet_catalog


@dataclass
class EvaluationContext:
    run_dir: Path
    benchmark_package_dir: Path
    dataset_id: str
    table_name: str
    db_path: Path
    build_manifest_v2: dict[str, Any]
    run_manifest: dict[str, Any]
    static_understanding: dict[str, Any]
    family_facet_catalog: dict[str, Any]
    query_specs: list[dict[str, Any]]
    question_bundles: list[dict[str, Any]]
    bundle_diversity_records: list[dict[str, Any]]
    query_execution_summaries: list[dict[str, Any]]
    set_curation_audit_v2: dict[str, Any]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
        except Exception:  # noqa: BLE001
            continue
    return rows


def _coerce_path(path_like: str | None) -> Path:
    if not path_like:
        return Path("")
    return Path(path_like)


def _load_query_specs(run_dir: Path, benchmark_package_dir: Path) -> list[dict[str, Any]]:
    pkg = _read_json(benchmark_package_dir / "queryspecs.json", {"queryspecs": []})
    if isinstance(pkg, dict) and isinstance(pkg.get("queryspecs"), list):
        rows = [item for item in pkg.get("queryspecs", []) if isinstance(item, dict)]
        if rows:
            return rows

    # Fallback: root jsonl (may include non-selected items in older runs).
    return _read_jsonl(run_dir / "query_specs.jsonl")


def _load_question_bundles(run_dir: Path, benchmark_package_dir: Path) -> list[dict[str, Any]]:
    pkg = _read_json(benchmark_package_dir / "question_bundles.json", {"bundles": []})
    if isinstance(pkg, dict) and isinstance(pkg.get("bundles"), list):
        rows = [item for item in pkg.get("bundles", []) if isinstance(item, dict)]
        if rows or (benchmark_package_dir / "question_bundles.json").exists():
            return rows

    pool = _read_json(run_dir / "question_bundle_pool.json", {"bundles": []})
    if isinstance(pool, dict) and isinstance(pool.get("bundles"), list):
        return [item for item in pool.get("bundles", []) if isinstance(item, dict)]
    return []


def _load_bundle_diversity_records(run_dir: Path, benchmark_package_dir: Path) -> list[dict[str, Any]]:
    package = _read_json(benchmark_package_dir / "bundle_diversity_matrix_v2.json", {})
    if isinstance(package, dict) and isinstance(package.get("bundles"), list):
        rows = [item for item in package.get("bundles", []) if isinstance(item, dict)]
        if rows:
            return rows

    return _read_jsonl(run_dir / "bundle_diversity_matrix_v2.jsonl")


def _load_query_execution_summaries(run_dir: Path, benchmark_package_dir: Path) -> list[dict[str, Any]]:
    package = _read_json(benchmark_package_dir / "query_execution_summaries_v2.json", {})
    if isinstance(package, dict) and isinstance(package.get("summaries"), list):
        rows = [item for item in package.get("summaries", []) if isinstance(item, dict)]
        if rows:
            return rows

    return _read_jsonl(run_dir / "query_execution_summaries_v2.jsonl")


def _fallback_question_id_from_text(text: str) -> str:
    normalized = " ".join(text.lower().split())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"rq_{digest}"


def _build_queryid_to_bundle_link(question_bundles: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for bundle in question_bundles:
        if not isinstance(bundle, dict):
            continue
        rq = bundle.get("research_question")
        if not isinstance(rq, dict):
            rq = {}

        question_id = str(rq.get("question_id") or "").strip()
        if not question_id:
            question_text = str(rq.get("question") or rq.get("question_text") or "").strip()
            if question_text:
                question_id = _fallback_question_id_from_text(question_text)

        family_id = str(rq.get("family_id") or rq.get("family") or bundle.get("family") or "").strip()
        intended_facet_id = str(rq.get("intended_facet_id") or "").strip()

        variants = bundle.get("variants")
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            spec = variant.get("query_spec")
            if not isinstance(spec, dict):
                continue
            query_id = str(spec.get("query_id") or "").strip()
            if not query_id:
                continue
            mapping[query_id] = {
                "question_id": question_id,
                "family_id": family_id,
                "intended_facet_id": intended_facet_id,
            }
    return mapping


def _infer_facet_id_from_family_and_role(family_id: str, variant_role: str) -> str:
    family = (family_id or "").strip()
    role = (variant_role or "").strip()

    if family == "subgroup_structure":
        if role == "ranked_signal_view":
            return "subgroup_rank_order"
        if role in {"contrastive_conditional_view", "filtered_stable_view"}:
            return "subgroup_conditional_contrast"
        return "subgroup_distribution_shift"

    if family == "conditional_dependency_structure":
        if role in {"ranked_signal_view", "contrastive_conditional_view", "filtered_stable_view"}:
            return "conditional_interaction_hotspots"
        if role in {"within_group_proportion", "focused_target_view", "collapsed_target_view"}:
            return "conditional_rate_shift"
        return "pairwise_conditional_dependency"

    if family == "tail_rarity_structure":
        if role == "ranked_signal_view":
            return "tail_ranked_signal"
        if role in {"rare_extreme_view", "filtered_stable_view"}:
            return "low_support_extremes"
        return "rare_target_concentration"

    if family == "missingness_structure":
        if role == "missing_target_interaction":
            return "missing_target_interaction"
        if role == "missing_rate_by_subgroup":
            return "missing_rate_by_subgroup"
        return "missing_indicator_distribution"

    if family == "cardinality_structure":
        if role == "ranked_signal_view":
            return "support_concentration"
        if role in {"focused_target_view", "collapsed_target_view", "within_group_proportion"}:
            return "target_cardinality_cross_section"
        return "value_imbalance_profile"

    return "unknown"


def _enrich_query_specs_with_bundle_links(
    query_specs: list[dict[str, Any]],
    question_bundles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not query_specs:
        return query_specs
    link_map = _build_queryid_to_bundle_link(question_bundles)
    if not link_map:
        return query_specs

    enriched: list[dict[str, Any]] = []
    for row in query_specs:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        query_id = str(item.get("query_id") or "").strip()
        link = link_map.get(query_id, {})
        if link:
            if not str(item.get("question_id") or "").strip() and link.get("question_id"):
                item["question_id"] = link["question_id"]
            if not str(item.get("stable_question_id") or "").strip() and str(item.get("question_id") or "").strip():
                item["stable_question_id"] = str(item.get("question_id"))
            if not str(item.get("family_id") or "").strip():
                linked_family = str(link.get("family_id") or "").strip()
                if linked_family:
                    item["family_id"] = linked_family
            if not str(item.get("intended_facet_id") or "").strip():
                linked_facet = str(link.get("intended_facet_id") or "").strip()
                if linked_facet:
                    item["intended_facet_id"] = linked_facet

        if not str(item.get("intended_facet_id") or "").strip():
            family_for_infer = str(item.get("family_id") or item.get("family") or "").strip()
            role_for_infer = str(item.get("variant_semantic_role") or "").strip()
            inferred = _infer_facet_id_from_family_and_role(family_for_infer, role_for_infer)
            if inferred and inferred != "unknown":
                item["intended_facet_id"] = inferred
        enriched.append(item)
    return enriched


def _coerce_family_facet_catalog(run_dir: Path, facet_wrapper: dict[str, Any]) -> dict[str, Any]:
    family_facet_catalog: dict[str, Any] = {}
    if isinstance(facet_wrapper, dict):
        if isinstance(facet_wrapper.get("catalog"), dict):
            family_facet_catalog = facet_wrapper.get("catalog", {})
        elif isinstance(facet_wrapper.get("families"), dict):
            family_facet_catalog = facet_wrapper

    if family_facet_catalog:
        return family_facet_catalog

    # Fallback to repo config; if absent, load defaults from helper.
    cfg_path: Path | None = None
    for parent in [run_dir] + list(run_dir.parents):
        candidate = parent / "config" / "family_facet_catalog_v0_1.yaml"
        if candidate.exists():
            cfg_path = candidate
            break
    return load_family_facet_catalog(cfg_path)


def load_evaluation_context(run_dir: Path) -> EvaluationContext:
    run_dir = run_dir.resolve()
    benchmark_package_dir = run_dir / "benchmark_package"

    run_manifest = _read_json(run_dir / "run_manifest.json", {})
    build_manifest_v2 = _read_json(run_dir / "build_manifest_v2.json", {})
    static_understanding = _read_json(run_dir / "static_understanding.json", {})

    facet_wrapper = _read_json(run_dir / "family_facet_catalog.json", {})
    family_facet_catalog = _coerce_family_facet_catalog(run_dir, facet_wrapper if isinstance(facet_wrapper, dict) else {})

    query_specs = _load_query_specs(run_dir, benchmark_package_dir)
    question_bundles = _load_question_bundles(run_dir, benchmark_package_dir)
    query_specs = _enrich_query_specs_with_bundle_links(query_specs, question_bundles)
    bundle_diversity_records = _load_bundle_diversity_records(run_dir, benchmark_package_dir)
    query_execution_summaries = _load_query_execution_summaries(run_dir, benchmark_package_dir)
    set_curation_audit_v2 = _read_json(run_dir / "set_curation_audit_v2.json", {})

    dataset_id = ""
    if isinstance(run_manifest, dict):
        dataset_id = str(run_manifest.get("dataset_id") or "")
    if not dataset_id and isinstance(build_manifest_v2, dict):
        dataset_id = str(build_manifest_v2.get("dataset_id") or "")
    if not dataset_id and isinstance(static_understanding, dict):
        dataset_id = str(static_understanding.get("dataset_id") or "")

    sqlite_obj = run_manifest.get("sqlite") if isinstance(run_manifest, dict) else {}
    db_path = _coerce_path(str((sqlite_obj or {}).get("db_path") or ""))
    table_name = str((sqlite_obj or {}).get("table_name") or "")

    if not db_path and isinstance(run_manifest, dict):
        db_path = _coerce_path(str(run_manifest.get("sqlite_db") or ""))

    if not table_name and query_specs:
        # Best-effort fallback for older manifests.
        table_name = dataset_id

    return EvaluationContext(
        run_dir=run_dir,
        benchmark_package_dir=benchmark_package_dir,
        dataset_id=dataset_id,
        table_name=table_name,
        db_path=db_path,
        build_manifest_v2=build_manifest_v2,
        run_manifest=run_manifest,
        static_understanding=static_understanding,
        family_facet_catalog=family_facet_catalog,
        query_specs=query_specs,
        question_bundles=question_bundles,
        bundle_diversity_records=bundle_diversity_records,
        query_execution_summaries=query_execution_summaries,
        set_curation_audit_v2=set_curation_audit_v2,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
