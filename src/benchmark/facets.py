"""Family facet catalog loader and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.benchmark.models import FIVE_FIXED_FAMILIES


def load_family_facet_catalog(path: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
            if isinstance(parsed, dict):
                payload = parsed

    schema_version = str(payload.get("schema_version") or "family_facet_catalog_v0_1")
    families_obj = payload.get("families") or {}

    normalized: dict[str, list[dict[str, Any]]] = {}
    for family in FIVE_FIXED_FAMILIES:
        facets = families_obj.get(family) if isinstance(families_obj, dict) else None
        if not isinstance(facets, list) or not facets:
            normalized[family] = [
                {
                    "facet_id": f"{family}_general",
                    "facet_name": f"{family} general",
                    "description": "General family-oriented structural probe.",
                    "minimum_evidence_requirements": ["grouped_support", "target_relevant_summary"],
                }
            ]
            continue
        normalized_facets: list[dict[str, Any]] = []
        for idx, facet in enumerate(facets, start=1):
            if not isinstance(facet, dict):
                continue
            facet_id = str(facet.get("facet_id") or f"{family}_facet_{idx}")
            normalized_facets.append(
                {
                    "facet_id": facet_id,
                    "facet_name": str(facet.get("facet_name") or facet_id),
                    "description": str(facet.get("description") or ""),
                    "minimum_evidence_requirements": list(facet.get("minimum_evidence_requirements") or []),
                    "anti_patterns": list(facet.get("anti_patterns") or []),
                }
            )
        if not normalized_facets:
            normalized_facets = [
                {
                    "facet_id": f"{family}_general",
                    "facet_name": f"{family} general",
                    "description": "General family-oriented structural probe.",
                    "minimum_evidence_requirements": ["grouped_support", "target_relevant_summary"],
                }
            ]
        normalized[family] = normalized_facets

    return {
        "schema_version": schema_version,
        "fixed_family_ontology": True,
        "families": normalized,
    }


def facets_for_family(catalog: dict[str, Any], family: str) -> list[dict[str, Any]]:
    families = catalog.get("families") if isinstance(catalog, dict) else {}
    if isinstance(families, dict):
        facets = families.get(family) or []
        if isinstance(facets, list):
            return [item for item in facets if isinstance(item, dict)]
    return []


def facet_ids_for_family(catalog: dict[str, Any], family: str) -> list[str]:
    return [str(item.get("facet_id")) for item in facets_for_family(catalog, family) if item.get("facet_id")]


def choose_facet_id(
    *,
    catalog: dict[str, Any],
    family: str,
    item_index: int,
    suggested_facet_id: str | None = None,
) -> str:
    facet_ids = facet_ids_for_family(catalog, family)
    if not facet_ids:
        return "unknown"
    if suggested_facet_id and suggested_facet_id in facet_ids:
        return suggested_facet_id
    return facet_ids[item_index % len(facet_ids)]


def catalog_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    families = catalog.get("families") if isinstance(catalog, dict) else {}
    summary: dict[str, Any] = {
        "schema_version": catalog.get("schema_version"),
        "fixed_family_ontology": catalog.get("fixed_family_ontology", True),
        "families": {},
    }
    if not isinstance(families, dict):
        return summary
    for family, facets in families.items():
        if not isinstance(facets, list):
            continue
        summary["families"][family] = [
            {"facet_id": item.get("facet_id"), "facet_name": item.get("facet_name")}
            for item in facets
            if isinstance(item, dict)
        ]
    return summary

