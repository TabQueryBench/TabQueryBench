"""Registry-backed evaluation exports for the v2 workload line."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tqb_query.analytics_contract import build_subitem_and_family_rows

from .loader import load_v2_query_rows
from .reporting import write_markdown_summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def evaluate_registry(
    *,
    registry_path: Path,
    dataset_id: str,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    query_rows = load_v2_query_rows(registry_path)
    accepted_rows = [row for row in query_rows if bool(row.get("accepted_for_eval"))]

    subitem_rows, family_rows = build_subitem_and_family_rows(
        query_rows=accepted_rows,
        context_fields={"dataset_id": dataset_id, "run_id": run_id},
        score_field="query_score",
        missingness_applicable=True,
    )

    metadata_completeness = {
        "family_id_present": sum(1 for row in query_rows if row.get("family_id")),
        "canonical_subitem_id_present": sum(1 for row in query_rows if row.get("canonical_subitem_id")),
        "intended_facet_id_present": sum(1 for row in query_rows if row.get("intended_facet_id")),
        "variant_semantic_role_present": sum(1 for row in query_rows if row.get("variant_semantic_role")),
    }
    inference_sources: dict[str, int] = {}
    for row in query_rows:
        key = str(row.get("subitem_inference_source") or "unknown")
        inference_sources[key] = inference_sources.get(key, 0) + 1

    summary = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "registry_path": str(registry_path.resolve()),
        "registered_query_count": len(query_rows),
        "accepted_query_count": len(accepted_rows),
        "metadata_completeness": metadata_completeness,
        "subitem_inference_source_counts": inference_sources,
    }

    _write_csv(output_dir / "accepted_query_rows_v2.csv", accepted_rows)
    _write_csv(output_dir / "subitem_eval_rows_v2.csv", subitem_rows)
    _write_csv(output_dir / "family_eval_rows_v2.csv", family_rows)
    (output_dir / "evaluation_summary_v2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(
        output_dir / "evaluation_summary_v2.md",
        title="V2 Registry Evaluation Summary",
        bullets=[
            f"dataset_id: `{dataset_id}`",
            f"run_id: `{run_id}`",
            f"registered_query_count: `{len(query_rows)}`",
            f"accepted_query_count: `{len(accepted_rows)}`",
            f"subitem_inference_source_counts: `{inference_sources}`",
        ],
        payload=summary,
    )
    return summary
