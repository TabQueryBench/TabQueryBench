"""Export the v2 contract tables into CSV and Markdown artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from .contract_spec import (
    DETERMINISTIC_ENUMERATION_RULES,
    QUERY_REGISTRY_FIELDS,
    TEMPLATE_CONTRACTS,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "workload_grounding_v2" / "contracts"
EVAL_DIR = ROOT / "Evaluation" / "subitem_workload_v2" / "final"


def _normalize_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "; ".join(value)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _normalize_value(value) for key, value in row.items()})


def _write_markdown(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# v2 Contract Tables",
        "",
        "This file records the agreed v2 workload design for the subitem coverage repair line.",
        "",
        "## Fixed design points",
        "",
        "- The eight agent-backed subitems must satisfy `dataset x subitem >= 5 accepted SQL`.",
        "- Missingness and cardinality are deterministic families with `enumerate_all_applicable` coverage.",
        "- `family_id` is fixed at the template layer.",
        "- `canonical_subitem_id`, `intended_facet_id`, and `variant_semantic_role` must be explicit in the v2 registry.",
        "- New v2 artifacts live under `data/workload_grounding_v2`, `logs/subitem_workload_v2`, `src/eval/subitem_workload_v2`, and `Evaluation/subitem_workload_v2`.",
        "",
        "## Template contract summary",
        "",
        f"- Total templates recorded: `{len(TEMPLATE_CONTRACTS)}`",
        f"- Legacy agent templates: `{sum(1 for row in TEMPLATE_CONTRACTS if row.realization_mode == 'agent')}`",
        f"- New deterministic templates: `{sum(1 for row in TEMPLATE_CONTRACTS if row.realization_mode == 'deterministic')}`",
        "",
        "## Deterministic rule summary",
        "",
        f"- Deterministic rules recorded: `{len(DETERMINISTIC_ENUMERATION_RULES)}`",
        "",
        "## Registry field summary",
        "",
        f"- Registry fields recorded: `{len(QUERY_REGISTRY_FIELDS)}`",
        "",
        "See the CSV siblings in the same folder for the machine-readable tables.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    template_rows = [asdict(row) for row in TEMPLATE_CONTRACTS]
    deterministic_rows = [asdict(row) for row in DETERMINISTIC_ENUMERATION_RULES]
    registry_rows = [asdict(row) for row in QUERY_REGISTRY_FIELDS]

    for base_dir in (DATA_DIR, EVAL_DIR):
        _write_csv(base_dir / "template_contract_matrix_v2.csv", template_rows)
        _write_csv(base_dir / "deterministic_enumeration_rules_v2.csv", deterministic_rows)
        _write_csv(base_dir / "query_registry_fields_v2.csv", registry_rows)

    _write_markdown(EVAL_DIR / "v2_contract_tables.md")


if __name__ == "__main__":
    main()
