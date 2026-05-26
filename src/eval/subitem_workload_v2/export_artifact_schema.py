"""Export the v2 artifact schema into machine-readable and markdown files."""

from __future__ import annotations

import csv
from pathlib import Path

from .artifact_schema import ARTIFACT_SCHEMA_NODES, artifact_schema_rows
from .paths import V2_CONTRACTS_DIR, V2_EVALUATION_FINAL_DIR


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path) -> None:
    lines = [
        "# v2 Artifact Schema",
        "",
        "This file documents the isolated artifact graph for the v2 subitem workload line.",
        "",
        f"- total_artifact_nodes: `{len(ARTIFACT_SCHEMA_NODES)}`",
        "",
    ]
    for node in ARTIFACT_SCHEMA_NODES:
        lines.append(f"- `{node.artifact_key}` -> `{node.relative_path}` ({node.producer})")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = artifact_schema_rows()
    _write_csv(V2_CONTRACTS_DIR / "artifact_schema_v2.csv", rows)
    _write_csv(V2_EVALUATION_FINAL_DIR / "artifact_schema_v2.csv", rows)
    _write_markdown(V2_EVALUATION_FINAL_DIR / "artifact_schema_v2.md")


if __name__ == "__main__":
    main()
