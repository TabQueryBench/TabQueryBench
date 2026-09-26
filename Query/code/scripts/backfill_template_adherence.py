#!/usr/bin/env python3
"""Backfill template adherence artifacts for existing legacy/v1 grounded run directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.workload_grounding.adherence import analyze_sql_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill grounding/template_adherence.json for existing legacy/v1 run directories.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more legacy/v1 grounded run directories under logs/runs/.",
    )
    parser.add_argument(
        "--template-library",
        default="data/workload_grounding/template_library_v1.jsonl",
        help="Template library JSONL used to resolve template ids.",
    )
    return parser.parse_args()


def load_template_lookup(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[obj["template_id"]] = obj
    return rows


def main() -> None:
    args = parse_args()
    template_lookup = load_template_lookup(Path(args.template_library))

    for raw_run_dir in args.run_dirs:
        run_dir = Path(raw_run_dir)
        selection = json.loads((run_dir / "grounding" / "selection.json").read_text(encoding="utf-8"))
        sql_text = (run_dir / "generated_sql.sql").read_text(encoding="utf-8")
        adherence = analyze_sql_queries(
            sql_queries=[sql_text],
            template_lookup=template_lookup,
            shortlist_ids=[item["template_id"] for item in selection.get("shortlist", [])],
        )
        output_path = run_dir / "grounding" / "template_adherence.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(adherence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.setdefault("grounding", {})["adherence"] = {
                "overall_adherence_score": adherence["overall_adherence_score"],
                "commented_query_count": adherence["commented_query_count"],
                "analyzed_query_count": adherence["analyzed_query_count"],
                "shortlist_violation_count": adherence["shortlist_violation_count"],
                "label_counts": adherence["label_counts"],
                "artifact_path": str(output_path.resolve()),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir.resolve()),
                    "output_path": str(output_path.resolve()),
                    "overall_adherence_score": adherence["overall_adherence_score"],
                    "label_counts": adherence["label_counts"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
