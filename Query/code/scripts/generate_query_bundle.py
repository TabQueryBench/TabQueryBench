#!/usr/bin/env python3
"""CLI for generating a query bundle from an arbitrary CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.workload_grounding.external_bundle import generate_query_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a TabQueryBench query bundle from a CSV and optional metadata JSON.")
    parser.add_argument("--csv", required=True, type=Path, help="Input CSV with a header row.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or existing directory for bundle artifacts.")
    parser.add_argument("--metadata", type=Path, help="Optional metadata JSON; supports dataset_id, table_name, and roles.")
    parser.add_argument("--dataset-id", help="Optional dataset ID overriding metadata and CSV filename.")
    parser.add_argument("--template-library", type=Path, help="Optional JSONL registry; defaults to the public v8 registry.")
    parser.add_argument("--max-queries", type=int, default=25, help="Maximum executable queries to retain (default: 25).")
    args = parser.parse_args()
    if args.max_queries < 1:
        parser.error("--max-queries must be at least 1")
    result = generate_query_bundle(csv_path=args.csv, output_dir=args.output_dir, metadata_path=args.metadata, dataset_id=args.dataset_id, template_library_path=args.template_library, max_queries=args.max_queries)
    print(f"query_bundle={result['bundle_path']}")
    print(f"manifest={result['manifest_path']}")
    print(f"accepted_queries={result['query_count']} skipped_templates={result['skipped_template_count']}")


if __name__ == "__main__":
    main()
