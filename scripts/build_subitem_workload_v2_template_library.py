#!/usr/bin/env python3
"""Build the isolated v2 template library JSONL."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.subitem_workload_v2.catalog import write_template_library_jsonl
from src.eval.subitem_workload_v2.export_contract_tables import main as export_contract_tables
from src.eval.subitem_workload_v2.paths import ensure_line_dirs, normalize_line_version, template_library_path


def parse_args() -> object:
    import argparse

    parser = argparse.ArgumentParser(description="Build the isolated subitem workload template library JSONL.")
    parser.add_argument("--line-version", type=str, choices=["v2", "v3", "v4"], default="v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    line_version = normalize_line_version(args.line_version)
    ensure_line_dirs(line_version)
    output_path = template_library_path(line_version)
    rows = write_template_library_jsonl(output_path)
    if line_version == "v2":
        export_contract_tables()
    print(f"[{line_version}-template-library] rows={len(rows)} output={output_path}")


if __name__ == "__main__":
    main()
