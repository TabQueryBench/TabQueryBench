from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dataset x family x subitem query count table for subitem workload v2."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir or (
        project_root / "Evaluation" / "subitem_workload_v2" / "final"
    )
    cmd = [
        sys.executable,
        str(project_root / "src" / "eval" / "subitem_workload_v2" / "query_count_wide_table.py"),
        "--project-root",
        str(project_root),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    print(f"[dataset-query-count-table] output_dir={output_dir}")


if __name__ == "__main__":
    main()
