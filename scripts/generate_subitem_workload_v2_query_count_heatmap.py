from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dataset-by-family query count heatmap for subitem workload v2."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
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
    input_csv = args.input_csv or (
        project_root / "Evaluation" / "subitem_workload_v2" / "final" / "dataset_family_query_token_summary.csv"
    )
    output_dir = args.output_dir or (
        project_root / "Evaluation" / "subitem_workload_v2" / "final"
    )
    cmd = [
        sys.executable,
        str(project_root / "src" / "eval" / "subitem_workload_v2" / "query_count_heatmap.py"),
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    print(f"[query-count-heatmap] output_dir={output_dir}")


if __name__ == "__main__":
    main()
