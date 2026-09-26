from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate token usage tables for subitem workload v2.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        required=True,
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
        project_root / "Evaluation" / "subitem_workload_v2" / "final" / "token_usage_snapshot"
    )
    cmd = [
        sys.executable,
        str(project_root / "src" / "eval" / "subitem_workload_v2" / "token_usage_report.py"),
        "--inventory-dir",
        str(project_root / "data" / "workload_grounding_v2" / "inventories"),
        "--run-root",
        str(project_root / "logs" / "subitem_workload_v2" / "runs"),
        "--run-ids",
        *args.run_ids,
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    print(f"[token-report] output_dir={output_dir}")


if __name__ == "__main__":
    main()
