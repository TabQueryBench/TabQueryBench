#!/usr/bin/env python3
"""Render basic LaTeX tables for v2 coverage CSV artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.subitem_workload_v2.tables import csv_to_latex_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LaTeX tables from v2 coverage CSV files.")
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--tex-path", type=Path, required=True)
    parser.add_argument("--caption", type=str, default="V2 subitem workload coverage")
    parser.add_argument("--label", type=str, default="tab:v2-subitem-workload-coverage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_to_latex_table(
        csv_path=args.csv_path,
        tex_path=args.tex_path,
        caption=args.caption,
        label=args.label,
    )
    print(f"[v2-table] csv={args.csv_path} tex={args.tex_path}")


if __name__ == "__main__":
    main()
