#!/usr/bin/env python3
"""Generate hyperparameter rank-response plots for query and distance scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_scoring.eval.hyper.rank_response_plots import generate_rank_response_plots


def main() -> None:
    output_root = PROJECT_ROOT.parent / "results" / "hyper"
    merged_csv = output_root / "hyper_run_metric_matrix.csv"
    result = generate_rank_response_plots(merged_csv, output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
