#!/usr/bin/env python3
"""Generate per-model hyperparameter metric card figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.hyper.model_metric_cards import generate_model_metric_cards


def main() -> None:
    output_root = PROJECT_ROOT / "Evaluation" / "hyper"
    result = generate_model_metric_cards(PROJECT_ROOT, output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
