#!/usr/bin/env python3
"""Compatibility wrapper for the time/score trade-off figure runner."""

from __future__ import annotations

import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "src" / "eval" / "time_score_tradeoff" / "runner.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
