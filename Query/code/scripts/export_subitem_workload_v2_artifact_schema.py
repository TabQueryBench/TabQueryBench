#!/usr/bin/env python3
"""Export the artifact schema for the v2 workload line."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.subitem_workload_v2.export_artifact_schema import main


if __name__ == "__main__":
    main()
