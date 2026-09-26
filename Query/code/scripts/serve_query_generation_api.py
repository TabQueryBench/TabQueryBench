#!/usr/bin/env python3
"""Run the TabQueryBench query-generation API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the TabQueryBench query-generation API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable development auto-reload")
    parser.add_argument("--log-level", default="info", choices=("critical", "error", "warning", "info", "debug", "trace"))
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("uvicorn and fastapi are required to run the API") from exc
    uvicorn.run("tqb_query.query_generation.api:app", host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)


if __name__ == "__main__":
    main()
