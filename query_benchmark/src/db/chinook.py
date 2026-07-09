"""Utilities for preparing the tutorial Chinook SQLite database."""

from __future__ import annotations

from pathlib import Path

import requests

CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"


def ensure_chinook_db(db_path: Path, download_url: str = CHINOOK_URL) -> Path:
    """Ensure Chinook.db exists locally, downloading if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        print(f"{db_path} already exists, skipping download.")
        return db_path

    response = requests.get(download_url, timeout=30)
    response.raise_for_status()
    db_path.write_bytes(response.content)
    print(f"Downloaded Chinook DB to {db_path}")
    return db_path
