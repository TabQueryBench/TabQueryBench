"""Thin reporting helpers for v2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_markdown_summary(path: Path, *, title: str, bullets: list[str], payload: dict[str, Any] | None = None) -> None:
    lines = [f"# {title}", ""]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    if payload:
        lines.extend(["", "```json", json.dumps(payload, indent=2, ensure_ascii=False), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
