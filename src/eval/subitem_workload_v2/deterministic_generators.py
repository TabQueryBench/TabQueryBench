"""Helpers for deterministic inventory slices."""

from __future__ import annotations

from typing import Any


def deterministic_items_from_inventory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in (payload.get("items") or [])
        if str(item.get("realization_mode") or "") == "deterministic"
    ]


def agent_items_from_inventory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in (payload.get("items") or [])
        if str(item.get("realization_mode") or "") == "agent"
    ]
