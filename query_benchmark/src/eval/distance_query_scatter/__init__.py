"""Distance-versus-SQL-score scatter plots."""

from __future__ import annotations

__all__ = ["run_distance_query_scatter"]


def __getattr__(name: str):
    if name == "run_distance_query_scatter":
        from .runner import run_distance_query_scatter

        return run_distance_query_scatter
    raise AttributeError(name)
