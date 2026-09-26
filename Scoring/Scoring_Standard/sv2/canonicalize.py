"""Conversion of raw answers and SQL result tables into SV2 canonical answer forms."""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .common import (
    COUNT_SUPPORT_DISTRIBUTION,
    NUMERIC_MAGNITUDE,
    RATE_SHARE_PROPORTION,
    SCALAR_KEY,
    TOPK_RANKING,
    ScoringError,
)
from .routing import template_routing_entry

_NUMERIC_TEXT = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")
_NON_FINITE_TEXT = frozenset({"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})
_SQL_LINE_COMMENT = re.compile(r"--[^\n]*")
_TRAILING_LIMIT = re.compile(r"\blimit\s+(\d+)\s*$", re.IGNORECASE)

KEYED_SCORERS = (NUMERIC_MAGNITUDE, RATE_SHARE_PROPORTION, COUNT_SUPPORT_DISTRIBUTION)


# ---------------------------------------------------------------------------
# Cell-level canonicalization
# ---------------------------------------------------------------------------


def _canonical_number(value: float) -> str:
    if value == 0:
        return "0"
    if float(value).is_integer() and abs(value) < 2**53:
        return str(int(value))
    return format(float(value), ".15g")


def canonical_key_cell(cell: Any) -> tuple[str, ...]:
    """Collision-safe, type-aware key cell.

    Numbers and numeric text share one representation so that the same group value
    materialized as INTEGER in one table and TEXT or REAL in the other still aligns.
    """
    if cell is None:
        return ("null",)
    if isinstance(cell, bool):
        return ("num", "1" if cell else "0")
    if isinstance(cell, int):
        return ("num", str(cell))
    if isinstance(cell, float):
        return ("num", _canonical_number(cell)) if math.isfinite(cell) else ("nonfinite", repr(cell))
    if isinstance(cell, (bytes, bytearray)):
        return ("bytes", bytes(cell).hex())
    text = str(cell)
    stripped = text.strip()
    if _NUMERIC_TEXT.match(stripped):
        number = float(stripped)
        if math.isfinite(number):
            return ("num", str(int(stripped)) if re.fullmatch(r"[-+]?\d+", stripped) else _canonical_number(number))
    return ("str", text)


def exact_key_cell(cell: Any) -> tuple[str, ...]:
    """Type-tagged raw key cell, used when numeric normalization would merge distinct values."""
    if cell is None:
        return ("null",)
    if isinstance(cell, bool):
        return ("bool", str(cell))
    if isinstance(cell, int):
        return ("int", str(cell))
    if isinstance(cell, float):
        return ("float", repr(cell))
    if isinstance(cell, (bytes, bytearray)):
        return ("bytes", bytes(cell).hex())
    return ("str", str(cell))


def canonical_key(row: Sequence[Any], indices: Sequence[int], *, exact: bool = False) -> tuple[tuple[str, ...], ...]:
    cell_key = exact_key_cell if exact else canonical_key_cell
    return tuple(cell_key(row[index]) for index in indices)


def _normalization_collides(rows: Sequence[Sequence[Any]], indices: Sequence[int]) -> bool:
    """True when two distinct raw keys in one result share a normalized key."""
    seen: dict[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]] = {}
    for row in rows:
        normalized = canonical_key(row, indices)
        exact = canonical_key(row, indices, exact=True)
        if seen.setdefault(normalized, exact) != exact:
            return True
    return False


def parse_numeric_cell(cell: Any, *, context: str) -> float | None:
    """SQL NULL -> None; finite numbers (or numeric text) -> float; anything else is invalid."""
    if cell is None:
        return None
    if isinstance(cell, bool):
        raise ScoringError("NON_NUMERIC_VALUE", f"{context}: boolean {cell!r} is not a numeric value.")
    if isinstance(cell, (int, float)):
        if not math.isfinite(cell):
            raise ScoringError("NON_FINITE_NUMERIC_VALUE", f"{context}: non-finite value {cell!r}.")
        return float(cell)
    text = str(cell).strip()
    if text.lower() in _NON_FINITE_TEXT:
        raise ScoringError("NON_FINITE_NUMERIC_VALUE", f"{context}: non-finite value {cell!r}.")
    if _NUMERIC_TEXT.match(text):
        return float(text)
    raise ScoringError("NON_NUMERIC_VALUE", f"{context}: cannot parse {cell!r} as a number.")


# ---------------------------------------------------------------------------
# Raw Python answers (public score_query API)
# ---------------------------------------------------------------------------


def _as_key(key: Hashable) -> tuple[Hashable, ...]:
    return key if isinstance(key, tuple) else (key,)


def canonicalize_answer(answer: Any, *, scorer_type: str, metadata: Mapping[str, Any] | None = None) -> Any:
    if scorer_type in (NUMERIC_MAGNITUDE, RATE_SHARE_PROPORTION):
        if answer is None or isinstance(answer, (int, float)) and not isinstance(answer, bool):
            return {SCALAR_KEY: answer}
        if isinstance(answer, Mapping):
            return {_as_key(key): value for key, value in answer.items()}
        if isinstance(answer, Sequence) and not isinstance(answer, (str, bytes)):
            return {(f"__pos_{index}__",): value for index, value in enumerate(answer)}
        raise ScoringError("UNSUPPORTED_ANSWER_FORM", f"Unsupported {scorer_type} answer {answer!r}.")
    if scorer_type == COUNT_SUPPORT_DISTRIBUTION:
        if isinstance(answer, Mapping):
            return [(_as_key(key), value) for key, value in answer.items()]
        if isinstance(answer, Sequence) and not isinstance(answer, (str, bytes)):
            entries = []
            for item in answer:
                if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                    raise ScoringError("UNSUPPORTED_ANSWER_FORM", f"Count entries must be (key, count) pairs, got {item!r}.")
                entries.append((_as_key(item[0]), item[1]))
            return entries
        raise ScoringError("UNSUPPORTED_ANSWER_FORM", f"Unsupported count answer {answer!r}.")
    if scorer_type == TOPK_RANKING:
        if isinstance(answer, Sequence) and not isinstance(answer, (str, bytes)):
            return list(answer)
        raise ScoringError("UNSUPPORTED_ANSWER_FORM", f"Ranking answers must be ordered sequences, got {answer!r}.")
    raise ScoringError("UNRESOLVED_SCORER_TYPE", f"Unknown scorer type {scorer_type!r}.")


# ---------------------------------------------------------------------------
# SQL result tables (analysis runner)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnPlan:
    scorer_type: str
    answer_shape: str
    columns: tuple[str, ...]
    value_index: int | None
    key_indices: tuple[int, ...]
    rank_index: int | None
    rank_direction: str
    diagnostic_columns: tuple[str, ...]
    requested_depth: int | None
    rate_scale: str
    plan_source: str

    def describe(self) -> dict[str, Any]:
        return {
            "plan_source": self.plan_source,
            "answer_shape": self.answer_shape,
            "value_column": self.columns[self.value_index] if self.value_index is not None else None,
            "key_columns": [self.columns[index] for index in self.key_indices],
            "rank_by": self.columns[self.rank_index] if self.rank_index is not None else None,
            "rank_direction": self.rank_direction if self.rank_index is not None else None,
            "diagnostic_columns": list(self.diagnostic_columns),
            "requested_depth": self.requested_depth,
            "rate_scale": self.rate_scale if self.scorer_type == RATE_SHARE_PROPORTION else None,
        }


def parse_sql_limit(sql: str) -> int | None:
    """Trailing LIMIT literal of the outermost statement, if any."""
    text = _SQL_LINE_COMMENT.sub("", str(sql or "")).strip().rstrip(";").strip()
    match = _TRAILING_LIMIT.search(text)
    return int(match.group(1)) if match else None


def _compatibility_entry(query: Mapping[str, Any], scorer_type: str, columns: Sequence[str]) -> dict[str, Any]:
    contract = query.get("semantic_result_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    lowered = {column.lower() for column in columns}
    primary = str(contract.get("primary_measure") or "")
    if not primary or primary.lower() not in lowered:
        raise ScoringError(
            "UNRESOLVED_VALUE_COLUMN",
            f"No template routing entry and primary measure {primary or None!r} is not a result column.",
        )
    measures = [str(item) for item in contract.get("measure_outputs") or [] if str(item).lower() != primary.lower()]
    if scorer_type == TOPK_RANKING:
        rank_by = str(contract.get("rank_by") or "")
        return {
            "answer_shape": "ranking",
            "rank_by": rank_by if rank_by.lower() in lowered else primary,
            "rank_direction": str(contract.get("rank_direction") or "desc"),
            "diagnostic_columns": [primary, *measures],
            "requested_depth": "sql_limit",
        }
    shape = "scalar" if str(contract.get("result_shape") or "").startswith("scalar") else "keyed"
    return {"answer_shape": shape, "value_columns": [primary], "diagnostic_columns": measures}


def build_column_plan(columns: Sequence[str], *, scorer_type: str, query: Mapping[str, Any]) -> ColumnPlan:
    entry = template_routing_entry(query.get("template_id"))
    plan_source = "template_routing"
    if entry is None or entry.get("scorer_type") != scorer_type:
        entry = _compatibility_entry(query, scorer_type, columns)
        plan_source = "semantic_result_contract"

    index_by_name: dict[str, int] = {}
    for index, column in enumerate(columns):
        index_by_name.setdefault(str(column).lower(), index)

    def find(name: str) -> int | None:
        return index_by_name.get(name.lower())

    answer_shape = str(entry.get("answer_shape") or "keyed")
    used: set[int] = set()

    value_index = None
    if answer_shape in ("scalar", "keyed"):
        candidates = [str(name) for name in entry.get("value_columns") or []]
        value_index = next((find(name) for name in candidates if find(name) is not None), None)
        if value_index is None:
            raise ScoringError("MISSING_OUTPUT_COLUMN", f"None of the value columns {candidates} is in the result {list(columns)}.")
        used.add(value_index)

    rank_index = None
    if answer_shape == "ranking":
        rank_by = entry.get("rank_by")
        if rank_by:
            rank_index = find(str(rank_by))
            if rank_index is None:
                raise ScoringError("MISSING_OUTPUT_COLUMN", f"Ranking column {rank_by!r} is not in the result {list(columns)}.")
            used.add(rank_index)

    diagnostic_columns = []
    for name in entry.get("diagnostic_columns") or []:
        index = find(str(name))
        if index is not None and index not in used:
            used.add(index)
            diagnostic_columns.append(columns[index])

    key_indices = tuple(index for index in range(len(columns)) if index not in used)
    if answer_shape == "scalar" and key_indices:
        raise ScoringError(
            "UNEXPECTED_OUTPUT_COLUMNS",
            f"Scalar answer has undeclared columns {[columns[index] for index in key_indices]}.",
        )
    if answer_shape == "ranking" and not key_indices:
        raise ScoringError("MISSING_RANKING_IDENTITY", "Ranking answer has no identity columns.")

    depth_rule = entry.get("requested_depth")
    if depth_rule == "sql_limit":
        requested_depth = parse_sql_limit(str(query.get("sql") or ""))
    elif isinstance(depth_rule, int) and not isinstance(depth_rule, bool):
        requested_depth = depth_rule
    else:
        requested_depth = None

    return ColumnPlan(
        scorer_type=scorer_type,
        answer_shape=answer_shape,
        columns=tuple(str(column) for column in columns),
        value_index=value_index,
        key_indices=key_indices,
        rank_index=rank_index,
        rank_direction=str(entry.get("rank_direction") or "desc").lower(),
        diagnostic_columns=tuple(diagnostic_columns),
        requested_depth=requested_depth,
        rate_scale=str(entry.get("rate_scale") or "unit"),
        plan_source=plan_source,
    )


def _rank_sort_key(value: float | None, direction: str) -> tuple[int, float]:
    if value is None:
        return (1, 0.0)
    return (0, -value if direction == "desc" else value)


def _answer_from_rows(
    rows: Sequence[Sequence[Any]], plan: ColumnPlan, *, side: str, exact_keys: bool
) -> tuple[Any, dict[str, Any]]:
    diagnostics: dict[str, Any] = {f"{side}_row_count": len(rows)}

    if plan.answer_shape == "scalar":
        if len(rows) != 1:
            raise ScoringError("SCALAR_ROW_COUNT", f"{side} scalar answer has {len(rows)} rows (expected 1).")
        cell = rows[0][plan.value_index]
        if plan.scorer_type == COUNT_SUPPORT_DISTRIBUTION:
            return [(SCALAR_KEY, parse_numeric_cell(cell, context=f"{side} count"))], diagnostics
        return {SCALAR_KEY: parse_numeric_cell(cell, context=f"{side} value")}, diagnostics

    if plan.answer_shape == "keyed":
        def key_for(position: int, row: Sequence[Any]) -> tuple[Hashable, ...]:
            if not plan.key_indices:
                return (f"__pos_{position}__",)
            return canonical_key(row, plan.key_indices, exact=exact_keys)

        if plan.scorer_type == COUNT_SUPPORT_DISTRIBUTION:
            entries = [
                (key_for(position, row), parse_numeric_cell(row[plan.value_index], context=f"{side} count"))
                for position, row in enumerate(rows)
            ]
            return entries, diagnostics
        answer: dict[tuple[Hashable, ...], float | None] = {}
        for position, row in enumerate(rows):
            key = key_for(position, row)
            if key in answer:
                raise ScoringError("DUPLICATE_KEY", f"{side} answer repeats semantic key {key!r}.")
            answer[key] = parse_numeric_cell(row[plan.value_index], context=f"{side} value for key {key!r}")
        return answer, diagnostics

    if plan.answer_shape == "ranking":
        items = []
        for row in rows:
            identity = canonical_key(row, plan.key_indices, exact=exact_keys)
            rank_value = (
                parse_numeric_cell(row[plan.rank_index], context=f"{side} ranking value")
                if plan.rank_index is not None
                else None
            )
            items.append((identity, rank_value))
        ordered = items
        if plan.rank_index is not None:
            # Deterministic tie-break: rank value in the declared direction, then identity ascending.
            ordered = sorted(items, key=lambda item: (_rank_sort_key(item[1], plan.rank_direction), item[0]))
        diagnostics[f"{side}_tie_reordered_positions"] = sum(1 for a, b in zip(items, ordered) if a[0] != b[0])
        diagnostics[f"{side}_rank_values_top"] = [value for _, value in ordered[:10]]
        return [identity for identity, _ in ordered], diagnostics

    raise ScoringError("UNSUPPORTED_ANSWER_SHAPE", f"Unknown answer shape {plan.answer_shape!r}.")


def answers_from_results(
    real_rows: Sequence[Sequence[Any]], syn_rows: Sequence[Sequence[Any]], plan: ColumnPlan
) -> tuple[Any, Any, dict[str, Any]]:
    """Canonical answers for both sides.

    Keys are normalized type-aware (numbers and numeric text align). If normalization would merge
    two distinct raw keys within either result, both sides use exact type-tagged keys instead.
    """
    exact_keys = bool(plan.key_indices) and plan.answer_shape in ("keyed", "ranking") and (
        _normalization_collides(real_rows, plan.key_indices) or _normalization_collides(syn_rows, plan.key_indices)
    )
    real, real_diag = _answer_from_rows(real_rows, plan, side="real", exact_keys=exact_keys)
    syn, syn_diag = _answer_from_rows(syn_rows, plan, side="synthetic", exact_keys=exact_keys)
    diagnostics = {"key_match_mode": "exact_typed" if exact_keys else "type_aware_numeric", **real_diag, **syn_diag}
    return real, syn, diagnostics
