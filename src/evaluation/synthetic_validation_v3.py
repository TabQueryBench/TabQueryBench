"""Deterministic synthetic-data validation metrics (protocol v0.3)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MISSING_TOKENS = {"", "null", "none", "nan", "na", "n/a", "<null>"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text.lower() in MISSING_TOKENS


def _parse_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    try:
        return float(text)
    except Exception:  # noqa: BLE001
        return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_mean(values: list[float]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


@dataclass
class ColumnStats:
    name: str
    row_count: int
    missing_count: int
    missing_rate: float
    non_missing_count: int
    distinct_non_missing: set[str]
    numeric_values: list[float]
    semantic_kind: str
    numeric_profile: str

    @property
    def distinct_non_missing_count(self) -> int:
        return len(self.distinct_non_missing)

    @property
    def duplicate_count(self) -> int:
        return max(0, self.non_missing_count - self.distinct_non_missing_count)

    @property
    def numeric_min(self) -> float | None:
        if not self.numeric_values:
            return None
        return min(self.numeric_values)

    @property
    def numeric_max(self) -> float | None:
        if not self.numeric_values:
            return None
        return max(self.numeric_values)


@dataclass
class ValidationContextV3:
    dataset_id: str
    columns: list[str]
    type_hints: dict[str, str]
    real_stats: dict[str, ColumnStats]
    real_row_count: int
    expected_columns: list[str]


def _load_field_type_hints(project_root: Path, dataset_id: str) -> dict[str, str]:
    field_registry_path = project_root / "data" / dataset_id / "metadata" / "field_registry.json"
    if not field_registry_path.exists():
        return {}
    try:
        payload = json.loads(field_registry_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

    hints: dict[str, str] = {}
    fields = payload.get("fields")
    if not isinstance(fields, list):
        return {}
    for item in fields:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        declared_type = str(item.get("declared_type") or "").strip().lower()
        semantic_type = str(item.get("semantic_type") or "").strip().lower()
        merged = semantic_type or declared_type
        if not merged:
            continue
        hints[name] = merged
    return hints


def _resolve_semantic_kind(type_hint: str, sample_values: list[str]) -> str:
    token = (type_hint or "").lower()
    if any(word in token for word in ["numeric", "integer", "float", "double", "decimal"]):
        return "numeric"
    if any(word in token for word in ["boolean", "categorical", "nominal", "ordinal", "string", "text"]):
        return "categorical"

    non_missing = [value for value in sample_values if not _is_missing(value)]
    if not non_missing:
        return "categorical"
    parseable = sum(1 for value in non_missing if _parse_float(value) is not None)
    ratio = parseable / max(1, len(non_missing))
    return "numeric" if ratio >= 0.95 else "categorical"


def _numeric_profile(kind: str, distinct_count: int) -> str:
    if kind != "numeric":
        return "non_numeric"
    if distinct_count > 20:
        return "continuous"
    return "discrete"


def _read_column_values(path: Path, expected_columns: list[str]) -> tuple[int, dict[str, list[str]]]:
    values_by_col: dict[str, list[str]] = {col: [] for col in expected_columns}
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        index_map = {str(name): idx for idx, name in enumerate(header)}
        for row in reader:
            row_count += 1
            for col in expected_columns:
                idx = index_map.get(col)
                if idx is None or idx >= len(row):
                    values_by_col[col].append("")
                else:
                    values_by_col[col].append(str(row[idx]))
    return row_count, values_by_col


def _build_stats(
    *,
    expected_columns: list[str],
    values_by_col: dict[str, list[str]],
    row_count: int,
    type_hints: dict[str, str],
) -> dict[str, ColumnStats]:
    stats: dict[str, ColumnStats] = {}
    for col in expected_columns:
        values = values_by_col.get(col, [])
        non_missing = [value for value in values if not _is_missing(value)]
        missing_count = max(0, len(values) - len(non_missing))
        distinct_non_missing = set(non_missing)

        semantic_kind = _resolve_semantic_kind(type_hints.get(col, ""), values)
        numeric_values: list[float] = []
        if semantic_kind == "numeric":
            numeric_values = [parsed for parsed in (_parse_float(value) for value in non_missing) if parsed is not None]
        numeric_profile = _numeric_profile(semantic_kind, len(distinct_non_missing))

        stats[col] = ColumnStats(
            name=col,
            row_count=row_count,
            missing_count=missing_count,
            missing_rate=(missing_count / max(1, row_count)),
            non_missing_count=len(non_missing),
            distinct_non_missing=distinct_non_missing,
            numeric_values=numeric_values,
            semantic_kind=semantic_kind,
            numeric_profile=numeric_profile,
        )
    return stats


def build_validation_context_v3(
    *,
    dataset_id: str,
    project_root: Path,
    real_csv_path: Path,
    expected_columns: list[str],
) -> ValidationContextV3:
    type_hints = _load_field_type_hints(project_root=project_root, dataset_id=dataset_id)
    row_count, values_by_col = _read_column_values(real_csv_path, expected_columns)
    real_stats = _build_stats(
        expected_columns=expected_columns,
        values_by_col=values_by_col,
        row_count=row_count,
        type_hints=type_hints,
    )
    return ValidationContextV3(
        dataset_id=dataset_id,
        columns=expected_columns,
        type_hints=type_hints,
        real_stats=real_stats,
        real_row_count=row_count,
        expected_columns=expected_columns,
    )


def _score_cardinality_range(
    *,
    real_stats: dict[str, ColumnStats],
    syn_stats: dict[str, ColumnStats],
) -> dict[str, Any]:
    discrete_cols = [
        col
        for col, stat in real_stats.items()
        if stat.distinct_non_missing_count > 10 and stat.numeric_profile != "continuous"
    ]
    continuous_cols = [col for col, stat in real_stats.items() if stat.numeric_profile == "continuous"]

    discrete_score = None
    discrete_details: dict[str, Any] = {"eligible_columns": discrete_cols}
    if discrete_cols:
        risky_cols = []
        total_entries = 0
        missing_entries = 0
        per_col: list[dict[str, Any]] = []
        for col in discrete_cols:
            real_values = real_stats[col].distinct_non_missing
            syn_values = syn_stats[col].distinct_non_missing
            missing_values = sorted(real_values - syn_values)
            risk = len(missing_values) > 0
            if risk:
                risky_cols.append(col)
            total_entries += len(real_values)
            missing_entries += len(missing_values)
            per_col.append(
                {
                    "column": col,
                    "real_distinct": len(real_values),
                    "syn_distinct": len(syn_values),
                    "missing_value_count": len(missing_values),
                    "risk_flag": risk,
                }
            )

        risk_ratio = len(risky_cols) / max(1, len(discrete_cols))
        value_missing_ratio = missing_entries / max(1, total_entries)
        indicator_1 = _clip01(1.0 - risk_ratio)
        indicator_2 = _clip01(1.0 - value_missing_ratio)
        discrete_score = _clip01((indicator_1 + indicator_2) / 2.0)
        discrete_details.update(
            {
                "indicator_1_column_risk_coverage_score": round(indicator_1, 6),
                "indicator_2_value_level_coverage_score": round(indicator_2, 6),
                "risk_column_ratio": round(risk_ratio, 6),
                "value_missing_ratio": round(value_missing_ratio, 6),
                "risky_columns": risky_cols,
                "total_value_entries": total_entries,
                "missing_value_entries": missing_entries,
                "per_column": per_col,
            }
        )

    continuous_score = None
    continuous_details: dict[str, Any] = {"eligible_columns": continuous_cols}
    if continuous_cols:
        per_col_scores: list[float] = []
        per_col_details: list[dict[str, Any]] = []
        for col in continuous_cols:
            real_min = real_stats[col].numeric_min
            real_max = real_stats[col].numeric_max
            syn_min = syn_stats[col].numeric_min
            syn_max = syn_stats[col].numeric_max

            if real_min is None or real_max is None or syn_min is None or syn_max is None:
                score = 0.0
                detail = {
                    "column": col,
                    "status": "invalid_numeric_range",
                    "real_min": real_min,
                    "real_max": real_max,
                    "syn_min": syn_min,
                    "syn_max": syn_max,
                    "score": score,
                }
            else:
                span = max(1e-9, real_max - real_min)
                min_dev = abs(syn_min - real_min) / span
                max_dev = abs(syn_max - real_max) / span
                score = _clip01(1.0 - ((min_dev + max_dev) / 2.0))
                detail = {
                    "column": col,
                    "real_min": real_min,
                    "real_max": real_max,
                    "syn_min": syn_min,
                    "syn_max": syn_max,
                    "min_deviation_ratio": round(min_dev, 6),
                    "max_deviation_ratio": round(max_dev, 6),
                    "score": round(score, 6),
                }
            per_col_scores.append(score)
            per_col_details.append(detail)

        continuous_score = _safe_mean(per_col_scores)
        if continuous_score is not None:
            continuous_score = _clip01(continuous_score)
        continuous_details.update(
            {
                "per_column": per_col_details,
                "continuous_score": (round(continuous_score, 6) if continuous_score is not None else None),
            }
        )

    channel_score = _safe_mean([v for v in [discrete_score, continuous_score] if v is not None])
    return {
        "score": (round(channel_score, 6) if channel_score is not None else None),
        "discrete_profile_score": (round(discrete_score, 6) if discrete_score is not None else None),
        "continuous_profile_score": (round(continuous_score, 6) if continuous_score is not None else None),
        "details": {
            "discrete_profile": discrete_details,
            "continuous_profile": continuous_details,
        },
    }


def _score_missing_introduction(
    *,
    real_stats: dict[str, ColumnStats],
    syn_stats: dict[str, ColumnStats],
) -> dict[str, Any]:
    c0 = [col for col, stat in real_stats.items() if stat.missing_count == 0]
    violations = [col for col in c0 if syn_stats[col].missing_count > 0]

    if not c0:
        return {
            "score": None,
            "status": "not_applicable",
            "details": {"reason": "no_real_complete_columns"},
        }

    if not violations:
        return {
            "score": 1.0,
            "status": "ok",
            "details": {"complete_columns": c0, "violations": []},
        }

    if len(violations) >= 2:
        return {
            "score": 0.0,
            "status": "hard_fail_multi_column_violation",
            "details": {
                "complete_columns": c0,
                "violations": violations,
                "violation_count": len(violations),
            },
        }

    violating_col = violations[0]
    missing_rate = syn_stats[violating_col].missing_rate
    base_penalty = 0.5
    smooth_penalty = min(0.5, 0.5 * (missing_rate / 0.01))
    score = _clip01(1.0 - base_penalty - smooth_penalty)
    return {
        "score": round(score, 6),
        "status": "single_column_violation",
        "details": {
            "complete_columns": c0,
            "violations": violations,
            "base_penalty": base_penalty,
            "smooth_penalty": round(smooth_penalty, 6),
            "missing_rate_violation_column": round(missing_rate, 6),
            "violation_column": violating_col,
        },
    }


def _score_uniqueness_integrity(
    *,
    real_stats: dict[str, ColumnStats],
    syn_stats: dict[str, ColumnStats],
) -> dict[str, Any]:
    eligible = [
        col
        for col, stat in real_stats.items()
        if stat.missing_count == 0 and stat.distinct_non_missing_count == stat.row_count
    ]

    cat_eligible = [col for col in eligible if real_stats[col].semantic_kind != "numeric"]
    num_eligible = [col for col in eligible if real_stats[col].semantic_kind == "numeric"]

    if not eligible:
        return {
            "score": None,
            "status": "not_applicable",
            "details": {"reason": "no_uniqueness_eligible_columns"},
        }

    cat_violations = [col for col in cat_eligible if syn_stats[col].duplicate_count > 0]
    cat_score = 1.0
    cat_details: dict[str, Any] = {
        "eligible_columns": cat_eligible,
        "violations": cat_violations,
    }
    if len(cat_violations) >= 2:
        cat_score = 0.0
        cat_details["status"] = "hard_fail_multi_categorical_violation"
    elif len(cat_violations) == 1:
        col = cat_violations[0]
        dup_count = syn_stats[col].duplicate_count
        dup_rate = dup_count / max(1, syn_stats[col].non_missing_count)
        base_penalty = 0.5
        smooth_penalty = min(0.5, 0.5 * (dup_rate / 0.01))
        cat_score = _clip01(1.0 - base_penalty - smooth_penalty)
        cat_details.update(
            {
                "status": "single_categorical_violation",
                "violation_column": col,
                "duplicate_count": dup_count,
                "duplicate_rate": round(dup_rate, 6),
                "base_penalty": base_penalty,
                "smooth_penalty": round(smooth_penalty, 6),
            }
        )
    else:
        cat_details["status"] = "ok"

    num_fail_cols = [col for col in num_eligible if syn_stats[col].duplicate_count > 10]
    num_score = 0.0 if num_fail_cols else 1.0
    num_details = {
        "eligible_columns": num_eligible,
        "tolerance_duplicate_threshold": 10,
        "violations": num_fail_cols,
        "status": ("hard_fail_numeric_duplicate" if num_fail_cols else "ok"),
    }

    branch_scores: list[float] = []
    if cat_eligible:
        branch_scores.append(cat_score)
    if num_eligible:
        branch_scores.append(num_score)
    score = min(branch_scores) if branch_scores else None

    return {
        "score": (round(score, 6) if score is not None else None),
        "status": "ok" if (score is not None and score > 0) else "hard_fail",
        "details": {
            "eligible_columns": eligible,
            "categorical_branch": {
                "score": round(cat_score, 6) if cat_eligible else None,
                **cat_details,
            },
            "numerical_branch": {
                "score": round(num_score, 6) if num_eligible else None,
                **num_details,
            },
        },
    }


def _score_impossible_state_placeholder() -> dict[str, Any]:
    return {
        "score": None,
        "status": "placeholder_not_implemented",
        "details": {
            "note": "Impossible-state validation remains a reserved independent channel in v0.3.",
        },
    }


def evaluate_synthetic_validation_v3(
    *,
    context: ValidationContextV3,
    synthetic_csv_path: Path,
) -> dict[str, Any]:
    syn_row_count, syn_values_by_col = _read_column_values(synthetic_csv_path, context.expected_columns)
    syn_stats = _build_stats(
        expected_columns=context.expected_columns,
        values_by_col=syn_values_by_col,
        row_count=syn_row_count,
        type_hints=context.type_hints,
    )

    cardinality_range = _score_cardinality_range(real_stats=context.real_stats, syn_stats=syn_stats)
    missing_intro = _score_missing_introduction(real_stats=context.real_stats, syn_stats=syn_stats)
    uniqueness = _score_uniqueness_integrity(real_stats=context.real_stats, syn_stats=syn_stats)
    impossible_state = _score_impossible_state_placeholder()

    return {
        "contract_version": "synthetic_validation_v3",
        "dataset_id": context.dataset_id,
        "synthetic_csv_path": str(synthetic_csv_path.resolve()),
        "row_count_real": context.real_row_count,
        "row_count_synthetic": syn_row_count,
        "validation_channels": {
            "cardinality_range": cardinality_range,
            "missing_introduction": missing_intro,
            "uniqueness_integrity": uniqueness,
            "impossible_state": impossible_state,
        },
        "validation_scores": {
            "cardinality_range_score": cardinality_range.get("score"),
            "missing_introduction_score": missing_intro.get("score"),
            "uniqueness_integrity_score": uniqueness.get("score"),
            "impossible_state_score": impossible_state.get("score"),
        },
    }
