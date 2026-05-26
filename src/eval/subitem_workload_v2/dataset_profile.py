"""Dataset role profiling helpers for the v2 workload line."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import DATA_DIR
from src.data.bundle import DatasetBundle, load_dataset_bundle
from src.db.csv_sqlite import SqliteMaterializationResult, materialize_dataset_to_sqlite
from src.workload_grounding.queryset_builder import FieldStats, build_field_stats


@dataclass(frozen=True)
class DatasetRoleProfile:
    dataset_id: str
    bundle: DatasetBundle
    sqlite_result: SqliteMaterializationResult
    field_stats: dict[str, FieldStats]
    row_count: int
    target_column: str | None
    groupable_cols: tuple[str, ...]
    numeric_cols: tuple[str, ...]
    low_card_cols: tuple[str, ...]
    high_card_cols: tuple[str, ...]
    temporal_cols: tuple[str, ...]
    missing_cols: tuple[str, ...]
    filterable_cols: tuple[str, ...]
    condition_cols: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
            "target_column": self.target_column,
            "groupable_cols": list(self.groupable_cols[:8]),
            "numeric_cols": list(self.numeric_cols[:8]),
            "low_card_cols": list(self.low_card_cols[:8]),
            "high_card_cols": list(self.high_card_cols[:8]),
            "temporal_cols": list(self.temporal_cols[:8]),
            "missing_cols": list(self.missing_cols[:8]),
            "filterable_cols": list(self.filterable_cols[:8]),
            "condition_cols": list(self.condition_cols[:8]),
        }


def _target_column_from_bundle(bundle: DatasetBundle) -> str | None:
    value = (
        bundle.dataset_semantics.get("target_column")
        or bundle.dataset_contract.get("target_column")
        or bundle.dataset_profile.get("target_column")
        or ""
    )
    text = str(value).strip()
    return text or None


def _row_count_from_bundle(bundle: DatasetBundle) -> int:
    return int(
        (bundle.dataset_contract.get("row_counts") or {}).get("main")
        or (bundle.dataset_profile.get("summary") or {}).get("n_rows")
        or 0
    )


def _missing_cols(bundle: DatasetBundle) -> list[str]:
    row_count = max(1, _row_count_from_bundle(bundle))
    contract_columns = {
        str(column.get("name") or "").strip(): column
        for column in (bundle.dataset_contract.get("columns") or [])
        if str(column.get("name") or "").strip()
    }
    results: list[str] = []
    for name, column in contract_columns.items():
        profile_stats = column.get("profile_stats") or {}
        missing_rate = float(profile_stats.get("missing_rate") or 0.0)
        missing_count = int(round(missing_rate * row_count))
        if missing_count > 0:
            results.append(name)
    return results


def _is_temporal(stats: FieldStats) -> bool:
    tokens = f"{stats.declared_type} {stats.semantic_type} {stats.name}".lower()
    return "date" in tokens or "time" in tokens or "timestamp" in tokens or "temporal" in tokens


def _is_high_card(stats: FieldStats, row_count: int) -> bool:
    return stats.distinct_count >= 20 or (row_count > 0 and stats.distinct_count / row_count >= 0.2)


def _is_low_card(stats: FieldStats) -> bool:
    return 1 < stats.distinct_count <= 8


def _is_groupable(stats: FieldStats, target_column: str | None) -> bool:
    if stats.use_for_groupby:
        return True
    if stats.use_as_target and stats.name == target_column and stats.distinct_count <= 10:
        return True
    return stats.is_categorical and stats.distinct_count <= 32


def _filterable(stats: FieldStats) -> bool:
    return stats.use_for_predicate or stats.is_numeric or stats.is_categorical


def load_dataset_role_profile(
    dataset_id: str,
    *,
    data_root: Path = DATA_DIR,
    use_cache: bool = True,
) -> DatasetRoleProfile:
    bundle = load_dataset_bundle(dataset_id=dataset_id, data_root=data_root, strict=True)
    sqlite_result = materialize_dataset_to_sqlite(bundle=bundle, use_cache=use_cache)
    field_stats = build_field_stats(bundle, sqlite_result.table_name, sqlite_result.db_path)
    target_column = _target_column_from_bundle(bundle)
    row_count = sqlite_result.row_count or _row_count_from_bundle(bundle)

    groupable_cols = [
        stats.name
        for stats in field_stats.values()
        if _is_groupable(stats, target_column)
    ]
    numeric_cols = [stats.name for stats in field_stats.values() if stats.is_numeric and stats.name != target_column]
    low_card_cols = [stats.name for stats in field_stats.values() if _is_low_card(stats)]
    high_card_cols = [stats.name for stats in field_stats.values() if _is_high_card(stats, row_count)]
    temporal_cols = [stats.name for stats in field_stats.values() if _is_temporal(stats)]
    filterable_cols = [stats.name for stats in field_stats.values() if _filterable(stats)]
    missing_cols = _missing_cols(bundle)

    condition_cols = list(low_card_cols)
    if target_column and target_column in field_stats and target_column not in condition_cols:
        condition_cols.insert(0, target_column)

    return DatasetRoleProfile(
        dataset_id=dataset_id,
        bundle=bundle,
        sqlite_result=sqlite_result,
        field_stats=field_stats,
        row_count=row_count,
        target_column=target_column,
        groupable_cols=tuple(groupable_cols),
        numeric_cols=tuple(numeric_cols),
        low_card_cols=tuple(low_card_cols),
        high_card_cols=tuple(high_card_cols),
        temporal_cols=tuple(temporal_cols),
        missing_cols=tuple(missing_cols),
        filterable_cols=tuple(filterable_cols),
        condition_cols=tuple(condition_cols),
    )
