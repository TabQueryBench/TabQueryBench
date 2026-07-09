"""Perturbation substrate for alignment/purity evaluation."""

from __future__ import annotations

import hashlib
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PerturbationVariant:
    variant_id: str
    kind: str  # real | boot | null | family
    family_id: str
    intensity: float
    repeat: int
    seed: int
    db_path: Path
    operators: list[str]
    notes: list[str]
    validity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "kind": self.kind,
            "family_id": self.family_id,
            "intensity": self.intensity,
            "repeat": self.repeat,
            "seed": self.seed,
            "db_path": str(self.db_path),
            "operators": self.operators,
            "notes": self.notes,
            "validity": self.validity or {},
        }


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed)] + [str(part) for part in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) % (2**31 - 1)


def _quote_ident(identifier: str) -> str:
    # Double-quote SQLite identifiers to support special characters (e.g. %).
    return '"' + str(identifier).replace('"', '""') + '"'


def _load_table(base_db_path: Path, table_name: str) -> tuple[str, list[str], list[dict[str, Any]]]:
    conn = sqlite3.connect(base_db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        if row is None or not row[0]:
            raise RuntimeError(f"Cannot find CREATE TABLE statement for table={table_name}")
        create_table_sql = str(row[0])

        quoted_table = _quote_ident(table_name)
        cols = [item[1] for item in cur.execute(f"PRAGMA table_info({quoted_table})").fetchall()]
        rows_raw = cur.execute(f"SELECT * FROM {quoted_table}").fetchall()
        rows: list[dict[str, Any]] = []
        for raw in rows_raw:
            rows.append({col: raw[col] for col in cols})
        return create_table_sql, cols, rows
    finally:
        conn.close()


def _write_table(db_path: Path, create_table_sql: str, table_name: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(create_table_sql)
        placeholders = ", ".join(["?" for _ in columns])
        quoted_table = _quote_ident(table_name)
        quoted_columns = ", ".join(_quote_ident(col) for col in columns)
        insert_sql = f"INSERT INTO {quoted_table} ({quoted_columns}) VALUES ({placeholders})"
        values = [[row.get(col) for col in columns] for row in rows]
        cur.executemany(insert_sql, values)
        conn.commit()
    finally:
        conn.close()


def _unique_non_null(rows: list[dict[str, Any]], column: str) -> list[Any]:
    values = []
    seen = set()
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _column_counts(rows: list[dict[str, Any]], column: str) -> Counter:
    counter = Counter()
    for row in rows:
        counter[row.get(column)] += 1
    return counter


def _choose_feature_columns(columns: list[str], target_column: str, key_fields: list[str]) -> list[str]:
    key_candidates = [str(col) for col in key_fields if col and col in columns and col != target_column]
    if key_candidates:
        dedup: list[str] = []
        for col in key_candidates:
            if col not in dedup:
                dedup.append(col)
        return dedup
    return [col for col in columns if col != target_column]


def _safe_missing_columns(
    *,
    columns: list[str],
    target_column: str,
    feature_columns: list[str],
    static_understanding: dict[str, Any],
) -> list[str]:
    role_map = static_understanding.get("field_roles") or {}
    if not isinstance(role_map, dict):
        role_map = {}

    protected_tokens = {"id", "key", "identifier", "index", "protected"}
    safe: list[str] = []
    for col in feature_columns:
        if col == target_column:
            continue
        role = str(role_map.get(col) or "").lower()
        if any(token in role for token in protected_tokens):
            continue
        if col not in safe:
            safe.append(col)

    if not safe:
        for col in columns:
            if col == target_column:
                continue
            role = str(role_map.get(col) or "").lower()
            if any(token in role for token in protected_tokens):
                continue
            safe.append(col)
    return safe


def _distribution(counter: Counter) -> dict[Any, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: (value / total) for key, value in counter.items()}


def _tv_distance(p: dict[Any, float], q: dict[Any, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0
    return 0.5 * sum(abs(float(p.get(key, 0.0)) - float(q.get(key, 0.0))) for key in keys)


def _cramers_v(x_vals: list[Any], y_vals: list[Any]) -> float:
    n = min(len(x_vals), len(y_vals))
    if n <= 1:
        return 0.0
    x_levels = list(dict.fromkeys(x_vals))
    y_levels = list(dict.fromkeys(y_vals))
    if len(x_levels) <= 1 or len(y_levels) <= 1:
        return 0.0

    x_index = {value: idx for idx, value in enumerate(x_levels)}
    y_index = {value: idx for idx, value in enumerate(y_levels)}
    table = [[0.0 for _ in y_levels] for _ in x_levels]
    for xv, yv in zip(x_vals[:n], y_vals[:n]):
        table[x_index[xv]][y_index[yv]] += 1.0

    row_sum = [sum(row) for row in table]
    col_sum = [sum(table[r][c] for r in range(len(x_levels))) for c in range(len(y_levels))]
    total = sum(row_sum)
    if total <= 0:
        return 0.0

    chi2 = 0.0
    for r in range(len(x_levels)):
        for c in range(len(y_levels)):
            expected = row_sum[r] * col_sum[c] / total if total > 0 else 0.0
            if expected <= 1e-12:
                continue
            chi2 += ((table[r][c] - expected) ** 2) / expected
    phi2 = chi2 / total
    k = min(len(x_levels) - 1, len(y_levels) - 1)
    if k <= 0:
        return 0.0
    return float((phi2 / k) ** 0.5)


def _family_statistics(
    *,
    rows: list[dict[str, Any]],
    target_column: str,
    feature_columns: list[str],
    safe_missing_columns: list[str],
) -> dict[str, float]:
    if not rows:
        return {
            "subgroup_structure": 0.0,
            "conditional_dependency_structure": 0.0,
            "tail_rarity_structure": 0.0,
            "missingness_structure": 0.0,
            "cardinality_structure": 0.0,
        }

    n = len(rows)
    primary_feature = feature_columns[0] if feature_columns else target_column
    secondary_feature = feature_columns[1] if len(feature_columns) > 1 else primary_feature

    # subgroup structure: target-distribution gap between top subgroup and complement
    subgroup_counts = Counter(row.get(primary_feature) for row in rows)
    top_group = subgroup_counts.most_common(1)[0][0] if subgroup_counts else None
    in_group = [row for row in rows if row.get(primary_feature) == top_group]
    out_group = [row for row in rows if row.get(primary_feature) != top_group]
    dist_in = _distribution(Counter(row.get(target_column) for row in in_group))
    dist_out = _distribution(Counter(row.get(target_column) for row in out_group))
    subgroup_stat = _tv_distance(dist_in, dist_out)

    # conditional dependency: cramer's V between primary feature and target
    x_vals = [row.get(primary_feature) for row in rows]
    y_vals = [row.get(target_column) for row in rows]
    conditional_stat = _cramers_v(x_vals, y_vals)

    # tail rarity: rare mass on highest-cardinality feature
    high_card_feature = primary_feature
    max_card = len({row.get(primary_feature) for row in rows})
    for col in feature_columns:
        card = len({row.get(col) for row in rows})
        if card > max_card:
            max_card = card
            high_card_feature = col
    counts = Counter(row.get(high_card_feature) for row in rows)
    rare_threshold = max(1, int(0.05 * n))
    rare_values = {value for value, count in counts.items() if value is not None and count <= rare_threshold}
    rare_mass = sum(1 for row in rows if row.get(high_card_feature) in rare_values) / max(1, n)

    # missingness: overall missing rate on safe columns
    miss_cols = safe_missing_columns if safe_missing_columns else [col for col in feature_columns if col != target_column]
    if miss_cols:
        total_cells = len(miss_cols) * n
        missing_cells = 0
        for row in rows:
            for col in miss_cols:
                if row.get(col) is None:
                    missing_cells += 1
        missing_stat = missing_cells / max(1, total_cells)
    else:
        missing_stat = 0.0

    # cardinality: distinct combo coverage on (primary, secondary)
    combos = {(row.get(primary_feature), row.get(secondary_feature)) for row in rows}
    cardinality_stat = len(combos) / max(1, n)

    return {
        "subgroup_structure": float(subgroup_stat),
        "conditional_dependency_structure": float(conditional_stat),
        "tail_rarity_structure": float(rare_mass),
        "missingness_structure": float(missing_stat),
        "cardinality_structure": float(cardinality_stat),
    }


def _target_floor(family_id: str, intensity: float) -> float:
    base = {
        "subgroup_structure": 0.04,
        "conditional_dependency_structure": 0.03,
        "tail_rarity_structure": 0.02,
        "missingness_structure": 0.015,
        "cardinality_structure": 0.02,
    }.get(family_id, 0.02)
    return max(0.005, base * max(0.1, float(intensity)))


def _evaluate_validity(
    *,
    family_id: str,
    intensity: float,
    baseline_stats: dict[str, float],
    variant_stats: dict[str, float],
    offtarget_ratio_max: float = 0.35,
) -> dict[str, Any]:
    if family_id not in baseline_stats or family_id not in variant_stats:
        return {
            "accepted": False,
            "target_change": 0.0,
            "offtarget_leakage": 0.0,
            "target_floor": 0.0,
            "offtarget_ratio_max": offtarget_ratio_max,
            "reason_codes": ["PERTURBATION_VALIDITY_FAMILY_UNKNOWN"],
        }

    changes = {
        fam: abs(float(variant_stats.get(fam, 0.0)) - float(baseline_stats.get(fam, 0.0)))
        for fam in baseline_stats.keys()
    }
    target_change = changes.get(family_id, 0.0)
    off_values = [value for fam, value in changes.items() if fam != family_id]
    off_leak = sum(off_values) / len(off_values) if off_values else 0.0
    floor = _target_floor(family_id, intensity)

    reason_codes: list[str] = []
    if target_change < floor:
        reason_codes.append("PERTURBATION_TARGET_CHANGE_TOO_WEAK")
    if off_leak > offtarget_ratio_max * max(target_change, 1e-9):
        reason_codes.append("PERTURBATION_OFFTARGET_LEAKAGE_HIGH")
    if not reason_codes:
        reason_codes.append("PERTURBATION_VALID")

    return {
        "accepted": not any(code != "PERTURBATION_VALID" for code in reason_codes),
        "target_change": float(target_change),
        "offtarget_leakage": float(off_leak),
        "target_floor": float(floor),
        "offtarget_ratio_max": float(offtarget_ratio_max),
        "changes_by_family": {fam: float(val) for fam, val in changes.items()},
        "reason_codes": reason_codes,
    }


def _apply_subgroup_structure(
    rows: list[dict[str, Any]],
    rng: random.Random,
    intensity: float,
    target_column: str,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    operators: list[str] = []
    notes: list[str] = []
    if not feature_columns:
        return operators, ["no_feature_columns_available"]

    subgroup_col = feature_columns[0]
    subgroup_values = _unique_non_null(rows, subgroup_col)
    if subgroup_values:
        # Subgroup proportion flattening by random reassignment.
        p = min(1.0, max(0.0, intensity))
        for row in rows:
            if rng.random() < p:
                row[subgroup_col] = rng.choice(subgroup_values)
        operators.append("subgroup_proportion_flattening")
        notes.append(f"column={subgroup_col}")

    # Subgroup-conditioned outcome shuffling.
    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row.get(subgroup_col)].append(idx)

    for _, indices in groups.items():
        if len(indices) <= 1:
            continue
        affected = max(1, int(len(indices) * intensity))
        selected = rng.sample(indices, k=min(affected, len(indices)))
        shuffled = [rows[idx].get(target_column) for idx in selected]
        rng.shuffle(shuffled)
        for idx, value in zip(selected, shuffled):
            rows[idx][target_column] = value
    operators.append("subgroup_conditioned_outcome_shuffling")
    return operators, notes


def _apply_conditional_dependency_structure(
    rows: list[dict[str, Any]],
    rng: random.Random,
    intensity: float,
    target_column: str,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    operators: list[str] = []
    notes: list[str] = []
    if not feature_columns:
        return operators, ["no_feature_columns_available"]

    cond_col = feature_columns[0]
    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row.get(cond_col)].append(idx)

    for _, indices in groups.items():
        if len(indices) <= 1:
            continue
        affected = max(1, int(len(indices) * intensity))
        selected = rng.sample(indices, k=min(affected, len(indices)))
        values = [rows[idx].get(target_column) for idx in selected]
        rng.shuffle(values)
        for idx, value in zip(selected, values):
            rows[idx][target_column] = value
    operators.append("conditional_target_shuffle")
    notes.append(f"condition_column={cond_col}")

    # Dependency attenuation: partial global randomization of target.
    global_targets = [row.get(target_column) for row in rows if row.get(target_column) is not None]
    if global_targets:
        p = min(1.0, intensity * 0.7)
        for row in rows:
            if rng.random() < p:
                row[target_column] = rng.choice(global_targets)
        operators.append("dependency_attenuation")
    return operators, notes


def _apply_tail_rarity_structure(
    rows: list[dict[str, Any]],
    rng: random.Random,
    intensity: float,
    target_column: str,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    operators: list[str] = []
    notes: list[str] = []
    if not feature_columns:
        return operators, ["no_feature_columns_available"]

    feature = max(feature_columns, key=lambda col: len(_unique_non_null(rows, col)))
    counts = _column_counts(rows, feature)
    sorted_counts = sorted((count, value) for value, count in counts.items() if value is not None)
    cutoff_idx = max(0, int(len(sorted_counts) * min(0.8, intensity)) - 1)
    cutoff = sorted_counts[cutoff_idx][0] if sorted_counts else 0
    rare_values = {value for value, count in counts.items() if value is not None and count <= cutoff}
    for row in rows:
        if row.get(feature) in rare_values:
            row[feature] = "__pooled_tail__"
    operators.append("rare_category_pooling")
    notes.append(f"feature={feature}")

    target_counts = _column_counts(rows, target_column)
    if target_counts:
        major = target_counts.most_common(1)[0][0]
        rare_targets = {
            value
            for value, count in target_counts.items()
            if value is not None and count <= max(1, int(len(rows) * 0.02 * max(0.1, intensity)))
        }
        for row in rows:
            if row.get(target_column) in rare_targets and rng.random() < intensity:
                row[target_column] = major
        operators.append("tail_clipping")
    return operators, notes


def _apply_missingness_structure(
    rows: list[dict[str, Any]],
    rng: random.Random,
    intensity: float,
    target_column: str,
    feature_columns: list[str],
    safe_missing_columns: list[str],
) -> tuple[list[str], list[str]]:
    operators: list[str] = []
    notes: list[str] = []
    candidates = [col for col in safe_missing_columns if col != target_column]
    if not candidates:
        return operators, ["no_feature_columns_available"]

    col_a = candidates[0]
    miss_rate = min(0.5, 0.1 + intensity * 0.4)
    for row in rows:
        if rng.random() < miss_rate:
            row[col_a] = None
    operators.append("marginal_missingness_randomization")
    notes.append(f"column={col_a};miss_rate={miss_rate:.3f}")

    if len(candidates) > 1:
        col_b = candidates[1]
        miss_rate_b = min(0.5, 0.05 + intensity * 0.35)
        for row in rows:
            if rng.random() < miss_rate_b:
                row[col_b] = None
        operators.append("co_missingness_break")
        notes.append(f"column={col_b};miss_rate={miss_rate_b:.3f}")
    return operators, notes


def _apply_cardinality_structure(
    rows: list[dict[str, Any]],
    intensity: float,
    target_column: str,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    operators: list[str] = []
    notes: list[str] = []
    candidates = [col for col in feature_columns if col != target_column]
    if not candidates:
        return operators, ["no_feature_columns_available"]

    col = max(candidates, key=lambda item: len(_unique_non_null(rows, item)))
    counts = _column_counts(rows, col)
    sorted_values = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    keep_count = max(1, int((1.0 - min(0.9, intensity)) * len(sorted_values)))
    keep_values = {value for value, _ in sorted_values[:keep_count]}
    for row in rows:
        if row.get(col) not in keep_values:
            row[col] = "__other__"
    operators.append("category_pooling_by_support")

    # Collapse very low-support categories.
    counts2 = _column_counts(rows, col)
    threshold = max(1, int(len(rows) * 0.01 * max(0.2, intensity)))
    low_support_values = {value for value, count in counts2.items() if value is not None and count <= threshold}
    for row in rows:
        if row.get(col) in low_support_values:
            row[col] = "__collapsed_low_support__"
    operators.append("low_support_bucket_collapse")
    notes.append(f"column={col};keep_count={keep_count};threshold={threshold}")
    return operators, notes


def _apply_null_variant(
    rows: list[dict[str, Any]],
    rng: random.Random,
    intensity: float,
    target_column: str,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    operators: list[str] = ["budget_matched_sham_edit"]
    notes: list[str] = []
    mutable = [col for col in feature_columns if col != target_column]
    if not mutable:
        return operators, ["no_mutable_columns_for_null_variant"]

    value_pool: dict[str, list[Any]] = {}
    for col in mutable:
        values = [row.get(col) for row in rows if row.get(col) is not None]
        value_pool[col] = values

    p = min(0.30, max(0.02, float(intensity) * 0.12))
    changed = 0
    for row in rows:
        if rng.random() >= p:
            continue
        col = rng.choice(mutable)
        values = value_pool.get(col) or []
        if not values:
            continue
        row[col] = rng.choice(values)
        changed += 1
    notes.append(f"edited_rows={changed};edit_prob={p:.3f}")
    return operators, notes


def _apply_bootstrap(rows: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if not rows:
        return rows, ["bootstrap_resample"], ["no_rows"]
    sampled = [dict(rng.choice(rows)) for _ in range(len(rows))]
    return sampled, ["bootstrap_resample"], []


def _materialize_variant(
    *,
    base_rows: list[dict[str, Any]],
    create_table_sql: str,
    columns: list[str],
    table_name: str,
    output_db_path: Path,
    family_id: str,
    kind: str,
    intensity: float,
    repeat: int,
    seed: int,
    target_column: str,
    feature_columns: list[str],
    safe_missing_columns: list[str],
    baseline_stats: dict[str, float],
) -> tuple[PerturbationVariant, list[dict[str, Any]]]:
    rows = [dict(row) for row in base_rows]
    rng = random.Random(seed)
    operators: list[str] = []
    notes: list[str] = []

    if kind == "boot":
        rows, operators, notes = _apply_bootstrap(rows, rng)
    elif kind == "null":
        operators, notes = _apply_null_variant(
            rows,
            rng,
            intensity=float(intensity),
            target_column=target_column,
            feature_columns=feature_columns,
        )
    elif kind == "family":
        if family_id == "subgroup_structure":
            operators, notes = _apply_subgroup_structure(rows, rng, intensity, target_column, feature_columns)
        elif family_id == "conditional_dependency_structure":
            operators, notes = _apply_conditional_dependency_structure(rows, rng, intensity, target_column, feature_columns)
        elif family_id == "tail_rarity_structure":
            operators, notes = _apply_tail_rarity_structure(rows, rng, intensity, target_column, feature_columns)
        elif family_id == "missingness_structure":
            operators, notes = _apply_missingness_structure(
                rows,
                rng,
                intensity,
                target_column,
                feature_columns,
                safe_missing_columns,
            )
        elif family_id == "cardinality_structure":
            operators, notes = _apply_cardinality_structure(rows, intensity, target_column, feature_columns)
        else:
            notes = [f"unknown_family_operator:{family_id}"]

    _write_table(output_db_path, create_table_sql, table_name, columns, rows)

    validity: dict[str, Any] | None = None
    if kind == "family":
        variant_stats = _family_statistics(
            rows=rows,
            target_column=target_column,
            feature_columns=feature_columns,
            safe_missing_columns=safe_missing_columns,
        )
        validity = _evaluate_validity(
            family_id=family_id,
            intensity=float(intensity),
            baseline_stats=baseline_stats,
            variant_stats=variant_stats,
        )
    elif kind in {"boot", "null"}:
        variant_stats = _family_statistics(
            rows=rows,
            target_column=target_column,
            feature_columns=feature_columns,
            safe_missing_columns=safe_missing_columns,
        )
        validity = {
            "accepted": True,
            "target_change": 0.0,
            "offtarget_leakage": 0.0,
            "target_floor": 0.0,
            "offtarget_ratio_max": 0.35,
            "changes_by_family": {
                fam: abs(float(variant_stats.get(fam, 0.0)) - float(baseline_stats.get(fam, 0.0)))
                for fam in baseline_stats.keys()
            },
            "reason_codes": ["CONTROL_VARIANT"],
        }

    variant_id = f"{kind}_{family_id}_i{intensity:.2f}_r{repeat}"
    return PerturbationVariant(
        variant_id=variant_id,
        kind=kind,
        family_id=family_id,
        intensity=float(intensity),
        repeat=repeat,
        seed=seed,
        db_path=output_db_path,
        operators=operators,
        notes=notes,
        validity=validity,
    ), rows


def build_perturbation_substrate(
    *,
    base_db_path: Path,
    table_name: str,
    static_understanding: dict[str, Any],
    output_dir: Path,
    intensities: list[float],
    repeats: int,
    base_seed: int,
    enabled_families: list[str],
    include_null: bool = True,
    include_boot: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    create_table_sql, columns, base_rows = _load_table(base_db_path, table_name)
    target_column = str(static_understanding.get("target_column") or "")
    if not target_column or target_column not in columns:
        # Conservative fallback: last column is often target in c2-like datasets.
        target_column = columns[-1]

    key_fields = static_understanding.get("key_fields") or []
    feature_columns = _choose_feature_columns(columns, target_column=target_column, key_fields=key_fields)
    safe_missing_columns = _safe_missing_columns(
        columns=columns,
        target_column=target_column,
        feature_columns=feature_columns,
        static_understanding=static_understanding,
    )
    baseline_stats = _family_statistics(
        rows=base_rows,
        target_column=target_column,
        feature_columns=feature_columns,
        safe_missing_columns=safe_missing_columns,
    )

    variants: list[PerturbationVariant] = []
    variants.append(
        PerturbationVariant(
            variant_id="real_base",
            kind="real",
            family_id="real",
            intensity=0.0,
            repeat=0,
            seed=base_seed,
            db_path=base_db_path,
            operators=[],
            notes=["reference_real_dataset"],
        )
    )

    # Optional bootstrap variants.
    if include_boot:
        for repeat in range(1, repeats + 1):
            seed = _stable_seed(base_seed, "boot", repeat)
            db_path = output_dir / f"boot_r{repeat}.sqlite"
            variant = _materialize_variant(
                base_rows=base_rows,
                create_table_sql=create_table_sql,
                columns=columns,
                table_name=table_name,
                output_db_path=db_path,
                family_id="boot",
                kind="boot",
                intensity=0.0,
                repeat=repeat,
                seed=seed,
                target_column=target_column,
                feature_columns=feature_columns,
                safe_missing_columns=safe_missing_columns,
                baseline_stats=baseline_stats,
            )
            variants.append(variant[0])

    if include_null:
        for intensity in intensities:
            for repeat in range(1, repeats + 1):
                seed = _stable_seed(base_seed, "null", intensity, repeat)
                db_path = output_dir / f"null_i{intensity:.2f}_r{repeat}.sqlite"
                variant = _materialize_variant(
                    base_rows=base_rows,
                    create_table_sql=create_table_sql,
                    columns=columns,
                    table_name=table_name,
                    output_db_path=db_path,
                    family_id="null",
                    kind="null",
                    intensity=float(intensity),
                    repeat=repeat,
                    seed=seed,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    safe_missing_columns=safe_missing_columns,
                    baseline_stats=baseline_stats,
                )
                variants.append(variant[0])

    for family_id in enabled_families:
        for intensity in intensities:
            for repeat in range(1, repeats + 1):
                seed = _stable_seed(base_seed, family_id, intensity, repeat)
                db_path = output_dir / f"family_{family_id}_i{intensity:.2f}_r{repeat}.sqlite"
                # Regenerate with alternate seed if validity fails.
                chosen_variant: PerturbationVariant | None = None
                for attempt in range(1, 5):
                    seed_attempt = _stable_seed(seed, "attempt", attempt)
                    db_path_attempt = output_dir / f"family_{family_id}_i{intensity:.2f}_r{repeat}_a{attempt}.sqlite"
                    candidate, _rows = _materialize_variant(
                        base_rows=base_rows,
                        create_table_sql=create_table_sql,
                        columns=columns,
                        table_name=table_name,
                        output_db_path=db_path_attempt,
                        family_id=family_id,
                        kind="family",
                        intensity=float(intensity),
                        repeat=repeat,
                        seed=seed_attempt,
                        target_column=target_column,
                        feature_columns=feature_columns,
                        safe_missing_columns=safe_missing_columns,
                        baseline_stats=baseline_stats,
                    )
                    chosen_variant = candidate
                    if bool((candidate.validity or {}).get("accepted", False)):
                        break
                if chosen_variant is not None:
                    variants.append(chosen_variant)

    valid_family_count = sum(
        1
        for item in variants
        if item.kind == "family" and bool((item.validity or {}).get("accepted", False))
    )
    total_family_count = sum(1 for item in variants if item.kind == "family")

    manifest = {
        "contract_version": "perturbation_manifest_v0_1",
        "base_db_path": str(base_db_path),
        "table_name": table_name,
        "base_seed": base_seed,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "safe_missing_columns": safe_missing_columns,
        "baseline_family_statistics": baseline_stats,
        "intensities": intensities,
        "repeats": repeats,
        "enabled_families": enabled_families,
        "validity_summary": {
            "accepted_family_variant_count": valid_family_count,
            "total_family_variant_count": total_family_count,
            "acceptance_rate": (valid_family_count / total_family_count) if total_family_count else 0.0,
            "offtarget_ratio_max": 0.35,
        },
        "variants": [item.to_dict() for item in variants],
    }
    return manifest
