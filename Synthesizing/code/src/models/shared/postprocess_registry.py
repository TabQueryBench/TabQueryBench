from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


MISSING_TEXT = {"", "__nan__", "nan", "na", "n/a", "none", "null", "<na>", "missing", "unknown", "?"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exact_tokens(field: dict[str, Any] | None) -> set[str] | None:
    """Fields from prepared datasets declare their missing tokens exactly (e.g. only "").

    For those, words such as "unknown" or "none" are real category values and must
    never be treated as missing. Legacy registries keep the permissive MISSING_TEXT.
    """
    if field and field.get("missing_tokens_exact"):
        return {str(x) for x in field.get("missing_tokens") or [""]}
    return None


def _is_missing(value: Any, exact: set[str] | None = None) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if exact is not None:
        return str(value).strip() in exact
    return str(value).strip().lower() in MISSING_TEXT


def _canonical(value: Any, exact: set[str] | None = None) -> str | None:
    if _is_missing(value, exact):
        return None
    if isinstance(value, bool):
        return f"BOOL::{int(value)}"
    text = str(value).strip()
    low = text.lower()
    if low in {"true", "false"}:
        return f"BOOL::{1 if low == 'true' else 0}"
    try:
        num = float(text)
    except Exception:
        return f"STR::{text}"
    if not math.isfinite(num):
        return f"STR::{text}"
    if abs(num - round(num)) < 1e-9:
        return f"INT::{int(round(num))}"
    return f"FLOAT::{num:.12g}"


def _load_registry(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("fields"), list):
        raise ValueError(f"field_registry has no fields list: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_actions(path: Path, actions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action, ensure_ascii=False, sort_keys=True) + "\n")


def _normalize_missing(series: pd.Series, field: dict[str, Any], actions: list[dict[str, Any]]) -> pd.Series:
    exact = _exact_tokens(field)
    before = int(series.isna().sum())
    as_text = series.astype("string")
    if exact is not None:
        mask = series.isna() | as_text.str.strip().isin(exact)
    else:
        missing_tokens = {str(x).strip().lower() for x in field.get("missing_tokens") or []}
        missing_tokens.update(MISSING_TEXT)
        mask = series.isna() | as_text.str.strip().str.lower().isin(missing_tokens)
    if mask.any():
        series = series.copy()
        series.loc[mask] = pd.NA
    changed = int(series.isna().sum()) - before
    if changed > 0:
        actions.append({"action": "preserve_missing_sentinel", "column": field["name"], "changed_cells": changed})
    return series


def _enforce_numeric(series: pd.Series, field: dict[str, Any], actions: list[dict[str, Any]]) -> pd.Series:
    name = field["name"]
    numeric = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & numeric.isna()
    if bad.any():
        examples = series.loc[bad].astype(str).head(8).tolist()
        raise ValueError(f"numeric column {name!r} contains non-numeric values: {examples}")
    finite_bad = numeric.notna() & ~numeric.apply(math.isfinite)
    if finite_bad.any():
        examples = series.loc[finite_bad].astype(str).head(8).tolist()
        raise ValueError(f"numeric column {name!r} contains non-finite values: {examples}")
    return numeric


def _enforce_range(series: pd.Series, field: dict[str, Any], policies: set[str], actions: list[dict[str, Any]]) -> pd.Series:
    name = field["name"]
    lo = field.get("numeric_min")
    hi = field.get("numeric_max")
    if lo is None and hi is None:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    mask = pd.Series(False, index=series.index)
    if lo is not None:
        mask = mask | (numeric < float(lo))
    if hi is not None:
        mask = mask | (numeric > float(hi))
    count = int(mask.fillna(False).sum())
    if count <= 0:
        return series
    if "reject_range" in policies and "clip_range" not in policies:
        examples = series.loc[mask].astype(str).head(8).tolist()
        raise ValueError(f"column {name!r} has {count} out-of-range values: {examples}")
    if "clip_range" in policies:
        clipped = numeric.clip(lower=float(lo) if lo is not None else None, upper=float(hi) if hi is not None else None)
        actions.append({"action": "clip_range", "column": name, "changed_cells": count, "min": lo, "max": hi})
        return clipped
    actions.append({"action": "warn_range", "column": name, "changed_cells": count, "min": lo, "max": hi})
    return series


def _domain_values(field: dict[str, Any], ref: pd.Series) -> list[Any]:
    values = field.get("domain_values") or []
    if values:
        return list(values)
    semantic = str(field.get("semantic_type") or "").lower()
    if semantic in {"categorical", "boolean", "ordinal", "id"}:
        return list(dict.fromkeys(ref.dropna().tolist()))
    return []


def _restore_domain(series: pd.Series, ref: pd.Series, field: dict[str, Any], actions: list[dict[str, Any]]) -> pd.Series:
    name = field["name"]
    exact = _exact_tokens(field)

    def canon(value: Any) -> str | None:
        return _canonical(value, exact)

    values = _domain_values(field, ref)
    if not values:
        return series
    canonical_to_value = {}
    for value in values:
        key = canon(value)
        if key is not None and key not in canonical_to_value:
            canonical_to_value[key] = value
    normalized = series.map(canon)
    bad = normalized.notna() & ~normalized.isin(set(canonical_to_value))
    if bad.any():
        numeric_codes = pd.to_numeric(series.loc[bad], errors="coerce")
        numeric_mask = numeric_codes.notna()
        if numeric_mask.any():
            bad_index = numeric_codes.index[numeric_mask]
            clipped = numeric_codes.loc[bad_index].round().clip(lower=0, upper=len(values) - 1).astype(int)
            series = series.copy()
            series.loc[bad_index] = [values[i] for i in clipped.tolist()]
            actions.append({"action": "restore_domain_encoded_index", "column": name, "changed_cells": int(len(bad_index))})
            normalized = series.map(canon)
            bad = normalized.notna() & ~normalized.isin(set(canonical_to_value))
    if bad.any():
        examples = series.loc[bad].astype(str).head(8).tolist()
        raise ValueError(f"categorical/domain column {name!r} has unmapped values: {examples}")
    restored = series.astype("object").copy()
    changed = 0
    for key, original in canonical_to_value.items():
        mask = normalized == key
        if mask.any():
            before = restored.loc[mask].astype(str)
            restored.loc[mask] = original
            after = restored.loc[mask].astype(str)
            changed += int((before != after).sum())
    if changed:
        actions.append({"action": "restore_domain", "column": name, "changed_cells": changed})
    return restored


def quality_warnings(ref_df: pd.DataFrame, out_df: pd.DataFrame, field_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Cheap distribution sanity checks. They never reject a run (short smoke runs are
    expected to be poor); they flag outputs that are structurally broken, e.g. numeric
    values collapsed onto the training min/max, or a categorical column far from the real mix."""
    warnings: list[dict[str, Any]] = []
    for col, field in field_map.items():
        if col not in out_df.columns or col not in ref_df.columns:
            continue
        semantic = str(field.get("semantic_type") or "").lower()
        real, syn = ref_df[col], out_df[col]
        real_null, syn_null = float(real.isna().mean()), float(syn.isna().mean())
        if abs(syn_null - real_null) > 0.25:
            warnings.append({"check": "missing_rate", "column": col, "real": round(real_null, 4), "synthetic": round(syn_null, 4)})
        if semantic in {"continuous", "integer"} or field.get("numeric_like"):
            r = pd.to_numeric(real, errors="coerce").dropna()
            s = pd.to_numeric(syn, errors="coerce").dropna()
            if len(r) < 2 or len(s) == 0 or r.min() == r.max():
                continue
            lo, hi = r.min(), r.max()
            real_edge = float(((r == lo) | (r == hi)).mean())
            syn_edge = float(((s <= lo) | (s >= hi)).mean())
            if syn_edge > max(0.5, real_edge + 0.4):
                warnings.append({"check": "numeric_boundary_pileup", "column": col, "real_fraction_at_min_max": round(real_edge, 4), "synthetic_fraction_at_min_max": round(syn_edge, 4)})
            if s.nunique() == 1 and r.nunique() > 1:
                warnings.append({"check": "numeric_constant", "column": col, "value": float(s.iloc[0])})
        elif semantic in {"categorical", "boolean", "ordinal", "id", "multilabel"}:
            r = real.dropna().astype(str).value_counts(normalize=True)
            s = syn.dropna().astype(str).value_counts(normalize=True)
            if r.empty or s.empty:
                continue
            if len(r) * 10 > syn.notna().sum():
                # Too many levels for the sample size: TVD would be dominated by sampling noise.
                continue
            tvd = 0.5 * float(r.subtract(s, fill_value=0).abs().sum())
            if tvd > 0.5:
                warnings.append({"check": "categorical_tvd", "column": col, "tvd": round(tvd, 4)})
    return warnings


def apply_field_registry_postprocess(
    *,
    model_name: str,
    dataset_id: str | None,
    output_csv: Path,
    ref_df: pd.DataFrame,
    out_df: pd.DataFrame,
    field_registry_path: Path,
    expected_rows: int,
) -> Path:
    registry = _load_registry(field_registry_path)
    expected_cols = list(ref_df.columns)
    status_path = output_csv.with_name("postprocess_run_status.json")
    actions_path = output_csv.with_name("postprocess_actions.jsonl")
    actions: list[dict[str, Any]] = []
    started_at = _now()
    try:
        missing = [col for col in expected_cols if col not in out_df.columns]
        extra = [col for col in out_df.columns if col not in expected_cols]
        if missing or extra:
            raise ValueError(f"generated CSV schema mismatch; missing={missing}, extra={extra}")
        if len(out_df) != int(expected_rows):
            raise ValueError(f"generated row count mismatch; got={len(out_df)}, expected={int(expected_rows)}")
        out_df = out_df[expected_cols].copy()

        field_map = {str(field.get("name")): field for field in registry.get("fields") or [] if field.get("name")}
        if set(expected_cols) - set(field_map):
            raise ValueError(f"field_registry missing columns: {sorted(set(expected_cols) - set(field_map))[:20]}")

        for col in expected_cols:
            field = field_map[col]
            policies = {str(item) for item in field.get("postprocess_policy") or []}
            semantic = str(field.get("semantic_type") or "").lower()
            series = out_df[col]
            if "preserve_missing_sentinel" in policies:
                series = _normalize_missing(series, field, actions)
            if semantic in {"continuous", "integer"} or field.get("numeric_like"):
                series = _enforce_numeric(series, field, actions)
                series = _enforce_range(series, field, policies, actions)
            if "round_integer" in policies or semantic == "integer" or field.get("integer_like"):
                numeric = pd.to_numeric(series, errors="coerce")
                bad = series.notna() & numeric.isna()
                if bad.any():
                    examples = series.loc[bad].astype(str).head(8).tolist()
                    raise ValueError(f"integer column {col!r} contains non-numeric values: {examples}")
                rounded = numeric.round().astype("Int64")
                changed = int(((numeric.notna()) & ((numeric - rounded.astype(float)).abs() > 1e-6)).sum())
                if changed:
                    actions.append({"action": "round_integer", "column": col, "changed_cells": changed})
                series = rounded
            if "restore_domain" in policies or semantic in {"categorical", "boolean", "ordinal", "id", "multilabel"}:
                series = _restore_domain(series, ref_df[col], field, actions)
            out_df[col] = series

        out_df.to_csv(output_csv, index=False, encoding="utf-8")
        try:
            quality = quality_warnings(ref_df, out_df, field_map)
        except Exception as exc:  # quality checks must never break generation
            quality = [{"check": "quality_check_error", "error": str(exc)}]
        summary = {
            "status": "accepted",
            "quality_warnings": quality,
            "model": model_name,
            "dataset_id": dataset_id or registry.get("dataset_id"),
            "output_csv": str(output_csv),
            "field_registry": str(field_registry_path),
            "expected_rows": int(expected_rows),
            "actual_rows": int(len(out_df)),
            "started_at_utc": started_at,
            "finished_at_utc": _now(),
            "action_count": len(actions),
        }
        _write_actions(actions_path, actions)
        _write_json(status_path, summary)
        return output_csv
    except Exception as exc:
        _write_actions(actions_path, actions)
        _write_json(
            status_path,
            {
                "status": "rejected",
                "model": model_name,
                "dataset_id": dataset_id or registry.get("dataset_id"),
                "output_csv": str(output_csv),
                "field_registry": str(field_registry_path),
                "expected_rows": int(expected_rows),
                "actual_rows": int(len(out_df)) if out_df is not None else None,
                "started_at_utc": started_at,
                "finished_at_utc": _now(),
                "error": str(exc),
                "action_count": len(actions),
            },
        )
        raise
