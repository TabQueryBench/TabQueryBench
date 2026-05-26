#!/usr/bin/env python3
"""Fix boolean semantic misclassifications for n1 and n8 preprocessing assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
F_PREPROC_ROOT = Path(r"F:\TabQueryBench\Data_HF\02_preprocessing_outputs")
BOOL_TOKENS = {"0", "1", "0.0", "1.0", "true", "false", "t", "f", "yes", "no", "y", "n"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _normalize_values(series: pd.Series) -> set[str]:
    values: set[str] = set()
    for raw in series.dropna().tolist():
        token = str(raw).strip().lower()
        if token:
            values.add(token)
    return values


def _values_fit_boolean_domain(values: set[str]) -> bool:
    return bool(values) and values.issubset(BOOL_TOKENS)


def _determine_bad_boolean_columns(dataset_id: str) -> dict[str, dict[str, Any]]:
    field_registry_path = PROJECT_ROOT / "data" / dataset_id / "metadata_core" / "field_registry.json"
    train_csv_path = PROJECT_ROOT / "data" / dataset_id / f"{dataset_id}-train.csv"
    field_registry = _load_json(field_registry_path)
    df = pd.read_csv(train_csv_path)

    bad_columns: dict[str, dict[str, Any]] = {}
    for field in field_registry.get("fields") or []:
        if field.get("declared_type") != "boolean":
            continue
        name = str(field.get("name") or "")
        if not name or name not in df.columns:
            continue
        values = _normalize_values(df[name])
        if _values_fit_boolean_domain(values):
            continue
        bad_columns[name] = {
            "unique_count": len(values),
            "example_values": sorted(values)[:10],
        }
    return bad_columns


def _update_contract_overrides(dataset_id: str, bad_columns: dict[str, dict[str, Any]]) -> None:
    metadata_dir = PROJECT_ROOT / "data" / dataset_id / "metadata"
    path = metadata_dir / "contract_overrides.json"
    payload = _load_json(path) if path.exists() else {"dataset_id": dataset_id, "status": "active", "overrides": {}}
    overrides = payload.setdefault("overrides", {})
    semantic_overrides = overrides.setdefault("semantic_type_overrides", {})
    for name in sorted(bad_columns):
        semantic_overrides[name] = "numeric"
    _dump_json(path, payload)


def _update_preprocessing_fixes(dataset_id: str, bad_columns: dict[str, dict[str, Any]]) -> None:
    metadata_dir = PROJECT_ROOT / "data" / dataset_id / "metadata"
    path = metadata_dir / "preprocessing_fixes.yaml"
    payload = _load_yaml(path) if path.exists() else {"dataset_id": dataset_id, "applied_fixes": []}
    if payload is None:
        payload = {"dataset_id": dataset_id, "applied_fixes": []}
    fixes = payload.setdefault("applied_fixes", [])
    fix_id = f"{dataset_id}_fix_boolean_domain_override"
    summary = (
        "Corrected non-boolean-valued fields that had been typed as boolean via train-domain validation: "
        + ", ".join(sorted(bad_columns))
        + "."
    )
    replacement = {
        "fix_id": fix_id,
        "status": "applied",
        "summary": summary,
    }
    updated = False
    for idx, item in enumerate(fixes):
        if str(item.get("fix_id") or "") == fix_id:
            fixes[idx] = replacement
            updated = True
            break
    if not updated:
        fixes.append(replacement)
    _dump_yaml(path, payload)


def _update_dataset_semantics(dataset_id: str, bad_columns: dict[str, dict[str, Any]]) -> None:
    path = PROJECT_ROOT / "data" / dataset_id / "metadata_core" / "dataset_semantics.yaml"
    payload = _load_yaml(path) or {}
    notes = list(payload.get("notes") or [])
    note = (
        "Resolved in package: fields with non-boolean train-domain values are forced to numeric after "
        f"value-domain validation ({', '.join(sorted(bad_columns))})."
    )
    if note not in notes:
        notes.append(note)
    payload["notes"] = notes
    _dump_yaml(path, payload)


def _update_contract_and_profile(dataset_id: str, bad_columns: dict[str, dict[str, Any]]) -> None:
    base = PROJECT_ROOT / "data" / "artifacts" / "data_core" / "tabular" / dataset_id
    contract_path = base / f"{dataset_id}-dataset_contract_v1.json"
    profile_path = base / f"{dataset_id}-dataset_profile.json"
    contract = _load_json(contract_path)
    profile = _load_json(profile_path)

    for column in contract.get("columns") or []:
        name = str(column.get("name") or "")
        if name in bad_columns:
            column["semantic_type"] = "numeric"

    for name, column_profile in (profile.get("column_profiles") or {}).items():
        if name in bad_columns:
            column_profile["inferred_type"] = "numerical"

    _dump_json(contract_path, contract)
    _dump_json(profile_path, profile)


def _update_field_registry(field_registry_path: Path, bad_columns: dict[str, dict[str, Any]]) -> None:
    payload = _load_json(field_registry_path)
    for field in payload.get("fields") or []:
        name = str(field.get("name") or "")
        if name not in bad_columns:
            continue
        field["declared_type"] = "numeric"
        if field.get("semantic_type") == "categorical_binary":
            field["semantic_type"] = "numeric_discrete"
        for evidence in field.get("evidence") or []:
            if evidence.get("type") == "contract":
                evidence["detail"] = "role=feature, semantic_type=numeric"
            elif evidence.get("type") == "profile":
                detail = str(evidence.get("detail") or "")
                unique_part = ""
                missing_part = ""
                for fragment in detail.split(","):
                    fragment = fragment.strip()
                    if fragment.startswith("unique_count="):
                        unique_part = fragment
                    elif fragment.startswith("missing_rate="):
                        missing_part = fragment
                pieces = ["inferred_type=numerical"]
                if unique_part:
                    pieces.append(unique_part)
                if missing_part:
                    pieces.append(missing_part)
                evidence["detail"] = ", ".join(pieces)
    _dump_json(field_registry_path, payload)


def _sync_to_f_drive(dataset_id: str) -> None:
    src_root = PROJECT_ROOT / "data" / dataset_id
    dst_root = F_PREPROC_ROOT / dataset_id
    copies = [
        ("metadata/contract_overrides.json", "metadata/contract_overrides.json"),
        ("metadata/preprocessing_fixes.yaml", "metadata/preprocessing_fixes.yaml"),
        ("metadata_core/field_registry.json", "metadata_core/field_registry.json"),
        ("metadata_core/dataset_semantics.yaml", "metadata_core/dataset_semantics.yaml"),
    ]
    for src_rel, dst_rel in copies:
        src = src_root / src_rel
        if not src.exists():
            continue
        dst = dst_root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    report: dict[str, Any] = {"datasets": {}}
    for dataset_id in ("n1", "n8"):
        bad_columns = _determine_bad_boolean_columns(dataset_id)
        _update_contract_overrides(dataset_id, bad_columns)
        _update_preprocessing_fixes(dataset_id, bad_columns)
        _update_dataset_semantics(dataset_id, bad_columns)
        _update_contract_and_profile(dataset_id, bad_columns)
        _update_field_registry(PROJECT_ROOT / "data" / dataset_id / "metadata_core" / "field_registry.json", bad_columns)
        _sync_to_f_drive(dataset_id)
        report["datasets"][dataset_id] = {
            "corrected_boolean_to_numeric_columns": sorted(bad_columns),
            "count": len(bad_columns),
        }
    report_path = PROJECT_ROOT / "tmp" / "n1_n8_preprocessing_semantic_fix_report_20260509.json"
    _dump_json(report_path, report)


if __name__ == "__main__":
    main()
