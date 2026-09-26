#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_recommendations(path: Path) -> dict[str, dict[str, str]]:
    by_dataset: dict[str, dict[str, str]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = str(row.get("dataset") or "").strip()
            column = str(row.get("column") or "").strip()
            suggested = str(row.get("suggested_semantic_type") or "").strip().lower()
            if not dataset or not column or suggested not in {"categorical", "boolean"}:
                continue
            by_dataset[dataset][column] = suggested
    return by_dataset


def _apply_contract_fix(contract: dict[str, Any], fixes: dict[str, str]) -> dict[str, Any]:
    changed: dict[str, str] = {}
    for column in contract.get("columns") or []:
        if not isinstance(column, dict):
            continue
        name = str(column.get("name") or "")
        suggested = fixes.get(name)
        if not suggested:
            continue
        if column.get("semantic_type") != suggested:
            column["semantic_type"] = suggested
            changed[name] = suggested
        # For low-cardinality categorical-like columns, median imputation is not appropriate.
        if suggested in {"categorical", "boolean"} and column.get("impute_strategy") != "mode":
            column["impute_strategy"] = "mode"
    notes = contract.setdefault("notes", [])
    note = "semantic_type corrections applied from discrete-issue review (2026-05-18)"
    if isinstance(notes, list) and note not in notes and changed:
        notes.append(note)
    return changed


def _apply_profile_fix(profile: dict[str, Any], fixes: dict[str, str]) -> dict[str, str]:
    changed: dict[str, str] = {}
    column_profiles = profile.get("column_profiles") or {}
    for name, col_profile in column_profiles.items():
        if not isinstance(col_profile, dict):
            continue
        suggested = fixes.get(str(name))
        if not suggested:
            continue
        inferred = "boolean" if suggested == "boolean" else "categorical"
        if col_profile.get("inferred_type") != inferred:
            col_profile["inferred_type"] = inferred
            changed[str(name)] = inferred
        warnings = col_profile.get("warnings")
        if isinstance(warnings, list):
            marker = "semantic_type_override_applied_20260518"
            if marker not in warnings:
                warnings.append(marker)
    warnings = profile.get("warnings")
    if isinstance(warnings, list):
        marker = "semantic_type corrections applied from discrete-issue review (2026-05-18)"
        if marker not in warnings and changed:
            warnings.append(marker)
    return changed


def _fix_dataset(root: Path, dataset: str, fixes: dict[str, str]) -> dict[str, Any]:
    base = root / "artifacts" / "data_core" / "tabular" / dataset
    contract_path = base / f"{dataset}-dataset_contract_v1.json"
    profile_path = base / f"{dataset}-dataset_profile.json"
    result = {
        "dataset": dataset,
        "root": str(root),
        "contract_path": str(contract_path),
        "profile_path": str(profile_path),
        "requested_column_count": len(fixes),
        "contract_changed_columns": [],
        "profile_changed_columns": [],
        "status": "ok",
    }
    if not contract_path.exists() or not profile_path.exists():
        result["status"] = "missing_artifacts"
        return result

    contract = _load_json(contract_path)
    profile = _load_json(profile_path)
    changed_contract = _apply_contract_fix(contract, fixes)
    changed_profile = _apply_profile_fix(profile, fixes)
    _dump_json(contract_path, contract)
    _dump_json(profile_path, profile)
    result["contract_changed_columns"] = sorted(changed_contract)
    result["profile_changed_columns"] = sorted(changed_profile)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root, e.g. data_train_as_main or /data/.../DatasetNew")
    parser.add_argument("--recommendations", required=True, help="CSV from preprocessing_issue_column_fixes_20260518")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rec_path = Path(args.recommendations).resolve()
    out_path = Path(args.out).resolve()

    recs = _load_recommendations(rec_path)
    results = [_fix_dataset(root, dataset, fixes) for dataset, fixes in sorted(recs.items())]
    summary = {
        "root": str(root),
        "dataset_count": len(results),
        "requested_column_fix_count": sum(item["requested_column_count"] for item in results),
        "contract_changed_column_count": sum(len(item["contract_changed_columns"]) for item in results),
        "profile_changed_column_count": sum(len(item["profile_changed_columns"]) for item in results),
        "missing_artifact_dataset_count": sum(1 for item in results if item["status"] == "missing_artifacts"),
        "datasets": results,
    }
    _dump_json(out_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
