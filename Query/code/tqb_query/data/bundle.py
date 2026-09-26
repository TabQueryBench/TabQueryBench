"""Standardized dataset bundle loader for SQL QA runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tqb_query.data.layout import (
    dataset_asset_candidates,
    dataset_contracts_dir,
    dataset_dir as resolve_dataset_dir,
    dataset_legacy_existing_artifacts_dir,
    dataset_legacy_metadata_dir,
    dataset_metadata_core_dir,
    dataset_metadata_optional_dir,
    dataset_raw_dir,
    dataset_source_dir,
    dataset_global_artifacts_dir,
    resolve_dataset_asset,
)


@dataclass
class DatasetBundle:
    dataset_id: str
    dataset_dir: Path
    raw_dir: Path
    metadata_dir: Path
    source_dir: Path
    existing_artifacts_dir: Path
    contracts_dir: Path
    metadata_core_dir: Path
    metadata_optional_dir: Path
    main_csv_path: Path
    dataset_profile_path: Path
    dataset_contract_path: Path
    dataset_description_path: Path
    dataset_semantics_path: Path
    field_registry_path: Path
    family_applicability_path: Path
    query_policy_path: Path
    validation_policy_path: Path
    risk_register_path: Path
    uncertainty_register_path: Path
    source_info_path: Path
    dataset_profile: dict[str, Any]
    dataset_contract: dict[str, Any]
    dataset_description: str
    dataset_semantics: dict[str, Any]
    field_registry: dict[str, Any]
    family_applicability: dict[str, Any]
    query_policy: dict[str, Any]
    validation_policy: dict[str, Any]
    risk_register: dict[str, Any]
    uncertainty_register: dict[str, Any]
    source_info: dict[str, Any]
    raw_csv_files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def loaded_files_summary(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_dir": str(self.dataset_dir),
            "contracts_dir": str(self.contracts_dir),
            "metadata_core_dir": str(self.metadata_core_dir),
            "metadata_optional_dir": str(self.metadata_optional_dir),
            "legacy_metadata_dir": str(self.metadata_dir),
            "legacy_existing_artifacts_dir": str(self.existing_artifacts_dir),
            "raw_csv_files": [str(p) for p in self.raw_csv_files],
            "main_csv_path": str(self.main_csv_path),
            "dataset_profile_path": str(self.dataset_profile_path),
            "dataset_contract_path": str(self.dataset_contract_path),
            "dataset_description_path": str(self.dataset_description_path),
            "dataset_semantics_path": str(self.dataset_semantics_path),
            "field_registry_path": str(self.field_registry_path),
            "family_applicability_path": str(self.family_applicability_path),
            "query_policy_path": str(self.query_policy_path),
            "validation_policy_path": str(self.validation_policy_path),
            "risk_register_path": str(self.risk_register_path),
            "uncertainty_register_path": str(self.uncertainty_register_path),
            "source_info_path": str(self.source_info_path),
            "warnings": self.warnings,
        }


def _resolve_first_existing(candidates: list[Path], label: str, strict: bool) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    if strict:
        lines = "\n".join(f"- {p}" for p in candidates)
        raise FileNotFoundError(f"Missing required {label}. Checked:\n{lines}")
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _collect_raw_csv_files(dataset_id: str, dataset_dir: Path, raw_dir: Path) -> list[Path]:
    raw_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if raw_files:
        return raw_files

    # Fallback for layouts where CSVs are placed directly under dataset directory.
    direct_files = sorted(dataset_dir.glob("*.csv"))
    if direct_files:
        return direct_files

    raise FileNotFoundError(f"No CSV files found for dataset {dataset_id} in {raw_dir} or {dataset_dir}.")


def load_dataset_bundle(dataset_id: str, data_root: Path, strict: bool = True) -> DatasetBundle:
    resolved_dataset_dir = resolve_dataset_dir(dataset_id, data_root)
    if not resolved_dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_dataset_dir}")

    raw_dir = dataset_raw_dir(dataset_id, data_root)
    metadata_dir = dataset_legacy_metadata_dir(dataset_id, data_root)
    source_dir = dataset_source_dir(dataset_id, data_root)
    existing_artifacts_dir = dataset_legacy_existing_artifacts_dir(dataset_id, data_root)
    contracts_dir = dataset_contracts_dir(dataset_id, data_root)
    metadata_core_dir = dataset_metadata_core_dir(dataset_id, data_root)
    metadata_optional_dir = dataset_metadata_optional_dir(dataset_id, data_root)
    global_artifacts_dir = dataset_global_artifacts_dir(dataset_id, data_root)

    warnings: list[str] = []
    if not metadata_dir.exists() and not metadata_core_dir.exists():
        msg = f"Metadata directories not found: {metadata_dir} and {metadata_core_dir}"
        if strict:
            raise FileNotFoundError(msg)
        warnings.append(msg)

    raw_csv_files = _collect_raw_csv_files(dataset_id, resolved_dataset_dir, raw_dir)
    main_csv_path = _resolve_first_existing(
        [path for path in [resolve_dataset_asset(dataset_id, "main_csv", data_root)] if path is not None],
        label="main CSV",
        strict=True,
    )
    assert main_csv_path is not None

    dataset_profile_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "dataset_profile", data_root),
        label="dataset profile JSON",
        strict=strict,
    )
    dataset_contract_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "dataset_contract", data_root),
        label="dataset contract JSON",
        strict=strict,
    )
    dataset_description_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "dataset_description", data_root),
        label="dataset description",
        strict=strict,
    )
    dataset_semantics_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "dataset_semantics", data_root),
        label="dataset semantics",
        strict=strict,
    )
    field_registry_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "field_registry", data_root),
        label="field registry",
        strict=strict,
    )
    query_policy_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "query_policy", data_root),
        label="query policy",
        strict=strict,
    )
    family_applicability_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "family_applicability", data_root),
        label="family applicability",
        strict=False,
    )
    validation_policy_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "validation_policy", data_root),
        label="validation policy",
        strict=strict,
    )
    risk_register_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "risk_register", data_root),
        label="risk register",
        strict=False,
    )
    uncertainty_register_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "uncertainty_register", data_root),
        label="uncertainty register",
        strict=False,
    )
    source_info_path = _resolve_first_existing(
        dataset_asset_candidates(dataset_id, "source_info", data_root),
        label="source info",
        strict=strict,
    )

    # Best-effort mode for non-critical files when strict=False.
    dataset_profile = _load_json(dataset_profile_path) if dataset_profile_path else {}
    dataset_contract = _load_json(dataset_contract_path) if dataset_contract_path else {}
    dataset_description = _load_text(dataset_description_path) if dataset_description_path else ""
    dataset_semantics = _load_yaml(dataset_semantics_path) if dataset_semantics_path else {}
    field_registry = _load_json(field_registry_path) if field_registry_path else {}
    family_applicability = _load_json(family_applicability_path) if family_applicability_path else {}
    query_policy = _load_yaml(query_policy_path) if query_policy_path else {}
    validation_policy = _load_yaml(validation_policy_path) if validation_policy_path else {}
    risk_register = _load_json(risk_register_path) if risk_register_path else {}
    uncertainty_register = _load_json(uncertainty_register_path) if uncertainty_register_path else {}
    source_info = _load_json(source_info_path) if source_info_path else {}

    if not dataset_profile_path:
        warnings.append("dataset_profile missing; loaded empty object.")
    if not dataset_contract_path:
        warnings.append("dataset_contract missing; loaded empty object.")
    if not family_applicability_path:
        warnings.append("family_applicability missing; loaded empty object.")
    if not risk_register_path:
        warnings.append("risk_register missing; loaded empty object.")
    if not uncertainty_register_path:
        warnings.append("uncertainty_register missing; loaded empty object.")

    return DatasetBundle(
        dataset_id=dataset_id,
        dataset_dir=resolved_dataset_dir,
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
        source_dir=source_dir,
        existing_artifacts_dir=existing_artifacts_dir,
        contracts_dir=contracts_dir,
        metadata_core_dir=metadata_core_dir,
        metadata_optional_dir=metadata_optional_dir,
        main_csv_path=main_csv_path,
        dataset_profile_path=dataset_profile_path or Path(""),
        dataset_contract_path=dataset_contract_path or Path(""),
        dataset_description_path=dataset_description_path or Path(""),
        dataset_semantics_path=dataset_semantics_path or Path(""),
        field_registry_path=field_registry_path or Path(""),
        family_applicability_path=family_applicability_path or Path(""),
        query_policy_path=query_policy_path or Path(""),
        validation_policy_path=validation_policy_path or Path(""),
        risk_register_path=risk_register_path or Path(""),
        uncertainty_register_path=uncertainty_register_path or Path(""),
        source_info_path=source_info_path or Path(""),
        dataset_profile=dataset_profile,
        dataset_contract=dataset_contract,
        dataset_description=dataset_description,
        dataset_semantics=dataset_semantics,
        field_registry=field_registry,
        family_applicability=family_applicability,
        query_policy=query_policy,
        validation_policy=validation_policy,
        risk_register=risk_register,
        uncertainty_register=uncertainty_register,
        source_info=source_info,
        raw_csv_files=raw_csv_files,
        warnings=warnings,
    )
