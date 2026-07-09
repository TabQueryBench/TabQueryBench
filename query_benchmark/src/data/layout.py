"""Dataset layout helpers for canonical and legacy asset resolution."""

from __future__ import annotations

from pathlib import Path

from src.config.settings import DATA_DIR

DATASET_CORE_ASSET_NAMES = {
    "main_csv",
    "dataset_profile",
    "dataset_contract",
    "dataset_description",
    "dataset_semantics",
    "field_registry",
    "query_policy",
    "validation_policy",
    "source_info",
}

DATASET_OPTIONAL_ASSET_NAMES = {
    "family_applicability",
    "risk_register",
    "uncertainty_register",
}


def dataset_dir(dataset_id: str, data_root: Path = DATA_DIR) -> Path:
    return data_root / dataset_id


def dataset_raw_dir(dataset_id: str, data_root: Path = DATA_DIR) -> Path:
    return dataset_dir(dataset_id, data_root) / "raw"


def dataset_source_dir(dataset_id: str, data_root: Path = DATA_DIR) -> Path:
    return dataset_dir(dataset_id, data_root) / "source"


def dataset_contracts_dir(dataset_id: str, data_root: Path = DATA_DIR) -> Path:
    return dataset_dir(dataset_id, data_root) / "contracts"


def dataset_metadata_core_dir(dataset_id: str, data_root: Path = DATA_DIR) -> Path:
    return dataset_dir(dataset_id, data_root) / "metadata_core"


def dataset_metadata_optional_dir(dataset_id: str, data_root: Path = DATA_DIR) -> Path:
    return dataset_dir(dataset_id, data_root) / "metadata_optional"


def dataset_legacy_metadata_dir(dataset_id: str, data_root: Path = DATA_DIR) -> Path:
    return dataset_dir(dataset_id, data_root) / "metadata"


def dataset_legacy_existing_artifacts_dir(dataset_id: str, data_root: Path = DATA_DIR) -> Path:
    return dataset_dir(dataset_id, data_root) / "existing_artifacts"


def dataset_global_artifacts_dir(dataset_id: str, data_root: Path = DATA_DIR) -> Path:
    return data_root / "artifacts" / "data_core" / "tabular" / dataset_id


def dataset_asset_candidates(dataset_id: str, asset_name: str, data_root: Path = DATA_DIR) -> list[Path]:
    ds_dir = dataset_dir(dataset_id, data_root)
    raw_dir = dataset_raw_dir(dataset_id, data_root)
    source_dir = dataset_source_dir(dataset_id, data_root)
    contracts_dir = dataset_contracts_dir(dataset_id, data_root)
    metadata_core_dir = dataset_metadata_core_dir(dataset_id, data_root)
    metadata_optional_dir = dataset_metadata_optional_dir(dataset_id, data_root)
    legacy_metadata_dir = dataset_legacy_metadata_dir(dataset_id, data_root)
    legacy_existing_artifacts_dir = dataset_legacy_existing_artifacts_dir(dataset_id, data_root)
    global_artifacts_dir = dataset_global_artifacts_dir(dataset_id, data_root)

    if asset_name == "main_csv":
        return [
            raw_dir / f"{dataset_id}-main.csv",
            ds_dir / f"{dataset_id}-main.csv",
        ]
    if asset_name == "dataset_profile":
        return [
            contracts_dir / "dataset_profile.json",
            legacy_existing_artifacts_dir / f"{dataset_id}-dataset_profile.json",
            global_artifacts_dir / f"{dataset_id}-dataset_profile.json",
        ]
    if asset_name == "dataset_contract":
        return [
            contracts_dir / "dataset_contract_v1.json",
            legacy_existing_artifacts_dir / f"{dataset_id}-dataset_contract_v1.json",
            global_artifacts_dir / f"{dataset_id}-dataset_contract_v1.json",
        ]
    if asset_name == "dataset_description":
        return [
            metadata_core_dir / "dataset_description.txt",
            legacy_metadata_dir / "dataset_description.txt",
        ]
    if asset_name == "dataset_semantics":
        return [
            metadata_core_dir / "dataset_semantics.yaml",
            legacy_metadata_dir / "dataset_semantics.yaml",
        ]
    if asset_name == "field_registry":
        return [
            metadata_core_dir / "field_registry.json",
            legacy_metadata_dir / "field_registry.json",
        ]
    if asset_name == "query_policy":
        return [
            metadata_core_dir / "query_policy.yaml",
            legacy_metadata_dir / "query_policy.yaml",
        ]
    if asset_name == "validation_policy":
        return [
            metadata_core_dir / "validation_policy.yaml",
            legacy_metadata_dir / "validation_policy.yaml",
        ]
    if asset_name == "family_applicability":
        return [
            metadata_optional_dir / "family_applicability.json",
            legacy_metadata_dir / "family_applicability.json",
        ]
    if asset_name == "risk_register":
        return [
            metadata_optional_dir / "risk_register.json",
            legacy_metadata_dir / "risk_register.json",
        ]
    if asset_name == "uncertainty_register":
        return [
            metadata_optional_dir / "uncertainty_register.json",
            legacy_metadata_dir / "uncertainty_register.json",
        ]
    if asset_name == "source_info":
        return [source_dir / "source_info.json"]
    raise KeyError(f"Unsupported dataset asset: {asset_name}")


def resolve_dataset_asset(dataset_id: str, asset_name: str, data_root: Path = DATA_DIR) -> Path | None:
    return next((path for path in dataset_asset_candidates(dataset_id, asset_name, data_root) if path.exists()), None)


def canonical_dataset_asset_path(dataset_id: str, asset_name: str, data_root: Path = DATA_DIR) -> Path:
    ds_dir = dataset_dir(dataset_id, data_root)
    if asset_name == "main_csv":
        return dataset_raw_dir(dataset_id, data_root) / f"{dataset_id}-main.csv"
    if asset_name == "dataset_profile":
        return dataset_contracts_dir(dataset_id, data_root) / "dataset_profile.json"
    if asset_name == "dataset_contract":
        return dataset_contracts_dir(dataset_id, data_root) / "dataset_contract_v1.json"
    if asset_name == "dataset_description":
        return dataset_metadata_core_dir(dataset_id, data_root) / "dataset_description.txt"
    if asset_name == "dataset_semantics":
        return dataset_metadata_core_dir(dataset_id, data_root) / "dataset_semantics.yaml"
    if asset_name == "field_registry":
        return dataset_metadata_core_dir(dataset_id, data_root) / "field_registry.json"
    if asset_name == "query_policy":
        return dataset_metadata_core_dir(dataset_id, data_root) / "query_policy.yaml"
    if asset_name == "validation_policy":
        return dataset_metadata_core_dir(dataset_id, data_root) / "validation_policy.yaml"
    if asset_name == "family_applicability":
        return dataset_metadata_optional_dir(dataset_id, data_root) / "family_applicability.json"
    if asset_name == "risk_register":
        return dataset_metadata_optional_dir(dataset_id, data_root) / "risk_register.json"
    if asset_name == "uncertainty_register":
        return dataset_metadata_optional_dir(dataset_id, data_root) / "uncertainty_register.json"
    if asset_name == "source_info":
        return dataset_source_dir(dataset_id, data_root) / "source_info.json"
    raise KeyError(f"Unsupported dataset asset: {asset_name}")
