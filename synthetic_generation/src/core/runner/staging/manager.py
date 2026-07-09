import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..dataset_loader import DatasetLoader


class StagingError(RuntimeError):
    def __init__(self, reason_code: str, detail: str):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass
class PublicGateResult:
    dataset_id: str
    target_column: str
    task_type: str
    column_schema: List[Dict[str, Any]]
    public_dir: Path
    features_json: Path
    report_json: Path
    normalized_schema_json: Path
    staged_manifest_json: Path
    staged_train_csv: Path
    staged_val_csv: Path
    staged_test_csv: Path


class AdapterStagingManager:
    DEFAULT_MISSING_TOKENS = ["", "?", "na", "n/a", "nan", "null", "none", "NA", "NULL", "NaN"]

    @staticmethod
    def _looks_like_field_registry(data: Dict[str, Any]) -> bool:
        version = str(data.get("schema_version") or "").strip().lower()
        if version.startswith("dataset_understanding_field_registry"):
            return True
        fields = data.get("fields")
        return isinstance(fields, list) and bool(fields)

    def __init__(self, loader: DatasetLoader):
        self.loader = loader

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _load_contract(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise StagingError("contract_missing", f"contract file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema_version") == "profile_contract_v1":
            return data
        if self._looks_like_field_registry(data):
            return self._contract_from_field_registry(data)
        raise StagingError(
            "contract_schema_invalid",
            f"schema_version expected profile_contract_v1, got {data.get('schema_version')}",
        )

        return data

    def _load_profile(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise StagingError("profile_missing", f"profile file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if self._looks_like_field_registry(data):
            fields = data.get("fields") or []
            columns = [str(field.get("name") or "").strip() for field in fields]
            columns = [name for name in columns if name]
            if not columns:
                raise StagingError("profile_missing", "field_registry contains no usable columns")
            return {"columns": columns}
        return data

    def _normalize_field_registry_semantic_type(self, field: Dict[str, Any]) -> str:
        semantic = str(field.get("semantic_type") or "").strip().lower()
        declared = str(field.get("declared_type") or "").strip().lower()
        raw = semantic or declared

        if raw in {"boolean", "categorical_binary", "binary"} or declared == "boolean":
            return "boolean"
        if "datetime" in raw or raw in {"timestamp", "date", "time"}:
            return "datetime"
        if raw in {"id", "id_like"} or declared in {"id", "id_like"}:
            return "id"
        if raw.startswith("numeric") or raw in {"integer", "continuous"} or declared == "numeric":
            return "numeric"
        if raw.startswith("categorical") or declared == "categorical":
            return "categorical"
        if raw == "text" or declared == "text":
            return "text"
        return "numeric"

    def _infer_task_type_from_registry(self, columns: List[Dict[str, Any]], target_column: str) -> str:
        for col in columns:
            if col["name"] != target_column:
                continue
            semantic = str(col.get("semantic_type") or "").lower()
            if semantic in {"categorical", "boolean", "id", "text"}:
                return "classification"
            return "regression"
        return "regression"

    def _contract_from_field_registry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        fields = data.get("fields") or []
        if not isinstance(fields, list) or not fields:
            raise StagingError("contract_schema_invalid", "field_registry.fields missing or invalid")

        columns: List[Dict[str, Any]] = []
        target_column = None
        for field in fields:
            name = str(field.get("name") or "").strip()
            if not name:
                continue
            role = (
                "target"
                if field.get("use_as_target")
                or str(field.get("role") or "").strip().lower() == "target"
                else "feature"
            )
            if role == "target" and target_column is None:
                target_column = name
            columns.append(
                {
                    "name": name,
                    "semantic_type": self._normalize_field_registry_semantic_type(field),
                    "role": role,
                    "missing_tokens": [],
                }
            )

        if not columns:
            raise StagingError("contract_schema_invalid", "field_registry contains no usable columns")
        if not target_column:
            raise StagingError("target_not_defined", "field_registry does not mark a target column")

        return {
            "schema_version": "profile_contract_v1",
            "dataset_id": data.get("dataset_id"),
            "target_column": target_column,
            "task_type": self._infer_task_type_from_registry(columns, target_column),
            "delimiter": ",",
            "encoding": "utf-8",
            "columns": columns,
        }

    def _column_type_to_features_dtype(self, semantic_type: str) -> str:
        s = str(semantic_type or "").lower()
        mapping = {
            "numeric": "continuous",
            "categorical": "categorical",
            "datetime": "timestamp",
            "text": "categorical",
            "id": "ID",
            "boolean": "binary",
        }
        return mapping.get(s, "continuous")

    @staticmethod
    def _read_csv_split(
        csv_path: Path,
        sep: str,
        encoding: str,
        missing_tokens: List[str],
    ) -> pd.DataFrame:
        return pd.read_csv(
            csv_path,
            sep=sep,
            encoding=encoding,
            na_values=missing_tokens,
            keep_default_na=True,
        )

    def _normalize_split(
        self,
        csv_path: Path,
        expected_columns: List[str],
        missing_tokens: List[str],
        delimiter: str,
        encoding: str,
    ) -> pd.DataFrame:
        try:
            df = self._read_csv_split(csv_path, delimiter, encoding, missing_tokens)
        except Exception as e:
            raise StagingError("csv_parser_error", f"read_csv failed for {csv_path}: {e}") from e

        cols = list(df.columns)
        if cols == expected_columns:
            return df

        # 常见情况：contract 的 delimiter 为 tab，但 CSV 实为逗号分隔，首行被读成单列列名。
        if len(cols) == 1 and len(expected_columns) > 1:
            lone = str(cols[0])
            if "," in lone:
                parts = [p.strip() for p in lone.split(",")]
                if parts == expected_columns:
                    try:
                        df2 = self._read_csv_split(csv_path, ",", encoding, missing_tokens)
                    except Exception as e:
                        raise StagingError(
                            "csv_parser_error",
                            f"fallback comma read failed for {csv_path}: {e}",
                        ) from e
                    if list(df2.columns) == expected_columns:
                        return df2

        if delimiter != ",":
            try:
                df_alt = self._read_csv_split(csv_path, ",", encoding, missing_tokens)
            except Exception:
                df_alt = None
            if df_alt is not None and list(df_alt.columns) == expected_columns:
                return df_alt

        raise StagingError(
            "split_header_mismatch",
            f"{csv_path.name}: expected columns {expected_columns}, got {cols}",
        )

    def _public_gate(
        self,
        dataset_id: str,
        train_csv: Path,
        val_csv: Path,
        test_csv: Path,
        profile_path: Path,
        contract_path: Path,
        gate_dir: Path,
    ) -> PublicGateResult:
        contract = self._load_contract(contract_path)
        profile = self._load_profile(profile_path)

        target_col = contract.get("target_column")
        task_type = str(contract.get("task_type") or "").strip().lower()
        columns = contract.get("columns") or []
        if not isinstance(columns, list) or not columns:
            raise StagingError("profile_header_mismatch", "contract columns missing or invalid")
        expected_columns = [c.get("name") for c in columns]
        if any(not c for c in expected_columns):
            raise StagingError("profile_header_mismatch", "contract has empty column name")

        if target_col not in expected_columns:
            raise StagingError("target_not_defined", f"target_column not in header: {target_col}")
        if task_type not in {"classification", "regression"}:
            raise StagingError("target_invalid", f"invalid task_type: {task_type}")

        profile_cols = profile.get("columns") or []
        if list(profile_cols) != expected_columns:
            raise StagingError(
                "profile_header_mismatch",
                f"profile columns mismatch for dataset={dataset_id}",
            )

        delimiter = contract.get("delimiter") or ","
        encoding = str(contract.get("encoding") or "utf-8").strip() or "utf-8"
        missing_tokens = list(dict.fromkeys(self.DEFAULT_MISSING_TOKENS))
        for c in columns:
            for token in (c.get("missing_tokens") or []):
                if token not in missing_tokens:
                    missing_tokens.append(token)

        train_df = self._normalize_split(
            train_csv, expected_columns, missing_tokens, delimiter, encoding
        )
        val_df = self._normalize_split(
            val_csv, expected_columns, missing_tokens, delimiter, encoding
        )
        test_df = self._normalize_split(
            test_csv, expected_columns, missing_tokens, delimiter, encoding
        )

        # PG005 semantic check: numeric must be castable after missing normalization.
        for col in columns:
            name = col["name"]
            semantic = str(col.get("semantic_type") or "").lower()
            if semantic != "numeric":
                continue
            s = pd.concat([train_df[name], val_df[name], test_df[name]], ignore_index=True)
            bad = pd.to_numeric(s, errors="coerce")
            bad_mask = s.notna() & bad.isna()
            if bad_mask.any():
                examples = s[bad_mask].astype(str).head(5).tolist()
                raise StagingError(
                    "profile_dtype_mismatch",
                    f"numeric column cast failed: {name}, examples={examples}",
                )

        public_dir = gate_dir.parent / "staged" / "public"
        public_dir.mkdir(parents=True, exist_ok=True)
        staged_train = public_dir / "train.csv"
        staged_val = public_dir / "val.csv"
        staged_test = public_dir / "test.csv"
        train_df.to_csv(staged_train, index=False, encoding="utf-8")
        val_df.to_csv(staged_val, index=False, encoding="utf-8")
        test_df.to_csv(staged_test, index=False, encoding="utf-8")

        features_rows = []
        for col in columns:
            name = col["name"]
            role = str(col.get("role") or "feature").lower()
            features_rows.append(
                {
                    "feature_name": name,
                    "data_type": self._column_type_to_features_dtype(col.get("semantic_type")),
                    "is_target": role == "target" or name == target_col,
                }
            )
        # enforce exactly one target in staged features
        target_count = sum(1 for r in features_rows if r["is_target"])
        if target_count != 1:
            raise StagingError(
                "target_invalid",
                f"expected exactly one target column in staged features, got {target_count}",
            )

        features_path = public_dir / "staged_features.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(features_rows, f, ensure_ascii=False, indent=2)

        gate_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "dataset_id": dataset_id,
            "status": "pass",
            "checks": [
                {"check_id": "PG001_csv_parse_ok", "status": "pass"},
                {"check_id": "PG002_split_header_consistent", "status": "pass"},
                {"check_id": "PG003_profile_header_match", "status": "pass"},
                {"check_id": "PG004_missing_token_normalized", "status": "pass"},
                {"check_id": "PG005_semantic_type_validated", "status": "pass"},
                {"check_id": "PG006_target_defined_and_valid", "status": "pass"},
            ],
            "target_column": target_col,
            "task_type": task_type,
            "input_splits": {
                "train": str(train_csv),
                "val": str(val_csv),
                "test": str(test_csv),
            },
        }
        report_path = gate_dir / "public_gate_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        schema_snapshot = {
            "dataset_id": dataset_id,
            "target_column": target_col,
            "task_type": task_type,
            "columns": columns,
        }
        schema_path = gate_dir / "normalized_schema_snapshot.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_snapshot, f, ensure_ascii=False, indent=2)

        staged_manifest = {
            "dataset_id": dataset_id,
            "target_column": target_col,
            "task_type": task_type,
            "train_csv": str(staged_train),
            "val_csv": str(staged_val),
            "test_csv": str(staged_test),
            "features_json": str(features_path),
            "public_gate_report": str(report_path),
            "column_schema": columns,
        }
        staged_manifest_path = gate_dir / "staged_input_manifest.json"
        with open(staged_manifest_path, "w", encoding="utf-8") as f:
            json.dump(staged_manifest, f, ensure_ascii=False, indent=2)

        return PublicGateResult(
            dataset_id=dataset_id,
            target_column=target_col,
            task_type=task_type,
            column_schema=columns,
            public_dir=public_dir,
            features_json=features_path,
            report_json=report_path,
            normalized_schema_json=schema_path,
            staged_manifest_json=staged_manifest_path,
            staged_train_csv=staged_train,
            staged_val_csv=staged_val,
            staged_test_csv=staged_test,
        )

    def _write_input_snapshot(
        self,
        out: Path,
        input_paths: Dict[str, Path],
        dataset_id: str,
        model_name: str,
    ) -> None:
        data: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "model": model_name,
            "inputs": {},
        }
        for k, p in input_paths.items():
            data["inputs"][k] = {
                "path": str(p),
                "exists": p.exists(),
                "size": p.stat().st_size if p.exists() else None,
                "sha256": self._sha256(p) if p.exists() else None,
            }
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def prepare(self, dataset_id: str, model_name: str, work_dir: Path) -> Dict[str, Any]:
        train_csv = self.loader.get_train_path(dataset_id)
        val_csv = self.loader.get_val_path(dataset_id)
        test_csv = self.loader.get_test_path(dataset_id)
        profile_path = self.loader.get_features_path(dataset_id)
        contract_path = self.loader.get_contract_path(dataset_id)

        inputs = {
            "train_csv": train_csv,
            "val_csv": val_csv,
            "test_csv": test_csv,
            "profile_json": profile_path,
            "contract_json": contract_path,
        }
        self._write_input_snapshot(work_dir / "input_snapshot.json", inputs, dataset_id, model_name)

        gate_dir = work_dir / "public_gate"
        gate = self._public_gate(
            dataset_id=dataset_id,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            profile_path=profile_path,
            contract_path=contract_path,
            gate_dir=gate_dir,
        )

        model_dir = work_dir / "staged" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_manifest = {
            "dataset_id": dataset_id,
            "model": model_name,
            "target_column": gate.target_column,
            "task_type": gate.task_type,
            "column_schema": gate.column_schema,
            "public_manifest": str(gate.staged_manifest_json),
            "train_csv": str(gate.staged_train_csv),
            "val_csv": str(gate.staged_val_csv),
            "test_csv": str(gate.staged_test_csv),
            "features_json": str(gate.features_json),
            "public_gate_report": str(gate.report_json),
        }
        model_manifest_path = model_dir / "model_input_manifest.json"
        with open(model_manifest_path, "w", encoding="utf-8") as f:
            json.dump(model_manifest, f, ensure_ascii=False, indent=2)

        adapter_report = {
            "adapter_ready_status": "pass",
            "adapter_fail_reason_code": None,
            "adapter_fail_detail": None,
            "adapter_transforms_applied": [],
            "model_input_manifest": str(model_manifest_path),
        }
        with open(model_dir / "adapter_report.json", "w", encoding="utf-8") as f:
            json.dump(adapter_report, f, ensure_ascii=False, indent=2)
        with open(model_dir / "adapter_transforms_applied.json", "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

        return {
            "public_gate_report": str(gate.report_json),
            "public_manifest": str(gate.staged_manifest_json),
            "model_manifest": str(model_manifest_path),
            "train_csv": str(gate.staged_train_csv),
            "features_json": str(gate.features_json),
            "target_column": gate.target_column,
            "task_type": gate.task_type,
        }
