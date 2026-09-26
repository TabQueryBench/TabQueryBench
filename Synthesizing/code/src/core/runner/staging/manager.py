import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..dataset_loader import DatasetLoader
from . import model_view


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
    field_registry_json: Path
    column_schema: List[Dict[str, Any]]
    public_dir: Path
    features_json: Path
    report_json: Path
    normalized_schema_json: Path
    staged_manifest_json: Path
    staged_train_csv: Path
    staged_val_csv: Path
    staged_test_csv: Path
    source_field_registry_json: Optional[Path] = None
    model_view_spec: Optional[Path] = None
    train_rows_original: int = 0
    train_sampling: Dict[str, Any] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)


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

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

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
        if raw == "integer":
            return "integer"
        if raw.startswith("numeric") or raw in {"continuous", "float", "number"} or declared == "numeric":
            return "continuous"
        if raw == "ordinal":
            return "ordinal"
        if raw == "multilabel":
            return "multilabel"
        if raw.startswith("categorical") or declared == "categorical":
            return "categorical"
        if raw == "text" or declared == "text":
            return "text"
        return "continuous"

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
        target_policy = data.get("target_policy") or {}
        official_target = data.get("official_target_column") or target_policy.get("official_target_column")
        adapter_primary = data.get("adapter_primary_target") or target_policy.get("adapter_primary_target") or official_target
        label_columns = list(data.get("label_columns") or target_policy.get("label_columns") or [])
        target_column = adapter_primary
        for field in fields:
            name = str(field.get("name") or "").strip()
            if not name:
                continue
            official_role = str(field.get("role") or "feature").strip().lower()
            adapter_role = "target" if adapter_primary and name == adapter_primary else "feature"
            columns.append(
                {
                    "name": name,
                    "semantic_type": self._normalize_field_registry_semantic_type(field),
                    "role": adapter_role,
                    "official_role": official_role,
                    "adapter_role": adapter_role,
                    "is_label_column": bool(field.get("is_label_column") or name in label_columns),
                    "storage_type": field.get("storage_type"),
                    "nullable": bool(field.get("nullable", False)),
                    "missing_tokens": field.get("missing_tokens") or [],
                    "missing_sentinel": field.get("missing_sentinel", ""),
                    "integer_like": bool(field.get("integer_like", False)),
                    "numeric_like": bool(field.get("numeric_like", False)),
                    "domain_values": field.get("domain_values") or [],
                    "numeric_min": field.get("numeric_min"),
                    "numeric_max": field.get("numeric_max"),
                    "postprocess_policy": field.get("postprocess_policy") or [],
                }
            )

        if not columns:
            raise StagingError("contract_schema_invalid", "field_registry contains no usable columns")
        if not target_column:
            raise StagingError("target_not_defined", "field_registry has no adapter_primary_target")
        if target_column not in {col["name"] for col in columns}:
            raise StagingError("target_not_defined", f"adapter_primary_target not in columns: {target_column}")

        return {
            "schema_version": "profile_contract_v1",
            "dataset_id": data.get("dataset_id"),
            "target_column": target_column,
            "official_target_column": official_target,
            "label_columns": label_columns,
            "adapter_primary_target": adapter_primary,
            "target_policy": target_policy,
            "task_type": data.get("task_type") or target_policy.get("task_type") or self._infer_task_type_from_registry(columns, target_column),
            "delimiter": ",",
            "encoding": "utf-8",
            "columns": columns,
        }

    def _column_type_to_features_dtype(self, semantic_type: str) -> str:
        s = str(semantic_type or "").lower()
        mapping = {
            "numeric": "continuous",
            "continuous": "continuous",
            "integer": "integer",
            "categorical": "categorical",
            "ordinal": "ordinal",
            "datetime": "timestamp",
            "text": "categorical",
            "id": "ID",
            "boolean": "binary",
            "multilabel": "categorical",
        }
        return mapping.get(s, "continuous")

    @staticmethod
    def _read_csv_split(
        csv_path: Path,
        sep: str,
        encoding: str,
        missing_tokens: List[str],
        keep_default_na: bool = True,
        dtype: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        return pd.read_csv(
            csv_path,
            sep=sep,
            encoding=encoding,
            na_values=missing_tokens,
            keep_default_na=keep_default_na,
            dtype=dtype,
            low_memory=False,
        )

    def _normalize_split(
        self,
        csv_path: Path,
        expected_columns: List[str],
        missing_tokens: List[str],
        delimiter: str,
        encoding: str,
        keep_default_na: bool = True,
        dtype: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        try:
            df = self._read_csv_split(csv_path, delimiter, encoding, missing_tokens, keep_default_na, dtype)
        except Exception as e:
            raise StagingError("csv_parser_error", f"read_csv failed for {csv_path}: {e}") from e

        cols = list(df.columns)
        if cols == expected_columns:
            return df

        if len(cols) == len(expected_columns):
            renamed = False
            if cols and str(cols[0]).startswith("Unnamed:") and str(expected_columns[0]).lower() in {"row_id", "id", "index"}:
                df = df.rename(columns={cols[0]: expected_columns[0]})
                renamed = True
            if renamed and list(df.columns) == expected_columns:
                return df

        if len(cols) == len(expected_columns):
            # Headerless CSVs are sometimes documented in the contract but stored
            # without a header row. If the first parsed "header" looks like data,
            # reread with header=None and apply the contract columns.
            expected_set = {str(c) for c in expected_columns}
            overlap = sum(1 for c in cols if str(c) in expected_set)
            numeric_like = 0
            for c in cols:
                try:
                    float(str(c))
                    numeric_like += 1
                except Exception:
                    pass
            looks_headerless = overlap == 0 and numeric_like >= max(1, len(cols) // 2)
            try:
                first_row = pd.read_csv(
                    csv_path,
                    sep=delimiter,
                    encoding=encoding,
                    na_values=missing_tokens,
                    keep_default_na=keep_default_na,
                    header=None,
                    nrows=1,
                )
            except Exception:
                first_row = None
            if (
                looks_headerless
                and first_row is not None
                and len(first_row.columns) == len(expected_columns)
            ):
                df_headerless = pd.read_csv(
                    csv_path,
                    sep=delimiter,
                    encoding=encoding,
                    na_values=missing_tokens,
                    keep_default_na=keep_default_na,
                    header=None,
                    names=expected_columns,
                )
                return df_headerless

        # 常见情况：contract 的 delimiter 为 tab，但 CSV 实为逗号分隔，首行被读成单列列名。
        if len(cols) == 1 and len(expected_columns) > 1:
            lone = str(cols[0])
            if "," in lone:
                parts = [p.strip() for p in lone.split(",")]
                if parts == expected_columns:
                    try:
                        df2 = self._read_csv_split(csv_path, ",", encoding, missing_tokens, keep_default_na, dtype)
                    except Exception as e:
                        raise StagingError(
                            "csv_parser_error",
                            f"fallback comma read failed for {csv_path}: {e}",
                        ) from e
                    if list(df2.columns) == expected_columns:
                        return df2

        if delimiter != ",":
            try:
                df_alt = self._read_csv_split(csv_path, ",", encoding, missing_tokens, keep_default_na, dtype)
            except Exception:
                df_alt = None
            if df_alt is not None and list(df_alt.columns) == expected_columns:
                return df_alt

        raise StagingError(
            "split_header_mismatch",
            f"{csv_path.name}: expected columns {expected_columns}, got {cols}",
        )

    @staticmethod
    def _resolve_max_train_rows(cli_value: Optional[int], generation: Dict[str, Any]) -> Optional[int]:
        """Priority: CLI/API argument > profile generation.max_train_rows > BENCHMARK_MAX_TRAIN_ROWS. 0 = off."""
        if cli_value is not None:
            return int(cli_value) if int(cli_value) > 0 else None
        profile_value = generation.get("max_train_rows")
        if profile_value:
            return int(profile_value)
        env_value = os.environ.get("BENCHMARK_MAX_TRAIN_ROWS", "").strip()
        if env_value:
            try:
                return int(env_value) if int(env_value) > 0 else None
            except ValueError:
                raise StagingError("config_invalid", f"BENCHMARK_MAX_TRAIN_ROWS must be an integer, got {env_value!r}")
        return None

    def _public_gate(
        self,
        dataset_id: str,
        train_csv: Path,
        val_csv: Path,
        test_csv: Path,
        profile_path: Path,
        contract_path: Path,
        gate_dir: Path,
        max_train_rows: Optional[int] = None,
    ) -> PublicGateResult:
        contract = self._load_contract(contract_path)
        profile = self._load_profile(profile_path)
        field_registry_path = contract_path if contract_path.name == "field_registry.json" else Path("")
        registry_raw = self._load_json(contract_path) if field_registry_path else {}
        prep_contract = registry_raw.get("preprocessing_contract") or {}
        exact_missing = bool(prep_contract.get("exact_missing_tokens"))

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
        dtype: Optional[Dict[str, Any]] = None
        if exact_missing:
            # Prepared datasets encode missing values as empty cells only; real category
            # values such as "NA", "None" or "unknown" must survive staging.
            missing_tokens = sorted({str(t) for c in columns for t in (c.get("missing_tokens") or [""])})
            keep_default_na = False
            dtype = {
                c["name"]: str
                for c in columns
                if str(c.get("semantic_type") or "").lower() not in {"continuous", "integer", "numeric"}
            }
        else:
            keep_default_na = True
            missing_tokens = list(dict.fromkeys(self.DEFAULT_MISSING_TOKENS))
            for c in columns:
                for token in (c.get("missing_tokens") or []):
                    if token not in missing_tokens:
                        missing_tokens.append(token)

        train_df = self._normalize_split(
            train_csv, expected_columns, missing_tokens, delimiter, encoding, keep_default_na, dtype
        )
        val_df = self._normalize_split(
            val_csv, expected_columns, missing_tokens, delimiter, encoding, keep_default_na, dtype
        )
        test_df = self._normalize_split(
            test_csv, expected_columns, missing_tokens, delimiter, encoding, keep_default_na, dtype
        )

        # PG005 semantic check: numeric must be castable after missing normalization.
        for col in columns:
            name = col["name"]
            semantic = str(col.get("semantic_type") or "").lower()
            if semantic not in {"numeric", "continuous", "integer"}:
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

        # PG008 optional training-row cap (off unless requested). Applied before the model view so
        # that the view's all-missing / constant column decisions see exactly the rows the model trains on.
        generation = dict(registry_raw.get("generation") or {})
        cap = self._resolve_max_train_rows(max_train_rows, generation)
        seed = int(os.environ.get("BENCHMARK_TRAIN_SAMPLE_SEED", "42"))
        train_rows_original = int(len(train_df))
        train_df, sampling = model_view.subsample_train(train_df, target_col, task_type, cap, seed)

        # PG007 model view: uniform model-facing table for prepared datasets.
        splits = {"train": train_df, "val": val_df, "test": test_df}
        model_columns = columns
        model_target = target_col
        view_spec_path: Optional[Path] = None
        adapter_registry_path = field_registry_path
        view_spec: Optional[Dict[str, Any]] = None
        if registry_raw and model_view.is_enabled_for(registry_raw):
            try:
                splits, model_columns, view_registry, view_spec = model_view.build_model_view(
                    splits, columns, registry_raw, target_col
                )
            except ValueError as e:
                raise StagingError("model_view_failed", str(e)) from e
            model_target = view_spec["target_column"]
            adapter_registry_path = public_dir / "model_field_registry.json"
            with open(adapter_registry_path, "w", encoding="utf-8") as f:
                json.dump(view_registry, f, ensure_ascii=False, indent=2)

        if view_spec is not None:
            view_spec["train_sampling"] = sampling
            view_spec_path = public_dir / "model_view_spec.json"
            with open(view_spec_path, "w", encoding="utf-8") as f:
                json.dump(view_spec, f, ensure_ascii=False, indent=2)

        staged_train = public_dir / "train.csv"
        staged_val = public_dir / "val.csv"
        staged_test = public_dir / "test.csv"
        splits["train"].to_csv(staged_train, index=False, encoding="utf-8")
        splits["val"].to_csv(staged_val, index=False, encoding="utf-8")
        splits["test"].to_csv(staged_test, index=False, encoding="utf-8")

        features_rows = []
        for col in model_columns:
            name = col["name"]
            role = str(col.get("role") or "feature").lower()
            features_rows.append(
                {
                    "feature_name": name,
                    "data_type": self._column_type_to_features_dtype(col.get("semantic_type")),
                    "is_target": role == "target" or name == model_target,
                    "semantic_type": col.get("semantic_type"),
                    "official_role": col.get("official_role"),
                    "adapter_role": col.get("adapter_role"),
                    "is_label_column": bool(col.get("is_label_column")),
                    "postprocess_policy": col.get("postprocess_policy") or [],
                    "nullable": bool(col.get("nullable", False)),
                    "integer_like": bool(col.get("integer_like", False)),
                    "numeric_like": bool(col.get("numeric_like", False)),
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
        checks = [
            {"check_id": "PG001_csv_parse_ok", "status": "pass"},
            {"check_id": "PG002_split_header_consistent", "status": "pass"},
            {"check_id": "PG003_profile_header_match", "status": "pass"},
            {"check_id": "PG004_missing_token_normalized", "status": "pass", "exact_missing_tokens": exact_missing},
            {"check_id": "PG005_semantic_type_validated", "status": "pass"},
            {"check_id": "PG006_target_defined_and_valid", "status": "pass"},
            {"check_id": "PG007_model_view", "status": "applied" if view_spec is not None else "skipped"},
            {"check_id": "PG008_train_row_cap", "status": sampling["sampling"], **sampling},
        ]
        report = {
            "dataset_id": dataset_id,
            "status": "pass",
            "checks": checks,
            "target_column": target_col,
            "model_target_column": model_target,
            "official_target_column": contract.get("official_target_column"),
            "label_columns": contract.get("label_columns") or [],
            "adapter_primary_target": contract.get("adapter_primary_target"),
            "target_policy": contract.get("target_policy") or {},
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
            "official_target_column": contract.get("official_target_column"),
            "label_columns": contract.get("label_columns") or [],
            "adapter_primary_target": contract.get("adapter_primary_target"),
            "target_policy": contract.get("target_policy") or {},
            "task_type": task_type,
            "columns": columns,
            "model_columns": model_columns,
        }
        schema_path = gate_dir / "normalized_schema_snapshot.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_snapshot, f, ensure_ascii=False, indent=2)

        staged_manifest = {
            "dataset_id": dataset_id,
            "target_column": model_target,
            "task_type": task_type,
            "train_csv": str(staged_train),
            "val_csv": str(staged_val),
            "test_csv": str(staged_test),
            "features_json": str(features_path),
            "field_registry_json": str(adapter_registry_path) if adapter_registry_path else None,
            "source_field_registry_json": str(field_registry_path) if field_registry_path else None,
            "model_view_spec": str(view_spec_path) if view_spec_path else None,
            "train_rows_original": train_rows_original,
            "train_sampling": sampling,
            "generation": generation,
            "public_gate_report": str(report_path),
            "column_schema": model_columns,
        }
        staged_manifest_path = gate_dir / "staged_input_manifest.json"
        with open(staged_manifest_path, "w", encoding="utf-8") as f:
            json.dump(staged_manifest, f, ensure_ascii=False, indent=2)

        return PublicGateResult(
            dataset_id=dataset_id,
            target_column=model_target,
            task_type=task_type,
            field_registry_json=adapter_registry_path,
            column_schema=model_columns,
            public_dir=public_dir,
            features_json=features_path,
            report_json=report_path,
            normalized_schema_json=schema_path,
            staged_manifest_json=staged_manifest_path,
            staged_train_csv=staged_train,
            staged_val_csv=staged_val,
            staged_test_csv=staged_test,
            source_field_registry_json=field_registry_path if field_registry_path else None,
            model_view_spec=view_spec_path,
            train_rows_original=train_rows_original,
            train_sampling=sampling,
            generation=generation,
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

    def prepare(
        self,
        dataset_id: str,
        model_name: str,
        work_dir: Path,
        max_train_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
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
            max_train_rows=max_train_rows,
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
            "field_registry_json": str(gate.field_registry_json) if gate.field_registry_json else None,
            "source_field_registry_json": str(gate.source_field_registry_json) if gate.source_field_registry_json else None,
            "model_view_spec": str(gate.model_view_spec) if gate.model_view_spec else None,
            "train_rows_original": gate.train_rows_original,
            "train_sampling": gate.train_sampling,
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
            "field_registry_json": str(gate.field_registry_json) if gate.field_registry_json else None,
            "model_view_spec": str(gate.model_view_spec) if gate.model_view_spec else None,
            "train_rows_original": gate.train_rows_original,
            "train_sampling": gate.train_sampling,
            "generation": gate.generation,
            "target_column": gate.target_column,
            "task_type": gate.task_type,
        }
