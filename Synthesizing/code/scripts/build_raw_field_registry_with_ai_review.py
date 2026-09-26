#!/usr/bin/env python3
"""Build per-column preprocessing metadata from raw CSV splits.

This script is the formal preprocessing entry point for raw_data/tabular_datasets.
It scans train/val/test for deterministic facts, optionally asks an AI reviewer
to classify each column, then writes auditable metadata under each dataset:

  metadata/field_ai_reviews.jsonl
  metadata_core/field_registry.json
  metadata_core/dataset_semantics.yaml

The generated registry is meant to be consumed before model training/generation,
not merely as a post-hoc repair artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - pyyaml is listed in requirements.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "raw_data" / "tabular_datasets"

NULL_TOKENS = {"", "na", "n/a", "null", "none", "nan", "?", "unknown", "missing", "__nan__"}
BOOL_TRUE = {"true", "t", "yes", "y", "1"}
BOOL_FALSE = {"false", "f", "no", "n", "0"}
BOOL_TOKENS = BOOL_TRUE | BOOL_FALSE
UNIQUE_CAP = 10000
DOMAIN_CAP = 5000
INT_TOL = 1e-9

SEMANTIC_TYPES = {
    "continuous",
    "integer",
    "ordinal",
    "categorical",
    "boolean",
    "id",
    "text",
    "datetime",
    "multilabel",
}
STORAGE_TYPES = {
    "integer_token",
    "float_token",
    "float_integer_token",
    "string_token",
    "string_boolean_token",
    "datetime_string_token",
    "empty_token",
}
POLICIES = {
    "round_integer",
    "restore_domain",
    "strip_whitespace",
    "warn_range",
    "clip_range",
    "reject_range",
    "warn_null_rate",
    "reject_all_null",
    "preserve_missing_sentinel",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[raw-field-registry] {message}", flush=True)


def norm_token(value: Any) -> str:
    return str(value).strip()


def is_null_token(value: Any) -> bool:
    return norm_token(value).lower() in NULL_TOKENS


def to_float(value: Any) -> float | None:
    text = norm_token(value).replace(",", "")
    if text.lower() in NULL_TOKENS:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def is_int_like_value(value: Any) -> bool:
    number = to_float(value)
    return number is not None and abs(number - round(number)) <= INT_TOL


def looks_id_like(name: str) -> bool:
    lowered = name.lower().replace(" ", "_")
    return (
        lowered == "id"
        or lowered.endswith("_id")
        or lowered.endswith("id")
        or "row_id" in lowered
        or "page_id" in lowered
        or "student_id" in lowered
        or "question_id" in lowered
        or "enrollee_id" in lowered
        or "recordid" in lowered
    )


def looks_code_like(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("code", "channel", "city", "region", "role_", "policy_sales"))


def looks_datetime_like_name(name: str) -> bool:
    lowered = name.lower()
    return "date" in lowered or lowered in {"year", "month", "day", "time", "timestamp"}


def looks_text_like_name(name: str) -> bool:
    lowered = name.lower().replace(" ", "_")
    return any(
        token in lowered
        for token in (
            "description",
            "overview",
            "summary",
            "comment",
            "review",
            "abstract",
            "plot",
            "synopsis",
        )
    )


def looks_multilabel_like_name(name: str) -> bool:
    lowered = name.lower().replace(" ", "_")
    return any(
        token in lowered
        for token in (
            "keyword",
            "tag",
            "genre",
            "cast",
            "director",
            "country",
            "listed_in",
            "category_list",
        )
    )


def looks_continuous_numeric_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("velocity", "index", "rate", "ratio", "score", "amount", "premium"))


def exact_domain_values(values: set[str]) -> list[str]:
    return sorted(values, key=lambda item: (to_float(item) is None, to_float(item) if to_float(item) is not None else item))


@dataclass
class ColumnProfile:
    name: str
    non_null: int = 0
    null: int = 0
    numeric: int = 0
    integer: int = 0
    boolish: int = 0
    dateish: int = 0
    min_value: float | None = None
    max_value: float | None = None
    unique: set[str] = field(default_factory=set)
    unique_over_cap: bool = False
    examples: list[str] = field(default_factory=list)
    non_numeric_examples: list[str] = field(default_factory=list)
    decimal_examples: list[str] = field(default_factory=list)
    token_has_decimal: bool = False
    whitespace_variant_count: int = 0
    comma_token_count: int = 0

    def update(self, value: Any) -> None:
        raw = "" if value is None else str(value)
        stripped = raw.strip()
        if raw != stripped and stripped:
            self.whitespace_variant_count += 1
        if is_null_token(raw):
            self.null += 1
            return
        self.non_null += 1
        if "," in stripped:
            self.comma_token_count += 1
        if len(self.examples) < 12 and stripped not in self.examples:
            self.examples.append(stripped)
        if not self.unique_over_cap:
            self.unique.add(stripped)
            if len(self.unique) > UNIQUE_CAP:
                self.unique_over_cap = True
                self.unique = set(list(self.unique)[:UNIQUE_CAP])

        lowered = stripped.lower()
        if lowered in BOOL_TOKENS:
            self.boolish += 1
        if parse_dateish(stripped):
            self.dateish += 1

        number = to_float(stripped)
        if number is not None:
            self.numeric += 1
            self.min_value = number if self.min_value is None else min(self.min_value, number)
            self.max_value = number if self.max_value is None else max(self.max_value, number)
            if is_int_like_value(stripped):
                self.integer += 1
                if re.search(r"\.0+$", stripped):
                    self.token_has_decimal = True
            else:
                self.token_has_decimal = self.token_has_decimal or "." in stripped
                if len(self.decimal_examples) < 8:
                    self.decimal_examples.append(stripped)
        elif len(self.non_numeric_examples) < 8:
            self.non_numeric_examples.append(stripped)

    @property
    def total(self) -> int:
        return self.non_null + self.null

    @property
    def null_rate(self) -> float:
        return self.null / self.total if self.total else 0.0

    @property
    def numeric_ratio(self) -> float:
        return self.numeric / self.non_null if self.non_null else 0.0

    @property
    def integer_ratio(self) -> float:
        return self.integer / self.non_null if self.non_null else 0.0

    @property
    def boolish_ratio(self) -> float:
        return self.boolish / self.non_null if self.non_null else 0.0

    @property
    def dateish_ratio(self) -> float:
        return self.dateish / self.non_null if self.non_null else 0.0

    @property
    def unique_count(self) -> int:
        return len(self.unique)


def parse_dateish(value: str) -> bool:
    if not value or len(value) < 6:
        return False
    patterns = [
        r"^[A-Za-z]+ \d{1,2}, \d{4}$",
        r"^\d{4}-\d{1,2}-\d{1,2}$",
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",
    ]
    return any(re.match(pattern, value) for pattern in patterns)


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return next(csv.reader(handle))


def scan_dataset(dataset_dir: Path, dataset_id: str) -> tuple[list[str], dict[str, ColumnProfile], dict[str, int]]:
    split_paths = {split: dataset_dir / f"{dataset_id}-{split}.csv" for split in ("train", "val", "test")}
    existing = {split: path for split, path in split_paths.items() if path.exists()}
    if "train" not in existing:
        raise FileNotFoundError(f"{dataset_id}: missing train split at {split_paths['train']}")
    header = read_header(existing["train"])
    profiles = {name: ColumnProfile(name) for name in header}
    row_counts: dict[str, int] = {}
    for split, path in existing.items():
        log(f"{dataset_id}/{split}: scanning {path.name}")
        row_counts[split] = 0
        last_log = time.monotonic()
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != header:
                raise ValueError(f"{dataset_id}-{split}: header mismatch")
            for row in reader:
                row_counts[split] += 1
                for name in header:
                    profiles[name].update(row.get(name, ""))
                if row_counts[split] % 100000 == 0 or time.monotonic() - last_log > 30:
                    log(f"{dataset_id}/{split}: {row_counts[split]} rows")
                    last_log = time.monotonic()
    return header, profiles, row_counts


def deterministic_field(dataset_id: str, profile: ColumnProfile, target_column: str | None) -> dict[str, Any]:
    name = profile.name
    lowered = name.lower()
    integer_like = profile.non_null > 0 and profile.integer_ratio >= 0.999
    numeric_like = profile.non_null > 0 and profile.numeric_ratio >= 0.999
    date_like = profile.non_null > 0 and profile.dateish_ratio >= 0.95
    bool_token_domain = profile.non_null > 0 and profile.unique_count <= 2 and {
        value.lower() for value in profile.unique
    }.issubset(BOOL_TOKENS)
    numeric_domain = {to_float(value) for value in profile.unique}
    boolean_numeric_domain = (
        profile.non_null > 0
        and profile.unique_count <= 2
        and None not in numeric_domain
        and {int(value) for value in numeric_domain}.issubset({0, 1})
        and all(abs(value - round(value)) <= INT_TOL for value in numeric_domain if value is not None)
    )
    boolean_domain = bool_token_domain or boolean_numeric_domain
    nullable = profile.null > 0
    unique_count = profile.unique_count
    domain_values: list[str] = []

    if profile.non_null == 0:
        semantic_type = "continuous"
        storage_type = "empty_token"
    elif date_like and looks_datetime_like_name(name):
        semantic_type = "datetime"
        storage_type = "datetime_string_token"
    elif numeric_like:
        storage_type = "float_integer_token" if integer_like and profile.token_has_decimal else ("integer_token" if integer_like else "float_token")
        if looks_id_like(name):
            semantic_type = "id"
        elif name == target_column or lowered in {"class", "target", "result", "price_range"} or lowered.startswith("label_"):
            semantic_type = "categorical" if integer_like and unique_count <= DOMAIN_CAP else "continuous"
        elif looks_code_like(name) and integer_like:
            semantic_type = "categorical"
        elif boolean_domain:
            semantic_type = "boolean"
        elif integer_like and unique_count <= 20 and not looks_continuous_numeric_name(name):
            semantic_type = "ordinal"
        elif integer_like:
            semantic_type = "integer"
        else:
            semantic_type = "continuous"
    else:
        storage_type = "string_token"
        if date_like and looks_datetime_like_name(name):
            semantic_type = "datetime"
            storage_type = "datetime_string_token"
        elif boolean_domain:
            semantic_type = "boolean"
            storage_type = "string_boolean_token" if bool_token_domain and not numeric_like else storage_type
        elif looks_text_like_name(name):
            semantic_type = "text"
        elif (
            looks_multilabel_like_name(name)
            and profile.comma_token_count / max(1, profile.non_null) >= 0.25
            and unique_count > 50
        ):
            semantic_type = "multilabel"
        elif unique_count <= DOMAIN_CAP and not profile.unique_over_cap:
            semantic_type = "categorical"
        else:
            semantic_type = "text"

    if semantic_type in {"boolean", "categorical", "ordinal"} and unique_count <= DOMAIN_CAP and not profile.unique_over_cap:
        domain_values = exact_domain_values(profile.unique)

    policies = {"warn_range", "warn_null_rate", "strip_whitespace"}
    if integer_like:
        policies.add("round_integer")
    if domain_values:
        policies.add("restore_domain")
    if nullable:
        policies.add("preserve_missing_sentinel")

    return {
        "name": name,
        "role": "target" if name == target_column or lowered in {"class", "target", "result"} or lowered.startswith("label_") else "feature",
        "semantic_type": semantic_type,
        "storage_type": storage_type,
        "nullable": nullable,
        "missing_tokens": sorted(NULL_TOKENS),
        "missing_sentinel": "",
        "raw_null_rate": round(profile.null_rate, 6),
        "integer_like": integer_like,
        "numeric_like": numeric_like,
        "datetime_like": date_like,
        "domain_values": domain_values,
        "numeric_min": profile.min_value,
        "numeric_max": profile.max_value,
        "unique_count": unique_count,
        "unique_over_cap": profile.unique_over_cap,
        "examples": profile.examples,
        "non_numeric_examples": profile.non_numeric_examples,
        "decimal_examples": profile.decimal_examples,
        "whitespace_variant_count": profile.whitespace_variant_count,
        "postprocess_policy": sorted(policies),
        "preprocess_policy": sorted({
            "trim_string_tokens",
            "preserve_raw_missing",
            "preserve_column_order",
            "write_field_registry",
        }),
        "review": {
            "source": "deterministic_profiler",
            "status": "pending_ai_review",
            "confidence": 0.7,
            "rationale": "Initial type inferred from raw train/val/test statistics.",
        },
    }


def build_ai_prompt(dataset_id: str, field: dict[str, Any]) -> str:
    payload = {
        "dataset_id": dataset_id,
        "column": {
            "name": field["name"],
            "role": field["role"],
            "deterministic_semantic_type": field["semantic_type"],
            "storage_type": field["storage_type"],
            "nullable": field["nullable"],
            "raw_null_rate": field["raw_null_rate"],
            "integer_like": field["integer_like"],
            "numeric_like": field["numeric_like"],
            "datetime_like": field["datetime_like"],
            "unique_count": field["unique_count"],
            "numeric_min": field["numeric_min"],
            "numeric_max": field["numeric_max"],
            "examples": field["examples"],
            "non_numeric_examples": field["non_numeric_examples"],
            "decimal_examples": field["decimal_examples"],
            "whitespace_variant_count": field["whitespace_variant_count"],
            "domain_sample": field["domain_values"][:40],
        },
    }
    return (
        "You are reviewing preprocessing metadata for a public tabular dataset.\n"
        "Classify this one column for synthetic-data training/generation.\n"
        "Return strict JSON only. Do not include markdown.\n\n"
        "Allowed semantic_type values: continuous, integer, ordinal, categorical, boolean, id, text, datetime, multilabel.\n"
        "Allowed storage_type values: integer_token, float_token, float_integer_token, string_token, string_boolean_token, datetime_string_token, empty_token.\n"
        "Allowed postprocess_policy values: round_integer, restore_domain, strip_whitespace, warn_range, clip_range, reject_range, warn_null_rate, reject_all_null, preserve_missing_sentinel.\n\n"
        "The rationale must be plain text; do not include raw JSON, braces, or unescaped double quotes inside it.\n\n"
        "JSON schema:\n"
        "{\"semantic_type\":\"...\",\"storage_type\":\"...\",\"role\":\"feature|target\","
        "\"domain_policy\":\"exact|range|free_text|datetime_parse|multilabel_tokens\","
        "\"postprocess_policy\":[\"...\"],\"confidence\":0.0,\"rationale\":\"short reason\"}\n\n"
        f"Column evidence:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return parse_json_object_fallback(match.group(0))
    return payload if isinstance(payload, dict) else None


def parse_json_object_fallback(text: str) -> dict[str, Any] | None:
    """Best-effort parser for near-JSON model replies with a malformed rationale."""
    payload: dict[str, Any] = {}
    for key in ("semantic_type", "storage_type", "role", "domain_policy"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        if match:
            payload[key] = match.group(1)
    policies_match = re.search(r'"postprocess_policy"\s*:\s*\[(.*?)\]', text, flags=re.S)
    if policies_match:
        payload["postprocess_policy"] = re.findall(r'"([^"]+)"', policies_match.group(1))
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if confidence_match:
        payload["confidence"] = float(confidence_match.group(1))
    rationale_match = re.search(r'"rationale"\s*:\s*"(.*)', text, flags=re.S)
    if rationale_match:
        payload["rationale"] = rationale_match.group(1).splitlines()[0].strip().rstrip('"} ')
    if {"semantic_type", "storage_type", "role"}.issubset(payload):
        payload.setdefault("domain_policy", "")
        payload.setdefault("postprocess_policy", [])
        payload.setdefault("confidence", 0.0)
        payload.setdefault("rationale", "Recovered from malformed near-JSON AI response.")
        return payload
    return None


def run_ai_review(provider: str, prompt: str, timeout_sec: int) -> tuple[dict[str, Any] | None, str | None]:
    if provider == "none":
        return None, None
    if provider != "claude":
        raise ValueError(f"Unsupported ai provider: {provider}")
    try:
        result = subprocess.run(
            ["claude", "-p", "--permission-mode", "dontAsk", "--effort", "low"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "ai_timeout"
    except FileNotFoundError:
        return None, "claude_cli_missing"
    if result.returncode != 0:
        return None, f"ai_exit_{result.returncode}: {result.stderr.strip()[:500]}"
    parsed = parse_json_object(result.stdout)
    if parsed is None:
        return None, f"ai_parse_failed: {result.stdout.strip()[:500]}"
    return parsed, None


def sanitize_ai_review(review: dict[str, Any]) -> dict[str, Any]:
    semantic_type = str(review.get("semantic_type") or "").strip()
    storage_type = str(review.get("storage_type") or "").strip()
    role = str(review.get("role") or "").strip()
    policies = [str(item).strip() for item in review.get("postprocess_policy") or []]
    return {
        "semantic_type": semantic_type if semantic_type in SEMANTIC_TYPES else "",
        "storage_type": storage_type if storage_type in STORAGE_TYPES else "",
        "role": role if role in {"feature", "target"} else "",
        "domain_policy": str(review.get("domain_policy") or "").strip(),
        "postprocess_policy": [item for item in policies if item in POLICIES],
        "confidence": float(review.get("confidence") or 0.0),
        "rationale": str(review.get("rationale") or "").strip()[:1000],
    }


def apply_ai_review(field: dict[str, Any], ai_review: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    updated = dict(field)
    if error:
        updated["review"] = {
            "source": "ai",
            "status": "error_kept_deterministic",
            "error": error,
            "confidence": field["review"]["confidence"],
            "rationale": field["review"]["rationale"],
        }
        return updated
    if not ai_review:
        return updated
    review = sanitize_ai_review(ai_review)
    if review["semantic_type"]:
        updated["semantic_type"] = review["semantic_type"]
    if review["storage_type"]:
        updated["storage_type"] = review["storage_type"]
    if review["role"]:
        updated["role"] = review["role"]
    if review["postprocess_policy"]:
        merged = set(updated.get("postprocess_policy") or [])
        merged.update(review["postprocess_policy"])
        updated["postprocess_policy"] = sorted(merged)
    updated["review"] = {
        "source": "ai",
        "status": "applied",
        "domain_policy": review["domain_policy"],
        "confidence": review["confidence"],
        "rationale": review["rationale"],
    }
    return updated


def load_manual_overrides(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    candidates = [
        dataset_dir / "metadata" / "field_overrides.json",
        dataset_dir / "metadata" / "contract_overrides.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        field_overrides = payload.get("field_overrides") or payload.get("overrides", {}).get("field_overrides") or {}
        if isinstance(field_overrides, dict):
            return {str(name): value for name, value in field_overrides.items() if isinstance(value, dict)}
    return {}


def load_cached_ai_reviews(reviews_path: Path, ai_provider: str) -> dict[str, dict[str, Any]]:
    if ai_provider == "none" or not reviews_path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    with reviews_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("ai_provider") != ai_provider or record.get("ai_error"):
                continue
            column = str(record.get("column") or "")
            ai_review = record.get("ai_review")
            if column and isinstance(ai_review, dict):
                cached[column] = ai_review
    return cached


def apply_manual_override(field: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return field
    updated = dict(field)
    for key in ("role", "semantic_type", "storage_type", "nullable", "missing_sentinel"):
        if key in override:
            updated[key] = override[key]
    if "domain_values" in override and isinstance(override["domain_values"], list):
        updated["domain_values"] = [str(value) for value in override["domain_values"]]
    if "postprocess_policy" in override and isinstance(override["postprocess_policy"], list):
        updated["postprocess_policy"] = sorted({str(value) for value in override["postprocess_policy"] if str(value) in POLICIES})
    if "preprocess_policy" in override and isinstance(override["preprocess_policy"], list):
        updated["preprocess_policy"] = sorted({str(value) for value in override["preprocess_policy"]})
    updated["review"] = {
        "source": "manual_override",
        "status": "applied",
        "confidence": 1.0,
        "rationale": str(override.get("rationale") or "Manual field override applied."),
    }
    return updated


def infer_target_column(fields: list[str]) -> str | None:
    for candidate in ("target", "class", "Result", "result", "price_range", "label_1"):
        if candidate in fields:
            return candidate
    label_cols = [name for name in fields if name.lower().startswith("label_")]
    if label_cols:
        return label_cols[0]
    return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def materialize_dataset(
    dataset_dir: Path,
    dataset_id: str,
    *,
    ai_provider: str,
    ai_timeout_sec: int,
    force_ai: bool,
    max_columns: int | None,
) -> dict[str, Any]:
    header, profiles, row_counts = scan_dataset(dataset_dir, dataset_id)
    target_column = infer_target_column(header)
    manual_overrides = load_manual_overrides(dataset_dir)
    metadata_dir = dataset_dir / "metadata"
    metadata_core_dir = dataset_dir / "metadata_core"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_core_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = metadata_dir / "field_ai_reviews.jsonl"
    cached_ai_reviews = load_cached_ai_reviews(reviews_path, ai_provider)

    fields: list[dict[str, Any]] = []
    with reviews_path.open("w", encoding="utf-8") as reviews_handle:
        for idx, name in enumerate(header):
            if max_columns is not None and idx >= max_columns:
                break
            field_payload = deterministic_field(dataset_id, profiles[name], target_column)
            prompt = build_ai_prompt(dataset_id, field_payload)
            ai_review = None
            ai_error = None
            ai_cached = False
            if ai_provider != "none" and (force_ai or not manual_overrides.get(name)):
                if not force_ai and name in cached_ai_reviews:
                    log(f"{dataset_id}/{name}: using cached {ai_provider} review ({idx + 1}/{len(header)})")
                    ai_review = cached_ai_reviews[name]
                    ai_cached = True
                else:
                    log(f"{dataset_id}/{name}: requesting {ai_provider} review ({idx + 1}/{len(header)})")
                    ai_review, ai_error = run_ai_review(ai_provider, prompt, ai_timeout_sec)
                    if ai_error:
                        log(f"{dataset_id}/{name}: {ai_provider} review failed: {ai_error}")
                    else:
                        log(f"{dataset_id}/{name}: {ai_provider} review completed")
            reviewed = apply_ai_review(field_payload, ai_review, ai_error)
            reviewed = apply_manual_override(reviewed, manual_overrides.get(name))
            fields.append(reviewed)
            reviews_handle.write(
                json.dumps(
                    {
                        "dataset_id": dataset_id,
                        "column": name,
                        "prompt": prompt,
                        "ai_provider": ai_provider,
                        "ai_cached": ai_cached,
                        "ai_error": ai_error,
                        "ai_review": ai_review,
                        "final_field": reviewed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            reviews_handle.flush()

    registry = {
        "dataset_id": dataset_id,
        "generated_at": now_iso(),
        "generated_by": "code/scripts/build_raw_field_registry_with_ai_review.py",
        "ai_provider": ai_provider,
        "target_column": target_column,
        "task_type": "classification" if any(field["role"] == "target" and field["semantic_type"] in {"categorical", "boolean", "ordinal"} for field in fields) else "unknown",
        "expected_rows": row_counts.get("train"),
        "row_counts": row_counts,
        "field_count": len(fields),
        "fields": fields,
        "preprocessing_contract": {
            "trim_string_tokens_before_training": True,
            "preserve_raw_missing_tokens": True,
            "write_model_input_manifest": True,
            "fail_closed_when_metadata_missing": True,
        },
    }
    write_json(metadata_core_dir / "field_registry.json", registry)
    write_yaml(
        metadata_core_dir / "dataset_semantics.yaml",
        {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "row_count": row_counts.get("train"),
            "notes": [
                "Generated from raw train/val/test splits with deterministic profiling plus optional per-column AI review.",
                "Field-level constraints are intended for preprocessing and model input manifests before training/generation.",
            ],
        },
    )
    return {
        "dataset_id": dataset_id,
        "field_count": len(fields),
        "row_counts": row_counts,
        "ai_reviewed_fields": sum(1 for field in fields if field.get("review", {}).get("source") == "ai"),
        "manual_override_fields": sum(1 for field in fields if field.get("review", {}).get("source") == "manual_override"),
        "registry_path": str((metadata_core_dir / "field_registry.json").resolve()),
        "reviews_path": str(reviews_path.resolve()),
    }


def list_dataset_ids(data_root: Path) -> list[str]:
    return [path.name for path in sorted(data_root.iterdir()) if path.is_dir() and not path.name.startswith(".")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw CSV field registries with optional per-column AI review.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-ids", default="", help="Comma-separated dataset ids. Defaults to all datasets.")
    parser.add_argument("--ai-provider", choices=["none", "claude"], default="none")
    parser.add_argument("--ai-timeout-sec", type=int, default=60)
    parser.add_argument("--force-ai", action="store_true", help="Ignore cached AI reviews and call AI again.")
    parser.add_argument("--max-columns", type=int, default=None, help="Debug/testing limit per dataset.")
    parser.add_argument("--report-path", type=Path, default=REPO_ROOT / "tmp" / "raw_field_registry_ai_review_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_ids = [item.strip() for item in args.dataset_ids.split(",") if item.strip()] or list_dataset_ids(args.data_root)
    summaries = []
    for dataset_id in dataset_ids:
        summaries.append(
            materialize_dataset(
                args.data_root / dataset_id,
                dataset_id,
                ai_provider=args.ai_provider,
                ai_timeout_sec=args.ai_timeout_sec,
                force_ai=args.force_ai,
                max_columns=args.max_columns,
            )
        )
    write_json(args.report_path, {"generated_at": now_iso(), "data_root": str(args.data_root.resolve()), "datasets": summaries})
    print(json.dumps({"dataset_count": len(summaries), "report_path": str(args.report_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    raise SystemExit(main())
