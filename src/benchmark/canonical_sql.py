"""Canonical SQL and stable identity helpers for benchmark contracts."""

from __future__ import annotations

import hashlib
import re


def stable_hash(text: str, length: int = 16) -> str:
    payload = (text or "").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def canonicalize_sql(sql: str) -> str:
    """Conservative SQL normalization for cross-run comparison.

    This function intentionally avoids aggressive SQL rewrites. It only removes
    superficial formatting differences and normalizes simple identifier quoting.
    """
    text = (sql or "").strip()
    if not text:
        return ""
    text = re.sub(r"```[a-zA-Z0-9_-]*", "", text)
    text = text.replace("```", "")
    text = text.strip().rstrip(";")
    text = re.sub(r"\s+", " ", text)
    # Normalize double-quoted simple identifiers: "foo" -> foo
    text = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r"\1", text)
    return text.lower()


def canonical_sql_hash(sql: str, length: int = 20) -> str:
    return stable_hash(canonicalize_sql(sql), length=length)


def stable_question_identity(
    *,
    dataset_id: str,
    family_id: str,
    intended_facet_id: str,
    question_text: str,
    length: int = 20,
) -> str:
    payload = "|".join(
        [
            (dataset_id or "").strip().lower(),
            (family_id or "").strip().lower(),
            (intended_facet_id or "unknown").strip().lower(),
            re.sub(r"\s+", " ", (question_text or "").strip().lower()),
        ]
    )
    return f"rqk_{stable_hash(payload, length=length)}"


def stable_query_identity(
    *,
    dataset_id: str,
    family_id: str,
    intended_facet_id: str,
    stable_question_id: str,
    variant_semantic_role: str,
    canonical_sql: str,
    length: int = 20,
) -> str:
    payload = "|".join(
        [
            (dataset_id or "").strip().lower(),
            (family_id or "").strip().lower(),
            (intended_facet_id or "unknown").strip().lower(),
            (stable_question_id or "").strip().lower(),
            (variant_semantic_role or "").strip().lower(),
            (canonical_sql or "").strip().lower(),
        ]
    )
    return f"qsk_{stable_hash(payload, length=length)}"

