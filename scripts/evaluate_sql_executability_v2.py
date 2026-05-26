#!/usr/bin/env python3
"""Estimate SQL executability quality after V2 relabeling and deduplication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_phase_c_sql_inventory import (  # noqa: E402
    CODE_FRAGMENT_PATTERNS,
    NONSTANDARD_QUERY_PATTERNS,
    SQL_START_PATTERN,
    leading_sql_candidate,
    standalone_sql_like,
)
from scripts.build_sql_high_corpus_phase_c_sql_inventory import clean_sql_text  # noqa: E402


DEFAULT_INPUT = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/dedup/master_sql_inventory_dedup_v2.csv"
)
DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404/v2_refinement")

NEW_FIELDS = [
    "detected_dialect_family_v2",
    "detected_dialect_markers_v2",
    "sql_text_for_eval_v2",
    "rewrite_applied_v2",
    "rewrite_type_v2",
    "executable_status_v2",
    "executable_reason_v2",
]

CSV_FIELDNAMES = [
    "own_id",
    "dataset_id",
    "dataset_name",
    "pass_count",
    "fail_count",
    "unknown_count",
    "rewrite_applied_count",
    "primary_rows",
    "non_primary_rows",
]

SQL_START_FALLBACK = re.compile(
    r"(?i)\b(with|select|insert\s+into|update|delete\s+from|create\s+"
    r"(or\s+replace\s+)?(table|view|database|schema|function|procedure|index|trigger)"
    r"|drop\s+(table|database|view|schema)|alter\s+table|truncate\s+table|use\s+\w+"
    r"|show\s+\w+|describe\s+\w+|explain\s+\w+|merge\s+into|call\s+\w+)\b"
)

UNBALANCED_TEMPLATE_PATTERNS = (
    "{{",
    "}}",
    "{%",
    "%}",
    "${",
    "<%",
    "%>",
)

DIALECT_MARKER_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "tsql": [
        ("tsql_go_batch", re.compile(r"(?im)^\s*go\s*;?\s*$")),
        ("tsql_use_database", re.compile(r"(?i)\buse\s+[a-z_][a-z0-9_$]*\b")),
        ("tsql_select_top", re.compile(r"(?i)^\s*select\s+(?:distinct\s+)?top\s*\(?\d+\)?\s+")),
        ("tsql_identity", re.compile(r"(?i)\bidentity\s*(?:\(|\b)")),
        ("tsql_getdate", re.compile(r"(?i)\bgetdate\s*\(\s*\)")),
        ("tsql_dateadd", re.compile(r"(?i)\bdateadd\s*\(")),
        ("tsql_datediff", re.compile(r"(?i)\bdatediff\s*\(")),
        ("tsql_isnull", re.compile(r"(?i)\bisnull\s*\(")),
        ("tsql_len", re.compile(r"(?i)\blen\s*\(")),
        ("tsql_object_id", re.compile(r"(?i)\bobject_id\s*\(")),
        ("tsql_sys_catalog", re.compile(r"(?i)\bsys\.")),
        ("tsql_scope_identity", re.compile(r"(?i)\bscope_identity\s*\(\s*\)")),
        ("tsql_create_or_alter", re.compile(r"(?i)\bcreate\s+or\s+alter\b")),
        ("tsql_square_identifiers", re.compile(r"\[[^\]]+\]")),
        ("tsql_output_clause", re.compile(r"(?i)\boutput\b")),
    ],
    "mysql": [
        ("mysql_auto_increment", re.compile(r"(?i)\bauto_increment\b")),
        ("mysql_engine_clause", re.compile(r"(?i)\)\s*engine\s*=")),
        ("mysql_default_charset", re.compile(r"(?i)\bdefault\s+charset\b")),
        ("mysql_collation", re.compile(r"(?i)\bcollate\s+[a-z0-9_]+")),
        ("mysql_show_statement", re.compile(r"(?i)^\s*show\s+\w+")),
        ("mysql_describe_statement", re.compile(r"(?i)^\s*describe\s+\w+")),
        ("mysql_delimiter", re.compile(r"(?i)^\s*delimiter\b")),
        ("mysql_replace_into", re.compile(r"(?i)^\s*replace\s+into\b")),
        ("mysql_unsigned", re.compile(r"(?i)\bunsigned\b")),
        ("mysql_backtick_identifier", re.compile(r"`[^`]+`")),
    ],
    "postgresql": [
        ("postgres_cast_operator", re.compile(r"::\s*[a-z_][a-z0-9_]*", re.IGNORECASE)),
        ("postgres_ilike", re.compile(r"(?i)\bilike\b")),
        ("postgres_serial", re.compile(r"(?i)\bserial\b")),
        ("postgres_returning", re.compile(r"(?i)\breturning\b")),
        ("postgres_now", re.compile(r"(?i)\bnow\s*\(\s*\)")),
        ("postgres_distinct_on", re.compile(r"(?i)\bdistinct\s+on\b")),
        ("postgres_array_agg", re.compile(r"(?i)\barray_agg\s*\(")),
        ("postgres_regexp_replace", re.compile(r"(?i)\bregexp_replace\s*\(")),
    ],
    "oracle": [
        ("oracle_varchar2", re.compile(r"(?i)\bvarchar2\b")),
        ("oracle_nvl", re.compile(r"(?i)\bnvl\s*\(")),
        ("oracle_decode", re.compile(r"(?i)\bdecode\s*\(")),
        ("oracle_rownum", re.compile(r"(?i)\brownum\b")),
        ("oracle_dual", re.compile(r"(?i)\bfrom\s+dual\b")),
        ("oracle_create_or_replace", re.compile(r"(?i)\bcreate\s+or\s+replace\b")),
        ("oracle_plsql_block", re.compile(r"(?is)^\s*begin\b.*\bend\s*;?\s*$")),
    ],
    "procedural_or_template": [
        ("templated_jinja", re.compile(r"\{\{|\}\}|\{%|%\}")),
        ("template_dollar", re.compile(r"\$\{")),
        ("session_variable", re.compile(r"(?i)@[a-z_][a-z0-9_]*")),
        ("procedure_definition", re.compile(r"(?i)\bcreate\s+(?:or\s+replace\s+|or\s+alter\s+)?procedure\b")),
        ("function_definition", re.compile(r"(?i)\bcreate\s+(?:or\s+replace\s+)?function\b")),
        ("if_begin_end_block", re.compile(r"(?is)^\s*if\b.*\bbegin\b")),
        ("merge_statement", re.compile(r"(?i)^\s*merge\s+")),
        ("declare_block", re.compile(r"(?i)^\s*declare\b")),
        ("exec_statement", re.compile(r"(?i)^\s*exec(?:ute)?\b")),
    ],
}

PASS_LIKE_ERRORS = (
    "no such table",
    "no such column",
    "ambiguous column name",
    "incorrect number of bindings supplied",
)
PASS_EXISTS_SUFFIX = "already exists"
UNKNOWN_LIKE_ERRORS = (
    "unknown database",
    "no such function",
    "no such collation sequence",
    "unrecognized token",
    "wrong number of arguments to function",
)
FAIL_LIKE_ERRORS = (
    "syntax error",
    "incomplete input",
)
UNSUPPORTED_SQLITE_PREFIXES = (
    "use ",
    "show ",
    "describe ",
)
PROCEDURAL_PREFIXES = (
    "if ",
    "go if ",
    "begin ",
    "declare ",
    "exec ",
    "execute ",
    "create or alter procedure",
    "create or replace procedure",
    "create procedure",
    "create function",
    "create or replace function",
    "merge ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate SQLite executability quality for the V2-deduplicated SQL "
            "inventory using syntax checks and conservative rewrites."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def strip_wrapping_quotes(text: str) -> str:
    cleaned = (text or "").strip()
    while True:
        previous = cleaned
        cleaned = re.sub(r'^\s*(?:"""|\'\'\'|["\'])\s*', "", cleaned)
        cleaned = re.sub(r'\s*(?:"""|\'\'\'|["\'])\s*$', "", cleaned)
        if cleaned == previous:
            return cleaned.strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_newlines(text)).strip()


def initial_sql_text(row: dict[str, str]) -> str:
    return (row.get("sql_text_clean") or "").strip() or (row.get("sql_text_raw") or "").strip()


def fallback_extract_sql(text: str) -> str:
    candidate = normalize_newlines(text or "").strip()
    if not candidate:
        return ""
    match = SQL_START_FALLBACK.search(candidate)
    if match and match.start() <= 450:
        return candidate[match.start():].strip()
    return candidate


def prepare_eval_text(text: str) -> str:
    candidate = clean_sql_text(text or "")
    candidate = normalize_newlines(candidate).replace("\ufeff", "").strip()
    candidate = strip_wrapping_quotes(candidate)
    candidate = re.sub(r"\\(?=\s)", "", candidate)
    lowered = candidate.lower().lstrip()
    if not lowered.startswith(PROCEDURAL_PREFIXES):
        leading_candidate = leading_sql_candidate(candidate)
        if leading_candidate:
            candidate = leading_candidate
    if (not candidate or not standalone_sql_like(candidate)) and not lowered.startswith(PROCEDURAL_PREFIXES):
        candidate = fallback_extract_sql(clean_sql_text(text or ""))
        candidate = strip_wrapping_quotes(candidate)
    candidate = normalize_newlines(candidate).strip()
    return candidate


def obvious_non_sql_fragment(text: str) -> bool:
    lowered = (text or "").lower()
    if any(pattern in lowered for pattern in NONSTANDARD_QUERY_PATTERNS):
        return True
    if any(pattern in lowered for pattern in CODE_FRAGMENT_PATTERNS):
        return True
    if any(pattern in text for pattern in UNBALANCED_TEMPLATE_PATTERNS):
        return True
    if not standalone_sql_like(text):
        starts = (text or "").lstrip().lower()
        if not any(
            starts.startswith(prefix)
            for prefix in (
                "select",
                "with",
                "insert",
                "update",
                "delete",
                "create",
                "alter",
                "drop",
                "use",
                "show",
                "describe",
                "merge",
                "call",
                "explain",
                "if",
                "begin",
                "declare",
                "exec",
                "execute",
            )
        ):
            return True
    return False


def bracket_balance_problem(text: str) -> bool:
    candidate = text or ""
    if candidate.count("{") != candidate.count("}"):
        return True
    if candidate.count("(") < candidate.count(")"):
        return True
    if candidate.count("'") % 2 == 1 and "''" not in candidate:
        return True
    return False


def detect_dialect_markers(text: str) -> tuple[str, list[str]]:
    markers: list[str] = []
    families: Counter[str] = Counter()
    for family, family_patterns in DIALECT_MARKER_PATTERNS.items():
        for marker_name, pattern in family_patterns:
            if pattern.search(text or ""):
                markers.append(marker_name)
                families[family] += 1
    if not families:
        return "", []
    family_name = ",".join(name for name, _count in families.most_common())
    return family_name, sorted(set(markers))


def normalize_bracket_identifiers(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = match.group(1).replace('"', '""')
        return f'"{inner}"'

    return re.sub(r"\[([^\]]+)\]", repl, text)


def rewrite_select_top(text: str) -> tuple[str, str]:
    pattern = re.compile(r"(?is)^\s*select\s+(distinct\s+)?top\s*\(?(\d+)\)?\s+")
    match = pattern.match(text)
    if not match:
        return text, ""
    distinct_part = match.group(1) or ""
    limit_value = match.group(2)
    remainder = text[match.end():].strip()
    if not remainder:
        return text, ""
    semicolon = remainder.rfind(";")
    if semicolon != -1:
        body = remainder[:semicolon].rstrip()
        suffix = remainder[semicolon:]
    else:
        body = remainder.rstrip()
        suffix = ""
    if re.search(r"(?i)\blimit\s+\d+\b", body):
        return text, ""
    rewritten = f"SELECT {distinct_part}{body} LIMIT {limit_value}{suffix}".strip()
    return rewritten, "select_top_to_limit"


def conservative_rewrite(text: str) -> tuple[str, list[str]]:
    rewritten = text
    rewrite_types: list[str] = []

    next_text = re.sub(r"(?im)^\s*go\s*;?\s*$", "", rewritten)
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("strip_go_batch")

    next_text = re.sub(
        r"(?is)^\s*go\s+(?=(if|create|alter|select|with|insert|update|delete|merge|declare|begin|exec(?:ute)?)\b)",
        "",
        rewritten,
        count=1,
    )
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("drop_leading_go_prefix")

    next_text = re.sub(r"(?is)^\s*use\s+[a-z_][a-z0-9_$]*\s*;\s*", "", rewritten)
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("drop_use_prefix")

    next_text = re.sub(r"\\(?=\s)", "", rewritten)
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("remove_line_continuation_backslashes")

    next_text = normalize_bracket_identifiers(rewritten)
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("normalize_square_brackets")

    next_text = re.sub(r"(?i)\bgetdate\s*\(\s*\)", "CURRENT_TIMESTAMP", rewritten)
    next_text = re.sub(r"(?i)\bnow\s*\(\s*\)", "CURRENT_TIMESTAMP", next_text)
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("portable_current_timestamp")

    next_text = re.sub(r"(?i)\bisnull\s*\(", "IFNULL(", rewritten)
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("isnull_to_ifnull")

    next_text = re.sub(r"(?i)\blen\s*\(", "LENGTH(", rewritten)
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("len_to_length")

    next_text = re.sub(r"(?i)\bcollate\s+[a-z0-9_]+", "", rewritten)
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("drop_named_collation")

    next_text = re.sub(r"(?i)\)\s*engine\s*=\s*[a-z0-9_]+", ")", rewritten)
    next_text = re.sub(r"(?i)\)\s*default\s+charset\s*=\s*[a-z0-9_]+", ")", next_text)
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("drop_mysql_table_options")

    next_text = re.sub(r"(?i)\bcreate\s+or\s+replace\s+view\b", "CREATE VIEW", rewritten)
    if next_text != rewritten:
        rewritten = next_text
        rewrite_types.append("create_or_replace_view_to_create_view")

    next_text, top_rewrite = rewrite_select_top(rewritten)
    if top_rewrite:
        rewritten = next_text
        rewrite_types.append(top_rewrite)

    rewritten = normalize_newlines(rewritten)
    rewritten = re.sub(r"\n{3,}", "\n\n", rewritten).strip()
    return rewritten, rewrite_types


def classify_sqlite_result(message: str) -> str:
    lowered = (message or "").lower()
    if not lowered:
        return "ok"
    if any(token in lowered for token in UNKNOWN_LIKE_ERRORS):
        return "unknown_like"
    if lowered.endswith(PASS_EXISTS_SUFFIX):
        return "pass_like"
    if any(token in lowered for token in PASS_LIKE_ERRORS):
        return "pass_like"
    if any(token in lowered for token in FAIL_LIKE_ERRORS):
        return "fail_like"
    return "other"


def execute_sqlite(text: str) -> dict[str, str]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("PRAGMA foreign_keys=OFF;")
        connection.executescript(text)
        return {"result": "ok", "message": ""}
    except sqlite3.Error as exc:
        message = str(exc).strip()
        return {"result": classify_sqlite_result(message), "message": message}
    finally:
        connection.close()


def reason_text(status: str, reason_code: str, markers: list[str], sqlite_message: str, rewrite_types: list[str]) -> str:
    fragments = [reason_code.replace("_", " ")]
    if rewrite_types:
        fragments.append("rewrite=" + ",".join(rewrite_types))
    if markers:
        fragments.append("dialect_markers=" + ",".join(markers[:8]))
    if sqlite_message:
        fragments.append("sqlite=" + sqlite_message)
    return "; ".join(fragments)


def evaluate_row(
    row: dict[str, str],
    eval_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sql_source_text = initial_sql_text(row)
    prepared = prepare_eval_text(sql_source_text)
    family, markers = detect_dialect_markers(prepared or sql_source_text)
    attempts: list[dict[str, Any]] = []

    if row.get("v2_specificity_label") == "reject_non_sql":
        reason_code = "reject_non_sql_v2"
        final_row = {
            "detected_dialect_family_v2": family,
            "detected_dialect_markers_v2": json.dumps(markers, ensure_ascii=False),
            "sql_text_for_eval_v2": prepared,
            "rewrite_applied_v2": "no",
            "rewrite_type_v2": "",
            "executable_status_v2": "fail",
            "executable_reason_v2": reason_text("fail", reason_code, markers, "", []),
        }
        attempts.append(
            {
                "attempt_index": 0,
                "attempt_kind": "short_circuit_reject",
                "rewrite_applied": False,
                "rewrite_types": [],
                "result": "fail",
                "reason_code": reason_code,
                "sqlite_result": "",
                "sqlite_message": "",
                "sql_text_for_eval": prepared,
            }
        )
        return final_row, {"family": family, "markers": markers, "attempts": attempts}

    if not prepared:
        reason_code = "empty_after_cleanup"
        final_row = {
            "detected_dialect_family_v2": family,
            "detected_dialect_markers_v2": json.dumps(markers, ensure_ascii=False),
            "sql_text_for_eval_v2": "",
            "rewrite_applied_v2": "no",
            "rewrite_type_v2": "",
            "executable_status_v2": "fail",
            "executable_reason_v2": reason_text("fail", reason_code, markers, "", []),
        }
        attempts.append(
            {
                "attempt_index": 0,
                "attempt_kind": "empty_after_cleanup",
                "rewrite_applied": False,
                "rewrite_types": [],
                "result": "fail",
                "reason_code": reason_code,
                "sqlite_result": "",
                "sqlite_message": "",
                "sql_text_for_eval": "",
            }
        )
        return final_row, {"family": family, "markers": markers, "attempts": attempts}

    non_sql_fragment = obvious_non_sql_fragment(prepared)
    delimiter_problem = bracket_balance_problem(prepared)

    def cached_eval(text: str) -> dict[str, Any]:
        if text not in eval_cache:
            eval_cache[text] = execute_sqlite(text)
        return eval_cache[text]

    original_eval = cached_eval(prepared)
    attempts.append(
        {
            "attempt_index": 0,
            "attempt_kind": "original",
            "rewrite_applied": False,
            "rewrite_types": [],
            "result": original_eval["result"],
            "reason_code": "",
            "sqlite_result": original_eval["result"],
            "sqlite_message": original_eval["message"],
            "sql_text_for_eval": prepared,
        }
    )

    rewritten, rewrite_types = conservative_rewrite(prepared)
    rewritten_eval: dict[str, Any] | None = None
    if rewrite_types and rewritten and rewritten != prepared:
        rewritten_eval = cached_eval(rewritten)
        attempts.append(
            {
                "attempt_index": 1,
                "attempt_kind": "conservative_rewrite",
                "rewrite_applied": True,
                "rewrite_types": rewrite_types,
                "result": rewritten_eval["result"],
                "reason_code": "",
                "sqlite_result": rewritten_eval["result"],
                "sqlite_message": rewritten_eval["message"],
                "sql_text_for_eval": rewritten,
            }
        )

    if original_eval["result"] in {"ok", "pass_like"}:
        reason_code = "sqlite_original_ok" if original_eval["result"] == "ok" else "sqlite_original_schema_only"
        return (
            {
                "detected_dialect_family_v2": family,
                "detected_dialect_markers_v2": json.dumps(markers, ensure_ascii=False),
                "sql_text_for_eval_v2": prepared,
                "rewrite_applied_v2": "no",
                "rewrite_type_v2": "",
                "executable_status_v2": "pass",
                "executable_reason_v2": reason_text("pass", reason_code, markers, original_eval["message"], []),
            },
            {"family": family, "markers": markers, "attempts": attempts},
        )

    if rewritten_eval and rewritten_eval["result"] in {"ok", "pass_like"}:
        reason_code = "sqlite_rewrite_ok" if rewritten_eval["result"] == "ok" else "sqlite_rewrite_schema_only"
        return (
            {
                "detected_dialect_family_v2": family,
                "detected_dialect_markers_v2": json.dumps(markers, ensure_ascii=False),
                "sql_text_for_eval_v2": rewritten,
                "rewrite_applied_v2": "yes",
                "rewrite_type_v2": ",".join(rewrite_types),
                "executable_status_v2": "pass",
                "executable_reason_v2": reason_text("pass", reason_code, markers, rewritten_eval["message"], rewrite_types),
            },
            {"family": family, "markers": markers, "attempts": attempts},
        )

    if non_sql_fragment:
        reason_code = "non_sql_or_code_fragment"
        return (
            {
                "detected_dialect_family_v2": family,
                "detected_dialect_markers_v2": json.dumps(markers, ensure_ascii=False),
                "sql_text_for_eval_v2": prepared,
                "rewrite_applied_v2": "no",
                "rewrite_type_v2": "",
                "executable_status_v2": "fail",
                "executable_reason_v2": reason_text("fail", reason_code, markers, original_eval["message"], []),
            },
            {"family": family, "markers": markers, "attempts": attempts},
        )

    if delimiter_problem or sqlite3.complete_statement(prepared) is False and original_eval["message"].lower().startswith("incomplete input"):
        reason_code = "truncated_or_incomplete_sql"
        return (
            {
                "detected_dialect_family_v2": family,
                "detected_dialect_markers_v2": json.dumps(markers, ensure_ascii=False),
                "sql_text_for_eval_v2": prepared,
                "rewrite_applied_v2": "no",
                "rewrite_type_v2": "",
                "executable_status_v2": "fail",
                "executable_reason_v2": reason_text("fail", reason_code, markers, original_eval["message"], []),
            },
            {"family": family, "markers": markers, "attempts": attempts},
        )

    if prepared.lower().startswith(UNSUPPORTED_SQLITE_PREFIXES):
        reason_code = "session_or_metadata_command_non_sqlite"
        return (
            {
                "detected_dialect_family_v2": family,
                "detected_dialect_markers_v2": json.dumps(markers, ensure_ascii=False),
                "sql_text_for_eval_v2": prepared,
                "rewrite_applied_v2": "no",
                "rewrite_type_v2": "",
                "executable_status_v2": "unknown",
                "executable_reason_v2": reason_text("unknown", reason_code, markers, original_eval["message"], []),
            },
            {"family": family, "markers": markers, "attempts": attempts},
        )

    if original_eval["result"] == "unknown_like" or (rewritten_eval and rewritten_eval["result"] == "unknown_like"):
        chosen_eval = rewritten_eval or original_eval
        reason_code = "dialect_specific_non_sqlite_syntax"
        return (
            {
                "detected_dialect_family_v2": family,
                "detected_dialect_markers_v2": json.dumps(markers, ensure_ascii=False),
                "sql_text_for_eval_v2": rewritten if rewritten_eval else prepared,
                "rewrite_applied_v2": "yes" if rewritten_eval else "no",
                "rewrite_type_v2": ",".join(rewrite_types) if rewritten_eval else "",
                "executable_status_v2": "unknown",
                "executable_reason_v2": reason_text("unknown", reason_code, markers, chosen_eval["message"], rewrite_types if rewritten_eval else []),
            },
            {"family": family, "markers": markers, "attempts": attempts},
        )

    if markers:
        chosen_eval = rewritten_eval or original_eval
        reason_code = "dialect_markers_persist_after_rewrite"
        return (
            {
                "detected_dialect_family_v2": family,
                "detected_dialect_markers_v2": json.dumps(markers, ensure_ascii=False),
                "sql_text_for_eval_v2": rewritten if rewritten_eval else prepared,
                "rewrite_applied_v2": "yes" if rewritten_eval else "no",
                "rewrite_type_v2": ",".join(rewrite_types) if rewritten_eval else "",
                "executable_status_v2": "unknown",
                "executable_reason_v2": reason_text("unknown", reason_code, markers, chosen_eval["message"], rewrite_types if rewritten_eval else []),
            },
            {"family": family, "markers": markers, "attempts": attempts},
        )

    reason_code = "sqlite_syntax_failure"
    chosen_eval = rewritten_eval or original_eval
    return (
        {
            "detected_dialect_family_v2": family,
            "detected_dialect_markers_v2": json.dumps(markers, ensure_ascii=False),
            "sql_text_for_eval_v2": rewritten if rewritten_eval else prepared,
            "rewrite_applied_v2": "yes" if rewritten_eval else "no",
            "rewrite_type_v2": ",".join(rewrite_types) if rewritten_eval else "",
            "executable_status_v2": "fail",
            "executable_reason_v2": reason_text("fail", reason_code, markers, chosen_eval["message"], rewrite_types if rewritten_eval else []),
        },
        {"family": family, "markers": markers, "attempts": attempts},
    )


def dataset_sort_key(item: tuple[str, Counter[str]]) -> tuple[int, str]:
    own_id, counter = item
    return (-(counter["pass"] + counter["fail"] + counter["unknown"]), own_id)


def render_summary(
    now_utc: str,
    input_path: Path,
    output_csv: Path,
    rewrite_jsonl: Path,
    dataset_counts: dict[str, Counter[str]],
    dataset_meta: dict[str, tuple[str, str]],
    global_counts: Counter[str],
    rewrite_counter: Counter[str],
    family_counter: Counter[str],
) -> str:
    lines = [
        "# V2 Executability Summary",
        "",
        f"- Generated at UTC: `{now_utc}`",
        f"- Input inventory: `{input_path.resolve()}`",
        f"- Output CSV: `{output_csv.resolve()}`",
        f"- Rewrite attempt ledger: `{rewrite_jsonl.resolve()}`",
        "",
        "## Global Counts",
        "",
        f"- Rows evaluated: {global_counts['rows']}",
        f"- Pass: {global_counts['pass']}",
        f"- Fail: {global_counts['fail']}",
        f"- Unknown: {global_counts['unknown']}",
        f"- Rows with rewrite applied: {global_counts['rewrite_applied']}",
        "",
        "## Rewrite Types",
        "",
    ]
    if rewrite_counter:
        for rewrite_type, count in rewrite_counter.most_common():
            lines.append(f"- `{rewrite_type}`: {count}")
    else:
        lines.append("- No rewrites were applied.")

    lines.extend(
        [
            "",
            "## Dialect Marker Families",
            "",
        ]
    )
    if family_counter:
        for family, count in family_counter.most_common():
            lines.append(f"- `{family}`: {count}")
    else:
        lines.append("- No dialect markers detected.")

    lines.extend(
        [
            "",
            "## Per-Dataset Counts",
            "",
            "| own_id | dataset_name | pass | fail | unknown | rewrite_applied | primary_rows | non_primary_rows |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for own_id, counter in sorted(dataset_counts.items(), key=dataset_sort_key):
        dataset_name = dataset_meta.get(own_id, ("", ""))[1]
        lines.append(
            "| {own_id} | {dataset_name} | {pass_count} | {fail_count} | {unknown_count} | "
            "{rewrite_applied} | {primary_rows} | {non_primary_rows} |".format(
                own_id=own_id,
                dataset_name=dataset_name,
                pass_count=counter["pass"],
                fail_count=counter["fail"],
                unknown_count=counter["unknown"],
                rewrite_applied=counter["rewrite_applied"],
                primary_rows=counter["primary_rows"],
                non_primary_rows=counter["non_primary_rows"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_root = args.output_root.resolve()
    output_dir = output_root / "execute"
    output_csv = output_dir / "sql_executability_v2.csv"
    rewrite_jsonl = output_dir / "sql_rewrite_attempts_v2.jsonl"
    summary_md = output_dir / "executability_summary_v2.md"
    manifest_json = output_dir / "run_manifest_v2_phase3.json"

    now_utc = utc_now_iso()
    input_rows = read_csv_rows(input_path)
    fieldnames = list(input_rows[0].keys()) + [field for field in NEW_FIELDS if field not in input_rows[0]]

    eval_cache: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_meta: dict[str, tuple[str, str]] = {}
    global_counts: Counter[str] = Counter()
    rewrite_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()

    for row in input_rows:
        own_id = (row.get("own_id") or "").strip()
        dataset_id = (row.get("dataset_id") or "").strip()
        dataset_name = (row.get("dataset_name") or "").strip()
        dataset_meta[own_id] = (dataset_id, dataset_name)

        evaluated, attempt_bundle = evaluate_row(row, eval_cache)
        enriched = dict(row)
        enriched.update(evaluated)
        output_rows.append(enriched)

        status = evaluated["executable_status_v2"]
        dataset_counts[own_id][status] += 1
        dataset_counts[own_id]["primary_rows" if row.get("is_primary_canonical") == "yes" else "non_primary_rows"] += 1
        global_counts["rows"] += 1
        global_counts[status] += 1

        if evaluated["rewrite_applied_v2"] == "yes":
            dataset_counts[own_id]["rewrite_applied"] += 1
            global_counts["rewrite_applied"] += 1
            for part in filter(None, evaluated["rewrite_type_v2"].split(",")):
                rewrite_counter[part] += 1

        family = attempt_bundle["family"]
        if family:
            family_counter[family] += 1

        attempt_rows.append(
            {
                "own_id": own_id,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "sql_item_id": row.get("sql_item_id", ""),
                "is_primary_canonical": row.get("is_primary_canonical", ""),
                "canonical_group_id": row.get("canonical_group_id", ""),
                "v2_specificity_label": row.get("v2_specificity_label", ""),
                "detected_dialect_family_v2": family,
                "detected_dialect_markers_v2": attempt_bundle["markers"],
                "selected_rewrite_applied_v2": evaluated["rewrite_applied_v2"],
                "selected_rewrite_type_v2": evaluated["rewrite_type_v2"],
                "selected_executable_status_v2": evaluated["executable_status_v2"],
                "selected_executable_reason_v2": evaluated["executable_reason_v2"],
                "attempts": attempt_bundle["attempts"],
            }
        )

    write_csv(output_csv, fieldnames, output_rows)
    write_jsonl(rewrite_jsonl, attempt_rows)

    summary_text = render_summary(
        now_utc=now_utc,
        input_path=input_path,
        output_csv=output_csv,
        rewrite_jsonl=rewrite_jsonl,
        dataset_counts=dataset_counts,
        dataset_meta=dataset_meta,
        global_counts=global_counts,
        rewrite_counter=rewrite_counter,
        family_counter=family_counter,
    )
    write_text(summary_md, summary_text)

    manifest_payload = {
        "phase": "v2_phase3_executability_estimation",
        "generated_at_utc": now_utc,
        "input": {
            "dedup_inventory_path": str(input_path),
            "dedup_inventory_sha256": sha256_file(input_path),
        },
        "summary": {
            "rows_evaluated": global_counts["rows"],
            "pass": global_counts["pass"],
            "fail": global_counts["fail"],
            "unknown": global_counts["unknown"],
            "rewrite_applied": global_counts["rewrite_applied"],
            "dialect_family_counts": dict(family_counter),
        },
        "outputs": [],
    }

    write_json(manifest_json, manifest_payload)

    outputs = [output_csv, rewrite_jsonl, summary_md, manifest_json]
    manifest_payload["outputs"] = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in outputs
    ]
    write_json(manifest_json, manifest_payload)

    for own_id in sorted(dataset_counts):
        counter = dataset_counts[own_id]
        print(
            f"{own_id}\tpass={counter['pass']}\tfail={counter['fail']}\tunknown={counter['unknown']}"
        )


if __name__ == "__main__":
    main()
