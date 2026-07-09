#!/usr/bin/env python3
"""Preprocess SQL corpus into benchmark-ready exemplar pools.

This script is intentionally independent from the existing SQL-corpus build pipeline.
It converts the current corpus into three buckets for benchmark construction usage:

1. core
2. convertible
3. exclude

Default input is the V2 deduplicated inventory, and defaults favor:
- keep-candidate rows only
- primary canonical rows only

Usage:
  python3 scripts/preprocess_sql_exemplars_for_benchmark.py
  python3 scripts/preprocess_sql_exemplars_for_benchmark.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_INPUT_CSV = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/dedup/master_sql_inventory_dedup_v2.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/preprocessed_sql_exemplars"
)

READONLY_HEAD_PAT = re.compile(r"^\s*(select|with)\b", flags=re.IGNORECASE)
MUTATING_OR_DDL_PAT = re.compile(
    r"\b(create|alter|drop|insert|update|delete|merge|truncate|grant|revoke|begin|commit|rollback)\b",
    flags=re.IGNORECASE,
)
PREAMBLE_PAT = re.compile(r"^\s*(use|set)\b", flags=re.IGNORECASE)
WRAPPER_HINT_PATTERNS = [
    re.compile(r"\bsqldf\s*\(", flags=re.IGNORECASE),
    re.compile(r"\bread\.csv\s*\(", flags=re.IGNORECASE),
    re.compile(r"\bhead\s*\(", flags=re.IGNORECASE),
    re.compile(r"^\s*#", flags=re.IGNORECASE),
    re.compile(r"```"),
]


@dataclass
class ProcessedRow:
    sql_item_id: str
    own_id: str
    dataset_id: str
    dataset_name: str
    source_url: str
    source_type: str
    source_title: str
    query_intent_label: str
    family_tag_guess: str
    v2_specificity_label: str
    v2_specificity_reason_code: str
    v2_keep_candidate: str
    is_primary_canonical: str
    canonical_group_id: str
    duplicate_type: str
    preprocess_bucket: str
    preprocess_reason: str
    conversion_applied: str
    extraction_confidence: str
    sql_text_input: str
    sql_text_prepared: str
    sql_text_prepared_hash: str
    notes: str

    def to_dict(self) -> dict[str, str]:
        return {
            "sql_item_id": self.sql_item_id,
            "own_id": self.own_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "query_intent_label": self.query_intent_label,
            "family_tag_guess": self.family_tag_guess,
            "v2_specificity_label": self.v2_specificity_label,
            "v2_specificity_reason_code": self.v2_specificity_reason_code,
            "v2_keep_candidate": self.v2_keep_candidate,
            "is_primary_canonical": self.is_primary_canonical,
            "canonical_group_id": self.canonical_group_id,
            "duplicate_type": self.duplicate_type,
            "preprocess_bucket": self.preprocess_bucket,
            "preprocess_reason": self.preprocess_reason,
            "conversion_applied": self.conversion_applied,
            "extraction_confidence": self.extraction_confidence,
            "sql_text_input": self.sql_text_input,
            "sql_text_prepared": self.sql_text_prepared,
            "sql_text_prepared_hash": self.sql_text_prepared_hash,
            "notes": self.notes,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess SQL corpus into benchmark-ready exemplar pools."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--only-keep-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When true, keep only rows with v2_keep_candidate=yes.",
    )
    parser.add_argument(
        "--only-primary-canonical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When true, keep only rows with is_primary_canonical=yes.",
    )
    parser.add_argument(
        "--strict-read-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When true, exclude converted SQL if mutating/DDL tokens remain.",
    )
    return parser.parse_args()


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "")
    )


def _strip_markdown_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[A-Za-z0-9_-]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw)
    return raw.strip()


def _remove_comment_prefix_lines(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("--"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _split_sql_statements(text: str) -> list[str]:
    statements: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
        elif ch == ";" and not in_single and not in_double:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _first_readonly_select_statement(text: str) -> str:
    cleaned = _remove_comment_prefix_lines(_strip_markdown_fence(_normalize_quotes(text)))
    if not cleaned:
        return ""
    for stmt in _split_sql_statements(cleaned):
        s = " ".join(stmt.split())
        if not s:
            continue
        if PREAMBLE_PAT.match(s):
            continue
        if READONLY_HEAD_PAT.match(s) and not MUTATING_OR_DDL_PAT.search(s):
            return stmt.strip() + ";"

    # fallback: recover a SELECT/WITH span from wrapper text
    m = re.search(r"\b(select|with)\b", cleaned, flags=re.IGNORECASE)
    if m:
        tail = cleaned[m.start() :].strip()
        candidate = _split_sql_statements(tail)[0] if _split_sql_statements(tail) else tail
        c = " ".join(candidate.split())
        if READONLY_HEAD_PAT.match(c) and not MUTATING_OR_DDL_PAT.search(c):
            return candidate.strip().rstrip(";") + ";"
    return ""


def _extract_sqldf_embedded_sql(text: str) -> str:
    norm = _normalize_quotes(text)
    # sqldf("SELECT ...")
    m = re.search(
        r"sqldf\s*\(\s*([\"'])(?P<sql>.*?)(?<!\\)\1\s*\)",
        norm,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ""
    sql = m.group("sql").replace('\\"', '"').replace("\\'", "'")
    sql = sql.strip()
    if not sql:
        return ""
    return sql.rstrip(";") + ";"


def _has_wrapper_hints(text: str) -> bool:
    return any(p.search(text) for p in WRAPPER_HINT_PATTERNS)


def _sql_hash(text: str) -> str:
    import hashlib

    norm = " ".join((text or "").strip().lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def classify_sql(
    sql_text: str,
    *,
    strict_read_only: bool,
) -> tuple[str, str, str, str, str]:
    """Return (bucket, reason, conversion_applied, prepared_sql, confidence)."""
    raw = (sql_text or "").strip()
    if not raw:
        return "exclude", "empty_sql", "no", "", "none"

    normalized = " ".join(_normalize_quotes(raw).lower().split())

    if PREAMBLE_PAT.match(normalized):
        return "exclude", "session_or_preamble", "no", "", "high"

    has_select = bool(re.search(r"\b(select|with)\b", normalized))
    has_mutating_or_ddl = bool(MUTATING_OR_DDL_PAT.search(normalized))
    has_wrappers = _has_wrapper_hints(raw)

    if has_mutating_or_ddl and not has_select:
        return "exclude", "ddl_or_mutating_dml", "no", "", "high"
    if has_mutating_or_ddl and has_select:
        # e.g., create procedure ... begin select ...
        if strict_read_only:
            return "exclude", "procedural_or_mixed_statement", "no", "", "high"

    if READONLY_HEAD_PAT.match(normalized) and not has_mutating_or_ddl and not has_wrappers:
        prepared = _first_readonly_select_statement(raw)
        if prepared:
            return "core", "read_only_select", "no", prepared, "high"
        return "exclude", "unparsable_select", "no", "", "low"

    # convertible: wrapper/comment embedded SQL
    extracted = _extract_sqldf_embedded_sql(raw)
    if not extracted:
        extracted = _first_readonly_select_statement(raw)
    if extracted:
        if strict_read_only and MUTATING_OR_DDL_PAT.search(" ".join(extracted.lower().split())):
            return "exclude", "converted_sql_not_read_only", "yes", "", "high"
        return "convertible", "wrapper_or_comment_embedded_sql", "yes", extracted, "medium"

    if has_select:
        return "exclude", "select_present_but_not_standalone", "no", "", "low"
    return "exclude", "fragment_or_non_sql", "no", "", "high"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    source_rows = _load_rows(args.input_csv)

    filtered_rows: list[dict[str, str]] = []
    for row in source_rows:
        if args.only_keep_candidate and (row.get("v2_keep_candidate") or "").strip().lower() != "yes":
            continue
        if args.only_primary_canonical and (row.get("is_primary_canonical") or "").strip().lower() != "yes":
            continue
        filtered_rows.append(row)

    processed: list[ProcessedRow] = []
    for row in filtered_rows:
        sql_input = (row.get("sql_text_clean") or row.get("sql_text_raw") or "").strip()
        bucket, reason, conversion_applied, prepared_sql, confidence = classify_sql(
            sql_input,
            strict_read_only=bool(args.strict_read_only),
        )
        processed.append(
            ProcessedRow(
                sql_item_id=(row.get("sql_item_id") or "").strip(),
                own_id=(row.get("own_id") or "").strip(),
                dataset_id=(row.get("dataset_id") or "").strip(),
                dataset_name=(row.get("dataset_name") or "").strip(),
                source_url=(row.get("source_url") or "").strip(),
                source_type=(row.get("source_type") or "").strip(),
                source_title=(row.get("source_title") or "").strip(),
                query_intent_label=(row.get("query_intent_label") or "").strip(),
                family_tag_guess=(row.get("family_tag_guess") or "").strip(),
                v2_specificity_label=(row.get("v2_specificity_label") or "").strip(),
                v2_specificity_reason_code=(row.get("v2_specificity_reason_code") or "").strip(),
                v2_keep_candidate=(row.get("v2_keep_candidate") or "").strip(),
                is_primary_canonical=(row.get("is_primary_canonical") or "").strip(),
                canonical_group_id=(row.get("canonical_group_id") or "").strip(),
                duplicate_type=(row.get("duplicate_type") or "").strip(),
                preprocess_bucket=bucket,
                preprocess_reason=reason,
                conversion_applied=conversion_applied,
                extraction_confidence=confidence,
                sql_text_input=sql_input,
                sql_text_prepared=prepared_sql,
                sql_text_prepared_hash=_sql_hash(prepared_sql) if prepared_sql else "",
                notes="",
            )
        )

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = [item.to_dict() for item in processed]
    core_rows = [row for row in all_rows if row["preprocess_bucket"] == "core"]
    convertible_rows = [row for row in all_rows if row["preprocess_bucket"] == "convertible"]
    exclude_rows = [row for row in all_rows if row["preprocess_bucket"] == "exclude"]

    # Usable exemplar pool: core + convertible with prepared SQL.
    usable_rows = [
        row
        for row in (core_rows + convertible_rows)
        if (row.get("sql_text_prepared") or "").strip()
    ]

    _write_csv(out_root / "preprocessed_sql_all.csv", all_rows)
    _write_csv(out_root / "preprocessed_sql_core.csv", core_rows)
    _write_csv(out_root / "preprocessed_sql_convertible.csv", convertible_rows)
    _write_csv(out_root / "preprocessed_sql_exclude.csv", exclude_rows)
    _write_csv(out_root / "benchmark_sql_exemplar_pool.csv", usable_rows)

    # Dataset-level pools.
    per_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in usable_rows:
        per_dataset[row["own_id"]].append(row)
    for own_id, rows in sorted(per_dataset.items()):
        _write_csv(out_root / "datasets" / own_id / "benchmark_sql_exemplar_pool.csv", rows)

    # Summary tables.
    by_dataset_counter: dict[str, Counter[str]] = defaultdict(Counter)
    by_reason_counter: Counter[str] = Counter()
    for row in all_rows:
        by_dataset_counter[row["own_id"]][row["preprocess_bucket"]] += 1
        by_reason_counter[row["preprocess_reason"]] += 1

    summary_rows: list[dict[str, str]] = []
    for own_id in sorted(by_dataset_counter.keys()):
        c = by_dataset_counter[own_id]
        summary_rows.append(
            {
                "own_id": own_id,
                "total_rows": str(sum(c.values())),
                "core_rows": str(c.get("core", 0)),
                "convertible_rows": str(c.get("convertible", 0)),
                "exclude_rows": str(c.get("exclude", 0)),
                "usable_rows": str(c.get("core", 0) + c.get("convertible", 0)),
            }
        )
    _write_csv(out_root / "preprocessed_sql_summary_by_dataset.csv", summary_rows)

    now_iso = datetime.now(timezone.utc).isoformat()
    summary_json = {
        "generated_at_utc": now_iso,
        "input_csv": str(args.input_csv.resolve()),
        "filters": {
            "only_keep_candidate": bool(args.only_keep_candidate),
            "only_primary_canonical": bool(args.only_primary_canonical),
            "strict_read_only": bool(args.strict_read_only),
        },
        "counts": {
            "input_rows": len(source_rows),
            "rows_after_filters": len(all_rows),
            "core": len(core_rows),
            "convertible": len(convertible_rows),
            "exclude": len(exclude_rows),
            "usable_pool": len(usable_rows),
        },
        "top_reasons": by_reason_counter.most_common(20),
        "output_files": {
            "all": str((out_root / "preprocessed_sql_all.csv").resolve()),
            "core": str((out_root / "preprocessed_sql_core.csv").resolve()),
            "convertible": str((out_root / "preprocessed_sql_convertible.csv").resolve()),
            "exclude": str((out_root / "preprocessed_sql_exclude.csv").resolve()),
            "usable_pool": str((out_root / "benchmark_sql_exemplar_pool.csv").resolve()),
            "dataset_summary": str((out_root / "preprocessed_sql_summary_by_dataset.csv").resolve()),
        },
    }
    (out_root / "preprocessed_sql_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Preprocessed SQL Exemplar Summary",
        "",
        f"- generated_at_utc: {now_iso}",
        f"- input_csv: `{args.input_csv}`",
        f"- only_keep_candidate: `{args.only_keep_candidate}`",
        f"- only_primary_canonical: `{args.only_primary_canonical}`",
        f"- strict_read_only: `{args.strict_read_only}`",
        "",
        "## Counts",
        f"- input_rows: {len(source_rows)}",
        f"- rows_after_filters: {len(all_rows)}",
        f"- core: {len(core_rows)}",
        f"- convertible: {len(convertible_rows)}",
        f"- exclude: {len(exclude_rows)}",
        f"- usable_pool(core+convertible): {len(usable_rows)}",
        "",
        "## Top Exclusion / Conversion Reasons",
    ]
    for reason, cnt in by_reason_counter.most_common(15):
        lines.append(f"- `{reason}`: {cnt}")
    lines.extend(
        [
            "",
            "## Output",
            f"- `preprocessed_sql_all.csv`",
            f"- `preprocessed_sql_core.csv`",
            f"- `preprocessed_sql_convertible.csv`",
            f"- `preprocessed_sql_exclude.csv`",
            f"- `benchmark_sql_exemplar_pool.csv`",
            f"- `preprocessed_sql_summary_by_dataset.csv`",
            f"- `preprocessed_sql_summary.json`",
            "",
            "## Recommended Agent Usage",
            "Use only `benchmark_sql_exemplar_pool.csv` as the direct exemplar source for prompt injection.",
        ]
    )
    (out_root / "preprocessed_sql_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] output_root={out_root.resolve()}")
    print(
        "[done] counts:",
        json.dumps(summary_json["counts"], ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
