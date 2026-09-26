"""SQL exemplar retrieval and adaptation for benchmark query realization."""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "group",
    "by",
    "order",
    "having",
    "limit",
    "as",
    "and",
    "or",
    "not",
    "in",
    "is",
    "null",
    "case",
    "when",
    "then",
    "else",
    "end",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "distinct",
    "over",
    "partition",
    "round",
    "asc",
    "desc",
    "on",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "with",
    "union",
    "all",
    "like",
    "between",
    "coalesce",
    "cast",
    "true",
    "false",
}

FORBIDDEN_TOKENS = {
    "information_schema",
    "sqlite_master",
    "pragma",
    "pg_catalog",
}


@dataclass
class ExemplarRecord:
    sql_item_id: str
    own_id: str
    source_url: str
    sql: str
    query_intent_label: str
    family_tag_guess: str


@dataclass
class ExemplarCandidate:
    sql_item_id: str
    own_id: str
    source_url: str
    sql: str
    origin_mode: str
    match_score: float
    transform_notes: list[str]


def _normalize_sql(sql: str) -> str:
    text = (sql or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    text = text.strip()
    if not text:
        return ""
    if not text.endswith(";"):
        text += ";"
    return text


def _split_sql_statements(text: str) -> list[str]:
    statements: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    for ch in text:
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            continue
        if ch == ";" and not in_single and not in_double:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _first_select_statement(sql: str) -> str:
    normalized = _normalize_sql(sql)
    if not normalized:
        return ""
    for stmt in _split_sql_statements(normalized):
        compact = " ".join(stmt.split()).lower()
        if compact.startswith("select ") or compact.startswith("with "):
            if any(token in compact for token in FORBIDDEN_TOKENS):
                continue
            return stmt.strip().rstrip(";") + ";"
    return ""


def _extract_table_names(sql: str) -> list[str]:
    compact = " ".join(sql.split())
    names = re.findall(r"\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)\b", compact, flags=re.IGNORECASE)
    names.extend(re.findall(r"\bjoin\s+([A-Za-z_][A-Za-z0-9_]*)\b", compact, flags=re.IGNORECASE))
    out: list[str] = []
    for name in names:
        lower = name.lower()
        if lower not in out:
            out.append(lower)
    return out


def _extract_aliases(sql: str) -> set[str]:
    compact = " ".join(sql.split())
    aliases = set(re.findall(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\b", compact, flags=re.IGNORECASE))
    aliases.update(re.findall(r"\bfrom\s+[A-Za-z_][A-Za-z0-9_]*\s+([A-Za-z_][A-Za-z0-9_]*)\b", compact, flags=re.IGNORECASE))
    return {item.lower() for item in aliases}


def _extract_identifier_tokens(sql: str) -> set[str]:
    tokens = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", sql))
    return {item.lower() for item in tokens}


def _replace_word(sql: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, sql, flags=re.IGNORECASE)


def _replace_table_names(sql: str, table_name: str) -> tuple[str, list[str], bool]:
    compact = " ".join(sql.split()).lower()
    if " join " in compact:
        return sql, ["rejected_join_detected"], False

    tables = _extract_table_names(sql)
    if not tables:
        return sql, ["rejected_no_table_detected"], False
    if len(tables) > 1:
        return sql, [f"rejected_multi_table:{','.join(tables)}"], False

    old = tables[0]
    updated = _replace_word(sql, old, table_name)
    note = "table_renamed" if old != table_name.lower() else "table_matched"
    return updated, [note], True


def _fuzzy_match_column(name: str, allowed_columns: list[str]) -> str | None:
    import difflib

    lowered = [item.lower() for item in allowed_columns]
    exact_map = {item.lower(): item for item in allowed_columns}
    if name in exact_map:
        return exact_map[name]
    matched = difflib.get_close_matches(name, lowered, n=1, cutoff=0.82)
    if not matched:
        return None
    return exact_map[matched[0]]


def _adapt_columns(
    sql: str,
    *,
    allowed_columns: list[str],
    preferred_columns: list[str],
) -> tuple[str, list[str], float, bool]:
    allowed_lower = {item.lower() for item in allowed_columns}
    aliases = _extract_aliases(sql)
    table_tokens = set(_extract_table_names(sql))
    stripped = re.sub(r"'([^']|'')*'", " ", sql)
    tokens = _extract_identifier_tokens(stripped)
    unknown = sorted(
        token
        for token in tokens
        if token not in allowed_lower
        and token not in SQL_KEYWORDS
        and token not in aliases
        and token not in table_tokens
        and token not in {"focus_rate", "support", "total_count", "bucket_rate", "focus_count", "target_bucket", "missing_rate"}
    )
    if not unknown:
        return sql, ["columns_matched"], 1.0, True

    preferred_map = {item.lower(): item for item in preferred_columns}
    updates: dict[str, str] = {}
    unresolved: list[str] = []
    for token in unknown:
        if token in preferred_map:
            updates[token] = preferred_map[token]
            continue
        matched = _fuzzy_match_column(token, allowed_columns)
        if matched is None:
            unresolved.append(token)
            continue
        updates[token] = matched

    if unresolved:
        return sql, [f"unresolved_columns:{','.join(unresolved[:6])}"], 0.35, False

    updated = sql
    for old, new in updates.items():
        if old == new.lower():
            continue
        updated = _replace_word(updated, old, new)
    confidence = max(0.45, 1.0 - 0.08 * len(updates))
    notes = [f"column_mapped:{old}->{new}" for old, new in sorted(updates.items())]
    return updated, notes, confidence, True


def _score_record(
    record: ExemplarRecord,
    *,
    dataset_id: str,
    family: str,
    role: str,
    question: str,
    related_fields: list[str],
    target_column: str,
) -> float:
    sql = " ".join(record.sql.lower().split())
    score = 0.0

    if record.own_id == dataset_id:
        score += 3.0
    if target_column and target_column.lower() in sql:
        score += 1.2
    for field in related_fields[:4]:
        if field.lower() in sql:
            score += 0.7
    if "group by" in sql:
        score += 0.6
    if role in {"within_group_proportion", "collapsed_target_view", "ranked_signal_view"} and (" over (" in sql or "round(" in sql or "/" in sql):
        score += 0.6
    if role == "rare_extreme_view" and ("order by" in sql and "asc" in sql):
        score += 0.6
    if family == "missingness_structure" and " is null" in sql:
        score += 0.8
    question_tokens = {tok for tok in re.findall(r"[a-zA-Z_]{3,}", question.lower()) if tok not in SQL_KEYWORDS}
    overlap = sum(1 for tok in question_tokens if tok in sql)
    score += min(1.0, 0.15 * overlap)
    return score


def _dedupe_candidates(candidates: list[ExemplarCandidate]) -> list[ExemplarCandidate]:
    seen: set[str] = set()
    out: list[ExemplarCandidate] = []
    for item in candidates:
        key = " ".join(item.sql.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


class SQLExemplarRepository:
    def __init__(self, records: list[ExemplarRecord], source_csv: Path) -> None:
        self.records = records
        self.source_csv = source_csv
        self.by_dataset: dict[str, list[ExemplarRecord]] = {}
        for item in records:
            self.by_dataset.setdefault(item.own_id, []).append(item)

    @classmethod
    def load(cls, pool_csv: Path) -> "SQLExemplarRepository":
        pool_csv = pool_csv.expanduser().resolve()
        if not pool_csv.exists():
            raise FileNotFoundError(f"SQL exemplar pool not found: {pool_csv}")
        csv.field_size_limit(sys.maxsize)
        records: list[ExemplarRecord] = []
        with pool_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sql = _first_select_statement(str(row.get("sql_text_prepared") or ""))
                if not sql:
                    continue
                compact = " ".join(sql.lower().split())
                if any(token in compact for token in FORBIDDEN_TOKENS):
                    continue
                records.append(
                    ExemplarRecord(
                        sql_item_id=str(row.get("sql_item_id") or ""),
                        own_id=str(row.get("own_id") or ""),
                        source_url=str(row.get("source_url") or ""),
                        sql=sql,
                        query_intent_label=str(row.get("query_intent_label") or ""),
                        family_tag_guess=str(row.get("family_tag_guess") or ""),
                    )
                )
        return cls(records=records, source_csv=pool_csv)

    def summary(self) -> dict[str, object]:
        by_dataset = {key: len(value) for key, value in sorted(self.by_dataset.items())}
        return {
            "source_csv": str(self.source_csv),
            "record_count": len(self.records),
            "dataset_count": len(by_dataset),
            "records_by_dataset": by_dataset,
        }

    def get_candidates(
        self,
        *,
        dataset_id: str,
        table_name: str,
        available_columns: list[str],
        family: str,
        role: str,
        question: str,
        related_fields: list[str],
        target_column: str,
        max_candidates: int = 4,
    ) -> list[ExemplarCandidate]:
        pool: list[ExemplarRecord] = list(self.by_dataset.get(dataset_id, []))
        if len(pool) < max_candidates:
            pool.extend(item for item in self.records if item.own_id != dataset_id)

        scored = sorted(
            (
                (
                    _score_record(
                        item,
                        dataset_id=dataset_id,
                        family=family,
                        role=role,
                        question=question,
                        related_fields=related_fields,
                        target_column=target_column,
                    ),
                    item,
                )
                for item in pool
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )

        candidates: list[ExemplarCandidate] = []
        preferred_columns = list(dict.fromkeys([target_column] + list(related_fields)))
        for score, record in scored[: max(40, max_candidates * 8)]:
            base = _first_select_statement(record.sql)
            if not base:
                continue
            base, table_notes, table_ok = _replace_table_names(base, table_name=table_name)
            if not table_ok:
                continue

            adapted, col_notes, confidence, col_ok = _adapt_columns(
                base,
                allowed_columns=available_columns,
                preferred_columns=preferred_columns,
            )
            if not col_ok:
                continue
            sql = _normalize_sql(adapted)
            if not sql:
                continue

            origin_mode = "direct_reuse" if record.own_id == dataset_id and confidence >= 0.95 else "template_adapt"
            candidates.append(
                ExemplarCandidate(
                    sql_item_id=record.sql_item_id,
                    own_id=record.own_id,
                    source_url=record.source_url,
                    sql=sql,
                    origin_mode=origin_mode,
                    match_score=round(score * confidence, 6),
                    transform_notes=table_notes + col_notes,
                )
            )
            if len(candidates) >= max_candidates:
                break
        return _dedupe_candidates(candidates)


def load_sql_exemplar_repository(pool_csv: Path | None) -> SQLExemplarRepository | None:
    if pool_csv is None:
        return None
    try:
        return SQLExemplarRepository.load(pool_csv)
    except Exception:
        return None


def extract_csv_columns(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return [str(item).strip() for item in header if str(item).strip()]


def iter_repo_rows(repo: SQLExemplarRepository | None) -> Iterable[ExemplarRecord]:
    if repo is None:
        return []
    return repo.records
