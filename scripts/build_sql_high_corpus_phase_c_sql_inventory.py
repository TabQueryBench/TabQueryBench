#!/usr/bin/env python3
"""Build Phase C SQL extraction artifacts for sql_high datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GLOBAL_SOURCE_INVENTORY = Path("logs/sql_high_corpus_build_20260404/global/all_source_inventory.csv")
DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404")
DEFAULT_SCOPE_CSV = Path("logs/sql_high_corpus_build_20260404/scope/high_datasets.csv")
USER_AGENT = "Mozilla/5.0 (compatible; SQLagent-PhaseC/1.0; +https://github.com/openai)"
OK_HTTP_STATUSES = {"200", "301", "302", "303", "307", "308"}
SEARCH_SOURCE_TYPES = {"github_repo_search", "github_code_search", "kaggle_code_search", "openml_task_search"}
DIRECT_CODE_PAGE_SOURCE_TYPES = {"github_repo", "github_file", "gist", "github_release", "kaggle_code_or_notebook"}
HTML_INSPECTION_SOURCE_TYPES = {
    "official_dataset_page",
    "official_api",
    "openml_api",
    "openml_task_page",
    "kaggle_code_page",
    "kaggle_data_page",
    "kaggle_overview_page",
    "readme_or_metadata",
    "paper",
    "kaggle_code_or_notebook",
}
SQL_FILE_SUFFIXES = {".sql", ".ddl", ".dml", ".psql"}
MARKDOWN_SUFFIXES = {".md", ".markdown", ".rmd", ".rst", ".txt"}
NOTEBOOK_SUFFIXES = {".ipynb"}
CODE_SUFFIXES = {
    ".py",
    ".r",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".php",
    ".scala",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
}
SKIP_DIR_NAMES = {
    ".git",
    ".github",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "target",
}
MAX_TEXT_FILE_BYTES = 2_000_000
MAX_HTML_BYTES = 2_000_000
TOKEN_STOPWORDS = {
    "and",
    "challenge",
    "classification",
    "competition",
    "data",
    "dataset",
    "datasets",
    "default",
    "detection",
    "for",
    "from",
    "high",
    "ii",
    "in",
    "kaggle",
    "learning",
    "machine",
    "ml",
    "of",
    "prediction",
    "risk",
    "sql",
    "the",
    "with",
}
SQL_INVENTORY_FIELDNAMES = [
    "own_id",
    "dataset_id",
    "dataset_name",
    "sql_item_id",
    "source_url",
    "source_type",
    "source_title",
    "sql_text_raw",
    "sql_text_clean",
    "sql_dialect_guess",
    "sql_complexity",
    "query_intent_label",
    "family_tag_guess",
    "dataset_specificity_label",
    "evidence_confidence",
    "executable_status",
    "extraction_notes",
    "retrieved_at_utc",
    "source_seed_url",
    "source_seed_type",
    "source_seed_title",
    "source_seed_http_status",
    "source_seed_specificity_hint",
    "source_seed_relevance_label",
    "source_file_path",
    "extraction_method",
    "sql_text_norm_hash",
    "is_near_duplicate",
    "duplicate_of_sql_item_id",
]
REJECTED_FIELDNAMES = [
    "own_id",
    "dataset_id",
    "dataset_name",
    "source_type",
    "source_url",
    "source_title",
    "retrieval_method",
    "http_status",
    "relevance_label",
    "dataset_specificity_hint",
    "has_sql_text",
    "notes",
    "inspection_outcome",
    "rejection_reason",
    "rejection_details",
    "retrieved_at_utc",
]


@dataclass(frozen=True)
class DatasetContext:
    own_id: str
    dataset_id: str
    dataset_name: str
    dataset_dir: Path
    source_inventory_path: Path


@dataclass(frozen=True)
class SourceRecord:
    own_id: str
    dataset_id: str
    dataset_name: str
    source_type: str
    source_url: str
    source_title: str
    retrieval_method: str
    http_status: str
    relevance_label: str
    dataset_specificity_hint: str
    has_sql_text: str
    notes: str


@dataclass
class FetchResult:
    url: str
    final_url: str
    http_status: str
    content_type: str
    title: str
    text: str
    error: str


@dataclass
class CloneResult:
    ok: bool
    path: Path | None
    branch: str
    error: str


@dataclass
class ExtractedBlock:
    text: str
    method: str
    source_file_path: str
    source_url: str
    source_type: str
    source_title: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract explicit SQL evidence from Phase B collected sources and write "
            "per-dataset/global Phase C inventories."
        )
    )
    parser.add_argument("--global-source-inventory", type=Path, default=DEFAULT_GLOBAL_SOURCE_INVENTORY)
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=25)
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
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_datasets(scope_csv: Path, output_root: Path) -> list[DatasetContext]:
    rows = read_csv_rows(scope_csv)
    datasets: list[DatasetContext] = []
    for row in rows:
        own_id = (row.get("own_id") or "").strip()
        dataset_dir = output_root / "datasets" / own_id
        source_inventory_path = dataset_dir / "sources" / "source_inventory.csv"
        datasets.append(
            DatasetContext(
                own_id=own_id,
                dataset_id=(row.get("dataset_id") or "").strip(),
                dataset_name=(row.get("dataset_name") or "").strip(),
                dataset_dir=dataset_dir,
                source_inventory_path=source_inventory_path,
            )
        )
    return datasets


def load_source_records(path: Path) -> list[SourceRecord]:
    rows = read_csv_rows(path)
    return [
        SourceRecord(
            own_id=(row.get("own_id") or "").strip(),
            dataset_id=(row.get("dataset_id") or "").strip(),
            dataset_name=(row.get("dataset_name") or "").strip(),
            source_type=(row.get("source_type") or "").strip(),
            source_url=(row.get("source_url") or "").strip(),
            source_title=(row.get("source_title") or "").strip(),
            retrieval_method=(row.get("retrieval_method") or "").strip(),
            http_status=(row.get("http_status") or "").strip(),
            relevance_label=(row.get("relevance_label") or "").strip(),
            dataset_specificity_hint=(row.get("dataset_specificity_hint") or "").strip(),
            has_sql_text=(row.get("has_sql_text") or "").strip(),
            notes=(row.get("notes") or "").strip(),
        )
        for row in rows
    ]


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in TOKEN_STOPWORDS
    ]


def dataset_tokens(dataset: DatasetContext) -> list[str]:
    tokens: list[str] = []
    for token in tokenize(dataset.dataset_name) + tokenize(dataset.dataset_id.split(":", 1)[-1]):
        if token not in tokens:
            tokens.append(token)
    return tokens


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def looks_like_binary(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:4096]
    return b"\x00" in sample


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    if looks_like_binary(data):
        raise ValueError(f"Binary file skipped: {path}")
    if len(data) > MAX_TEXT_FILE_BYTES:
        raise ValueError(f"File too large to scan as text: {path}")
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def fetch_text(url: str, timeout_seconds: int) -> FetchResult:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(MAX_HTML_BYTES + 1)
            if len(body) > MAX_HTML_BYTES:
                body = body[:MAX_HTML_BYTES]
            content_type = response.headers.get("Content-Type", "")
            final_url = response.geturl()
            status = str(getattr(response, "status", 200))
            charset = response.headers.get_content_charset() or "utf-8"
            text = body.decode(charset, errors="replace")
            title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
            title = html.unescape(normalize_whitespace(title_match.group(1))) if title_match else ""
            return FetchResult(
                url=url,
                final_url=final_url,
                http_status=status,
                content_type=content_type,
                title=title,
                text=text,
                error="",
            )
    except urllib.error.HTTPError as error:
        try:
            body = error.read(MAX_HTML_BYTES + 1)
        except Exception:
            body = b""
        text = body.decode("utf-8", errors="replace")
        title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        title = html.unescape(normalize_whitespace(title_match.group(1))) if title_match else ""
        return FetchResult(
            url=url,
            final_url=url,
            http_status=str(error.code),
            content_type=error.headers.get("Content-Type", "") if error.headers else "",
            title=title,
            text=text,
            error=str(error),
        )
    except Exception as error:
        return FetchResult(
            url=url,
            final_url=url,
            http_status="",
            content_type="",
            title="",
            text="",
            error=str(error),
        )


def strip_html_tags(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(without_tags)


def extract_html_code_blocks(text: str) -> list[str]:
    without_scripts = re.sub(r"<script.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    without_styles = re.sub(r"<style.*?</style>", " ", without_scripts, flags=re.IGNORECASE | re.DOTALL)
    blocks: list[str] = []
    for pattern in (r"<pre[^>]*>(.*?)</pre>", r"<code[^>]*>(.*?)</code>"):
        for match in re.finditer(pattern, without_styles, flags=re.IGNORECASE | re.DOTALL):
            block = strip_html_tags(match.group(1))
            block = block.strip()
            if block:
                blocks.append(block)
    return blocks


def strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```[^\n]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def remove_sql_comments(text: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    without_line = re.sub(r"(?m)--[^\n]*$", " ", without_block)
    return without_line


def looks_like_sql_text(text: str) -> bool:
    candidate = strip_markdown_code_fences(text)
    candidate = candidate.strip()
    if not candidate:
        return False
    lowered = candidate.lower()
    start_pattern = re.compile(
        r"^\s*(with|select|insert\s+into|update|delete\s+from|create\s+(or\s+replace\s+)?"
        r"(table|view|database|schema|procedure|function|index|trigger)|drop\s+"
        r"(table|database|schema|view)|alter\s+table|truncate\s+table|use\s+\w+|"
        r"show\s+\w+|describe\s+\w+|explain\s+\w+|call\s+\w+|merge\s+into|copy\s+\w+)",
        re.IGNORECASE | re.DOTALL,
    )
    if start_pattern.search(candidate):
        return True
    keyword_hits = sum(
        1
        for pattern in (
            r"\bselect\b",
            r"\bfrom\b",
            r"\bwhere\b",
            r"\bgroup\s+by\b",
            r"\border\s+by\b",
            r"\bjoin\b",
            r"\bcreate\s+table\b",
            r"\binsert\s+into\b",
            r"\bupdate\b",
            r"\bdelete\s+from\b",
            r"\bwith\b",
        )
        if re.search(pattern, lowered, re.IGNORECASE)
    )
    return keyword_hits >= 2 and (";" in candidate or "\n" in candidate)


def split_sql_statements(text: str) -> list[str]:
    content = strip_markdown_code_fences(text)
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        return []

    statements: list[str] = []
    buffer: list[str] = []
    in_single = False
    in_double = False
    in_backtick = False
    in_line_comment = False
    in_block_comment = False
    index = 0
    while index < len(content):
        char = content[index]
        next_char = content[index + 1] if index + 1 < len(content) else ""

        if in_line_comment:
            buffer.append(char)
            if char == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            buffer.append(char)
            if char == "*" and next_char == "/":
                buffer.append(next_char)
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue

        if not in_single and not in_double and not in_backtick:
            if char == "-" and next_char == "-":
                buffer.append(char)
                buffer.append(next_char)
                in_line_comment = True
                index += 2
                continue
            if char == "/" and next_char == "*":
                buffer.append(char)
                buffer.append(next_char)
                in_block_comment = True
                index += 2
                continue

        if char == "'" and not in_double and not in_backtick:
            if in_single and next_char == "'":
                buffer.append(char)
                buffer.append(next_char)
                index += 2
                continue
            in_single = not in_single
            buffer.append(char)
            index += 1
            continue

        if char == '"' and not in_single and not in_backtick:
            if in_double and next_char == '"':
                buffer.append(char)
                buffer.append(next_char)
                index += 2
                continue
            in_double = not in_double
            buffer.append(char)
            index += 1
            continue

        if char == "`" and not in_single and not in_double:
            in_backtick = not in_backtick
            buffer.append(char)
            index += 1
            continue

        if char == ";" and not in_single and not in_double and not in_backtick:
            buffer.append(char)
            statement = "".join(buffer).strip()
            if statement and looks_like_sql_text(statement):
                statements.append(statement)
            buffer = []
            index += 1
            continue

        buffer.append(char)
        index += 1

    trailing = "".join(buffer).strip()
    if trailing and looks_like_sql_text(trailing):
        statements.append(trailing)

    if not statements and looks_like_sql_text(content):
        statements.append(content)
    return statements


def clean_sql_text(text: str) -> str:
    cleaned = strip_markdown_code_fences(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    if cleaned.startswith("%%sql"):
        cleaned = cleaned.split("\n", 1)[1].strip() if "\n" in cleaned else ""
    if cleaned.startswith("%sql"):
        cleaned = cleaned.split("\n", 1)[1].strip() if "\n" in cleaned else ""
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalized_sql_hash(text: str) -> str:
    normalized = clean_sql_text(text)
    normalized = remove_sql_comments(normalized)
    normalized = normalized.replace("`", "")
    normalized = re.sub(r"\s+", " ", normalized).strip().rstrip(";").lower()
    normalized = re.sub(r"\s*([(),=<>+\-/*])\s*", r"\1", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def sql_dialect_guess(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("auto_increment", "engine=", "delimiter ", "str_to_date", "use ")):
        return "mysql"
    if any(token in lowered for token in (" serial ", " ilike ", "::", "language plpgsql", "returning ", "array_agg")):
        return "postgresql"
    if any(token in lowered for token in ("integer primary key autoincrement", "pragma ", "sqlite_")):
        return "sqlite"
    if any(token in lowered for token in (" nvarchar", " top ", "go\n", "[dbo]", "identity(")):
        return "tsql"
    if any(token in lowered for token in ("varchar2", "number(", "begin\n", "end;", "pl/sql")):
        return "oracle"
    return "generic_sql"


def sql_complexity(text: str) -> str:
    lowered = remove_sql_comments(text).lower()
    advanced_signals = (
        " over(",
        " partition by ",
        " recursive ",
        " union all ",
        " create procedure ",
        " create function ",
        " case when ",
    )
    moderate_signals = (
        " join ",
        " group by ",
        " order by ",
        " having ",
        " foreign key",
        " subquery",
        " distinct ",
        " limit ",
    )
    if any(signal in lowered for signal in advanced_signals):
        return "advanced"
    join_count = lowered.count(" join ")
    if join_count >= 2:
        return "advanced"
    if any(signal in lowered for signal in moderate_signals):
        return "moderate"
    return "simple"


def query_intent_label(text: str) -> str:
    lowered = clean_sql_text(text).lower()
    if lowered.startswith("create table"):
        return "ddl_create_table"
    if lowered.startswith("create database"):
        return "ddl_create_database"
    if lowered.startswith("create view"):
        return "ddl_create_view"
    if lowered.startswith("create procedure") or lowered.startswith("create function"):
        return "ddl_program_definition"
    if lowered.startswith("drop database"):
        return "ddl_drop_database"
    if lowered.startswith("drop table"):
        return "ddl_drop_table"
    if lowered.startswith("alter table"):
        return "ddl_alter_table"
    if lowered.startswith("insert into"):
        return "dml_insert"
    if lowered.startswith("update"):
        return "dml_update"
    if lowered.startswith("delete from"):
        return "dml_delete"
    if lowered.startswith("use "):
        return "session_use"
    if lowered.startswith("with ") or lowered.startswith("select"):
        return "dml_select"
    if lowered.startswith("call "):
        return "procedure_call"
    return "other_sql"


def family_tag_guess(text: str) -> str:
    lowered = remove_sql_comments(clean_sql_text(text)).lower()
    if "create table" in lowered or "alter table" in lowered or "drop table" in lowered:
        return "schema_ddl"
    if "insert into" in lowered or "copy " in lowered or "load data" in lowered:
        return "etl_load"
    if " join " in lowered:
        return "join_analysis"
    if " group by " in lowered:
        return "aggregation"
    if " over(" in lowered or " partition by " in lowered:
        return "window_analytics"
    if " where " in lowered:
        return "filtering"
    if lowered.startswith("select"):
        return "basic_projection"
    return "misc_sql"


def build_gist_anchor(filename: str) -> str:
    return safe_slug(filename.replace(".", "-"))


def infer_dataset_specificity(
    dataset: DatasetContext,
    source: SourceRecord,
    context_text: str,
) -> str:
    hint = source.dataset_specificity_hint or "unknown"
    if hint in {"strict", "weak", "collision_risk"}:
        base_hint = hint
    else:
        base_hint = "unknown"

    tokens = dataset_tokens(dataset)
    haystack = " ".join(
        [
            source.source_url,
            source.source_title,
            context_text,
        ]
    ).lower()
    overlap = sum(1 for token in tokens if token in haystack)

    normalized_dataset_name = normalize_whitespace(dataset.dataset_name).lower()
    exact_name_match = normalized_dataset_name and normalized_dataset_name in haystack

    if base_hint == "strict":
        return "strict"
    if base_hint == "weak":
        return "strict" if exact_name_match or overlap >= 3 else "weak"
    if base_hint == "collision_risk":
        return "weak" if exact_name_match and overlap >= 3 else "collision_risk"
    if exact_name_match or overlap >= 3:
        return "strict"
    if overlap >= 1:
        return "weak"
    return "collision_risk"


def evidence_confidence(source_type: str, specificity: str, extraction_method: str) -> str:
    score = 0
    if source_type in {"github_file", "gist"}:
        score += 2
    elif source_type in {"github_repo", "kaggle_code_or_notebook", "kaggle_code_page"}:
        score += 1

    if specificity == "strict":
        score += 2
    elif specificity == "weak":
        score += 1

    if extraction_method in {"markdown_fence", "html_code_block", "source_string_literal"}:
        score -= 1

    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def build_extraction_notes(
    *,
    source: SourceRecord,
    extraction_method: str,
    source_file_path: str,
    specificity: str,
) -> str:
    note_parts = [f"extraction_method={extraction_method}"]
    if source_file_path:
        note_parts.append(f"source_file_path={source_file_path}")
    if source.dataset_specificity_hint and source.dataset_specificity_hint != specificity:
        note_parts.append(
            f"specificity_inferred_from_phase_b_hint={source.dataset_specificity_hint}->{specificity}"
        )
    elif source.dataset_specificity_hint:
        note_parts.append(f"specificity_matches_phase_b_hint={specificity}")
    if source.notes:
        note_parts.append(f"phase_b_notes={normalize_whitespace(source.notes)}")
    return "; ".join(note_parts)


def github_url_components(url: str) -> dict[str, str]:
    parsed = urllib.parse.urlsplit(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.lower() != "github.com" or len(parts) < 2:
        return {}
    components = {
        "owner": parts[0],
        "repo": parts[1],
        "kind": "repo",
        "branch": "",
        "subpath": "",
        "base_repo_url": f"https://github.com/{parts[0]}/{parts[1]}",
    }
    if len(parts) >= 3:
        if parts[2] == "tree":
            components["kind"] = "tree"
            components["branch"] = parts[3] if len(parts) >= 4 else ""
            components["subpath"] = "/".join(parts[4:]) if len(parts) >= 5 else ""
        elif parts[2] == "blob":
            components["kind"] = "blob"
            components["branch"] = parts[3] if len(parts) >= 4 else ""
            components["subpath"] = "/".join(parts[4:]) if len(parts) >= 5 else ""
        elif parts[2] == "releases":
            components["kind"] = "releases"
    return components


def github_raw_url(url: str) -> str:
    info = github_url_components(url)
    if info.get("kind") != "blob":
        return url
    return (
        f"https://raw.githubusercontent.com/{info['owner']}/{info['repo']}/"
        f"{info['branch']}/{info['subpath']}"
    )


def github_file_url(base_repo_url: str, branch: str, relative_path: str) -> str:
    return f"{base_repo_url}/blob/{branch}/{relative_path.replace(os.sep, '/')}"


def ensure_repo_clone(
    *,
    cache: dict[tuple[str, str], CloneResult],
    temp_root: Path,
    clone_url: str,
    branch: str,
    timeout_seconds: int,
) -> CloneResult:
    key = (clone_url, branch)
    if key in cache:
        return cache[key]

    target_dir = temp_root / hashlib.sha256(f"{clone_url}|{branch}".encode("utf-8")).hexdigest()[:16]
    command = ["git", "clone", "--depth", "1"]
    if branch:
        command.extend(["--branch", branch])
    command.extend([clone_url, str(target_dir)])
    clone_timeout = max(timeout_seconds * 4, 120)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=clone_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        clone_result = CloneResult(
            ok=False,
            path=None,
            branch="",
            error=f"git clone timed out after {clone_timeout} seconds",
        )
        cache[key] = clone_result
        return clone_result
    if result.returncode != 0 and branch:
        fallback_command = ["git", "clone", "--depth", "1", clone_url, str(target_dir)]
        try:
            result = subprocess.run(
                fallback_command,
                capture_output=True,
                text=True,
                timeout=clone_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            clone_result = CloneResult(
                ok=False,
                path=None,
                branch="",
                error=f"git clone timed out after {clone_timeout} seconds",
            )
            cache[key] = clone_result
            return clone_result
    if result.returncode != 0:
        clone_result = CloneResult(
            ok=False,
            path=None,
            branch="",
            error=normalize_whitespace(result.stderr or result.stdout or "git clone failed"),
        )
        cache[key] = clone_result
        return clone_result

    branch_result = subprocess.run(
        ["git", "-C", str(target_dir), "branch", "--show-current"],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    branch_name = normalize_whitespace(branch_result.stdout) or branch or "HEAD"
    clone_result = CloneResult(ok=True, path=target_dir, branch=branch_name, error="")
    cache[key] = clone_result
    return clone_result


def iter_repo_files(root: Path) -> Iterable[Path]:
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        for filename in filenames:
            path = Path(current_root) / filename
            if path.name.startswith(".") and path.parent.name != "":
                continue
            yield path


def markdown_code_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(text):
        language = normalize_whitespace(match.group(1))
        block = match.group(2).strip()
        if block:
            blocks.append((language, block))
    return blocks


def source_code_string_literals(text: str) -> list[str]:
    matches: list[str] = []
    triple_pattern = re.compile(r'(?s)(?:[rubfRUBF]{0,3})("""|\'\'\')(.*?)\1')
    single_pattern = re.compile(r'(?s)(?:[rubfRUBF]{0,3})(["\'])(.{20,}?)\1')
    for pattern in (triple_pattern, single_pattern):
        for match in pattern.finditer(text):
            value = match.group(2)
            if looks_like_sql_text(value):
                matches.append(value)
    return matches


def extract_blocks_from_markdown(
    *,
    text: str,
    source_url: str,
    source_type: str,
    source_title: str,
    source_file_path: str,
) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    for language, block in markdown_code_blocks(text):
        language_lower = language.lower()
        if "sql" in language_lower or looks_like_sql_text(block):
            blocks.append(
                ExtractedBlock(
                    text=block,
                    method="markdown_fence",
                    source_file_path=source_file_path,
                    source_url=source_url,
                    source_type=source_type,
                    source_title=source_title,
                )
            )
    return blocks


def extract_blocks_from_notebook(
    *,
    text: str,
    source_url: str,
    source_type: str,
    source_title: str,
    source_file_path: str,
) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return blocks

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source") or [])
        cell_path = f"{source_file_path}#cell{index}"
        if cell_type == "markdown":
            blocks.extend(
                extract_blocks_from_markdown(
                    text=source,
                    source_url=source_url,
                    source_type=source_type,
                    source_title=source_title,
                    source_file_path=cell_path,
                )
            )
            continue
        if cell_type != "code":
            continue
        stripped = source.lstrip()
        if stripped.startswith("%%sql"):
            sql_body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
            if looks_like_sql_text(sql_body):
                blocks.append(
                    ExtractedBlock(
                        text=sql_body,
                        method="ipynb_sql_magic",
                        source_file_path=cell_path,
                        source_url=source_url,
                        source_type=source_type,
                        source_title=source_title,
                    )
                )
        elif stripped.startswith("%sql"):
            sql_body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
            if looks_like_sql_text(sql_body):
                blocks.append(
                    ExtractedBlock(
                        text=sql_body,
                        method="ipynb_sql_magic",
                        source_file_path=cell_path,
                        source_url=source_url,
                        source_type=source_type,
                        source_title=source_title,
                    )
                )
        for literal in source_code_string_literals(source):
            blocks.append(
                ExtractedBlock(
                    text=literal,
                    method="ipynb_string_literal",
                    source_file_path=cell_path,
                    source_url=source_url,
                    source_type=source_type,
                    source_title=source_title,
                )
            )
    return blocks


def extract_blocks_from_source_text(
    *,
    text: str,
    source_url: str,
    source_type: str,
    source_title: str,
    source_file_path: str,
) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    for literal in source_code_string_literals(text):
        blocks.append(
            ExtractedBlock(
                text=literal,
                method="source_string_literal",
                source_file_path=source_file_path,
                source_url=source_url,
                source_type=source_type,
                source_title=source_title,
            )
        )
    return blocks


def extract_blocks_from_html_source(
    *,
    html_text: str,
    source_url: str,
    source_type: str,
    source_title: str,
) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    for index, block in enumerate(extract_html_code_blocks(html_text), start=1):
        if looks_like_sql_text(block):
            blocks.append(
                ExtractedBlock(
                    text=block,
                    method="html_code_block",
                    source_file_path=f"html_block_{index}",
                    source_url=source_url,
                    source_type=source_type,
                    source_title=source_title,
                )
            )
    return blocks


def extract_blocks_from_text_file(
    *,
    file_path: Path,
    source_url: str,
    source_type: str,
    source_title: str,
    logical_file_path: str,
) -> list[ExtractedBlock]:
    suffix = file_path.suffix.lower()
    text = read_text_file(file_path)
    if suffix in SQL_FILE_SUFFIXES or file_path.name.lower().endswith(".sql"):
        return [
            ExtractedBlock(
                text=text,
                method="sql_file",
                source_file_path=logical_file_path,
                source_url=source_url,
                source_type=source_type,
                source_title=source_title,
            )
        ]
    if suffix in MARKDOWN_SUFFIXES:
        return extract_blocks_from_markdown(
            text=text,
            source_url=source_url,
            source_type=source_type,
            source_title=source_title,
            source_file_path=logical_file_path,
        )
    if suffix in NOTEBOOK_SUFFIXES:
        return extract_blocks_from_notebook(
            text=text,
            source_url=source_url,
            source_type=source_type,
            source_title=source_title,
            source_file_path=logical_file_path,
        )
    if suffix in CODE_SUFFIXES or "sql" in file_path.name.lower():
        return extract_blocks_from_source_text(
            text=text,
            source_url=source_url,
            source_type=source_type,
            source_title=source_title,
            source_file_path=logical_file_path,
        )
    return []


def rejected_row(source: SourceRecord, outcome: str, reason: str, details: str) -> dict[str, str]:
    return {
        "own_id": source.own_id,
        "dataset_id": source.dataset_id,
        "dataset_name": source.dataset_name,
        "source_type": source.source_type,
        "source_url": source.source_url,
        "source_title": source.source_title,
        "retrieval_method": source.retrieval_method,
        "http_status": source.http_status,
        "relevance_label": source.relevance_label,
        "dataset_specificity_hint": source.dataset_specificity_hint,
        "has_sql_text": source.has_sql_text,
        "notes": source.notes,
        "inspection_outcome": outcome,
        "rejection_reason": reason,
        "rejection_details": details,
        "retrieved_at_utc": utc_now_iso(),
    }


def blocks_to_candidate_rows(
    *,
    dataset: DatasetContext,
    source: SourceRecord,
    blocks: list[ExtractedBlock],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        statements = split_sql_statements(block.text)
        for statement in statements:
            raw_text = clean_sql_text(statement)
            if not raw_text:
                continue
            clean_text = normalize_whitespace(remove_sql_comments(raw_text)) or normalize_whitespace(raw_text)
            if not clean_text:
                clean_text = raw_text
            specificity = infer_dataset_specificity(
                dataset=dataset,
                source=source,
                context_text=f"{block.source_file_path} {block.source_title} {raw_text[:500]}",
            )
            rows.append(
                {
                    "own_id": dataset.own_id,
                    "dataset_id": dataset.dataset_id,
                    "dataset_name": dataset.dataset_name,
                    "sql_item_id": "",
                    "source_url": block.source_url,
                    "source_type": block.source_type,
                    "source_title": block.source_title,
                    "sql_text_raw": raw_text,
                    "sql_text_clean": clean_text,
                    "sql_dialect_guess": sql_dialect_guess(raw_text),
                    "sql_complexity": sql_complexity(raw_text),
                    "query_intent_label": query_intent_label(raw_text),
                    "family_tag_guess": family_tag_guess(raw_text),
                    "dataset_specificity_label": specificity,
                    "evidence_confidence": evidence_confidence(block.source_type, specificity, block.method),
                    "executable_status": "unknown",
                    "extraction_notes": build_extraction_notes(
                        source=source,
                        extraction_method=block.method,
                        source_file_path=block.source_file_path,
                        specificity=specificity,
                    ),
                    "retrieved_at_utc": utc_now_iso(),
                    "source_seed_url": source.source_url,
                    "source_seed_type": source.source_type,
                    "source_seed_title": source.source_title,
                    "source_seed_http_status": source.http_status,
                    "source_seed_specificity_hint": source.dataset_specificity_hint,
                    "source_seed_relevance_label": source.relevance_label,
                    "source_file_path": block.source_file_path,
                    "extraction_method": block.method,
                    "sql_text_norm_hash": normalized_sql_hash(raw_text),
                    "is_near_duplicate": "no",
                    "duplicate_of_sql_item_id": "",
                }
            )
    return rows


def extract_from_github_file(
    *,
    dataset: DatasetContext,
    source: SourceRecord,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    raw_url = github_raw_url(source.source_url)
    fetch = fetch_text(raw_url, timeout_seconds)
    if fetch.http_status not in OK_HTTP_STATUSES:
        return [], rejected_row(
            source,
            "rejected",
            "github_file_fetch_failed",
            fetch.error or f"http_status={fetch.http_status or 'unknown'}",
        )

    info = github_url_components(source.source_url)
    logical_path = info.get("subpath", Path(urllib.parse.urlsplit(source.source_url).path).name)
    logical_title = f"{logical_path} · {info.get('owner', '')}/{info.get('repo', '')}".strip(" ·")
    text_path = Path(logical_path)
    suffix = text_path.suffix.lower()
    blocks: list[ExtractedBlock] = []
    if suffix in SQL_FILE_SUFFIXES or text_path.name.lower().endswith(".sql"):
        blocks.append(
            ExtractedBlock(
                text=fetch.text,
                method="github_raw_file",
                source_file_path=logical_path,
                source_url=source.source_url,
                source_type="github_file",
                source_title=logical_title or source.source_title,
            )
        )
    elif suffix in MARKDOWN_SUFFIXES:
        temp_path = Path(logical_path)
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_path_on_disk = Path(temp_dir_name) / temp_path.name
            temp_path_on_disk.write_text(fetch.text, encoding="utf-8")
            blocks.extend(
                extract_blocks_from_text_file(
                    file_path=temp_path_on_disk,
                    source_url=source.source_url,
                    source_type="github_file",
                    source_title=logical_title or source.source_title,
                    logical_file_path=logical_path,
                )
            )
    else:
        if looks_like_sql_text(fetch.text):
            blocks.append(
                ExtractedBlock(
                    text=fetch.text,
                    method="github_raw_file",
                    source_file_path=logical_path,
                    source_url=source.source_url,
                    source_type="github_file",
                    source_title=logical_title or source.source_title,
                )
            )

    rows = blocks_to_candidate_rows(dataset=dataset, source=source, blocks=blocks)
    if rows:
        return rows, None
    return [], rejected_row(
        source,
        "rejected",
        "github_file_contains_no_explicit_sql",
        "Fetched raw file successfully but no explicit SQL statements were detected.",
    )


def extract_from_gist(
    *,
    dataset: DatasetContext,
    source: SourceRecord,
    clone_cache: dict[tuple[str, str], CloneResult],
    temp_root: Path,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    clone_url = source.source_url.rstrip("/") + ".git"
    clone = ensure_repo_clone(
        cache=clone_cache,
        temp_root=temp_root,
        clone_url=clone_url,
        branch="",
        timeout_seconds=timeout_seconds,
    )
    if not clone.ok or not clone.path:
        return [], rejected_row(
            source,
            "rejected",
            "gist_clone_failed",
            clone.error or "Unable to clone gist repository.",
        )

    blocks: list[ExtractedBlock] = []
    for file_path in iter_repo_files(clone.path):
        try:
            logical_path = file_path.relative_to(clone.path).as_posix()
            file_title = f"{logical_path} · gist"
            blocks.extend(
                extract_blocks_from_text_file(
                    file_path=file_path,
                    source_url=f"{source.source_url}#file-{build_gist_anchor(file_path.name)}",
                    source_type="gist",
                    source_title=file_title,
                    logical_file_path=logical_path,
                )
            )
        except Exception:
            continue

    rows = blocks_to_candidate_rows(dataset=dataset, source=source, blocks=blocks)
    if rows:
        return rows, None
    return [], rejected_row(
        source,
        "rejected",
        "gist_scanned_no_explicit_sql",
        "Cloned gist successfully but found no explicit SQL statements.",
    )


def extract_from_github_repo_like(
    *,
    dataset: DatasetContext,
    source: SourceRecord,
    clone_cache: dict[tuple[str, str], CloneResult],
    temp_root: Path,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    info = github_url_components(source.source_url)
    if not info:
        return [], rejected_row(
            source,
            "rejected",
            "github_source_parse_failed",
            "Could not parse owner/repo components from source URL.",
        )

    clone_branch = info.get("branch", "")
    clone = ensure_repo_clone(
        cache=clone_cache,
        temp_root=temp_root,
        clone_url=info["base_repo_url"] + ".git",
        branch=clone_branch,
        timeout_seconds=timeout_seconds,
    )
    if not clone.ok or not clone.path:
        page_rows, page_rejected = extract_from_html_page(
            dataset=dataset,
            source=source,
            timeout_seconds=timeout_seconds,
        )
        if page_rows:
            return page_rows, None
        details = clone.error or "Unable to clone repository."
        if page_rejected is not None:
            details = f"{details}; html_fallback={page_rejected['rejection_reason']}"
        return [], rejected_row(
            source,
            "rejected",
            "github_repo_clone_failed",
            details,
        )

    scan_root = clone.path
    if info.get("kind") == "tree" and info.get("subpath"):
        scan_root = clone.path / info["subpath"]
        if not scan_root.exists():
            return [], rejected_row(
                source,
                "rejected",
                "github_tree_subpath_missing",
                f"Subpath does not exist after clone: {info['subpath']}",
            )

    blocks: list[ExtractedBlock] = []
    for file_path in iter_repo_files(scan_root):
        try:
            logical_path = file_path.relative_to(clone.path).as_posix()
            source_url = github_file_url(info["base_repo_url"], clone.branch or clone_branch or "HEAD", logical_path)
            source_title = f"{logical_path} · {info['owner']}/{info['repo']}"
            blocks.extend(
                extract_blocks_from_text_file(
                    file_path=file_path,
                    source_url=source_url,
                    source_type="github_file",
                    source_title=source_title,
                    logical_file_path=logical_path,
                )
            )
        except Exception:
            continue

    rows = blocks_to_candidate_rows(dataset=dataset, source=source, blocks=blocks)
    if rows:
        return rows, None
    return [], rejected_row(
        source,
        "rejected",
        "github_repo_scanned_no_machine_readable_sql",
        "Repository cloned successfully but no explicit machine-readable SQL was detected in scanned files.",
    )


def extract_from_html_page(
    *,
    dataset: DatasetContext,
    source: SourceRecord,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    fetch = fetch_text(source.source_url, timeout_seconds)
    if fetch.http_status not in OK_HTTP_STATUSES:
        return [], rejected_row(
            source,
            "rejected",
            "page_fetch_failed",
            fetch.error or f"http_status={fetch.http_status or 'unknown'}",
        )
    blocks = extract_blocks_from_html_source(
        html_text=fetch.text,
        source_url=source.source_url,
        source_type=source.source_type,
        source_title=fetch.title or source.source_title,
    )
    rows = blocks_to_candidate_rows(dataset=dataset, source=source, blocks=blocks)
    if rows:
        return rows, None
    return [], rejected_row(
        source,
        "rejected",
        "no_explicit_sql_in_page_source",
        "Fetched the page successfully but found no explicit SQL in HTML code/pre blocks.",
    )


def process_source(
    *,
    dataset: DatasetContext,
    source: SourceRecord,
    clone_cache: dict[tuple[str, str], CloneResult],
    temp_root: Path,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    if source.http_status not in OK_HTTP_STATUSES:
        return [], rejected_row(
            source,
            "rejected",
            f"source_not_reachable_in_phase_b_status_{source.http_status or 'unknown'}",
            "Phase B recorded this source as unreachable or broken, so no Phase C extraction was attempted.",
        )

    if source.source_type in SEARCH_SOURCE_TYPES:
        return [], rejected_row(
            source,
            "rejected",
            "discovery_search_page_only",
            "Search/discovery pages are kept for provenance but are not treated as direct SQL evidence sources.",
        )

    if source.source_type == "github_file":
        return extract_from_github_file(dataset=dataset, source=source, timeout_seconds=timeout_seconds)

    if source.source_type == "gist":
        return extract_from_gist(
            dataset=dataset,
            source=source,
            clone_cache=clone_cache,
            temp_root=temp_root,
            timeout_seconds=timeout_seconds,
        )

    if source.source_type in {"github_repo", "github_release"}:
        return extract_from_github_repo_like(
            dataset=dataset,
            source=source,
            clone_cache=clone_cache,
            temp_root=temp_root,
            timeout_seconds=timeout_seconds,
        )

    if source.source_type in HTML_INSPECTION_SOURCE_TYPES | DIRECT_CODE_PAGE_SOURCE_TYPES:
        return extract_from_html_page(dataset=dataset, source=source, timeout_seconds=timeout_seconds)

    return [], rejected_row(
        source,
        "rejected",
        "unsupported_source_type",
        "This source type is not supported by the Phase C extractor.",
    )


def assign_sql_item_ids(dataset_rows: list[dict[str, Any]], own_id: str) -> None:
    for index, row in enumerate(dataset_rows, start=1):
        row["sql_item_id"] = f"{own_id}_sql_{index:04d}"


def mark_duplicates(all_rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[row["sql_text_norm_hash"]].append(row)
    for rows in groups.values():
        if len(rows) <= 1:
            continue
        canonical_id = rows[0]["sql_item_id"]
        for index, row in enumerate(rows):
            if index == 0:
                continue
            row["is_near_duplicate"] = "yes"
            row["duplicate_of_sql_item_id"] = canonical_id


def build_dataset_inventory_md(
    *,
    dataset: DatasetContext,
    sql_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
) -> str:
    if not sql_rows:
        return "\n".join(
            [
                f"# SQL Inventory for {dataset.dataset_name} (`{dataset.own_id}`)",
                "",
                f"- Dataset id: `{dataset.dataset_id}`",
                "- SQL items extracted: 0",
                f"- Rejected or non-SQL sources: {len(rejected_rows)}",
                "",
                "No explicit machine-readable SQL was extracted from the collected Phase B sources for this dataset.",
                "",
            ]
        )

    counts_by_source_type = Counter(row["source_type"] for row in sql_rows)
    counts_by_specificity = Counter(row["dataset_specificity_label"] for row in sql_rows)
    counts_by_confidence = Counter(row["evidence_confidence"] for row in sql_rows)
    counts_by_complexity = Counter(row["sql_complexity"] for row in sql_rows)
    duplicate_count = sum(1 for row in sql_rows if row["is_near_duplicate"] == "yes")
    unique_hash_count = len({row["sql_text_norm_hash"] for row in sql_rows})

    lines = [
        f"# SQL Inventory for {dataset.dataset_name} (`{dataset.own_id}`)",
        "",
        f"- Dataset id: `{dataset.dataset_id}`",
        f"- SQL items extracted: {len(sql_rows)}",
        f"- Unique normalized SQL hashes: {unique_hash_count}",
        f"- Duplicate rows retained: {duplicate_count}",
        f"- Rejected or non-SQL sources: {len(rejected_rows)}",
        "",
        "## Counts by source_type",
        "",
    ]
    for key, value in sorted(counts_by_source_type.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Counts by dataset_specificity_label", ""])
    for key, value in sorted(counts_by_specificity.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Counts by evidence_confidence", ""])
    for key, value in sorted(counts_by_confidence.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Counts by sql_complexity", ""])
    for key, value in sorted(counts_by_complexity.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## SQL Items", ""])
    for row in sql_rows:
        lines.extend(
            [
                f"### {row['sql_item_id']}",
                "",
                f"- Source type: `{row['source_type']}`",
                f"- Source URL: `{row['source_url']}`",
                f"- Specificity: `{row['dataset_specificity_label']}`",
                f"- Confidence: `{row['evidence_confidence']}`",
                f"- Duplicate: `{row['is_near_duplicate']}`",
                f"- Intent: `{row['query_intent_label']}`",
                f"- Family tag: `{row['family_tag_guess']}`",
                "",
                "```sql",
                row["sql_text_raw"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def build_global_summary(
    *,
    datasets: list[DatasetContext],
    sql_rows_by_dataset: dict[str, list[dict[str, Any]]],
    rejected_rows_by_dataset: dict[str, list[dict[str, Any]]],
    all_rows: list[dict[str, Any]],
) -> str:
    duplicate_count = sum(1 for row in all_rows if row["is_near_duplicate"] == "yes")
    zero_sql_datasets = [dataset.own_id for dataset in datasets if not sql_rows_by_dataset.get(dataset.own_id)]
    counts_by_source_type = Counter(row["source_type"] for row in all_rows)
    counts_by_specificity = Counter(row["dataset_specificity_label"] for row in all_rows)
    counts_by_confidence = Counter(row["evidence_confidence"] for row in all_rows)
    counts_by_complexity = Counter(row["sql_complexity"] for row in all_rows)
    counts_by_dialect = Counter(row["sql_dialect_guess"] for row in all_rows)

    lines = [
        "# Phase C SQL Extraction Summary",
        "",
        f"- Generated at UTC: `{utc_now_iso()}`",
        f"- Dataset count: {len(datasets)}",
        f"- Total SQL items: {len(all_rows)}",
        f"- Duplicate SQL rows retained: {duplicate_count}",
        f"- Datasets with zero extracted SQL: {len(zero_sql_datasets)}",
        "",
        "## Per-Dataset SQL Counts",
        "",
    ]
    for dataset in datasets:
        lines.append(
            f"- `{dataset.own_id}` - {dataset.dataset_name}: "
            f"{len(sql_rows_by_dataset.get(dataset.own_id, []))} SQL items; "
            f"{len(rejected_rows_by_dataset.get(dataset.own_id, []))} rejected/non-SQL sources"
        )
    if zero_sql_datasets:
        lines.extend(["", "## Zero-SQL Datasets", ""])
        for own_id in zero_sql_datasets:
            dataset = next(item for item in datasets if item.own_id == own_id)
            lines.append(f"- `{own_id}` - {dataset.dataset_name}")
    for title, counter in (
        ("Counts by source_type", counts_by_source_type),
        ("Counts by dataset_specificity_label", counts_by_specificity),
        ("Counts by evidence_confidence", counts_by_confidence),
        ("Counts by sql_complexity", counts_by_complexity),
        ("Counts by sql_dialect_guess", counts_by_dialect),
    ):
        lines.extend(["", f"## {title}", ""])
        for key, value in sorted(counter.items()):
            lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    output_root = args.output_root.resolve()
    scope_csv = args.scope_csv.resolve()
    global_source_inventory = args.global_source_inventory.resolve()
    global_dir = output_root / "global"
    script_path = Path(__file__).resolve()

    datasets = load_datasets(scope_csv, output_root)
    global_source_rows = read_csv_rows(global_source_inventory)

    sql_rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    rejected_rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    clone_cache: dict[tuple[str, str], CloneResult] = {}

    with tempfile.TemporaryDirectory(prefix="sql_phase_c_") as temp_dir_name:
        temp_root = Path(temp_dir_name)
        for dataset in datasets:
            sources = load_source_records(dataset.source_inventory_path)
            dataset_sql_rows: list[dict[str, Any]] = []
            dataset_rejected_rows: list[dict[str, Any]] = []

            for source in sources:
                extracted_rows, rejected = process_source(
                    dataset=dataset,
                    source=source,
                    clone_cache=clone_cache,
                    temp_root=temp_root,
                    timeout_seconds=args.timeout_seconds,
                )
                if extracted_rows:
                    dataset_sql_rows.extend(extracted_rows)
                elif rejected is not None:
                    dataset_rejected_rows.append(rejected)

            assign_sql_item_ids(dataset_sql_rows, dataset.own_id)
            sql_rows_by_dataset[dataset.own_id] = dataset_sql_rows
            rejected_rows_by_dataset[dataset.own_id] = dataset_rejected_rows
            all_rows.extend(dataset_sql_rows)

    mark_duplicates(all_rows)

    for dataset in datasets:
        dataset_sql_rows = sql_rows_by_dataset[dataset.own_id]
        dataset_rejected_rows = rejected_rows_by_dataset[dataset.own_id]
        sql_dir = dataset.dataset_dir / "sql"
        raw_sql_candidates_path = sql_dir / "raw_sql_candidates.jsonl"
        sql_inventory_csv_path = sql_dir / "sql_inventory.csv"
        sql_inventory_md_path = sql_dir / "sql_inventory.md"
        rejected_csv_path = sql_dir / "rejected_or_non_sql_sources.csv"

        write_jsonl(
            raw_sql_candidates_path,
            [
                {
                    **row,
                    "dataset_folder": dataset.dataset_dir.name,
                }
                for row in dataset_sql_rows
            ],
        )
        write_csv(sql_inventory_csv_path, SQL_INVENTORY_FIELDNAMES, dataset_sql_rows)
        sql_inventory_md_path.write_text(
            build_dataset_inventory_md(
                dataset=dataset,
                sql_rows=dataset_sql_rows,
                rejected_rows=dataset_rejected_rows,
            ),
            encoding="utf-8",
        )
        write_csv(rejected_csv_path, REJECTED_FIELDNAMES, dataset_rejected_rows)

    master_sql_inventory_path = global_dir / "master_sql_inventory_all.csv"
    sql_extraction_summary_path = global_dir / "sql_extraction_summary.md"
    manifest_path = global_dir / "run_manifest_phase_c.json"

    write_csv(master_sql_inventory_path, SQL_INVENTORY_FIELDNAMES, all_rows)
    sql_extraction_summary_path.write_text(
        build_global_summary(
            datasets=datasets,
            sql_rows_by_dataset=sql_rows_by_dataset,
            rejected_rows_by_dataset=rejected_rows_by_dataset,
            all_rows=all_rows,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": "C",
        "phase_name": "sql_evidence_extraction_and_inventory_build",
        "generated_at_utc": utc_now_iso(),
        "script_path": str(script_path),
        "rerun_command": (
            f"python3 {script_path} "
            f"--global-source-inventory {global_source_inventory} "
            f"--scope-csv {scope_csv} "
            f"--output-root {output_root}"
        ),
        "inputs": {
            "global_source_inventory_path": str(global_source_inventory),
            "global_source_inventory_sha256": sha256_file(global_source_inventory),
            "global_source_inventory_row_count": len(global_source_rows),
            "scope_csv_path": str(scope_csv),
            "scope_csv_sha256": sha256_file(scope_csv),
            "dataset_count": len(datasets),
        },
        "outputs": {
            "master_sql_inventory_all_csv": str(master_sql_inventory_path),
            "sql_extraction_summary_md": str(sql_extraction_summary_path),
            "run_manifest_phase_c_json": str(manifest_path),
        },
        "counts": {
            "total_sql_items": len(all_rows),
            "total_unique_sql_hashes": len({row["sql_text_norm_hash"] for row in all_rows}),
            "total_duplicate_rows": sum(1 for row in all_rows if row["is_near_duplicate"] == "yes"),
            "total_rejected_or_non_sql_source_rows": sum(len(rows) for rows in rejected_rows_by_dataset.values()),
        },
        "per_dataset": [
            {
                "own_id": dataset.own_id,
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "sql_item_count": len(sql_rows_by_dataset[dataset.own_id]),
                "rejected_or_non_sql_source_count": len(rejected_rows_by_dataset[dataset.own_id]),
                "sql_inventory_csv": str(dataset.dataset_dir / "sql" / "sql_inventory.csv"),
                "sql_inventory_md": str(dataset.dataset_dir / "sql" / "sql_inventory.md"),
                "raw_sql_candidates_jsonl": str(dataset.dataset_dir / "sql" / "raw_sql_candidates.jsonl"),
                "rejected_or_non_sql_sources_csv": str(dataset.dataset_dir / "sql" / "rejected_or_non_sql_sources.csv"),
            }
            for dataset in datasets
        ],
    }
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
