#!/usr/bin/env python3
"""Build Phase B source-discovery artifacts for sql_high datasets."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCOPE_CSV = Path("logs/sql_high_corpus_build_20260404/scope/high_datasets.csv")
DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404")
DEFAULT_DEEP_RESEARCH_DOC = Path("doc/dataset_deep_research_active51_20260403.md")
DEFAULT_EXTERNAL_RESEARCH_DOC = Path("doc/candidate_plus_external_dataset_research_20260403.md")
USER_AGENT = "Mozilla/5.0 (compatible; SQLagent-PhaseB/1.0; +https://github.com/openai)"
REQUIRED_FIELDS = [
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
]
STOPWORDS = {
    "and",
    "data",
    "dataset",
    "datasets",
    "for",
    "from",
    "the",
    "with",
    "prediction",
    "challenge",
    "classification",
    "regression",
    "analysis",
    "customer",
}


@dataclass(frozen=True)
class DatasetRow:
    own_id: str
    dataset_id: str
    dataset_name: str
    dataset_link: str
    source_type: str
    source_group: str
    row: dict[str, str]


@dataclass
class FetchResult:
    original_url: str
    final_url: str
    http_status: str
    content_type: str
    title: str
    body_text: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover official, task-context, Kaggle, GitHub, and SQL-related sources "
            "for every sql_high dataset listed in the Phase A scope file."
        )
    )
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--deep-research-doc", type=Path, default=DEFAULT_DEEP_RESEARCH_DOC)
    parser.add_argument("--external-research-doc", type=Path, default=DEFAULT_EXTERNAL_RESEARCH_DOC)
    parser.add_argument("--github-repo-results", type=int, default=3)
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_scope(scope_csv: Path) -> list[DatasetRow]:
    with scope_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    datasets: list[DatasetRow] = []
    for row in rows:
        datasets.append(
            DatasetRow(
                own_id=(row.get("own_id") or "").strip(),
                dataset_id=(row.get("dataset_id") or "").strip(),
                dataset_name=(row.get("dataset_name") or "").strip(),
                dataset_link=(row.get("dataset_link") or "").strip(),
                source_type=(row.get("source_type") or "").strip(),
                source_group=(row.get("source_group") or "").strip(),
                row=row,
            )
        )
    return datasets


def normalize_url(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    parsed = urllib.parse.urlsplit(text)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    query = parsed.query
    fragment = ""
    if netloc == "www.bing.com" and parsed.scheme == "http":
        scheme = "https"
    if netloc == "www.kaggle.com" and path.startswith("/c/"):
        slug = path.split("/", 2)[2]
        path = f"/competitions/{slug}"
    return urllib.parse.urlunsplit((scheme, netloc, path, query, fragment))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_markdown_links(text: str) -> list[tuple[str, str]]:
    return [(label.strip(), normalize_url(url)) for label, url in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text or "")]


def keep_seed_url(url: str) -> bool:
    if not url:
        return False
    netloc = urllib.parse.urlsplit(url).netloc.lower()
    if not netloc:
        return False
    return netloc in {
        "archive.ics.uci.edu",
        "www.openml.org",
        "huggingface.co",
        "www.kaggle.com",
        "github.com",
        "gist.github.com",
        "mode.com",
        "www.postgresql.org",
        "doi.org",
    }


def parse_deep_research_doc(path: Path) -> dict[str, list[dict[str, str]]]:
    seeds: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in read_text(path).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 10:
            continue
        own_id = parts[0]
        if not own_id or own_id.lower() == "own_id":
            continue
        official_cell = parts[-2]
        sql_cell = parts[-1]
        for label, url in extract_markdown_links(official_cell):
            if not keep_seed_url(url):
                continue
            seeds[own_id].append(
                {
                    "source_url": url,
                    "source_title_hint": "",
                    "retrieval_method": f"seed:{path}",
                    "source_type": classify_source_type(url, official_context=True),
                    "relevance_label": default_relevance_for_url(url, official_context=True),
                    "dataset_specificity_hint": default_specificity_for_url(url, official_context=True),
                    "has_sql_text": default_has_sql_text(url, official_context=True),
                    "notes": f"Seeded from {path.name} official/context column (label {label}).",
                }
            )
        for label, url in extract_markdown_links(sql_cell):
            if not keep_seed_url(url):
                continue
            seeds[own_id].append(
                {
                    "source_url": url,
                    "source_title_hint": "",
                    "retrieval_method": f"seed:{path}",
                    "source_type": classify_source_type(url, official_context=False),
                    "relevance_label": default_relevance_for_url(url, official_context=False),
                    "dataset_specificity_hint": default_specificity_for_url(url, official_context=False),
                    "has_sql_text": default_has_sql_text(url, official_context=False),
                    "notes": f"Seeded from {path.name} SQL/resource column (label {label}).",
                }
            )
    return seeds


def parse_external_research_doc(path: Path) -> dict[str, list[dict[str, str]]]:
    seeds: dict[str, list[dict[str, str]]] = defaultdict(list)
    current_dataset_id = ""
    current_dataset_name = ""
    for line in read_text(path).splitlines():
        heading_match = re.match(r"###\s+(.+?)\s+\(`([^`]+)`\)", line.strip())
        if heading_match:
            current_dataset_name = heading_match.group(1).strip()
            current_dataset_id = heading_match.group(2).strip()
            continue
        if not current_dataset_id:
            continue
        stripped = line.strip()
        if stripped.startswith("- 下游任务证据"):
            for label, url in extract_markdown_links(stripped):
                if not keep_seed_url(url):
                    continue
                seeds[current_dataset_id].append(
                    {
                        "source_url": url,
                        "source_title_hint": current_dataset_name,
                        "retrieval_method": f"seed:{path}",
                        "source_type": classify_source_type(url, official_context=True),
                        "relevance_label": default_relevance_for_url(url, official_context=True),
                        "dataset_specificity_hint": default_specificity_for_url(url, official_context=True),
                        "has_sql_text": default_has_sql_text(url, official_context=True),
                        "notes": f"Seeded from {path.name} downstream-task evidence line.",
                    }
                )
        if stripped.startswith("- Query/SQL证据"):
            for label, url in extract_markdown_links(stripped):
                if not keep_seed_url(url):
                    continue
                seeds[current_dataset_id].append(
                    {
                        "source_url": url,
                        "source_title_hint": current_dataset_name,
                        "retrieval_method": f"seed:{path}",
                        "source_type": classify_source_type(url, official_context=False),
                        "relevance_label": default_relevance_for_url(url, official_context=False),
                        "dataset_specificity_hint": default_specificity_for_url(url, official_context=False),
                        "has_sql_text": default_has_sql_text(url, official_context=False),
                        "notes": f"Seeded from {path.name} SQL evidence line.",
                    }
                )
    return seeds


def classify_source_type(url: str, *, official_context: bool) -> str:
    parsed = urllib.parse.urlsplit(url)
    netloc = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()
    if netloc == "archive.ics.uci.edu":
        if "/api/" in path or "dataset?id=" in query:
            return "official_api"
        return "official_dataset_page"
    if netloc == "www.openml.org":
        if "/api/" in path:
            return "openml_api"
        if re.match(r"/t/\d+", path):
            return "openml_task_page"
        if "type=task" in query:
            return "openml_task_search"
        return "official_dataset_page"
    if netloc == "www.kaggle.com":
        if path.startswith("/code") and "search=" in query:
            return "kaggle_code_search"
        if path.endswith("/code") or path.startswith("/code"):
            return "kaggle_code_page" if official_context else "kaggle_code_or_notebook"
        if path.endswith("/data"):
            return "kaggle_data_page"
        if path.endswith("/overview"):
            return "kaggle_overview_page"
        if "/competitions/" in path or "/datasets/" in path:
            return "official_dataset_page"
    if netloc == "github.com":
        if path == "/search":
            if "type=code" in query:
                return "github_code_search"
            return "github_repo_search"
        if "/blob/" in path:
            return "github_file"
        if "/releases" in path:
            return "github_release"
        return "github_repo"
    if netloc == "gist.github.com":
        return "gist"
    if netloc == "huggingface.co":
        if "/blob/" in path:
            return "readme_or_metadata"
        return "official_dataset_page"
    if netloc == "mode.com" or netloc == "www.postgresql.org":
        return "tutorial_blog"
    if netloc == "doi.org":
        return "paper"
    return "other"


def default_relevance_for_url(url: str, *, official_context: bool) -> str:
    source_type = classify_source_type(url, official_context=official_context)
    if source_type in {
        "official_dataset_page",
        "official_api",
        "openml_api",
        "openml_task_page",
        "kaggle_code_page",
        "kaggle_data_page",
        "kaggle_overview_page",
        "github_file",
        "gist",
        "paper",
    }:
        return "high" if official_context or source_type not in {"github_file", "gist"} else "medium"
    if source_type in {"github_repo", "kaggle_code_or_notebook", "tutorial_blog", "readme_or_metadata"}:
        return "medium"
    return "low"


def default_specificity_for_url(url: str, *, official_context: bool) -> str:
    source_type = classify_source_type(url, official_context=official_context)
    if source_type in {
        "official_dataset_page",
        "official_api",
        "openml_api",
        "openml_task_page",
        "kaggle_code_page",
        "kaggle_data_page",
        "kaggle_overview_page",
        "readme_or_metadata",
        "paper",
    }:
        return "strict"
    if source_type in {"tutorial_blog", "github_repo_search", "github_code_search", "kaggle_code_search"}:
        return "weak"
    return "unknown"


def default_has_sql_text(url: str, *, official_context: bool) -> str:
    source_type = classify_source_type(url, official_context=official_context)
    parsed = urllib.parse.urlsplit(url)
    lowered = f"{parsed.path.lower()}?{parsed.query.lower()}"
    if source_type in {"github_file", "gist", "tutorial_blog"}:
        return "yes"
    if source_type in {"kaggle_code_page", "kaggle_code_or_notebook", "github_repo", "github_repo_search", "github_code_search", "kaggle_code_search"}:
        return "partial"
    if "sql" in lowered:
        return "partial"
    return "no"


def dataset_query_tokens(dataset: DatasetRow) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", dataset.dataset_name.lower())
    slug = re.findall(r"[a-z0-9]+", dataset.dataset_id.lower().split("/")[-1])
    tokens = []
    for token in raw + slug:
        if token in STOPWORDS:
            continue
        if len(token) < 3 and token not in {"sql", "uci"}:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens[:8]


def github_search_query(dataset: DatasetRow) -> str:
    name = dataset.dataset_name
    dataset_id = dataset.dataset_id.lower()
    if dataset_id.startswith("uci:"):
        return f"\"{name}\" uci sql"
    if dataset_id.startswith("openml:"):
        return f"\"{name}\" openml sql"
    if dataset_id.startswith("hf:"):
        return f"\"{name}\" sql"
    return f"\"{name}\" kaggle sql"


def kaggle_search_query(dataset: DatasetRow) -> str:
    return f"{dataset.dataset_name} sql"


def urlopen_text(url: str, timeout_seconds: int) -> FetchResult:
    last_error = ""
    for attempt in range(3):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                final_url = normalize_url(response.geturl())
                status = str(getattr(response, "status", "") or response.getcode() or "")
                content_type = response.headers.get("Content-Type", "")
                raw = response.read(512 * 1024)
                body = raw.decode("utf-8", errors="ignore")
                if status == "429" and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return FetchResult(
                    original_url=url,
                    final_url=final_url,
                    http_status=status,
                    content_type=content_type,
                    title=extract_title(body),
                    body_text=body,
                    error="",
                )
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read(256 * 1024)
                body = raw.decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            if exc.code == 429 and attempt < 2:
                time.sleep(2 * (attempt + 1))
                last_error = str(exc)
                continue
            return FetchResult(
                original_url=url,
                final_url=normalize_url(exc.geturl() or url),
                http_status=str(exc.code),
                content_type=str(exc.headers.get("Content-Type", "")),
                title=extract_title(body),
                body_text=body,
                error=str(exc),
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
                continue
            return FetchResult(
                original_url=url,
                final_url=normalize_url(url),
                http_status="ERROR",
                content_type="",
                title="",
                body_text="",
                error=str(exc),
            )
    return FetchResult(
        original_url=url,
        final_url=normalize_url(url),
        http_status="ERROR",
        content_type="",
        title="",
        body_text="",
        error=last_error or "Unknown fetch error",
    )


def extract_title(body: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", body, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return clean_text(match.group(1))
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.IGNORECASE | re.DOTALL)
    if h1_match:
        return clean_text(h1_match.group(1))
    return ""


def clean_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def fetch_many(urls: list[str], timeout_seconds: int, workers: int) -> dict[str, FetchResult]:
    results: dict[str, FetchResult] = {}
    if not urls:
        return results
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(urlopen_text, url, timeout_seconds): url
            for url in urls
        }
        for future in as_completed(future_map):
            url = future_map[future]
            results[url] = future.result()
    return results


def build_derived_sources(dataset: DatasetRow) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    dataset_link = normalize_url(dataset.dataset_link)
    sources.append(
        {
            "source_url": dataset_link,
            "source_title_hint": dataset.dataset_name,
            "retrieval_method": "derived:phase_a_scope_dataset_link",
            "source_type": classify_source_type(dataset_link, official_context=True),
            "relevance_label": "high",
            "dataset_specificity_hint": "strict",
            "has_sql_text": "no",
            "notes": "Official dataset page from Phase A scope file.",
        }
    )

    dataset_id = dataset.dataset_id.lower()
    if dataset_id.startswith("uci:"):
        uci_id = dataset.dataset_id.split(":", 1)[1]
        api_url = normalize_url(f"https://archive.ics.uci.edu/api/dataset?id={uci_id}")
        sources.append(
            {
                "source_url": api_url,
                "source_title_hint": f"UCI Dataset API for id={uci_id}",
                "retrieval_method": "derived:uci_api_from_dataset_id",
                "source_type": "official_api",
                "relevance_label": "high",
                "dataset_specificity_hint": "strict",
                "has_sql_text": "no",
                "notes": "Structured UCI API endpoint derived from dataset_id.",
            }
        )

    if dataset_id.startswith("openml:"):
        openml_id = dataset.dataset_id.split(":", 1)[1]
        task_api = normalize_url(f"https://www.openml.org/api/v1/json/task/list/data_id/{openml_id}")
        task_search = normalize_url(f"https://www.openml.org/search?type=task&data_id={openml_id}")
        sources.extend(
            [
                {
                    "source_url": task_api,
                    "source_title_hint": f"OpenML task list API for data_id={openml_id}",
                    "retrieval_method": "derived:openml_task_api_from_dataset_id",
                    "source_type": "openml_api",
                    "relevance_label": "high",
                    "dataset_specificity_hint": "strict",
                    "has_sql_text": "no",
                    "notes": "OpenML task-list API derived from dataset_id.",
                },
                {
                    "source_url": task_search,
                    "source_title_hint": f"OpenML task search for data_id={openml_id}",
                    "retrieval_method": "derived:openml_task_search_from_dataset_id",
                    "source_type": "openml_task_search",
                    "relevance_label": "high",
                    "dataset_specificity_hint": "strict",
                    "has_sql_text": "no",
                    "notes": "OpenML task search page derived from dataset_id.",
                },
            ]
        )

    if "kaggle.com" in urllib.parse.urlsplit(dataset_link).netloc.lower():
        parsed = urllib.parse.urlsplit(dataset_link)
        path = parsed.path.rstrip("/")
        if path.startswith("/competitions/"):
            base = normalize_url(f"https://www.kaggle.com{path}")
            sources.extend(
                [
                    {
                        "source_url": base,
                        "source_title_hint": dataset.dataset_name,
                        "retrieval_method": "derived:kaggle_competition_base",
                        "source_type": "official_dataset_page",
                        "relevance_label": "high",
                        "dataset_specificity_hint": "strict",
                        "has_sql_text": "no",
                        "notes": "Canonical Kaggle competition page.",
                    },
                    {
                        "source_url": normalize_url(base + "/overview"),
                        "source_title_hint": dataset.dataset_name,
                        "retrieval_method": "derived:kaggle_competition_overview",
                        "source_type": "kaggle_overview_page",
                        "relevance_label": "high",
                        "dataset_specificity_hint": "strict",
                        "has_sql_text": "no",
                        "notes": "Kaggle competition overview page for downstream-task context.",
                    },
                    {
                        "source_url": normalize_url(base + "/data"),
                        "source_title_hint": dataset.dataset_name,
                        "retrieval_method": "derived:kaggle_competition_data",
                        "source_type": "kaggle_data_page",
                        "relevance_label": "high",
                        "dataset_specificity_hint": "strict",
                        "has_sql_text": "no",
                        "notes": "Kaggle competition data page.",
                    },
                    {
                        "source_url": normalize_url(base + "/code"),
                        "source_title_hint": dataset.dataset_name,
                        "retrieval_method": "derived:kaggle_competition_code",
                        "source_type": "kaggle_code_page",
                        "relevance_label": "high",
                        "dataset_specificity_hint": "strict",
                        "has_sql_text": "partial",
                        "notes": "Kaggle competition code hub; notebook-level SQL presence varies.",
                    },
                ]
            )
        if path.startswith("/datasets/"):
            base = normalize_url(f"https://www.kaggle.com{path}")
            sources.extend(
                [
                    {
                        "source_url": base,
                        "source_title_hint": dataset.dataset_name,
                        "retrieval_method": "derived:kaggle_dataset_base",
                        "source_type": "official_dataset_page",
                        "relevance_label": "high",
                        "dataset_specificity_hint": "strict",
                        "has_sql_text": "no",
                        "notes": "Canonical Kaggle dataset page.",
                    },
                    {
                        "source_url": normalize_url(base + "/code"),
                        "source_title_hint": dataset.dataset_name,
                        "retrieval_method": "derived:kaggle_dataset_code",
                        "source_type": "kaggle_code_page",
                        "relevance_label": "high",
                        "dataset_specificity_hint": "strict",
                        "has_sql_text": "partial",
                        "notes": "Kaggle dataset code hub; notebook-level SQL presence varies.",
                    },
                ]
            )

    if dataset_id.startswith("hf:") or "huggingface.co" in urllib.parse.urlsplit(dataset_link).netloc.lower():
        readme_url = normalize_url(dataset_link.rstrip("/") + "/blob/main/README.md")
        sources.append(
            {
                "source_url": readme_url,
                "source_title_hint": f"{dataset.dataset_name} README",
                "retrieval_method": "derived:huggingface_readme",
                "source_type": "readme_or_metadata",
                "relevance_label": "high",
                "dataset_specificity_hint": "strict",
                "has_sql_text": "no",
                "notes": "Hugging Face dataset README for dataset context.",
            }
        )

    github_repo_search = normalize_url(
        "https://github.com/search?q="
        + urllib.parse.quote_plus(github_search_query(dataset))
        + "&type=repositories"
    )
    github_code_search = normalize_url(
        "https://github.com/search?q="
        + urllib.parse.quote_plus(github_search_query(dataset))
        + "&type=code"
    )
    kaggle_code_search = normalize_url(
        "https://www.kaggle.com/code?search=" + urllib.parse.quote_plus(kaggle_search_query(dataset))
    )
    sources.extend(
        [
            {
                "source_url": github_repo_search,
                "source_title_hint": f"GitHub repository search for {dataset.dataset_name}",
                "retrieval_method": "derived:github_repo_search",
                "source_type": "github_repo_search",
                "relevance_label": "low",
                "dataset_specificity_hint": "collision_risk",
                "has_sql_text": "partial",
                "notes": "Direct GitHub repository search path retained as a weak but reproducible discovery source.",
            },
            {
                "source_url": github_code_search,
                "source_title_hint": f"GitHub code search for {dataset.dataset_name}",
                "retrieval_method": "derived:github_code_search",
                "source_type": "github_code_search",
                "relevance_label": "low",
                "dataset_specificity_hint": "collision_risk",
                "has_sql_text": "partial",
                "notes": "Direct GitHub code search path retained as a weak but reproducible discovery source.",
            },
            {
                "source_url": kaggle_code_search,
                "source_title_hint": f"Kaggle code search for {dataset.dataset_name}",
                "retrieval_method": "derived:kaggle_code_search",
                "source_type": "kaggle_code_search",
                "relevance_label": "low",
                "dataset_specificity_hint": "collision_risk",
                "has_sql_text": "partial",
                "notes": "Kaggle code search path retained as a weak but reproducible discovery source.",
            },
        ]
    )
    return sources


def add_candidate(raw_rows: list[dict[str, str]], dataset: DatasetRow, payload: dict[str, str]) -> None:
    row = {
        "own_id": dataset.own_id,
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.dataset_name,
        "source_type": payload["source_type"],
        "source_url": normalize_url(payload["source_url"]),
        "source_title": payload.get("source_title_hint", ""),
        "retrieval_method": payload["retrieval_method"],
        "http_status": "",
        "relevance_label": payload["relevance_label"],
        "dataset_specificity_hint": payload["dataset_specificity_hint"],
        "has_sql_text": payload["has_sql_text"],
        "notes": payload["notes"],
    }
    if row["source_url"]:
        raw_rows.append(row)


def parse_github_repo_results(search_html: str, max_results: int) -> list[str]:
    repo_urls: list[str] = []
    for href in re.findall(r'href="(/[^"/<>\s]+/[^"/<>\s]+)"', search_html):
        path = href.strip("/")
        parts = path.split("/")
        if len(parts) != 2:
            continue
        if parts[0] in {
            "search",
            "topics",
            "collections",
            "sponsors",
            "settings",
            "features",
            "orgs",
            "organizations",
            "marketplace",
            "about",
            "explore",
        }:
            continue
        repo_url = normalize_url("https://github.com/" + path)
        if repo_url not in repo_urls:
            repo_urls.append(repo_url)
        if len(repo_urls) >= max_results:
            break
    return repo_urls


def specificity_for_repo(dataset: DatasetRow, repo_url: str, title: str) -> str:
    haystack = f"{repo_url.lower()} {title.lower()}"
    tokens = dataset_query_tokens(dataset)
    matches = [token for token in tokens if token in haystack]
    if len(matches) >= 2:
        return "strict"
    if len(matches) == 1:
        return "weak"
    return "collision_risk"


def relevance_for_repo(repo_url: str, specificity: str) -> str:
    lower = repo_url.lower()
    if specificity == "strict" and "sql" in lower:
        return "high"
    if specificity == "strict":
        return "medium"
    if specificity == "weak":
        return "medium"
    return "low"


def has_sql_for_repo(repo_url: str, title: str) -> str:
    lower = f"{repo_url.lower()} {title.lower()}"
    if ".sql" in lower or " sql" in lower or "-sql" in lower or "_sql" in lower:
        return "yes"
    if "database" in lower or "query" in lower:
        return "partial"
    return "partial"


def extract_uci_doi_urls(body_text: str) -> list[str]:
    urls = []
    for doi in re.findall(r"https?://doi\.org/10\.[A-Za-z0-9./_-]+", body_text, flags=re.IGNORECASE):
        cleaned = doi.rstrip(").,;:]}\\\"'")
        normalized = normalize_url(cleaned)
        if normalized not in urls:
            urls.append(normalized)
    return urls


def extract_openml_task_urls(body_text: str, limit: int = 3) -> list[str]:
    urls = []
    for task_id in re.findall(r'href="/t/(\d+)"', body_text):
        task_url = normalize_url(f"https://www.openml.org/t/{task_id}")
        if task_url not in urls:
            urls.append(task_url)
        if len(urls) >= limit:
            break
    return urls


def infer_title(row: dict[str, str], fetch_result: FetchResult) -> str:
    if fetch_result.title:
        return fetch_result.title
    if row["source_title"]:
        return row["source_title"]
    source_type = row["source_type"]
    url = row["source_url"]
    if source_type == "official_api":
        return f"Official API endpoint for {row['dataset_name']}"
    if source_type == "openml_api":
        return f"OpenML task API for {row['dataset_name']}"
    if source_type == "github_repo_search":
        return f"GitHub repository search for {row['dataset_name']}"
    if source_type == "github_code_search":
        return f"GitHub code search for {row['dataset_name']}"
    if source_type == "kaggle_code_search":
        return f"Kaggle code search for {row['dataset_name']}"
    if source_type == "paper":
        return f"Paper or supplement link for {row['dataset_name']}"
    return url


def dedupe_dataset_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = normalize_url(row["source_url"])
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def aggregate_global_rows(dataset_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rows in dataset_rows.values():
        for row in rows:
            buckets[row["source_url"]].append(row)

    def combine(field: str, values: list[str]) -> str:
        unique = []
        for value in values:
            if value and value not in unique:
                unique.append(value)
        return " ; ".join(unique)

    def combine_relevance(values: list[str]) -> str:
        priority = {"high": 0, "medium": 1, "low": 2}
        filtered = [value for value in values if value]
        return sorted(filtered, key=lambda value: priority.get(value, 99))[0] if filtered else ""

    def combine_has_sql(values: list[str]) -> str:
        ordered = [value for value in values if value]
        if "yes" in ordered:
            return "yes"
        if "partial" in ordered:
            return "partial"
        return ordered[0] if ordered else ""

    def combine_specificity(values: list[str]) -> str:
        priority = {"strict": 0, "weak": 1, "collision_risk": 2, "unknown": 3}
        filtered = [value for value in values if value]
        return sorted(filtered, key=lambda value: priority.get(value, 99))[0] if filtered else ""

    rows: list[dict[str, str]] = []
    for source_url, group in sorted(buckets.items(), key=lambda item: item[0]):
        rows.append(
            {
                "own_id": combine("own_id", [row["own_id"] for row in group]),
                "dataset_id": combine("dataset_id", [row["dataset_id"] for row in group]),
                "dataset_name": combine("dataset_name", [row["dataset_name"] for row in group]),
                "source_type": combine("source_type", [row["source_type"] for row in group]),
                "source_url": source_url,
                "source_title": combine("source_title", [row["source_title"] for row in group]),
                "retrieval_method": combine("retrieval_method", [row["retrieval_method"] for row in group]),
                "http_status": combine("http_status", [row["http_status"] for row in group]),
                "relevance_label": combine_relevance([row["relevance_label"] for row in group]),
                "dataset_specificity_hint": combine_specificity([row["dataset_specificity_hint"] for row in group]),
                "has_sql_text": combine_has_sql([row["has_sql_text"] for row in group]),
                "notes": combine("notes", [row["notes"] for row in group]),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def write_dataset_notes(path: Path, dataset: DatasetRow, raw_rows: list[dict[str, str]], deduped_rows: list[dict[str, str]]) -> None:
    counts_by_type = Counter(row["source_type"] for row in deduped_rows)
    counts_by_relevance = Counter(row["relevance_label"] for row in deduped_rows)
    strongest_rows = [row for row in deduped_rows if row["relevance_label"] == "high"][:10]
    weak_rows = [row for row in deduped_rows if row["dataset_specificity_hint"] in {"weak", "collision_risk"}][:10]

    lines = [
        f"# Source Notes: {dataset.own_id}",
        "",
        f"- Dataset: `{dataset.dataset_name}`",
        f"- Dataset ID: `{dataset.dataset_id}`",
        f"- Official link from Phase A: `{dataset.dataset_link}`",
        f"- Raw source candidate count: `{len(raw_rows)}`",
        f"- Unique source URL count: `{len(deduped_rows)}`",
        f"- Relevance counts: `high={counts_by_relevance.get('high', 0)}`, `medium={counts_by_relevance.get('medium', 0)}`, `low={counts_by_relevance.get('low', 0)}`",
        "",
        "## Counts by Source Type",
        "",
        markdown_table(
            ["source_type", "count"],
            [[source_type, str(count)] for source_type, count in sorted(counts_by_type.items())],
        ),
        "",
        "## Strongest Sources",
        "",
    ]
    if strongest_rows:
        lines.extend(
            [
                f"- `{row['source_type']}` | {row['source_title']} | {row['source_url']} | sql={row['has_sql_text']}"
                for row in strongest_rows
            ]
        )
    else:
        lines.append("- No high-relevance sources were identified.")

    lines.extend(["", "## Weak or Collision-Risk Sources", ""])
    if weak_rows:
        lines.extend(
            [
                f"- `{row['source_type']}` | {row['source_title']} | {row['source_url']} | specificity={row['dataset_specificity_hint']}"
                for row in weak_rows
            ]
        )
    else:
        lines.append("- No weak or collision-risk sources remained after deduplication.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- GitHub search pages were used to harvest additional repo candidates where available.",
            "- Kaggle code pages and code-search URLs were retained even when notebook-level SQL could not be confirmed at Phase B.",
            "- Generic SQL tutorials are labeled weak where they provide concrete SQL but are not dataset-specific.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_progress(path: Path, dataset_rows: dict[str, list[dict[str, str]]], dataset_raw_rows: dict[str, list[dict[str, str]]], datasets: list[DatasetRow]) -> None:
    total_unique = sum(len(rows) for rows in dataset_rows.values())
    lines = [
        "# Phase B Progress",
        "",
        f"- Generated at UTC: `{utc_now_iso()}`",
        f"- Dataset count: `{len(datasets)}`",
        f"- Sum of per-dataset unique source counts: `{total_unique}`",
        "",
        markdown_table(
            ["own_id", "dataset_name", "raw_candidates", "unique_urls", "high", "medium", "low"],
            [
                [
                    dataset.own_id,
                    dataset.dataset_name,
                    str(len(dataset_raw_rows.get(dataset.own_id, []))),
                    str(len(dataset_rows.get(dataset.own_id, []))),
                    str(sum(1 for row in dataset_rows.get(dataset.own_id, []) if row["relevance_label"] == "high")),
                    str(sum(1 for row in dataset_rows.get(dataset.own_id, []) if row["relevance_label"] == "medium")),
                    str(sum(1 for row in dataset_rows.get(dataset.own_id, []) if row["relevance_label"] == "low")),
                ]
                for dataset in datasets
            ],
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    scope_csv = args.scope_csv.resolve()
    output_root = args.output_root.resolve()
    deep_doc = args.deep_research_doc.resolve()
    external_doc = args.external_research_doc.resolve()
    script_path = Path(__file__).resolve()

    datasets = read_scope(scope_csv)
    datasets_by_own_id = {dataset.own_id: dataset for dataset in datasets}
    datasets_by_dataset_id = {dataset.dataset_id: dataset for dataset in datasets}

    deep_seeds = parse_deep_research_doc(deep_doc)
    external_seeds = parse_external_research_doc(external_doc)

    raw_rows_by_own_id: dict[str, list[dict[str, str]]] = defaultdict(list)

    for dataset in datasets:
        for payload in build_derived_sources(dataset):
            add_candidate(raw_rows_by_own_id[dataset.own_id], dataset, payload)

        for payload in deep_seeds.get(dataset.own_id, []):
            add_candidate(raw_rows_by_own_id[dataset.own_id], dataset, payload)

        for payload in external_seeds.get(dataset.dataset_id, []):
            add_candidate(raw_rows_by_own_id[dataset.own_id], dataset, payload)

    official_urls_to_fetch: list[str] = []
    github_search_urls_to_fetch: list[str] = []
    for dataset in datasets:
        for row in raw_rows_by_own_id[dataset.own_id]:
            if row["source_type"] in {"official_dataset_page", "official_api", "openml_api", "openml_task_page"}:
                if row["source_url"] not in official_urls_to_fetch:
                    official_urls_to_fetch.append(row["source_url"])
            if row["source_type"] == "github_repo_search":
                if row["source_url"] not in github_search_urls_to_fetch:
                    github_search_urls_to_fetch.append(row["source_url"])

    official_fetch = fetch_many(official_urls_to_fetch, args.timeout_seconds, args.fetch_workers)

    for dataset in datasets:
        for row in list(raw_rows_by_own_id[dataset.own_id]):
            if row["source_type"] == "official_dataset_page" and "archive.ics.uci.edu" in row["source_url"]:
                result = official_fetch.get(row["source_url"])
                if result:
                    for doi_url in extract_uci_doi_urls(result.body_text):
                        add_candidate(
                            raw_rows_by_own_id[dataset.own_id],
                            dataset,
                            {
                                "source_url": doi_url,
                                "source_title_hint": f"Dataset DOI for {dataset.dataset_name}",
                                "retrieval_method": "derived:uci_page_doi_extraction",
                                "source_type": "paper",
                                "relevance_label": "medium",
                                "dataset_specificity_hint": "strict",
                                "has_sql_text": "no",
                                "notes": "DOI link extracted from the UCI official dataset page for downstream-task context.",
                            },
                        )
            if row["source_type"] == "official_dataset_page" and "www.openml.org" in row["source_url"]:
                result = official_fetch.get(row["source_url"])
                if result:
                    for task_url in extract_openml_task_urls(result.body_text):
                        add_candidate(
                            raw_rows_by_own_id[dataset.own_id],
                            dataset,
                            {
                                "source_url": task_url,
                                "source_title_hint": f"OpenML task page for {dataset.dataset_name}",
                                "retrieval_method": "derived:openml_dataset_page_task_extraction",
                                "source_type": "openml_task_page",
                                "relevance_label": "high",
                                "dataset_specificity_hint": "strict",
                                "has_sql_text": "no",
                                "notes": "OpenML task page extracted from the OpenML dataset page.",
                            },
                        )

    github_search_fetch = fetch_many(github_search_urls_to_fetch, args.timeout_seconds, args.fetch_workers)
    github_repo_candidates: dict[str, list[str]] = {}
    for dataset in datasets:
        search_url = normalize_url(
            "https://github.com/search?q="
            + urllib.parse.quote_plus(github_search_query(dataset))
            + "&type=repositories"
        )
        search_result = github_search_fetch.get(search_url)
        if not search_result:
            continue
        repo_urls = parse_github_repo_results(search_result.body_text, args.github_repo_results)
        github_repo_candidates[dataset.own_id] = repo_urls

    repo_urls_to_fetch = sorted({url for urls in github_repo_candidates.values() for url in urls})
    repo_fetch = fetch_many(repo_urls_to_fetch, args.timeout_seconds, args.fetch_workers)

    for dataset in datasets:
        for repo_url in github_repo_candidates.get(dataset.own_id, []):
            repo_result = repo_fetch.get(repo_url)
            repo_title = repo_result.title if repo_result else ""
            specificity = specificity_for_repo(dataset, repo_url, repo_title)
            add_candidate(
                raw_rows_by_own_id[dataset.own_id],
                dataset,
                {
                    "source_url": repo_url,
                    "source_title_hint": repo_title or repo_url.rsplit("/", 1)[-1],
                    "retrieval_method": "derived:github_repo_search_top_results",
                    "source_type": "github_repo",
                    "relevance_label": relevance_for_repo(repo_url, specificity),
                    "dataset_specificity_hint": specificity,
                    "has_sql_text": has_sql_for_repo(repo_url, repo_title),
                    "notes": f"Top GitHub repository result harvested from repository search query `{github_search_query(dataset)}`.",
                },
            )

    unique_urls_to_fetch = sorted(
        {
            row["source_url"]
            for rows in raw_rows_by_own_id.values()
            for row in rows
            if row["source_url"]
        }
    )
    fetch_cache = fetch_many(unique_urls_to_fetch, args.timeout_seconds, args.fetch_workers)

    final_rows_by_own_id: dict[str, list[dict[str, str]]] = {}
    raw_jsonl_by_own_id: dict[str, list[dict[str, str]]] = {}

    for dataset in datasets:
        raw_rows = raw_rows_by_own_id[dataset.own_id]
        enriched_raw_rows: list[dict[str, str]] = []
        for row in raw_rows:
            result = fetch_cache.get(row["source_url"])
            enriched = dict(row)
            if result:
                enriched["http_status"] = result.http_status
                enriched["source_url"] = result.final_url or enriched["source_url"]
                enriched["source_title"] = infer_title(enriched, result)
                if result.error:
                    enriched["notes"] = f"{enriched['notes']} Fetch error: {result.error}"
            else:
                enriched["source_title"] = enriched["source_title"] or enriched["source_url"]
            enriched_raw_rows.append(enriched)
        deduped_rows = dedupe_dataset_rows(enriched_raw_rows)
        final_rows_by_own_id[dataset.own_id] = deduped_rows
        raw_jsonl_by_own_id[dataset.own_id] = enriched_raw_rows

    global_inventory_rows = aggregate_global_rows(final_rows_by_own_id)

    for dataset in datasets:
        dataset_root = output_root / "datasets" / dataset.own_id / "sources"
        raw_jsonl_path = dataset_root / "raw_source_candidates.jsonl"
        inventory_csv_path = dataset_root / "source_inventory.csv"
        notes_path = dataset_root / "source_notes.md"
        write_jsonl(raw_jsonl_path, raw_jsonl_by_own_id[dataset.own_id])
        write_csv(inventory_csv_path, final_rows_by_own_id[dataset.own_id])
        write_dataset_notes(notes_path, dataset, raw_jsonl_by_own_id[dataset.own_id], final_rows_by_own_id[dataset.own_id])

    global_dir = output_root / "global"
    all_source_inventory_path = global_dir / "all_source_inventory.csv"
    progress_path = global_dir / "progress_phase_b.md"
    manifest_path = global_dir / "run_manifest_phase_b.json"
    write_csv(all_source_inventory_path, global_inventory_rows)
    write_progress(progress_path, final_rows_by_own_id, raw_jsonl_by_own_id, datasets)

    manifest = {
        "phase": "B",
        "phase_name": "sql_high_source_discovery",
        "generated_at_utc": utc_now_iso(),
        "script_path": str(script_path),
        "scope_csv": str(scope_csv),
        "seed_docs": [str(deep_doc), str(external_doc)],
        "dataset_count": len(datasets),
        "per_dataset_unique_source_counts": {
            dataset.own_id: len(final_rows_by_own_id[dataset.own_id])
            for dataset in datasets
        },
        "global_unique_source_url_count": len(global_inventory_rows),
        "github_repo_results_per_dataset": args.github_repo_results,
        "fetch_workers": args.fetch_workers,
        "timeout_seconds": args.timeout_seconds,
        "outputs": {
            "all_source_inventory_csv": str(all_source_inventory_path),
            "run_manifest_phase_b_json": str(manifest_path),
            "progress_phase_b_md": str(progress_path),
            "per_dataset_sources_root": str(output_root / "datasets"),
        },
        "notes": [
            "Phase B keeps weak links when discovered, but labels them via relevance and specificity fields.",
            "URLs are deduplicated within each dataset inventory and again in the global inventory.",
            "GitHub repository candidates were live-harvested from GitHub repository search result pages.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for dataset in datasets:
        print(f"{dataset.own_id}\t{dataset.dataset_name}\t{len(final_rows_by_own_id[dataset.own_id])}")
    print(f"TOTAL_UNIQUE_SOURCE_URLS\t{len(global_inventory_rows)}")
    print("PHASE B DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
