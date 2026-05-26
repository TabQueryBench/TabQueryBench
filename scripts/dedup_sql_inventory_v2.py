#!/usr/bin/env python3
"""Normalize SQL and perform provenance-preserving V2 deduplication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_phase_c_sql_inventory import (  # noqa: E402
    extract_table_tokens,
    leading_sql_candidate,
)


DEFAULT_INPUT = Path(
    "logs/sql_high_corpus_build_20260404/v2_refinement/reclassify/master_sql_inventory_reclassified_v2.csv"
)
DEFAULT_OUTPUT_ROOT = Path("logs/sql_high_corpus_build_20260404/v2_refinement")

NEW_FIELDS = [
    "phase_c_is_near_duplicate",
    "phase_c_duplicate_of_sql_item_id",
    "sql_canonical_v2",
    "sql_fingerprint_v2",
    "is_primary_canonical",
    "canonical_group_id",
    "duplicate_type",
    "duplicate_of_sql_item_id",
]

MAPPING_FIELDNAMES = [
    "own_id",
    "dataset_id",
    "dataset_name",
    "canonical_group_id",
    "canonical_sql_item_id",
    "canonical_source_url",
    "canonical_v2_specificity_label",
    "canonical_v2_keep_candidate",
    "duplicate_sql_item_id",
    "duplicate_source_url",
    "duplicate_v2_specificity_label",
    "duplicate_v2_keep_candidate",
    "duplicate_type",
    "sql_fingerprint_v2",
    "sql_canonical_v2",
]

V2_LABEL_PRIORITY = {
    "strict": 1,
    "weak": 2,
    "collision_risk": 3,
    "reject_non_sql": 4,
    "": 5,
}
TIER_PRIORITY = {
    "tier_1_official": 1,
    "tier_2_primary_code": 2,
    "tier_3_secondary_explanatory": 3,
    "tier_4_low_trust": 4,
    "": 5,
}
CONFIDENCE_PRIORITY = {
    "high": 1,
    "medium": 2,
    "low": 3,
    "": 4,
}
EXECUTABLE_PRIORITY = {
    "pass": 1,
    "unknown": 2,
    "fail": 3,
    "": 4,
}


@dataclass
class RowState:
    row: dict[str, str]
    index: int
    own_id: str
    dataset_id: str
    dataset_name: str
    sql_item_id: str
    source_url: str
    sql_base_text: str
    raw_exact_hash: str
    sql_canonical_v2: str
    sql_fingerprint_v2: str
    near_signature_v2: str
    leading_keyword: str
    literal_signature: tuple[str, ...]
    table_signature: tuple[str, ...]
    token_sequence: tuple[str, ...]
    token_count: int
    group_root: int = -1
    canonical_group_id: str = ""
    is_primary_canonical: str = ""
    duplicate_type: str = ""
    duplicate_of_sql_item_id: str = ""


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and deduplicate the V2-reclassified SQL inventory while "
            "preserving canonical provenance mappings."
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_int(value: str | int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = value.strip()
    if not text:
        return 0
    return int(text)


def normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def base_sql_text(row: dict[str, str]) -> str:
    return (row.get("sql_text_clean") or "").strip() or (row.get("sql_text_raw") or "")


def leading_keyword(text: str) -> str:
    match = re.match(r"\s*([a-z]+)", text or "", flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def strip_wrapping_quotes(text: str) -> str:
    cleaned = text
    while True:
        previous = cleaned
        cleaned = re.sub(r'^\s*(?:""|\'\'|"""|\'\'\')\s*', "", cleaned)
        cleaned = re.sub(r'\s*(?:""|\'\'|"""|\'\'\')\s*$', "", cleaned)
        if cleaned == previous:
            return cleaned


def canonicalize_sql(text: str) -> str:
    candidate = normalize_newlines(text or "")
    candidate = leading_sql_candidate(candidate)
    candidate = strip_wrapping_quotes(candidate)
    candidate = candidate.replace("\\r\\n", " ").replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    candidate = re.sub(r"(?is)/\*.*?\*/", " ", candidate)
    candidate = re.sub(r"(?m)^\s*--.*?$", " ", candidate)
    candidate = re.sub(r"(?m)^\s*#.*?$", " ", candidate)
    candidate = re.sub(r"(?im)(^|[\n;])\s*go\s*(?=$|[\n;])", r"\1 ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    candidate = re.sub(r"\s*([(),;])\s*", r"\1 ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip().lower()
    candidate = re.sub(r";+\s*$", "", candidate).strip()
    return candidate


def near_signature(text: str) -> str:
    candidate = canonicalize_sql(text)
    candidate = re.sub(r"\bleft\s+outer\s+join\b", "left join", candidate)
    candidate = re.sub(r"\bright\s+outer\s+join\b", "right join", candidate)
    candidate = re.sub(r"\bfull\s+outer\s+join\b", "full join", candidate)
    candidate = re.sub(r"\binner\s+join\b", "join", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


def literal_signature(text: str) -> tuple[str, ...]:
    candidate = text or ""
    literals = re.findall(r"'[^']*'|\"[^\"]*\"|\b\d+(?:\.\d+)?\b", candidate)
    return tuple(sorted(literals))


def token_sequence(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9_]+", text or ""))


def sequence_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    return SequenceMatcher(a=left, b=right).ratio()


def row_priority(state: RowState, normalized_group_size: int, exact_group_size: int) -> tuple[Any, ...]:
    row = state.row
    return (
        0 if (row.get("v2_keep_candidate") or "").strip() == "yes" else 1,
        V2_LABEL_PRIORITY.get((row.get("v2_specificity_label") or "").strip(), 99),
        TIER_PRIORITY.get((row.get("v2_source_credibility_tier") or "").strip(), 99),
        CONFIDENCE_PRIORITY.get((row.get("evidence_confidence") or "").strip(), 99),
        EXECUTABLE_PRIORITY.get((row.get("executable_status") or "").strip(), 99),
        -normalized_group_size,
        -exact_group_size,
        len(state.sql_canonical_v2),
        len((row.get("source_url") or "").strip()),
        state.sql_item_id,
    )


def build_row_states(rows: list[dict[str, str]]) -> list[RowState]:
    states: list[RowState] = []
    for index, row in enumerate(rows):
        own_id = (row.get("own_id") or "").strip()
        dataset_id = (row.get("dataset_id") or "").strip()
        dataset_name = (row.get("dataset_name") or "").strip()
        sql_item_id = (row.get("sql_item_id") or "").strip()
        source_url = (row.get("source_url") or "").strip()
        base_text = base_sql_text(row)
        canonical = canonicalize_sql(base_text)
        fingerprint = short_hash(canonical)
        near_sig = near_signature(base_text)
        table_sig = tuple(sorted(set(token.lower() for token in extract_table_tokens(canonical))))
        tokens = token_sequence(canonical)
        states.append(
            RowState(
                row=row,
                index=index,
                own_id=own_id,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                sql_item_id=sql_item_id,
                source_url=source_url,
                sql_base_text=base_text,
                raw_exact_hash=short_hash(normalize_newlines(base_text).strip()),
                sql_canonical_v2=canonical,
                sql_fingerprint_v2=fingerprint,
                near_signature_v2=near_sig,
                leading_keyword=leading_keyword(canonical),
                literal_signature=literal_signature(canonical),
                table_signature=table_sig,
                token_sequence=tokens,
                token_count=len(tokens),
            )
        )
    return states


def group_states_by_dataset(states: list[RowState]) -> dict[str, list[RowState]]:
    grouped: dict[str, list[RowState]] = defaultdict(list)
    for state in states:
        grouped[state.own_id].append(state)
    return grouped


def near_duplicate_match(left: RowState, right: RowState) -> bool:
    if left.sql_fingerprint_v2 == right.sql_fingerprint_v2:
        return False
    if left.leading_keyword != right.leading_keyword:
        return False
    if left.table_signature != right.table_signature:
        return False
    if left.literal_signature != right.literal_signature:
        return False
    if left.token_count < 3 or right.token_count < 3:
        return False
    if abs(left.token_count - right.token_count) > max(2, int(0.10 * max(left.token_count, right.token_count))):
        return False
    if left.near_signature_v2 == right.near_signature_v2:
        return True
    return sequence_similarity(left.token_sequence, right.token_sequence) >= 0.965


def assign_groups(dataset_states: list[RowState]) -> list[list[RowState]]:
    index_map = {state.index: idx for idx, state in enumerate(dataset_states)}
    union = UnionFind(len(dataset_states))

    normalized_buckets: dict[str, list[RowState]] = defaultdict(list)
    near_buckets: dict[tuple[str, tuple[str, ...], tuple[str, ...]], list[RowState]] = defaultdict(list)
    for state in dataset_states:
        normalized_buckets[state.sql_fingerprint_v2].append(state)
        near_buckets[(state.leading_keyword, state.table_signature, state.literal_signature)].append(state)

    for bucket in normalized_buckets.values():
        if len(bucket) <= 1:
            continue
        first_local = index_map[bucket[0].index]
        for state in bucket[1:]:
            union.union(first_local, index_map[state.index])

    for bucket in near_buckets.values():
        if len(bucket) <= 1:
            continue
        for left_idx in range(len(bucket)):
            for right_idx in range(left_idx + 1, len(bucket)):
                left = bucket[left_idx]
                right = bucket[right_idx]
                if near_duplicate_match(left, right):
                    union.union(index_map[left.index], index_map[right.index])

    grouped: dict[int, list[RowState]] = defaultdict(list)
    for state in dataset_states:
        root = union.find(index_map[state.index])
        state.group_root = root
        grouped[root].append(state)
    return list(grouped.values())


def annotate_group(group_id: str, group_states: list[RowState]) -> list[dict[str, Any]]:
    normalized_sizes = Counter(state.sql_fingerprint_v2 for state in group_states)
    exact_sizes = Counter((state.sql_fingerprint_v2, state.raw_exact_hash) for state in group_states)
    primary = min(
        group_states,
        key=lambda state: row_priority(
            state,
            normalized_group_size=normalized_sizes[state.sql_fingerprint_v2],
            exact_group_size=exact_sizes[(state.sql_fingerprint_v2, state.raw_exact_hash)],
        ),
    )

    mapping_rows: list[dict[str, Any]] = []
    for state in group_states:
        state.canonical_group_id = group_id
        if state.sql_item_id == primary.sql_item_id:
            state.is_primary_canonical = "yes"
            state.duplicate_type = ""
            state.duplicate_of_sql_item_id = ""
            continue
        state.is_primary_canonical = "no"
        state.duplicate_of_sql_item_id = primary.sql_item_id
        if state.raw_exact_hash == primary.raw_exact_hash:
            duplicate_type = "exact"
        elif state.sql_fingerprint_v2 == primary.sql_fingerprint_v2:
            duplicate_type = "normalized"
        else:
            duplicate_type = "near"
        state.duplicate_type = duplicate_type
        mapping_rows.append(
            {
                "own_id": state.own_id,
                "dataset_id": state.dataset_id,
                "dataset_name": state.dataset_name,
                "canonical_group_id": group_id,
                "canonical_sql_item_id": primary.sql_item_id,
                "canonical_source_url": primary.source_url,
                "canonical_v2_specificity_label": (primary.row.get("v2_specificity_label") or "").strip(),
                "canonical_v2_keep_candidate": (primary.row.get("v2_keep_candidate") or "").strip(),
                "duplicate_sql_item_id": state.sql_item_id,
                "duplicate_source_url": state.source_url,
                "duplicate_v2_specificity_label": (state.row.get("v2_specificity_label") or "").strip(),
                "duplicate_v2_keep_candidate": (state.row.get("v2_keep_candidate") or "").strip(),
                "duplicate_type": duplicate_type,
                "sql_fingerprint_v2": primary.sql_fingerprint_v2,
                "sql_canonical_v2": primary.sql_canonical_v2,
            }
        )
    return mapping_rows


def build_summary_markdown(
    *,
    input_path: Path,
    states: list[RowState],
    per_dataset_rows: list[dict[str, Any]],
    global_counts: dict[str, Any],
    output_csv: Path,
    mapping_csv: Path,
) -> str:
    lines = [
        "# V2 Dedup Summary",
        "",
        f"- Generated at UTC: `{utc_now_iso()}`",
        f"- Input inventory: `{input_path.resolve()}`",
        f"- Annotated dedup inventory: `{output_csv.resolve()}`",
        f"- Duplicate mapping ledger: `{mapping_csv.resolve()}`",
        "- `master_sql_inventory_dedup_v2.csv` preserves all rows and annotates canonical membership; filter `is_primary_canonical=yes` to obtain the deduplicated active view.",
        "",
        "## Global Reduction",
        "",
        f"- Input rows: {global_counts['input_rows']}",
        f"- Primary canonical rows: {global_counts['primary_rows']}",
        f"- Duplicate rows dropped from deduplicated view: {global_counts['duplicate_rows']}",
        f"- Global reduction ratio: {global_counts['reduction_ratio']:.3f}",
        f"- Duplicate type counts: exact={global_counts['exact_duplicates']}, normalized={global_counts['normalized_duplicates']}, near={global_counts['near_duplicates']}",
        f"- Keep-candidate rows before dedup: {global_counts['keep_candidate_before']}",
        f"- Keep-candidate primary rows after dedup: {global_counts['keep_candidate_after']}",
        "",
        "## Per-Dataset Reduction",
        "",
        "| own_id | dataset_name | input_rows | primary_rows | duplicates_dropped | reduction_ratio | exact | normalized | near | keep_before | keep_after |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in per_dataset_rows:
        lines.append(
            f"| {row['own_id']} | {row['dataset_name']} | {row['input_rows']} | {row['primary_rows']} | "
            f"{row['duplicates_dropped']} | {row['reduction_ratio']:.3f} | {row['exact_duplicates']} | "
            f"{row['normalized_duplicates']} | {row['near_duplicates']} | {row['keep_candidate_before']} | {row['keep_candidate_after']} |"
        )
    return "\n".join(lines)


def build_manifest(
    *,
    args: argparse.Namespace,
    output_paths: list[Path],
    global_counts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "v2_phase2_deduplicate_reclassified_sql_inventory",
        "generated_at_utc": utc_now_iso(),
        "input": {
            "reclassified_inventory_path": str(args.input.resolve()),
            "reclassified_inventory_sha256": sha256_file(args.input),
        },
        "summary": global_counts,
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / "dedup"
    dedup_csv_path = output_dir / "master_sql_inventory_dedup_v2.csv"
    mapping_csv_path = output_dir / "dedup_mapping_v2.csv"
    summary_path = output_dir / "dedup_summary.md"
    manifest_path = output_dir / "run_manifest_v2_phase2.json"

    rows = read_csv_rows(args.input)
    states = build_row_states(rows)
    dataset_groups = group_states_by_dataset(states)

    mapping_rows: list[dict[str, Any]] = []
    group_sort_records: list[tuple[str, list[RowState]]] = []
    for own_id, dataset_states in dataset_groups.items():
        groups = assign_groups(dataset_states)
        groups.sort(key=lambda group: min(state.sql_item_id for state in group))
        for index, group in enumerate(groups, start=1):
            group_id = f"{own_id}_cg_{index:04d}"
            mapping_rows.extend(annotate_group(group_id, group))
            group_sort_records.append((group_id, group))

    output_rows: list[dict[str, Any]] = []
    per_dataset: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "own_id": "",
        "dataset_name": "",
        "input_rows": 0,
        "primary_rows": 0,
        "duplicates_dropped": 0,
        "exact_duplicates": 0,
        "normalized_duplicates": 0,
        "near_duplicates": 0,
        "keep_candidate_before": 0,
        "keep_candidate_after": 0,
    })
    for state in states:
        own_id = state.own_id
        summary = per_dataset[own_id]
        summary["own_id"] = own_id
        summary["dataset_name"] = state.dataset_name
        summary["input_rows"] += 1
        if (state.row.get("v2_keep_candidate") or "").strip() == "yes":
            summary["keep_candidate_before"] += 1
        if state.is_primary_canonical == "yes":
            summary["primary_rows"] += 1
            if (state.row.get("v2_keep_candidate") or "").strip() == "yes":
                summary["keep_candidate_after"] += 1
        else:
            summary["duplicates_dropped"] += 1
            summary[f"{state.duplicate_type}_duplicates"] += 1

        output_row = dict(state.row)
        output_row["phase_c_is_near_duplicate"] = output_row.get("is_near_duplicate", "")
        output_row["phase_c_duplicate_of_sql_item_id"] = output_row.get("duplicate_of_sql_item_id", "")
        output_row["sql_canonical_v2"] = state.sql_canonical_v2
        output_row["sql_fingerprint_v2"] = state.sql_fingerprint_v2
        output_row["is_primary_canonical"] = state.is_primary_canonical
        output_row["canonical_group_id"] = state.canonical_group_id
        output_row["duplicate_type"] = state.duplicate_type
        output_row["duplicate_of_sql_item_id"] = state.duplicate_of_sql_item_id
        output_rows.append(output_row)

    for summary in per_dataset.values():
        summary["reduction_ratio"] = (
            summary["duplicates_dropped"] / summary["input_rows"]
            if summary["input_rows"]
            else 0.0
        )

    per_dataset_rows = sorted(
        per_dataset.values(),
        key=lambda row: (-row["duplicates_dropped"], -row["reduction_ratio"], row["own_id"]),
    )

    global_counts = {
        "input_rows": len(states),
        "primary_rows": sum(1 for state in states if state.is_primary_canonical == "yes"),
        "duplicate_rows": sum(1 for state in states if state.is_primary_canonical == "no"),
        "reduction_ratio": (
            sum(1 for state in states if state.is_primary_canonical == "no") / len(states)
            if states
            else 0.0
        ),
        "exact_duplicates": sum(1 for state in states if state.duplicate_type == "exact"),
        "normalized_duplicates": sum(1 for state in states if state.duplicate_type == "normalized"),
        "near_duplicates": sum(1 for state in states if state.duplicate_type == "near"),
        "keep_candidate_before": sum(
            1 for state in states if (state.row.get("v2_keep_candidate") or "").strip() == "yes"
        ),
        "keep_candidate_after": sum(
            1
            for state in states
            if state.is_primary_canonical == "yes"
            and (state.row.get("v2_keep_candidate") or "").strip() == "yes"
        ),
    }

    output_fieldnames = list(rows[0].keys())
    for field in NEW_FIELDS:
        if field not in output_fieldnames:
            output_fieldnames.append(field)
    write_csv(dedup_csv_path, output_fieldnames, output_rows)
    write_csv(mapping_csv_path, MAPPING_FIELDNAMES, mapping_rows)
    write_text(
        summary_path,
        build_summary_markdown(
            input_path=args.input,
            states=states,
            per_dataset_rows=per_dataset_rows,
            global_counts=global_counts,
            output_csv=dedup_csv_path,
            mapping_csv=mapping_csv_path,
        ),
    )

    manifest_payload = build_manifest(
        args=args,
        output_paths=[dedup_csv_path, mapping_csv_path, summary_path],
        global_counts=global_counts,
    )
    write_json(manifest_path, manifest_payload)
    manifest_payload["outputs"] = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in [dedup_csv_path, mapping_csv_path, summary_path, manifest_path]
    ]
    write_json(manifest_path, manifest_payload)

    print(str(dedup_csv_path.resolve()))
    print(str(mapping_csv_path.resolve()))
    print(str(summary_path.resolve()))
    print(str(manifest_path.resolve()))
    print("")
    print("DUPLICATE REDUCTION")
    print(
        f"global\tinput={global_counts['input_rows']}\tprimary={global_counts['primary_rows']}\t"
        f"dropped={global_counts['duplicate_rows']}\treduction_ratio={global_counts['reduction_ratio']:.3f}\t"
        f"exact={global_counts['exact_duplicates']}\tnormalized={global_counts['normalized_duplicates']}\tnear={global_counts['near_duplicates']}"
    )
    for row in sorted(per_dataset_rows, key=lambda item: item["own_id"]):
        print(
            f"{row['own_id']}\t{row['dataset_name']}\tinput={row['input_rows']}\tprimary={row['primary_rows']}\t"
            f"dropped={row['duplicates_dropped']}\treduction_ratio={row['reduction_ratio']:.3f}\t"
            f"exact={row['exact_duplicates']}\tnormalized={row['normalized_duplicates']}\tnear={row['near_duplicates']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
