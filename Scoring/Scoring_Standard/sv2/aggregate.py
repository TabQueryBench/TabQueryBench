"""Run-level SV2 summaries: scorer-type macro means, query-weighted means, and invalid accounting.

Command line (re-summarize an analysis run)::

    PYTHONPATH=Scoring/code python3 -m tqb_scoring.standards.sv2.aggregate \
        runs/<run>/summaries/analysis_query_scores__all_datasets.jsonl \
        --group-by model_id --output sv2_summary_by_model.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import SCORE_METHOD, SCORER_TYPES


def _is_valid(row: Mapping[str, Any], score_field: str, valid_field: str) -> bool:
    valid = row.get(valid_field)
    if valid is None:
        details = row.get("semantic_details")
        valid = details.get("valid") if isinstance(details, Mapping) else None
    return bool(valid) and row.get(score_field) is not None


def _error_code(row: Mapping[str, Any], error_field: str) -> str:
    code = row.get(error_field)
    if code is None:
        details = row.get("semantic_details")
        code = details.get("error_code") if isinstance(details, Mapping) else None
    return str(code or "UNKNOWN")


def summarize_query_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = (),
    score_field: str = "query_score",
    scorer_field: str = "scorer_type",
    valid_field: str = "semantic_valid",
    error_field: str = "semantic_error_code",
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field) or "") for field in group_fields)].append(row)

    summaries = []
    for group_key, items in sorted(groups.items()):
        by_type: dict[str, list[float]] = defaultdict(list)
        valid_scores: list[float] = []
        invalid_reasons: Counter[str] = Counter()
        for row in items:
            if _is_valid(row, score_field, valid_field):
                score = float(row[score_field])
                valid_scores.append(score)
                by_type[str(row.get(scorer_field) or "")].append(score)
            else:
                invalid_reasons[_error_code(row, error_field)] += 1

        type_means = {scorer: math.fsum(scores) / len(scores) for scorer, scores in by_type.items() if scores}
        summary: dict[str, Any] = dict(zip(group_fields, group_key))
        summary.update(
            {
                "score_method": SCORE_METHOD,
                "n_queries_total": len(items),
                "n_queries_valid": len(valid_scores),
                "n_queries_invalid": len(items) - len(valid_scores),
                "valid_query_fraction": len(valid_scores) / len(items) if items else None,
                "query_weighted_mean": math.fsum(valid_scores) / len(valid_scores) if valid_scores else None,
                "macro_over_scorer_types_mean": (
                    math.fsum(type_means.values()) / len(type_means) if type_means else None
                ),
            }
        )
        for scorer in SCORER_TYPES:
            summary[f"{scorer}_mean"] = type_means.get(scorer)
            summary[f"{scorer}_valid_count"] = len(by_type.get(scorer, []))
        summary["invalid_reason_counts"] = json.dumps(dict(sorted(invalid_reasons.items())), sort_keys=True)
        summaries.append(summary)
    return summaries


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize SV2 query scores from analysis query-score JSONL files.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--group-by", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = (row for path in args.inputs for row in _read_jsonl(path))
    summaries = summarize_query_scores(rows, group_fields=args.group_by)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        if summaries:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    print(f"wrote {len(summaries)} rows to {args.output}")


if __name__ == "__main__":
    main()
