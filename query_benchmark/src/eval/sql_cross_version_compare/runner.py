"""Cross-version ranking comparison across v2/v3/v4 SQL analysis runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from src.eval.common import (
    DEFAULT_SQL_SOURCE_VERSION,
    PROVENANCE_CONTRACT_VERSION,
    make_task_run_dir,
    normalize_sql_source_version,
    now_run_tag,
    read_json,
    sql_source_family,
    sql_source_label,
    sql_source_line_version,
    write_csv,
    write_json,
)
from src.eval.final_outputs import normalize_standard_model_id, task_final_root
from src.evaluation.rank_stability import (
    _kendall_tau,
    _pairwise_reversal_ratio,
    _rank_map,
    _rank_models,
    _spearman_rho,
    _topk_overlap,
)

TASK_NAME = "sql_cross_version_compare"


def _dataset_sort_key(dataset_id: str) -> tuple[str, int, str]:
    text = str(dataset_id or "").strip().lower()
    prefix = text[:1]
    suffix = text[1:]
    try:
        numeric = int(suffix)
    except Exception:
        numeric = 10**9
    return (prefix, numeric, text)


def _normalize_compare_label(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        raise ValueError("Empty comparison label is not allowed.")
    try:
        return normalize_sql_source_version(text)
    except Exception:
        return text


def _maybe_normalize_version(label: str) -> str | None:
    try:
        return normalize_sql_source_version(label)
    except Exception:
        return None


def _load_analysis_rows(analysis_run_dir: Path) -> list[dict[str, Any]]:
    path = analysis_run_dir / "summaries" / "analysis_query_scores__all_datasets.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis query summary: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _resolve_analysis_sql_source_metadata(analysis_run_dir: Path, query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = read_json(analysis_run_dir / "manifest.json", {}) or {}
    sample_row = query_rows[0] if query_rows else {}
    version = str(manifest.get("sql_source_version") or sample_row.get("sql_source_version") or "")
    if not version and query_rows:
        version = DEFAULT_SQL_SOURCE_VERSION
    normalized = normalize_sql_source_version(version or DEFAULT_SQL_SOURCE_VERSION)
    return {
        "comparison_label": "",
        "run_tag": str(manifest.get("run_tag") or analysis_run_dir.name),
        "analysis_run_dir": str(analysis_run_dir.resolve()),
        "provenance_contract_version": str(
            manifest.get("provenance_contract_version")
            or sample_row.get("provenance_contract_version")
            or PROVENANCE_CONTRACT_VERSION
        ),
        "sql_source_family": str(
            manifest.get("sql_source_family")
            or sample_row.get("sql_source_family")
            or sql_source_family(normalized)
        ),
        "sql_source_line_version": str(
            manifest.get("sql_source_line_version")
            or sample_row.get("sql_source_line_version")
            or sql_source_line_version(normalized)
        ),
        "sql_source_version": normalized,
        "sql_source_label": str(manifest.get("sql_source_label") or sample_row.get("sql_source_label") or sql_source_label(normalized)),
        "sql_source_root": str(manifest.get("sql_source_root") or sample_row.get("sql_source_root") or ""),
        "dataset_count": int(manifest.get("dataset_count") or 0),
        "asset_count": int(manifest.get("asset_count") or 0),
    }


def _version_pair(left: str, right: str) -> str:
    return f"{left}_vs_{right}"


def _group_rows_by_dataset(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset_id = str(row.get("dataset_id") or "").strip()
        if dataset_id:
            out[dataset_id].append(row)
    return out


def _score_from_row(row: dict[str, Any]) -> float | None:
    value = row.get("query_score")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _ranking_for_rows(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("synthetic_exec_ok") is False:
            continue
        model_id = normalize_standard_model_id(row.get("model_id"))
        score = _score_from_row(row)
        if not model_id or score is None:
            continue
        scores[model_id].append(score)
    averaged = {model_id: mean(values) for model_id, values in scores.items() if values}
    return _rank_models(averaged)


def _compare_rankings(reference: list[tuple[str, float]], candidate: list[tuple[str, float]], top_k: int) -> dict[str, Any] | None:
    ref_scores = {model: score for model, score in reference}
    cand_scores = {model: score for model, score in candidate}
    common_models = sorted(set(ref_scores.keys()) & set(cand_scores.keys()))
    if len(common_models) < 2:
        return None
    ref_common = {model: ref_scores[model] for model in common_models}
    cand_common = {model: cand_scores[model] for model in common_models}
    ref_ranked = _rank_models(ref_common)
    cand_ranked = _rank_models(cand_common)
    ref_order = [model for model, _ in ref_ranked]
    cand_order = [model for model, _ in cand_ranked]
    reversal_ratio, _ = _pairwise_reversal_ratio(ref_order, cand_order)
    return {
        "common_model_count": len(common_models),
        "kendall_tau": round(_kendall_tau(ref_order, cand_order), 6),
        "spearman_rho": round(_spearman_rho(_rank_map(ref_common), _rank_map(cand_common)), 6),
        "champion_same": bool(ref_order and cand_order and ref_order[0] == cand_order[0]),
        "top_k_overlap": round(_topk_overlap(ref_order, cand_order, top_k), 6),
        "pairwise_reversal_ratio": round(reversal_ratio, 6),
        "reference_top_model": ref_order[0] if ref_order else "",
        "candidate_top_model": cand_order[0] if cand_order else "",
    }


def _stable_query_key(row: dict[str, Any]) -> str:
    for key in ("query_identity_stable_key", "stable_query_id", "query_id", "question_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _build_version_dataset_summary(
    *,
    version: str,
    rows_by_dataset: dict[str, list[dict[str, Any]]],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset_id, rows in sorted(rows_by_dataset.items(), key=lambda item: _dataset_sort_key(item[0])):
        stable_query_keys = {_stable_query_key(row) for row in rows if _stable_query_key(row)}
        model_ids = {
            normalize_standard_model_id(row.get("model_id"))
            for row in rows
            if normalize_standard_model_id(row.get("model_id"))
        }
        families = {str(row.get("family_id") or "").strip() for row in rows if row.get("family_id")}
        templates = {str(row.get("template_id") or "").strip() for row in rows if row.get("template_id")}
        output.append(
            {
                "comparison_label": version,
                "sql_source_version": meta.get("sql_source_version"),
                "sql_source_label": meta.get("sql_source_label"),
                "run_tag": meta.get("run_tag"),
                "dataset_id": dataset_id,
                "row_count": len(rows),
                "shared_query_key_count": len(stable_query_keys),
                "family_count": len(families),
                "template_count": len(templates),
                "model_count": len(model_ids),
            }
        )
    return output


def _aggregate_metric_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(field) or "") for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        payload = {field: value for field, value in zip(group_fields, key)}
        payload.update(
            {
                "comparison_count": len(items),
                "shared_query_count": len({str(item.get("stable_query_key") or "") for item in items if item.get("stable_query_key")}),
                "avg_kendall_tau": round(mean(float(item["kendall_tau"]) for item in items), 6),
                "avg_spearman_rho": round(mean(float(item["spearman_rho"]) for item in items), 6),
                "champion_retention_rate": round(
                    mean(1.0 if bool(item["champion_same"]) else 0.0 for item in items),
                    6,
                ),
                "avg_top_k_overlap": round(mean(float(item["top_k_overlap"]) for item in items), 6),
                "avg_pairwise_reversal_ratio": round(
                    mean(float(item["pairwise_reversal_ratio"]) for item in items),
                    6,
                ),
            }
        )
        payload["rank_stability_score"] = round(
            (
                payload["avg_kendall_tau"]
                + payload["avg_spearman_rho"]
                + payload["champion_retention_rate"]
                + payload["avg_top_k_overlap"]
                + (1.0 - payload["avg_pairwise_reversal_ratio"])
            )
            / 5.0,
            6,
        )
        output.append(payload)
    return output


def _build_cross_version_report(
    *,
    manifest: dict[str, Any],
    overall_rows: list[dict[str, Any]],
    shared_dataset_rows: list[dict[str, Any]],
) -> str:
    version_lines = [
        f"- `{version}`: `{meta['sql_source_label']}` from `{meta['analysis_run_dir']}`"
        for version, meta in sorted((manifest.get("versions") or {}).items())
    ]
    overall_lines = [
        f"- `{row['version_pair']}` / `{row['dataset_id']}`: kendall={row['kendall_tau']}, spearman={row['spearman_rho']}, top_k={row['top_k_overlap']}"
        for row in overall_rows[:18]
    ]
    shared_lines = [
        f"- `{row['version_pair']}` / `{row['dataset_id']}`: shared_queries={row['shared_query_count']}, rank_stability_score={row['rank_stability_score']}"
        for row in shared_dataset_rows[:18]
    ]
    return "\n".join(
        [
            "# SQL Cross-Version Comparison",
            "",
            "Input analysis runs:",
            "",
            *(version_lines or ["- none"]),
            "",
            f"- common_dataset_count: `{manifest.get('common_dataset_count')}`",
            f"- top_k: `{manifest.get('top_k')}`",
            "",
            "## Overall ranking comparison (sample)",
            "",
            *(overall_lines or ["- none"]),
            "",
            "## Shared-query comparison (sample)",
            "",
            *(shared_lines or ["- none"]),
            "",
        ]
    )


def run_sql_cross_version_compare(
    *,
    run_tag: str,
    analysis_runs: dict[str, Path],
    top_k: int = 3,
    publish_final: bool = True,
) -> dict[str, Any]:
    if len(analysis_runs) < 2:
        raise ValueError("Need at least two analysis runs for cross-version comparison.")

    normalized_runs = {
        _normalize_compare_label(version): Path(path)
        for version, path in analysis_runs.items()
    }
    run_dir = make_task_run_dir(TASK_NAME, run_tag)

    version_rows: dict[str, list[dict[str, Any]]] = {}
    version_meta: dict[str, dict[str, Any]] = {}
    rows_by_dataset_and_version: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    coverage_rows: list[dict[str, Any]] = []
    for version, analysis_run_dir in sorted(normalized_runs.items()):
        rows = _load_analysis_rows(analysis_run_dir)
        meta = _resolve_analysis_sql_source_metadata(analysis_run_dir, rows)
        meta["comparison_label"] = version
        requested_version = _maybe_normalize_version(version)
        if requested_version and meta.get("sql_source_version") != requested_version:
            raise ValueError(
                f"Analysis run provenance mismatch: requested={requested_version}, actual={meta.get('sql_source_version')}, run_dir={analysis_run_dir.resolve()}"
            )
        version_rows[version] = rows
        version_meta[version] = meta
        grouped = _group_rows_by_dataset(rows)
        for dataset_id, dataset_rows in grouped.items():
            rows_by_dataset_and_version[dataset_id][version] = dataset_rows
        coverage_rows.extend(_build_version_dataset_summary(version=version, rows_by_dataset=grouped, meta=meta))

    common_datasets = sorted(
        [
            dataset_id
            for dataset_id, version_map in rows_by_dataset_and_version.items()
            if len(version_map) >= 2
        ],
        key=_dataset_sort_key,
    )

    overall_rows: list[dict[str, Any]] = []
    shared_query_rows: list[dict[str, Any]] = []
    for dataset_id in common_datasets:
        version_map = rows_by_dataset_and_version[dataset_id]
        available_versions = sorted(version_map.keys())
        for left_version, right_version in combinations(available_versions, 2):
            pair_id = _version_pair(left_version, right_version)
            left_rows = version_map[left_version]
            right_rows = version_map[right_version]
            overall_metrics = _compare_rankings(
                _ranking_for_rows(left_rows),
                _ranking_for_rows(right_rows),
                top_k,
            )
            if overall_metrics is not None:
                overall_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "version_pair": pair_id,
                        "left_version": left_version,
                        "left_sql_source_version": version_meta[left_version].get("sql_source_version"),
                        "left_label": version_meta[left_version].get("sql_source_label"),
                        "left_run_tag": version_meta[left_version].get("run_tag"),
                        "right_version": right_version,
                        "right_sql_source_version": version_meta[right_version].get("sql_source_version"),
                        "right_label": version_meta[right_version].get("sql_source_label"),
                        "right_run_tag": version_meta[right_version].get("run_tag"),
                        **overall_metrics,
                    }
                )

            left_query_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
            right_query_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in left_rows:
                key = _stable_query_key(row)
                if key:
                    left_query_map[key].append(row)
            for row in right_rows:
                key = _stable_query_key(row)
                if key:
                    right_query_map[key].append(row)

            shared_keys = sorted(set(left_query_map.keys()) & set(right_query_map.keys()))
            for stable_key in shared_keys:
                metrics = _compare_rankings(
                    _ranking_for_rows(left_query_map[stable_key]),
                    _ranking_for_rows(right_query_map[stable_key]),
                    top_k,
                )
                if metrics is None:
                    continue
                left_sample = left_query_map[stable_key][0]
                right_sample = right_query_map[stable_key][0]
                shared_query_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "version_pair": pair_id,
                        "left_version": left_version,
                        "left_sql_source_version": version_meta[left_version].get("sql_source_version"),
                        "left_run_tag": version_meta[left_version].get("run_tag"),
                        "right_version": right_version,
                        "right_sql_source_version": version_meta[right_version].get("sql_source_version"),
                        "right_run_tag": version_meta[right_version].get("run_tag"),
                        "stable_query_key": stable_key,
                        "family_id": str(left_sample.get("family_id") or right_sample.get("family_id") or ""),
                        "template_id": str(left_sample.get("template_id") or right_sample.get("template_id") or ""),
                        "left_query_id": str(left_sample.get("query_id") or ""),
                        "right_query_id": str(right_sample.get("query_id") or ""),
                        **metrics,
                    }
                )

    shared_dataset_rows = _aggregate_metric_rows(shared_query_rows, ["version_pair", "dataset_id", "left_version", "right_version"])
    shared_family_rows = _aggregate_metric_rows(
        shared_query_rows,
        ["version_pair", "dataset_id", "family_id", "left_version", "right_version"],
    )
    shared_template_rows = _aggregate_metric_rows(
        shared_query_rows,
        ["version_pair", "dataset_id", "template_id", "left_version", "right_version"],
    )
    pair_summary_rows = _aggregate_metric_rows(shared_query_rows, ["version_pair", "left_version", "right_version"])

    write_csv(run_dir / "summaries" / "sql_version_coverage__all_datasets.csv", coverage_rows)
    write_csv(run_dir / "summaries" / "sql_version_pair_overall_by_dataset.csv", overall_rows)
    write_csv(run_dir / "summaries" / "sql_version_pair_shared_query_by_query.csv", shared_query_rows)
    write_csv(run_dir / "summaries" / "sql_version_pair_shared_query_by_dataset.csv", shared_dataset_rows)
    write_csv(run_dir / "summaries" / "sql_version_pair_shared_query_by_family.csv", shared_family_rows)
    write_csv(run_dir / "summaries" / "sql_version_pair_shared_query_by_template.csv", shared_template_rows)
    write_csv(run_dir / "summaries" / "sql_version_pair_summary.csv", pair_summary_rows)

    manifest = {
        "task": TASK_NAME,
        "run_tag": run_tag,
        "top_k": top_k,
        "version_count": len(version_meta),
        "common_dataset_count": len(common_datasets),
        "common_datasets": common_datasets,
        "versions": version_meta,
        "overall_dataset_comparison_count": len(overall_rows),
        "shared_query_comparison_count": len(shared_query_rows),
    }

    final_manifest: dict[str, Any] | None = None
    if publish_final:
        final_dir = task_final_root(TASK_NAME)
        final_dir.mkdir(parents=True, exist_ok=True)
        report_path = final_dir / "sql_cross_version_compare_summary.md"
        report_path.write_text(
            _build_cross_version_report(
                manifest=manifest,
                overall_rows=overall_rows,
                shared_dataset_rows=shared_dataset_rows,
            ),
            encoding="utf-8",
        )
        for name in [
            "sql_version_coverage__all_datasets.csv",
            "sql_version_pair_overall_by_dataset.csv",
            "sql_version_pair_shared_query_by_dataset.csv",
            "sql_version_pair_shared_query_by_family.csv",
            "sql_version_pair_shared_query_by_template.csv",
            "sql_version_pair_summary.csv",
        ]:
            src = run_dir / "summaries" / name
            if src.exists():
                (final_dir / name).write_bytes(src.read_bytes())
        final_manifest = {
            "task": TASK_NAME,
            "run_dir": str(run_dir.resolve()),
            "final_dir": str(final_dir.resolve()),
            "summary_note": str(report_path.resolve()),
        }
        write_json(final_dir / "sql_cross_version_compare_final_manifest.json", final_manifest)
        manifest["final_outputs"] = final_manifest
    else:
        manifest["final_outputs"] = None

    write_json(run_dir / "manifest.json", manifest)
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "coverage_rows": coverage_rows,
        "overall_rows": overall_rows,
        "shared_dataset_rows": shared_dataset_rows,
        "pair_summary_rows": pair_summary_rows,
    }


def _parse_analysis_runs(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        version, sep, path_text = item.partition("=")
        version = _normalize_compare_label(version.strip())
        if not sep or not path_text.strip():
            raise ValueError(
                f"Invalid --analysis-run value: {item!r}. Expected format like v3=Evaluation/analysis/runs/rankstab_v3_full_20260506 or vr=/path/to/custom/run"
            )
        out[version] = Path(path_text.strip())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SQL analysis runs across versions (v2/v3/v4).")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag.")
    parser.add_argument(
        "--analysis-run",
        action="append",
        default=[],
        help="Versioned analysis run in the form v2=/path/to/run. Repeat for v3/v4.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k overlap cutoff.")
    parser.add_argument("--skip-final-publish", action="store_true", help="Skip writing shared final outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_runs = _parse_analysis_runs(list(args.analysis_run or []))
    result = run_sql_cross_version_compare(
        run_tag=args.run_tag or now_run_tag(),
        analysis_runs=analysis_runs,
        top_k=max(1, int(args.top_k)),
        publish_final=not args.skip_final_publish,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
