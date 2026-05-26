from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TEXTUAL_MARKERS = {
    "identifier_string",
    "dna_sequence",
    "categorical",
    "boolean",
    "text",
    "free_text",
    "nominal",
    "identifier",
}
MISSING_TEXT = {"", "null", "none", "nan", "na", "n/a", "<null>"}
MODEL_DECODE_FIRST = {"tabsyn"}
MODEL_DECODE_SORTED = {"tabbyflow", "tabdiff", "forestdiffusion", "tabpfgen", "tabddpm"}
MODEL_DECODE_SOFT_INTEGER_TOLERANCE = {"forestdiffusion": 0.25}


@dataclass
class AssetRecord:
    section: str
    dataset_id: str
    model_id: str
    run_id: str
    run_dir: Path
    root_name: str


def _is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in MISSING_TEXT


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.map(_is_missing)


def _numeric_like_ratio(series: pd.Series) -> float:
    non_missing = series[~_missing_mask(series)]
    if non_missing.empty:
        return 0.0
    parsed = pd.to_numeric(non_missing.astype(str), errors="coerce")
    return float(parsed.notna().mean())


def _integer_like_ratio(series: pd.Series, tolerance: float = 1e-8) -> float:
    non_missing = series[~_missing_mask(series)]
    if non_missing.empty:
        return 0.0
    parsed = pd.to_numeric(non_missing.astype(str), errors="coerce").dropna()
    if parsed.empty:
        return 0.0
    return float(((parsed - parsed.round()).abs() <= tolerance).mean())


def _read_csv_with_sniff(path: Path) -> pd.DataFrame:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:32768]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except Exception:
        sep = ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _load_field_meta(metadata_root: Path, dataset_id: str) -> dict[str, dict[str, Any]]:
    path = metadata_root / dataset_id / "metadata_core" / "field_registry.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for field in payload.get("fields", []):
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        out[name] = {
            "semantic_type": str(field.get("semantic_type") or "").strip().lower(),
            "declared_type": str(field.get("declared_type") or "").strip().lower(),
            "value_order": [str(item) for item in field.get("value_order") or []],
        }
    return out


def _bucket_for_column(meta: dict[str, Any], real_series: pd.Series) -> str:
    semantic = str(meta.get("semantic_type") or "").lower()
    declared = str(meta.get("declared_type") or "").lower()
    raw_numeric_ratio = _numeric_like_ratio(real_series)
    raw_integer_ratio = _integer_like_ratio(real_series)
    if any(marker in semantic for marker in TEXTUAL_MARKERS):
        if raw_numeric_ratio < 0.95:
            return "textual_categorical"
        if raw_integer_ratio >= 0.95 or meta.get("value_order"):
            return "discrete_numeric"
        return "continuous_numeric"
    if "numeric_discrete" in semantic or "identifier_numeric" in semantic:
        if raw_numeric_ratio < 0.95:
            return "textual_categorical"
        return "discrete_numeric" if raw_integer_ratio >= 0.95 or meta.get("value_order") else "continuous_numeric"
    if "numeric" in semantic or declared in {"numeric", "integer", "float", "double", "decimal"}:
        if raw_numeric_ratio >= 0.95:
            return "discrete_numeric" if raw_integer_ratio >= 0.95 else "continuous_numeric"
        return "textual_categorical"
    if raw_numeric_ratio < 0.95:
        return "textual_categorical"
    if raw_integer_ratio >= 0.95:
        return "discrete_numeric"
    return "continuous_numeric"


def _tabsyn_order(real_series: pd.Series) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for value in real_series.tolist():
        token = "<NA>" if _is_missing(value) else str(value)
        if token not in seen:
            seen.add(token)
            order.append(token)
    return order


def _sorted_order(real_series: pd.Series) -> list[str]:
    values = {str(value) for value in real_series[~_missing_mask(real_series)].tolist()}
    return sorted(values, key=str)


def _build_dataset_context(dataset_id: str, real_data_root: Path, metadata_root: Path) -> dict[str, Any]:
    real_path = real_data_root / dataset_id / f"{dataset_id}-train.csv"
    real_df = _read_csv_with_sniff(real_path)
    field_meta = _load_field_meta(metadata_root, dataset_id)
    columns: dict[str, dict[str, Any]] = {}
    for column in real_df.columns:
        real_series = real_df[column]
        meta = field_meta.get(column, {})
        bucket = _bucket_for_column(meta, real_series)
        info: dict[str, Any] = {
            "meta": meta,
            "bucket": bucket,
            "real_missing_rate": float(_missing_mask(real_series).mean()),
        }
        if bucket == "textual_categorical":
            missing_mask = _missing_mask(real_series)
            info["real_text_set"] = {str(v) for v in real_series[~missing_mask].tolist()}
            info["tabsyn_order"] = _tabsyn_order(real_series)
            info["sorted_order"] = _sorted_order(real_series)
            info["real_numeric_ratio"] = _numeric_like_ratio(real_series)
        columns[column] = info
    return {"dataset_id": dataset_id, "real_df": real_df, "columns": columns}


def _audit_textual(column_info: dict[str, Any], syn_series: pd.Series) -> dict[str, Any]:
    syn_missing_mask = _missing_mask(syn_series)
    syn_non_missing = syn_series[~syn_missing_mask]
    if syn_non_missing.empty:
        return {"status": "all_missing", "note": "synthetic_all_missing", "overlap": None, "numeric_ratio": 0.0}
    syn_numeric_ratio = _numeric_like_ratio(syn_series)
    real_numeric_ratio = float(column_info.get("real_numeric_ratio", 0.0))
    syn_text_set = {str(v) for v in syn_non_missing.tolist()}
    if syn_text_set:
        overlap = len(column_info.get("real_text_set", set()) & syn_text_set) / len(syn_text_set)
    else:
        overlap = None
    if real_numeric_ratio < 0.95 and syn_numeric_ratio > 0.95 and (overlap or 0.0) == 0.0:
        return {
            "status": "suspected_not_decoded",
            "note": "real_textual_syn_numeric_no_overlap",
            "overlap": overlap,
            "numeric_ratio": syn_numeric_ratio,
        }
    if overlap is None:
        return {"status": "unknown", "note": "no_non_missing_values", "overlap": overlap, "numeric_ratio": syn_numeric_ratio}
    if overlap >= 0.95:
        status = "verified_direct_compare"
    elif overlap >= 0.50:
        status = "likely_ok_direct_compare"
    else:
        status = "has_overlap_gap"
    return {"status": status, "note": f"overlap={overlap:.3f}", "overlap": overlap, "numeric_ratio": syn_numeric_ratio}


def _decode_textual_column(model_id: str, column_info: dict[str, Any], syn_series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    syn_missing_mask = _missing_mask(syn_series)
    syn_non_missing = syn_series[~syn_missing_mask]
    parsed = pd.to_numeric(syn_non_missing.astype(str), errors="coerce")
    integer_tolerance = float(MODEL_DECODE_SOFT_INTEGER_TOLERANCE.get(model_id, 1e-8))
    if (
        syn_non_missing.empty
        or parsed.notna().mean() < 0.95
        or ((parsed.dropna() - parsed.dropna().round()).abs() <= integer_tolerance).mean() < 0.95
    ):
        return syn_series, {"changed_cells": 0, "restored_missing": 0, "decoded_column": False, "strategy": ""}
    if model_id in MODEL_DECODE_FIRST:
        order = column_info["tabsyn_order"]
        strategy = "tabsyn_first_seen"
        allow_out_of_range_missing = False
    elif model_id in MODEL_DECODE_SORTED:
        order = column_info["sorted_order"]
        strategy = "sorted_unique"
        allow_out_of_range_missing = True
    else:
        return syn_series, {"changed_cells": 0, "restored_missing": 0, "decoded_column": False, "strategy": ""}
    if not order:
        return syn_series, {"changed_cells": 0, "restored_missing": 0, "decoded_column": False, "strategy": strategy}

    parsed_all = pd.to_numeric(syn_series, errors="coerce")
    int_mask = parsed_all.notna() & ((parsed_all - parsed_all.round()).abs() <= integer_tolerance)
    indices = parsed_all.round().astype("Int64")
    in_range_mask = int_mask & indices.ge(0) & indices.lt(len(order))
    modify_mask = in_range_mask.copy()
    updated = syn_series.astype("object").copy()
    if in_range_mask.any():
        mapping = {idx: (pd.NA if token == "<NA>" else token) for idx, token in enumerate(order)}
        updated.loc[in_range_mask] = indices.loc[in_range_mask].map(mapping).astype("object")
    restored_missing = int((in_range_mask & updated.isna()).sum())
    if allow_out_of_range_missing:
        out_of_range_mask = int_mask & ~in_range_mask
        if out_of_range_mask.any():
            updated.loc[out_of_range_mask] = pd.NA
            restored_missing += int(out_of_range_mask.sum())
            modify_mask = modify_mask | out_of_range_mask
    changed_cells = int(modify_mask.sum())
    return updated, {
        "changed_cells": changed_cells,
        "restored_missing": restored_missing,
        "decoded_column": changed_cells > 0 or restored_missing > 0,
        "strategy": strategy,
    }


def _enumerate_assets(root: Path) -> list[AssetRecord]:
    out: list[AssetRecord] = []
    if not root.exists():
        return out
    if root.name == "main":
        for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            for model_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
                for run_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                    out.append(
                        AssetRecord(
                            section="main",
                            dataset_id=dataset_dir.name,
                            model_id=model_dir.name.lower(),
                            run_id=run_dir.name,
                            run_dir=run_dir,
                            root_name=root.name,
                        )
                    )
    else:
        for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            for model_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
                runs_dir = model_dir / "runs"
                if not runs_dir.exists():
                    continue
                for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
                    out.append(
                        AssetRecord(
                            section="hyper_parameter_tuning",
                            dataset_id=dataset_dir.name,
                            model_id=model_dir.name.lower(),
                            run_id=run_dir.name,
                            run_dir=run_dir,
                            root_name=root.name,
                        )
                    )
    return out


def _choose_csv(run_dir: Path, dataset_id: str, model_id: str, real_row_count: int) -> Path | None:
    candidates = [p for p in run_dir.glob("*.csv") if p.is_file()]
    if not candidates:
        candidates = [p for p in run_dir.rglob("synthetic_data/*.csv") if p.is_file()]
    if not candidates:
        return None
    scored: list[tuple[int, int, str, Path]] = []
    model_prefixes = [f"{model_id}-{dataset_id}-", f"rtf-{dataset_id}-", f"forest-{dataset_id}-"]
    for path in candidates:
        match = re.search(rf"^(?:{re.escape(model_id)}|rtf|forest)-{re.escape(dataset_id)}-(\\d+)-", path.name)
        if match:
            row_count = int(match.group(1))
        else:
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    row_count = max(sum(1 for _ in handle) - 1, 0)
            except Exception:
                row_count = -1
        prefix_match = int(any(path.name.startswith(prefix) for prefix in model_prefixes))
        exact_rows = int(row_count == real_row_count)
        scored.append((exact_rows, prefix_match, row_count, path.name, path))
    scored.sort(reverse=True)
    return scored[0][-1]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-root", default="/data/jialinzhang/SyntheData0523/main")
    ap.add_argument("--hyper-root", default="/data/jialinzhang/SyntheData0523/hyper_parameter_tuning")
    ap.add_argument("--real-data-root", default="/data/jialinzhang/Datasets/tabular_datasets")
    ap.add_argument("--metadata-root", default="/data/jialinzhang/TabQueryBench/code_snapshot/data")
    ap.add_argument("--output-dir", default="/data/jialinzhang/TabQueryBench/tmp/encoding_space_repair_20260524")
    ap.add_argument("--filter-manifest", default="")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    main_root = Path(args.main_root)
    hyper_root = Path(args.hyper_root)
    real_data_root = Path(args.real_data_root)
    metadata_root = Path(args.metadata_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = _enumerate_assets(main_root)
    if hyper_root.exists():
        assets.extend(_enumerate_assets(hyper_root))
    if args.filter_manifest:
        manifest_df = pd.read_csv(args.filter_manifest)
        wanted = {
            (str(row["dataset_id"]), str(row["model_id"]).lower(), str(row["run_id"]))
            for _, row in manifest_df.iterrows()
        }
        assets = [asset for asset in assets if (asset.dataset_id, asset.model_id, asset.run_id) in wanted]
    assets.sort(key=lambda a: (a.dataset_id, a.model_id, a.run_id))

    dataset_cache: dict[str, dict[str, Any]] = {}
    asset_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    repaired_asset_count = 0
    repaired_column_count = 0

    for idx, asset in enumerate(assets, start=1):
        if asset.dataset_id not in dataset_cache:
            dataset_cache[asset.dataset_id] = _build_dataset_context(asset.dataset_id, real_data_root, metadata_root)
        dataset_context = dataset_cache[asset.dataset_id]
        real_df = dataset_context["real_df"]
        csv_path = _choose_csv(asset.run_dir, asset.dataset_id, asset.model_id, len(real_df))
        if csv_path is None:
            asset_rows.append(
                {
                    "dataset_id": asset.dataset_id,
                    "model_id": asset.model_id,
                    "run_id": asset.run_id,
                    "section": asset.section,
                    "csv_path": "",
                    "candidate_columns": 0,
                    "repaired_columns": 0,
                    "actions": "missing_csv",
                    "modified": False,
                }
            )
            continue
        syn_df = _read_csv_with_sniff(csv_path)
        asset_candidate_count = 0
        asset_repaired_columns = 0
        actions: list[str] = []

        for column in real_df.columns:
            if column not in syn_df.columns:
                continue
            column_info = dataset_context["columns"].get(column, {})
            if column_info.get("bucket") != "textual_categorical":
                continue
            before = _audit_textual(column_info, syn_df[column])
            if before["status"] != "suspected_not_decoded":
                continue
            asset_candidate_count += 1
            updated, repair_info = _decode_textual_column(asset.model_id, column_info, syn_df[column])
            after = _audit_textual(column_info, updated)
            fixable = bool(repair_info["decoded_column"]) and (after["overlap"] or 0.0) > (before["overlap"] or 0.0)
            applied = False
            if args.apply and fixable:
                syn_df[column] = updated
                applied = True
                asset_repaired_columns += 1
                repaired_column_count += 1
                actions.append(f"{column}:{repair_info['strategy']}")
            column_rows.append(
                {
                    "dataset_id": asset.dataset_id,
                    "model_id": asset.model_id,
                    "run_id": asset.run_id,
                    "section": asset.section,
                    "csv_path": str(csv_path),
                    "column": column,
                    "before_status": before["status"],
                    "before_note": before["note"],
                    "before_overlap": before["overlap"],
                    "before_numeric_ratio": before["numeric_ratio"],
                    "strategy": repair_info["strategy"],
                    "changed_cells": repair_info["changed_cells"],
                    "restored_missing": repair_info["restored_missing"],
                    "after_status": after["status"],
                    "after_note": after["note"],
                    "after_overlap": after["overlap"],
                    "after_numeric_ratio": after["numeric_ratio"],
                    "fixable": fixable,
                    "applied": applied,
                }
            )

        if args.apply and asset_repaired_columns > 0:
            syn_df.to_csv(csv_path, index=False)
            repaired_asset_count += 1

        asset_rows.append(
            {
                "dataset_id": asset.dataset_id,
                "model_id": asset.model_id,
                "run_id": asset.run_id,
                "section": asset.section,
                "csv_path": str(csv_path),
                "candidate_columns": asset_candidate_count,
                "repaired_columns": asset_repaired_columns,
                "actions": " | ".join(actions),
                "modified": asset_repaired_columns > 0,
            }
        )
        if idx % 50 == 0:
            print(f"[encoding-repair] scanned {idx}/{len(assets)} assets; repaired_assets={repaired_asset_count}; repaired_columns={repaired_column_count}")

    candidate_rows = [row for row in column_rows if row["before_status"] == "suspected_not_decoded"]
    fixable_rows = [row for row in candidate_rows if row["fixable"]]
    applied_rows = [row for row in candidate_rows if row["applied"]]
    remaining_rows = [row for row in candidate_rows if not row["applied"]]

    _write_csv(output_dir / "asset_summary.csv", asset_rows)
    _write_csv(output_dir / "column_candidates.csv", candidate_rows)
    _write_csv(output_dir / "fixable_candidates.csv", fixable_rows)
    _write_csv(output_dir / "applied_repairs.csv", applied_rows)
    _write_csv(output_dir / "remaining_unapplied_candidates.csv", remaining_rows)

    by_model = Counter((row["model_id"] for row in candidate_rows))
    by_model_fixable = Counter((row["model_id"] for row in fixable_rows))
    by_dataset = Counter((row["dataset_id"] for row in candidate_rows))
    summary_rows = []
    for model_id in sorted(set(by_model) | set(by_model_fixable)):
        summary_rows.append(
            {
                "model_id": model_id,
                "candidate_columns": by_model.get(model_id, 0),
                "fixable_columns": by_model_fixable.get(model_id, 0),
            }
        )
    _write_csv(output_dir / "summary_by_model.csv", summary_rows)
    _write_csv(
        output_dir / "summary_by_dataset.csv",
        [{"dataset_id": dataset_id, "candidate_columns": count} for dataset_id, count in sorted(by_dataset.items())],
    )

    report_lines = [
        "# Server Encoding-Space Repair Audit",
        "",
        f"- scanned_assets: {len(assets)}",
        f"- candidate_columns: {len(candidate_rows)}",
        f"- fixable_columns: {len(fixable_rows)}",
        f"- applied_columns: {len(applied_rows)}",
        f"- modified_assets: {repaired_asset_count}",
        "",
        "## Candidate Columns by Model",
    ]
    for row in summary_rows:
        report_lines.append(f"- `{row['model_id']}`: candidates={row['candidate_columns']}, fixable={row['fixable_columns']}")
    report_lines += [
        "",
        "## Output Files",
        "- `asset_summary.csv`",
        "- `column_candidates.csv`",
        "- `fixable_candidates.csv`",
        "- `applied_repairs.csv`",
        "- `remaining_unapplied_candidates.csv`",
        "- `summary_by_model.csv`",
        "- `summary_by_dataset.csv`",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
