#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_helper_module(path: Path):
    spec = importlib.util.spec_from_file_location("fix_decode_helper", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        sval = str(value).strip()
        if sval == "":
            return None
        out = float(sval)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


class SupportSpec:
    def __init__(self, dataset: str, column: str, raw_series: pd.Series, helper: Any):
        self.dataset = dataset
        self.column = column
        self.raw_series = raw_series
        self.helper = helper
        self.normalized = helper._normalize_cat_series(raw_series)
        uniq = list(pd.unique(self.normalized))
        self.support_all = sorted(str(v) for v in uniq)
        self.support_nonempty = [v for v in self.support_all if v != ""]
        self.empty_allowed = "" in self.support_all
        self.numeric_pairs: list[tuple[float, str]] = []
        numeric_ok = True
        for sval in self.support_nonempty:
            num = _safe_float(sval)
            if num is None:
                numeric_ok = False
                break
            self.numeric_pairs.append((float(num), sval))
        self.numeric_support = numeric_ok and bool(self.numeric_pairs)
        if self.numeric_support:
            self.numeric_pairs = sorted(self.numeric_pairs)
            self.numeric_values = np.asarray([k for k, _ in self.numeric_pairs], dtype=float)
            self.numeric_labels = [v for _, v in self.numeric_pairs]
        else:
            self.numeric_values = np.asarray([], dtype=float)
            self.numeric_labels = []


def _build_support_specs(rowdata_root: Path, datasets: list[str], helper: Any) -> dict[str, dict[str, SupportSpec]]:
    out: dict[str, dict[str, SupportSpec]] = {}
    for dataset in sorted(set(datasets)):
        train_csv = rowdata_root / dataset / f"{dataset}-train.csv"
        df = helper._load_df(train_csv)
        out[dataset] = {col: SupportSpec(dataset, col, df[col], helper) for col in df.columns}
    return out


def _repair_with_numeric_support(series: pd.Series, spec: SupportSpec) -> tuple[pd.Series, dict[str, Any]]:
    support_set = set(spec.support_all)
    out_values: list[str] = []
    changed = 0
    for value in series.tolist():
        sval = "" if pd.isna(value) else str(value)
        if sval == "nan":
            sval = ""
        if sval in support_set:
            out_values.append(sval)
            continue
        num = _safe_float(value)
        if num is None:
            repaired = "" if spec.empty_allowed else spec.numeric_labels[0]
        else:
            idx = int(np.argmin(np.abs(spec.numeric_values - float(num))))
            repaired = spec.numeric_labels[idx]
        out_values.append(repaired)
        if repaired != sval:
            changed += 1
    out = pd.Series(out_values, index=series.index, name=series.name)
    remaining = int((~out.isin(list(support_set))).sum())
    return out, {
        "strategy": "numeric_support_projection",
        "mapping_size": len(spec.numeric_labels),
        "mapping_sample": json.dumps({str(k): v for k, v in spec.numeric_pairs[:10]}, ensure_ascii=False),
        "changed_count": changed,
        "remaining_bad_count": remaining,
    }


def _repair_with_categorical_code_map(series: pd.Series, spec: SupportSpec) -> tuple[pd.Series, dict[str, Any]]:
    support_set = set(spec.support_all)
    decode_support = spec.support_nonempty[:]
    if not decode_support:
        decode_support = spec.support_all[:]
    if not decode_support:
        return series.copy(), {
            "strategy": "empty_support_noop",
            "mapping_size": 0,
            "changed_count": 0,
            "remaining_bad_count": None,
        }

    out_values: list[str] = []
    changed = 0
    for value in series.tolist():
        sval = "" if pd.isna(value) else str(value)
        if sval == "nan":
            sval = ""
        if sval in support_set:
            out_values.append(sval)
            continue
        num = _safe_float(value)
        if num is None:
            repaired = "" if spec.empty_allowed else decode_support[0]
        else:
            code = int(round(float(num)))
            if code < 0:
                code = 0
            if code >= len(decode_support):
                code = len(decode_support) - 1
            repaired = decode_support[code]
        out_values.append(repaired)
        if repaired != sval:
            changed += 1
    out = pd.Series(out_values, index=series.index, name=series.name)
    remaining = int((~out.isin(list(support_set))).sum())
    return out, {
        "strategy": "categorical_code_decode_with_clamp",
        "mapping_size": len(decode_support),
        "mapping_sample": json.dumps({str(i): v for i, v in enumerate(decode_support[:10])}, ensure_ascii=False),
        "changed_count": changed,
        "remaining_bad_count": remaining,
    }


def _choose_and_repair(series: pd.Series, spec: SupportSpec, pattern: str) -> tuple[pd.Series, dict[str, Any]]:
    if spec.numeric_support:
        return _repair_with_numeric_support(series, spec)
    return _repair_with_categorical_code_map(series, spec)


def _sync_6000_csv_to_hf(hf_repo: Path, dataset: str, model: str, run_id: str, final_csv: Path) -> Path:
    target_dir = hf_repo / "syntheticSuccess" / dataset / model / run_id
    if not target_dir.exists():
        return Path("")
    target_csv = target_dir / final_csv.name
    if not target_csv.exists():
        candidates = [p for p in target_dir.glob("*.csv") if p.parent == target_dir]
        if len(candidates) == 1:
            target_csv = candidates[0]
        else:
            raise FileNotFoundError(f"HF target csv missing for {dataset}/{model}/{run_id}: {final_csv.name}")
    target_csv.write_bytes(final_csv.read_bytes())
    return target_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-csv", required=True)
    ap.add_argument("--validation-csv", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--rowdata-root", default="/data/jialinzhang/TabQueryBench/RowData")
    ap.add_argument("--hf-repo-root", default="/data/jialinzhang/hf_repo_server_sync_20260518b")
    ap.add_argument("--helper-script", required=True)
    args = ap.parse_args()

    helper = _load_helper_module(Path(args.helper_script))
    plan = pd.read_csv(args.plan_csv)
    plan = plan[plan["fixability"].isin(["high", "medium"])].copy()
    val = pd.read_csv(args.validation_csv)
    val = val[["dataset", "model", "source", "run_id", "run_dir", "synthetic_csv", "bad_discrete_cols"]]
    targets = plan.merge(val, on=["dataset", "model", "source", "run_id"], how="left", validate="one_to_one")

    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, str, str]] = set()
    if manifest_out.exists():
        existing = pd.read_csv(manifest_out)
        if not existing.empty:
            results = existing.to_dict("records")
            done_keys = {
                (str(r["dataset"]), str(r["model"]), str(r["source"]), str(r["run_id"]))
                for _, r in existing.iterrows()
                if str(r.get("status", "")).startswith("repaired")
            }
            if done_keys:
                targets = targets[
                    ~targets.apply(
                        lambda r: (str(r["dataset"]), str(r["model"]), str(r["source"]), str(r["run_id"])) in done_keys,
                        axis=1,
                    )
                ].reset_index(drop=True)

    support_cache = _build_support_specs(Path(args.rowdata_root), targets["dataset"].tolist(), helper)
    hf_repo = Path(args.hf_repo_root)

    total = len(targets)
    for idx, (_, row) in enumerate(targets.iterrows(), start=1):
        dataset = str(row["dataset"])
        model = str(row["model"])
        source = str(row["source"])
        run_id = str(row["run_id"])
        run_dir = Path(str(row["run_dir"]))
        target_csv = Path(str(row["synthetic_csv"]))
        print(f"[postfix] {idx}/{total} source={source} dataset={dataset} model={model} run_id={run_id}", flush=True)
        record: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "source": source,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "target_csv": str(target_csv),
            "pattern": str(row["pattern"]),
            "fixability": str(row["fixability"]),
            "status": "pending",
        }
        try:
            if pd.isna(row.get("run_dir")) or pd.isna(row.get("synthetic_csv")):
                raise FileNotFoundError("missing run_dir or synthetic_csv after merge")
            syn_df = helper._load_df(target_csv)
            write_target, is_overlay = helper._prepare_output_target(run_dir, target_csv)
            col_reports: list[dict[str, Any]] = []
            repaired = False
            bad_cols = [c for c in str(row["bad_discrete_cols"]).split("|") if c]
            for col in bad_cols:
                if col not in syn_df.columns:
                    col_reports.append({"column": col, "status": "missing_in_synthetic_csv"})
                    continue
                spec = support_cache[dataset][col]
                fixed, info = _choose_and_repair(syn_df[col], spec, str(row["pattern"]))
                syn_df[col] = fixed
                info["column"] = col
                info["status"] = "ok" if info.get("remaining_bad_count") == 0 else "remaining_bad_values"
                col_reports.append(info)
                repaired = True

            if repaired:
                write_target.parent.mkdir(parents=True, exist_ok=True)
                syn_df.to_csv(write_target, index=False)
                final_csv = helper._finalize_overlay(run_dir, write_target) if is_overlay else write_target
            else:
                final_csv = write_target

            hf_target_csv = ""
            if source == "6000":
                hf_target = _sync_6000_csv_to_hf(hf_repo, dataset, model, run_id, final_csv)
                hf_target_csv = str(hf_target) if str(hf_target) else ""
            else:
                hf_target_csv = str(final_csv)

            remaining = sum(
                int(item.get("remaining_bad_count") or 0)
                for item in col_reports
                if item.get("remaining_bad_count") is not None
            )
            record["status"] = "repaired" if repaired and remaining == 0 else "repaired_with_remaining"
            record["column_reports_json"] = json.dumps(col_reports, ensure_ascii=False)
            record["remaining_bad_total"] = remaining
            record["final_csv_path"] = str(final_csv)
            record["hf_target_csv"] = hf_target_csv
        except Exception as e:
            record["status"] = "failed"
            record["error"] = f"{type(e).__name__}: {e}"
            print(f"[postfix][error] source={source} dataset={dataset} model={model} run_id={run_id} error={type(e).__name__}: {e}", flush=True)

        results.append(record)
        helper._write_results_checkpoint(results, manifest_out)
        print(f"[postfix][done] {idx}/{total} source={source} dataset={dataset} model={model} run_id={run_id} status={record['status']}", flush=True)

    out_df = pd.DataFrame(results)
    out_df.to_csv(manifest_out, index=False)
    summary = {
        "target_count": int(len(targets)),
        "repaired_count": int((out_df["status"] == "repaired").sum()) if not out_df.empty else 0,
        "repaired_with_remaining_count": int((out_df["status"] == "repaired_with_remaining").sum()) if not out_df.empty else 0,
        "failed_count": int((out_df["status"] == "failed").sum()) if not out_df.empty else 0,
    }
    manifest_out.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
