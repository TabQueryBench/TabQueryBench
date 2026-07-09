import csv
import json
from pathlib import Path


ROOT = Path.cwd()
MATRIX_CSV = ROOT / "tmp" / "generated_data_source_matrix_20260506.csv"
OUTPUT_JSON = ROOT / "tmp" / "th_final_candidates_audit_20260509.json"
OUTPUT_CSV = ROOT / "tmp" / "th_final_candidates_audit_20260509.csv"
OUTPUT_MD = ROOT / "tmp" / "th_final_candidates_audit_20260509.md"

TARGET_DATASETS = {"c2", "m4", "n3"}
TARGET_MODELS = {
    "arf",
    "bayesnet",
    "ctgan",
    "forestdiffusion",
    "realtabformer",
    "tabbyflow",
    "tabddpm",
    "tabdiff",
    "tabpfgen",
    "tabsyn",
    "tvae",
}

SOURCE_ROOTS = {
    "5": ROOT / "SynOutput-5090",
    "6": ROOT / "SynOutput",
    "R": ROOT / "remote-output-Benchmark-trainonly-v1",
    "H": ROOT / "hyperparameter" / "output-Benchmark-trainonly-v1",
}

CSV_SUFFIX_BLACKLIST = ("__real.csv", "__train.csv", "__val.csv", "__test.csv")
SOURCE_PREFERENCE = {"6": 0, "5": 1, "R": 2, "H": 3}


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def summarize_csv(real_csv: Path, synthetic_csv: Path):
    real_header, real_rows = read_csv_rows(real_csv)
    syn_header, syn_rows = read_csv_rows(synthetic_csv)
    duplicate_rows = len(syn_rows) - len({tuple(r) for r in syn_rows})
    constant_cols = 0
    for idx in range(len(syn_header)):
        values = {row[idx] for row in syn_rows if idx < len(row)}
        if len(values) <= 1:
            constant_cols += 1
    return {
        "rows": len(syn_rows),
        "cols": len(syn_header),
        "duplicate_rows": duplicate_rows,
        "constant_cols": constant_cols,
        "columns_match_real_exact": syn_header == real_header,
        "row_count_matches_real_train": len(syn_rows) == len(real_rows),
    }


def find_runtime_result(path: Path):
    direct = path / "runtime_result.json"
    if direct.exists():
        return direct
    meta = path / "meta" / "runtime_result.json"
    if meta.exists():
        return meta
    return None


def find_synthetic_csv(path: Path, runtime: dict):
    artifact_name = Path(runtime.get("artifacts", {}).get("synthetic_csv", "")).name
    if artifact_name:
        direct = path / artifact_name
        if direct.exists():
            return direct
        for p in path.rglob(artifact_name):
            if p.is_file():
                return p
    candidates = [
        p
        for p in path.rglob("*.csv")
        if p.is_file() and not any(p.name.endswith(s) for s in CSV_SUFFIX_BLACKLIST)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def consolidated_candidates(source: str, dataset: str, model: str):
    model_dir = SOURCE_ROOTS[source] / dataset / model
    if not model_dir.exists():
        return []
    meta_dir = model_dir / "metadata"
    synth_dir = model_dir / "synthetic_data"
    if not meta_dir.exists() or not synth_dir.exists():
        return []
    runtime_files = sorted(meta_dir.glob("*__runtime_result.json"))
    out = []
    for runtime_path in runtime_files:
        try:
            runtime = read_json(runtime_path)
        except Exception:
            continue
        if runtime.get("generate_status") != "success":
            continue
        synthetic_path = find_synthetic_csv(synth_dir, runtime)
        if synthetic_path is None:
            continue
        out.append(
            {
                "source": source,
                "run_id": runtime.get("run_id") or runtime_path.stem,
                "run_ref": repo_rel(runtime_path),
                "runtime_path": runtime_path,
                "synthetic_path": synthetic_path,
                "train_duration_sec": runtime.get("timings", {}).get("train", {}).get("duration_sec"),
                "generate_duration_sec": runtime.get("timings", {}).get("generate", {}).get("duration_sec"),
            }
        )
    return out


def run_dir_candidates(source: str, dataset: str, model: str):
    model_dir = SOURCE_ROOTS[source] / dataset / model
    if not model_dir.exists():
        return []
    out = []
    for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        runtime_path = find_runtime_result(run_dir)
        if runtime_path is None:
            continue
        try:
            runtime = read_json(runtime_path)
        except Exception:
            continue
        if runtime.get("generate_status") != "success":
            continue
        synthetic_path = find_synthetic_csv(run_dir, runtime)
        if synthetic_path is None:
            continue
        out.append(
            {
                "source": source,
                "run_id": runtime.get("run_id") or run_dir.name,
                "run_ref": repo_rel(run_dir),
                "runtime_path": runtime_path,
                "synthetic_path": synthetic_path,
                "train_duration_sec": runtime.get("timings", {}).get("train", {}).get("duration_sec"),
                "generate_duration_sec": runtime.get("timings", {}).get("generate", {}).get("duration_sec"),
            }
        )
    return out


def healthy(summary: dict) -> bool:
    return (
        summary["columns_match_real_exact"]
        and summary["row_count_matches_real_train"]
        and summary["constant_cols"] == 0
    )


def candidate_sort_key(cand: dict):
    summary = cand["summary"]
    return (
        0 if healthy(summary) else 1,
        0 if summary["columns_match_real_exact"] else 1,
        0 if summary["row_count_matches_real_train"] else 1,
        summary["constant_cols"],
        summary["duplicate_rows"],
        SOURCE_PREFERENCE.get(cand["source"], 99),
        -(cand["train_duration_sec"] or -1),
    )


def pick_recommendation(candidates):
    if not candidates:
        return None
    ranked = sorted(candidates, key=candidate_sort_key)
    best = ranked[0]
    summary = best["summary"]
    reason_bits = []
    if healthy(summary):
        reason_bits.append("healthy")
    else:
        reason_bits.append("best_available")
    reason_bits.append(f"source={best['source']}")
    reason_bits.append(f"dup={summary['duplicate_rows']}")
    reason_bits.append(f"const={summary['constant_cols']}")
    if best["train_duration_sec"] is not None:
        reason_bits.append(f"train={best['train_duration_sec']}")
    return best, ", ".join(reason_bits)


def main():
    with MATRIX_CSV.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    audits = []
    for row in rows:
        dataset = row["dataset"]
        if dataset not in TARGET_DATASETS:
            continue
        real_csv = ROOT / "data" / dataset / f"{dataset}-train.csv"
        for model in TARGET_MODELS:
            cell = row.get(model, "")
            if "T/H=" not in cell:
                continue
            source_codes = cell.split("=", 1)[1].split("-")
            candidates = []
            for source in source_codes:
                if source == "5" or source == "6":
                    source_candidates = consolidated_candidates(source, dataset, model)
                else:
                    source_candidates = run_dir_candidates(source, dataset, model)
                for cand in source_candidates:
                    cand["summary"] = summarize_csv(real_csv, cand["synthetic_path"])
                    candidates.append(cand)
            recommendation = pick_recommendation(candidates)
            if recommendation is None:
                audits.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "matrix_cell": cell,
                        "candidate_count": 0,
                        "recommended_source": None,
                        "recommended_run": None,
                        "reason": "no_success_csv_candidate",
                        "candidates": [],
                    }
                )
                continue
            best, reason = recommendation
            audits.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "matrix_cell": cell,
                    "candidate_count": len(candidates),
                    "recommended_source": best["source"],
                    "recommended_run": best["run_id"],
                    "recommended_run_ref": best["run_ref"],
                    "reason": reason,
                    "recommended_summary": best["summary"],
                    "candidates": [
                        {
                            "source": cand["source"],
                            "run_id": cand["run_id"],
                            "run_ref": cand["run_ref"],
                            "train_duration_sec": cand["train_duration_sec"],
                            "generate_duration_sec": cand["generate_duration_sec"],
                            "summary": cand["summary"],
                        }
                        for cand in sorted(candidates, key=candidate_sort_key)
                    ],
                }
            )

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(audits, f, ensure_ascii=False, indent=2)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "matrix_cell",
                "candidate_count",
                "recommended_source",
                "recommended_run",
                "recommended_run_ref",
                "reason",
                "duplicate_rows",
                "constant_cols",
                "columns_match_real_exact",
                "row_count_matches_real_train",
            ],
        )
        writer.writeheader()
        for item in audits:
            summary = item.get("recommended_summary", {})
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    "model": item["model"],
                    "matrix_cell": item["matrix_cell"],
                    "candidate_count": item["candidate_count"],
                    "recommended_source": item["recommended_source"],
                    "recommended_run": item.get("recommended_run"),
                    "recommended_run_ref": item.get("recommended_run_ref"),
                    "reason": item["reason"],
                    "duplicate_rows": summary.get("duplicate_rows"),
                    "constant_cols": summary.get("constant_cols"),
                    "columns_match_real_exact": summary.get("columns_match_real_exact"),
                    "row_count_matches_real_train": summary.get("row_count_matches_real_train"),
                }
            )

    lines = [
        "# T/H final candidate audit",
        "",
        "Rules: prefer healthy CSVs; among comparable candidates prefer `6`, then `5`, then `R`, and only then `H`.",
        "",
        "| dataset | model | cell | recommend | run | duplicates | const cols | schema | row count |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in audits:
        summary = item.get("recommended_summary", {})
        lines.append(
            "| {dataset} | {model} | {cell} | {source} | {run} | {dup} | {const} | {schema} | {rows} |".format(
                dataset=item["dataset"],
                model=item["model"],
                cell=item["matrix_cell"],
                source=item["recommended_source"] or "none",
                run=item.get("recommended_run") or "none",
                dup=summary.get("duplicate_rows", "n/a"),
                const=summary.get("constant_cols", "n/a"),
                schema=summary.get("columns_match_real_exact", "n/a"),
                rows=summary.get("row_count_matches_real_train", "n/a"),
            )
        )
    with OUTPUT_MD.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    counts = {}
    for item in audits:
        src = item["recommended_source"] or "none"
        counts[src] = counts.get(src, 0) + 1
    print(json.dumps({"combo_count": len(audits), "recommended_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
