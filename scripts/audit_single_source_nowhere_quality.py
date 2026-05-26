import csv
import json
from pathlib import Path


ROOT = Path(r"D:\dpan\Uni\Project\HKUNAISS\SQLagent")
MATRIX_CSV = ROOT / "tmp" / "generated_data_source_matrix_20260506.csv"
OUT_JSON = ROOT / "tmp" / "single_source_nowhere_quality_audit_20260509.json"
OUT_CSV = ROOT / "tmp" / "single_source_nowhere_quality_audit_20260509.csv"
OUT_MD = ROOT / "tmp" / "single_source_nowhere_quality_audit_20260509.md"

EXCLUDE_MODELS = {"codi", "cdtd", "goggle"}
SOURCE_ROOTS = {
    "5": ROOT / "SynOutput-5090",
    "6": ROOT / "SynOutput",
    "R": ROOT / "remote-output-Benchmark-trainonly-v1",
}
CSV_SUFFIX_BLACKLIST = ("__real.csv", "__train.csv", "__val.csv", "__test.csv")
SOURCE_PREFERENCE = {"6": 0, "5": 1, "R": 2}


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
        rows = list(csv.reader(f))
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


def healthy(summary: dict) -> bool:
    return (
        summary["columns_match_real_exact"]
        and summary["row_count_matches_real_train"]
        and summary["constant_cols"] == 0
    )


def candidate_sort_key(cand: dict):
    s = cand["summary"]
    return (
        0 if healthy(s) else 1,
        0 if s["columns_match_real_exact"] else 1,
        0 if s["row_count_matches_real_train"] else 1,
        s["constant_cols"],
        s["duplicate_rows"],
        -(cand.get("train_duration_sec") or -1),
        SOURCE_PREFERENCE.get(cand["source"], 99),
    )


def find_runtime_result(run_dir: Path):
    direct = run_dir / "runtime_result.json"
    if direct.exists():
        return direct
    meta = run_dir / "meta" / "runtime_result.json"
    if meta.exists():
        return meta
    return None


def find_synthetic_csv(base_dir: Path, runtime: dict):
    basename = Path(runtime.get("artifacts", {}).get("synthetic_csv", "")).name
    if basename:
        direct = base_dir / basename
        if direct.exists():
            return direct
        for p in base_dir.rglob(basename):
            if p.is_file():
                return p
    candidates = [
        p
        for p in base_dir.rglob("*.csv")
        if p.is_file() and not any(p.name.endswith(suf) for suf in CSV_SUFFIX_BLACKLIST)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def consolidated_candidates(source: str, dataset: str, model: str):
    model_dir = SOURCE_ROOTS[source] / dataset / model
    alias_used = False
    if not model_dir.exists() and model == "realtabformer":
        alias = SOURCE_ROOTS[source] / dataset / "rtf"
        if alias.exists():
            model_dir = alias
            alias_used = True
    if not model_dir.exists():
        return [], "source_model_dir_missing"
    meta_dir = model_dir / "metadata"
    synth_dir = model_dir / "synthetic_data"
    if not meta_dir.exists() or not synth_dir.exists():
        return [], "insufficient_clean_metadata_or_artifact"
    runtime_files = sorted(meta_dir.glob("*__runtime_result.json"))
    out = []
    for runtime_path in runtime_files:
        try:
            runtime = read_json(runtime_path)
        except Exception:
            continue
        if runtime.get("generate_status") != "success":
            continue
        synthetic_csv = find_synthetic_csv(synth_dir, runtime)
        if synthetic_csv is None:
            continue
        out.append(
            {
                "source": source,
                "run_id": runtime.get("run_id") or runtime_path.stem,
                "run_ref": repo_rel(runtime_path),
                "synthetic_path": synthetic_csv,
                "train_duration_sec": runtime.get("timings", {}).get("train", {}).get("duration_sec"),
                "generate_duration_sec": runtime.get("timings", {}).get("generate", {}).get("duration_sec"),
            }
        )
    if out:
        return out, None
    if alias_used:
        return [], "no_success_csv_candidate_under_alias"
    return [], "no_success_csv_candidate"


def run_dir_candidates(source: str, dataset: str, model: str):
    model_dir = SOURCE_ROOTS[source] / dataset / model
    if not model_dir.exists():
        return [], "source_model_dir_missing"
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
        synthetic_csv = find_synthetic_csv(run_dir, runtime)
        if synthetic_csv is None:
            continue
        out.append(
            {
                "source": source,
                "run_id": runtime.get("run_id") or run_dir.name,
                "run_ref": repo_rel(run_dir),
                "synthetic_path": synthetic_csv,
                "train_duration_sec": runtime.get("timings", {}).get("train", {}).get("duration_sec"),
                "generate_duration_sec": runtime.get("timings", {}).get("generate", {}).get("duration_sec"),
            }
        )
    if out:
        return out, None
    return [], "no_success_csv_candidate"


def gather_candidates(source: str, dataset: str, model: str):
    if source in {"5", "6"}:
        return consolidated_candidates(source, dataset, model)
    if source == "R":
        return run_dir_candidates(source, dataset, model)
    return [], "unsupported_source"


def main():
    with MATRIX_CSV.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    audits = []
    for row in rows:
        dataset = row["dataset"]
        real_csv = ROOT / "data" / dataset / f"{dataset}-train.csv"
        for model, cell in row.items():
            if model == "dataset" or model in EXCLUDE_MODELS or not cell:
                continue
            if "=" in cell:
                continue
            sources = cell.split("-")
            if len(sources) != 1:
                continue
            source = sources[0]
            candidates, reason = gather_candidates(source, dataset, model)
            audit = {
                "dataset": dataset,
                "model": model,
                "source": source,
                "cell": cell,
                "candidate_count": len(candidates),
            }
            if not candidates:
                audit.update(
                    {
                        "status": "missing_or_unusable",
                        "meets_requirements": False,
                        "reason": reason,
                        "recommended_run": None,
                        "recommended_run_ref": None,
                        "summary": None,
                    }
                )
                audits.append(audit)
                continue
            for cand in candidates:
                cand["summary"] = summarize_csv(real_csv, cand["synthetic_path"])
            ranked = sorted(candidates, key=candidate_sort_key)
            best = ranked[0]
            audit.update(
                {
                    "status": "pass" if healthy(best["summary"]) else "needs_review",
                    "meets_requirements": healthy(best["summary"]),
                    "reason": (
                        "healthy_best_candidate"
                        if healthy(best["summary"])
                        else "best_candidate_has_quality_issues"
                    ),
                    "recommended_run": best["run_id"],
                    "recommended_run_ref": best["run_ref"],
                    "summary": best["summary"],
                    "all_candidates": [
                        {
                            "run_id": cand["run_id"],
                            "run_ref": cand["run_ref"],
                            "train_duration_sec": cand["train_duration_sec"],
                            "generate_duration_sec": cand["generate_duration_sec"],
                            "summary": cand["summary"],
                        }
                        for cand in ranked
                    ],
                }
            )
            audits.append(audit)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(audits, f, ensure_ascii=False, indent=2)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "source",
                "cell",
                "candidate_count",
                "status",
                "meets_requirements",
                "reason",
                "recommended_run",
                "recommended_run_ref",
                "duplicate_rows",
                "constant_cols",
                "columns_match_real_exact",
                "row_count_matches_real_train",
            ],
        )
        writer.writeheader()
        for item in audits:
            s = item.get("summary") or {}
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    "model": item["model"],
                    "source": item["source"],
                    "cell": item["cell"],
                    "candidate_count": item["candidate_count"],
                    "status": item["status"],
                    "meets_requirements": item["meets_requirements"],
                    "reason": item["reason"],
                    "recommended_run": item.get("recommended_run"),
                    "recommended_run_ref": item.get("recommended_run_ref"),
                    "duplicate_rows": s.get("duplicate_rows"),
                    "constant_cols": s.get("constant_cols"),
                    "columns_match_real_exact": s.get("columns_match_real_exact"),
                    "row_count_matches_real_train": s.get("row_count_matches_real_train"),
                }
            )

    counts = {"pass": 0, "needs_review": 0, "missing_or_unusable": 0}
    for item in audits:
        counts[item["status"]] += 1

    lines = [
        "# Single-source nowhere quality audit",
        "",
        "Rule: inspect combinations with exactly one source code and no current F/T/H destination prefix; excluded models `codi/cdtd/goggle` are omitted.",
        "",
        f"- combo_count: `{len(audits)}`",
        f"- pass: `{counts['pass']}`",
        f"- needs_review: `{counts['needs_review']}`",
        f"- missing_or_unusable: `{counts['missing_or_unusable']}`",
        "",
        "| dataset | model | source | status | run | duplicates | const cols | schema | row count | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in audits:
        s = item.get("summary") or {}
        lines.append(
            "| {dataset} | {model} | {source} | {status} | {run} | {dup} | {const} | {schema} | {rows} | {reason} |".format(
                dataset=item["dataset"],
                model=item["model"],
                source=item["source"],
                status=item["status"],
                run=item.get("recommended_run") or "none",
                dup=s.get("duplicate_rows", "n/a"),
                const=s.get("constant_cols", "n/a"),
                schema=s.get("columns_match_real_exact", "n/a"),
                rows=s.get("row_count_matches_real_train", "n/a"),
                reason=item["reason"],
            )
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"combo_count": len(audits), **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
