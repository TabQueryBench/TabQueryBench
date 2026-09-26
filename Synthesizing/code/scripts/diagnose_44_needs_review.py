import csv
import json
from pathlib import Path


ROOT = Path(r"D:\dpan\Uni\Project\HKUNAISS\SQLagent")
AUDIT_JSON = ROOT / "tmp" / "single_source_nowhere_quality_audit_20260509.json"
OUT_JSON = ROOT / "tmp" / "needs_review_44_diagnosis_20260509.json"
OUT_MD = ROOT / "tmp" / "needs_review_44_diagnosis_20260509.md"
OUT_CSV = ROOT / "tmp" / "needs_review_44_diagnosis_20260509.csv"
SOURCE_ROOTS = {
    "5": ROOT / "SynOutput-5090",
    "6": ROOT / "SynOutput",
    "R": ROOT / "remote-output-Benchmark-trainonly-v1",
}
CSV_SUFFIX_BLACKLIST = ("__real.csv", "__train.csv", "__val.csv", "__test.csv")


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1:]


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


def resolve_synthetic_path(item: dict):
    source = item["source"]
    dataset = item["dataset"]
    model = item["model"]
    run_ref = item["recommended_run_ref"]
    if source in {"5", "6"}:
        runtime_path = ROOT / Path(run_ref)
        runtime = read_json(runtime_path)
        model_dir = SOURCE_ROOTS[source] / dataset / model
        if not model_dir.exists() and model == "realtabformer":
            alias = SOURCE_ROOTS[source] / dataset / "rtf"
            if alias.exists():
                model_dir = alias
        synth_dir = model_dir / "synthetic_data"
        return find_synthetic_csv(synth_dir, runtime), runtime_path
    if source == "R":
        run_dir = ROOT / Path(run_ref)
        runtime_path = find_runtime_result(run_dir)
        runtime = read_json(runtime_path)
        return find_synthetic_csv(run_dir, runtime), runtime_path
    return None, None


def classify_schema(real_header, syn_header):
    if syn_header == real_header:
        return None, [], []
    real_set = set(real_header)
    syn_set = set(syn_header)
    missing = [col for col in real_header if col not in syn_set]
    extra = [col for col in syn_header if col not in real_set]
    if not missing and not extra and len(real_header) == len(syn_header):
        return "column_order_mismatch", missing, extra
    return "column_name_set_mismatch", missing, extra


def main():
    audits = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    targets = [item for item in audits if item.get("status") == "needs_review"]
    diagnoses = []

    for item in targets:
        synthetic_path, runtime_path = resolve_synthetic_path(item)
        real_csv = ROOT / "data" / item["dataset"] / f"{item['dataset']}-train.csv"
        real_header, real_rows = read_csv(real_csv)
        syn_header, syn_rows = read_csv(synthetic_path)
        schema_issue, missing_cols, extra_cols = classify_schema(real_header, syn_header)
        summary = item["summary"]

        generation_issues = []
        if summary["constant_cols"] > 0:
            generation_issues.append(f"{summary['constant_cols']} suspicious constant columns")
        if not summary["row_count_matches_real_train"]:
            generation_issues.append(
                f"row count mismatch: synthetic={summary['rows']} real_train={len(real_rows)}"
            )
        if summary["duplicate_rows"] > 0:
            generation_issues.append(f"{summary['duplicate_rows']} duplicate rows observed")

        if schema_issue and generation_issues:
            root_cause = "mixed"
            likely_origin = "both schema/postprocess and generation"
        elif schema_issue:
            root_cause = "schema_postprocess"
            likely_origin = "preprocessing/postprocess schema restoration"
        else:
            root_cause = "generation"
            likely_origin = "generation output quality"

        diagnosis = {
            "dataset": item["dataset"],
            "model": item["model"],
            "source": item["source"],
            "run": item["recommended_run"],
            "runtime_ref": repo_rel(runtime_path) if runtime_path else item["recommended_run_ref"],
            "synthetic_csv": repo_rel(synthetic_path) if synthetic_path else None,
            "root_cause": root_cause,
            "likely_origin": likely_origin,
            "schema_issue": schema_issue,
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "summary": summary,
            "generation_issues": generation_issues,
            "note": None,
        }

        if schema_issue == "column_order_mismatch":
            diagnosis["note"] = "Same column set, but order differs from raw train schema."
        elif schema_issue == "column_name_set_mismatch":
            diagnosis["note"] = "Generated CSV column names/set do not match raw train schema."
        elif generation_issues:
            diagnosis["note"] = "Schema aligns, but generated content quality violates current release rule."

        diagnoses.append(diagnosis)

    OUT_JSON.write_text(json.dumps(diagnoses, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "source",
                "run",
                "root_cause",
                "likely_origin",
                "schema_issue",
                "missing_columns_count",
                "extra_columns_count",
                "constant_cols",
                "duplicate_rows",
                "row_count_matches_real_train",
                "note",
            ],
        )
        writer.writeheader()
        for d in diagnoses:
            writer.writerow(
                {
                    "dataset": d["dataset"],
                    "model": d["model"],
                    "source": d["source"],
                    "run": d["run"],
                    "root_cause": d["root_cause"],
                    "likely_origin": d["likely_origin"],
                    "schema_issue": d["schema_issue"],
                    "missing_columns_count": len(d["missing_columns"]),
                    "extra_columns_count": len(d["extra_columns"]),
                    "constant_cols": d["summary"]["constant_cols"],
                    "duplicate_rows": d["summary"]["duplicate_rows"],
                    "row_count_matches_real_train": d["summary"]["row_count_matches_real_train"],
                    "note": d["note"],
                }
            )

    counts = {"schema_postprocess": 0, "generation": 0, "mixed": 0}
    for d in diagnoses:
        counts[d["root_cause"]] += 1

    lines = [
        "# 44 needs_review diagnosis",
        "",
        f"- total: `{len(diagnoses)}`",
        f"- schema_postprocess: `{counts['schema_postprocess']}`",
        f"- generation: `{counts['generation']}`",
        f"- mixed: `{counts['mixed']}`",
        "",
    ]
    for d in diagnoses:
        s = d["summary"]
        lines.extend(
            [
                f"## {d['dataset']} / {d['model']}",
                f"- source: `{d['source']}`",
                f"- run: `{d['run']}`",
                f"- root_cause: `{d['root_cause']}`",
                f"- likely_origin: `{d['likely_origin']}`",
                f"- schema_issue: `{d['schema_issue'] or 'none'}`",
                f"- constant_cols: `{s['constant_cols']}`",
                f"- duplicate_rows: `{s['duplicate_rows']}`",
                f"- row_count_matches_real_train: `{s['row_count_matches_real_train']}`",
                f"- note: `{d['note'] or ''}`",
            ]
        )
        if d["missing_columns"]:
            lines.append(f"- missing_columns: `{', '.join(d['missing_columns'][:12])}`")
        if d["extra_columns"]:
            lines.append(f"- extra_columns: `{', '.join(d['extra_columns'][:12])}`")
        if d["generation_issues"]:
            lines.append(f"- generation_issues: `{'; '.join(d['generation_issues'])}`")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
