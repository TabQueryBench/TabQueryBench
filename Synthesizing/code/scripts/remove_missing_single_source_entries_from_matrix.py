import csv
import json
from pathlib import Path


ROOT = Path(r"D:\dpan\Uni\Project\HKUNAISS\SQLagent")
AUDIT_JSON = ROOT / "tmp" / "single_source_nowhere_quality_audit_20260509.json"
MATRIX_CSV = ROOT / "tmp" / "generated_data_source_matrix_20260506.csv"
MATRIX_MD = ROOT / "tmp" / "generated_data_source_matrix_20260506.md"
REPORT_JSON = ROOT / "tmp" / "removed_missing_single_source_entries_20260509.json"


def main():
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    targets = [
        {
            "dataset": item["dataset"],
            "model": item["model"],
            "source": item["source"],
            "reason": item["reason"],
        }
        for item in audit
        if item.get("status") == "missing_or_unusable"
    ]
    target_map = {(item["dataset"], item["model"]): item for item in targets}

    with MATRIX_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    removed = []
    for row in rows:
        dataset = row["dataset"]
        for model in fieldnames:
            if model == "dataset":
                continue
            key = (dataset, model)
            if key not in target_map:
                continue
            old_value = row.get(model, "")
            if old_value:
                removed.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "old_value": old_value,
                        "source": target_map[key]["source"],
                        "reason": target_map[key]["reason"],
                    }
                )
            row[model] = ""

    with MATRIX_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Generated Data Source Matrix (updated 2026-05-09)",
        "",
        "Legend: source codes `5=SynOutput-5090`, `6=SynOutput`, `H=hyperparameter`, `R=remote-output-Benchmark-trainonly-v1`; destination prefixes `F=final_csv`, `T=time_cost`, `H=hyper_parameter`.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(h, "") for h in fieldnames) + " |")
    MATRIX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "removed_count": len(removed),
        "removed": removed,
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
