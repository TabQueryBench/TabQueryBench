import csv
import json
from pathlib import Path


ROOT = Path.cwd()
DIAG_JSON = ROOT / "tmp" / "needs_review_44_diagnosis_20260509.json"
OUT_JSON = ROOT / "tmp" / "column_order_repair_report_20260509.json"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def reorder_file(csv_path: Path, target_header):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if set(fieldnames) != set(target_header):
        return {
            "status": "failed",
            "reason": "column_set_mismatch",
            "current_header": fieldnames,
            "target_header": target_header,
        }

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=target_header)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in target_header})
    tmp_path.replace(csv_path)
    return {"status": "repaired", "row_count": len(rows)}


def main():
    diagnoses = json.loads(DIAG_JSON.read_text(encoding="utf-8"))
    targets = [d for d in diagnoses if d.get("schema_issue") == "column_order_mismatch"]
    report = {"target_count": len(targets), "repaired": [], "failed": []}

    for item in targets:
        real_csv = ROOT / "data" / item["dataset"] / f"{item['dataset']}-train.csv"
        synthetic_csv = ROOT / item["synthetic_csv"]
        target_header, _ = read_csv(real_csv)
        result = reorder_file(synthetic_csv, target_header)
        payload = {
            "dataset": item["dataset"],
            "model": item["model"],
            "synthetic_csv": item["synthetic_csv"],
            **result,
        }
        if result["status"] == "repaired":
            report["repaired"].append(payload)
        else:
            report["failed"].append(payload)

    report["repaired_count"] = len(report["repaired"])
    report["failed_count"] = len(report["failed"])
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"target_count": report["target_count"], "repaired_count": report["repaired_count"], "failed_count": report["failed_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
