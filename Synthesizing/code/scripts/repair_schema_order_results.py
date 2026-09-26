#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "tmp" / "schema_order_repair_selected5_20260510.json"
REPORT_MD = REPO_ROOT / "tmp" / "schema_order_repair_selected5_20260510.md"

TARGETS: List[Tuple[str, str]] = [
    ("m4", "tabbyflow"),
    ("m4", "tabddpm"),
    ("m4", "tabdiff"),
    ("m6", "tabbyflow"),
    ("m6", "tabddpm"),
]

SEARCH_ROOTS = [
    REPO_ROOT / "SynOutput",
    REPO_ROOT / "SynOutput-5090",
    REPO_ROOT / "remote-output-Benchmark-trainonly-v1",
    REPO_ROOT / "hyperparameter" / "output-Benchmark-trainonly-v1",
]


def load_expected_header(dataset: str) -> List[str]:
    train_csv = REPO_ROOT / "data" / dataset / f"{dataset}-train.csv"
    with train_csv.open("r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def should_consider_csv(path: Path) -> bool:
    if path.name.lower() == "loss.csv":
        return False
    if path.name.endswith("-train.csv") or path.name.endswith("-val.csv") or path.name.endswith("-test.csv"):
        return False
    return True


def reorder_csv(path: Path, expected_header: List[str]) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return {"status": "skipped", "reason": "no_header"}
        current_header = list(reader.fieldnames)
        if current_header == expected_header:
            return {"status": "skipped", "reason": "already_correct"}
        if len(current_header) != len(expected_header):
            return {
                "status": "ignored",
                "reason": "column_count_mismatch",
                "current_header": current_header,
            }
        if set(current_header) != set(expected_header):
            return {
                "status": "ignored",
                "reason": "column_set_mismatch",
                "current_header": current_header,
            }
        rows = list(reader)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=expected_header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in expected_header})
    return {
        "status": "repaired",
        "from_header": current_header,
        "to_header": expected_header,
        "rows": len(rows),
    }


def main() -> int:
    report: Dict[str, object] = {
        "targets": [],
        "repaired_files": 0,
        "skipped_already_correct": 0,
        "ignored_nonmatching": 0,
    }
    md_lines = ["# Schema Order Repair For Selected Results", ""]

    for dataset, model in TARGETS:
        expected = load_expected_header(dataset)
        target_summary = {
            "dataset": dataset,
            "model": model,
            "expected_header": expected,
            "files": [],
        }
        md_lines.append(f"## `{dataset}/{model}`")
        md_lines.append("")
        md_lines.append(f"- Expected header: `{','.join(expected)}`")
        md_lines.append("")

        for root in SEARCH_ROOTS:
            combo_root = root / dataset / model
            if not combo_root.exists():
                continue
            for path in sorted(combo_root.rglob("*.csv")):
                if not should_consider_csv(path):
                    continue
                result = reorder_csv(path, expected)
                item = {
                    "path": str(path),
                    "root": str(root),
                    **result,
                }
                target_summary["files"].append(item)
                status = result["status"]
                if status == "repaired":
                    report["repaired_files"] = int(report["repaired_files"]) + 1
                    md_lines.append(f"- Repaired: `{path}`")
                elif status == "skipped" and result["reason"] == "already_correct":
                    report["skipped_already_correct"] = int(report["skipped_already_correct"]) + 1
                else:
                    report["ignored_nonmatching"] = int(report["ignored_nonmatching"]) + 1
                    md_lines.append(f"- Ignored `{path}`: `{result['reason']}`")
        report["targets"].append(target_summary)
        md_lines.append("")

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["repaired_files", "skipped_already_correct", "ignored_nonmatching"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
