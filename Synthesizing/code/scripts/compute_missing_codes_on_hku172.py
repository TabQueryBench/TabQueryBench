#!/usr/bin/env python3
import csv
from pathlib import Path


def has_log(run_dir: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() == ".log" for p in run_dir.rglob("*.log"))


def is_synthetic_csv(p: Path, run_dir: Path) -> bool:
    name = p.name.lower()
    if not name.endswith(".csv"):
        return False
    banned = {"train.csv", "test.csv", "val.csv", "real.csv", "ctgan_train_prepared.csv"}
    if name in banned:
        return False
    if p.parent == run_dir:
        return True
    parts = {part.lower() for part in p.parts}
    if "synthetic" in parts:
        return True
    return False


def has_synthetic_csv(run_dir: Path) -> bool:
    return any(p.is_file() and is_synthetic_csv(p, run_dir) for p in run_dir.rglob("*.csv"))


def has_weight(run_dir: Path) -> bool:
    exts = {".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".bin", ".safetensors", ".joblib", ".onnx", ".model"}
    for p in run_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            return True
        name = p.name.lower()
        if "best-disc-model" in name or "last-epoch-model" in name or name.startswith("model_"):
            return True
    return False


def main():
    in_csv = Path("/tmp/hku172_success_vs_matrix_20260516/missing_from_matrix.csv")
    out_csv = Path("/tmp/hku172_success_vs_matrix_20260516/missing_with_codes.csv")
    rows = list(csv.DictReader(in_csv.open()))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) + ["code", "has_log", "has_synthetic_csv", "has_weight"],
        )
        writer.writeheader()
        for row in rows:
            run_dir = Path(row["example_run_dir"])
            l = int(has_log(run_dir))
            s = int(has_synthetic_csv(run_dir))
            w = int(has_weight(run_dir))
            row["code"] = f"{l}{s}{w}"
            row["has_log"] = str(l)
            row["has_synthetic_csv"] = str(s)
            row["has_weight"] = str(w)
            writer.writerow(row)
    print(out_csv)


if __name__ == "__main__":
    main()
