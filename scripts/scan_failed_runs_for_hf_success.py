#!/usr/bin/env python3
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


ROOTS = [
    ("benchmark", Path("/data/jialinzhang/SynthesizePipeline-server/output-Benchmark-trainonly-v1")),
    ("specialized", Path("/data/jialinzhang/SynthesizePipeline-server/output-SpecializedModels")),
]


def human_size(n: int) -> str:
    if n <= 0:
        return "0B"
    units = ["B", "K", "M", "G", "T"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(x)}{u}"
            return f"{x:.0f}{u}" if x >= 10 else f"{x:.1f}{u}"
        x /= 1024
    return f"{n}B"


def dir_size_bytes(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            fp = Path(dirpath) / fn
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def load_success_matrix(matrix_path: Path):
    rows = list(csv.DictReader(matrix_path.open()))
    models = [c for c in rows[0].keys() if c != "dataset"]
    datasets = [r["dataset"] for r in rows]
    success_combos = {(r["dataset"], m) for r in rows for m in models if r[m] in ("110", "111")}
    return rows, datasets, models, success_combos


def classify_run(run_dir: Path):
    rr = run_dir / "runtime_result.json"
    if rr.exists():
        try:
            data = json.loads(rr.read_text())
        except Exception:
            return "runtime_result_unparsed", "", ""
        train_status = str(data.get("train_status", ""))
        generate_status = str(data.get("generate_status", ""))
        if generate_status == "success":
            return None, train_status, generate_status
        state = f"{train_status}|{generate_status}" if (train_status or generate_status) else "runtime_result_unparsed"
        return state, train_status, generate_status
    return "missing_runtime_result", "", ""


def main():
    matrix_path = Path(os.environ["HF_SUCCESS_MATRIX"])
    output_dir = Path(os.environ["HF_FAILED_AUDIT_OUT"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _, datasets, models, success_combos = load_success_matrix(matrix_path)

    summary = defaultdict(lambda: {"count": 0, "size": 0})
    details = []

    for ds, model in sorted(success_combos):
        for root_name, root in ROOTS:
            model_dir = root / ds / model
            if not model_dir.is_dir():
                continue
            for child in sorted(model_dir.iterdir()):
                if not child.is_dir():
                    continue
                state, train_status, generate_status = classify_run(child)
                if state is None:
                    continue
                size = dir_size_bytes(child)
                summary[(ds, model)]["count"] += 1
                summary[(ds, model)]["size"] += size
                details.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "root": root_name,
                        "run_id": child.name,
                        "run_dir": str(child),
                        "status": state,
                        "train_status": train_status,
                        "generate_status": generate_status,
                        "size_bytes": size,
                    }
                )

    matrix_out = output_dir / "hf_success_failed_runs_matrix.csv"
    with matrix_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + models)
        for ds in datasets:
            row = [ds]
            for model in models:
                item = summary.get((ds, model))
                row.append(f"{item['count']}-{human_size(item['size'])}" if item and item["count"] > 0 else "")
            writer.writerow(row)

    long_out = output_dir / "hf_success_failed_runs_long.csv"
    with long_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "failed_run_count", "failed_total_size_bytes", "failed_total_size_human"],
        )
        writer.writeheader()
        for ds in datasets:
            for model in models:
                item = summary.get((ds, model))
                if item and item["count"] > 0:
                    writer.writerow(
                        {
                            "dataset": ds,
                            "model": model,
                            "failed_run_count": item["count"],
                            "failed_total_size_bytes": item["size"],
                            "failed_total_size_human": human_size(item["size"]),
                        }
                    )

    detail_out = output_dir / "hf_success_failed_runs_detail.csv"
    with detail_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "root", "run_id", "run_dir", "status", "train_status", "generate_status", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(details)

    summary_json = {
        "success_combo_count": len(success_combos),
        "combos_with_failed_runs": sum(1 for v in summary.values() if v["count"] > 0),
        "failed_run_total_count": sum(v["count"] for v in summary.values()),
        "failed_run_total_size_bytes": sum(v["size"] for v in summary.values()),
        "roots_scanned": [str(r[1]) for r in ROOTS],
        "notes": [
            "Only dataset/model combinations marked 110 or 111 in HF success matrix were scanned.",
            "Runs with generate_status=success were treated as successful and excluded.",
            "Runs missing runtime_result.json are counted as incomplete/failed for cleanup review.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False))

    print(matrix_out)
    print(long_out)
    print(detail_out)
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
