#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOTS = [
    ("benchmark", Path("/data/jialinzhang/SynthesizePipeline-server/output-Benchmark-trainonly-v1")),
    ("specialized", Path("/data/jialinzhang/SynthesizePipeline-server/output-SpecializedModels")),
]


def main():
    matrix_json = Path("/tmp/hf_matrix_filled_20260515_v2.json")
    out = Path("/tmp/hku172_success_vs_matrix_20260516")
    out.mkdir(parents=True, exist_ok=True)

    matrix = json.loads(matrix_json.read_text())
    filled = {(x["dataset"], x["model"]) for x in matrix["filled"]}

    success_by_combo = defaultdict(list)
    for root_name, root in ROOTS:
        if not root.is_dir():
            continue
        for ds_dir in sorted(root.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            for model_dir in sorted(ds_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                for run_dir in sorted(model_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    rr = run_dir / "runtime_result.json"
                    if not rr.exists():
                        continue
                    try:
                        data = json.loads(rr.read_text())
                    except Exception:
                        continue
                    if str(data.get("generate_status", "")) == "success":
                        success_by_combo[(ds, model)].append(
                            {
                                "root": root_name,
                                "run_id": run_dir.name,
                                "run_dir": str(run_dir),
                                "train_status": str(data.get("train_status", "")),
                                "generate_status": str(data.get("generate_status", "")),
                            }
                        )

    live_rows = []
    missing_rows = []
    for (ds, model), runs in sorted(success_by_combo.items()):
        live_rows.append(
            {
                "dataset": ds,
                "model": model,
                "success_run_count_on_hku172": len(runs),
                "roots": sorted({r["root"] for r in runs}),
                "example_run_id": runs[0]["run_id"],
                "example_run_dir": runs[0]["run_dir"],
            }
        )
        if (ds, model) not in filled:
            missing_rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "success_run_count_on_hku172": len(runs),
                    "roots": ";".join(sorted({r["root"] for r in runs})),
                    "example_run_id": runs[0]["run_id"],
                    "example_run_dir": runs[0]["run_dir"],
                    "train_status": runs[0]["train_status"],
                    "generate_status": runs[0]["generate_status"],
                }
            )

    (out / "live_success_combos.json").write_text(json.dumps(live_rows, indent=2, ensure_ascii=False))
    with (out / "missing_from_matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "success_run_count_on_hku172",
                "roots",
                "example_run_id",
                "example_run_dir",
                "train_status",
                "generate_status",
            ],
        )
        writer.writeheader()
        writer.writerows(missing_rows)

    summary = {
        "hku172_live_success_combo_count": len(success_by_combo),
        "matrix_filled_combo_count": len(filled),
        "missing_combo_count": len(missing_rows),
        "roots_scanned": [str(r[1]) for r in ROOTS],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(out / "missing_from_matrix.csv")
    print(out / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
