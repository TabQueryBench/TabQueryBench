#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_COLS = ["bucket", "dataset", "model", "run_id"]


def _load_done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.startswith("repaired")]
    return {
        tuple(str(row[col]) for col in KEY_COLS)
        for _, row in df.iterrows()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feasibility-csv", required=True)
    ap.add_argument("--done-manifest", action="append", default=[])
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    feasibility = pd.read_csv(args.feasibility_csv)
    feasibility = feasibility[feasibility["recoverable"].astype(str) == "yes"].copy()
    if "work_estimate" not in feasibility.columns:
        feasibility["work_estimate"] = pd.to_numeric(
            feasibility.get("bad_col_count", 1), errors="coerce"
        ).fillna(1).astype(int)

    done_keys: set[tuple[str, str, str, str]] = set()
    for manifest in args.done_manifest:
        done_keys |= _load_done_keys(Path(manifest))

    if done_keys:
        feasibility = feasibility[
            ~feasibility.apply(
                lambda r: tuple(str(r[col]) for col in KEY_COLS) in done_keys, axis=1
            )
        ].reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = [[] for _ in range(args.num_shards)]
    weights = [0] * args.num_shards

    for _, row in feasibility.sort_values("work_estimate", ascending=False).iterrows():
        idx = min(range(args.num_shards), key=lambda i: weights[i])
        shards[idx].append(row)
        weights[idx] += int(row["work_estimate"])

    shard_counts: list[int] = []
    for i, rows in enumerate(shards, start=1):
        shard_df = pd.DataFrame(rows, columns=feasibility.columns)
        shard_path = out_dir / f"decode_recovery_shard_{i}.csv"
        shard_df.to_csv(shard_path, index=False)
        shard_counts.append(int(len(shard_df)))

    summary = {
        "feasibility_total": int(len(pd.read_csv(args.feasibility_csv))),
        "done_key_count": int(len(done_keys)),
        "remaining_count": int(len(feasibility)),
        "num_shards": int(args.num_shards),
        "shard_counts": shard_counts,
        "shard_weights": weights,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
