"""CLI for dataset preparation.

  PYTHONPATH=Synthesizing/code/src python -m core.prepare run <dataset_dir|profile.yaml|parent_dir> ... [--out-root DIR]
  PYTHONPATH=Synthesizing/code/src python -m core.prepare check <dataset_dir|profile.yaml> ...
  PYTHONPATH=Synthesizing/code/src python -m core.prepare draft --data data.csv --dataset-id X [--target COL] [-o profile.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import PrepareError, default_out_root, find_profiles, prepare_many
from .profile import ProfileError, draft_profile, load_profile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.prepare", description="Prepare datasets from profile.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="validate, clean, split and write runner-ready datasets")
    p_run.add_argument("paths", nargs="+", type=Path, help="profile.yaml, dataset dir, or a parent dir of dataset dirs")
    p_run.add_argument("--out-root", type=Path, default=None, help="default: BENCHMARK_NEW_DATASET_ROOT or code/DatasetNew")

    p_check = sub.add_parser("check", help="validate profile.yaml structure only")
    p_check.add_argument("paths", nargs="+", type=Path)

    p_draft = sub.add_parser("draft", help="infer a draft profile.yaml from a CSV for human review")
    p_draft.add_argument("--data", type=Path, required=True)
    p_draft.add_argument("--dataset-id", required=True)
    p_draft.add_argument("--target", default=None)
    p_draft.add_argument("--delimiter", default=",")
    p_draft.add_argument("-o", "--output", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.command == "draft":
        text = draft_profile(args.data, args.dataset_id, args.target, args.delimiter)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
            print(f"draft written: {args.output}")
        else:
            sys.stdout.write(text)
        return 0

    if args.command == "check":
        ok = True
        for path in find_profiles(args.paths):
            try:
                prof = load_profile(path)
                print(f"OK   {prof.dataset_id}: {len(prof.columns)} columns, target={prof.target_column} ({prof.task_type})")
            except ProfileError as exc:
                ok = False
                print(f"FAIL {exc}")
        return 0 if ok else 1

    out_root = args.out_root or default_out_root()
    try:
        results = prepare_many(args.paths, out_root)
    except PrepareError as exc:
        print(f"FAIL {exc}")
        return 1
    failed = 0
    for r in results:
        if r.get("status") == "pass":
            rows = r["split"]["rows"]
            print(
                f"PASS {r['dataset_id']}: main={r['output']['main_rows']} train={rows['train']} val={rows['val']} "
                f"test={rows['test']} cols={r['output']['columns']} warnings={len(r['warnings'])} -> {r['out_dir']}"
            )
            for w in r["warnings"]:
                print(f"     warn: {w}")
        else:
            failed += 1
            print(f"FAIL {r.get('profile')}\n     {r.get('error')}")
    summary = Path(out_root) / "prepare_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"summary: {summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
