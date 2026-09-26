#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_ROOTS = {"syntheticSuccess", "5090-Success", "timecost", "hyperparameter"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-csv', required=True)
    ap.add_argument('--hf-repo', required=True)
    ap.add_argument('--out-csv', required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    repo = Path(args.hf_repo)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        status = str(row.get('status', ''))
        if status not in {'replaced', 'repaired'}:
            results.append({
                'bucket': row['bucket'], 'dataset': row['dataset'], 'model': row['model'], 'run_id': row['run_id'],
                'status': 'skipped', 'reason': status, 'updated_paths_json': '[]'
            })
            continue

        src = Path(row['final_csv_path'])
        if not src.exists():
            results.append({
                'bucket': row['bucket'], 'dataset': row['dataset'], 'model': row['model'], 'run_id': row['run_id'],
                'status': 'failed', 'reason': 'final_csv_missing', 'updated_paths_json': '[]'
            })
            continue

        suffix = Path(row['dataset']) / row['model'] / row['run_id'] / src.name
        matches = []
        for p in repo.rglob(src.name):
            parts = p.relative_to(repo).parts
            if not parts or parts[0] not in ALLOWED_ROOTS:
                continue
            rel = Path(*parts[1:])
            if rel == suffix:
                matches.append(p)

        if not matches:
            results.append({
                'bucket': row['bucket'], 'dataset': row['dataset'], 'model': row['model'], 'run_id': row['run_id'],
                'status': 'failed', 'reason': 'no_hf_match', 'updated_paths_json': '[]'
            })
            continue

        for dst in matches:
            shutil.copy2(src, dst)

        results.append({
            'bucket': row['bucket'], 'dataset': row['dataset'], 'model': row['model'], 'run_id': row['run_id'],
            'status': 'updated', 'reason': '',
            'updated_paths_json': json.dumps([str(p.relative_to(repo)) for p in matches], ensure_ascii=False)
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    summary = {
        'total_manifest_rows': int(len(manifest)),
        'updated_rows': int((out_df['status'] == 'updated').sum()),
        'skipped_rows': int((out_df['status'] == 'skipped').sum()),
        'failed_rows': int((out_df['status'] == 'failed').sum()),
    }
    out_csv.with_suffix('.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
