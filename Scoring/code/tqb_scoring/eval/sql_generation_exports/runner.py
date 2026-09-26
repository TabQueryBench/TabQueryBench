from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable

from tqb_scoring.eval.common import sql_source_label
from tqb_query.subitem_workload_v2.paths import SUPPORTED_LINE_VERSIONS, runs_root

REPO_ROOT = Path(__file__).resolve().parents[3]
V1_RUNS_ROOT = REPO_ROOT.parent.parent / "Query" / "code" / "logs" / "runs"
EXPORT_ROOT = REPO_ROOT / "Exports" / "sql_generation"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iter_v1_runs() -> Iterable[Path]:
    for run_dir in sorted(V1_RUNS_ROOT.iterdir()):
        if run_dir.is_dir() and (run_dir / "generated_sql.sql").exists():
            yield run_dir


def iter_current_dataset_sql_dirs(line_version: str) -> Iterable[tuple[str, str, Path, Path]]:
    current_runs_root = runs_root(line_version)
    if not current_runs_root.exists():
        return
    for run_dir in sorted(current_runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for dataset_dir in sorted(run_dir.iterdir()):
            sql_dir = dataset_dir / "sql"
            if dataset_dir.is_dir() and sql_dir.is_dir():
                yield run_dir.name, dataset_dir.name, dataset_dir, sql_dir


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def export_v1() -> list[dict]:
    rows: list[dict] = []
    base = EXPORT_ROOT / "v1_legacy"
    sql_root = ensure_dir(base / "sql")
    manifest_root = ensure_dir(base / "manifests")
    summary_root = ensure_dir(base / "summaries")

    for run_dir in iter_v1_runs():
        manifest = safe_read_json(run_dir / "run_manifest.json")
        dataset_id = manifest.get("dataset_id") or run_dir.name.split("_", 1)[0]
        run_id = run_dir.name

        target_sql_dir = ensure_dir(sql_root / dataset_id / run_id)
        export_sql_path = target_sql_dir / "generated_sql.sql"
        shutil.copy2(run_dir / "generated_sql.sql", export_sql_path)

        target_manifest_dir = ensure_dir(manifest_root / dataset_id / run_id)
        for name in [
            "run_manifest.json",
            "usage_summary.json",
            "query_results.jsonl",
            "final_answer.txt",
            "trace.jsonl",
        ]:
            copy_if_exists(run_dir / name, target_manifest_dir / name)

        cli_dir = run_dir / "cli"
        if cli_dir.is_dir():
            target_cli_dir = ensure_dir(target_manifest_dir / "cli")
            for cli_file in cli_dir.iterdir():
                if cli_file.is_file():
                    shutil.copy2(cli_file, target_cli_dir / cli_file.name)

        export_manifest = {
            "sql_source_version": "v1",
            "sql_source_label": "v1_legacy",
            "version_line": "v1_legacy",
            "dataset_id": dataset_id,
            "run_id": run_id,
            "source_run_dir": str(run_dir),
            "source_sql_file": str((run_dir / "generated_sql.sql").resolve()),
            "export_sql_file": str(export_sql_path.resolve()),
            "mode": manifest.get("mode", ""),
            "status": manifest.get("status", ""),
            "engine": manifest.get("engine", ""),
            "model": manifest.get("model", ""),
        }
        (target_manifest_dir / "export_manifest.json").write_text(
            json.dumps(export_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        row = {
            "sql_source_version": "v1",
            "sql_source_label": "v1_legacy",
            "version_line": "v1_legacy",
            "dataset_id": dataset_id,
            "run_id": run_id,
            "mode": manifest.get("mode", ""),
            "status": manifest.get("status", ""),
            "engine": manifest.get("engine", ""),
            "model": manifest.get("model", ""),
            "sql_file_count": 1,
            "source_run_dir": str(run_dir),
            "export_sql_dir": str(target_sql_dir),
        }
        rows.append(row)

    write_csv(summary_root / "v1_legacy_sql_export_index.csv", rows)
    write_summary_md(base / "README.md", "v1_legacy", rows)
    return rows


def export_current(line_version: str) -> list[dict]:
    rows: list[dict] = []
    version_label = sql_source_label(line_version)
    base = EXPORT_ROOT / version_label
    sql_root = ensure_dir(base / "sql")
    manifest_root = ensure_dir(base / "manifests")
    summary_root = ensure_dir(base / "summaries")

    for run_id, dataset_id, dataset_dir, sql_dir in iter_current_dataset_sql_dirs(line_version):
        target_sql_dir = ensure_dir(sql_root / dataset_id / run_id)
        sql_files = sorted(sql_dir.glob("*.sql"))
        for sql_file in sql_files:
            shutil.copy2(sql_file, target_sql_dir / sql_file.name)

        target_manifest_dir = ensure_dir(manifest_root / dataset_id / run_id)
        metadata = {
            "sql_source_version": line_version,
            "sql_source_label": version_label,
            "version_line": version_label,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "source_dataset_dir": str(dataset_dir),
            "source_sql_dir": str(sql_dir),
            "sql_file_count": len(sql_files),
        }
        (target_manifest_dir / "export_manifest.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        artifact_dir = dataset_dir / "artifacts"
        if artifact_dir.is_dir():
            artifact_entries = sorted(p.name for p in artifact_dir.iterdir())
            (target_manifest_dir / "artifact_entries.txt").write_text(
                "\n".join(artifact_entries), encoding="utf-8"
            )

        rows.append(
            {
                "sql_source_version": line_version,
                "sql_source_label": version_label,
                "version_line": version_label,
                "dataset_id": dataset_id,
                "run_id": run_id,
                "sql_file_count": len(sql_files),
                "source_dataset_dir": str(dataset_dir),
                "source_sql_dir": str(sql_dir),
                "export_sql_dir": str(target_sql_dir),
            }
        )

    write_csv(summary_root / f"{version_label}_sql_export_index.csv", rows)
    write_summary_md(base / "README.md", version_label, rows)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_md(path: Path, line_name: str, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    dataset_count = len({row["dataset_id"] for row in rows})
    run_count = len({row["run_id"] for row in rows})
    sql_count = sum(int(row.get("sql_file_count", 0)) for row in rows)
    text = "\n".join(
        [
            f"# {line_name} SQL Export",
            "",
            f"- datasets: `{dataset_count}`",
            f"- runs: `{run_count}`",
            f"- exported SQL files: `{sql_count}`",
            "",
            "This directory is a normalized SQL-only export layer.",
            "Do not treat it as the source-of-truth run log; the original logs remain under `logs/`.",
            "",
            "Use the sibling `summaries/*.csv` files to locate the original run roots.",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_root_readme(v1_rows: list[dict], current_rows_by_version: dict[str, list[dict]]) -> None:
    ensure_dir(EXPORT_ROOT)
    current_layout_lines = [
        f"- `{sql_source_label(version)}/`",
        f"  - normalized export of current dataset-scoped SQL bundles from `logs/subitem_workload_{version}/runs/*/<dataset>/sql/`",
    ]
    current_count_lines = [
        f"- `{sql_source_label(version)}`: `{len({r['run_id'] for r in rows})}` runs, `{len({r['dataset_id'] for r in rows})}` dataset-run bundles, `{sum(int(r['sql_file_count']) for r in rows)}` SQL files"
        for version, rows in current_rows_by_version.items()
    ]
    text = "\n".join(
        [
            "# SQL Generation Exports",
            "",
            "This folder separates legacy and current SQL-generation outputs into non-overlapping export roots.",
            "",
            "## Layout",
            "",
            "- `v1_legacy/`",
            "  - normalized export of legacy single-run SQL from `logs/runs/*/generated_sql.sql`",
            *current_layout_lines,
            "",
            "## Counts",
            "",
            f"- `v1_legacy`: `{len({r['run_id'] for r in v1_rows})}` runs, `{len({r['dataset_id'] for r in v1_rows})}` datasets, `{sum(int(r['sql_file_count']) for r in v1_rows)}` SQL files",
            *current_count_lines,
            "",
            "## Rule",
            "",
            "When you need current benchmark SQL, check the newest `v*_current/` export first.",
            "Only use `v1_legacy/` when you are explicitly tracing the older generation line.",
        ]
    )
    (EXPORT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    v1_rows = export_v1()
    current_rows_by_version = {version: export_current(version) for version in SUPPORTED_LINE_VERSIONS}
    write_root_readme(v1_rows, current_rows_by_version)
    print(f"Exported v1 rows: {len(v1_rows)}")
    for version, rows in current_rows_by_version.items():
        print(f"Exported {sql_source_label(version)} rows: {len(rows)}")
    print(f"Output root: {EXPORT_ROOT}")


if __name__ == "__main__":
    main()
