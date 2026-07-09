"""Export dataset-level and template-level legacy/v1 run distribution tables.

This task reads successful legacy/v1 run summaries under ``logs/runs`` and writes:

- dataset_distribution.csv / .tex
- template_distribution.csv / .tex

Outputs are stored under ``Evaluation/run_distribution/runs/<run_tag>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.common import DATA_ROOT, LOGS_ROOT, make_task_run_dir, now_run_tag, write_csv, write_json


ENGINE_CLI = "cli"
ENGINE_CLI_ALL = "cli-all"
SUPPORTED_ENGINES = {ENGINE_CLI, ENGINE_CLI_ALL}
TEMPLATE_LIBRARY_PATH = DATA_ROOT / "workload_grounding" / "template_library_v1.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CLI / CLI-All run distribution tables.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag for the output directory.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _best_score(summary_path: Path, payload: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(payload.get("completed_question_count") or 0),
        int(payload.get("question_count") or 0),
        int(summary_path.stat().st_mtime),
    )


def _load_best_run_summaries() -> dict[tuple[str, str], tuple[Path, dict[str, Any]]]:
    best: dict[tuple[str, str], tuple[tuple[int, int, int], Path, dict[str, Any]]] = {}
    for summary_path in LOGS_ROOT.rglob("batch_summary.json"):
        payload = _read_json(summary_path)
        if not isinstance(payload, dict):
            continue
        dataset_id = payload.get("dataset_id")
        engine = payload.get("engine")
        if not dataset_id or engine not in SUPPORTED_ENGINES:
            continue
        if payload.get("completed_question_count") is None:
            continue
        key = (str(dataset_id), str(engine))
        score = _best_score(summary_path, payload)
        current = best.get(key)
        if current is None or score > current[0]:
            best[key] = (score, summary_path, payload)
    return {key: (path, payload) for key, (_, path, payload) in best.items()}


def _load_template_library() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with TEMPLATE_LIBRARY_PATH.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            rows.append(
                {
                    "template_id": str(item.get("template_id") or ""),
                    "template_name": str(item.get("template_name") or item.get("title") or item.get("template_id") or ""),
                }
            )
    return rows


def _extract_dataset_row(dataset_id: str, payload: dict[str, Any] | None, summary_path: Path | None) -> dict[str, Any]:
    template_counts = dict(payload.get("template_problem_counts") or {}) if payload else {}
    cli_all_inventory = dict(payload.get("cli_all_inventory") or {}) if payload else {}
    if payload and payload.get("engine") == ENGINE_CLI_ALL:
        template_count = int(
            cli_all_inventory.get("selected_template_count")
            or len(template_counts)
            or 0
        )
        problem_count = int(
            cli_all_inventory.get("problem_count")
            or payload.get("question_count")
            or 0
        )
    else:
        template_count = int(len(template_counts))
        problem_count = int((payload or {}).get("question_count") or 0)
    sql_count = int((payload or {}).get("total_generated_sql_count") or 0)
    return {
        "dataset_id": dataset_id,
        "template_count": template_count,
        "problem_count": problem_count,
        "sql_count": sql_count,
        "valid_summary": bool(template_count or problem_count or sql_count),
        "summary_path": str(summary_path.resolve()) if summary_path else "",
    }


def build_dataset_distribution_rows(
    best_summaries: dict[tuple[str, str], tuple[Path, dict[str, Any]]]
) -> list[dict[str, Any]]:
    dataset_ids = sorted({dataset_id for dataset_id, _engine in best_summaries})
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        cli_path, cli_payload = best_summaries.get((dataset_id, ENGINE_CLI), (None, None))
        cli_all_path, cli_all_payload = best_summaries.get((dataset_id, ENGINE_CLI_ALL), (None, None))
        cli = _extract_dataset_row(dataset_id, cli_payload, cli_path)
        cli_all = _extract_dataset_row(dataset_id, cli_all_payload, cli_all_path)
        rows.append(
            {
                "dataset_id": dataset_id,
                "cli_templates": cli["template_count"],
                "cli_problems": cli["problem_count"],
                "cli_sql": cli["sql_count"],
                "cli_valid_summary": cli["valid_summary"],
                "cli_summary_path": cli["summary_path"],
                "cli_all_templates": cli_all["template_count"],
                "cli_all_problems": cli_all["problem_count"],
                "cli_all_sql": cli_all["sql_count"],
                "cli_all_valid_summary": cli_all["valid_summary"],
                "cli_all_summary_path": cli_all["summary_path"],
            }
        )
    return rows


def build_template_distribution_rows(
    best_summaries: dict[tuple[str, str], tuple[Path, dict[str, Any]]],
    template_library: list[dict[str, str]],
) -> list[dict[str, Any]]:
    per_engine: dict[str, dict[str, dict[str, int]]] = {
        ENGINE_CLI: {},
        ENGINE_CLI_ALL: {},
    }
    for engine in SUPPORTED_ENGINES:
        for template in template_library:
            per_engine[engine][template["template_id"]] = {"dataset_count": 0, "problem_count": 0}

    for (_dataset_id, engine), (_path, payload) in best_summaries.items():
        template_counts = dict(payload.get("template_problem_counts") or {})
        for template_id, raw_problem_count in template_counts.items():
            problem_count = int(raw_problem_count or 0)
            if problem_count <= 0:
                continue
            bucket = per_engine[engine].setdefault(template_id, {"dataset_count": 0, "problem_count": 0})
            bucket["dataset_count"] += 1
            bucket["problem_count"] += problem_count

    rows: list[dict[str, Any]] = []
    for template in template_library:
        template_id = template["template_id"]
        cli_stats = per_engine[ENGINE_CLI].get(template_id, {"dataset_count": 0, "problem_count": 0})
        cli_all_stats = per_engine[ENGINE_CLI_ALL].get(template_id, {"dataset_count": 0, "problem_count": 0})
        rows.append(
            {
                "template_id": template_id,
                "template_name": template["template_name"],
                "cli_dataset_count": cli_stats["dataset_count"],
                "cli_problem_count": cli_stats["problem_count"],
                "cli_all_dataset_count": cli_all_stats["dataset_count"],
                "cli_all_problem_count": cli_all_stats["problem_count"],
                "total_dataset_count": cli_stats["dataset_count"] + cli_all_stats["dataset_count"],
                "total_problem_count": cli_stats["problem_count"] + cli_all_stats["problem_count"],
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["total_dataset_count"]),
            -int(row["total_problem_count"]),
            str(row["template_id"]),
        )
    )
    return rows


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _render_longtable(
    *,
    caption: str,
    label: str,
    columns: list[tuple[str, str]],
    rows: list[dict[str, Any]],
) -> str:
    colspec = "".join(spec for _header, spec in columns)
    headers = " & ".join(_latex_escape(header) for header, _spec in columns) + r" \\"
    lines = [
        r"\begin{longtable}{" + colspec + "}",
        r"\caption{" + _latex_escape(caption) + r"}\label{" + _latex_escape(label) + r"}\\",
        r"\hline",
        headers,
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        headers,
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot",
        r"\hline",
        r"\endlastfoot",
    ]
    for row in rows:
        values = []
        for header, _spec in columns:
            key = _column_key_from_header(header)
            values.append(_latex_escape(row.get(key, "")))
        lines.append(" & ".join(values) + r" \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def _column_key_from_header(header: str) -> str:
    return header.lower().replace(" ", "_").replace("-", "_")


def write_tex_table(path: Path, *, caption: str, label: str, columns: list[tuple[str, str]], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "% Requires: \\usepackage{longtable}\n"
        + _render_longtable(caption=caption, label=label, columns=columns, rows=rows)
    )
    path.write_text(content, encoding="utf-8")


def run_export(*, run_tag: str) -> dict[str, Any]:
    run_dir = make_task_run_dir("run_distribution", run_tag)
    best_summaries = _load_best_run_summaries()
    template_library = _load_template_library()

    dataset_rows = build_dataset_distribution_rows(best_summaries)
    template_rows = build_template_distribution_rows(best_summaries, template_library)

    dataset_csv = run_dir / "dataset_distribution.csv"
    dataset_tex = run_dir / "dataset_distribution.tex"
    template_csv = run_dir / "template_distribution.csv"
    template_tex = run_dir / "template_distribution.tex"

    write_csv(
        dataset_csv,
        dataset_rows,
        fieldnames=[
            "dataset_id",
            "cli_templates",
            "cli_problems",
            "cli_sql",
            "cli_valid_summary",
            "cli_summary_path",
            "cli_all_templates",
            "cli_all_problems",
            "cli_all_sql",
            "cli_all_valid_summary",
            "cli_all_summary_path",
        ],
    )
    write_csv(
        template_csv,
        template_rows,
        fieldnames=[
            "template_id",
            "template_name",
            "cli_dataset_count",
            "cli_problem_count",
            "cli_all_dataset_count",
            "cli_all_problem_count",
            "total_dataset_count",
            "total_problem_count",
        ],
    )
    write_tex_table(
        dataset_tex,
        caption="Dataset-level CLI and CLI-All run distribution.",
        label="tab:dataset_run_distribution",
        columns=[
            ("dataset_id", "l"),
            ("cli_templates", "r"),
            ("cli_problems", "r"),
            ("cli_sql", "r"),
            ("cli_all_templates", "r"),
            ("cli_all_problems", "r"),
            ("cli_all_sql", "r"),
        ],
        rows=dataset_rows,
    )
    write_tex_table(
        template_tex,
        caption="Template-level CLI and CLI-All selection and problem distribution.",
        label="tab:template_run_distribution",
        columns=[
            ("template_id", "l"),
            ("template_name", "l"),
            ("cli_dataset_count", "r"),
            ("cli_problem_count", "r"),
            ("cli_all_dataset_count", "r"),
            ("cli_all_problem_count", "r"),
            ("total_dataset_count", "r"),
        ],
        rows=template_rows,
    )

    manifest = {
        "status": "ok",
        "run_tag": run_tag,
        "run_dir": str(run_dir.resolve()),
        "sql_source_version": "v1",
        "sql_source_label": "v1_legacy",
        "source_logs_root": str(LOGS_ROOT.resolve()),
        "dataset_row_count": len(dataset_rows),
        "template_row_count": len(template_rows),
        "outputs": {
            "dataset_csv": str(dataset_csv.resolve()),
            "dataset_tex": str(dataset_tex.resolve()),
            "template_csv": str(template_csv.resolve()),
            "template_tex": str(template_tex.resolve()),
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or now_run_tag()
    manifest = run_export(run_tag=run_tag)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
