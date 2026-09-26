from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqb_scoring.eval.common import list_dataset_ids, load_latest_sql_queries


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT.parent.parent / "Query" / "code" / "data"
OUTPUT_ROOT = DATA_ROOT / "sql_result_role_annotations_v1"
VERSIONS = ("v2", "v3", "v4")
SHARD_COUNT = 10


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dataset_sort_key(dataset_id: str) -> tuple[str, int, str]:
    text = str(dataset_id or "").strip().lower()
    prefix = text[:1]
    suffix = text[1:]
    try:
        number = int(suffix)
    except Exception:
        number = 10**9
    return prefix, number, text


def _query_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_key": f"{row.get('sql_source_version','')}::{row.get('query_id','')}",
        "sql_source_version": row.get("sql_source_version"),
        "sql_source_label": row.get("sql_source_label"),
        "query_id": row.get("query_id"),
        "question_id": row.get("question_id"),
        "stable_question_id": row.get("stable_question_id"),
        "query_identity_stable_key": row.get("query_identity_stable_key"),
        "template_id": row.get("template_id"),
        "template_name": row.get("template_name"),
        "family_id": row.get("family_id"),
        "intended_facet_id": row.get("intended_facet_id"),
        "variant_semantic_role": row.get("variant_semantic_role"),
        "sql_model": row.get("model"),
        "sql_engine": row.get("engine"),
        "sql_origin_path": row.get("sql_origin_path"),
        "sql": row.get("sql"),
    }


def _output_template(dataset_id: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "annotation_contract_version": "sql_result_role_annotation_v1",
        "annotation_scope": "query_result_columns",
        "query_annotations": [
            {
                "annotation_key": "",
                "sql_source_version": "",
                "query_id": "",
                "query_identity_stable_key": "",
                "family_id": "",
                "template_id": "",
                "template_name": "",
                "result_column_roles": [
                    {
                        "column_name": "",
                        "role": "key",
                        "reason": "",
                    }
                ],
                "result_key_columns": [],
                "result_measure_columns": [],
                "reasoning_short": "",
                "confidence": 0.0,
                "needs_review": False,
            }
        ],
    }


def _top_level_readme() -> str:
    return "\n".join(
        [
            "# SQL Result Role Annotations v1",
            "",
            "This directory scaffolds AI-assisted labeling for SQL result-column roles.",
            "",
            "Goal:",
            "- produce deterministic `result_key_columns` and `result_measure_columns` for query-result comparison",
            "- replace regex-only key guessing in `key_set_score` / `profile_score` with explicit annotations",
            "",
            "Structure:",
            "- `README.md`: overview",
            "- `schema/annotation_contract_v1.md`: output contract and labeling rules",
            "- `registry/dataset_registry.json`: all dataset manifests",
            "- `registry/shards/shard_XX.json`: ten balanced shard lists for parallel annotation",
            "- `datasets/<dataset_id>/input_manifest.json`: dataset-level context and file pointers",
            "- `datasets/<dataset_id>/queries/queries_v2.jsonl`: version-specific query inputs",
            "- `datasets/<dataset_id>/outputs/sql_result_roles_ai_v1.json`: target output file to be filled by AI",
            "- `prompts/codex_cli_gpt54_parallel_prompt.md`: ready-to-paste prompt for a new Codex window",
            "",
            "Important rules:",
            "- prefer SQL structure and dataset semantics over column-name heuristics",
            "- label output columns in result-order",
            "- treat aggregated numeric outputs like `count`, `avg`, `rate`, `ratio`, `support` as measure columns unless there is a dataset-specific reason otherwise",
            "- use `needs_review=true` when a query is genuinely ambiguous",
            "",
        ]
    )


def _contract_doc() -> str:
    return "\n".join(
        [
            "# Annotation Contract v1",
            "",
            "Each dataset output file should be valid JSON and follow this shape:",
            "",
            "```json",
            "{",
            '  "dataset_id": "c2",',
            '  "annotation_contract_version": "sql_result_role_annotation_v1",',
            '  "annotation_scope": "query_result_columns",',
            '  "query_annotations": [',
            "    {",
            '      "annotation_key": "v2::v2q_c2_xxx",',
            '      "sql_source_version": "v2",',
            '      "query_id": "v2q_c2_xxx",',
            '      "query_identity_stable_key": "c2::v2q_c2_xxx",',
            '      "family_id": "conditional_dependency_structure",',
            '      "template_id": "tpl_xxx",',
            '      "template_name": "Filtered Two-Dimensional Group Count",',
            '      "result_column_roles": [',
            '        {"column_name": "buying", "role": "key", "reason": "GROUP BY dimension"},',
            '        {"column_name": "lug_boot", "role": "key", "reason": "GROUP BY dimension"},',
            '        {"column_name": "row_count", "role": "measure", "reason": "COUNT(*) aggregate"}',
            "      ],",
            '      "result_key_columns": ["buying", "lug_boot"],',
            '      "result_measure_columns": ["row_count"],',
            '      "reasoning_short": "Two grouping columns define the support tuples; count is the aggregate measure.",',
            '      "confidence": 0.98,',
            '      "needs_review": false',
            "    }",
            "  ]",
            "}",
            "```",
            "",
            "Labeling rules:",
            "- `key`: result columns that define the structural support tuples being compared across real vs synthetic outputs.",
            "- `measure`: aggregated or derived value columns whose magnitudes summarize each key tuple.",
            "- preserve result order in `result_column_roles`.",
            "- if a query returns only structural columns and no aggregate measure, leave `result_measure_columns` empty.",
            "- if a query returns an ordering/helper column that is still a derived numeric summary, mark it as `measure`.",
            "- prefer SQL semantics (`GROUP BY`, aggregate expressions, window outputs, aliases) over raw column names.",
            "- consult dataset semantics and field registry when deciding whether a selected column is a structural dimension.",
            "- set `needs_review=true` when the query is ambiguous or the role split is not trustworthy.",
            "",
        ]
    )


def _prompt_text() -> str:
    return "\n".join(
        [
            "你现在在 `D:\\dpan\\Uni\\Project\\HKUNAISS\\SQLagent` 仓库里工作。",
            "",
            "目标：完成 `data/sql_result_role_annotations_v1/` 下面 49 个 dataset 的 AI 标注，产出每个 dataset 的 `outputs/sql_result_roles_ai_v1.json`。",
            "",
            "要求：",
            "1. 使用 GPT-5.4。",
            "2. 并行 10 个任务处理 49 个数据集。",
            "3. 只处理 `data/sql_result_role_annotations_v1/registry/shards/` 里定义的 shard；每个 shard 内按 manifest 完成对应 datasets。",
            "4. 每个 dataset 都读取：",
            "   - `input_manifest.json`",
            "   - `queries/queries_v2.jsonl`",
            "   - `queries/queries_v3.jsonl`",
            "   - `queries/queries_v4.jsonl`",
            "   - `data/<dataset_id>/metadata_core/field_registry.json`",
            "   - `data/<dataset_id>/metadata_core/dataset_semantics.yaml`",
            "   - `data/<dataset_id>/metadata_core/query_policy.yaml`（如果存在）",
            "5. 你要为每条 query 标注：",
            "   - `result_column_roles`",
            "   - `result_key_columns`",
            "   - `result_measure_columns`",
            "   - `reasoning_short`",
            "   - `confidence`",
            "   - `needs_review`",
            "6. 判断原则：优先使用 SQL 结构语义（GROUP BY、aggregate、window output、alias），不要只靠列名猜。",
            "7. 输出必须严格符合 `data/sql_result_role_annotations_v1/schema/annotation_contract_v1.md`。",
            "8. 不要改 evaluation 代码；只写这些 JSON 标注文件，必要时可以额外写每个 dataset 的简短 notes 文件。",
            "9. 先完成全部标注文件，再统一做一次轻量校验：",
            "   - JSON 可解析",
            "   - 每条 query 都有 annotation",
            "   - `result_key_columns` 与 `result_measure_columns` 不重叠",
            "10. 如果某些 query 明显无法可靠判断，保留输出，但设 `needs_review=true` 并解释原因。",
            "",
            "工作建议：",
            "- 以 shard 为并行单元，一共 10 个。",
            "- 每个 shard 完成后立刻写回对应 dataset 的输出文件，不要把 49 个结果都攒到最后。",
            "- 对明显模板化的 query，保持同一 dataset 内前后一致。",
            "",
            "开始前先快速阅读：",
            "- `data/sql_result_role_annotations_v1/README.md`",
            "- `data/sql_result_role_annotations_v1/schema/annotation_contract_v1.md`",
            "- `data/sql_result_role_annotations_v1/registry/dataset_registry.json`",
            "",
            "完成后请给出：",
            "- 哪些 datasets 已完成",
            "- 哪些 query 被标成 `needs_review=true`",
            "- 任何你发现的结构性歧义",
            "",
        ]
    )


def build_scaffold(output_root: Path) -> dict[str, Any]:
    dataset_ids = sorted(list_dataset_ids(), key=_dataset_sort_key)

    datasets_root = output_root / "datasets"
    registry_root = output_root / "registry"
    shards_root = registry_root / "shards"
    prompts_root = output_root / "prompts"
    schema_root = output_root / "schema"

    _write_text(output_root / "README.md", _top_level_readme())
    _write_text(schema_root / "annotation_contract_v1.md", _contract_doc())
    _write_text(prompts_root / "codex_cli_gpt54_parallel_prompt.md", _prompt_text())

    dataset_registry: list[dict[str, Any]] = []
    shard_buckets: list[list[str]] = [[] for _ in range(SHARD_COUNT)]

    for index, dataset_id in enumerate(dataset_ids):
        shard_buckets[index % SHARD_COUNT].append(dataset_id)
        data_dir = DATA_ROOT / dataset_id
        dataset_root = datasets_root / dataset_id
        queries_root = dataset_root / "queries"
        outputs_root = dataset_root / "outputs"

        context = {
            "dataset_id": dataset_id,
            "data_dir": str(data_dir.resolve()),
            "train_csv_path": str((data_dir / f"{dataset_id}-train.csv").resolve()),
            "main_csv_path": str((data_dir / f"{dataset_id}-main.csv").resolve()),
            "metadata_paths": {
                "field_registry": str((data_dir / "metadata_core" / "field_registry.json").resolve()),
                "dataset_semantics": str((data_dir / "metadata_core" / "dataset_semantics.yaml").resolve()),
                "dataset_description": str((data_dir / "metadata_core" / "dataset_description.txt").resolve()),
                "query_policy": str((data_dir / "metadata_core" / "query_policy.yaml").resolve()),
                "family_applicability": str((data_dir / "metadata_optional" / "family_applicability.json").resolve()),
                "source_info": str((data_dir / "source" / "source_info.json").resolve()),
            },
            "output_path": str((outputs_root / "sql_result_roles_ai_v1.json").resolve()),
            "versions": {},
        }

        for version in VERSIONS:
            queries_path = queries_root / f"queries_{version}.jsonl"
            error_note = None
            payloads: list[dict[str, Any]] = []
            try:
                rows = load_latest_sql_queries(
                    dataset_id=dataset_id,
                    engines=("cli",),
                    include_all_statements=True,
                    sql_source_version=version,
                )
                payloads = [_query_payload(row) for row in rows]
            except Exception as exc:  # noqa: BLE001
                error_note = str(exc)
            queries_root.mkdir(parents=True, exist_ok=True)
            with queries_path.open("w", encoding="utf-8") as handle:
                for row in payloads:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            context["versions"][version] = {
                "query_count": len(payloads),
                "queries_path": str(queries_path.resolve()),
                "load_error": error_note,
            }

        _write_json(dataset_root / "input_manifest.json", context)
        _write_json(outputs_root / "output_template.json", _output_template(dataset_id))

        dataset_registry.append(
            {
                "dataset_id": dataset_id,
                "input_manifest_path": str((dataset_root / "input_manifest.json").resolve()),
                "output_path": context["output_path"],
                "versions": context["versions"],
            }
        )

    _write_json(registry_root / "dataset_registry.json", dataset_registry)

    shard_payloads: list[dict[str, Any]] = []
    for index, dataset_bucket in enumerate(shard_buckets, start=1):
        shard_name = f"shard_{index:02d}"
        payload = {
            "shard_name": shard_name,
            "dataset_count": len(dataset_bucket),
            "dataset_ids": dataset_bucket,
            "input_manifest_paths": [
                str((datasets_root / dataset_id / "input_manifest.json").resolve())
                for dataset_id in dataset_bucket
            ],
        }
        _write_json(shards_root / f"{shard_name}.json", payload)
        shard_payloads.append(payload)

    _write_json(registry_root / "shard_registry.json", shard_payloads)

    return {
        "output_root": str(output_root.resolve()),
        "dataset_count": len(dataset_ids),
        "shard_count": SHARD_COUNT,
        "dataset_registry_path": str((registry_root / "dataset_registry.json").resolve()),
        "prompt_path": str((prompts_root / "codex_cli_gpt54_parallel_prompt.md").resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scaffold for AI SQL result role annotations.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_scaffold(args.output_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
