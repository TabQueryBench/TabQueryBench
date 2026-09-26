"""Run artifact writer for per-run trace and outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunArtifactWriter:
    def __init__(self, runs_root: Path, run_id: str) -> None:
        self.run_id = run_id
        self.run_dir = runs_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._sql_header_metadata: dict[str, str] = {}

        self.trace_path = self.run_dir / "trace.jsonl"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.final_answer_path = self.run_dir / "final_answer.txt"
        self.generated_sql_path = self.run_dir / "generated_sql.sql"
        self.query_results_path = self.run_dir / "query_results.jsonl"
        self.usage_summary_path = self.run_dir / "usage_summary.json"

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def append_trace(self, event: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_final_answer(self, text: str) -> None:
        self.final_answer_path.write_text(text, encoding="utf-8")

    def set_sql_header_metadata(self, metadata: dict[str, Any] | None) -> None:
        cleaned: dict[str, str] = {}
        for key, value in (metadata or {}).items():
            key_text = str(key or "").strip()
            value_text = str(value or "").strip()
            if not key_text or not value_text:
                continue
            cleaned[key_text] = value_text
        self._sql_header_metadata = cleaned

    def _apply_sql_header_metadata(self, sql_text: str) -> str:
        if not self._sql_header_metadata:
            return sql_text
        body_lines = str(sql_text or "").strip().splitlines()
        while body_lines:
            stripped = body_lines[0].strip()
            if not stripped.startswith("--"):
                break
            lowered = stripped.lower()
            if any(lowered.startswith(f"-- {field.lower()}:") for field in self._sql_header_metadata):
                body_lines.pop(0)
                continue
            break
        sql_body = "\n".join(body_lines).strip()
        header_lines = [f"-- {field}: {value}" for field, value in self._sql_header_metadata.items()]
        if not header_lines:
            return sql_body
        if not sql_body:
            return "\n".join(header_lines) + "\n"
        return "\n".join(header_lines + [sql_body])

    def write_generated_sql(self, sql_list: list[str]) -> None:
        if not sql_list:
            self.generated_sql_path.write_text("", encoding="utf-8")
            return
        content = ";\n\n".join(s.rstrip(";\n ") for s in sql_list if s.strip()) + ";\n"
        content = self._apply_sql_header_metadata(content)
        if content and not content.endswith("\n"):
            content += "\n"
        self.generated_sql_path.write_text(content, encoding="utf-8")

    def write_query_results(self, rows: list[dict[str, Any]]) -> None:
        with self.query_results_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_usage_summary(self, usage_summary: dict[str, Any]) -> None:
        with self.usage_summary_path.open("w", encoding="utf-8") as f:
            json.dump(usage_summary, f, ensure_ascii=False, indent=2)

    def write_json(self, relative_path: str, payload: Any) -> Path:
        output_path = self.run_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return output_path

    def write_jsonl(self, relative_path: str, rows: list[dict[str, Any]]) -> Path:
        output_path = self.run_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return output_path

    def write_text(self, relative_path: str, content: str) -> Path:
        output_path = self.run_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path
