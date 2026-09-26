"""Lightweight table exporters for v2 CSV artifacts."""

from __future__ import annotations

import csv
from pathlib import Path


def csv_to_latex_table(
    *,
    csv_path: Path,
    tex_path: Path,
    caption: str,
    label: str,
) -> None:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    headers = list(rows[0].keys()) if rows else []
    body_lines = []
    for row in rows:
        values = [str(row.get(header, "")).replace("_", "\\_") for header in headers]
        body_lines.append(" & ".join(values) + r" \\")

    column_spec = "l" * max(1, len(headers))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(header.replace("_", r"\_") for header in headers) + r" \\",
        r"\hline",
        *body_lines,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
