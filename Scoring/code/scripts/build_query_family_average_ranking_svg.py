#!/usr/bin/env python3
"""Build an editable SVG ranking chart for average query fidelity.

The chart computes the mean of the five query-family scores:
Subgroup, Conditional, Tail, Missingness, Cardinality.

Input source:
  tmp_remote_paper/figures/time_cost/final/model_global_summary_generated.tex

Style source:
  tqb_scoring/eval/overview_regenerated/runner.py
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from statistics import mean
from typing import Any
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_TEX = ROOT / "tmp_remote_paper/figures/time_cost/final/model_global_summary_generated.tex"
STYLE_SOURCE = ROOT / "tqb_scoring/eval/overview_regenerated/runner.py"
DEFAULT_OUTPUT = ROOT / "tmp/query_family_average_ranking.svg"

FAMILY_COLUMNS = {
    "Subgroup": 4,
    "Conditional": 5,
    "Tail": 6,
    "Missingness": 7,
    "Cardinality": 8,
}

MEDAL_COLORS = {
    1: ("#d8a321", "#fff4cf"),
    2: ("#7b8597", "#eef2f8"),
    3: ("#b56a3a", "#f7e3d5"),
}


def parse_runner_literals(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text())
    wanted = {"MODEL_LABELS", "MODEL_SHORT_LABELS", "MODEL_COLORS"}
    found: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"Missing constants in {path}: {sorted(missing)}")
    return found


def parse_float(token: str) -> float:
    cleaned = re.sub(r"[^0-9.\-]+", "", token)
    return float(cleaned)


def parse_summary_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if "&" not in line or not line.endswith(r"\\"):
            continue
        if line.startswith(("\\toprule", "\\midrule", "\\bottomrule", "Model ")):
            continue

        parts = [part.strip() for part in line.split("&")]
        if len(parts) < 9:
            continue
        if parts[0] == "Model" or parts[0].startswith("\\"):
            continue

        family_scores = {
            family: parse_float(parts[index].replace("\\", "").strip())
            for family, index in FAMILY_COLUMNS.items()
        }
        rows.append(
            {
                "model_label": parts[0],
                "family_scores": family_scores,
                "average_score": mean(family_scores.values()),
            }
        )
    if not rows:
        raise RuntimeError(f"No data rows parsed from {path}")
    return rows


def build_inverse_label_map(model_labels: dict[str, str]) -> dict[str, str]:
    inverse: dict[str, str] = {}
    for model_id, label in model_labels.items():
        inverse[label.lower()] = model_id
    return inverse


def with_model_style(
    rows: list[dict[str, Any]],
    model_labels: dict[str, str],
    model_short_labels: dict[str, str],
    model_colors: dict[str, str],
) -> list[dict[str, Any]]:
    inverse = build_inverse_label_map(model_labels)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        key = row["model_label"].lower()
        model_id = inverse.get(key)
        if model_id is None:
            raise RuntimeError(f"Could not map model label to model_id: {row['model_label']}")
        enriched.append(
            {
                **row,
                "model_id": model_id,
                "display_label": model_labels[model_id],
                "short_label": model_short_labels.get(model_id, model_labels[model_id]),
                "color": model_colors[model_id],
            }
        )
    enriched.sort(key=lambda item: item["average_score"], reverse=True)
    for rank, row in enumerate(enriched, start=1):
        row["rank"] = rank
    return enriched


def rect(x: float, y: float, width: float, height: float, rx: float, fill: str, stroke: str | None = None, stroke_width: float = 1.0, opacity: float | None = None) -> str:
    attrs = [
        f'x="{x:.1f}"',
        f'y="{y:.1f}"',
        f'width="{width:.1f}"',
        f'height="{height:.1f}"',
        f'rx="{rx:.1f}"',
        f'fill="{fill}"',
    ]
    if stroke:
        attrs.append(f'stroke="{stroke}"')
        attrs.append(f'stroke-width="{stroke_width:.1f}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity:.3f}"')
    return f"<rect {' '.join(attrs)} />"


def text(x: float, y: float, value: str, size: float, fill: str, weight: str = "400", anchor: str = "start", letter_spacing: float | None = None) -> str:
    extra = f' letter-spacing="{letter_spacing:.2f}"' if letter_spacing is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size:.1f}" font-weight="{weight}" fill="{fill}"{extra}>{escape(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, stroke_width: float = 1.0, dasharray: str | None = None) -> str:
    dash = f' stroke-dasharray="{dasharray}"' if dasharray else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}"{dash} />'
    )


def circle(cx: float, cy: float, r: float, fill: str, stroke: str | None = None, stroke_width: float = 1.0) -> str:
    attrs = [f'cx="{cx:.1f}"', f'cy="{cy:.1f}"', f'r="{r:.1f}"', f'fill="{fill}"']
    if stroke:
        attrs.append(f'stroke="{stroke}"')
        attrs.append(f'stroke-width="{stroke_width:.1f}"')
    return f"<circle {' '.join(attrs)} />"


def build_svg(rows: list[dict[str, Any]], width: int = 760, transparent: bool = True) -> str:
    bg = "none" if transparent else "#ffffff"
    row_h = 28
    chart_top = 72
    chart_bottom_pad = 18
    height = chart_top + row_h * len(rows) + chart_bottom_pad

    left_pad = 18
    chart_x = 224
    chart_w = width - chart_x - 34
    bar_h = 17

    title_color = "#4b4ed8"
    text_color = "#1a1f2c"
    muted = "#6d7488"
    max_score = max(row["average_score"] for row in rows)

    pieces = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Average query fidelity ranking</title>",
        "<desc id=\"desc\">Horizontal ranking chart showing each model's average query fidelity across subgroup, conditional, tail, missingness, and cardinality families.</desc>",
        "<style>",
        "text { font-family: Arial, \"Helvetica Neue\", Helvetica, sans-serif; dominant-baseline: middle; }",
        ".label { fill: #1a1f2c; font-weight: 600; }",
        ".muted { fill: #6d7488; }",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}" />',
        text(left_pad, 28, "5-Family Average Ranking", 24, title_color, weight="700"),
    ]

    for row in rows:
        rank = row["rank"]
        row_y = chart_top + (rank - 1) * row_h
        bar_y = row_y - bar_h / 2
        bar_len = chart_w * (row["average_score"] / max_score)
        medal = MEDAL_COLORS.get(rank)

        if medal:
            medal_stroke, medal_fill = medal
            pieces.append(circle(30, row_y, 13, medal_fill, medal_stroke, 2.0))
            pieces.append(text(30, row_y + 0.5, str(rank), 13, medal_stroke, weight="700", anchor="middle"))
        else:
            pieces.append(circle(30, row_y, 12, "#f4f6fb", "#cbd3e4", 1.3))
            pieces.append(text(30, row_y + 0.5, str(rank), 12, "#6d7488", weight="700", anchor="middle"))

        pieces.append(text(56, row_y + 0.5, row["short_label"], 15, text_color, weight="600"))

        if medal:
            medal_stroke, _ = medal
            pieces.append(rect(chart_x, bar_y, bar_len, bar_h, 8, row["color"], medal_stroke, 2.0))
        else:
            pieces.append(rect(chart_x, bar_y, bar_len, bar_h, 8, row["color"]))

        pieces.append(text(chart_x + bar_len + 10, row_y + 0.5, f"{row['average_score']:.3f}", 13.5, text_color, weight="700"))
    pieces.append("</svg>")
    return "\n".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SVG ranking chart for average query fidelity.")
    parser.add_argument("--summary-tex", type=Path, default=SUMMARY_TEX)
    parser.add_argument("--style-source", type=Path, default=STYLE_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=760)
    parser.add_argument("--opaque-background", action="store_true", help="Fill the SVG background with white instead of transparent.")
    args = parser.parse_args()

    literals = parse_runner_literals(args.style_source)
    rows = parse_summary_rows(args.summary_tex)
    ranked_rows = with_model_style(rows, literals["MODEL_LABELS"], literals["MODEL_SHORT_LABELS"], literals["MODEL_COLORS"])
    svg = build_svg(ranked_rows, width=args.width, transparent=not args.opaque_background)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"Wrote {args.output}")
    print("Top 3:")
    for row in ranked_rows[:3]:
        print(f"  {row['rank']}. {row['display_label']} - {row['average_score']:.4f}")


if __name__ == "__main__":
    main()
