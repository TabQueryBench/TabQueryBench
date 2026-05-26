#!/usr/bin/env python3
"""Render the combined tail-submetric preview as a native PGFPlots figure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_tail_metric_model_grid import _load_model_threshold, _metric_pivot, _model_order
from scripts.render_tail_stress_main_figure import DECOMP_COLORS


METRICS = [
    ("tail_set_consistency", "Tail set consistency"),
    ("tail_mass_similarity", "Tail mass similarity"),
    ("tail_concentration_consistency", "Tail concentration consistency"),
]

COLOR_NAMES = {
    "tail_set_consistency": "tailsetmetric",
    "tail_mass_similarity": "tailmassmetric",
    "tail_concentration_consistency": "tailconcmetric",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--embedded-output", type=Path, required=True)
    parser.add_argument("--standalone-output", type=Path)
    parser.add_argument("--paper-embedded-output", type=Path)
    return parser


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_embedded_tex(df: pd.DataFrame) -> str:
    model_order = _model_order(df)
    payloads: list[dict[str, object]] = []

    for metric, label in METRICS:
        pivot = _metric_pivot(df, metric, model_order)
        values = pivot.to_numpy(dtype=float)
        payloads.append(
            {
                "metric": metric,
                "label": label,
                "mean": np.nanmean(values, axis=1),
                "q25": np.nanquantile(values, 0.25, axis=1),
                "q75": np.nanquantile(values, 0.75, axis=1),
                "vmin": np.nanmin(values, axis=1),
                "vmax": np.nanmax(values, axis=1),
            }
        )

    offsets = [-0.24, 0.0, 0.24]
    box_width = 0.18

    lines: list[str] = []
    lines.append(r"\definecolor{tailsetmetric}{HTML}{" + DECOMP_COLORS["tail_set_consistency_mean"].lstrip("#") + "}")
    lines.append(r"\definecolor{tailmassmetric}{HTML}{" + DECOMP_COLORS["tail_mass_similarity_mean"].lstrip("#") + "}")
    lines.append(r"\definecolor{tailconcmetric}{HTML}{" + DECOMP_COLORS["tail_concentration_consistency_mean"].lstrip("#") + "}")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(r"    width=17.6cm,")
    lines.append(r"    height=6.9cm,")
    lines.append(r"    xmin=-0.65, xmax=" + _fmt(len(model_order) - 0.35) + ",")
    lines.append(r"    ymin=0.0, ymax=1.0,")
    lines.append(r"    xtick={0,...," + str(len(model_order) - 1) + r"},")
    lines.append(r"    xticklabels={" + ",".join(model_order) + r"},")
    lines.append(r"    xticklabel style={rotate=30, anchor=east, font=\small},")
    lines.append(r"    yticklabel style={font=\small},")
    lines.append(r"    xlabel={Model},")
    lines.append(r"    ylabel={Score},")
    lines.append(r"    axis line style={draw=gray!65, line width=0.8pt},")
    lines.append(r"    tick style={black, line width=0.8pt},")
    lines.append(r"    grid=major,")
    lines.append(r"    major grid style={draw=gray!22, dashed},")
    lines.append(r"    axis background/.style={fill=white},")
    lines.append(r"    label style={font=\small},")
    lines.append(r"    clip=false,")
    lines.append(r"]")

    for boundary in range(len(model_order) - 1):
        x = boundary + 0.5
        lines.append(
            r"\draw[gray!22, line width=0.6pt] (axis cs:"
            + _fmt(x)
            + r",0.0) -- (axis cs:"
            + _fmt(x)
            + r",1.0);"
        )

    for payload, offset in zip(payloads, offsets):
        metric = str(payload["metric"])
        color_name = COLOR_NAMES[metric]
        means = np.asarray(payload["mean"], dtype=float)
        q25 = np.asarray(payload["q25"], dtype=float)
        q75 = np.asarray(payload["q75"], dtype=float)
        vmin = np.asarray(payload["vmin"], dtype=float)
        vmax = np.asarray(payload["vmax"], dtype=float)

        for idx in range(len(model_order)):
            if np.isnan(means[idx]):
                continue
            xpos = idx + offset
            left = xpos - box_width / 2.0
            right = xpos + box_width / 2.0
            low = vmin[idx]
            high = vmax[idx]
            cap_left = xpos - 0.045
            cap_right = xpos + 0.045
            box_bottom = q25[idx]
            box_top = max(q25[idx] + 1e-6, q75[idx])
            mean = means[idx]

            lines.append(
                r"\path[fill="
                + color_name
                + r", fill opacity=0.28, draw=none] (axis cs:"
                + _fmt(left)
                + ","
                + _fmt(box_bottom)
                + r") rectangle (axis cs:"
                + _fmt(right)
                + ","
                + _fmt(box_top)
                + r");"
            )
            lines.append(
                r"\draw["
                + color_name
                + r", line width=1.15pt] (axis cs:"
                + _fmt(xpos)
                + ","
                + _fmt(low)
                + r") -- (axis cs:"
                + _fmt(xpos)
                + ","
                + _fmt(high)
                + r");"
            )
            lines.append(
                r"\draw["
                + color_name
                + r", line width=1.15pt] (axis cs:"
                + _fmt(cap_left)
                + ","
                + _fmt(low)
                + r") -- (axis cs:"
                + _fmt(cap_right)
                + ","
                + _fmt(low)
                + r");"
            )
            lines.append(
                r"\draw["
                + color_name
                + r", line width=1.15pt] (axis cs:"
                + _fmt(cap_left)
                + ","
                + _fmt(high)
                + r") -- (axis cs:"
                + _fmt(cap_right)
                + ","
                + _fmt(high)
                + r");"
            )
            lines.append(
                r"\addplot[only marks, mark=square*, mark size=2.2pt, color="
                + color_name
                + r"] coordinates {("
                + _fmt(xpos)
                + ","
                + _fmt(mean)
                + r")};"
            )

    lines.append(
        r"\node[anchor=west, font=\small] at (rel axis cs:0.01,1.035) {"
        r"\textcolor{tailsetmetric}{Tail set consistency}"
        r"\quad"
        r"\textcolor{tailmassmetric}{Tail mass similarity}"
        r"\quad"
        r"\textcolor{tailconcmetric}{Tail concentration consistency}"
        r"};"
    )
    lines.append(
        r"\node[anchor=east, font=\small, text=gray!70!black] at (rel axis cs:0.995,1.035) {"
        r"square = mean \quad whisker = min--max \quad box = IQR"
        r"};"
    )
    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    lines.append("")
    return "\n".join(lines)


def _build_standalone_tex(embedded_name: str) -> str:
    return "\n".join(
        [
            r"\documentclass[tikz,border=6pt]{standalone}",
            r"\usepackage{pgfplots}",
            r"\pgfplotsset{compat=1.18}",
            r"\begin{document}",
            r"\input{" + embedded_name + r"}",
            r"\end{document}",
            "",
        ]
    )


def main() -> int:
    args = _build_parser().parse_args()
    df = _load_model_threshold(args.tables_dir)
    embedded_tex = _build_embedded_tex(df)
    _write(args.embedded_output, embedded_tex)

    if args.paper_embedded_output:
        _write(args.paper_embedded_output, embedded_tex)

    if args.standalone_output:
        standalone_tex = _build_standalone_tex(args.embedded_output.name)
        _write(args.standalone_output, standalone_tex)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
