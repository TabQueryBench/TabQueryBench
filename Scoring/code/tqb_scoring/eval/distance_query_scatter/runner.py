"""Scatter plots comparing JSD distance against SQL-family scores."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqb_scoring.eval.common import (
    DEFAULT_SQL_SOURCE_VERSION,
    PROVENANCE_CONTRACT_VERSION,
    make_task_run_dir,
    normalize_sql_source_version,
    now_run_tag,
    read_json,
    sql_source_family,
    sql_source_label,
    sql_source_line_version,
    write_csv,
    write_json,
)
from tqb_scoring.eval.final_outputs import (
    copy_files,
    find_latex_compiler,
    task_version_final_dir,
    write_json as write_final_json,
    write_versioned_final_readme,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT.parent / "results"
TASK_NAME = "distance_query_scatter"
FINAL_TASK_NAME = "distance_query_scatter"

EXCLUDED_MODELS = {
    "cdtd",
    "codi",
    "goggle",
}

MODEL_LABELS = {
    "arf": "ARF",
    "bayesnet": "BayesNet",
    "ctgan": "CTGAN",
    "forestdiffusion": "ForestDiffusion",
    "realtabformer": "RealTabFormer",
    "rtf": "RealTabFormer",
    "tabbyflow": "TabbyFlow",
    "tabddpm": "TabDDPM",
    "tabdiff": "TabDiff",
    "tabpfgen": "TabPFGen",
    "tabsyn": "TabSyn",
    "tvae": "TVAE",
}

MODEL_ALIASES = {
    "rtf": "realtabformer",
}

MODEL_COLORS = {
    "realtabformer": "#332288",
    "tvae": "#4477AA",
    "forestdiffusion": "#228833",
    "tabddpm": "#EE7733",
    "tabsyn": "#66CCEE",
    "tabdiff": "#AA3377",
    "ctgan": "#EE6677",
    "arf": "#777777",
    "bayesnet": "#CCBB44",
    "tabpfgen": "#009988",
    "tabbyflow": "#882255",
}

FAMILY_SPECS = [
    ("subgroup_structure", "subgroup", "Subgroup score", "Subgroup"),
    ("conditional_dependency_structure", "conditional", "Conditional score", "Conditional"),
    ("tail_rarity_structure", "tail", "Tail score", "Tail"),
]

DATASET_GROUP_SPECS = [
    ("c", "C datasets", "square*"),
    ("m", "M datasets", "x"),
    ("n", "N datasets", "triangle*"),
]

VERSION_SPECS = [
    ("grouped_lines", "Grouped Means with Connecting Lines"),
    ("tradeoff_errorbars", "Trade-off Scatter with Bidirectional Error Bars"),
]


def _attach_context(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**context, **row} for row in rows]


@dataclass
class PrefixPoint:
    family_id: str
    family_slug: str
    family_title: str
    model_id: str
    model_label: str
    model_color: str
    dataset_group: str
    dataset_group_label: str
    mark: str
    mean_family_score: float
    mean_jsd: float
    matched_dataset_count: int
    dataset_ids: list[str]

    def to_row(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "family_slug": self.family_slug,
            "family_title": self.family_title,
            "model_id": self.model_id,
            "model_label": self.model_label,
            "model_color": self.model_color,
            "dataset_group": self.dataset_group,
            "dataset_group_label": self.dataset_group_label,
            "mark": self.mark,
            "mean_family_score": round(self.mean_family_score, 6),
            "mean_jsd": round(self.mean_jsd, 6),
            "matched_dataset_count": self.matched_dataset_count,
            "dataset_ids": ",".join(self.dataset_ids),
        }


@dataclass
class TradeoffPoint:
    family_id: str
    family_slug: str
    family_title: str
    model_id: str
    model_label: str
    model_color: str
    center_family_score: float
    center_jsd: float
    x_error: float
    y_error: float
    group_count: int
    groups_present: list[str]

    def to_row(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "family_slug": self.family_slug,
            "family_title": self.family_title,
            "model_id": self.model_id,
            "model_label": self.model_label,
            "model_color": self.model_color,
            "center_family_score": round(self.center_family_score, 6),
            "center_jsd": round(self.center_jsd, 6),
            "x_error": round(self.x_error, 6),
            "y_error": round(self.y_error, 6),
            "group_count": self.group_count,
            "groups_present": ",".join(self.groups_present),
        }


@dataclass
class PrefixTradeoffPoint:
    family_id: str
    family_slug: str
    family_title: str
    model_id: str
    model_label: str
    model_color: str
    dataset_group: str
    dataset_group_label: str
    center_family_score: float
    center_jsd: float
    x_error: float
    y_error: float
    dataset_count: int
    dataset_ids: list[str]

    def to_row(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "family_slug": self.family_slug,
            "family_title": self.family_title,
            "model_id": self.model_id,
            "model_label": self.model_label,
            "model_color": self.model_color,
            "dataset_group": self.dataset_group,
            "dataset_group_label": self.dataset_group_label,
            "center_family_score": round(self.center_family_score, 6),
            "center_jsd": round(self.center_jsd, 6),
            "x_error": round(self.x_error, 6),
            "y_error": round(self.y_error, 6),
            "dataset_count": self.dataset_count,
            "dataset_ids": ",".join(self.dataset_ids),
        }


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_model_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    return MODEL_ALIASES.get(text, text)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _model_color(model_id: str) -> str | None:
    return MODEL_COLORS.get(_normalize_model_id(model_id))


def _dataset_group(dataset_id: str) -> str | None:
    text = str(dataset_id or "").strip().lower()
    if not text:
        return None
    prefix = text[:1]
    return prefix if prefix in {"c", "m", "n"} else None


def _group_label(group_key: str) -> str:
    mapping = {key: label for key, label, _ in DATASET_GROUP_SPECS}
    return mapping[group_key]


def _group_mark(group_key: str) -> str:
    mapping = {key: mark for key, _, mark in DATASET_GROUP_SPECS}
    return mapping[group_key]


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na", "<null>"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:
        return datetime.min


def _model_sort_key(model_id: str) -> tuple[int, str]:
    normalized = _normalize_model_id(model_id)
    if normalized == "real":
        return (0, "")
    return (1, _model_label(normalized).lower(), normalized)


def _dataset_sort_key(dataset_id: str) -> tuple[str, int, str]:
    text = str(dataset_id or "").strip().lower()
    prefix = text[:1]
    suffix = text[1:]
    try:
        numeric = int(suffix)
    except Exception:
        numeric = 10**9
    return (prefix, numeric, text)


def _resolve_existing_run_dir(task_name: str, explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Explicit {task_name} run dir does not exist: {candidate}")

    latest_path = OUTPUT_ROOT / task_name / "LATEST_RUN.json"
    if latest_path.exists():
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        run_dir_value = str(payload.get("run_dir") or "").strip()
        if run_dir_value:
            candidate = Path(run_dir_value)
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
            if candidate.exists():
                return candidate
        run_tag = str(payload.get("run_tag") or "").strip()
        if run_tag:
            candidate = (OUTPUT_ROOT / task_name / "runs" / run_tag).resolve()
            if candidate.exists():
                return candidate

    runs_root = OUTPUT_ROOT / task_name / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"No runs directory found for task '{task_name}': {runs_root}")

    candidates = [path.resolve() for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found for task '{task_name}' under {runs_root}")
    candidates.sort(key=lambda path: path.name, reverse=True)
    return candidates[0]


def _pick_latest_row(rows: list[dict[str, Any]], *, numeric_fields: list[str]) -> dict[str, Any]:
    def _score(row: dict[str, Any]) -> tuple[datetime, int]:
        timestamp = _parse_timestamp(row.get("timestamp_utc"))
        richness = sum(1 for field in numeric_fields if _to_float(row.get(field)) is not None)
        return (timestamp, richness)

    return max(rows, key=_score)


def _load_distance_rows(distance_run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    summary_path = distance_run_dir / "summaries" / "distance_summary__all_datasets.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Distance summary missing: {summary_path}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _read_csv_rows(summary_path):
        dataset_id = str(row.get("dataset_id") or "").strip()
        model_id = _normalize_model_id(row.get("model_id"))
        jsd = _to_float(row.get("jensen_shannon_distance"))
        if (
            not dataset_id
            or not model_id
            or model_id in EXCLUDED_MODELS
            or _model_color(model_id) is None
            or jsd is None
            or _dataset_group(dataset_id) is None
        ):
            continue
        key = (dataset_id, model_id)
        grouped.setdefault(key, []).append(dict(row, model_id=model_id))

    return {
        key: _pick_latest_row(rows, numeric_fields=["jensen_shannon_distance", "overall_fidelity_score"])
        for key, rows in grouped.items()
    }


def _load_family_rows(analysis_run_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    datasets_root = analysis_run_dir / "datasets"
    if not datasets_root.exists():
        raise FileNotFoundError(f"Analysis dataset directory missing: {datasets_root}")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    valid_families = {family_id for family_id, _, _, _ in FAMILY_SPECS}
    for csv_path in sorted(datasets_root.glob("*/analysis_family_scores__*.csv")):
        for row in _read_csv_rows(csv_path):
            dataset_id = str(row.get("dataset_id") or csv_path.parent.name).strip()
            family_id = str(row.get("family_id") or "").strip()
            model_id = _normalize_model_id(row.get("model_id"))
            family_score = _to_float(row.get("family_score"))
            if (
                not dataset_id
                or _dataset_group(dataset_id) is None
                or not family_id
                or family_id not in valid_families
                or not model_id
                or model_id in EXCLUDED_MODELS
                or _model_color(model_id) is None
                or family_score is None
            ):
                continue
            key = (dataset_id, model_id, family_id)
            grouped.setdefault(key, []).append(dict(row, dataset_id=dataset_id, model_id=model_id))

    return {
        key: _pick_latest_row(rows, numeric_fields=["family_score", "query_count"])
        for key, rows in grouped.items()
    }


def _average(values: list[float]) -> float:
    return float(statistics.mean(values))


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _ci95_radius(values: list[float]) -> float:
    clean = [float(value) for value in values]
    if len(clean) <= 1:
        return 0.0
    return 1.96 * float(statistics.stdev(clean)) / (len(clean) ** 0.5)


def _rankdata(values: list[float]) -> list[float]:
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[cursor][0]:
            end += 1
        avg_rank = (cursor + end + 2) / 2.0
        for inner in range(cursor, end + 1):
            ranks[ordered[inner][1]] = avg_rank
        cursor = end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = _average(xs)
    mean_y = _average(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (den_x * den_y) ** 0.5
    if denom <= 0:
        return None
    return num / denom


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _build_family_structures(
    distance_rows: dict[tuple[str, str], dict[str, Any]],
    family_rows: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[
    dict[str, dict[str, list[PrefixPoint]]],
    dict[str, list[TradeoffPoint]],
    dict[str, dict[str, list[PrefixTradeoffPoint]]],
    list[dict[str, Any]],
]:
    prefix_points_by_family: dict[str, dict[str, list[PrefixPoint]]] = {}
    tradeoff_points_by_family: dict[str, list[TradeoffPoint]] = {}
    prefix_tradeoff_points_by_family: dict[str, dict[str, list[PrefixTradeoffPoint]]] = {}
    correlation_rows: list[dict[str, Any]] = []

    for family_id, family_slug, _, family_title in FAMILY_SPECS:
        by_model_group: dict[str, dict[str, list[tuple[str, float, float]]]] = {}
        for (dataset_id, model_id, row_family_id), family_row in family_rows.items():
            if row_family_id != family_id:
                continue
            distance_row = distance_rows.get((dataset_id, model_id))
            if distance_row is None:
                continue
            family_score = _to_float(family_row.get("family_score"))
            jsd = _to_float(distance_row.get("jensen_shannon_distance"))
            group_key = _dataset_group(dataset_id)
            if family_score is None or jsd is None or group_key is None:
                continue
            by_model_group.setdefault(model_id, {}).setdefault(group_key, []).append((dataset_id, family_score, jsd))

        family_model_points: dict[str, list[PrefixPoint]] = {}
        family_tradeoffs: list[TradeoffPoint] = []
        family_prefix_tradeoffs: dict[str, list[PrefixTradeoffPoint]] = {group_key: [] for group_key, _, _ in DATASET_GROUP_SPECS}
        for model_id in sorted(by_model_group.keys(), key=_model_sort_key):
            group_map = by_model_group[model_id]
            prefix_points: list[PrefixPoint] = []
            for group_key, _, _ in DATASET_GROUP_SPECS:
                values = group_map.get(group_key, [])
                if not values:
                    continue
                dataset_ids = sorted({dataset_id for dataset_id, _, _ in values}, key=_dataset_sort_key)
                prefix_points.append(
                    PrefixPoint(
                        family_id=family_id,
                        family_slug=family_slug,
                        family_title=family_title,
                        model_id=model_id,
                        model_label=_model_label(model_id),
                        model_color=_model_color(model_id) or "#000000",
                        dataset_group=group_key,
                        dataset_group_label=_group_label(group_key),
                        mark=_group_mark(group_key),
                        mean_family_score=_average([score for _, score, _ in values]),
                        mean_jsd=_average([jsd for _, _, jsd in values]),
                        matched_dataset_count=len(dataset_ids),
                        dataset_ids=dataset_ids,
                    )
                )
                family_prefix_tradeoffs[group_key].append(
                    PrefixTradeoffPoint(
                        family_id=family_id,
                        family_slug=family_slug,
                        family_title=family_title,
                        model_id=model_id,
                        model_label=_model_label(model_id),
                        model_color=_model_color(model_id) or "#000000",
                        dataset_group=group_key,
                        dataset_group_label=_group_label(group_key),
                        center_family_score=_average([score for _, score, _ in values]),
                        center_jsd=_average([jsd for _, _, jsd in values]),
                        x_error=_ci95_radius([score for _, score, _ in values]),
                        y_error=_ci95_radius([jsd for _, _, jsd in values]),
                        dataset_count=len(dataset_ids),
                        dataset_ids=dataset_ids,
                    )
                )
            if not prefix_points:
                continue
            family_model_points[model_id] = prefix_points

            xs = [point.mean_family_score for point in prefix_points]
            ys = [point.mean_jsd for point in prefix_points]
            center_x = _average(xs)
            center_y = _average(ys)
            family_tradeoffs.append(
                TradeoffPoint(
                    family_id=family_id,
                    family_slug=family_slug,
                    family_title=family_title,
                    model_id=model_id,
                    model_label=_model_label(model_id),
                    model_color=_model_color(model_id) or "#000000",
                    center_family_score=center_x,
                    center_jsd=center_y,
                    x_error=max(abs(value - center_x) for value in xs) if len(xs) > 1 else 0.0,
                    y_error=max(abs(value - center_y) for value in ys) if len(ys) > 1 else 0.0,
                    group_count=len(prefix_points),
                    groups_present=[point.dataset_group for point in prefix_points],
                )
            )

        prefix_points_by_family[family_id] = family_model_points
        tradeoff_points_by_family[family_id] = family_tradeoffs
        prefix_tradeoff_points_by_family[family_id] = {
            group_key: sorted(points, key=lambda item: _model_sort_key(item.model_id))
            for group_key, points in family_prefix_tradeoffs.items()
            if points
        }
        correlation_rows.append(
            {
                "family_id": family_id,
                "family_slug": family_slug,
                "family_title": family_title,
                "model_count": len(family_tradeoffs),
                "pearson_r": _round_or_none(
                    _pearson([point.center_family_score for point in family_tradeoffs], [point.center_jsd for point in family_tradeoffs]),
                    6,
                )
                if len(family_tradeoffs) >= 2
                else None,
                "spearman_rho": _round_or_none(
                    _spearman([point.center_family_score for point in family_tradeoffs], [point.center_jsd for point in family_tradeoffs]),
                    6,
                )
                if len(family_tradeoffs) >= 2
                else None,
            }
        )

    return prefix_points_by_family, tradeoff_points_by_family, prefix_tradeoff_points_by_family, correlation_rows


def _escape_tex(text: str) -> str:
    out = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _format_num(value: float) -> str:
    return f"{value:.6f}"


def _tikz_color_name(model_id: str) -> str:
    return "model" + "".join(ch for ch in _normalize_model_id(model_id) if ch.isalnum())


def _family_prefix_points(points_by_model: dict[str, list[PrefixPoint]]) -> list[PrefixPoint]:
    points: list[PrefixPoint] = []
    for model_id in sorted(points_by_model.keys(), key=_model_sort_key):
        points.extend(points_by_model[model_id])
    return points


def _color_definitions_from_prefix(points_by_model: dict[str, list[PrefixPoint]]) -> list[str]:
    points = _family_prefix_points(points_by_model)
    seen: set[str] = set()
    lines: list[str] = []
    for point in points:
        color_name = _tikz_color_name(point.model_id)
        if color_name in seen:
            continue
        seen.add(color_name)
        lines.append(r"\definecolor{" + color_name + r"}{HTML}{" + point.model_color.lstrip("#").upper() + r"}")
    return lines


def _color_definitions_from_tradeoff(points: list[TradeoffPoint]) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for point in points:
        color_name = _tikz_color_name(point.model_id)
        if color_name in seen:
            continue
        seen.add(color_name)
        lines.append(r"\definecolor{" + color_name + r"}{HTML}{" + point.model_color.lstrip("#").upper() + r"}")
    return lines


def _color_definitions_from_prefix_tradeoff(points: list[PrefixTradeoffPoint]) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for point in points:
        color_name = _tikz_color_name(point.model_id)
        if color_name in seen:
            continue
        seen.add(color_name)
        lines.append(r"\definecolor{" + color_name + r"}{HTML}{" + point.model_color.lstrip("#").upper() + r"}")
    return lines


def _legend_models_tabular(model_ids: list[str]) -> str:
    rows: list[str] = []
    left = model_ids[::2]
    right = model_ids[1::2]
    total_rows = max(len(left), len(right))
    for idx in range(total_rows):
        cells: list[str] = []
        for part in (left, right):
            if idx < len(part):
                model_id = part[idx]
                color_name = _tikz_color_name(model_id)
                label = _escape_tex(_model_label(model_id))
                cells.append(r"\textcolor{" + color_name + r"}{\rule{10pt}{1.2pt}} " + label)
            else:
                cells.append("")
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join(rows)


def _legend_shapes_tabular() -> str:
    rows: list[str] = []
    for group_key, label, mark in DATASET_GROUP_SPECS:
        mark_style = "mark size=2.4pt"
        rows.append(
            r"\raisebox{0.2ex}{\tikz{\draw[black, "
            + mark_style
            + ", mark="
            + mark
            + "] plot coordinates {(0,0)};}} "
            + _escape_tex(label)
            + r" \\"
        )
    return "\n".join(rows)


def _better_arrow_block() -> str:
    return "\n".join(
        [
            r"\path[draw=green!60!black, -{Stealth[length=1.8mm,width=1.2mm]}, line width=0.55pt]",
            r"(axis cs:0.018,0.988) -- node[pos=0.50, above, sloped, font=\tiny\itshape, text=green!45!black] {better} (axis cs:0.102,0.925);",
        ]
    )


def _shared_axis_options(*, title: str, xlabel: str) -> list[str]:
    return [
        r"\begin{axis}[",
        f"title={{{_escape_tex(title)}}},",
        f"xlabel={{{_escape_tex(xlabel)}}},",
        r"ylabel={Jensen-Shannon distance},",
        r"xmin=0, xmax=1,",
        r"ymin=0, ymax=1,",
        r"width=12.8cm,",
        r"height=8.8cm,",
        r"grid=both,",
        r"major grid style={draw=gray!25},",
        r"minor grid style={draw=gray!10},",
        r"tick label style={font=\small},",
        r"label style={font=\small},",
        r"title style={font=\normalsize},",
        r"clip=false,",
        r"enlarge x limits=0.08,",
        r"enlarge y limits=0.08,",
        r"]",
        _better_arrow_block(),
    ]


def _model_legend_node(model_ids: list[str]) -> str:
    return (
        r"\node[anchor=north east, draw=gray!40, rounded corners=2pt, fill=white, fill opacity=0.92, text opacity=1, inner sep=4pt, font=\scriptsize] "
        r"at (rel axis cs:0.985,0.985) {"
        r"\begin{tabular}{ll}"
        + "\n"
        + _legend_models_tabular(model_ids)
        + "\n"
        + r"\end{tabular}"
        + r"};"
    )


def _shape_legend_node() -> str:
    return (
        r"\node[anchor=south east, draw=gray!40, rounded corners=2pt, fill=white, fill opacity=0.92, text opacity=1, inner sep=4pt, font=\scriptsize] "
        r"at (rel axis cs:0.985,0.02) {"
        r"\begin{tabular}{l}"
        + "\n"
        + _legend_shapes_tabular()
        + "\n"
        + r"\end{tabular}"
        + r"};"
    )


def _tradeoff_note_node() -> str:
    return (
        r"\node[anchor=south east, draw=gray!40, rounded corners=2pt, fill=white, fill opacity=0.92, text opacity=1, inner sep=4pt, font=\scriptsize, align=left] "
        r"at (rel axis cs:0.985,0.02) {"
        + _escape_tex("Marker = overall mean; error bars = max deviation across C/M/N means")
        + r"};"
    )


def _prefix_tradeoff_note_node(group_label: str) -> str:
    return (
        r"\node[anchor=south east, draw=gray!40, rounded corners=2pt, fill=white, fill opacity=0.92, text opacity=1, inner sep=4pt, font=\scriptsize, align=left] "
        r"at (rel axis cs:0.985,0.02) {"
        + _escape_tex(group_label + "; error bars = 95% CI across matched datasets")
        + r"};"
    )


def _axis_block_grouped_lines(points_by_model: dict[str, list[PrefixPoint]], *, title: str, xlabel: str) -> str:
    model_ids = sorted(points_by_model.keys(), key=_model_sort_key)
    lines = _shared_axis_options(
        title=title,
        xlabel=xlabel,
    )
    for model_id in model_ids:
        points = sorted(points_by_model[model_id], key=lambda item: [spec[0] for spec in DATASET_GROUP_SPECS].index(item.dataset_group))
        color_name = _tikz_color_name(model_id)
        if len(points) >= 2:
            coords = " ".join(f"({_format_num(point.mean_family_score)},{_format_num(point.mean_jsd)})" for point in points)
            lines.append(r"\addplot[no marks, color=" + color_name + r", line width=0.9pt] coordinates {" + coords + r"};")
        for point in points:
            lines.append(
                r"\addplot[only marks, color="
                + color_name
                + r", mark="
                + point.mark
                + r", mark size=2.8pt, line width=0.9pt] coordinates {("
                + _format_num(point.mean_family_score)
                + ","
                + _format_num(point.mean_jsd)
                + r")};"
            )
    lines.append(_model_legend_node(model_ids))
    lines.append(_shape_legend_node())
    lines.append(r"\end{axis}")
    return "\n".join(lines)


def _axis_block_tradeoff(points: list[TradeoffPoint], *, title: str, xlabel: str) -> str:
    model_ids = [point.model_id for point in sorted(points, key=lambda item: _model_sort_key(item.model_id))]
    lines = _shared_axis_options(
        title=title,
        xlabel=xlabel,
    )
    for point in sorted(points, key=lambda item: _model_sort_key(item.model_id)):
        color_name = _tikz_color_name(point.model_id)
        lines.append(
            r"\addplot+[only marks, color="
            + color_name
            + r", mark=x, mark size=3.4pt, line width=1.0pt, error bars/.cd, x dir=both, x explicit, y dir=both, y explicit, error bar style={line width=0.8pt}] "
            r"coordinates {("
            + _format_num(point.center_family_score)
            + ","
            + _format_num(point.center_jsd)
            + r") +- ("
            + _format_num(point.x_error)
            + ","
            + _format_num(point.y_error)
            + r")};"
        )
    lines.append(_model_legend_node(model_ids))
    lines.append(_tradeoff_note_node())
    lines.append(r"\end{axis}")
    return "\n".join(lines)


def _axis_block_prefix_tradeoff(points: list[PrefixTradeoffPoint], *, title: str, xlabel: str, group_label: str) -> str:
    model_ids = [point.model_id for point in sorted(points, key=lambda item: _model_sort_key(item.model_id))]
    lines = _shared_axis_options(
        title=title,
        xlabel=xlabel,
    )
    for point in sorted(points, key=lambda item: _model_sort_key(item.model_id)):
        color_name = _tikz_color_name(point.model_id)
        lines.append(
            r"\addplot+[only marks, color="
            + color_name
            + r", mark=*, mark size=2.8pt, line width=0.95pt, error bars/.cd, x dir=both, x explicit, y dir=both, y explicit, error bar style={line width=0.78pt}] "
            r"coordinates {("
            + _format_num(point.center_family_score)
            + ","
            + _format_num(point.center_jsd)
            + r") +- ("
            + _format_num(point.x_error)
            + ","
            + _format_num(point.y_error)
            + r")};"
        )
    lines.append(_model_legend_node(model_ids))
    lines.append(_prefix_tradeoff_note_node(group_label))
    lines.append(r"\end{axis}")
    return "\n".join(lines)


def _wrap_tex_document(content: str, *, title: str, color_lines: list[str]) -> str:
    return "\n".join(
        [
            r"\documentclass[tikz,border=6pt]{standalone}",
            r"\usetikzlibrary{arrows.meta}",
            r"\usepackage{pgfplots}",
            r"\pgfplotsset{compat=1.18}",
            *color_lines,
            r"\begin{document}",
            content,
            r"\end{document}",
            "",
        ]
    )


def _write_tex_outputs(
    run_dir: Path,
    prefix_points_by_family: dict[str, dict[str, list[PrefixPoint]]],
    tradeoff_points_by_family: dict[str, list[TradeoffPoint]],
    prefix_tradeoff_points_by_family: dict[str, dict[str, list[PrefixTradeoffPoint]]],
) -> list[Path]:
    latex_dir = run_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for family_id, family_slug, xlabel, title in FAMILY_SPECS:
        grouped_tex_path = latex_dir / f"distance_vs_{family_slug}__grouped_lines.tex"
        grouped_axis = "\n".join(
            [
                r"\begin{tikzpicture}",
                _axis_block_grouped_lines(
                    prefix_points_by_family.get(family_id, {}),
                    title=f"JSD vs {title} score",
                    xlabel=xlabel,
                ),
                r"\end{tikzpicture}",
            ]
        )
        grouped_tex_path.write_text(
            _wrap_tex_document(
                grouped_axis,
                title="README-fixed model colors; group markers summarize C / M / N dataset families",
                color_lines=_color_definitions_from_prefix(prefix_points_by_family.get(family_id, {})),
            ),
            encoding="utf-8",
        )
        written.append(grouped_tex_path)

        tradeoff_tex_path = latex_dir / f"distance_vs_{family_slug}__tradeoff_errorbars.tex"
        tradeoff_axis = "\n".join(
            [
                r"\begin{tikzpicture}",
                _axis_block_tradeoff(
                    tradeoff_points_by_family.get(family_id, []),
                    title=f"JSD vs {title} score",
                    xlabel=xlabel,
                ),
                r"\end{tikzpicture}",
            ]
        )
        tradeoff_tex_path.write_text(
            _wrap_tex_document(
                tradeoff_axis,
                title="README-fixed model colors; error bars capture the C / M / N trade-off spread",
                color_lines=_color_definitions_from_tradeoff(tradeoff_points_by_family.get(family_id, [])),
            ),
            encoding="utf-8",
        )
        written.append(tradeoff_tex_path)

        for group_key, group_label, _ in DATASET_GROUP_SPECS:
            prefix_points = prefix_tradeoff_points_by_family.get(family_id, {}).get(group_key, [])
            if not prefix_points:
                continue
            prefix_tradeoff_tex_path = latex_dir / f"distance_vs_{family_slug}__tradeoff_errorbars__{group_key}.tex"
            prefix_tradeoff_axis = "\n".join(
                [
                    r"\begin{tikzpicture}",
                    _axis_block_prefix_tradeoff(
                        prefix_points,
                        title=f"JSD vs {title} score within {group_label}",
                        xlabel=xlabel,
                        group_label=group_label,
                    ),
                    r"\end{tikzpicture}",
                ]
            )
            prefix_tradeoff_tex_path.write_text(
                _wrap_tex_document(
                    prefix_tradeoff_axis,
                    title=f"README-fixed model colors; {group_label.lower()} only",
                    color_lines=_color_definitions_from_prefix_tradeoff(prefix_points),
                ),
                encoding="utf-8",
            )
            written.append(prefix_tradeoff_tex_path)

    return written


def _detect_latex_engine(explicit: str | None) -> list[str] | None:
    compiler = find_latex_compiler(explicit)
    if compiler is not None:
        return [str(compiler)]
    return None


def _compile_tex(engine: list[str], tex_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_name = Path(engine[0]).name.lower()
    if "tectonic" in engine_name:
        command = engine + ["--outdir", str(out_dir), str(tex_path)]
    elif "latexmk" in engine_name:
        command = engine + ["-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={out_dir}", str(tex_path)]
    else:
        command = engine + ["-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out_dir}", str(tex_path)]
    subprocess.run(command, cwd=tex_path.parent, check=True, text=True, capture_output=True)
    pdf_path = out_dir / f"{tex_path.stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected compiled PDF missing: {pdf_path}")
    return pdf_path


def _render_pdf_to_png(pdf_path: Path, png_path: Path) -> Path:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyMuPDF (fitz) is required to render PNG previews.") from exc

    png_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    if len(doc) <= 0:
        raise RuntimeError(f"Cannot render empty PDF: {pdf_path}")

    chosen_index = 0
    best_ratio = float("inf")
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(0.8, 0.8), alpha=False)
        samples = pix.samples
        total_pixels = len(samples) // 3
        if total_pixels <= 0:
            ratio = 1.0
        else:
            white_pixels = 0
            for idx in range(0, len(samples), 3):
                if samples[idx] > 245 and samples[idx + 1] > 245 and samples[idx + 2] > 245:
                    white_pixels += 1
            ratio = white_pixels / total_pixels
        if ratio < best_ratio:
            best_ratio = ratio
            chosen_index = page_index

    chosen_page = doc.load_page(chosen_index)
    pix = chosen_page.get_pixmap(matrix=fitz.Matrix(2.6, 2.6), alpha=False)
    pix.save(png_path)
    doc.close()
    return png_path


def _resolve_analysis_sql_source_metadata(analysis_run_dir: Path) -> dict[str, str]:
    manifest = read_json(analysis_run_dir / "manifest.json", {}) or {}
    version = normalize_sql_source_version(str(manifest.get("sql_source_version") or DEFAULT_SQL_SOURCE_VERSION))
    return {
        "provenance_contract_version": str(
            manifest.get("provenance_contract_version") or PROVENANCE_CONTRACT_VERSION
        ),
        "real_reference_split": str(manifest.get("real_reference_split") or "train"),
        "sql_source_family": str(manifest.get("sql_source_family") or sql_source_family(version)),
        "sql_source_line_version": str(manifest.get("sql_source_line_version") or sql_source_line_version(version)),
        "sql_source_version": version,
        "sql_source_label": str(manifest.get("sql_source_label") or sql_source_label(version)),
        "sql_source_description": str(manifest.get("sql_source_description") or ""),
        "sql_source_root": str(manifest.get("sql_source_root") or ""),
        "sql_source_registry_root": str(manifest.get("sql_source_registry_root") or ""),
    }


SQL_SOURCE_SUMMARY_FIELDS = [
    "provenance_contract_version",
    "real_reference_split",
    "sql_source_family",
    "sql_source_line_version",
    "sql_source_version",
    "sql_source_label",
    "sql_source_description",
    "sql_source_root",
    "sql_source_registry_root",
]


def _write_distance_query_scatter_final_bundle(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    artifact_paths: list[Path],
) -> dict[str, Any]:
    sql_source_version = str(manifest.get("sql_source_version") or DEFAULT_SQL_SOURCE_VERSION)
    final_dir = task_version_final_dir(FINAL_TASK_NAME, sql_source_version)
    final_dir.mkdir(parents=True, exist_ok=True)
    write_versioned_final_readme(
        task_name=FINAL_TASK_NAME,
        title="distance_query_scatter final outputs",
        summary="Versioned final bundles for JSD-versus-SQL-family tradeoff plots derived from the analysis scorer.",
        notes=[
            "This task already emits TeX, compiled PDF, and PNG plot variants; the final bundle just collects the important outputs by SQL source version.",
        ],
    )
    summary_note = "\n".join(
        [
            "# Distance Query Scatter Final Bundle",
            "",
            f"- analysis_run_dir: `{manifest['analysis_run_dir']}`",
            f"- distance_run_dir: `{manifest['distance_run_dir']}`",
            f"- sql_source: `{manifest['sql_source_label']}` (`{manifest['sql_source_version']}`)",
            f"- sql_source_family: `{manifest.get('sql_source_family') or ''}`",
            f"- grouped_point_count: `{manifest['grouped_point_count']}`",
            f"- tradeoff_point_count: `{manifest['tradeoff_point_count']}`",
            f"- prefix_tradeoff_point_count: `{manifest['prefix_tradeoff_point_count']}`",
            "",
        ]
    )
    summary_note_path = final_dir / "distance_query_scatter_summary.md"
    summary_note_path.write_text(summary_note, encoding="utf-8")
    copy_files(final_dir, artifact_paths)
    write_final_json(final_dir / "distance_query_scatter_run_manifest.json", manifest)
    final_manifest = {
        "task": FINAL_TASK_NAME,
        "run_tag": manifest.get("run_tag"),
        "run_dir": str(run_dir.resolve()),
        "final_dir": str(final_dir.resolve()),
        "provenance_contract_version": manifest.get("provenance_contract_version"),
        "real_reference_split": manifest.get("real_reference_split"),
        "sql_source_family": manifest.get("sql_source_family"),
        "sql_source_line_version": manifest.get("sql_source_line_version"),
        "sql_source_version": sql_source_version,
        "sql_source_label": manifest.get("sql_source_label"),
        "summary_note": str(summary_note_path.resolve()),
        "artifact_count": len([path for path in artifact_paths if path.exists()]),
    }
    write_final_json(final_dir / "distance_query_scatter_final_manifest.json", final_manifest)
    return final_manifest


def run_distance_query_scatter(
    *,
    run_tag: str,
    analysis_run_dir: Path | None = None,
    distance_run_dir: Path | None = None,
    compile_pdf: bool = True,
    latex_engine: str | None = None,
    publish_final: bool = True,
) -> dict[str, Any]:
    run_dir = make_task_run_dir(TASK_NAME, run_tag)
    resolved_analysis_run_dir = _resolve_existing_run_dir("analysis", analysis_run_dir)
    resolved_distance_run_dir = _resolve_existing_run_dir("distance", distance_run_dir)
    sql_source_meta = _resolve_analysis_sql_source_metadata(resolved_analysis_run_dir)
    sql_source_context = {
        field: sql_source_meta.get(field)
        for field in SQL_SOURCE_SUMMARY_FIELDS
        if sql_source_meta.get(field) not in (None, "")
    }

    distance_rows = _load_distance_rows(resolved_distance_run_dir)
    family_rows = _load_family_rows(resolved_analysis_run_dir)
    (
        prefix_points_by_family,
        tradeoff_points_by_family,
        prefix_tradeoff_points_by_family,
        correlation_rows,
    ) = _build_family_structures(distance_rows, family_rows)

    prefix_rows = _attach_context([
        point.to_row()
        for family_id, _, _, _ in FAMILY_SPECS
        for model_id in sorted(prefix_points_by_family.get(family_id, {}).keys(), key=_model_sort_key)
        for point in prefix_points_by_family.get(family_id, {}).get(model_id, [])
    ], sql_source_context)
    tradeoff_rows = _attach_context([
        point.to_row()
        for family_id, _, _, _ in FAMILY_SPECS
        for point in sorted(tradeoff_points_by_family.get(family_id, []), key=lambda item: _model_sort_key(item.model_id))
    ], sql_source_context)
    prefix_tradeoff_rows = _attach_context([
        point.to_row()
        for family_id, _, _, _ in FAMILY_SPECS
        for group_key, _, _ in DATASET_GROUP_SPECS
        for point in prefix_tradeoff_points_by_family.get(family_id, {}).get(group_key, [])
    ], sql_source_context)
    correlation_rows = _attach_context(correlation_rows, sql_source_context)
    if not prefix_rows or not tradeoff_rows:
        raise RuntimeError("No matched model points were found between distance and analysis outputs.")

    write_csv(run_dir / "summaries" / "distance_query_scatter_grouped_points.csv", prefix_rows)
    write_csv(run_dir / "summaries" / "distance_query_scatter_tradeoff_points.csv", tradeoff_rows)
    write_csv(run_dir / "summaries" / "distance_query_scatter_prefix_tradeoff_points.csv", prefix_tradeoff_rows)
    write_csv(run_dir / "summaries" / "distance_query_scatter_correlations.csv", correlation_rows)

    tex_paths = _write_tex_outputs(run_dir, prefix_points_by_family, tradeoff_points_by_family, prefix_tradeoff_points_by_family)
    pdf_paths: list[Path] = []
    png_paths: list[Path] = []
    compile_info: dict[str, Any] = {"compile_pdf": compile_pdf, "latex_engine": None}
    if compile_pdf:
        engine = _detect_latex_engine(latex_engine)
        if engine is None:
            raise RuntimeError("No LaTeX engine found. Install tectonic/pdflatex or pass --latex-engine.")
        compile_info["latex_engine"] = engine[0]
        pdf_dir = run_dir / "pdf"
        png_dir = run_dir / "png"
        for tex_path in tex_paths:
            pdf_path = _compile_tex(engine, tex_path, pdf_dir)
            pdf_paths.append(pdf_path)
            png_paths.append(_render_pdf_to_png(pdf_path, png_dir / f"{pdf_path.stem}.png"))

    included_models = sorted({row["model_id"] for row in tradeoff_rows}, key=_model_sort_key)
    manifest = {
        "task": TASK_NAME,
        "run_tag": run_tag,
        "analysis_run_dir": str(resolved_analysis_run_dir),
        "distance_run_dir": str(resolved_distance_run_dir),
        **sql_source_meta,
        "excluded_models": sorted(EXCLUDED_MODELS),
        "included_models": included_models,
        "family_count": len(FAMILY_SPECS),
        "version_count": len(VERSION_SPECS),
        "grouped_point_count": len(prefix_rows),
        "tradeoff_point_count": len(tradeoff_rows),
        "prefix_tradeoff_point_count": len(prefix_tradeoff_rows),
        "plot_files": [str(path.resolve()) for path in tex_paths],
        "pdf_files": [str(path.resolve()) for path in pdf_paths],
        "png_files": [str(path.resolve()) for path in png_paths],
        "compile": compile_info,
    }
    if publish_final:
        final_manifest = _write_distance_query_scatter_final_bundle(
            run_dir=run_dir,
            manifest=manifest,
            artifact_paths=[
                run_dir / "summaries" / "distance_query_scatter_grouped_points.csv",
                run_dir / "summaries" / "distance_query_scatter_tradeoff_points.csv",
                run_dir / "summaries" / "distance_query_scatter_prefix_tradeoff_points.csv",
                run_dir / "summaries" / "distance_query_scatter_correlations.csv",
                *tex_paths,
                *pdf_paths,
                *png_paths,
            ],
        )
        manifest["final_outputs"] = final_manifest
    else:
        manifest["final_outputs"] = None
    write_json(run_dir / "manifest.json", manifest)
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "grouped_points": prefix_rows,
        "tradeoff_points": tradeoff_rows,
        "correlations": correlation_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSD-versus-SQL-family scatter plots.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag.")
    parser.add_argument("--analysis-run-dir", type=Path, default=None, help="Optional analysis run dir.")
    parser.add_argument("--distance-run-dir", type=Path, default=None, help="Optional distance run dir.")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF compilation.")
    parser.add_argument("--latex-engine", type=str, default=None, help="Explicit LaTeX engine.")
    parser.add_argument("--skip-final-publish", action="store_true", help="Skip writing shared final outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_distance_query_scatter(
        run_tag=args.run_tag or now_run_tag(),
        analysis_run_dir=args.analysis_run_dir,
        distance_run_dir=args.distance_run_dir,
        compile_pdf=not args.skip_pdf,
        latex_engine=args.latex_engine,
        publish_final=not args.skip_final_publish,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
