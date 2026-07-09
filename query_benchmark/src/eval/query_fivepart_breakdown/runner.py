#!/usr/bin/env python3
"""Build a unified five-part query breakdown contract snapshot.

This task does not change the canonical scoring pipeline. It produces a
paper-facing inventory that reconciles the repository's current
five-family query taxonomy with the partially implemented analytics
breakdown stack.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
except Exception:  # pragma: no cover - preview rendering is best-effort
    plt = None
    FancyBboxPatch = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.eval.analytics_contract import CANONICAL_ANALYTICS_SUBITEMS
except Exception:  # pragma: no cover - fallback keeps task self-contained
    CANONICAL_ANALYTICS_SUBITEMS = {
        "subgroup_structure": [
            "internal_profile_stability",
            "subgroup_size_stability",
        ],
        "conditional_dependency_structure": [
            "dependency_strength_similarity",
            "direction_consistency",
            "slice_level_consistency",
        ],
        "tail_rarity_structure": [
            "tail_set_consistency",
            "tail_mass_similarity",
            "tail_concentration_consistency",
        ],
        "missingness_structure": [
            "marginal_missing_rate_consistency",
            "co_missingness_pattern_consistency",
        ],
    }


OUTPUT_ROOT = PROJECT_ROOT / "Evaluation" / "query_fivepart_breakdown"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"

CARDINALITY_ANALYTICS_SUBITEMS = [
    "support_rank_profile_consistency",
    "high_cardinality_response_stability",
]


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, DATA_DIR, FIG_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _path_exists(rel_path: str) -> bool:
    return (PROJECT_ROOT / rel_path).exists()


def _canonical_family_rows() -> list[dict[str, Any]]:
    return [
        {
            "part_order": 1,
            "family_id": "subgroup_structure",
            "paper_label": "Subgroup structure",
            "short_label": "Subgroup",
            "layer": "query-centric family",
            "canonical_status": "implemented",
            "subitem_status": "frozen",
            "canonical_subitems": list(CANONICAL_ANALYTICS_SUBITEMS["subgroup_structure"]),
            "subitem_count": len(CANONICAL_ANALYTICS_SUBITEMS["subgroup_structure"]),
            "aggregation_rule": "mean(internal_profile_stability, subgroup_size_stability)",
            "taxonomy_anchor": "src/benchmark/models.py; doc/families_v0_1_1.yaml",
            "scoring_anchor": "doc/analytics_family_subitem_contract_v1.md; src/eval/analytics_contract.py",
            "current_breakdown_code": "src/eval/query_fivepart_breakdown/subgroup_breakdown/runner.py",
            "current_breakdown_output": "Evaluation/query_fivepart_breakdown/subgroup_breakdown/final",
            "current_repo_state": "Dedicated breakdown exists and matches the current four-family analytics contract.",
            "reporting_recommendation": "Keep as part 1 of the five-part query vector.",
            "gap_note": "",
        },
        {
            "part_order": 2,
            "family_id": "conditional_dependency_structure",
            "paper_label": "Conditional dependency structure",
            "short_label": "Conditional",
            "layer": "query-centric family",
            "canonical_status": "implemented",
            "subitem_status": "frozen",
            "canonical_subitems": list(CANONICAL_ANALYTICS_SUBITEMS["conditional_dependency_structure"]),
            "subitem_count": len(CANONICAL_ANALYTICS_SUBITEMS["conditional_dependency_structure"]),
            "aggregation_rule": "mean(dependency_strength_similarity, direction_consistency, slice_level_consistency)",
            "taxonomy_anchor": "src/benchmark/models.py; doc/families_v0_1_1.yaml",
            "scoring_anchor": "doc/analytics_family_subitem_contract_v1.md; src/eval/analytics_contract.py",
            "current_breakdown_code": "src/eval/query_fivepart_breakdown/conditional_breakdown/runner.py",
            "current_breakdown_output": "Evaluation/query_fivepart_breakdown/conditional_breakdown/final",
            "current_repo_state": "Dedicated breakdown exists and follows the canonical three-branch decomposition.",
            "reporting_recommendation": "Keep as part 2 of the five-part query vector.",
            "gap_note": "",
        },
        {
            "part_order": 3,
            "family_id": "tail_rarity_structure",
            "paper_label": "Tail / rarity structure",
            "short_label": "Tail / Rarity",
            "layer": "query-centric family",
            "canonical_status": "implemented_with_doc_inconsistency",
            "subitem_status": "implemented_as_three",
            "canonical_subitems": list(CANONICAL_ANALYTICS_SUBITEMS["tail_rarity_structure"]),
            "subitem_count": len(CANONICAL_ANALYTICS_SUBITEMS["tail_rarity_structure"]),
            "aggregation_rule": "mean(tail_set_consistency, tail_mass_similarity, tail_concentration_consistency)",
            "taxonomy_anchor": "src/benchmark/models.py; doc/families_v0_1_1.yaml",
            "scoring_anchor": "doc/analytics_family_subitem_contract_v1.md; src/eval/analytics_contract.py",
            "current_breakdown_code": "src/eval/query_fivepart_breakdown/tail_breakdown/runner.py",
            "current_breakdown_output": "Evaluation/query_fivepart_breakdown/tail_breakdown/final",
            "current_repo_state": "Dedicated breakdown exists, but the versioned contract document contains a 4-vs-3 tail count inconsistency while code uses 3 subitems.",
            "reporting_recommendation": "Use the implemented three-subitem tail layout until a versioned contract revision lands.",
            "gap_note": "Tail count mismatch across doc sections should be fixed in a later contract revision, not ad hoc in plotting.",
        },
        {
            "part_order": 4,
            "family_id": "missingness_structure",
            "paper_label": "Missingness structure",
            "short_label": "Missingness",
            "layer": "query-centric family",
            "canonical_status": "implemented_with_placeholder_fallback",
            "subitem_status": "frozen",
            "canonical_subitems": list(CANONICAL_ANALYTICS_SUBITEMS["missingness_structure"]),
            "subitem_count": len(CANONICAL_ANALYTICS_SUBITEMS["missingness_structure"]),
            "aggregation_rule": "mean(marginal_missing_rate_consistency, co_missingness_pattern_consistency) with explicit N/A gate",
            "taxonomy_anchor": "src/benchmark/models.py; doc/families_v0_1_1.yaml",
            "scoring_anchor": "doc/analytics_family_subitem_contract_v1.md; src/eval/analytics_contract.py; tests/run_comissing_condition_eval.py",
            "current_breakdown_code": "src/eval/query_fivepart_breakdown/missingness_breakdown/runner.py",
            "current_breakdown_output": "Evaluation/query_fivepart_breakdown/missingness_breakdown/final",
            "current_repo_state": "A dedicated breakdown runner now exists. If the current unified analysis export has no missingness query rows, it emits explicit standardized placeholder artifacts instead of leaving the slot empty.",
            "reporting_recommendation": "Keep as part 4 and surface the placeholder/NA status explicitly until missingness queries are present in the unified run.",
            "gap_note": "This runner preserves structure now, but real missingness figures still depend on the upstream analysis export containing missingness query rows.",
        },
        {
            "part_order": 5,
            "family_id": "cardinality_structure",
            "paper_label": "Cardinality structure",
            "short_label": "Cardinality",
            "layer": "merged query/validation breakdown",
            "canonical_status": "implemented_merged_layer",
            "subitem_status": "merged_layer_reuse",
            "canonical_subitems": list(CARDINALITY_ANALYTICS_SUBITEMS),
            "subitem_count": len(CARDINALITY_ANALYTICS_SUBITEMS),
            "aggregation_rule": "mean(support_rank_profile_consistency, high_cardinality_response_stability)",
            "taxonomy_anchor": "src/benchmark/models.py; doc/families_v0_1_1.yaml; doc/query_metrics_protocol.md",
            "scoring_anchor": "README.md; src/eval/query_fivepart_breakdown/cardinality/runner.py",
            "current_breakdown_code": "src/eval/query_fivepart_breakdown/cardinality/runner.py",
            "current_breakdown_output": "Evaluation/query_fivepart_breakdown/cardinality/final",
            "current_repo_state": "This is the merged cardinality breakdown path now shared by the former validation/query layers, and the existing code/results can be used directly.",
            "reporting_recommendation": "Keep as part 5 directly and reuse the existing cardinality breakdown folder plus its final artifacts.",
            "gap_note": "",
        },
    ]


def _coverage_rows() -> list[dict[str, Any]]:
    rows = []
    for family in _canonical_family_rows():
        rows.append(
            {
                "family_id": family["family_id"],
                "paper_label": family["paper_label"],
                "scope": family["layer"],
                "part_order": family["part_order"],
                "code_runner": family["current_breakdown_code"],
                "code_exists": _path_exists(family["current_breakdown_code"]) if family["current_breakdown_code"] else False,
                "result_dir": family["current_breakdown_output"],
                "result_dir_exists": _path_exists(family["current_breakdown_output"]) if family["current_breakdown_output"] else False,
                "final_dir_exists": _path_exists(f"{family['current_breakdown_output']}/README.md") if family["current_breakdown_output"] else False,
                "status_label": family["canonical_status"],
                "status_note": family["current_repo_state"],
            }
        )
    return rows


def _subitem_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in _canonical_family_rows():
        family_id = family["family_id"]
        scoring_anchor = family["scoring_anchor"]
        for index, subitem_id in enumerate(family["canonical_subitems"], start=1):
            rows.append(
                {
                    "family_id": family_id,
                    "paper_label": family["paper_label"],
                    "part_order": family["part_order"],
                    "subitem_order": index,
                    "subitem_id": subitem_id,
                    "subitem_status": family["subitem_status"],
                    "scoring_anchor": scoring_anchor,
                    "aggregation_rule": family["aggregation_rule"],
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _escape_tex(text: str) -> str:
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
    out = str(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _status_color(status: str) -> str:
    if "implemented" in status and "partial" not in status and "inconsistency" not in status:
        return "green!10"
    if "partial" in status or "inconsistency" in status:
        return "orange!12"
    return "red!10"


def _status_text(status: str) -> str:
    mapping = {
        "implemented": "Implemented",
        "implemented_merged_layer": "Implemented (merged layer)",
        "implemented_with_doc_inconsistency": "Implemented / doc mismatch",
        "partially_implemented": "Partial",
        "taxonomy_defined_partial_scoring": "Taxonomy only / partial scoring",
    }
    return mapping.get(status, status.replace("_", " "))


def _figure_text(family: dict[str, Any]) -> str:
    item_line = "; ".join(str(item).replace("_", r"\_") for item in family["canonical_subitems"])
    status_line = _status_text(str(family["canonical_status"]))
    return (
        rf"\textbf{{{family['part_order']}. {family['paper_label']}}}"
        + r"\\"
        + rf"\textit{{Breakdown:}} {item_line}"
        + r"\\"
        + rf"\textit{{Status:}} {status_line}"
    )


def _write_overview_tex(path: Path, families: list[dict[str, Any]]) -> None:
    lines = [
        r"\documentclass[tikz,border=6pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{xcolor}",
        r"\usetikzlibrary{positioning}",
        r"\begin{document}",
        r"\begin{tikzpicture}[font=\sffamily]",
    ]
    y_step = -3.05
    for index, family in enumerate(families):
        y_pos = index * y_step
        fill = _status_color(str(family["canonical_status"]))
        lines.extend(
            [
                rf"\node[draw=black!65, rounded corners=3pt, line width=0.6pt, fill={fill},",
                r"      text width=16.2cm, align=left, inner sep=6pt, anchor=north west]",
                rf"      (box{index}) at (0,{y_pos}) {{{_figure_text(family)}}};",
            ]
        )
    lines.extend([r"\end{tikzpicture}", r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table_tex(path: Path, families: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{p{0.06\linewidth}p{0.23\linewidth}p{0.34\linewidth}p{0.27\linewidth}}",
        r"\toprule",
        r"Part & Family & Recommended breakdown & Current repo state \\",
        r"\midrule",
    ]
    for family in families:
        breakdown = ", ".join(family["canonical_subitems"])
        current = family["current_repo_state"]
        lines.append(
            " & ".join(
                [
                    str(family["part_order"]),
                    _escape_tex(family["paper_label"]),
                    _escape_tex(breakdown),
                    _escape_tex(current),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_preview(png_path: Path, pdf_fallback_path: Path | None, families: list[dict[str, Any]]) -> None:
    if plt is None or FancyBboxPatch is None:  # pragma: no cover
        return
    fig_height = 2.1 * len(families) + 0.8
    fig, ax = plt.subplots(figsize=(13.5, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top = 0.965
    box_h = 0.17
    gap = 0.02
    color_map = {
        "implemented": "#DFF2E1",
        "implemented_with_doc_inconsistency": "#FDE8C8",
        "partially_implemented": "#FDE8C8",
        "taxonomy_defined_partial_scoring": "#F9D9D9",
    }
    for index, family in enumerate(families):
        y0 = top - index * (box_h + gap) - box_h
        color = color_map.get(str(family["canonical_status"]), "#EFEFEF")
        rect = FancyBboxPatch(
            (0.03, y0),
            0.94,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.01",
            linewidth=1.0,
            edgecolor="#555555",
            facecolor=color,
        )
        ax.add_patch(rect)
        title = f"{family['part_order']}. {family['paper_label']}"
        breakdown = "Breakdown: " + ", ".join(family["canonical_subitems"])
        status = "Status: " + _status_text(str(family["canonical_status"]))
        ax.text(0.05, y0 + box_h - 0.035, title, ha="left", va="top", fontsize=13, fontweight="bold")
        ax.text(0.05, y0 + box_h - 0.082, breakdown, ha="left", va="top", fontsize=10.5)
        ax.text(0.05, y0 + 0.03, status, ha="left", va="bottom", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    if pdf_fallback_path is not None:
        fig.savefig(pdf_fallback_path, bbox_inches="tight")
    plt.close(fig)


def _try_compile_tex(tex_path: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["latexmk", "-pdf", tex_path.name],
            cwd=tex_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "latexmk not available"
    return proc.returncode == 0, proc.stdout[-1200:]


def _build_report(families: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> str:
    dedicated = [item for item in families if item["current_breakdown_code"]]
    implemented = [item for item in families if str(item["canonical_status"]).startswith("implemented")]
    partial = [item for item in families if not str(item["canonical_status"]).startswith("implemented")]
    lines = [
        "# Unified Five-Part Query Breakdown",
        "",
        "## Recommended fixed order",
        "",
        "1. `subgroup_structure`",
        "2. `conditional_dependency_structure`",
        "3. `tail_rarity_structure`",
        "4. `missingness_structure`",
        "5. `cardinality_structure`",
        "",
        "## One-sentence contract",
        "",
        "Use a fixed five-part query breakdown, where `cardinality_structure` is already merged into the existing cardinality path and can be reused directly as part 5.",
        "",
        "## Current repository reality",
        "",
        f"- Dedicated breakdown folders already exist for `{', '.join(item['family_id'] for item in dedicated)}`.",
        f"- Clean reusable implementations without additional contract caveats currently cover `{', '.join(item['family_id'] for item in implemented)}`.",
        "- `tail_rarity_structure` already has a dedicated breakdown, but the contract document still contains a 4-vs-3 wording inconsistency while code/results use 3 subitems.",
        "",
        "## Recommended part-by-part breakdown",
        "",
    ]
    if partial:
        lines.insert(
            len(lines) - 3,
            f"- The remaining explicit gap is `{', '.join(item['family_id'] for item in partial)}`.",
        )
    else:
        lines.insert(
            len(lines) - 3,
            "- All five categories now have dedicated breakdown folders and standardized `final/must_do/` bundles.",
        )
    for family in families:
        lines.extend(
            [
                f"### {family['part_order']}. {family['paper_label']}",
                "",
                f"- Canonical subitems: `{', '.join(family['canonical_subitems'])}`",
                f"- Aggregation: `{family['aggregation_rule']}`",
                f"- Current state: {family['current_repo_state']}",
                f"- Recommended reporting: {family['reporting_recommendation']}",
                f"- Main anchors: `{family['taxonomy_anchor']}` / `{family['scoring_anchor']}`",
            ]
        )
        if family["current_breakdown_code"]:
            lines.append(f"- Existing dedicated artifact: `{family['current_breakdown_code']}` -> `{family['current_breakdown_output']}`")
        if family["gap_note"]:
            lines.append(f"- Gap note: {family['gap_note']}")
        lines.append("")

    lines.extend(
        [
            "## Suggested paper wording",
            "",
            "> We use a fixed five-part query breakdown: subgroup structure, conditional dependency structure, tail/rarity structure, missingness structure, and cardinality structure. In the current repository, the cardinality axis is already merged into the existing cardinality breakdown path, so its code and final artifacts can be reused directly as part 5.",
            "",
            "## What not to mix together",
            "",
            "- Do not merge `subgroup_structure` and `conditional_dependency_structure` at the manuscript taxonomy layer if the goal is to present a five-part query breakdown.",
            "- Do not create a second duplicate cardinality folder; reuse the merged `src/eval/query_fivepart_breakdown/cardinality` and `Evaluation/query_fivepart_breakdown/cardinality/final` path directly.",
            "- Do not silently score `missingness_structure` as zero on datasets with no native missing signal; keep the explicit `N/A` policy visible.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_readme() -> str:
    return textwrap.dedent(
        """\
        # Query Five-Part Breakdown

        This directory contains a repository-wide reconciliation of the current five-part query taxonomy and the currently implemented breakdown artifacts.

        ## What this task does

        - freezes a recommended five-part query breakdown order,
        - records the canonical subitems currently available for each part,
        - records the now-aligned five-family breakdown coverage and standardized must-do bundle convention,
        - records that the cardinality axis is already merged into the existing reusable cardinality breakdown path,
        - exports a paper-facing overview figure, CSV contract snapshot, and markdown report.

        ## Re-run

        ```bash
        python src/eval/query_fivepart_breakdown/runner.py
        ```

        ## Main outputs

        - `data/fivepart_breakdown_contract.csv`
        - `data/fivepart_breakdown_coverage.csv`
        - `data/fivepart_breakdown_subitems.csv`
        - `analysis_report.md`
        - `figures/query_fivepart_breakdown_overview.tex`
        - `figures/query_fivepart_breakdown_overview.pdf`
        - `figures/query_fivepart_breakdown_overview.png`
        - nested child folders such as `conditional_breakdown/final/`, `subgroup_breakdown/final/`, `tail_breakdown/final/`, `cardinality/final/`, and `missingness_breakdown/final/`
        """
    )


def run_query_fivepart_breakdown() -> dict[str, Any]:
    _ensure_dirs()

    families = _canonical_family_rows()
    coverage_rows = _coverage_rows()
    subitem_rows = _subitem_rows()

    contract_csv = DATA_DIR / "fivepart_breakdown_contract.csv"
    coverage_csv = DATA_DIR / "fivepart_breakdown_coverage.csv"
    subitems_csv = DATA_DIR / "fivepart_breakdown_subitems.csv"
    report_md = OUTPUT_ROOT / "analysis_report.md"
    readme_md = OUTPUT_ROOT / "README.md"
    figure_tex = FIG_DIR / "query_fivepart_breakdown_overview.tex"
    figure_pdf = FIG_DIR / "query_fivepart_breakdown_overview.pdf"
    figure_png = FIG_DIR / "query_fivepart_breakdown_overview.png"
    table_tex = TABLE_DIR / "query_fivepart_breakdown_summary.tex"

    _write_csv(contract_csv, families)
    _write_csv(coverage_csv, coverage_rows)
    _write_csv(subitems_csv, subitem_rows)
    report_md.write_text(_build_report(families, coverage_rows), encoding="utf-8")
    readme_md.write_text(_build_readme(), encoding="utf-8")
    _write_overview_tex(figure_tex, families)
    _write_table_tex(table_tex, families)

    compile_ok, compile_note = _try_compile_tex(figure_tex)
    if not compile_ok:
        _render_preview(figure_png, figure_pdf, families)
    else:
        _render_preview(figure_png, None, families)

    manifest = {
        "task": "query_fivepart_breakdown",
        "family_count": len(families),
        "dedicated_breakdown_family_count": sum(1 for item in families if item["current_breakdown_code"]),
        "implemented_query_family_breakdown_count": sum(
            1
            for item in families
            if str(item["canonical_status"]).startswith("implemented")
        ),
        "partial_or_gap_family_count": sum(
            1
            for item in families
            if not str(item["canonical_status"]).startswith("implemented")
        ),
        "compile_overview_tex_success": compile_ok,
        "compile_overview_tex_note": compile_note,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = run_query_fivepart_breakdown()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
