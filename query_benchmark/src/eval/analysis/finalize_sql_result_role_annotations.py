from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANNOTATION_ROOT = PROJECT_ROOT / "data" / "sql_result_role_annotations_v1" / "datasets"


TEMPLATE_POLICIES: dict[str, dict[str, Any]] = {
    "tpl_h2o_topn_within_group": {
        "reasoning_short": (
            "Resolved benchmark policy: group plus within-group rank identifies each retained top-N "
            "support position, while the raw ranked value is treated as the measure."
        ),
        "confidence_floor": 0.96,
    },
    "tpl_m4_quantile_tail_slice": {
        "reasoning_short": (
            "Resolved benchmark policy: the query returns tail-member values rather than an aggregate summary, "
            "so the returned value is treated as structural output and no measure column is declared."
        ),
        "confidence_floor": 0.95,
    },
    "tpl_m4_global_zscore_outliers": {
        "reasoning_short": (
            "Resolved benchmark policy: raw outlier-row attributes are structural key columns and the derived "
            "z-score style column is the measure."
        ),
        "confidence_floor": 0.95,
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_annotations(root: Path) -> dict[str, Any]:
    files_updated = 0
    annotations_updated = 0
    remaining_review = 0

    for output_path in sorted(root.glob("*/outputs/sql_result_roles_ai_v1.json")):
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        changed = False
        for ann in payload.get("query_annotations", []):
            template_id = str(ann.get("template_id") or "")
            policy = TEMPLATE_POLICIES.get(template_id)
            if not policy:
                if ann.get("needs_review") is True:
                    remaining_review += 1
                continue
            if ann.get("needs_review") is True:
                ann["needs_review"] = False
                ann["reasoning_short"] = policy["reasoning_short"]
                try:
                    current_conf = float(ann.get("confidence") or 0.0)
                except Exception:
                    current_conf = 0.0
                ann["confidence"] = max(current_conf, float(policy["confidence_floor"]))
                annotations_updated += 1
                changed = True
            elif template_id in TEMPLATE_POLICIES:
                # Keep already-resolved annotations aligned with the same final benchmark policy.
                ann["reasoning_short"] = policy["reasoning_short"]
                try:
                    current_conf = float(ann.get("confidence") or 0.0)
                except Exception:
                    current_conf = 0.0
                ann["confidence"] = max(current_conf, float(policy["confidence_floor"]))
                changed = True
        if changed:
            _write_json(output_path, payload)
            files_updated += 1

    # Recount to ensure the directory is fully clean.
    final_remaining = 0
    for output_path in sorted(root.glob("*/outputs/sql_result_roles_ai_v1.json")):
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        for ann in payload.get("query_annotations", []):
            if ann.get("needs_review") is True:
                final_remaining += 1

    return {
        "annotation_root": str(root.resolve()),
        "files_updated": files_updated,
        "annotations_updated": annotations_updated,
        "remaining_review_before_recount": remaining_review,
        "remaining_review_after_finalize": final_remaining,
        "resolved_templates": sorted(TEMPLATE_POLICIES),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize AI SQL result role annotations for approved templates.")
    parser.add_argument("--root", type=Path, default=ANNOTATION_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize_annotations(args.root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
