from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Scoring" / "code"))

from tqb_query.subitem_workload_v2 import inventory as inv
from tqb_query.subitem_workload_v2.paths import line_version_family, normalize_line_version, workload_data_root
from tqb_query.workload_grounding.problem_planner import PlannerQuotaError
from tqb_query.workload_grounding.v10_versions import (
    format_grounding_version,
    resolve_model_version,
    resolve_v10_model_version,
    validate_grounding_version,
)
from tqb_scoring.eval.common import normalize_sql_source_version, sql_source_label


def _stats(name: str, *, numeric: bool, categorical: bool, values: list[tuple[object, int]]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        declared_type="numeric" if numeric else "string",
        semantic_type="numeric" if numeric else "categorical",
        field_role="feature",
        field_tags=[],
        is_numeric=numeric,
        is_categorical=categorical,
        distinct_count=len(values),
        top_values=values,
        min_value=1.0 if numeric else None,
        max_value=9.0 if numeric else None,
        q33=3.0 if numeric else None,
        q50=5.0 if numeric else None,
        q66=6.0 if numeric else None,
        q75=7.0 if numeric else None,
    )


def _row(index: int) -> dict[str, object]:
    return {
        "template_id": f"tpl_{index:02d}",
        "template_name": f"template {index}",
        "family_id": inv.CORE_AGENT_FAMILIES[index % len(inv.CORE_AGENT_FAMILIES)],
        "gate_priority": "primary",
        "binding_roles": ["group_col"],
        "supported_canonical_subitem_ids": ["slice_level_consistency"],
        "allowed_variant_roles": ["count_distribution"],
        "intent": "group count",
        "sql_skeleton": "SELECT {group_col}, COUNT(*) FROM {table} GROUP BY {group_col}",
    }


class FakeSelector:
    def __init__(self, selected: list[str] | Exception) -> None:
        self.selected = selected
        self.requests: list[dict[str, object]] = []

    def select_templates_for_binding(self, **kwargs: object) -> list[str]:
        self.requests.append(kwargs)
        if isinstance(self.selected, Exception):
            raise self.selected
        return list(self.selected)


class V11AgentSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = SimpleNamespace(
            sqlite_result=SimpleNamespace(table_name="main"),
            row_count=20,
            target_column="label",
            groupable_cols=("category", "label"),
            numeric_cols=("amount",),
            high_card_cols=("category",),
            condition_cols=("label",),
            missing_cols=(),
            temporal_cols=(),
            filterable_cols=("label", "amount"),
            field_stats={
                "category": _stats("category", numeric=False, categorical=True, values=[("a", 10), ("b", 10)]),
                "label": _stats("label", numeric=False, categorical=True, values=[("yes", 12), ("no", 8)]),
                "amount": _stats("amount", numeric=True, categorical=False, values=[(5, 3), (7, 2)]),
            },
            summary=lambda: {"dataset_id": "c-test"},
        )

    def _select(self, rows: list[dict[str, object]], planner: FakeSelector):
        with patch.object(inv, "_agent_template_rows", return_value=rows), patch.object(
            inv, "_template_binding_possible", return_value=True
        ):
            return inv._select_agent_templates(
                dataset_id="c-test",
                profile=self.profile,
                planner_kind="agent-select-bind",
                planner_model="glm-5.3",
                ai_cli_preset="auto",
                ai_cli_command="",
                selection_policy="agent_selected_rule_count",
                agent_planner=planner,
            )

    def test_versions_paths_and_scoring_labels(self) -> None:
        # line.vendor.model[-run]_modelname; vendor 1 Anthropic, 2 Z.AI, 3 OpenAI.
        v11 = lambda model, **kw: resolve_model_version(model, line_family="v11", **kw).grounding_version
        self.assertEqual(v11("opus5"), "v11.1.1_claude-opus-5")
        self.assertEqual(v11("sonnet"), "v11.1.2_claude-sonnet-4-6")
        self.assertEqual(v11("fable"), "v11.1.3_claude-fable-5-1")
        self.assertEqual(v11("claude-haiku-4-5-20251001"), "v11.1.4_claude-haiku-4-5")
        self.assertEqual(v11("flash"), "v11.2.1_glm-5.3-flash")
        self.assertEqual(v11("glm"), "v11.2.2_glm-5.3")
        self.assertEqual(
            [v11(m) for m in ("gpt-5.4", "gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")],
            ["v11.3.1_gpt-5.4", "v11.3.2_gpt-5.5", "v11.3.3_gpt-5.6-sol", "v11.3.4_gpt-5.6-terra", "v11.3.5_gpt-5.6-luna"],
        )
        # sonnet-5 artifacts predate the table; they keep slot 1.2 under their own name.
        self.assertEqual(v11("sonnet5"), "v11.1.2_claude-sonnet-5")
        # A repeat run is named by a run suffix, given with or without the model part.
        repeat = resolve_model_version("glm", line_family="v11", grounding_version="v11.2.2-2")
        self.assertEqual(
            (repeat.grounding_version, repeat.run, repeat.mapping_source),
            ("v11.2.2-2_glm-5.3", 2, "repeat_run_version"),
        )
        self.assertEqual(v11("flash", grounding_version="v11.2.1-3_glm-5.3-flash"), "v11.2.1-3_glm-5.3-flash")
        self.assertEqual(format_grounding_version("v10", "haiku", 2), "v10.1.4-2_claude-haiku-4-5")
        for bad in (
            "v11.2.1",            # flash's slot, not glm's
            "v11.2.2_glm-5.3-flash",  # right slot, wrong model
            "v10.2.2",            # wrong line
            "v11.2.2-1",          # the first run has no suffix
            "v11.5.2",            # the retired GLM slot
        ):
            with self.assertRaises(ValueError, msg=bad):
                v11("glm", grounding_version=bad)
        with self.assertRaises(ValueError):
            v11("glm-5.2")  # no slot
        with self.assertRaises(ValueError):
            validate_grounding_version("v11.2.2", "v11")  # a full name must carry the model
        self.assertEqual(resolve_v10_model_version("opus5").grounding_version, "v10.1.1_claude-opus-5")
        version = "v11.2.2-2_glm-5.3"
        self.assertEqual(line_version_family(normalize_line_version(version)), "v11")
        self.assertTrue(str(workload_data_root(version)).endswith(f"workload_grounding_v11/variants/{version}"))
        self.assertEqual(inv.selection_policy_for_line_version(version), "agent_selected_rule_count")
        self.assertEqual(inv.selection_policy_for_line_version("v10.2.2_glm-5.3"), "all_applicable_dense")
        with self.assertRaises(ValueError):
            normalize_line_version("v11.2.2")  # a bare slot is not an artifact version
        self.assertEqual(normalize_sql_source_version(f"subitem_workload_v11/variants/{version}/runs/r"), version)
        self.assertEqual(normalize_sql_source_version(f"sv2_{version}_20260919"), version)
        self.assertEqual(sql_source_label(version), f"{version}_model_grounded")
        self.assertEqual(sql_source_label("v11"), "v11_current")

    def test_agent_choice_is_validated_and_capped_at_target(self) -> None:
        rows = [_row(index) for index in range(14)]
        chosen = ["tpl_invented", "tpl_00", "tpl_00"] + [f"tpl_{index:02d}" for index in range(1, 13)]
        planner = FakeSelector(chosen)
        selected, modes, deficits, summary = self._select(rows, planner)
        self.assertEqual([row["template_id"] for row in selected], [f"tpl_{index:02d}" for index in range(12)])
        self.assertEqual(set(modes.values()), {"agent_selected"})
        selection = summary["template_selection"]
        self.assertEqual(selection["invalid_template_ids"], ["tpl_invented"])
        self.assertEqual(selection["duplicate_template_ids"], ["tpl_00"])
        self.assertEqual(selection["truncated_template_ids"], ["tpl_12"])
        self.assertEqual((selection["min_templates"], selection["target_templates"]), (10, 12))
        self.assertEqual(deficits, [])
        request = planner.requests[0]
        self.assertEqual(len(request["candidates"]), 14)
        self.assertIn("sql_skeleton", request["candidates"][0])

    def test_short_agent_choice_is_backfilled_to_minimum_with_deficit(self) -> None:
        rows = [_row(index) for index in range(14)]
        selected, modes, deficits, summary = self._select(rows, FakeSelector(["tpl_13", "tpl_12", "tpl_11"]))
        self.assertEqual(len(selected), 10)
        self.assertEqual([row["template_id"] for row in selected[:3]], ["tpl_13", "tpl_12", "tpl_11"])
        self.assertEqual(list(modes.values()).count("agent_selected"), 3)
        self.assertEqual(list(modes.values()).count("rule_backfill"), 7)
        self.assertEqual(deficits[0]["reason"], "agent_selected_templates_below_minimum")
        self.assertEqual(len(summary["template_selection"]["rule_backfilled_template_ids"]), 7)

    def test_small_candidate_pool_caps_rule_count(self) -> None:
        rows = [_row(index) for index in range(8)]
        selected, _modes, deficits, summary = self._select(rows, FakeSelector([row["template_id"] for row in rows]))
        self.assertEqual(len(selected), 8)
        self.assertEqual((summary["template_selection"]["min_templates"], summary["template_selection"]["target_templates"]), (8, 8))
        self.assertEqual([row["reason"] for row in deficits], ["insufficient_agent_templates_for_minimum"])

    def test_quota_error_during_selection_propagates(self) -> None:
        with self.assertRaises(PlannerQuotaError):
            self._select([_row(index) for index in range(12)], FakeSelector(PlannerQuotaError("limit")))

    def test_select_then_bind_inventory_path(self) -> None:
        row = {
            "template_id": "tpl_test",
            "template_name": "test",
            "intent": "group target rate",
            "sql_skeleton": (
                "SELECT {group_col}, AVG(CASE WHEN {target_col} = {target_value} THEN 1.0 ELSE 0 END) "
                "FROM {table} GROUP BY {group_col} LIMIT {top_k}"
            ),
            "required_roles": ["group_col", "target_col", "target_value", "top_k"],
            "constraints": [],
            "role_constraints": {"distinct_roles": [["group_col", "target_col"]]},
            "family_id": "conditional_dependency_structure",
            "gate_priority": "primary",
            "extended_family": False,
            "realization_mode": "agent",
            "binding_roles": ["group_col", "target_col", "target_value", "top_k"],
            "supported_canonical_subitem_ids": ["slice_level_consistency"],
            "allowed_variant_roles": ["conditional_rate_view"],
            "semantic_result_contract": {},
        }

        class FakePlanner:
            def __init__(self, **kwargs: object) -> None:
                self.summary = {"calls": 0}
                self.kwargs = kwargs

            def select_templates_for_binding(self, **_kwargs: object) -> list[str]:
                self.summary["calls"] += 1
                return ["tpl_test"]

            def bind_template_placeholders(self, _request: dict[str, object]) -> dict[str, object]:
                self.summary["calls"] += 1
                return {"group_col": "category", "target_col": "label", "target_value": "yes", "top_k": 5}

        with TemporaryDirectory() as temp_dir, patch.object(inv, "_agent_template_rows", return_value=[row]), patch.object(
            inv, "_template_binding_possible", return_value=True
        ), patch("tqb_query.workload_grounding.problem_planner.CLIProblemPlanner", FakePlanner), patch.object(
            inv, "logs_root", return_value=Path(temp_dir)
        ):
            items, _deficits, templates, usage = inv._agent_items_for_dataset(
                "c-test", self.profile, planner_kind="agent-select-bind", planner_model="opus5",
                ai_cli_preset="auto", ai_cli_command="", selection_policy="agent_selected_rule_count",
                grounding_version="", agent_bind_problems_per_template=1,
            )
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].binding_validation["valid"])
        self.assertEqual(items[0].template_selection_mode, "agent_selected")
        self.assertEqual(items[0].model_provenance["grounding_version"], "v11.1.1_claude-opus-5")
        self.assertEqual(items[0].model_provenance["line_version"], "v11")
        self.assertEqual(templates[0]["selection_mode"], "agent_selected")
        self.assertEqual(usage["planner_kind"], "agent-select-bind")
        self.assertEqual(usage["calls"], 2)
        self.assertEqual(usage["template_selection"]["agent_selected_template_ids"], ["tpl_test"])


if __name__ == "__main__":
    unittest.main()
