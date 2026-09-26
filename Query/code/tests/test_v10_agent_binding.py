from __future__ import annotations

from types import MethodType, SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Scoring" / "code"))

from tqb_query.agent.local_sql_runner import instantiate_template_sql
from tqb_scoring.eval.common import normalize_sql_source_version, sql_source_label
from tqb_query.subitem_workload_v2.inventory import selection_policy_for_line_version
from tqb_query.subitem_workload_v2.inventory import _agent_items_for_dataset
from tqb_query.subitem_workload_v2.paths import line_version_family, normalize_line_version, workload_data_root
from tqb_query.workload_grounding.agent_binding import build_agent_binding_request, validate_agent_bindings
from tqb_query.workload_grounding.problem_planner import CLIProblemPlanner
from tqb_query.workload_grounding.v10_versions import resolve_v10_model_version


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


class V10AgentBindingTest(unittest.TestCase):
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
        )
        self.template = {
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
        }

    def test_model_versions_and_dotted_paths(self) -> None:
        self.assertEqual(resolve_v10_model_version("gpt-5.5").grounding_version, "v10.3.2_gpt-5.5")
        self.assertEqual(resolve_v10_model_version("opus5").grounding_version, "v10.1.1_claude-opus-5")
        self.assertEqual(resolve_v10_model_version("sonnet").grounding_version, "v10.1.2_claude-sonnet-4-6")
        self.assertEqual(resolve_v10_model_version("sonnet5").grounding_version, "v10.1.2_claude-sonnet-5")
        self.assertEqual(resolve_v10_model_version("fable 5.1").grounding_version, "v10.1.3_claude-fable-5-1")
        self.assertEqual(resolve_v10_model_version("haiku").grounding_version, "v10.1.4_claude-haiku-4-5")
        self.assertEqual(resolve_v10_model_version("glm").resolved_model, "glm-5.3")
        self.assertEqual(resolve_v10_model_version("GLM-5.3").grounding_version, "v10.2.2_glm-5.3")
        version = "v10.3.1_gpt-5.4"
        self.assertEqual(normalize_line_version("V10.3.1_gpt-5.4"), version)
        self.assertEqual(line_version_family(version), "v10")
        self.assertTrue(str(workload_data_root(version)).endswith(f"/variants/{version}"))
        self.assertEqual(selection_policy_for_line_version(version), "all_applicable_dense")
        self.assertEqual(normalize_sql_source_version(f"subitem_workload_v10/variants/{version}"), version)
        self.assertEqual(sql_source_label(version), f"{version}_model_grounded")

    def test_binding_contract_accepts_only_allowed_choices(self) -> None:
        request = build_agent_binding_request(
            dataset_id="c-test", template_row=self.template, profile=self.profile, grounding_index=0
        )
        good = {"group_col": "category", "target_col": "label", "target_value": "yes", "top_k": 5}
        self.assertTrue(validate_agent_bindings(good, request).valid)
        bad = {**good, "group_col": "invented", "target_value": "maybe", "extra": 1}
        issue_codes = {issue.code for issue in validate_agent_bindings(bad, request).issues}
        self.assertEqual(issue_codes, {"column_not_allowed_for_role", "value_not_observed", "unknown_binding_key"})

    def test_cli_binding_method_enforces_top_level_shape(self) -> None:
        planner = object.__new__(CLIProblemPlanner)
        planner._invoke_json = MethodType(lambda _self, **_kwargs: {"bindings": {"group_col": "category"}}, planner)
        self.assertEqual(
            planner.bind_template_placeholders({"case_id": "x", "template": {"template_id": "t"}}),
            {"group_col": "category"},
        )
        planner._invoke_json = MethodType(
            lambda _self, **_kwargs: {"bindings": {"group_col": "category"}, "sql": "SELECT 1"}, planner
        )
        self.assertEqual(planner.bind_template_placeholders({"case_id": "x", "template": {}}), {})

    def test_strict_renderer_rejects_missing_or_invalid_bindings(self) -> None:
        lookup = {"tpl": {"sql_skeleton": "SELECT {group_col} FROM {table} WHERE {group_col} {predicate_op} {predicate_value}"}}
        with self.assertRaises(KeyError):
            instantiate_template_sql(
                template_id="tpl", template_lookup=lookup, question_record={"bindings": {"group_col": "x"}},
                table_name="main", strict=True,
            )
        with self.assertRaises(ValueError):
            instantiate_template_sql(
                template_id="tpl", template_lookup=lookup,
                question_record={"bindings": {"group_col": "x", "predicate_op": "DROP", "predicate_value": 1}},
                table_name="main", strict=True,
            )

    def test_agent_bind_inventory_path_keeps_model_output_and_provenance(self) -> None:
        row = {
            **self.template,
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
            def __init__(self, **_kwargs: object) -> None:
                self.summary = {"calls": 0}

            def bind_template_placeholders(self, _request: dict[str, object]) -> dict[str, object]:
                self.summary["calls"] += 1
                return {"group_col": "category", "target_col": "label", "target_value": "yes", "top_k": 5}

        selection = ([row], {"tpl_test": "all_applicable_dense"}, [], {"planner_kind": "agent-bind"})
        with TemporaryDirectory() as temp_dir, patch(
            "tqb_query.subitem_workload_v2.inventory._select_agent_templates", return_value=selection
        ), patch("tqb_query.workload_grounding.problem_planner.CLIProblemPlanner", FakePlanner), patch(
            "tqb_query.subitem_workload_v2.inventory.logs_root", return_value=Path(temp_dir)
        ):
            items, deficits, _templates, usage = _agent_items_for_dataset(
                "c-test", self.profile, planner_kind="agent-bind", planner_model="gpt-5.5",
                ai_cli_preset="auto", ai_cli_command="", selection_policy="all_applicable_dense",
                grounding_version="v10.3.2", agent_bind_problems_per_template=1,
            )
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].binding_validation["valid"])
        self.assertEqual(items[0].raw_agent_bindings, items[0].bindings)
        self.assertEqual(items[0].realization_mode, "agent_bind")
        self.assertEqual(items[0].model_provenance["grounding_version"], "v10.3.2_gpt-5.5")
        self.assertEqual(items[0].model_provenance["ai_cli_preset"], "codex")
        self.assertEqual(usage["calls"], 1)
        self.assertTrue(any(row["reason"] == "planned_agent_sql_below_minimum" for row in deficits))


if __name__ == "__main__":
    unittest.main()
