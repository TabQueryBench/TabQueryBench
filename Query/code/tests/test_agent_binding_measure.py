from __future__ import annotations

import unittest
from types import SimpleNamespace

from tqb_query.workload_grounding.agent_binding import build_agent_binding_request


def _stats(name: str, *, minimum: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        declared_type="numeric",
        semantic_type="numeric",
        field_role="feature",
        field_tags=[],
        is_numeric=True,
        is_categorical=False,
        distinct_count=10,
        top_values=[(1, 3), (2, 2)],
        min_value=minimum,
        max_value=100.0,
        q33=1.0,
        q50=2.0,
        q66=3.0,
        q75=4.0,
    )


SHARE_TEMPLATE = {
    "template_id": "tpl_share",
    "template_name": "share",
    "sql_skeleton": (
        "SELECT {group_col}, {item_col}, SUM({measure_col}) * 100.0 / "
        "SUM(SUM({measure_col})) OVER (PARTITION BY {group_col}) AS share_within_group FROM {table} GROUP BY 1, 2"
    ),
    "required_roles": ["group_col", "item_col", "measure_col"],
    "constraints": [],
    "role_constraints": {},
    "semantic_result_contract": {
        "scorer_type": "keyed_rate_profile",
        "primary_measure": "share_within_group",
        "measure_outputs": ["share_within_group"],
    },
}

AVERAGE_TEMPLATE = {
    **SHARE_TEMPLATE,
    "template_id": "tpl_avg",
    "sql_skeleton": "SELECT {group_col}, AVG({measure_col}) AS measure_mean FROM {table} GROUP BY 1",
    "required_roles": ["group_col", "measure_col"],
    "semantic_result_contract": {
        "scorer_type": "keyed_numeric_aggregate",
        "primary_measure": "measure_mean",
        "measure_outputs": ["measure_mean"],
    },
}

# Contract wording says "rate", but the measure is standardized, not divided by its own total:
# signed columns are exactly what this template is for.
ZSCORE_TEMPLATE = {
    **SHARE_TEMPLATE,
    "template_id": "tpl_zscore",
    "sql_skeleton": (
        "WITH base AS (SELECT CAST({measure_col} AS FLOAT) AS measure_value FROM {table}) "
        "SELECT AVG(CASE WHEN ABS(measure_value) > 3 THEN 1 ELSE 0 END) AS outlier_rate FROM base"
    ),
    "required_roles": ["measure_col"],
    "semantic_result_contract": {
        "scorer_type": "rate_share_proportion",
        "primary_measure": "outlier_rate",
        "measure_outputs": ["outlier_rate"],
    },
}


class MeasureColumnConstraintTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = SimpleNamespace(
            sqlite_result=SimpleNamespace(table_name="main"),
            row_count=100,
            target_column="label",
            groupable_cols=("job", "age"),
            numeric_cols=("balance", "amount"),
            low_card_cols=("job",),
            high_card_cols=("age",),
            continuous_numeric_cols=("balance", "amount"),
            temporal_cols=(),
            missing_cols=(),
            filterable_cols=("balance", "amount"),
            condition_cols=("label",),
            field_stats={
                "balance": _stats("balance", minimum=-2000.0),  # signed: invalid for a share
                "amount": _stats("amount", minimum=0.0),
                "job": _stats("job", minimum=None),
                "age": _stats("age", minimum=18.0),
                "label": _stats("label", minimum=None),
            },
        )

    def _measure_options(self, template: dict[str, object]) -> list[str]:
        request = build_agent_binding_request(
            dataset_id="d1", template_row=template, profile=self.profile, grounding_index=0
        )
        roles = (request["allowed_bindings"] or {}).get("column_roles") or {}
        return list(roles.get("measure_col") or [])

    def test_share_template_excludes_signed_measures(self) -> None:
        options = self._measure_options(SHARE_TEMPLATE)
        self.assertNotIn("balance", options, "a column with negative values cannot form a share")
        self.assertIn("amount", options)

    def test_non_share_template_keeps_signed_measures(self) -> None:
        options = self._measure_options(AVERAGE_TEMPLATE)
        self.assertIn("balance", options, "averaging a signed column stays valid")
        self.assertIn("amount", options)

    def test_rate_wording_alone_does_not_restrict(self) -> None:
        options = self._measure_options(ZSCORE_TEMPLATE)
        self.assertIn("balance", options, "a z-score outlier rate needs signed input")


if __name__ == "__main__":
    unittest.main()
