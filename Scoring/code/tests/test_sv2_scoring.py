"""SV2 contract tests: fixed examples (contract §24), properties (§25), routing, adapters, aggregation."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqb_scoring.standards import load_standard  # noqa: E402
from tqb_scoring.standards.sv2 import score_execution_results, score_query  # noqa: E402
from tqb_scoring.standards.sv2.aggregate import summarize_query_scores  # noqa: E402
from tqb_scoring.standards.sv2.canonicalize import parse_sql_limit  # noqa: E402
from tqb_scoring.standards.sv2.common import ScoringError, checked_unit_interval  # noqa: E402
from tqb_scoring.standards.sv2.numeric_magnitude import normalized_smape_similarity  # noqa: E402
from tqb_scoring.standards.sv2.routing import load_template_routing, resolve_sv2_scorer  # noqa: E402
from tqb_scoring.standards.sv2.topk_ranking import rbo_ext  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
NUM = "numeric_magnitude"
CNT = "count_support_distribution"
RATE = "rate_share_proportion"
RANK = "topk_ranking"


def score(real, syn, scorer_type, **config):
    return score_query(real, syn, scorer_type=scorer_type, scorer_config=config or None)


def exec_result(columns, rows, ok=True, error=None):
    return SimpleNamespace(ok=ok, columns=columns, rows=rows, error=error)


def manual_rbo(real, syn, depth, p=0.9):
    real = list(real[:depth]) + [("real_pad", i) for i in range(depth - len(real[:depth]))]
    syn = list(syn[:depth]) + [("syn_pad", i) for i in range(depth - len(syn[:depth]))]
    agreements = [len(set(real[:d]) & set(syn[:d])) / d for d in range(1, depth + 1)]
    return sum((1 - p) * p ** (d - 1) * agreements[d - 1] for d in range(1, depth + 1)) + p**depth * agreements[-1]


class NumericMagnitudeTest(unittest.TestCase):
    def assertScore(self, result, expected):
        self.assertTrue(result["valid"], result)
        # Contract examples are printed to 10 decimals.
        self.assertAlmostEqual(result["query_score"], expected, places=9)
        self.assertEqual(result["semantic_query_score"], result["query_score"])

    def test_fixed_examples(self):
        self.assertScore(score(100, 100, NUM), 1.0)
        self.assertScore(score(100, 80, NUM), 1 - 20 / 180)
        self.assertScore(score(2, 3, NUM), 0.8)
        self.assertScore(score(0, 0, NUM), 1.0)
        self.assertScore(score(0, 5, NUM), 0.0)
        self.assertScore(score(-10, -8, NUM), 1 - 2 / 18)
        self.assertScore(score(1, -1, NUM), 0.0)

    def test_keyed_and_missing_keys(self):
        self.assertScore(score({"A": 100, "B": 50, "C": 20}, {"A": 90, "B": 40, "C": 20}, NUM), 0.9454191033)
        self.assertScore(score({"A": 100, "B": 50}, {"A": 90, "B": 40}, NUM), 0.9181286549)
        self.assertScore(score({"A": 10, "B": 20}, {"A": 10}, NUM), 0.5)
        self.assertScore(score({}, {}, NUM), 1.0)
        self.assertScore(score({"A": 1.0}, {}, NUM), 0.0)

    def test_metadata_fields(self):
        result = score(100, 80, NUM)
        self.assertEqual(result["metric"], "normalized_smape_similarity")
        self.assertEqual(result["semantic_query_score_method"], "sv2")
        self.assertEqual(result["score_contract_version"], "sv2_four_primary_v1")

    def test_null_handling(self):
        self.assertScore(score({"A": None}, {"A": None}, NUM), 1.0)
        self.assertScore(score({"A": None}, {"A": 5.0}, NUM), 0.0)

    def test_missing_key_is_not_zero_value(self):
        self.assertScore(score({"A": 0.0, "B": 0.0}, {"A": 0.0}, NUM), 0.5)

    def test_unkeyed_vector_is_not_sorted(self):
        self.assertScore(score([10.0, 20.0], [20.0, 10.0], NUM), 1 - 10 / 30)

    def test_invalid_inputs_are_not_zero(self):
        for bad, code in ((math.nan, "NON_FINITE_NUMERIC_VALUE"), (math.inf, "NON_FINITE_NUMERIC_VALUE"), ("abc", "NON_NUMERIC_VALUE")):
            result = score({"A": 1.0}, {"A": bad}, NUM)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["query_score"])
            self.assertEqual(result["error_code"], code)


class CountSupportDistributionTest(unittest.TestCase):
    def test_fixed_examples(self):
        self.assertAlmostEqual(score({"A": 50, "B": 50}, {"A": 5, "B": 5}, CNT)["query_score"], 1.0)
        self.assertAlmostEqual(score({"A": 50, "B": 30, "C": 20}, {"A": 40, "B": 40, "C": 20}, CNT)["query_score"], 0.9)
        self.assertAlmostEqual(score({"A": 100}, {"B": 100}, CNT)["query_score"], 0.0)
        self.assertEqual(score({}, {}, CNT)["query_score"], 1.0)
        self.assertEqual(score({"A": 0}, {"A": 0}, CNT)["query_score"], 1.0)
        self.assertEqual(score({"A": 10}, {}, CNT)["query_score"], 0.0)

    def test_total_mass_is_diagnostic_only(self):
        result = score({"A": 50, "B": 50}, {"A": 500, "B": 500}, CNT)
        self.assertEqual(result["query_score"], 1.0)
        self.assertEqual(result["diagnostics"]["synthetic_total_mass"], 1000.0)

    def test_duplicate_keys_are_summed(self):
        result = score([("A", 10), ("A", 20), ("B", 30)], {"A": 30, "B": 30}, CNT)
        self.assertEqual(result["query_score"], 1.0)

    def test_invalid_counts(self):
        self.assertEqual(score({"A": -1}, {"A": 1}, CNT)["error_code"], "NEGATIVE_COUNT")
        self.assertEqual(score({"A": None}, {"A": 1}, CNT)["error_code"], "NULL_COUNT")
        self.assertEqual(score({"A": math.nan}, {"A": 1}, CNT)["error_code"], "NON_FINITE_NUMERIC_VALUE")


class RateShareProportionTest(unittest.TestCase):
    def test_fixed_examples(self):
        self.assertAlmostEqual(score(0.30, 0.25, RATE)["query_score"], 0.95)
        self.assertAlmostEqual(score({"A": 0.2, "B": 0.6, "C": 0.9}, {"A": 0.25, "B": 0.5, "C": 0.9}, RATE)["query_score"], 0.95)
        self.assertAlmostEqual(score({"A": 0.2, "B": 0.8}, {"A": 0.2}, RATE)["query_score"], 0.5)

    def test_range_validation(self):
        result = score(1.2, 0.8, RATE)
        self.assertFalse(result["valid"])
        self.assertEqual(result["error_code"], "RATE_OUT_OF_RANGE")
        self.assertEqual(score(1 + 1e-13, 1.0, RATE)["query_score"], 1.0)

    def test_percent_scale_requires_explicit_declaration(self):
        self.assertFalse(score(35, 30, RATE)["valid"])
        self.assertAlmostEqual(score(35, 30, RATE, rate_scale="percent")["query_score"], 0.95)

    def test_mae_not_relative(self):
        self.assertAlmostEqual(score(0.10, 0.20, RATE)["query_score"], score(0.80, 0.90, RATE)["query_score"])


class TopkRankingTest(unittest.TestCase):
    def test_fixed_examples(self):
        self.assertAlmostEqual(score(["A", "B", "C"], ["A", "B", "C"], RANK)["query_score"], 1.0)
        self.assertAlmostEqual(score(["A", "B", "C"], ["D", "E", "F"], RANK)["query_score"], 0.0)
        self.assertEqual(score(["A"], ["A"], RANK)["query_score"], 1.0)
        self.assertEqual(score(["A"], ["B"], RANK)["query_score"], 0.0)
        self.assertEqual(score([], [], RANK)["query_score"], 1.0)
        self.assertEqual(score([], ["A"], RANK)["query_score"], 0.0)

    def test_unequal_lengths_match_manual_formula(self):
        real, syn = ["A", "B", "C", "D", "E"], ["A", "C", "F"]
        self.assertAlmostEqual(rbo_ext(real, syn), manual_rbo(real, syn, 5), places=12)
        self.assertAlmostEqual(rbo_ext(["A", "B", "C"], ["A"]), manual_rbo(["A", "B", "C"], ["A"], 3), places=12)
        self.assertAlmostEqual(rbo_ext(["A", "B", "C"], ["A"]), 0.1 * (1 + 0.9 * 0.5 + 0.81 / 3) + 0.729 / 3, places=12)

    def test_requested_depth_truncates_and_pads(self):
        real, syn = ["A", "B", "C", "D"], ["B", "A"]
        self.assertAlmostEqual(rbo_ext(real, syn, requested_depth=3), manual_rbo(real, syn, 3), places=12)

    def test_requested_depth_is_capped_by_observed_lists(self):
        self.assertEqual(rbo_ext(["A", "B"], ["A", "B"], requested_depth=12), 1.0)
        self.assertAlmostEqual(rbo_ext(["A", "B"], ["A"], requested_depth=12), manual_rbo(["A", "B"], ["A"], 2), places=12)

    def test_compound_identities_and_duplicates(self):
        self.assertEqual(score([("US", "A"), ("CA", "B")], [("US", "A"), ("CA", "B")], RANK)["query_score"], 1.0)
        self.assertEqual(score(["A", "A"], ["A"], RANK)["error_code"], "DUPLICATE_RANKING_IDENTITY")

    def test_padding_never_matches_real_values(self):
        self.assertEqual(rbo_ext(["__SYN_PAD_1__"], []), 0.0)


class RoutingTest(unittest.TestCase):
    def test_routing_priority_and_compatibility(self):
        self.assertEqual(resolve_sv2_scorer({"sv2_scorer_type": RATE, "template_id": "tpl_h2o_group_sum"})[0], RATE)
        self.assertEqual(resolve_sv2_scorer({"template_id": "tpl_h2o_group_sum"}), (NUM, "template_routing"))
        self.assertEqual(resolve_sv2_scorer({"old_scorer_type": "ratio"})[0], NUM)
        self.assertEqual(resolve_sv2_scorer({"old_scorer_type": "keyed_numeric_aggregate"})[0], NUM)
        self.assertEqual(resolve_sv2_scorer({"old_scorer_type": "scalar"})[0], NUM)
        self.assertEqual(resolve_sv2_scorer({"old_scorer_type": "scalar", "semantic_value_type": "probability"})[0], RATE)
        self.assertEqual(resolve_sv2_scorer({"old_scorer_type": "distribution_cardinality_profile"})[0], CNT)
        for hint in ("topk_ranking", "topk_tailk_ranking", "topk_ranked_measure", "argmax_selection"):
            self.assertEqual(resolve_sv2_scorer({"old_scorer_type": hint})[0], RANK)

    def test_ambiguous_hints_fail_explicitly(self):
        with self.assertRaises(ScoringError) as ctx:
            resolve_sv2_scorer({"semantic_result_contract": {"scorer_type": "tail_topn_value_curve"}})
        self.assertEqual(ctx.exception.code, "UNRESOLVED_SCORER_TYPE")
        self.assertEqual(score_query(1, 1)["error_code"], "UNRESOLVED_SCORER_TYPE")

    def test_every_template_is_routed(self):
        import json

        routing = load_template_routing()
        for library in ("workload_grounding_v8/template_library_v8.jsonl", "workload_grounding_v9/template_library_v9.jsonl"):
            path = REPO_ROOT / "Query" / "code" / "data" / library
            if not path.exists():
                continue
            template_ids = {json.loads(line)["template_id"] for line in path.read_text().splitlines() if line.strip()}
            self.assertFalse(template_ids - set(routing), f"unrouted templates in {library}")


class PropertyTest(unittest.TestCase):
    rng = random.Random(20260915)

    def random_map(self, rate=False, count=False):
        keys = self.rng.sample("ABCDEFGH", self.rng.randint(0, 6))
        if count:
            return {key: float(self.rng.randint(0, 50)) for key in keys}
        if rate:
            return {key: self.rng.random() for key in keys}
        return {key: self.rng.choice([0.0, self.rng.uniform(-1e6, 1e6), self.rng.uniform(-1, 1)]) for key in keys}

    def random_ranking(self):
        return self.rng.sample("ABCDEFGHIJ", self.rng.randint(0, 8))

    def test_bounded_identity_symmetry(self):
        cases = (
            (NUM, lambda: self.random_map()),
            (RATE, lambda: self.random_map(rate=True)),
            (CNT, lambda: self.random_map(count=True)),
            (RANK, self.random_ranking),
        )
        for scorer_type, generate in cases:
            for _ in range(300):
                x, y = generate(), generate()
                forward = score(x, y, scorer_type)["query_score"]
                backward = score(y, x, scorer_type)["query_score"]
                self.assertTrue(0.0 <= forward <= 1.0)
                self.assertEqual(score(x, x, scorer_type)["query_score"], 1.0)
                self.assertAlmostEqual(forward, backward, places=12)

    def test_disjoint_counts_and_opposite_signs_and_rate_error(self):
        for _ in range(200):
            left = {f"L{i}": self.rng.randint(1, 9) for i in range(self.rng.randint(1, 4))}
            right = {f"R{i}": self.rng.randint(1, 9) for i in range(self.rng.randint(1, 4))}
            self.assertEqual(score(left, right, CNT)["query_score"], 0.0)
            x = self.rng.uniform(1e-6, 1e6) * self.rng.choice([1, -1])
            self.assertEqual(normalized_smape_similarity(x, -x), 0.0)
            r, s = self.rng.random(), self.rng.random()
            self.assertAlmostEqual(score(r, s, RATE)["query_score"], 1 - abs(r - s), places=12)

    def test_checked_unit_interval(self):
        self.assertEqual(checked_unit_interval(-1e-13), 0.0)
        with self.assertRaises(ScoringError):
            checked_unit_interval(1.1)


class ExecutionAdapterTest(unittest.TestCase):
    def test_keyed_numeric_template(self):
        query = {"template_id": "tpl_h2o_group_sum", "sql": "SELECT g, SUM(x) AS total_measure FROM t GROUP BY g"}
        real = exec_result(["g", "total_measure"], [["A", 100], ["B", 50]])
        syn = exec_result(["g", "total_measure"], [[" A", 1], ["B", 40]])
        result = score_execution_results(real, syn, query)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["query_score"], (0 + 0 + (1 - 10 / 90)) / 3)
        self.assertEqual(result["diagnostics"]["key_columns"], ["g"])

    def test_numeric_text_keys_align(self):
        query = {"template_id": "tpl_m4_group_avg_numeric"}
        real = exec_result(["k", "avg_measure"], [[1, 2.0], [2, 4.0]])
        syn = exec_result(["k", "avg_measure"], [["1", 2.0], ["2.0", 4.0]])
        self.assertEqual(score_execution_results(real, syn, query)["query_score"], 1.0)

    def test_normalization_collision_falls_back_to_exact_keys(self):
        query = {"template_id": "tpl_m4_group_avg_numeric"}
        real = exec_result(["k", "avg_measure"], [[0, 1.0], ["0", 2.0]])
        syn = exec_result(["k", "avg_measure"], [[0, 1.0], ["0", 2.0]])
        result = score_execution_results(real, syn, query)
        self.assertEqual(result["query_score"], 1.0)
        self.assertEqual(result["diagnostics"]["key_match_mode"], "exact_typed")

    def test_percent_share_template(self):
        query = {"template_id": "tpl_tpcds_within_group_share"}
        columns = ["g", "i", "total_measure", "share_within_group"]
        real = exec_result(columns, [["a", "x", 10, 60.0], ["a", "y", 5, 40.0]])
        syn = exec_result(columns, [["a", "x", 10, 50.0], ["a", "y", 5, 50.0]])
        self.assertAlmostEqual(score_execution_results(real, syn, query)["query_score"], 0.9)

    def test_ranking_template_uses_sql_limit_and_tie_break(self):
        query = {"template_id": "tpl_clickbench_group_distinct_topk", "sql": "SELECT g, COUNT(DISTINCT e) AS distinct_entities FROM t GROUP BY g ORDER BY distinct_entities DESC LIMIT 2;"}
        real = exec_result(["g", "distinct_entities"], [["B", 5], ["A", 5]])
        syn = exec_result(["g", "distinct_entities"], [["A", 5], ["B", 5]])
        result = score_execution_results(real, syn, query)
        self.assertEqual(result["query_score"], 1.0)
        self.assertEqual(result["diagnostics"]["requested_depth"], 2)
        self.assertEqual(result["diagnostics"]["real_tie_reordered_positions"], 2)

    def test_scalar_rate_template(self):
        query = {"template_id": "tpl_missing_marginal_rate_profile"}
        columns = ["total_rows", "missing_rows", "missing_rate"]
        result = score_execution_results(exec_result(columns, [[10, 3, 0.3]]), exec_result(columns, [[20, 5, 0.25]]), query)
        self.assertAlmostEqual(result["query_score"], 0.95)

    def test_invalid_execution_and_columns(self):
        query = {"template_id": "tpl_h2o_group_sum"}
        ok = exec_result(["g", "total_measure"], [["A", 1]])
        failed = exec_result([], [], ok=False, error="no such column")
        self.assertEqual(score_execution_results(ok, failed, query)["error_code"], "SYNTHETIC_EXECUTION_FAILED")
        mismatch = exec_result(["g", "other"], [["A", 1]])
        self.assertEqual(score_execution_results(ok, mismatch, query)["error_code"], "COLUMN_MISMATCH")
        duplicate = exec_result(["g", "total_measure"], [["A", 1], ["A", 2]])
        self.assertEqual(score_execution_results(ok, duplicate, query)["error_code"], "DUPLICATE_KEY")

    def test_runner_interface(self):
        module = load_standard("sv2")
        query = {"template_id": "tpl_threshold_rarity_cdf"}
        columns = ["empirical_cdf_at_threshold"]
        value, detail = module.compare_semantic_execution_results(exec_result(columns, [[0.4]]), exec_result(columns, [[1.4]]), query=query)
        self.assertIsNone(value)
        self.assertEqual(detail["validity_status"], "invalid")
        value, detail = module.compare_semantic_execution_results(exec_result(columns, [[0.4]]), exec_result(columns, [[0.5]]), query=query)
        self.assertAlmostEqual(value, 0.9)
        self.assertEqual(detail["component_scores"], {"one_minus_mae": value})

    def test_parse_sql_limit(self):
        self.assertEqual(parse_sql_limit("SELECT * FROM t ORDER BY x DESC LIMIT 12;"), 12)
        self.assertEqual(parse_sql_limit("-- LIMIT 5\nSELECT * FROM (SELECT * FROM t LIMIT 3) ORDER BY x"), None)


class AggregateTest(unittest.TestCase):
    def test_macro_and_query_weighted_summaries(self):
        rows = [
            {"model_id": "m", "scorer_type": NUM, "query_score": 1.0, "semantic_valid": True},
            {"model_id": "m", "scorer_type": NUM, "query_score": 0.5, "semantic_valid": True},
            {"model_id": "m", "scorer_type": RATE, "query_score": 0.0, "semantic_valid": True},
            {"model_id": "m", "scorer_type": RATE, "query_score": None, "semantic_valid": False, "semantic_error_code": "RATE_OUT_OF_RANGE"},
        ]
        (summary,) = summarize_query_scores(rows, group_fields=("model_id",))
        self.assertEqual(summary["n_queries_total"], 4)
        self.assertEqual(summary["n_queries_invalid"], 1)
        self.assertAlmostEqual(summary["query_weighted_mean"], 0.5)
        self.assertAlmostEqual(summary["macro_over_scorer_types_mean"], 0.375)
        self.assertEqual(summary["invalid_reason_counts"], '{"RATE_OUT_OF_RANGE": 1}')


if __name__ == "__main__":
    unittest.main()
