from types import SimpleNamespace
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqb_scoring.standards import load_standard  # noqa: E402

compare_semantic_execution_results = load_standard("spq_v9").compare_semantic_execution_results


def _exec(columns, rows):
    return SimpleNamespace(ok=True, columns=columns, rows=rows)


def _score(real, syn, query):
    score, detail = compare_semantic_execution_results(real, syn, query=query)
    return score, detail


class QuerySemanticScoringV9Test(unittest.TestCase):
    def test_scalar_numeric_uses_max_denominator(self):
        score, detail = _score(
            _exec(["avg_value"], [[100.0]]),
            _exec(["avg_value"], [[80.0]]),
            {"scorer_type": "scalar_numeric"},
        )

        self.assertAlmostEqual(score, 0.8)
        self.assertEqual(detail["semantic_query_score_method"], "spq")
        self.assertEqual(detail["score_contract_version"], "spq_v9_single_primary")
        self.assertEqual(detail["primary_metric"], "scalar_similarity")

    def test_count_support_distribution_uses_tvd_over_union_keys(self):
        score, detail = _score(
            _exec(["group", "support"], [["a", 30], ["b", 70]]),
            _exec(["group", "support"], [["a", 50], ["c", 50]]),
            {"scorer_type": "count_support_distribution"},
        )

        self.assertAlmostEqual(score, 0.3)
        self.assertEqual(detail["primary_metric"], "support_distribution_similarity")
        self.assertAlmostEqual(detail["component_scores"]["key_f1"], 0.5)

    def test_rate_similarity_penalizes_missing_union_keys(self):
        score, detail = _score(
            _exec(["group", "missing_rate"], [["a", 0.1], ["b", 0.2]]),
            _exec(["group", "missing_rate"], [["a", 0.3], ["c", 0.4]]),
            {"scorer_type": "rate_share_proportion"},
        )

        self.assertAlmostEqual(score, (0.8 + 0.0 + 0.0) / 3.0)
        self.assertEqual(detail["primary_metric"], "rate_similarity")

    def test_ratio_similarity_uses_min_over_max(self):
        score, detail = _score(
            _exec(["group", "ratio"], [["a", 2.0], ["b", 0.0], ["c", 0.0]]),
            _exec(["group", "ratio"], [["a", 4.0], ["b", 0.0], ["c", 3.0]]),
            {"scorer_type": "ratio"},
        )

        self.assertAlmostEqual(score, (0.5 + 1.0 + 0.0) / 3.0)
        self.assertEqual(detail["primary_metric"], "ratio_similarity")

    def test_topk_primary_score_is_rbo_not_measure_similarity(self):
        score, detail = _score(
            _exec(["item", "value"], [["a", 100], ["b", 90], ["c", 80]]),
            _exec(["item", "value"], [["a", 1], ["c", 1], ["b", 1]]),
            {"scorer_type": "topk_ranking"},
        )

        self.assertAlmostEqual(score, detail["component_scores"]["rbo_similarity"])
        self.assertEqual(detail["primary_metric"], "rbo_similarity")
        self.assertLess(detail["component_scores"]["measure_similarity"], score)


if __name__ == "__main__":
    unittest.main()
