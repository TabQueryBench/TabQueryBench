from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tqb_query.query_generation.pipeline import QueryGenerationOptions, QueryGenerationPipeline
from tqb_query.query_generation.preprocessing import preprocess_csv
from tqb_query.query_generation.providers import (
    BindingProviderUnavailable,
    BindingResponse,
    HeuristicBindingProvider,
    candidate_first_bindings,
)


ROOT = Path(__file__).resolve().parents[2]


class RecordingProvider:
    def __init__(self, invalid: bool = False) -> None:
        self.invalid = invalid
        self.requests: list[dict[str, Any]] = []

    def bind(self, request: dict[str, Any]) -> BindingResponse:
        self.requests.append(request)
        bindings = {"invented": "value"} if self.invalid else candidate_first_bindings(request)
        return BindingResponse(bindings=bindings, model="recording-provider", usage={"input_tokens": 1})


class UnavailableProvider:
    def bind(self, request: dict[str, Any]) -> BindingResponse:
        raise BindingProviderUnavailable("provider unavailable")


def write_fixture(path: Path, *, include_missing: bool = True) -> None:
    coupon = "" if include_missing else "NONE"
    path.write_text(
        "region,channel,customer_id,revenue,returned,event_date,coupon\n"
        f"east,web,c1,10,no,2024-01-01,{coupon}\n"
        "west,store,c2,20,yes,2024-02-01,SAVE\n"
        "east,web,c3,15,no,2024-03-01,SAVE\n"
        f"west,web,c4,12,no,2024-04-01,{coupon}\n"
        "north,store,c5,24,yes,2024-05-01,HELLO\n"
        f"south,web,c6,18,no,2024-06-01,{coupon}\n",
        encoding="utf-8",
    )


class PreprocessingTest(unittest.TestCase):
    def test_arbitrary_csv_produces_typed_job_local_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            write_fixture(source)
            profile = preprocess_csv(source, output_dir=root / "job", dataset_id="orders-1", target_column="returned")
            self.assertIn("revenue", profile.numeric_cols)
            self.assertIn("event_date", profile.temporal_cols)
            self.assertIn("coupon", profile.missing_cols)
            self.assertEqual(profile.sqlite_result.table_name, "orders_1")
            self.assertTrue(profile.sqlite_result.db_path.is_file())

    def test_duplicate_headers_and_unknown_target_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            source.write_text("a,a\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                preprocess_csv(source, output_dir=root / "job", dataset_id="x")
            source.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target"):
                preprocess_csv(source, output_dir=root / "job", dataset_id="x", target_column="missing")

    def test_mixed_numeric_column_materializes_as_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            source.write_text("group,value\na,1\nb,unknown\nc,3\n", encoding="utf-8")
            profile = preprocess_csv(source, output_dir=root / "job", dataset_id="mixed")
            self.assertFalse(profile.field_stats["value"].is_numeric)
            self.assertTrue(profile.sqlite_result.db_path.is_file())

    def test_unique_numeric_measure_is_not_misclassified_as_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            source.write_text("group,revenue\n" + "".join(f"g{i % 3},{i}.5\n" for i in range(30)), encoding="utf-8")
            profile = preprocess_csv(source, output_dir=root / "job", dataset_id="numeric")
            self.assertIn("revenue", profile.numeric_cols)
            self.assertNotEqual(profile.field_stats["revenue"].semantic_type, "identifier")

    def test_whitespace_header_is_rejected_instead_of_becoming_null(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            source.write_text(" group,value\na,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "whitespace"):
                preprocess_csv(source, output_dir=root / "job", dataset_id="headers")


class QueryGenerationPipelineTest(unittest.TestCase):
    def test_all_templates_are_accepted_or_explained(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            write_fixture(source)
            provider = RecordingProvider()
            result = QueryGenerationPipeline(binding_provider=provider).run(
                csv_path=source,
                output_dir=root / "out",
                options=QueryGenerationOptions(dataset_id="orders", target_column="returned", grounding_attempts=1),
            )
            manifest = result["manifest"]
            self.assertEqual(manifest["template_count"], 49)
            accounted = {query["template_id"] for query in result["queries"]} | {
                skipped["template_id"] for skipped in result["skipped_templates"]
            }
            self.assertEqual(len(accounted), 49)
            self.assertGreater(len(provider.requests), 0)
            self.assertTrue(all(query["grounding"]["model"] in {"recording-provider", "deterministic"} for query in result["queries"]))
            serialized = json.dumps(result)
            self.assertNotIn(str(root), serialized)
            self.assertTrue((root / "out/query_bundle.json").is_file())
            self.assertTrue((root / "out/grounding_records.jsonl").is_file())

    def test_invalid_ai_bindings_skip_templates_without_aborting_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            write_fixture(source, include_missing=False)
            provider = RecordingProvider(invalid=True)
            result = QueryGenerationPipeline(binding_provider=provider).run(
                csv_path=source,
                output_dir=root / "out",
                options=QueryGenerationOptions(dataset_id="orders", target_column="returned", grounding_attempts=1),
            )
            self.assertGreater(len(result["queries"]), 0)  # deterministic templates still run
            self.assertTrue(any("binding_validation_failed" in row["reasons"] for row in result["skipped_templates"]))

    def test_provider_outage_fails_the_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            write_fixture(source)
            with self.assertRaisesRegex(BindingProviderUnavailable, "unavailable"):
                QueryGenerationPipeline(binding_provider=UnavailableProvider()).run(
                    csv_path=source,
                    output_dir=root / "out",
                    options=QueryGenerationOptions(dataset_id="orders", target_column="returned", grounding_attempts=1),
                )


if __name__ == "__main__":
    unittest.main()
