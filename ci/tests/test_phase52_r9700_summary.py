import importlib.util
import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "ci/tools/aggregate_phase52_r9700.py"
SCHEMA_PATH = ROOT / "ci/schema/phase52-r9700-kv-commit-summary-v1.schema.json"
SUMMARY_PATH = ROOT / "ci/matrix/phase52-r9700-kv-commit-summary-v1.json"
spec = importlib.util.spec_from_file_location("aggregate_phase52_r9700", TOOL_PATH)
assert spec is not None and spec.loader is not None
aggregate_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate_tool)


def kv_sample(case_id: str) -> dict[str, object]:
    long_case = case_id == "long-100000"
    capacity = 131_072 if long_case else 10_003
    observed = 100_001 if long_case else 10_002
    memory_kind = "contiguous-resident" if long_case else "virtual-contiguous"
    mapped = capacity if long_case else 10_240
    committed = 268_435_456 if long_case else 20_971_520
    return {
        "memory": {
            "kv": {
                "kv_layer_count": 1,
                "committed_kv_bytes": committed * 2,
                "layers": [
                    {
                        "layer": 3,
                        "logical_capacity_tokens": capacity,
                        "observed_length_tokens": observed,
                        "memory_kind": memory_kind,
                        "physical_page_bytes": 2_097_152,
                        "tokens_per_page": 1_024,
                        "mapped_token_capacity": mapped,
                        "committed_bytes_per_plane": committed,
                    }
                ],
            }
        }
    }


class Phase52R9700SummaryTests(unittest.TestCase):
    def test_schema_is_well_formed(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)

    def test_statistics_accept_only_phase52_protocol_counts(self) -> None:
        self.assertEqual(aggregate_tool.stats([1, 2, 3], "short")["count"], 3)
        self.assertEqual(aggregate_tool.stats(range(1, 11), "standard")["count"], 10)
        for values in ([1, 2], [1, 2, 3, 4]):
            with self.assertRaises(aggregate_tool.Phase52Error):
                aggregate_tool.stats(values, "wrong")

    def test_kv_memory_contract_distinguishes_short_and_long_provider(self) -> None:
        short = aggregate_tool.validate_kv_memory(
            kv_sample("long-10001"), "long-10001", "short"
        )
        long = aggregate_tool.validate_kv_memory(
            kv_sample("long-100000"), "long-100000", "long"
        )
        self.assertEqual(short["memory_kind"], "virtual-contiguous")
        self.assertEqual(long["memory_kind"], "contiguous-resident")

        invalid = kv_sample("long-100000")
        invalid["memory"]["kv"]["layers"][0]["observed_length_tokens"] = 100_000
        with self.assertRaises(aggregate_tool.Phase52Error):
            aggregate_tool.validate_kv_memory(invalid, "long-100000", "invalid")

    def test_published_summary_matches_schema(self) -> None:
        if not SUMMARY_PATH.exists():
            self.skipTest("Phase 52 GPU summary has not been published yet")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(summary)
        self.assertEqual([row["case_id"] for row in summary["rows"]], list(aggregate_tool.CASES))
        self.assertTrue(all(row["resources"]["baseline_restored"] for row in summary["rows"]))


if __name__ == "__main__":
    unittest.main()
