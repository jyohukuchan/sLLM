from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]


class Phase27WeightStreamSummaryTests(unittest.TestCase):
    def test_negative_discovery_is_schema_valid_and_does_not_claim_adoption(self) -> None:
        schema = json.loads(
            (ROOT / "ci/schema/phase27-weight-stream-summary-v1.schema.json").read_text()
        )
        summary = json.loads(
            (ROOT / "ci/matrix/phase27-weight-stream-summary-v1.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(summary),
            key=lambda error: list(error.absolute_path),
        )
        self.assertEqual(errors, [])
        rows = {row["target"]: row for row in summary["production_baseline"]}
        profiles = {row["target"]: row for row in summary["profile_accounting"]}
        self.assertEqual(set(rows), {"gfx1030", "gfx1201"})
        self.assertEqual(set(profiles), set(rows))
        for row in rows.values():
            self.assertAlmostEqual(
                row["throughput_ratio"],
                row["sllm_decode_tokens_per_second"] / row["llama_decode_tokens_per_second"],
            )
            self.assertEqual(row["comparison_class"], "E1-system-equivalent")
        for row in profiles.values():
            self.assertAlmostEqual(
                row["projection_latency_delta"],
                row["sllm_projection_ns_per_token"] / row["llama_projection_ns_per_token"] - 1,
            )
            expected_sllm_rate = (
                row["mandatory_projection_bytes_per_token"] / row["sllm_projection_ns_per_token"]
            )
            expected_llama_rate = (
                row["mandatory_projection_bytes_per_token"] / row["llama_projection_ns_per_token"]
            )
            self.assertAlmostEqual(row["sllm_effective_weight_stream_gbps"], expected_sllm_rate)
            self.assertAlmostEqual(row["llama_effective_weight_stream_gbps"], expected_llama_rate)
            self.assertIn("not decode-only", row["projection_excluded_scope"])
            self.assertIn("no cross-engine ratio", row["projection_excluded_scope"])
        self.assertLess(profiles["gfx1030"]["projection_latency_delta"], 0)
        self.assertGreater(profiles["gfx1201"]["projection_latency_delta"], 0.05)
        self.assertFalse(summary["candidate_accounting"]["candidate_frozen"])
        disposition = summary["disposition"]
        self.assertFalse(disposition["production_source_changed"])
        self.assertFalse(disposition["candidate_adopted"])
        self.assertFalse(disposition["cross_engine_winner_claimed"])


if __name__ == "__main__":
    unittest.main()
