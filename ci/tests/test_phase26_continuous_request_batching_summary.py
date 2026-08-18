from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]


class Phase26ContinuousRequestBatchingSummaryTests(unittest.TestCase):
    def test_rejected_candidate_is_schema_valid_and_does_not_claim_gpu_batching(self) -> None:
        schema = json.loads(
            (ROOT / "ci/schema/phase26-continuous-request-batching-summary-v1.schema.json").read_text()
        )
        summary = json.loads(
            (ROOT / "ci/matrix/phase26-continuous-request-batching-summary-v1.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(summary),
            key=lambda error: list(error.absolute_path),
        )
        self.assertEqual(errors, [])
        self.assertEqual({row["target"] for row in summary["baseline"]}, {"gfx1030", "gfx1201"})
        for row in summary["baseline"]:
            self.assertAlmostEqual(
                row["concurrency_2_finish_ratio"],
                row["concurrency_2_last_ns"] / row["concurrency_2_first_ns"],
            )
            self.assertAlmostEqual(
                row["aggregate_rps_change_from_single"],
                2 * row["single_http_median_ns"] / row["concurrency_2_last_ns"] - 1,
            )
            self.assertEqual(row["gpu_batch_rows"], 1)
        self.assertTrue(summary["host_contract"]["retained"])
        self.assertFalse(summary["host_contract"]["production_connected"])
        self.assertFalse(summary["gpu_feasibility"]["production_batched_decode_implemented"])
        disposition = summary["disposition"]
        self.assertFalse(disposition["continuous_request_batching_adopted"])
        self.assertFalse(disposition["gpu_b_greater_than_one_claimed"])
        self.assertFalse(disposition["throughput_threshold_claimed"])


if __name__ == "__main__":
    unittest.main()
