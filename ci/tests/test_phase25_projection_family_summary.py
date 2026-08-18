from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]


class Phase25ProjectionFamilySummaryTests(unittest.TestCase):
    def test_negative_discovery_is_schema_valid_and_bounded(self) -> None:
        schema = json.loads(
            (ROOT / "ci/schema/phase25-projection-family-summary-v1.schema.json").read_text()
        )
        summary = json.loads(
            (ROOT / "ci/matrix/phase25-projection-family-summary-v1.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(summary),
            key=lambda error: list(error.absolute_path),
        )
        self.assertEqual(errors, [])
        self.assertEqual({row["target"] for row in summary["profiles"]}, {"gfx1030", "gfx1201"})
        for row in summary["profiles"]:
            self.assertAlmostEqual(
                row["projection_device_share"],
                row["projection_device_time_ns"] / row["kernel_device_time_ns"],
            )
            self.assertAlmostEqual(
                row["profiled_launch_average_ns"],
                row["hip_launch_total_ns"] / row["hip_launch_calls"],
            )
            self.assertLess(row["gate_up_launch_only_upper_bound_fraction_of_tpot"], 0.05)
        accounting = summary["candidate_accounting"]
        self.assertLess(accounting["maximum_shared_activation_fraction_of_weight_traffic"], 0.001)
        self.assertTrue(accounting["gate_up_upper_bound_below_five_percent_on_both_targets"])
        self.assertFalse(summary["disposition"]["production_source_changed"])
        self.assertTrue(summary["disposition"]["phase26_unblocked"])


if __name__ == "__main__":
    unittest.main()
