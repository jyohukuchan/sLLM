import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]


class Phase28NonprojectionSummaryTests(unittest.TestCase):
    def test_explicit_exception_is_bounded_and_arithmetic_is_consistent(self) -> None:
        schema = json.loads((ROOT / "ci/schema/phase28-nonprojection-summary-v1.schema.json").read_text())
        summary = json.loads((ROOT / "ci/matrix/phase28-nonprojection-summary-v1.json").read_text())
        Draft202012Validator.check_schema(schema)
        self.assertEqual(list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(summary)), [])
        self.assertIn(">=5%", summary["adoption_rule"])
        self.assertEqual(summary["decision"], "ADOPTED_BY_EXPLICIT_USER_EXCEPTION")
        self.assertTrue(summary["implementation"]["shared_path"])
        self.assertFalse(summary["implementation"]["target_split_added"])
        for row in summary["exact_step_baseline"]:
            self.assertAlmostEqual(row["nonprojection_ns_per_step"], row["device_ns_per_step"] - row["projection_ns_per_step"], delta=2)
        for row in summary["candidate"]:
            expected = (row["candidate_tokens_per_second"] / row["baseline_tokens_per_second"] - 1) * 100
            self.assertAlmostEqual(row["full_model_improvement_percent"], expected)
            self.assertLess(row["full_model_improvement_percent"], 5)
            self.assertTrue(row["token_ids_equal"] and row["all_dispatches_hip"] and row["cleanup_valid"])
            self.assertFalse(row["fallback_used"])


if __name__ == "__main__":
    unittest.main()
