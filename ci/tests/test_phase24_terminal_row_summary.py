from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]


class Phase24TerminalRowSummaryTests(unittest.TestCase):
    def test_checked_in_summary_matches_schema_and_adoption_contract(self) -> None:
        schema = json.loads(
            (ROOT / "ci/schema/phase24-terminal-row-summary-v1.schema.json").read_text()
        )
        summary = json.loads(
            (ROOT / "ci/matrix/phase24-terminal-row-summary-v1.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(
                schema, format_checker=FormatChecker()
            ).iter_errors(summary),
            key=lambda error: list(error.absolute_path),
        )
        self.assertEqual(errors, [])
        expected_cases = {
            (target, case)
            for target in ("gfx1030", "gfx1201")
            for case in ("P0", "P1", "P2", "P3", "D0")
        }
        self.assertEqual(
            {(row["target"], row["case"]) for row in summary["measurements"]},
            expected_cases,
        )
        threshold = summary["frozen_contract"]["minimum_improvement_any_pattern"]
        for row in summary["measurements"]:
            self.assertAlmostEqual(
                row["e2e_improvement"],
                1 - row["candidate_e2e_ns"] / row["baseline_e2e_ns"],
            )
            self.assertEqual(row["no_regression_pass"], row["e2e_improvement"] >= 0)
            self.assertEqual(
                row["five_percent_pass"], row["e2e_improvement"] >= threshold
            )
        self.assertTrue(
            all(row["no_regression_pass"] for row in summary["measurements"])
        )
        self.assertTrue(
            any(row["five_percent_pass"] for row in summary["measurements"])
        )
        memory = summary["mechanism_and_memory"]
        self.assertEqual(
            memory["workspace_reduction_bytes"],
            memory["baseline_workspace_high_water_bytes"]
            - memory["candidate_workspace_high_water_bytes"],
        )
        self.assertTrue(summary["disposition"]["production_candidate_retained"])
        self.assertFalse(summary["disposition"]["candidate_source_reverted"])
        self.assertTrue(summary["disposition"]["shared_target_path_retained"])
        self.assertFalse(summary["disposition"]["target_specific_split_added"])


if __name__ == "__main__":
    unittest.main()
