"""Contract tests for the Phase 43 protocol profile fixture."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))
import validate_phase43_profiles as validator  # noqa: E402


class Phase43ProtocolProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = validator.load(validator.FIXTURE)
        cls.schema = validator.load(validator.SCHEMA)

    def test_checked_in_contract_passes(self) -> None:
        validator.validate(self.fixture, self.schema)

    def test_exact_profile_and_case_coverage(self) -> None:
        self.assertEqual(
            {profile["id"] for profile in self.fixture["profiles"]},
            {"openai-responses-v1", "anthropic-messages-v1"},
        )
        cases = self.fixture["cases"]
        self.assertGreaterEqual(len(cases["positive"]), 8)
        self.assertGreaterEqual(len(cases["rejection"]), 12)
        self.assertEqual(
            len({case["id"] for group in cases.values() for case in group}),
            sum(len(group) for group in cases.values()),
        )

    def test_pin_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["spec_pins"]["openai"]["commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "OpenAI pin changed"):
            validator.validate(mutated, self.schema)

    def test_limit_and_admission_mutations_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["limits"]["parallel_calls"]["max"] = 32
        with self.assertRaisesRegex(ValueError, "parallel call limit changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["cases"]["rejection"][0]["expected"]["admission"] = "after_gpu_admission"
        with self.assertRaisesRegex(ValueError, "validation is not pre-admission"):
            validator.validate(mutated, self.schema)

    def test_json_loader_rejects_duplicate_and_nonfinite_values(self) -> None:
        duplicate = validator.FIXTURE.parent / ".phase43-duplicate-test.json"
        duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                validator.load(duplicate)
        finally:
            duplicate.unlink()

        nonfinite = validator.FIXTURE.parent / ".phase43-nonfinite-test.json"
        nonfinite.write_text('{"value": NaN}', encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                validator.load(nonfinite)
        finally:
            nonfinite.unlink()

    def test_schema_is_draft_2020_12_and_matches_fixture(self) -> None:
        self.assertEqual(self.schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(self.schema["$id"], self.fixture["$schema"])
        self.assertIn("no_execution_boundary", self.schema["required"])
        self.assertIn("cases", self.schema["required"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
