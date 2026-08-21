"""Mutation and contract tests for the Phase 44 machine profile."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))
import validate_phase44_profiles as validator  # noqa: E402


class Phase44ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = validator.load(validator.FIXTURE)
        cls.schema = validator.load(validator.SCHEMA)

    def test_checked_in_contract_passes(self) -> None:
        validator.validate(self.fixture, self.schema)

    def test_exact_case_coverage_and_pre_gpu_rejections(self) -> None:
        cases = self.fixture["cases"]
        self.assertEqual(
            {case["id"] for case in cases["positive"]}, validator.POSITIVE_IDS
        )
        self.assertEqual(
            {case["id"] for case in cases["rejection"]}, validator.REJECTION_IDS
        )
        self.assertEqual(
            {case["surface"] for case in cases["positive"]},
            {"template", "reasoning", "cli", "checkpoint"},
        )
        self.assertEqual(
            {case["surface"] for case in cases["rejection"]},
            {"template", "reasoning", "cli", "checkpoint", "security"},
        )
        self.assertTrue(
            all(
                case["expected"]["admission"] == "before_gpu_admission"
                for case in cases["rejection"]
            )
        )

    def test_llama_and_minijinja_pin_mutations_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["spec_pins"]["llama_cpp"]["commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "llama.cpp commit changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["spec_pins"]["minijinja"]["version"] = "2.23.0"
        with self.assertRaisesRegex(ValueError, "MiniJinja version changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["spec_pins"]["minijinja"]["features"].remove("multi_template")
        with self.assertRaisesRegex(ValueError, "MiniJinja feature allowlist changed"):
            validator.validate(mutated, self.schema)

    def test_template_limit_and_capability_mutations_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["template_profile"]["source_limits"]["fuel_instructions"] = 1000001
        with self.assertRaisesRegex(ValueError, "template limits changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["template_profile"]["forbidden_capabilities"].remove("process")
        with self.assertRaisesRegex(ValueError, "forbidden template capabilities changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["template_profile"]["identity"]["checkpoint_binds"] = ["template_digest"]
        with self.assertRaisesRegex(ValueError, "template identity changed"):
            validator.validate(mutated, self.schema)

    def test_reasoning_and_cli_semantic_mutations_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["reasoning_control"]["budget"]["max"] = 8192
        with self.assertRaisesRegex(ValueError, "reasoning budget changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["reasoning_control"]["wire_mapping"]["responses_reasoning_effort"]["low"] = 1
        with self.assertRaisesRegex(ValueError, "reasoning wire mapping changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["interactive_cli"]["prompt_source_conflicts"].pop()
        with self.assertRaisesRegex(ValueError, "prompt-source conflict matrix changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["interactive_cli"]["checkpoint"]["identity_fields"].remove("plan_digest")
        with self.assertRaisesRegex(ValueError, "checkpoint policy changed"):
            validator.validate(mutated, self.schema)

    def test_case_boundary_and_admission_mutations_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        case = next(
            row
            for row in mutated["cases"]["rejection"]
            if row["id"] == "template-source-over-limit"
        )
        case["expected"]["admission"] = "after_gpu_admission"
        with self.assertRaisesRegex(ValueError, "validation is not pre-GPU admission"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        case = next(
            row
            for row in mutated["cases"]["positive"]
            if row["id"] == "reasoning-enabled-budget-one"
        )
        case["input"]["budget"] = 2
        with self.assertRaisesRegex(ValueError, "reasoning-enabled-budget-one: boundary changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["cases"]["rejection"][0]["id"] = "duplicate-case"
        with self.assertRaisesRegex(ValueError, "rejection case identity set changed"):
            validator.validate(mutated, self.schema)

    def test_json_loader_rejects_duplicate_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase44-contract-") as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                validator.load(duplicate)

            nonfinite = Path(directory) / "nonfinite.json"
            nonfinite.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                validator.load(nonfinite)

    def test_schema_is_draft_2020_12_and_matches_fixture(self) -> None:
        self.assertEqual(
            self.schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertEqual(self.schema["$id"], self.fixture["$schema"])
        self.assertIn("template_profile", self.schema["required"])
        self.assertIn("reasoning_control", self.schema["required"])
        self.assertIn("interactive_cli", self.schema["required"])
        self.assertIn("cases", self.schema["required"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
