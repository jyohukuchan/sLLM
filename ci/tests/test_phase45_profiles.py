"""Mutation and contract tests for the Phase 45 machine profile."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))
import validate_phase45_profiles as validator  # noqa: E402


class Phase45ProfileTests(unittest.TestCase):
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
        self.assertTrue(
            all(
                case["expected"]["admission"] == "before_gpu_admission"
                for case in cases["rejection"]
            )
        )

    def test_pins_and_scope_mutations_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["spec_pins"]["llama_cpp"]["commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "llama.cpp commit changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["spec_pins"]["artifact_source"] = "network"
        with self.assertRaisesRegex(ValueError, "artifact source policy changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["scope"]["fallback"] = True
        with self.assertRaisesRegex(ValueError, "scope changed"):
            validator.validate(mutated, self.schema)

    def test_adapter_and_control_bounds_are_pinned(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["lora"]["rank"]["max"] = 512
        with self.assertRaisesRegex(ValueError, "LoRA policy changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["lora"]["scale"]["min"] = -32.0
        with self.assertRaisesRegex(ValueError, "LoRA policy changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["control_vectors"]["overlap"] = "compose"
        with self.assertRaisesRegex(ValueError, "control-vector policy changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["lora"]["disabled_identity"] = "adapter:empty-v2"
        with self.assertRaisesRegex(ValueError, "LoRA policy changed"):
            validator.validate(mutated, self.schema)

    def test_identity_registry_and_router_mutations_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["identity"]["fields"].remove("derived_plan_digest")
        with self.assertRaisesRegex(ValueError, "identity policy changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["registry"]["max_loaded_models"] = 17
        with self.assertRaisesRegex(ValueError, "registry policy changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["registry"]["lru"] = "evict-active"
        with self.assertRaisesRegex(ValueError, "registry policy changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["router"]["admin_action"] = "path-or-alias"
        with self.assertRaisesRegex(ValueError, "router policy changed"):
            validator.validate(mutated, self.schema)

    def test_manifest_cli_and_verification_boundaries_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["manifest"]["network"] = True
        with self.assertRaisesRegex(ValueError, "manifest policy changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["cli_server"]["admin_arguments"] = "path"
        with self.assertRaisesRegex(ValueError, "CLI/server boundary changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["verification"]["mi300x_real"] = "pass"
        with self.assertRaisesRegex(ValueError, "verification matrix changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["verification"]["full_model_smoke"] = "pass"
        with self.assertRaisesRegex(ValueError, "verification matrix changed"):
            validator.validate(mutated, self.schema)

    def test_case_boundary_mutations_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        case = next(
            row
            for row in mutated["cases"]["rejection"]
            if row["id"] == "lora-rank-zero"
        )
        case["expected"]["admission"] = "after_gpu_admission"
        with self.assertRaisesRegex(ValueError, "validation is not pre-GPU admission"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        case = next(
            row
            for row in mutated["cases"]["positive"]
            if row["id"] == "lora-qwen-bf16-single-scale-one"
        )
        case["expected"]["result"] = "rejected"
        with self.assertRaisesRegex(ValueError, "positive result changed"):
            validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.fixture)
        mutated["cases"]["rejection"][0]["id"] = "duplicate-case"
        with self.assertRaisesRegex(ValueError, "rejection case identity set changed"):
            validator.validate(mutated, self.schema)

    def test_json_loader_rejects_duplicate_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase45-contract-") as directory:
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
        for required in (
            "spec_pins",
            "scope",
            "lora",
            "control_vectors",
            "identity",
            "registry",
            "router",
            "manifest",
            "cli_server",
            "verification",
            "cases",
        ):
            self.assertIn(required, self.schema["required"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
