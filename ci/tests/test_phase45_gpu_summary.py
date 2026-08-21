"""Mutation and evidence-boundary tests for the Phase 45 GPU summary."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))
import validate_phase45_gpu_summary as validator  # noqa: E402


class Phase45GpuSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = validator.load(validator.SUMMARY)
        cls.schema = validator.load(validator.SCHEMA)

    def test_checked_in_summary_passes(self) -> None:
        validator.validate(self.summary, self.schema)

    def test_exact_target_and_case_coverage(self) -> None:
        self.assertEqual({row["target"] for row in self.summary["targets"]}, {"gfx1030", "gfx1201"})
        self.assertEqual(set(self.summary["cases"]), {"disabled", "lora", "control", "combined"})
        self.assertFalse(self.summary["raw_artifacts_tracked"])

    def test_mutations_are_rejected(self) -> None:
        for field, value, message in (
            ("fallback", True, "fallback claim changed"),
            ("hip_only", False, "HIP-only claim changed"),
            ("resident_bytes", 1, "resident bytes changed"),
            ("final_allocations", 1, "allocation cleanup changed"),
        ):
            mutated = copy.deepcopy(self.summary)
            mutated["targets"][0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                validator.validate(mutated, self.schema)

        mutated = copy.deepcopy(self.summary)
        mutated["gfx942"] = "pass"
        with self.assertRaisesRegex(ValueError, "gfx942 boundary changed"):
            validator.validate(mutated, self.schema)


if __name__ == "__main__":
    unittest.main(verbosity=2)
