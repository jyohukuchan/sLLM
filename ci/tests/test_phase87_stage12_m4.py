#!/usr/bin/env python3
"""Focused host tests for Stage 12 M4 identity and width math."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))
import phase87_stage12_acceptance as acceptance  # noqa: E402
import phase87_stage12_m4 as m4  # noqa: E402


class Phase87Stage12M4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.identity = acceptance.load_identity()

    def _m1(self, width: int) -> dict:
        return {
            "schema_version": "mtp-expected-acceptance-v2",
            "state": "PASS",
            "width": width,
            "target": "gfx1030",
            "model_sha256": "a" * 64,
            "manifest_sha256": self.identity["manifest_sha256"],
            "per_prompt": [
                {
                    "case_id": case_id,
                    "prompt_sha256_baseline": "sha256:" + self.identity["frozen_prefixes"][case_id]["prompt_tokens_le_i32_sha256"],
                    "prompt_sha256_candidate": "sha256:" + self.identity["frozen_prefixes"][case_id]["prompt_tokens_le_i32_sha256"],
                    "by_proposal_step": {
                        f"step{step}": {"baseline": 0.8, "candidate": 0.85}
                        for step in range(1, width + 1)
                    },
                }
                for case_id in sorted(self.identity["conditions"])
            ],
        }

    def _m3(self, target: str = "gfx1030") -> dict:
        return {
            "state": "PASS",
            "target": target,
            "model_sha256": "a" * 64,
            "execution_binary_sha256": "b" * 64,
            "manifest_sha256": self.identity["manifest_sha256"],
            "per_prompt": [
                {
                    "case_id": case_id,
                    "prompt_sha256": "sha256:" + self.identity["frozen_prefixes"][case_id]["prompt_tokens_le_i32_sha256"],
                    "median": {
                        "draft_ms_per_block": 3.0,
                        "non_draft_ms_per_block": 7.0,
                        "total_ms_per_block": 10.0,
                        "blocks": 10,
                    },
                }
                for case_id in sorted(self.identity["conditions"])
            ],
        }

    def test_width4_m4_contains_per_prompt_rows(self) -> None:
        result = m4.calculate_from_records(
            self._m1(4), self._m3(), self._m3(), 4, self.identity
        )
        self.assertEqual(result["schema_version"], "phase87-stage12-m4-v2")
        self.assertEqual(result["conditions"], 26)
        self.assertEqual(len(result["per_prompt"]), 26)
        self.assertAlmostEqual(result["per_prompt"][0]["baseline_expected_tokens_per_block"], 3.3616)

    def test_missing_m1_step_is_rejected_before_m4(self) -> None:
        m1 = self._m1(3)
        del m1["per_prompt"][0]["by_proposal_step"]["step3"]
        with self.assertRaisesRegex(acceptance.Stage12IdentityError, "proposal steps differ"):
            m4.calculate_from_records(m1, self._m3(), self._m3(), 3, self.identity)

    def test_missing_m1_frozen_prompt_hash_is_rejected(self) -> None:
        m1 = self._m1(3)
        del m1["per_prompt"][0]["prompt_sha256_baseline"]
        with self.assertRaisesRegex(acceptance.Stage12IdentityError, "frozen prompt identity differs"):
            m4.calculate_from_records(m1, self._m3(), self._m3(), 3, self.identity)

    def test_mismatched_m3_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(acceptance.Stage12IdentityError, "target identities differ"):
            m4.calculate_from_records(self._m1(2), self._m3("gfx1030"), self._m3("gfx1201"), 2, self.identity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
