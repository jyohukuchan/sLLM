#!/usr/bin/env python3
"""Focused host tests for the Stage 12 v2 benchmark identity."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))
import phase87_stage12_acceptance as acceptance  # noqa: E402
import phase87_stage12_m4 as m4  # noqa: E402


class Phase87Stage12AcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.identity = acceptance.load_identity()

    def _prompt_rows(self, width: int, offset: float = 0.0) -> list[dict]:
        rows = []
        for index, case_id in enumerate(sorted(self.identity["conditions"])):
            rows.append({
                "case_id": case_id,
                "prompt_sha256": "sha256:" + self.identity["frozen_prefixes"][case_id]["prompt_tokens_le_i32_sha256"],
                "output_prefix_sha256": "sha256:" + self.identity["frozen_prefixes"][case_id]["output_prefix_tokens_le_i32_sha256"],
                "by_proposal_step": {
                    f"step{step}": 0.50 + offset + step * 0.01 + index * 0.0001
                    for step in range(1, width + 1)
                },
            })
        return rows

    def _m1_report(self, width: int, target: str = "gfx1030", offset: float = 0.0) -> dict:
        return {
            "schema_version": "mtp-expected-acceptance-v2-input",
            "state": "PASS",
            "target": target,
            "model_sha256": "a" * 64,
            "width": width,
            "manifest_sha256": self.identity["manifest_sha256"],
            "cleanup": {"zero": True},
            "support_transform": {"top_k": 20, "top_p": 0.95},
            "per_prompt": self._prompt_rows(width, offset),
        }

    def test_width_boundaries_and_cumulative_products(self) -> None:
        self.assertAlmostEqual(acceptance.expected_tokens_per_block([0.8, 0.7], 2), 2.36)
        self.assertAlmostEqual(acceptance.expected_tokens_per_block([0.8, 0.7, 0.6], 3), 2.696)
        self.assertAlmostEqual(acceptance.expected_tokens_per_block([0.8, 0.7, 0.6, 0.5], 4), 2.864)
        for width in (1, 5):
            with self.assertRaises(acceptance.Stage12IdentityError):
                acceptance.validate_width(width)

    def test_missing_proposal_step_is_rejected(self) -> None:
        with self.assertRaisesRegex(acceptance.Stage12IdentityError, "missing proposal step"):
            acceptance.expected_tokens_per_block({"step1": 0.8, "step3": 0.6}, 3)
        report = self._m1_report(3)
        del report["per_prompt"][0]["by_proposal_step"]["step2"]
        with self.assertRaises(acceptance.Stage12IdentityError):
            acceptance._normalise_prompt_steps(report["per_prompt"], 3, None)

    def test_mismatched_m1_target_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._m1_report(2, "gfx1030")
            candidate = self._m1_report(2, "gfx1201")
            base_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            base_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaisesRegex(acceptance.Stage12IdentityError, "targets differ"):
                acceptance.calculate(base_path, candidate_path, 2)

    def test_compact_m1_without_frozen_prompt_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._m1_report(2)
            del baseline["per_prompt"][0]["prompt_sha256"]
            candidate = self._m1_report(2)
            base_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            base_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaisesRegex(acceptance.Stage12IdentityError, "compact prompt/token hash differs"):
                acceptance.calculate(base_path, candidate_path, 2)

    def test_v2_manifest_reuses_frozen_v1_tree(self) -> None:
        self.assertEqual(self.identity["source_manifest_path"].name, "mtp-bench-v1.json")
        self.assertEqual(len(self.identity["conditions"]), 26)
        self.assertEqual(self.identity["manifest"]["fixture_reuse"]["copied_files"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
