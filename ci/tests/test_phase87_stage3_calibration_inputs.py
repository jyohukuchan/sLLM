#!/usr/bin/env python3
"""Focused host tests for the Phase 87 Stage 3 calibration corpus."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ci/tools/phase87_stage3_calibration_inputs.py"
SPEC = importlib.util.spec_from_file_location("phase87_stage3_calibration_inputs", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Phase87Stage3CalibrationInputsTest(unittest.TestCase):
    def test_checked_in_outputs_are_deterministic_and_disjoint(self) -> None:
        rendered, manifest, summary = module.build_artifacts()
        self.assertEqual(
            rendered,
            (ROOT / "ci/fixtures/phase87-stage3-calibration-v1/rendered/inputs.jsonl").read_bytes(),
        )
        self.assertEqual(
            manifest,
            (ROOT / "ci/fixtures/phase87-stage3-calibration-v1/manifest.json").read_bytes(),
        )
        self.assertEqual(len(summary["cases"]), 6)
        self.assertEqual(summary["manifest"]["exclusion"]["overlap_result"]["normalized_text_overlap_count"], 0)

    def test_normalized_prompt_collision_is_rejected(self) -> None:
        fixture = ROOT / "ci/fixtures/phase87-stage3-calibration-v1"
        exclusions = module._load_mtp_exclusions(
            ROOT / "ci/matrix/mtp-bench-v1.json", ROOT / "ci/fixtures/mtp-bench-v1"
        )
        source_text = next(iter(exclusions["source_texts"]))
        case = {
            "id": "intentional-collision",
            "raw_prompt_sha256": module.sha256_bytes(source_text.encode("utf-8")),
            "normalized_text": source_text,
            "normalized_text_sha256": module.sha256_bytes(source_text.encode("utf-8")),
            "rendered_prompt": source_text,
            "rendered_prompt_sha256": module.sha256_bytes(source_text.encode("utf-8")),
        }
        with self.assertRaises(module.CalibrationInputError):
            module._check_case_disjointness([case], exclusions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
