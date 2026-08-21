#!/usr/bin/env python3
"""Validate the compact Phase 42 exact-GPU evidence summary."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "ci/matrix/phase42-inference-gpu-summary-v1.json"


class Phase42InferenceGpuSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(SUMMARY.read_text(encoding="utf-8"))

    def test_identity_and_exact_target_matrix(self) -> None:
        self.assertEqual(self.document["schema"], "sllm-phase42-inference-gpu-summary-v1")
        rows = self.document["rows"]
        self.assertEqual(
            {(row["target"], row["model"]) for row in rows},
            {
                ("gfx1030", "qwen"),
                ("gfx1030", "gemma"),
                ("gfx1201", "qwen"),
                ("gfx1201", "gemma"),
            },
        )
        self.assertTrue(all(row["result"] == "PASS" for row in rows))
        self.assertTrue(all(row["target_only"] for row in rows))
        self.assertTrue(all(not row["fallback_used"] for row in rows))

    def test_embedding_oracles(self) -> None:
        models = self.document["models"]
        expected_tokens = self.document["scope"]["input_tokens"]
        for row in self.document["rows"]:
            with self.subTest(target=row["target"], model=row["model"]):
                self.assertEqual(row["rows"], 1)
                self.assertEqual(row["dimension"], models[row["model"]]["hidden_dimension"])
                self.assertTrue(row["finite"])
                self.assertTrue(math.isfinite(row["l2_norm"]))
                self.assertLessEqual(abs(row["l2_norm"] - 1.0), 2e-9)
                self.assertEqual(row["prompt_tokens"], expected_tokens)
                self.assertEqual(row["total_tokens"], expected_tokens)
                self.assertRegex(row["binary_sha256"], r"^[0-9a-f]{64}$")

    def test_mi300x_is_compile_only(self) -> None:
        gfx942 = self.document["gfx942"]
        self.assertEqual(gfx942["target"], "gfx942:sramecc+:xnack-")
        self.assertEqual(gfx942["result"], "compile-only PASS")
        self.assertEqual(gfx942["real_gpu_result"], "deferred")


if __name__ == "__main__":
    unittest.main()
