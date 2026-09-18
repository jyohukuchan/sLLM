#!/usr/bin/env python3
"""Analytic integration tests for qwen38_kld_compare.py.

The fixtures are intentionally tiny, but use the same manifest, little-endian
FP32 raw-logit files, and vocabulary filtering as the full-model runner.  Each
test invokes the command line program in a subprocess so the checks cover file
validation and output serialization as well as the numerical calculation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COMPARE = ROOT / "ci" / "tools" / "qwen38_kld_compare.py"


def token_hash(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class CompareFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="qwen38-kld-compare-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_inputs(self, token_ids: list[int], positions: list[int], valid_vocab: list[int]) -> tuple[Path, Path]:
        inputs = self.root / "inputs.json"
        inputs.write_text(json.dumps({"cases": [{"id": "case", "token_ids": token_ids, "positions": positions}]}))
        vocab = self.root / "vocab.json"
        vocab.write_text(json.dumps({"valid_token_ids": valid_vocab}))
        return inputs, vocab

    def write_engine(
        self,
        name: str,
        logits: np.ndarray,
        token_ids: list[int],
        positions: list[int],
        hash_override: str | None = None,
        positions_override: list[int] | None = None,
    ) -> Path:
        root = self.root / name
        root.mkdir()
        matrix = np.asarray(logits, dtype="<f4")
        raw_name = "case.f32"
        matrix.tofile(root / raw_name)
        case_positions = positions if positions_override is None else positions_override
        metadata = {
            "engine": name,
            "cases": [{
                "id": "case",
                "shape": list(matrix.shape),
                "positions": case_positions,
                "input_token_ids_sha256": token_hash(token_ids) if hash_override is None else hash_override,
                "nonfinite_count": 0,
                "logits_file": raw_name,
            }],
        }
        (root / "manifest.json").write_text(json.dumps(metadata))
        return root

    def run_compare(
        self,
        reference_logits: np.ndarray,
        candidate_logits: np.ndarray,
        token_ids: list[int],
        positions: list[int],
        valid_vocab: list[int],
        *,
        candidate_hash: str | None = None,
        candidate_positions: list[int] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        inputs, vocab = self.write_inputs(token_ids, positions, valid_vocab)
        reference = self.write_engine("reference", reference_logits, token_ids, positions)
        candidate = self.write_engine(
            "candidate",
            candidate_logits,
            token_ids,
            positions,
            hash_override=candidate_hash,
            positions_override=candidate_positions,
        )
        output = self.root / "result.json"
        return subprocess.run(
            [
                sys.executable,
                str(COMPARE),
                "--reference", str(reference),
                "--candidate", str(candidate),
                "--inputs", str(inputs),
                "--vocab", str(vocab),
                "--output", str(output),
                "--rows-per-block", "2",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def successful_result(self, *args, **kwargs) -> dict:
        process = self.run_compare(*args, **kwargs)
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads((self.root / "result.json").read_text())

    def test_identical_logits_have_zero_kl(self) -> None:
        logits = np.asarray([[0.0, 1.0, -2.0, 3.0], [1.5, -0.5, 0.25, 2.0], [-1.0, 0.75, 2.5, -0.25]], dtype=np.float32)
        result = self.successful_result(logits, logits.copy(), [0, 1, 2], [0, 1, 2], [0, 1, 2, 3])
        self.assertEqual(result["all_rows"]["count"], 3)
        self.assertTrue(np.allclose(result["cases"][0]["position_kld"], 0.0, atol=1e-12))
        self.assertEqual(result["all_rows"]["max"], 0.0)

    def test_three_category_kl_matches_manual_oracle(self) -> None:
        # p = (1, 2, 3) / 6 and q = (3, 2, 1) / 6, so KL(p||q) = log(3)/3.
        ref_valid = np.log(np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
        cand_valid = np.log(np.asarray([3.0, 2.0, 1.0], dtype=np.float64))
        reference = np.asarray([[ref_valid[0], 77.0, ref_valid[1], -88.0, ref_valid[2]]], dtype=np.float32)
        candidate = np.asarray([[cand_valid[0], -999.0, cand_valid[1], 1234.0, cand_valid[2]]], dtype=np.float32)
        result = self.successful_result(reference, candidate, [0, 2], [0], [0, 2, 4])
        expected = float(np.sum((np.asarray([1.0, 2.0, 3.0]) / 6.0) *
                               (np.log(np.asarray([1.0, 2.0, 3.0]) / 6.0) -
                                np.log(np.asarray([3.0, 2.0, 1.0]) / 6.0))))
        self.assertAlmostEqual(result["cases"][0]["position_kld"][0], expected, places=6)

    def test_additive_logit_offset_is_softmax_invariant(self) -> None:
        reference = np.asarray([[0.0, 1.0, -1.0], [2.0, -3.0, 0.5]], dtype=np.float32)
        candidate = reference + np.asarray([[10000.0], [-10000.0]], dtype=np.float32)
        result = self.successful_result(reference, candidate, [0, 1, 2], [0, 1], [0, 1, 2])
        self.assertLess(result["all_rows"]["max"], 1e-10)

    def test_padded_vocabulary_is_excluded_from_kl_and_top1(self) -> None:
        # IDs 1 and 3 are outside the valid vocabulary and have huge candidate
        # logits.  They must not affect either KLD or the selected top-1 token.
        reference = np.asarray([[0.0, -500.0, 0.0, -500.0, 0.0]], dtype=np.float32)
        candidate = np.asarray([[0.0, 10000.0, 0.0, 9999.0, 0.0]], dtype=np.float32)
        result = self.successful_result(reference, candidate, [0, 2], [0], [0, 2, 4])
        case = result["cases"][0]
        self.assertLess(case["position_kld"][0], 1e-10)
        self.assertEqual(case["top1_agreement"], 1.0)

    def test_token_hash_mismatch_is_rejected(self) -> None:
        logits = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)
        process = self.run_compare(logits, logits, [0, 1], [0], [0, 1, 2], candidate_hash="0" * 64)
        self.assertNotEqual(process.returncode, 0)

    def test_position_mismatch_is_rejected(self) -> None:
        logits = np.asarray([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]], dtype=np.float32)
        process = self.run_compare(
            logits,
            logits,
            [0, 1, 2],
            [0, 2],
            [0, 1, 2],
            candidate_positions=[0, 1],
        )
        self.assertNotEqual(process.returncode, 0)

    def test_sparse_final_position_has_no_next_token_nll(self) -> None:
        logits = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)
        result = self.successful_result(logits, logits.copy(), [0, 1, 2], [2], [0, 1, 2])
        case = result["cases"][0]
        self.assertEqual(case["positions"], [2])
        self.assertIsNone(case["reference_mean_next_token_nll"])
        self.assertIsNone(case["candidate_mean_next_token_nll"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
