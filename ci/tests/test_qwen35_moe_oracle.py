from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))

from qwen35_moe_oracle import (  # noqa: E402
    MoeOracleError,
    decode_e8m0,
    decode_mxfp4_rows,
    encode_e2m1,
    mxfp4_even_scale_codes,
    mxfp4_w4a4_matmul,
    quantize_mxfp4_rows,
    round_to_bf16,
    route_logits,
    router_logits,
    shared_expert_gate,
)


class Qwen35MoeOracleTests(unittest.TestCase):
    def test_ties_use_ascending_expert_id_and_stable_grouping(self) -> None:
        result = route_logits(np.zeros((2, 10), dtype=np.float32), selected=3)
        np.testing.assert_array_equal(result["expert_ids"], [[0, 1, 2], [0, 1, 2]])
        np.testing.assert_array_equal(result["expert_counts"], [2, 2, 2, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result["expert_offsets"], [0, 2, 4, 6, 6, 6, 6, 6, 6, 6, 6])
        np.testing.assert_array_equal(result["grouped_token_ids"], [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(result["grouped_topk_slots"], [0, 0, 1, 1, 2, 2])
        np.testing.assert_allclose(result["expert_weights"].sum(axis=1), 1.0, rtol=0, atol=2e-6)

    def test_required_token_boundaries_and_extreme_skew(self) -> None:
        for tokens in (1, 2, 3, 7, 8, 31, 32, 33):
            logits = np.full((tokens, 256), -20.0, dtype=np.float32)
            logits[:, 248:] = np.arange(8, dtype=np.float32)
            result = route_logits(logits)
            np.testing.assert_array_equal(result["expert_ids"][0], [255, 254, 253, 252, 251, 250, 249, 248])
            self.assertEqual(int(result["expert_counts"][255]), tokens)
            self.assertEqual(int(result["expert_offsets"][-1]), tokens * 8)

    def test_bf16_router_and_shared_gate(self) -> None:
        hidden = np.array([[1.001, -0.5, 0.25]], dtype=np.float32)
        weight = np.array([[0.5, 1.0, -2.0], [-1.0, 0.5, 0.25]], dtype=np.float32)
        expected = round_to_bf16(hidden) @ round_to_bf16(weight).T
        np.testing.assert_array_equal(router_logits(hidden, weight), expected.astype(np.float32))
        gate = shared_expert_gate(hidden, weight[:1])
        self.assertEqual(gate.shape, (1, 1))
        self.assertTrue(0.0 < float(gate[0, 0]) < 1.0)

    def test_nonfinite_and_bad_shape_fail_closed(self) -> None:
        with self.assertRaises(MoeOracleError):
            route_logits(np.array([[0.0, np.nan]], dtype=np.float32), selected=1)
        with self.assertRaises(MoeOracleError):
            router_logits(np.zeros((2, 3), dtype=np.float32), np.zeros((4, 2), dtype=np.float32))

    def test_mxfp4_even_scale_threshold_and_special_codes(self) -> None:
        maxima = np.array([0.0, 1.749, 1.75, 3.499, 3.5], dtype=np.float32)
        np.testing.assert_array_equal(mxfp4_even_scale_codes(maxima), [0, 125, 126, 126, 127])
        np.testing.assert_array_equal(
            decode_e8m0(np.array([0, 125, 126, 127, 254], dtype=np.uint8)),
            np.array([2.0**-127, 0.25, 0.5, 1.0, 2.0**127], dtype=np.float32),
        )
        with self.assertRaises(MoeOracleError):
            decode_e8m0(np.array([255], dtype=np.uint8))

    def test_mxfp4_half_even_codes_and_non_aligned_rows(self) -> None:
        values = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75], dtype=np.float32)
        np.testing.assert_array_equal(encode_e2m1(values), [0, 2, 2, 4, 4, 6, 6, 10])
        for columns in (31, 32, 33):
            source = np.arange(3 * columns, dtype=np.float32).reshape(3, columns) / np.float32(17.0)
            encoded = quantize_mxfp4_rows(source)
            self.assertEqual(encoded["packed"].shape, (3, (columns + 1) // 2))
            self.assertEqual(encoded["scale_codes"].shape, (3, (columns + 31) // 32))
            np.testing.assert_array_equal(
                decode_mxfp4_rows(encoded["packed"], encoded["scale_codes"], columns),
                encoded["decoded"],
            )

    def test_mxfp4_w4a4_stage_oracle_is_finite(self) -> None:
        rng = np.random.default_rng(1903)
        for tokens, hidden, output in ((1, 31, 7), (3, 32, 8), (7, 33, 9)):
            activation = rng.normal(0, 0.5, size=(tokens, hidden)).astype(np.float32)
            weight = rng.normal(0, 0.2, size=(output, hidden)).astype(np.float32)
            result = mxfp4_w4a4_matmul(activation, weight)
            self.assertEqual(result["output"].shape, (tokens, output))
            self.assertTrue(np.isfinite(result["output"]).all())


if __name__ == "__main__":
    unittest.main()
