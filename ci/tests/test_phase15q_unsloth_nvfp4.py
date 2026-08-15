from __future__ import annotations

import unittest

import numpy as np

from ci.tools import analyze_unsloth_nvfp4 as analysis
from ci.tools import import_unsloth_nvfp4_sidecar as importer


class Phase15QUnslothNvfp4Tests(unittest.TestCase):
    def test_independent_numeric_tables_cover_codes_and_ties(self) -> None:
        codes = np.arange(16, dtype=np.uint8)
        expected = np.asarray(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(analysis.decode_e2m1(codes), expected)
        ties = np.asarray([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
        self.assertEqual(analysis.encode_e2m1(ties).tolist(), [0, 2, 2, 4, 4, 6, 6])

    def test_sample_quantizer_handles_zero_and_non_aligned_blocks(self) -> None:
        source = np.asarray(
            [[0.0] * 16, [(-1.0 if index & 1 else 1.0) * (index + 1) / 17 for index in range(16)]],
            dtype=np.float32,
        )
        candidate = analysis.quantize_sample(source, 1.0 / (448.0 * 6.0))
        self.assertTrue(np.isfinite(candidate).all())
        self.assertTrue(np.all(candidate[0] == 0.0))
        self.assertGreater(np.count_nonzero(candidate[1]), 0)

    def test_import_inventory_is_exact_and_ordered(self) -> None:
        names = importer.mlp_names()
        self.assertEqual(len(names), 144)
        self.assertEqual(names[0], "model.language_model.layers.0.mlp.down_proj.weight")
        self.assertEqual(names[1], "model.language_model.layers.0.mlp.gate_proj.weight")
        self.assertEqual(names[-1], "model.language_model.layers.47.mlp.up_proj.weight")
        self.assertEqual(len(set(names)), len(names))


if __name__ == "__main__":
    unittest.main()
