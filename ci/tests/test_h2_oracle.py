#!/usr/bin/env python3
"""Independent tiny NumPy oracle cases; no model or GPU-sized tensor is used."""

from __future__ import annotations

import json
import os
import sys
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


class TinyOracleTests(unittest.TestCase):
    def test_boundary_vector_addition(self) -> None:
        rng = np.random.default_rng(314159)
        for size in (0, 1, 3, 7, 17, 37, 73):
            left = rng.standard_normal(size, dtype=np.float32)
            right = rng.standard_normal(size, dtype=np.float32)
            expected = [np.float32(float(a) + float(b)) for a, b in zip(left, right)]
            actual = np.add(left, right).tolist()
            self.assertEqual(len(actual), size)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_stable_softmax_reference(self) -> None:
        for values in ([-3.0, 0.0, 3.0], [1000.0, 1001.0], [-1000.0, -999.0], [0.0]):
            vector = np.asarray(values, dtype=np.float64)
            shifted = vector - np.max(vector)
            exponent = np.exp(shifted)
            actual = exponent / np.sum(exponent)
            self.assertTrue(np.isfinite(actual).all())
            self.assertAlmostEqual(float(np.sum(actual)), 1.0, places=15)
            self.assertEqual(int(np.argmax(actual)), int(np.argmax(vector)))

    def test_kv_page_index_boundaries(self) -> None:
        page = 16
        for position in (0, 1, 15, 16, 17, 37, 73):
            page_index, offset = divmod(position, page)
            self.assertEqual(page_index * page + offset, position)
            self.assertGreaterEqual(offset, 0)
            self.assertLess(offset, page)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if os.environ.get("ULLM_EMIT_TEST_COUNTS") == "1":
        selected = result.testsRun
        failed = len(result.failures) + len(result.errors)
        skipped = len(result.skipped)
        print(
            "ULLM_UNITTEST_COUNTS="
            + json.dumps(
                {
                    "collected": selected,
                    "selected": selected,
                    "passed": selected - failed - skipped,
                    "failed": failed,
                    "skipped": skipped,
                    "deselected": 0,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
