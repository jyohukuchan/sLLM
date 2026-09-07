"""Numerical and fail-closed checks for fixed-input capture comparisons."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

_spec = importlib.util.spec_from_file_location(
    "phase79_compare_logits", Path(__file__).parents[1] / "tools/phase79_compare_logits.py"
)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


def capture():
    return {
        "schema_version": "phase79-gemma-fixed-logits-v1", "state": "CAPTURED",
        "cleanup_current_bytes": 0, "cleanup_retryable": 0, "cleanup_durable": 0,
        "model_fingerprint": "model", "target": "gfx1030", "device_index": 0,
        "kv_encoding": "fp16",
        "fixture": [{"id": "case", "prompt": [2, 3, 4], "continuation": [5]}],
        "rows": [{"id": "case", "kernel_dispatch_count": 3, "fallback_used": False,
                  "positions": [{"position": 2, "logits": [0.0, 0.0]},
                                {"position": 3, "logits": [1.0, 0.0]}]}],
    }


class ComparisonTests(unittest.TestCase):
    def test_analytic_kl_and_large_logit_shift(self):
        result = module.compare_logits([0.0, 0.0], [math.log(3), 0.0])
        self.assertAlmostEqual(result["kld_reference_to_candidate"], math.log(4/3)/2)
        shifted = module.compare_logits([10000.0, 10000.0], [10000+math.log(3), 10000.0])
        self.assertAlmostEqual(shifted["kld_reference_to_candidate"], result["kld_reference_to_candidate"], places=10)

    def test_comparison_does_not_promote_quality(self):
        result = module.compare(capture(), capture())
        self.assertEqual(result["summary"]["count"], 2)
        self.assertEqual(result["summary"]["max_kld"], 0)
        self.assertEqual(result["quality_verdict"], "not_evaluated")

    def test_incomplete_or_mismatched_capture_is_rejected(self):
        changes = [
            lambda v: v.update(target="gfx1201"),
            lambda v: v.update(rows=[]),
            lambda v: v["rows"][0]["positions"].pop(),
            lambda v: v["rows"][0].update(kernel_dispatch_count=0),
            lambda v: v["rows"][0].update(fallback_used=True),
            lambda v: v.update(cleanup_current_bytes=1),
            lambda v: v["fixture"][0]["continuation"].append(6),
            lambda v: v["rows"][0]["positions"][1].update(position=2),
            lambda v: v["rows"][0]["positions"][0].update(logits=[math.nan, 0]),
        ]
        for change in changes:
            candidate = copy.deepcopy(capture())
            change(candidate)
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.compare(capture(), candidate)


if __name__ == "__main__":
    unittest.main()
