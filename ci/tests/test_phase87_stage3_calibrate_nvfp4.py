"""Check the Stage 3 scale-site contract and GPU report fail-closed path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools/phase87_stage3_calibrate_nvfp4.py"
SPEC = importlib.util.spec_from_file_location("stage3_calibrate", TOOL)
assert SPEC is not None and SPEC.loader is not None
stage3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage3)


def valid_report() -> dict:
    observations = {
        "mtp.concat.output": 12.0,
        "layer.64.input_rmsnorm.output": 16.0,
        "layer.64.full.sigmoid_mul.output": 24.0,
        "layer.64.post_attention_rmsnorm.output": 32.0,
        "layer.64.mlp.silu_mul.output": 48.0,
    }
    return {
        "state": "PASS",
        "target": "gfx1201",
        "device_index": 0,
        "series": "bf16-calibration",
        "companion_encoding": None,
        "cleanup": {"zero": True},
        "secondary_mtp": {
            "suite_file_sha256": "sha256:suite",
            "warmups": 0,
            "measured": 1,
            "seed": 123,
            "selected_cases": ["held-out"],
            "entries": [{
                "case_id": "held-out",
                "prompt_sha256": "sha256:prompt",
                "prompt_token_count": 10,
                "output_tokens": 32,
                "run": {
                    "generated_tokens": list(range(32)),
                    "audit": {
                        "selected_backend": "hip",
                        "target": "gfx1201",
                        "fallback_used": False,
                        "all_dispatches_hip": True,
                        "kernel_dispatch_count": 1,
                    },
                    "mtp": {
                        "draft_fallback_used": False,
                        "draft_all_dispatches_hip": True,
                    },
                    "allocation_after_request_drop": {
                        "request_state": {"current_bytes": 0},
                        "workspace": {"current_bytes": 0},
                        "poisoned": False,
                    },
                    "phase87_stage3_calibration_amax": observations,
                },
            }],
        },
    }


class CalibrationTest(unittest.TestCase):
    def test_o_projection_has_a_separate_activation_site(self) -> None:
        report = valid_report()
        maxima = stage3.collect(report, "sha256:suite", {"held-out": ("sha256:prompt", 10, 32)})
        assert maxima["layer.64.full.sigmoid_mul.output"] == 24.0
        self.assertEqual(
            stage3.WEIGHT_SITE["mtp.layers.0.self_attn.o_proj.weight"],
            "layer.64.full.sigmoid_mul.output",
        )
        self.assertNotEqual(
            stage3.WEIGHT_SITE["mtp.layers.0.self_attn.o_proj.weight"],
            stage3.WEIGHT_SITE["mtp.layers.0.self_attn.q_proj.weight"],
        )

    def test_missing_site_and_fallback_do_not_calibrate(self) -> None:
        report = valid_report()
        del report["secondary_mtp"]["entries"][0]["run"][
            "phase87_stage3_calibration_amax"
        ]["layer.64.mlp.silu_mul.output"]
        with self.assertRaisesRegex(ValueError, "incomplete activation sites"):
            stage3.collect(report, "sha256:suite", {"held-out": ("sha256:prompt", 10, 32)})
        report = valid_report()
        report["secondary_mtp"]["entries"][0]["run"]["audit"]["fallback_used"] = True
        with self.assertRaisesRegex(ValueError, "HIP-only"):
            stage3.collect(report, "sha256:suite", {"held-out": ("sha256:prompt", 10, 32)})
        report = valid_report()
        report["secondary_mtp"]["entries"][0]["prompt_sha256"] = "sha256:another"
        with self.assertRaisesRegex(ValueError, "prompt identity"):
            stage3.collect(report, "sha256:suite", {"held-out": ("sha256:prompt", 10, 32)})


if __name__ == "__main__":
    unittest.main()
