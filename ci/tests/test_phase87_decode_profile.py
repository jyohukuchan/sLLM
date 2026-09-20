"""Checks for byte accounting and ambiguous profile attribution."""

import importlib.util
from pathlib import Path
import unittest
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "tools/phase87_decode_profile.py"
SPEC = importlib.util.spec_from_file_location("phase87_decode_profile", SOURCE)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class DecodeAccountingTests(unittest.TestCase):
    def test_kv_fp8_attention_is_not_a_weight_matmul(self):
        self.assertEqual(profile._family("sllm_causal_attention_mxfp8_v1"), "full_attention")
        self.assertEqual(profile._family("sllm_kv_append_mxfp8_v1"), "kv_cache")
        self.assertEqual(profile._family("sllm_matmul_bf16_to_fp8_v1"), "activation_quantize")

    def test_nvfp4_rows_round_independently_and_scales_are_scalar(self):
        # Three weight rows and two activation rows: each has nine packed
        # value bytes and two scale bytes. Two tensor scales add eight bytes.
        self.assertEqual(profile._estimate_bytes("nvfp4", 2, 17, 3), 63)
        self.assertEqual(profile._estimate_bytes("nvfp4", 3, 16, 3), 62)
        self.assertEqual(profile._estimate_bytes("nvfp4", 3, 15, 3), 62)

    def test_repeated_dispatches_each_read_their_payload(self):
        item = {"kernel_name": "matmul", "grid_x": 256,
                "dispatch_count": 7, "duration_ns": 7000}
        specs = [profile._parse_shape_spec("matmul=2x17x3:nvfp4")]
        profile._attach_estimates([item], specs)
        self.assertEqual(item["logical_read_bytes_per_dispatch_estimate"], 63)
        self.assertEqual(item["logical_read_bytes_estimate"], 441)

    def test_conflicting_shape_rules_are_rejected(self):
        specs = [("mat*", None, 1, 16, 32, "fp8"),
                 ("matmul", 256, 1, 32, 32, "fp8")]
        with self.assertRaises(profile.Phase87ProfileError):
            profile._shape_for("matmul", 256, specs)

    def test_roctx_function_column_and_boundary_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marker.csv"
            path.write_text("Domain,Function,Start_Timestamp,End_Timestamp\n"
                            "MARKER_CORE_RANGE_API,sllm_phase87_decode_mtp,100,200\n")
            rows = [{"dispatch_id": 7, "start_ns": 110, "end_ns": 190,
                     "name": "matmul"}]
            selected, segment = profile._segment_from_marker(rows, path, "decode_mtp", -1)
            self.assertEqual(len(selected), 1)
            self.assertEqual(segment["timestamp_gap_ns"], 20)
            rows[0]["start_ns"] = 90
            with self.assertRaises(profile.Phase87ProfileError):
                profile._segment_from_marker(rows, path, "decode_mtp", -1)


if __name__ == "__main__":
    unittest.main()
