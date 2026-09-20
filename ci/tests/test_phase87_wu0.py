"""Check WU0 logical-byte units and signed versus selectable headroom."""

import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "tools/summarize_phase87_wu0.py"
SPEC = importlib.util.spec_from_file_location("wu0_summary", PATH)
wu0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wu0)


class WU0AccountingTests(unittest.TestCase):
    def test_resident_weight_scales_and_packed_row_tails(self):
        self.assertEqual(wu0.weight_bytes("fp8", 5120, 17408), 89_198_592)
        self.assertEqual(wu0.weight_bytes("nvfp4", 5120, 17408), 50_135_044)
        self.assertEqual(wu0.weight_bytes("nvfp4", 17, 3), 3 * 9 + 3 * 2 + 4)

    def test_negative_read_reference_margin_remains_visible(self):
        shape = {"m": 1, "k": 1024, "n": 1024, "encoding": "fp8"}
        row = {"family": "fp8_matmul", "invocation_count": 127,
               "duration_ms_per_transition": 1.0, "shape_candidates": [shape]}
        probe = {(1, 1024, 1024, "fp8"):
                 {"read_gbps": {"median": 0.5}, "payload_bytes": 1_053_700}}
        result = wu0.headroom_for_row({"mtp": False, "committed_decode_transitions": 127}, row, probe)
        self.assertAlmostEqual(result["time_floor_ms_per_token"], 2.105344)
        self.assertAlmostEqual(result["headroom_signed_ms"], -1.105344)
        self.assertEqual(result["headroom_positive_ms"], 0)
        self.assertTrue(result["probe_slower_than_current_kernel"])

    def test_half_cutoff_uses_positive_shape_margins_not_net_sum(self):
        rows = [{"family": "fp8_matmul", "headroom": {
            "state": "exact", "observed_ms_per_token": observed,
            "time_floor_ms_per_token": floor, "headroom_signed_ms": observed - floor,
            "headroom_positive_ms": max(0, observed - floor)}}
            for observed, floor in [(4, 2), (1, 2)]]
        result = wu0.aggregate_headroom(rows)["fp8_matmul"]
        self.assertEqual(result["headroom_signed_ms"], 1)
        self.assertEqual(result["headroom_positive_ms"], 2)
        self.assertEqual(result["cutoff_half_positive_ms"], 1)

    def test_quantizer_shape_maps_to_bf16_input_bytes(self):
        row = {"shape_candidates": [{"m": 3, "k": 5120, "n": 1, "encoding": "bytes"}]}
        probe = {(1, 30720, 1, "bytes"): {"payload_bytes": 30720,
                                          "read_gbps": {"median": 20.0}}}
        result = wu0.stage0_read_match(row, probe)
        self.assertEqual(result["matched_payloads"], [30720])


if __name__ == "__main__":
    unittest.main()
