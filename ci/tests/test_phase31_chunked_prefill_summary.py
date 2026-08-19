import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "ci/matrix/phase31-chunked-prefill-summary-v1.json"
SCHEMA = ROOT / "ci/schema/phase31-chunked-prefill-summary-v1.schema.json"


class Phase31SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(SUMMARY.read_text())
        cls.schema = json.loads(SCHEMA.read_text())

    def test_schema(self):
        jsonschema.Draft202012Validator(self.schema).validate(self.summary)

    def test_selector_policy_is_deterministic_and_fail_closed(self):
        policy = self.summary["policy"]
        self.assertEqual(policy["total_vram_at_most_16_gib_chunk_tokens"], 512)
        self.assertEqual(policy["total_vram_above_16_gib_candidates_descending"], [16384, 8192, 4096, 2048, 512])
        self.assertEqual(policy["selection_time"], "before_device_allocation_or_dispatch")
        self.assertFalse(policy["runtime_oom_retry"])
        self.assertEqual(policy["default_kv_cache_encoding"], "fp16")

    def test_arena_reduction_exceeds_acceptance_threshold(self):
        workspace = self.summary["workspace"]
        for prefix in ("input_10001", "input_16385"):
            self.assertLess(workspace[f"{prefix}_arena_high_water_bytes"], workspace[f"{prefix}_separate_allocation_bytes"])
            self.assertGreaterEqual(workspace[f"{prefix}_reduction_percent"], 5.0)
        self.assertTrue(workspace["intermediate_chunk_terminal_fence"])
        self.assertFalse(workspace["intermediate_lm_head_or_argmax"])

    def test_long_context_has_both_targets_and_real_multichunk(self):
        rows = self.summary["full_model"]
        self.assertTrue(any(row["target"] == "gfx1201" and row["input_tokens"] >= 10001 for row in rows))
        self.assertTrue(any(row["target"] == "gfx1030" and row["input_tokens"] >= 10001 for row in rows))
        self.assertTrue(any(row["target"] == "gfx1201" and row["final_source_revalidated"] for row in rows))
        self.assertTrue(any(row["target"] == "gfx1030" and row["final_source_revalidated"] for row in rows))
        self.assertTrue(any(row["input_tokens"] == 16385 and row["selected_chunk_tokens"] == 16384 and row["chunk_count"] == 2 for row in rows))
        for row in rows:
            self.assertLess(row["required_bytes"], row["available_bytes"])
            self.assertTrue(row["hip_only"])
            self.assertFalse(row["fallback"])
            self.assertEqual(row["cleanup_failures"], 0)

    def test_low_bit_long_context_and_service_are_not_default_promotion(self):
        rows = self.summary["full_model"]
        self.assertTrue(any(row["target"] == "gfx1201" and row["input_tokens"] >= 10001 and row["kv_cache_encoding"] == "fp8" for row in rows))
        self.assertTrue(any(row["target"] == "gfx1030" and row["input_tokens"] >= 10001 and row["kv_cache_encoding"] == "fp8" for row in rows))
        self.assertEqual(self.summary["service"]["kv_cache_encoding"], "fp8")
        self.assertTrue(self.summary["service"]["sse_done"])
        self.assertLess(
            self.summary["service"]["placement_incremental_required_bytes"],
            self.summary["service"]["placement_full_required_bytes"],
        )
        self.assertLess(
            self.summary["service"]["placement_incremental_required_bytes"],
            self.summary["service"]["placement_available_memory_bytes"],
        )
        self.assertEqual(self.summary["service"]["shutdown_current_request_and_workspace_bytes"], 0)
        self.assertEqual(self.summary["policy"]["default_kv_cache_encoding"], "fp16")


if __name__ == "__main__":
    unittest.main()
