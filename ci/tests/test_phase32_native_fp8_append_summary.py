import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "ci/matrix/phase32-native-fp8-append-summary-v1.json"
SCHEMA = ROOT / "ci/schema/phase32-native-fp8-append-summary-v1.schema.json"


class Phase32SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(SUMMARY.read_text())
        cls.schema = json.loads(SCHEMA.read_text())

    def test_schema(self):
        jsonschema.Draft202012Validator(self.schema).validate(self.summary)

    def test_prototype_is_bit_exact_and_target_scoped(self):
        build = self.summary["build"]
        self.assertTrue(build["gfx1201_compile"])
        self.assertTrue(build["gfx1030_compile"])
        self.assertTrue(build["wrong_target_load_rejected"])
        self.assertNotEqual(build["wrong_target_exit_code"], 0)
        self.assertGreater(build["production_gfx1201_native_instruction_occurrences"], 0)
        self.assertEqual(build["production_gfx1030_native_instruction_occurrences"], 0)
        self.assertEqual(build["production_oracle_cases_passed"], build["production_oracle_cases_total"])
        operator = self.summary["operator"]
        self.assertEqual(operator["gfx1201"]["mismatch_sum"], 0)
        self.assertEqual(operator["gfx1201"]["exhaustive_bf16_codes_across_key_value"], 65536)
        self.assertEqual(operator["gfx1030"]["mismatch_sum"], 0)
        self.assertEqual(operator["gfx1030"]["providers"], ["software"])
        self.assertFalse(operator["gfx1030"]["native_available"])

    def test_ai_judgment_adopts_the_low_maintenance_scalar_path(self):
        operator = self.summary["operator"]["gfx1201"]
        self.assertGreater(operator["native_scalar_min_time_reduction_percent"], 0.0)
        for row in self.summary["operator"]["primary_rows"]:
            self.assertGreater(row["time_reduction_percent"], 0.0)
            self.assertLess(row["native_scalar_median_ns"], row["software_median_ns"])
        policy = self.summary["policy"]
        self.assertEqual(policy["decision_authority"], "assigned-ai-contextual-judgment")
        self.assertFalse(policy["fixed_performance_threshold"])
        self.assertTrue(policy["production_candidate_integrated"])
        self.assertTrue(policy["production_source_changed"])
        self.assertEqual(policy["gfx1201_provider"], "native-scalar")
        self.assertEqual(self.summary["candidate_decisions"]["native_packed"], "REJECTED_ADDITIONAL_WORKGROUP_STORE_AND_TAIL_COMPLEXITY_NOT_JUSTIFIED")

    def test_small_full_model_share_is_diagnostic_not_a_fixed_gate(self):
        for row in self.summary["full_model"]:
            calculated_share = 100.0 * row["append_total_device_ns"] / row["timing_ns"]
            self.assertAlmostEqual(row["append_share_percent"], calculated_share, places=8)
            self.assertLess(row["perfect_elimination_ceiling_percent"], 1.0)
            self.assertFalse(row["fallback"])
            self.assertEqual(row["cleanup_failures"], 0)
        append = self.summary["production_append"]
        calculated_reduction = 100.0 * (append["software_total_device_ns"] - append["native_scalar_total_device_ns"]) / append["software_total_device_ns"]
        self.assertAlmostEqual(append["time_reduction_percent"], calculated_reduction, places=8)
        self.assertGreater(append["time_reduction_percent"], 0.0)
        self.assertTrue(append["candidate_full_model_saving_is_below_timing_noise"])
        for row in self.summary["production_validation"]:
            self.assertEqual(row["generated_token_ids"], [1228, 1228])
            self.assertFalse(row["fallback"])
            self.assertEqual(row["cleanup_failures"], 0)
        service = self.summary["service"]
        self.assertEqual(service["prompt_tokens"], 10013)
        self.assertEqual(service["non_stream_content"], "It")
        self.assertEqual(service["sse_content"], "It")
        self.assertTrue(service["sse_done"])
        self.assertEqual(service["fallbacks"], 0)
        self.assertEqual(service["shutdown_current_bytes"], 0)
        self.assertEqual(service["shutdown_request_state_bytes"], 0)
        self.assertEqual(service["shutdown_workspace_bytes"], 0)
        self.assertEqual(service["cleanup_failures"], 0)
        self.assertEqual(self.summary["policy"]["default_kv_cache_encoding"], "fp16")


if __name__ == "__main__":
    unittest.main()
