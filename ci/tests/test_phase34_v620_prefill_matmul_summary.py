import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "ci/matrix/phase34-v620-prefill-matmul-summary-v1.json"
SCHEMA = ROOT / "ci/schema/phase34-v620-prefill-matmul-summary-v1.schema.json"


class Phase34SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(SUMMARY.read_text())
        cls.schema = json.loads(SCHEMA.read_text())

    def test_schema(self):
        jsonschema.Draft202012Validator(self.schema).validate(self.summary)

    def test_adoption_is_exact_shape_scoped_and_fail_closed(self):
        self.assertEqual(self.summary["state"], "COMPLETE")
        decisions = self.summary["decisions"]
        self.assertEqual(decisions["gfx1030_long_prefill_hipblas"], "ADOPTED_SHAPE_SCOPED")
        self.assertEqual(decisions["performance_authority"], "assigned-ai-contextual-judgment")
        routing = self.summary["routing"]
        self.assertEqual(routing["target"], "gfx1030")
        self.assertEqual(routing["main_min_m"], 128)
        self.assertEqual(routing["full_attention_kv_min_m"], 1024)
        self.assertEqual(len(routing["main_shapes"]), 5)
        self.assertFalse(routing["gfx1030_hipblaslt_created"])
        self.assertFalse(routing["runtime_failure_fallback"])

    def test_operator_and_full_model_gain_support_adoption(self):
        weighted = self.summary["operator"]["weighted_10001_projection"]
        self.assertEqual(weighted["call_count"], 248)
        self.assertLess(weighted["hipblas_ns"], weighted["tiled16_ns"])
        self.assertGreater(weighted["reduction_percent"], 80.0)
        final = self.summary["full_model"]["gfx1030_fp16_10001"]
        self.assertLess(final["final_ns"], final["baseline_ns"])
        self.assertGreater(final["reduction_percent"], 60.0)
        self.assertEqual(final["generated_token_ids"], [2064, 5686])
        self.assertTrue(final["hip_only"])
        self.assertFalse(final["fallback"])
        self.assertEqual(final["cleanup_failures"], 0)

    def test_numerical_class_and_mtp_fix_are_explicit(self):
        numerical = self.summary["numerical"]
        self.assertTrue(numerical["classification"].startswith("N1_"))
        self.assertEqual(numerical["selected_solution_global_split_u"], 1)
        self.assertFalse(numerical["selected_solution_uses_global_atomic_combine"])
        self.assertTrue(numerical["provider_repeats_deterministic"])
        for row in numerical["stress_cases"]:
            self.assertEqual(row["tiled_bound_violations"], 0)
            self.assertEqual(row["hipblas_bound_violations"], 0)
        mtp = self.summary["mtp_fix"]
        self.assertEqual(mtp["numerical_classification"], "N0_CORRECTNESS_RESTORATION")
        self.assertFalse(mtp["normal_prefill_compaction_changed"])

    def test_resources_and_regression_controls_close_out(self):
        resources = self.summary["resources"]
        self.assertEqual(resources["gfx1030_context_hipblas_handles_added"], 1)
        self.assertEqual(resources["gfx1030_context_hipblaslt_handles_added"], 0)
        self.assertEqual(resources["explicit_hipblas_workspace_bytes"], 0)
        self.assertEqual(resources["request_workspace_arena_bytes_before"], resources["request_workspace_arena_bytes_after"])
        verification = self.summary["verification"]
        self.assertTrue(verification["gfx1030_release_build"])
        self.assertTrue(verification["gfx1201_release_build"])
        self.assertTrue(verification["gfx942_compile_only"])
        self.assertTrue(verification["gfx1030_openai_lifecycle"])
        self.assertTrue(verification["wrong_target_load_rejected"])
        self.assertNotEqual(verification["wrong_target_exit_code"], 0)

        service = self.summary["service"]
        self.assertEqual(service["prompt_tokens"], 10001)
        self.assertEqual(service["non_stream_content"], service["sse_content"])
        self.assertTrue(service["sse_done"])
        self.assertTrue(service["usage_equal"])
        self.assertEqual(service["disconnect_outcome"], "cancelled")
        self.assertEqual(service["recovery_content"], "Hello")
        self.assertTrue(service["all_dispatches_hip"])
        self.assertEqual(service["fallbacks"], 0)
        self.assertEqual(service["shutdown_current_bytes"], 0)
        self.assertEqual(service["shutdown_request_state_bytes"], 0)
        self.assertEqual(service["shutdown_workspace_bytes"], 0)
        self.assertEqual(service["retryable_cleanup"], 0)
        self.assertEqual(service["durable_quarantine"], 0)


if __name__ == "__main__":
    unittest.main()
