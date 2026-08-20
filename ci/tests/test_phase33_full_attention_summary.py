import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "ci/matrix/phase33-full-attention-summary-v1.json"
SCHEMA = ROOT / "ci/schema/phase33-full-attention-summary-v1.schema.json"


class Phase33SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(SUMMARY.read_text())
        cls.schema = json.loads(SCHEMA.read_text())

    def test_schema(self):
        jsonschema.Draft202012Validator(self.schema).validate(self.summary)

    def test_correctness_and_target_scope_are_fail_closed(self):
        correctness = self.summary["correctness"]
        self.assertEqual(correctness["cases_passed"], correctness["cases_total"])
        self.assertEqual(correctness["cases_total"], 2 * 4 * 29)
        self.assertEqual(correctness["fallbacks"], 0)
        self.assertEqual(correctness["cleanup_failures"], 0)
        self.assertEqual(self.summary["routing"]["targets"], ["gfx1030", "gfx1201"])
        self.assertFalse(self.summary["routing"]["runtime_failure_fallback"])
        verification = self.summary["verification"]
        self.assertTrue(verification["gfx942_compile_only"])
        self.assertTrue(verification["wrong_target_load_rejected"])
        self.assertNotEqual(verification["wrong_target_exit_code"], 0)

    def test_c2_is_adopted_by_contextual_judgment_and_every_scoped_row_improves(self):
        decisions = self.summary["decisions"]
        self.assertEqual(decisions["c2_prefill_gqa4"], "ADOPTED")
        self.assertEqual(decisions["performance_authority"], "assigned-ai-contextual-judgment")
        self.assertFalse(decisions["fixed_threshold"])
        for target in ["gfx1201", "gfx1030"]:
            lower, upper = self.summary["operator"]["c2"][f"{target}_reduction_percent_m64_to_m257"]
            self.assertGreater(lower, 0.0)
            self.assertGreaterEqual(upper, lower)

    def test_c1_is_explicitly_adopted_as_n2_and_finally_revalidated(self):
        self.assertEqual(self.summary["state"], "COMPLETE")
        self.assertEqual(
            self.summary["decisions"]["c1_decode_wave_split"],
            "ADOPTED",
        )
        self.assertTrue(self.summary["correctness"]["c1_numerical_class"].startswith("N2_"))
        self.assertFalse(self.summary["correctness"]["final_identity_revalidation_pending"])
        self.assertTrue(self.summary["verification"]["final_release_rerun_complete"])
        final_smoke = self.summary["full_model"]["final_identity_smoke"]
        self.assertEqual(final_smoke["gfx1201_generated_token_id"], 1228)
        self.assertEqual(final_smoke["gfx1030_generated_token_id"], 1228)
        self.assertTrue(final_smoke["hip_only"])
        self.assertFalse(final_smoke["fallback"])
        self.assertEqual(final_smoke["cleanup_failures"], 0)
        for target_rows in self.summary["operator"]["c1"].values():
            for row in target_rows:
                self.assertGreater(row["reduction_percent"], 0.0)
                self.assertLess(row["candidate_ns"], row["baseline_ns"])

    def test_matrix_candidate_and_service_closeout_are_explicit(self):
        matrix = self.summary["matrix_candidate"]
        self.assertEqual(matrix["actual_matrix_instruction_occurrences"], 0)
        self.assertFalse(matrix["external_code_reused"])
        self.assertEqual(self.summary["decisions"]["c3_gfx1201_matrix"], "REJECTED_TILE_SHAPE_SCOPE_MISMATCH")
        service = self.summary["service"]
        self.assertTrue(service["sse_done"])
        self.assertEqual(service["disconnect_outcome"], "cancelled")
        self.assertEqual(service["shutdown_current_bytes"], 0)
        self.assertEqual(service["shutdown_request_state_bytes"], 0)
        self.assertEqual(service["shutdown_workspace_bytes"], 0)
        self.assertEqual(service["retryable_cleanup"], 0)
        self.assertEqual(service["durable_quarantine"], 0)


if __name__ == "__main__":
    unittest.main()
