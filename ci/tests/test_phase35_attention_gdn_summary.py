import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "ci/matrix/phase35-attention-gdn-summary-v1.json"
SCHEMA = ROOT / "ci/schema/phase35-attention-gdn-summary-v1.schema.json"


class Phase35SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(SUMMARY.read_text())
        cls.schema = json.loads(SCHEMA.read_text())

    def test_schema(self):
        jsonschema.Draft202012Validator(self.schema).validate(self.summary)

    def test_shared_routes_are_bounded_and_fail_closed(self):
        self.assertEqual(self.summary["state"], "COMPLETE")
        self.assertEqual(self.summary["decisions"]["attention"], "ADOPTED_SHARED_SHAPE_SCOPED")
        self.assertEqual(self.summary["decisions"]["gdn"], "ADOPTED_SHARED_SHAPE_SCOPED")
        self.assertEqual(self.summary["routing"]["targets"], ["gfx1030", "gfx1201"])
        self.assertEqual(self.summary["routing"]["attention"]["minimum_query_count"], 128)
        self.assertEqual(self.summary["routing"]["gdn"]["minimum_token_count"], 128)
        self.assertEqual(self.summary["routing"]["gdn"]["recurrent_grid_size"], 1024)
        self.assertFalse(self.summary["routing"]["runtime_failure_fallback"])

    def test_numerical_evidence_covers_targets_encodings_and_boundaries(self):
        numerical = self.summary["numerical"]
        self.assertTrue(numerical["attention_classification"].startswith("N1_"))
        self.assertTrue(numerical["gdn_classification"].startswith("N1_"))
        self.assertEqual(numerical["generated_token_ids"], [2064, 5686])
        self.assertEqual(numerical["attention_g1"]["total_cases_passed"], 232)
        self.assertEqual(numerical["gdn_g2"]["token_counts"][-3:], [127, 128, 129])
        self.assertTrue(numerical["gdn_g2"]["state_publication_exact"])
        self.assertEqual(numerical["fallbacks"], 0)
        self.assertEqual(numerical["cleanup_failures"], 0)

    def test_both_tracks_and_combined_result_improve_both_targets(self):
        performance = self.summary["performance"]
        for track in ["attention_only", "gdn_only", "combined_final_source"]:
            for target in ["gfx1030", "gfx1201"]:
                row = performance[track][target]
                self.assertLess(row["candidate_ns"], row["baseline_ns"])
                self.assertGreater(row["reduction_percent"], 0.0)
        self.assertGreater(performance["gfx1030_profile_seconds"]["full_attention_reduction_percent"], 60.0)
        self.assertGreater(performance["gfx1030_profile_seconds"]["gdn_reduction_percent"], 90.0)

    def test_resources_builds_and_provenance_are_explicit(self):
        resources = self.summary["resources"]
        self.assertEqual(resources["attention_global_scratch_bytes_added"], 0)
        self.assertEqual(resources["workspace_arena_bytes_before"], resources["workspace_arena_bytes_after"])
        self.assertFalse(resources["state_layout_migration"])
        verification = self.summary["verification"]
        self.assertTrue(verification["gfx1030_release_build"])
        self.assertTrue(verification["gfx1201_release_build"])
        self.assertTrue(verification["gfx942_compile_only"])
        self.assertTrue(verification["wrong_target_load_rejected"])
        self.assertTrue(verification["final_full_model_hip_only"])
        self.assertFalse(verification["final_full_model_fallback"])
        self.assertTrue(verification["final_full_model_cleanup_zero"])
        self.assertEqual(self.summary["provenance"]["gdn_notice"], "llama-cpp-phase35-gdn-column-state-001")

        service = self.summary["service"]
        self.assertEqual(service["prompt_tokens"], 10001)
        self.assertEqual(service["non_stream_content"], service["sse_content"])
        self.assertTrue(service["sse_done"])
        self.assertTrue(service["usage_equal"])
        self.assertTrue(service["all_dispatches_hip"])
        self.assertEqual(service["fallbacks"], 0)
        self.assertEqual(service["shutdown_current_bytes"], 0)
        self.assertEqual(service["shutdown_request_state_bytes"], 0)
        self.assertEqual(service["shutdown_workspace_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
