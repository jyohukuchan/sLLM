import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "ci/matrix/phase30-rdna4-attention-kv-summary-v1.json"
SCHEMA = ROOT / "ci/schema/phase30-rdna4-attention-kv-summary-v1.schema.json"


class Phase30SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(SUMMARY.read_text())
        cls.schema = json.loads(SCHEMA.read_text())

    def test_schema(self):
        jsonschema.Draft202012Validator(self.schema).validate(self.summary)

    def test_adoption_is_target_scoped_and_non_api(self):
        routing = self.summary["routing"]
        self.assertEqual(routing["target"], "gfx1201")
        self.assertEqual(routing["control_target"], "gfx1030")
        self.assertEqual(routing["control_provider"], "baseline")
        self.assertFalse(routing["public_api_changed"])
        self.assertFalse(routing["kv_format_changed"])

    def test_adopted_mechanisms_have_direct_proof(self):
        candidates = self.summary["candidates"]
        probe = candidates["native_fp8_read"]["all_code_probe"]
        self.assertEqual(probe, {"codes": 256, "nan_codes": 2, "mismatches": 0, "fallback": False})
        self.assertEqual(candidates["wave_provider"]["decision"], "ADOPTED")
        self.assertLess(candidates["wave_provider"]["candidate_barriers_per_key"], candidates["wave_provider"]["baseline_barriers_per_key"])
        self.assertEqual(candidates["native_fp8_append"]["decision"], "REJECTED")
        self.assertFalse(candidates["prefill_matrix_provider"]["matrix_isa_claimed"])

    def test_full_model_adoption_pattern_exceeds_threshold(self):
        pattern = next(row for row in self.summary["full_model"] if row["classification"] == "ADOPTION_PATTERN")
        self.assertGreaterEqual(pattern["processes"], 3)
        self.assertGreaterEqual(pattern["ttft_improvement_percent"], 5.0)
        self.assertGreaterEqual(pattern["decode_tps_improvement_percent"], 5.0)

    def test_correctness_has_both_targets_and_encodings(self):
        rows = self.summary["correctness"]
        self.assertEqual({(row["target"], row["encoding"]) for row in rows}, {("gfx1030", "fp16"), ("gfx1030", "fp8"), ("gfx1201", "fp16"), ("gfx1201", "fp8")})
        for row in rows:
            self.assertEqual(row["passed"], row["cases"])
            self.assertEqual(row["fallbacks"], 0)
            self.assertEqual(row["cleanup_failures"], 0)


if __name__ == "__main__":
    unittest.main()
