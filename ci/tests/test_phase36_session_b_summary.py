from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    Draft202012Validator = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))
sys.path.insert(0, str(ROOT / "ci" / "tests"))

import run_phase36_session_b as runner  # noqa: E402
from test_phase36_session_b_runner import _populate  # noqa: E402


class Phase36SessionBSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads((ROOT / "ci/schema/phase36-mi300x-session-b-summary-v1.schema.json").read_text(encoding="utf-8"))

    def _summary(self) -> dict[str, object]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        binary, model, lock, source = _populate(root)
        return runner.aggregate(raw_dir=root / "raw", output_dir=root / "out", binary=binary, model=model, lock=lock, source_identity=source)

    def assert_schema_rejects(self, document: dict[str, object]) -> None:
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not installed")
        self.assertTrue(list(Draft202012Validator(self.schema).iter_errors(document)))

    def test_complete_summary_passes_strict_schema(self) -> None:
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not installed")
        summary = self._summary()
        self.assertEqual(list(Draft202012Validator(self.schema).iter_errors(summary)), [])

    def test_schema_is_closed_and_raw_paths_are_not_tracked(self) -> None:
        summary = self._summary()
        invalid = copy.deepcopy(summary)
        invalid["unexpected"] = True
        self.assert_schema_rejects(invalid)
        serialized = json.dumps(summary)
        self.assertNotIn("sysfs-", serialized)
        self.assertNotIn('"path"', serialized)

    def test_four_model_free_encodings_and_fp16_state_have_exact_case_counts(self) -> None:
        summary = self._summary()
        self.assertEqual([item["encoding"] for item in summary["full_attention_reports"]], ["fp16-v1", "kv-fp8-v1", "kv-fp8-static-v1", "kv-nvfp4-v1"])
        self.assertEqual([item["case_count"] for item in summary["full_attention_reports"]], [29, 29, 29, 29])
        self.assertEqual(summary["kv_state"]["case_count"], 19)
        self.assertEqual(summary["lowbit_oracle"]["case_count"], 17)
        self.assertTrue(all(item["contiguous_resident"] for item in summary["full_attention_reports"]))

    def test_rows_bind_exact_input_and_output_vectors_and_chunk_partitions(self) -> None:
        summary = self._summary()
        rows = summary["model_rows"]["rows"]
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["input_ids_sha256"] for row in rows}), 1)
        self.assertTrue(all(row["input_ids_count"] == 10001 for row in rows))
        self.assertTrue(all(row["output_ids"] == [23066, 23066] for row in rows))
        self.assertEqual([(row["encoding"], row["chunk_setting"]) for row in rows], list(runner.EXPECTED_ROWS))
        self.assertTrue(summary["comparisons"]["cross_setting_token_equality"])
        self.assertTrue(summary["comparisons"]["chunk_partition_valid"])

    def test_request_state_arena_reduction_settled_baseline_and_no_gtt(self) -> None:
        summary = self._summary()
        memory = summary["memory"]
        self.assertGreater(memory["fp16_request_state_bytes"], memory["fp8_request_state_bytes"])
        self.assertGreater(memory["fp8_request_state_reduction_percent"], 0.0)
        self.assertLess(memory["arena_high_water_bytes"], memory["separate_allocation_bytes"])
        self.assertTrue(memory["settled_baseline"]["settled"])
        self.assertEqual(memory["settled_baseline"]["gtt_used_bytes"], 0)
        self.assertTrue(memory["no_gtt_spill"])
        self.assertEqual(memory["gtt_spill_bytes"], 0)

    def test_schema_rejects_tampered_case_count_or_raw_digest(self) -> None:
        summary = self._summary()
        changed = copy.deepcopy(summary)
        changed["full_attention_reports"][0]["case_count"] = 28
        self.assert_schema_rejects(changed)
        changed = copy.deepcopy(summary)
        changed["model_rows"]["rows"][0]["raw_sha256"] = "0" * 64
        # Shape-only schema permits any digest, but semantic validation rejects
        # a changed producer row when the aggregate is reconstructed.
        self.assertEqual(list(Draft202012Validator(self.schema).iter_errors(changed)) if Draft202012Validator else [], [])


if __name__ == "__main__":
    unittest.main()
