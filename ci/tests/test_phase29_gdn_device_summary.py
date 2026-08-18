import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "phase29_gdn_device", ROOT / "ci/tools/phase29_gdn_device.py"
)
assert SPEC and SPEC.loader
PHASE29 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE29)


def kernel_row(dispatch: int, name: str, duration: int = 10) -> dict[str, str]:
    return {
        "Dispatch_Id": str(dispatch),
        "Kernel_Name": name,
        "Start_Timestamp": str(dispatch * 100),
        "End_Timestamp": str(dispatch * 100 + duration),
        "Workgroup_Size_X": "128",
        "Grid_Size_X": "4096",
        "LDS_Block_Size": "2096",
        "Scratch_Size": "0",
        "VGPR_Count": "32",
        "SGPR_Count": "48",
    }


class Phase29GdnDeviceTests(unittest.TestCase):
    def test_summary_schema_and_adoption_arithmetic(self) -> None:
        schema = json.loads((ROOT / "ci/schema/phase29-gdn-device-summary-v1.schema.json").read_text())
        summary = json.loads((ROOT / "ci/matrix/phase29-gdn-device-summary-v1.json").read_text())
        Draft202012Validator.check_schema(schema)
        errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(summary))
        self.assertEqual(errors, [])
        self.assertFalse(summary["metric_contract"]["full_model_is_gate"])
        self.assertTrue(all(row["non_regressing"] for row in summary["patterns"]))
        self.assertTrue(any(row["improvement_percent"] >= 5 for row in summary["patterns"]))
        self.assertEqual(summary["decision"], "ADOPTED_ANALYTIC_ERROR_REDUCTION")
        self.assertEqual(summary["numerical_policy"]["classification"], "N1")
        self.assertEqual(summary["numerical_policy"]["analytic_error_direction"], "reduced")
        self.assertFalse(summary["numerical_policy"]["token_exactness_is_gate"])
        self.assertFalse(summary["numerical_policy"]["high_precision_provider_required"])
        self.assertTrue(summary["correctness"]["short16_token_records_equal"])
        self.assertFalse(summary["correctness"]["long128_token_records_equal"])
        self.assertFalse(summary["implementation"]["candidate_removed"])
        self.assertTrue(summary["implementation"]["shared_path"])
        self.assertEqual(
            {(row["target"], row["pattern"]) for row in summary["patterns"]},
            {(target, pattern) for target in ("gfx1030", "gfx1201") for pattern in ("B0", "B1", "B2")},
        )
        for row in summary["patterns"]:
            expected = (1 - row["candidate_process_p50_ns"] / row["baseline_process_p50_ns"]) * 100
            self.assertAlmostEqual(row["improvement_percent"], expected)

    def test_step_extraction_includes_split_gdn_family(self) -> None:
        rows = []
        dispatch = 0
        for _request in range(14):
            for token in range(16):
                dispatch += 1
                rows.append(kernel_row(dispatch, "sllm_linear_attention_gdn_prepare_v1", 3))
                dispatch += 1
                rows.append(kernel_row(dispatch, "sllm_linear_attention_gdn_core_v1", 5))
                dispatch += 1
                rows.append(kernel_row(dispatch, "sllm_linear_attention_gdn_finalize_v1", 7))
                dispatch += 1
                rows.append(kernel_row(dispatch, "sllm_argmax_bf16_f32_v1", 2))
        steps = PHASE29.extract_decode_steps(rows)
        self.assertEqual(len(steps), 210)
        self.assertEqual({step["device_ns"] for step in steps}, {15})
        self.assertEqual({step["calls"] for step in steps}, {3})

    def test_missing_argmax_or_gdn_fails_closed(self) -> None:
        with self.assertRaises(PHASE29.Phase29Error):
            PHASE29.extract_decode_steps([kernel_row(1, "unrelated")])
        rows = [kernel_row(index + 1, "sllm_argmax_bf16_f32_v1") for index in range(224)]
        with self.assertRaises(PHASE29.Phase29Error):
            PHASE29.extract_decode_steps(rows)

    def test_run_report_checks_protocol_and_tokens(self) -> None:
        rows = []
        dispatch = 0
        for _request in range(14):
            for _token in range(16):
                dispatch += 1
                rows.append(kernel_row(dispatch, "sllm_linear_attention_recurrent_gated_norm_v1", 10))
                dispatch += 1
                rows.append(kernel_row(dispatch, "sllm_argmax_bf16_f32_v1", 2))
        token_record = {"input_token_ids": [1], "generated_token_ids": [2], "visible_token_ids": [2], "decode_input_token_ids": []}
        raw = {
            "state": "PASS",
            "config": {"input_token_count": 17, "max_new_tokens": 16, "warmups": 3, "measured": 10},
            "audit": {"target": "gfx1030", "selected_backend": "hip", "all_dispatches_hip": True, "fallback_used": False},
            "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "all_requests_dropped": True},
            "measured": {"samples": [{"tokens": token_record} for _ in range(10)]},
        }
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.csv"
            raw_path = Path(directory) / "raw.json"
            with trace_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            raw_path.write_text(json.dumps(raw))
            report = PHASE29.build_run_report(
                trace_path, raw_path, target="gfx1030", pattern="B0", variant="candidate", process_index=1
            )
        self.assertEqual(report["committed_decode_steps"], 210)
        self.assertEqual(report["gdn_device_p50_ns"], 10)
        self.assertEqual(report["gdn_calls_per_step"], 1)


if __name__ == "__main__":
    unittest.main()
