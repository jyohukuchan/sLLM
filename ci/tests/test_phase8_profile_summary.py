import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class Phase8ProfileSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(
            (ROOT / "ci/matrix/phase8-profile-summary-v1.json").read_text()
        )

    def test_targets_and_protocol_are_exact(self) -> None:
        self.assertEqual(self.document["schema_version"], "phase8-profile-summary-v1")
        self.assertEqual(self.document["protocol"]["warmups"], 3)
        self.assertEqual(self.document["protocol"]["measured"], 10)
        self.assertEqual(
            [target["target"] for target in self.document["targets"]],
            ["gfx1030", "gfx1201"],
        )

    def test_profiles_are_fallback_free_bounded_summaries(self) -> None:
        for target in self.document["targets"]:
            short = target["short_odd"]["phase8"]
            surrogate = target["surrogate_32x32"]
            self.assertEqual(short["submissions_per_request"], 7956)
            self.assertEqual(short["kernels_per_request"], 8364)
            self.assertEqual(surrogate["submissions_per_request"], 14976)
            self.assertEqual(surrogate["kernels_per_request"], 15744)
            self.assertFalse(short["fallback_used"])
            self.assertTrue(short["cleanup_terminal_zero"])
            self.assertFalse(surrogate["fallback_used"])
            self.assertTrue(surrogate["cleanup_terminal_zero"])
            self.assertEqual(len(short["raw_report_sha256"]), 64)
            self.assertEqual(len(surrogate["raw_report_sha256"]), 64)
            self.assertGreater(short["prefill_tokens_per_second_median"], 0.0)
            self.assertGreater(short["decode_tokens_per_second_median"], 0.0)

    def test_provider_selection_is_target_specific(self) -> None:
        by_target = {row["target"]: row for row in self.document["targets"]}
        self.assertEqual(
            by_target["gfx1030"]["decode_matmul_1x2560x9216"]["selected_kernel_id"],
            3,
        )
        self.assertEqual(
            by_target["gfx1201"]["decode_matmul_1x2560x9216"]["selected_kernel_id"],
            4,
        )
        self.assertEqual(self.document["protocol"]["matmul_workspace_bytes"], 0)

    def test_a6_integration_records_are_identity_bound_and_fallback_free(self) -> None:
        integration = self.document["integration"]
        self.assertEqual(
            [row["target"] for row in integration["builds"]],
            ["gfx1030", "gfx1201"],
        )
        self.assertEqual(len(integration["model_spot_checks"]), 4)
        self.assertEqual(len(integration["llama_cpp_short_odd"]), 2)
        for section in ("builds", "model_spot_checks", "llama_cpp_short_odd"):
            for row in integration[section]:
                for key, value in row.items():
                    if key.endswith("sha256"):
                        self.assertEqual(len(value), 64)
        for row in integration["openai_service"]:
            self.assertEqual(row["result"], "PASS")
            self.assertFalse(row["fallback_used"])
            self.assertTrue(row["cleanup_terminal_zero"])
            self.assertEqual(row["capacity_boundaries"], [1023, 1024, 1025])

        o2 = integration["o2"]
        self.assertTrue(o2["health"]["ecc_zero"])
        self.assertTrue(o2["health"]["processes_terminal_zero"])
        self.assertTrue(o2["health"]["vram_returned_to_pre_run_baseline"])
        expected_cases = [
            "minimum",
            "short-odd",
            "boundary-255",
            "boundary-256",
            "boundary-257",
            "prefill-long",
            "decode-long",
        ]
        for target in o2["targets"]:
            self.assertEqual([case["case"] for case in target["cases"]], expected_cases)
            for case in target["cases"]:
                self.assertFalse(case["fallback_used"])
                self.assertTrue(case["cleanup_terminal_zero"])
                self.assertEqual(len(case["raw_sha256"]), 64)
                self.assertGreater(case["prefill_tps"], 0.0)
                self.assertGreater(case["e2e_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
