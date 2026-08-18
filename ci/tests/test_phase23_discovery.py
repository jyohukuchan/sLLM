from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError  # noqa: E402
import run_phase23_discovery as phase23  # noqa: E402


class Phase23DiscoveryTests(unittest.TestCase):
    def test_checked_in_summary_matches_schema(self) -> None:
        schema = json.loads(
            (ROOT / "ci/schema/phase23-performance-discovery-summary-v1.schema.json").read_text()
        )
        summary = json.loads(
            (ROOT / "ci/matrix/phase23-performance-discovery-summary-v1.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(
                schema, format_checker=FormatChecker()
            ).iter_errors(summary),
            key=lambda error: list(error.absolute_path),
        )
        self.assertEqual(errors, [])
        opportunities = summary["opportunities"]
        self.assertEqual([item["rank"] for item in opportunities], list(range(1, len(opportunities) + 1)))
        opportunity_ids = {item["id"] for item in opportunities}
        self.assertTrue(set(summary["phase24_shortlist"]) <= opportunity_ids)
        self.assertEqual(
            summary["identity"]["runner_sha256"],
            phase23.sha256_file(ROOT / "ci/tools/run_phase23_discovery.py"),
        )

    def test_distribution_keeps_bounds_median_and_mad(self) -> None:
        self.assertEqual(
            phase23.distribution([9, 1, 5, 7]),
            {"count": 4, "min": 1, "median": 6.0, "max": 9, "mad": 2.0},
        )
        with self.assertRaises(ContractError):
            phase23.distribution([])

    def test_profiler_categories_preserve_semantic_priority(self) -> None:
        cases = {
            "matmul_bf16_fp32_tiled16_kernel": "prefill_matmul",
            "Cijk_Ailk_Bljk_SB_MT128x128x16": "hipblas_gemm",
            "matmul_bf16_decode_v4": "decode_or_recurrent_matvec",
            "linear_attention_gdn_decode": "linear_attention",
            "causal_attention_decode": "full_attention_and_kv",
            "argmax_bf16": "sampling",
            "rmsnorm_bf16": "normalization",
            "elementwise_add": "elementwise_and_embedding",
            "rocclr_copyBuffer": "runtime_copy_fill",
            "unknown_kernel": "other",
        }
        for kernel, expected in cases.items():
            with self.subTest(kernel=kernel):
                self.assertEqual(phase23.profiler_category(kernel), expected)

    def test_profile_aggregate_is_bounded_and_digest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "profile"
            output = root / "out"
            profile.mkdir()
            self._write_csv(
                profile / "case_kernel_stats.csv",
                ["Name", "Calls", "TotalDurationNs"],
                [
                    ["matmul_bf16_fp32_tiled16_kernel", "2", "800"],
                    ["linear_attention_gdn_decode", "4", "200"],
                ],
            )
            self._write_csv(
                profile / "case_hip_api_stats.csv",
                ["Name", "Calls", "TotalDurationNs"],
                [["hipLaunchKernel", "6", "50"]],
            )
            self._write_csv(
                profile / "case_memory_copy_stats.csv",
                ["Name", "Calls", "TotalDurationNs"],
                [["Host to Device", "1", "25"]],
            )
            self._write_csv(
                profile / "case_kernel_trace.csv",
                ["Name", "StartNs", "EndNs"],
                [["matmul_bf16_fp32_tiled16_kernel", "0", "800"]],
            )

            result = phase23.aggregate_profile(profile, "gfx1030", output)

            self.assertEqual(result["state"], "PASS")
            self.assertEqual(result["kernel"]["total_duration_ns"], 1000)
            self.assertEqual(result["kernel"]["categories"][0]["category"], "prefill_matmul")
            self.assertEqual(result["kernel"]["categories"][0]["device_time_share"], 0.8)
            encoded = json.loads((output / "profile-aggregate.json").read_text())
            self.assertEqual(encoded, result)
            self.assertEqual(
                (output / "profile-aggregate.json.sha256").read_text().strip(),
                phase23.sha256_file(output / "profile-aggregate.json"),
            )

    @staticmethod
    def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
