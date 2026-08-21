from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))

import phase36_session_d_profile as profile  # noqa: E402


def _execution(*, include_input_ids: bool = True) -> dict[str, object]:
    document: dict[str, object] = {
        "state": "PASS",
        "pass": True,
        "target": "gfx942",
        "selected_backend": "hip",
        "all_dispatches_hip": True,
        "fallback_used": False,
        "cpu_fallback_used": False,
        "partial_offload": False,
        "usage": {"prompt_tokens": profile.INPUT_TOKENS, "completion_tokens": profile.OUTPUT_TOKENS},
        "generated_token_ids": list(profile.EXPECTED_OUTPUT_IDS),
        "execution": {"mtp_draft_width_requested": 0, "host_wall_ns": 1000},
        "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "terminal_zero": True},
    }
    if include_input_ids:
        document["input_token_ids"] = [profile.EXPECTED_INPUT_ID] * profile.INPUT_TOKENS
    else:
        document["input_ids_sha256"] = hashlib.sha256(
            profile.canonical_bytes([profile.EXPECTED_INPUT_ID] * profile.INPUT_TOKENS)
        ).hexdigest()
    return document


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _populate(root: Path) -> tuple[Path, Path]:
    profile_dir = root / "rocprof"
    profile_dir.mkdir()
    _write_csv(
        profile_dir / "case_kernel_stats.csv",
        ["Name", "Calls", "TotalDurationNs"],
        [
            ["Cijk_Ailk_Bljk_SB_MT128x128x16", 1, 800],
            ["causal_attention_prefill_gqa4_qtile4_kernel", 1, 300],
            ["sllm_linear_attention_recurrent_column_state_v2", 1, 200],
            ["runtime_unknown_kernel", 1, 100],
        ],
    )
    _write_csv(
        profile_dir / "case_kernel_trace.csv",
        ["Kernel_Name", "Dispatch_Id", "Start_Timestamp", "End_Timestamp"],
        [
            ["Cijk_Ailk_Bljk_SB_MT128x128x16", 1, 100, 300],
            ["causal_attention_prefill_gqa4_qtile4_kernel", 2, 200, 250],
            ["sllm_linear_attention_recurrent_column_state_v2", 3, 400, 550],
            ["runtime_unknown_kernel", 4, 450, 500],
        ],
    )
    _write_csv(
        profile_dir / "case_hip_api_stats.csv",
        ["Name", "Calls", "TotalDurationNs"],
        [["hipLaunchKernel", 6, 50]],
    )
    _write_csv(
        profile_dir / "case_memory_copy_stats.csv",
        ["Name", "Calls", "TotalDurationNs"],
        [["Host to Device", 1, 25]],
    )
    execution_path = root / "execution.json"
    execution_path.write_text(json.dumps(_execution()), encoding="utf-8")
    return profile_dir, execution_path


class Phase36SessionDProfileTests(unittest.TestCase):
    def test_classification_is_single_bucket_and_unknown_is_other(self) -> None:
        self.assertEqual(profile.classify_kernel("sllm_matmul_bf16_fp32_decode_v4"), "projection")
        self.assertEqual(profile.classify_kernel("causal_attention_prefill_gqa4_qtile4_kernel"), "full_attention")
        self.assertEqual(profile.classify_kernel("sllm_linear_attention_column_postprocess_v2"), "gdn")
        self.assertEqual(profile.classify_kernel("unrecognized_kernel"), "mtp_or_other")
        with self.assertRaises(profile.SessionDProfileError):
            profile.classify_kernel("sllm_matmul_linear_attention_ambiguous")

    def test_non_aligned_trace_intervals_use_union_and_external_wall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_dir, execution = _populate(Path(temporary))
            result = profile.aggregate_profile(profile_dir, execution, host_wall_ns=1000)
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(result["kernel"]["total_duration_ns"], 1400)
        self.assertEqual(result["kernel"]["trace_interval_union_ns"], 350)
        self.assertEqual(result["kernel_external"]["external_ns"], 650)
        self.assertEqual(result["kernel"]["categories"][0]["total_duration_ns"], 800)
        self.assertEqual(result["kernel"]["categories"][3]["kernel_names"], ["runtime_unknown_kernel"])
        self.assertAlmostEqual(result["kernel"]["category_share_sum"], 1.0)

    def test_stats_calls_and_trace_dispatches_must_close_per_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_dir, execution = _populate(Path(temporary))
            stats = profile_dir / "case_kernel_stats.csv"
            text = stats.read_text(encoding="utf-8").replace(
                "Cijk_Ailk_Bljk_SB_MT128x128x16,1,800",
                "Cijk_Ailk_Bljk_SB_MT128x128x16,2,800",
            )
            stats.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(profile.SessionDProfileError, "do not close"):
                profile.aggregate_profile(profile_dir, execution)

    def test_input_id_digest_variant_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_dir, execution = _populate(root)
            document = json.loads(execution.read_text())
            document.pop("input_token_ids", None)
            document["input_token_count"] = profile.INPUT_TOKENS
            document["input_ids_sha256"] = hashlib.sha256(
                profile.canonical_bytes([profile.EXPECTED_INPUT_ID] * profile.INPUT_TOKENS)
            ).hexdigest()
            execution.write_text(json.dumps(document), encoding="utf-8")
            result = profile.aggregate_profile(profile_dir, execution)
        self.assertEqual(result["execution"]["input_ids_mode"], "digest")
        self.assertEqual(result["kernel_external"]["state"], "available")

    def test_kernel_external_is_explicitly_unavailable_without_host_wall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_dir, execution = _populate(root)
            document = json.loads(execution.read_text())
            document["execution"].pop("host_wall_ns")
            execution.write_text(json.dumps(document), encoding="utf-8")
            result = profile.aggregate_profile(profile_dir, execution)
        self.assertEqual(result["kernel_external"]["state"], "unavailable")
        self.assertIn("not supplied", result["kernel_external"]["reason"])

    def test_duplicate_dispatch_and_duplicate_stats_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_dir, execution = _populate(root)
            path = profile_dir / "case_kernel_trace.csv"
            with path.open("a", encoding="utf-8") as stream:
                stream.write("runtime_unknown_kernel,1,600,601\n")
            with self.assertRaisesRegex(profile.SessionDProfileError, "duplicate Dispatch_Id"):
                profile.aggregate_profile(profile_dir, execution)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_dir, execution = _populate(root)
            path = profile_dir / "case_kernel_stats.csv"
            with path.open("a", encoding="utf-8") as stream:
                stream.write("runtime_unknown_kernel,1,1\n")
            with self.assertRaisesRegex(profile.SessionDProfileError, "duplicate kernel Name"):
                profile.aggregate_profile(profile_dir, execution)

    def test_non_positive_duration_is_rejected_at_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_dir, execution = _populate(root)
            path = profile_dir / "case_kernel_trace.csv"
            text = path.read_text(encoding="utf-8").replace(",100,300", ",300,300")
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(profile.SessionDProfileError, "duration must be positive"):
                profile.aggregate_profile(profile_dir, execution)

    def test_execution_contract_rejects_target_fallback_output_or_mtp_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_dir, execution = _populate(root)
            document = json.loads(execution.read_text())
            document["fallback_used"] = True
            execution.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(profile.SessionDProfileError, "fallback"):
                profile.aggregate_profile(profile_dir, execution)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_dir, execution = _populate(root)
            document = json.loads(execution.read_text())
            document["execution"]["mtp_draft_width_requested"] = 1
            execution.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(profile.SessionDProfileError, "MTP"):
                profile.aggregate_profile(profile_dir, execution)

    def test_raw_digests_and_optional_output_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_dir, execution = _populate(root)
            output = root / "summary.json"
            result = profile.aggregate_profile(profile_dir, execution, output_path=output)
            encoded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(encoded, result)
            self.assertEqual(
                result["raw_sha256"]["kernel_stats"]["sha256"],
                profile.sha256_file(profile_dir / "case_kernel_stats.csv"),
            )
            self.assertEqual(result["raw_sha256"]["execution_json"]["path"], "execution.json")


if __name__ == "__main__":
    unittest.main()
