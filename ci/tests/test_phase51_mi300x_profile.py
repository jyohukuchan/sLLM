from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))

import phase51_mi300x_profile as profile  # noqa: E402


def _sample(sample_index: int) -> dict[str, object]:
    return {
        "tokens": {
            "input_token_ids": [profile.EXPECTED_INPUT_ID] * profile.INPUT_TOKENS,
            "generated_token_ids": list(profile.EXPECTED_OUTPUT_IDS),
            "visible_token_ids": list(profile.EXPECTED_OUTPUT_IDS),
            "decode_input_token_ids": [profile.EXPECTED_INPUT_ID],
        },
        "audit": {
            "selected_backend": "hip",
            "target": "gfx942",
            "model_fingerprint": profile.MODEL_FINGERPRINT,
            "fallback_used": False,
            "all_dispatches_hip": True,
        },
        "stop": {
            "version": 1,
            "reason_version": 1,
            "kind": "max_new_tokens",
            "token_id": None,
        },
        "cleanup": {
            "sample_index": sample_index,
            "request_dropped": True,
            "allocator_cleanup_validated": True,
            "retryable_cleanup": 0,
            "durable_quarantine": 0,
        },
    }


def _execution() -> dict[str, object]:
    control = _sample(0)
    return {
        "benchmark_schema_version": "engine-performance-direct-v2",
        "state": "PASS",
        "lane": "direct",
        "identities": {
            "engine": "sllm",
            "backend": "hip",
            "target": "gfx942",
            "model": {"lock_fingerprint": profile.MODEL_FINGERPRINT},
            "binding": {"model_fingerprint": profile.MODEL_FINGERPRINT},
        },
        "config": {
            "input_token_ids": [profile.EXPECTED_INPUT_ID] * profile.INPUT_TOKENS,
            "input_token_count": profile.INPUT_TOKENS,
            "max_new_tokens": profile.OUTPUT_TOKENS,
            "context_length": profile.CONTEXT_LENGTH,
            "warmups": profile.WARMUPS,
            "measured": profile.MEASURED,
        },
        "row": {
            "row_id": "phase51-mi300x-sllm-long-10001",
            "case_id": "long-10001",
            "input_token_ids": [profile.EXPECTED_INPUT_ID] * profile.INPUT_TOKENS,
            "input_token_count": profile.INPUT_TOKENS,
            "requested_output_tokens": profile.OUTPUT_TOKENS,
        },
        "audit": {
            "selected_backend": "hip",
            "target": "gfx942",
            "model_fingerprint": profile.MODEL_FINGERPRINT,
            "fallback_used": False,
            "all_dispatches_hip": True,
        },
        "correctness_control": control,
        "warmups": {"count": 1, "samples": [copy.deepcopy(control)]},
        "measured": {"count": 3, "samples": [_sample(index) for index in range(3)]},
        "cleanup": {
            "all_requests_dropped": True,
            "retryable_cleanup": 0,
            "durable_quarantine": 0,
        },
        "session_cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
    }


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
        [["hipLaunchKernel", 4, 50]],
    )
    _write_csv(
        profile_dir / "case_memory_copy_stats.csv",
        ["Name", "Calls", "TotalDurationNs"],
        [["Host to Device", 1, 25]],
    )
    execution_path = root / "execution.json"
    execution_path.write_text(json.dumps(_execution()), encoding="utf-8")
    return profile_dir, execution_path


class Phase51MI300XProfileTests(unittest.TestCase):
    def test_current_direct_output_without_mtp_passes_without_inventing_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_dir, execution = _populate(Path(temporary))
            result = profile.aggregate_profile(profile_dir, execution, host_wall_ns=1000)
        self.assertEqual(result["schema_version"], "phase51-mi300x-profile-v1")
        self.assertEqual(result["state"], "PASS")
        self.assertNotIn("mtp_draft_width", result["execution"])
        self.assertEqual(result["execution"]["mtp"]["evidence_state"], "unavailable")
        self.assertEqual(result["execution"]["mtp"]["source"], "not-emitted")
        self.assertIs(result["execution"]["mtp"]["validation_claimed"], False)
        self.assertEqual(result["execution"]["model_fingerprint"], profile.MODEL_FINGERPRINT)

    def test_categories_interval_union_and_external_wall_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_dir, execution = _populate(Path(temporary))
            result = profile.aggregate_profile(profile_dir, execution, host_wall_ns=1000)
        self.assertEqual(result["kernel"]["total_duration_ns"], 1400)
        self.assertEqual(result["kernel"]["trace_interval_union_ns"], 350)
        self.assertEqual(result["kernel_external"]["external_ns"], 650)
        self.assertEqual(result["kernel"]["categories"][0]["total_duration_ns"], 800)
        self.assertEqual(result["kernel"]["categories"][3]["kernel_names"], ["runtime_unknown_kernel"])
        self.assertAlmostEqual(result["kernel"]["category_share_sum"], 1.0)

    def test_classification_is_single_bucket_and_unknown_is_other(self) -> None:
        self.assertEqual(profile.classify_kernel("sllm_matmul_bf16_fp32_decode_v4"), "projection")
        self.assertEqual(profile.classify_kernel("causal_attention_prefill_gqa4_qtile4_kernel"), "full_attention")
        self.assertEqual(profile.classify_kernel("sllm_linear_attention_column_postprocess_v2"), "gdn")
        self.assertEqual(profile.classify_kernel("unrecognized_kernel"), "mtp_or_other")
        with self.assertRaises(profile.Phase51MI300XProfileError):
            profile.classify_kernel("sllm_matmul_linear_attention_ambiguous")

    def test_direct_contract_drift_fails_closed(self) -> None:
        mutations = (
            ("target", lambda value: value["identities"].__setitem__("target", "gfx941")),
            ("backend", lambda value: value["identities"].__setitem__("backend", "cpu")),
            ("fallback", lambda value: value["audit"].__setitem__("fallback_used", True)),
            ("input", lambda value: value["config"]["input_token_ids"].__setitem__(-1, 1)),
            ("output", lambda value: value["measured"]["samples"][0]["tokens"].__setitem__("generated_token_ids", [1, 2])),
            ("cleanup", lambda value: value["session_cleanup"].__setitem__("retryable_cleanup", 1)),
            ("fingerprint", lambda value: value["identities"]["model"].__setitem__("lock_fingerprint", "sha256:" + "0" * 64)),
            ("schema", lambda value: value.__setitem__("benchmark_schema_version", "engine-performance-direct-v1")),
            ("case", lambda value: value["row"].__setitem__("case_id", "short-odd")),
            ("context", lambda value: value["config"].__setitem__("context_length", 10002)),
            ("config-warmups", lambda value: value["config"].__setitem__("warmups", 2)),
            ("config-measured", lambda value: value["config"].__setitem__("measured", 2)),
            ("warmup-count", lambda value: value["warmups"].__setitem__("count", 2)),
            ("warmup-list", lambda value: value["warmups"]["samples"].append(_sample(1))),
            ("measured-count", lambda value: value["measured"].__setitem__("count", 2)),
            ("measured-list", lambda value: value["measured"]["samples"].pop()),
            ("stop", lambda value: value["measured"]["samples"][1]["stop"].__setitem__("kind", "stop_token")),
            ("output-count", lambda value: value["measured"]["samples"][2]["tokens"].__setitem__("generated_token_ids", [profile.EXPECTED_INPUT_ID])),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                profile_dir, execution = _populate(Path(temporary))
                document = json.loads(execution.read_text(encoding="utf-8"))
                mutate(document)
                execution.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(profile.Phase51MI300XProfileError):
                    profile.aggregate_profile(profile_dir, execution)

    def test_missing_required_execution_evidence_fails_closed(self) -> None:
        removals = (
            ("target", lambda value: value["identities"].pop("target")),
            ("backend", lambda value: value["identities"].pop("backend")),
            ("fallback", lambda value: value["audit"].pop("fallback_used")),
            ("input", lambda value: value["config"].pop("input_token_ids")),
            ("output", lambda value: [sample["tokens"].pop("generated_token_ids") for section in ("correctness_control",) for sample in [value[section]]]),
            ("cleanup", lambda value: value.pop("session_cleanup")),
        )
        for label, remove in removals:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                profile_dir, execution = _populate(Path(temporary))
                document = json.loads(execution.read_text(encoding="utf-8"))
                remove(document)
                if label == "input":
                    document["row"].pop("input_token_ids", None)
                    document["correctness_control"]["tokens"].pop("input_token_ids", None)
                    for section in ("warmups", "measured"):
                        for sample in document[section]["samples"]:
                            sample["tokens"].pop("input_token_ids", None)
                if label == "output":
                    for section in ("warmups", "measured"):
                        for sample in document[section]["samples"]:
                            sample["tokens"].pop("generated_token_ids", None)
                execution.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(profile.Phase51MI300XProfileError):
                    profile.aggregate_profile(profile_dir, execution)

    def test_stats_trace_dispatches_must_close_per_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_dir, execution = _populate(Path(temporary))
            stats = profile_dir / "case_kernel_stats.csv"
            text = stats.read_text(encoding="utf-8").replace(
                "Cijk_Ailk_Bljk_SB_MT128x128x16,1,800",
                "Cijk_Ailk_Bljk_SB_MT128x128x16,2,800",
            )
            stats.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(profile.Phase51MI300XProfileError, "do not close"):
                profile.aggregate_profile(profile_dir, execution)

    def test_malformed_or_missing_raw_csv_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_dir, execution = _populate(Path(temporary))
            stats = profile_dir / "case_kernel_stats.csv"
            stats.write_text("Name,Calls\nkernel,1,3\n", encoding="utf-8")
            with self.assertRaises(profile.Phase51MI300XProfileError):
                profile.aggregate_profile(profile_dir, execution)

        with tempfile.TemporaryDirectory() as temporary:
            profile_dir, execution = _populate(Path(temporary))
            (profile_dir / "case_kernel_trace.csv").unlink()
            with self.assertRaisesRegex(profile.Phase51MI300XProfileError, "found 0"):
                profile.aggregate_profile(profile_dir, execution)

    def test_raw_digests_and_optional_output_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_dir, execution = _populate(root)
            output = root / "summary.json"
            result = profile.aggregate_profile(profile_dir, execution, output_path=output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
            self.assertEqual(
                result["raw_sha256"]["kernel_stats"]["sha256"],
                profile.sha256_file(profile_dir / "case_kernel_stats.csv"),
            )
            self.assertEqual(result["raw_sha256"]["execution_json"]["path"], "execution.json")


if __name__ == "__main__":
    unittest.main()
