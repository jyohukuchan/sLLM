from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, canonical_bytes  # noqa: E402
import engine_performance_common as contracts  # noqa: E402
import run_engine_performance_render as render  # noqa: E402
import run_engine_performance as direct_runner  # noqa: E402
from ci.tests.test_engine_performance_schema import build_configuration_for, evidence_for, monitor_capture_for, result_for as direct_result_for  # noqa: E402


def render_row(target: str = "gfx1030") -> dict[str, object]:
    matrix, _ = render.load_matrix()
    return copy.deepcopy(next(row for row in matrix["rows"] if row["target"] == target))


def render_result_for(target: str = "gfx1030") -> dict[str, object]:
    row = render_row(target)
    direct_row = {
        "row_id": "engine-performance-direct-4b-gfx1030-short-odd",
        "model_size": "4B",
        "case_id": "short-odd",
        "input_token_sequence": "short-odd",
        "input_token_ids": [1, 3, 17, 37, 73, 255, 256, 257, 2, 5, 11, 19, 23, 29, 31, 41, 43],
        "input_tokens": 17,
        "requested_output_tokens": 17,
        "target": "gfx1030",
        "timeout_seconds": 5400,
    }
    result = direct_result_for(direct_row)
    result["benchmark_schema_version"] = render.VERSION
    result["lane"] = "render-tokenize"
    result["lane_definition"] = "CLI end-to-end: request start includes chat render and tokenizer encode"
    result["row"] = {
        "row_id": row["row_id"], "model_size": "4B", "case_id": "chat-hello",
        "input_token_ids": list(render.INPUT_TOKEN_IDS), "input_token_count": 13,
        "requested_output_tokens": 17,
    }
    result["identities"]["target"] = target
    result["config"] = {
        "input_token_ids": list(render.INPUT_TOKEN_IDS), "input_token_count": 13,
        "max_new_tokens": 17, "greedy": True, "warmups": 3, "measured": 10,
        "tokenizer": True, "render": True,
        "stop_policy": {"stop_token_ids": [248046, 248044], "visible_stop_tokens": False},
    }
    result["memory"]["model_resident_high_water_bytes"] = result["memory"]["resident_vram_bytes"]
    result["memory"]["resident_vram_source"] = "model_resident_allocator_high_water"
    result["memory"]["model_ready"] = {
        "model_resident": {"current_bytes": 1000, "high_water_bytes": 1200},
        "request_state": {"current_bytes": 0, "high_water_bytes": 0},
        "workspace": {"current_bytes": 0, "high_water_bytes": 0},
        "current_bytes": 1000, "high_water_bytes": 1200, "poisoned": False,
    }
    result["memory"]["after_model_drop"] = {
        "model_resident": {"current_bytes": 0, "high_water_bytes": 3600},
        "request_state": {"current_bytes": 0, "high_water_bytes": 0},
        "workspace": {"current_bytes": 0, "high_water_bytes": 0},
        "current_bytes": 0, "high_water_bytes": 3600, "poisoned": False,
    }
    for group in ("warmups", "measured"):
        for sample in result[group]["samples"]:
            sample["execution_path"] = "timed-production"
            sample["timing_instrumentation"] = "on"
            sample["tokens"]["input_token_ids"] = list(render.INPUT_TOKEN_IDS)
            sample["derived"]["prefill_tokens_per_second"] = 13 * 1_000_000_000 / sample["derived"]["prefill_ns"]
            sample["audit"]["target"] = target
            sample["cleanup"]["allocator_cleanup_validated"] = True
            after_cleanup = sample["memory"]["after_cleanup"]
            after_cleanup["model_resident"] = {"current_bytes": 1000, "high_water_bytes": 1200}
            after_cleanup["request_state"] = {"current_bytes": 0, "high_water_bytes": 1200}
            after_cleanup["workspace"] = {"current_bytes": 0, "high_water_bytes": 1200}
            after_cleanup["current_bytes"] = 1000
    result["audit"].update({
        "target": target,
        "correctness_control_request_count": 1,
        "total_request_count": 14,
    })
    result["cleanup"] = {
        "correctness_control_request_count": 1, "warmup_request_count": 3, "measured_request_count": 10,
        "request_cleanup_count": 14, "performance_sample_count": 13, "all_requests_dropped": True,
        "correctness_control_dropped": True, "retryable_cleanup": 0, "durable_quarantine": 0,
    }
    first = result["warmups"]["samples"][0]
    result["correctness_control"] = {
        "label": "correctness-only", "execution_path": "normal-untimed", "timing_instrumentation": "off",
        "included_in_performance_statistics": False,
        "tokens": copy.deepcopy(first["tokens"]), "stop": copy.deepcopy(first["stop"]),
        "audit": copy.deepcopy(first["audit"]),
        "memory": {"request_start": copy.deepcopy(first["memory"]["request_start"]), "after_cleanup": copy.deepcopy(first["memory"]["after_cleanup"])},
        "cleanup": {"request_dropped": True, "allocator_cleanup_validated": True},
        "comparison": {
            "mode": "exact", "scope": "every_warmup_and_measured_sample",
            "token_fields": ["input_token_ids", "generated_token_ids", "visible_token_ids", "decode_input_token_ids"],
            "stop_fields": ["version", "reason_version", "kind", "token_id"],
            "dispatch_fields": ["selected_backend", "target", "device_index", "model_fingerprint", "plan_digest", "fallback_used", "all_dispatches_hip", "submission_count", "kernel_dispatch_count", "segment_count", "boundary_count"],
            "dispatch_count_rule": "exact_when_token_and_stop_fields_match",
        },
    }
    return result


def observation(target: str) -> dict[str, object]:
    return {
        "selected_device": contracts.expected_device(target),
        "health": {"available": True, "reliable": True, "state": "OK", "ras_uncorrectable_count": 0},
        "process": {"available": True, "reliable": True, "state": "CLEAN", "gpu_processes": [], "residual_runner_children": []},
    }


def allowlisted_observation(target: str, *, present: bool) -> dict[str, object]:
    value = observation(target)
    pid = 4242
    records = []
    if present:
        records = [{
            "process_info": {
                "name": "inert-target", "pid": pid,
                "memory_usage": {
                    "gtt_mem": {"value": 4096, "unit": "B"},
                    "cpu_mem": {"value": 0, "unit": "B"},
                    "vram_mem": {"value": 4096, "unit": "B"},
                },
                "mem_usage": {"value": 8192, "unit": "B"},
                "usage": {"gfx": {"value": 0, "unit": "ns"}, "enc": {"value": 0, "unit": "ns"}},
                "sdma_usage": {"value": 0, "unit": "us"},
                "cu_occupancy": "N/A", "evicted_time": {"value": 0, "unit": "ms"},
            },
        }]
    value["process"]["gpu_processes"] = direct_runner._allowed_process_observation(records, (pid,))  # type: ignore[index]
    return value


def evidence(target: str) -> dict[str, object]:
    return direct_runner._build_evidence(
        evidence_for(target), evidence_for(target), monitor_capture_for(target), target,
        {"path": direct_runner.AMD_SMI_EXECUTABLE, "tool_version": "test", "library_version": "test", "rocm_version": "7.14.0"},
    )


def manifest_for(root: Path, target: str, raw: dict[str, object]) -> Path:
    row = render_row(target)
    raw_path = root / f"raw-{target}.json"
    raw_path.write_bytes(canonical_bytes(raw))
    subprocess = __import__("subprocess")
    source_base_revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    semantic_tree = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"], text=True).strip()
    manifest = {
        "benchmark_schema_version": render.VERSION, "record_kind": "evidence_manifest", "state": "PASS",
        "required": False, "failure_reason": None, "row_id": row["row_id"], "claims": dict(render.CLAIMS),
        "matrix": {"path": str(render.MATRIX_PATH), "matrix_id": render.VERSION, "sha256": render.load_matrix()[1]},
        "binary": {"path": str(root / "sllm"), "sha256": "1" * 64, "bytes": 1},
        "build_identity": {"path": str(root / "build.json"), "sha256": "2" * 64, "source_root": str(ROOT), "source_base_revision": source_base_revision, "semantic_tree": semantic_tree, "build_inputs_digest": "sha256:" + "5" * 64, "build_configuration": build_configuration_for(target), "target": target, "backend": "hip", "rocm_release": "7.14.0", "rocm_root": "/opt/rocm/core-7.14", "binary_sha256": "1" * 64},
        "model_lock": {"path": str(ROOT / "docs/models/locks/qwen3.5-4b-bf16.json"), "sha256": "6" * 64, "fingerprint": contracts.expected_model("4B")["lock_fingerprint"]},
        "model_cache": {"path": str(root / "cache"), "sha256": "7" * 64},
        "raw_artifact": {"path": str(raw_path), "sha256": __import__("hashlib").sha256(raw_path.read_bytes()).hexdigest(), "bytes": raw_path.stat().st_size},
        "observations": {"pre": observation(target), "post": observation(target)}, "evidence": evidence(target),
        "execution": {"exit_code": 0, "timed_out": False, "timeout_seconds": 5400, "stderr_bytes": 0, "term_sent": False, "kill_sent": False, "process_group_gone": True},
        "cleanup": {"pre_process_clean": True, "post_process_clean": True, "process_group_gone": True, "retryable_cleanup": 0, "durable_quarantine": 0},
    }
    path = root / f"manifest-{target}.json"
    path.write_bytes(canonical_bytes(manifest))
    return path


class RenderPerformanceTests(unittest.TestCase):
    def test_closed_matrix_binds_frontend_prompt_and_exact_ids(self) -> None:
        matrix, _ = render.load_matrix()
        self.assertEqual([row["target"] for row in matrix["rows"]], ["gfx1030", "gfx1201"])
        self.assertEqual(matrix["render_contract"]["rendered_prompt"], render.RENDERED_PROMPT)
        self.assertEqual(matrix["render_contract"]["input_token_ids"], render.INPUT_TOKEN_IDS)
        self.assertEqual(matrix["protocol"]["warmup_requests"], 3)
        self.assertEqual(matrix["protocol"]["measured_requests"], 10)
        changed = copy.deepcopy(matrix)
        changed["render_contract"]["input_token_ids"][0] += 1
        with self.assertRaises(ContractError):
            render.validate_matrix_document(changed)

    def test_expected_command_uses_render_tokenize_and_message_spelling(self) -> None:
        command = render._expected_command(Path("/tmp/sllm"), render_row(), Path("/lock"), Path("/cache"))
        self.assertEqual(command[0:5], ["/tmp/sllm", "benchmark", "--lane", "render-tokenize", "--lock"])
        self.assertIn("--message", command)
        self.assertEqual(command[command.index("--message") + 1], "user:Hello")
        self.assertNotIn("--input-token-ids", command)
        self.assertEqual(command[command.index("--thinking") + 1], "disabled")

    def test_result_contract_rejects_lane_ids_counts_and_math_drift(self) -> None:
        row = render_row()
        result = render_result_for()
        render.validate_cli_result(result, row)
        for mutation in (
            lambda value: value.__setitem__("lane", "direct"),
            lambda value: value["config"]["input_token_ids"].__setitem__(0, 0),
            lambda value: value["measured"]["samples"].pop(),
            lambda value: value["measured"]["samples"][0]["derived"].__setitem__("ttft_ns", 999999),
            lambda value: value["correctness_control"]["audit"].__setitem__("fallback_used", True),
        ):
            changed = copy.deepcopy(result)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                render.validate_cli_result(changed, row)

    def test_aggregate_keeps_raw_digests_timing_samples_and_run_level_vram(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-render-aggregate-") as directory:
            root = Path(directory)
            raw_1030 = render_result_for("gfx1030")
            raw_1201 = render_result_for("gfx1201")
            manifests = [manifest_for(root, "gfx1030", raw_1030), manifest_for(root, "gfx1201", raw_1201)]
            summary = __import__("aggregate_engine_performance_render").aggregate_manifests(manifests, root / "out", verify_external_digests=False)
            self.assertEqual(summary["counts"]["collected_samples"], 20)
            self.assertEqual(set(summary["rows"][0]["metrics"]), {"ttft_ns", "prefill_ns", "tpot_ns", "decode_token_per_s", "e2e_ns", "resident_vram_bytes", "peak_vram_bytes"})
            self.assertEqual(summary["rows"][0]["metrics"]["ttft_ns"]["count"], 10)
            self.assertEqual(summary["rows"][0]["metrics"]["resident_vram_bytes"]["count"], 1)
            self.assertEqual(summary["rows"][0]["metrics"]["peak_vram_bytes"]["count"], 1)
            self.assertTrue(summary["rows"][0]["raw_result_sha256"])
            self.assertEqual(summary["identity"]["build_identity_by_target"]["gfx1030"]["target"], "gfx1030")
            self.assertEqual(summary["identity"]["build_identity_by_target"]["gfx1201"]["build_configuration"]["CMAKE_HIP_ARCHITECTURES"], "gfx1201")
            self.assertEqual(summary["graph_csv"]["row_count"], 14)

    def test_render_aggregate_accepts_recorded_inert_allowlist_presence_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-render-inert-allowlist-") as directory:
            root = Path(directory)
            manifests = [manifest_for(root, target, render_result_for(target)) for target in ("gfx1030", "gfx1201")]
            for path in manifests:
                document = json.loads(path.read_text(encoding="utf-8"))
                target = document["build_identity"]["target"]
                document["observations"] = {
                    "pre": allowlisted_observation(target, present=True),
                    "post": allowlisted_observation(target, present=False),
                }
                path.write_bytes(canonical_bytes(document))
            summary = __import__("aggregate_engine_performance_render").aggregate_manifests(
                manifests, root / "out", verify_external_digests=False,
            )
            self.assertEqual(summary["state"], "PASS")

    def test_render_aggregate_rejects_mixed_source_revision_and_tree(self) -> None:
        subprocess = __import__("subprocess")
        parent_revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD^"], text=True).strip()
        parent_tree = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", f"{parent_revision}^{{tree}}"], text=True).strip()
        for key, replacement in (("source_base_revision", parent_revision), ("semantic_tree", parent_tree)):
            with tempfile.TemporaryDirectory(prefix="sllm-render-mixed-source-") as directory:
                root = Path(directory)
                manifests = [manifest_for(root, target, render_result_for(target)) for target in ("gfx1030", "gfx1201")]
                document = json.loads(manifests[0].read_text(encoding="utf-8"))
                document["build_identity"][key] = replacement
                manifests[0].write_bytes(canonical_bytes(document))
                with self.subTest(key=key), self.assertRaises(ContractError):
                    __import__("aggregate_engine_performance_render").aggregate_manifests(manifests, root / "out", verify_external_digests=False)

    def test_render_validation_failure_publishes_nothing_and_sidecars_are_no_replace(self) -> None:
        aggregator = __import__("aggregate_engine_performance_render")
        with tempfile.TemporaryDirectory(prefix="sllm-render-atomic-validation-") as directory:
            root = Path(directory)
            manifests = [manifest_for(root, target, render_result_for(target)) for target in ("gfx1030", "gfx1201")]
            output = root / "out"
            with mock.patch.object(contracts, "schema_validate", side_effect=ContractError("synthetic schema failure")):
                with self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, output, verify_external_digests=False)
            self.assertFalse(output.exists())

        for sidecar in ("summary.json.sha256", "graph.csv.sha256"):
            with tempfile.TemporaryDirectory(prefix="sllm-render-sidecar-no-replace-") as directory:
                root = Path(directory)
                manifests = [manifest_for(root, target, render_result_for(target)) for target in ("gfx1030", "gfx1201")]
                output = root / "out"
                output.mkdir()
                existing = output / sidecar
                existing.write_bytes(b"existing-sidecar\n")
                with self.subTest(sidecar=sidecar), self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, output, verify_external_digests=False)
                self.assertEqual(existing.read_bytes(), b"existing-sidecar\n")
                self.assertEqual(sorted(path.name for path in output.iterdir()), [sidecar])

    def test_render_aggregate_rejects_build_configuration_target_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-render-build-config-") as directory:
            root = Path(directory)
            manifests = [manifest_for(root, target, render_result_for(target)) for target in ("gfx1030", "gfx1201")]
            document = json.loads(manifests[0].read_text(encoding="utf-8"))
            document["build_identity"]["build_configuration"]["CMAKE_HIP_ARCHITECTURES"] = "gfx1201"
            manifests[0].write_bytes(canonical_bytes(document))
            with self.assertRaises(ContractError):
                __import__("aggregate_engine_performance_render").aggregate_manifests(manifests, root / "out", verify_external_digests=False)


if __name__ == "__main__":
    unittest.main()
