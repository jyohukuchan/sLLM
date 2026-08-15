from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError  # noqa: E402
import engine_performance_common as contracts  # noqa: E402


def build_configuration_for(target: str) -> dict[str, str]:
    return contracts.expected_build_configuration(target)


def build_identity_for(row: dict[str, object], *, binary_sha256: str = "1" * 64) -> dict[str, object]:
    import subprocess
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    semantic_tree = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"], text=True).strip()
    return {
        "path": "/tmp/build-identity.json", "sha256": "2" * 64,
        "source_root": str(ROOT), "source_base_revision": revision, "semantic_tree": semantic_tree,
        "build_inputs_digest": "sha256:" + "5" * 64,
        "build_configuration": build_configuration_for(str(row["target"])),
        "target": row["target"], "backend": "hip",
        "rocm_release": "7.14.0", "rocm_root": "/opt/rocm/core-7.14", "binary_sha256": binary_sha256,
    }


def _evidence_static(target: str) -> dict[str, object]:
    device = contracts.expected_device(target)
    return {
        "target": target, "product": device["product"], "gpu_bdf": device["gpu_bdf"], "gpu_uuid": device["gpu_uuid"],
        "physical_hip_index": device["physical_hip_index"], "amd_smi_gpu_index": 0 if target == "gfx1030" else 1,
        "driver_version": "6.16.13", "kernel_version": "6.17.0-35-generic",
        "profile": {"current": "BOOTUP_DEFAULT", "available_profiles": ["BOOTUP_DEFAULT"], "digest": "sha256:" + "6" * 64},
        "limits": {"values": {
            "ppt0": {
                "socket_power_limit": {"value": 250, "unit": "W"},
                "max_power_limit": {"value": 250, "unit": "W"},
                "min_power_limit": {"value": 250, "unit": "W"},
            },
            "slowdown_edge_temperature": {"value": 100, "unit": "C"},
            "slowdown_hotspot_temperature": {"value": 100, "unit": "C"},
            "slowdown_vram_temperature": {"value": 100, "unit": "C"},
        }, "digest": "sha256:" + "7" * 64},
        "clock_levels": {"values": {"sys": {"frequency_levels": {"Level 0": {"value": "500", "unit": "MHz"}}}}, "digest": "sha256:" + "8" * 64},
        "vram_total_mb": 30704,
    }


def _evidence_metric() -> dict[str, object]:
    return {
        "temperature_c": {"edge": 35, "hotspot": 40, "mem": 37}, "gfx_clock_mhz": 1500, "mem_clock_mhz": 1000,
        "power_w": 100, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO", "throttle_status": "UNTHROTTLED",
        "vram_used_mb": 1000, "vram_total_mb": 30704, "ecc_uncorrectable": 0, "metric_digest": "sha256:" + "9" * 64,
    }


def _evidence_vram(target: str = "gfx1030") -> dict[str, object]:
    return {"source": "amd-smi monitor -v", "gpu": 0 if target == "gfx1030" else 1, "used_mb": 1000, "free_mb": 29704, "total_mb": 30704, "percent": 3.25}


def evidence_for(target: str) -> dict[str, object]:
    static = _evidence_static(target)
    metric = _evidence_metric()
    vram = _evidence_vram(target)
    return {"static": static, "metric": metric, "vram_auxiliary": vram, "process_state": "CLEAN", "violation": {"power_statuses": ["UNTHROTTLED"], "explicit_violation": False, "accumulator_available": False, "accumulator_reason": "AMD-SMI test limitation", "accumulator_digest": "sha256:" + "a" * 64}}


def monitor_capture_for(target: str, *, pid: int = 123) -> dict[str, object]:
    metric = _evidence_metric()
    vram = _evidence_vram(target)
    loader_paths = ["/opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000", "/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0"]
    loader_digest = "sha256:" + __import__("hashlib").sha256(contracts.canonical_bytes(loader_paths)).hexdigest()
    violation = {"power_statuses": ["UNTHROTTLED"], "explicit_violation": False, "accumulator_available": False, "accumulator_reason": "AMD-SMI test limitation", "accumulator_digest": "sha256:" + "b" * 64}
    loader = {"required_rocm_release": "7.14.0", "expected_root": "/opt/rocm/core-7.14", "resolved_paths": loader_paths, "path_digest": loader_digest, "library_digests": {path: "c" * 64 for path in loader_paths}, "process_ids": [pid]}
    sample = {"timestamp_ns": 1, "metric": metric, "vram_auxiliary": vram, "process": {"state": "OWNED", "pids": [pid]}, "loader_path_digest": loader_digest, "violation": violation}
    return {"pid": pid, "process_group_gone": True, "monitor": {"samples": [sample], "errors": [], "loader": loader, "loaders": [copy.deepcopy(loader)]}}


def first_row() -> dict[str, object]:
    matrix, _ = contracts.load_matrix()
    return copy.deepcopy(matrix["rows"][0])


def _snapshot(model_current: int, model_high_water: int, request_current: int = 0, workspace_current: int = 0, total_high_water: int | None = None) -> dict[str, object]:
    model = {"current_bytes": model_current, "high_water_bytes": model_high_water}
    request = {"current_bytes": request_current, "high_water_bytes": max(request_current, 150)}
    workspace = {"current_bytes": workspace_current, "high_water_bytes": max(workspace_current, 75)}
    return {
        "model_resident": model,
        "request_state": request,
        "workspace": workspace,
        "current_bytes": model_current + request_current + workspace_current,
        "high_water_bytes": total_high_water if total_high_water is not None else max(model_high_water, request["high_water_bytes"], workspace["high_water_bytes"]),
        "poisoned": False,
    }


def result_for(row: dict[str, object] | None = None) -> dict[str, object]:
    row = contracts.resolved_row(row or first_row())
    model = contracts.expected_model(row["model_size"])
    device = contracts.expected_device(row["target"])
    output_count = int(row["requested_output_tokens"])
    samples: list[dict[str, object]] = []
    for sample_index in range(13):
        base = 1_000_000_000 + sample_index * 1_000_000
        first = base + 700
        later = [first + 100 * index for index in range(1, output_count)]
        last = later[-1] if later else first
        stop = last + 100
        cleanup = stop + 500
        generated = [7] * output_count
        prefill_ns = 400
        decode_tokens = output_count - 1
        decode_rate = decode_tokens * 1_000_000_000 / (last - first) if decode_tokens else None
        sample = {
            "execution_path": "timed-production",
            "timing_instrumentation": "on",
            "events": {
                "request_start_ns": base,
                "prefill_submit_ns": base + 100,
                "prefill_complete_ns": base + 500,
                "first_token_ns": first,
                "later_token_publications_ns": later,
                "stop_ns": stop,
                "cleanup_ns": cleanup,
            },
            "derived": {
                "ttft_ns": first - base,
                "prefill_ns": prefill_ns,
                "prefill_tokens_per_second": len(row["input_token_ids"]) * 1_000_000_000 / prefill_ns,
                "e2e_ns": cleanup - base,
                "tpot_ns": [100] * len(later),
                "decode_tokens": decode_tokens,
                "decode_tokens_per_second": decode_rate,
            },
            "tokens": {
                "input_token_ids": row["input_token_ids"],
                "generated_token_ids": generated,
                "visible_token_ids": generated,
                "decode_input_token_ids": generated[:-1],
            },
            "stop": {"version": 1, "reason_version": 1, "kind": "max_new_tokens", "token_id": None},
            "audit": {
                "selected_backend": "hip",
                "target": row["target"],
                "device_index": 0,
                "model_fingerprint": model["lock_fingerprint"],
                "plan_digest": "sha256:" + "9" * 64,
                "fallback_used": False,
                "submission_count": 1,
                "kernel_dispatch_count": 1,
                "segment_count": 1,
                "boundary_count": 1,
                "all_dispatches_hip": True,
            },
            "memory": {
                "request_start": _snapshot(1000, 1200, 100, 75, 1400),
                "after_cleanup": _snapshot(1000, 1200, 0, 0, 1400),
            },
            "cleanup": {
                "sample_index": sample_index if sample_index < 3 else sample_index - 3,
                "request_dropped": True,
                "allocator_cleanup_validated": True,
                "retryable_cleanup": 0,
                "durable_quarantine": 0,
            },
        }
        samples.append(sample)
    return {
        "benchmark_schema_version": "engine-performance-direct-v1",
        "state": "PASS",
        "lane": "direct",
        "lane_definition": "pretokenized direct engine: request start excludes render/tokenize",
        "row": {
            "row_id": row["row_id"],
            "model_size": row["model_size"],
            "case_id": row["case_id"],
            "input_token_ids": row["input_token_ids"],
            "input_token_count": row["input_tokens"],
            "requested_output_tokens": row["requested_output_tokens"],
        },
        "identities": {
            "engine": "sllm",
            "backend": "hip",
            "session_id": 1,
            "device_index": 0,
            "target": row["target"],
            "model": {
                "model_size": row["model_size"],
                "repo_id": model["repo_id"],
                "resolved_revision": model["resolved_revision"],
                "lock_fingerprint": model["lock_fingerprint"],
            },
            "binding": {
                "model_fingerprint": model["lock_fingerprint"],
                "plan_digest": "sha256:" + "9" * 64,
            },
        },
        "model_load": {"event": "model_load", "start_ns": 0, "model_ready_ns": 110, "duration_ns": 110, "load_count": 1},
        "config": {
            "input_token_ids": row["input_token_ids"],
            "input_token_count": row["input_tokens"],
            "max_new_tokens": row["requested_output_tokens"],
            "greedy": True,
            "warmups": 3,
            "measured": 10,
            "tokenizer": False,
            "render": False,
            "stop_policy": {"stop_token_ids": [248046, 248044], "visible_stop_tokens": False},
        },
        "memory": {
            "model_ready": _snapshot(1000, 1200, 0, 0, 1200),
            "after_model_drop": _snapshot(0, 1200, 0, 0, 3600),
            "model_resident_high_water_bytes": 1200,
            "resident_vram_bytes": 1200,
            "resident_vram_source": "model_resident_allocator_high_water",
            "peak_vram_bytes": 3600,
            "peak_source": "runtime_allocator",
        },
        "audit": {
            "selected_backend": "hip",
            "target": row["target"],
            "device_index": 0,
            "submission_count": 13,
            "kernel_dispatch_count": 13,
            "segment_count": 13,
            "boundary_count": 13,
            "fallback_used": False,
            "all_dispatches_hip": True,
            "model_load_count": 1,
            "weight_encoding": "bf16",
            "fp8_provider": None,
            "request_model_load_count": 0,
            "model_reused": True,
            "sample_count": 13,
            "correctness_control_request_count": 1,
            "total_request_count": 14,
        },
        "cleanup": {
            "correctness_control_request_count": 1,
            "warmup_request_count": 3,
            "measured_request_count": 10,
            "request_cleanup_count": 14,
            "performance_sample_count": 13,
            "all_requests_dropped": True,
            "correctness_control_dropped": True,
            "retryable_cleanup": 0,
            "durable_quarantine": 0,
        },
        "session_cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
        "correctness_control": {
            "label": "correctness-only",
            "execution_path": "normal-untimed",
            "timing_instrumentation": "off",
            "included_in_performance_statistics": False,
            "tokens": {
                "input_token_ids": row["input_token_ids"],
                "generated_token_ids": [7] * output_count,
                "visible_token_ids": [7] * output_count,
                "decode_input_token_ids": [7] * (output_count - 1),
            },
            "stop": {"version": 1, "reason_version": 1, "kind": "max_new_tokens", "token_id": None},
            "audit": {
                "selected_backend": "hip", "target": row["target"], "device_index": 0,
                "model_fingerprint": model["lock_fingerprint"], "plan_digest": "sha256:" + "9" * 64,
                "fallback_used": False, "submission_count": 1, "kernel_dispatch_count": 1,
                "segment_count": 1, "boundary_count": 1,
                "all_dispatches_hip": True,
            },
            "memory": {
                "request_start": _snapshot(1000, 1200, 100, 75, 1400),
                "after_cleanup": _snapshot(1000, 1200, 0, 0, 1400),
            },
            "cleanup": {"request_dropped": True, "allocator_cleanup_validated": True},
            "comparison": {
                "mode": "exact", "scope": "every_warmup_and_measured_sample",
                "token_fields": ["input_token_ids", "generated_token_ids", "visible_token_ids", "decode_input_token_ids"],
                "stop_fields": ["version", "reason_version", "kind", "token_id"],
                "dispatch_fields": ["selected_backend", "target", "device_index", "model_fingerprint", "plan_digest", "fallback_used", "all_dispatches_hip", "submission_count", "kernel_dispatch_count", "segment_count", "boundary_count"],
                "dispatch_count_rule": "exact_when_token_and_stop_fields_match",
            },
        },
        "warmups": {"count": 3, "samples": samples[:3]},
        "measured": {"count": 10, "samples": samples[3:]},
    }


class EnginePerformanceSchemaTests(unittest.TestCase):
    def test_fixed_matrix_has_exact_ids_boundaries_and_realistic_timeouts(self) -> None:
        matrix, _ = contracts.load_matrix()
        self.assertEqual(len(matrix["rows"]), 22)
        self.assertEqual(matrix["revision"], 4)
        sequences = {item["sequence_id"]: item for item in matrix["token_sequences"]}
        self.assertEqual(sequences["short-odd"]["input_token_ids"], [1, 3, 17, 37, 73, 255, 256, 257, 2, 5, 11, 19, 23, 29, 31, 41, 43])
        self.assertEqual(len(sequences["boundary-255"]["input_token_ids"]), 255)
        self.assertEqual(len(sequences["prefill-long"]["input_token_ids"]), 1024)
        self.assertEqual(min(row["timeout_seconds"] for row in matrix["rows"]), 3600)
        self.assertGreater(max(row["timeout_seconds"] for row in matrix["rows"]), 600)
        self.assertEqual(matrix["protocol"]["warmup_requests"], 3)
        self.assertEqual(matrix["protocol"]["measured_requests"], 10)
        self.assertEqual(matrix["targets"][1]["gpu_bdf"], "0000:07:00.0")
        self.assertEqual(matrix["claims"], contracts.CLAIMS)

    def test_direct_and_aggregate_schemas_are_draft_2020_12_and_closed(self) -> None:
        for path in (contracts.DIRECT_SCHEMA_PATH, contracts.AGGREGATE_SCHEMA_PATH):
            schema = json.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            self.assertTrue(list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors({})), path)

    def test_valid_cli_result_is_schema_and_semantics_valid(self) -> None:
        row = first_row()
        result = result_for(row)
        contracts.validate_cli_result(result, row)
        self.assertEqual(len(result["measured"]["samples"]), 10)
        self.assertEqual(result["config"]["input_token_ids"], contracts.resolved_row(row)["input_token_ids"])

    def test_direct_schema_accepts_explicit_nvfp4_packed_dequant_identity(self) -> None:
        result = result_for(first_row())
        result["audit"]["weight_encoding"] = "nvfp4-e2m1-block16-e4m3fn-tensor-f32"
        result["audit"]["fp8_provider"] = "nvfp4-packed-dequant"
        contracts.schema_validate(result, contracts.DIRECT_SCHEMA_PATH, "NVFP4 direct result")

        result["audit"]["fp8_provider"] = "implicit-nvfp4"
        with self.assertRaises(ContractError):
            contracts.schema_validate(result, contracts.DIRECT_SCHEMA_PATH, "invalid NVFP4 direct result")

    def test_stop_semantics_reject_short_max_and_nonterminal_stop_token(self) -> None:
        policy = {"stop_token_ids": [248046, 248044], "visible_stop_tokens": False}
        contracts.validate_stop_semantics(
            [7, 8], {"version": 1, "reason_version": 1, "kind": "max_new_tokens", "token_id": None},
            policy, 2, "test",
        )
        contracts.validate_stop_semantics(
            [7, 248044], {"version": 1, "reason_version": 1, "kind": "stop_token", "token_id": 248044},
            policy, 3, "test",
        )
        invalid = (
            ([7], {"version": 1, "reason_version": 1, "kind": "max_new_tokens", "token_id": None}, 2),
            ([7, 8], {"version": 1, "reason_version": 1, "kind": "stop_token", "token_id": 248044}, 3),
            ([248044, 7], {"version": 1, "reason_version": 1, "kind": "stop_token", "token_id": 248044}, 3),
            ([7, 8], {"version": 2, "reason_version": 1, "kind": "max_new_tokens", "token_id": None}, 2),
        )
        for generated, stop, maximum in invalid:
            with self.subTest(generated=generated, stop=stop), self.assertRaises(ContractError):
                contracts.validate_stop_semantics(generated, stop, policy, maximum, "test")

    def test_snapshot_rejects_category_high_water_above_total(self) -> None:
        snapshot = _snapshot(1000, 1200, 0, 0, 1400)
        snapshot["request_state"]["high_water_bytes"] = 1401
        with self.assertRaises(ContractError):
            contracts.validate_snapshot(snapshot, "test snapshot")

    def test_direct_schema_rejects_stale_stop_protocol_version(self) -> None:
        row = first_row()
        result = result_for(row)
        result["correctness_control"]["stop"]["version"] = 2
        with self.assertRaises(ContractError):
            contracts.validate_cli_result(result, row)

    def test_build_configuration_is_closed_exact_and_target_bound(self) -> None:
        row = first_row()
        identity = build_identity_for(row)
        manifest = {
            "row_id": row["row_id"],
            "binary": {"sha256": identity["binary_sha256"]},
            "build_identity": identity,
        }
        contracts.validate_build_configuration(identity["build_configuration"], str(row["target"]))
        for mutation in (
            lambda value: value["build_configuration"].pop("cargo_profile"),
            lambda value: value["build_configuration"].__setitem__("unknown", "value"),
            lambda value: value["build_configuration"].__setitem__("CMAKE_HIP_ARCHITECTURES", "gfx1201"),
        ):
            changed = copy.deepcopy(identity)
            mutation(changed)
            manifest["build_identity"] = changed
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                contracts.validate_manifest_evidence(manifest)

    def test_render_tokenize_lane_is_not_a_direct_matrix_result(self) -> None:
        result = result_for(first_row())
        result["lane"] = "render-tokenize"
        result["lane_definition"] = "CLI end-to-end: request start includes chat render and tokenizer encode"
        with self.assertRaises(ContractError):
            contracts.validate_cli_result(result, first_row())

    def test_matrix_missing_duplicate_nonmonotonic_and_wrong_target_fail(self) -> None:
        matrix, _ = contracts.load_matrix()
        mutations = []
        missing = copy.deepcopy(matrix); missing["rows"].pop(); mutations.append(missing)
        duplicate = copy.deepcopy(matrix); duplicate["rows"].append(copy.deepcopy(duplicate["rows"][0])); mutations.append(duplicate)
        nonmonotonic = copy.deepcopy(matrix); nonmonotonic["rows"][1], nonmonotonic["rows"][2] = nonmonotonic["rows"][2], nonmonotonic["rows"][1]; mutations.append(nonmonotonic)
        wrong_target = copy.deepcopy(matrix); wrong_target["targets"][0]["target"] = "gfx1201"; mutations.append(wrong_target)
        changed_ids = copy.deepcopy(matrix); changed_ids["token_sequences"][1]["input_token_ids"][0] = 2; mutations.append(changed_ids)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(ContractError):
                    contracts.validate_matrix_document(changed)


if __name__ == "__main__":
    unittest.main()
