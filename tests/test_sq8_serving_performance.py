import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-performance.py"
GIT_COMMIT = "a" * 40
BINARY_SHA256 = "b" * 64
WORKER_PID = 4_242
VRAM_BYTES = 24 * 1024**3
M128_PREFILL_MODE = "m128-chunk128"
M128_INPUT_SCHEMA_VERSION = "ullm.sq8.serving_performance.raw.v2"
M128_PREFILL_CHUNK_TOKENS = 128
M128_PREFILL_IMPLEMENTATION = "sq8.fixed-m128-cached-prefix.v1"


def load_module():
    spec = importlib.util.spec_from_file_location("sq8_serving_performance", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.value = 1_000


def raw_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact_json(value) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def prefill_config(module, prefill_mode: str):
    if prefill_mode == module.PREFILL_MODE:
        return {
            "schema_version": module.INPUT_SCHEMA_VERSION,
            "prefill_chunk_tokens": module.PREFILL_CHUNK_TOKENS,
            "prefill_implementation": module.PREFILL_IMPLEMENTATION,
        }
    if prefill_mode == M128_PREFILL_MODE:
        return {
            "schema_version": M128_INPUT_SCHEMA_VERSION,
            "prefill_chunk_tokens": M128_PREFILL_CHUNK_TOKENS,
            "prefill_implementation": M128_PREFILL_IMPLEMENTATION,
        }
    raise AssertionError(f"unsupported test prefill mode: {prefill_mode}")


def prompt_execution_calls(prompt_tokens: int, chunk_tokens: int) -> int:
    return prompt_tokens // chunk_tokens + prompt_tokens % chunk_tokens


def active_snapshot(
    module, request_id: str, prompt_tokens: int, generated: int, cache_len: int, status: str
):
    return {
        "status": status,
        "active_request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_processed": prompt_tokens,
        "generated_tokens": generated,
        "cache_lengths": [cache_len] * module.STACK_LAYERS,
        "scheduler_active": 1,
        "scheduler_waiting": 0,
        "block_size_tokens": module.BLOCK_TOKENS,
        "total_blocks": module.CACHE_BLOCKS,
        "free_blocks": 0,
        "allocated_blocks": module.CACHE_BLOCKS,
        "free_runs": 0,
        "largest_free_run": 0,
    }


def ready_snapshot(module):
    return {
        "status": "ready",
        "active_request_id": None,
        "prompt_tokens": 0,
        "prompt_tokens_processed": 0,
        "generated_tokens": 0,
        "cache_lengths": [0] * module.STACK_LAYERS,
        "scheduler_active": 0,
        "scheduler_waiting": 0,
        "block_size_tokens": module.BLOCK_TOKENS,
        "total_blocks": module.CACHE_BLOCKS,
        "free_blocks": module.CACHE_BLOCKS,
        "allocated_blocks": 0,
        "free_runs": 1,
        "largest_free_run": module.CACHE_BLOCKS,
    }


def vram_capture(module, clock: Clock):
    command_start = clock.value
    command_end = command_start + 10
    captured = command_end + 5
    clock.value = captured + 10
    raw = compact_json(
        [
            {
                "gpu": module.AMD_SMI_GPU_INDEX,
                "process_list": [
                    {
                        "process_info": {
                            "pid": WORKER_PID,
                            "mem_usage": {"value": VRAM_BYTES, "unit": "B"},
                        }
                    }
                ],
            }
        ]
    )
    return {
        "amd_smi_command_start_elapsed_ns": command_start,
        "amd_smi_command_end_elapsed_ns": command_end,
        "captured_elapsed_ns": captured,
        "worker_pid": WORKER_PID,
        "amd_smi_gpu_index": module.AMD_SMI_GPU_INDEX,
        "amd_smi_mem_usage_bytes": VRAM_BYTES,
        "amd_smi_process_raw_json": raw,
        "amd_smi_process_raw_sha256": raw_sha(raw),
        "kfd_gpu_id": module.R9700_KFD_GPU_ID,
        "kfd_vram_bytes": VRAM_BYTES,
        "kfd_processes": [
            {"pid": 111, "vram_bytes": 0},
            {"pid": WORKER_PID, "vram_bytes": VRAM_BYTES},
        ],
        "unrelated_positive_kfd_pids": [],
    }


def metric_capture(module, clock: Clock):
    command_start = clock.value
    command_end = command_start + 10
    clock.value = command_end + 10
    raw = compact_json(
        {
            "gpu_data": [
                {
                    "gpu": module.AMD_SMI_GPU_INDEX,
                    "temperature": {"hotspot": {"value": 55.0, "unit": "C"}},
                    "power": {"socket_power": {"value": 180.0, "unit": "W"}},
                    "clock": {
                        "gfx_0": {"clk": {"value": 2200.0, "unit": "MHz"}},
                        "mem_0": {"clk": {"value": 1250.0, "unit": "MHz"}},
                        "fclk_0": {"clk": {"value": 1800.0, "unit": "MHz"}},
                    },
                }
            ]
        }
    )
    return {
        "command_start_elapsed_ns": command_start,
        "command_end_elapsed_ns": command_end,
        "captured_elapsed_ns": command_end,
        "gpu_index": module.AMD_SMI_GPU_INDEX,
        "hotspot_temperature_c": 55.0,
        "socket_power_w": 180.0,
        "gfx_clock_mhz": 2200.0,
        "memory_clock_mhz": 1250.0,
        "fabric_clock_mhz": 1800.0,
        "raw_json": raw,
        "raw_sha256": raw_sha(raw),
    }


def ttft_sample(
    module,
    clock: Clock,
    prompt_tokens: int,
    phase: str,
    index: int,
    ttft_ns: int,
    prefill_chunk_tokens: int,
):
    request_id = f"perf-ttft-p{prompt_tokens:04d}-{phase}-{index}"
    start = clock.value
    first = start + ttft_ns
    cancel_set = first + 10
    cancellation = cancel_set + 10
    reset_start = cancellation + 10
    reset_end = reset_start + 100
    clock.value = reset_end + 10
    calls = prompt_execution_calls(prompt_tokens, prefill_chunk_tokens)
    result = {
        "phase": phase,
        "sample_index": index,
        "request_id": request_id,
        "request_start_elapsed_ns": start,
        "first_token_elapsed_ns": first,
        "ttft_ns": ttft_ns,
        "first_token_id": 123,
        "first_token_cache_len": prompt_tokens,
        "prompt_execution_calls": calls,
        "prompt_progress_events": calls - 1,
        "first_token_snapshot": active_snapshot(
            module, request_id, prompt_tokens, 1, prompt_tokens, "decoding"
        ),
        "cancel_set_elapsed_ns": cancel_set,
        "cancellation_observed_elapsed_ns": cancellation,
        "cancellation_snapshot": active_snapshot(
            module, request_id, prompt_tokens, 1, prompt_tokens, "cancelling"
        ),
        "reset_start_elapsed_ns": reset_start,
        "reset_end_elapsed_ns": reset_end,
        "reset_ns": reset_end - reset_start,
        "release_outcome": "cancelled",
        "release_generated_tokens": 1,
        "release_reset_complete": True,
        "post_reset_snapshot": ready_snapshot(module),
    }
    result["vram_after_reset"] = vram_capture(module, clock)
    return result


def ttft_case(
    module,
    clock: Clock,
    prompt_tokens: int,
    ttft_ns: int,
    prefill_chunk_tokens: int,
):
    p50, p95 = module.TTFT_LIMITS[prompt_tokens]
    before = metric_capture(module, clock)
    samples = [
        ttft_sample(
            module,
            clock,
            prompt_tokens,
            "warmup",
            index,
            ttft_ns,
            prefill_chunk_tokens,
        )
        for index in range(2)
    ]
    samples.extend(
        ttft_sample(
            module,
            clock,
            prompt_tokens,
            "measured",
            index,
            ttft_ns,
            prefill_chunk_tokens,
        )
        for index in range(5)
    )
    after = metric_capture(module, clock)
    return {
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": module.TTFT_MAX_NEW_TOKENS,
        "prompt_token_pattern": module.PROMPT_TOKEN_PATTERN,
        "prompt_token_ids_u32_le_sha256": module.ascending_prompt_sha256(prompt_tokens),
        "p50_limit_seconds": p50,
        "p95_limit_seconds": p95,
        "metric_before": before,
        "samples": samples,
        "metric_after": after,
    }


def decode_sample(
    module,
    clock: Clock,
    phase: str,
    index: int,
    interval_ns: int,
    prefill_chunk_tokens: int,
):
    request_id = f"perf-decode-p0032-g0064-{phase}-{index}"
    start = clock.value
    first = start + 1_000_000_000
    generated = []
    for generated_index in range(module.DECODE_GENERATED_TOKENS):
        available = first + generated_index * interval_ns
        generated.append(
            {
                "generated_index": generated_index,
                "token_id": 100 + generated_index,
                "cache_len": module.DECODE_PROMPT_TOKENS + generated_index,
                "available_elapsed_ns": available,
                "terminal_reason": (
                    "length" if generated_index == module.DECODE_GENERATED_TOKENS - 1 else None
                ),
            }
        )
    last = generated[-1]["available_elapsed_ns"]
    reset_start = last + 10
    reset_end = reset_start + 100
    clock.value = reset_end + 10
    prompt_calls = prompt_execution_calls(module.DECODE_PROMPT_TOKENS, prefill_chunk_tokens)
    result = {
        "phase": phase,
        "sample_index": index,
        "request_id": request_id,
        "request_start_elapsed_ns": start,
        "first_token_elapsed_ns": first,
        "last_token_elapsed_ns": last,
        "ttft_ns": first - start,
        "decode_duration_ns": last - first,
        "prompt_execution_calls": prompt_calls,
        "prompt_progress_events": prompt_calls - 1,
        "execution_calls": prompt_calls + module.DECODE_TIMED_TOKENS,
        "generated": generated,
        "terminal_snapshot": active_snapshot(module, request_id, 32, 64, 95, "finishing"),
        "reset_start_elapsed_ns": reset_start,
        "reset_end_elapsed_ns": reset_end,
        "reset_ns": reset_end - reset_start,
        "release_outcome": "length",
        "release_generated_tokens": 64,
        "release_reset_complete": True,
        "post_reset_snapshot": ready_snapshot(module),
    }
    result["vram_after_reset"] = vram_capture(module, clock)
    return result


def decode_case(module, clock: Clock, interval_ns: int, prefill_chunk_tokens: int):
    before = metric_capture(module, clock)
    samples = [
        decode_sample(
            module,
            clock,
            "warmup",
            index,
            interval_ns,
            prefill_chunk_tokens,
        )
        for index in range(2)
    ]
    samples.extend(
        decode_sample(
            module,
            clock,
            "measured",
            index,
            interval_ns,
            prefill_chunk_tokens,
        )
        for index in range(5)
    )
    after = metric_capture(module, clock)
    return {
        "prompt_tokens": module.DECODE_PROMPT_TOKENS,
        "max_new_tokens": module.DECODE_GENERATED_TOKENS,
        "generated_tokens": module.DECODE_GENERATED_TOKENS,
        "prompt_token_pattern": module.PROMPT_TOKEN_PATTERN,
        "prompt_token_ids_u32_le_sha256": module.ascending_prompt_sha256(
            module.DECODE_PROMPT_TOKENS
        ),
        "decode_tokens_per_sample": module.DECODE_TIMED_TOKENS,
        "p50_tokens_per_second_minimum": module.DECODE_P50_TOKENS_PER_SECOND_MINIMUM,
        "p95_inter_token_seconds_maximum": module.DECODE_P95_INTER_TOKEN_SECONDS_MAXIMUM,
        "p95_inter_token_pooling": "all_measured_inter_token_intervals",
        "metric_before": before,
        "samples": samples,
        "metric_after": after,
    }


def environment(module):
    version = "\n".join(module.REQUIRED_AMD_SMI_VERSION_LINES) + "\namdgpu version: diagnostic\n"
    gpu_list = compact_json(
        [
            {
                "gpu": module.AMD_SMI_GPU_INDEX,
                "bdf": module.R9700_BDF,
                "uuid": module.R9700_UUID,
                "kfd_id": module.R9700_KFD_GPU_ID,
            }
        ]
    )
    return {
        "hip_visible_devices": "1",
        "amd_smi_version_raw": version,
        "amd_smi_version_raw_sha256": raw_sha(version),
        "amd_smi_list_raw_json": gpu_list,
        "amd_smi_list_raw_sha256": raw_sha(gpu_list),
        "target_gpu_index": module.AMD_SMI_GPU_INDEX,
        "target_gpu_bdf": module.R9700_BDF,
        "target_gpu_uuid": module.R9700_UUID,
        "target_kfd_gpu_id": module.R9700_KFD_GPU_ID,
    }


def valid_document(
    module,
    *,
    ttft_overrides=None,
    decode_interval_ns=50_000_000,
    prefill_mode=None,
):
    ttft_overrides = ttft_overrides or {}
    prefill_mode = prefill_mode or module.PREFILL_MODE
    config = prefill_config(module, prefill_mode)
    chunk_tokens = config["prefill_chunk_tokens"]
    clock = Clock()
    initial = vram_capture(module, clock)
    cases = [
        ttft_case(
            module,
            clock,
            prompt,
            ttft_overrides.get(prompt, 1_000_000_000),
            chunk_tokens,
        )
        for prompt in module.TTFT_LIMITS
    ]
    decode = decode_case(module, clock, decode_interval_ns, chunk_tokens)
    final = vram_capture(module, clock)
    return {
        "schema_version": config["schema_version"],
        "runner_git_commit": GIT_COMMIT,
        "runner_worktree_clean": True,
        "runner_binary_sha256": BINARY_SHA256,
        "prefill_mode": prefill_mode,
        "prefill_chunk_tokens": chunk_tokens,
        "prefill_implementation": config["prefill_implementation"],
        "warmup_runs": module.WARMUP_RUNS,
        "measured_runs": module.MEASURED_RUNS,
        "percentile_method": module.PERCENTILE_METHOD,
        "vram_policy": module.VRAM_POLICY,
        "timer": {
            "clock": "std::time::Instant_monotonic",
            "ttft_start": "immediately_before_session.start",
            "ttft_end": "immediately_after_first_token_return_before_snapshot",
            "fixture_construction_included": False,
            "model_load_included": False,
            "cleanup_included": False,
        },
        "sampling": {
            "method": "greedy_temperature_zero",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "seed": 0,
            "eos_token_ids": list(module.EOS_TOKEN_IDS),
        },
        "load_seconds": 20.0,
        "artifact_content_sha256": module.EXPECTED_ARTIFACT_SHA256,
        "package_manifest_sha256": module.EXPECTED_PACKAGE_SHA256,
        "device": {
            "device_id": 0,
            "backend": "hip",
            "name": "AMD Radeon Graphics",
            "gcn_arch_name": "gfx1201",
            "compute_major": 12,
            "compute_minor": 0,
            "total_global_mem": 34_208_743_424,
        },
        "kv_cache_bytes": module.KV_CACHE_BYTES,
        "cache_blocks": module.CACHE_BLOCKS,
        "context_tokens": module.CONTEXT_TOKENS,
        "environment": environment(module),
        "initial_vram": initial,
        "ttft_cases": cases,
        "decode_case": decode,
        "final_vram": final,
        "final_snapshot": ready_snapshot(module),
    }


def write_document(path: Path, value) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


def replace_raw(capture: dict, raw_key: str, sha_key: str, value) -> None:
    raw = compact_json(value)
    capture[raw_key] = raw
    capture[sha_key] = raw_sha(raw)


class Sq8ServingPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def validate(self, path: Path, prefill_mode=None):
        if prefill_mode is None:
            return self.module.validate_result(path, GIT_COMMIT, BINARY_SHA256)
        return self.module.validate_result(path, GIT_COMMIT, BINARY_SHA256, prefill_mode)

    def assert_rejected(self, mutate, pattern: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance.json"
            document = valid_document(self.module)
            mutate(document)
            write_document(path, document)
            with self.assertRaisesRegex(self.module.ValidationError, pattern):
                self.validate(path)

    def test_recomputes_all_metrics_without_producer_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance.json"
            write_document(path, valid_document(self.module))
            result = self.validate(path)
            self.assertTrue(result["passed"])
            self.assertEqual(
                [item["prompt_tokens"] for item in result["ttft_cases"]],
                [32, 128, 512, 2048, 3584],
            )
            self.assertAlmostEqual(result["decode_case"]["p50_tokens_per_second"], 20.0)
            self.assertAlmostEqual(result["decode_case"]["p95_inter_token_seconds"], 0.05)
            self.assertEqual(result["decode_case"]["measured_inter_token_interval_count"], 315)

    def test_recomputes_m128_raw_v2_with_short_prompt_m1_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance-m128.json"
            document = valid_document(self.module, prefill_mode=M128_PREFILL_MODE)
            write_document(path, document)

            result = self.validate(path, M128_PREFILL_MODE)

            self.assertTrue(result["passed"])
            self.assertEqual(document["schema_version"], M128_INPUT_SCHEMA_VERSION)
            self.assertEqual(document["prefill_chunk_tokens"], M128_PREFILL_CHUNK_TOKENS)
            for case in document["ttft_cases"]:
                expected_calls = prompt_execution_calls(
                    case["prompt_tokens"], M128_PREFILL_CHUNK_TOKENS
                )
                for sample in case["samples"]:
                    self.assertEqual(sample["prompt_execution_calls"], expected_calls)
                    self.assertEqual(sample["prompt_progress_events"], expected_calls - 1)
            p32 = document["ttft_cases"][0]["samples"][0]
            self.assertEqual(p32["prompt_execution_calls"], 32)
            self.assertEqual(p32["prompt_progress_events"], 31)
            decode = document["decode_case"]["samples"][0]
            self.assertEqual(decode["prompt_execution_calls"], 32)
            self.assertEqual(decode["prompt_progress_events"], 31)
            self.assertEqual(decode["execution_calls"], 95)

    def test_prefill_mode_strictly_binds_raw_schema_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m8_path = root / "performance-m8.json"
            m128_path = root / "performance-m128.json"
            write_document(m8_path, valid_document(self.module))
            write_document(
                m128_path,
                valid_document(self.module, prefill_mode=M128_PREFILL_MODE),
            )

            with self.assertRaisesRegex(self.module.ValidationError, "schema/model/runtime contract"):
                self.validate(m128_path)
            with self.assertRaisesRegex(self.module.ValidationError, "schema/model/runtime contract"):
                self.validate(m8_path, M128_PREFILL_MODE)

    def test_rejects_m128_short_prompt_call_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance-m128.json"
            document = valid_document(self.module, prefill_mode=M128_PREFILL_MODE)
            document["decode_case"]["samples"][0]["prompt_execution_calls"] = 1
            write_document(path, document)

            with self.assertRaisesRegex(self.module.ValidationError, "decode execution contract"):
                self.validate(path, M128_PREFILL_MODE)

    def test_cli_selects_m128_raw_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance-m128.json"
            write_document(path, valid_document(self.module, prefill_mode=M128_PREFILL_MODE))
            run = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    str(path),
                    "--expected-runner-git-commit",
                    GIT_COMMIT,
                    "--expected-binary-sha256",
                    BINARY_SHA256,
                    "--prefill-mode",
                    M128_PREFILL_MODE,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("passed=true", run.stdout)

    def test_rejects_dirty_build(self):
        self.assert_rejected(
            lambda value: value.__setitem__("runner_worktree_clean", False),
            "clean runner build identity",
        )

    def test_rejects_timer_or_vram_policy_drift(self):
        self.assert_rejected(
            lambda value: value.__setitem__("vram_policy", "stability_gate"),
            "schema/model/runtime contract",
        )
        self.assert_rejected(
            lambda value: value["timer"].__setitem__("model_load_included", True),
            "timer",
        )

    def test_rejects_environment_raw_hash_tampering(self):
        self.assert_rejected(
            lambda value: value["environment"].__setitem__("amd_smi_version_raw", "changed"),
            "version raw SHA-256",
        )

    def test_rejects_environment_gpu_identity_tampering(self):
        def mutate(value):
            env = value["environment"]
            replace_raw(
                env,
                "amd_smi_list_raw_json",
                "amd_smi_list_raw_sha256",
                [{"gpu": 2, "bdf": "bad", "uuid": self.module.R9700_UUID, "kfd_id": 51545}],
            )

        self.assert_rejected(mutate, "frozen R9700 environment")

    def test_rejects_ttft_prompt_hash_tampering(self):
        self.assert_rejected(
            lambda value: value["ttft_cases"][2].__setitem__(
                "prompt_token_ids_u32_le_sha256", "0" * 64
            ),
            "TTFT case contract",
        )

    def test_rejects_ttft_phase_order_tampering(self):
        self.assert_rejected(
            lambda value: value["ttft_cases"][0]["samples"][0].__setitem__(
                "phase", "measured"
            ),
            "TTFT execution contract",
        )

    def test_rejects_ttft_formula_tampering(self):
        self.assert_rejected(
            lambda value: value["ttft_cases"][0]["samples"][3].__setitem__("ttft_ns", 7),
            "TTFT execution contract",
        )

    def test_rejects_cancel_timestamp_tampering(self):
        def mutate(value):
            sample = value["ttft_cases"][0]["samples"][0]
            sample["cancel_set_elapsed_ns"] = sample["first_token_elapsed_ns"] - 1

        self.assert_rejected(mutate, "not strictly increasing")

    def test_rejects_active_snapshot_cache_drift(self):
        self.assert_rejected(
            lambda value: value["ttft_cases"][1]["samples"][2]["first_token_snapshot"][
                "cache_lengths"
            ].__setitem__(19, 1),
            "snapshot contract",
        )

    def test_rejects_vram_raw_hash_tampering(self):
        self.assert_rejected(
            lambda value: value["ttft_cases"][0]["samples"][0]["vram_after_reset"].__setitem__(
                "amd_smi_process_raw_json", "[]"
            ),
            "raw SHA-256",
        )

    def test_rejects_vram_raw_pid_tampering(self):
        def mutate(value):
            capture = value["initial_vram"]
            replace_raw(
                capture,
                "amd_smi_process_raw_json",
                "amd_smi_process_raw_sha256",
                [
                    {
                        "gpu": 2,
                        "process_list": [
                            {
                                "process_info": {
                                    "pid": 999,
                                    "mem_usage": {"value": VRAM_BYTES, "unit": "B"},
                                }
                            }
                        ],
                    }
                ],
            )

        self.assert_rejected(mutate, "VRAM identity differs")

    def test_rejects_unrelated_positive_kfd_process(self):
        def mutate(value):
            capture = value["initial_vram"]
            capture["kfd_processes"].append({"pid": 9999, "vram_bytes": 1})

        self.assert_rejected(mutate, "isolated KFD VRAM ownership")

    def test_rejects_metric_raw_summary_tampering(self):
        self.assert_rejected(
            lambda value: value["decode_case"]["metric_before"].__setitem__(
                "gfx_clock_mhz", 1.0
            ),
            "differs from raw AMD SMI JSON",
        )

    def test_rejects_metric_unit_tampering(self):
        def mutate(value):
            metric = value["ttft_cases"][0]["metric_before"]
            raw = json.loads(metric["raw_json"])
            raw["gpu_data"][0]["power"]["socket_power"]["unit"] = "mW"
            replace_raw(metric, "raw_json", "raw_sha256", raw)

        self.assert_rejected(mutate, "unit must be W")

    def test_rejects_decode_token_transition_tampering(self):
        self.assert_rejected(
            lambda value: value["decode_case"]["samples"][3]["generated"][17].__setitem__(
                "cache_len", 1
            ),
            "token/cache transition differs",
        )

    def test_rejects_decode_availability_tampering(self):
        def mutate(value):
            generated = value["decode_case"]["samples"][0]["generated"]
            generated[5]["available_elapsed_ns"] = generated[4]["available_elapsed_ns"]

        self.assert_rejected(mutate, "not strictly increasing")

    def test_rejects_equal_command_timestamps(self):
        def mutate(value):
            metric = value["ttft_cases"][0]["metric_before"]
            metric["command_end_elapsed_ns"] = metric["command_start_elapsed_ns"]
            metric["captured_elapsed_ns"] = metric["command_start_elapsed_ns"]

        self.assert_rejected(mutate, "not strictly increasing")

    def test_rejects_ttft_eos_first_token(self):
        self.assert_rejected(
            lambda value: value["ttft_cases"][0]["samples"][0].__setitem__(
                "first_token_id", self.module.EOS_TOKEN_IDS[0]
            ),
            "TTFT execution contract",
        )

    def test_rejects_decode_eos_token(self):
        self.assert_rejected(
            lambda value: value["decode_case"]["samples"][0]["generated"][7].__setitem__(
                "token_id", self.module.EOS_TOKEN_IDS[1]
            ),
            "token/cache transition differs",
        )

    def test_rejects_decode_release_tampering(self):
        self.assert_rejected(
            lambda value: value["decode_case"]["samples"][0].__setitem__(
                "release_reset_complete", False
            ),
            "decode execution contract",
        )

    def test_ttft_gate_failure_writes_output_and_cli_exits_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "performance.json"
            output = root / "validation.json"
            write_document(
                evidence,
                valid_document(self.module, ttft_overrides={32: 2_800_000_000}),
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    str(evidence),
                    "--expected-runner-git-commit",
                    GIT_COMMIT,
                    "--expected-binary-sha256",
                    BINARY_SHA256,
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 1, run.stderr)
            validation = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(validation["passed"])
            self.assertTrue(any("prompt 32 TTFT p50" in item for item in validation["gate_errors"]))

    def test_decode_gate_failure_returns_all_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance.json"
            write_document(path, valid_document(self.module, decode_interval_ns=110_000_000))
            result = self.validate(path)
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["decode_case"]["measured_tokens_per_second"]), 5)
            self.assertTrue(any("decode" in item for item in result["gate_errors"]))

    def test_percentile_uses_linear_rank_interpolation(self):
        self.assertEqual(self.module.percentile([0.0, 1.0, 2.0, 3.0, 4.0], 0.50), 2.0)
        self.assertAlmostEqual(
            self.module.percentile([0.0, 1.0, 2.0, 3.0, 4.0], 0.95), 3.8
        )

    def test_rejects_symlink_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            link = root / "evidence.json"
            write_document(target, valid_document(self.module))
            os.symlink(target, link)
            with self.assertRaisesRegex(self.module.ValidationError, "regular file"):
                self.validate(link)


if __name__ == "__main__":
    unittest.main()
