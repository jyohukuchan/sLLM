import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-generation-benchmark.py"
TOKENS = [353, 10, 4999, 1725, 15, 16, 17, 18]


def token_hash(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def u128_distribution(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "max": max(values),
    }


def f64_distribution(values: list[float]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def valid_payload() -> dict:
    samples = []
    for index in range(10):
        reset = 5_000_000 + index * 100_000
        first = 100_000_000 + index * 1_000_000
        decode = 280_000_000 + index * 2_000_000
        runtime = first + decode
        request_call = runtime + 4_000_000 + index * 50_000
        steady_cycle = reset + request_call + 500_000 + index * 10_000
        samples.append(
            {
                "sample_index": index,
                "generated_token_ids": list(TOKENS),
                "token_ids_sha256": token_hash(TOKENS),
                "finish_reason": "length",
                "feedback_verified": True,
                "allocation_released": True,
                "fallback_used": False,
                "host_staging_used": False,
                "reset_latency_ns": reset,
                "request_call_wall_latency_ns": request_call,
                "steady_state_cycle_wall_latency_ns": steady_cycle,
                "runtime_time_to_first_token_ns": first,
                "runtime_request_latency_ns": runtime,
                "runtime_decode_elapsed_ns": decode,
                "runtime_decode_tokens_per_second": 7e9 / decode,
            }
        )
    series = {
        key: [sample[key] for sample in samples]
        for key in (
            "reset_latency_ns",
            "request_call_wall_latency_ns",
            "steady_state_cycle_wall_latency_ns",
            "runtime_time_to_first_token_ns",
            "runtime_request_latency_ns",
            "runtime_decode_elapsed_ns",
            "runtime_decode_tokens_per_second",
        )
    }
    aggregate_cycle = sum(series["steady_state_cycle_wall_latency_ns"])
    aggregate_seconds = float(aggregate_cycle) / 1_000_000_000.0
    return {
        "schema_version": "ullm.sq8.generation_benchmark.v1",
        "passed": True,
        "benchmark_mode": "audited_generation_gate",
        "source": {
            "name": "Qwen/Qwen3-14B-FP8",
            "artifact_content_sha256": (
                "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147"
            ),
            "package_manifest_sha256": (
                "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb"
            ),
            "model_revision": "9a283b4a5efbc09ce247e0ae5b02b744739e525a",
            "promotion_result_sha256": (
                "a9a1a4158a55cbb04a8da411b2dee5f676b149654df88f29926878bdaf9b28e0"
            ),
        },
        "workload": {
            "prompt_token_ids": list(range(1, 9)),
            "prompt_position_ids": list(range(8)),
            "expected_generated_token_ids": list(TOKENS),
            "prompt_tokens": 8,
            "generated_tokens": 8,
            "context_tokens": 16,
            "batch_size": 1,
            "sampling": "greedy_temperature_zero",
            "configured_min_new_tokens": 0,
            "max_new_tokens": 8,
            "ignore_eos": False,
            "early_stop_on_eos": True,
            "eos_token_id": 151645,
            "finish_reason": "length",
            "attention": "causal",
            "bos_inserted": False,
            "chat_template_applied": False,
            "detokenization": False,
        },
        "execution": {
            "device": {
                "runtime_index": 1,
                "backend_device_id": 0,
                "backend": "hip",
                "name": "AMD Radeon Graphics",
                "gcn_arch_name": "",
                "compute_major": 12,
                "compute_minor": 0,
                "total_global_mem": 34208743424,
            },
            "profile": "rdna4_w8a8_block_ck",
            "required_hip_kernel_env": [
                "ULLM_REQUIRE_HIP_ADD_KERNEL",
                "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
                "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
                "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
                "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
                "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
                "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
                "ULLM_REQUIRE_HIP_ROPE_KERNEL",
                "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
                "ULLM_REQUIRE_HIP_TOP1_KERNEL",
            ],
            "hip_visible_devices": "1",
        },
        "measurement_scope": {
            "timer_start": "before_reset_synchronized",
            "timer_end": "after_run_fixed_synchronized_returns",
            "model_load_included": False,
            "cache_reset_included": True,
            "runtime_contract_validation_included": True,
            "full_logits_readback_included": True,
            "final_hidden_readback_included": True,
            "top10_host_scan_included": True,
            "detokenization_included": False,
        },
        "comparison": {
            "scope_caveat_metrics": [
                "steady_state_cycle_wall_latency_ns",
                "requests_per_second",
                "generated_tokens_per_second",
                "total_tokens_per_second",
            ],
            "unavailable_on_vllm": [
                "runtime_time_to_first_token_ns",
                "runtime_decode_elapsed_ns",
                "runtime_decode_tokens_per_second",
            ],
            "production_engine_comparison_eligible": False,
            "interpretation": (
                "same fixed token workload with different timer scopes; uLLM includes "
                "reset and audited full-hidden/full-logits readback, so rates are "
                "diagnostic rather than production-equivalent"
            ),
        },
        "warmups": 3,
        "repeats": 10,
        "promotion_run_counted_as_first_warmup": True,
        "load_excluded_from_primary_throughput": True,
        "reset_excluded_from_primary_throughput": False,
        "samples": samples,
        "aggregate": {
            "percentile_method": "linear_interpolation_rank_(n-1)*p",
            "throughput_denominator": (
                "sum_of_measured_reset_plus_run_fixed_cycle_wall_latencies"
            ),
            "aggregate_measured_cycle_wall_ns": aggregate_cycle,
            "aggregate_measured_seconds": aggregate_seconds,
            "requests_per_second": 10.0 / aggregate_seconds,
            "generated_tokens_per_second": 80.0 / aggregate_seconds,
            "total_tokens_per_second": 160.0 / aggregate_seconds,
            "steady_state_cycle_wall_latency_ns": u128_distribution(
                series["steady_state_cycle_wall_latency_ns"]
            ),
            "reset_latency_ns": u128_distribution(series["reset_latency_ns"]),
            "request_call_wall_latency_ns": u128_distribution(
                series["request_call_wall_latency_ns"]
            ),
            "runtime_time_to_first_token_ns": u128_distribution(
                series["runtime_time_to_first_token_ns"]
            ),
            "runtime_request_latency_ns": u128_distribution(
                series["runtime_request_latency_ns"]
            ),
            "runtime_decode_elapsed_ns": u128_distribution(
                series["runtime_decode_elapsed_ns"]
            ),
            "runtime_decode_tokens_per_second": f64_distribution(
                series["runtime_decode_tokens_per_second"]
            ),
        },
        "exact_tokens_all_samples": True,
        "token_hash_stable": True,
        "feedback_verified_all_samples": True,
        "allocation_released_all_samples": True,
        "fallback_used": False,
        "host_staging_used": False,
    }


class Sq8GenerationBenchmarkValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location(
            "sq8_generation_benchmark_validator", VALIDATOR
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load validator: {VALIDATOR}")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def run_validator(self, path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(VALIDATOR), str(path)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def write_payload(self, root: Path, payload: dict) -> Path:
        path = root / "benchmark.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    def assert_rejects(self, mutate, expected_error: str) -> None:
        payload = valid_payload()
        mutate(payload)
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_validator(self.write_payload(Path(temporary), payload))
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected_error, result.stderr)

    def test_generated_valid_payload_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_validator(
                self.write_payload(Path(temporary), valid_payload())
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "passed=true benchmark_mode=audited_generation_gate warmups=3 repeats=10",
                result.stdout,
            )
            self.assertIn("steady_cycle_p50_ms=", result.stdout)

    def test_percentile_uses_linear_interpolation(self) -> None:
        values = [float(value) for value in range(10)]
        self.assertEqual(self.module.percentile(values, 0.50), 4.5)
        self.assertAlmostEqual(self.module.percentile(values, 0.95), 8.55)

    def test_unknown_top_level_key_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("untrusted", True),
            "benchmark keys differ",
        )

    def test_missing_nested_key_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"]["steady_state_cycle_wall_latency_ns"].pop(
                "mean"
            ),
            "aggregate.steady_state_cycle_wall_latency_ns keys differ",
        )

    def test_unknown_sample_key_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][0].__setitem__("device", "R9700"),
            "samples[0] keys differ",
        )

    def test_duplicate_json_key_is_rejected(self) -> None:
        raw = json.dumps(valid_payload(), indent=2)
        needle = '  "schema_version": "ullm.sq8.generation_benchmark.v1",'
        self.assertIn(needle, raw)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text(raw.replace(needle, needle + "\n" + needle, 1), encoding="utf-8")
            result = self.run_validator(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate JSON key: schema_version", result.stderr)

    def test_nan_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"].__setitem__("requests_per_second", math.nan),
            "non-finite JSON number is forbidden: NaN",
        )

    def test_infinity_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][0].__setitem__(
                "runtime_decode_tokens_per_second", math.inf
            ),
            "non-finite JSON number is forbidden: Infinity",
        )

    def test_boolean_cannot_replace_integer(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][0].__setitem__("sample_index", False),
            "samples[0].sample_index must be an integer",
        )

    def test_integer_cannot_replace_f64(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"].__setitem__("aggregate_measured_seconds", 4),
            "must be a JSON floating-point number",
        )

    def test_benchmark_mode_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("benchmark_mode", "serving_throughput"),
            "benchmark_mode differs",
        )

    def test_source_name_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["source"].__setitem__("name", "Qwen/Qwen3-14B"),
            "source.name differs",
        )

    def test_source_hash_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["source"].__setitem__("artifact_content_sha256", "0" * 64),
            "source.artifact_content_sha256 differs",
        )

    def test_promotion_result_hash_must_be_lowercase_sha256(self) -> None:
        self.assert_rejects(
            lambda value: value["source"].__setitem__(
                "promotion_result_sha256", "not-a-sha256"
            ),
            "source.promotion_result_sha256 must be a lowercase SHA-256 digest",
        )

    def test_zero_promotion_result_hash_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["source"].__setitem__(
                "promotion_result_sha256", "0" * 64
            ),
            "source.promotion_result_sha256 differs",
        )

    def test_other_valid_promotion_result_hash_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["source"].__setitem__(
                "promotion_result_sha256",
                "cafd46e09d7f42e95dc021fc5d1a45e2dc54ab78f8f2afabfe261dac4971be04",
            ),
            "source.promotion_result_sha256 differs",
        )

    def test_workload_position_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["workload"]["prompt_position_ids"].__setitem__(7, 8),
            "workload.prompt_position_ids differs",
        )

    def test_workload_expected_token_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["workload"]["expected_generated_token_ids"].__setitem__(0, 354),
            "workload.expected_generated_token_ids differs",
        )

    def test_workload_count_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["workload"].__setitem__("configured_min_new_tokens", 8),
            "workload.configured_min_new_tokens differs",
        )

    def test_workload_eos_semantics_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["workload"].__setitem__("ignore_eos", True),
            "workload.ignore_eos differs",
        )

    def test_workload_detokenization_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["workload"].__setitem__("detokenization", True),
            "workload.detokenization differs",
        )

    def test_execution_device_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["execution"]["device"].__setitem__(
                "compute_major", 11
            ),
            "execution.device.compute_major differs",
        )

    def test_execution_device_type_is_strict(self) -> None:
        self.assert_rejects(
            lambda value: value["execution"]["device"].__setitem__(
                "runtime_index", True
            ),
            "execution.device.runtime_index must be an integer",
        )

    def test_execution_profile_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["execution"].__setitem__(
                "profile", "reference_w8a8"
            ),
            "execution.profile differs",
        )

    def test_execution_guard_substitution_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["execution"]["required_hip_kernel_env"].__setitem__(
                0, "ULLM_REQUIRE_HIP_FAKE_KERNEL"
            ),
            "execution.required_hip_kernel_env differs",
        )

    def test_execution_guard_count_is_strict(self) -> None:
        self.assert_rejects(
            lambda value: value["execution"]["required_hip_kernel_env"].pop(),
            "must contain exactly 10 entries",
        )

    def test_execution_guard_order_is_strict(self) -> None:
        def mutate(value) -> None:
            guards = value["execution"]["required_hip_kernel_env"]
            guards[0], guards[1] = guards[1], guards[0]

        self.assert_rejects(mutate, "execution.required_hip_kernel_env differs")

    def test_hip_visible_devices_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["execution"].__setitem__("hip_visible_devices", "0"),
            "execution.hip_visible_devices must equal",
        )

    def test_measurement_timer_start_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["measurement_scope"].__setitem__(
                "timer_start", "before_run_fixed"
            ),
            "measurement_scope.timer_start differs",
        )

    def test_measurement_reset_scope_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["measurement_scope"].__setitem__(
                "cache_reset_included", False
            ),
            "measurement_scope.cache_reset_included differs",
        )

    def test_comparison_scope_metric_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["comparison"]["scope_caveat_metrics"].__setitem__(
                0, "request_call_wall_latency_ns"
            ),
            "comparison.scope_caveat_metrics differs",
        )

    def test_comparison_unavailable_metric_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["comparison"]["unavailable_on_vllm"].__setitem__(
                0, "runtime_request_latency_ns"
            ),
            "comparison.unavailable_on_vllm differs",
        )

    def test_comparison_eligibility_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["comparison"].__setitem__(
                "production_engine_comparison_eligible", True
            ),
            "production_engine_comparison_eligible must be false",
        )

    def test_comparison_interpretation_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["comparison"].__setitem__(
                "interpretation", "production serving throughput"
            ),
            "comparison.interpretation differs",
        )

    def test_warmup_count_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("warmups", 2), "warmups differs"
        )

    def test_repeat_count_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("repeats", 9), "repeats differs"
        )

    def test_promotion_warmup_semantics_are_required(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__(
                "promotion_run_counted_as_first_warmup", False
            ),
            "promotion_run_counted_as_first_warmup differs",
        )

    def test_load_exclusion_semantics_are_required(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__(
                "load_excluded_from_primary_throughput", False
            ),
            "load_excluded_from_primary_throughput differs",
        )

    def test_reset_must_not_be_excluded_from_primary_throughput(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__(
                "reset_excluded_from_primary_throughput", True
            ),
            "reset_excluded_from_primary_throughput differs",
        )

    def test_sample_count_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"].pop(),
            "samples must contain exactly 10 entries",
        )

    def test_sample_index_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][4].__setitem__("sample_index", 5),
            "samples[4].sample_index is out of order",
        )

    def test_sample_token_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][2]["generated_token_ids"].__setitem__(3, 1726),
            "samples[2].generated_token_ids differ",
        )

    def test_sample_hash_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][7].__setitem__("token_ids_sha256", "0" * 64),
            "samples[7].token_ids_sha256 does not match",
        )

    def test_sample_finish_reason_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][1].__setitem__("finish_reason", "eos"),
            "samples[1].finish_reason differs",
        )

    def test_sample_feedback_false_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][3].__setitem__("feedback_verified", False),
            "samples[3].feedback_verified differs",
        )

    def test_sample_allocation_not_released_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][3].__setitem__("allocation_released", False),
            "samples[3].allocation_released differs",
        )

    def test_sample_fallback_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][5].__setitem__("fallback_used", True),
            "samples[5].fallback_used differs",
        )

    def test_sample_host_staging_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][6].__setitem__("host_staging_used", True),
            "samples[6].host_staging_used differs",
        )

    def test_cycle_shorter_than_reset_plus_call_is_rejected(self) -> None:
        def mutate(value) -> None:
            sample = value["samples"][0]
            sample["steady_state_cycle_wall_latency_ns"] = (
                sample["reset_latency_ns"] + sample["request_call_wall_latency_ns"] - 1
            )

        self.assert_rejects(mutate, "steady cycle must cover reset plus request-call")

    def test_request_call_shorter_than_runtime_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][0].__setitem__(
                "request_call_wall_latency_ns",
                value["samples"][0]["runtime_request_latency_ns"] - 1,
            ),
            "must satisfy request-call wall >= runtime request >= TTFT",
        )

    def test_runtime_shorter_than_ttft_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][1].__setitem__(
                "runtime_request_latency_ns",
                value["samples"][1]["runtime_time_to_first_token_ns"] - 1,
            ),
            "must satisfy request-call wall >= runtime request >= TTFT",
        )

    def test_decode_difference_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][3].__setitem__(
                "runtime_decode_elapsed_ns",
                value["samples"][3]["runtime_decode_elapsed_ns"] + 1,
            ),
            "must equal runtime minus TTFT",
        )

    def test_sample_decode_rate_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["samples"][5].__setitem__(
                "runtime_decode_tokens_per_second",
                value["samples"][5]["runtime_decode_tokens_per_second"] + 0.01,
            ),
            "samples[5].runtime_decode_tokens_per_second is inconsistent",
        )

    def test_aggregate_cycle_sum_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"].__setitem__(
                "aggregate_measured_cycle_wall_ns",
                value["aggregate"]["aggregate_measured_cycle_wall_ns"] + 1,
            ),
            "does not equal the sample cycle sum",
        )

    def test_aggregate_seconds_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"].__setitem__(
                "aggregate_measured_seconds",
                value["aggregate"]["aggregate_measured_seconds"] + 0.01,
            ),
            "aggregate.aggregate_measured_seconds does not match",
        )

    def test_aggregate_rate_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"].__setitem__(
                "generated_tokens_per_second",
                value["aggregate"]["generated_tokens_per_second"] + 0.01,
            ),
            "aggregate.generated_tokens_per_second does not match",
        )

    def test_throughput_denominator_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"].__setitem__(
                "throughput_denominator", "sum_of_request_call_wall_latencies"
            ),
            "aggregate.throughput_denominator differs",
        )

    def test_distribution_count_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"]["reset_latency_ns"].__setitem__("count", 9),
            "aggregate.reset_latency_ns.count does not match",
        )

    def test_distribution_mean_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"]["request_call_wall_latency_ns"].__setitem__(
                "mean",
                value["aggregate"]["request_call_wall_latency_ns"]["mean"] + 1.0,
            ),
            "mean/percentiles do not match samples",
        )

    def test_distribution_p95_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"]["runtime_request_latency_ns"].__setitem__(
                "p95", value["aggregate"]["runtime_request_latency_ns"]["p95"] + 1.0
            ),
            "mean/percentiles do not match samples",
        )

    def test_f64_distribution_mean_corruption_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value["aggregate"]["runtime_decode_tokens_per_second"].__setitem__(
                "mean",
                value["aggregate"]["runtime_decode_tokens_per_second"]["mean"] + 0.01,
            ),
            "runtime_decode_tokens_per_second.mean does not match",
        )

    def test_aggregate_feedback_claim_false_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("feedback_verified_all_samples", False),
            "feedback_verified_all_samples differs",
        )

    def test_aggregate_exact_token_claim_false_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("exact_tokens_all_samples", False),
            "exact_tokens_all_samples differs",
        )

    def test_aggregate_token_hash_claim_false_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("token_hash_stable", False),
            "token_hash_stable differs",
        )

    def test_aggregate_allocation_claim_false_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("allocation_released_all_samples", False),
            "allocation_released_all_samples differs",
        )

    def test_aggregate_fallback_claim_true_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("fallback_used", True),
            "fallback_used differs",
        )

    def test_aggregate_host_staging_claim_true_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("host_staging_used", True),
            "host_staging_used differs",
        )

    def test_passed_false_is_rejected(self) -> None:
        self.assert_rejects(
            lambda value: value.__setitem__("passed", False),
            "passed does not match independently derived benchmark gates",
        )


if __name__ == "__main__":
    unittest.main()
