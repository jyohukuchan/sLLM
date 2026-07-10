#!/usr/bin/env python3
"""Validate the fixed Qwen3-14B SQ8 audited generation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ullm.sq8.generation_benchmark.v1"
WARMUPS = 3
REPEATS = 10
PROMPT_TOKEN_IDS = list(range(1, 9))
PROMPT_POSITION_IDS = list(range(8))
GENERATED_TOKEN_IDS = [353, 10, 4999, 1725, 15, 16, 17, 18]
PROMPT_TOKENS = 8
GENERATED_TOKENS = 8
DECODE_TOKENS = GENERATED_TOKENS - 1
CONTEXT_TOKENS = PROMPT_TOKENS + GENERATED_TOKENS
EOS_TOKEN_ID = 151645
TOKEN_IDS_SHA256 = "58af80297882940ac9695b0f425dac6c768a495e7e02d96c5eda79d921793fd6"
EXPECTED_SOURCE = {
    "name": "Qwen/Qwen3-14B-FP8",
    "artifact_content_sha256": "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147",
    "package_manifest_sha256": "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb",
    "model_revision": "9a283b4a5efbc09ce247e0ae5b02b744739e525a",
    "promotion_result_sha256": "a9a1a4158a55cbb04a8da411b2dee5f676b149654df88f29926878bdaf9b28e0",
}
EXPECTED_DEVICE = {
    "runtime_index": 1,
    "backend_device_id": 0,
    "backend": "hip",
    "name": "AMD Radeon Graphics",
    "gcn_arch_name": "",
    "compute_major": 12,
    "compute_minor": 0,
    "total_global_mem": 34208743424,
}
EXPECTED_HIP_GUARDS = [
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
]
EXPECTED_MEASUREMENT_SCOPE = {
    "timer_start": "before_reset_synchronized",
    "timer_end": "after_run_fixed_synchronized_returns",
    "model_load_included": False,
    "cache_reset_included": True,
    "runtime_contract_validation_included": True,
    "full_logits_readback_included": True,
    "final_hidden_readback_included": True,
    "top10_host_scan_included": True,
    "detokenization_included": False,
}
EXPECTED_SCOPE_CAVEAT_METRICS = [
    "steady_state_cycle_wall_latency_ns",
    "requests_per_second",
    "generated_tokens_per_second",
    "total_tokens_per_second",
]
EXPECTED_UNAVAILABLE_ON_VLLM = [
    "runtime_time_to_first_token_ns",
    "runtime_decode_elapsed_ns",
    "runtime_decode_tokens_per_second",
]
EXPECTED_INTERPRETATION = (
    "same fixed token workload with different timer scopes; uLLM includes reset and "
    "audited full-hidden/full-logits readback, so rates are diagnostic rather than "
    "production-equivalent"
)
PERCENTILE_METHOD = "linear_interpolation_rank_(n-1)*p"
THROUGHPUT_DENOMINATOR = (
    "sum_of_measured_reset_plus_run_fixed_cycle_wall_latencies"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
U128_MAX = (1 << 128) - 1


class ValidationError(ValueError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite_constant(value: str) -> None:
    fail(f"non-finite JSON number is forbidden: {value}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"benchmark file does not exist: {path}")
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        fail(f"failed to read benchmark JSON: {error}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except ValidationError:
        raise
    except json.JSONDecodeError as error:
        fail(f"failed to parse benchmark JSON: {error}")
    if not isinstance(value, dict):
        fail("benchmark must contain a JSON object")
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        fail(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def exact_list(value: Any, length: int, label: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        fail(f"{label} must contain exactly {length} entries")
    return value


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    return value


def unsigned_integer(value: Any, label: str, maximum: int = U128_MAX) -> int:
    parsed = integer(value, label)
    if parsed < 0 or parsed > maximum:
        fail(f"{label} is outside its unsigned integer range")
    return parsed


def floating(value: Any, label: str) -> float:
    if not isinstance(value, float):
        fail(f"{label} must be a JSON floating-point number")
    if not math.isfinite(value):
        fail(f"{label} must be finite")
    return value


def boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{label} must be boolean")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        fail(f"{label} must be a string")
    return value


def sha256_value(value: Any, label: str) -> str:
    parsed = string(value, label)
    if SHA256_RE.fullmatch(parsed) is None:
        fail(f"{label} must be a lowercase SHA-256 digest")
    return parsed


def integer_list(value: Any, length: int, label: str) -> list[int]:
    raw = exact_list(value, length, label)
    return [unsigned_integer(item, f"{label}[{index}]") for index, item in enumerate(raw)]


def string_list(value: Any, expected: list[str], label: str) -> list[str]:
    raw = exact_list(value, len(expected), label)
    parsed = [string(item, f"{label}[{index}]") for index, item in enumerate(raw)]
    if parsed != expected:
        fail(f"{label} differs from the fixed comparison scope")
    return parsed


def token_ids_sha256(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if value > 0xFFFFFFFF:
            fail(f"token ID does not fit u32: {value}")
        digest.update(struct.pack("<I", value))
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        fail("percentile input is invalid")
    ordered = sorted(values)
    if any(not math.isfinite(value) for value in ordered):
        fail("percentile input contains a non-finite value")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numbers_match(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)


def validate_source(value: Any) -> None:
    source = exact_keys(value, set(EXPECTED_SOURCE), "source")
    for key, expected in EXPECTED_SOURCE.items():
        actual = string(source[key], f"source.{key}")
        if key.endswith("_sha256"):
            sha256_value(actual, f"source.{key}")
        if actual != expected:
            fail(f"source.{key} differs from the fixed source identity")


def validate_execution(value: Any) -> None:
    execution = exact_keys(
        value,
        {"device", "profile", "required_hip_kernel_env", "hip_visible_devices"},
        "execution",
    )
    device = exact_keys(execution["device"], set(EXPECTED_DEVICE), "execution.device")
    for key, expected in EXPECTED_DEVICE.items():
        if isinstance(expected, int):
            actual: Any = integer(device[key], f"execution.device.{key}")
        else:
            actual = string(device[key], f"execution.device.{key}")
        if actual != expected:
            fail(f"execution.device.{key} differs from the isolated R9700 identity")
    if string(execution["profile"], "execution.profile") != "rdna4_w8a8_block_ck":
        fail("execution.profile differs from the fixed CK profile")
    string_list(
        execution["required_hip_kernel_env"],
        EXPECTED_HIP_GUARDS,
        "execution.required_hip_kernel_env",
    )
    if string(execution["hip_visible_devices"], "execution.hip_visible_devices") != "1":
        fail("execution.hip_visible_devices must equal the isolated runtime index 1")


def validate_workload(value: Any) -> None:
    workload = exact_keys(
        value,
        {
            "prompt_token_ids",
            "prompt_position_ids",
            "expected_generated_token_ids",
            "prompt_tokens",
            "generated_tokens",
            "context_tokens",
            "batch_size",
            "sampling",
            "configured_min_new_tokens",
            "max_new_tokens",
            "ignore_eos",
            "early_stop_on_eos",
            "eos_token_id",
            "finish_reason",
            "attention",
            "bos_inserted",
            "chat_template_applied",
            "detokenization",
        },
        "workload",
    )
    expected_lists = {
        "prompt_token_ids": PROMPT_TOKEN_IDS,
        "prompt_position_ids": PROMPT_POSITION_IDS,
        "expected_generated_token_ids": GENERATED_TOKEN_IDS,
    }
    for key, expected in expected_lists.items():
        actual = integer_list(workload[key], len(expected), f"workload.{key}")
        if actual != expected:
            fail(f"workload.{key} differs from the fixed workload")
    expected_integers = {
        "prompt_tokens": PROMPT_TOKENS,
        "generated_tokens": GENERATED_TOKENS,
        "context_tokens": CONTEXT_TOKENS,
        "batch_size": 1,
        "configured_min_new_tokens": 0,
        "max_new_tokens": GENERATED_TOKENS,
        "eos_token_id": EOS_TOKEN_ID,
    }
    for key, expected in expected_integers.items():
        if unsigned_integer(workload[key], f"workload.{key}") != expected:
            fail(f"workload.{key} differs from the fixed workload")
    expected_strings = {
        "sampling": "greedy_temperature_zero",
        "finish_reason": "length",
        "attention": "causal",
    }
    for key, expected in expected_strings.items():
        if string(workload[key], f"workload.{key}") != expected:
            fail(f"workload.{key} differs from the fixed workload semantics")
    expected_booleans = {
        "ignore_eos": False,
        "early_stop_on_eos": True,
        "bos_inserted": False,
        "chat_template_applied": False,
        "detokenization": False,
    }
    for key, expected in expected_booleans.items():
        if boolean(workload[key], f"workload.{key}") is not expected:
            fail(f"workload.{key} differs from the fixed workload semantics")


def validate_measurement_scope(value: Any) -> None:
    scope = exact_keys(value, set(EXPECTED_MEASUREMENT_SCOPE), "measurement_scope")
    for key, expected in EXPECTED_MEASUREMENT_SCOPE.items():
        if isinstance(expected, bool):
            actual: Any = boolean(scope[key], f"measurement_scope.{key}")
        else:
            actual = string(scope[key], f"measurement_scope.{key}")
        if actual != expected:
            fail(f"measurement_scope.{key} differs from the primary timer scope")


def validate_comparison(value: Any) -> None:
    comparison = exact_keys(
        value,
        {
            "scope_caveat_metrics",
            "unavailable_on_vllm",
            "production_engine_comparison_eligible",
            "interpretation",
        },
        "comparison",
    )
    string_list(
        comparison["scope_caveat_metrics"],
        EXPECTED_SCOPE_CAVEAT_METRICS,
        "comparison.scope_caveat_metrics",
    )
    string_list(
        comparison["unavailable_on_vllm"],
        EXPECTED_UNAVAILABLE_ON_VLLM,
        "comparison.unavailable_on_vllm",
    )
    if boolean(
        comparison["production_engine_comparison_eligible"],
        "comparison.production_engine_comparison_eligible",
    ) is not False:
        fail("comparison.production_engine_comparison_eligible must be false")
    if string(comparison["interpretation"], "comparison.interpretation") != EXPECTED_INTERPRETATION:
        fail("comparison.interpretation differs from the audited-run caveat")


def validate_samples(value: Any) -> dict[str, list[Any]]:
    samples = exact_list(value, REPEATS, "samples")
    collected: dict[str, list[Any]] = {
        "reset_latency_ns": [],
        "request_call_wall_latency_ns": [],
        "steady_state_cycle_wall_latency_ns": [],
        "runtime_time_to_first_token_ns": [],
        "runtime_request_latency_ns": [],
        "runtime_decode_elapsed_ns": [],
        "runtime_decode_tokens_per_second": [],
        "exact_token_match": [],
        "token_hash_match": [],
        "feedback_verified": [],
        "allocation_released": [],
        "fallback_used": [],
        "host_staging_used": [],
    }
    expected_token_hash = token_ids_sha256(GENERATED_TOKEN_IDS)
    if expected_token_hash != TOKEN_IDS_SHA256:
        fail("internal fixed token hash is inconsistent")
    for sample_index, raw_sample in enumerate(samples):
        label = f"samples[{sample_index}]"
        sample = exact_keys(
            raw_sample,
            {
                "sample_index",
                "generated_token_ids",
                "token_ids_sha256",
                "finish_reason",
                "feedback_verified",
                "allocation_released",
                "fallback_used",
                "host_staging_used",
                "reset_latency_ns",
                "request_call_wall_latency_ns",
                "steady_state_cycle_wall_latency_ns",
                "runtime_time_to_first_token_ns",
                "runtime_request_latency_ns",
                "runtime_decode_elapsed_ns",
                "runtime_decode_tokens_per_second",
            },
            label,
        )
        if unsigned_integer(sample["sample_index"], f"{label}.sample_index") != sample_index:
            fail(f"{label}.sample_index is out of order")
        tokens = integer_list(
            sample["generated_token_ids"], GENERATED_TOKENS, f"{label}.generated_token_ids"
        )
        exact_token_match = tokens == GENERATED_TOKEN_IDS
        if not exact_token_match:
            fail(f"{label}.generated_token_ids differ from the fixed token sequence")
        token_hash = sha256_value(sample["token_ids_sha256"], f"{label}.token_ids_sha256")
        token_hash_match = (
            token_hash == expected_token_hash and token_hash == token_ids_sha256(tokens)
        )
        if not token_hash_match:
            fail(f"{label}.token_ids_sha256 does not match its token bytes")
        if string(sample["finish_reason"], f"{label}.finish_reason") != "length":
            fail(f"{label}.finish_reason differs from the fixed completion")
        parsed_flags = {
            key: boolean(sample[key], f"{label}.{key}")
            for key in (
                "feedback_verified",
                "allocation_released",
                "fallback_used",
                "host_staging_used",
            )
        }
        expected_flags = {
            "feedback_verified": True,
            "allocation_released": True,
            "fallback_used": False,
            "host_staging_used": False,
        }
        for key, expected in expected_flags.items():
            if parsed_flags[key] is not expected:
                fail(f"{label}.{key} differs from the measured execution contract")

        reset = unsigned_integer(sample["reset_latency_ns"], f"{label}.reset_latency_ns")
        call_wall = unsigned_integer(
            sample["request_call_wall_latency_ns"], f"{label}.request_call_wall_latency_ns"
        )
        cycle_wall = unsigned_integer(
            sample["steady_state_cycle_wall_latency_ns"],
            f"{label}.steady_state_cycle_wall_latency_ns",
        )
        first = unsigned_integer(
            sample["runtime_time_to_first_token_ns"],
            f"{label}.runtime_time_to_first_token_ns",
        )
        runtime = unsigned_integer(
            sample["runtime_request_latency_ns"], f"{label}.runtime_request_latency_ns"
        )
        decode = unsigned_integer(
            sample["runtime_decode_elapsed_ns"], f"{label}.runtime_decode_elapsed_ns"
        )
        if min(reset, call_wall, cycle_wall, first, runtime, decode) <= 0:
            fail(f"{label} latency counters must be positive")
        measured_parts = reset + call_wall
        if measured_parts > U128_MAX:
            fail(f"{label} reset plus request-call timing overflows u128")
        if cycle_wall < measured_parts:
            fail(f"{label} steady cycle must cover reset plus request-call wall time")
        if call_wall < runtime or runtime < first:
            fail(f"{label} must satisfy request-call wall >= runtime request >= TTFT")
        if decode != runtime - first:
            fail(f"{label}.runtime_decode_elapsed_ns must equal runtime minus TTFT")
        decode_tps = floating(
            sample["runtime_decode_tokens_per_second"],
            f"{label}.runtime_decode_tokens_per_second",
        )
        expected_decode_tps = DECODE_TOKENS * 1e9 / decode
        if decode_tps <= 0.0 or not numbers_match(decode_tps, expected_decode_tps):
            fail(f"{label}.runtime_decode_tokens_per_second is inconsistent")

        collected["reset_latency_ns"].append(reset)
        collected["request_call_wall_latency_ns"].append(call_wall)
        collected["steady_state_cycle_wall_latency_ns"].append(cycle_wall)
        collected["runtime_time_to_first_token_ns"].append(first)
        collected["runtime_request_latency_ns"].append(runtime)
        collected["runtime_decode_elapsed_ns"].append(decode)
        collected["runtime_decode_tokens_per_second"].append(decode_tps)
        collected["exact_token_match"].append(exact_token_match)
        collected["token_hash_match"].append(token_hash_match)
        for key, parsed in parsed_flags.items():
            collected[key].append(parsed)
    return collected


def validate_u128_distribution(value: Any, samples: list[int], label: str) -> None:
    distribution = exact_keys(value, {"count", "min", "mean", "p50", "p95", "max"}, label)
    count = unsigned_integer(distribution["count"], f"{label}.count")
    minimum = unsigned_integer(distribution["min"], f"{label}.min")
    maximum = unsigned_integer(distribution["max"], f"{label}.max")
    mean = floating(distribution["mean"], f"{label}.mean")
    p50 = floating(distribution["p50"], f"{label}.p50")
    p95 = floating(distribution["p95"], f"{label}.p95")
    as_f64 = [float(sample) for sample in sorted(samples)]
    expected = {
        "mean": sum(as_f64) / len(as_f64),
        "p50": percentile(as_f64, 0.50),
        "p95": percentile(as_f64, 0.95),
    }
    if count != len(samples) or count != REPEATS:
        fail(f"{label}.count does not match samples")
    if minimum != min(samples) or maximum != max(samples):
        fail(f"{label} min/max do not match samples")
    if not all(numbers_match(actual, expected[key]) for key, actual in (("mean", mean), ("p50", p50), ("p95", p95))):
        fail(f"{label} mean/percentiles do not match samples")


def validate_f64_distribution(value: Any, samples: list[float], label: str) -> None:
    distribution = exact_keys(value, {"count", "min", "mean", "p50", "p95", "max"}, label)
    count = unsigned_integer(distribution["count"], f"{label}.count")
    parsed = {
        key: floating(distribution[key], f"{label}.{key}")
        for key in ("min", "mean", "p50", "p95", "max")
    }
    if count != len(samples) or count != REPEATS:
        fail(f"{label}.count does not match samples")
    if any(item <= 0.0 for item in parsed.values()):
        fail(f"{label} values must be positive")
    ordered = sorted(samples)
    expected = {
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }
    for key, expected_value in expected.items():
        if not numbers_match(parsed[key], expected_value):
            fail(f"{label}.{key} does not match samples")


def validate_aggregate(value: Any, samples: dict[str, list[Any]]) -> int:
    aggregate = exact_keys(
        value,
        {
            "percentile_method",
            "throughput_denominator",
            "aggregate_measured_cycle_wall_ns",
            "aggregate_measured_seconds",
            "requests_per_second",
            "generated_tokens_per_second",
            "total_tokens_per_second",
            "steady_state_cycle_wall_latency_ns",
            "reset_latency_ns",
            "request_call_wall_latency_ns",
            "runtime_time_to_first_token_ns",
            "runtime_request_latency_ns",
            "runtime_decode_elapsed_ns",
            "runtime_decode_tokens_per_second",
        },
        "aggregate",
    )
    if string(aggregate["percentile_method"], "aggregate.percentile_method") != PERCENTILE_METHOD:
        fail("aggregate.percentile_method differs from the fixed interpolation method")
    if string(aggregate["throughput_denominator"], "aggregate.throughput_denominator") != THROUGHPUT_DENOMINATOR:
        fail("aggregate.throughput_denominator differs from the reset-inclusive cycle")
    expected_cycle_wall = sum(samples["steady_state_cycle_wall_latency_ns"])
    if expected_cycle_wall <= 0 or expected_cycle_wall > U128_MAX:
        fail("aggregate measured cycle wall sum is outside u128")
    reported_wall = unsigned_integer(
        aggregate["aggregate_measured_cycle_wall_ns"],
        "aggregate.aggregate_measured_cycle_wall_ns",
    )
    if reported_wall != expected_cycle_wall:
        fail("aggregate.aggregate_measured_cycle_wall_ns does not equal the sample cycle sum")
    expected_seconds = float(expected_cycle_wall) / 1_000_000_000.0
    reported_seconds = floating(
        aggregate["aggregate_measured_seconds"], "aggregate.aggregate_measured_seconds"
    )
    if reported_seconds <= 0.0 or not numbers_match(reported_seconds, expected_seconds):
        fail("aggregate.aggregate_measured_seconds does not match cycle nanoseconds")
    expected_rates = {
        "requests_per_second": float(REPEATS) / expected_seconds,
        "generated_tokens_per_second": float(REPEATS * GENERATED_TOKENS) / expected_seconds,
        "total_tokens_per_second": float(REPEATS * CONTEXT_TOKENS) / expected_seconds,
    }
    for key, expected in expected_rates.items():
        actual = floating(aggregate[key], f"aggregate.{key}")
        if actual <= 0.0 or not numbers_match(actual, expected):
            fail(f"aggregate.{key} does not match the measured cycle wall sum")
    u128_distributions = {
        "steady_state_cycle_wall_latency_ns": "steady_state_cycle_wall_latency_ns",
        "reset_latency_ns": "reset_latency_ns",
        "request_call_wall_latency_ns": "request_call_wall_latency_ns",
        "runtime_time_to_first_token_ns": "runtime_time_to_first_token_ns",
        "runtime_request_latency_ns": "runtime_request_latency_ns",
        "runtime_decode_elapsed_ns": "runtime_decode_elapsed_ns",
    }
    for aggregate_key, samples_key in u128_distributions.items():
        validate_u128_distribution(
            aggregate[aggregate_key], samples[samples_key], f"aggregate.{aggregate_key}"
        )
    validate_f64_distribution(
        aggregate["runtime_decode_tokens_per_second"],
        samples["runtime_decode_tokens_per_second"],
        "aggregate.runtime_decode_tokens_per_second",
    )
    return expected_cycle_wall


def validate(path: Path) -> dict[str, Any]:
    result = load_json(path)
    exact_keys(
        result,
        {
            "schema_version",
            "passed",
            "benchmark_mode",
            "source",
            "workload",
            "execution",
            "measurement_scope",
            "comparison",
            "warmups",
            "repeats",
            "promotion_run_counted_as_first_warmup",
            "load_excluded_from_primary_throughput",
            "reset_excluded_from_primary_throughput",
            "samples",
            "aggregate",
            "exact_tokens_all_samples",
            "token_hash_stable",
            "feedback_verified_all_samples",
            "allocation_released_all_samples",
            "fallback_used",
            "host_staging_used",
        },
        "benchmark",
    )
    if string(result["schema_version"], "schema_version") != SCHEMA_VERSION:
        fail("schema_version is invalid")
    if string(result["benchmark_mode"], "benchmark_mode") != "audited_generation_gate":
        fail("benchmark_mode differs from the audited generation gate")
    validate_source(result["source"])
    validate_workload(result["workload"])
    validate_execution(result["execution"])
    validate_measurement_scope(result["measurement_scope"])
    validate_comparison(result["comparison"])
    if unsigned_integer(result["warmups"], "warmups") != WARMUPS:
        fail("warmups differs from the fixed benchmark contract")
    if unsigned_integer(result["repeats"], "repeats") != REPEATS:
        fail("repeats differs from the fixed benchmark contract")
    expected_top_semantics = {
        "promotion_run_counted_as_first_warmup": True,
        "load_excluded_from_primary_throughput": True,
        "reset_excluded_from_primary_throughput": False,
    }
    for key, expected in expected_top_semantics.items():
        if boolean(result[key], key) is not expected:
            fail(f"{key} differs from the primary benchmark scope")
    samples = validate_samples(result["samples"])
    aggregate_cycle_wall = validate_aggregate(result["aggregate"], samples)
    derived_flags = {
        "exact_tokens_all_samples": all(samples["exact_token_match"]),
        "token_hash_stable": all(samples["token_hash_match"]),
        "feedback_verified_all_samples": all(samples["feedback_verified"]),
        "allocation_released_all_samples": all(samples["allocation_released"]),
        "fallback_used": any(samples["fallback_used"]),
        "host_staging_used": any(samples["host_staging_used"]),
    }
    for key, expected in derived_flags.items():
        if boolean(result[key], key) is not expected:
            fail(f"{key} differs from the independently validated samples")
    derived_passed = (
        derived_flags["exact_tokens_all_samples"]
        and derived_flags["token_hash_stable"]
        and derived_flags["feedback_verified_all_samples"]
        and derived_flags["allocation_released_all_samples"]
        and not derived_flags["fallback_used"]
        and not derived_flags["host_staging_used"]
    )
    if boolean(result["passed"], "passed") is not derived_passed:
        fail("passed does not match independently derived benchmark gates")
    return {
        "aggregate_cycle_wall_ns": aggregate_cycle_wall,
        "generated_tokens_per_second": result["aggregate"]["generated_tokens_per_second"],
        "cycle_p50_ns": result["aggregate"]["steady_state_cycle_wall_latency_ns"]["p50"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate(args.benchmark.expanduser().resolve())
    print(
        "passed=true benchmark_mode=audited_generation_gate "
        f"warmups={WARMUPS} repeats={REPEATS} "
        f"aggregate_cycle_wall_ms={summary['aggregate_cycle_wall_ns'] / 1e6:.6f} "
        f"steady_cycle_p50_ms={summary['cycle_p50_ns'] / 1e6:.6f} "
        f"generated_tokens_per_second={summary['generated_tokens_per_second']:.6f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
