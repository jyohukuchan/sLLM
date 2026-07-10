#!/usr/bin/env python3
"""Validate the fixed Qwen3-14B SQ8 M=8, G=8 generation result."""

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


SCHEMA_VERSION = "ullm.sq8.generation.v1"
TRUSTED_RESULT_SHA256 = "cafd46e09d7f42e95dc021fc5d1a45e2dc54ab78f8f2afabfe261dac4971be04"
PROMPT_TOKEN_IDS = list(range(1, 9))
PROMPT_POSITION_IDS = list(range(8))
GENERATED_TOKEN_IDS = [353, 10, 4999, 1725, 15, 16, 17, 18]
GENERATED_TOKEN_IDS_SHA256 = "58af80297882940ac9695b0f425dac6c768a495e7e02d96c5eda79d921793fd6"
LAYERS = 40
HIDDEN_SIZE = 5120
HEAD_DIM = 128
VOCAB_SIZE = 151936
MAX_NEW_TOKENS = 8
EOS_TOKEN_ID = 151645
TOP_K = 10
UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024
MAX_RELATIVE_L2 = 0.20
MIN_COSINE_SIMILARITY = 0.98
MIN_TOP_10_OVERLAP = 3
EXPECTED_LAYER_NORMS_CANONICAL_SHA256 = (
    "5e44e9cc6f75e1dccf0578331397f223c58eda385d43bab40eb75c569e944a6a"
)
EXPECTED_VLLM_ORACLE_CANONICAL_SHA256 = (
    "39c990e22e80eda207097c65fa59e7b4c4c0dcb70c65add0c81f087429a8bcdb"
)
EXPECTED_SOURCE = {
    "artifact_content_sha256": "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147",
    "artifact_config_sha256": "c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793",
    "artifact_index_sha256": "6a9c8e17744118347080916d8f673b881941cf42989ee77266b14dc2062a7151",
    "package_manifest_sha256": "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb",
    "vllm_oracle_metadata_sha256": "5fc03a28cd15409e84a7fd23fd51c0cbd6ec9cf8761a66d1f5ede7ddfe3226a0",
    "model_revision": "9a283b4a5efbc09ce247e0ae5b02b744739e525a",
}
EXPECTED_PAYLOAD_HASHES = {
    "embedding_payload_sha256": "720549f9f5bae3c8a6520d03f8f07ad79ff9a1885d639d6a9e09afc8550cc0f5",
    "final_norm_payload_sha256": "00e8c9b2696bc683b0a9b2403aecf41ba3555875b1483829b2f154ca563b6e2c",
    "lm_head_payload_sha256": "8488b7e7753b3ba708efa81b2426feb67196c793eee1fd226dfd14f0915b4b6f",
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
EXPECTED_COUNTERS = {
    "embedding_gather_calls": 15,
    "prompt_embedding_d2d_copies": 8,
    "stack_input_d2d_copies": 8,
    "projection_calls": 2240,
    "activation_quantizations": 1280,
    "layer_d2d_copies": 320,
    "kv_write_calls": 600,
    "paged_attention_calls": 280,
    "model_head_calls": 8,
    "model_head_d2d_copies": 1,
    "result_readback_count": 30,
    "execution_synchronization_count": 17,
    "scheduler_prefill_completions": 1,
    "scheduler_prefill_token_records": 1,
    "scheduler_decode_advances": 7,
    "scheduler_release_calls": 1,
    "identity_check_count": 3,
}
EXPECTED_HIP_GUARDS = {
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
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


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


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_result(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        fail(f"result file does not exist: {path}")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        fail(f"failed to read result JSON: {error}")
    result_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except ValidationError:
        raise
    except json.JSONDecodeError as error:
        fail(f"failed to parse result JSON: {error}")
    if not isinstance(value, dict):
        fail("result must contain a JSON object")
    return value, result_sha256


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


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        fail(f"{label} must be finite")
    return parsed


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
    return [integer(item, f"{label}[{index}]") for index, item in enumerate(raw)]


def numbers_match(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def gate_verdict(
    metrics: dict[str, Any], max_relative_l2: float, min_cosine: float
) -> bool:
    return (
        metrics["nonfinite_count"] == 0
        and math.isfinite(float(metrics["relative_l2"]))
        and float(metrics["relative_l2"]) <= max_relative_l2
        and math.isfinite(float(metrics["cosine_similarity"]))
        and float(metrics["cosine_similarity"]) >= min_cosine
    )


def validate_health(value: Any, elements: int, label: str) -> dict[str, Any]:
    health = exact_keys(
        value,
        {"elements", "nonfinite", "minimum", "maximum", "max_abs", "f32_le_sha256"},
        label,
    )
    if integer(health["elements"], f"{label}.elements") != elements:
        fail(f"{label}.elements is invalid")
    if integer(health["nonfinite"], f"{label}.nonfinite") != 0:
        fail(f"{label} reports non-finite values")
    minimum = finite_number(health["minimum"], f"{label}.minimum")
    maximum = finite_number(health["maximum"], f"{label}.maximum")
    max_abs = finite_number(health["max_abs"], f"{label}.max_abs")
    if minimum > maximum or max_abs < 0.0:
        fail(f"{label} range is invalid")
    if not math.isclose(
        max_abs, max(abs(minimum), abs(maximum)), rel_tol=0.0, abs_tol=1e-6
    ):
        fail(f"{label}.max_abs does not match minimum/maximum")
    sha256_value(health["f32_le_sha256"], f"{label}.f32_le_sha256")
    return health


def validate_metrics(value: Any, elements: int, label: str) -> dict[str, Any]:
    metrics = exact_keys(
        value,
        {
            "element_count",
            "nonfinite_count",
            "mse",
            "max_abs",
            "relative_l2",
            "cosine_similarity",
        },
        label,
    )
    if integer(metrics["element_count"], f"{label}.element_count") != elements:
        fail(f"{label}.element_count is invalid")
    if integer(metrics["nonfinite_count"], f"{label}.nonfinite_count") != 0:
        fail(f"{label} reports non-finite values")
    mse = finite_number(metrics["mse"], f"{label}.mse")
    max_abs = finite_number(metrics["max_abs"], f"{label}.max_abs")
    relative_l2 = finite_number(metrics["relative_l2"], f"{label}.relative_l2")
    cosine = finite_number(metrics["cosine_similarity"], f"{label}.cosine_similarity")
    if mse < 0.0 or max_abs < 0.0 or relative_l2 < 0.0:
        fail(f"{label} contains a negative error metric")
    if cosine < -1.0000001 or cosine > 1.0000001:
        fail(f"{label}.cosine_similarity is outside [-1, 1]")
    return metrics


def validate_gate(value: Any, elements: int, label: str) -> bool:
    gate = exact_keys(
        value,
        {"metrics", "max_relative_l2", "min_cosine_similarity", "passed"},
        label,
    )
    metrics = validate_metrics(gate["metrics"], elements, f"{label}.metrics")
    max_relative_l2 = finite_number(
        gate["max_relative_l2"], f"{label}.max_relative_l2"
    )
    min_cosine = finite_number(
        gate["min_cosine_similarity"], f"{label}.min_cosine_similarity"
    )
    if max_relative_l2 != MAX_RELATIVE_L2 or min_cosine != MIN_COSINE_SIMILARITY:
        fail(f"{label} thresholds differ from the fixed contract")
    derived = gate_verdict(metrics, max_relative_l2, min_cosine)
    if boolean(gate["passed"], f"{label}.passed") is not derived:
        fail(f"{label}.passed does not match its metrics")
    if not derived:
        fail(f"{label} misses its numerical gate")
    return derived


def validate_top_10(value: Any, label: str) -> list[tuple[int, float]]:
    entries = exact_list(value, TOP_K, label)
    parsed: list[tuple[int, float]] = []
    seen: set[int] = set()
    for rank, raw_entry in enumerate(entries):
        entry = exact_keys(raw_entry, {"token_id", "logit"}, f"{label}[{rank}]")
        token_id = integer(entry["token_id"], f"{label}[{rank}].token_id")
        logit = finite_number(entry["logit"], f"{label}[{rank}].logit")
        if token_id < 0 or token_id >= VOCAB_SIZE or token_id in seen:
            fail(f"{label}[{rank}] has an invalid or duplicate token ID")
        seen.add(token_id)
        if parsed:
            previous_id, previous_logit = parsed[-1]
            if logit > previous_logit or (
                logit == previous_logit and token_id < previous_id
            ):
                fail(f"{label} is not ordered by descending logit and ascending token ID")
        parsed.append((token_id, logit))
    return parsed


def validate_source(value: Any) -> None:
    source = exact_keys(value, set(EXPECTED_SOURCE), "source")
    for key, expected in EXPECTED_SOURCE.items():
        actual = string(source[key], f"source.{key}")
        if key.endswith("_sha256"):
            sha256_value(actual, f"source.{key}")
        if actual != expected:
            fail(f"source.{key} differs from the fixed source identity")


def validate_input(value: Any) -> None:
    input_data = exact_keys(
        value,
        {
            "prompt_token_ids",
            "prompt_position_ids",
            "max_new_tokens",
            "eos_token_id",
            "sampling",
        },
        "input",
    )
    prompt = integer_list(
        input_data["prompt_token_ids"], len(PROMPT_TOKEN_IDS), "input.prompt_token_ids"
    )
    positions = integer_list(
        input_data["prompt_position_ids"],
        len(PROMPT_POSITION_IDS),
        "input.prompt_position_ids",
    )
    if prompt != PROMPT_TOKEN_IDS or positions != PROMPT_POSITION_IDS:
        fail("input prompt tokens or positions differ from the fixed contract")
    if integer(input_data["max_new_tokens"], "input.max_new_tokens") != MAX_NEW_TOKENS:
        fail("input.max_new_tokens differs from the fixed contract")
    if integer(input_data["eos_token_id"], "input.eos_token_id") != EOS_TOKEN_ID:
        fail("input.eos_token_id differs from the fixed contract")
    if string(input_data["sampling"], "input.sampling") != "greedy_temperature_zero":
        fail("input.sampling differs from the fixed greedy contract")


def expected_payload_chunks(payload_bytes: int) -> int:
    return (payload_bytes + UPLOAD_CHUNK_BYTES - 1) // UPLOAD_CHUNK_BYTES


def validate_payload_record(
    value: Any, expected_name: str, expected_shape: list[int], label: str
) -> None:
    record = exact_keys(
        value,
        {
            "tensor_name",
            "dtype",
            "shape",
            "elements",
            "payload_bytes",
            "payload_sha256",
            "verified_chunks",
        },
        label,
    )
    if string(record["tensor_name"], f"{label}.tensor_name") != expected_name:
        fail(f"{label}.tensor_name is invalid")
    if string(record["dtype"], f"{label}.dtype") != "BF16":
        fail(f"{label}.dtype is invalid")
    shape = integer_list(record["shape"], len(expected_shape), f"{label}.shape")
    if shape != expected_shape:
        fail(f"{label}.shape is invalid")
    elements = math.prod(expected_shape)
    payload_bytes = elements * 2
    if integer(record["elements"], f"{label}.elements") != elements:
        fail(f"{label}.elements does not match shape")
    if integer(record["payload_bytes"], f"{label}.payload_bytes") != payload_bytes:
        fail(f"{label}.payload_bytes does not match BF16 shape")
    sha256_value(record["payload_sha256"], f"{label}.payload_sha256")
    if (
        integer(record["verified_chunks"], f"{label}.verified_chunks")
        != expected_payload_chunks(payload_bytes)
    ):
        fail(f"{label}.verified_chunks does not cover its payload exactly")


def validate_payloads(value: Any) -> None:
    payloads = exact_keys(
        value,
        {
            "layer_count",
            "layer_norm_tensor_count",
            "layer_norms",
            "embedding_payload_sha256",
            "final_norm_payload_sha256",
            "lm_head_payload_sha256",
            "upload_chunk_bytes",
        },
        "payloads",
    )
    if integer(payloads["layer_count"], "payloads.layer_count") != LAYERS:
        fail("payloads.layer_count differs from the fixed contract")
    if (
        integer(payloads["layer_norm_tensor_count"], "payloads.layer_norm_tensor_count")
        != LAYERS * 4
    ):
        fail("payloads.layer_norm_tensor_count differs from the fixed contract")
    norms = exact_list(payloads["layer_norms"], LAYERS * 4, "payloads.layer_norms")
    suffixes = [
        ("input_layernorm.weight", [HIDDEN_SIZE]),
        ("post_attention_layernorm.weight", [HIDDEN_SIZE]),
        ("self_attn.q_norm.weight", [HEAD_DIM]),
        ("self_attn.k_norm.weight", [HEAD_DIM]),
    ]
    for layer_index in range(LAYERS):
        for within_layer, (suffix, shape) in enumerate(suffixes):
            record_index = layer_index * 4 + within_layer
            validate_payload_record(
                norms[record_index],
                f"model.layers.{layer_index}.{suffix}",
                shape,
                f"payloads.layer_norms[{record_index}]",
            )
    if canonical_sha256(norms) != EXPECTED_LAYER_NORMS_CANONICAL_SHA256:
        fail("payloads.layer_norms differ from the fixed verified package payloads")
    for key, expected in EXPECTED_PAYLOAD_HASHES.items():
        actual = sha256_value(payloads[key], f"payloads.{key}")
        if actual != expected:
            fail(f"payloads.{key} differs from the fixed verified payload")
    if integer(payloads["upload_chunk_bytes"], "payloads.upload_chunk_bytes") != UPLOAD_CHUNK_BYTES:
        fail("payloads.upload_chunk_bytes differs from the fixed contract")


def validate_device(value: Any) -> None:
    device = exact_keys(value, set(EXPECTED_DEVICE), "device")
    for key, expected in EXPECTED_DEVICE.items():
        if isinstance(expected, int):
            actual: Any = integer(device[key], f"device.{key}")
        else:
            actual = string(device[key], f"device.{key}")
        if actual != expected:
            fail(f"device.{key} differs from the isolated R9700 HIP identity")


def u32_le_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def validate_generation(value: Any) -> list[int]:
    generation = exact_keys(
        value,
        {
            "request_id",
            "generated_token_ids",
            "expected_generated_token_ids",
            "generated_token_ids_u32_le_sha256",
            "decode_input_token_ids",
            "decode_positions",
            "completion_reason",
            "final_kv_len",
            "released_kv_blocks",
            "allocation_released",
            "feedback_verified",
            "exact_token_sequence",
        },
        "generation",
    )
    if integer(generation["request_id"], "generation.request_id") != 1:
        fail("generation.request_id differs from the fixed request")
    generated = integer_list(
        generation["generated_token_ids"], MAX_NEW_TOKENS, "generation.generated_token_ids"
    )
    expected = integer_list(
        generation["expected_generated_token_ids"],
        MAX_NEW_TOKENS,
        "generation.expected_generated_token_ids",
    )
    if generated != GENERATED_TOKEN_IDS or expected != GENERATED_TOKEN_IDS:
        fail("generation token sequence differs from the fixed vLLM oracle")
    token_hash = sha256_value(
        generation["generated_token_ids_u32_le_sha256"],
        "generation.generated_token_ids_u32_le_sha256",
    )
    if token_hash != GENERATED_TOKEN_IDS_SHA256 or token_hash != u32_le_sha256(generated):
        fail("generation token SHA-256 does not match the generated token bytes")
    decode_inputs = integer_list(
        generation["decode_input_token_ids"],
        MAX_NEW_TOKENS - 1,
        "generation.decode_input_token_ids",
    )
    decode_positions = integer_list(
        generation["decode_positions"],
        MAX_NEW_TOKENS - 1,
        "generation.decode_positions",
    )
    expected_positions = list(range(len(PROMPT_TOKEN_IDS), len(PROMPT_TOKEN_IDS) + 7))
    derived_feedback = (
        decode_inputs == generated[:-1] and decode_positions == expected_positions
    )
    if not derived_feedback:
        fail("generation decode inputs or positions do not prove token feedback")
    if string(generation["completion_reason"], "generation.completion_reason") != "max_new_tokens":
        fail("generation.completion_reason differs from the fixed contract")
    final_kv_len = len(PROMPT_TOKEN_IDS) + len(decode_inputs)
    if integer(generation["final_kv_len"], "generation.final_kv_len") != final_kv_len:
        fail("generation.final_kv_len is inconsistent with prompt and decode writes")
    if integer(generation["released_kv_blocks"], "generation.released_kv_blocks") != 1:
        fail("generation.released_kv_blocks differs from the fixed allocation")
    for key, derived in {
        "allocation_released": True,
        "feedback_verified": derived_feedback,
        "exact_token_sequence": generated == expected == GENERATED_TOKEN_IDS,
    }.items():
        if boolean(generation[key], f"generation.{key}") is not derived:
            fail(f"generation.{key} is inconsistent")
    return generated


def validate_steps(value: Any, generated: list[int]) -> list[tuple[int, int, int]]:
    steps = exact_list(value, MAX_NEW_TOKENS, "steps")
    timings: list[tuple[int, int, int]] = []
    oracle_records: list[dict[str, Any]] = []
    for step_index, raw_step in enumerate(steps):
        label = f"steps[{step_index}]"
        step = exact_keys(
            raw_step,
            {
                "step_index",
                "phase",
                "input_token_id",
                "cache_position",
                "cache_len_after",
                "output_token_id",
                "expected_output_token_id",
                "output_logit",
                "device_final_hidden_health",
                "vllm_final_hidden_health",
                "device_logits_health",
                "vllm_logits_health",
                "vllm_final_hidden_sha256",
                "vllm_logits_sha256",
                "final_hidden",
                "logits",
                "device_top_10",
                "vllm_top_10",
                "top_1_exact",
                "top_10_overlap",
                "minimum_top_10_overlap",
                "started_at_ns",
                "completed_at_ns",
                "latency_ns",
                "passed",
            },
            label,
        )
        if integer(step["step_index"], f"{label}.step_index") != step_index:
            fail(f"{label} is out of order")
        expected_phase = "prefill" if step_index == 0 else "decode"
        if string(step["phase"], f"{label}.phase") != expected_phase:
            fail(f"{label}.phase is invalid")
        if step_index == 0:
            if step["input_token_id"] is not None or step["cache_position"] is not None:
                fail(f"{label} prefill input token and cache position must be null")
            expected_cache_len = len(PROMPT_TOKEN_IDS)
        else:
            if integer(step["input_token_id"], f"{label}.input_token_id") != generated[step_index - 1]:
                fail(f"{label}.input_token_id does not feed back the previous output")
            expected_position = len(PROMPT_TOKEN_IDS) + step_index - 1
            if integer(step["cache_position"], f"{label}.cache_position") != expected_position:
                fail(f"{label}.cache_position is invalid")
            expected_cache_len = expected_position + 1
        if integer(step["cache_len_after"], f"{label}.cache_len_after") != expected_cache_len:
            fail(f"{label}.cache_len_after is invalid")
        output_token_id = integer(step["output_token_id"], f"{label}.output_token_id")
        expected_output = integer(
            step["expected_output_token_id"], f"{label}.expected_output_token_id"
        )
        if output_token_id != generated[step_index] or expected_output != generated[step_index]:
            fail(f"{label} output token differs from the fixed generation sequence")
        output_logit = finite_number(step["output_logit"], f"{label}.output_logit")

        device_hidden_health = validate_health(
            step["device_final_hidden_health"], HIDDEN_SIZE, f"{label}.device_final_hidden_health"
        )
        vllm_hidden_health = validate_health(
            step["vllm_final_hidden_health"], HIDDEN_SIZE, f"{label}.vllm_final_hidden_health"
        )
        device_logits_health = validate_health(
            step["device_logits_health"], VOCAB_SIZE, f"{label}.device_logits_health"
        )
        vllm_logits_health = validate_health(
            step["vllm_logits_health"], VOCAB_SIZE, f"{label}.vllm_logits_health"
        )
        vllm_hidden_hash = sha256_value(
            step["vllm_final_hidden_sha256"], f"{label}.vllm_final_hidden_sha256"
        )
        vllm_logits_hash = sha256_value(
            step["vllm_logits_sha256"], f"{label}.vllm_logits_sha256"
        )
        if vllm_hidden_hash != vllm_hidden_health["f32_le_sha256"]:
            fail(f"{label} vLLM final-hidden hashes disagree")
        if vllm_logits_hash != vllm_logits_health["f32_le_sha256"]:
            fail(f"{label} vLLM logits hashes disagree")
        hidden_passed = validate_gate(step["final_hidden"], HIDDEN_SIZE, f"{label}.final_hidden")
        logits_passed = validate_gate(step["logits"], VOCAB_SIZE, f"{label}.logits")

        device_top = validate_top_10(step["device_top_10"], f"{label}.device_top_10")
        vllm_top = validate_top_10(step["vllm_top_10"], f"{label}.vllm_top_10")
        if not math.isclose(device_top[0][1], output_logit, rel_tol=0.0, abs_tol=1e-6):
            fail(f"{label}.output_logit does not match device top-1")
        if not math.isclose(
            device_top[0][1],
            finite_number(device_logits_health["maximum"], f"{label} device maximum"),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            fail(f"{label} device top-1 does not match logits maximum")
        derived_top_1 = (
            device_top[0][0] == vllm_top[0][0] == output_token_id == expected_output
        )
        if boolean(step["top_1_exact"], f"{label}.top_1_exact") is not derived_top_1:
            fail(f"{label}.top_1_exact is inconsistent")
        overlap = len(
            {token_id for token_id, _ in device_top}
            & {token_id for token_id, _ in vllm_top}
        )
        if integer(step["top_10_overlap"], f"{label}.top_10_overlap") != overlap:
            fail(f"{label}.top_10_overlap is inconsistent")
        minimum_overlap = integer(
            step["minimum_top_10_overlap"], f"{label}.minimum_top_10_overlap"
        )
        if minimum_overlap != MIN_TOP_10_OVERLAP:
            fail(f"{label}.minimum_top_10_overlap differs from the fixed contract")

        started = integer(step["started_at_ns"], f"{label}.started_at_ns")
        completed = integer(step["completed_at_ns"], f"{label}.completed_at_ns")
        latency = integer(step["latency_ns"], f"{label}.latency_ns")
        if started < 0 or completed <= started or latency <= 0:
            fail(f"{label} timing interval must be positive")
        if completed - started != latency:
            fail(f"{label}.latency_ns does not match its endpoints")
        if step_index == 0 and started != 0:
            fail("steps[0].started_at_ns must start the relative request clock at zero")
        if timings and started != timings[-1][1]:
            fail(f"{label}.started_at_ns does not follow the previous step")
        timings.append((started, completed, latency))

        derived_passed = (
            hidden_passed
            and logits_passed
            and derived_top_1
            and overlap >= minimum_overlap
        )
        if boolean(step["passed"], f"{label}.passed") is not derived_passed:
            fail(f"{label}.passed does not match independently derived gates")
        if not derived_passed:
            fail(f"{label} misses its independently derived contract")
        oracle_records.append(
            {
                "expected_output_token_id": expected_output,
                "vllm_final_hidden_health": vllm_hidden_health,
                "vllm_logits_health": vllm_logits_health,
                "vllm_final_hidden_sha256": vllm_hidden_hash,
                "vllm_logits_sha256": vllm_logits_hash,
                "vllm_top_10": step["vllm_top_10"],
            }
        )
        sha256_value(
            device_hidden_health["f32_le_sha256"], f"{label} device hidden hash"
        )
    if canonical_sha256(oracle_records) != EXPECTED_VLLM_ORACLE_CANONICAL_SHA256:
        fail("steps vLLM reference values differ from the fixed trusted oracle")
    return timings


def validate_execution(value: Any, generated: list[int]) -> None:
    execution = exact_keys(
        value,
        {
            "runtime_status_after_load",
            "runtime_status_after_run",
            "profile",
            "stack_steps",
            "counters",
            "final_cache_lengths",
            "generated_token_ids_sha256",
            "required_hip_kernel_env",
            "feedback_verified",
            "allocation_released",
            "fallback_used",
            "host_staging_used",
        },
        "execution",
    )
    expected_strings = {
        "runtime_status_after_load": "ready",
        "runtime_status_after_run": "completed",
        "profile": "rdna4_w8a8_block_ck",
    }
    for key, expected in expected_strings.items():
        if string(execution[key], f"execution.{key}") != expected:
            fail(f"execution.{key} differs from the fixed HIP execution contract")
    stack_steps = exact_list(execution["stack_steps"], MAX_NEW_TOKENS, "execution.stack_steps")
    stack_totals = {
        "projection_calls": 0,
        "activation_quantizations": 0,
        "layer_d2d_copies": 0,
        "kv_write_calls": 0,
        "paged_attention_calls": 0,
        "input_d2d_copies": 0,
    }
    for step_index, raw_step in enumerate(stack_steps):
        label = f"execution.stack_steps[{step_index}]"
        step = exact_keys(
            raw_step,
            {
                "phase",
                "position",
                "sequence_len",
                "cache_len",
                "projection_calls",
                "activation_quantizations",
                "layer_d2d_copies",
                "kv_write_calls",
                "paged_attention_calls",
                "input_d2d_copies",
                "all_ck",
                "fallback_used",
                "host_staging_used",
            },
            label,
        )
        is_prefill = step_index == 0
        expected_values = {
            "phase": "prefill" if is_prefill else "decode",
            "position": 0 if is_prefill else len(PROMPT_TOKEN_IDS) + step_index - 1,
            "sequence_len": len(PROMPT_TOKEN_IDS) if is_prefill else 1,
            "cache_len": len(PROMPT_TOKEN_IDS) + step_index,
            "projection_calls": 280,
            "activation_quantizations": 160,
            "layer_d2d_copies": 40,
            "kv_write_calls": 320 if is_prefill else 40,
            "paged_attention_calls": 0 if is_prefill else 40,
            "input_d2d_copies": 0 if is_prefill else 1,
        }
        if string(step["phase"], f"{label}.phase") != expected_values["phase"]:
            fail(f"{label}.phase is invalid")
        for key in (
            "position",
            "sequence_len",
            "cache_len",
            "projection_calls",
            "activation_quantizations",
            "layer_d2d_copies",
            "kv_write_calls",
            "paged_attention_calls",
            "input_d2d_copies",
        ):
            actual = integer(step[key], f"{label}.{key}")
            if actual != expected_values[key]:
                fail(f"{label}.{key} differs from the fixed stack-step contract")
            if key in stack_totals:
                stack_totals[key] += actual
        for key, expected in {
            "all_ck": True,
            "fallback_used": False,
            "host_staging_used": False,
        }.items():
            if boolean(step[key], f"{label}.{key}") is not expected:
                fail(f"{label}.{key} differs from the HIP-only contract")

    counters = exact_keys(execution["counters"], set(EXPECTED_COUNTERS), "execution.counters")
    for key, expected in EXPECTED_COUNTERS.items():
        if integer(counters[key], f"execution.counters.{key}") != expected:
            fail(f"execution.counters.{key} differs from the fixed generation contract")
    for stack_key, counter_key in {
        "projection_calls": "projection_calls",
        "activation_quantizations": "activation_quantizations",
        "layer_d2d_copies": "layer_d2d_copies",
        "kv_write_calls": "kv_write_calls",
        "paged_attention_calls": "paged_attention_calls",
    }.items():
        if stack_totals[stack_key] != counters[counter_key]:
            fail(f"execution stack-step totals disagree with counters.{counter_key}")
    cache_lengths = integer_list(
        execution["final_cache_lengths"], LAYERS, "execution.final_cache_lengths"
    )
    if cache_lengths != [15] * LAYERS:
        fail("execution.final_cache_lengths must contain 15 for all 40 layers")
    generated_hash = sha256_value(
        execution["generated_token_ids_sha256"], "execution.generated_token_ids_sha256"
    )
    if generated_hash != GENERATED_TOKEN_IDS_SHA256 or generated_hash != u32_le_sha256(generated):
        fail("execution generated-token hash disagrees with generation")
    guards = exact_list(
        execution["required_hip_kernel_env"],
        len(EXPECTED_HIP_GUARDS),
        "execution.required_hip_kernel_env",
    )
    parsed_guards = [
        string(guard, f"execution.required_hip_kernel_env[{index}]")
        for index, guard in enumerate(guards)
    ]
    if len(set(parsed_guards)) != len(parsed_guards) or set(parsed_guards) != EXPECTED_HIP_GUARDS:
        fail("execution.required_hip_kernel_env differs from the exact HIP guard set")
    for key, expected in {
        "feedback_verified": True,
        "allocation_released": True,
        "fallback_used": False,
        "host_staging_used": False,
    }.items():
        if boolean(execution[key], f"execution.{key}") is not expected:
            fail(f"execution.{key} differs from the fixed execution contract")


def validate_timing(value: Any, step_timings: list[tuple[int, int, int]]) -> int:
    timing = exact_keys(
        value,
        {
            "request_count",
            "prompt_tokens",
            "generated_tokens",
            "time_to_first_token_ns",
            "request_latency_ns",
            "decode_elapsed_ns",
            "requests_per_second",
            "generated_tokens_per_second",
            "total_tokens_per_second",
            "decode_tokens_per_second",
        },
        "timing",
    )
    expected_counts = {
        "request_count": 1,
        "prompt_tokens": len(PROMPT_TOKEN_IDS),
        "generated_tokens": len(GENERATED_TOKEN_IDS),
    }
    for key, expected in expected_counts.items():
        if integer(timing[key], f"timing.{key}") != expected:
            fail(f"timing.{key} differs from the measured workload")
    time_to_first = step_timings[0][2]
    request_latency = step_timings[-1][1] - step_timings[0][0]
    decode_elapsed = sum(interval[2] for interval in step_timings[1:])
    expected_ns = {
        "time_to_first_token_ns": time_to_first,
        "request_latency_ns": request_latency,
        "decode_elapsed_ns": decode_elapsed,
    }
    for key, expected in expected_ns.items():
        actual = integer(timing[key], f"timing.{key}")
        if actual != expected or actual <= 0:
            fail(f"timing.{key} does not match step timing endpoints")
    if request_latency != time_to_first + decode_elapsed:
        fail("timing request interval does not partition into prefill and decode")
    expected_rates = {
        "requests_per_second": 1e9 / request_latency,
        "generated_tokens_per_second": len(GENERATED_TOKEN_IDS) * 1e9 / request_latency,
        "total_tokens_per_second": (
            len(PROMPT_TOKEN_IDS) + len(GENERATED_TOKEN_IDS)
        )
        * 1e9
        / request_latency,
        "decode_tokens_per_second": (len(GENERATED_TOKEN_IDS) - 1)
        * 1e9
        / decode_elapsed,
    }
    for key, expected in expected_rates.items():
        actual = finite_number(timing[key], f"timing.{key}")
        if actual <= 0.0 or not numbers_match(actual, expected):
            fail(f"timing.{key} does not match its nanosecond counters")
    return request_latency


def validate_allocator(value: Any) -> None:
    allocator = exact_keys(value, {"before", "after_release"}, "allocator")
    expected_snapshot = {
        "block_size_tokens": 16,
        "total_blocks": 1,
        "free_blocks": 1,
        "allocated_blocks": 0,
        "free_runs": 1,
        "largest_free_run": 1,
    }
    for snapshot_name in ("before", "after_release"):
        label = f"allocator.{snapshot_name}"
        snapshot = exact_keys(allocator[snapshot_name], set(expected_snapshot), label)
        parsed: dict[str, int] = {}
        for key, expected in expected_snapshot.items():
            parsed[key] = integer(snapshot[key], f"{label}.{key}")
            if parsed[key] != expected:
                fail(f"{label}.{key} differs from the fully released allocator state")
        if parsed["free_blocks"] + parsed["allocated_blocks"] != parsed["total_blocks"]:
            fail(f"{label} block accounting is inconsistent")
        if parsed["largest_free_run"] > parsed["free_blocks"]:
            fail(f"{label}.largest_free_run exceeds free blocks")


def validate(path: Path, contract_only: bool = False) -> dict[str, Any]:
    result, result_sha256 = load_result(path)
    trusted = not contract_only
    if trusted and result_sha256 != TRUSTED_RESULT_SHA256:
        fail(
            "result SHA-256 does not match the promotion trust anchor; "
            "use --contract-only only for an untrusted self-consistency rerun"
        )
    exact_keys(
        result,
        {
            "schema_version",
            "passed",
            "source",
            "input",
            "payloads",
            "device",
            "generation",
            "steps",
            "execution",
            "timing",
            "allocator",
        },
        "result",
    )
    if string(result["schema_version"], "schema_version") != SCHEMA_VERSION:
        fail("schema_version is invalid")
    validate_source(result["source"])
    validate_input(result["input"])
    validate_payloads(result["payloads"])
    validate_device(result["device"])
    generated = validate_generation(result["generation"])
    step_timings = validate_steps(result["steps"], generated)
    validate_execution(result["execution"], generated)
    request_latency_ns = validate_timing(result["timing"], step_timings)
    validate_allocator(result["allocator"])
    derived_passed = True
    if boolean(result["passed"], "passed") is not derived_passed:
        fail("top-level passed does not match independently derived gates")
    return {
        "trusted": trusted,
        "mode": "promotion" if trusted else "contract-only",
        "request_latency_ns": request_latency_ns,
        "first_token": generated[0],
        "last_token": generated[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="validate an untrusted self-consistent rerun without the fixed result SHA-256",
    )
    parser.add_argument("result", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate(
        args.result.expanduser().resolve(), contract_only=args.contract_only
    )
    print(
        "passed=true "
        f"mode={summary['mode']} trusted={str(summary['trusted']).lower()} "
        f"steps={MAX_NEW_TOKENS} prompt_tokens={len(PROMPT_TOKEN_IDS)} "
        f"generated_tokens={len(GENERATED_TOKEN_IDS)} final_kv_len=15 "
        f"projections={EXPECTED_COUNTERS['projection_calls']} "
        f"activation_quantizations={EXPECTED_COUNTERS['activation_quantizations']} "
        f"first_token={summary['first_token']} last_token={summary['last_token']} "
        f"request_latency_ms={summary['request_latency_ns'] / 1e6:.6f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
