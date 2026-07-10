#!/usr/bin/env python3
"""Validate the fixed Qwen3-14B SQ8 M=8 full-model promotion result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ullm.sq8.full_model.v1"
TRUSTED_RESULT_SHA256 = "bcc7355ed9deed102ae620bf237466d2aebd78b897dd776fcb088377d4241647"
SEQUENCE_LEN = 8
HIDDEN_SIZE = 5120
HEAD_DIM = 128
VOCAB_SIZE = 151936
INTERMEDIATE_SIZE = 17408
LAYERS = 40
PROJECTIONS = 280
ACTIVATION_QUANTIZATIONS = 160
TOP_K = 10
UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024
EXPECTED_PAYLOADS_CANONICAL_SHA256 = (
    "301ec735327d78529c917ae1d5aac111c462cab993a0343b64c4150411a52e2e"
)
EXPECTED_VLLM_LAYER_HASH_LIST_SHA256 = (
    "d5ed858af2c40abf50ea23b8a09be8f36b90c32c3ef3b2b79541a08e9f1f8981"
)
EXPECTED_SOURCE_SCALARS = {
    "artifact_content_sha256": "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147",
    "artifact_config_sha256": "c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793",
    "artifact_index_sha256": "6a9c8e17744118347080916d8f673b881941cf42989ee77266b14dc2062a7151",
    "package_manifest_sha256": "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb",
    "vllm_oracle_metadata_sha256": "5caafcd2c976482dd01e51b537593d8924d381a8a9ab076b2082325e22fea39e",
    "model_revision": "9a283b4a5efbc09ce247e0ae5b02b744739e525a",
    "vllm_final_hidden_sha256": "a6772963cee66d8429eaa7b4e72e2594345b1a6613a06a1bf67660b4f02aa9a7",
    "vllm_logits_sha256": "24c93f3fbe0fc3d2a101c782f0e181be1206cabd56e900814a608d2a09fd268e",
}
EXPECTED_INPUT = {
    "sequence_len": 8,
    "token_ids": list(range(1, 9)),
    "position_ids": list(range(8)),
    "embedding_tensor": "model.embed_tokens.weight",
    "selected_embedding_f32_le_sha256": (
        "3504c3cd7d4aa4893b49085b065893d541af453bdf6d6a7cf654e3190329436b"
    ),
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
EXPECTED_VLLM_TOP_10 = [
    (353, 13.8125),
    (3764, 11.875),
    (25010, 10.8125),
    (220, 10.0625),
    (5572, 9.5625),
    (671, 9.375),
    (3014, 9.125),
    (374, 9.0625),
    (262, 8.9375),
    (16, 8.8125),
]
EXPECTED_DISPATCH_COUNTS = {
    "mem_v1_default_tile_16x128x128": 160,
    "mem_v1_default_tile_16x128x256": 40,
    "mem_v1_kpadding_tile_16x128x256": 80,
}
EXPECTED_GUARDS = [
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
]
EXPECTED_VRAM_COMPONENTS = {
    "artifact_weight_and_scale_bytes": 13213670400,
    "layer_norm_f32_bytes": 1679360,
    "shared_stack_workspace_bytes": 3989504,
    "resident_stack_hidden_bytes": 163840,
    "model_head_resident_bytes": 1556493824,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"failed to load result JSON: {error}")
    if not isinstance(value, dict):
        fail("result must contain a JSON object")
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    if set(value) != expected:
        fail(
            f"{label} keys differ: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
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
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite")
    return result


def boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{label} must be boolean")
    return value


def sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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
    expected_max_abs = max(abs(minimum), abs(maximum))
    if not math.isclose(max_abs, expected_max_abs, rel_tol=0.0, abs_tol=1e-6):
        fail(f"{label}.max_abs does not match minimum/maximum")
    sha256_value(health["f32_le_sha256"], f"{label}.f32_le_sha256")
    return health


def validate_gate(
    value: Any,
    elements: int,
    expected_max_relative_l2: float,
    expected_min_cosine: float,
    label: str,
) -> bool:
    gate = exact_keys(
        value,
        {"metrics", "max_relative_l2", "min_cosine", "passed"},
        label,
    )
    metrics = validate_metrics(gate["metrics"], elements, f"{label}.metrics")
    max_relative_l2 = finite_number(
        gate["max_relative_l2"], f"{label}.max_relative_l2"
    )
    min_cosine = finite_number(gate["min_cosine"], f"{label}.min_cosine")
    if max_relative_l2 != expected_max_relative_l2 or min_cosine != expected_min_cosine:
        fail(f"{label} thresholds differ from the fixed contract")
    derived = gate_verdict(metrics, max_relative_l2, min_cosine)
    if boolean(gate["passed"], f"{label}.passed") is not derived:
        fail(f"{label}.passed does not match its metrics")
    if not derived:
        fail(f"{label} misses its numerical gate")
    return derived


def validate_source(value: Any) -> list[str]:
    source = exact_keys(
        value,
        set(EXPECTED_SOURCE_SCALARS) | {"vllm_layer_output_sha256"},
        "source",
    )
    for key, expected in EXPECTED_SOURCE_SCALARS.items():
        if source[key] != expected:
            fail(f"source.{key} differs from the fixed source identity")
    layers = exact_list(
        source["vllm_layer_output_sha256"], LAYERS, "source.vllm_layer_output_sha256"
    )
    for index, digest in enumerate(layers):
        sha256_value(digest, f"source.vllm_layer_output_sha256[{index}]")
    if canonical_sha256(layers) != EXPECTED_VLLM_LAYER_HASH_LIST_SHA256:
        fail("source vLLM layer hash list differs from the trusted oracle")
    return layers


def validate_input(value: Any) -> None:
    exact_keys(value, set(EXPECTED_INPUT), "input")
    if value != EXPECTED_INPUT:
        fail("input differs from the fixed M=8 token and embedding contract")


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
    if (
        record["tensor_name"] != expected_name
        or record["dtype"] != "BF16"
        or record["shape"] != expected_shape
    ):
        fail(f"{label} tensor identity, dtype, or shape is invalid")
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


def validate_payloads(value: Any) -> tuple[int, int]:
    payloads = exact_keys(value, {"embedding", "layer_norms", "final_norm", "lm_head"}, "payloads")
    validate_payload_record(
        payloads["embedding"],
        "model.embed_tokens.weight",
        [VOCAB_SIZE, HIDDEN_SIZE],
        "payloads.embedding",
    )
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
    validate_payload_record(
        payloads["final_norm"], "model.norm.weight", [HIDDEN_SIZE], "payloads.final_norm"
    )
    validate_payload_record(
        payloads["lm_head"],
        "lm_head.weight",
        [VOCAB_SIZE, HIDDEN_SIZE],
        "payloads.lm_head",
    )
    if canonical_sha256(payloads) != EXPECTED_PAYLOADS_CANONICAL_SHA256:
        fail("payload identities differ from the fixed verified package payloads")
    resident_checks = len(norms) + 2
    all_records = resident_checks + 1
    if resident_checks != 162 or all_records != 163:
        fail("internal payload accounting is inconsistent")
    return resident_checks, all_records


def validate_device(value: Any) -> None:
    exact_keys(value, set(EXPECTED_DEVICE), "device")
    if value != EXPECTED_DEVICE:
        fail("device differs from the isolated R9700 HIP identity")


def validate_cpu_oracle(value: Any) -> list[str]:
    oracle = exact_keys(value, {"elapsed_ms", "layer_output_f32_le_sha256"}, "cpu_oracle")
    if finite_number(oracle["elapsed_ms"], "cpu_oracle.elapsed_ms") <= 0.0:
        fail("cpu_oracle.elapsed_ms must be positive")
    hashes = exact_list(
        oracle["layer_output_f32_le_sha256"], LAYERS, "cpu_oracle.layer_output_f32_le_sha256"
    )
    for index, digest in enumerate(hashes):
        sha256_value(digest, f"cpu_oracle.layer_output_f32_le_sha256[{index}]")
    if len(set(hashes)) != LAYERS:
        fail("CPU oracle layer hashes must be unique")
    return hashes


def validate_layers(
    value: Any, cpu_hashes: list[str], vllm_hashes: list[str]
) -> bool:
    layers = exact_list(value, LAYERS, "layer_boundaries")
    all_passed = True
    for layer_index, raw_layer in enumerate(layers):
        label = f"layer_boundaries[{layer_index}]"
        layer = exact_keys(
            raw_layer,
            {
                "layer_index",
                "optimized_health",
                "cpu_sq8_f32_le_sha256",
                "vllm_f32_le_sha256",
                "optimized_vs_cpu_sq8",
                "optimized_vs_vllm",
            },
            label,
        )
        if integer(layer["layer_index"], f"{label}.layer_index") != layer_index:
            fail(f"{label} is out of order")
        validate_health(layer["optimized_health"], SEQUENCE_LEN * HIDDEN_SIZE, f"{label}.optimized_health")
        if layer["cpu_sq8_f32_le_sha256"] != cpu_hashes[layer_index]:
            fail(f"{label} CPU oracle hash does not match cpu_oracle")
        if layer["vllm_f32_le_sha256"] != vllm_hashes[layer_index]:
            fail(f"{label} vLLM hash does not match source")
        max_relative_l2 = 0.08 if layer_index == LAYERS - 1 else 0.10
        min_cosine = 0.997 if layer_index == LAYERS - 1 else 0.995
        all_passed &= validate_gate(
            layer["optimized_vs_cpu_sq8"],
            SEQUENCE_LEN * HIDDEN_SIZE,
            max_relative_l2,
            min_cosine,
            f"{label}.optimized_vs_cpu_sq8",
        )
        validate_metrics(
            layer["optimized_vs_vllm"],
            SEQUENCE_LEN * HIDDEN_SIZE,
            f"{label}.optimized_vs_vllm",
        )
    return all_passed


def validate_topk(value: Any, label: str) -> list[tuple[int, float]]:
    entries = exact_list(value, TOP_K, label)
    parsed: list[tuple[int, float]] = []
    seen = set()
    for rank, raw_entry in enumerate(entries):
        entry = exact_keys(raw_entry, {"token_id", "logit"}, f"{label}[{rank}]")
        token_id = integer(entry["token_id"], f"{label}[{rank}].token_id")
        logit = finite_number(entry["logit"], f"{label}[{rank}].logit")
        if token_id < 0 or token_id >= VOCAB_SIZE or token_id in seen:
            fail(f"{label}[{rank}] has an invalid or duplicate token ID")
        seen.add(token_id)
        if parsed:
            previous_id, previous_logit = parsed[-1]
            if logit > previous_logit or (logit == previous_logit and token_id < previous_id):
                fail(f"{label} is not ordered by descending logit and ascending token ID")
        parsed.append((token_id, logit))
    return parsed


def validate_final_head(value: Any) -> bool:
    final = exact_keys(
        value,
        {
            "resident_validation_layer39_matches_audit_bits",
            "device_final_hidden_health",
            "device_logits_health",
            "cpu_final_hidden_f32_le_sha256",
            "cpu_logits_f32_le_sha256",
            "device_vs_cpu_final_hidden",
            "device_vs_cpu_logits",
            "device_vs_vllm_final_hidden",
            "device_vs_vllm_logits",
            "device_top_10",
            "cpu_top_10",
            "vllm_top_10",
            "device_top_1",
            "cpu_top_1",
            "vllm_top_1",
            "device_vllm_top_10_overlap",
            "top_1_contract_passed",
            "passed",
        },
        "final_head",
    )
    resident_match = boolean(
        final["resident_validation_layer39_matches_audit_bits"],
        "final_head.resident_validation_layer39_matches_audit_bits",
    )
    device_hidden_health = validate_health(
        final["device_final_hidden_health"], HIDDEN_SIZE, "final_head.device_final_hidden_health"
    )
    device_logits_health = validate_health(
        final["device_logits_health"], VOCAB_SIZE, "final_head.device_logits_health"
    )
    sha256_value(final["cpu_final_hidden_f32_le_sha256"], "final_head.cpu_final_hidden_f32_le_sha256")
    sha256_value(final["cpu_logits_f32_le_sha256"], "final_head.cpu_logits_f32_le_sha256")
    gates = [
        validate_gate(
            final["device_vs_cpu_final_hidden"],
            HIDDEN_SIZE,
            0.002,
            0.999999,
            "final_head.device_vs_cpu_final_hidden",
        ),
        validate_gate(
            final["device_vs_cpu_logits"],
            VOCAB_SIZE,
            0.002,
            0.999999,
            "final_head.device_vs_cpu_logits",
        ),
        validate_gate(
            final["device_vs_vllm_final_hidden"],
            HIDDEN_SIZE,
            0.15,
            0.99,
            "final_head.device_vs_vllm_final_hidden",
        ),
        validate_gate(
            final["device_vs_vllm_logits"],
            VOCAB_SIZE,
            0.15,
            0.99,
            "final_head.device_vs_vllm_logits",
        ),
    ]
    device_top = validate_topk(final["device_top_10"], "final_head.device_top_10")
    cpu_top = validate_topk(final["cpu_top_10"], "final_head.cpu_top_10")
    vllm_top = validate_topk(final["vllm_top_10"], "final_head.vllm_top_10")
    if vllm_top != EXPECTED_VLLM_TOP_10:
        fail("final_head.vllm_top_10 differs from the trusted vLLM oracle")
    if not math.isclose(
        device_top[0][1],
        finite_number(device_logits_health["maximum"], "device logits maximum"),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        fail("device top-1 logit does not match device logits maximum")
    device_top_1 = integer(final["device_top_1"], "final_head.device_top_1")
    cpu_top_1 = integer(final["cpu_top_1"], "final_head.cpu_top_1")
    vllm_top_1 = integer(final["vllm_top_1"], "final_head.vllm_top_1")
    if device_top_1 != device_top[0][0] or cpu_top_1 != cpu_top[0][0] or vllm_top_1 != vllm_top[0][0]:
        fail("final_head top-1 fields do not match their top-10 lists")
    derived_top_1 = device_top_1 == cpu_top_1 == vllm_top_1 == 353
    if boolean(final["top_1_contract_passed"], "final_head.top_1_contract_passed") is not derived_top_1:
        fail("final_head.top_1_contract_passed is inconsistent")
    overlap = len({token_id for token_id, _ in device_top} & {token_id for token_id, _ in vllm_top})
    if integer(final["device_vllm_top_10_overlap"], "final_head.device_vllm_top_10_overlap") != overlap:
        fail("final_head.device_vllm_top_10_overlap is inconsistent")
    derived_passed = resident_match and all(gates) and derived_top_1 and overlap >= 5
    if boolean(final["passed"], "final_head.passed") is not derived_passed:
        fail("final_head.passed does not match independently derived gates")
    if not derived_passed:
        fail("final_head misses its independently derived contract")
    sha256_value(device_hidden_health["f32_le_sha256"], "device final hidden hash")
    return derived_passed


def validate_timing_summary(value: Any, repeats: int, label: str) -> list[float]:
    summary = exact_keys(value, {"samples_ms", "p50_ms", "p95_ms"}, label)
    raw_samples = exact_list(summary["samples_ms"], repeats, f"{label}.samples_ms")
    samples = [finite_number(sample, f"{label}.samples_ms[{index}]") for index, sample in enumerate(raw_samples)]
    if any(sample <= 0.0 for sample in samples):
        fail(f"{label} samples must be positive")
    for key, fraction in (("p50_ms", 0.50), ("p95_ms", 0.95)):
        reported = finite_number(summary[key], f"{label}.{key}")
        expected = percentile(samples, fraction)
        if not math.isclose(reported, expected, rel_tol=0.0, abs_tol=1e-12):
            fail(f"{label}.{key} does not match its samples")
    return samples


def validate_timing(value: Any) -> tuple[int, int, float]:
    timing = exact_keys(
        value,
        {
            "warmups",
            "repeats",
            "input_upload_excluded",
            "inter_stage_host_validation_excluded",
            "stack_final_synchronization_included",
            "head_readback_decode_and_validation_included",
            "full_stack_and_head",
            "stack",
            "head",
        },
        "timing",
    )
    warmups = integer(timing["warmups"], "timing.warmups")
    repeats = integer(timing["repeats"], "timing.repeats")
    if warmups != 3 or repeats != 10:
        fail("timing warmup/repeat counts differ from the fixed contract")
    flags = {
        "input_upload_excluded": True,
        "inter_stage_host_validation_excluded": True,
        "stack_final_synchronization_included": True,
        "head_readback_decode_and_validation_included": True,
    }
    for key, expected in flags.items():
        if boolean(timing[key], f"timing.{key}") is not expected:
            fail(f"timing.{key} has the wrong measurement semantics")
    full = validate_timing_summary(timing["full_stack_and_head"], repeats, "timing.full_stack_and_head")
    stack = validate_timing_summary(timing["stack"], repeats, "timing.stack")
    head = validate_timing_summary(timing["head"], repeats, "timing.head")
    for index, (full_ms, stack_ms, head_ms) in enumerate(zip(full, stack, head, strict=True)):
        overhead_ms = full_ms - stack_ms - head_ms
        if overhead_ms < -1e-9 or overhead_ms > 1.0:
            fail(f"timing sample {index} is inconsistent with stack plus head")
    p50 = finite_number(timing["full_stack_and_head"]["p50_ms"], "timing full p50")
    return warmups, repeats, p50


def validate_execution(value: Any, warmups: int, repeats: int) -> bool:
    execution = exact_keys(
        value,
        {
            "sequence_len",
            "stack_invocations_per_timed_sample",
            "layers",
            "projections",
            "activation_quantizations",
            "layer_d2d_copies",
            "stack_execution_synchronizations",
            "head_d2d_copies",
            "head_rmsnorm_calls",
            "head_bf16_matvec_calls",
            "head_result_readbacks",
            "head_execution_synchronizations",
            "timed_path_fallback_used",
            "timed_path_host_staging_used",
            "layerwise_audit_is_non_timed",
            "layerwise_audit_host_staging_used",
            "layerwise_audit_readbacks",
            "fresh_input_uploads_for_validation_and_timing",
            "input_ready_state_checks",
            "output_ready_state_checks",
            "validated_stack_reports",
            "validated_head_reports",
            "timed_output_hash_stability_checks",
            "dispatch_implementation_counts",
            "required_hip_kernel_env",
            "passed",
        },
        "execution",
    )
    expected_runs = 1 + warmups + repeats
    expected_integers = {
        "sequence_len": SEQUENCE_LEN,
        "stack_invocations_per_timed_sample": 1,
        "layers": LAYERS,
        "projections": PROJECTIONS,
        "activation_quantizations": ACTIVATION_QUANTIZATIONS,
        "layer_d2d_copies": LAYERS,
        "stack_execution_synchronizations": 1,
        "head_d2d_copies": 1,
        "head_rmsnorm_calls": 1,
        "head_bf16_matvec_calls": 1,
        "head_result_readbacks": 2,
        "head_execution_synchronizations": 1,
        "layerwise_audit_readbacks": LAYERS,
        "fresh_input_uploads_for_validation_and_timing": expected_runs,
        "input_ready_state_checks": expected_runs,
        "output_ready_state_checks": expected_runs,
        "validated_stack_reports": expected_runs,
        "validated_head_reports": expected_runs,
        "timed_output_hash_stability_checks": warmups + repeats,
    }
    for key, expected in expected_integers.items():
        if integer(execution[key], f"execution.{key}") != expected:
            fail(f"execution.{key} differs from the derived contract")
    expected_flags = {
        "timed_path_fallback_used": False,
        "timed_path_host_staging_used": False,
        "layerwise_audit_is_non_timed": True,
        "layerwise_audit_host_staging_used": True,
    }
    for key, expected in expected_flags.items():
        if boolean(execution[key], f"execution.{key}") is not expected:
            fail(f"execution.{key} differs from the derived contract")
    dispatch = exact_keys(
        execution["dispatch_implementation_counts"],
        set(EXPECTED_DISPATCH_COUNTS),
        "execution.dispatch_implementation_counts",
    )
    parsed_dispatch = {
        key: integer(value, f"execution.dispatch_implementation_counts.{key}")
        for key, value in dispatch.items()
    }
    if parsed_dispatch != EXPECTED_DISPATCH_COUNTS or sum(parsed_dispatch.values()) != PROJECTIONS:
        fail("execution dispatch counts do not cover exactly 280 projections")
    if execution["required_hip_kernel_env"] != EXPECTED_GUARDS:
        fail("execution.required_hip_kernel_env differs from the HIP-only contract")
    derived_passed = True
    if boolean(execution["passed"], "execution.passed") is not derived_passed:
        fail("execution.passed does not match independently checked counters")
    return derived_passed


def validate_vram(value: Any, device_total: int) -> bool:
    vram = exact_keys(
        value,
        {
            "device_total_global_mem",
            "artifact_weight_and_scale_bytes",
            "layer_norm_f32_bytes",
            "shared_stack_workspace_bytes",
            "resident_stack_hidden_bytes",
            "model_head_resident_bytes",
            "minimum_accounted_resident_bytes",
            "unaccounted_device_bytes",
            "excludes_allocator_and_backend_overhead",
            "fits_device",
        },
        "vram",
    )
    if integer(vram["device_total_global_mem"], "vram.device_total_global_mem") != device_total:
        fail("vram.device_total_global_mem does not match device")
    components = {}
    for key, expected in EXPECTED_VRAM_COMPONENTS.items():
        components[key] = integer(vram[key], f"vram.{key}")
        if components[key] != expected:
            fail(f"vram.{key} differs from the independently derived component")
    minimum = sum(components.values())
    if integer(vram["minimum_accounted_resident_bytes"], "vram.minimum_accounted_resident_bytes") != minimum:
        fail("vram.minimum_accounted_resident_bytes is not the component sum")
    unaccounted = max(device_total - minimum, 0)
    if integer(vram["unaccounted_device_bytes"], "vram.unaccounted_device_bytes") != unaccounted:
        fail("vram.unaccounted_device_bytes is inconsistent")
    if boolean(
        vram["excludes_allocator_and_backend_overhead"],
        "vram.excludes_allocator_and_backend_overhead",
    ) is not True:
        fail("vram must state that allocator/backend overhead is excluded")
    derived_fit = minimum < device_total
    if boolean(vram["fits_device"], "vram.fits_device") is not derived_fit:
        fail("vram.fits_device does not match the independent fit calculation")
    if not derived_fit:
        fail("minimum accounted residency does not fit the device")
    return derived_fit


def validate(path: Path, contract_only: bool = False) -> dict[str, Any]:
    if not path.is_file():
        fail(f"result file does not exist: {path}")
    try:
        result_sha256 = sha256_file(path)
    except OSError as error:
        fail(f"failed to hash result: {error}")
    trusted = not contract_only
    if trusted and result_sha256 != TRUSTED_RESULT_SHA256:
        fail(
            "result SHA-256 does not match the promotion trust anchor; "
            "use --contract-only only for an untrusted self-consistency rerun"
        )
    result = load_json(path)
    exact_keys(
        result,
        {
            "schema_version",
            "passed",
            "source",
            "input",
            "payloads",
            "device",
            "cpu_oracle",
            "layer_boundaries",
            "final_head",
            "execution",
            "timing",
            "vram",
        },
        "result",
    )
    if result["schema_version"] != SCHEMA_VERSION:
        fail("schema_version is invalid")
    vllm_hashes = validate_source(result["source"])
    validate_input(result["input"])
    resident_payloads, payload_records = validate_payloads(result["payloads"])
    validate_device(result["device"])
    cpu_hashes = validate_cpu_oracle(result["cpu_oracle"])
    layers_passed = validate_layers(result["layer_boundaries"], cpu_hashes, vllm_hashes)
    final_passed = validate_final_head(result["final_head"])
    warmups, repeats, full_p50_ms = validate_timing(result["timing"])
    execution_passed = validate_execution(result["execution"], warmups, repeats)
    vram_fit = validate_vram(result["vram"], EXPECTED_DEVICE["total_global_mem"])
    derived_passed = layers_passed and final_passed and execution_passed and vram_fit
    if boolean(result["passed"], "passed") is not derived_passed:
        fail("top-level passed does not match independently derived gates")
    if not derived_passed:
        fail("full-model result misses an independently derived gate")
    return {
        "trusted": trusted,
        "mode": "promotion" if trusted else "contract-only",
        "resident_payloads": resident_payloads,
        "payload_records": payload_records,
        "full_p50_ms": full_p50_ms,
        "top1": result["final_head"]["device_top_1"],
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
        f"layers={LAYERS} resident_payloads={summary['resident_payloads']} "
        f"payload_records={summary['payload_records']} projections={PROJECTIONS} "
        f"activation_quantizations={ACTIVATION_QUANTIZATIONS} "
        f"hash_stability_checks=13 top1={summary['top1']} "
        f"full_p50_ms={summary['full_p50_ms']:.6f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
