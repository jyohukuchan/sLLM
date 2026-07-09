#!/usr/bin/env python3

import hashlib
import json
import math
import sys
from pathlib import Path


EXPECTED_TENSORS = {
    "input_norm",
    "q_projected",
    "k_projected",
    "v_projected",
    "q_norm",
    "k_norm",
    "q_rope",
    "k_rope",
    "attention",
    "o_projected",
    "attention_residual",
    "post_attention_norm",
    "gate_projected",
    "up_projected",
    "silu_gate_mul_up",
    "down_projected",
    "output",
}
EXPECTED_ACTIVATIONS = {
    "input_norm_qkv": (8, 5120),
    "attention_o": (8, 5120),
    "post_norm_gate_up": (8, 5120),
    "mlp_down": (8, 17408),
}
EXPECTED_DISPATCH = [
    ("q_m8", 8, 5120, 5120, "MemV1DefaultTile16x128x128"),
    ("gate_m8", 8, 17408, 5120, "MemV1KPaddingTile16x128x256"),
    ("gate_m128", 128, 17408, 5120, "MemV1DefaultTile16x256x128"),
    ("down_m8", 8, 5120, 17408, "MemV1DefaultTile16x128x256"),
]
REQUIRED_GUARDS = [
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
]


def fail(message: str) -> None:
    raise ValueError(message)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"failed to load {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} must contain a JSON object")
    return value


def finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        fail(f"{label} must be finite")
    return converted


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_timing(result: dict, name: str, repeats: int) -> None:
    timing = result.get(name)
    if not isinstance(timing, dict):
        fail(f"{name} is missing")
    samples_raw = timing.get("samples_ms")
    if not isinstance(samples_raw, list) or len(samples_raw) != repeats:
        fail(f"{name}.samples_ms must contain {repeats} samples")
    samples = [finite_number(value, f"{name}.samples_ms") for value in samples_raw]
    if any(value <= 0.0 for value in samples):
        fail(f"{name} samples must be positive")
    for key, fraction in (("p50_ms", 0.50), ("p95_ms", 0.95)):
        reported = finite_number(timing.get(key), f"{name}.{key}")
        expected = percentile(samples, fraction)
        if not math.isclose(reported, expected, rel_tol=0.0, abs_tol=1.0e-12):
            fail(f"{name}.{key} does not match its samples")


def validate_tensor_checks(result: dict) -> None:
    checks = result.get("tensor_checks")
    if not isinstance(checks, dict) or set(checks) != EXPECTED_TENSORS:
        fail("tensor_checks does not contain the exact 17-stage set")
    for name, check in checks.items():
        if not isinstance(check, dict):
            fail(f"tensor_checks.{name} must be an object")
        metrics = check.get("metrics")
        if not isinstance(metrics, dict):
            fail(f"tensor_checks.{name}.metrics is missing")
        nonfinite = metrics.get("nonfinite_count")
        relative_l2 = finite_number(metrics.get("relative_l2"), f"{name}.relative_l2")
        cosine = finite_number(metrics.get("cosine_similarity"), f"{name}.cosine")
        max_relative_l2 = finite_number(check.get("max_relative_l2"), f"{name}.limit")
        min_cosine = finite_number(check.get("min_cosine"), f"{name}.min_cosine")
        expected_pass = (
            nonfinite == 0
            and relative_l2 <= max_relative_l2
            and cosine >= min_cosine
        )
        if check.get("passed") is not expected_pass or not expected_pass:
            fail(f"tensor check {name} failed or has an inconsistent verdict")


def validate_activations(result: dict) -> None:
    checks = result.get("activation_checks")
    if not isinstance(checks, dict) or set(checks) != set(EXPECTED_ACTIVATIONS):
        fail("activation_checks does not contain the exact four-stage set")
    for name, (m, k) in EXPECTED_ACTIVATIONS.items():
        check = checks[name]
        if (
            check.get("m") != m
            or check.get("k") != k
            or check.get("encoded_byte_exact") is not True
            or check.get("scale_bit_exact") is not True
            or check.get("passed") is not True
        ):
            fail(f"activation check {name} is not bit exact")


def validate_layer(result: dict) -> None:
    if result.get("schema_version") != "ullm.sq8.layer.v1" or result.get("passed") is not True:
        fail("layer result schema or verdict is invalid")
    if result.get("layer_index") != 0 or result.get("sequence_len") != 8:
        fail("layer result must be layer 0, M=8")
    if result.get("position_offset") != 0:
        fail("isolated layer result must use position_offset=0")
    device = result.get("device", {})
    if (
        device.get("backend") != "hip"
        or device.get("compute_major") != 12
        or device.get("compute_minor") != 0
    ):
        fail("layer result device is not the isolated RDNA4 HIP device")
    contracts = result.get("contracts", {})
    if (
        contracts.get("optimized_profile") != "rdna4_w8a8_block_ck"
        or contracts.get("reference_profile") != "reference_w8a16_block2d"
        or contracts.get("projection_output") != "bf16_rne_then_f32"
        or contracts.get("activation_quantizations") != 4
        or contracts.get("projection_calls") != 7
        or contracts.get("fallback_used") is not False
        or contracts.get("timed_path_host_staging") is not False
        or contracts.get("required_hip_kernel_env") != REQUIRED_GUARDS
    ):
        fail("layer execution contract is incomplete")
    repeats = contracts.get("repeats")
    if not isinstance(repeats, int) or repeats < 3:
        fail("layer result repeat count is invalid")
    validate_timing(result, "optimized_timing", repeats)
    validate_timing(result, "reference_timing", repeats)
    optimized_p50 = finite_number(result["optimized_timing"]["p50_ms"], "optimized p50")
    reference_p50 = finite_number(result["reference_timing"]["p50_ms"], "reference p50")
    speedup = finite_number(result.get("optimized_speedup"), "optimized_speedup")
    if not math.isclose(speedup, reference_p50 / optimized_p50, abs_tol=1.0e-12):
        fail("optimized_speedup does not match p50 timings")
    if speedup <= 1.0:
        fail("optimized layer is not faster than reference")
    validate_tensor_checks(result)
    validate_activations(result)
    health = result.get("optimized_output_health", {})
    if health.get("elements") != 40960 or health.get("nonfinite") != 0:
        fail("optimized output health is invalid")
    oracle = result.get("oracle_trace", {})
    if (
        oracle.get("artifact_content_sha256") != result.get("artifact_content_sha256")
        or oracle.get("sequence_len") != 8
        or oracle.get("output_f32_le_sha256")
        != oracle.get("tensors", {}).get("output_hidden", {}).get("f32_le_sha256")
    ):
        fail("oracle identity or output trace is invalid")


def validate_dispatch(dispatch: dict) -> None:
    if (
        dispatch.get("schema_version") != "ullm.sq8.ck_runtime_dispatch.v1"
        or dispatch.get("device_arch") != "gfx1201"
        or dispatch.get("fallback_used") is not False
        or dispatch.get("passed") is not True
    ):
        fail("dispatch result header is invalid")
    cases = dispatch.get("cases")
    if not isinstance(cases, list) or len(cases) != len(EXPECTED_DISPATCH):
        fail("dispatch result must contain four cases")
    for case, expected in zip(cases, EXPECTED_DISPATCH, strict=True):
        label, m, n, k, implementation = expected
        if (
            case.get("label") != label
            or case.get("m") != m
            or case.get("n") != n
            or case.get("k") != k
            or case.get("implementation") != implementation
            or case.get("quantized_byte_exact") is not True
            or case.get("scale_bit_exact") is not True
            or case.get("nonfinite") != 0
            or case.get("passed") is not True
        ):
            fail(f"dispatch case {label} is invalid")
        if finite_number(case.get("relative_l2"), f"{label}.relative_l2") > 0.005:
            fail(f"dispatch case {label} exceeds relative L2 gate")
        if finite_number(case.get("cosine"), f"{label}.cosine") < 0.9999:
            fail(f"dispatch case {label} misses cosine gate")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate-sq8-layer-result.py RESULT_DIR", file=sys.stderr)
        return 2
    result_dir = Path(sys.argv[1])
    layer_path = result_dir / "layer0-m8.json"
    layer = load_json(layer_path)
    environment = load_json(result_dir / "environment.json")
    dispatch = load_json(result_dir / "dispatch-runtime-validation.json")
    validate_layer(layer)
    validate_dispatch(dispatch)
    if environment.get("source", {}).get("artifact_content_sha256") != layer.get(
        "artifact_content_sha256"
    ):
        fail("environment and layer artifact hashes differ")
    expected_hash = environment.get("result", {}).get("sha256")
    actual_hash = hashlib.sha256(layer_path.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        fail("layer result SHA-256 does not match environment.json")
    print(
        "passed=true "
        f"tensor_checks={len(layer['tensor_checks'])} "
        f"activation_checks={len(layer['activation_checks'])} "
        f"dispatch_cases={len(dispatch['cases'])} "
        f"speedup={layer['optimized_speedup']:.6f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
