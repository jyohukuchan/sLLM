#!/usr/bin/env python3
"""Validate SQ8 shape-expansion results and freeze measured dispatch choices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


M_GRID = (1, 2, 4, 8, 16, 32, 128)
DEFAULT_LABELS = ("q", "o", "k", "v", "gate", "up", "down")
FIXTURE_SCHEMA = "sq8-optimized-fixture-v0.2"
COMPONENT_SCHEMA = "ullm.sq8.ck_component.v2"
REFERENCE_SCHEMA = "sq8-reference-batch-benchmark-v0.2"
RELATIVE_L2_LIMIT = 5.0e-3
COSINE_LIMIT = 0.9999


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"failed to read {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    converted = float(value)
    require(converted == converted and abs(converted) != float("inf"), f"{label} is not finite")
    return converted


def validate_component(
    result: dict[str, Any],
    path: Path,
    fixture: dict[str, Any],
    cache_state: str,
) -> None:
    prefix = str(path)
    require(result.get("schema_version") == COMPONENT_SCHEMA, f"{prefix}: wrong schema")
    require(result.get("status") == "passed" and result.get("passed") is True, f"{prefix}: not passed")
    require(result.get("fallback") == "not_used", f"{prefix}: fallback used")
    expected_shape = {"m": fixture["m"], "n": fixture["n"], "k": fixture["k"]}
    require(result.get("shape") == expected_shape, f"{prefix}: shape differs from fixture")
    timing = result.get("timing")
    require(isinstance(timing, dict), f"{prefix}: timing missing")
    require(timing.get("cache_state") == cache_state, f"{prefix}: wrong cache state")
    require(timing.get("warmups") == 10 and timing.get("repeats") == 50, f"{prefix}: wrong sample policy")
    inclusive = timing.get("quant_plus_gemm")
    require(isinstance(inclusive, dict), f"{prefix}: inclusive timing missing")
    require(finite_number(inclusive.get("p50_ms"), f"{prefix} p50") > 0.0, f"{prefix}: p50 must be positive")
    require(finite_number(inclusive.get("tflops_p50"), f"{prefix} TFLOP/s") > 0.0, f"{prefix}: TFLOP/s must be positive")

    quantization = result.get("activation_quantization_check")
    require(isinstance(quantization, dict), f"{prefix}: activation check missing")
    require(quantization.get("passed") is True, f"{prefix}: activation check failed")
    require(quantization.get("fp8_byte_exact") is True, f"{prefix}: activation FP8 differs")
    require(quantization.get("scale_bit_exact") is True, f"{prefix}: activation scales differ")

    correctness = result.get("correctness")
    require(isinstance(correctness, dict), f"{prefix}: correctness missing")
    require(correctness.get("nonfinite") == 0, f"{prefix}: output is non-finite")
    relative_l2 = finite_number(correctness.get("relative_l2"), f"{prefix} relative L2")
    cosine = finite_number(correctness.get("cosine"), f"{prefix} cosine")
    require(relative_l2 <= RELATIVE_L2_LIMIT, f"{prefix}: relative L2 gate failed")
    require(cosine >= COSINE_LIMIT, f"{prefix}: cosine gate failed")

    candidates = result.get("candidate_measurements")
    require(isinstance(candidates, list), f"{prefix}: candidates missing")
    valid_candidates = [candidate for candidate in candidates if candidate.get("numerically_valid") is True]
    require(result.get("instance_count") == 38, f"{prefix}: unexpected candidate count")
    require(result.get("supported_count") == 6, f"{prefix}: unexpected supported count")
    require(len(valid_candidates) == 6, f"{prefix}: not all supported candidates are valid")
    selected = result.get("selected_instance")
    require(isinstance(selected, dict), f"{prefix}: selected instance missing")
    require(isinstance(selected.get("group"), str), f"{prefix}: selected group missing")
    require(isinstance(selected.get("instance"), str), f"{prefix}: selected type missing")

    eviction = result.get("cache_eviction")
    require(isinstance(eviction, dict), f"{prefix}: cache eviction report missing")
    if cache_state == "target_buffers_evicted":
        require(eviction.get("enabled") is True, f"{prefix}: eviction is disabled")
        require(eviction.get("bytes") == 256 * 1024 * 1024, f"{prefix}: wrong eviction bytes")
        require(eviction.get("outside_timed_region") is True, f"{prefix}: eviction is timed")
        require(eviction.get("validation_checksum_matches") is True, f"{prefix}: eviction checksum failed")
    else:
        require(eviction.get("enabled") is False, f"{prefix}: warm result enabled eviction")


def validate_reference(result: dict[str, Any], path: Path, fixture: dict[str, Any]) -> None:
    prefix = str(path)
    require(result.get("schema_version") == REFERENCE_SCHEMA, f"{prefix}: wrong schema")
    require(result.get("passed") is True, f"{prefix}: not passed")
    require(result.get("fallback_state") == "not_used", f"{prefix}: fallback used")
    require(result.get("profile") == "reference_w8a16", f"{prefix}: wrong profile")
    require(result.get("tensor") == fixture["tensor"]["name"], f"{prefix}: tensor mismatch")
    require(result.get("artifact_content_sha256") == fixture["artifact_content_sha256"], f"{prefix}: artifact mismatch")
    require(result.get("input_f32_le_sha256") == fixture["activation"]["sha256"], f"{prefix}: input hash mismatch")
    require(result.get("m") == fixture["m"], f"{prefix}: M mismatch")
    require(result.get("n") == fixture["n"] and result.get("k") == fixture["k"], f"{prefix}: shape mismatch")
    require(result.get("output_nonfinite") == 0, f"{prefix}: output is non-finite")
    device = result.get("device")
    require(isinstance(device, dict), f"{prefix}: device missing")
    require(device.get("backend") == "hip", f"{prefix}: not HIP")
    require(device.get("compute_major") == 12 and device.get("compute_minor") == 0, f"{prefix}: not gfx12")
    require(device.get("isolated_visibility") is True, f"{prefix}: device is not isolated")
    timing = result.get("timing")
    require(isinstance(timing, dict), f"{prefix}: timing missing")
    require(timing.get("cache_state") == "warm_repeated_same_buffers", f"{prefix}: wrong cache state")
    require(timing.get("warmups") == 10 and timing.get("repeats") == 50, f"{prefix}: wrong sample policy")
    require(finite_number(timing.get("p50_ms"), f"{prefix} p50") > 0.0, f"{prefix}: p50 must be positive")
    correctness = result.get("correctness")
    require(isinstance(correctness, dict) and correctness.get("passed") is True, f"{prefix}: oracle check failed")


def validate_case(results_dir: Path, label: str, m: int) -> dict[str, Any]:
    fixture_path = results_dir / f"fixture-{label}-m{m}.json"
    warm_path = results_dir / f"optimized-{label}-m{m}.json"
    evicted_path = results_dir / f"optimized-evicted-{label}-m{m}.json"
    reference_path = results_dir / f"reference-{label}-m{m}.json"
    fixture = load_json(fixture_path)
    require(fixture.get("schema_version") == FIXTURE_SCHEMA, f"{fixture_path}: wrong schema")
    require(fixture.get("m") == m, f"{fixture_path}: M mismatch")
    require(
        isinstance(fixture.get("n"), int) and fixture["n"] > 0,
        f"{fixture_path}: N missing or invalid",
    )
    require(
        isinstance(fixture.get("k"), int) and fixture["k"] > 0,
        f"{fixture_path}: K missing or invalid",
    )
    require(
        isinstance(fixture.get("artifact_content_sha256"), str),
        f"{fixture_path}: artifact hash missing",
    )
    tensor = fixture.get("tensor")
    activation = fixture.get("activation")
    thresholds = fixture.get("thresholds")
    require(isinstance(tensor, dict), f"{fixture_path}: tensor report missing")
    require(isinstance(tensor.get("name"), str), f"{fixture_path}: tensor name missing")
    require(isinstance(tensor.get("weight_sha256"), str), f"{fixture_path}: weight hash missing")
    require(
        isinstance(tensor.get("weight_scale_sha256"), str),
        f"{fixture_path}: scale hash missing",
    )
    require(isinstance(activation, dict), f"{fixture_path}: activation report missing")
    require(isinstance(activation.get("sha256"), str), f"{fixture_path}: activation hash missing")
    require(isinstance(thresholds, dict), f"{fixture_path}: thresholds missing")
    require(
        thresholds.get("relative_l2_max") == RELATIVE_L2_LIMIT,
        f"{fixture_path}: relative L2 threshold changed",
    )
    require(
        thresholds.get("cosine_min") == COSINE_LIMIT,
        f"{fixture_path}: cosine threshold changed",
    )

    warm = load_json(warm_path)
    evicted = load_json(evicted_path)
    reference = load_json(reference_path)
    validate_component(warm, warm_path, fixture, "warm_repeated_same_buffers")
    validate_component(evicted, evicted_path, fixture, "target_buffers_evicted")
    validate_reference(reference, reference_path, fixture)
    require(warm["correctness"] == evicted["correctness"], f"{label} M={m}: correctness changed with cache mode")

    warm_ms = float(warm["timing"]["quant_plus_gemm"]["p50_ms"])
    evicted_ms = float(evicted["timing"]["quant_plus_gemm"]["p50_ms"])
    reference_ms = float(reference["timing"]["p50_ms"])
    return {
        "label": label,
        "tensor": fixture["tensor"]["name"],
        "artifact_content_sha256": fixture["artifact_content_sha256"],
        "weight_sha256": fixture["tensor"]["weight_sha256"],
        "scale_sha256": fixture["tensor"]["weight_scale_sha256"],
        "m": m,
        "n": fixture["n"],
        "k": fixture["k"],
        "warm": {
            "p50_ms": warm_ms,
            "tflops_p50": warm["timing"]["quant_plus_gemm"]["tflops_p50"],
            "selected_group": warm["selected_instance"]["group"],
            "selected_instance": warm["selected_instance"]["instance"],
        },
        "evicted": {
            "p50_ms": evicted_ms,
            "tflops_p50": evicted["timing"]["quant_plus_gemm"]["tflops_p50"],
        },
        "reference": {"p50_ms": reference_ms},
        "warm_speedup": reference_ms / warm_ms,
        "evicted_vs_warm_reference": reference_ms / evicted_ms,
        "relative_l2": warm["correctness"]["relative_l2"],
        "cosine": warm["correctness"]["cosine"],
    }


def build_reports(results_dir: Path, labels: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = [validate_case(results_dir, label, m) for label in labels for m in M_GRID]
    summaries = []
    for label in labels:
        label_cases = [case for case in cases if case["label"] == label]
        by_m = {case["m"]: case for case in label_cases}
        summaries.append(
            {
                "label": label,
                "tensor": label_cases[0]["tensor"],
                "n": label_cases[0]["n"],
                "k": label_cases[0]["k"],
                "warm_m8_over_m2_throughput_ratio": (
                    by_m[8]["warm"]["tflops_p50"] / by_m[2]["warm"]["tflops_p50"]
                ),
                "evicted_m8_over_m2_throughput_ratio": (
                    by_m[8]["evicted"]["tflops_p50"] / by_m[2]["evicted"]["tflops_p50"]
                ),
                "warm_m8_speedup": by_m[8]["warm_speedup"],
                "evicted_m8_vs_warm_reference": by_m[8]["evicted_vs_warm_reference"],
                "maximum_relative_l2": max(case["relative_l2"] for case in label_cases),
                "minimum_cosine": min(case["cosine"] for case in label_cases),
            }
        )

    metrics = {
        "schema_version": "sq8-shape-expansion-metrics-v0.1",
        "results_dir": str(results_dir),
        "labels": list(labels),
        "m_grid": list(M_GRID),
        "case_count": len(cases),
        "optimized_result_count": len(cases) * 2,
        "reference_result_count": len(cases),
        "thresholds": {
            "relative_l2_max": RELATIVE_L2_LIMIT,
            "cosine_min": COSINE_LIMIT,
            "recommended_m8_over_m2_throughput_ratio_min": 2.5,
        },
        "summaries": summaries,
        "aggregate": {
            "maximum_relative_l2": max(case["relative_l2"] for case in cases),
            "minimum_cosine": min(case["cosine"] for case in cases),
            "minimum_warm_speedup": min(case["warm_speedup"] for case in cases),
            "minimum_evicted_vs_warm_reference": min(
                case["evicted_vs_warm_reference"] for case in cases
            ),
            "minimum_warm_m8_over_m2_throughput_ratio": min(
                summary["warm_m8_over_m2_throughput_ratio"] for summary in summaries
            ),
            "minimum_evicted_m8_over_m2_throughput_ratio": min(
                summary["evicted_m8_over_m2_throughput_ratio"] for summary in summaries
            ),
            "all_passed": True,
            "all_fallback": "not_used",
        },
    }
    dispatch = {
        "schema_version": "sq8-ck-dispatch-table-v0.1",
        "source_results_dir": str(results_dir),
        "profile": "rdna4_w8a8_block",
        "device_arch": "gfx1201",
        "measured_m_values": list(M_GRID),
        "unmeasured_shape_or_m_policy": "reject_optimized_dispatch",
        "cases": [
            {
                "label": case["label"],
                "tensor": case["tensor"],
                "artifact_content_sha256": case["artifact_content_sha256"],
                "weight_sha256": case["weight_sha256"],
                "scale_sha256": case["scale_sha256"],
                "m": case["m"],
                "n": case["n"],
                "k": case["k"],
                "selected_group": case["warm"]["selected_group"],
                "selected_instance": case["warm"]["selected_instance"],
                "warm_p50_ms": case["warm"]["p50_ms"],
                "warm_tflops_p50": case["warm"]["tflops_p50"],
            }
            for case in cases
        ],
    }
    return metrics, dispatch


def atomic_write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ValidationError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise ValidationError(f"refusing to replace existing output {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--label", action="append", dest="labels")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--dispatch-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    labels = tuple(args.labels or DEFAULT_LABELS)
    require(len(labels) == len(set(labels)), "--label values must be unique")
    metrics_output = args.metrics_output or args.results_dir / "metrics.json"
    dispatch_output = args.dispatch_output or args.results_dir / "dispatch-table.json"
    require(metrics_output != dispatch_output, "metrics and dispatch outputs must differ")
    metrics, dispatch = build_reports(args.results_dir, labels)
    atomic_write_json(metrics_output, metrics, args.overwrite)
    try:
        atomic_write_json(dispatch_output, dispatch, args.overwrite)
    except Exception:
        if not args.overwrite and metrics_output.exists():
            metrics_output.unlink()
        raise
    print(
        json.dumps(
            {
                "status": "passed",
                "case_count": metrics["case_count"],
                "metrics_output": str(metrics_output),
                "dispatch_output": str(dispatch_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True))
        raise SystemExit(2) from error
