#!/usr/bin/env python3
"""Run and validate the Phase 6 A1 AMD KV/attention comparison."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_vattention_a0 as a0

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ci/tools/vattention_a1_compare.hip.cpp"
PRODUCTION_SOURCE = ROOT / "ci/tools/vattention_a1_production_probe.cpp"
PROTOCOL = "sllm-vattention-a1-compare-v1"
REPORT_VERSION = "sllm-vattention-a1-report-v1"
AGGREGATE_VERSION = "sllm-vattention-a1-aggregate-v1"
QUERY_LENGTHS = [1, 37]
KV_LENGTHS = [255, 256, 257, 1023, 1024, 1025]
MODES = ["contiguous", "vattention", "paged"]
KERNEL_SYMBOL = "sllm_a1_online_attention_proxy_v1"
OUTPUT_ABS_TOLERANCE = 0.016
CROSS_MODE_ABS_TOLERANCE = 0.004


class A1Error(RuntimeError):
    """A fail-closed A1 contract or execution error."""


def _int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise A1Error(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise A1Error(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise A1Error(f"{label} must be a finite nonnegative number")
    return converted


def _bf16_to_float(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _float_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


@functools.lru_cache(maxsize=None)
def _reference(query_length: int, kv_length: int) -> np.ndarray:
    query_indices = np.arange(query_length * 16 * 256, dtype=np.int64)
    query = (((query_indices * 17 + 13) % 29) - 14).astype(np.float32) / 32.0
    query = query.reshape(query_length, 16, 256)
    kv_indices = np.arange(kv_length * 4 * 256, dtype=np.int64)
    key = (((kv_indices * 19 + 7) % 31) - 15).astype(np.float32) / 32.0
    value = (((kv_indices * 23 + 3) % 37) - 18).astype(np.float32) / 32.0
    key = key.reshape(kv_length, 4, 256)
    value = value.reshape(kv_length, 4, 256)
    output = np.empty((query_length, 16, 256), dtype=np.float32)
    for row in range(query_length):
        query_position = kv_length - query_length + row
        for query_head in range(16):
            kv_head = query_head // 4
            scores = key[: query_position + 1, kv_head].astype(np.float64) @ (
                query[row, query_head].astype(np.float64) / 16.0
            )
            scores -= np.max(scores)
            weights = np.exp(scores)
            weights /= np.sum(weights)
            output[row, query_head] = weights @ value[: query_position + 1, kv_head]
    return _bf16_to_float(_float_to_bf16(output.reshape(-1)))


def validate_probe(document: Any, target: str, artifact_dir: Path) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise A1Error("probe output must be a JSON object")
    expected_top = {
        "protocol", "state", "device", "shape", "algorithm", "vmm",
        "warmup_iterations", "measured_iterations", "results", "fallback_used",
        "cleanup_complete",
    }
    if set(document) != expected_top:
        raise A1Error("probe output has missing or unexpected fields")
    if document["protocol"] != PROTOCOL or document["state"] != "PASS":
        raise A1Error("probe did not report the canonical PASS protocol")
    if document["fallback_used"] is not False or document["cleanup_complete"] is not True:
        raise A1Error("probe used fallback or did not clean up")

    expected_device = a0.CANONICAL[target]
    device = document["device"]
    if device != {
        "logical_index": 0,
        "product": expected_device["product"],
        "target": target,
        "bdf": expected_device["bdf"],
        "vmm_supported": True,
    }:
        raise A1Error("probe device identity is not the canonical target")
    if document["shape"] != {
        "q_heads": 16,
        "kv_heads": 4,
        "head_dim": 256,
        "logical_capacity": 4096,
        "paged_block_tokens": 256,
        "query_lengths": QUERY_LENGTHS,
        "kv_lengths": KV_LENGTHS,
    }:
        raise A1Error("Qwen model-free shape or boundary set drifted")
    if document["algorithm"] != {
        "class": "FA2-style tiled online-softmax proxy",
        "kernel_symbol": KERNEL_SYMBOL,
        "contiguous_and_vattention_same_kernel": True,
        "kv_layout": "token-major",
        "causal_alignment": "bottom-right",
    }:
        raise A1Error("algorithm/accessor contract drifted")
    vmm = document["vmm"]
    if not isinstance(vmm, dict) or set(vmm) != {
        "minimum_page_bytes", "recommended_page_bytes", "selected_page_bytes",
    }:
        raise A1Error("VMM granularity is malformed")
    minimum = _int(vmm["minimum_page_bytes"], "minimum VMM page", minimum=1)
    recommended = _int(vmm["recommended_page_bytes"], "recommended VMM page", minimum=1)
    if recommended % minimum != 0 or vmm["selected_page_bytes"] != recommended:
        raise A1Error("VMM page selection is not canonical")
    if document["warmup_iterations"] != 3 or document["measured_iterations"] != 9:
        raise A1Error("timing iteration contract drifted")

    results = document["results"]
    if not isinstance(results, list) or len(results) != len(MODES) * len(QUERY_LENGTHS) * len(KV_LENGTHS):
        raise A1Error("comparison result matrix is incomplete")
    expected_cases = [(mode, q, k) for mode in MODES for q in QUERY_LENGTHS for k in KV_LENGTHS]
    observed_cases: list[tuple[str, int, int]] = []
    first_mode_outputs: dict[tuple[int, int], np.ndarray] = {}
    cross_mode_max_abs_error = 0.0
    oracle_cache: dict[tuple[int, int], np.ndarray] = {}
    validated_results: list[dict[str, Any]] = []
    expected_keys = {
        "mode", "mode_id", "query_length", "kv_length", "setup_us", "grow_us",
        "kernel_p50_us", "kernel_p95_us", "logical_bytes", "committed_bytes",
        "metadata_bytes", "observed_vram_delta_bytes", "output_file",
        "nonidentity_block_table",
    }
    for row in results:
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise A1Error("comparison row is malformed")
        mode = row["mode"]
        q = row["query_length"]
        k = row["kv_length"]
        if mode not in MODES or q not in QUERY_LENGTHS or k not in KV_LENGTHS:
            raise A1Error("comparison row has an unknown mode or shape")
        if row["mode_id"] != MODES.index(mode):
            raise A1Error("comparison mode id drifted")
        observed_cases.append((mode, q, k))
        p50 = _number(row["kernel_p50_us"], "kernel p50")
        p95 = _number(row["kernel_p95_us"], "kernel p95")
        if p50 <= 0 or p95 < p50:
            raise A1Error("kernel timing is invalid")
        _number(row["setup_us"], "setup latency")
        grow = _number(row["grow_us"], "grow latency")
        logical = _int(row["logical_bytes"], "logical bytes", minimum=1)
        committed = _int(row["committed_bytes"], "committed bytes", minimum=1)
        metadata = _int(row["metadata_bytes"], "metadata bytes")
        _int(row["observed_vram_delta_bytes"], "observed VRAM delta")
        if logical != 2 * 4096 * 4 * 256 * 2:
            raise A1Error("logical byte accounting drifted")
        used_plane = k * 4 * 256 * 2
        if mode == "contiguous":
            if committed != logical or metadata != 0 or grow != 0:
                raise A1Error("contiguous accounting is invalid")
        elif mode == "vattention":
            expected_commit = 2 * ((used_plane + recommended - 1) // recommended) * recommended
            if committed != expected_commit or metadata != 0 or grow <= 0:
                raise A1Error("vAttention sparse commitment is invalid")
        else:
            blocks = (k + 255) // 256
            expected_commit = 2 * blocks * 256 * 4 * 256 * 2
            if committed != expected_commit or metadata != blocks * 4 or grow != 0:
                raise A1Error("paged KV accounting is invalid")
            if row["nonidentity_block_table"] is not (blocks > 1):
                raise A1Error("paged comparison did not exercise the required block table")
        if mode != "paged" and row["nonidentity_block_table"] is not False:
            raise A1Error("non-paged mode reported a block table")

        filename = row["output_file"]
        expected_filename = f"{mode}-q{q}-k{k}.bf16"
        if filename != expected_filename or Path(filename).name != filename:
            raise A1Error("output artifact name is non-canonical")
        path = artifact_dir / filename
        expected_size = q * 16 * 256 * 2
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_size:
            raise A1Error("output artifact is missing or has the wrong size")
        raw_bytes = path.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        key = (q, k)
        actual = _bf16_to_float(np.frombuffer(raw_bytes, dtype="<u2"))
        prior = first_mode_outputs.setdefault(key, actual)
        cross_error = float(np.max(np.abs(actual - prior)))
        cross_mode_max_abs_error = max(cross_mode_max_abs_error, cross_error)
        if not math.isfinite(cross_error) or cross_error > CROSS_MODE_ABS_TOLERANCE:
            raise A1Error(f"KV access modes diverged for q={q} k={k}: {cross_error}")
        reference = oracle_cache.setdefault(key, _reference(q, k))
        maximum_error = float(np.max(np.abs(actual - reference)))
        if not math.isfinite(maximum_error) or maximum_error > OUTPUT_ABS_TOLERANCE:
            raise A1Error(f"NumPy oracle failed for {mode} q={q} k={k}: {maximum_error}")
        validated = dict(row)
        validated["output_sha256"] = digest
        validated["oracle_max_abs_error"] = maximum_error
        validated_results.append(validated)
    if observed_cases != expected_cases:
        raise A1Error("comparison rows are missing, duplicated, or reordered")
    validated_document = dict(document)
    validated_document["results"] = validated_results
    validated_document["oracle"] = {
        "implementation": "NumPy float64 softmax/matmul with BF16 output rounding",
        "absolute_tolerance": OUTPUT_ABS_TOLERANCE,
        "cross_mode_absolute_tolerance": CROSS_MODE_ABS_TOLERANCE,
        "cross_mode_max_abs_error": cross_mode_max_abs_error,
        "all_modes_numerically_equivalent": True,
    }
    return validated_document


def validate_production_probe(document: Any, target: str) -> dict[str, Any]:
    expected = {
        "protocol": "sllm-vattention-a1-production-v1",
        "state": "PASS",
        "target": target,
        "layout": "token-major",
        "memory_kind": "virtual-contiguous",
        "boundary_tokens": [1023, 1024, 1025],
        "committed_bytes_per_plane": [2097152, 2097152, 4194304],
        "unmapped_readback_rejected": True,
        "numerical_oracle": True,
        "fallback_used": False,
        "cleanup_complete": True,
    }
    if document != expected:
        raise A1Error("production KV probe did not satisfy the exact A1 contract")
    return document


def run_production_probe(
    target: str, build_dir: Path, environment: dict[str, str]
) -> dict[str, Any]:
    native_build = build_dir / f"production-{target}"
    configure_command = [
        "/usr/bin/cmake", "-S", str(ROOT / "native/hip"), "-B", str(native_build),
        "-DSLLM_ENABLE_PUBLIC_HIP_RUNTIME=ON",
        "-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++",
        "-DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++",
        "-DSLLM_HIP_COMPILER_LOGICAL=/opt/rocm/bin/amdclang++",
        f"-DCMAKE_HIP_ARCHITECTURES={target}",
        f"-DSLLM_HIP_COMPILE_TARGET={target}",
        "-DSLLM_HIP_CODEGEN_FEATURES=co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0",
        "-DROCM_PATH=/opt/rocm",
    ]
    configured = a0._run_bounded(configure_command, timeout=180)
    if configured.returncode != 0:
        raise A1Error(f"production CMake configure failed: {configured.stderr.decode(errors='replace').strip()}")
    build_command = ["/usr/bin/cmake", "--build", str(native_build), "-j2"]
    built = a0._run_bounded(build_command, timeout=300)
    if built.returncode != 0:
        raise A1Error(f"production HIP build failed: {built.stderr.decode(errors='replace').strip()}")
    binary = build_dir / f"vattention-a1-production-{target}"
    link_command = [
        str(a0.COMPILER), "-std=c++17", "-O3", "-DNDEBUG", "-Iinclude",
        "-Inative/hip/src", str(PRODUCTION_SOURCE),
        str(native_build / "libsllm_hip_stub.a"), "-L/opt/rocm/core-7.14/lib",
        "-lamdhip64", "-pthread", "--hip-link", f"--offload-arch={target}",
        "-o", str(binary),
    ]
    linked = a0._run_bounded(link_command, timeout=180)
    if linked.returncode != 0:
        raise A1Error(f"production probe link failed: {linked.stderr.decode(errors='replace').strip()}")
    executed = a0._run_bounded([str(binary), target], env=environment, timeout=120)
    if executed.returncode != 0:
        raise A1Error(f"production probe failed: {executed.stderr.decode(errors='replace').strip()}")
    lines = executed.stdout.splitlines()
    if len(lines) != 1:
        raise A1Error("production probe output is not exactly one JSON line")
    try:
        document = json.loads(lines[0])
    except (UnicodeError, ValueError) as exc:
        raise A1Error("production probe output is invalid JSON") from exc
    return {
        "result": validate_production_probe(document, target),
        "identity": {
            "probe_source": str(PRODUCTION_SOURCE.relative_to(ROOT)),
            "probe_source_sha256": a0.sha256_file(PRODUCTION_SOURCE),
            "public_runtime_sha256": a0.sha256_file(ROOT / "native/hip/src/public_runtime.hip.cpp"),
            "kv_kernel_sha256": a0.sha256_file(ROOT / "native/hip/src/kv_state_kernel.hip.cpp"),
            "binary_sha256": a0.sha256_file(binary),
        },
        "commands": {
            "configure": configure_command,
            "build": build_command,
            "link": link_command,
        },
        "stderr": executed.stderr.decode("utf-8"),
    }


def run_target(target: str, build_dir: Path) -> dict[str, Any]:
    mapping = a0.validate_canonical_mapping(a0._amd_smi_list(), target)
    process_before = a0._gpu_processes(mapping["amd_smi_index"])
    health_before = a0._gpu_health(mapping["amd_smi_index"])
    binary = build_dir / f"vattention-a1-{target}"
    artifact_dir = build_dir / f"artifacts-{target}"
    artifact_dir.mkdir()
    compile_command = [
        str(a0.COMPILER), "-std=c++17", "-O3", "-DNDEBUG", "-x", "hip",
        f"--offload-arch={target}", "-mcode-object-version=6", "-mno-wavefrontsize64",
        "--hip-link", "--rtlib=compiler-rt", "-unwindlib=libgcc", str(SOURCE),
        "-o", str(binary),
    ]
    compiled = a0._run_bounded(compile_command, timeout=180)
    if compiled.returncode != 0:
        raise A1Error(f"HIP compile failed for {target}: {compiled.stderr.decode(errors='replace').strip()}")
    environment = os.environ.copy()
    for selector in ("CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "ROCR_VISIBLE_DEVICES"):
        environment.pop(selector, None)
    environment["HIP_VISIBLE_DEVICES"] = str(mapping["physical_hip_index"])
    environment["LD_LIBRARY_PATH"] = str(a0.RUNTIME_LIBRARY_DIR)
    started_at = datetime.now(timezone.utc)
    executed = a0._run_bounded([str(binary), target, str(artifact_dir)], env=environment, timeout=600)
    finished_at = datetime.now(timezone.utc)
    if executed.returncode != 0:
        raise A1Error(f"A1 comparison failed for {target}: {executed.stderr.decode(errors='replace').strip()}")
    lines = executed.stdout.splitlines()
    if len(lines) != 1:
        raise A1Error("A1 comparison output is not exactly one JSON line")
    try:
        probe = json.loads(lines[0])
    except (UnicodeError, ValueError) as exc:
        raise A1Error("A1 comparison output is invalid JSON") from exc
    validated = validate_probe(probe, target, artifact_dir)
    production_probe = run_production_probe(target, build_dir, environment)
    process_after = a0._gpu_processes(mapping["amd_smi_index"])
    health_after = a0._gpu_health(mapping["amd_smi_index"])
    return {
        "report_version": REPORT_VERSION,
        "state": "PASS",
        "target": target,
        "canonical_device": a0.CANONICAL[target],
        "routing": mapping,
        "toolchain": {
            "rocm_release": "7.14.0",
            "rocm_root": str(a0.ROCM_ROOT),
            "compiler": str(a0.COMPILER),
            "compile_command": compile_command,
        },
        "identity": {
            "source": str(SOURCE.relative_to(ROOT)),
            "source_sha256": a0.sha256_file(SOURCE),
            "binary_sha256": a0.sha256_file(binary),
        },
        "execution": {
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
            "timeout_seconds": 600,
            "stderr": executed.stderr.decode("utf-8"),
            "process_before": process_before,
            "process_after": process_after,
            "health_before": health_before,
            "health_after": health_after,
        },
        "probe": validated,
        "production_probe": production_probe,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=tuple(a0.CANONICAL), default=list(a0.CANONICAL))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise A1Error("output directory must be a regular directory")
    if (not SOURCE.is_file() or SOURCE.is_symlink() or not PRODUCTION_SOURCE.is_file()
            or PRODUCTION_SOURCE.is_symlink()):
        raise A1Error("canonical A1 source is missing")
    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sllm-vattention-a1-") as temporary:
        build_dir = Path(temporary)
        for target in args.targets:
            report = run_target(target, build_dir)
            (output_dir / f"vattention-a1-{target}.json").write_bytes(a0.canonical_bytes(report))
            reports.append(report)
    if args.targets == list(a0.CANONICAL) and [report["target"] for report in reports] != list(a0.CANONICAL):
        raise A1Error("canonical dual-GPU report order is incomplete")
    aggregate = {
        "aggregate_version": AGGREGATE_VERSION,
        "state": "PASS" if len(reports) == len(args.targets) else "FAIL",
        "host": {"kernel": platform.release(), "platform": platform.platform()},
        "targets": args.targets,
        "reports": reports,
        "source_sha256": a0.sha256_file(SOURCE),
    }
    aggregate_bytes = a0.canonical_bytes(aggregate)
    aggregate_path = output_dir / "vattention-a1-aggregate.json"
    aggregate_path.write_bytes(aggregate_bytes)
    print(f"vAttention A1: PASS ({', '.join(args.targets)})")
    print(f"aggregate={aggregate_path} sha256={a0.sha256_bytes(aggregate_bytes)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (A1Error, a0.A0Error) as error:
        print(f"vAttention A1: FAIL: {error}", file=os.sys.stderr)
        raise SystemExit(1)
