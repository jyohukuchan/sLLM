#!/usr/bin/env python3
"""Build and run the Phase 6 A0 HIP VMM PoC on canonical AMD GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ci/tools/vattention_a0_probe.hip.cpp"
ROCM_ROOT = Path("/opt/rocm/core-7.14")
COMPILER = Path("/opt/rocm/bin/amdclang++")
RUNTIME_LIBRARY_DIR = ROCM_ROOT / "lib"
PROTOCOL = "sllm-vattention-a0-probe-v1"
REPORT_VERSION = "sllm-vattention-a0-report-v1"
AGGREGATE_VERSION = "sllm-vattention-a0-aggregate-v1"
MAX_OUTPUT_BYTES = 1024 * 1024

CANONICAL: dict[str, dict[str, Any]] = {
    "gfx1030": {
        "target": "gfx1030",
        "product": "AMD Radeon Pro V620",
        "bdf": "0000:03:00.0",
        "hip_uuid": "GPU-76a08c022586fed6",
        "physical_hip_index": 1,
    },
    "gfx1201": {
        "target": "gfx1201",
        "product": "AMD Radeon AI PRO R9700",
        "bdf": "0000:07:00.0",
        "hip_uuid": "GPU-a8e9ddefa2d60f55",
        "physical_hip_index": 2,
    },
}


class A0Error(RuntimeError):
    """A fail-closed A0 contract or execution error."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise A0Error(f"required input must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_amd_smi_list(text: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.fullmatch(r"GPU: ([0-9]+)", line)
        if match:
            if current is not None:
                target = current.get("HIP_UUID")
                if isinstance(target, str):
                    records[target] = current
            current = {"amd_smi_index": int(match.group(1))}
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = value.strip()
    if current is not None and isinstance(current.get("HIP_UUID"), str):
        records[current["HIP_UUID"]] = current
    return records


def validate_canonical_mapping(text: str, target: str) -> dict[str, Any]:
    expected = CANONICAL[target]
    records = parse_amd_smi_list(text)
    record = records.get(expected["hip_uuid"])
    if record is None:
        raise A0Error(f"canonical HIP UUID is absent from amd-smi: {expected['hip_uuid']}")
    if record.get("BDF") != expected["bdf"]:
        raise A0Error(f"canonical BDF drift for {target}: {record.get('BDF')}")
    try:
        physical_hip_index = int(record.get("HIP_ID", ""))
    except ValueError as exc:
        raise A0Error(f"invalid HIP_ID for {target}") from exc
    if physical_hip_index != expected["physical_hip_index"]:
        raise A0Error(f"canonical HIP index drift for {target}: {physical_hip_index}")
    return {
        "amd_smi_index": record["amd_smi_index"],
        "bdf": record["BDF"],
        "hip_uuid": record["HIP_UUID"],
        "physical_hip_index": physical_hip_index,
    }


def _int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise A0Error(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise A0Error(f"{label} must be a finite nonnegative number")
    converted = float(value)
    if converted == float("inf") or converted != converted:
        raise A0Error(f"{label} must be finite")
    return converted


def validate_probe(document: Any, target: str) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise A0Error("probe output must be a JSON object")
    expected_top = {
        "protocol", "state", "device", "granularity", "primitive", "qwen_shape",
        "latency_us", "memory_info", "fallback_used", "cleanup_complete",
    }
    if set(document) != expected_top:
        raise A0Error("probe output has missing or unexpected top-level fields")
    if document["protocol"] != PROTOCOL or document["state"] != "PASS":
        raise A0Error("probe protocol/state is not a PASS result")
    if document["fallback_used"] is not False or document["cleanup_complete"] is not True:
        raise A0Error("probe used a fallback or did not complete cleanup")

    expected = CANONICAL[target]
    device = document["device"]
    if not isinstance(device, dict) or set(device) != {
        "logical_index", "product", "target", "bdf", "vmm_supported",
    }:
        raise A0Error("probe device identity is malformed")
    if device != {
        "logical_index": 0,
        "product": expected["product"],
        "target": target,
        "bdf": expected["bdf"],
        "vmm_supported": True,
    }:
        raise A0Error("probe device identity does not match the canonical target")

    granularity = document["granularity"]
    if not isinstance(granularity, dict) or set(granularity) != {
        "minimum_bytes", "recommended_bytes", "selected_physical_page_bytes",
    }:
        raise A0Error("granularity result is malformed")
    minimum = _int(granularity["minimum_bytes"], "minimum granularity", minimum=1)
    recommended = _int(granularity["recommended_bytes"], "recommended granularity", minimum=1)
    selected = _int(granularity["selected_physical_page_bytes"], "selected page", minimum=1)
    if recommended % minimum != 0 or selected != recommended:
        raise A0Error("selected physical page is not the recommended multiple of minimum granularity")

    primitive = document["primitive"]
    if not isinstance(primitive, dict) or set(primitive) != {
        "reserved_bytes", "mapped_pages", "contiguous_kernel_oracle", "remap_oracle",
        "event_synchronized_before_unmap", "nonaligned_byte_offset",
    }:
        raise A0Error("primitive result is malformed")
    if primitive != {
        "reserved_bytes": selected * 3,
        "mapped_pages": 3,
        "contiguous_kernel_oracle": True,
        "remap_oracle": True,
        "event_synchronized_before_unmap": True,
        "nonaligned_byte_offset": 37,
    }:
        raise A0Error("primitive reserve/map/remap/lifetime oracle did not pass canonically")

    qwen = document["qwen_shape"]
    expected_qwen_keys = {
        "model", "full_attention_layers", "regions", "kv_heads", "head_dim", "element_bytes",
        "bytes_per_token_per_region", "logical_token_capacity", "tokens_per_physical_page",
        "logical_reserved_bytes", "requested_physical_bytes", "observed_physical_commit_bytes",
        "virtual_reserve_physical_delta_bytes", "activated_pages_per_step", "boundary_tokens",
    }
    if not isinstance(qwen, dict) or set(qwen) != expected_qwen_keys:
        raise A0Error("Qwen model-free shape result is malformed")
    fixed_qwen = {
        "model": "Qwen/Qwen3.5-4B", "full_attention_layers": 8, "regions": 16,
        "kv_heads": 4, "head_dim": 256, "element_bytes": 2,
        "bytes_per_token_per_region": 2048, "logical_token_capacity": 4096,
        "activated_pages_per_step": 16,
    }
    for key, value in fixed_qwen.items():
        if qwen.get(key) != value:
            raise A0Error(f"Qwen model-free shape drift: {key}")
    tokens_per_page = selected // 2048
    if tokens_per_page < 2 or qwen["tokens_per_physical_page"] != tokens_per_page:
        raise A0Error("Qwen tokens-per-page is invalid")
    if qwen["boundary_tokens"] != [tokens_per_page - 1, tokens_per_page, tokens_per_page + 1, 37]:
        raise A0Error("Qwen B-1/B/B+1 and non-aligned boundary cases are missing")
    logical = _int(qwen["logical_reserved_bytes"], "logical reserve", minimum=1)
    requested = _int(qwen["requested_physical_bytes"], "requested physical bytes", minimum=1)
    observed = _int(qwen["observed_physical_commit_bytes"], "observed physical commit", minimum=1)
    reserve_delta = _int(qwen["virtual_reserve_physical_delta_bytes"], "reserve physical delta")
    if logical != 16 * 4096 * 2048 or requested != 16 * selected:
        raise A0Error("Qwen logical/physical byte accounting is not canonical")
    if observed >= logical or reserve_delta > selected:
        raise A0Error("physical commitment is not sparse relative to logical capacity")

    latency = document["latency_us"]
    if not isinstance(latency, dict) or set(latency) != {
        "warmup_iterations", "measured_iterations", "activate_p50", "activate_p95",
        "create_p50", "create_p95", "map_p50", "map_p95", "set_access_p50",
        "set_access_p95", "deactivate_p50", "deactivate_p95", "unmap_p50", "unmap_p95",
        "release_p50", "release_p95",
    }:
        raise A0Error("latency result is malformed")
    if latency["warmup_iterations"] != 5 or latency["measured_iterations"] != 101:
        raise A0Error("latency iteration counts are not canonical")
    for operation in ("activate", "create", "map", "set_access", "deactivate", "unmap", "release"):
        p50 = _number(latency[f"{operation}_p50"], f"{operation} p50")
        p95 = _number(latency[f"{operation}_p95"], f"{operation} p95")
        if p95 < p50:
            raise A0Error(f"{operation} p95 is lower than p50")

    memory = document["memory_info"]
    expected_memory_keys = {
        "total_bytes", "free_before_bytes", "free_after_primitive_reserve_bytes",
        "free_before_qwen_reserve_bytes", "free_after_qwen_reserve_bytes",
        "free_after_first_create_bytes", "free_after_first_map_bytes", "free_after_cleanup_bytes",
    }
    if not isinstance(memory, dict) or set(memory) != expected_memory_keys:
        raise A0Error("memory-info result is malformed")
    for key in expected_memory_keys:
        _int(memory[key], f"memory info {key}")
    if memory["free_before_qwen_reserve_bytes"] - memory["free_after_qwen_reserve_bytes"] != reserve_delta:
        raise A0Error("reported virtual reserve physical delta is inconsistent")
    if memory["free_after_qwen_reserve_bytes"] - memory["free_after_first_create_bytes"] != observed:
        raise A0Error("reported physical commitment is inconsistent")
    if abs(memory["free_after_first_create_bytes"] - memory["free_after_first_map_bytes"]) > selected:
        raise A0Error("mapping unexpectedly added more than one page beyond physical handle creation")
    if memory["free_after_cleanup_bytes"] + selected < memory["free_before_qwen_reserve_bytes"]:
        raise A0Error("physical memory was not restored after cleanup")
    return document


def _run_bounded(argv: list[str], *, env: dict[str, str] | None = None, timeout: int = 120) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise A0Error(f"command timed out: {argv[0]}") from exc
    if len(stdout) > MAX_OUTPUT_BYTES or len(stderr) > MAX_OUTPUT_BYTES:
        raise A0Error(f"command exceeded bounded output: {argv[0]}")
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _amd_smi_list() -> str:
    result = _run_bounded(["/usr/bin/amd-smi", "list", "-e"], timeout=30)
    if result.returncode != 0:
        raise A0Error(f"amd-smi list failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout.decode("utf-8")


def _gpu_processes(amd_smi_index: int) -> Any:
    result = _run_bounded(
        ["/usr/bin/amd-smi", "process", "-g", str(amd_smi_index), "--json"], timeout=30,
    )
    if result.returncode != 0:
        raise A0Error(f"amd-smi process failed: {result.stderr.decode(errors='replace').strip()}")
    try:
        document = json.loads(result.stdout)
    except (UnicodeError, ValueError) as exc:
        raise A0Error("amd-smi process returned invalid JSON") from exc
    return document


def validate_health(document: Any, amd_smi_index: int) -> dict[str, Any]:
    if not isinstance(document, dict) or not isinstance(document.get("gpu_data"), list):
        raise A0Error("amd-smi metric output is malformed")
    rows = document["gpu_data"]
    if len(rows) != 1 or not isinstance(rows[0], dict) or rows[0].get("gpu") != amd_smi_index:
        raise A0Error("amd-smi metric output does not identify the requested GPU")
    row = rows[0]
    temperature = row.get("temperature")
    ecc = row.get("ecc")
    power = row.get("power")
    usage = row.get("usage")
    if not all(isinstance(value, dict) for value in (temperature, ecc, power, usage)):
        raise A0Error("amd-smi metric output lacks health sections")
    temperatures: dict[str, int] = {}
    for key in ("edge", "hotspot", "mem"):
        reading = temperature.get(key)
        if not isinstance(reading, dict) or reading.get("unit") != "C":
            raise A0Error(f"amd-smi metric lacks temperature reading: {key}")
        temperatures[key] = _int(reading.get("value"), f"temperature {key}")
    ecc_fields = (
        "total_uncorrectable_count", "total_deferred_count", "cache_uncorrectable_count",
    )
    for key in ecc_fields:
        if _int(ecc.get(key), f"ECC {key}") != 0:
            raise A0Error(f"amd-smi reports a nonzero ECC health counter: {key}")
    gfx_activity = usage.get("gfx_activity")
    socket_power = power.get("socket_power")
    if not isinstance(gfx_activity, dict) or gfx_activity.get("unit") != "%":
        raise A0Error("amd-smi metric lacks gfx activity")
    if not isinstance(socket_power, dict) or socket_power.get("unit") != "W":
        raise A0Error("amd-smi metric lacks socket power")
    return {
        "gpu": amd_smi_index,
        "temperature_c": temperatures,
        "ecc": {key: ecc[key] for key in ecc_fields},
        "gfx_activity_percent": _int(gfx_activity.get("value"), "gfx activity"),
        "socket_power_w": _int(socket_power.get("value"), "socket power"),
        "throttle_status": power.get("throttle_status"),
    }


def _gpu_health(amd_smi_index: int) -> dict[str, Any]:
    result = _run_bounded(
        ["/usr/bin/amd-smi", "metric", "-g", str(amd_smi_index), "--json"], timeout=30,
    )
    if result.returncode != 0:
        raise A0Error(f"amd-smi metric failed: {result.stderr.decode(errors='replace').strip()}")
    try:
        document = json.loads(result.stdout)
    except (UnicodeError, ValueError) as exc:
        raise A0Error("amd-smi metric returned invalid JSON") from exc
    return validate_health(document, amd_smi_index)


def run_target(target: str, build_dir: Path) -> dict[str, Any]:
    mapping = validate_canonical_mapping(_amd_smi_list(), target)
    process_before = _gpu_processes(mapping["amd_smi_index"])
    health_before = _gpu_health(mapping["amd_smi_index"])
    binary = build_dir / f"vattention-a0-{target}"
    compile_command = [
        str(COMPILER), "-std=c++17", "-O3", "-DNDEBUG", "-x", "hip",
        f"--offload-arch={target}", "-mcode-object-version=6", "-mno-wavefrontsize64",
        "--hip-link", "--rtlib=compiler-rt", "-unwindlib=libgcc", str(SOURCE), "-o", str(binary),
    ]
    compiled = _run_bounded(compile_command, timeout=120)
    if compiled.returncode != 0:
        raise A0Error(f"HIP compile failed for {target}: {compiled.stderr.decode(errors='replace').strip()}")
    binary_sha256 = sha256_file(binary)

    environment = os.environ.copy()
    for selector in ("CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "ROCR_VISIBLE_DEVICES"):
        environment.pop(selector, None)
    environment["HIP_VISIBLE_DEVICES"] = str(mapping["physical_hip_index"])
    environment["LD_LIBRARY_PATH"] = str(RUNTIME_LIBRARY_DIR)
    started_at = datetime.now(timezone.utc)
    executed = _run_bounded([str(binary), target], env=environment, timeout=120)
    finished_at = datetime.now(timezone.utc)
    if executed.returncode != 0:
        raise A0Error(f"A0 probe failed for {target}: {executed.stderr.decode(errors='replace').strip()}")
    lines = executed.stdout.splitlines()
    if len(lines) != 1:
        raise A0Error(f"A0 probe output for {target} is not exactly one JSON line")
    try:
        probe = json.loads(lines[0])
    except (UnicodeError, ValueError) as exc:
        raise A0Error(f"A0 probe output for {target} is invalid JSON") from exc
    validate_probe(probe, target)
    process_after = _gpu_processes(mapping["amd_smi_index"])
    health_after = _gpu_health(mapping["amd_smi_index"])

    return {
        "report_version": REPORT_VERSION,
        "state": "PASS",
        "target": target,
        "canonical_device": CANONICAL[target],
        "routing": mapping,
        "toolchain": {
            "rocm_release": "7.14.0",
            "rocm_root": str(ROCM_ROOT),
            "compiler": str(COMPILER),
            "compile_command": compile_command,
        },
        "identity": {
            "source": str(SOURCE.relative_to(ROOT)),
            "source_sha256": sha256_file(SOURCE),
            "binary_sha256": binary_sha256,
        },
        "execution": {
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
            "timeout_seconds": 120,
            "stderr": executed.stderr.decode("utf-8"),
            "process_before": process_before,
            "process_after": process_after,
            "health_before": health_before,
            "health_after": health_after,
        },
        "probe": probe,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=tuple(CANONICAL), default=list(CANONICAL))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise A0Error("output directory must be a regular directory")
    if not SOURCE.is_file() or SOURCE.is_symlink() or not COMPILER.exists() or not RUNTIME_LIBRARY_DIR.is_dir():
        raise A0Error("canonical source/toolchain input is missing")

    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sllm-vattention-a0-") as temporary:
        build_dir = Path(temporary)
        for target in args.targets:
            report = run_target(target, build_dir)
            report_path = output_dir / f"vattention-a0-{target}.json"
            report_path.write_bytes(canonical_bytes(report))
            reports.append(report)

    aggregate = {
        "aggregate_version": AGGREGATE_VERSION,
        "state": "PASS" if len(reports) == len(args.targets) else "FAIL",
        "host": {"kernel": platform.release(), "platform": platform.platform()},
        "targets": args.targets,
        "reports": reports,
        "source_sha256": sha256_file(SOURCE),
    }
    if args.targets == list(CANONICAL) and [report["target"] for report in reports] != list(CANONICAL):
        raise A0Error("canonical dual-GPU report order is incomplete")
    aggregate_path = output_dir / "vattention-a0-aggregate.json"
    aggregate_bytes = canonical_bytes(aggregate)
    aggregate_path.write_bytes(aggregate_bytes)
    print(f"vAttention A0: PASS ({', '.join(args.targets)})")
    print(f"aggregate={aggregate_path} sha256={sha256_bytes(aggregate_bytes)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A0Error as error:
        print(f"vAttention A0: FAIL: {error}", file=os.sys.stderr)
        raise SystemExit(1)
