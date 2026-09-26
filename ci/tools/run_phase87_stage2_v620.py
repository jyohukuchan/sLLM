#!/usr/bin/env python3
"""Build and run the Phase 87 WU-2V V620 FP8 C1 probe.

The probe compares a test-only wave-uniform weight-base candidate with the
production gfx1030 ID82 M=1 launcher for K=6144,N=5120.  This runner keeps the
exact GPU lease, build inputs, source identity, and fail-closed JSON report;
the HIP process owns the independent oracle and AB/BA/AB timing protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("native/hip/tests/phase87_stage2_v620_probe.hip.cpp")
UUID = "GPU-76a08c022586fed6"
QWEN_STATUS = "/home/homelab1/.local/bin/qwen38-subagent-server"
ROCM_COMPILER = "/opt/rocm/bin/amdclang++"
SOURCE_PATHS = (
    SOURCE,
    Path("native/hip/src/matmul_kernel_internal.hpp"),
    Path("native/lowp/include/lowp/detail/lowp_kernel_internal.hpp"),
    Path("native/lowp/include/lowp/detail/bf16_helpers.inc"),
    Path("native/lowp/include/lowp/detail/low_precision_block_codec.hpp"),
    Path("ci/tools/run_phase87_stage2_v620.py"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_device(uuid: str) -> Path:
    matches = []
    wanted = uuid.removeprefix("GPU-").lower()
    for unique_id in Path("/sys/class/drm").glob("card[0-9]*/device/unique_id"):
        if unique_id.read_text().strip().lower() == wanted:
            matches.append(unique_id.parent)
    if len(matches) != 1:
        raise RuntimeError(f"exact GPU UUID is not unique: {uuid} ({len(matches)})")
    return matches[0]


def source_hashes(archive: Path) -> dict[str, str]:
    paths = list(SOURCE_PATHS) + [archive]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("missing build input: " + ", ".join(map(str, missing)))
    return {str(path): digest(path) for path in paths}


def finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


def finite_nonnegative(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"blank JSONL line {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def validate_case(path: Path, pattern: int, benchmark: bool) -> dict:
    rows = read_jsonl(path)
    identities = [row for row in rows if row.get("kind") == "identity"]
    oracles = [row for row in rows if row.get("kind") == "oracle"]
    performances = [row for row in rows if row.get("kind") == "performance"]
    cleanups = [row for row in rows if row.get("kind") == "cleanup"]
    if len(identities) != 1 or len(oracles) != 1 or len(cleanups) != 1:
        raise RuntimeError(f"identity/oracle/cleanup rows are incomplete: {path}")
    identity = identities[0]
    for key, wanted in (
        ("target", "gfx1030"),
        ("m", 1),
        ("k", 6144),
        ("n", 5120),
        ("pattern", pattern),
    ):
        if identity.get(key) != wanted:
            raise RuntimeError(f"identity mismatch for {path}: {identity}")
    if identity.get("control_launcher") != "production_public_id82":
        raise RuntimeError(f"control is not production ID82 for {path}")
    if identity.get("control_symbol") != (
        "sllm_matmul_fp8_outer_decode_gfx1030_lds_lut_m1_k6144n5120_v1"
    ):
        raise RuntimeError(f"wrong production control symbol for {path}")
    if identity.get("gpu_execution") is not True or identity.get("fallback_used") is not False:
        raise RuntimeError(f"GPU/fallback contract failed for {path}")
    if not isinstance(identity.get("weight_pool_bytes"), int) or identity["weight_pool_bytes"] < 512 * 2**20:
        raise RuntimeError(f"weight pool is below 512 MiB for {path}")
    if not isinstance(identity.get("copies"), int) or identity["copies"] < 1:
        raise RuntimeError(f"invalid weight pool copy count for {path}")

    oracle = oracles[0]
    if oracle.get("state") != "PASS":
        raise RuntimeError(f"oracle failed for {path}: {oracle}")
    for key in (
        "finite",
        "guard",
        "repeat",
        "candidate_control_bitwise",
        "quantizer_bitwise",
    ):
        if oracle.get(key) is not True:
            raise RuntimeError(f"oracle {key} failed for {path}: {oracle}")
    for key in ("max_ulp", "control_max_ulp"):
        if not isinstance(oracle.get(key), int) or not 0 <= oracle[key] <= 4:
            raise RuntimeError(f"oracle ULP bound failed for {path}: {oracle}")
    for key in ("max_abs", "control_max_abs"):
        if not finite_nonnegative(oracle.get(key)):
            raise RuntimeError(f"oracle absolute error is invalid for {path}: {oracle}")

    expected_performance = 1 if benchmark else 0
    if len(performances) != expected_performance:
        raise RuntimeError(f"performance row count mismatch for {path}")
    if benchmark:
        performance = performances[0]
        if (
            performance.get("state") != "PASS"
            or performance.get("warmup_ms") != 300
            or performance.get("samples") != 27
            or performance.get("rounds") != 3
            or performance.get("order") != "AB-BA-AB"
        ):
            raise RuntimeError(f"performance protocol mismatch for {path}: {performance}")
        for key in (
            "control_quant_ms",
            "control_dot_ms",
            "control_total_ms",
            "candidate_quant_ms",
            "candidate_dot_ms",
            "candidate_total_ms",
        ):
            if not finite_positive(performance.get(key)):
                raise RuntimeError(f"performance timing is invalid for {path}: {performance}")
        for key in (
            "control_quant_samples_ms",
            "control_dot_samples_ms",
            "control_total_samples_ms",
            "candidate_quant_samples_ms",
            "candidate_dot_samples_ms",
            "candidate_total_samples_ms",
        ):
            values = performance.get(key)
            if not isinstance(values, list) or len(values) != 27 or any(
                not finite_positive(value) for value in values
            ):
                raise RuntimeError(f"performance sample vector is invalid for {path}")
    cleanup = cleanups[0]
    if cleanup.get("state") != "PASS" or cleanup.get("live_allocations") != 0:
        raise RuntimeError(f"cleanup failed for {path}: {cleanup}")
    if len(rows) != 3 + expected_performance:
        raise RuntimeError(f"unexpected JSONL row count for {path}: {len(rows)}")
    return {
        "oracle": oracle,
        "performance": performances[0] if performances else None,
        "cleanup": cleanup,
        "row_count": len(rows),
    }


def build_command(source: Path, archive: Path, binary: Path) -> list[str]:
    return [
        ROCM_COMPILER,
        "-O3",
        "-ffp-contract=off",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-x",
        "hip",
        "--offload-arch=gfx1030",
        "--hip-link",
        "-I",
        "include",
        "-I",
        "native/hip/src",
        "-I",
        "native/lowp/include",
        str(source),
        "-x",
        "none",
        str(archive),
        "-L/opt/rocm/lib",
        "-lamdhip64",
        "-o",
        str(binary),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowp-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", choices=("0", "1", "2", "all"), default="all")
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    archive = args.lowp_archive.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    if not archive.is_file():
        parser.error(f"lowp archive is required: {archive}")
    exact_device(UUID)
    output.mkdir(parents=True)
    cases = (0, 1, 2) if args.pattern == "all" else (int(args.pattern),)
    binary = output / "phase87-stage2-v620-gfx1030"
    before = source_hashes(archive)
    command = build_command(SOURCE, archive, binary)
    build = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    (output / "build.stdout").write_text(build.stdout)
    (output / "build.stderr").write_text(build.stderr)
    if build.returncode != 0:
        print(f"build failed; see {output / 'build.stderr'}", file=sys.stderr)
        return 2

    status = subprocess.check_output([QWEN_STATUS, "status"], text=True)
    if "process:  stopped" not in status:
        raise RuntimeError("local Qwen must be stopped before V620 GPU measurement")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLLM_")
        and key not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    }
    env.update(
        {
            "ROCR_VISIBLE_DEVICES": UUID,
            "LD_LIBRARY_PATH": "/opt/rocm/lib",
            "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING": "0",
        }
    )

    jobs: list[dict] = []
    started = time.time()
    state = "complete"
    error = None
    try:
        for pattern in cases:
            label = f"pattern{pattern}"
            raw_path = output / f"{label}.jsonl"
            stderr_path = output / f"{label}.stderr"
            command_run = [
                str(binary),
                "--target",
                "gfx1030",
                "--pattern",
                str(pattern),
                "--bench",
                "0" if args.no_bench else "1",
            ]
            job = {
                "pattern": pattern,
                "command": command_run,
                "raw_jsonl": str(raw_path),
                "stderr": str(stderr_path),
                "started": time.time(),
            }
            jobs.append(job)
            with raw_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(command_run, cwd=REPO, env=env, stdout=stdout, stderr=stderr)
                job["pid"] = process.pid
                job["exit_code"] = process.wait()
            job["ended"] = time.time()
            job["raw_sha256"] = digest(raw_path)
            if job["exit_code"] != 0:
                raise RuntimeError(f"probe exited {job['exit_code']}: {label}")
            job["validation"] = validate_case(raw_path, pattern, not args.no_bench)
    except Exception as exc:  # noqa: BLE001 - report the fail-closed state
        state = "failed"
        error = str(exc)

    after = source_hashes(archive)
    report = {
        "schema": "phase87-stage2-v620-execution-v1",
        "state": state,
        "target": "gfx1030",
        "uuid": UUID,
        "lowp_archive": str(archive),
        "lowp_archive_sha256": digest(archive),
        "binary": str(binary),
        "binary_sha256": digest(binary) if binary.is_file() else None,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "run_inputs_unchanged": before == after,
        "compile_command": command,
        "benchmark": not args.no_bench,
        "patterns": list(cases),
        "jobs": jobs,
        "started": started,
        "ended": time.time(),
    }
    if error is not None:
        report["error"] = error
    if not report["run_inputs_unchanged"]:
        report["state"] = "failed"
    (output / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Phase87 WU-2V V620: {report['state']} patterns={len(jobs)}/{len(cases)}")
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
