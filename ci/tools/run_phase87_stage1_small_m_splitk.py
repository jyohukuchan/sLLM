#!/usr/bin/env python3
"""Build and run the bounded Phase 87 Stage 1 V620 M=2/3 split-K probe."""

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
SOURCE = Path("native/hip/tests/phase87_stage1_small_m_splitk_probe.hip.cpp")
TARGET = "gfx1030"
UUID = "GPU-76a08c022586fed6"
QWEN_STATUS = "/home/homelab1/.local/bin/qwen38-subagent-server"
ROCM_COMPILER = "/opt/rocm/bin/amdclang++"
SOURCE_PATHS = (
    SOURCE,
    Path("native/lowp/src/nvfp4_decode_scale_lut.inc"),
    Path("native/hip/src/matmul_kernel_internal.hpp"),
    Path("native/lowp/include/lowp/detail/lowp_kernel_internal.hpp"),
    Path("ci/tools/run_phase87_stage1_small_m_splitk.py"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_device(uuid: str) -> Path:
    wanted = uuid.removeprefix("GPU-").lower()
    matches = [
        unique_id.parent
        for unique_id in Path("/sys/class/drm").glob("card[0-9]*/device/unique_id")
        if unique_id.read_text().strip().lower() == wanted
    ]
    if len(matches) != 1:
        raise RuntimeError(f"exact GPU UUID is not unique: {uuid} ({len(matches)})")
    return matches[0]


def source_hashes(archive: Path) -> dict[str, str]:
    paths = list(SOURCE_PATHS) + [archive]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("missing build input: " + ", ".join(map(str, missing)))
    return {str(path): digest(path) for path in paths}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"blank JSONL line {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL line {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


def validate_case(path: Path, shape: tuple[int, int], m: int, benchmark: bool) -> dict:
    rows = read_jsonl(path)
    identities = [row for row in rows if row.get("kind") == "identity"]
    resources = [row for row in rows if row.get("kind") == "resource"]
    oracles = [row for row in rows if row.get("kind") == "oracle"]
    performances = [row for row in rows if row.get("kind") == "performance"]
    cleanups = [row for row in rows if row.get("kind") == "cleanup"]
    if len(identities) != 1 or len(resources) != 1 or len(oracles) != 1 or len(cleanups) != 1:
        raise RuntimeError(f"identity/resource/oracle/cleanup rows incomplete: {path}")
    identity = identities[0]
    wanted = {"target": TARGET, "m": m, "k": shape[0], "n": shape[1]}
    if any(identity.get(key) != value for key, value in wanted.items()):
        raise RuntimeError(f"identity mismatch for {path}: {identity}")
    if (
        identity.get("candidate") != "id94_splitk2_tm1"
        or identity.get("control_launcher") != "production_id94"
        or identity.get("control_variant") != 94
        or identity.get("split_k") != 2
        or identity.get("gpu_execution") is not True
        or identity.get("fallback_used") is not False
        or not isinstance(identity.get("weight_pool_bytes"), int)
        or identity["weight_pool_bytes"] < 512 * 2**20
    ):
        raise RuntimeError(f"identity contract failed for {path}: {identity}")
    resource = resources[0]
    for key in (
        "producer_vgpr",
        "producer_static_lds",
        "producer_active_blocks",
        "reducer_vgpr",
        "reducer_active_blocks",
    ):
        if not isinstance(resource.get(key), int) or resource[key] <= 0:
            raise RuntimeError(f"resource contract failed for {path}: {resource}")
    oracle = oracles[0]
    if oracle.get("state") != "PASS" or any(
        oracle.get(key) is not True for key in ("finite", "guard", "repeat")
    ):
        raise RuntimeError(f"oracle contract failed for {path}: {oracle}")
    for key in ("candidate_max_ulp", "control_max_ulp"):
        if not isinstance(oracle.get(key), int) or oracle[key] > 8:
            raise RuntimeError(f"oracle ULP bound failed for {path}: {oracle}")
    expected_performance = 2 if benchmark else 0
    if len(performances) != expected_performance:
        raise RuntimeError(f"performance row count mismatch for {path}")
    for performance in performances:
        if (
            performance.get("state") != "PASS"
            or performance.get("warmup_ms") != 300
            or performance.get("samples") != 27
            or performance.get("rounds") != 3
            or performance.get("order") != "AB-BA-AB"
            or performance.get("mode") not in ("isolated", "gqa32")
        ):
            raise RuntimeError(f"performance protocol mismatch for {path}: {performance}")
        for key in (
            "control_kv_ms",
            "candidate_kv_ms",
            "control_quant_ms",
            "candidate_quant_ms",
            "control_dot_ms",
            "candidate_dot_ms",
            "control_total_ms",
            "candidate_total_ms",
        ):
            if not positive(performance.get(key)):
                raise RuntimeError(f"invalid timing for {path}: {performance}")
        for key in (
            "control_kv_samples_ms",
            "candidate_kv_samples_ms",
            "control_quant_samples_ms",
            "candidate_quant_samples_ms",
            "control_dot_samples_ms",
            "candidate_dot_samples_ms",
            "control_total_samples_ms",
            "candidate_total_samples_ms",
        ):
            samples = performance.get(key)
            if not isinstance(samples, list) or len(samples) != 27 or any(
                not positive(value) for value in samples
            ):
                raise RuntimeError(f"invalid raw timing samples for {path}: {key}")
    if cleanups[0].get("state") != "PASS" or cleanups[0].get("live_allocations") != 0:
        raise RuntimeError(f"cleanup failed for {path}: {cleanups[0]}")
    if len(rows) != 4 + expected_performance:
        raise RuntimeError(f"unexpected row count for {path}: {len(rows)}")
    return {"identity": identity, "resource": resource, "oracle": oracle, "performance": performances}


def build_command(archive: Path, binary: Path) -> list[str]:
    return [
        ROCM_COMPILER,
        "-D__HIP_ROCclr__=1",
        f'-DSLLM_HIP_COMPILE_TARGET="{TARGET}"',
        "-O3",
        "-ffp-contract=off",
        "-std=gnu++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"--offload-arch={TARGET}",
        "-mcode-object-version=6",
        "-mno-wavefrontsize64",
        "--hip-link",
        "-I",
        "include",
        "-I",
        "native/hip/src",
        "-I",
        "native/lowp/include",
        "-x",
        "hip",
        str(SOURCE),
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
    parser.add_argument("--shape", choices=("wide", "down", "both"), default="wide")
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()
    archive = args.lowp_archive.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    if not archive.is_file():
        parser.error(f"lowp archive is required: {archive}")
    device_path = exact_device(UUID)
    output.mkdir(parents=True)
    shape_cases = ((0, (5120, 17408), 2), (1, (17408, 5120), 3))
    if args.shape == "wide":
        shape_cases = (shape_cases[0],)
    elif args.shape == "down":
        shape_cases = (shape_cases[1],)
    binary = output / "phase87-stage1-small-m-splitk-gfx1030"
    before = source_hashes(archive)
    command = build_command(archive, binary)
    build = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    (output / "build.stdout").write_text(build.stdout)
    (output / "build.stderr").write_text(build.stderr)
    if build.returncode != 0:
        print(f"build failed; see {output / 'build.stderr'}", file=sys.stderr)
        return 2
    service_status = subprocess.check_output([QWEN_STATUS, "status"], text=True)
    if "process:  stopped" not in service_status:
        raise RuntimeError("local Qwen must be stopped before gfx1030 measurement")
    level_path = device_path / "power_dpm_force_performance_level"
    level_before = level_path.read_text().strip()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLLM_")
        and key not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    }
    env.update({"ROCR_VISIBLE_DEVICES": UUID, "LD_LIBRARY_PATH": "/opt/rocm/lib"})
    jobs: list[dict] = []
    state = "complete"
    error = None
    started = time.time()
    try:
        for shape_index, shape, m in shape_cases:
            label = f"shape{shape_index}-m{m}-k{shape[0]}-n{shape[1]}"
            raw_path = output / f"{label}.jsonl"
            stderr_path = output / f"{label}.stderr"
            command_run = [
                str(binary),
                "--target",
                TARGET,
                "--shape",
                str(shape_index),
                "--bench",
                "0" if args.no_bench else "1",
            ]
            job = {
                "shape": {"m": m, "k": shape[0], "n": shape[1]},
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
            job["validation"] = validate_case(raw_path, shape, m, not args.no_bench)
    except Exception as exc:  # noqa: BLE001 - preserve fail-closed report
        state = "failed"
        error = str(exc)
    level_after = level_path.read_text().strip()
    after = source_hashes(archive)
    report = {
        "schema": "phase87-stage1-small-m-splitk-execution-v1",
        "state": state,
        "target": TARGET,
        "uuid": UUID,
        "device_sysfs": str(device_path),
        "qwen_service_status": service_status,
        "performance_level_before": level_before,
        "performance_level_after": level_after,
        "performance_level_restored": level_before == level_after,
        "lowp_archive": str(archive),
        "lowp_archive_sha256": digest(archive),
        "binary": str(binary),
        "binary_sha256": digest(binary) if binary.is_file() else None,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "run_inputs_unchanged": before == after,
        "compile_command": command,
        "benchmark": not args.no_bench,
        "shapes": [{"index": index, "m": m, "k": shape[0], "n": shape[1]} for index, shape, m in shape_cases],
        "jobs": jobs,
        "started": started,
        "ended": time.time(),
    }
    if error is not None:
        report["error"] = error
    if not report["performance_level_restored"] or not report["run_inputs_unchanged"]:
        report["state"] = "failed"
    (output / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Phase87 Stage1 small-M split-K {TARGET}: {report['state']} jobs={len(jobs)}/{len(shape_cases)}")
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
